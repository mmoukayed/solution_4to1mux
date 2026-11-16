
"""One-arm RGB-D grasp waypoints, controller results and contact verification.

Gripper commands use the trajectory topic; the simulation clamp is topic-only.
The live profile also lists the hardware's gripper action endpoints.
This local IK/waypoint controller does not provide whole-body collision planning.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformListener, TransformException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .ros_topics import configure_ros
from .geometry import ArmKinematics, pose_xyz, rotation, transform_matrix, yaw

GRIPPER_JOINT = 'gripper_right_finger_joint'
GRIPPER_GRASP_FRAME = 'gripper_right_grasping_link'
ARM_JOINTS = [f'arm_right_{i}_joint' for i in range(1, 8)]


def entity_name(value):
    return value if isinstance(value, str) else getattr(value, 'name', '')


class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')
        self.ros = configure_ros(self)
        topics = self.ros.topics
        simulation = self.ros.profile == 'simulation'
        for name, value in {
            # The sim clamp's joint convention is opposite to the live robot:
            # sim 0.0 is closed and 0.069 is fully open; live 0.0 is open.
            'gripper_open': 0.066 if simulation else 0.0,
            'gripper_closed': 0.0 if simulation else 0.79,
            # Insert past the measured spine rather than asking the base to
            # violate its shelf keepout. Keep the lateral target centred on
            # perception; the former 1.5 cm right bias led to left-only contact.
            'grasp_insert_depth': 0.085, 'grasp_right_offset': 0.0,
            'pregrasp_distance': 0.10,
            'grasp_correction_limit': 0.12,
            'grasp_position_tolerance': 0.012,
            'grasp_orientation_tolerance': 0.10,
            'lift_distance': 0.025, 'extract_distance': 0.10,
            # Must match navigation.grasp_base_return_advance.  This offsets
            # the prealignment target while the base is backed out.
            'grasp_base_advance': 0.28,
            # Top-row pickup stays folded during base return, then approaches
            # above the target so the wrist does not sweep across the middle row.
            'top_book_clearance_height': 0.16,
            'top_book_front_standoff': 0.10,
            'top_book_arm_clearance': 0.06,
            # Keep the existing hover height while releasing the upright book
            # higher above the rim, with a shorter torso-only descent.
            'place_base_advance': 0.35, 'place_clearance_height': 0.10,
            'place_torso_hover': 0.35,
            'place_position_tolerance': 0.025, 'place_orientation_tolerance': 0.10,
            'place_correction_limit': 0.12, 'delivery_contact_hold': 0.5,
            'target_max_age': 2.0, 'gripper_timeout': 8.0,
            'placement_target_max_age': 30.0,
            'grip_contact_hold': 1.0,
            'motion_timeout_margin': 8.0, 'planning_timeout': 60.0,
            # The measured bin point is near the far half of its opening;
            # release 22 cm back toward the robot for a centred drop.
            'bin_inset': 0.22, 'bin_release_height': 0.31,
        }.items():
            self.declare_parameter(name, value)
        gripper_limit = 0.069 if simulation else 0.803
        for key in ('gripper_open', 'gripper_closed'):
            if not 0 <= self.p(key) <= gripper_limit:
                raise ValueError(
                    f'{key} must lie in the {self.ros.profile} gripper interval '
                    f'[0, {gripper_limit:.3f}]')
        if self.p('gripper_open') == self.p('gripper_closed'):
            raise ValueError('gripper_open and gripper_closed must be different positions')
        for key in ('grasp_correction_limit', 'grasp_position_tolerance',
                    'grasp_orientation_tolerance', 'top_book_front_standoff',
                    'top_book_arm_clearance', 'place_clearance_height',
                    'place_position_tolerance', 'place_orientation_tolerance',
                    'place_correction_limit', 'delivery_contact_hold'):
            if not math.isfinite(self.p(key)) or self.p(key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not math.isfinite(self.p('top_book_clearance_height')) or self.p('top_book_clearance_height') < 0:
            raise ValueError('top_book_clearance_height must be finite and nonnegative')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.model = None
        self.model_error = ''
        self.base_to_grasp = None
        self.joints = {}
        self.joint_stamp = None
        self.targets = {}
        self.book_row = None
        self.contacts = {}
        self.holding_book = None
        self.bin_contact_stamp = None
        self.arm = ActionClient(self, FollowJointTrajectory, self.ros.actions['right_arm_trajectory'])
        self.torso = ActionClient(self, FollowJointTrajectory, self.ros.actions['torso_trajectory'])
        self.gripper_pub = self.create_publisher(JointTrajectory, topics['right_gripper_command'], 10)
        self.ready_pub = self.create_publisher(Bool, topics['manipulation_ready'], 10)
        self.holding_pub = self.create_publisher(Bool, topics['holding_book'], 10)
        self.failure_pub = self.create_publisher(String, topics['manipulation_failure'], QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.grab_pub = self.create_publisher(Bool, topics['grab_complete'], 10)
        self.prepared_pub = self.create_publisher(Bool, topics['grab_prepared'], 10)
        self.place_pub = self.create_publisher(Bool, topics['place_complete'], 10)
        self.place_prepared_pub = self.create_publisher(Bool, topics['place_prepared'], 10)
        self.carry_raised_pub = self.create_publisher(Bool, topics['carry_raised'], 10)
        self.create_subscription(String, topics['robot_description'], self.on_description,
                                   QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Int32, topics['book_row'], self.on_book_row,
                                   QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointState, topics['joint_states'], self.on_joints, qos_profile_sensor_data)
        for name in ('book', 'bin'):
            self.create_subscription(PoseStamped, topics[f'{name}_pose'], lambda m, n=name: self.targets.update({n: m}), 10)
        self.create_subscription(Bool, topics['request_grab'], lambda m: self.request('grab') if m.data else None, 10)
        self.create_subscription(Bool, topics['request_prepare_grab'], lambda m: self.request('prepare_grab') if m.data else None, 10)
        self.create_subscription(Bool, topics['request_finish_grab'], lambda m: self.request('finish_grab') if m.data else None, 10)
        self.create_subscription(Bool, topics['request_place'], lambda m: self.request('place') if m.data else None, 10)
        self.create_subscription(Bool, topics['request_prepare_place'], lambda m: self.request('prepare_place') if m.data else None, 10)
        self.create_subscription(Bool, topics['request_finish_place'], lambda m: self.request('finish_place') if m.data else None, 10)
        self.create_subscription(Bool, topics['request_raise_carry'], lambda m: self.request('raise_carry') if m.data else None, 10)
        self.create_subscription(Bool, topics['cancel_manipulation'], lambda m: self.abort('Cancelled by navigation') if m.data and self.active else None, 10)
        if topics['contacts'] is not None or topics['bin_contacts'] is not None:
            from ros_gz_interfaces.msg import Contacts
            if topics['contacts'] is not None:
                self.create_subscription(Contacts, topics['contacts'], self.on_contacts, qos_profile_sensor_data)
            if topics['bin_contacts'] is not None:
                self.create_subscription(Contacts, topics['bin_contacts'], self.on_bin_contacts, qos_profile_sensor_data)
        if topics['contacts'] is None or topics['bin_contacts'] is None:
            self.get_logger().warn(
                'This profile has no simulated grasp/bin contact feedback. '
                'Hardware grasp and delivery verification still need an adapter or implementation.')
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.plan_future = None
        self.arm_in_shelf = False
        self.retreat_target = None
        self.close_target = None
        self.grasp_feedback_started = None
        self.placement_hover_target = None
        self.placement_release_target = None
        self.placement_feedback_started = None
        self.release_started = None
        self.delivery_contact_since = None
        self.active = None
        self.phase = None
        self.phases = []
        self.deferred_plan = None
        self.handles = []
        self.pending = 0
        self.token = 0
        self.deadline = 0.0
        self.warned_at = -100.0
        # Publish readiness on wall time, even while Gazebo sim time is paused.
        # Motion deadlines still use the simulator clock in tick().
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.1, self.tick, clock=self.wall_clock)
        self.get_logger().info(
            f'Manipulation BUILD 5.28 (vertical centred bin release): {self.ros.profile} gripper positions '
            f'open={self.p("gripper_open"):.3f}, closed={self.p("gripper_closed"):.3f}.')

    def p(self, name):
        return self.get_parameter(name).value

    def now(self):
        return self.get_clock().now().nanoseconds/1e9

    def fresh(self, stamp, age=1.0):
        return stamp is not None and 0 <= self.now()-stamp <= age

    def on_description(self, msg):
        if self.model is None:
            try:
                self.model = ArmKinematics(msg.data, base=self.ros.frames['base'],
                                           tip=self.ros.frames['arm_tip'], profile=self.ros.profile)
                self.model.limits.validate([GRIPPER_JOINT],
                                           [[self.p('gripper_open')], [self.p('gripper_closed')]])
                self.get_logger().info(f'Loaded right-arm/torso kinematics and limits from {self.ros.topics["robot_description"]}.')
                try:
                    self.base_to_grasp = self.model.fixed_transform(
                        self.ros.frames['arm_tip'], GRIPPER_GRASP_FRAME)
                    self.get_logger().info(
                        f'Loaded fixed grasp-frame transform from '
                        f'{self.ros.frames["arm_tip"]} to {GRIPPER_GRASP_FRAME}.')
                except ValueError as error:
                    self.get_logger().warn(
                        f'{error}; manipulation requests will remain blocked until the URDF '
                        'provides the grasp-frame transform.')
            except (ValueError, AttributeError) as error:
                self.model = self.base_to_grasp = None
                self.model_error = str(error)
                self.get_logger().error(f'Cannot load arm model: {error}')

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))
        self.joint_stamp = Time.from_msg(msg.header.stamp).nanoseconds/1e9

    def on_book_row(self, msg):
        self.book_row = int(msg.data) if 1 <= msg.data <= 4 else None

    def on_contacts(self, msg):
        for contact in msg.contacts:
            a, b = entity_name(contact.collision1), entity_name(contact.collision2)
            for gripper, book in ((a, b), (b, a)):
                if 'gripper_right_' not in gripper or 'book' not in book:
                    continue
                side = 'left' if 'left' in gripper else 'right' if 'right' in gripper else None
                if side:
                    self.contacts[(book, side)] = self.now()

    def held_book(self):
        books = {book for book, _ in self.contacts}
        for book in books:
            if all(self.fresh(self.contacts.get((book, side)), 0.5) for side in ('left', 'right')):
                return book
        return None

    def on_bin_contacts(self, msg):
        # A collision with the robot itself is not evidence that the book was delivered.
        if not self.holding_book:
            return
        for contact in msg.contacts:
            names = [entity_name(contact.collision1), entity_name(contact.collision2)]
            if self.holding_book in names and any('collection_bin' in n for n in names):
                self.bin_contact_stamp = self.now()

    def readiness_issues(self):
        issues = []
        if self.model is None:
            detail = f'URDF model not loaded from {self.ros.topics["robot_description"]}'
            if self.model_error:
                detail += f' ({self.model_error})'
            issues.append(detail)
        elif self.base_to_grasp is None:
            issues.append(
                f'URDF has no fixed transform from {self.ros.frames["arm_tip"]} '
                f'to {GRIPPER_GRASP_FRAME}')
        if not self.fresh(self.joint_stamp):
            issues.append(f'no fresh joint feedback on {self.ros.topics["joint_states"]}')
        elif self.model is not None:
            missing = [name for name in self.model.names+[GRIPPER_JOINT]
                       if name not in self.joints]
            if missing:
                issues.append(f'{self.ros.topics["joint_states"]} is missing {missing}')
        if not self.arm.server_is_ready():
            issues.append(f'arm action unavailable: {self.ros.actions["right_arm_trajectory"]}')
        if not self.torso.server_is_ready():
            issues.append(f'torso action unavailable: {self.ros.actions["torso_trajectory"]}')
        if self.gripper_pub.get_subscription_count() == 0:
            issues.append(f'no subscriber on gripper topic {self.ros.topics["right_gripper_command"]}')
        return issues

    def ready(self):
        return not self.readiness_issues()

    def request(self, sequence):
        if self.active:
            return
        issues = self.readiness_issues()
        if issues:
            self.report(sequence, False, 'Readiness blocked: '+'; '.join(issues)+'.')
            return
        if sequence in ('place', 'prepare_place', 'finish_place', 'raise_carry') and not self.holding_book:
            self.report(sequence, False, 'Cannot place: no verified grasp.')
            return
        if sequence == 'raise_carry':
            self.active = sequence
            self.token += 1
            self.arm_in_shelf = False
            self.retreat_target = None
            self.phase = 'planning'
            self.deadline = self.now()+self.p('planning_timeout')
            current = np.array([self.joints[n] for n in self.model.names])
            self.plan_future = self.pool.submit(self.plan_carry_raise, current)
            self.get_logger().info('raise_carry: solving a raised carry pose within the loaded robot limits.')
            return
        if sequence == 'prepare_place':
            self.placement_hover_target = None
            self.placement_release_target = None
        target = self.targets.get('book' if sequence in ('grab', 'prepare_grab', 'finish_grab') else 'bin')
        target_age = (self.p('placement_target_max_age') if sequence in ('place', 'prepare_place', 'finish_place')
                      else self.p('target_max_age'))
        if target is None or target.header.frame_id != self.ros.frames['odom'] or not self.fresh(Time.from_msg(target.header.stamp).nanoseconds/1e9, target_age):
            self.report(sequence, False, f'No fresh target pose in {self.ros.frames["odom"]}; inspect RGB-D perception.')
            return
        try:
            tf = self.tf.lookup_transform(self.ros.frames['base'], self.ros.frames['odom'], Time())
        except TransformException as error:
            self.report(sequence, False, f'Cannot transform target to robot base: {error}')
            return
        matrix = transform_matrix(tf.transform)
        point = (matrix @ np.r_[pose_xyz(target), 1])[:3]
        normal_world = np.array([math.cos(yaw(target.pose.orientation)), math.sin(yaw(target.pose.orientation)), 0])
        normal = matrix[:3, :3] @ normal_world
        normal[2] = 0
        normal /= np.linalg.norm(normal)
        self.active = sequence
        self.token += 1
        self.arm_in_shelf = False
        if sequence != 'finish_grab':
            self.retreat_target = None
        if sequence in ('grab', 'prepare_grab', 'finish_grab'):
            self.close_target = None
            self.grasp_feedback_started = None
        current = np.array([self.joints[n] for n in self.model.names])
        params = {key: self.p(key) for key in ('grasp_insert_depth', 'grasp_right_offset', 'pregrasp_distance', 'grasp_correction_limit', 'grasp_position_tolerance', 'grasp_orientation_tolerance', 'lift_distance', 'extract_distance', 'grasp_base_advance', 'top_book_clearance_height', 'top_book_front_standoff', 'top_book_arm_clearance', 'place_base_advance', 'place_clearance_height', 'place_torso_hover', 'place_correction_limit', 'bin_inset', 'bin_release_height')}
        self.deferred_plan = None
        self.phases = []
        if sequence == 'prepare_grab' and self.book_row == 1:
            # Raise first, before IK is attempted, so an unreachable seed can
            # never prevent the torso from moving to the top-row reach height.
            self.deferred_plan = (sequence, point, normal, params)
            torso_target = current.copy()
            torso_target[0] = self.model.upper[0]
            if abs(torso_target[0]-current[0]) > 0.005:
                duration = self.model.duration(current, torso_target, 1.0)
                self.phases = [('raise_torso_for_top_book', 'torso',
                                torso_target, duration)]
            self.next_phase()
            return
        self.phase = 'planning'
        self.deadline = self.now()+self.p('planning_timeout')
        self.plan_future = self.pool.submit(self.plan, sequence, point, normal, current, params)
        self.get_logger().info(f'{sequence}: planning for camera target in base frame {point.round(3)}')

    def grasp_pose(self, joints):
        pose = self.model.fk(joints)
        if self.base_to_grasp is not None:
            pose = pose @ self.base_to_grasp
        return pose

    def grasp_alignment_issue(self):
        """Require measured open fingers at the intended grasp, not just action success."""
        if self.close_target is None:
            return 'No planned finger pose is available.'
        if (self.grasp_feedback_started is None or not self.fresh(self.joint_stamp) or
                self.joint_stamp < self.grasp_feedback_started):
            return 'Waiting for joint feedback after insertion.'
        names = self.model.names+[GRIPPER_JOINT]
        if any(name not in self.joints for name in names):
            return 'Missing arm, torso or gripper joint feedback.'
        current = np.array([self.joints[name] for name in self.model.names])
        if not np.isfinite(current).all() or not math.isfinite(self.joints[GRIPPER_JOINT]):
            return 'Nonfinite arm or gripper joint feedback.'
        actual = self.grasp_pose(current)
        delta = self.close_target[:3, 3]-actual[:3, 3]
        position_error = float(np.linalg.norm(delta))
        angle = math.acos(float(np.clip(
            (np.trace(self.close_target[:3, :3].T @ actual[:3, :3])-1.0)/2.0, -1.0, 1.0)))
        if (position_error > self.p('grasp_position_tolerance') or
                angle > self.p('grasp_orientation_tolerance')):
            return (f'Finger pose misses the book by {position_error:.3f} m '
                    f'(target minus measured xyz={delta.round(3).tolist()} m), '
                    f'orientation error {math.degrees(angle):.1f} deg; redock.')
        if abs(self.joints[GRIPPER_JOINT]-self.p('gripper_open')) >= 0.006:
            return 'Gripper is not confirmed open; refusing to close at an unverified insertion.'
        return ''

    def plan(self, sequence, point, normal, current, params):
        """Compute all waypoints before any actuator moves; never guess joint angles."""
        if not np.isfinite(point).all() or point[0] < 0.2:
            raise ValueError(f'Target is not in front of the arm at reachable distance: {point}')
        reach_distance = point[0]
        if sequence == 'prepare_grab' and self.book_row != 1:
            # This phase runs after the base has backed out. Validate the
            # compensated arm target that will be reached after the planned
            # base advance, rather than rejecting the intentionally larger
            # book-to-base distance at the staging pose.
            reach_distance += normal[0]*(params['grasp_insert_depth']-
                                         params['grasp_base_advance'])
        if sequence in ('grab', 'prepare_grab', 'finish_grab') and not 0.45 <= reach_distance <= 1.05:
            # A grasp target closer than ~0.45 m means the base ended up jammed
            # against the shelf; swinging the arm from there hits shelf boards.
            raise ValueError(f'{sequence} reach target is {reach_distance:.2f} m in base frame, need 0.45-1.05 m; redock.')
        # +y is robot-left in base_footprint. Keep the default target neutral;
        # a nonzero grasp_right_offset remains available for measured tuning.
        point = point+np.array([0.0, -params.get('grasp_right_offset', 0.0), 0.0]) if sequence in ('grab', 'prepare_grab', 'finish_grab') else point
        if sequence == 'prepare_grab' and self.book_row == 1:
            return self.plan_top_pregrasp(point, normal, current, params)
        if sequence == 'finish_place':
            return self.plan_finish_place(point, normal, current, params)
        facing = rotation([0, 0, 1], math.atan2(normal[1], normal[0]))
        if sequence == 'grab':
            grasp = point+normal*params['grasp_insert_depth']
            pre = grasp-normal*params['pregrasp_distance']
            lifted = grasp+np.array([0, 0, params['lift_distance']])
            extracted = lifted-normal*params['extract_distance']
            targets = [('pregrasp', pre), ('insert', grasp), ('lift', lifted), ('extract', extracted)]
        elif sequence == 'prepare_grab':
            # Top-row preparation is handled by plan_top_pregrasp above.
            # Lower rows compensate for the upcoming base advance.
            targets = [('prealign', point+normal*(params['grasp_insert_depth']-params['grasp_base_advance']))]
        elif sequence == 'finish_grab':
            grasp = point+normal*params['grasp_insert_depth']
            actual_grasp = self.grasp_pose(current)
            correction = float(np.linalg.norm(grasp-actual_grasp[:3, 3]))
            if correction > params['grasp_correction_limit']:
                raise ValueError(f'Gripper needs {correction:.3f} m of correction after base return '
                                 f'(limit {params["grasp_correction_limit"]:.3f} m); redock.')
            lifted = grasp+np.array([0, 0, params['lift_distance']])
            extracted = lifted-normal*params['extract_distance']
            # Base return can stop short at its tolerance or shelf keepout.
            # Correct from the new RGB-D target for every row before closing.
            targets = [('insert', grasp), ('lift', lifted), ('extract', extracted)]
        elif sequence == 'prepare_place':
            # Perception's bin normal points from the camera/robot toward the
            # bin. Move against it to centre the book in the opening. The base
            # has already reached its calibrated standoff; the torso supplies
            # the final straight down motion after this hover pose is reached.
            release = point-normal*params['bin_inset']+np.array([0, 0, params['bin_release_height']])
            hover = release+np.array([0, 0, params['place_clearance_height']])
            targets = [('crane_hover', hover)]
        else:
            release = point-normal*params['bin_inset']+np.array([0, 0, params['bin_release_height']])
            targets = [('over_bin', release+np.array([0, 0, params['place_clearance_height']])), ('lower_to_bin', release)]
        errors = []
        candidates = []
        # These are grasp-frame orientations. Both rolls keep the held book
        # vertical. Quarter-turn wrist fallbacks would lay it across the rim.
        rolls = (math.pi, 0.0)
        if sequence == 'finish_grab':
            # Keep the prepared wrist roll; do not flip between shelf boards.
            rolls = [min(rolls, key=lambda roll: np.linalg.norm(
                facing @ rotation([1, 0, 0], roll)-actual_grasp[:3, :3]))]
        for roll in rolls:
            grasp_orientation = facing @ rotation([1, 0, 0], roll)
            if self.base_to_grasp is not None:
                # `targets` are expressed at the URDF grasp frame. Convert
                # them back to the configured arm-tip link so IK places the
                # fingers and their approach axis at the book, not the wrist
                # base (whose fixed grasp-frame joint is translated/rotated).
                orientation = grasp_orientation @ self.base_to_grasp[:3, :3].T
                offset = self.base_to_grasp[:3, 3]
            else:
                orientation = grasp_orientation
                offset = None
            q = current.copy()
            if sequence == 'prepare_place':
                q[0] = min(self.model.upper[0], params['place_torso_hover'])
            candidate = []
            try:
                for name, xyz in targets:
                    fixed_torso = sequence == 'prepare_place'
                    fixed = {0: q[0]} if fixed_torso else None
                    tip_xyz = xyz if offset is None else xyz-orientation @ offset
                    if sequence == 'finish_grab':
                        actual = self.grasp_pose(q)
                        angle = math.acos(float(np.clip(
                            (np.trace(grasp_orientation.T @ actual[:3, :3])-1)/2, -1, 1)))
                        if (name == 'insert' and
                                np.linalg.norm(xyz-actual[:3, 3]) <= params.get('grasp_position_tolerance', .012) and
                                angle <= params.get('grasp_orientation_tolerance', .10)):
                            # Closing an already aligned hand needs no new IK.
                            path = np.atleast_2d(q.copy())
                        else:
                            # Allow bounded torso travel when it is needed to
                            # stay on the current arm branch near a joint limit.
                            path = self.model.cartesian_path(tip_xyz, orientation, q)
                        times = self.model.limits.times(self.model.names, q, path, minimum=1.0)
                        candidate.append((name, 'motion', path if len(path) > 1 else path[0], float(times[-1])))
                        q = path[-1]
                    else:
                        solved = self.model.solve(tip_xyz, orientation, q, fixed=fixed)
                        if candidate and np.max(np.abs(solved[1:]-q[1:])) > 1.4:
                            raise ValueError(f'{name}: IK branch jump; redock before retrying.')
                        candidate.append((name, 'motion', solved, self.model.duration(q, solved, 1.0)))
                        q = solved
                # Both parallel-jaw rolls can grasp the same book. Keep the
                # wrist nearer its current orientation by selecting the valid
                # complete path with the least estimated joint travel time,
                # rather than accepting the first roll that happens to solve.
                candidates.append((sum(step[3] for step in candidate), candidate, grasp_orientation))
            except ValueError as error:
                errors.append(str(error))
        if not candidates:
            if sequence == 'finish_grab':
                raise ValueError('No continuous grasp path within robot limits; redock. '+'; '.join(errors))
            raise ValueError('; '.join(errors))
        _, motions, chosen_orientation = min(candidates, key=lambda item: item[0])
        if sequence in ('grab', 'finish_grab'):
            self.close_target = np.eye(4)
            self.close_target[:3, 3] = point+normal*params['grasp_insert_depth']
            self.close_target[:3, :3] = chosen_orientation
            confirmation = ('confirm_grasp_alignment', 'grasp_feedback', self.close_target, 2.0)
        if sequence == 'grab':
            # Remember the pregrasp joints: they are the known-collision-free
            # retreat if anything fails while the fingers are inside the shelf.
            self.retreat_target = motions[0][2]
            return [('open_gripper', 'open', None, 1.0), motions[0], motions[1],
                    confirmation, ('close_gripper', 'close', None, 1.0), motions[2], motions[3],
                    ('verify_book_held', 'verify', None, 0.8)]
        if sequence == 'prepare_grab':
            return [('open_gripper', 'open', None, 1.0), motions[0]]
        if sequence == 'finish_grab':
            if self.book_row == 1 and self.retreat_target is None:
                raise ValueError('Top-row pregrasp joints were not retained; prepare_grab must complete before finish_grab.')
            if self.book_row != 1:
                # Save the measured position before correction, not an
                # extraction/lift target from a different point on the shelf.
                self.retreat_target = current.copy()
            return [('open_gripper', 'open', None, 1.0), motions[0], confirmation,
                    ('close_gripper', 'close', None, 1.0), motions[1], motions[2],
                    ('verify_book_held', 'verify', None, 0.8)]
        if sequence == 'prepare_place':
            self.placement_hover_target = motions[-1][2].copy()
            self.placement_release_target = np.eye(4)
            self.placement_release_target[:3, :3] = chosen_orientation
            self.placement_release_target[:3, 3] = release
            return motions
        self.placement_release_target = np.eye(4)
        self.placement_release_target[:3, :3] = chosen_orientation
        self.placement_release_target[:3, 3] = release
        return motions+[('confirm_bin_release', 'place_feedback', self.placement_release_target, 2.),
                        ('release_book', 'open', None, 1.0),
                        ('verify_delivery', 'delivery', None, 3.0)]

    def plan_finish_place(self, point, normal, current, params):
        """Refresh bin alignment while high, then lower only the torso."""
        if self.placement_hover_target is None or self.placement_release_target is None:
            raise ValueError('No staged vertical bin pose; run prepare_place first.')
        release = point-normal*params['bin_inset']+np.array([0., 0., params['bin_release_height']])
        hover = release+np.array([0., 0., params['place_clearance_height']])
        actual = self.grasp_pose(current)
        if np.linalg.norm(hover-actual[:3, 3]) > params['place_correction_limit']:
            raise ValueError('Bin alignment changed too far; prepare placement again before lowering.')
        grasp_orientation = self.placement_release_target[:3, :3].copy()
        orientation = grasp_orientation @ self.base_to_grasp[:3, :3].T
        tip_xyz = hover-orientation @ self.base_to_grasp[:3, 3]
        if np.linalg.norm(hover-actual[:3, 3]) < .008 and np.linalg.norm(actual[:3, :3]-grasp_orientation) < .06:
            path = np.atleast_2d(current.copy())
        else:
            path = self.model.cartesian_path(tip_xyz, orientation, current, fixed={0: current[0]})
        hover_q = path[-1]
        lowered = hover_q.copy()
        lowered[0] -= params['place_clearance_height']
        self.model.limits.validate(self.model.names, lowered)
        if params['place_clearance_height'] < .05:
            raise ValueError('Placement needs at least 5 cm of checked torso descent.')
        self.placement_release_target = np.eye(4)
        self.placement_release_target[:3, :3] = grasp_orientation
        self.placement_release_target[:3, 3] = release
        hover_pose = self.placement_release_target.copy()
        hover_pose[:3, 3] = hover
        times = self.model.limits.times(self.model.names, current, path, minimum=1.)
        return [('align_vertical_over_bin', 'motion', path if len(path) > 1 else path[0], float(times[-1])),
                ('confirm_bin_hover', 'place_hover_feedback', hover_pose, 2.),
                ('lower_to_bin', 'torso', lowered, self.model.duration(hover_q, lowered)),
                ('confirm_bin_release', 'place_feedback', self.placement_release_target, 2.),
                ('release_book', 'open', None, 1.), ('verify_delivery', 'delivery', None, 3.),
                ('clear_bin', 'torso', hover_q, self.model.duration(lowered, hover_q))]

    def placement_alignment_issue(self, target):
        if target is None:
            return 'No verified vertical release target.'
        if (self.placement_feedback_started is None or not self.fresh(self.joint_stamp) or
                self.joint_stamp < self.placement_feedback_started):
            return 'Waiting for fresh placement joint feedback.'
        if not self.holding_book or self.held_book() != self.holding_book:
            return 'Held book contact is not confirmed before release.'
        if any(name not in self.joints for name in self.model.names):
            return 'Missing placement arm or torso feedback.'
        current = np.array([self.joints[name] for name in self.model.names])
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            return 'Nonfinite placement pose or joint feedback.'
        actual = self.grasp_pose(current)
        if not np.isfinite(actual).all():
            return 'Nonfinite measured placement transform.'
        tilt = math.acos(float(np.clip(abs(actual[2, 2]), 0., 1.)))
        error = float(np.linalg.norm(target[:3, 3]-actual[:3, 3]))
        angle = math.acos(float(np.clip((np.trace(target[:3, :3].T @ actual[:3, :3])-1)/2, -1., 1.)))
        if tilt > self.p('place_orientation_tolerance'):
            return f'Book is not vertical ({math.degrees(tilt):.1f} deg tilt).'
        if error > self.p('place_position_tolerance') or angle > self.p('place_orientation_tolerance'):
            return f'Bin placement pose is misaligned by {error:.3f} m / {math.degrees(angle):.1f} deg.'
        return ''

    def top_path_margin(self, start, path, point, normal, arm_clearance):
        """Keep the sampled arm/wrist path in front of the book's shelf plane.

        The grasp frame may approach the spine; the thicker arm and wrist need
        full clearance. This plane check does not replace mesh collision checks.
        """
        distance = float(point @ normal)
        margin = float('inf')
        before = start
        for target in np.atleast_2d(path):
            count = max(3, int(np.ceil(np.max(np.abs(target-before))/0.04))+1)
            for fraction in np.linspace(0., 1., count):
                q = before+fraction*(target-before)
                points = self.model.arm_points(q, self.base_to_grasp)
                margin = min(margin, distance-arm_clearance-float(np.max(points[:-1] @ normal)),
                             distance-float(points[-1] @ normal))
            before = target
        return margin

    def plan_top_pregrasp(self, point, normal, current, params):
        """Find the highest reachable complete approach at the safe torso limit."""
        height = params['top_book_clearance_height']
        standoff = params['top_book_front_standoff']
        arm_clearance = params.get('top_book_arm_clearance', .06)
        if (not np.isfinite([height, standoff, arm_clearance]).all() or height < 0 or
                min(standoff, arm_clearance) <= 0):
            raise ValueError('Top-row geometry needs nonnegative height and positive clearances.')
        grasp = point+normal*params['grasp_insert_depth']
        pre = grasp-normal*params['pregrasp_distance']
        facing = rotation([0, 0, 1], math.atan2(normal[1], normal[0]))
        errors = []
        for extra_height in np.linspace(height, 0., 5) if height > 0 else [0.]:
            above = pre+np.array([0., 0., extra_height])
            front = above-normal*standoff
            targets = ([('top_front_clearance', front), ('top_clearance', above), ('pregrasp', pre)]
                       if extra_height > 0 else [('top_front_clearance', front), ('pregrasp', pre)])
            candidates = []
            for roll in (math.pi, 0.):
                grasp_orientation = facing @ rotation([1, 0, 0], roll)
                orientation = (grasp_orientation if self.base_to_grasp is None else
                               grasp_orientation @ self.base_to_grasp[:3, :3].T)
                offset = np.zeros(3) if self.base_to_grasp is None else orientation @ self.base_to_grasp[:3, 3]
                seed = current.copy()
                seed[0] = self.model.upper[0]
                candidate, before = [], current.copy()
                try:
                    for index, (name, xyz) in enumerate(targets):
                        if index == 0:
                            try:
                                target = self.model.solve(xyz-offset, orientation, seed, fixed={0: seed[0]})
                                path = np.atleast_2d(target)
                            except ValueError as error:
                                path, first_error = None, str(error)
                            if path is None or self.top_path_margin(before, path, point, normal, arm_clearance) < -0.003:
                                # IK seeds are not commanded poses. Try two
                                # other bounded solutions before discarding a
                                # reachable target with an unsafe first branch.
                                for initial in (np.clip(np.zeros_like(seed), self.model.lower, self.model.upper),
                                                (self.model.lower+self.model.upper)/2):
                                    initial[0] = seed[0]
                                    try:
                                        target = self.model.solve(xyz-offset, orientation, initial,
                                                                  fixed={0: seed[0]}, attempts=2)
                                    except ValueError:
                                        continue
                                    alternative = np.atleast_2d(target)
                                    if self.top_path_margin(before, alternative, point, normal, arm_clearance) >= -0.003:
                                        path = alternative
                                        break
                            if path is None:
                                raise ValueError(first_error)
                        else:
                            path = self.model.cartesian_path(xyz-offset, orientation, seed, fixed={0: seed[0]})
                        if self.top_path_margin(before, path, point, normal, arm_clearance) < -0.003:
                            raise ValueError('sampled arm path enters the shelf-front plane')
                        times = self.model.limits.times(self.model.names, before, path, minimum=1.)
                        candidate.append((name, 'motion', path if len(path) > 1 else path[0], float(times[-1])))
                        seed, before = path[-1], path[-1]
                    candidates.append((sum(phase[3] for phase in candidate), candidate))
                except ValueError as error:
                    errors.append(f'{extra_height:.3f} m hover, roll {roll:.2f}, {name}: {error}')
            if candidates:
                motions = min(candidates, key=lambda item: item[0])[1]
                self.retreat_target = np.atleast_2d(motions[-1][2])[-1].copy()
                self.top_pregrasp_height = float(extra_height)
                return [('open_gripper', 'open', None, 1.)]+motions
        raise ValueError('No reachable shelf-clear top-row pregrasp. '+'; '.join(errors))

    def plan_carry_raise(self, current):
        """Solve a raised carry pose beside the shoulder within this robot's limits.

        Navigation has already withdrawn from the shelf. This is a joint-limit
        and geometric endpoint check, not a mesh collision planner.
        """
        shoulder = self.model.arm_points(current)[0]
        candidates, errors = [], []
        for forward in (0.25, 0.15):
            grasp = shoulder+np.array([forward, -0.20, 0.25])
            for roll in (math.pi, 0.0):
                facing = rotation([1, 0, 0], roll)
                orientation = facing if self.base_to_grasp is None else facing @ self.base_to_grasp[:3, :3].T
                xyz = grasp if self.base_to_grasp is None else grasp-orientation @ self.base_to_grasp[:3, 3]
                try:
                    target = self.model.solve(xyz, orientation, current, fixed={0: current[0]})
                    self.model.limits.validate(self.model.names, target)
                    points = self.model.arm_points(target, self.base_to_grasp)
                    if points[3:, 1].max() > -abs(shoulder[1])-0.04:
                        raise ValueError('Carry endpoint crosses the torso-side envelope.')
                    duration = self.model.duration(current, target)
                    candidates.append((duration, target))
                except ValueError as error:
                    errors.append(str(error))
        if not candidates:
            raise ValueError('No carry pose within robot limits: '+'; '.join(errors))
        duration, target = min(candidates, key=lambda item: item[0])
        return [('raise_carried_book', 'motion', target, duration)]

    def trajectory(self, names, positions, duration, times=None):
        msg = JointTrajectory()
        msg.joint_names = list(names)
        path = np.atleast_2d(np.asarray(positions, dtype=float))
        self.model.limits.validate(names, path)
        if any(name not in self.joints for name in names):
            raise ValueError('Missing current joint feedback for trajectory timing.')
        start = [self.joints[name] for name in names]
        if times is None:
            times = self.model.limits.times(names, start, path, minimum=duration)
        else:
            times = np.asarray(times, dtype=float)
            required = self.model.limits.times(names, start, path)
            if (times.shape != (len(path),) or not np.isfinite(times).all() or
                    np.any(np.diff(np.r_[0.0, times]) < np.diff(np.r_[0.0, required])-1e-9)):
                raise ValueError('Trajectory exceeds robot joint velocity limits or has invalid times.')
        for values, elapsed in zip(path, times):
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in values]
            stamp = int(math.ceil(float(elapsed)*1e9))
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(stamp, 10**9)
            msg.points.append(point)
        return msg

    def next_phase(self):
        self.phase_started = self.now()
        self.handles = []
        if not self.phases:
            if self.deferred_plan is not None:
                if not self.fresh(self.joint_stamp):
                    self.abort('Joint feedback went stale after the top-row torso lift; refusing arm motion.')
                    return
                sequence, point, normal, params = self.deferred_plan
                self.deferred_plan = None
                current = np.array([self.joints[n] for n in self.model.names])
                self.phase = 'planning'
                self.phase_name = 'planning_after_torso_lift'
                self.deadline = self.now()+self.p('planning_timeout')
                self.plan_future = self.pool.submit(
                    self.plan, sequence, point, normal, current, params)
                self.get_logger().info(
                    f'{sequence}: torso lift complete; planning the top-row pregrasp '
                    f'from joint feedback {current[0]:.3f} m.')
                return
            sequence = self.active
            self.active = self.phase = None
            self.report(sequence, True, 'Confirmed sequence complete.')
            if sequence in ('grab', 'finish_grab'):
                self.retreat_target = None
            return
        self.phase_name, self.phase, self.phase_target, duration = self.phases.pop(0)
        # Fingers cross the shelf front at 'insert' and are clear again once
        # 'extract' has succeeded (i.e. when 'verify_book_held' starts).
        if self.active in ('grab', 'finish_grab'):
            if self.phase_name == 'insert':
                self.arm_in_shelf = True
            elif self.phase_name == 'close_gripper' and self.active == 'finish_grab':
                self.arm_in_shelf = True
            elif self.phase_name == 'verify_book_held':
                self.arm_in_shelf = False
        self.get_logger().info(f'{self.active}: {self.phase_name}')
        if self.phase in ('motion', 'torso'):
            if not self.fresh(self.joint_stamp):
                self.abort('Joint feedback went stale before motion.')
                return
            path = np.atleast_2d(self.phase_target)
            # Recheck all positions and segment speeds before either actuator
            # is commanded, using the actual start rather than the planned seed.
            start = np.array([self.joints[n] for n in self.model.names])
            try:
                times = (self.model.limits.times(['torso_lift_joint'], start[:1], path[:, :1], minimum=duration)
                         if self.phase == 'torso' else
                         self.model.limits.times(self.model.names, start, path, minimum=duration))
            except ValueError as error:
                self.abort(f'Robot joint limits rejected {self.phase_name}: {error}')
                return
            self.deadline = self.now()+float(times[-1])+self.p('motion_timeout_margin')
            clients = (((self.torso, ['torso_lift_joint'], path[:, :1]),)
                       if self.phase == 'torso' else
                       ((self.torso, ['torso_lift_joint'], path[:, :1]),
                        (self.arm, ARM_JOINTS, path[:, 1:])))
            self.pending = len(clients)
            token = self.token
            for client, names, values in clients:
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = self.trajectory(names, values, duration, times=times)
                future = client.send_goal_async(goal)
                future.add_done_callback(lambda f, t=token: self.goal_response(f, t))
        elif self.phase in ('open', 'close'):
            value = self.p('gripper_open' if self.phase == 'open' else 'gripper_closed')
            if self.phase_name == 'release_book':
                issue = self.placement_alignment_issue(self.placement_release_target)
                if issue:
                    self.abort(f'Refusing book release: {issue}')
                    return
                self.release_started = self.now()
                self.bin_contact_stamp = None
                self.delivery_contact_since = None
            if self.phase == 'close':
                issue = self.grasp_alignment_issue()
                if issue:
                    self.abort(f'Refusing gripper close: {issue}')
                    return
                # Do not let contact from a previous approach/attempt count as
                # proof of a grasp. The close phase needs fresh bilateral data.
                self.contacts.clear()
            try:
                command = self.trajectory([GRIPPER_JOINT], [value], duration)
            except ValueError as error:
                self.abort(f'Robot gripper limits rejected {self.phase_name}: {error}')
                return
            self.gripper_pub.publish(command)
            self.deadline = self.now()+self.p('gripper_timeout')
            self.contact_since = None
        elif self.phase == 'grasp_feedback':
            self.grasp_feedback_started = self.now()
            self.deadline = self.now()+duration
        elif self.phase in ('place_hover_feedback', 'place_feedback'):
            self.placement_feedback_started = self.now()
            self.deadline = self.now()+duration
        else:
            self.deadline = self.now()+duration+3.0

    def goal_response(self, future, token):
        try:
            handle = future.result()
            if token != self.token or self.active is None:
                if handle.accepted:
                    handle.cancel_goal_async()
                return
            if not handle.accepted:
                self.abort(f'{self.phase_name}: controller rejected trajectory.')
                return
            self.handles.append(handle)
            handle.get_result_async().add_done_callback(lambda f, t=token: self.motion_result(f, t))
        except Exception as error:
            if token == self.token:
                self.abort(f'Action response failed: {error}')

    def motion_result(self, future, token):
        if token != self.token or self.active is None:
            return
        try:
            response = future.result()
            if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code != 0:
                self.abort(f'{self.phase_name}: status={response.status}, error={response.result.error_code}: {response.result.error_string}')
                return
            self.pending -= 1
            if self.pending == 0:
                if self.phase == 'torso':
                    actual = self.joints.get('torso_lift_joint')
                    target = float(self.phase_target[0])
                    if (not self.fresh(self.joint_stamp) or actual is None or
                            abs(actual-target) > 0.02):
                        # The action result may arrive just before the next
                        # joint-state sample. Wait briefly for measured lift;
                        # never start the arm using only the planned height.
                        self.phase = 'torso_feedback'
                        self.phase_name = 'confirm_torso_motion'
                        self.deadline = self.now()+3.0
                        return
                    self.get_logger().info(
                        f'{self.active}: torso motion confirmed at {actual:.3f} m.')
                self.next_phase()
        except Exception as error:
            self.abort(f'Action result failed: {error}')

    def tick(self):
        self.ready_pub.publish(Bool(data=self.ready()))
        self.holding_pub.publish(Bool(data=bool(self.holding_book and self.held_book() == self.holding_book)))
        if self.active is None:
            if not self.ready() and self.now()-self.warned_at > 5.0:
                self.warned_at = self.now()
                self.get_logger().warn(f'Waiting for robot/controllers: model={self.model is not None}, joints={self.fresh(self.joint_stamp)}, arm={self.arm.server_is_ready()}, torso={self.torso.server_is_ready()}, gripper_subscribers={self.gripper_pub.get_subscription_count()}')
            return
        if self.now() > self.deadline:
            detail = ''
            if self.phase == 'grasp_feedback':
                detail = ' '+self.grasp_alignment_issue()
            elif self.phase in ('place_hover_feedback', 'place_feedback'):
                detail = ' '+self.placement_alignment_issue(self.phase_target)
            elif self.phase == 'close':
                seen = {f'{book.split("::")[0]}:{side}': f'{self.now()-stamp:.1f}s ago'
                        for (book, side), stamp in self.contacts.items()}
                finger_position = self.joints.get(GRIPPER_JOINT)
                position_text = ('unknown' if finger_position is None else
                                 f'{finger_position:.4f} m')
                detail = (f' Gripper joint at timeout: {position_text}; last fresh-phase '
                          f'finger/book contacts: {seen or "none"}. If the joint reached 0.0 m '
                          'without bilateral contact, the fingers closed beside/in front of the '
                          'book; if it stayed open, inspect the gripper trajectory controller '
                          f'and {self.ros.topics["joint_states"]}.')
            self.abort(f'{self.phase}: timed out.{detail} Check controller feedback, contacts, target pose and Gazebo.')
            return
        if self.phase == 'planning':
            if self.plan_future.done():
                try:
                    self.phases = self.plan_future.result()
                    if self.active == 'prepare_grab' and self.book_row == 1:
                        self.get_logger().info(
                            f'Top-row approach planned with {self.top_pregrasp_height:.3f} m extra hover; '
                            'torso remains at its loaded safe upper limit.')
                    if self.active == 'prepare_place':
                        hover = next((phase[2] for phase in self.phases
                                      if phase[1] == 'motion'), None)
                        if hover is None:
                            raise ValueError('The bin hover plan did not contain a joint waypoint.')
                        self.placement_hover_target = np.atleast_2d(hover)[-1].copy()
                    self.next_phase()
                except Exception as error:
                    self.abort(f'No valid grasp/placement plan: {error}')
        elif self.phase == 'torso_feedback':
            actual = self.joints.get('torso_lift_joint')
            target = float(self.phase_target[0])
            if (self.fresh(self.joint_stamp) and actual is not None and
                    abs(actual-target) <= 0.02):
                self.get_logger().info(
                    f'{self.active}: torso motion confirmed at {actual:.3f} m.')
                self.next_phase()
        elif self.phase == 'open':
            if self.fresh(self.joint_stamp) and abs(self.joints.get(GRIPPER_JOINT, -1)-self.p('gripper_open')) < 0.006 and self.now()-self.phase_started > 0.8:
                self.next_phase()
        elif self.phase == 'grasp_feedback':
            if not self.grasp_alignment_issue():
                self.get_logger().info('Open fingers confirmed at the measured book grasp pose; closing gripper.')
                self.next_phase()
        elif self.phase in ('place_hover_feedback', 'place_feedback'):
            if not self.placement_alignment_issue(self.phase_target):
                self.get_logger().info(f'{self.active}: vertical book pose confirmed over the bin.')
                self.next_phase()
        elif self.phase == 'close':
            # This is a position trajectory. Once the one close trajectory
            # finishes, the controller holds its target; republishing a shorter
            # trajectory here can repeatedly restart/preempt the close motion.
            book = self.held_book()
            if book:
                if self.contact_since is None:
                    self.contact_since = self.now()
                if self.now()-self.contact_since >= self.p('grip_contact_hold'):
                    self.holding_book = book
                    self.next_phase()
            else:
                self.contact_since = None
        elif self.phase == 'verify':
            if self.now()-self.phase_started > 0.5:
                if self.held_book() == self.holding_book:
                    self.next_phase()
                else:
                    self.abort('Book contact lost after lift/extraction; grasp not confirmed.')
        elif self.phase == 'delivery':
            if (self.release_started is not None and self.bin_contact_stamp is not None and
                    self.bin_contact_stamp >= self.release_started and self.fresh(self.bin_contact_stamp, .5) and
                    self.fresh(self.joint_stamp) and
                    abs(self.joints.get(GRIPPER_JOINT, -1)-self.p('gripper_open')) < .006 and
                    self.held_book() != self.holding_book):
                if self.delivery_contact_since is None:
                    self.delivery_contact_since = self.now()
                if self.now()-self.delivery_contact_since >= self.p('delivery_contact_hold'):
                    self.holding_book = None
                    self.next_phase()
            else:
                self.delivery_contact_since = None

    def abort(self, reason):
        sequence = self.active
        self.token += 1
        for handle in self.handles:
            handle.cancel_goal_async()
        if self.plan_future:
            self.plan_future.cancel()
        self.phases = []
        self.deferred_plan = None
        self.active = self.phase = None
        self.retract_if_in_shelf()
        if sequence:
            self.report(sequence, False, reason)

    def retract_if_in_shelf(self):
        """After a failed grasp, never leave the fingers wedged between shelf boards.

        The pregrasp joints were solved on the straight approach line, so going
        back to them is a straight pull-out, not a new sideways swing.
        """
        retreat = getattr(self, 'retreat_target', None)
        if not getattr(self, 'arm_in_shelf', False) or retreat is None:
            return
        self.arm_in_shelf = False
        if self.model is None or not self.fresh(self.joint_stamp):
            self.get_logger().error('Cannot retract from shelf: no model or stale joint feedback.')
            return
        self.get_logger().warn('Grasp ended with the fingers inside the shelf; opening gripper and retracting to the pregrasp joints.')
        try:
            self.gripper_pub.publish(self.trajectory([GRIPPER_JOINT], [self.p('gripper_open')], 1.0))
            start = np.array([self.joints[n] for n in self.model.names])
            duration = self.model.duration(start, retreat)
            for client, names, values in ((self.torso, ['torso_lift_joint'], retreat[:1]),
                                          (self.arm, ARM_JOINTS, retreat[1:])):
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = self.trajectory(names, values, duration)
                client.send_goal_async(goal)
        except Exception as error:
            self.get_logger().error(f'Retract after failed grasp could not be sent: {error}')
        self.holding_book = None

    def report(self, sequence, success, reason):
        if not success:
            self.failure_pub.publish(String(data=f'{sequence}: {reason}'))
        publisher = (self.carry_raised_pub if sequence == 'raise_carry' else
                     self.prepared_pub if sequence == 'prepare_grab' else
                     self.place_prepared_pub if sequence == 'prepare_place' else
                     self.grab_pub if sequence in ('grab', 'finish_grab') else self.place_pub)
        publisher.publish(Bool(data=success))
        try:
            (self.get_logger().info if success else self.get_logger().error)(f'{sequence}: {reason}')
        except ValueError as error:
            # rcutils can reject a severity change while a callback is already
            # unwinding an exception. The failure has already been published;
            # never let diagnostic logging kill the manipulation node.
            if 'severity' not in str(error).lower():
                raise


def main():
    rclpy.init()
    node = ManipulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.active:
            node.abort('Node shutting down')
        node.pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        rclpy.shutdown()


