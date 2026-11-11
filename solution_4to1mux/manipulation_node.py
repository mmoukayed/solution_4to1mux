
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
# User-calibrated high carry pose.  This is arm-only: torso_lift_joint remains
# at its measured current value.
HIGH_CARRY_RIGHT_ARM = np.array([0.0, 1.25, 0.0, -1.0, 0.2, 1.7, 0.0])


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
            'lift_distance': 0.025, 'extract_distance': 0.10,
            # Must match navigation.grasp_base_return_advance.  This offsets
            # the prealignment target while the base is backed out.
            'grasp_base_advance': 0.28,
            # Top-row pickup stays folded during base return, then approaches
            # above the target so the wrist does not sweep across the middle row.
            'top_book_clearance_height': 0.16,
            'top_book_front_standoff': 0.10,
            # Must match navigation.place_base_advance.  The high pre-place
            # target is compensated while the base is still before the table.
            'place_base_advance': 0.35, 'place_clearance_height': 0.20,
            'place_torso_hover': 0.35, 'place_torso_drop': 0.0,
            'target_max_age': 2.0, 'gripper_timeout': 8.0,
            'placement_target_max_age': 30.0,
            'grip_contact_hold': 1.0,
            'motion_timeout_margin': 8.0, 'planning_timeout': 60.0,
            # The measured bin point is near the far half of its opening;
            # release 22 cm back toward the robot for a centred drop.
            'bin_inset': 0.22, 'bin_release_height': 0.21,
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
        self.grip_history = []   # (time, gripper position): the grasp check where no contact topic exists
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
        self.placement_hover_target = None
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
            f'Manipulation BUILD 5.19: {self.ros.profile} gripper positions '
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
                                           tip=self.ros.frames['arm_tip'])
                self.get_logger().info(f'Loaded right-arm/torso kinematics and limits from {self.ros.topics["robot_description"]}.')
                try:
                    self.base_to_grasp = self.model.fixed_transform(
                        self.ros.frames['arm_tip'], GRIPPER_GRASP_FRAME)
                    self.get_logger().info(
                        f'Loaded fixed grasp-frame transform from '
                        f'{self.ros.frames["arm_tip"]} to {GRIPPER_GRASP_FRAME}.')
                except ValueError as error:
                    # The live TIAGo Pro URDF has no grasping link: use the simulator's
                    # fixed gripper_right_grasping_frame_joint (xyz 0 0 0.157157, rpy 0 -1.57 0).
                    c, s = np.cos(-1.57), np.sin(-1.57)
                    self.base_to_grasp = np.array([[c, 0., s, 0.], [0., 1., 0., 0.],
                                                   [-s, 0., c, 0.157157], [0., 0., 0., 1.]])
                    self.get_logger().warn(f'{error}; using the simulator grasp-frame offset.')
            except (ValueError, AttributeError) as error:
                self.model_error = str(error)
                self.get_logger().error(f'Cannot load arm model: {error}')

    def on_joints(self, msg):
        self.joints.update(zip(msg.name, msg.position))
        self.joint_stamp = Time.from_msg(msg.header.stamp).nanoseconds/1e9
        if GRIPPER_JOINT in msg.name:
            self.grip_history.append((self.joint_stamp, self.joints[GRIPPER_JOINT]))
            self.grip_history = [h for h in self.grip_history if h[0] > self.joint_stamp-0.6]

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
        if self.ros.topics['contacts'] is None:
            # The live robot has no contact topic: the fingers held still for 0.5 s, stopped
            # clearly short of fully shut (0.79) and clearly closed from open (0.0), is a book.
            if not self.fresh(self.joint_stamp, 0.5) or len(self.grip_history) < 5:
                return None
            span = self.grip_history[-1][0]-self.grip_history[0][0]
            qs = [q for _, q in self.grip_history]
            if span >= 0.5 and max(qs)-min(qs) < 0.004 and 0.35 <= qs[-1] <= 0.75:
                return 'book_by_gripper_position'
            return None
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
            self.get_logger().info('raise_carry: moving the held book to the predefined high right-arm carry pose.')
            return
        if sequence == 'prepare_place':
            self.placement_hover_target = None
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
        current = np.array([self.joints[n] for n in self.model.names])
        params = {key: self.p(key) for key in ('grasp_insert_depth', 'grasp_right_offset', 'pregrasp_distance', 'lift_distance', 'extract_distance', 'grasp_base_advance', 'top_book_clearance_height', 'top_book_front_standoff', 'place_base_advance', 'place_clearance_height', 'place_torso_hover', 'place_torso_drop', 'bin_inset', 'bin_release_height')}
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
        if sequence == 'finish_place':
            if self.placement_hover_target is None:
                raise ValueError('No staged crane pose is available; run prepare_place before finish_place.')
            if params['place_torso_drop'] < 0.01:
                # release from the hover: lowering the torso put the upper arm on the table edge
                return [('release_book', 'open', None, 1.0),
                        ('verify_delivery', 'delivery', None, 3.0)]
            lowered = self.placement_hover_target.copy()
            lowered[0] = max(self.model.lower[0],
                             lowered[0]-params['place_torso_drop'])
            if self.placement_hover_target[0]-lowered[0] < 0.05:
                raise ValueError('Torso has less than 5 cm of safe downward travel from the bin hover pose.')
            duration = self.model.duration(current, lowered, 1.0)
            motions = [('lower_to_bin', 'motion', lowered, duration)]
            return motions+[('release_book', 'open', None, 1.0),
                            ('verify_delivery', 'delivery', None, 3.0)]
        facing = rotation([0, 0, 1], math.atan2(normal[1], normal[0]))
        if sequence == 'grab':
            grasp = point+normal*params['grasp_insert_depth']
            pre = grasp-normal*params['pregrasp_distance']
            lifted = grasp+np.array([0, 0, params['lift_distance']])
            extracted = lifted-normal*params['extract_distance']
            targets = [('pregrasp', pre), ('insert', grasp), ('lift', lifted), ('extract', extracted)]
        elif sequence == 'prepare_grab':
            if self.book_row == 1:
                # The base has already returned to the dock for a top-row grab.
                # Reach from outside and above, then descend to pregrasp.
                grasp = point+normal*params['grasp_insert_depth']
                pre = grasp-normal*params['pregrasp_distance']
                above = pre+np.array([0.0, 0.0, params['top_book_clearance_height']])
                front = above-normal*params['top_book_front_standoff']
                targets = [('top_front_clearance', front), ('top_clearance', above),
                           ('pregrasp', pre)]
            else:
                # At the backed-out base pose, solve for the eventual dock pose.
                targets = [('prealign', point+normal*(params['grasp_insert_depth']-params['grasp_base_advance']))]
        elif sequence == 'finish_grab':
            grasp = point+normal*params['grasp_insert_depth']
            lifted = grasp+np.array([0, 0, params['lift_distance']])
            extracted = lifted-normal*params['extract_distance']
            targets = ([('insert', grasp), ('lift', lifted), ('extract', extracted)]
                       if self.book_row == 1 else [('lift', lifted), ('extract', extracted)])
        elif sequence == 'prepare_place':
            # Perception's bin normal points from the camera/robot toward the
            # bin. Move against it to centre the book in the opening. The base
            # has already reached its calibrated standoff; the torso supplies
            # the final straight down motion after this hover pose is reached.
            release = point-normal*params['bin_inset']+np.array([0, 0, params['bin_release_height']])
            hover = release+np.array([0, 0, params['place_clearance_height']])
            # pass high over the bin first: straight from the carry pose the forearm
            # swung down through the bin's rim
            targets = [('crane_high', hover+np.array([0, 0, 0.20])), ('crane_hover', hover)]
        elif sequence == 'finish_place':
            release = point-normal*params['bin_inset']+np.array([0, 0, params['bin_release_height']])
            targets = [('lower_to_bin', release)]
        else:
            release = point-normal*params['bin_inset']+np.array([0, 0, params['bin_release_height']])
            targets = [('over_bin', release+np.array([0, 0, 0.06])), ('lower_to_bin', release)]
        errors = []
        candidates = []
        # Parallel fingers can grasp with either 0 or 180 degree wrist roll.
        # A bin release may also require the wrist sideways while lowering from
        # the calibrated upright carry pose, so try quarter turns as well.
        rolls = ((math.pi, 0.0, math.pi/2, -math.pi/2)
                 if sequence in ('place', 'prepare_place', 'finish_place')
                 else (math.pi, 0.0))
        for roll in rolls:
            grasp_sequence = sequence in ('grab', 'prepare_grab', 'finish_grab')
            grasp_orientation = facing @ rotation([1, 0, 0], roll)
            if grasp_sequence and self.base_to_grasp is not None:
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
            elif sequence == 'prepare_grab' and self.book_row == 1:
                # Top-row IK needs the torso at its upper travel limit. Solve
                # with that value fixed; the torso is raised in its own phase
                # before any arm waypoint begins.
                q[0] = self.model.upper[0]
            candidate = []
            try:
                for name, xyz in targets:
                    fixed_torso = (sequence == 'prepare_place' or
                                   sequence == 'finish_grab' and self.book_row == 1 or
                                   sequence == 'prepare_grab' and self.book_row == 1)
                    fixed = {0: q[0]} if fixed_torso else None
                    tip_xyz = xyz if offset is None else xyz-orientation @ offset
                    solved = self.model.solve(tip_xyz, orientation, q, fixed=fixed)
                    if candidate and np.max(np.abs(solved[1:]-q[1:])) > 1.4:
                        raise ValueError(f'{name}: IK branch jump; adjust docking pose before retrying.')
                    candidate.append((name, 'motion', solved, self.model.duration(q, solved, 1.0)))
                    q = solved
                # Both parallel-jaw rolls can grasp the same book. Keep the
                # wrist nearer its current orientation by selecting the valid
                # complete path with the least estimated joint travel time,
                # rather than accepting the first roll that happens to solve.
                candidates.append((sum(step[3] for step in candidate), candidate))
            except ValueError as error:
                errors.append(str(error))
        motions = min(candidates, key=lambda item: item[0])[1] if candidates else None
        if motions is None:
            raise ValueError('; '.join(errors))
        if sequence == 'grab':
            # Remember the pregrasp joints: they are the known-collision-free
            # retreat if anything fails while the fingers are inside the shelf.
            self.retreat_target = motions[0][2]
            return [('open_gripper', 'open', None, 1.0), motions[0], motions[1],
                    ('close_gripper', 'close', None, 1.0), motions[2], motions[3],
                    ('verify_book_held', 'verify', None, 0.8)]
        if sequence == 'prepare_grab':
            if self.book_row == 1:
                # Keep the solved pregrasp as the safe retreat if insertion fails.
                self.retreat_target = motions[-1][2]
                return [('open_gripper', 'open', None, 1.0)]+motions
            return [('open_gripper', 'open', None, 1.0), motions[0]]
        if sequence == 'finish_grab':
            if self.book_row == 1:
                if self.retreat_target is None:
                    raise ValueError('Top-row pregrasp joints were not retained; prepare_grab must complete before finish_grab.')
                return [motions[0], ('close_gripper', 'close', None, 1.0),
                        motions[1], motions[2], ('verify_book_held', 'verify', None, 0.8)]
            self.retreat_target = motions[-1][2]
            return [('close_gripper', 'close', None, 1.0), motions[0], motions[1],
                    ('verify_book_held', 'verify', None, 0.8)]
        if sequence == 'prepare_place':
            return motions
        if sequence == 'finish_place':
            return motions+[('release_book', 'open', None, 1.0),
                            ('verify_delivery', 'delivery', None, 3.0)]
        return motions+[('release_book', 'open', None, 1.0),
                        ('verify_delivery', 'delivery', None, 3.0)]

    def plan_carry_raise(self, current):
        """Move the held book to the fixed top-shelf carry pose, without torso motion."""
        target = current.copy()
        target[0] = current[0]
        target[1:] = HIGH_CARRY_RIGHT_ARM
        # the live arm's joint 2 stops at 1.134 rad (the simulator allowed 1.25): stay inside its limits
        target = np.clip(target, self.model.lower, self.model.upper)
        if np.max(np.abs(target-current)) <= 0.005:
            return []
        return [('raise_carried_book', 'motion', target, self.model.duration(current, target, 1.0))]

    def trajectory(self, names, positions, duration):
        msg = JointTrajectory()
        msg.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int(round((duration-int(duration))*1e9))
        if point.time_from_start.nanosec >= 1000000000:
            point.time_from_start.sec += 1
            point.time_from_start.nanosec = 0
        msg.points = [point]
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
            q = self.phase_target
            # Recalculate timing from actual joint feedback; respect slow torso travel.
            start = np.array([self.joints[n] for n in self.model.names])
            duration = max(duration, self.model.duration(start, q))
            self.deadline = self.now()+duration+self.p('motion_timeout_margin')
            clients = (((self.torso, ['torso_lift_joint'], q[:1]),)
                       if self.phase == 'torso' else
                       ((self.torso, ['torso_lift_joint'], q[:1]),
                        (self.arm, ARM_JOINTS, q[1:])))
            self.pending = len(clients)
            token = self.token
            for client, names, values in clients:
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = self.trajectory(names, values, duration)
                future = client.send_goal_async(goal)
                future.add_done_callback(lambda f, t=token: self.goal_response(f, t))
        elif self.phase in ('open', 'close'):
            value = self.p('gripper_open' if self.phase == 'open' else 'gripper_closed')
            if self.phase == 'close':
                # Do not let contact from a previous approach/attempt count as
                # proof of a grasp. The close phase needs fresh bilateral data.
                self.contacts.clear()
            self.gripper_pub.publish(self.trajectory([GRIPPER_JOINT], [value], duration))
            self.deadline = self.now()+self.p('gripper_timeout')
            self.contact_since = None
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
                        self.phase_name = 'confirm_torso_lift'
                        self.deadline = self.now()+3.0
                        return
                    self.get_logger().info(
                        f'Top-row torso lift confirmed at {actual:.3f} m; starting arm approach.')
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
            if self.phase == 'close':
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
                    if self.active == 'prepare_place':
                        hover = next((phase[2] for phase in reversed(self.phases)
                                      if phase[1] == 'motion'), None)
                        if hover is None:
                            raise ValueError('The bin hover plan did not contain a joint waypoint.')
                        self.placement_hover_target = hover.copy()
                    self.next_phase()
                except Exception as error:
                    self.abort(f'No valid grasp/placement plan: {error}')
        elif self.phase == 'torso_feedback':
            actual = self.joints.get('torso_lift_joint')
            target = float(self.phase_target[0])
            if (self.fresh(self.joint_stamp) and actual is not None and
                    abs(actual-target) <= 0.02):
                self.get_logger().info(
                    f'Top-row torso lift confirmed at {actual:.3f} m; starting arm approach.')
                self.next_phase()
        elif self.phase == 'open':
            if self.fresh(self.joint_stamp) and abs(self.joints.get(GRIPPER_JOINT, -1)-self.p('gripper_open')) < 0.006 and self.now()-self.phase_started > 0.8:
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
            if self.ros.topics['bin_contacts'] is None:
                # nothing on the live robot reports the book in the bin: released is done
                if self.now()-self.phase_started > 1.0:
                    self.holding_book = None
                    self.next_phase()
            elif self.bin_contact_stamp is not None and self.bin_contact_stamp >= self.phase_started-1.5:
                self.holding_book = None
                self.next_phase()

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


