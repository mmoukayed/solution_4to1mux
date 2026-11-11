"""Sensor-driven mecanum navigation with bounded searches and visible failures."""
import math
import time
from enum import Enum, auto

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformListener, TransformException
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .ros_topics import configure_ros
from .geometry import (body_velocity, forward_command, pose_xyz, row_for_height,
                       scan_points, transform_matrix, wrap, yaw)


# The live robot's resting tuck (measured on the robot), inside the base footprint.
# The simulators start with both arms held out to the sides instead.
TUCK_POSES = {
    'left': (0.3595, -1.8291, 0.4695, -2.3492, 0.0, -1.2008, 0.0),
    'right': (-0.3595, -1.8291, -0.4695, -2.3492, 0.0, -1.2008, 0.0),
}
DOCK_STOW_POSES = {
    # joint 2 stops at 1.134 rad on the live arm and /erc wants 0.05 of margin (was 1.25)
    # the left arm is never used: it stays in the resting tuck. Stowed out to the side it
    # swept the bin when the base turned beside the bin table.
    'left': TUCK_POSES['left'],
    'right': (0.0, 1.07, 0.0, -1.0, 0.4, 0.0, 0.0),
}


class State(Enum):
    FIND_SHELF = auto()
    SCAN_COLUMNS = auto()
    APPROACH_COLUMN = auto()
    FIND_BOOK = auto()
    ALIGN_SHELF_FOR_DOCK = auto()
    STOW_ARMS_FOR_DOCK = auto()
    ALIGN_BOOK = auto()
    BACK_OUT_FOR_GRASP = auto()
    WAIT_FOR_PREGRASP = auto()
    RETURN_FOR_GRAB = auto()
    REACQUIRE_BOOK = auto()
    BACK_OUT_FOR_VIEW = auto()
    WAIT_FOR_GRAB = auto()
    RETREAT_FROM_SHELF = auto()
    WAIT_FOR_CARRY_RAISE = auto()
    RETURN_TO_START = auto()
    FIND_BIN = auto()
    APPROACH_BIN = auto()
    BACK_OUT_FOR_BIN_TURN = auto()
    WAIT_FOR_PLACE_PREP = auto()
    APPROACH_BIN_FOR_PLACE = auto()
    WAIT_FOR_PLACE = auto()
    DONE = auto()
    FAILED = auto()


class NavigationNode(Node):
    # Class-level default so offline tests on bare instances see the field too.
    blocked_since = None

    def __init__(self):
        super().__init__('navigation_node')
        self.ros = configure_ros(self)
        topics = self.ros.topics
        defaults = {
            'shelf_column_number': 1, 'linear_speed': 0.35, 'strafe_speed': 0.12,
            'rotate_speed': 0.4, 'search_rotate_speed': 0.3,
            'search_direction': 'right',
            # The fixed bin is normally behind the west-facing start pose;
            # begin its complete sweep counterclockwise.
            'bin_search_direction': 'left',
            'phase_timeout': 90.0, 'grab_timeout': 120.0, 'place_timeout': 90.0,
            'reading_max_age': 1.0,
            'book_standoff': 0.68, 'book_lateral_offset': -0.18,
            'bin_standoff': 0.95, 'bin_final_advance_max': 0.65,
            'bin_turn_backoff': 0.25, 'max_bin_turn_recoveries': 2, 'carry_turn_clearance': 0.85,
            'bin_turn_backoff_timeout': 8.0,
            'place_base_advance': 0.35,
            'obstacle_margin': 0.12,
            # Minimum base-centre distance from the shelf-front plane while
            # docking.  This deliberately leaves more than the footprint
            # margin: a depth-biased book point must not put the bumper (or
            # the parked arm) against the shelf before the grasp starts.
            'shelf_keepout_distance': 0.62,
            # Before folding the arms, square the mobile base to the shelf and
            # correct its lateral position at low speed while still staged.
            'shelf_alignment_standoff': 1.35,
            'shelf_alignment_strafe_speed': 0.08,
            'shelf_alignment_timeout': 15.0,
            # Low-row arm links sweep close to the shelf even in the dock
            # stow pose. Keep the base farther back while ALIGN_BOOK runs.
            'lower_book_standoff': 0.92, 'lower_book_keepout_distance': 0.84,
            'top_book_marker_offset': 0.335, 'row_spacing': 0.33,
            'column_half_width': 0.43,
            'arm_stow_timeout': 20.0, 'arm_stow_settle_time': 9.0,
            'scan_half_width': 2.6, 'scan_step': 0.65, 'scan_observe_time': 1.0,
            'docking_speed': 0.12, 'book_search_timeout': 8.0,
            'align_block_timeout': 3.0,
            'grasp_base_retreat': 0.25,
            # Keep a 3 cm buffer above the shelf keepout while returning to
            # the book. Manipulation uses the same 0.28 m compensation.
            'grasp_base_return_advance': 0.28,
            # The arm's calibrated high-carry pose sweeps a large arc.  A
            # bottom-row grasp needs this full straight withdrawal before that
            # pose is commanded, otherwise its elbow can still hit the shelf.
            'carry_raise_retreat': 0.80,
            'max_book_recoveries': 2, 'head_scan_dwell': 1.4,
            'book_view_backoff': 0.35,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        for key in ('linear_speed', 'strafe_speed', 'rotate_speed', 'search_rotate_speed', 'docking_speed',
                    'shelf_alignment_strafe_speed'):
            if not 0 < self.p(key) <= 1.0:
                raise ValueError(f'{key} must be >0 and <=1')
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.pose = self.start = None
        self.odom_stamp = None
        self.observations = {}
        self.scans = {}
        self.result = None
        self.holding_stamp = None
        self.state = State.FIND_SHELF
        self.entered = self.now()
        self.last_head = None
        self.warned = {}
        self.goal = None
        self.book_recoveries = 0
        self.return_detour_side = 0
        self.bin_turn_recoveries = 0
        self.bin_turn_retreat_total = 0.0
        self.last_command = (0.0, 0.0, 0.0)
        self.actual_speed = 0.0
        self.arm_joints = {}
        self.arm_joint_stamp = None
        self.arm_stow_last = None
        self.arms_tucked = False
        self.arm_tuck_last = None
        self.arm_tuck_secs = 2.0
        self.last_gate = 'clear'
        self.blocked_since = None
        self.active_shelf_keepout_distance = self.p('shelf_keepout_distance')
        self.book_dock_minimum = self.p('shelf_keepout_distance')
        self.current_book_row = None
        self.failure_reason = ''
        self.manip_failure = ''
        self.book_diagnostics = 'No book diagnostics received yet.'
        self.cmd_pub = self.create_publisher(Twist, topics['cmd_vel'], 10)
        self.head_pub = self.create_publisher(JointTrajectory, topics['head_command'], 10)
        self.left_arm_pub = self.create_publisher(JointTrajectory, topics['left_arm_command'], 10)
        self.right_arm_pub = self.create_publisher(JointTrajectory, topics['right_arm_command'], 10)
        self.bin_mode_pub = self.create_publisher(Bool, topics['looking_for_bin'], 10)
        self.grab_pub = self.create_publisher(Bool, topics['request_grab'], 10)
        self.prepare_grab_pub = self.create_publisher(Bool, topics['request_prepare_grab'], 10)
        self.finish_grab_pub = self.create_publisher(Bool, topics['request_finish_grab'], 10)
        self.raise_carry_pub = self.create_publisher(Bool, topics['request_raise_carry'], 10)
        self.place_pub = self.create_publisher(Bool, topics['request_place'], 10)
        self.prepare_place_pub = self.create_publisher(Bool, topics['request_prepare_place'], 10)
        self.finish_place_pub = self.create_publisher(Bool, topics['request_finish_place'], 10)
        self.cancel_pub = self.create_publisher(Bool, topics['cancel_manipulation'], 10)
        self.failure_pub = self.create_publisher(String, topics['solution_failure'], QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.state_pub = self.create_publisher(String, topics['solution_state'], QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.book_row_pub = self.create_publisher(
            Int32, topics['book_row'],
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, topics['odom'], self.on_odom, qos_profile_sensor_data)
        self.create_subscription(JointState, topics['joint_states'], self.on_arm_joints, qos_profile_sensor_data)
        for name in ('shelf', 'shelf_unit', 'book', 'bin'):
            self.create_subscription(PoseStamped, topics[f'{name}_pose'], lambda m, n=name: self.on_pose(n, m), 10)
        for name in ('front', 'rear'):
            self.create_subscription(LaserScan, topics[f'scan_{name}'], lambda m, n=name: self.on_scan(n, m), qos_profile_sensor_data)
        self.create_subscription(Bool, topics['holding_book'], self.on_holding, 10)
        self.create_subscription(String, topics['manipulation_failure'], self.on_manip_failure, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, topics['book_diagnostics'], self.on_book_diagnostics, 10)
        if topics['controller_rejections'] is not None:
            self.create_subscription(String, topics['controller_rejections'], self.on_controller_rejection, 10)
        self.create_subscription(Bool, topics['grab_complete'], lambda m: self.on_result(m, State.WAIT_FOR_GRAB), 10)
        self.create_subscription(Bool, topics['grab_prepared'], lambda m: self.on_result(m, State.WAIT_FOR_PREGRASP), 10)
        self.create_subscription(Bool, topics['carry_raised'], lambda m: self.on_result(m, State.WAIT_FOR_CARRY_RAISE), 10)
        self.create_subscription(Bool, topics['place_complete'], lambda m: self.on_result(m, State.WAIT_FOR_PLACE), 10)
        self.create_subscription(Bool, topics['place_prepared'], lambda m: self.on_result(m, State.WAIT_FOR_PLACE_PREP), 10)
        self.wall_sample, self.sim_sample = time.monotonic(), self.now()
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self.tick)
        self.create_timer(5.0, self.wall_status, clock=self.wall_clock)
        self.get_logger().info(f'Navigation BUILD 5.17 (pre-stow shelf-parallel alignment): target column {self.p("shelf_column_number")}, forward cap {self.p("linear_speed")} m/s')
        self.transition(State.FIND_SHELF)
        self.look_for_shelf(0.0)

    def p(self, name):
        return self.get_parameter(name).value

    def now(self):
        return self.get_clock().now().nanoseconds/1e9

    def fresh(self, stamp, age=None):
        return stamp is not None and 0 <= self.now()-stamp <= (self.p('reading_max_age') if age is None else age)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.pose = np.array([p.x, p.y, yaw(msg.pose.pose.orientation)])
        self.actual_speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.odom_stamp = Time.from_msg(msg.header.stamp).nanoseconds/1e9
        if self.start is None:
            self.start = self.pose.copy()
            if self.state == State.FIND_SHELF and self.search_origin is None:
                # Startup already entered FIND_SHELF. Anchor its sweep to the
                # first measured heading, and its timeout to the running clock
                # so a first /clock jump from zero cannot expire the search.
                self.search_origin = self.pose[2]
                self.entered = self.search_pause = self.now()

    def on_arm_joints(self, msg):
        self.arm_joints.update(zip(msg.name, msg.position))
        self.arm_joint_stamp = Time.from_msg(msg.header.stamp).nanoseconds/1e9

    def on_pose(self, name, msg):
        if msg.header.frame_id == self.ros.frames['odom'] and np.isfinite(pose_xyz(msg)).all():
            self.observations[name] = msg

    def on_holding(self, msg):
        if msg.data:
            self.holding_stamp = self.now()

    def on_manip_failure(self, msg):
        self.manip_failure = msg.data

    def on_book_diagnostics(self, msg):
        self.book_diagnostics = msg.data

    def on_controller_rejection(self, msg):
        # Topic trajectory rejections have no action result. Surface the
        # organiser's reason without attributing another client's rejection
        # to our current action or triggering an unrelated state transition.
        self.get_logger().error(
            f'Controller rejection on {self.ros.topics["controller_rejections"]}: {msg.data}')

    def on_result(self, msg, expected):
        if self.state == expected:
            self.result = msg.data

    def on_scan(self, name, msg):
        try:
            tf = self.tf.lookup_transform(self.ros.frames['base'], msg.header.frame_id, Time.from_msg(msg.header.stamp))
        except TransformException:
            return
        values = np.asarray(msg.ranges, dtype=float)
        if not np.any((np.isfinite(values) & (values >= msg.range_min)) | np.isposinf(values)):
            return  # All-NaN scan is a sensor failure, not an empty arena.
        points = scan_points(msg.ranges, msg.angle_min, msg.angle_increment,
                             msg.range_min, msg.range_max, transform_matrix(tf.transform))
        # Ignore returns from inside the robot's own footprint.
        points = points[(np.abs(points[:, 0]) > 0.36) | (np.abs(points[:, 1]) > 0.32)]
        self.scans[name] = (Time.from_msg(msg.header.stamp).nanoseconds/1e9, points)

    def warn(self, key, text):
        if self.now()-self.warned.get(key, -100) > 3:
            self.get_logger().warn(text)
            self.warned[key] = self.now()

    def wall_status(self):
        wall, sim = time.monotonic(), self.now()
        rtf = (sim-self.sim_sample)/max(wall-self.wall_sample, 1e-6)
        self.wall_sample, self.sim_sample = wall, sim
        cmd_speed = math.hypot(*self.last_command[:2])
        book = self.observations.get('book')
        book_age = 'none' if book is None else f'{self.now()-Time.from_msg(book.header.stamp).nanoseconds/1e9:.1f}s'
        remaining = 'none' if self.goal is None or self.pose is None else f'{np.linalg.norm(self.goal[:2]-self.pose[:2]):.2f}m'
        heading_error = ('none' if self.goal is None or self.pose is None else
                         f'{math.degrees(wrap(self.goal[2]-self.pose[2])):+.1f}deg')
        self.get_logger().info(
            f'State={self.state.name}; sim/wall={rtf:.2f}; command={cmd_speed:.2f}m/s; '
            f'turn={self.last_command[2]:+.2f}rad/s; odom={self.actual_speed:.2f}m/s; '
            f'remaining={remaining}; heading_error={heading_error}; book_age={book_age}; gate={self.last_gate}')
        if self.state == State.FAILED:
            self.get_logger().error(f'Failure retained on {self.ros.topics["solution_failure"]}: {self.failure_reason}')
        if rtf < 0.5:
            self.get_logger().warn('Gazebo is paused or slower than real time. Increasing /cmd_vel cannot speed up physics; check GPU/rendering.')

    def stop(self):
        self.last_command = (0.0, 0.0, 0.0)
        self.cmd_pub.publish(Twist())

    def wall_reference(self):
        """(point, into-shelf normal) of the shelf-front plane, or None.

        Marker plates are flush with the shelf front and book spines sit on it.
        The shelf is static in odom, so a stale observation is still a valid
        plane. Only the confirmed book/column poses are used: the shelf_unit
        centroid's heading can be a mere line-of-sight bearing, and a tilted
        wall through a distant centroid can cut across a valid docking path.
        The dock goal and the wall must come from the same plane family so the
        goal always stays reachable on the robot side of the wall.
        """
        # A bin sits on a table whose top edge can be invisible to the base
        # LiDAR. During placement use its observed front plane as the virtual
        # wall; otherwise use the confirmed shelf marker.
        if self.state in (State.FIND_BIN, State.APPROACH_BIN,
                          State.BACK_OUT_FOR_BIN_TURN,
                          State.WAIT_FOR_PLACE_PREP, State.APPROACH_BIN_FOR_PLACE,
                          State.WAIT_FOR_PLACE):
            msg = self.observations.get('bin')
            if msg is not None:
                heading = yaw(msg.pose.orientation)
                return pose_xyz(msg)[:2], np.array([math.cos(heading), math.sin(heading)])
        # The confirmed overhead marker is flush with the shelf front and is
        # a more stable wall reference than a colour/depth book point.  Keep
        # the book as a fallback for the close-up case where the marker has
        # left the camera view; this does not change perception itself.
        for name in ('shelf', 'book'):
            msg = self.observations.get(name)
            if msg is not None:
                heading = yaw(msg.pose.orientation)
                return pose_xyz(msg)[:2], np.array([math.cos(heading), math.sin(heading)])
        return None

    def shelf_wall_points(self, reference):
        """Synthetic base-frame obstacle points along the shelf-front plane.

        The ERC shelf is nearly invisible to the base LiDAR: at its 0.13 m scan
        plane an aisle presents only the top edge of the bottom board, so the
        clearance gate reports 'clear' while the bumper is about to hit the
        shelf. These virtual returns make the existing clearance/turn gates see
        the shelf like a wall.
        """
        if reference is None or self.pose is None:
            return np.empty((0, 2))
        p, normal = reference
        tangent = np.array([-normal[1], normal[0]])
        world = p[None, :]+np.linspace(-2.7, 2.7, 55)[:, None]*tangent[None, :]+0.03*normal[None, :]
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        rel = world-self.pose[:2]
        return np.c_[c*rel[:, 0]+s*rel[:, 1], -s*rel[:, 0]+c*rel[:, 1]]

    def clamped_dock_goal(self, goal):
        """Never accept a docking goal closer to the shelf plane than the standoff allows.

        A colour blob seen deep inside a shelf gap can otherwise produce a dock
        goal at or beyond the shelf front, which the LiDAR cannot veto.
        """
        reference = self.wall_reference()
        if reference is None:
            return goal
        p, normal = reference
        minimum = max(self.p('book_standoff')-0.10,
                      self.p('shelf_keepout_distance'),
                      getattr(self, 'active_shelf_keepout_distance',
                              self.p('shelf_keepout_distance')),
                      getattr(self, 'book_dock_minimum',
                              self.p('shelf_keepout_distance')))
        depth = (goal[:2]-p) @ normal   # negative = robot side of the plane
        if depth > -minimum:
            goal = goal.copy()
            goal[:2] -= (depth+minimum)*normal
            self.warn('deep_goal', f'Dock goal was {depth:+.2f} m of the shelf plane; clamped to {minimum:.2f} m in front. Book depth estimate is suspect.')
        return goal

    def dock_constraints_for_book(self, book):
        """Increase low-row clearance and estimate a safe base centre in-column."""
        shelf = self.observations.get('shelf')
        row = None
        if shelf is not None:
            row = row_for_height(
                shelf.pose.position.z, book.pose.position.z,
                self.p('top_book_marker_offset'), self.p('row_spacing'))
        standoff = self.p('book_standoff')
        keepout = self.p('shelf_keepout_distance')
        if row is not None and row >= 3:
            standoff = max(standoff, self.p('lower_book_standoff'))
            keepout = max(keepout, self.p('lower_book_keepout_distance'))
        self.current_book_row = row
        self.active_shelf_keepout_distance = keepout
        self.book_dock_minimum = max(standoff-0.10, keepout)
        if row is not None:
            self.book_row_pub.publish(Int32(data=row))
        return standoff

    def constrain_goal_to_column(self, goal):
        """Keep the robot footprint within the target aisle while the arm reaches sideways."""
        shelf = self.observations.get('shelf')
        if shelf is None:
            return goal
        heading = yaw(shelf.pose.orientation)
        tangent = np.array([-math.sin(heading), math.cos(heading)])
        relative = goal[:2]-pose_xyz(shelf)[:2]
        lateral = float(relative @ tangent)
        robot_half_width = (0.36*abs(math.sin(goal[2]-heading))
                            + 0.32*abs(math.cos(goal[2]-heading)))
        safe_lateral = max(0.02, self.p('column_half_width')-robot_half_width-0.03)
        bounded = float(np.clip(lateral, -safe_lateral, safe_lateral))
        if abs(bounded-lateral) > 1e-3:
            goal = goal.copy()
            goal[:2] += (bounded-lateral)*tangent
            self.warn('column_lateral_clamp',
                      f'Base alignment shifted {abs(bounded-lateral):.2f} m toward the column centre to clear its divider.')
        return goal

    def move(self, vx=0.0, vy=0.0, omega=0.0, allow_turn_recovery=True):
        # Require fresh front AND rear scans for holonomic translation/rotation.
        if len(self.scans) < 2 or not all(self.fresh(v[0]) for v in self.scans.values()):
            self.stop()
            self.last_gate = 'stale_or_missing_lidar'
            self.warn('scan', f'Waiting for fresh front/rear LiDAR in {self.ros.frames["base"]}; no blind creep.')
            return
        points = np.concatenate([v[1] for v in self.scans.values()])
        scan_count = len(points)
        reference = self.wall_reference()
        self.last_gate = 'clear'
        hard_shelf_blocked = False
        # A sampled virtual wall protects the footprint, but a position-only
        # docking goal can still overshoot it by one controller tick.  Guard
        # the shelf normal directly as well: once at the conservative dock
        # distance, no command may have a component into the shelf.
        if reference is not None and self.pose is not None:
            wall_point, normal = reference
            c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
            world_velocity = np.array([c*vx-s*vy, s*vx+c*vy])
            distance = float(-(self.pose[:2]-wall_point) @ normal)
            keepout = getattr(self, 'active_shelf_keepout_distance',
                              self.p('shelf_keepout_distance'))
            if world_velocity @ normal > 1e-6 and distance <= keepout:
                vx = vy = 0.0
                hard_shelf_blocked = True
                self.last_gate = f'shelf_keepout_hard_{distance:.2f}m'
                if self.blocked_since is None:
                    self.blocked_since = self.now()
                self.warn('hard_shelf_gate',
                          f'Shelf keepout reached at {distance:.2f} m (minimum {keepout:.2f} m); suppressing inward docking motion.')
        wall = self.shelf_wall_points(reference)
        if len(wall):
            points = np.concatenate([points, wall]) if scan_count else wall
        speed = math.hypot(vx, vy)
        if speed > 0 and len(points):
            direction = np.array([vx, vy])/speed
            side = np.array([-direction[1], direction[0]])
            extent = abs(direction[0])*0.36+abs(direction[1])*0.32
            width = abs(direction[1])*0.36+abs(direction[0])*0.32+0.03
            along, across = points @ direction, points @ side
            in_path = (along > 0) & (np.abs(across) < width)
            if in_path.any():
                indices = np.where(in_path)[0]
                nearest = indices[int(np.argmin(along[in_path]))]
                source = 'shelf_keepout' if nearest >= scan_count else 'obstacle_clearance'
                clearance = float(along[in_path].min()-extent)
                if clearance < self.p('obstacle_margin'):
                    self.stop()
                    self.last_gate = f'{source}_{clearance:.2f}m'
                    if self.blocked_since is None:
                        self.blocked_since = self.now()
                    self.warn('blocked', f'Path blocked ({source}): {clearance:.2f} m from footprint; state {self.state.name}. This is not successful docking.')
                    return
                # Limit absolute speed by clearance. Multiplying an already
                # tapered docking command caused an unnecessary second slowdown.
                clearance_cap = 1.5*max(0.0, clearance-self.p('obstacle_margin'))
                scale = min(1.0, clearance_cap/speed)
                vx, vy = vx*scale, vy*scale
                if scale < 1.0:
                    self.last_gate = f'clearance_speed_limit_{clearance_cap:.2f}m/s'
        # While the book is carried the arm sweeps well outside the base, and the bin
        # table's top overhangs between legs the LiDAR plane cannot see: turn only with
        # every LiDAR return outside the swept radius.
        carrying = self.state in (State.RETREAT_FROM_SHELF, State.WAIT_FOR_CARRY_RAISE, State.RETURN_TO_START,
                                  State.FIND_BIN, State.APPROACH_BIN, State.BACK_OUT_FOR_BIN_TURN)
        turn_radius = self.p('carry_turn_clearance') if carrying else 0.49
        if abs(omega) > 0 and len(points) and np.linalg.norm(points, axis=1).min() < turn_radius:
            omega = 0.0
            self.last_gate = 'turning_envelope_blocked'
            self.warn('turn', 'Obstacle inside turning envelope; rotation stopped.')
            if allow_turn_recovery and vx == 0.0 and vy == 0.0:
                # Rotation needed but blocked, and no translation commanded either:
                # slide slowly straight away from the nearest return, if that path is clear.
                nearest = points[int(np.argmin(np.linalg.norm(points, axis=1)))]
                away = -nearest/max(float(np.linalg.norm(nearest)), 1e-6)
                side = np.array([-away[1], away[0]])
                along, across = points @ away, points @ side
                ahead = along[(along > 0) & (np.abs(across) < 0.35)]
                clearance = float(ahead.min())-0.36 if len(ahead) else 10.0
                if clearance > self.p('obstacle_margin')+0.05:
                    vx, vy = 0.08*float(away[0]), 0.08*float(away[1])
                    self.last_gate = 'turn_blocked_moving_away'
                    self.warn('turn_back', 'Rotation blocked; sliding away from the nearest obstacle to regain turning room.')
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(omega)
        if not hard_shelf_blocked and any(abs(v) > 1e-6 for v in (vx, vy, omega)):
            self.blocked_since = None
        self.last_command = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_pub.publish(msg)

    def transition(self, state):
        self.stop()
        self.blocked_since = None
        self.state, self.entered = state, self.now()
        if state == State.FIND_SHELF:
            self.active_shelf_keepout_distance = self.p('shelf_keepout_distance')
            self.book_dock_minimum = self.p('shelf_keepout_distance')
            self.current_book_row = None
            self.book_row_pub.publish(Int32(data=-1))
        if state == State.STOW_ARMS_FOR_DOCK:
            self.arm_stow_last = None
        if state == State.RETURN_TO_START:
            self.return_detour_side = 0
        self.state_pub.publish(String(data=state.name))
        self.get_logger().info(f'-> {state.name}')
        if state in (State.FIND_SHELF, State.FIND_BIN):
            self.search_origin = self.pose[2] if self.pose is not None else None
            search_direction = self.p('bin_search_direction') if state == State.FIND_BIN else self.p('search_direction')
            direction = -1 if search_direction == 'right' else 1
            self.search_angles = [direction*math.radians(a) for a in (70, -70, 155, -155)]
            self.search_index = 0
            self.search_pause = self.now()+(0.5 if state == State.FIND_BIN else 0.0)
        if state == State.SCAN_COLUMNS:
            shelf_unit = self.observations['shelf_unit']
            self.scan_heading = yaw(shelf_unit.pose.orientation)
            shelf_normal = np.array([math.cos(self.scan_heading), math.sin(self.scan_heading)])
            # Center the sweep on the measured marker cluster, not wherever the
            # robot happened to be when the first non-target sign was spotted.
            # The marker normal points into the shelf, so stand 1.8 m in front.
            self.scan_origin = pose_xyz(shelf_unit)[:2]-1.8*shelf_normal
            self.scan_direction = -1 if self.p('search_direction') == 'right' else 1
            self.scan_offset = 0.0
            self.scan_pause_until = None
            self.scan_observe_started = None
            # First stop in front of the marker cluster and observe; only then
            # begin the lateral column sweep.
            self.scan_goal = np.r_[self.scan_origin, self.scan_heading]

    def advance_scan_waypoint(self):
        limit = self.p('scan_half_width')
        if self.scan_direction*self.scan_offset >= limit-1e-6:
            self.scan_direction *= -1
        self.scan_offset = float(np.clip(self.scan_offset+self.scan_direction*self.p('scan_step'), -limit, limit))
        left = np.array([-math.sin(self.scan_heading), math.cos(self.scan_heading)])
        # Fixed waypoint. Do not shift it sideways on every timer tick.
        self.scan_goal = np.r_[self.scan_origin+self.scan_offset*left, self.scan_heading]
        self.scan_pause_until = None
        self.scan_observe_started = None

    def scan_for_book(self, elapsed):
        self.stop()
        self.last_gate = 'waiting_for_fresh_book_pose'
        cached = self.observations.get('book')
        if cached is not None and elapsed < self.p('head_scan_dwell'):
            self.look(cached)
        else:
            # Hold each head pose long enough to settle and collect three detections.
            tilts = (0.28, 0.05, -0.25, -0.50, -0.78)
            offset = self.p('head_scan_dwell') if cached is not None else 0.0
            index = int(max(0.0, elapsed-offset)/self.p('head_scan_dwell')) % len(tilts)
            self.look(tilt=tilts[index])
        self.warn('book', f'Book reacquisition: {self.book_diagnostics}')

    def tuck_arms_at_start(self):
        """Both arms into the resting tuck, slowly; True once there (or settled)."""
        names = {side: [f'arm_{side}_{i}_joint' for i in range(1, 8)] for side in ('left', 'right')}
        if not self.fresh(self.arm_joint_stamp) or any(
                name not in self.arm_joints for side in names.values() for name in side):
            self.warn('arm_feedback', 'Waiting for fresh left/right arm joint feedback before tucking.')
            return False
        if all(abs(self.arm_joints[name]-target) < 0.08
               for side, joint_names in names.items()
               for name, target in zip(joint_names, TUCK_POSES[side])):
            self.get_logger().info('Both arms in the resting tuck.')
            return True
        if self.arm_tuck_last is not None:
            if self.now()-self.arm_tuck_last < self.arm_tuck_secs+1.0:
                return False
            self.warn('arm_tuck_settle', 'Arm tuck settled without exact joint feedback; continuing.')
            return True
        self.arm_tuck_secs = 2.0
        for side, publisher in (('left', self.left_arm_pub), ('right', self.right_arm_pub)):
            move = max(abs(self.arm_joints[n]-q) for n, q in zip(names[side], TUCK_POSES[side]))
            secs = max(2.0, move/0.25)
            self.arm_tuck_secs = max(self.arm_tuck_secs, secs)
            msg = JointTrajectory()
            msg.joint_names = names[side]
            point = JointTrajectoryPoint()
            point.positions = list(TUCK_POSES[side])
            point.time_from_start.sec = int(secs)
            point.time_from_start.nanosec = int((secs % 1)*1e9)
            msg.points = [point]
            publisher.publish(msg)
        self.arm_tuck_last = self.now()
        self.get_logger().info(f'Tucking both arms before moving the base ({self.arm_tuck_secs:.1f} s).')
        return False

    def stow_arms_for_dock(self):
        """Fold both parked arms before the base enters the shelf keepout.

        These are the calibrated, full seven-joint trajectories rather than
        partial-joint edits.  The controller receives exactly the same shape
        of command used for live calibration, and docking starts only after
        feedback confirms the complete pose.
        """
        names = {
            'left': [f'arm_left_{i}_joint' for i in range(1, 8)],
            'right': [f'arm_right_{i}_joint' for i in range(1, 8)],
        }
        if not self.fresh(self.arm_joint_stamp) or any(
                name not in self.arm_joints for side in names.values() for name in side):
            self.warn('arm_feedback', 'Waiting for fresh left/right arm joint feedback before shelf docking.')
            return False
        if all(abs(self.arm_joints[name]-target) < 0.08
               for side, joint_names in names.items()
               for name, target in zip(joint_names, DOCK_STOW_POSES[side])):
            return True
        if self.arm_stow_last is not None:
            # Gazebo can visibly complete a trajectory while its constrained
            # joint feedback remains a little off the requested values.  The
            # topic command already has a two-second duration; after one full
            # settle window and still-fresh feedback, continue rather than
            # re-publishing forever and blocking at the shelf.
            if self.now()-self.arm_stow_last < self.p('arm_stow_settle_time'):
                return False
            self.warn('arm_stow_settle',
                      'Arm stow trajectory settled without exact joint feedback; continuing with the calibrated parked pose.')
            return True
        for side, publisher in (('left', self.left_arm_pub), ('right', self.right_arm_pub)):
            msg = JointTrajectory()
            msg.joint_names = names[side]
            point = JointTrajectoryPoint()
            point.positions = list(DOCK_STOW_POSES[side])
            # /erc rejects segments over 0.60 rad/s (joints 1-2) / 0.75: time it at 0.25 rad/s
            move = max(abs(self.arm_joints[n]-q) for n, q in zip(names[side], DOCK_STOW_POSES[side]))
            secs = max(2.0, move/0.25)
            point.time_from_start.sec = int(secs)
            point.time_from_start.nanosec = int((secs % 1)*1e9)
            msg.points = [point]
            publisher.publish(msg)
        self.arm_stow_last = self.now()
        self.get_logger().info('Tucking the left arm and stowing the right before ALIGN_BOOK.')
        return False

    def recover_book_view(self):
        # Only before a grasp: never move away after a partially executed arm motion.
        if self.book_recoveries >= self.p('max_book_recoveries'):
            self.fail(f'Book could not be reacquired after a bounded head scan and back-off. {self.book_diagnostics}')
            return
        self.book_recoveries += 1
        heading = self.pose[2]
        retreat = self.p('book_view_backoff')
        self.goal = np.r_[self.pose[:2]-retreat*np.array([math.cos(heading), math.sin(heading)]), heading]
        self.get_logger().warn(
            f'Book scan found no stable target; backing straight out {retreat:.2f} m '
            f'for a wider view ({self.book_recoveries}/{self.p("max_book_recoveries")}).')
        self.transition(State.BACK_OUT_FOR_VIEW)

    def fail(self, reason):
        self.failure_reason = reason
        self.failure_pub.publish(String(data=reason))
        self.get_logger().error(reason)
        self.cancel_pub.publish(Bool(data=True))
        self.transition(State.FAILED)

    def observation(self, name, maximum_age=2.0):
        msg = self.observations.get(name)
        if msg is None or not self.fresh(Time.from_msg(msg.header.stamp).nanoseconds/1e9, maximum_age):
            return None
        return msg

    def look(self, target=None, tilt=0.15):
        if target is not None:
            try:
                tf = self.tf.lookup_transform(self.ros.frames['head'], self.ros.frames['odom'], Time())
                p = (transform_matrix(tf.transform) @ np.r_[pose_xyz(target), 1])[:3]
                tilt = math.atan2(p[2]-0.25, max(math.hypot(p[0], p[1]), 0.1))
            except TransformException:
                pass
        tilt = float(np.clip(tilt, -0.9, 0.30))
        if self.last_head is not None and abs(self.last_head-tilt) < 0.04:
            return
        if self.head_pub.get_subscription_count() == 0:
            return
        self.last_head = tilt
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0, tilt]
        point.time_from_start.sec = 1
        msg.points = [point]
        self.head_pub.publish(msg)

    def look_for_shelf(self, elapsed):
        # The shelf numbers sit above the book rows. Sweep the camera pitch as
        # the base scans so acquisition still works when the initial head pose
        # puts the overhead markers outside the camera's vertical field of view.
        tilts = (0.28, 0.12, -0.08)
        tilt = tilts[int(max(0.0, elapsed) / self.p('head_scan_dwell')) % len(tilts)]
        self.look(tilt=tilt)

    def search(self):
        if self.now() < self.search_pause:
            self.stop()
            return
        target = self.search_origin+self.search_angles[self.search_index]
        error = wrap(target-self.pose[2])
        if abs(error) < 0.06:
            self.stop()
            self.search_index = (self.search_index+1) % len(self.search_angles)
            self.search_pause = self.now()+0.6
        else:
            self.move(omega=float(np.clip(1.5*error, -self.p('search_rotate_speed'), self.p('search_rotate_speed'))))

    def drive_to(self, goal, tolerance=0.04, docking=False, keep_heading=False):
        dx, dy = goal[:2]-self.pose[:2]
        if not docking and not keep_heading:
            vx, vy, omega, arrived = forward_command(
                dx, dy, self.pose[2], goal[2], self.p('linear_speed'),
                self.p('rotate_speed'), tolerance)
            if arrived:
                self.stop()
                return True
            self.move(vx, vy, omega)
            return False
        error = wrap(goal[2]-self.pose[2])
        if math.hypot(dx, dy) < tolerance and abs(error) < 0.04:
            self.stop()
            return True
        omega = float(np.clip(1.6*error, -self.p('rotate_speed'), self.p('rotate_speed')))
        vx, vy = body_velocity(dx, dy, self.pose[2], self.p('docking_speed'), gain=1.8)
        if keep_heading:
            # A reverse retreat must not first turn the carried book into the shelf.
            vy = 0.0
        elif abs(error) > 0.20:
            vx = vy = 0.0
        else:
            vy = float(np.clip(vy, -self.p('strafe_speed'), self.p('strafe_speed')))
        self.move(vx, vy, omega)
        return False

    def return_path_clearance(self, direction, horizon):
        """Estimate LiDAR clearance along a short base-frame translation."""
        points = np.concatenate([scan[1] for scan in self.scans.values()])
        wall = self.shelf_wall_points(self.wall_reference())
        if len(wall):
            points = np.concatenate([points, wall])
        direction = np.asarray(direction, dtype=float)
        direction /= max(np.linalg.norm(direction), 1e-9)
        side = np.array([-direction[1], direction[0]])
        extent = abs(direction[0])*0.36+abs(direction[1])*0.32
        width = abs(direction[1])*0.36+abs(direction[0])*0.32+0.03
        along, across = points @ direction, points @ side
        in_path = (along > 0) & (along < horizon+extent) & (np.abs(across) < width)
        if not in_path.any():
            return float('inf')
        return float(along[in_path].min()-extent)

    def drive_return_to_start(self):
        """Take a direct reverse/strafe route, detouring around LiDAR obstacles."""
        delta = self.start[:2]-self.pose[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.10:
            # No turn back to the start heading: the start sits beside the bin table and the
            # bin search turns anyway (through the carried-arm clearance check).
            self.stop()
            return True
        if len(self.scans) < 2 or not all(self.fresh(scan[0]) for scan in self.scans.values()):
            self.stop()
            self.last_gate = 'stale_or_missing_lidar'
            self.warn('return_lidar', 'Waiting for fresh front/rear LiDAR before returning to the start.')
            return False

        desired_world = delta/distance
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        horizon = min(1.0, distance)
        offsets = (0.0, math.radians(20), -math.radians(20), math.radians(35),
                   -math.radians(35), math.radians(50), -math.radians(50),
                   math.radians(65), -math.radians(65), math.radians(80),
                   -math.radians(80))
        options = []
        for offset in offsets:
            angle = math.atan2(desired_world[1], desired_world[0])+offset
            world_direction = np.array([math.cos(angle), math.sin(angle)])
            body_direction = np.array([c*world_direction[0]+s*world_direction[1],
                                       -s*world_direction[0]+c*world_direction[1]])
            progress = float(world_direction @ desired_world)
            if progress < 0.15:
                continue
            clearance = self.return_path_clearance(body_direction, horizon)
            if clearance < self.p('obstacle_margin'):
                continue
            score = abs(offset)+0.025/max(clearance, 0.05)
            options.append((score, offset, body_direction))

        direct_clear = any(offset == 0.0 for _, offset, _ in options)
        if direct_clear:
            self.return_detour_side = 0
            options = [option for option in options if option[1] == 0.0]
        else:
            left = [option for option in options if option[1] > 0]
            right = [option for option in options if option[1] < 0]
            if self.return_detour_side > 0 and left:
                options = left
            elif self.return_detour_side < 0 and right:
                options = right
            elif left or right:
                best_left = min((option[0] for option in left), default=float('inf'))
                best_right = min((option[0] for option in right), default=float('inf'))
                self.return_detour_side = 1 if best_left < best_right else -1
                options = left if self.return_detour_side > 0 else right

        if not options:
            self.stop()
            self.last_gate = 'return_path_blocked'
            self.warn('return_blocked', 'No short obstacle-free return direction is available; holding position.')
            return False

        _, offset, direction = min(options, key=lambda option: option[0])
        speed_cap = self.p('linear_speed')
        if abs(direction[1]) > 1e-6:
            speed_cap = min(speed_cap, self.p('strafe_speed')/abs(direction[1]))
        speed = min(speed_cap, 1.5*distance)
        self.move(float(direction[0]*speed), float(direction[1]*speed), 0.0)
        if offset and self.last_gate == 'clear':
            self.last_gate = f'return_detour_{"left" if offset > 0 else "right"}'
        return False

    def rotate_to(self, heading):
        """Align heading without changing the already-valid base position."""
        error = wrap(heading-self.pose[2])
        if abs(error) < 0.04:
            self.stop()
            return True
        omega = float(np.clip(1.6*error, -self.p('rotate_speed'), self.p('rotate_speed')))
        # Keep rotation itself translation-free. APPROACH_BIN handles a blocked
        # turn with a separate, bounded retreat and recomputes the final advance.
        self.move(omega=omega, allow_turn_recovery=False)
        return False

    def back_out_for_bin_turn(self):
        """Make room for a blocked bin turn without reversing indefinitely."""
        if self.bin_turn_recoveries >= self.p('max_bin_turn_recoveries'):
            self.fail('Bin rotation is still blocked after the bounded clearance retreats; inspect front/rear LiDAR around the base.')
            return
        # Check the entire planned reverse path before beginning the recovery.
        # move() continues checking both sensors and obstacles on every tick.
        if len(self.scans) < 2 or not all(self.fresh(v[0]) for v in self.scans.values()):
            self.stop()
            self.last_gate = 'stale_or_missing_lidar'
            return
        points = np.concatenate([v[1] for v in self.scans.values()])
        wall = self.shelf_wall_points(self.wall_reference())
        points = np.concatenate([points, wall])
        behind = points[(points[:, 0] < 0) & (np.abs(points[:, 1]) < 0.35)]
        clearance = float(-behind[:, 0].max())-0.36 if len(behind) else float('inf')
        retreat = self.p('bin_turn_backoff')
        if clearance < retreat+self.p('obstacle_margin'):
            self.fail(f'Bin rotation blocked and rear clearance {clearance:.2f} m is too small for a {retreat:.2f} m retreat.')
            return
        self.bin_turn_recoveries += 1
        self.bin_approach_heading = self.goal[2]
        self.bin_retreat_origin = self.pose[:2].copy()
        heading = self.pose[2]
        self.goal = np.r_[self.pose[:2]-retreat*np.array([math.cos(heading), math.sin(heading)]), heading]
        self.get_logger().warn(
            f'Bin turn blocked; backing straight out {retreat:.2f} m before retrying '
            f'({self.bin_turn_recoveries}/{self.p("max_bin_turn_recoveries")}).')
        self.transition(State.BACK_OUT_FOR_BIN_TURN)

    def drive_bin_turn_backoff(self):
        # Translate along the saved heading without any simultaneous rotation.
        heading = np.array([math.cos(self.goal[2]), math.sin(self.goal[2])])
        remaining = float((self.pose[:2]-self.goal[:2]) @ heading)
        if remaining <= 0.02:
            self.stop()
            return True
        self.move(vx=-min(0.08, 1.8*remaining), allow_turn_recovery=False)
        return False

    def drive_forward_to(self, goal, tolerance=0.03):
        """Advance along the current heading only; never back away from a bin."""
        heading = np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
        remaining = float((goal[:2]-self.pose[:2]) @ heading)
        if remaining <= tolerance:
            self.stop()
            return True
        self.move(vx=min(self.p('docking_speed'), 1.8*remaining))
        return False

    def shelf_limited_return_arrived(self, goal):
        """Accept a close return stopped solely by the hard shelf boundary."""
        return (self.last_gate.startswith('shelf_keepout') and
                math.hypot(*(goal[:2]-self.pose[:2])) <= 0.06 and
                abs(wrap(goal[2]-self.pose[2])) < 0.04)

    @staticmethod
    def target_goal(msg, standoff, lateral=0.0):
        heading = yaw(msg.pose.orientation)
        normal = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-normal[1], normal[0]])
        p = pose_xyz(msg)[:2]-normal*standoff-left*lateral
        return np.r_[p, heading]

    def tick(self):
        elapsed = self.now()-self.entered
        self.bin_mode_pub.publish(Bool(data=self.state in (
            State.RETURN_TO_START, State.FIND_BIN, State.APPROACH_BIN,
            State.BACK_OUT_FOR_BIN_TURN,
            State.WAIT_FOR_PLACE_PREP, State.APPROACH_BIN_FOR_PLACE, State.WAIT_FOR_PLACE)))
        if self.state in (State.DONE, State.FAILED):
            self.stop()
            return
        if not self.arms_tucked:
            # Before the base moves at all: arms held out sweep whatever is beside the start
            # (the bin table) on the first search turn.
            self.stop()
            self.last_gate = 'tucking_arms'
            self.arms_tucked = self.tuck_arms_at_start()
            return
        if self.state == State.FIND_SHELF:
            self.look_for_shelf(elapsed)
        if not self.fresh(self.odom_stamp):
            self.stop()
            self.last_gate = 'stale_or_missing_odometry'
            self.warn('odom', 'Waiting for fresh odometry.')
            return
        if self.state in (State.RETREAT_FROM_SHELF, State.WAIT_FOR_CARRY_RAISE, State.RETURN_TO_START,
                          State.FIND_BIN, State.APPROACH_BIN, State.BACK_OUT_FOR_BIN_TURN, State.WAIT_FOR_PLACE_PREP,
                          State.APPROACH_BIN_FOR_PLACE) and not self.fresh(self.holding_stamp, 1.5):
            self.fail('Book contact lost while carrying, or manipulation feedback stopped.')
            return
        if self.state in (State.FIND_BOOK, State.REACQUIRE_BOOK) and elapsed > self.p('book_search_timeout'):
            self.recover_book_view()
            return
        limit = (self.p('grab_timeout') if self.state == State.WAIT_FOR_GRAB else
                 self.p('place_timeout') if self.state == State.WAIT_FOR_PLACE else
                 self.p('shelf_alignment_timeout') if self.state == State.ALIGN_SHELF_FOR_DOCK else
                 self.p('phase_timeout'))
        if elapsed > limit:
            self.fail(f'{self.state.name} timed out after {elapsed:.1f} simulation seconds; inspect preceding warnings.')
            return
        if self.state in (State.FIND_SHELF, State.SCAN_COLUMNS):
            observing = self.state == State.SCAN_COLUMNS and self.scan_pause_until is not None
            if self.state == State.SCAN_COLUMNS:
                self.look_for_shelf(self.now()-self.scan_observe_started if observing else elapsed)
            target = self.observation('shelf')
            if target is not None:
                self.goal = self.target_goal(target, 1.15)
                self.transition(State.APPROACH_COLUMN)
            elif self.state == State.FIND_SHELF:
                if self.observation('shelf_unit') is not None:
                    self.transition(State.SCAN_COLUMNS)
                else:
                    self.search()
            else:
                self.goal = self.scan_goal
                if self.scan_pause_until is not None:
                    self.stop()
                    self.last_gate = f'observing_columns_{max(0.0, self.scan_pause_until-self.now()):.1f}s'
                    if self.now() >= self.scan_pause_until:
                        self.advance_scan_waypoint()
                elif self.drive_to(self.scan_goal):
                    # One full pitch sweep, including head settling and three
                    # stable observations at each pitch, with the base stopped.
                    self.scan_observe_started = self.now()
                    self.scan_pause_until = self.now()+max(
                        self.p('scan_observe_time'), 3*self.p('head_scan_dwell'))
                    self.last_gate = 'observing_columns'
        elif self.state == State.APPROACH_COLUMN:
            if self.drive_to(self.goal):
                self.transition(State.FIND_BOOK)
        elif self.state in (State.FIND_BOOK, State.REACQUIRE_BOOK):
            book = self.observation('book')
            if book is not None:
                standoff = self.dock_constraints_for_book(book)
                self.goal = self.target_goal(
                    book, standoff, self.p('book_lateral_offset'))
                shelf = self.observation('shelf')
                if shelf is not None:
                    self.goal[2] = yaw(shelf.pose.orientation)
                self.goal = self.constrain_goal_to_column(self.goal)
                self.goal = self.clamped_dock_goal(self.goal)
                self.get_logger().info(
                    f'Book dock: row={self.current_book_row}, standoff={standoff:.2f} m, '
                    f'shelf_keepout={self.active_shelf_keepout_distance:.2f} m.')
                self.get_logger().info('Aligning the base parallel to the shelf before moving the arms.')
                self.transition(State.ALIGN_SHELF_FOR_DOCK)
            else:
                self.scan_for_book(elapsed)
        elif self.state == State.ALIGN_SHELF_FOR_DOCK:
            if (self.blocked_since is not None and
                    self.now()-self.blocked_since > self.p('align_block_timeout')):
                self.fail(f'Shelf-parallel alignment blocked ({self.last_gate}); refusing to stow arms against the shelf.')
                return
            reference = self.wall_reference()
            if reference is not None:
                wall_point, normal = reference
                shelf_distance = float(-(self.pose[:2]-wall_point) @ normal)
                safe_distance = max(self.p('shelf_alignment_standoff'),
                                    self.active_shelf_keepout_distance+0.25)
                if shelf_distance < safe_distance-0.025:
                    speed = min(self.p('shelf_alignment_strafe_speed'),
                                1.5*(safe_distance-shelf_distance))
                    world_velocity = -normal*speed
                    c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
                    vx = c*world_velocity[0]+s*world_velocity[1]
                    vy = -s*world_velocity[0]+c*world_velocity[1]
                    self.move(vx=vx, vy=vy, allow_turn_recovery=False)
                    return
            heading_error = wrap(self.goal[2]-self.pose[2])
            if abs(heading_error) > 0.025:
                omega = float(np.clip(1.6*heading_error,
                                      -self.p('rotate_speed'), self.p('rotate_speed')))
                self.move(omega=omega, allow_turn_recovery=False)
                return
            tangent = np.array([-math.sin(self.goal[2]), math.cos(self.goal[2])])
            lateral_error = float((self.goal[:2]-self.pose[:2]) @ tangent)
            if abs(lateral_error) <= 0.025:
                self.transition(State.STOW_ARMS_FOR_DOCK)
                return
            speed = min(self.p('shelf_alignment_strafe_speed'), 1.5*abs(lateral_error))
            self.move(vy=math.copysign(speed, lateral_error), allow_turn_recovery=False)
        elif self.state == State.STOW_ARMS_FOR_DOCK:
            self.stop()
            if self.stow_arms_for_dock():
                self.transition(State.ALIGN_BOOK)
            elif elapsed > self.p('arm_stow_timeout'):
                self.fail(f'Arm stow timed out before shelf docking; inspect {self.ros.topics["joint_states"]} and both arm trajectory controllers.')
        elif self.state == State.BACK_OUT_FOR_VIEW:
            if self.drive_to(self.goal, keep_heading=True):
                self.transition(State.FIND_BOOK)
        elif self.state == State.ALIGN_BOOK:
            cached = self.observations.get('book')
            self.look(cached)
            # Keep the goal consistent with the current wall plane, and never
            # sit out the 90 s phase timeout against a blocked docking path —
            # back out and reacquire instead.
            self.goal = self.constrain_goal_to_column(self.goal)
            self.goal = self.clamped_dock_goal(self.goal)
            arrived = self.drive_to(self.goal, tolerance=0.025, docking=True)
            if arrived or self.shelf_limited_return_arrived(self.goal):
                if not arrived:
                    self.get_logger().warn('ALIGN_BOOK reached the shelf keepout boundary; continuing with the compensated staged grasp.')
                if self.observation('book') is None:
                    self.transition(State.REACQUIRE_BOOK)
                    return
                heading = self.pose[2]
                retreat = self.p('grasp_base_retreat')
                self.goal = np.r_[self.pose[:2]-retreat*np.array([math.cos(heading), math.sin(heading)]), heading]
                self.transition(State.BACK_OUT_FOR_GRASP)
            elif self.blocked_since is not None and self.now()-self.blocked_since > self.p('align_block_timeout'):
                self.get_logger().warn(f'Docking path blocked for {self.p("align_block_timeout"):.0f} s (gate={self.last_gate}); backing out to reacquire the book.')
                self.recover_book_view()
        elif self.state == State.BACK_OUT_FOR_GRASP:
            if self.drive_to(self.goal, keep_heading=True):
                if self.current_book_row == 1:
                    # Keep the arms folded for the base return. The top-row
                    # reach is planned from above only after the base is still.
                    heading = self.pose[2]
                    advance = self.p('grasp_base_return_advance')
                    self.goal = np.r_[self.pose[:2]+advance*np.array([math.cos(heading), math.sin(heading)]), heading]
                    self.transition(State.RETURN_FOR_GRAB)
                else:
                    self.result = None
                    self.transition(State.WAIT_FOR_PREGRASP)
                    self.prepare_grab_pub.publish(Bool(data=True))
        elif self.state == State.WAIT_FOR_PREGRASP:
            self.stop()
            if self.result is False:
                self.fail(f'Pregrasp failed: {self.manip_failure or self.ros.topics["manipulation_failure"]}')
            elif self.result is True:
                self.result = None
                if self.current_book_row == 1:
                    self.transition(State.WAIT_FOR_GRAB)
                    self.finish_grab_pub.publish(Bool(data=True))
                else:
                    heading = self.pose[2]
                    advance = self.p('grasp_base_return_advance')
                    self.goal = np.r_[self.pose[:2]+advance*np.array([math.cos(heading), math.sin(heading)]), heading]
                    self.transition(State.RETURN_FOR_GRAB)
        elif self.state == State.RETURN_FOR_GRAB:
            arrived = self.drive_to(self.goal, keep_heading=True)
            if arrived or self.shelf_limited_return_arrived(self.goal):
                if not arrived:
                    self.get_logger().warn('Return-to-grab stopped by the shelf keepout at the safe boundary; starting the compensated arm close.')
                self.result = None
                if self.current_book_row == 1:
                    self.transition(State.WAIT_FOR_PREGRASP)
                    self.prepare_grab_pub.publish(Bool(data=True))
                else:
                    self.transition(State.WAIT_FOR_GRAB)
                    self.finish_grab_pub.publish(Bool(data=True))
        elif self.state == State.WAIT_FOR_GRAB:
            self.stop()
            if self.result is False:
                if 'redock' in self.manip_failure and self.book_recoveries < self.p('max_book_recoveries'):
                    # Manipulation refused because the base is jammed too close to
                    # the shelf. Back straight out and dock again instead of failing.
                    self.book_recoveries += 1
                    heading = self.pose[2]
                    self.goal = np.r_[self.pose[:2]-0.35*np.array([math.cos(heading), math.sin(heading)]), heading]
                    self.get_logger().warn('Grab refused: base too close to the shelf. Backing out 0.35 m to re-dock.')
                    self.result = None
                    self.transition(State.BACK_OUT_FOR_VIEW)
                    return
                self.fail(f'Grasp failed: {self.manip_failure or self.ros.topics["manipulation_failure"]}')
            elif self.result is True:
                # The upright carry configuration moves its elbow through a
                # broad arc. Back straight out first, especially for a book
                # taken from the bottom shelf, before changing arm shape.
                heading = self.pose[2]
                retreat = self.p('carry_raise_retreat')
                self.goal = np.r_[self.pose[:2]-retreat*np.array([math.cos(heading), math.sin(heading)]), heading]
                self.get_logger().info(f'Grasp verified; withdrawing {retreat:.2f} m before the high carry pose.')
                self.transition(State.RETREAT_FROM_SHELF)
        elif self.state == State.RETREAT_FROM_SHELF:
            if self.drive_to(self.goal, keep_heading=True):
                self.result = None
                self.transition(State.WAIT_FOR_CARRY_RAISE)
                self.raise_carry_pub.publish(Bool(data=True))
        elif self.state == State.WAIT_FOR_CARRY_RAISE:
            self.stop()
            if self.result is False:
                self.fail(f'Carry raise failed: {self.manip_failure or "see manipulation log"}')
            elif self.result is True:
                self.transition(State.RETURN_TO_START)
        elif self.state == State.RETURN_TO_START:
            self.look(tilt=-0.1)
            if self.drive_return_to_start():
                self.transition(State.FIND_BIN)
        elif self.state == State.FIND_BIN:
            self.look(tilt=-0.1)
            target = self.observation('bin')
            if target is not None:
                self.goal = self.target_goal(target, self.p('bin_standoff'), self.p('book_lateral_offset'))
                self.bin_turn_recoveries = 0
                self.bin_turn_retreat_total = 0.0
                self.transition(State.APPROACH_BIN)
            else:
                self.search()
        elif self.state == State.APPROACH_BIN:
            self.look(self.observations.get('bin'))
            # Face the bin at the return pose, making a bounded retreat only
            # when LiDAR prevents rotation. Recompute distance after recovery.
            if self.rotate_to(self.goal[2]):
                # The high carry pose cannot reach a bin 1.53 m away (the
                # observed failure was a 0.425 m IK position error). Advance
                # only toward the bin until the calibrated 0.95 m arm standoff
                # is reached; unlike target-goal docking this cannot reverse.
                bin_pose = self.observations.get('bin')
                heading = np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
                distance = (float((pose_xyz(bin_pose)[:2]-self.pose[:2]) @ heading)
                            if bin_pose is not None else self.p('bin_standoff'))
                advance = float(np.clip(distance-self.p('bin_standoff'), 0.0,
                                         self.p('bin_final_advance_max')+self.bin_turn_retreat_total))
                self.goal = np.r_[self.pose[:2]+advance*heading, self.pose[2]]
                self.transition(State.APPROACH_BIN_FOR_PLACE)
            elif self.last_gate == 'turning_envelope_blocked':
                self.back_out_for_bin_turn()
        elif self.state == State.BACK_OUT_FOR_BIN_TURN:
            self.look(self.observations.get('bin'))
            if self.drive_bin_turn_backoff():
                self.bin_turn_retreat_total += float(np.linalg.norm(self.pose[:2]-self.bin_retreat_origin))
                self.goal = np.r_[self.pose[:2], self.bin_approach_heading]
                self.transition(State.APPROACH_BIN)
            elif elapsed > self.p('bin_turn_backoff_timeout'):
                self.fail(f'Bin-turn retreat could not finish within {self.p("bin_turn_backoff_timeout"):.0f} s; gate={self.last_gate}.')
        elif self.state == State.APPROACH_BIN_FOR_PLACE:
            if self.drive_forward_to(self.goal):
                self.result = None
                self.transition(State.WAIT_FOR_PLACE_PREP)
                self.prepare_place_pub.publish(Bool(data=True))
        elif self.state == State.WAIT_FOR_PLACE_PREP:
            self.stop()
            if self.result is False:
                self.fail(f'Bin hover failed: {self.manip_failure or self.ros.topics["manipulation_failure"]}')
            elif self.result is True:
                self.result = None
                self.transition(State.WAIT_FOR_PLACE)
                self.finish_place_pub.publish(Bool(data=True))
        elif self.state == State.WAIT_FOR_PLACE:
            self.stop()
            if self.result is False:
                self.fail(f'Placement failed: {self.manip_failure or "manipulation returned false without a reason; inspect its preceding log."}')
            elif self.result is True:
                self.transition(State.DONE)


def main():
    rclpy.init()
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.cancel_pub.publish(Bool(data=True))
        node.destroy_node()
        rclpy.shutdown()
