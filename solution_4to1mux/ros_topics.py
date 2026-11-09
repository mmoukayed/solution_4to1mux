"""Edit ROS names here; select a profile with robot_profile:=simulation|real.

TOPICS, ACTIONS and FRAMES contain the current simulator names. PROFILES can
override any entry by its key. Shared names (including inter-node messages)
are inherited automatically, keeping publishers and subscribers in agreement.
The real profile follows the ERC 2026 Live Phase TIAGo Pro Interface Guide,
sections 2-3. Optional interfaces unavailable in a profile are set to None.
"""
from dataclasses import dataclass


DEFAULT_PROFILE = 'simulation'

TOPICS = {
    # Robot input and control (message types stay defined in the nodes).
    'cmd_vel': '/cmd_vel',
    'odom': '/odom',
    'joint_states': '/joint_states',
    'robot_description': '/robot_description',
    'scan_front': '/scan_front_raw',
    'scan_rear': '/scan_rear_raw',
    'scan_merged': None,  # Live robot: filtered 360-degree sensor_msgs/LaserScan.
    'base_imu': '/base_imu',  # Not available on the live robot.
    'camera_rgb': '/head_front_camera/head_front_camera/color/image_raw',
    # camera_depth is the image consumed by perception. On the live robot it
    # must be aligned to colour to use camera_info's colour intrinsics.
    'camera_depth': '/head_front_camera/head_front_camera/depth/image_rect_raw',
    # The simulator bridge publishes CameraInfo under the same doubled
    # namespace as its colour image topic.
    'camera_info': '/head_front_camera/head_front_camera/color/camera_info',
    'camera_depth_raw': '/head_front_camera/depth/image_rect_raw',
    'camera_depth_aligned': None,
    'camera_points': '/head_front_camera/depth/color/points',
    'camera_rgb_compressed': None,
    'camera_depth_compressed': None,
    'camera_aligned_depth_compressed': None,
    'head_command': '/head_controller/joint_trajectory',
    'left_arm_command': '/arm_left_controller/joint_trajectory',
    'right_arm_command': '/arm_right_controller/joint_trajectory',
    'torso_command': '/torso_controller/joint_trajectory',
    'left_gripper_command': '/gripper_left_controller/joint_trajectory',
    'right_gripper_command': '/gripper_right_controller/joint_trajectory',
    'controller_rejections': None,  # Live checking-layer diagnostics: std_msgs/String.
    # These currently require ros_gz_interfaces/msg/Contacts messages.
    'contacts': '/contacts',
    'bin_contacts': '/bin_contacts',

    # Perception and navigation exchange.
    'looking_for_bin': '/perception/looking_for_bin',
    'perception_ready': '/perception/ready',
    'shelf_pose': '/perception/shelf_pose',
    'shelf_unit_pose': '/perception/shelf_unit_pose',
    'book_pose': '/perception/book_pose',
    'bin_pose': '/perception/bin_pose',
    'book_diagnostics': '/perception/book_diagnostics',
    'shelf_column_identification': '/erc/shelf_column_identification',
    'shelf_row_identification': '/erc/shelf_row_identification',
    'solution_state': '/solution/state',
    'solution_failure': '/solution/failure',
    'book_row': '/solution/book_row',

    # Navigation and manipulation exchange.
    'manipulation_ready': '/manipulation/ready',
    'holding_book': '/manipulation/holding_book',
    'manipulation_failure': '/manipulation/failure',
    'grab_complete': '/manipulation/grab_complete',
    'grab_prepared': '/manipulation/grab_prepared',
    'place_complete': '/manipulation/place_complete',
    'place_prepared': '/manipulation/place_prepared',
    'carry_raised': '/manipulation/carry_raised',
    'request_grab': '/manipulation/request_grab',
    'request_prepare_grab': '/manipulation/request_prepare_grab',
    'request_finish_grab': '/manipulation/request_finish_grab',
    'request_place': '/manipulation/request_place',
    'request_prepare_place': '/manipulation/request_prepare_place',
    'request_finish_place': '/manipulation/request_finish_place',
    'request_raise_carry': '/manipulation/request_raise_carry',
    'cancel_manipulation': '/manipulation/cancel',

    # Used by ROS/TF internally; solution.launch.py applies their remappings.
    'tf': '/tf',
    'tf_static': '/tf_static',
    'clock': '/clock',
}

ACTIONS = {
    'left_arm_trajectory': '/arm_left_controller/follow_joint_trajectory',
    'right_arm_trajectory': '/arm_right_controller/follow_joint_trajectory',
    'torso_trajectory': '/torso_controller/follow_joint_trajectory',
    'head_trajectory': '/head_controller/follow_joint_trajectory',
    # The existing simulation uses topic-only gripper clamps.
    'left_gripper_trajectory': None,
    'right_gripper_trajectory': None,
}

FRAMES = {
    'odom': 'odom',
    'base': 'base_footprint',
    'head': 'head_1_link',
    'depth_camera': 'head_front_camera_depth_optical_frame',
    'colour_camera': 'head_front_camera_color_optical_frame',
    'scan_merged': 'virtual_base_laser_link',
    'scan_front': 'base_front_laser_link',
    'scan_rear': 'base_rear_laser_link',
    'arm_tip': 'gripper_right_base_link',
}

PROFILES = {
    'simulation': {
        'use_sim_time': True,
        'topics': {},
        'actions': {},
        'frames': {},
    },
    'real': {
        'use_sim_time': False,
        # Live interface guide, sections 2-3. Joint commands must use the
        # organiser's /erc checking layer. Keep these prefixes intact.
        'topics': {
            'odom': '/mobile_base_controller/odom',
            'scan_merged': '/scan',
            'base_imu': None,
            'camera_rgb': '/head_front_camera/color/image_raw',
            'camera_info': '/head_front_camera/color/camera_info',
            'camera_depth': '/head_front_camera/aligned_depth_to_color/image_raw',
            'camera_depth_raw': '/head_front_camera/depth/image_rect_raw',
            'camera_depth_aligned': '/head_front_camera/aligned_depth_to_color/image_raw',
            'camera_points': '/head_front_camera/depth/color/points',
            # Expand the guide's image-transport suffixes against each image topic.
            'camera_rgb_compressed': '/head_front_camera/color/image_raw/compressed',
            'camera_depth_compressed': '/head_front_camera/depth/image_rect_raw/compressedDepth',
            'camera_aligned_depth_compressed': '/head_front_camera/aligned_depth_to_color/image_raw/compressedDepth',
            'head_command': '/erc/head_controller/joint_trajectory',
            'left_arm_command': '/erc/arm_left_controller/joint_trajectory',
            'right_arm_command': '/erc/arm_right_controller/joint_trajectory',
            'torso_command': '/erc/torso_controller/joint_trajectory',
            'left_gripper_command': '/erc/gripper_left_controller/joint_trajectory',
            'right_gripper_command': '/erc/gripper_right_controller/joint_trajectory',
            'controller_rejections': '/erc/rejections',
            'contacts': None,
            'bin_contacts': None,
        },
        'actions': {
            'left_arm_trajectory': '/erc/arm_left_controller/follow_joint_trajectory',
            'right_arm_trajectory': '/erc/arm_right_controller/follow_joint_trajectory',
            'torso_trajectory': '/erc/torso_controller/follow_joint_trajectory',
            'head_trajectory': '/erc/head_controller/follow_joint_trajectory',
            'left_gripper_trajectory': '/erc/gripper_left_controller/follow_joint_trajectory',
            'right_gripper_trajectory': '/erc/gripper_right_controller/follow_joint_trajectory',
        },
        'frames': {
            # The supplied TF dump includes this optical frame. Aligned depth
            # is projected into the colour camera, unlike the raw depth image.
            'depth_camera': 'head_front_camera_color_optical_frame',
        },
    },
}

# Keep these source names fixed: the libraries themselves subscribe to them.
_BUILTIN_TOPICS = {'tf': '/tf', 'tf_static': '/tf_static', 'clock': '/clock'}

_OPTIONAL_INTERFACES = {
    'topics': {'scan_merged', 'base_imu', 'camera_depth_aligned',
               'camera_rgb_compressed', 'camera_depth_compressed',
               'camera_aligned_depth_compressed', 'controller_rejections',
               'contacts', 'bin_contacts'},
    'actions': {'left_gripper_trajectory', 'right_gripper_trajectory'},
    'frames': set(),
}


@dataclass
class RosConfig:
    profile: str
    topics: dict
    actions: dict
    frames: dict
    use_sim_time: bool

    def builtin_remappings(self):
        """Remap library-owned subscriptions to this profile's topic names."""
        return [(source, self.topics[key]) for key, source in _BUILTIN_TOPICS.items()]


def get_ros_config(profile=DEFAULT_PROFILE):
    """Return a fresh copy of a profile; reject misspelled keys before startup."""
    if profile not in PROFILES:
        raise ValueError(f'Unknown robot_profile {profile!r}; choose from {", ".join(PROFILES)}.')
    settings = PROFILES[profile]
    unknown = set(settings)-{'topics', 'actions', 'frames', 'use_sim_time'}
    if unknown:
        raise ValueError(f'{profile}: unknown settings {sorted(unknown)}')
    if not isinstance(settings.get('use_sim_time'), bool):
        raise ValueError(f'{profile}.use_sim_time must be True or False')
    resolved = {}
    for section, defaults in (('topics', TOPICS), ('actions', ACTIONS), ('frames', FRAMES)):
        overrides = settings.get(section, {})
        unknown = set(overrides)-set(defaults)
        if unknown:
            raise ValueError(f'{profile}.{section}: unknown keys {sorted(unknown)}')
        values = {**defaults, **overrides}
        for key, value in values.items():
            if value is None and key in _OPTIONAL_INTERFACES[section]:
                continue
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ValueError(f'{profile}.{section}.{key} must be a nonempty ROS name without spaces')
        resolved[section] = values
    return RosConfig(profile=profile, use_sim_time=settings['use_sim_time'], **resolved)


def configure_ros(node):
    """Select the same name profile in each ROS node, including ros2 run."""
    profile = node.declare_parameter('robot_profile', DEFAULT_PROFILE).value
    config = get_ros_config(profile)
    node.get_logger().info(f'ROS interface profile: {profile} (ros_topics.py)')
    return config
