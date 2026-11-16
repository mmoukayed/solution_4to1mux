"""ERC entry point; all nodes share one ROS interface profile and clock."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from solution_4to1mux.ros_topics import DEFAULT_PROFILE, get_ros_config


def generate_launch_description():
    arguments = {
        'shelf_column_number': ('1', int),
        'book_colour': ('red', str),
        'book_text': ('', str),
        'linear_speed': ('0.85', float),
        'strafe_speed': ('0.25', float),
        'rotate_speed': ('0.85', float),
        'search_rotate_speed': ('0.45', float),
        'search_direction': ('right', str),
        'book_standoff': ('0.68', float),
        'book_lateral_offset': ('-0.18', float),
        'grab_timeout': ('120.0', float),
        'place_timeout': ('90.0', float),
        'phase_timeout': ('90.0', float),
        'perception_rate': ('8.0', float),
        'marker_min_height': ('1.75', float),
        'marker_max_height': ('2.55', float),
        'top_book_marker_offset': ('0.335', float),
        'row_spacing': ('0.33', float),
        'column_half_width': ('0.43', float),
        'lower_book_standoff': ('0.92', float),
        'lower_book_keepout_distance': ('0.84', float),
        'shelf_alignment_standoff': ('1.35', float),
        'shelf_alignment_strafe_speed': ('0.12', float),
        'shelf_alignment_timeout': ('15.0', float),
        'arm_stow_timeout': ('30.0', float),
        'arm_stow_settle_time': ('0.3', float),
        'arm_stow_joint_tolerance': ('0.06', float),
        'arm_stow_front_limit': ('0.40', float),
        'arm_stow_half_width': ('0.55', float),
        'arm_stow_link_radius': ('0.06', float),
        'arm_stow_clearance': ('0.08', float),
        'grasp_correction_limit': ('0.12', float),
        'grasp_position_tolerance': ('0.012', float),
        'grasp_orientation_tolerance': ('0.10', float),
        'top_book_clearance_height': ('0.16', float),
        'top_book_front_standoff': ('0.10', float),
        'top_book_arm_clearance': ('0.06', float),
        'docking_speed': ('0.25', float),
        'book_search_timeout': ('8.0', float),
        'max_book_recoveries': ('2', int),
        'book_view_backoff': ('0.35', float),
    }
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=kind)
              for name, (_, kind) in arguments.items()}
    declarations = [
        DeclareLaunchArgument('robot_profile', default_value=DEFAULT_PROFILE,
                              description='Profile name from solution_4to1mux/ros_topics.py'),
        DeclareLaunchArgument('use_sim_time', default_value='auto',
                              choices=['auto', 'true', 'false'],
                              description='auto uses the clock setting in the selected robot profile'),
    ]+[DeclareLaunchArgument(name, default_value=value)
       for name, (value, _) in arguments.items()]

    def launch_nodes(context):
        profile = LaunchConfiguration('robot_profile').perform(context)
        config = get_ros_config(profile)
        clock = LaunchConfiguration('use_sim_time').perform(context)
        if clock not in ('auto', 'true', 'false'):
            raise ValueError('use_sim_time must be auto, true, or false')
        shared = {'robot_profile': profile,
                  'use_sim_time': config.use_sim_time if clock == 'auto' else clock == 'true'}

        def node(executable, names):
            return Node(package='solution_4to1mux', executable=executable,
                        parameters=[{**shared, **{name: params[name] for name in names}}],
                        remappings=config.builtin_remappings(),
                        output='screen', emulate_tty=True)

        return [
            node('perception_node', ['shelf_column_number', 'book_colour', 'book_text', 'perception_rate',
                                     'marker_min_height', 'marker_max_height', 'top_book_marker_offset',
                                     'row_spacing', 'column_half_width']),
            node('navigation_node', ['shelf_column_number', 'linear_speed', 'strafe_speed',
                                    'rotate_speed', 'search_rotate_speed', 'search_direction',
                                    'book_standoff', 'book_lateral_offset', 'grab_timeout',
                                    'place_timeout', 'phase_timeout', 'docking_speed',
                                    'book_search_timeout', 'max_book_recoveries',
                                    'book_view_backoff', 'top_book_marker_offset', 'row_spacing',
                                    'column_half_width', 'lower_book_standoff',
                                    'lower_book_keepout_distance',
                                    'shelf_alignment_standoff',
                                    'shelf_alignment_strafe_speed',
                                    'shelf_alignment_timeout',
                                    'arm_stow_timeout', 'arm_stow_settle_time',
                                    'arm_stow_joint_tolerance', 'arm_stow_front_limit',
                                    'arm_stow_half_width', 'arm_stow_link_radius',
                                    'arm_stow_clearance']),
            node('manipulation_node', ['grasp_correction_limit', 'grasp_position_tolerance',
                                       'grasp_orientation_tolerance', 'top_book_clearance_height',
                                       'top_book_front_standoff', 'top_book_arm_clearance']),
        ]

    return LaunchDescription(declarations+[OpaqueFunction(function=launch_nodes)])
