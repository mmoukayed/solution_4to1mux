"""ERC entry point; all nodes share one ROS interface profile and clock."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from solution_4to1mux.ros_topics import DEFAULT_PROFILE, get_ros_config
from solution_4to1mux.alignment import MOTION_DEFAULTS


def generate_launch_description():
    arguments = {
        'shelf_column_number': ('1', int),
        'book_colour': ('', str),
        'book_color': ('', str),
        'book_text': ('', str),
        'search_direction': ('right', str),
        'book_standoff': ('0.68', float),
        'book_lateral_offset': ('-0.18', float),
        'grab_timeout': ('120.0', float),
        'place_timeout': ('90.0', float),
        'phase_timeout': ('240.0', float),
        'perception_rate': ('8.0', float),
        'marker_min_height': ('auto', str),
        'marker_max_height': ('2.55', float),
        'top_book_marker_offset': ('0.335', float),
        'row_spacing': ('0.33', float),
        'column_half_width': ('0.43', float),
        'lower_book_standoff': ('0.92', float),
        'lower_book_keepout_distance': ('0.84', float),
        'shelf_alignment_standoff': ('1.35', float),
        'shelf_alignment_timeout': ('40.0', float),
        'book_search_timeout': ('20.0', float),
        'max_book_recoveries': ('2', int),
        'book_view_backoff': ('0.35', float),
        # how far either side of the shelf numbers the column sweep goes: 1.3 keeps it on a small platform,
        # a long shelf on open floor needs ~2.6 to reach its end columns
        'scan_half_width': ('1.3', float),
        'grasp_base_return_advance': ('0.28', float),
        'motion_speed_scale': ('0.60', float),
        'sensor_loss_timeout': ('3.0', float),
        **{name: (str(value), float) for name, value in MOTION_DEFAULTS.items()},
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
        colour = LaunchConfiguration('book_colour').perform(context).strip().lower()
        alias = LaunchConfiguration('book_color').perform(context).strip().lower()
        if colour and alias and colour != alias:
            raise ValueError('Conflicting book_colour and book_color; specify one colour.')
        colour = colour or alias or 'red'
        if colour not in ('red', 'green', 'blue', 'yellow'):
            raise ValueError(f'Invalid book colour: {colour}')
        minimum = LaunchConfiguration('marker_min_height').perform(context).strip()
        minimum = (1.60 if profile == 'real' else 1.75) if minimum == 'auto' else float(minimum)
        shared = {'robot_profile': profile,
                  'use_sim_time': config.use_sim_time if clock == 'auto' else clock == 'true'}

        def node(executable, names, extra=None):
            return Node(package='solution_4to1mux', executable=executable,
                        parameters=[{**shared, **{name: params[name] for name in names}, **(extra or {})}],
                        remappings=config.builtin_remappings(),
                        output='screen', emulate_tty=True)

        return [
            node('perception_node', ['shelf_column_number', 'book_colour', 'book_text', 'perception_rate',
                                     'marker_min_height', 'marker_max_height', 'top_book_marker_offset',
                                     'row_spacing', 'column_half_width'], {'book_colour': colour, 'marker_min_height': minimum}),
            node('navigation_node', ['shelf_column_number', 'scan_half_width', 'linear_speed', 'strafe_speed',
                                    'rotate_speed', 'search_rotate_speed', 'search_direction',
                                    'book_standoff', 'book_lateral_offset', 'grab_timeout',
                                    'place_timeout', 'phase_timeout', 'docking_speed',
                                    'book_search_timeout', 'max_book_recoveries',
                                    'book_view_backoff', 'top_book_marker_offset', 'row_spacing',
                                    'column_half_width', 'lower_book_standoff',
                                    'lower_book_keepout_distance',
                                    'shelf_alignment_standoff',
                                    'shelf_alignment_strafe_speed',
                                    'shelf_alignment_timeout', 'grasp_base_return_advance', 'sensor_loss_timeout', *MOTION_DEFAULTS]),
            node('manipulation_node', ['motion_speed_scale'],
                 {'grasp_base_advance': params['grasp_base_return_advance']}),
        ]

    return LaunchDescription(declarations+[OpaqueFunction(function=launch_nodes)])
