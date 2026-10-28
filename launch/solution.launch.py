"""ERC entry point; all nodes share simulation time and typed arguments."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = {
        'shelf_column_number': ('1', int),
        'book_colour': ('red', str),
        'use_sim_time': ('true', bool),
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
        'docking_speed': ('0.25', float),
        'book_search_timeout': ('24.0', float),
        'max_book_recoveries': ('2', int),
    }
    params = {name: ParameterValue(LaunchConfiguration(name), value_type=kind)
              for name, (_, kind) in arguments.items()}
    declarations = [DeclareLaunchArgument(name, default_value=value)
                    for name, (value, _) in arguments.items()]
    def node(executable, names):
        return Node(package='solution_4to1mux', executable=executable,
                    parameters=[{name: params[name] for name in ['use_sim_time']+names}],
                    output='screen', emulate_tty=True)
    return LaunchDescription(declarations+[
        node('perception_node', ['shelf_column_number', 'book_colour', 'perception_rate',
                                 'marker_min_height', 'marker_max_height', 'top_book_marker_offset']),
        node('navigation_node', ['shelf_column_number', 'linear_speed', 'strafe_speed',
                                'rotate_speed', 'search_rotate_speed', 'search_direction',
                                'book_standoff', 'book_lateral_offset', 'grab_timeout',
                                'place_timeout', 'phase_timeout', 'docking_speed',
                                'book_search_timeout', 'max_book_recoveries']),
        node('manipulation_node', []),
    ])
