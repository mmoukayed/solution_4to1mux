from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    shelf_column_number = LaunchConfiguration('shelf_column_number')
    book_colour = LaunchConfiguration('book_colour')

    return LaunchDescription([
        DeclareLaunchArgument('shelf_column_number', default_value='1'),
        DeclareLaunchArgument('book_colour', default_value='red'),

        Node(
            package='solution_4to1mux',
            executable='perception_node',
            parameters=[{'shelf_column_number': shelf_column_number,
                         'book_colour': book_colour}],
            output='screen',
        ),
        Node(
            package='solution_4to1mux',
            executable='navigation_node',
            output='screen',
        ),
        Node(
            package='solution_4to1mux',
            executable='manipulation_node',
            output='screen',
        ),
        Node(
            package='solution_4to1mux',
            executable='solution_coordinator',
            output='screen',
        ),
    ])