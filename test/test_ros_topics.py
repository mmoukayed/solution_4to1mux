"""Profile/launch checks without ROS, NumPy or Gazebo installed.

Run: python3 -m unittest discover -s test -p test_ros_topics.py -v
"""
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from solution_4to1mux.ros_topics import DEFAULT_PROFILE, PROFILES, configure_ros, get_ros_config


ROOT = Path(__file__).resolve().parents[1]


def launch_nodes(**overrides):
    """Execute the real launch factory using recording ROS launch stand-ins."""
    modules = {name: ModuleType(name) for name in (
        'launch', 'launch.actions', 'launch.substitutions',
        'launch_ros', 'launch_ros.actions', 'launch_ros.parameter_descriptions')}

    class LaunchConfiguration:
        def __init__(self, name):
            self.name = name

        def perform(self, context):
            return context[self.name]

    modules['launch'].LaunchDescription = lambda entities: entities
    modules['launch.actions'].DeclareLaunchArgument = lambda name, **kw: NS(name=name, **kw)
    modules['launch.actions'].OpaqueFunction = lambda function: NS(function=function)
    modules['launch.substitutions'].LaunchConfiguration = LaunchConfiguration
    modules['launch_ros.actions'].Node = lambda **kw: NS(**kw)
    modules['launch_ros.parameter_descriptions'].ParameterValue = lambda value, **kw: NS(value=value, **kw)
    with patch.dict(sys.modules, modules):
        factory = runpy.run_path(str(ROOT/'launch/solution.launch.py'))['generate_launch_description']
        entities = factory()
        context = {entity.name: entity.default_value for entity in entities if hasattr(entity, 'name')}
        context.update(overrides)
        return entities[-1].function(context)


class RosTopicProfiles(unittest.TestCase):
    def test_simulation_keeps_existing_camera_and_controller_endpoints(self):
        config = get_ros_config('simulation')
        self.assertTrue(config.use_sim_time)
        self.assertEqual(config.topics['camera_depth'], '/head_front_camera/depth/image_rect_raw')
        self.assertEqual(config.topics['camera_info'], '/head_front_camera/color/camera_info')
        self.assertEqual(config.topics['right_arm_command'], '/arm_right_controller/joint_trajectory')
        self.assertEqual(config.topics['right_gripper_command'], '/erc/gripper_right_controller/joint_trajectory')
        self.assertEqual(config.actions['right_arm_trajectory'], '/erc/arm_right_controller/follow_joint_trajectory')
        self.assertEqual(config.frames['arm_tip'], 'gripper_right_base_link')

    def test_real_profile_inherits_inter_node_topics(self):
        sim, real = get_ros_config('simulation'), get_ros_config('real')
        self.assertFalse(real.use_sim_time)
        self.assertEqual(real.actions['torso_trajectory'], '/erc/torso_controller/follow_joint_trajectory')
        for key in ('book_pose', 'request_grab', 'grab_complete', 'holding_book', 'book_row'):
            self.assertEqual(sim.topics[key], real.topics[key])

    def test_real_joint_commands_all_use_live_checking_layer(self):
        config = get_ros_config('real')
        for key, controller in (
                ('left_arm', 'arm_left'), ('right_arm', 'arm_right'),
                ('torso', 'torso'), ('head', 'head'),
                ('left_gripper', 'gripper_left'), ('right_gripper', 'gripper_right')):
            with self.subTest(controller=controller):
                prefix = f'/erc/{controller}_controller/'
                self.assertEqual(config.topics[key+'_command'], prefix+'joint_trajectory')
                self.assertEqual(config.actions[key+'_trajectory'], prefix+'follow_joint_trajectory')

    def test_real_camera_depth_matches_colour_projection(self):
        config = get_ros_config('real')
        self.assertEqual(config.topics['camera_rgb'], '/head_front_camera/color/image_raw')
        self.assertEqual(config.topics['camera_info'], '/head_front_camera/color/camera_info')
        self.assertEqual(config.topics['camera_depth'], '/head_front_camera/aligned_depth_to_color/image_raw')
        self.assertEqual(config.topics['camera_depth'], config.topics['camera_depth_aligned'])
        self.assertEqual(config.frames['depth_camera'], 'head_front_camera_color_optical_frame')
        self.assertEqual(config.topics['camera_depth_raw'], '/head_front_camera/depth/image_rect_raw')
        self.assertEqual(config.topics['camera_points'], '/head_front_camera/depth/color/points')
        self.assertEqual(config.topics['camera_rgb_compressed'], config.topics['camera_rgb']+'/compressed')
        self.assertEqual(config.topics['camera_depth_compressed'], config.topics['camera_depth_raw']+'/compressedDepth')
        self.assertEqual(config.topics['camera_aligned_depth_compressed'], config.topics['camera_depth']+'/compressedDepth')

    def test_real_mobility_and_feedback_match_live_guide(self):
        config = get_ros_config('real')
        for key, expected in {
                'cmd_vel': '/cmd_vel', 'odom': '/mobile_base_controller/odom',
                'scan_merged': '/scan', 'scan_front': '/scan_front_raw', 'scan_rear': '/scan_rear_raw',
                'joint_states': '/joint_states', 'robot_description': '/robot_description',
                'controller_rejections': '/erc/rejections', 'tf': '/tf', 'tf_static': '/tf_static'}.items():
            with self.subTest(topic=key):
                self.assertEqual(config.topics[key], expected)
        self.assertEqual(config.frames['scan_merged'], 'virtual_base_laser_link')

    def test_unavailable_interfaces_are_explicit(self):
        real, sim = get_ros_config('real'), get_ros_config('simulation')
        for key in ('contacts', 'bin_contacts', 'base_imu'):
            self.assertIsNone(real.topics[key])
            self.assertEqual(sim.topics[key], '/'+key)
        self.assertIsNone(sim.topics['scan_merged'])
        self.assertIsNone(sim.topics['controller_rejections'])

    def test_default_config_follows_selected_default_profile(self):
        self.assertEqual(get_ros_config(), get_ros_config(DEFAULT_PROFILE))

    def test_configurations_are_independent_copies(self):
        first = get_ros_config('simulation')
        first.topics['book_pose'] = '/temporary/book'
        first.frames['odom'] = 'temporary_odom'
        self.assertEqual(get_ros_config('simulation').topics['book_pose'], '/perception/book_pose')
        self.assertEqual(get_ros_config('simulation').frames['odom'], 'odom')

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown robot_profile'):
            get_ros_config('reall')

    def test_invalid_override_is_rejected(self):
        for section, overrides in (('topics', {'typo': '/camera'}),
                                   ('topics', {'odom': None}),
                                   ('actions', {'torso_trajectory': ''}),
                                   ('frames', {'base': 'invalid frame'})):
            with self.subTest(section=section), patch.dict(
                    PROFILES, {'bad': {'use_sim_time': False, section: overrides}}):
                with self.assertRaises(ValueError):
                    get_ros_config('bad')

    def test_node_profile_uses_ros_parameter_override(self):
        node = NS(declare_parameter=lambda name, default: NS(value='real'),
                  get_logger=lambda: NS(info=lambda text: None))
        self.assertEqual(configure_ros(node).profile, 'real')

    def test_tf_remap_source_stays_the_library_default(self):
        config = get_ros_config()
        config.topics.update(tf='/robot/tf', tf_static='/robot/tf_static', clock='/sim/clock')
        self.assertEqual(dict(config.builtin_remappings()), {
            '/tf': '/robot/tf', '/tf_static': '/robot/tf_static', '/clock': '/sim/clock'})


class ProfileLaunch(unittest.TestCase):
    def test_default_launch_uses_selected_default_in_every_node(self):
        nodes = launch_nodes()
        self.assertEqual({node.executable for node in nodes}, {
            'perception_node', 'navigation_node', 'manipulation_node'})
        for node in nodes:
            self.assertEqual(node.parameters[0]['robot_profile'], DEFAULT_PROFILE)
            self.assertIs(node.parameters[0]['use_sim_time'], get_ros_config(DEFAULT_PROFILE).use_sim_time)

    def test_simulation_launch_uses_simulated_time_in_every_node(self):
        for node in launch_nodes(robot_profile='simulation'):
            self.assertEqual(node.parameters[0]['robot_profile'], 'simulation')
            self.assertIs(node.parameters[0]['use_sim_time'], True)

    def test_real_launch_uses_system_time_in_every_node(self):
        for node in launch_nodes(robot_profile='real'):
            self.assertEqual(node.parameters[0]['robot_profile'], 'real')
            self.assertIs(node.parameters[0]['use_sim_time'], False)

    def test_explicit_clock_choice_overrides_profile(self):
        for profile, clock, expected in (('real', 'true', True), ('simulation', 'false', False)):
            for node in launch_nodes(robot_profile=profile, use_sim_time=clock):
                self.assertIs(node.parameters[0]['use_sim_time'], expected)

    def test_custom_tf_and_clock_are_remapped_for_all_nodes(self):
        custom = {'use_sim_time': True, 'topics': {
            'tf': '/test/tf', 'tf_static': '/test/tf_static', 'clock': '/test/clock'}}
        with patch.dict(PROFILES, {'custom': custom}):
            for node in launch_nodes(robot_profile='custom'):
                self.assertEqual(dict(node.remappings), {
                    '/tf': '/test/tf', '/tf_static': '/test/tf_static', '/clock': '/test/clock'})
                self.assertEqual(node.parameters[0]['robot_profile'], 'custom')

    def test_bad_profile_or_clock_is_rejected_before_launch(self):
        for overrides in ({'robot_profile': 'misspelled'}, {'use_sim_time': 'yes'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                launch_nodes(**overrides)


if __name__ == '__main__':
    unittest.main()
