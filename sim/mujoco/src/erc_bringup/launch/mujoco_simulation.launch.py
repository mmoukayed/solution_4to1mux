"""
Competition simulation.

Brings up the arena and the robot on MuJoCo: the controllers from
controller_params.yaml, /cmd_vel driving the omni base through
mecanum_drive_controller and the four wheels, /joint_states from
joint_state_broadcaster, the head RealSense on the competition topic names,
and the ERC_SEED-driven arena layout.

Two artefacts are generated at launch rather than checked in, because both
have to carry absolute paths and the arena also has to carry the seed:

    generate_mujoco_urdf.py   competition URDF -> MuJoCo-flavoured URDF
    generate_mujoco_world.py  arena spec       -> arena MJCF

    ros2 launch erc_bringup mujoco_simulation.launch.py
    ERC_SEED=7 ros2 launch erc_bringup mujoco_simulation.launch.py headless:=false
"""

import atexit
import os
import shutil
import subprocess
import tempfile

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
    get_package_prefix,
)
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# The robot starts on the start zone, yawed 90 degrees, as the arena spec has it.
START_ZONE_XYZ = '0 0 0'
START_ZONE_YAW = '1.5708'


def launch_setup(context, *args, **kwargs):
    headless = LaunchConfiguration('headless').perform(context)
    camera_rate = LaunchConfiguration('camera_rate').perform(context)
    sim_speed = LaunchConfiguration('sim_speed').perform(context)
    plugins_file = LaunchConfiguration('plugins').perform(context)
    scene_override = LaunchConfiguration('scene').perform(context)
    seed = LaunchConfiguration('seed').perform(context)
    rnd = LaunchConfiguration('round').perform(context)
    book_mu = LaunchConfiguration('book_mu').perform(context)
    layout = LaunchConfiguration('layout').perform(context) or os.environ.get('ERC_LAYOUT', '')

    bringup_share = get_package_share_directory('erc_bringup')
    scripts = os.path.join(get_package_prefix('erc_bringup'), 'lib', 'erc_bringup')
    omni_assets = os.path.join(
        get_package_share_directory('omni_base_description'), 'mujoco', 'assets')

    # A private directory per launch. Fixed /tmp names collide between two
    # users on a shared machine and race between two concurrent launches,
    # and the converter reads these files from a separate process.
    workdir = tempfile.mkdtemp(prefix='erc_mujoco_')
    atexit.register(shutil.rmtree, workdir, True)

    # ── The robot: competition URDF, retargeted at MuJoCo ──
    # A drawn layout moves the start inside the start zone and turns it; the
    # world generator owns the draw, so the robot and the arena agree.
    spawn_xyz, spawn_yaw = START_ZONE_XYZ, START_ZONE_YAW
    if layout and not layout.lstrip('-').isdigit():
        raise RuntimeError(f'layout:={layout} is not a whole number')
    if layout and scene_override:
        raise RuntimeError('layout:= draws the arena the robot is placed in; it cannot '
                           'be combined with scene:=, whose arena is fixed')
    if layout:
        got = subprocess.run(
            [os.path.join(scripts, 'generate_mujoco_world.py'), '--layout', layout,
             '--print-start'], check=True, capture_output=True, text=True).stdout.split()
        spawn_xyz, spawn_yaw = ' '.join(got[:3]), got[3]
    urdf_out = os.path.join(workdir, 'robot.urdf')
    try:
        subprocess.run(
            [os.path.join(scripts, 'generate_mujoco_urdf.py'),
             '-o', urdf_out,
             '--headless', headless,
             '--camera-rate', camera_rate,
             '--sim-speed-factor', sim_speed,
             '--spawn-xyz', spawn_xyz,
             '--spawn-yaw', spawn_yaw],
            check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            'Could not build the MuJoCo robot description. It is derived from '
            'the competition URDF at erc_description/urdf/tiago_pro.urdf; if '
            'that file is missing, generate it first with:\n'
            '    ros2 run erc_bringup generate_urdf.py') from exc
    with open(urdf_out) as fh:
        robot_description = fh.read()

    # ── The arena ──
    if scene_override:
        if not os.path.exists(scene_override):
            raise RuntimeError(
                f'scene:={scene_override} does not exist. Pass an absolute path '
                f'to an MJCF file, or omit the argument to generate the arena.')
        scene_path = scene_override
    else:
        scene_path = os.path.join(workdir, 'world.xml')
        cmd = [os.path.join(scripts, 'generate_mujoco_world.py'), '-o', scene_path]
        if seed:
            cmd += ['--seed', seed]
        if rnd:
            cmd += ['--round', rnd]
        if book_mu:
            cmd += ['--book-mu', book_mu]
        if layout:
            cmd += ['--layout', layout]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                'Could not build the arena. It is generated from the meshes and '
                'textures in erc_description; check that package is built.'
            ) from exc

    manager_cfg = os.path.join(
        bringup_share, 'config', 'controller_manager_cfg.yaml')
    controller_params = os.path.join(
        bringup_share, 'config', 'controller_params.yaml')
    plugins_cfg = os.path.join(bringup_share, 'config', plugins_file)

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': ParameterValue(robot_description,
                                                         value_type=str),
                     'use_sim_time': True}],
    )

    converter = Node(
        package='mujoco_ros2_control',
        executable='robot_description_to_mjcf.sh',
        output='both',
        emulate_tty=True,
        arguments=[
            '--publish_topic', '/mujoco_robot_description',
            '--no-fuse',
            '--add_free_joint',
            '--asset_dir', omni_assets,
            '--scene', scene_path,
        ],
    )

    control = Node(
        package='mujoco_ros2_control',
        executable='ros2_control_node',
        output='both',
        emulate_tty=True,
        parameters=[
            {'use_sim_time': True},
            manager_cfg,
            plugins_cfg,
            # Last wins, so this is what makes camera_rate:= mean anything;
            # the value in mujoco_plugins.yaml would otherwise always win.
            {'mujoco_plugins.mujoco_camera_plugin.camera_publish_rate':
                float(camera_rate)},
        ],
        remappings=[('~/robot_description', '/robot_description')],
    )

    # The depth's camera_info is the depth camera's own (mujoco_plugins.yaml,
    # head_front_camera_depth): the colour's used to be relayed onto it when
    # one camera rendered both, and that would now give the depth the
    # colour's intrinsics.

    # mecanum_drive_controller listens on its own private reference topic; the
    # competition drives the base on /cmd_vel. It reads Twist rather than
    # TwistStamped because controller_params.yaml sets use_stamped_vel: false.
    cmd_vel_relay = Node(
        package='topic_tools',
        executable='relay',
        name='cmd_vel_relay',
        arguments=['/cmd_vel',
                   '/mecanum_drive_controller/reference_unstamped'],
        parameters=[{'use_sim_time': True}],
    )

    # IMUSensorBroadcaster publishes on its own private topic; the competition
    # exposes the base IMU as /base_imu.
    imu_relay = Node(
        package='topic_tools',
        executable='relay',
        name='imu_relay',
        arguments=['/imu_sensor_broadcaster/imu', '/base_imu'],
        parameters=[{'use_sim_time': True}],
    )

    odom_relay = Node(
        package='erc_bringup',
        executable='odom_relay.py',
        output='both',
        parameters=[{'use_sim_time': True}],
    )

    def spawner(name, params=None):
        args = [name, '-c', '/controller_manager',
                '--controller-manager-timeout', '120']
        if params:
            args += ['-p', params]
        return Node(package='controller_manager', executable='spawner',
                    arguments=args, output='both')

    controllers = TimerAction(period=5.0, actions=[
        spawner('joint_state_broadcaster'),
        spawner('imu_sensor_broadcaster', controller_params),
        spawner('mecanum_drive_controller', controller_params),
        spawner('arm_left_controller', controller_params),
        spawner('arm_right_controller', controller_params),
        spawner('head_controller', controller_params),
        spawner('torso_controller', controller_params),
        spawner('gripper_left_controller_raw', controller_params),
        spawner('gripper_right_controller_raw', controller_params),
    ])

    gripper_clamp = TimerAction(period=9.0, actions=[
        Node(package='erc_bringup', executable='gripper_command_clamp.py',
             parameters=[{'use_sim_time': True}], output='both'),
    ])

    # The camera plugin publishes depth but no point cloud. The `sensors`
    # package rebuilds it with a RealSense-D435 noise model; its topic defaults
    # already match what this launch publishes. Delayed so it starts after the
    # camera it consumes, rather than idling on an absent camera_info.
    depth_cloud = []
    try:
        sensors_launch = os.path.join(
            get_package_share_directory('sensors'), 'launch', 'depth_to_cloud.launch.py')
    except PackageNotFoundError:
        sensors_launch = None
    if sensors_launch and os.path.exists(sensors_launch):
        depth_cloud.append(TimerAction(period=12.0, actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sensors_launch),
                condition=IfCondition(LaunchConfiguration('depth_cloud')))]))
    else:
        print('[mujoco_simulation] depth cloud NOT started - the `sensors` '
              'package is unavailable. Build it with: '
              'colcon build --packages-select sensors --symlink-install')

    # topic_tools relay picks its publisher QoS from the first publisher it
    # discovers on the input topic. Started at t=0 it can come up before the
    # camera and the broadcaster exist and settle on a QoS nothing downstream
    # matches - which showed up as depth_to_cloud silently never receiving
    # camera_info and publishing no cloud at all. Start them once their inputs
    # are real.
    relays = TimerAction(period=8.0, actions=[imu_relay,
                                              cmd_vel_relay])

    return [rsp, converter, control, relays, odom_relay,
            controllers, gripper_clamp, *depth_cloud]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('camera_rate', default_value='30.0'),
        DeclareLaunchArgument('sim_speed', default_value='-1.0',
                              description='-1 follows the viewer slowdown setting'),
        DeclareLaunchArgument('plugins', default_value='mujoco_plugins.yaml'),
        DeclareLaunchArgument('scene', default_value='',
                              description='Absolute path to an MJCF scene to use '
                                          'instead of the generated arena'),
        DeclareLaunchArgument('seed', default_value='',
                              description='Arena layout seed (default: $ERC_SEED)'),
        DeclareLaunchArgument('round', default_value='',
                              description='Competition round layout, 1 or 2 '
                                          '(default: $ERC_ROUND or 1)'),
        DeclareLaunchArgument('layout', default_value='',
                              description='table and start pose draw (default $ERC_LAYOUT; '
                                          'empty: the arena spec)'),
        DeclareLaunchArgument('book_mu', default_value='',
                              description='Book sliding friction '
                                          '(default: $ERC_BOOK_MU or the generator\'s)'),
        DeclareLaunchArgument('depth_cloud', default_value='true',
                              description='Rebuild the head depth point cloud '
                                          'via sensors/depth_to_cloud'),
        OpaqueFunction(function=launch_setup),
    ])
