#!/usr/bin/env python3
"""
Generate the competition URDF for the TIAGo Pro robot.

Calls PAL's xacro with the competition configuration and applies the two
patches the robot itself needs: the wheels are removed from <ros2_control> as
commanded joints and re-added as state-only, and the head camera is retargeted
from PAL's stock module to the Intel RealSense D435 the competition robot
carries. The result is written to erc_description/urdf/tiago_pro.urdf.

This file describes the ROBOT, not the simulator. The MuJoCo-specific
retarget - hardware plugin, actuators, four-bar loop closures, camera
placement, gravity compensation - is applied on top of this by
generate_mujoco_urdf.py at launch, so this URDF stays the single description
both the simulator and any real-robot tooling start from.

Run inside the container after building:
    ros2 run erc_bringup generate_urdf.py

Use --help to see all available configuration options and their valid values.

Examples:
    # Default competition config (dual-arm, pro grippers)
    ros2 run erc_bringup generate_urdf.py

    # Single left arm only
    ros2 run erc_bringup generate_urdf.py --arm_type_right no-arm

    # Navigation-only testing, no arms
    ros2 run erc_bringup generate_urdf.py --arm_type_left no-arm --arm_type_right no-arm
"""

import argparse
import math
import os
import re
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory

# The wheels are left state-only here because that is what the competition's
# Gazebo build wants: there the gz mecanum plugin drives them from outside
# ros2_control. The MuJoCo retarget gives them a velocity command interface of
# its own (generate_mujoco_urdf.py, command_wheels), so this file stays the
# description of the robot rather than of either simulator.
WHEEL_JOINTS = ('wheel_front_right_joint', 'wheel_front_left_joint',
                'wheel_rear_right_joint', 'wheel_rear_left_joint')
WHEEL_POSITIONS = {'wheel_front_right_joint': 'front_right',
                   'wheel_front_left_joint': 'front_left',
                   'wheel_rear_right_joint': 'rear_right',
                   'wheel_rear_left_joint': 'rear_left'}

# The head camera is an Intel RealSense D435, and this is the ONE place its
# optics are stated: generate_mujoco_urdf.py reads them back out of the sensor
# block below rather than carrying a second copy, so what the simulator renders
# and what the description says cannot drift apart.
#
# 640x360 is a native 16:9 D435 depth mode and 87 deg is the datasheet
# horizontal field of view (~87x56, against the datasheet's 87x58). The clip
# opens PAL's short 0.3-3.0 m range to the module's own 0.2-8.0 m so the shelf
# at ~4 m is in range.
HEAD_CAMERA_LINK = 'head_front_camera_link'
HEAD_CAMERA_W, HEAD_CAMERA_H = 640, 360
HEAD_CAMERA_HFOV = 1.5184364492350666    # 87 deg
HEAD_CAMERA_NEAR, HEAD_CAMERA_FAR = 0.2, 8.0


def patch_head_camera(urdf):
    """Retarget PAL's stock head camera sensors to the competition's D435.

    Rewrites values rather than matching PAL's own, so the result is the same
    whichever module --camera_model selected. Scoped to the head camera's own
    <gazebo> block, so a wrist camera added by --has_wrist_camera keeps its.
    """
    m = re.search(r'<gazebo reference="' + HEAD_CAMERA_LINK + r'">.*?</gazebo>',
                  urdf, re.S)
    if not m:
        sys.exit(f'ERROR: no <gazebo reference="{HEAD_CAMERA_LINK}"> block to '
                 f'retarget; the head camera would keep PAL\'s stock optics.')

    # The vertical field of view PAL emits alongside the horizontal one is not
    # an independent degree of freedom - a pinhole camera has one FoV and an
    # aspect ratio - so it is derived here rather than stated, and cannot
    # contradict the numbers above.
    vfov = 2 * math.atan(math.tan(HEAD_CAMERA_HFOV / 2)
                         * HEAD_CAMERA_H / HEAD_CAMERA_W)

    # Both head sensors - the colour camera and the RGB-D one - are retargeted
    # together and stay pixel-aligned, because a depth sample is read at a
    # colour pixel's coordinates downstream.
    block = m.group(0)
    for pattern, repl, expected in (
        (r'<width>\d+</width>', f'<width>{HEAD_CAMERA_W}</width>', 2),
        (r'<height>\d+</height>', f'<height>{HEAD_CAMERA_H}</height>', 2),
        (r'<horizontal_fov>[^<]+</horizontal_fov>',
         f'<horizontal_fov>{HEAD_CAMERA_HFOV}</horizontal_fov>', 2),
        (r'<vertical_fov>[^<]+</vertical_fov>',
         f'<vertical_fov>{vfov}</vertical_fov>', 2),
        # The rendered image follows the ROS optical convention (Z forward, X
        # right, Y down), but PAL stamps some modules with the camera BODY
        # frame. Every consumer that trusts the header then reads it through
        # the wrong axes: a point "1 m in front" is placed 1 m up. The optical
        # frames are already in the description; point the sensors at them.
        # Matched by role, not by PAL's current value, so a module that already
        # names the optical frame is left correct rather than failing the count.
        (r'<gz_frame_id>head_front_camera_(color|depth)(?:_optical)?_frame</gz_frame_id>',
         r'<gz_frame_id>head_front_camera_\1_optical_frame</gz_frame_id>', 2),
        # The simulator publishes RGB8, whatever byte order the module declares.
        (r'<format>[^<]+</format>', '<format>R8G8B8</format>', 2),
    ):
        block, n = re.subn(pattern, repl, block)
        if n != expected:
            sys.exit(f'ERROR: head camera retarget: {pattern} matched {n} '
                     f'times, expected {expected}. A half-retargeted camera '
                     f'renders at optics nothing downstream agrees with.')

    # The range clip is the depth sensor's alone. Scoped to <depth_camera>
    # because some modules put a <clip> on the colour camera as well, where it
    # means the render distance rather than the sensing range.
    depth = re.search(r'<depth_camera>.*?</depth_camera>', block, re.S)
    if not depth:
        sys.exit('ERROR: head camera retarget: no <depth_camera> to give a '
                 'sensing range to; this module has no depth sensor.')
    clipped = depth.group(0)
    for pattern, repl in ((r'<near>[^<]+</near>', f'<near>{HEAD_CAMERA_NEAR}</near>'),
                          (r'<far>[^<]+</far>', f'<far>{HEAD_CAMERA_FAR}</far>')):
        clipped, n = re.subn(pattern, repl, clipped)
        if n != 1:
            sys.exit(f'ERROR: head camera retarget: {pattern} matched {n} '
                     f'times inside <depth_camera>, expected 1.')
    block = block[:depth.start()] + clipped + block[depth.end():]

    return urdf[:m.start()] + block + urdf[m.end():]


def generate_urdf(**xacro_args):
    """Generate and patch the TIAGo Pro URDF. Returns the URDF string."""

    tiago_dir = get_package_share_directory('tiago_pro_description')
    bringup_dir = get_package_share_directory('erc_bringup')

    controller_cfg = os.path.join(
        bringup_dir, 'config', 'controller_manager_cfg.yaml')

    # ── Generate URDF via xacro ──
    xacro_path = os.path.join(tiago_dir, 'robots', 'tiago_pro.urdf.xacro')
    cmd = ['xacro', xacro_path]
    for key, val in xacro_args.items():
        cmd.append(f'{key}:={val}')
    urdf = subprocess.check_output(cmd).decode('utf-8')

    # ── Swap PAL's controller config with ours ──
    pal_cfg = os.path.join(
        tiago_dir, 'ros2_control', 'gazebo_controller_manager_cfg.yaml')
    urdf = urdf.replace(pal_cfg, controller_cfg)

    # ── Wheels: commanded -> state-only ──
    for wheel in WHEEL_JOINTS:
        position = WHEEL_POSITIONS[wheel]
        joint_block = (
            f'<joint name="{wheel}">\n'
            f'      <command_interface name="velocity"/>\n'
            f'      <state_interface name="position"/>\n'
            f'    </joint>\n'
            f'    <transmission name="wheel_{position}_trans">\n'
            f'      <plugin>transmission_interface/SimpleTransmission</plugin>\n'
            f'      <actuator name="wheel_{position}_actuator" role="actuator1"/>\n'
            f'      <joint name="{wheel}" role="joint1">\n'
            f'        <mechanical_reduction>1.0</mechanical_reduction>\n'
            f'      </joint>\n'
            f'    </transmission>\n'
        )
        if joint_block in urdf:
            urdf = urdf.replace(joint_block, '', 1)
        else:
            print(f'WARNING: could not strip {wheel} from <ros2_control> — '
                  f'block format may have changed.')

    state_only = ''
    for wheel in WHEEL_JOINTS:
        state_only += (
            f'    <joint name="{wheel}">\n'
            f'      <state_interface name="position"/>\n'
            f'      <state_interface name="velocity"/>\n'
            f'    </joint>\n'
        )
    urdf = urdf.replace('</ros2_control>', state_only + '  </ros2_control>', 1)

    # ── Head camera: PAL's stock module -> the competition's D435 ──
    urdf = patch_head_camera(urdf)

    return urdf


def main():
    parser = argparse.ArgumentParser(
        description='Generate the ERC 2026 TIAGo Pro URDF.')

    # ── Robot configuration ──
    parser.add_argument('--arm_type_left', default='tiago-pro-s',
                        choices=['tiago-pro-s', 'no-arm'],
                        help='Left arm type (default: tiago-pro-s)')
    parser.add_argument('--arm_type_right', default='tiago-pro-s',
                        choices=['tiago-pro-s', 'no-arm'],
                        help='Right arm type (default: tiago-pro-s)')
    parser.add_argument('--end_effector_left', default='pal-pro-gripper',
                        choices=['pal-pro-gripper', 'no-end-effector'],
                        help='Left end effector (default: pal-pro-gripper)')
    parser.add_argument('--end_effector_right', default='pal-pro-gripper',
                        choices=['pal-pro-gripper', 'no-end-effector'],
                        help='Right end effector (default: pal-pro-gripper)')
    parser.add_argument('--ft_sensor_left', default=None,
                        choices=['no-ft-sensor'],
                        help='Force torque sensor left arm [no-ft-sensor]')
    parser.add_argument('--ft_sensor_right', default=None,
                        choices=['no-ft-sensor'],
                        help='Force torque sensor right arm [no-ft-sensor]')
    parser.add_argument('--wrist_model_left', default=None,
                        help='Wrist model left arm [wrist-2010, wrist-2017]')
    parser.add_argument('--wrist_model_right', default=None,
                        help='Wrist model right arm [wrist-2010, wrist-2017]')
    parser.add_argument('--camera_model', default=None,
                        help='Head camera model [realsense-d435i, realsense-d435]')
    parser.add_argument('--laser_model', default=None,
                        help='LiDAR model [sick-571, sick-561, sick-tim, no-laser]')
    parser.add_argument('--base_type', default=None,
                        help='Base type [omni_base]')

    # ── Boolean flags ──
    parser.add_argument('--has_wrist_camera', action='store_true',
                        help='Mount a wrist camera link on the end effector. Adds '
                             'the link only - nothing renders it, so it is a '
                             'kinematic placeholder [flag, default: off]')

    args = parser.parse_args()

    xacro_args = {
        'arm_type_left': args.arm_type_left,
        'arm_type_right': args.arm_type_right,
        'end_effector_left': args.end_effector_left,
        'end_effector_right': args.end_effector_right,
        'has_teleop_arms': 'False',
        'is_public_sim': 'True',
        'use_sim_time': 'True',
        # PAL's arg name is about their plugin choice, but it also selects the
        # command-interface set: their 'classic' default adds velocity and
        # effort interfaces alongside position, and a position-controlled
        # simulator rejects both. Keep it on the position-only branch.
        'gazebo_version': 'gazebo',
    }

    optional = {
        'ft_sensor_left': args.ft_sensor_left,
        'ft_sensor_right': args.ft_sensor_right,
        'wrist_model_left': args.wrist_model_left,
        'wrist_model_right': args.wrist_model_right,
        'camera_model': args.camera_model,
        'laser_model': args.laser_model,
        'base_type': args.base_type,
    }
    for key, val in optional.items():
        if val is not None:
            xacro_args[key] = val

    if args.has_wrist_camera:
        xacro_args['has_wrist_camera'] = 'True'

    print('[generate_urdf] Configuration:')
    for key, val in xacro_args.items():
        print(f'  {key}: {val}')

    urdf = generate_urdf(**xacro_args)

    # Prefer the source tree so the URDF is visible and editable on the host.
    # With --symlink-install the share directory is a symlink to it anyway; the
    # fallback only matters for a non-symlink build, where the file lands in the
    # install space and the next colcon build would overwrite it.
    share_urdf = os.path.join(
        get_package_share_directory('erc_description'), 'urdf')
    source_urdf = os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..', '..', 'erc_description', 'urdf'))
    if os.path.isdir(os.path.dirname(source_urdf)):
        output_dir = source_urdf
    else:
        output_dir = share_urdf
        print('[generate_urdf] WARNING: writing into the install space; a '
              'rebuild will overwrite this. Build with --symlink-install.')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'tiago_pro.urdf')

    with open(output_path, 'w') as f:
        f.write(urdf)

    print(f'[generate_urdf] Written to {os.path.realpath(output_path)}')
    print(f'[generate_urdf] URDF size: {len(urdf):,} bytes')


if __name__ == '__main__':
    main()
