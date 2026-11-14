#!/usr/bin/env python3
"""Pre-flight for the live robot: is everything the three nodes need actually arriving, at a usable rate,
with sane timestamps? Read-only: it only listens. Run it in the same container/shell as the launch:

    python3 scripts/preflight.py            # robot_profile real, listens 8 s
    python3 scripts/preflight.py --secs 15

Exit code 0 = all PASS (WARNs allowed), 1 = something the run needs is missing.
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from control_msgs.action import FollowJointTrajectory
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from solution_4to1mux.ros_topics import get_ros_config

# what each stream must reach for the stack to work (Hz): (key, type, FAIL below, WARN below)
STREAMS = [
    ('joint_states', JointState, 10.0, 30.0),
    ('odom', Odometry, 5.0, 15.0),
    ('scan_front', LaserScan, 3.0, 8.0),
    ('scan_rear', LaserScan, 3.0, 8.0),
    ('camera_rgb', Image, 3.0, 10.0),
    ('camera_info', CameraInfo, 3.0, 10.0),
    ('camera_depth_raw', Image, 1.0, 5.0),
]
ACTIONS = ['right_arm_trajectory', 'torso_trajectory', 'head_trajectory', 'left_arm_trajectory']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='real')
    ap.add_argument('--secs', type=float, default=8.0)
    ap.add_argument('--sim-time', action='store_true', help='stamps are simulation time (twin / simulator)')
    args = ap.parse_args()
    cfg = get_ros_config(args.profile)
    topics, frames = cfg.topics, cfg.frames

    rclpy.init()
    from rclpy.parameter import Parameter
    node = rclpy.create_node('solution_preflight', parameter_overrides=[
        Parameter('use_sim_time', Parameter.Type.BOOL, args.sim_time)])
    stats = {}
    results = []

    def add(level, what, detail):
        results.append((level, what, detail))

    def listen(key, kind, topic):
        st = stats.setdefault(key, {'topic': topic, 'n': 0, 'age': [], 'sizes': set(), 'first': None})

        def cb(msg):
            now = time.monotonic()
            st['n'] += 1
            st['first'] = st['first'] or now
            st['last'] = now
            header = getattr(msg, 'header', None)
            if header is not None and (header.stamp.sec or header.stamp.nanosec):
                stamp = header.stamp.sec+header.stamp.nanosec*1e-9
                st['age'].append(node.get_clock().now().nanoseconds*1e-9-stamp)
            if kind is Image:
                st['sizes'].add(f'{msg.width}x{msg.height} {msg.encoding}')
            if kind is JointState:
                st.setdefault('names', set()).update(msg.name)
        node.create_subscription(kind, topic, cb, qos_profile_sensor_data)

    for key, kind, _, _ in STREAMS:
        if topics.get(key):
            listen(key, kind, topics[key])
    depth_info = topics['camera_depth_raw'].rsplit('/', 1)[0]+'/camera_info'
    listen('depth_camera_info', CameraInfo, depth_info)

    desc = {}
    node.create_subscription(String, topics['robot_description'], lambda m: desc.setdefault('urdf', m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    clients = {k: ActionClient(node, FollowJointTrajectory, cfg.actions[k]) for k in ACTIONS if cfg.actions.get(k)}
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    print(f'listening {args.secs:.0f} s on the {args.profile} profile ...', flush=True)
    t_end = time.monotonic()+args.secs
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    # streams
    for key, kind, fail_hz, warn_hz in STREAMS+[('depth_camera_info', CameraInfo, 1.0, 5.0)]:
        st = stats.get(key)
        if st is None:
            continue
        if st['n'] < 2:
            add('FAIL', key, f'{st["topic"]}: {st["n"]} messages in {args.secs:.0f} s')
            continue
        hz = (st['n']-1)/max(st['last']-st['first'], 1e-3)
        level = 'FAIL' if hz < fail_hz else 'WARN' if hz < warn_hz else 'PASS'
        detail = f'{st["topic"]}: {hz:.1f} Hz'
        if st['sizes']:
            detail += ' [' + ', '.join(sorted(st['sizes'])) + ']'
        if st['age']:
            age = sorted(st['age'])[len(st['age'])//2]
            detail += f', stamp age {age:+.2f} s'
            # the nodes judge freshness by now - stamp: a clock offset makes every pose "stale" (or never stale)
            if abs(age) > 2.0:
                level = 'FAIL'
                detail += ' (laptop vs robot clock offset? sync the clocks)'
            elif abs(age) > 0.5 and level == 'PASS':
                level = 'WARN'
        add(level, key, detail)
    if len(stats.get('camera_depth_raw', {}).get('sizes', ())) > 1:
        add('WARN', 'depth sizes', 'depth arrives in more than one size (decimated copies): handled, but note it')

    names = stats.get('joint_states', {}).get('names', set())
    need = {'torso_lift_joint', 'head_1_joint', 'head_2_joint', 'arm_right_1_joint', 'gripper_right_finger_joint'}
    missing = sorted(n for n in need if n not in names)
    if names:
        add('FAIL' if missing else 'PASS', 'joint names', f'missing {missing}' if missing else f'{len(names)} joints')

    add('PASS' if 'urdf' in desc else 'FAIL', 'robot_description',
        f'{len(desc["urdf"])} chars' if 'urdf' in desc else f'nothing on {topics["robot_description"]}')

    for key, client in clients.items():
        ok = client.wait_for_server(timeout_sec=2.0)
        level = 'PASS' if ok else ('WARN' if key == 'left_arm_trajectory' else 'FAIL')
        add(level, key, f'{cfg.actions[key]} {"ready" if ok else "NOT available"}')
    for key in ('right_gripper_command', 'head_command'):
        topic = topics.get(key)
        if topic:
            subs = node.count_subscribers(topic)
            add('PASS' if subs else 'FAIL', key, f'{topic}: {subs} subscriber(s)')

    for parent, child in ((frames.get('odom', 'odom'), frames.get('base', 'base_footprint')),
                          (frames.get('base', 'base_footprint'), frames.get('depth_camera'))):
        if not child:
            continue
        try:
            tf_buffer.lookup_transform(parent, child, Time(), timeout=Duration(seconds=1.0))
            add('PASS', 'tf', f'{parent} -> {child}')
        except Exception as error:
            add('FAIL', 'tf', f'{parent} -> {child}: {type(error).__name__}')

    width = max(len(w) for _, w, _ in results)
    for level, what, detail in results:
        print(f'{level:4}  {what:<{width}}  {detail}')
    fails = sum(1 for level, _, _ in results if level == 'FAIL')
    print(f'\n{"READY" if not fails else "NOT READY"}: {fails} fail(s), '
          f'{sum(1 for level, _, _ in results if level == "WARN")} warning(s)')
    node.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
