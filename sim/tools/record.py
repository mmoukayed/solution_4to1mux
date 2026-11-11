#!/usr/bin/env python3
"""Record the arena camera and the robot's head camera as JPEG frames, each with its SIMULATION
time stamp, until stopped (Ctrl-C / SIGTERM). make_video.py turns them into a video that plays
at simulation speed, so a slow computer still gives a real-time video.

    python3 record.py OUT_DIR [--every 2]

OUT_DIR/arena and OUT_DIR/head get fNNNNN.jpg plus stamps.txt ("frame sim_time").
"""
import argparse
import os
import signal

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

TOPICS = {'arena': '/spectator/color',
          'head': '/head_front_camera/head_front_camera/color/image_raw'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--every', type=int, default=2, help='keep every Nth frame of each camera')
    a = ap.parse_args()
    rclpy.init()
    node = Node('arena_recorder')
    state = {}

    def recorder(name, topic):
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        st = state[name] = dict(n=0, k=0, stamps=open(os.path.join(d, 'stamps.txt'), 'w'))

        def cb(m):
            st['n'] += 1
            if st['n'] % a.every:
                return
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            if m.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(d, 'f%05d.jpg' % st['k']), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            st['stamps'].write('%d %.3f\n' % (st['k'], m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec))
            st['stamps'].flush()
            st['k'] += 1
        node.create_subscription(Image, topic, cb, qos_profile_sensor_data)

    for name, topic in TOPICS.items():
        recorder(name, topic)
    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())
    print('recording', ', '.join(TOPICS.values()), '->', a.out, flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    print('saved', ', '.join('%s %d frames' % (k, v['k']) for k, v in state.items()))


if __name__ == '__main__':
    main()
