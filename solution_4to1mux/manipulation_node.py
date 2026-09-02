#!/usr/bin/env python3
"""
manipulation_node.py — solution_4to1mux (ERC 2026)

Open-loop pick-and-place sequencer using the left arm + gripper only, per the
competition's single-arm rule.

Trigger points:
  - Starts the pick sequence when navigation_node reports state 'AT_SHELF'
  - Starts the place sequence when navigation_node reports state 'AT_BIN'
  - Publishes /manipulation/status = 'book_grasped' / 'book_placed' so
    navigation_node knows when to move on.

IMPORTANT: the joint angles below are PLACEHOLDERS copied in the same shape
as the README's `ros2 topic pub` examples. You need to jog the real arm in
the sim (via RViz + MoveIt, or by publishing test trajectories like those
examples) to find working pre-grasp / grasp joint sets for each shelf row
height, then replace ROW_JOINT_TARGETS and PRE_PLACE_POSE below. Swap this
whole node for a MoveIt 2 motion-planning pipeline later if open-loop jogging
isn't reliable enough once real book positions vary.
"""

import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String, Int32
from builtin_interfaces.msg import Duration


ARM_JOINTS = [f'arm_left_{i}_joint' for i in range(1, 8)]
GRIPPER_JOINT = ['gripper_left_finger_joint']

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

HOME_POSE = [0.2, 0.3, -0.2, 1.4, 0.0, 0.3, 0.0]

# PLACEHOLDER joint sets per shelf row (1-4). Replace with tuned values —
# these just illustrate the structure (pre_grasp = hovering near the row,
# grasp = extended to the book).
ROW_JOINT_TARGETS = {
    1: {'pre_grasp': [0.4, 0.5, -0.3, 1.2, 0.0, 0.4, 0.0], 'grasp': [0.55, 0.5, -0.3, 1.0, 0.0, 0.4, 0.0]},
    2: {'pre_grasp': [0.3, 0.4, -0.3, 1.1, 0.0, 0.4, 0.0], 'grasp': [0.45, 0.4, -0.3, 0.9, 0.0, 0.4, 0.0]},
    3: {'pre_grasp': [0.2, 0.3, -0.3, 1.0, 0.0, 0.4, 0.0], 'grasp': [0.35, 0.3, -0.3, 0.8, 0.0, 0.4, 0.0]},
    4: {'pre_grasp': [0.1, 0.2, -0.3, 0.9, 0.0, 0.4, 0.0], 'grasp': [0.25, 0.2, -0.3, 0.7, 0.0, 0.4, 0.0]},
}

PRE_PLACE_POSE = [0.0, 0.6, 0.0, 1.3, 0.0, 0.3, 0.0]  # arm out over the bin

TORSO_HEIGHTS = {1: 0.30, 2: 0.20, 3: 0.10, 4: 0.0}  # metres, low row -> low torso


class ManipulationNode(Node):
    def __init__(self):
        super().__init__('manipulation_node')

        self.target_row = None
        self.nav_state = None
        self.sequence = []          # queue of (joint_group, positions, duration_s) steps
        self.step_start_time = None
        self.step_duration = 0.0
        self.current_step = None
        self.busy = False

        self.did_pick_sequence = False
        self.did_place_sequence = False

        self.create_subscription(String, '/navigation/status', self.nav_status_cb, 10)
        self.create_subscription(Int32, '/erc/shelf_row_identification', self.row_cb, 10)

        self.arm_pub = self.create_publisher(JointTrajectory, '/arm_left_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(JointTrajectory, '/gripper_left_controller/joint_trajectory', 10)
        self.torso_pub = self.create_publisher(JointTrajectory, '/torso_controller/joint_trajectory', 10)
        self.status_pub = self.create_publisher(String, '/manipulation/status', 10)

        self.timer = self.create_timer(0.2, self.tick)
        self.get_logger().info('ManipulationNode up, waiting for shelf row + navigation state')

    # ------------------------------------------------------------------
    def row_cb(self, msg: Int32):
        self.target_row = msg.data

    def nav_status_cb(self, msg: String):
        self.nav_state = msg.data

        if self.nav_state == 'AT_SHELF' and not self.did_pick_sequence and not self.busy:
            self._queue_pick_sequence()
            self.did_pick_sequence = True

        if self.nav_state == 'AT_BIN' and not self.did_place_sequence and not self.busy:
            self._queue_place_sequence()
            self.did_place_sequence = True

    # ------------------------------------------------------------------
    def _queue_pick_sequence(self):
        if self.target_row is None:
            self.get_logger().warn('AT_SHELF but no row identified yet — defaulting to row 2')
        row = self.target_row or 2
        targets = ROW_JOINT_TARGETS.get(row, ROW_JOINT_TARGETS[2])
        torso_h = TORSO_HEIGHTS.get(row, 0.15)

        self.sequence = [
            ('torso', [torso_h], 2.0),
            ('gripper', [GRIPPER_OPEN], 1.0),
            ('arm', targets['pre_grasp'], 2.5),
            ('arm', targets['grasp'], 2.0),
            ('gripper', [GRIPPER_CLOSED], 1.0),
            ('arm', targets['pre_grasp'], 2.0),   # retract with book
            ('arm', HOME_POSE, 2.5),
            ('__done__', 'book_grasped', 0.0),
        ]
        self.busy = True
        self.get_logger().info(f'Queued pick sequence for row {row}')

    def _queue_place_sequence(self):
        self.sequence = [
            ('arm', PRE_PLACE_POSE, 2.5),
            ('gripper', [GRIPPER_OPEN], 1.0),
            ('arm', HOME_POSE, 2.5),
            ('__done__', 'book_placed', 0.0),
        ]
        self.busy = True
        self.get_logger().info('Queued place sequence')

    # ------------------------------------------------------------------
    def tick(self):
        if not self.busy or (not self.sequence and self.current_step is None):
            return

        now = self.get_clock().now()

        if self.current_step is None:
            self.current_step = self.sequence.pop(0)
            group, positions, duration = self.current_step

            if group == '__done__':
                self.status_pub.publish(String(data=positions))
                self.get_logger().info(f'Manipulation sequence complete: {positions}')
                self.current_step = None
                self.busy = False
                return

            self._send_trajectory(group, positions, duration)
            self.step_start_time = now
            self.step_duration = duration

        else:
            elapsed = (now - self.step_start_time).nanoseconds / 1e9
            if elapsed >= self.step_duration:
                self.current_step = None  # advance to next queued step next tick

    def _send_trajectory(self, group, positions, duration_s):
        msg = JointTrajectory()
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration_s), nanosec=int((duration_s % 1) * 1e9))
        msg.points = [point]

        if group == 'arm':
            msg.joint_names = ARM_JOINTS
            self.arm_pub.publish(msg)
        elif group == 'gripper':
            msg.joint_names = GRIPPER_JOINT
            self.gripper_pub.publish(msg)
        elif group == 'torso':
            msg.joint_names = ['torso_lift_joint']
            self.torso_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()