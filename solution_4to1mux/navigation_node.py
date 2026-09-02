#!/usr/bin/env python3
"""
navigation_node.py — solution_4to1mux (ERC 2026)

Simple state-machine navigation:
  1. ALIGN_COLUMN     — rotate/strafe to centre the target shelf column marker
                         in view (visual servo, driven by perception_node's
                         /perception/column_target).
  2. APPROACH_SHELF   — drive forward using the front LiDAR until within
                         grasp range of the shelf.
  3. AT_SHELF         — hold position, wait for manipulation_node to report
                         the book is grasped.
  4. RETURN_TO_START  — drive back to the recorded start pose using odometry.
  5. ALIGN_BIN        — rotate/strafe to centre the red bin
                         (/perception/bin_target).
  6. AT_BIN           — hold position, wait for manipulation_node to report
                         the book is placed.
  7. DONE             — trial ends when /bin_contacts fires.

This is a hand-rolled state machine rather than full Nav2, since the
environment has no pre-built map. Swap in Nav2 (AMCL + a saved map of the
room) later for proper obstacle-aware planning if the LiDAR safety-stop used
here isn't robust enough.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def yaw_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # --- Tunables ---
        self.shelf_stop_distance = 0.9     # metres, front LiDAR range to stop at
        self.centring_gain = 0.0025        # rad/s per pixel of horizontal error
        self.linear_speed = 0.2
        self.position_tolerance = 0.08     # metres
        self.yaw_tolerance = 0.05          # radians

        self.state = 'ALIGN_COLUMN'
        self.start_pose = None             # (x, y, yaw) captured on first odom msg
        self.current_pose = None
        self.column_target_px = None
        self.bin_target_px = None
        self.front_range = float('inf')
        self.manipulation_status = 'idle'

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan_front_raw', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Point, '/perception/column_target', self.column_target_cb, 10)
        self.create_subscription(Point, '/perception/bin_target', self.bin_target_cb, 10)
        self.create_subscription(String, '/manipulation/status', self.manip_status_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('NavigationNode up, state=ALIGN_COLUMN')

    # ------------------------------------------------------------------
    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.current_pose = (x, y, yaw)
        if self.start_pose is None:
            self.start_pose = (x, y, yaw)
            self.get_logger().info(f'Captured start pose: {self.start_pose}')

    def scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        centre = n // 2
        window = msg.ranges[max(0, centre - 10):centre + 10]
        valid = [r for r in window if msg.range_min < r < msg.range_max]
        self.front_range = min(valid) if valid else float('inf')

    def column_target_cb(self, msg: Point):
        self.column_target_px = (msg.x, msg.z)  # (pixel_x, image_width)

    def bin_target_cb(self, msg: Point):
        self.bin_target_px = (msg.x, msg.z)

    def manip_status_cb(self, msg: String):
        self.manipulation_status = msg.data

    # ------------------------------------------------------------------
    def control_loop(self):
        if self.current_pose is None:
            return  # wait for first odom message

        twist = Twist()

        if self.state == 'ALIGN_COLUMN':
            if self._centre_on(self.column_target_px, twist):
                self.state = 'APPROACH_SHELF'
                self.get_logger().info('Column centred -> APPROACH_SHELF')

        elif self.state == 'APPROACH_SHELF':
            if self.front_range > self.shelf_stop_distance:
                twist.linear.x = self.linear_speed
            else:
                self.state = 'AT_SHELF'
                self.get_logger().info('Reached shelf -> AT_SHELF, waiting on manipulation')

        elif self.state == 'AT_SHELF':
            if self.manipulation_status == 'book_grasped':
                self.state = 'RETURN_TO_START'
                self.get_logger().info('Book grasped -> RETURN_TO_START')

        elif self.state == 'RETURN_TO_START':
            if self._drive_to(self.start_pose, twist):
                self.state = 'ALIGN_BIN'
                self.get_logger().info('Back at start -> ALIGN_BIN')

        elif self.state == 'ALIGN_BIN':
            if self._centre_on(self.bin_target_px, twist):
                self.state = 'AT_BIN'
                self.get_logger().info('Bin centred -> AT_BIN, waiting on manipulation')

        elif self.state == 'AT_BIN':
            if self.manipulation_status == 'book_placed':
                self.state = 'DONE'
                self.get_logger().info('Book placed -> DONE')

        elif self.state == 'DONE':
            pass  # nothing left to do; trial ends when /bin_contacts fires

        self.cmd_pub.publish(twist)
        self.status_pub.publish(String(data=self.state))

    # ------------------------------------------------------------------
    def _centre_on(self, target_px, twist: Twist) -> bool:
        """Rotate until target_px is within a small band of image centre.
        Returns True once centred (and stops the robot that tick)."""
        if target_px is None:
            twist.angular.z = 0.15  # slow search rotation if we haven't seen the target yet
            return False

        px, width = target_px
        error = px - (width / 2.0)
        if abs(error) < width * 0.03:  # within 3% of image width -> considered centred
            return True

        twist.angular.z = -self.centring_gain * error
        return False

    def _drive_to(self, target_pose, twist: Twist) -> bool:
        tx, ty, tyaw = target_pose
        cx, cy, cyaw = self.current_pose
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)

        if dist > self.position_tolerance:
            heading_to_target = math.atan2(dy, dx)
            heading_error = self._wrap(heading_to_target - cyaw)
            # mecanum base: drive straight toward target in the robot frame
            twist.linear.x = self.linear_speed * math.cos(heading_error)
            twist.linear.y = self.linear_speed * math.sin(heading_error)
            return False

        yaw_error = self._wrap(tyaw - cyaw)
        if abs(yaw_error) > self.yaw_tolerance:
            twist.angular.z = 0.5 * yaw_error
            return False

        return True

    @staticmethod
    def _wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()