"""Live RGB-D target detection; no Gazebo model poses or fixed column ordering."""
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformListener, TransformException

from .digit_detector import DigitDetector
from .geometry import depth_median, pose_xyz, row_for_height, transform_matrix, yaw

COLOR_RANGES = {
    'green': [((40, 60, 40), (85, 255, 255))],
    'blue': [((95, 60, 40), (140, 255, 255))],
    'yellow': [((18, 60, 40), (38, 255, 255))],
    'red': [((0, 60, 40), (10, 255, 255)), ((170, 60, 40), (180, 255, 255))],
}
SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


def colour_boxes(image, colour, minimum_area=12):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for lower, upper in COLOR_RANGES[colour]:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= minimum_area]


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        defaults = {
            'shelf_column_number': 1, 'book_colour': 'red', 'perception_rate': 8.0,
            'erc_images_dir': str(Path(__file__).resolve().parents[1] / 'erc_images'),
            'template_dir': str(Path(get_package_share_directory('solution_4to1mux')) / 'templates'),
            'depth_frame': 'head_front_camera_depth_optical_frame',
            'depth_max_skew': 0.20, 'observation_max_age': 0.8,
            'column_half_width': 0.43, 'top_book_marker_offset': 0.665,
            'row_spacing': 0.33, 'bin_min_area_fraction': 0.006,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value.lower()
        if self.target_column not in range(1, 6) or self.target_colour not in COLOR_RANGES:
            raise ValueError('shelf_column_number must be 1..5; book_colour red/green/blue/yellow')
        self.images_dir = Path(self.get_parameter('erc_images_dir').value)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.detector = DigitDetector(template_dir=self.get_parameter('template_dir').value,
                                      min_area=25, match_threshold=0.55)
        self.bridge = CvBridge()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.rgb = self.depth = self.info = None
        self.processed = None
        self.mode = 'shelf'
        self.column_pose = None
        self.candidate_column_pose = None
        self.candidate_column_hits = 0
        self.column_normal_locked = False
        self.column_hits = 0
        self.reported = set()
        self.warned_at = -100.0
        self.latest_book = None
        self.book_hits = 0
        prefix = '/head_front_camera/head_front_camera'
        self.create_subscription(Image, prefix+'/color/image_raw', self.on_rgb, SENSOR_QOS)
        self.create_subscription(Image, prefix+'/depth/image_rect_raw', self.on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, prefix+'/depth/camera_info', self.on_info, SENSOR_QOS)
        self.create_subscription(Bool, '/perception/looking_for_bin', self.on_mode, 10)
        self.ready_pub = self.create_publisher(Bool, '/perception/ready', 10)
        self.shelf_pub = self.create_publisher(PoseStamped, '/perception/shelf_pose', 10)
        self.unit_pub = self.create_publisher(PoseStamped, '/perception/shelf_unit_pose', 10)
        self.book_diag_pub = self.create_publisher(String, '/perception/book_diagnostics', 10)
        self.book_pub = self.create_publisher(PoseStamped, '/perception/book_pose', 10)
        self.bin_pub = self.create_publisher(PoseStamped, '/perception/bin_pose', 10)
        self.column_pub = self.create_publisher(Int32, '/erc/shelf_column_identification', 10)
        self.row_pub = self.create_publisher(Int32, '/erc/shelf_row_identification', 10)
        self.create_timer(1.0/self.get_parameter('perception_rate').value, self.process)
        self.get_logger().info(f'RGB-D perception: column {self.target_column}, {self.target_colour}; latest-frame queue')

    def on_rgb(self, msg):
        self.rgb = msg

    def on_depth(self, msg):
        self.depth = msg

    def on_info(self, msg):
        self.info = msg

    def on_mode(self, msg):
        self.mode = 'bin' if msg.data else 'shelf'

    def warn(self, text):
        now = self.get_clock().now().nanoseconds / 1e9
        if now-self.warned_at > 3.0:
            self.get_logger().warn(text)
            self.warned_at = now

    def point(self, px, py):
        dh, dw = self.depth_image.shape
        h, w = self.image.shape[:2]
        u, v = px*dw/w, py*dh/h
        z = depth_median(self.depth_image, u, v)
        if z is None:
            return None
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        if fx <= 0 or fy <= 0:
            return None
        # Gazebo colour/depth streams share the virtual sensor projection.
        return (self.camera_to_odom @ np.array([(u-cx)*z/fx, (v-cy)*z/fy, z, 1.]))[:3]

    def make_pose(self, xyz, heading=0.0):
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.depth.header.stamp
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, xyz)
        msg.pose.orientation.z = math.sin(heading/2)
        msg.pose.orientation.w = math.cos(heading/2)
        return msg

    def process(self):
        if self.rgb is None or self.depth is None or self.info is None:
            self.warn('Waiting for RGB, depth and depth CameraInfo; check camera topics/QoS.')
            return
        stamp = Time.from_msg(self.rgb.header.stamp)
        key = stamp.nanoseconds
        if key == self.processed:
            return
        depth_stamp = Time.from_msg(self.depth.header.stamp)
        if abs((stamp-depth_stamp).nanoseconds)/1e9 > self.get_parameter('depth_max_skew').value:
            self.warn('RGB/depth timestamps too far apart; waiting for matching frames.')
            return
        if (self.get_clock().now()-stamp).nanoseconds/1e9 > self.get_parameter('observation_max_age').value:
            self.warn('Camera frames are stale; check Gazebo real-time factor and use_sim_time.')
            return
        try:
            tf = self.tf.lookup_transform('odom', self.get_parameter('depth_frame').value, depth_stamp)
            self.camera_to_odom = transform_matrix(tf.transform)
        except TransformException as error:
            self.warn(f'Waiting for camera -> odom TF: {error}')
            return
        self.image = self.bridge.imgmsg_to_cv2(self.rgb, 'bgr8')
        raw = self.bridge.imgmsg_to_cv2(self.depth, 'passthrough')
        self.depth_image = raw.astype(np.float32) / (1000.0 if self.depth.encoding == '16UC1' else 1.0)
        self.processed = key
        self.ready_pub.publish(Bool(data=True))
        if self.mode == 'bin':
            self.process_bin()
        else:
            self.process_shelf()
            self.process_book()

    def process_shelf(self):
        candidates = []
        for digit, score, box in self.detector.detect(self.image):
            x, y, w, h = box
            p = self.point(x+w/2, y+h/2)
            # Markers are overhead; rejecting low shapes avoids reading books as digits.
            if p is not None and 1.95 < p[2] < 2.55:
                candidates.append((digit, score, box, p))
        if not candidates:
            return
        points = np.array([c[3] for c in candidates])
        centre = points.mean(axis=0)
        toward = centre[:2]-self.camera_to_odom[:2, 3]
        normal = toward/max(np.linalg.norm(toward), 1e-6)
        reliable_normal = False
        if len(points) >= 2:
            i, j = np.unravel_index(np.argmax(np.linalg.norm(points[:, None, :2]-points[None, :, :2], axis=2)), (len(points), len(points)))
            tangent = points[i, :2]-points[j, :2]
            if np.linalg.norm(tangent) > 0.5:
                reliable_normal = True
                normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
                if normal @ toward < 0:
                    normal *= -1
        heading = math.atan2(normal[1], normal[0])
        self.unit_pub.publish(self.make_pose(centre, heading))
        target = next((c for c in candidates if c[0] == self.target_column), None)
        if target is None:
            return
        _, score, box, p = target
        if self.confirm_column(p, heading, reliable_normal):
            self.shelf_pub.publish(self.column_pose)
            if 'column' not in self.reported:
                self.column_pub.publish(Int32(data=self.target_column))
                self.save(box, 'shelf_column')
                self.reported.add('column')
                self.get_logger().info(f'Confirmed column {self.target_column} from live image (score {score:.2f})')

    def confirm_column(self, point, heading, reliable_normal):
        # Keep the confirmed reference when the overhead sign leaves the view.
        # One bad crop must not reset book detection just as docking finishes.
        if self.column_pose is not None and np.linalg.norm(point-pose_xyz(self.column_pose)) > 0.20:
            return False
        previous = self.candidate_column_pose
        if previous is not None and np.linalg.norm(point-pose_xyz(previous)) < 0.15:
            self.candidate_column_hits += 1
        else:
            self.candidate_column_hits = 1
        self.candidate_column_pose = self.make_pose(point, heading)
        if self.candidate_column_hits < 3:
            return False
        if self.column_normal_locked and not reliable_normal:
            heading = yaw(self.column_pose.pose.orientation)
        self.column_pose = self.make_pose(point, heading)
        self.column_hits = self.candidate_column_hits
        self.column_normal_locked = self.column_normal_locked or reliable_normal
        return True

    def book_status(self, text):
        self.book_diag_pub.publish(String(data=text))

    def process_book(self):
        if self.column_pose is None:
            self.book_status('waiting_for_confirmed_column')
            return
        marker = pose_xyz(self.column_pose)
        heading = yaw(self.column_pose.pose.orientation)
        normal = np.array([math.cos(heading), math.sin(heading)])
        tangent = np.array([-normal[1], normal[0]])
        candidates = []
        rejected = {'shape': 0, 'depth': 0, 'column': 0, 'plane': 0, 'row': 0}
        plane_offsets = []
        boxes = colour_boxes(self.image, self.target_colour)
        for box in boxes:
            x, y, w, h = box
            if h < 1.3*w:
                rejected['shape'] += 1
                continue
            p = self.point(x+w/2, y+h/2)
            if p is None:
                rejected['depth'] += 1
                continue
            delta = p[:2]-marker[:2]
            if abs(delta @ tangent) > self.get_parameter('column_half_width').value:
                rejected['column'] += 1
                continue
            # Book spines sit on the marker plane (the shelf front). A colour blob
            # measured much deeper is a reflection/gap hit inside the shelf, and
            # accepting it would aim the dock at the shelf front. Deep rejection is
            # strict; the robot side is generous because the confirmed column pose
            # can carry depth bias from far away, and navigation clamps the dock
            # goal against the shelf plane anyway.
            plane_offset = float(delta @ normal)
            if not -0.35 <= plane_offset <= 0.20:
                rejected['plane'] += 1
                plane_offsets.append(round(plane_offset, 2))
                continue
            row = row_for_height(marker[2], p[2], self.get_parameter('top_book_marker_offset').value,
                                 self.get_parameter('row_spacing').value)
            if row is not None:
                candidates.append((box, p, row))
            else:
                rejected['row'] += 1
        # Exactly one target-colour book per column; ambiguity must not select a neighbour.
        if len(candidates) != 1:
            self.book_hits = 0
            detail = f'; plane_offsets_m={plane_offsets}' if plane_offsets else ''
            self.book_status(f'colour={self.target_colour}; blobs={len(boxes)}; valid_in_column={len(candidates)}; rejected={rejected}{detail}')
            return
        box, p, row = candidates[0]
        if self.latest_book is not None and np.linalg.norm(p-self.latest_book) < 0.08:
            self.book_hits += 1
        else:
            self.book_hits = 1
        self.latest_book = p
        if self.book_hits < 3:
            self.book_status(f'confirming_book: {self.book_hits}/3 observations')
            return
        self.book_status(f'book_confirmed: row={row}, xyz={p.round(3).tolist()}')
        self.book_pub.publish(self.make_pose(p, heading))
        if 'row' not in self.reported:
            self.row_pub.publish(Int32(data=row))
            self.save(box, 'book_row')
            self.reported.add('row')
            self.get_logger().info(f'Confirmed {self.target_colour} book in target column, row {row}, xyz={p.round(3)}')

    def process_bin(self):
        frame_area = self.image.shape[0]*self.image.shape[1]
        candidates = []
        for box in colour_boxes(self.image, 'red', frame_area*self.get_parameter('bin_min_area_fraction').value):
            x, y, w, h = box
            if w < 1.15*h:
                continue
            p = self.point(x+w/2, y+h/2)
            if p is not None and 0.60 < p[2] < 1.20:
                candidates.append((w*h, box, p))
        if not candidates:
            return
        _, box, p = max(candidates, key=lambda c: c[0])
        d = p[:2]-self.camera_to_odom[:2, 3]
        self.bin_pub.publish(self.make_pose(p, math.atan2(d[1], d[0])))
        if 'bin' not in self.reported:
            self.save(box, 'collection_bin')
            self.reported.add('bin')

    def save(self, box, label):
        x, y, w, h = box
        image = self.image.copy()
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        cv2.putText(image, timestamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        if not cv2.imwrite(str(self.images_dir/f'{label}_{timestamp}.png'), image):
            self.get_logger().error(f'Could not write annotated image to {self.images_dir}')


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
