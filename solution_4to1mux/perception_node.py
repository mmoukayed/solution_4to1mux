"""Live RGB-D target detection; no Gazebo model poses or fixed column ordering."""
import csv
import io
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
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


def normalized_label(text):
    return ''.join(character for character in text.casefold() if character.isalnum())


def label_matches(candidate, target):
    candidate, target = normalized_label(candidate), normalized_label(target)
    if candidate == target:
        return True
    return (len(target) >= 5 and abs(len(candidate)-len(target)) <= 1
            and SequenceMatcher(None, candidate, target).ratio() >= 0.90)


def read_text_label_boxes(crop, target):
    """Read exact target labels printed vertically along book spines.

    Tesseract runs in a worker thread so book search continues while OCR is
    processing. The returned boxes use coordinates in the unrotated crop.
    """
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.equalizeHist(gray)
    scale = 3
    matches = []
    rotations = (
        ('ccw', cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ('cw', cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
    )
    for direction, rotated in rotations:
        enlarged = cv2.resize(rotated, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        success, encoded = cv2.imencode('.png', enlarged)
        if not success:
            continue
        result = subprocess.run(
            ['tesseract', 'stdin', 'stdout', '--psm', '11', '-l', 'eng', 'tsv'],
            input=encoded.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=1.5)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors='replace').strip()
                               or 'Tesseract returned a non-zero status.')
        lines = {}
        reader = csv.DictReader(io.StringIO(result.stdout.decode(errors='replace')),
                                delimiter='\t')
        for word in reader:
            if word.get('level') != '5' or not word.get('text', '').strip():
                continue
            try:
                confidence = float(word['conf'])
                x, y = int(word['left']), int(word['top'])
                width, height = int(word['width']), int(word['height'])
                key = tuple(int(word[name]) for name in
                            ('page_num', 'block_num', 'par_num', 'line_num'))
            except (KeyError, TypeError, ValueError):
                continue
            if confidence < 18:
                continue
            group = lines.setdefault(key, {'words': [], 'confidence': [],
                                           'left': x, 'top': y,
                                           'right': x+width, 'bottom': y+height})
            group['words'].append((int(word['word_num']), word['text']))
            group['confidence'].append(confidence)
            group['left'] = min(group['left'], x)
            group['top'] = min(group['top'], y)
            group['right'] = max(group['right'], x+width)
            group['bottom'] = max(group['bottom'], y+height)
        for group in lines.values():
            text = ''.join(word for _, word in sorted(group['words']))
            if not label_matches(text, target):
                continue
            left = group['left']/scale
            top = group['top']/scale
            width = (group['right']-group['left'])/scale
            height = (group['bottom']-group['top'])/scale
            crop_height, crop_width = gray.shape
            if direction == 'ccw':
                box = (crop_width-top-height, left, height, width)
            else:
                box = (top, crop_height-left-width, height, width)
            confidence = float(np.mean(group['confidence']))
            matches.append((tuple(int(round(value)) for value in box), confidence, text))
    return matches


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
            'shelf_column_number': 1, 'book_colour': 'red', 'book_text': '',
            'perception_rate': 8.0, 'book_text_ocr_period': 0.45,
            'erc_images_dir': str(Path(__file__).resolve().parents[1] / 'erc_images'),
            'template_dir': str(Path(get_package_share_directory('solution_4to1mux')) / 'templates'),
            'depth_frame': 'head_front_camera_depth_optical_frame',
            'depth_max_skew': 0.20, 'observation_max_age': 0.8,
            # Current simulator: marker centre 1.93 m, top active book 1.595 m.
            'marker_min_height': 1.75, 'marker_max_height': 2.55,
            'column_half_width': 0.43, 'top_book_marker_offset': 0.335,
            'row_spacing': 0.33, 'bin_min_area_fraction': 0.006,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value.lower()
        self.target_text = str(self.get_parameter('book_text').value).strip()
        if self.target_column not in range(1, 6) or (not self.target_text and self.target_colour not in COLOR_RANGES):
            raise ValueError('shelf_column_number must be 1..5; set book_colour to red/green/blue/yellow or provide book_text')
        if self.target_text and shutil.which('tesseract') is None:
            raise RuntimeError('book_text requires Tesseract OCR. Install it in the simulator container with: apt install tesseract-ocr')
        self.images_dir = Path(self.get_parameter('erc_images_dir').value)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        template_dir = Path(self.get_parameter('template_dir').value)
        source_templates = Path(__file__).resolve().parents[1] / 'templates'
        complete_templates = lambda path: all((path / f'{digit}.png').is_file()
                                              for digit in range(1, 6))
        if not complete_templates(template_dir) and complete_templates(source_templates):
            self.get_logger().warn(
                f'Installed shelf templates are incomplete at {template_dir}; '
                f'using source templates at {source_templates}.')
            template_dir = source_templates
        self.detector = DigitDetector(template_dir=str(template_dir),
                                      min_area=25, match_threshold=0.55)
        if self.detector.using_synthetic:
            missing = ', '.join(map(str, sorted(set(range(1, 6))-
                                                self.detector.real_template_digits)))
            self.get_logger().error(
                f'Shelf digit detector is using synthetic templates; missing {missing} '
                f'in {template_dir}. Correct shelf recognition may fail.')
        else:
            self.get_logger().info(f'Loaded all five shelf digit templates from {template_dir}.')
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
        self.ocr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='book-text-ocr')
        self.ocr_future = None
        self.ocr_pending_context = None
        self.ocr_result_context = None
        self.ocr_boxes = []
        self.ocr_sequence = 0
        self.ocr_handled = 0
        self.ocr_submit_at = -100.0
        
        self.create_subscription(Image, '/head_front_camera/color/image_raw', self.on_rgb, SENSOR_QOS)
        self.create_subscription(Image, '/head_front_camera/depth/image_rect_raw', self.on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, '/head_front_camera/color/camera_info', self.on_info, SENSOR_QOS)
        
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
        self.get_logger().info(
            f'Perception BUILD 5.11: column {self.target_column}, '
            f'{"text="+self.target_text if self.target_text else self.target_colour}; '
            f'marker height {self.get_parameter("marker_min_height").value:.2f}-'
            f'{self.get_parameter("marker_max_height").value:.2f} m')

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
        return self.point_in_frame(px, py, self.image.shape[:2], self.depth_image,
                                   self.camera_to_odom, self.info)

    @staticmethod
    def point_in_frame(px, py, image_shape, depth_image, camera_to_odom, info):
        dh, dw = depth_image.shape
        h, w = image_shape
        u, v = px*dw/w, py*dh/h
        z = depth_median(depth_image, u, v)
        if z is None:
            return None
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        if fx <= 0 or fy <= 0:
            return None
        # Gazebo colour/depth streams share the virtual sensor projection.
        return (camera_to_odom @ np.array([(u-cx)*z/fx, (v-cy)*z/fy, z, 1.]))[:3]

    def text_search_crop(self):
        """Project the confirmed column and four book rows into the RGB image."""
        marker = pose_xyz(self.column_pose)
        heading = yaw(self.column_pose.pose.orientation)
        tangent = np.array([-math.sin(heading), math.cos(heading)])
        top = marker[2]-self.get_parameter('top_book_marker_offset').value
        spacing = self.get_parameter('row_spacing').value
        bottom = top-3*spacing
        inverse = np.linalg.inv(self.camera_to_odom)
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        pixels = []
        for lateral in (-0.43, 0.43):
            for height in (top+0.18, bottom-0.18):
                world = np.r_[marker[:2]+lateral*tangent, height, 1.0]
                camera = inverse @ world
                if camera[2] <= 0.1:
                    continue
                pixels.append((fx*camera[0]/camera[2]+cx,
                               fy*camera[1]/camera[2]+cy))
        if len(pixels) < 2:
            return None
        image_height, image_width = self.image.shape[:2]
        margin = 12
        x0 = max(0, int(math.floor(min(point[0] for point in pixels)))-margin)
        y0 = max(0, int(math.floor(min(point[1] for point in pixels)))-margin)
        x1 = min(image_width, int(math.ceil(max(point[0] for point in pixels)))+margin)
        y1 = min(image_height, int(math.ceil(max(point[1] for point in pixels)))+margin)
        if x1-x0 < 12 or y1-y0 < 12:
            return None
        return self.image[y0:y1, x0:x1].copy(), (x0, y0)

    def update_text_search(self):
        if self.ocr_future is not None and self.ocr_future.done():
            try:
                self.ocr_boxes = self.ocr_future.result()
            except Exception as error:
                self.ocr_boxes = []
                self.warn('book_text_ocr', f'Text OCR failed: {error}')
            self.ocr_result_context = self.ocr_pending_context
            self.ocr_sequence += 1
            self.ocr_future = None
        if (self.ocr_future is not None or
                self.get_clock().now().nanoseconds/1e9-self.ocr_submit_at <
                self.get_parameter('book_text_ocr_period').value):
            return
        search = self.text_search_crop()
        if search is None:
            return
        crop, origin = search
        self.ocr_pending_context = {
            'origin': origin,
            'image_shape': self.image.shape[:2],
            'depth_image': self.depth_image.copy(),
            'camera_to_odom': self.camera_to_odom.copy(),
            'info': self.info,
        }
        self.ocr_future = self.ocr_pool.submit(read_text_label_boxes, crop,
                                               self.target_text)
        self.ocr_submit_at = self.get_clock().now().nanoseconds/1e9

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
        detections = self.detector.detect(self.image)
        candidates = []
        rejected = {'depth': 0, 'height': 0}
        measured_heights = []
        min_height = self.get_parameter('marker_min_height').value
        max_height = self.get_parameter('marker_max_height').value
        for digit, score, box in detections:
            x, y, w, h = box
            p = self.point(x+w/2, y+h/2)
            if p is None or not np.isfinite(p).all():
                rejected['depth'] += 1
                continue
            measured_heights.append((digit, round(float(p[2]), 3)))
            # Reject book spines while allowing the current 1.93 m marker mount
            # and the older 2.26 m mounting height.
            if min_height < p[2] < max_height:
                candidates.append((digit, score, box, p))
            else:
                rejected['height'] += 1
        if not candidates:
            labels = sorted({digit for digit, _, _ in detections})
            if labels:
                self.warn(
                    f'Shelf search recognized labels {labels}; rejected={rejected}; '
                    f'heights_m={measured_heights}; allowed={min_height:.2f}-{max_height:.2f} m.')
            else:
                self.warn('Shelf search cannot recognize any number markers in the current camera view.')
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
        # A single sign only gives its bearing from the camera, not the shelf
        # normal. Navigation uses this pose to choose a lateral scan direction,
        # so publish it only when two separated signs establish the shelf line.
        if reliable_normal:
            self.unit_pub.publish(self.make_pose(centre, heading))
        target = next((c for c in candidates if c[0] == self.target_column), None)
        if target is None:
            labels = sorted({candidate[0] for candidate in candidates})
            if reliable_normal:
                self.warn(f'Shelf search sees columns {labels}; waiting for requested column {self.target_column}.')
            else:
                self.warn(f'Shelf search sees labels {labels} but needs two separated markers to estimate the shelf direction; continuing the rotational search.')
            return
        _, score, box, p = target
        if self.confirm_column(p, heading, reliable_normal):
            self.shelf_pub.publish(self.column_pose)
            if 'column' not in self.reported:
                self.column_pub.publish(Int32(data=self.target_column))
                self.save(box, 'shelf_column')
                self.reported.add('column')
                self.get_logger().info(f'Confirmed column {self.target_column} from live image (score {score:.2f})')
        else:
            self.warn(f'Requested column {self.target_column} is visible; stabilizing its measured pose.')

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
        if self.target_text:
            self.update_text_search()
            if self.ocr_sequence == self.ocr_handled:
                return
            self.ocr_handled = self.ocr_sequence
            boxes = self.ocr_boxes
            context = self.ocr_result_context
            if context is None:
                self.book_hits = 0
                return
            x0, y0 = context['origin']
            for local_box, confidence, recognized in boxes:
                x, y, w, h = local_box
                box = (x+x0, y+y0, w, h)
                p = self.point_in_frame(
                    box[0]+w/2, box[1]+h/2, context['image_shape'],
                    context['depth_image'], context['camera_to_odom'], context['info'])
                if p is None:
                    rejected['depth'] += 1
                    continue
                candidates.append((box, p, None, confidence, recognized))
        else:
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
                candidates.append((box, p, None, 100.0, self.target_colour))

        valid = []
        for box, p, _, confidence, recognized in candidates:
            delta = p[:2]-marker[:2]
            if abs(delta @ tangent) > self.get_parameter('column_half_width').value:
                rejected['column'] += 1
                continue
            # Book spines sit on the marker plane (the shelf front). A candidate
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
                valid.append((box, p, row, confidence, recognized))
            else:
                rejected['row'] += 1
        # The two 90-degree OCR passes can report the same spine. Merge those
        # boxes by measured 3D position, while preserving genuine ambiguity.
        valid.sort(key=lambda candidate: candidate[3], reverse=True)
        candidates = []
        for candidate in valid:
            if not any(np.linalg.norm(candidate[1]-kept[1]) < 0.12 for kept in candidates):
                candidates.append(candidate)
        if len(candidates) != 1:
            self.book_hits = 0
            detail = f'; plane_offsets_m={plane_offsets}' if plane_offsets else ''
            target = f'text={self.target_text!r}' if self.target_text else f'colour={self.target_colour}'
            self.book_status(f'{target}; candidates={len(boxes)}; valid_in_column={len(candidates)}; rejected={rejected}{detail}')
            return
        box, p, row, confidence, recognized = candidates[0]
        if self.latest_book is not None and np.linalg.norm(p-self.latest_book) < 0.08:
            self.book_hits += 1
        else:
            self.book_hits = 1
        self.latest_book = p
        required_hits = 2 if self.target_text else 3
        if self.book_hits < required_hits:
            self.book_status(f'confirming_book: {self.book_hits}/{required_hits} observations')
            return
        target = f'text={self.target_text!r}' if self.target_text else f'colour={self.target_colour}'
        self.book_status(f'book_confirmed: {target}, row={row}, xyz={p.round(3).tolist()}')
        self.book_pub.publish(self.make_pose(p, heading))
        if 'row' not in self.reported:
            self.row_pub.publish(Int32(data=row))
            self.save(box, 'book_row')
            self.reported.add('row')
            self.get_logger().info(
                f'Confirmed {target} book in target column, row {row}, '
                f'confidence={confidence:.0f}, recognized={recognized!r}, xyz={p.round(3)}')

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
