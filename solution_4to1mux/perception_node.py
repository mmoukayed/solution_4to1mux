"""Live RGB-D target detection; no Gazebo model poses or fixed column ordering."""
import csv
import io
import math
import shutil
import subprocess
from collections import deque
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
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformListener, TransformException

from .digit_detector import DigitDetector
from .ros_topics import configure_ros
from .geometry import depth_median, pose_xyz, row_for_height, transform_matrix, yaw

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


# Hue/saturation bands measured on 60 labelled real-robot frames (slots 1-2, MBR library): the old
# wide bands found every book but also 970 false boxes (wood panelling and floor read red, the
# library's book wall blue). STRICT is the measured band with a margin; RELAXED is for a dark frame
# after a camera restart flips the exposure preset (books read dull: saturation 56-90, red ~orange).
STRICT = {'red': [((0, 100, 80), (8, 255, 255)), ((172, 100, 80), (180, 255, 255))],
          'green': [((58, 80, 50), (80, 255, 255))], 'yellow': [((16, 100, 110), (28, 255, 255))],
          'blue': [((93, 140, 80), (108, 255, 255))]}
RELAXED = {'red': [((0, 45, 45), (16, 255, 255)), ((168, 45, 45), (180, 255, 255))],
           'green': [((48, 45, 40), (85, 255, 255))], 'yellow': [((16, 45, 70), (36, 255, 255))],
           'blue': [((90, 50, 40), (112, 255, 255))]}


def normalise_exposure(image, mean_lo=110, sd_lo=40):
    """A dark or flat frame (the robot's exposure preset after a camera restart): grey-world white
    balance on the brightest unsaturated pixels (the white shelf), then stretch the shelf to white.
    A normal frame is returned unchanged."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= mean_lo and gray.std() >= sd_lo:
        return image
    f = image.astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ref = (hsv[..., 1] < 90) & (gray > np.percentile(gray, 60))
    if ref.sum() < 500:
        ref = gray > np.percentile(gray, 80)
    m = f[ref].mean(axis=0)
    f = f*(m.mean()/np.maximum(m, 1))
    f = f*(235.0/max(np.percentile(f[ref].mean(axis=1), 90), 1))
    return np.clip(f, 0, 255).astype(np.uint8)


def _raw_boxes(image, bands, minimum_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for lower, upper in bands:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= minimum_area], hsv


def _in_white_bay(hsv, boxes, need=0.55, max_saturation=60, min_side=7):
    """A competition book stands in a white bay: most of a ring around it is unsaturated. Wood,
    people and the library's own book wall are not (real frames: 546 -> 16 false red boxes)."""
    h_img, w_img = hsv.shape[:2]
    kept = []
    for (x, y, w, h) in boxes:
        if min(w, h) < min_side:
            continue
        pad = max(4, int(0.6*min(w, h)))
        x0, y0 = max(0, x-pad), max(0, y-pad//2)
        x1, y1 = min(w_img, x+w+pad), min(h_img, y+h+pad//2)
        ring = hsv[y0:y1, x0:x1, 1].astype(np.int32)
        inner = np.zeros(ring.shape, bool)
        inner[y-y0:y-y0+h, x-x0:x-x0+w] = True
        values = ring[~inner]
        if values.size and (values < max_saturation).mean() >= need:
            kept.append((x, y, w, h))
    return kept


def colour_boxes(image, colour, minimum_area=60, books=True):
    """Boxes of `colour`. Books: the strict band, kept only inside a white bay; on a dark frame
    with nothing found, the relaxed band (stricter ring). The bin (books=False): the strict band."""
    dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() < 110
    image = normalise_exposure(image)
    boxes, hsv = _raw_boxes(image, STRICT[colour], minimum_area)
    if not books:
        return boxes
    kept = _in_white_bay(hsv, boxes)
    if not kept and dark:
        boxes, hsv = _raw_boxes(image, RELAXED[colour], minimum_area)
        kept = _in_white_bay(hsv, boxes, need=0.65)
    return kept


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.ros = configure_ros(self)
        topics = self.ros.topics
        defaults = {
            'shelf_column_number': 1, 'book_colour': 'red', 'book_text': '',
            'perception_rate': 8.0, 'book_text_ocr_period': 0.45,
            'erc_images_dir': str(Path(__file__).resolve().parents[1] / 'erc_images'),
            'template_dir': str(Path(get_package_share_directory('solution_4to1mux')) / 'templates'),
            # registered depth is in the colour camera's pixels: project from the colour frame
            'depth_frame': self.ros.frames['colour_camera'],
            'depth_max_skew': 0.20, 'observation_max_age': 0.8,
            'tf_latest_fallback_tolerance': 0.10,
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
        if self.target_column not in range(1, 6) or (not self.target_text and self.target_colour not in STRICT):
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
        self.depth_buffer = deque(maxlen=12)
        self.depth_infos = {}
        self.last_pair = None
        self.head_tilt = 0.0            # measured head pitch error (rad), from the floor
        self.tilt_samples = 0
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
        
        self.create_subscription(Image, topics['camera_rgb'], self.on_rgb, SENSOR_QOS)
        # Raw depth, registered to colour here (depth intrinsics + the depth->colour TF): the robot's
        # depth comes from its own camera (wider view) and at two sizes on one topic (848x480 and
        # the decimated 172x96, slot 3); scaling pixel coordinates by the size ratio is wrong for both.
        self.create_subscription(Image, topics['camera_depth_raw'], self.on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, topics['camera_depth_raw'].rsplit('/', 1)[0]+'/camera_info',
                                 self.on_depth_info, SENSOR_QOS)
        self.create_subscription(CameraInfo, topics['camera_info'], self.on_info, SENSOR_QOS)
        
        self.create_subscription(Bool, topics['looking_for_bin'], self.on_mode, 10)
        self.ready_pub = self.create_publisher(Bool, topics['perception_ready'], 10)
        self.shelf_pub = self.create_publisher(PoseStamped, topics['shelf_pose'], 10)
        self.unit_pub = self.create_publisher(PoseStamped, topics['shelf_unit_pose'], 10)
        self.book_diag_pub = self.create_publisher(String, topics['book_diagnostics'], 10)
        self.book_pub = self.create_publisher(PoseStamped, topics['book_pose'], 10)
        self.bin_pub = self.create_publisher(PoseStamped, topics['bin_pose'], 10)
        self.column_pub = self.create_publisher(Int32, topics['shelf_column_identification'], 10)
        self.row_pub = self.create_publisher(Int32, topics['shelf_row_identification'], 10)
        self.create_timer(1.0/self.get_parameter('perception_rate').value, self.process)
        self.get_logger().info(
            f'Perception BUILD 5.13: column {self.target_column}, '
            f'{"text="+self.target_text if self.target_text else self.target_colour}; '
            f'marker height {self.get_parameter("marker_min_height").value:.2f}-'
            f'{self.get_parameter("marker_max_height").value:.2f} m')

    def on_rgb(self, msg):
        self.rgb = msg

    def on_depth(self, msg):
        self.depth = msg
        self.depth_buffer.append(msg)

    def on_depth_info(self, msg):
        self.depth_infos[(msg.width, msg.height)] = msg

    def pick_depth(self, stamp):
        """The depth frame nearest the colour frame: full-size preferred, a decimated one only when
        no full frame is near; the window widens when nothing has paired for a while (the robot's
        camera drops to a few Hz)."""
        now = self.get_clock().now().nanoseconds/1e9
        wide = self.last_pair is None or now-self.last_pair > 5.0
        limit = 0.6 if wide else self.get_parameter('depth_max_skew').value
        best = None
        for msg in self.depth_buffer:
            dt = abs((stamp-Time.from_msg(msg.header.stamp)).nanoseconds)/1e9
            if dt > limit:
                continue
            score = dt+(0.0 if msg.width >= 400 else 0.3)
            if best is None or score < best[0]:
                best = (score, msg)
        return None if best is None else best[1]

    def depth_K(self, msg):
        info = self.depth_infos.get((msg.width, msg.height))
        if info is not None:
            return np.array(info.k, float).reshape(3, 3)
        full = [i for (w, h), i in self.depth_infos.items() if w >= 400]
        if not full:
            return None
        K = np.array(full[0].k, float).reshape(3, 3).copy()
        K[:2] *= msg.width/float(full[0].width)
        return K

    def register(self, depth_m, K_d, T_cd, K_c, width, height):
        """Depth (its own camera) -> a depth image in the colour camera's pixels, nearest surface
        per pixel, small holes filled from neighbours."""
        step = 2 if depth_m.shape[1] >= 400 else 1
        d = depth_m[::step, ::step]
        v, u = np.nonzero(np.isfinite(d) & (d > 0.15) & (d < 8.0))
        z = d[v, u]
        u, v = u*step, v*step
        pts = np.stack([(u-K_d[0, 2])*z/K_d[0, 0], (v-K_d[1, 2])*z/K_d[1, 1], z])
        P = T_cd[:3, :3] @ pts + T_cd[:3, 3:4]
        ok = P[2] > 0.15
        P = P[:, ok]
        uc = np.round(K_c[0, 0]*P[0]/P[2]+K_c[0, 2]).astype(int)
        vc = np.round(K_c[1, 1]*P[1]/P[2]+K_c[1, 2]).astype(int)
        inside = (uc >= 0) & (uc < width) & (vc >= 0) & (vc < height)
        uc, vc, zc = uc[inside], vc[inside], P[2, inside]
        order = np.argsort(-zc)                 # far first, so the nearest surface is written last
        reg = np.zeros((height, width), np.float32)
        reg[vc[order], uc[order]] = zc[order]
        k = max(3, int(round(width/depth_m.shape[1]*step))*2+1)
        filled = cv2.erode(np.where(reg > 0, reg, 1e3).astype(np.float32), np.ones((k, k), np.uint8))
        reg = np.where((reg == 0) & (filled < 1e3), filled, reg)
        return reg

    def measure_tilt(self, depth_m, K_d, T_od):
        """The head's pitch error from the floor: the floor is z = 0 in odom, so the camera rotation
        about its x axis that puts the most floor points on it is the encoder's error (the real
        robot read its head tilt 4.3 deg off at slot 3). Smoothed; only with enough floor seen."""
        step = 4 if depth_m.shape[1] >= 400 else 1
        d = depth_m[::step, ::step]
        v, u = np.nonzero(np.isfinite(d) & (d > 0.4) & (d < 3.5))
        if v.size < 400:
            return
        z = d[v, u]
        pts = np.stack([(u*step-K_d[0, 2])*z/K_d[0, 0], (v*step-K_d[1, 2])*z/K_d[1, 1], z])
        world = T_od[:3, :3] @ pts + T_od[:3, 3:4]
        near_floor = np.abs(world[2]) < 0.25
        if near_floor.sum() < 400:
            return
        pts = pts[:, near_floor]
        best = (0, 0.0)
        for delta in np.arange(-0.14, 0.1401, 0.002):
            c, s_ = math.cos(delta), math.sin(delta)
            R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
            zz = (T_od[2, :3] @ (R @ pts)) + T_od[2, 3]
            n = int((np.abs(zz) < 0.02).sum())
            if n > best[0]:
                best = (n, delta)
        if best[0] < 300:
            return
        a = 0.3 if self.tilt_samples > 3 else 0.6
        self.head_tilt = (1-a)*self.head_tilt+a*best[1]
        self.tilt_samples += 1
        if self.tilt_samples in (1, 5, 20) or self.tilt_samples % 50 == 0:
            self.get_logger().info(f'Head tilt from the floor: {math.degrees(self.head_tilt):+.2f} deg '
                                   f'({best[0]} floor points, sample {self.tilt_samples})')

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
        msg.header.frame_id = self.ros.frames['odom']
        msg.header.stamp = self.depth.header.stamp
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, xyz)
        msg.pose.orientation.z = math.sin(heading/2)
        msg.pose.orientation.w = math.cos(heading/2)
        return msg

    def process(self):
        if self.rgb is None or self.depth is None or self.info is None:
            missing = [label for label, message in (
                ('RGB', self.rgb), ('depth', self.depth), ('CameraInfo', self.info))
                if message is None]
            self.warn(
                f'Waiting for {", ".join(missing)}; verify the subscribed topic names '
                f'and QoS (RGB={self.ros.topics["camera_rgb"]}, '
                f'depth={self.ros.topics["camera_depth"]}, '
                f'CameraInfo={self.ros.topics["camera_info"]}).')
            return
        stamp = Time.from_msg(self.rgb.header.stamp)
        key = stamp.nanoseconds
        if key == self.processed:
            return
        depth_msg = self.pick_depth(stamp)
        if depth_msg is None:
            self.warn('No depth frame near this colour frame yet; waiting for matching frames.')
            return
        K_d = self.depth_K(depth_msg)
        if K_d is None:
            self.warn('Waiting for the depth camera_info.')
            return
        depth_stamp = Time.from_msg(depth_msg.header.stamp)
        self.last_pair = self.get_clock().now().nanoseconds/1e9
        if (self.get_clock().now()-stamp).nanoseconds/1e9 > self.get_parameter('observation_max_age').value:
            self.warn('Camera frames are stale; check Gazebo real-time factor and use_sim_time.')
            return
        target_frame = self.ros.frames['odom']
        camera_frame = self.get_parameter('depth_frame').value
        try:
            tf = self.tf.lookup_transform(target_frame, camera_frame, stamp)
        except TransformException as error:
            # Sensor images can be stamped a few milliseconds after the most
            # recent /tf sample. Use the latest transform only when it is very
            # close in time; otherwise keep waiting for correctly aligned TF.
            try:
                latest = self.tf.lookup_transform(target_frame, camera_frame, Time())
            except TransformException:
                self.warn(f'Waiting for {camera_frame} -> {target_frame} TF: {error}')
                return
            lag = (stamp-Time.from_msg(latest.header.stamp)).nanoseconds/1e9
            tolerance = self.get_parameter('tf_latest_fallback_tolerance').value
            if not 0.0 <= lag <= tolerance:
                self.warn(f'Waiting for {camera_frame} -> {target_frame} TF at image time; '
                          f'latest transform differs by {lag*1000:.1f} ms: {error}')
                return
            tf = latest
            self.warn(f'Image is {lag*1000:.1f} ms ahead of TF; using the latest transform.')
        try:
            T_od = transform_matrix(self.tf.lookup_transform(
                target_frame, depth_msg.header.frame_id, depth_stamp, timeout=Duration(seconds=0.05)).transform)
        except TransformException as error:
            self.warn(f'Waiting for the depth camera TF: {error}')
            return
        raw = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = raw.astype(np.float32)/(1000.0 if depth_msg.encoding == '16UC1' else 1.0)
        self.measure_tilt(depth_m, K_d, T_od)
        c, s_ = math.cos(self.head_tilt), math.sin(self.head_tilt)
        tilt = np.eye(4)
        tilt[1:3, 1:3] = [[c, -s_], [s_, c]]
        self.camera_to_odom = transform_matrix(tf.transform) @ tilt
        # depth -> colour through odom, each at its own stamp: the pair may be up to 0.6 s apart while
        # the base turns, and a fixed depth->colour transform would smear every point by that motion
        T_cd = np.linalg.inv(self.camera_to_odom) @ (T_od @ tilt)
        self.image = self.bridge.imgmsg_to_cv2(self.rgb, 'bgr8')
        K_c = np.array(self.info.k, float).reshape(3, 3)
        self.depth_image = self.register(depth_m, K_d, T_cd, K_c, self.image.shape[1], self.image.shape[0])
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
        for box in colour_boxes(self.image, 'red', frame_area*self.get_parameter('bin_min_area_fraction').value, books=False):
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
