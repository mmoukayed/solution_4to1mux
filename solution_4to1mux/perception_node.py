"""Live RGB-D target detection; no Gazebo model poses or fixed column ordering."""
import math
import json
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformListener, TransformException

from .text_detection import normalized_label, label_matches, read_text_label_boxes, WHITE_BOOK_TITLES
from .book_geometry import measure_book, divider_bounds
from .target_geometry import masked_surface
from .digit_detector import DigitDetector
from .ros_topics import configure_ros
from .alignment import angle_delta, fit_shelf_line
from .geometry import depth_median, pose_xyz, row_for_height, transform_matrix, yaw

SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


# Bands apply to the unmodified camera feed. The November live frames include
# olive green (H=41..43) and pale blue (S=90), which the previous bands excluded.
# Shelf context, spine shape and the confirmed 3D bay constrain book candidates;
# bins use their own stricter mask below and never these relaxed book bands.
STRICT = {'red': [((0, 80, 70), (10, 255, 255)), ((170, 80, 70), (180, 255, 255))],
          'green': [((35, 45, 45), (85, 255, 255))], 'yellow': [((16, 70, 90), (32, 255, 255))],
          'blue': [((90, 60, 60), (120, 255, 255))]}
RELAXED = {'red': [((0, 45, 45), (16, 255, 255)), ((168, 45, 45), (180, 255, 255))],
           'green': [((35, 35, 40), (85, 255, 255))], 'yellow': [((16, 45, 70), (32, 255, 255))],
           'blue': [((90, 45, 40), (120, 255, 255))]}


def _raw_boxes(image, bands, minimum_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], np.uint8)
    for lower, upper in bands:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= minimum_area], hsv


def _in_white_bay(hsv, boxes, need=0.55, max_saturation=90, min_side=4):
    """Require a mostly neutral shelf surround, allowing the live feed's pink cast."""
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
    """Detect original camera colours without white balance or brightness stretching.

    Books require a white bay; dark frames may use relaxed thresholds on the
    same unmodified pixels. books=False is also used for the held-book check.
    """
    dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() < 110
    boxes, hsv = _raw_boxes(image, STRICT[colour], minimum_area)
    if not books:
        return boxes
    kept = _in_white_bay(hsv, boxes)
    if not kept and dark:
        boxes, hsv = _raw_boxes(image, RELAXED[colour], minimum_area)
        kept = _in_white_bay(hsv, boxes, need=0.65)
    return kept


def bin_colour_candidates(image, minimum_area):
    """Require a broad, mostly saturated red surface instead of a red contour alone."""
    bands = [((0, 150, 70), (8, 255, 255)), ((172, 150, 70), (180, 255, 255))]
    boxes, hsv = _raw_boxes(image, bands, minimum_area)
    mask = np.zeros(image.shape[:2], np.uint8)
    for lower, upper in bands:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    kept = []
    for x, y, w, h in boxes:
        if w < 1.15*h:
            continue
        if np.count_nonzero(mask[y:y+h, x:x+w]) / float(w*h) < 0.60:
            continue
        kept.append((x, y, w, h))
    return kept, mask


def bin_surface_point(box, mask, depth, camera_to_odom, info):
    """Use depth supported by red pixels, excluding holes and background at the box centre."""
    x, y, w, h = box
    # register() produces depth at the RGB resolution. Do not guess a mapping.
    if depth.shape != mask.shape:
        return None
    red = cv2.erode(mask[y:y+h, x:x+w], np.ones((3, 3), np.uint8)) > 0
    roi = depth[y:y+h, x:x+w]
    valid = red & np.isfinite(roi) & (roi > 0.15) & (roi <= 3.0)
    if valid.sum() < max(20, 0.20*red.sum()):
        return None
    z = float(np.median(roi[valid]))
    surface = valid & (np.abs(roi-z) <= 0.15)
    if surface.sum() < 0.60*valid.sum():
        return None
    height, width = mask.shape
    fx = info.k[0]*width/(info.width or width)
    fy = info.k[4]*height/(info.height or height)
    cx = info.k[2]*width/(info.width or width)
    cy = info.k[5]*height/(info.height or height)
    if fx <= 0 or fy <= 0 or not (0.15 <= w*z/fx <= 1.0 and 0.08 <= h*z/fy <= 0.60):
        return None
    rows, cols = np.nonzero(surface)
    values = roi[surface]
    camera_point = np.array([np.median((cols+x-cx)*values/fx),
                             np.median((rows+y-cy)*values/fy), np.median(values), 1.])
    return (camera_to_odom @ camera_point)[:3]


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        cv2.setNumThreads(1)
        self.ros = configure_ros(self)
        topics = self.ros.topics
        defaults = {
            'shelf_column_number': 1, 'book_colour': 'red', 'book_text': '',
            'perception_rate': 8.0, 'book_text_ocr_period': 0.45, 'book_text_colour_hint': '',
            'erc_images_dir': str(Path.cwd() / 'erc_images'),
            'template_dir': str(Path(get_package_share_directory('solution_4to1mux')) / 'templates'),
            # registered depth is in the colour camera's pixels: project from the colour frame
            'depth_frame': self.ros.frames['colour_camera'],
            'depth_max_skew': 0.20, 'observation_max_age': 0.8,
            'tf_latest_fallback_tolerance': 0.10,
            # Current simulator: marker centre 1.93 m, top active book 1.595 m.
            'marker_min_height': 1.60 if self.ros.profile == 'real' else 1.75, 'marker_max_height': 2.55,
            'column_half_width': 0.43, 'top_book_marker_offset': 0.335,
            'row_spacing': 0.33, 'bin_min_area_fraction': 0.002,
            'floor_calibration_period': 1.0,
            # the gripper's visible centroid in its own frame (calibrated on the fault-free twin)
            'hand_frame': 'gripper_right_base_link', 'hand_bias': [-0.038, 0.020, -0.116],
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        for key in ('perception_rate', 'row_spacing', 'column_half_width'):
            value = self.get_parameter(key).value
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        bounds = [self.get_parameter(key).value for key in ('marker_min_height', 'marker_max_height')]
        if not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError('Marker height bounds must be finite, positive and ordered.')
        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value.lower()
        self.target_text = str(self.get_parameter('book_text').value).strip()
        if self.target_column not in range(1, 6) or (not self.target_text and self.target_colour not in STRICT):
            raise ValueError('shelf_column_number must be 1..5; set book_colour to red/green/blue/yellow or provide book_text')
        if self.target_text and normalized_label(self.target_text) not in [normalized_label(t) for t in WHITE_BOOK_TITLES]:
            raise ValueError('book_text must be ROBOT, LIBRARY, AGENT or FUTURE')
        self.text_colour_hint = str(self.get_parameter('book_text_colour_hint').value).strip().lower()
        if self.text_colour_hint and self.text_colour_hint not in STRICT:
            raise ValueError('book_text_colour_hint must be empty or red/green/blue/yellow')
        if self.target_text and len(normalized_label(self.target_text)) < 2:
            raise ValueError('book_text must contain at least two letters/digits')
        self.text_book_colour = None
        self.text_grasp_origin = self.text_grasp_normal = None
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
        self.bin_candidate, self.bin_stamp, self.bin_hits = None, None, 0
        self.bin_candidate = None
        self.bin_hits = 0
        self.bin_candidate_stamp = None
        self.column_pose = None
        self.candidate_column_pose = None
        self.candidate_column_hits = 0
        self.column_normal_locked = False
        self.last_floor_calibration = -100.0
        self.observation_stamp = None
        self.filtered_book = None
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
        self.ocr_readings = []
        self.held_ocr_handled = -1
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
        self.create_subscription(Bool, topics['measure_hand'], self.on_measure_hand, 10)
        self.create_subscription(Bool, topics['check_held'], self.on_check_held, 10)
        self.create_subscription(PoseStamped, topics['grasp_target_pose'], self.on_grasp_target,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.held_pub = self.create_publisher(Bool, topics['book_held'], 10)
        self.held_until = 0.0
        self.held_votes = []
        self.hand_pub = self.create_publisher(PointStamped, topics['hand_offset'], 10)
        self.hand_until = 0.0
        self.hand_samples = []
        self.ready_pub = self.create_publisher(Bool, topics['perception_ready'], 10)
        self.shelf_pub = self.create_publisher(PoseStamped, topics['shelf_pose'], 10)
        self.unit_pub = self.create_publisher(PoseStamped, topics['shelf_unit_pose'], 10)
        self.book_diag_pub = self.create_publisher(String, topics['book_diagnostics'], 10)
        self.geometry_pub = self.create_publisher(String, topics['book_geometry'], 10)
        self.book_pub = self.create_publisher(PoseStamped, topics['book_pose'], 10)
        self.bin_pub = self.create_publisher(PoseStamped, topics['bin_pose'], 10)
        self.column_pub = self.create_publisher(Int32, topics['shelf_column_identification'], 10)
        self.row_pub = self.create_publisher(Int32, topics['shelf_row_identification'], 10)
        self.create_timer(1.0/self.get_parameter('perception_rate').value, self.process)
        self.get_logger().info(
            f'Perception BUILD 10.0: column {self.target_column}, '
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
        K[0] *= msg.width/float(full[0].width)
        K[1] *= msg.height/float(full[0].height)
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
        reg = np.full((height, width), np.inf, np.float32)
        np.minimum.at(reg, (vc, uc), zc)
        reg[~np.isfinite(reg)] = 0.0
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

    def on_measure_hand(self, msg):
        if msg.data:
            self.hand_until = self.get_clock().now().nanoseconds/1e9+4.5
            self.hand_samples = []

    def measure_hand(self, depth_m, K_d, T_od, stamp):
        """The auto-calibrated grasp: the gripper seen in depth vs where the arm's kinematics put it.
        Both the book and the hand are measured through this camera, so the camera's own error (the
        real robot's head camera sits ~3.5 cm and ~1 deg off its TF) cancels, and the arm's encoder
        error is measured too. Returns (correction in odom, implied bias in the hand frame) or None."""
        try:
            T_oh = transform_matrix(self.tf.lookup_transform(self.ros.frames['odom'], self.get_parameter('hand_frame').value,
                                                             stamp, timeout=Duration(seconds=0.05)).transform)
        except TransformException:
            return None
        bias = np.array(self.get_parameter('hand_bias').value, float)
        expected = T_oh[:3, 3]+T_oh[:3, :3] @ bias
        step = 2 if depth_m.shape[1] >= 400 else 1
        d = depth_m[::step, ::step]
        v, u = np.nonzero(np.isfinite(d) & (d > 0.2) & (d < 1.6))
        z = d[v, u]
        pts = np.stack([(u*step-K_d[0, 2])*z/K_d[0, 0], (v*step-K_d[1, 2])*z/K_d[1, 1], z])
        P = (T_od[:3, :3] @ pts + T_od[:3, 3:4]).T
        c = expected.copy()
        for radius in (0.14, 0.11, 0.09, 0.09):
            near = P[np.linalg.norm(P-c, axis=1) < radius]
            if len(near) < 80:
                return None
            c = near.mean(axis=0)
        correction = expected-c
        implied = T_oh[:3, :3].T @ (c-T_oh[:3, 3])
        if np.linalg.norm(correction) > 0.10:
            self.warn(f'Hand measurement {correction.round(3)} m is implausible; ignored '
                      f'(gripper centroid in its frame {implied.round(3)}; hand_bias may need calibrating).')
            return None
        return correction, implied

    def on_check_held(self, msg):
        now = self.get_clock().now().nanoseconds/1e9
        if msg.data and now >= self.held_until:
            self.held_until = now+12.0
            self.held_votes = []
            self.held_text_stamp = None

    def book_at_hand(self, stamp):
        """Is a book of the target colour within 18 cm of the grasp point (the fingers)? None when the
        hand's pose is unknown."""
        try:
            T_oh = transform_matrix(self.tf.lookup_transform(self.ros.frames['odom'], self.get_parameter('hand_frame').value,
                                                             stamp, timeout=Duration(seconds=0.05)).transform)
        except TransformException:
            return None
        grasp = T_oh[:3, :3] @ np.array([0.0, 0.0, 0.157]) + T_oh[:3, 3]
        camera = np.linalg.inv(self.camera_to_odom) @ np.r_[grasp, 1.]
        if camera[2] <= 0.15:
            return None
        height, width = self.image.shape[:2]
        sx, sy = width/(self.info.width or width), height/(self.info.height or height)
        fx, fy = self.info.k[0]*sx, self.info.k[4]*sy
        cx, cy = self.info.k[2]*sx, self.info.k[5]*sy
        u, v = fx*camera[0]/camera[2]+cx, fy*camera[1]/camera[2]+cy
        if not 0 <= u < self.image.shape[1] or not 0 <= v < self.image.shape[0]:
            return None
        colour = getattr(self, 'text_book_colour', None) if self.target_text else self.target_colour
        if colour is None:
            return None
        for (x, y, w, h) in colour_boxes(self.image, colour, books=False):
            q = self.point(x+w/2, y+h/2)
            if (q is not None and np.linalg.norm(q-grasp) < 0.18
                    and (not self.target_text or self.text_book_extracted(q))):
                return True
        return False

    def on_grasp_target(self, msg):
        if self.target_text and msg.header.frame_id == self.ros.frames['odom'] and np.isfinite(pose_xyz(msg)).all():
            self.text_grasp_origin = pose_xyz(msg).copy()
            heading = yaw(msg.pose.orientation)
            self.text_grasp_normal = np.array([math.cos(heading), math.sin(heading), 0.])

    def text_book_extracted(self, point):
        # Reading the original book still sitting behind an empty gripper is
        # not evidence of a successful grasp. Require motion out of its bay.
        origin, normal = self.text_grasp_origin, self.text_grasp_normal
        if origin is None or normal is None:
            return False
        delta = point-origin
        return bool(np.linalg.norm(delta) > .05 and delta @ normal < -.035)

    def text_spine_box(self, image, label_box):
        """Associate an OCR label with a full coloured spine, never colour alone."""
        x, y, w, h = label_box
        centre = (x+w/2, y+h/2)
        colours = [self.text_colour_hint] if self.text_colour_hint else list(STRICT)
        candidates = []
        for colour in colours:
            for box in colour_boxes(image, colour):
                bx, by, bw, bh = box
                if bh < 1.3*bw or not (bx <= centre[0] <= bx+bw and by <= centre[1] <= by+bh):
                    continue
                overlap = max(0, min(x+w, bx+bw)-max(x, bx))*max(0, min(y+h, by+bh)-max(y, by))
                if overlap >= .60*w*h:
                    candidates.append((box, colour))
        return candidates[0] if len(candidates) == 1 else None

    def hand_text_crop(self):
        """Read the held title in a small projected hand ROI at this RGB stamp."""
        stamp = Time.from_msg(self.observation_stamp)
        try:
            tf = self.tf.lookup_transform(self.ros.frames['odom'], self.get_parameter('hand_frame').value,
                                          stamp, timeout=Duration(seconds=.05))
        except TransformException:
            return None
        hand = transform_matrix(tf.transform)
        grasp = (hand @ np.array([0., 0., .157, 1.]))[:3]
        camera = np.linalg.inv(self.camera_to_odom) @ np.r_[grasp, 1.]
        if camera[2] <= .15:
            return None
        height, width = self.image.shape[:2]
        fx, fy = self.info.k[0]*width/(self.info.width or width), self.info.k[4]*height/(self.info.height or height)
        cx, cy = self.info.k[2]*width/(self.info.width or width), self.info.k[5]*height/(self.info.height or height)
        u, v = fx*camera[0]/camera[2]+cx, fy*camera[1]/camera[2]+cy
        if not 0 <= u < width or not 0 <= v < height:
            return None
        rx, ry = fx*.28/camera[2], fy*.28/camera[2]
        x0, x1 = max(0, int(u-rx)), min(width, int(u+rx)+1)
        y0, y1 = max(0, int(v-ry)), min(height, int(v+ry)+1)
        if x1-x0 < 12 or y1-y0 < 12:
            return None
        return self.image[y0:y1, x0:x1].copy(), (x0, y0), grasp

    def record_held_vote(self, vote, required=3):
        if vote is None:
            return
        self.held_votes.append(bool(vote))
        if len(self.held_votes) >= required:
            held = sum(self.held_votes) >= required//2+1
            self.held_pub.publish(Bool(data=held))
            self.get_logger().info(f'Grasp check: {sum(self.held_votes)}/{len(self.held_votes)} fresh votes -> {"held" if held else "NOT held"}')
            self.held_until = 0.

    def verify_book_at_hand(self, stamp):
        if not self.target_text or self.text_book_colour is not None:
            self.record_held_vote(self.book_at_hand(stamp))
            return
        # White/neutral spines: require the exact title at the hand instead
        # of silently using the unrelated default red colour.
        self.update_text_search('hand')
        if self.ocr_sequence == self.held_ocr_handled:
            return
        self.held_ocr_handled = self.ocr_sequence
        context = self.ocr_result_context
        if context is None or context.get('purpose') != 'hand' or context.get('grasp') is None:
            return
        age = (self.get_clock().now()-Time.from_msg(context['stamp'])).nanoseconds/1e9
        if not 0 <= age <= 2.0:
            return
        captured = (context['stamp'].sec, context['stamp'].nanosec)
        if captured == getattr(self, 'held_text_stamp', None):
            return
        self.held_text_stamp = captured
        x0, y0 = context['origin']
        for (x, y, w, h), _, recognized in self.ocr_boxes:
            if not label_matches(recognized, self.target_text):
                continue
            point = self.point_in_frame(x+x0+w/2, y+y0+h/2, context['image_shape'],
                                        context['depth_image'], context['camera_to_odom'], context['info'])
            if (point is not None and np.linalg.norm(point-context['grasp']) < .18
                    and self.text_book_extracted(point)):
                self.record_held_vote(True, required=2)
                break

    def on_mode(self, msg):
        mode = 'bin' if msg.data else 'shelf'
        if mode != self.mode:
            self.bin_candidate = None
            self.bin_hits = 0
            self.bin_stamp = None
        self.mode = mode

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
        fx, fy = info.k[0]*dw/(info.width or w), info.k[4]*dh/(info.height or h)
        cx, cy = info.k[2]*dw/(info.width or w), info.k[5]*dh/(info.height or h)
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
        height, width = self.image.shape[:2]
        sx, sy = width/(self.info.width or width), height/(self.info.height or height)
        fx, fy = self.info.k[0]*sx, self.info.k[4]*sy
        cx, cy = self.info.k[2]*sx, self.info.k[5]*sy
        pixels = []
        half_width = self.get_parameter('column_half_width').value
        for lateral in (-half_width, half_width):
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

    def update_text_search(self, purpose='shelf'):
        if self.ocr_future is not None and self.ocr_future.done():
            try:
                result = self.ocr_future.result()
                self.ocr_boxes, self.ocr_readings = result.matches, result.readings
            except Exception as error:
                self.ocr_boxes, self.ocr_readings = [], []
                self.warn(f'Text OCR failed: {error}')
            self.ocr_result_context = self.ocr_pending_context
            self.ocr_sequence += 1
            self.ocr_future = None
        if (self.ocr_future is not None or
                self.get_clock().now().nanoseconds/1e9-self.ocr_submit_at <
                self.get_parameter('book_text_ocr_period').value):
            return
        search = self.hand_text_crop() if purpose == 'hand' else self.text_search_crop()
        if search is None:
            return
        crop, origin = search[:2]
        self.ocr_pending_context = {
            'origin': origin, 'purpose': purpose, 'rgb_crop': crop, 'rgb_image': self.image.copy(),
            'grasp': search[2] if purpose == 'hand' else None,
            'image_shape': self.image.shape[:2],
            'depth_image': self.depth_image.copy(),
            'camera_to_odom': self.camera_to_odom.copy(),
            'info': self.info,
            'stamp': self.observation_stamp,
        }
        self.ocr_future = self.ocr_pool.submit(read_text_label_boxes, crop,
                                               self.target_text)
        self.ocr_submit_at = self.get_clock().now().nanoseconds/1e9

    def make_pose(self, xyz, heading=0.0):
        msg = PoseStamped()
        msg.header.frame_id = self.ros.frames['odom']
        msg.header.stamp = self.observation_stamp or self.depth.header.stamp
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
        if not 0 <= (self.get_clock().now()-stamp).nanoseconds/1e9 <= self.get_parameter('observation_max_age').value:
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
        now = self.get_clock().now().nanoseconds/1e9
        if now-self.last_floor_calibration >= self.get_parameter('floor_calibration_period').value:
            self.measure_tilt(depth_m, K_d, T_od)
            self.last_floor_calibration = now
        c, s_ = math.cos(self.head_tilt), math.sin(self.head_tilt)
        tilt = np.eye(4)
        tilt[1:3, 1:3] = [[c, -s_], [s_, c]]
        self.camera_to_odom = transform_matrix(tf.transform) @ tilt
        # depth -> colour through odom, each at its own stamp: the pair may be up to 0.6 s apart while
        # the base turns, and a fixed depth->colour transform would smear every point by that motion
        T_cd = np.linalg.inv(self.camera_to_odom) @ (T_od @ tilt)
        self.image = self.bridge.imgmsg_to_cv2(self.rgb, 'bgr8')
        self.surface_masks = {}
        K_c = np.array(self.info.k, float).reshape(3, 3).copy()
        K_c[0] *= self.image.shape[1]/float(self.info.width or self.image.shape[1])
        K_c[1] *= self.image.shape[0]/float(self.info.height or self.image.shape[0])
        self.observation_stamp = self.rgb.header.stamp
        self.depth_image = self.register(depth_m, K_d, T_cd, K_c, self.image.shape[1], self.image.shape[0])
        if self.get_clock().now().nanoseconds/1e9 < self.held_until:
            self.verify_book_at_hand(stamp)
        if self.get_clock().now().nanoseconds/1e9 < self.hand_until:
            c_, s__ = math.cos(self.head_tilt), math.sin(self.head_tilt)
            tilt_ = np.eye(4)
            tilt_[1:3, 1:3] = [[c_, -s__], [s__, c_]]
            m = self.measure_hand(depth_m, K_d, T_od @ tilt_, depth_stamp)
            if m is not None:
                # the book again, in this same frame and from this same distance (the camera's yaw error
                # shifts things sideways in proportion to range: book at 1.8 m and hand at 0.6 m do not cancel)
                shift = np.zeros(3)
                if self.latest_book is not None and not self.target_text:
                    best = None
                    for (x, y, w, h) in colour_boxes(self.image, self.target_colour):
                        q = self.point(x+w/2, y+h/2)
                        if q is not None and np.linalg.norm(q[:2]-self.latest_book[:2]) < 0.15:
                            if best is None or np.linalg.norm(q[:2]-self.latest_book[:2]) < np.linalg.norm(best[:2]-self.latest_book[:2]):
                                best = q
                    if best is not None:
                        shift[:2] = best[:2]-self.latest_book[:2]
                self.hand_samples.append((m[0]+shift, m[1], shift))
            if len(self.hand_samples) >= 4:
                corr = np.median([a for a, _, _ in self.hand_samples], axis=0)
                implied = np.median([b for _, b, _ in self.hand_samples], axis=0)
                book_shift = np.median([c for _, _, c in self.hand_samples], axis=0)
                try:
                    R = transform_matrix(self.tf.lookup_transform(self.ros.frames['base'], target_frame, Time()).transform)[:3, :3]
                except TransformException:
                    R = None
                if R is not None:
                    corr_base = R @ corr
                    msg = PointStamped()
                    msg.header.stamp = depth_msg.header.stamp
                    msg.header.frame_id = self.ros.frames['base']
                    msg.point.x, msg.point.y, msg.point.z = (float(v) for v in corr_base)
                    self.hand_pub.publish(msg)
                    self.get_logger().info(f'Hand measured: grasp correction {corr_base.round(3)} m in the base frame '
                                           f'(of which the book re-measured beside the hand: {(R @ book_shift).round(3)}) '
                                           f'(gripper centroid in its frame {implied.round(3)}; hand_bias {self.get_parameter("hand_bias").value})')
                self.hand_until = 0.0
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
            if self.column_pose is not None:
                return  # The static plane is retained while looking down at the book.
            labels = sorted({digit for digit, _, _ in detections})
            if labels:
                self.warn(
                    f'Shelf search recognized labels {labels}; rejected={rejected}; '
                    f'heights_m={measured_heights}; allowed={min_height:.2f}-{max_height:.2f} m.')
            else:
                self.warn('Shelf search cannot recognize any number markers in the current camera view.')
            return
        # Multiple contours for one digit must not vote for a fictitious shelf line.
        unique = {}
        for candidate in candidates:
            if candidate[0] not in unique or candidate[1] > unique[candidate[0]][1]:
                unique[candidate[0]] = candidate
        candidates = list(unique.values())
        points = np.array([c[3] for c in candidates])
        if len(points) >= 3:
            good = np.abs(points[:, 2]-np.median(points[:, 2])) < 0.12
            candidates = [c for c, keep in zip(candidates, good) if keep]
            points = np.array([c[3] for c in candidates])
        if not len(points):
            return
        centre = points.mean(axis=0)
        toward = centre[:2]-self.camera_to_odom[:2, 3]
        heading = math.atan2(toward[1], toward[0])
        fitted = fit_shelf_line(points, self.camera_to_odom[:2, 3])
        reliable_normal = fitted is not None
        if reliable_normal:
            centre, heading, inliers = fitted
            candidates = [c for c, keep in zip(candidates, inliers) if keep]
        # A single sign only gives its bearing from the camera, not the shelf
        # normal. Navigation uses this pose to choose a lateral scan direction,
        # so publish it only when two separated signs establish the shelf line.
        if reliable_normal:
            self.unit_pub.publish(self.make_pose(centre, heading))
            last = getattr(self, 'logged_unit', None)
            if last is None or np.linalg.norm(centre[:2]-last[0]) > 0.2 or abs(math.remainder(heading-last[1], 2*math.pi)) > 0.17:
                self.logged_unit = (centre[:2].copy(), heading)
                self.get_logger().info(f'Shelf line from markers {sorted(c[0] for c in candidates)}: centre '
                                       f'({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}) facing {math.degrees(heading):.0f} deg')
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
        previous = self.candidate_column_pose
        if self.column_pose is not None and np.linalg.norm(point-pose_xyz(self.column_pose)) > 0.45:
            return False
        consistent = previous is not None and np.linalg.norm(point-pose_xyz(previous)) < 0.15
        if reliable_normal and previous is not None:
            consistent = consistent and abs(angle_delta(heading, yaw(previous.pose.orientation))) < 0.30
        self.candidate_column_hits = self.candidate_column_hits+1 if consistent else 1
        self.candidate_column_pose = self.make_pose(point, heading)
        if self.candidate_column_hits < 3 or not (reliable_normal or self.column_normal_locked):
            return False
        if self.column_pose is not None:
            point = 0.75*pose_xyz(self.column_pose)+0.25*point
            old_heading = yaw(self.column_pose.pose.orientation)
            if reliable_normal:
                heading = old_heading+0.25*float(np.clip(angle_delta(heading, old_heading), -0.17, 0.17))
            else:
                heading = old_heading
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
            if self.get_clock().now().nanoseconds/1e9 < self.held_until:
                return
            self.update_text_search()
            if self.ocr_sequence == self.ocr_handled:
                return
            self.ocr_handled = self.ocr_sequence
            boxes = self.ocr_boxes
            context = self.ocr_result_context
            if context is None or context.get('purpose', 'shelf') != 'shelf':
                self.book_hits = 0
                return
            source_stamp = context['stamp']
            if source_stamp is None or not 0 <= (self.get_clock().now()-Time.from_msg(source_stamp)).nanoseconds/1e9 <= 2.0:
                self.book_hits = 0
                self.book_status('OCR result too old; waiting for a current frame')
                return
            captured = (source_stamp.sec, source_stamp.nanosec)
            if captured == getattr(self, 'last_text_book_stamp', None):
                return
            self.last_text_book_stamp = captured
            x0, y0 = context['origin']
            candidate_colours, candidate_surfaces = {}, {}
            vertical = [entry for entry in boxes if entry[0][3] > 1.3*entry[0][2]]
            if not vertical:
                self.book_status('Title not yet confirmed on a vertical spine; waiting for a front view.')
                return
            boxes = vertical
            for local_box, confidence, recognized in boxes:
                if not label_matches(recognized, self.target_text):
                    continue
                x, y, w, h = local_box
                label_box = (x+x0, y+y0, w, h)
                surface = measure_book(context['rgb_image'], context['depth_image'],
                    context['camera_to_odom'], context['info'], label_box, np.r_[normal, 0.], white=True)
                if surface is None:
                    rejected['depth'] += 1
                    continue
                box, p = tuple(surface['box']), np.array(surface['point'])
                candidate_colours[box] = None
                candidate_surfaces[box] = surface
                candidates.append((box, p, None, confidence, recognized))
        else:
            source_stamp = self.observation_stamp
            boxes = colour_boxes(self.image, self.target_colour)
            crop = self.text_search_crop()
            if crop is not None:
                image, (x0, y0) = crop
                local, _ = _raw_boxes(image, RELAXED[self.target_colour], 60)
                for x, y, w, h in local:
                    box = (x+x0, y+y0, w, h)
                    if not any(abs(box[0]-old[0]) < 5 and abs(box[1]-old[1]) < 5 for old in boxes):
                        boxes.append(box)
            for box in boxes:
                x, y, w, h = box
                if h < 1.3*w:
                    rejected['shape'] += 1
                    continue
                surface = self.colour_surface(box, self.target_colour)
                if surface is None:
                    rejected['depth'] += 1
                    continue
                p, _, _ = surface
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
            def same_detection(kept):
                if np.linalg.norm(candidate[1]-kept[1]) >= .12:
                    return False
                if not self.target_text:
                    return True
                x, y, w, h = candidate[0]
                a, b, c, d = kept[0]
                overlap = max(0, min(x+w, a+c)-max(x, a))*max(0, min(y+h, b+d)-max(y, b))
                return overlap > .5*min(w*h, c*d)
            if not any(same_detection(kept) for kept in candidates):
                candidates.append(candidate)
        if len(candidates) != 1:
            self.book_misses = getattr(self, 'book_misses', 0)+1
            if self.book_misses > 1:
                self.book_hits = 0
            detail = f'; plane_offsets_m={plane_offsets}' if plane_offsets else ''
            if self.target_text:
                detail += f'; OCR_readings={self.ocr_readings}'
            target = f'text={self.target_text!r}' if self.target_text else f'colour={self.target_colour}'
            self.book_status(f'{target}; candidates={len(boxes)}; valid_in_column={len(candidates)}; rejected={rejected}{detail}')
            return
        box, p, row, confidence, recognized = candidates[0]
        observed = Time.from_msg(source_stamp).nanoseconds/1e9
        last = getattr(self, 'last_book_observed', None)
        if (self.latest_book is not None and np.linalg.norm(p-self.latest_book) < 0.08
                and (last is None or 0 <= observed-last <= 1.0)):
            self.book_hits += 1
        else:
            self.book_hits = 1
        self.latest_book = p
        self.book_misses = 0
        self.last_book_observed = observed
        required_hits = 3
        if self.book_hits < required_hits:
            self.book_status(f'confirming_book: {self.book_hits}/{required_hits} observations')
            return
        if self.target_text:
            self.text_book_colour = candidate_colours.get(box)
        target = f'text={self.target_text!r}' if self.target_text else f'colour={self.target_colour}'
        self.book_status(f'book_confirmed: {target}, row={row}, xyz={p.round(3).tolist()}')
        previous = self.filtered_book
        self.filtered_book = (0.65*previous+0.35*p if previous is not None and
                              np.linalg.norm(p-previous) < 0.12 else p.copy())
        pose = self.make_pose(self.filtered_book, heading)
        pose.header.stamp = source_stamp
        geometry = self.publish_book_geometry(box, p, source_stamp,
                candidate_surfaces.get(box) if self.target_text else None,
                context if self.target_text else None)
        if not geometry:
            return
        if isinstance(geometry, dict):
            # Target the measured front-surface centre for colour books too,
            # keeping the grasp target and its divider envelope consistent.
            self.filtered_book = self.latest_book = np.array(geometry['point'])
            pose = self.make_pose(self.filtered_book, heading)
            pose.header.stamp = source_stamp
        self.book_pub.publish(pose)
        if 'row' not in self.reported:
            self.row_pub.publish(Int32(data=row))
            self.save(box, 'book_row')
            self.reported.add('row')
            self.get_logger().info(
                f'Confirmed {target} book in target column, row {row}, '
                f'confidence={confidence:.0f}, recognized={recognized!r}, xyz={p.round(3)}')

    def publish_book_geometry(self, box, point, stamp, surface=None, context=None):
        image = self.image if context is None else context['rgb_image']
        depth = self.depth_image if context is None else context['depth_image']
        transform = self.camera_to_odom if context is None else context['camera_to_odom']
        info = self.info if context is None else context['info']
        heading = yaw(self.column_pose.pose.orientation)
        normal = np.array([math.cos(heading), math.sin(heading), 0.])
        if surface is None:
            surface = measure_book(image, depth, transform, info, box, normal)
        if surface is None or np.linalg.norm(np.asarray(surface['point'])-point) > .08:
            self.book_status('Book surface lacks stable depth/size support; holding for a clearer view.')
            return False
        shelf = pose_xyz(self.column_pose)
        left, right, source = divider_bounds(depth, transform, info, image.shape[:2], shelf, normal,
                                            point[2], self.get_parameter('column_half_width').value)
        geometry = {**surface, 'shelf_point': shelf.tolist(), 'normal': normal.tolist(),
                    'left': left, 'right': right, 'divider_source': source,
                    'stamp': stamp.sec+stamp.nanosec/1e9, 'frame': self.ros.frames['odom']}
        self.geometry_pub.publish(String(data=json.dumps(geometry)))
        return geometry

    def colour_surface(self, box, colour, maximum_depth=8.0):
        masks = getattr(self, 'surface_masks', {})
        if colour not in masks:
            image = self.image
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = np.zeros(image.shape[:2], np.uint8)
            for lower, upper in RELAXED[colour]:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks[colour] = mask
            self.surface_masks = masks
        mask = masks[colour]
        return masked_surface(box, mask, self.depth_image, self.camera_to_odom,
                              self.info, maximum_depth)

    def process_bin(self):
        frame_area = self.image.shape[0]*self.image.shape[1]
        image = self.image
        # A dark red rim and an oblique view need a wider band than a book.
        bands = [((0, 80, 45), (16, 255, 255)), ((168, 80, 45), (180, 255, 255))]
        boxes, _ = _raw_boxes(image, bands, frame_area*self.get_parameter('bin_min_area_fraction').value)
        candidates = []
        rejected = {'shape': 0, 'surface_depth': 0, 'size_or_height': 0}
        for box in boxes:
            x, y, w, h = box
            if not .65 <= w/max(h, 1) <= 6.0:
                rejected['shape'] += 1
                continue
            surface = self.colour_surface(box, 'red', maximum_depth=4.0)
            if surface is None:
                rejected['surface_depth'] += 1
                continue
            p, metric_width, metric_height = surface
            if not (.60 < p[2] < 1.20 and .20 <= metric_width <= 1.20 and
                    .08 <= metric_height <= .80 and
                    np.linalg.norm(p[:2]-self.camera_to_odom[:2, 3]) > .40):
                rejected['size_or_height'] += 1
                continue
            candidates.append((w*h, box, p))
        if len(candidates) != 1:
            self.bin_hits = 0
            self.bin_candidate = None
            self.warn(f'Red drop-box search: candidates={len(boxes)}, valid={len(candidates)}, rejected={rejected}; waiting for one surface-supported box.')
            return
        _, box, p = candidates[0]
        stamp = (self.observation_stamp.sec, self.observation_stamp.nanosec)
        if stamp == self.bin_stamp:
            return
        previous = self.bin_candidate
        consistent = (previous is not None and self.bin_stamp is not None and
                      0 < stamp[0]+stamp[1]/1e9-(self.bin_stamp[0]+self.bin_stamp[1]/1e9) <= 1.0 and
                      np.linalg.norm(p-previous) <= .18)
        self.bin_hits = self.bin_hits+1 if consistent else 1
        self.bin_candidate = .65*previous+.35*p if consistent else p.copy()
        self.bin_stamp = stamp
        if self.bin_hits < 3:
            return
        p = self.bin_candidate
        d = p[:2]-self.camera_to_odom[:2, 3]
        self.bin_pub.publish(self.make_pose(p, math.atan2(d[1], d[0])))
        if 'bin' not in self.reported:
            self.save(box, 'collection_bin')
            self.reported.add('bin')
            self.get_logger().info(f'Confirmed red drop box on coloured surface: xyz={p.round(3)}, camera_range={np.linalg.norm(d):.2f} m')

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
        node.ocr_pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
