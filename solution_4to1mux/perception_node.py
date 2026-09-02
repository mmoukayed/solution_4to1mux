#!/usr/bin/env python3
"""
perception_node.py — solution_4to1mux (ERC 2026)

Responsibilities:
  - Detect the overhead numeral marker (1-5) above the target shelf column
  - Detect the target book by colour on the shelf
  - Detect the red collection bin
  - Publish scoring topics (/erc/shelf_column_identification, /erc/shelf_row_identification)
  - Publish lightweight guidance data for navigation_node / manipulation_node to consume
  - Save timestamped annotated images to erc_images/ for scoring

Communication contract with the other two nodes:
  /perception/phase          (std_msgs/String)  current search phase
  /perception/column_target  (geometry_msgs/Point) x=pixel_x, z=image_width of target column digit
  /perception/book_target    (geometry_msgs/Point) x=pixel_x, z=image_width of target book
  /perception/bin_target     (geometry_msgs/Point) x=pixel_x, z=image_width of red bin
  /erc/shelf_column_identification (std_msgs/Int32) scoring topic
  /erc/shelf_row_identification    (std_msgs/Int32) scoring topic

navigation_node drives phase changes indirectly by reaching each location;
wire set_phase() calls in from navigation_node's status callback if you want
tighter coupling than the fixed order used here.
"""

import os
from datetime import datetime

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Point


# HSV ranges — TUNE THESE against the actual sim lighting before competition day.
# Grab a few frames with `ros2 run image_view image_saver` and sample pixel
# values in GIMP/an HSV picker to refine.
COLOUR_RANGES = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'blue':   [((100, 120, 70), (130, 255, 255))],
    'green':  [((40, 80, 70), (80, 255, 255))],
    'yellow': [((20, 100, 100), (35, 255, 255))],
}

# Assumes ros2 launch is run from the repo root, so this lands inside the
# team's git repo as required by the rules. Verify this before submitting.
IMAGE_SAVE_DIR = os.path.join(os.getcwd(), 'erc_images')


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('book_colour', 'red')
        self.target_column = self.get_parameter('shelf_column_number').value
        self.target_colour = self.get_parameter('book_colour').value.lower()

        self.bridge = CvBridge()
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

        self.column_reported = False
        self.row_reported = False
        self.phase = 'search_column'  # search_column -> search_book -> search_bin

        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.column_pub = self.create_publisher(Int32, '/erc/shelf_column_identification', 10)
        self.row_pub = self.create_publisher(Int32, '/erc/shelf_row_identification', 10)

        self.phase_pub = self.create_publisher(String, '/perception/phase', 10)
        self.column_bearing_pub = self.create_publisher(Point, '/perception/column_target', 10)
        self.book_pixel_pub = self.create_publisher(Point, '/perception/book_target', 10)
        self.bin_pixel_pub = self.create_publisher(Point, '/perception/bin_target', 10)

        # Let navigation_node tell us when it's safe to move on to the next phase
        self.create_subscription(String, '/navigation/status', self.nav_status_cb, 10)

        self.get_logger().info(
            f'PerceptionNode up — target column={self.target_column}, colour={self.target_colour}'
        )

    # ------------------------------------------------------------------
    def nav_status_cb(self, msg: String):
        # Advance perception's phase in step with navigation's state machine.
        state = msg.data
        if state == 'APPROACH_SHELF' and self.phase == 'search_column':
            self.phase = 'search_book'
        elif state in ('RETURN_TO_START', 'ALIGN_BIN') and self.phase == 'search_book':
            self.phase = 'search_bin'

    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated = frame.copy()

        if self.phase == 'search_column':
            self._handle_column_search(frame, annotated)
        elif self.phase == 'search_book':
            self._handle_book_search(frame, annotated)
        elif self.phase == 'search_bin':
            self._handle_bin_search(frame, annotated)

        self.phase_pub.publish(String(data=self.phase))

    # ------------------------------------------------------------------
    def _handle_column_search(self, frame, annotated):
        digit, box = self._find_shelf_digit(frame)
        if digit is None:
            return

        cv2.rectangle(annotated, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(annotated, str(digit), (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if digit == self.target_column:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            self.column_bearing_pub.publish(Point(x=cx, y=cy, z=float(frame.shape[1])))

            if not self.column_reported:
                self.column_pub.publish(Int32(data=digit))
                self._save_image(annotated, 'shelf_column')
                self.column_reported = True
                self.get_logger().info(f'Found target column {digit} — reported +1/+2 points')

    def _handle_book_search(self, frame, annotated):
        mask, contour = self._colour_mask_largest_contour(frame, self.target_colour)
        if contour is None:
            return

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 255), 2)

        row = self._estimate_row_from_y(y + h / 2.0, frame.shape[0])

        cx, cy = x + w / 2.0, y + h / 2.0
        self.book_pixel_pub.publish(Point(x=cx, y=cy, z=float(frame.shape[1])))

        if not self.row_reported:
            self.row_pub.publish(Int32(data=row))
            self._save_image(annotated, 'target_book')
            self.row_reported = True
            self.get_logger().info(f'Found target book row {row} — reported +1/+2 points')

    def _handle_bin_search(self, frame, annotated):
        mask, contour = self._colour_mask_largest_contour(frame, 'red', min_area=1500)
        if contour is None:
            return
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx, cy = x + w / 2.0, y + h / 2.0
        self.bin_pixel_pub.publish(Point(x=cx, y=cy, z=float(frame.shape[1])))

    # ------------------------------------------------------------------
    def _find_shelf_digit(self, frame):
        """
        Lightweight digit finder: threshold the marker band, find blob
        contours, classify each with template matching against digits
        rendered at runtime with cv2.putText (no external OCR dependency,
        so it's guaranteed to work in the unmodified competition Docker image).

        TODO before competition:
          - Crop to the actual marker band (markers are overhead — you may
            need to tilt head_1_joint/head_2_joint up during this phase).
          - Tune the rendered template font/size/thickness to match the
            real marker's appearance.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_digit, best_score, best_box = None, -1.0, None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h < 15 or w < 8 or h > 150:
                continue
            roi = thresh[y:y + h, x:x + w]
            digit, score = self._match_digit_template(roi)
            if score > best_score:
                best_digit, best_score, best_box = digit, score, (x, y, x + w, y + h)

        if best_score < 0.45:  # confidence gate — tune against real footage
            return None, None
        return best_digit, best_box

    def _match_digit_template(self, roi):
        best_digit, best_score = None, -1.0
        h, w = roi.shape
        if h == 0 or w == 0:
            return None, -1.0
        for digit in range(1, 6):
            template = np.zeros((h, w), dtype=np.uint8)
            cv2.putText(template, str(digit), (int(w * 0.1), int(h * 0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, h / 30.0, 255, 2)
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max()) if result.size else -1.0
            if score > best_score:
                best_digit, best_score = digit, score
        return best_digit, best_score

    def _colour_mask_largest_contour(self, frame, colour_name, min_area=400):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOUR_RANGES[colour_name]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return mask, None
        return mask, largest

    @staticmethod
    def _estimate_row_from_y(y_centre, frame_height):
        # Books occupy the middle 4 of 6 shelf rows. Split the visible band
        # into 4 equal strips top-to-bottom -> rows 1-4. TUNE against real footage.
        band_top, band_bottom = frame_height * 0.15, frame_height * 0.85
        fraction = (y_centre - band_top) / max(band_bottom - band_top, 1.0)
        row = int(np.clip(fraction * 4, 0, 3)) + 1
        return row

    def _save_image(self, image, label):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(IMAGE_SAVE_DIR, f'{label}_{ts}.png')
        cv2.imwrite(path, image)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()