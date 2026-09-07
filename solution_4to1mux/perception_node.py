import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs import msg
from cv_bridge import CvBridge
import cv2
import numpy as np
import easyocr

class PerceptionNode(Node):
    def process_image(self, msg: msg.Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # OCR
        results = self.reader.readtext(
            cv_image,
            detail=1,
            allowlist="0123456789"
        )

        # Print detected numbers
        detection_img = cv_image.copy()
        for bbox, text, confidence in results:
            if self.shelf_column_num == int(text):
                cv2.rectangle(detection_img, bbox[0], bbox[2], (0,255,0))
            else:
                cv2.rectangle(detection_img, bbox[0], bbox[2], (255,0,0))
            self.get_logger().info(f"Number: {text}, confidence: {confidence:.2f}")
            # self.get_logger().info(f"{bbox}")
        # Now cv_image is a normal OpenCV/numpy image
        cv2.imwrite("rgb_cam_num_bounds.png", detection_img) # Saving image with bounding boxes
        cv2.imwrite("rgb_cam.png", cv_image)
    def process_depth(self, msg: msg.Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # depth_display = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_display = np.clip(cv_image, 0.0, 10.0)

        depth_display = (
            (10.0 - depth_display) / 10.0 * 255
        ).astype(np.uint8)

        # h, w = cv_image.shape
        # depth = cv_image[h // 2, w // 2]
        # self.get_logger().info(f'Center depth: {depth:.2f} m')
        
        cv2.imwrite("depth_cam.png", depth_display)

    def __init__(self):
        super().__init__('perception_node')
        self.declare_parameter('shelf_column_number', -1)
        self.shelf_column_num = self.get_parameter('shelf_column_number').get_parameter_value().integer_value
        self.timer = self.create_timer(0.1, self.tick)
        self.bridge = CvBridge()
        self.camera = self.create_subscription(msg.Image, "/head_front_camera/head_front_camera/color/image_raw", self.process_image, 10)
        self.depth = self.create_subscription(msg.Image, "/head_front_camera/head_front_camera/depth/image_rect_raw", self.process_depth, 10)
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    def tick(self):
        msg = Twist()

def main():
    rclpy.init()
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


    '''

    color isolation program, useful for identifying books by color
    import cv2
import numpy as np

# 1. Load image and convert to HSV
img = cv2.imread('objects.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. Define upper and lower HSV thresholds for Blue
# [Hue, Saturation, Value]
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# 3. Create a binary mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 4. Bitwise-AND mask and original image to isolate color
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Isolated Blue Objects', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

    '''