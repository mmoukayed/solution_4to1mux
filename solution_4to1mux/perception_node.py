import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs import msg
from cv_bridge import CvBridge
import cv2
import numpy as np

class PerceptionNode(Node):
    def process_image(self, msg: msg.Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Now cv_image is a normal OpenCV/numpy image
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
        self.timer = self.create_timer(0.1, self.tick)
        self.bridge = CvBridge()
        self.camera = self.create_subscription(msg.Image, "/head_front_camera/head_front_camera/color/image_raw", self.process_image, 10)
        self.depth = self.create_subscription(msg.Image, "/head_front_camera/head_front_camera/depth/image_rect_raw", self.process_depth, 10)

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