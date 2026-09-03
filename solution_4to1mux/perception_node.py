import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs import msg
from cv_bridge import CvBridge
import cv2

class PerceptionNode(Node):
    def process_image(self, msg: msg.Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Now cv_image is a normal OpenCV/numpy image
        cv2.imshow("perceptino", cv_image)
        cv2.waitKey(1)


    def __init__(self):
        self.timer = self.create_timer(0.1, self.tick)
        self.bridge = CvBridge()
        self.camera = self.create_subscription(msg.Image, "/head_front_camera/head_front_camera/color/image_raw", self.process_image, 10)

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