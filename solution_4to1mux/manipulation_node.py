import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs import msg
from cv_bridge import CvBridge
import cv2

class ManipulationNode(Node):

    def __init__(self):
        super().__init__('manipulation_node')
        self.timer = self.create_timer(0.1, self.tick)
        self.bridge = CvBridge()

    def tick(self):
        msg = Twist()

def main():
    rclpy.init()
    node = ManipulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()