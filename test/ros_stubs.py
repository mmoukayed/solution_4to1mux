"""Import-only ROS stand-ins for offline controller-logic tests.

These do not simulate ROS transport, controller dynamics, TF, or Gazebo.
"""
import sys
import types
from types import SimpleNamespace as NS


class Message:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Twist:
    def __init__(self):
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class PoseStamped:
    def __init__(self):
        self.header = NS(frame_id='odom', stamp=NS(sec=0, nanosec=0))
        self.pose = NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


class Time:
    def __init__(self, nanoseconds=0, **kwargs):
        self.nanoseconds = nanoseconds

    @classmethod
    def from_msg(cls, msg):
        return cls(msg.sec*10**9+msg.nanosec)

    def __sub__(self, other):
        return Time(self.nanoseconds-other.nanoseconds)


class JointTrajectory:
    def __init__(self):
        self.joint_names = []
        self.points = []


class JointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.time_from_start = NS(sec=0, nanosec=0)


class Pub:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    def get_subscription_count(self):
        return 1


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, text):
        self.messages.append(text)

    warn = error = info


def install():
    before = dict(sys.modules)
    modules = {
        'rclpy': {}, 'rclpy.node': {'Node': object},
        'rclpy.action': {'ActionClient': Message},
        'rclpy.clock': {'Clock': Message, 'ClockType': NS(STEADY_TIME=1)},
        'rclpy.time': {'Time': Time},
        'rclpy.qos': {'QoSProfile': Message, 'DurabilityPolicy': NS(TRANSIENT_LOCAL=1),
                      'ReliabilityPolicy': NS(BEST_EFFORT=1), 'qos_profile_sensor_data': Message()},
        'geometry_msgs.msg': {'PoseStamped': PoseStamped, 'Twist': Twist},
        'nav_msgs.msg': {'Odometry': Message},
        'sensor_msgs.msg': {'LaserScan': Message, 'JointState': Message, 'CameraInfo': Message, 'Image': Message},
        'std_msgs.msg': {'Bool': Message, 'String': Message, 'Int32': Message},
        'trajectory_msgs.msg': {'JointTrajectory': JointTrajectory, 'JointTrajectoryPoint': JointTrajectoryPoint},
        'control_msgs.action': {'FollowJointTrajectory': NS(Goal=Message)},
        'action_msgs.msg': {'GoalStatus': NS(STATUS_SUCCEEDED=4)},
        'tf2_ros': {'Buffer': Message, 'TransformListener': Message, 'TransformException': RuntimeError},
        'ros_gz_interfaces.msg': {'Contacts': Message},
        'ament_index_python.packages': {'get_package_share_directory': lambda name: '.'},
        'cv_bridge': {'CvBridge': Message},
    }
    for name, symbols in modules.items():
        parts = name.split('.')
        for i in range(1, len(parts)+1):
            key = '.'.join(parts[:i])
            if key not in sys.modules:
                sys.modules[key] = types.ModuleType(key)
            if i > 1:
                setattr(sys.modules['.'.join(parts[:i-1])], parts[i-1], sys.modules[key])
        sys.modules[name].__dict__.update(symbols)

    def restore():
        roots = {name.split('.')[0] for name in modules}
        for key in list(sys.modules):
            if key.split('.')[0] in roots:
                if key in before:
                    sys.modules[key] = before[key]
                else:
                    sys.modules.pop(key, None)
    return restore
