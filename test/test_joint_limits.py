"""URDF/live checking-layer bounds and outgoing command speed checks."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from solution_4to1mux.geometry import ArmKinematics
from solution_4to1mux.joint_limits import JointLimits, LIVE_CAPS
from solution_4to1mux.ros_topics import get_ros_config
from ros_stubs import install, Pub, Logger

restore_ros = install()
from solution_4to1mux.manipulation_node import ManipulationNode
from solution_4to1mux.navigation_node import NavigationNode
restore_ros()


@pytest.fixture(scope='module')
def urdf():
    return (Path(__file__).parent/'data/erc_both_arm_chains.urdf').read_text()


@pytest.fixture(scope='module')
def live_urdf(urdf):
    # The live arm-1/arm-5 sides differ from the simulator. Supply the guide's
    # hard bounds as a synthetic live URDF, retaining the model's own speeds.
    root = ET.fromstring(urdf)
    for joint in root.findall('joint'):
        if joint.get('name') not in LIVE_CAPS:
            continue
        lo, hi, _ = LIVE_CAPS[joint.get('name')]
        joint.find('limit').set('lower', str(lo))
        joint.find('limit').set('upper', str(hi))
        safety = joint.find('safety_controller')
        if safety is not None:
            joint.remove(safety)
    return ET.tostring(root, encoding='unicode')


def test_real_limits_include_guide_margins_and_asymmetric_arms(live_urdf):
    limits = JointLimits(live_urdf, 'real')
    for name, (lo, hi, speed) in LIVE_CAPS.items():
        if name not in limits.by_name:
            continue
        bound = limits.by_name[name]
        margin = 0.0 if name.startswith('gripper_') else .05
        assert bound.lower >= lo+margin-1e-9
        assert bound.upper <= hi-margin+1e-9
        if speed is not None:
            assert bound.velocity <= speed
    assert limits.by_name['arm_right_1_joint'].upper > 4.
    assert limits.by_name['arm_left_1_joint'].upper < .5
    assert limits.by_name['arm_right_5_joint'].lower < -3.
    assert limits.by_name['arm_left_5_joint'].lower > -2.
    assert limits.by_name['torso_lift_joint'].lower == pytest.approx(.05)
    assert limits.by_name['torso_lift_joint'].upper == pytest.approx(.30)


def test_live_caps_never_widen_narrower_urdf_limits(live_urdf):
    root = ET.fromstring(live_urdf)
    joint = root.find("joint[@name='arm_right_3_joint']/limit")
    joint.set('upper', '0.40')
    joint.set('velocity', '0.20')
    limits = JointLimits(ET.tostring(root, encoding='unicode'), 'real')
    bound = limits.by_name['arm_right_3_joint']
    assert bound.upper == pytest.approx(.35)
    assert bound.velocity == pytest.approx(.20)


def test_incompatible_urdf_and_live_envelopes_are_refused(live_urdf):
    root = ET.fromstring(live_urdf)
    joint = root.find("joint[@name='head_2_joint']/limit")
    joint.set('lower', '2.0')
    joint.set('upper', '3.0')
    with pytest.raises(ValueError, match='conflicting limits'):
        JointLimits(ET.tostring(root, encoding='unicode'), 'real')


@pytest.mark.parametrize('profile', ['simulation', 'real'])
def test_former_carry_angle_is_rejected_before_command(urdf, profile):
    limits = JointLimits(urdf, profile)
    with pytest.raises(ValueError, match='arm_right_2_joint.*outside'):
        limits.validate(['arm_right_2_joint'], [1.25])


@pytest.mark.parametrize('bad', [[float('nan')], [float('inf')], [0., 1.]])
def test_bad_command_data_is_refused(urdf, bad):
    with pytest.raises(ValueError):
        JointLimits(urdf).validate(['torso_lift_joint'], bad)


def test_real_segment_timing_respects_slowest_joint(live_urdf):
    limits = JointLimits(live_urdf, 'real')
    names = ['torso_lift_joint', 'arm_right_1_joint', 'arm_right_3_joint']
    start = np.array([.05, 0., 0.])
    path = np.array([[.15, .8, .9], [.30, 1., 1.5]])
    times = limits.times(names, start, path)
    velocity = np.array([limits.by_name[name].velocity for name in names])
    actual = np.abs(np.diff(np.vstack([start, path]), axis=0))/np.diff(np.r_[0., times])[:, None]
    assert np.all(actual <= .7*velocity+1e-9)


def test_send_rejects_out_of_range_or_overspeed_waypoints(live_urdf):
    node = ManipulationNode.__new__(ManipulationNode)
    node.model = ArmKinematics(live_urdf, profile='real')
    node.joints = {'arm_right_1_joint': 0.}
    with pytest.raises(ValueError, match='outside'):
        node.trajectory(['arm_right_1_joint'], [-1.], 1.)
    with pytest.raises(ValueError, match='velocity limits'):
        node.trajectory(['arm_right_1_joint'], [[.5], [1.]], 1., times=[.1, .2])
    command = node.trajectory(['arm_right_1_joint'], [1.], .01)
    p = command.points[-1]
    elapsed = p.time_from_start.sec+p.time_from_start.nanosec/1e9
    assert elapsed >= 1./(.7*.6)


def test_head_uses_loaded_bounds_and_measured_start(live_urdf):
    node = NavigationNode.__new__(NavigationNode)
    node.head_limits = JointLimits(live_urdf, 'real')
    node.arm_joints = {'head_1_joint': 1., 'head_2_joint': -.7}
    node.arm_joint_stamps = dict.fromkeys(node.arm_joints, 10.)
    node.now = lambda: 10.
    node.p = lambda key: 1.0
    node.head_pub, node.last_head = Pub(), None
    node.look(tilt=1.)
    command = node.head_pub.messages[-1]
    point = command.points[-1]
    node.head_limits.validate(command.joint_names, point.positions)
    elapsed = point.time_from_start.sec+point.time_from_start.nanosec/1e9
    actual = np.abs(np.array(point.positions)-np.array([1., -.7]))/elapsed
    assert np.max(actual) <= .7*.5+1e-9


def test_stale_head_feedback_cannot_start_motion(live_urdf):
    node = NavigationNode.__new__(NavigationNode)
    node.head_limits = JointLimits(live_urdf, 'real')
    node.arm_joint_stamps = {}
    node.now = lambda: 10.
    node.head_pub, node.last_head = Pub(), None
    node.look()
    assert not node.head_pub.messages
