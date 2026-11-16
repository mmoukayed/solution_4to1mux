"""Vertical book release, bin recentering and measured release interlocks."""
import math
from pathlib import Path

import numpy as np
import pytest

from solution_4to1mux.geometry import ArmKinematics, rotation
from solution_4to1mux.ros_topics import get_ros_config
from ros_stubs import install, Logger, Pub

restore_ros = install()
from solution_4to1mux.manipulation_node import ManipulationNode, GRIPPER_JOINT
restore_ros()


@pytest.fixture(scope='module')
def model():
    return ArmKinematics((Path(__file__).parent/'data/erc_right_arm_chain.urdf').read_text())


def node_for_model(model):
    node = ManipulationNode.__new__(ManipulationNode)
    node.ros = get_ros_config('simulation')
    node.model = model
    node.base_to_grasp = model.fixed_transform('gripper_right_base_link', 'gripper_right_grasping_link')
    node.placement_hover_target = node.placement_release_target = None
    return node


def parameters():
    return dict(bin_inset=.22, bin_release_height=.31, place_clearance_height=.10,
                place_torso_hover=.35, place_correction_limit=.12)


@pytest.fixture(scope='module')
def prepared(model):
    node = node_for_model(model)
    point, normal = np.array([.95, 0., .95]), np.array([1., 0., 0.])
    current = np.r_[.15, np.zeros(7)]
    phases = node.plan('prepare_place', point, normal, current, parameters())
    return node, point, normal, phases


@pytest.mark.parametrize('bin_height,lateral', [(.85, -.18), (.95, 0.), (1.195, 0.)])
def test_hover_centres_the_grasp_frame_and_keeps_book_vertical(model, bin_height, lateral):
    node = node_for_model(model)
    point, normal = np.array([.95, lateral, bin_height]), np.array([1., 0., 0.])
    params = parameters()
    phases = node.plan('prepare_place', point, normal, np.r_[.15, np.zeros(7)], params)
    q = phases[-1][2]
    actual = node.grasp_pose(q)
    expected = point-normal*.22+np.array([0., 0., .41])
    assert np.linalg.norm(actual[:3, 3]-expected) < .008
    assert abs(actual[2, 2]) > math.cos(.05)
    node.model.limits.validate(node.model.names, q)
    # The old code treated this target as a wrist pose, losing this offset.
    assert np.linalg.norm(model.fk(q)[:3, 3]-expected) > .14


def test_old_quarter_turn_wrist_pose_makes_book_horizontal_and_off_centre(model):
    node = node_for_model(model)
    old_wrist_rotation = rotation([1., 0., 0.], math.pi/2)
    old_grasp_rotation = old_wrist_rotation @ node.base_to_grasp[:3, :3]
    old_offset = old_wrist_rotation @ node.base_to_grasp[:3, 3]
    assert abs(old_grasp_rotation[2, 2]) < .01
    assert abs(old_offset[1]) > .15


def test_finish_realigns_to_fresh_bin_target_then_lowers_only_torso(prepared):
    original, point, normal, prep_phases = prepared
    node = node_for_model(original.model)
    node.placement_hover_target = original.placement_hover_target.copy()
    node.placement_release_target = original.placement_release_target.copy()
    hover_q = prep_phases[-1][2]
    updated = point+np.array([.02, -.015, 0.])
    phases = node.plan('finish_place', updated, normal, hover_q.copy(), parameters())
    assert [p[0] for p in phases] == [
        'align_vertical_over_bin', 'confirm_bin_hover', 'lower_to_bin',
        'confirm_bin_release', 'release_book', 'verify_delivery', 'clear_bin']
    corrected = np.atleast_2d(phases[0][2])[-1]
    lower = phases[2][2]
    assert phases[2][1] == phases[-1][1] == 'torso'
    assert lower[1:] == pytest.approx(corrected[1:])
    assert corrected[0]-lower[0] == pytest.approx(.10)
    target = updated-normal*.22+np.array([0., 0., .31])
    actual = node.grasp_pose(lower)
    assert np.linalg.norm(actual[:3, 3]-target) < .008
    assert abs(actual[2, 2]) > math.cos(.05)
    node.model.limits.validate(node.model.names, lower)


def test_large_bin_target_shift_refuses_descent(prepared):
    node, point, normal, phases = prepared
    with pytest.raises(ValueError, match='alignment changed too far'):
        node.plan('finish_place', point+np.array([.2, 0., 0.]), normal,
                  phases[-1][2], parameters())


@pytest.fixture
def release_node(prepared):
    original, point, normal, prep = prepared
    node = node_for_model(original.model)
    node.placement_hover_target = original.placement_hover_target.copy()
    node.placement_release_target = original.placement_release_target.copy()
    phases = node.plan('finish_place', point, normal, prep[-1][2].copy(), parameters())
    q = next(phase[2] for phase in phases if phase[0] == 'lower_to_bin')
    node.joints = dict(zip(node.model.names, q))
    node.joints[GRIPPER_JOINT] = .01
    node.joint_stamp, node.placement_feedback_started = 10., 9.9
    node.now = lambda: 10.1
    node.p = dict(place_position_tolerance=.025, place_orientation_tolerance=.10,
                  gripper_open=.066, gripper_closed=0., gripper_timeout=8.,
                  delivery_contact_hold=.5).__getitem__
    node.holding_book = 'book_A'
    node.contacts = {('book_A', 'left'): 10., ('book_A', 'right'): 10.}
    node.active, node.phase = 'finish_place', None
    node.phases = [('release_book', 'open', None, 1.)]
    node.gripper_pub = Pub()
    node.get_logger = lambda: Logger()
    node.failures = []
    node.abort = node.failures.append
    node.bin_contact_stamp = 9.8
    node.ready = lambda: True
    node.ready_pub, node.holding_pub = Pub(), Pub()
    return node


def test_off_centre_book_does_not_get_released(release_node):
    node = release_node
    node.placement_release_target[1, 3] += .10
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'misaligned' in node.failures[0]


def test_horizontal_book_does_not_get_released(release_node):
    node = release_node
    pose = node.placement_release_target.copy()
    pose[:3, :3] = rotation([1., 0., 0.], math.pi/2)
    node.grasp_pose = lambda joints: pose
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'not vertical' in node.failures[0]


def test_stale_feedback_does_not_get_released(release_node):
    node = release_node
    node.joint_stamp = 9.8
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'fresh placement' in node.failures[0]


def test_missing_book_contact_does_not_get_released(release_node):
    node = release_node
    node.contacts = {}
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'book contact' in node.failures[0]


@pytest.mark.parametrize('source', ['joints', 'transform'])
def test_nonfinite_placement_feedback_does_not_get_released(release_node, source):
    node = release_node
    if source == 'joints':
        node.joints['arm_right_3_joint'] = float('nan')
    else:
        node.grasp_pose = lambda joints: np.full((4, 4), float('nan'))
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'Nonfinite' in node.failures[0]


def test_confirmed_vertical_release_opens_once_and_clears_old_rim_contact(release_node):
    node = release_node
    node.next_phase()
    assert not node.failures
    assert len(node.gripper_pub.messages) == 1
    assert node.gripper_pub.messages[0].points[0].positions == [.066]
    assert node.bin_contact_stamp is None
    assert node.release_started == 10.1


def delivery_node(node):
    node.phase, node.phase_started, node.deadline = 'delivery', 10., 20.
    node.release_started = 10.
    node.delivery_contact_since = None
    node.joints[GRIPPER_JOINT] = .066
    node.contacts = {}
    node.completed = []
    node.next_phase = lambda: node.completed.append(True)
    return node


def test_delivery_needs_sustained_contact_after_opening(release_node):
    node = delivery_node(release_node)
    node.bin_contact_stamp = 9.9
    node.tick()
    assert not node.completed  # Old rim contact before opening is insufficient.
    node.bin_contact_stamp, node.joint_stamp = 10.2, 10.2
    node.now = lambda: 10.2
    node.tick()
    assert not node.completed
    node.bin_contact_stamp, node.joint_stamp = 10.8, 10.8
    node.now = lambda: 10.8
    node.tick()
    assert node.completed == [True] and node.holding_book is None


def test_transient_bin_brush_does_not_count_as_delivery(release_node):
    node = delivery_node(release_node)
    node.bin_contact_stamp, node.joint_stamp = 10.2, 10.2
    node.now = lambda: 10.2
    node.tick()
    node.joint_stamp = 10.8
    node.now = lambda: 10.8
    node.tick()
    assert not node.completed and node.delivery_contact_since is None
