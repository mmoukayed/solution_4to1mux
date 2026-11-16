"""Grasp correction and measured close interlocks; no ROS transport or motion."""
import math
from pathlib import Path

import numpy as np
import pytest

from ros_stubs import install, Logger, Pub
from solution_4to1mux.geometry import ArmKinematics, rotation
from solution_4to1mux.ros_topics import get_ros_config

restore_ros = install()
from solution_4to1mux.manipulation_node import ManipulationNode, GRIPPER_JOINT
restore_ros()


@pytest.fixture(scope='module')
def model():
    return ArmKinematics((Path(__file__).parent/'data/erc_right_arm_chain.urdf').read_text())


def bare_manipulation(model):
    node = ManipulationNode.__new__(ManipulationNode)
    node.ros = get_ros_config('simulation')
    node.model = model
    node.base_to_grasp = model.fixed_transform(
        'gripper_right_base_link', 'gripper_right_grasping_link')
    node.book_row = 3
    node.close_target = None
    node.retreat_target = None
    return node


def solve_grasp(node, xyz, roll=0.0):
    orientation = rotation([1, 0, 0], roll) @ node.base_to_grasp[:3, :3].T
    tip_xyz = xyz-orientation @ node.base_to_grasp[:3, 3]
    return node.model.solve(tip_xyz, orientation, np.zeros(8))


@pytest.mark.parametrize('row,height,roll,gap', [
    (1, 1.595, math.pi, .10), (2, 1.265, 0.0, .03),
    (3, .935, math.pi, .03), (4, .605, 0.0, .03)])
def test_finish_corrects_short_base_return_before_closing(model, row, height, roll, gap):
    node = bare_manipulation(model)
    node.book_row = row
    book = np.array([.68, -.18, height])
    intended_grasp = book+np.array([.085, 0., 0.])
    # Base navigation accepts a dock 3 cm short of its commanded return.
    current = solve_grasp(node, intended_grasp-np.array([gap, 0., 0.]), roll=roll)
    node.retreat_target = current.copy()
    params = dict(grasp_insert_depth=.085, grasp_right_offset=0.,
                  grasp_correction_limit=.12, lift_distance=.025, extract_distance=.10)
    phases = node.plan('finish_grab', book, np.array([1., 0., 0.]), current, params)
    assert [phase[0] for phase in phases] == [
        'open_gripper', 'insert', 'confirm_grasp_alignment', 'close_gripper',
        'lift', 'extract', 'verify_book_held']
    inserted = np.atleast_2d(phases[1][2])[-1]
    actual = node.grasp_pose(inserted)
    assert np.linalg.norm(actual[:3, 3]-intended_grasp) < .008
    assert np.linalg.norm(actual[:3, 3]-node.grasp_pose(current)[:3, 3]) > .02
    model.limits.validate(model.names, inserted)
    assert np.linalg.norm(actual[:3, :3]-node.grasp_pose(current)[:3, :3]) < .12
    assert node.retreat_target == pytest.approx(current)


def test_captured_insert_branch_jump_has_a_continuous_solution(model):
    import json
    sample = json.loads((Path(__file__).parent/'data/grasp_branch_jump.json').read_text())
    node = bare_manipulation(model)
    current = np.array([sample['joints'][name] for name in model.names])
    book, normal = np.array(sample['point']), np.array(sample['normal'])
    phases = node.plan('finish_grab', book, normal, current, dict(
        grasp_insert_depth=.085, grasp_correction_limit=.12, lift_distance=.025, extract_distance=.10))
    before = current.copy()
    for _, kind, values, _ in phases:
        if kind != 'motion':
            continue
        path = np.atleast_2d(values)
        model.limits.validate(model.names, path)
        steps = np.diff(np.vstack([before, path]), axis=0)
        assert np.abs(steps[:, 1:]).max() <= .350001
        assert np.abs(steps[:, 0]).max() <= .010001
        before = path[-1]
    inserted = np.atleast_2d(phases[1][2])[-1]
    assert np.linalg.norm(node.grasp_pose(inserted)[:3, 3]-(book+.085*normal)) <= .012


def test_already_aligned_insert_reuses_measured_joints(model):
    node = bare_manipulation(model)
    book = np.array([.68, -.18, .935])
    current = solve_grasp(node, book+np.array([.085, 0., 0.]), roll=math.pi)
    phases = node.plan('finish_grab', book, np.array([1., 0., 0.]), current, dict(
        grasp_insert_depth=.085, grasp_correction_limit=.12, lift_distance=.025, extract_distance=.10))
    assert phases[1][0] == 'insert'
    assert phases[1][2] == pytest.approx(current)


def test_large_post_return_error_requires_redocking(model):
    node = bare_manipulation(model)
    current = solve_grasp(node, np.array([.60, -.18, .935]))
    params = dict(grasp_insert_depth=.085, grasp_correction_limit=.12)
    with pytest.raises(ValueError, match='correction.*redock'):
        node.plan('finish_grab', np.array([.68, -.18, .935]),
                  np.array([1., 0., 0.]), current, params)


@pytest.fixture
def aligned_node(model):
    node = bare_manipulation(model)
    current = solve_grasp(node, np.array([.765, -.18, .935]))
    node.joints = dict(zip(model.names, current))
    node.joints[GRIPPER_JOINT] = .066
    node.close_target = node.grasp_pose(current)
    node.grasp_feedback_started = 9.95
    node.joint_stamp = 10.0
    node.now = lambda: 10.1
    node.p = dict(grasp_position_tolerance=.012, grasp_orientation_tolerance=.10,
                  gripper_open=.066, gripper_closed=0., gripper_timeout=8.,
                  grip_contact_hold=1.).__getitem__
    node.active = 'finish_grab'
    node.contacts = {}
    node.holding_book = None
    node.phase = None
    node.phases = [('close_gripper', 'close', None, 1.)]
    node.gripper_pub = Pub()
    node.get_logger = lambda: Logger()
    node.failures = []
    node.abort = node.failures.append
    return node


def test_close_refused_if_measured_fingers_still_short_of_book(aligned_node):
    node = aligned_node
    node.close_target[:3, 3] += [.03, 0., 0.]
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'misses the book' in node.failures[0]


def test_close_refused_if_wrist_is_tilted(aligned_node):
    node = aligned_node
    node.close_target[:3, :3] = rotation([0, 1, 0], math.pi/6) @ node.close_target[:3, :3]
    node.next_phase()
    assert not node.gripper_pub.messages
    assert node.failures and 'orientation error' in node.failures[0]


def test_close_requires_feedback_after_insertion(aligned_node):
    node = aligned_node
    node.joint_stamp = 9.9  # Fresh, but predates the final insertion.
    node.next_phase()
    assert not node.gripper_pub.messages
    assert 'after insertion' in node.failures[0]


def test_close_requires_confirmed_open_fingers(aligned_node):
    node = aligned_node
    node.joints[GRIPPER_JOINT] = 0.
    node.next_phase()
    assert not node.gripper_pub.messages
    assert 'not confirmed open' in node.failures[0]


def test_aligned_open_gripper_closes_once_and_still_requires_contacts(aligned_node):
    node = aligned_node
    node.next_phase()
    assert not node.failures
    assert len(node.gripper_pub.messages) == 1
    assert node.gripper_pub.messages[0].points[0].positions == [0.]
    node.ready = lambda: True
    node.ready_pub, node.holding_pub = Pub(), Pub()
    node.joints[GRIPPER_JOINT] = 0.
    node.now = lambda: 11.5
    node.tick()
    assert node.phase == 'close' and node.holding_book is None
    assert len(node.gripper_pub.messages) == 1


def test_feedback_phase_waits_for_new_measurement_then_closes(aligned_node):
    node = aligned_node
    node.phases.insert(0, ('confirm_grasp_alignment', 'grasp_feedback', node.close_target, 2.))
    node.ready = lambda: True
    node.ready_pub, node.holding_pub = Pub(), Pub()
    node.next_phase()
    node.tick()
    assert node.phase == 'grasp_feedback' and not node.gripper_pub.messages
    node.joint_stamp = 10.2
    node.now = lambda: 10.25
    node.tick()
    assert node.phase == 'close' and len(node.gripper_pub.messages) == 1
    assert not node.failures


def test_failed_insertion_reports_error_without_closing(aligned_node):
    node = aligned_node
    node.close_target[:3, 3] += [.03, 0., 0.]
    node.phases.insert(0, ('confirm_grasp_alignment', 'grasp_feedback', node.close_target, 2.))
    node.ready = lambda: True
    node.ready_pub, node.holding_pub = Pub(), Pub()
    node.next_phase()
    node.now = lambda: 12.5
    node.joint_stamp = 12.5
    node.tick()
    assert not node.gripper_pub.messages
    assert node.failures and 'target minus measured xyz' in node.failures[0]
