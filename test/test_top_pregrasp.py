"""Top-row reach regression: hover may fail even when the book is reachable."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from solution_4to1mux.arm_stow import ArmStowPlanner
from solution_4to1mux.geometry import rotation
from ros_stubs import install

restore_ros = install()
from solution_4to1mux.manipulation_node import ManipulationNode
restore_ros()


def configuration(torso_limit):
    root = ET.fromstring((Path(__file__).parent/'data/erc_both_arm_chains.urdf').read_text())
    torso = root.find("joint[@name='torso_lift_joint']")
    torso.find('limit').set('upper', str(torso_limit))
    torso.find('safety_controller').set('soft_upper_limit', str(torso_limit))
    planner = ArmStowPlanner(ET.tostring(root, encoding='unicode'),
                            'base_footprint', 'gripper_right_base_link')
    joints = {name: 0. for model in planner.models.values() for name in model.names}
    current = planner.plan(joints, [1., 0.], 1.35, .62).targets['right']
    current[0] = planner.models['right'].upper[0]
    node = ManipulationNode.__new__(ManipulationNode)
    node.model = planner.models['right']
    node.base_to_grasp = node.model.grasp_transform
    node.book_row = 1
    params = dict(grasp_insert_depth=.085, pregrasp_distance=.10,
                  top_book_clearance_height=.16, top_book_front_standoff=.10,
                  top_book_arm_clearance=.06)
    # This produces the reported wrist waypoints near x=.421/.521, z=1.748.
    book = np.array([.693, -.109, 1.588])
    normal = np.array([1., 0., 0.])
    return node, current, book, normal, params


@pytest.fixture(scope='module')
def approach():
    node, current, book, normal, params = configuration(.35)
    phases = node.plan('prepare_grab', book, normal, current.copy(), params)
    return node, current, book, normal, params, phases


def test_reported_height_miss_is_not_accepted_by_relaxing_ik_tolerance(approach):
    node, *_ = approach
    orientation = rotation([1, 0, 0], math.pi) @ node.base_to_grasp[:3, :3].T
    seed = np.zeros(8)
    seed[0] = node.model.upper[0]
    with pytest.raises(ValueError, match='IK failed'):
        node.model.solve([.521, -.109, 1.748], orientation, seed,
                         fixed={0: seed[0]}, attempts=3)


def test_reachable_complete_top_approach_is_selected_within_loaded_limits(approach):
    node, current, book, normal, params, phases = approach
    assert 0 <= node.top_pregrasp_height < params['top_book_clearance_height']
    assert phases[0][1] == 'open'
    assert phases[-1][0] == 'pregrasp'
    for _, kind, values, _ in phases:
        if kind == 'motion':
            path = np.atleast_2d(values)
            node.model.limits.validate(node.model.names, path)
            assert path[:, 0] == pytest.approx(current[0])
    target = np.atleast_2d(phases[-1][2])[-1]
    expected = book+normal*(params['grasp_insert_depth']-params['pregrasp_distance'])
    assert np.linalg.norm(node.grasp_pose(target)[:3, 3]-expected) < .008
    assert node.retreat_target == pytest.approx(target)


def test_entire_interpolated_arm_approach_stays_in_front_of_shelf(approach):
    node, before, book, normal, params, phases = approach
    before = before.copy()
    for _, kind, values, _ in phases:
        if kind != 'motion':
            continue
        for target in np.atleast_2d(values):
            for fraction in np.linspace(0., 1., 80):
                q = before+fraction*(target-before)
                points = node.model.arm_points(q, node.base_to_grasp)
                assert np.max(points[:-1] @ normal)+params['top_book_arm_clearance'] <= book @ normal+.0031
                assert points[-1] @ normal <= book @ normal+.0031
            before = target


def test_stricter_torso_limit_does_not_get_widened_to_force_a_path():
    node, current, book, normal, params = configuration(.30)
    with pytest.raises(ValueError, match='No reachable shelf-clear top-row pregrasp'):
        node.plan('prepare_grab', book, normal, current, params)
    assert node.model.upper[0] == pytest.approx(.30)
    assert node.model.limits.by_name['torso_lift_joint'].upper == pytest.approx(.30)
