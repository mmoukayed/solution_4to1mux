"""Finite torso clearance and safe stow from a centreline-crossing reach."""
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from solution_4to1mux.arm_stow import ArmStowPlanner
from solution_4to1mux.torso_clearance import TorsoClearance, _mesh_bounds


DATA = Path(__file__).parent/'data'
BOX = [(np.array([-.2, -.2, .3]), np.array([.25, .2, 1.0]))]


def test_forward_reach_can_cross_robot_centreline():
    points = np.array([[.8, -.3, .8], [.8, .15, .8]])
    assert TorsoClearance.segment_margins(points, BOX, .06)[0] == pytest.approx(.49)


def test_link_through_torso_is_rejected_even_if_both_endpoints_are_outside():
    points = np.array([[.1, -.5, .8], [.1, .5, .8]])
    assert all(TorsoClearance.segment_margins(np.vstack([p, p]), BOX, .06)[0] > 0
               for p in points)
    assert TorsoClearance.segment_margins(points, BOX, .06)[0] < -.1


def test_above_torso_reach_is_clear_and_near_front_still_requires_capsule_margin():
    assert TorsoClearance.segment_margins(np.array([[0., 0., 1.2], [.1, 0., 1.2]]), BOX, .06)[0] > 0
    assert TorsoClearance.segment_margins(np.array([[.28, -.1, .8], [.28, .1, .8]]), BOX, .06)[0] < 0


def test_saved_extended_arm_stows_with_finite_body_and_shelf_clearance():
    planner = ArmStowPlanner((DATA/'erc_both_arm_chains.urdf').read_text(),
                            'base_footprint', 'gripper_right_base_link')
    sample = json.loads((DATA/'grasp_branch_jump.json').read_text())
    joints = {'torso_lift_joint': sample['joints']['torso_lift_joint']}
    for model in planner.models.values():
        joints.update(zip(model.names[1:], [0., .9, 0., -1., 0., 0., 0.]))
    joints.update(sample['joints'])
    right = planner.models['right']
    q = np.array([joints[name] for name in right.names])
    points = planner.points(right, q)
    # This is the false-positive condition in the old infinite side plane.
    assert np.min(-points[3:, 1]-(abs(points[0, 1])+.06)) < -.25
    assert len(planner.torso.boxes) == 3
    plan = planner.plan(joints, [1., 0.], 1.35, .62)
    for side, model in planner.models.items():
        start = np.array([joints[name] for name in model.names])
        target = plan.targets[side]
        model.limits.validate(model.names[1:], target[1:])
        assert target[0] == start[0]
        boxes = planner.torso.bounds(model, start)
        for fraction in np.linspace(0., 1., 120):
            points = planner.points(model, start+fraction*(target-start))
            assert TorsoClearance.segment_margins(points[3:], boxes, .06).min() >= -.0031
            assert points[:, 0].max()+.06+.08 <= 1.35+.0031
            assert points[:, 2].min()-.06 >= .08-.0031


def test_stl_mesh_bounds_support_binary_and_ascii(tmp_path):
    vertices = [[-1., 2., 3.], [4., -5., 6.], [0., 1., -2.]]
    path = tmp_path/'body.stl'
    path.write_bytes(bytes(80)+struct.pack('<I', 1)+struct.pack('<12fH', 0., 0., 1.,
                                                             *np.asarray(vertices).ravel(), 0))
    lower, upper = _mesh_bounds(str(path))
    assert lower == pytest.approx([-1., -5., -2.])
    assert upper == pytest.approx([4., 2., 6.])
    path.write_text('solid body\n'+'\n'.join('vertex '+' '.join(map(str, p)) for p in vertices)+'\nendsolid\n')
    assert _mesh_bounds(str(path))[0] == pytest.approx(lower)


def test_joint_only_urdf_uses_finite_fallback():
    root = ET.fromstring((DATA/'erc_both_arm_chains.urdf').read_text())
    for link in root.findall('link'):
        root.remove(link)
    planner = ArmStowPlanner(ET.tostring(root, encoding='unicode'),
                            'base_footprint', 'gripper_right_base_link')
    assert not planner.torso.boxes
    model = planner.models['right']
    bounds = planner.torso.bounds(model, np.zeros(8))
    points = np.array([[.8, -.1, .8], [.9, .1, .8]])
    assert TorsoClearance.segment_margins(points, bounds, .06).min() > 0


def test_real_torso_intrusion_is_not_relaxed_to_accept_an_invalid_start():
    planner = ArmStowPlanner((DATA/'erc_both_arm_chains.urdf').read_text(),
                            'base_footprint', 'gripper_right_base_link')
    joints = {'torso_lift_joint': .15}
    for model in planner.models.values():
        joints.update(zip(model.names[1:], [0., .9, 0., -1., 0., 0., 0.]))
    right = planner.models['right']
    values = [.15, -.4007633212, -2.2470231567, .3441820608, -.2785952825,
              -1.4654985624, -.9648584281, -1.5905880847]
    right.limits.validate(right.names, values)
    joints.update(zip(right.names, values))
    with pytest.raises(ValueError, match='right arm.*finite torso clearance'):
        planner.plan(joints, [1., 0.], 1.35, .62)


def test_lift_moves_upper_torso_bounds_and_keeps_fixed_base_bounds_stationary():
    planner = ArmStowPlanner((DATA/'erc_both_arm_chains.urdf').read_text(),
                            'base_footprint', 'gripper_right_base_link')
    model = planner.models['right']
    low = np.zeros(8)
    high = low.copy()
    high[0] = .30
    low_boxes, high_boxes = planner.torso.bounds(model, low), planner.torso.bounds(model, high)
    assert high_boxes[0] == pytest.approx(np.asarray(low_boxes[0]))
    assert high_boxes[1][0]-low_boxes[1][0] == pytest.approx([0., 0., .30])
