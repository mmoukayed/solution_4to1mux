"""Model-based stow and controller-feedback regression checks (no ROS transport)."""
from concurrent.futures import Future
from pathlib import Path
import numpy as np
import pytest

from solution_4to1mux.arm_stow import ArmStowPlanner, StowPlan
from solution_4to1mux.ros_topics import get_ros_config
from ros_stubs import install, Logger, Pub

restore_ros = install()
from solution_4to1mux.navigation_node import NavigationNode, State
restore_ros()


@pytest.fixture(scope='module')
def planner():
    urdf = (Path(__file__).parent/'data/erc_both_arm_chains.urdf').read_text()
    return ArmStowPlanner(urdf, 'base_footprint', 'gripper_right_base_link')


def measured_joints(planner, torso=0.15):
    joints = {'torso_lift_joint': torso}
    for model in planner.models.values():
        joints.update(zip(model.names[1:], [0.0, 0.9, 0.0, -1.0, 0.0, 0.0, 0.0]))
    return joints


@pytest.mark.parametrize('torso', [0.0, 0.30])
def test_stow_uses_both_urdf_chains_and_preserves_measured_torso(planner, torso):
    joints = measured_joints(planner, torso)
    plan = planner.plan(joints, [1.0, 0.0], 1.35, 0.62)
    for side, q in plan.targets.items():
        model = planner.models[side]
        assert np.all(q >= model.lower) and np.all(q <= model.upper)
        assert q[0] == pytest.approx(torso)
        points = planner.points(model, q)
        assert points[:, 0].max()+0.06 <= 0.402
        assert np.abs(points[:, 1]).max()+0.06 <= 0.552
        start = np.array([joints[name] for name in model.names])
        assert plan.duration >= model.duration(start, q)
        for a in np.linspace(0.0, 1.0, 60):
            path = planner.points(model, start+a*(q-start))
            assert path[:, 0].max()+0.06+0.08 < 1.35


def test_already_compact_arms_keep_their_measured_pose(planner):
    first = planner.plan(measured_joints(planner), [1.0, 0.0], 1.35, 0.62)
    joints = {name: value for side, model in planner.models.items()
              for name, value in zip(model.names, first.targets[side])}
    second = planner.plan(joints, [1.0, 0.0], 1.35, 0.62)
    for side in planner.models:
        assert second.targets[side] == pytest.approx(first.targets[side])


@pytest.mark.parametrize('torso', [0.0, 0.15])
def test_stow_accepts_default_extended_arms_below_transport_height(planner, torso):
    # The simulator starts both arms at zero with a lowered torso. This is
    # clear of the ground but below the desired final transport envelope.
    joints = {name: 0.0 for model in planner.models.values() for name in model.names}
    joints['torso_lift_joint'] = torso
    plan = planner.plan(joints, [1.0, 0.0], 1.35, 0.62)
    for side, model in planner.models.items():
        start = np.array([joints[name] for name in model.names])
        target = plan.targets[side]
        assert target[0] == torso
        assert np.all(target >= model.lower) and np.all(target <= model.upper)
        end = planner.points(model, target)
        assert end[:, 2].min()-0.06 >= 0.30-0.002
        assert end[:, 0].max()+0.06 <= 0.402
        assert np.abs(end[:, 1]).max()+0.06 <= 0.552
        for fraction in np.linspace(0.0, 1.0, 120):
            points = planner.points(model, start+fraction*(target-start))
            assert points[:, 0].max()+0.06+0.08 <= 1.35+0.003
            assert points[:, 2].min()-0.06 >= 0.08-0.003


@pytest.mark.parametrize('obstacle', ['shelf', 'ground'])
def test_extended_start_does_not_relax_physical_obstacle_clearance(planner, obstacle):
    joints = {name: 0.0 for model in planner.models.values() for name in model.names}
    wall_distance = 1.35
    if obstacle == 'shelf':
        wall_distance = 0.95  # Extended fingers already cross this shelf plane.
    else:
        joints['arm_left_2_joint'] = -0.5  # Fingertips below the ground.
    with pytest.raises(ValueError, match=f'insufficient {obstacle} clearance'):
        planner.plan(joints, [1.0, 0.0], wall_distance, 0.62)


def test_stow_rejects_too_close_shelf_and_joint_limit_violation(planner):
    joints = measured_joints(planner)
    with pytest.raises(ValueError, match='shelf clearance|No compact'):
        planner.plan(joints, [1.0, 0.0], 0.15, 0.15)
    joints['arm_left_2_joint'] = 10.0
    with pytest.raises(ValueError, match='URDF limits'):
        planner.plan(joints, [1.0, 0.0], 1.35, 0.62)


def nav_with_finished_plan(planner):
    node = NavigationNode.__new__(NavigationNode)
    node.ros = get_ros_config('simulation')
    node.state = State.STOW_ARMS_FOR_DOCK
    node.stow_planner = planner
    node.arm_joints = measured_joints(planner)
    node.arm_joint_stamps = dict.fromkeys(node.arm_joints, 10.0)
    node.arm_stow_start = node.arm_joints.copy()
    node.arm_stow_origin = node.pose = np.zeros(3)
    node.arm_stow_future = Future()
    targets = {side: np.array([node.arm_joints[name] for name in model.names])
               for side, model in planner.models.items()}
    # Use a valid target different from the measured pose to model a stalled arm.
    targets['right'][1] -= 0.25
    node.arm_stow_future.set_result(StowPlan(targets, 3.0))
    node.arm_stow_plan = None
    node.arm_stow_last = node.arm_stow_confirmed_since = None
    node.left_arm_pub, node.right_arm_pub = Pub(), Pub()
    node.now = lambda: 10.0
    node.fresh = lambda stamp: stamp is not None and 0 <= node.now()-stamp <= 1.0
    node.p = {'arm_stow_joint_tolerance': 0.06, 'arm_stow_settle_time': 0.3}.__getitem__
    node.warn = lambda *args: None
    node.get_logger = lambda: Logger()
    node.fail = lambda reason: pytest.fail(reason)
    return node


def test_stow_does_not_accept_elapsed_time_without_joint_confirmation(planner):
    node = nav_with_finished_plan(planner)
    assert not node.stow_arms_for_dock()
    node.now = lambda: 15.0
    node.arm_joint_stamps = dict.fromkeys(node.arm_joints, 15.0)
    assert not node.stow_arms_for_dock()
    assert len(node.left_arm_pub.messages) == len(node.right_arm_pub.messages) == 1
    for side, model in planner.models.items():
        node.arm_joints.update(zip(model.names, node.arm_stow_plan.targets[side]))
    assert not node.stow_arms_for_dock()  # First confirmed feedback sample.
    node.now = lambda: 15.4
    node.arm_joint_stamps = dict.fromkeys(node.arm_joints, 15.4)
    assert node.stow_arms_for_dock()


def test_stow_refuses_stale_individual_arm_feedback(planner):
    node = nav_with_finished_plan(planner)
    node.arm_joint_stamps['arm_left_4_joint'] = 1.0
    assert not node.stow_arms_for_dock()
    assert not node.left_arm_pub.messages and not node.right_arm_pub.messages


def test_stow_refuses_joints_that_changed_while_planning(planner):
    node = nav_with_finished_plan(planner)
    failures = []
    node.fail = failures.append
    node.arm_joints['arm_right_2_joint'] += 0.2
    assert not node.stow_arms_for_dock()
    assert failures and 'outdated trajectory' in failures[0]
    assert not node.right_arm_pub.messages


def test_stow_failure_replaces_active_motion_with_a_measured_hold(planner):
    node = nav_with_finished_plan(planner)
    assert not node.stow_arms_for_dock()
    node.failure_pub, node.cancel_pub, node.state_pub = Pub(), Pub(), Pub()
    node.transition = lambda state: setattr(node, 'state', state)
    NavigationNode.fail(node, 'Arm stow timed out.')
    assert node.state == State.FAILED
    for side, publisher in (('left', node.left_arm_pub), ('right', node.right_arm_pub)):
        assert len(publisher.messages) == 2
        hold = publisher.messages[-1]
        assert hold.points[0].positions == pytest.approx(
            [node.arm_joints[name] for name in planner.models[side].names[1:]])
