"""Regression cases derived from the 14 November logs; no live ROS required."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_regressions import (
    ROOT, ArmKinematics, ManipulationNode, PerceptionNode, State,
    bare_nav, bare_perception_tracker, make_heading_pose, shelf_perception,
)
from test_ros_topics import launch_nodes
from ros_stubs import Message, Pub, Time
from solution_4to1mux.alignment import MOTION_DEFAULTS, angle_delta, fit_shelf_line, read_joint_limits


def tune(node, **overrides):
    original = node.p
    node.p = lambda name: overrides[name] if name in overrides else original(name)


def test_yellow_spelling_from_log_reaches_perception():
    for name in ('book_color', 'book_colour'):
        nodes = launch_nodes(**{name: 'yellow'}, robot_profile='real')
        perception = next(n for n in nodes if n.executable == 'perception_node')
        assert perception.parameters[0]['book_colour'] == 'yellow'


def test_conflicting_colours_fail_instead_of_selecting_wrong_book():
    with pytest.raises(ValueError, match='Conflicting'):
        launch_nodes(book_color='yellow', book_colour='red')


def test_default_target_is_still_red_and_limits_shared_with_node():
    nodes, arguments = launch_nodes(_return_arguments=True)
    defaults = {a.name: a.default_value for a in arguments if hasattr(a, 'name')}
    for key, value in MOTION_DEFAULTS.items():
        assert float(defaults[key]) == value
    perception = next(n for n in nodes if n.executable == 'perception_node')
    assert perception.parameters[0]['book_colour'] == 'red'
    nav = next(n for n in nodes if n.executable == 'navigation_node').parameters[0]
    manip = next(n for n in nodes if n.executable == 'manipulation_node').parameters[0]
    assert nav['grasp_base_return_advance'].value.name == manip['grasp_base_advance'].value.name


def test_robust_line_rejects_outlier_instead_of_rotating_toward_it():
    heading = math.radians(65)
    normal = np.array([math.cos(heading), math.sin(heading)])
    tangent = np.array([-normal[1], normal[0]])
    centre = np.array([-4.8, 5.5])
    points = [np.r_[centre+offset*tangent+noise*normal, 1.71]
              for offset, noise in [(-1., .01), (0., -.01), (1., .01), (1.5, 0.)]]
    points.append(np.r_[centre+1.2*normal, 1.71])
    fitted = fit_shelf_line(points, centre-2.*normal)
    assert fitted is not None
    _, measured, inliers = fitted
    assert abs(angle_delta(measured, heading)) < math.radians(2)
    assert inliers.sum() == 4 and not inliers[-1]


def test_single_marker_never_defines_shelf_normal():
    assert fit_shelf_line([[3., 0., 1.71]], np.zeros(2)) is None
    n = bare_perception_tracker()
    for _ in range(4):
        assert not n.confirm_column(np.array([3., 0., 1.71]), .4, False)


def test_marker_heights_in_log_are_accepted():
    n = shelf_perception(1.71)
    values = dict(marker_min_height=1.60, marker_max_height=2.55,
                  column_half_width=.43, top_book_marker_offset=.335, row_spacing=.33)
    n.get_parameter = lambda key: NS(value=values[key])
    for _ in range(3):
        n.process_shelf()
    assert n.column_pub.messages[-1].data == 2


def test_normals_are_smoothed_and_preserved_when_only_one_marker_visible():
    n = bare_perception_tracker()
    point = np.array([-4.8, 5.5, 1.71])
    for _ in range(3):
        n.confirm_column(point, math.radians(74), True)
    old = n.column_pose.pose.orientation
    old_heading = 2*math.atan2(old.z, old.w)
    n.confirm_column(point, math.radians(63), True)
    updated = n.column_pose.pose.orientation
    heading = 2*math.atan2(updated.z, updated.w)
    assert math.radians(63) < heading < old_heading
    n.confirm_column(point, -1., False)
    result = n.column_pose.pose.orientation
    assert 2*math.atan2(result.z, result.w) == pytest.approx(heading)


def test_column_approach_tracks_new_heading_from_log():
    n = bare_nav()
    n.state = State.APPROACH_COLUMN
    old = make_heading_pose([-4.8, 5.5, 1.71], math.radians(74))
    new = make_heading_pose([-4.7, 5.45, 1.72], math.radians(63))
    n.goal = n.target_goal(old, 1.35)
    n.observations['shelf'] = new
    for _ in range(30):
        n.update_column_goal()
    expected = n.target_goal(new, 1.35)
    assert n.goal == pytest.approx(expected, abs=.001)


def test_docking_uses_one_frozen_plane_after_arm_stow():
    n = bare_nav()
    n.state = State.ALIGN_BOOK
    n.dock_reference = make_heading_pose([2., 0., 1.71], 0.)
    n.observations['shelf'] = make_heading_pose([2.2, .1, 1.71], .2)
    point, normal = n.wall_reference()
    assert point == pytest.approx([2., 0.])
    assert normal == pytest.approx([1., 0.])


def test_alignment_converges_with_live_shelf_heading_update():
    n = bare_nav()
    tune(n, **MOTION_DEFAULTS)
    n.look = lambda *args, **kwargs: None
    actual_heading = math.radians(63)
    normal = np.array([math.cos(actual_heading), math.sin(actual_heading)])
    tangent = np.array([-normal[1], normal[0]])
    marker = np.array([-4.7, 5.45])
    n.pose = np.r_[marker-1.15*normal+.20*tangent, actual_heading+.20]
    n.dock_reference = make_heading_pose([*marker, 1.71], math.radians(74))
    n.selected_book = make_heading_pose([*marker, 1.045], math.radians(74))
    n.goal = n.target_goal(n.selected_book, .68, -.18)
    n.state, n.entered = State.ALIGN_SHELF_FOR_DOCK, 10.
    for step in range(800):
        clock = 10.+step*.05
        n.now = lambda: clock
        n.odom_stamp = clock
        n.scans = {name: (clock, data[1]) for name, data in n.scans.items()}
        shelf = make_heading_pose([*marker, 1.71], actual_heading)
        book = make_heading_pose([*marker, 1.045], actual_heading)
        stamp = NS(sec=int(clock), nanosec=int((clock-int(clock))*1e9))
        shelf.header.stamp = book.header.stamp = stamp
        n.observations.update(shelf=shelf, book=book)
        n.tick()
        if n.state != State.ALIGN_SHELF_FOR_DOCK:
            break
        vx, vy, omega = n.last_command
        c, s = math.cos(n.pose[2]), math.sin(n.pose[2])
        n.pose += np.array([c*vx-s*vy, s*vx+c*vy, omega])*.05
    assert n.state == State.STOW_ARMS_FOR_DOCK, n.log.messages[-5:]
    assert abs(angle_delta(n.pose[2], actual_heading)) <= .025
    assert abs((n.goal[:2]-n.pose[:2]) @ tangent) <= .025
    assert -(n.pose[:2]-marker) @ normal >= 1.325
    assert not n.left_arm_pub.messages and not n.right_arm_pub.messages


def test_arm_pose_does_not_resume_based_on_time_alone():
    n = bare_nav()
    n.state = State.STOW_ARMS_FOR_DOCK
    n.arm_stow_last = 1.
    assert not n.stow_arms_for_dock()
    assert not n.cmd_pub.messages


def test_actual_urdf_limits_are_used_for_both_arm_commands():
    n = bare_nav()
    xml = (ROOT/'test/data/erc_full_robot.urdf').read_text()
    n.arm_limits = read_joint_limits(xml)
    n.state = State.STOW_ARMS_FOR_DOCK
    assert not n.stow_arms_for_dock()
    for publisher in (n.left_arm_pub, n.right_arm_pub):
        msg = publisher.messages[-1]
        point = msg.points[0]
        duration = point.time_from_start.sec+point.time_from_start.nanosec/1e9
        for name, q in zip(msg.joint_names, point.positions):
            lo, hi, velocity = n.arm_limits[name]
            assert lo <= q <= hi
            assert abs(q-n.arm_joints[name])/duration <= min(.4, .8*velocity)+1e-8


def test_grasp_return_speed_is_limited_and_heading_fixed():
    n = bare_nav()
    n.state = State.RETURN_FOR_GRAB
    n.drive_to(np.array([.3, 0., .025]), keep_heading=True)
    assert 0 < n.last_command[0] <= .08
    assert n.last_command[1:] == (0., 0.)


def test_acceleration_and_immediate_obstacle_stop():
    n = bare_nav()
    tune(n, **MOTION_DEFAULTS)
    for step in range(35):
        clock = 10.+step*.05
        n.now = lambda: clock
        n.scans = {name: (clock, data[1]) for name, data in n.scans.items()}
        previous = n.last_command[0]
        n.move(.45)
        assert n.last_command[0]-previous <= .35*.05+1e-8
    assert n.last_command[0] == pytest.approx(.45)
    n.scans['front'] = (clock, np.array([[.43, 0.]]))
    n.move(.45)
    assert n.last_command == (0., 0., 0.)


def test_full_margin_remains_in_effect_for_unreachable_old_top_fixture():
    model = ArmKinematics((ROOT/'test/data/erc_right_arm_chain.urdf').read_text())
    n = ManipulationNode.__new__(ManipulationNode)
    n.model, n.base_to_grasp = model, None
    params = dict(grasp_insert_depth=.025, pregrasp_distance=.06,
                  lift_distance=.025, extract_distance=.10)
    with pytest.raises(ValueError, match='IK failed'):
        n.plan('grab', np.array([.68, -.18, 1.595]), np.array([1., 0., 0.]), np.zeros(8), params)
    assert model.upper[0] == pytest.approx(.29)


def test_top_path_guard_uses_shelf_normal_and_requires_link_models():
    n = ManipulationNode.__new__(ManipulationNode)
    pose = np.eye(4)
    pose[:3, 3] = [0., .63, 1.15]
    n.link_models = [NS(names=['torso'], fk=lambda q: pose)]*4
    with pytest.raises(ValueError, match='shelf face'):
        n.check_shelf_clear(np.zeros(8), [np.zeros(8)], np.array([0., .66, 1.37]), np.array([0., 1., 0.]))
    n.link_models = []
    with pytest.raises(ValueError, match='all four'):
        n.check_shelf_clear(np.zeros(8), [np.zeros(8)], np.array([.66, 0., 1.37]))


def test_solver_can_reject_converged_but_unsafe_solution():
    model = ArmKinematics((ROOT/'test/data/erc_right_arm_chain.urdf').read_text())
    q = (model.lower+model.upper)/2
    actual = model.fk(q)
    with pytest.raises(ValueError, match='paths rejected'):
        model.solve(actual[:3, 3], actual[:3, :3], q, attempts=1, accept=lambda candidate: False)


def test_missing_visual_confirmation_does_not_count_as_grasp_success():
    n = ManipulationNode.__new__(ManipulationNode)
    n.active, n.phase, n.finger_feedback = 'finish_grab', 'verify', False
    n.phase_started, n.deadline = -6., 20.
    n.joint_stamp = 10.
    n.now = lambda: 10.
    n.holding_book = 'tentative'
    n.held_book = lambda: n.holding_book
    n.ready = lambda: True
    n.ready_pub, n.holding_pub = Pub(), Pub()
    n.seen_held = None
    errors = []
    n.abort = errors.append
    n.tick()
    assert errors and 'unverified' in errors[0]


def test_repeated_visual_requests_do_not_erase_existing_votes():
    n = PerceptionNode.__new__(PerceptionNode)
    n.held_until, n.held_votes = 20., [True, True]
    n.get_clock = lambda: NS(now=lambda: Time(nanoseconds=10_000_000_000))
    n.on_check_held(Message(data=True))
    assert n.held_votes == [True, True] and n.held_until == 20.


def test_registered_depth_uses_nearest_surface_when_points_overlap():
    n = PerceptionNode.__new__(PerceptionNode)
    depth = np.array([[2., 1.], [4., 3.]], dtype=np.float32)
    Kd = np.eye(3)
    Kc = np.diag([.1, .1, 1.])
    registered = n.register(depth, Kd, np.eye(4), Kc, 2, 2)
    assert registered[0, 0] == pytest.approx(1.)


def test_prepare_grab_uses_navigation_snapshot_not_latest_occluded_blob():
    from concurrent.futures import Future
    from test_regressions import target
    n = ManipulationNode.__new__(ManipulationNode)
    n.active = n.plan_future = None
    n.ros = bare_nav().ros
    n.readiness_issues = lambda: []
    n.selected_grasp_target = target(.75, -.07, 1.2)
    n.targets = {'book': target(9., 9., 9.)}
    n.selected_grasp_geometry = dict(point=[.75,-.07,1.2], shelf_point=[.75,0.,2.], normal=[1.,0.,0.], left=-.4, right=.4, width=.04, height=.24, stamp=10., depth_supported=True)
    n.now = lambda: 10.
    n.book_row, n.token, n.hand_corr = 2, 0, None
    n.model = NS(names=['torso_lift_joint']+[f'arm_right_{i}_joint' for i in range(1, 8)])
    n.joints = {name: 0. for name in n.model.names}
    values = dict(grasp_insert_depth=.085, grasp_right_offset=0., pregrasp_distance=.1,
                  lift_distance=.025, extract_distance=.1, grasp_base_advance=.28,
                  top_book_clearance_height=.12, top_book_front_standoff=.1,
                  tf_max_age=1.0, gripper_half_width=.065, divider_margin=.015, place_base_advance=.35, place_clearance_height=.2, place_torso_hover=.35,
                  place_torso_drop=0., bin_inset=.2, bin_release_height=.21,
                  bin_release_z=1.08, bin_release_x=.78, planning_timeout=60.)
    n.p = values.__getitem__
    tf = NS(header=NS(stamp=NS(sec=10, nanosec=0)), transform=NS(translation=NS(x=0., y=0., z=0.), rotation=NS(x=0., y=0., z=0., w=1.)))
    n.tf = NS(lookup_transform=lambda *args: tf)
    submitted = []
    n.pool = NS(submit=lambda *args: submitted.append(args) or Future())
    n.get_logger = lambda: NS(info=lambda text: None)
    n.request('prepare_grab')
    assert submitted[0][2] == pytest.approx([.75, -.07, 1.2])
    assert n.grab_target is n.selected_grasp_target


@pytest.mark.parametrize('row,z,standoff', [(2, 1.265, .68), (3, .935, .92), (4, .605, .92)])
def test_calibrated_middle_lower_grasp_plans_keep_controller_limits(row, z, standoff):
    xml = (ROOT/'test/data/erc_full_robot.urdf').read_text()
    model = ArmKinematics(xml)
    n = ManipulationNode.__new__(ManipulationNode)
    n.model, n.book_row = model, row
    n.base_to_grasp = model.fixed_transform('gripper_right_base_link', 'gripper_right_grasping_link')
    current = np.clip(np.r_[.1, [0., 1.07, 0., -1., .4, 0., 0.]], model.lower, model.upper)
    params = dict(grasp_insert_depth=.085, pregrasp_distance=.1, lift_distance=.025,
                  extract_distance=.1, grasp_base_advance=.28)
    prepare = n.plan('prepare_grab', np.array([standoff+.25, -.07, z]), np.array([1., 0., 0.]), current, params)
    prepared = next(p[2] for p in reversed(prepare) if p[1] == 'motion')
    finish = n.plan('finish_grab', np.array([standoff-.03, -.07, z]), np.array([1., 0., 0.]), prepared, params)
    insert = next(p[2] for p in finish if p[0] == 'insert')
    actual = model.fk(insert) @ n.base_to_grasp
    assert actual[:3, 3] == pytest.approx([standoff-.03+.085, -.07, z], abs=.008)
    for _, kind, q, _ in prepare+finish:
        if kind == 'motion':
            assert np.all(q >= model.lower) and np.all(q <= model.upper)


def test_live_height_cutoff_does_not_change_simulation_calibration():
    for profile, expected in [('real', 1.60), ('simulation', 1.75)]:
        nodes = launch_nodes(robot_profile=profile)
        perception = next(n for n in nodes if n.executable == 'perception_node')
        assert perception.parameters[0]['marker_min_height'] == expected
    nodes = launch_nodes(robot_profile='real', marker_min_height='1.65')
    perception = next(n for n in nodes if n.executable == 'perception_node')
    assert perception.parameters[0]['marker_min_height'] == 1.65
