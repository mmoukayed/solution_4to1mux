"""Run offline: python3 -m pytest -q test/test_regressions.py.

Pure calculations and mocked callbacks only; not a Gazebo success-rate test.
"""
import math
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from ros_stubs import install, Logger, Message, PoseStamped, Pub, Time
restore_ros = install()
from solution_4to1mux.geometry import (ArmKinematics, body_velocity, depth_median, forward_command,
                                      rotation, row_for_height, scan_points, wrap)
from solution_4to1mux.digit_detector import DigitDetector
from solution_4to1mux.navigation_node import NavigationNode, State
from solution_4to1mux.manipulation_node import ManipulationNode
from solution_4to1mux.perception_node import PerceptionNode
restore_ros()
# Retain the imported stand-in classes for these tests only; do not poison other imports.
import sys
for module in ('navigation_node', 'manipulation_node', 'perception_node'):
    sys.modules.pop('solution_4to1mux.'+module, None)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def model():
    return ArmKinematics((ROOT/'test/data/erc_right_arm_chain.urdf').read_text())


def bare_nav():
    n = NavigationNode.__new__(NavigationNode)
    values = dict(reading_max_age=1.0, obstacle_margin=0.12, linear_speed=0.85,
                  strafe_speed=0.25, rotate_speed=0.85, search_rotate_speed=0.45,
                  phase_timeout=90, search_direction='right', docking_speed=.25,
                  book_search_timeout=24., head_scan_dwell=1.8, max_book_recoveries=1,
                  scan_half_width=2.6, scan_step=.65, scan_observe_time=1.0,
                  shelf_keepout_distance=.62, arm_stow_timeout=10.0,
                  arm_stow_settle_time=2.5, grasp_base_retreat=.25,
                  grasp_base_return_advance=.28)
    n.p = values.__getitem__
    n.now = lambda: 10.0
    n.pose = np.array([0., 0., 0.])
    n.odom_stamp = 10.
    n.entered = 9.
    n.scans = {'front': (10., np.empty((0, 2))), 'rear': (10., np.empty((0, 2)))}
    n.cmd_pub = Pub()
    n.state_pub = Pub()
    n.failure_pub = Pub()
    n.manip_failure = ''
    n.book_diagnostics = 'test diagnostics'
    n.book_recoveries = 0
    n.cancel_pub = Pub()
    n.prepare_grab_pub, n.finish_grab_pub, n.grab_pub = Pub(), Pub(), Pub()
    n.bin_mode_pub = Pub()
    n.warned = {}
    n.log = Logger()
    n.get_logger = lambda: n.log
    n.observations = {}
    n.arm_joints = {f'arm_{side}_{index}_joint': 0.0
                    for side in ('left', 'right') for index in range(1, 8)}
    n.arm_joint_stamp = 10.0
    n.arm_stow_last = None
    n.left_arm_pub, n.right_arm_pub = Pub(), Pub()
    n.state = State.FIND_SHELF
    return n


def target(x=1., y=0., z=1., stamp=10):
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
    msg.header.stamp.sec = stamp
    return msg


def test_scan_forward_accounts_for_mount_yaw():
    matrix = np.eye(4)
    matrix[:3, :3] = rotation([0, 0, 1], -math.pi/4)
    # The +45 degree LASER beam is straight ahead in the BASE frame.
    points = scan_points([2], math.pi/4, 0.1, 0.05, 25, matrix)
    assert np.allclose(points, [[2, 0]], atol=1e-8)


def test_speed_vector_respects_diagonal_limit():
    assert math.hypot(*body_velocity(5, 5, 0, .65)) == pytest.approx(.65)
    assert body_velocity(1, 0, math.pi/2, .65) == pytest.approx((0, -.65))


def test_depth_rejects_nonfinite_and_background():
    depth = np.full((3, 3), np.inf)
    depth[0, 0] = np.nan
    depth[1, 1] = 0
    assert depth_median(depth, 1, 1) is None
    depth[1, 1] = 1.4
    assert depth_median(depth, 1, 1) == pytest.approx(1.4)


@pytest.mark.parametrize('row,z', [(1, 1.595), (2, 1.265), (3, .935), (4, .605)])
def test_row_is_world_height_not_image_quarter(row, z):
    assert row_for_height(2.26, z) == row
    # Moving the odom origin vertically must not change the row number.
    assert row_for_height(2.46, z+.2) == row


def test_observed_target_skips_search_rotation():
    n = bare_nav()
    n.look = lambda *args, **kwargs: None
    n.observations['shelf'] = target(3., 1., 2.26)
    n.tick()
    assert n.state == State.APPROACH_COLUMN
    assert all(m.angular.z == 0 for m in n.cmd_pub.messages)


def test_shortest_turn_across_pi_boundary():
    n = bare_nav()
    n.pose[2] = math.radians(179)
    n.drive_to(np.array([0, 0, math.radians(-175)]))
    command = n.cmd_pub.messages[-1]
    assert 0 < command.angular.z < .3
    assert abs(wrap(math.radians(-175)-math.radians(179))) == pytest.approx(math.radians(6))


def test_stale_scans_stop_instead_of_creeping():
    n = bare_nav()
    n.scans['front'] = (1., np.empty((0, 2)))
    n.move(.65)
    assert n.cmd_pub.messages[-1].linear.x == 0


def test_blocked_approach_does_not_report_arrival():
    n = bare_nav()
    n.scans['front'] = (10., np.array([[.40, 0.]]))
    arrived = n.drive_to(np.array([2., 0., 0.]))
    assert not arrived
    assert n.cmd_pub.messages[-1].linear.x == 0


def test_failed_grasp_never_returns_empty():
    n = bare_nav()
    n.p = lambda key: 120.0 if key == 'grab_timeout' else 1.0
    n.state = State.WAIT_FOR_GRAB
    n.result = False
    n.tick()
    assert n.state == State.FAILED
    assert n.cancel_pub.messages[-1].data is True


def test_wrong_column_larger_colour_blob_is_rejected():
    n = PerceptionNode.__new__(PerceptionNode)
    n.target_colour = 'red'
    n.image = np.zeros((160, 250, 3), np.uint8)
    n.image[40:100, 50:65] = (0, 0, 255)       # correct column
    n.image[30:150, 180:215] = (0, 0, 255)     # larger WRONG column
    n.column_pose = target(3., 0., 2.26)
    n.column_hits = 3
    n.point = lambda x, y: np.array([3.0, 0.1 if x < 100 else 1.0, .935])
    values = {'column_half_width': .43, 'top_book_marker_offset': .665, 'row_spacing': .33}
    n.get_parameter = lambda key: NS(value=values[key])
    n.latest_book, n.book_hits = None, 0
    n.book_pub, n.row_pub, n.book_diag_pub = Pub(), Pub(), Pub()
    n.reported = set()
    n.saved = []
    n.save = lambda b, label: n.saved.append((b, label))
    n.make_pose = lambda p, h: target(*p)
    n.get_logger = lambda: Logger()
    for _ in range(3):
        n.process_book()
    assert len(n.book_pub.messages) == 1
    assert n.book_pub.messages[0].pose.position.y == pytest.approx(.1)
    assert n.saved[0][0][0] == 50
    assert n.row_pub.messages[0].data == 3


def test_real_sim_digit_templates_match_supplied_frame():
    d = DigitDetector(str(ROOT/'templates'), min_area=25, match_threshold=.55)
    image = cv2.imread(str(ROOT/'erc_images/shelf_column_20260912_120246_121458.png'))
    results = d.detect(image)
    assert not d.using_synthetic
    assert any(digit == 2 and 420 < box[0] < 470 for digit, _, box in results)
    assert {digit for digit, _, box in results if box[1] > 20} == {2, 3, 4, 5}


@pytest.mark.parametrize('height', [.605, .935, 1.265, 1.595])
def test_grasp_waypoints_reach_all_four_rows_within_limits(model, height):
    n = ManipulationNode.__new__(ManipulationNode)
    n.model = model
    params = dict(grasp_insert_depth=.025, pregrasp_distance=.06, lift_distance=.025,
                  extract_distance=.10, bin_inset=.12, bin_release_height=.21)
    phases = n.plan('grab', np.array([.68, -.18, height]), np.array([1., 0., 0.]), np.zeros(8), params)
    expected = {'pregrasp': [.645, -.18, height], 'insert': [.705, -.18, height],
                'lift': [.705, -.18, height+.025], 'extract': [.605, -.18, height+.025]}
    for name, kind, q, _ in phases:
        if kind != 'motion':
            continue
        assert np.all(q >= model.lower) and np.all(q <= model.upper)
        assert np.linalg.norm(model.fk(q)[:3, 3]-expected[name]) < .008


def test_torso_duration_respects_actual_urdf_limit(model):
    start, end = np.zeros(8), np.zeros(8)
    end[0] = .35
    assert model.duration(start, end) >= .35/.035


def test_unreachable_target_rejected_before_motion(model):
    with pytest.raises(ValueError, match='IK failed'):
        model.solve([5, 0, 5], np.eye(3), np.zeros(8), attempts=1, iterations=15)


def test_grasp_needs_both_fingers_on_same_book():
    n = ManipulationNode.__new__(ManipulationNode)
    n.now = lambda: 10.
    n.contacts = {('book_A', 'left'): 10., ('book_B', 'right'): 10.}
    assert n.held_book() is None
    n.contacts[('book_A', 'right')] = 10.
    assert n.held_book() == 'book_A'
    n.now = lambda: 11.
    assert n.held_book() is None


def test_rejected_action_fails_instead_of_waiting_for_nav_timeout():
    n = ManipulationNode.__new__(ManipulationNode)
    n.token = 1
    n.active = 'grab'
    n.phase_name = 'pregrasp'
    failures = []
    n.abort = failures.append
    n.goal_response(NS(result=lambda: NS(accepted=False)), 1)
    assert failures and 'rejected' in failures[0]


def test_late_action_acceptance_after_cancel_is_cancelled():
    n = ManipulationNode.__new__(ManipulationNode)
    n.token, n.active = 2, None
    cancelled = []
    handle = NS(accepted=True, cancel_goal_async=lambda: cancelled.append(True))
    n.goal_response(NS(result=lambda: handle), 1)
    assert cancelled == [True]


def test_cancelled_result_with_zero_error_is_not_success():
    n = ManipulationNode.__new__(ManipulationNode)
    n.token, n.active, n.phase_name = 1, 'grab', 'insert'
    failures = []
    n.abort = failures.append
    n.motion_result(NS(result=lambda: NS(status=5, result=NS(error_code=0, error_string='cancelled'))), 1)
    assert failures


@pytest.mark.parametrize('goal', [(2., 1., 0.), (2., -1., math.pi/2),
                                 (-1., .2, 0.), (0., -2., math.pi)])
def test_forward_travel_reaches_goal_without_strafing(goal):
    pose = np.zeros(3)
    for _ in range(2000):
        vx, vy, omega, arrived = forward_command(
            goal[0]-pose[0], goal[1]-pose[1], pose[2], goal[2], .85, .85)
        assert vy == 0.0
        assert 0 <= vx <= .85
        if arrived:
            break
        pose[0] += vx*math.cos(pose[2])*.05
        pose[1] += vx*math.sin(pose[2])*.05
        pose[2] = wrap(pose[2]+omega*.05)
    assert arrived
    assert np.linalg.norm(pose[:2]-np.array(goal[:2])) <= .04
    assert abs(wrap(pose[2]-goal[2])) <= .04


def test_sideways_destination_causes_turn_before_forward_drive():
    n = bare_nav()
    n.drive_to(np.array([0., 2., 0.]))
    command = n.cmd_pub.messages[-1]
    assert command.linear.x == 0
    assert command.linear.y == 0
    assert command.angular.z > 0


def test_clear_straight_travel_uses_configured_cap():
    n = bare_nav()
    n.drive_to(np.array([3., 0., 0.]))
    assert n.cmd_pub.messages[-1].linear.x == pytest.approx(.85)
    assert n.cmd_pub.messages[-1].linear.y == 0


def test_small_docking_alignment_still_uses_mecanum_at_low_speed():
    n = bare_nav()
    n.drive_to(np.array([.03, .10, 0.]), docking=True)
    command = n.cmd_pub.messages[-1]
    assert command.linear.y > 0
    assert math.hypot(command.linear.x, command.linear.y) <= .25
    assert command.angular.z == 0


def test_retreat_reverses_without_turning_carried_book():
    n = bare_nav()
    n.drive_to(np.array([-.20, 0., 0.]), keep_heading=True)
    command = n.cmd_pub.messages[-1]
    assert command.linear.x < 0
    assert command.linear.y == 0
    assert command.angular.z == 0


def test_clearance_does_not_multiply_an_already_slow_docking_command():
    n = bare_nav()
    n.scans['front'] = (10., np.array([[.60, 0.]]))
    n.move(.05)
    assert n.cmd_pub.messages[-1].linear.x == pytest.approx(.05)


def test_scan_waypoint_does_not_move_with_robot_every_tick():
    n = bare_nav()
    n.observations['shelf_unit'] = target(3., 0., 2.26)
    n.look = lambda *args, **kwargs: None
    n.transition(State.SCAN_COLUMNS)
    goal = n.scan_goal.copy()
    for y in (0., -.1, -.2):
        n.pose[1] = y
        n.tick()
        assert np.array_equal(n.scan_goal, goal)
        assert n.cmd_pub.messages[-1].linear.y == 0


def test_scan_waypoints_remain_bounded_on_both_sides():
    n = bare_nav()
    n.observations['shelf_unit'] = target(3., 0., 2.26)
    n.transition(State.SCAN_COLUMNS)
    offsets = []
    for _ in range(30):
        offsets.append(n.scan_offset)
        n.advance_scan_waypoint()
    assert min(offsets) == pytest.approx(-2.6)
    assert max(offsets) == pytest.approx(2.6)


def test_stale_cached_book_does_not_pin_camera_forever():
    n = bare_nav()
    n.observations['book'] = target(1., 0., .93, stamp=1)
    looked = []
    n.look = lambda *args, **kwargs: looked.append((args, kwargs))
    n.scan_for_book(0.5)
    n.scan_for_book(2.0)
    n.scan_for_book(4.0)
    assert looked[0][0] == (n.observations['book'],)
    assert looked[1][1]['tilt'] != looked[2][1]['tilt']
    assert all(msg.linear.x == 0 and msg.linear.y == 0 for msg in n.cmd_pub.messages)


def test_missing_book_at_dock_enters_reacquisition_not_grab_or_failure():
    n = bare_nav()
    n.state = State.ALIGN_BOOK
    n.goal = n.pose.copy()
    n.look = lambda *args, **kwargs: None
    n.tick()
    assert n.state == State.REACQUIRE_BOOK
    assert not n.failure_pub.messages


def test_both_arms_fold_before_base_enters_align_book():
    n = bare_nav()
    n.look = lambda *args, **kwargs: None
    n.state = State.FIND_BOOK
    n.observations['book'] = target(1., 0., .935)
    n.tick()
    assert n.state == State.STOW_ARMS_FOR_DOCK
    n.tick()
    expected = ((n.left_arm_pub, [0.0, 1.25, 0.0, -1.0, -0.4, 0.0, 0.0]),
                (n.right_arm_pub, [0.0, 1.25, 0.0, -1.0, 0.4, 0.0, 0.0]))
    for publisher, pose in expected:
        command = publisher.messages[-1]
        assert len(command.joint_names) == 7
        assert command.points[0].positions == pytest.approx(pose)
    for side, pose in (('left', expected[0][1]), ('right', expected[1][1])):
        for index, value in enumerate(pose, 1):
            n.arm_joints[f'arm_{side}_{index}_joint'] = value
    n.tick()
    assert n.state == State.ALIGN_BOOK


def test_calibrated_arm_stow_settle_does_not_wait_for_exact_gazebo_feedback():
    n = bare_nav()
    n.look = lambda *args, **kwargs: None
    n.state = State.FIND_BOOK
    n.observations['book'] = target(1., 0., .935)
    n.tick()
    n.tick()  # Publish the two-second calibrated trajectories.
    n.now = lambda: 12.6
    n.arm_joint_stamp = 12.6
    n.tick()
    assert n.state == State.ALIGN_BOOK


def test_grasp_stages_retreat_prealign_and_base_advance_before_close():
    n = bare_nav()
    n.look = lambda *args, **kwargs: None
    n.state, n.goal = State.ALIGN_BOOK, n.pose.copy()
    n.observations['book'] = target(.68, 0., .935)
    n.tick()
    assert n.state == State.BACK_OUT_FOR_GRASP
    assert n.goal[0] == pytest.approx(-.25)
    n.pose[0] = -.25
    n.tick()
    assert n.state == State.WAIT_FOR_PREGRASP
    assert n.prepare_grab_pub.messages[-1].data
    n.result = True
    n.tick()
    assert n.state == State.RETURN_FOR_GRAB
    assert n.goal[0] == pytest.approx(.03)
    n.pose[0] = .03
    n.tick()
    assert n.state == State.WAIT_FOR_GRAB
    assert n.finish_grab_pub.messages[-1].data


def test_book_view_recovery_is_bounded_and_does_not_retry_a_grasp():
    n = bare_nav()
    n.state = State.REACQUIRE_BOOK
    n.entered = -20.
    n.tick()
    assert n.state == State.BACK_OUT_FOR_VIEW
    assert n.book_recoveries == 1
    assert n.goal[0] == pytest.approx(-.2)
    n.state = State.REACQUIRE_BOOK
    n.entered = -20.
    n.tick()
    assert n.state == State.FAILED
    assert 'reacquired' in n.failure_pub.messages[-1].data


def bare_perception_tracker():
    n = PerceptionNode.__new__(PerceptionNode)
    n.column_pose = None
    n.candidate_column_pose = None
    n.candidate_column_hits = 0
    n.column_normal_locked = False
    n.make_pose = lambda p, heading: make_heading_pose(p, heading)
    return n


def make_heading_pose(point, heading):
    result = target(*point)
    result.pose.orientation.z = math.sin(heading/2)
    result.pose.orientation.w = math.cos(heading/2)
    return result


def test_one_bad_marker_does_not_destroy_confirmed_column():
    n = bare_perception_tracker()
    for _ in range(3):
        n.confirm_column(np.array([3., 0., 2.26]), 0., True)
    confirmed = n.column_pose
    assert not n.confirm_column(np.array([3., 1., 2.26]), 0., True)
    assert n.column_pose is confirmed


def test_single_visible_sign_does_not_replace_measured_shelf_normal_with_bearing():
    n = bare_perception_tracker()
    for _ in range(3):
        n.confirm_column(np.array([3., 0., 2.26]), 0., True)
    for _ in range(3):
        n.confirm_column(np.array([3., 0., 2.26]), .6, False)
    assert n.column_pose.pose.orientation.z == pytest.approx(0.)


def test_manipulation_failure_reason_published_with_false_result():
    n = ManipulationNode.__new__(ManipulationNode)
    n.failure_pub, n.grab_pub, n.place_pub = Pub(), Pub(), Pub()
    n.get_logger = lambda: Logger()
    n.report('grab', False, 'close: no bilateral book contact')
    assert n.grab_pub.messages[-1].data is False
    assert 'bilateral book contact' in n.failure_pub.messages[-1].data


def test_virtual_shelf_wall_blocks_forward_motion_lidar_cannot_see():
    # The ERC shelf presents almost nothing at the 0.13 m LiDAR plane inside an
    # aisle, so the scan-based gate alone would report 'clear' into a collision.
    n = bare_nav()
    n.observations = {'shelf': target(0.40, 0., 2.26)}
    n.move(vx=0.5)
    assert n.cmd_pub.messages[-1].linear.x == 0.0
    assert n.last_gate.startswith('shelf_keepout')
    n.observations = {}
    n.move(vx=0.5)
    assert n.cmd_pub.messages[-1].linear.x > 0.0


def test_shelf_wall_does_not_block_normal_docking_standoff():
    n = bare_nav()
    n.observations = {'shelf': target(1.5, 0., 2.26)}  # plane 1.5 m ahead
    n.move(vx=0.25)
    assert n.cmd_pub.messages[-1].linear.x > 0.0


def test_dock_goal_clamped_when_book_reads_too_deep():
    n = bare_nav()
    n.p = dict(book_standoff=0.68, shelf_keepout_distance=.62).__getitem__
    n.observations = {'shelf': target(3.0, 0., 2.26)}
    clamped = n.clamped_dock_goal(np.array([2.9, 0., 0.]))
    assert clamped[0] == pytest.approx(3.0-0.62)
    ok = np.array([2.30, 0., 0.])
    assert np.allclose(n.clamped_dock_goal(ok), ok)


def test_deep_colour_blob_inside_shelf_is_rejected():
    n = PerceptionNode.__new__(PerceptionNode)
    n.target_colour = 'red'
    n.image = np.zeros((160, 250, 3), np.uint8)
    n.image[40:100, 50:65] = (0, 0, 255)
    n.column_pose = target(3., 0., 2.26)
    n.column_hits = 3
    depth_x = [3.40]  # 0.4 m behind the marker plane: a gap hit inside the shelf
    n.point = lambda x, y: np.array([depth_x[0], 0.1, .935])
    values = {'column_half_width': .43, 'top_book_marker_offset': .665, 'row_spacing': .33}
    n.get_parameter = lambda key: NS(value=values[key])
    n.latest_book, n.book_hits = None, 0
    n.book_pub, n.row_pub, n.book_diag_pub = Pub(), Pub(), Pub()
    n.reported = set()
    n.save = lambda b, label: None
    n.make_pose = lambda p, h: target(*p)
    n.get_logger = lambda: Logger()
    for _ in range(3):
        n.process_book()
    assert not n.book_pub.messages
    assert 'plane' in n.book_diag_pub.messages[-1].data
    assert 'plane_offsets_m' in n.book_diag_pub.messages[-1].data
    depth_x[0] = 2.83  # 0.17 m on the robot side: tolerated column-depth bias
    for _ in range(3):
        n.process_book()
    assert len(n.book_pub.messages) == 1


def test_wall_ignores_bearing_based_shelf_unit_centroid():
    n = bare_nav()
    n.observations = {'shelf_unit': target(0.40, 0., 2.26)}
    assert n.wall_reference() is None
    n.move(vx=0.5)
    assert n.cmd_pub.messages[-1].linear.x > 0.0


def test_confirmed_marker_plane_wins_over_depth_biased_book_for_keepout():
    n = bare_nav()
    # The book point may be accepted up to 20 cm behind the marker plane.
    # It must never move the physical shelf wall inward with it.
    n.observations = {'shelf': target(1.0, 0., 2.26),
                      'book': target(1.2, 0., .935)}
    point, _ = n.wall_reference()
    assert point[0] == pytest.approx(1.0)


def test_hard_shelf_keepout_blocks_inward_command_at_safe_dock_distance():
    n = bare_nav()
    n.pose[0] = .62
    n.observations = {'shelf': target(1.24, 0., 2.26)}
    n.move(vx=.10)
    assert n.cmd_pub.messages[-1].linear.x == 0.0
    assert n.last_gate.startswith('shelf_keepout_hard')
    assert n.blocked_since == 10.0
    # A retreat is always permitted once the gate has stopped docking.
    n.move(vx=-.10)
    assert n.cmd_pub.messages[-1].linear.x < 0.0


def test_blocked_docking_is_tracked_for_recovery_not_waited_out():
    n = bare_nav()
    n.observations = {'shelf': target(0.40, 0., 2.26)}
    n.move(vx=0.5)
    assert n.blocked_since == 10.0
    n.observations = {}
    n.move(vx=0.5)
    assert n.blocked_since is None


def test_grab_refused_when_base_jammed_against_shelf(model):
    n = ManipulationNode.__new__(ManipulationNode)
    n.model = model
    params = dict(grasp_insert_depth=.04, pregrasp_distance=.10, lift_distance=.025,
                  extract_distance=.10, bin_inset=.12, bin_release_height=.21)
    with pytest.raises(ValueError, match='redock'):
        n.plan('grab', np.array([.30, -.18, .935]), np.array([1., 0., 0.]), np.zeros(8), params)


def test_refused_jammed_grab_backs_out_and_redocks_instead_of_failing():
    n = bare_nav()
    values = dict(reading_max_age=1.0, phase_timeout=90., grab_timeout=120., place_timeout=90.,
                  max_book_recoveries=2, book_search_timeout=24.)
    n.p = values.__getitem__
    n.state = State.WAIT_FOR_GRAB
    n.result = False
    n.manip_failure = 'grab: No valid grasp/placement plan: Dock distance wrong for a safe grasp: book at 0.30 m in base frame, need 0.45-1.05 m; redock.'
    n.tick()
    assert n.state == State.BACK_OUT_FOR_VIEW
    assert n.book_recoveries == 1
    assert not n.failure_pub.messages
    # once the recovery budget is spent, it fails visibly instead of looping
    n.state = State.WAIT_FOR_GRAB
    n.result = False
    n.book_recoveries = 2
    n.tick()
    assert n.state == State.FAILED
    assert 'redock' in n.failure_pub.messages[-1].data
