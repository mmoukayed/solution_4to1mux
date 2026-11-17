"""Replays of the distant drop target and repeated book-view recoveries."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from test_regressions import (
    ROOT, ArmKinematics, HIGH_CARRY_RIGHT_ARM, ManipulationNode, PerceptionNode,
    State, bare_nav, make_heading_pose, target,
)
from ros_stubs import Logger, Pub
from solution_4to1mux.geometry import pose_xyz
from solution_4to1mux.target_geometry import masked_surface


def bin_scene(width=120, colour=(0, 0, 255)):
    n = PerceptionNode.__new__(PerceptionNode)
    n.image = np.full((200, 300, 3), 255, np.uint8)
    x = 150-width//2
    cv2.rectangle(n.image, (x, 60), (x+width, 140), colour, 8)
    n.depth_image = np.full((200, 300), 7.9, np.float32)
    red = (n.image[..., 2] > n.image[..., 1]+20) & (n.image[..., 1] == 0)
    n.depth_image[red] = 2.
    n.camera_to_odom = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.],
                                [0., -1., 0., .9], [0., 0., 0., 1.]])
    n.info = NS(k=[400., 0., 150., 0., 400., 100., 0., 0., 1.], width=300, height=200)
    n.get_parameter = lambda name: NS(value=.002)
    n.bin_candidate, n.bin_hits, n.bin_stamp = None, 0, None
    n.bin_pub = Pub()
    n.reported = set()
    n.save = lambda *args: None
    n.log = Logger()
    n.get_logger = lambda: n.log
    n.warn = n.log.warn
    n.make_pose = lambda p, heading: make_heading_pose(p, heading)
    n.observation_stamp = NS(sec=10, nanosec=0)
    return n, red


def advance(n, i):
    n.observation_stamp = NS(sec=10+i//8, nanosec=(i % 8)*125000000)
    n.surface_masks = {}
    n.process_bin()


@pytest.mark.parametrize('width,colour', [(120, (0, 0, 255)), (55, (0, 0, 65))])
def test_hollow_and_oblique_dark_red_rims_use_surface_not_background(width, colour):
    n, _ = bin_scene(width, colour)
    for i in range(3):
        advance(n, i)
    assert len(n.bin_pub.messages) == 1
    assert pose_xyz(n.bin_pub.messages[-1]) == pytest.approx([2., -.0025, .8975], abs=.025)
    assert 'coloured surface' in n.log.messages[-1]


def test_far_red_background_is_not_a_drop_box():
    n, _ = bin_scene()
    n.depth_image[:] = 7.9
    for i in range(4):
        advance(n, i)
    assert not n.bin_pub.messages
    assert 'surface_depth' in n.log.messages[-1]


def test_colour_size_rejects_a_held_red_book():
    n, _ = bin_scene(width=30)
    for i in range(3):
        advance(n, i)
    assert not n.bin_pub.messages


def test_same_frame_cannot_confirm_drop_box():
    n, _ = bin_scene()
    for _ in range(5):
        advance(n, 0)
    assert n.bin_hits == 1 and not n.bin_pub.messages


def test_bin_jump_and_gaps_restart_confirmation():
    n, red = bin_scene()
    advance(n, 0)
    advance(n, 1)
    n.depth_image[red] = 2.5
    advance(n, 2)
    assert n.bin_hits == 1 and not n.bin_pub.messages
    advance(n, 12)
    assert n.bin_hits == 1 and not n.bin_pub.messages


def test_bin_mode_heartbeat_does_not_reset_track_but_mode_change_does():
    n, _ = bin_scene()
    n.mode = 'bin'
    advance(n, 0)
    n.on_mode(NS(data=True))
    assert n.bin_hits == 1
    n.on_mode(NS(data=False))
    assert n.bin_hits == 0 and n.bin_candidate is None


def test_registration_holes_and_scattered_depth_do_not_invent_surface():
    n, red = bin_scene()
    n.depth_image[red] = np.linspace(.21, 3.99, red.sum())
    advance(n, 0)
    assert n.bin_candidate is None
    n.depth_image[red] = np.nan
    advance(n, 1)
    assert n.bin_candidate is None


def test_masked_book_depth_ignores_a_letter_gap_at_box_centre():
    mask = np.zeros((100, 100), np.uint8)
    mask[15:85, 40:60] = 255
    depth = np.full((100, 100), 6., np.float32)
    depth[mask > 0] = 1.2
    mask[48:53, 48:53] = 0
    depth[48:53, 48:53] = 6.
    info = NS(k=[100., 0., 50., 0., 100., 50., 0., 0., 1.], width=100, height=100)
    p, width, height = masked_surface((40, 15, 20, 70), mask, depth, np.eye(4), info)
    assert p == pytest.approx([0., 0., 1.2])
    assert width == pytest.approx(.24) and height == pytest.approx(.84)


def test_bin_pose_from_log_cannot_start_approach_or_arm_swing():
    n = bare_nav()
    n.state = State.FIND_BIN
    n.observations['bin'] = make_heading_pose([8.431, .277, .666], 0.)
    n.look = lambda *args, **kwargs: None
    n.search = lambda: None
    n.tick()
    assert n.state == State.FIND_BIN and not n.prepare_place_pub.messages


def test_stale_bin_stops_approach_and_reacquires_without_arm_request():
    n = bare_nav()
    n.state = State.APPROACH_BIN
    n.goal = np.array([.5, 0., 0.])
    n.observations['bin'] = target(1., 0., .85, stamp=1)
    n.tick()
    assert n.state == State.APPROACH_BIN and not n.prepare_place_pub.messages
    assert n.last_command == (0., 0., 0.)
    n.now = lambda: 13.1
    n.odom_stamp = n.holding_stamp = 13.1
    n.scans = {name: (13.1, points) for name, (_, points) in n.scans.items()}
    n.tick()
    assert n.state == State.FIND_BIN


def test_bin_centre_update_requires_lateral_alignment_before_prepare():
    n = bare_nav()
    n.state = State.APPROACH_BIN
    n.goal = np.array([0., .0, 0.])
    n.observations['bin'] = make_heading_pose([1., .277, .85], 0.)
    n.look = lambda *args, **kwargs: None
    n.tick()
    assert n.state == State.APPROACH_BIN and not n.prepare_place_pub.messages
    assert n.last_command[1] > 0
    n.pose[1] = .277+.18
    n.tick()
    assert n.state == State.WAIT_FOR_PLACE_PREP
    assert n.place_target_pub.messages[-1] is n.placement_bin
    assert pose_xyz(n.placement_bin)[1]-n.pose[1] == pytest.approx(-.18)


def test_final_approach_uses_frozen_box_and_never_releases_on_expired_target():
    n = bare_nav()
    n.state = State.APPROACH_BIN_FOR_PLACE
    n.goal = n.pose.copy()
    n.placement_bin = make_heading_pose([.8, -.18, .85], 0.)
    n.observations['bin'] = make_heading_pose([8.431, .277, .666], 0.)
    n.tick()
    assert n.state == State.WAIT_FOR_PLACE
    assert n.place_target_pub.messages[-1] is n.placement_bin
    n.state, n.entered, n.holding_stamp, n.odom_stamp = State.APPROACH_BIN_FOR_PLACE, 41., 41., 41.
    n.now = lambda: 41.
    n.scans = {name: (41., points) for name, (_, points) in n.scans.items()}
    count = len(n.finish_place_pub.messages)
    n.tick()
    assert n.state == State.FAILED and len(n.finish_place_pub.messages) == count


def test_changed_fresh_book_realigns_without_consuming_backout_budget():
    n = bare_nav()
    n.state = State.ALIGN_BOOK
    n.goal = n.pose.copy()
    n.look = lambda *args: None
    n.drive_to = lambda *args, **kwargs: True
    n.selected_book = make_heading_pose([1., -.1, 1.2], 0.)
    n.observations['book'] = make_heading_pose([1.15, -.1, 1.2], 0.)
    n.tick()
    assert n.state == State.REACQUIRE_BOOK and n.book_recoveries == 0
    assert not n.grasp_target_pub.messages


def test_clipped_white_bay_uses_projected_column_but_keeps_wrong_column_rejection():
    from test_regressions import shelf_perception
    n = shelf_perception(1.93)
    for _ in range(3):
        n.process_shelf()
    # No white ring: the old detector returns no candidate. The identified
    # column ROI is allowed to contribute, but its 3D column gate still applies.
    crop = np.zeros((120, 60, 3), np.uint8)
    crop[20:100, 20:40] = (0, 0, 255)
    n.text_search_crop = lambda: (crop, (0, 0))
    n.colour_surface = lambda *args, **kwargs: (np.array([3., 1., 1.265]), .05, .20)
    with patch.dict(PerceptionNode.process_book.__globals__, colour_boxes=lambda *args: []):
        for _ in range(3):
            n.process_book()
    assert n.book_pub.messages
    n.book_pub.messages.clear()
    n.colour_surface = lambda *args, **kwargs: (np.array([3., 2., 1.265]), .05, .20)
    with patch.dict(PerceptionNode.process_book.__globals__, colour_boxes=lambda *args: []):
        for _ in range(3):
            n.process_book()
    assert not n.book_pub.messages


def test_calibrated_bin_hover_at_right_arm_lateral_offset_has_valid_ik():
    model = ArmKinematics((ROOT/'test/data/erc_full_robot.urdf').read_text())
    n = ManipulationNode.__new__(ManipulationNode)
    n.model, n.book_row = model, 2
    n.base_to_grasp = model.fixed_transform('gripper_right_base_link', 'gripper_right_grasping_link')
    current = np.clip(np.r_[.1, HIGH_CARRY_RIGHT_ARM], model.lower, model.upper)
    params = dict(bin_inset=.2, bin_release_x=.78, bin_release_z=1.08, place_torso_hover=.35)
    phases = n.plan('prepare_place', np.array([1.1, -.18, .85]), np.array([1., 0., 0.]), current, params)
    for _, kind, q, _ in phases:
        assert kind == 'motion'
        assert np.all(q >= model.lower) and np.all(q <= model.upper)
    actual = model.fk(phases[-1][2]) @ n.base_to_grasp
    assert actual[:3, 3] == pytest.approx([.78, -.18, 1.08], abs=.008)


def placement_request(point, sequence='finish_place', stamp=10, hover=True):
    n = ManipulationNode.__new__(ManipulationNode)
    n.ros = bare_nav().ros
    n.active = n.plan_future = None
    n.holding_book = 'confirmed_book'
    n.now = lambda: 10.
    n.token, n.hand_corr = 0, None
    n.readiness_issues = lambda: []
    n.targets = {'bin': target(*point, stamp=stamp)}
    n.targets['bin'].header.frame_id = n.ros.frames['odom']
    n.selected_place_target = None
    n.model = NS(names=['torso_lift_joint']+[f'arm_right_{i}_joint' for i in range(1, 8)])
    n.joints = {name: 0. for name in n.model.names}
    n.placement_hover_target = np.zeros(8) if hover else None
    n.placement_prepared_at = 9. if hover else None
    params = dict(placement_target_max_age=30., place_torso_drop=0., tf_max_age=1.,
                  place_lateral_offset=-.18, bin_release_x=.78)
    n.p = params.__getitem__
    tf = NS(header=NS(stamp=NS(sec=10, nanosec=0)), transform=NS(translation=NS(x=0., y=0., z=0.), rotation=NS(x=0., y=0., z=0., w=1.)))
    n.tf = NS(lookup_transform=lambda *args: tf)
    reports, phases = [], []
    n.report = lambda *args: reports.append(args)
    n.next_phase = lambda: phases.extend(n.phases)
    n.request(sequence)
    return n, reports, phases


@pytest.mark.parametrize('point', [[8.431, .277, .666], [1.8, -.18, .85], [.8, .277, .85]])
def test_implausible_or_misaligned_bin_cannot_trigger_gripper_open(point):
    n, reports, phases = placement_request(point)
    assert not phases and reports[-1][1] is False
    assert n.holding_book == 'confirmed_book'


def test_release_requires_fresh_box_and_completed_measured_hover():
    _, reports, phases = placement_request([.8, -.18, .85], hover=False)
    assert reports[-1][1] is False and not phases
    _, reports, phases = placement_request([.8, -.18, .85], stamp=-30)
    assert reports[-1][1] is False and not phases
    n, reports, phases = placement_request([.8, -.18, .85])
    assert not reports and phases[0][0] == 'release_book'
    assert n.active == 'finish_place'


def test_one_book_frame_dropout_preserves_recent_votes_without_publishing_stale_pose():
    from test_regressions import shelf_perception
    n = shelf_perception(1.93)
    for _ in range(3):
        n.process_shelf()
    n.image[40:100, 50:65] = (0, 0, 255)
    n.colour_surface = lambda *args, **kw: (np.array([3., 1., 1.265]), .05, .20)
    for i in range(2):
        n.observation_stamp = NS(sec=10, nanosec=i*125000000)
        n.process_book()
    assert n.book_hits == 2 and not n.book_pub.messages
    with patch.dict(PerceptionNode.process_book.__globals__, colour_boxes=lambda *args: []):
        n.process_book()
    assert n.book_hits == 2 and not n.book_pub.messages
    n.observation_stamp = NS(sec=10, nanosec=375000000)
    n.process_book()
    assert len(n.book_pub.messages) == 1
    n.book_pub.messages.clear()
    # Even a matching point needs new evidence after a long interruption.
    n.observation_stamp = NS(sec=13, nanosec=0)
    n.process_book()
    assert n.book_hits == 1 and not n.book_pub.messages


def test_launch_keeps_bin_lateral_alignment_shared_between_base_and_arm():
    from test_ros_topics import launch_nodes
    nodes = launch_nodes(robot_profile='real', bin_lateral_offset='-0.16')
    nav = next(n for n in nodes if n.executable == 'navigation_node')
    manip = next(n for n in nodes if n.executable == 'manipulation_node')
    shared = nav.parameters[0]['bin_lateral_offset']
    assert manip.parameters[0]['place_lateral_offset'] is shared
    assert shared.value_type(shared.value.perform({'bin_lateral_offset': '-0.16'})) == -.16
