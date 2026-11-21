"""Photo OCR, synthetic RGB-D boundaries, divider path checks, and gentle IK."""
from copy import deepcopy
import shutil
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from test_regressions import ROOT, ArmKinematics, ManipulationNode, NavigationNode, HIGH_CARRY_RIGHT_ARM, bare_nav, make_heading_pose
from solution_4to1mux.book_geometry import measure_book, divider_bounds, grasp_fits, check_divider_points
from solution_4to1mux.text_detection import read_text_label_boxes, title_from_reading


@pytest.mark.skipif(shutil.which('tesseract') is None, reason='Tesseract required')
@pytest.mark.parametrize('title', ['ROBOT', 'LIBRARY', 'AGENT', 'FUTURE'])
def test_actual_supplied_photo_recognizes_each_printed_spine(title):
    image = cv2.imread(str(ROOT/'test/data/white_books_reference.png'))
    result = read_text_label_boxes(image, title)
    assert result.matches, result.readings
    assert any(h > 1.3*w for (x, y, w, h), _, _ in result.matches)
    assert all(word.upper() == title for _, _, word in result.matches)


@pytest.mark.parametrize('raw,score,expected', [('ROBOT', 90, 'ROBOT'), ('RROBOT', 40, None), ('ROBOOT', 90, 'ROBOT'), ('HELLO', 99, None)])
def test_closed_vocabulary_correction_is_bounded_and_score_gated(raw, score, expected):
    assert title_from_reading(raw, score) == expected


def white_scene(board=False):
    image = np.full((200, 300, 3), 235, np.uint8)
    depth = np.full((200, 300), 1.30, np.float32)
    image[52:148, 140:160] = 255
    image[70:130, 147:153] = 20
    depth[52:148, 140:160] = 1.
    if board:
        depth[148:160] = 1.
    info = NS(k=[400., 0., 150., 0., 400., 100., 0., 0., 1.], width=300, height=200)
    transform = np.array([[0.,0.,1.,0.],[-1.,0.,0.,0.],[0.,-1.,0.,1.2],[0.,0.,0.,1.]])
    return image, depth, transform, info


@pytest.mark.parametrize('board', [False, True])
def test_white_book_separates_from_white_backboard_and_touching_bottom_board(board):
    image, depth, transform, info = white_scene(board)
    surface = measure_book(image, depth, transform, info, (145, 70, 10, 60), [1.,0.,0.], white=True)
    assert surface is not None
    assert surface['point'][0] == pytest.approx(1.)
    assert surface['width'] == pytest.approx(.0475, abs=.006)
    assert .22 < surface['height'] < .29
    assert surface['point'][2] == pytest.approx(1.2, abs=.025)


@pytest.mark.parametrize('bad', ['no_depth', 'same_plane', 'oversized', 'sparse'])
def test_depth_does_not_fabricate_a_white_book_from_the_shelf(bad):
    image, depth, transform, info = white_scene()
    if bad == 'no_depth': depth[:] = np.nan
    elif bad == 'same_plane': depth[:] = 1.
    elif bad == 'oversized': depth[52:148, 80:220] = 1.
    else:
        depth[:] = np.nan
        depth[80:82, 148:152] = 1.
    assert measure_book(image, depth, transform, info, (145,70,10,60), [1.,0.,0.], white=True) is None


def geometry(y=.30, stamp=10.):
    return dict(point=[.68,y,1.265], shelf_point=[.68,0.,1.93], normal=[1.,0.,0.],
                left=-.43,right=.43,width=.04,height=.24,stamp=stamp,depth_supported=True,frame='odom')


@pytest.mark.parametrize('y,expected', [(-.34, True),(.34,True),(-.40,False),(.40,False)])
def test_open_finger_envelope_refuses_flush_divider_grasps(y, expected):
    assert grasp_fits(geometry(y)) is expected


def test_unobserved_dividers_are_conservatively_inset():
    image, depth, transform, info = white_scene()
    left, right, source = divider_bounds(depth, transform, info, image.shape[:2], [.68,0.,1.93], [1.,0.,0.],1.265)
    assert left == pytest.approx(-.40) and right == pytest.approx(.40)
    assert source == 'calibrated_with_margin'


@pytest.mark.parametrize('y', [-.32, .32])
def test_edge_base_alignment_keeps_front_clear_and_book_on_right_arm_side(y):
    n = bare_nav()
    n.geometry_for_book = NavigationNode.geometry_for_book.__get__(n)
    n.book_geometry = geometry(y)
    n.selected_book = make_heading_pose([.68,y,1.265],0.)
    n.dock_reference = make_heading_pose([.68,0.,1.93],0.)
    goal = n.target_goal(n.selected_book,.68,-.18)
    actual = n.constrain_goal_to_column(goal)
    assert actual == pytest.approx(goal)
    assert y-actual[1] == pytest.approx(-.18)
    assert actual[0] == pytest.approx(0.)


def test_stale_geometry_cannot_enable_extra_edge_lateral_motion():
    n = bare_nav()
    n.geometry_for_book = NavigationNode.geometry_for_book.__get__(n)
    n.book_geometry = geometry(.32, stamp=1.)
    n.selected_book = make_heading_pose([.68,.32,1.265],0.)
    n.dock_reference = make_heading_pose([.68,0.,1.93],0.)
    actual = n.constrain_goal_to_column(n.target_goal(n.selected_book,.68,-.18))
    assert abs(actual[1]) <= .08+1e-8


@pytest.mark.parametrize('point,radius,safe', [([.80,.42,1.2],.045,False),([.80,0.,1.2],.065,True),([.20,.42,1.2],.065,True)])
def test_divider_capsules_include_radius_and_allow_retracted_arm(point,radius,safe):
    if safe:
        check_divider_points([point],[radius],geometry())
    else:
        with pytest.raises(ValueError,match='Divider'):
            check_divider_points([point],[radius],geometry())


@pytest.fixture(scope='module')
def calibrated_arm():
    xml = (ROOT/'test/data/erc_full_robot.urdf').read_text()
    model = ArmKinematics(xml)
    n = ManipulationNode.__new__(ManipulationNode)
    n.model, n.book_row = model, 2
    n.base_to_grasp = model.fixed_transform('gripper_right_base_link','gripper_right_grasping_link')
    n.link_models = [ArmKinematics(xml,tip=f'arm_right_{i}_link') for i in (3,4,5,6)]
    n.p = dict(gripper_half_width=.065,divider_margin=.015).__getitem__
    return n


@pytest.mark.parametrize('y', [-.32,.32])
@pytest.mark.parametrize('row,z,standoff', [(2,1.265,.68),(3,.935,.92),(4,.605,.92)])
def test_full_calibrated_edge_grasp_path_keeps_side_board_clearance(calibrated_arm,y,row,z,standoff):
    n = calibrated_arm
    n.book_row = row
    # The base aligns 18 cm robot-left of the selected book while staying
    # outside the shelf face. Geometry remains frozen in world and transformed.
    base_y = y+.18
    current = np.clip(np.r_[.1,[0.,1.07,0.,-1.,.4,0.,0.]],n.model.lower,n.model.upper)
    params = dict(grasp_insert_depth=.085,pregrasp_distance=.10,lift_distance=.025,extract_distance=.10,grasp_base_advance=.28)
    g = geometry(y)
    g['point'] = [standoff+.25,-.18,z]
    g['shelf_point'] = [standoff+.25,-base_y,1.93]
    params['shelf_geometry'] = g
    prepare = n.plan('prepare_grab',np.array([standoff+.25,-.18,z]),np.array([1.,0.,0.]),current,params)
    prepared = next(p[2] for p in reversed(prepare) if p[1]=='motion')
    params = deepcopy(params)
    params['shelf_geometry']['point'][0] -= .28
    params['shelf_geometry']['shelf_point'][0] -= .28
    finish = n.plan('finish_grab',np.array([standoff-.03,-.18,z]),np.array([1.,0.,0.]),prepared,params)
    assert any(name=='insert' for name,_,_,_ in finish)


def gentle_params():
    return dict(bin_inset=.20,bin_release_x=.78,bin_release_z=1.08,place_torso_hover=.35,
                gentle_drop_distance=.08,gentle_lower_speed=.025,gentle_open_duration=3.,bin_rim_height=.95,table_surface_height=.72)


def test_gentle_release_uses_slow_arm_descent_fixed_torso_and_delayed_open(calibrated_arm):
    n = calibrated_arm
    current = np.clip(np.r_[.10,HIGH_CARRY_RIGHT_ARM],n.model.lower,n.model.upper)
    params = gentle_params()
    hover = n.plan('prepare_place',np.array([1.1,-.18,.85]),np.array([1.,0.,0.]),current,params)[-1][2]
    phases = n.plan_gentle_place(hover,np.array([.86,-.18,.85]),params)
    previous = (n.model.fk(hover)@n.base_to_grasp)[:3,3]
    for name,kind,q,duration in phases:
        if kind != 'motion':continue
        actual = (n.model.fk(q)@n.base_to_grasp)[:3,3]
        assert q[0] == pytest.approx(hover[0])
        assert actual[:2] == pytest.approx(previous[:2],abs=.008)
        assert 0 < previous[2]-actual[2] <= duration*.025+.001
        assert actual[2] >= params['bin_rim_height']+.035-1e-3
        previous = actual
    assert previous[2] == pytest.approx(1.,abs=.008)
    assert phases[-2][1:] == ('gentle_open',None,3.)


@pytest.mark.parametrize('point', [[8.,-.18,.85],[.86,.4,.85],[.86,-.18,.3]])
def test_bad_bin_pose_cannot_trigger_gentle_drop(calibrated_arm,point):
    n=calibrated_arm
    q=np.clip(np.r_[.2,HIGH_CARRY_RIGHT_ARM],n.model.lower,n.model.upper)
    with pytest.raises(ValueError):n.plan_gentle_place(q,np.array(point),gentle_params())


def test_white_launch_accepts_titles_and_gives_ocr_settling_time():
    from test_ros_topics import launch_nodes
    nodes = launch_nodes(robot_profile='real',book_colour='white',book_text='LIBRARY')
    nav=next(n for n in nodes if n.executable=='navigation_node')
    assert nav.parameters[0]['head_scan_dwell']==6.
    assert nav.parameters[0]['book_search_timeout']==36.


def white_perception():
    from test_regressions import PerceptionNode
    from ros_stubs import Logger, Pub, Time
    n=PerceptionNode.__new__(PerceptionNode)
    n.image,n.depth_image,n.camera_to_odom,n.info=white_scene()
    n.camera_to_odom[2,3]=1.265
    n.ros=bare_nav().ros
    n.target_text,n.target_colour,n.text_colour_hint,n.text_book_colour='LIBRARY','white','',None
    n.column_pose=make_heading_pose([1.,0.,1.93],0.)
    n.held_until=0.
    n.ocr_sequence,n.ocr_handled=1,0
    n.ocr_readings=['LIBRARY']
    n.ocr_boxes=[((145,70,10,60),95.,'LIBRARY')]
    n.ocr_result_context=dict(rgb_image=n.image.copy(),rgb_crop=n.image.copy(),depth_image=n.depth_image.copy(),
        camera_to_odom=n.camera_to_odom.copy(),info=n.info,image_shape=n.image.shape[:2],origin=(0,0),
        stamp=NS(sec=10,nanosec=125000000),purpose='shelf')
    n.update_text_search=lambda *args:None
    n.get_clock=lambda:NS(now=lambda:Time(10800000000))
    values=dict(column_half_width=.43,top_book_marker_offset=.335,row_spacing=.33)
    n.get_parameter=lambda key:NS(value=values[key])
    n.latest_book,n.filtered_book,n.book_hits=None,None,0
    n.book_pub,n.geometry_pub,n.row_pub,n.book_diag_pub,n.held_pub=Pub(),Pub(),Pub(),Pub(),Pub()
    n.held_votes,n.held_ocr_handled=[],-1
    n.reported=set();n.save=lambda *args:None
    n.get_logger=lambda:Logger()
    n.make_pose=lambda p,h:make_heading_pose(p,h)
    n.observation_stamp=NS(sec=10,nanosec=500000000)
    return n


def test_white_title_pipeline_publishes_depth_surface_and_source_timestamp_after_three_frames():
    import json
    n=white_perception()
    n.depth_image[:]=np.nan  # Only the captured OCR depth is valid.
    for i in range(1,4):
        n.ocr_sequence=i
        n.ocr_result_context['stamp']=NS(sec=10,nanosec=i*125000000)
        n.process_book()
    assert len(n.book_pub.messages)==1
    pose=n.book_pub.messages[-1]
    assert pose.header.stamp.nanosec==375000000
    assert pose.pose.position.x==pytest.approx(1.)
    assert pose.pose.position.z==pytest.approx(1.265,abs=.01)
    geometry=json.loads(n.geometry_pub.messages[-1].data)
    assert geometry['depth_supported'] is True and geometry['height']>.20
    assert geometry['stamp']==pytest.approx(10.375)
    assert n.row_pub.messages[-1].data==2


def test_white_title_without_depth_or_with_stale_capture_is_not_published():
    n=white_perception()
    n.ocr_result_context['depth_image'][:]=np.nan
    for i in range(1,4):
        n.ocr_sequence=i
        n.ocr_result_context['stamp']=NS(sec=10,nanosec=i*125000000)
        n.process_book()
    assert not n.book_pub.messages
    n.ocr_result_context['stamp']=NS(sec=1,nanosec=0)
    n.ocr_sequence=4;n.process_book()
    assert not n.book_pub.messages


def test_neutral_spine_at_hand_needs_two_new_reads_and_measured_extraction():
    n=white_perception()
    n.held_until=22.
    n.text_grasp_origin=np.array([1.10,0.,1.265])
    n.text_grasp_normal=np.array([1.,0.,0.])
    n.ocr_result_context.update(purpose='hand',grasp=np.array([1.,0.,1.265]))
    n.verify_book_at_hand(None)
    assert not n.held_pub.messages
    n.verify_book_at_hand(None)
    assert len(n.held_votes)==1
    n.ocr_sequence=2;n.ocr_result_context['stamp']=NS(sec=10,nanosec=250000000)
    n.verify_book_at_hand(None)
    assert n.held_pub.messages[-1].data is True
    n=white_perception();n.held_until=22.
    n.text_grasp_origin=np.array([1.,0.,1.265]);n.text_grasp_normal=np.array([1.,0.,0.])
    n.ocr_result_context.update(purpose='hand',grasp=np.array([1.,0.,1.265]))
    n.verify_book_at_hand(None)
    assert not n.held_votes


def test_gentle_open_waits_full_ramp_even_when_live_finger_feedback_is_frozen():
    from test_runtime_faults import bare_fault_manipulation
    n,_,_,_=bare_fault_manipulation()
    n.phase='gentle_open';n.phase_duration=3.;n.phase_started=10.;n.deadline=20.
    n.finger_feedback=False
    advanced=[];n.next_phase=lambda:advanced.append(True)
    n.now=lambda:12.;n.joint_stamp=12.;n.tick()
    assert not advanced
    n.now=lambda:13.6;n.joint_stamp=13.6;n.tick()
    assert advanced==[True]


def test_broad_side_title_alone_does_not_choose_a_white_book_grasp():
    n=white_perception()
    n.ocr_boxes=[((140,80,30,10),95.,'LIBRARY')]
    n.process_book()
    assert not n.book_pub.messages and 'vertical spine' in n.book_diag_pub.messages[-1].data


def test_checked_edge_alignment_is_not_undone_when_stow_briefly_occludes_the_book():
    from test_regressions import State
    n=bare_nav();n.geometry_for_book=NavigationNode.geometry_for_book.__get__(n)
    n.now=lambda:17.
    n.book_geometry=None;n.dock_geometry=geometry(.32,stamp=10.)
    n.selected_book=make_heading_pose([.68,.32,1.265],0.)
    n.dock_reference=make_heading_pose([.68,0.,1.93],0.)
    n.state=State.ALIGN_BOOK
    desired=n.target_goal(n.selected_book,.68,-.18)
    assert n.constrain_goal_to_column(desired)==pytest.approx(desired)
