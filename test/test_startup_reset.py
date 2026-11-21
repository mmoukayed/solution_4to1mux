"""Beginning-of-run recovery from the previous run's extended arm/high torso."""
import pytest

from test_regressions import ROOT, State, bare_nav
from ros_stubs import Message, Pub
from solution_4to1mux.alignment import read_joint_limits


TUCK_POSES = bare_nav().tuck_arms_at_start.__globals__['TUCK_POSES']


def reset_node():
    n = bare_nav()
    previous = n.p
    config = dict(startup_feedback_timeout=45., startup_motion_margin=8.,
                  startup_torso_height=.10, startup_torso_speed=.07)
    n.p = lambda key: config[key] if key in config else previous(key)
    n.state = State.RESET_ROBOT
    n.startup_complete = n.arms_tucked = False
    n.startup_phase = 'WAIT_FEEDBACK'
    n.startup_phase_started = 0.
    n.startup_deadline = n.startup_settled_since = None
    n.startup_joint_sent = False
    n.fake_wall = 0.
    n.startup_now = lambda: n.fake_wall
    n.now = lambda: 10.+n.fake_wall
    n.torso_pub, n.head_pub = Pub(), Pub()
    n.arm_limits = read_joint_limits((ROOT/'test/data/erc_full_robot.urdf').read_text())
    for side in ('left', 'right'):
        n.arm_joints.update({f'arm_{side}_{i}_joint': value for i, value in enumerate(TUCK_POSES[side], 1)})
    # A new process receives the previous run's physical finishing pose:
    # right arm extended, raised torso, and head aimed down at the bin.
    n.arm_joints.update({f'arm_right_{i}_joint': v for i, v in enumerate([0., 1.0, 0., -1., .4, 0., 0.], 1)})
    n.arm_joints.update(torso_lift_joint=.35, head_1_joint=.6, head_2_joint=-.5)
    n.arm_tuck_secs = 2.
    n.joint_observed_at = {name: 10. for name in n.arm_joints}
    return n


def refresh(n, advance=0.):
    n.fake_wall += advance
    now = n.now()
    n.arm_joint_stamp = n.odom_stamp = now
    n.joint_observed_at = {name: now for name in n.arm_joints}
    n.scans = {side: (now, points) for side, (_, points) in n.scans.items()}


def measured_goal(n, publisher):
    message = publisher.messages[-1]
    n.arm_joints.update(zip(message.joint_names, message.points[0].positions))
    refresh(n)


def settle(n):
    n.tick()
    refresh(n, .30)
    n.tick()


def assert_stopped(n):
    assert n.cmd_pub.messages
    assert all(m.linear.x == m.linear.y == m.angular.z == 0 for m in n.cmd_pub.messages)


def test_relaunch_from_finished_pose_resets_arms_then_torso_then_head_before_navigation():
    n = reset_node()
    original_base = n.pose.copy()
    n.tick()
    assert n.startup_phase == 'TUCK_ARMS'
    n.tick()
    assert n.left_arm_pub.messages and n.right_arm_pub.messages
    assert not n.torso_pub.messages and not n.head_pub.messages
    # Time passing alone cannot approve the arm tuck.
    refresh(n, 1.)
    n.tick()
    assert n.startup_phase == 'TUCK_ARMS'
    measured_goal(n, n.right_arm_pub)
    measured_goal(n, n.left_arm_pub)
    settle(n)
    assert n.startup_phase == 'RESET_TORSO' and n.arms_tucked
    n.tick()
    torso = n.torso_pub.messages[-1]
    assert torso.points[0].positions == pytest.approx([.10])
    assert not n.head_pub.messages
    measured_goal(n, n.torso_pub)
    settle(n)
    assert n.startup_phase == 'RESET_HEAD'
    n.tick()
    assert n.head_pub.messages[-1].points[0].positions == pytest.approx([0., min(.28, n.arm_limits['head_2_joint'][1])])
    measured_goal(n, n.head_pub)
    settle(n)
    assert n.startup_complete and n.state == State.FIND_SHELF
    assert n.pose == pytest.approx(original_base)
    assert n.start == pytest.approx(original_base)
    assert n.search_origin == pytest.approx(original_base[2])
    assert_stopped(n)
    # The reset does not run again at mission completion or change the end pose.
    counts = tuple(len(p.messages) for p in (n.right_arm_pub, n.torso_pub, n.head_pub))
    n.state = State.DONE
    n.tick()
    assert counts == tuple(len(p.messages) for p in (n.right_arm_pub, n.torso_pub, n.head_pub))


def test_ready_pose_does_not_need_redundant_joint_commands():
    n = reset_node()
    for side in ('left', 'right'):
        n.arm_joints.update({f'arm_{side}_{i}_joint': value for i, value in enumerate(TUCK_POSES[side], 1)})
    n.arm_joints.update(torso_lift_joint=.10, head_1_joint=0., head_2_joint=.28)
    for _ in range(9):
        refresh(n, .3)
        n.tick()
        if n.startup_complete:
            break
    assert n.startup_complete
    assert not n.left_arm_pub.messages and not n.right_arm_pub.messages
    assert not n.torso_pub.messages and not n.head_pub.messages
    assert_stopped(n)


def test_late_controller_subscription_does_not_consume_one_shot_tuck():
    n = reset_node()
    connected = [False]
    n.right_arm_pub.get_subscription_count = lambda: int(connected[0])
    n.tick()
    assert n.startup_phase == 'WAIT_FEEDBACK' and n.arm_tuck_last is None
    assert not n.right_arm_pub.messages and not n.torso_pub.messages
    refresh(n, 5.)
    connected[0] = True
    n.tick()
    n.tick()
    assert n.startup_phase == 'TUCK_ARMS' and len(n.right_arm_pub.messages) == 1
    assert n.arm_tuck_last is not None
    assert_stopped(n)


def test_tuck_itself_waits_for_both_command_subscribers():
    n = reset_node()
    n.left_arm_pub.get_subscription_count = lambda: 0
    assert n.tuck_arms_at_start() is False
    assert n.arm_tuck_last is None and not n.right_arm_pub.messages


@pytest.mark.parametrize('missing', ['torso_lift_joint', 'arm_right_4_joint', 'head_2_joint'])
def test_fresh_unrelated_joint_messages_cannot_hide_missing_reset_joint(missing):
    n = reset_node()
    n.joint_observed_at[missing] = 1.
    n.tick()
    assert n.startup_phase == 'WAIT_FEEDBACK'
    assert not n.right_arm_pub.messages and not n.torso_pub.messages
    assert_stopped(n)


def test_wait_for_controllers_has_bounded_wall_clock_timeout():
    n = reset_node()
    n.head_pub.get_subscription_count = lambda: 0
    refresh(n, 46.)
    n.tick()
    assert n.state == State.FAILED and not n.startup_complete
    assert 'controllers' in n.failure_pub.messages[-1].data
    assert_stopped(n)


def test_unchanged_joint_feedback_does_not_approve_reset_after_timeout():
    n = reset_node()
    n.tick()
    n.tick()
    refresh(n, 60.)
    n.tick()
    assert n.state == State.FAILED and not n.startup_complete
    assert not n.torso_pub.messages and not n.head_pub.messages
    assert_stopped(n)


def test_no_repeated_topic_goal_while_reset_motion_is_in_progress():
    n = reset_node()
    n.tick()
    n.tick()
    for _ in range(5):
        refresh(n, .1)
        n.tick()
    assert len(n.right_arm_pub.messages) == 1
    assert not n.torso_pub.messages


def test_torso_command_is_bounded_and_respects_urdf_velocity():
    n = reset_node()
    n.startup_stage('RESET_TORSO')
    n.tick()
    goal = n.torso_pub.messages[-1].points[0]
    duration = goal.time_from_start.sec+goal.time_from_start.nanosec/1e9
    lo, hi, velocity = n.arm_limits['torso_lift_joint']
    assert lo <= goal.positions[0] <= hi
    assert .25/duration <= min(.07, .8*velocity)+1e-8
    assert_stopped(n)


def test_controller_rejection_stops_startup_without_progressing():
    n = reset_node()
    n.startup_stage('RESET_TORSO')
    n.on_controller_rejection(Message(data='torso_controller REJECTED goal'))
    n.tick()
    assert n.state == State.FAILED
    assert not n.torso_pub.messages and not n.head_pub.messages
    assert_stopped(n)


def test_arm_goal_outside_urdf_bounds_is_never_sent_at_startup():
    n = reset_node()
    n.tick()
    n.tick()
    for publisher in (n.left_arm_pub, n.right_arm_pub):
        msg = publisher.messages[-1]
        for name, value in zip(msg.joint_names, msg.points[0].positions):
            lo, hi, _ = n.arm_limits[name]
            assert lo <= value <= hi


def test_startup_parameters_are_exposed_by_launch():
    from test_ros_topics import launch_nodes
    nodes = launch_nodes(robot_profile='real', startup_torso_height='0.12')
    nav = next(node for node in nodes if node.executable == 'navigation_node')
    value = nav.parameters[0]['startup_torso_height']
    assert value.value_type(value.value.perform({'startup_torso_height': '0.12'})) == .12


def test_arm_torso_and_head_feedback_loss_keeps_reset_paused():
    n = reset_node()
    n.startup_stage('RESET_HEAD')
    n.joint_observed_at['torso_lift_joint'] = 0.
    n.tick()
    assert not n.head_pub.messages and not n.startup_complete
    refresh(n, 1.)
    n.tick()
    assert len(n.head_pub.messages) == 1 and not n.startup_complete
    assert_stopped(n)


def test_partial_joint_message_does_not_refresh_unreported_positions():
    from types import SimpleNamespace as NS
    n = reset_node()
    n.joint_observed_at['arm_right_4_joint'] = 1.
    n.on_arm_joints(NS(name=['head_1_joint', 'arm_right_4_joint'], position=[0.],
                       header=NS(stamp=NS(sec=10, nanosec=0))))
    assert n.joint_observed_at['head_1_joint'] == 10.
    assert n.joint_observed_at['arm_right_4_joint'] == 1.
    n.tick()
    assert n.startup_phase == 'WAIT_FEEDBACK' and not n.right_arm_pub.messages
