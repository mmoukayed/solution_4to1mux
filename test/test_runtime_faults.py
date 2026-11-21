"""Fault replay from the final insertion/deactivation in the supplied output."""
import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from ros_stubs import Logger, Message, Pub, install
from test_regressions import ROOT, ManipulationNode, State, bare_nav
from solution_4to1mux.build_info import source_fingerprint
from solution_4to1mux.runtime_health import current_transform, is_controller_deactivation


FAILURE = 'Current goal cancelled during deactivate transition.'


def bare_fault_manipulation():
    n = ManipulationNode.__new__(ManipulationNode)
    n.ros = bare_nav().ros
    n.now = lambda: 10.
    n.token, n.active = 7, 'finish_grab'
    n.phase, n.phase_name = 'motion', 'insert'
    n.joint_stamp, n.arm_in_shelf = 9.95, True
    n.controller_fault = ''
    n.holding_book = 'already_held_book'
    n.phases = [('lift', 'motion', np.zeros(8), 1.)]
    n.pending_request, n.deferred_plan = 'prepare_grab', ('old plan',)
    n.grasp_prepared, n.selected_grasp_target = True, object()
    n.plan_future = None
    cancellations = []
    n.handles = [NS(cancel_goal_async=lambda: cancellations.append(True))]
    n.robot_fault_pub, n.gripper_pub = Pub(), Pub()
    n.ready_pub, n.holding_pub = Pub(), Pub()
    n.ready = lambda: False
    n.held_book = lambda: n.holding_book
    n.get_logger = lambda: Logger()
    retractions, reports = [], []
    n.retract_if_in_shelf = lambda: retractions.append(True)
    n.report = lambda *args: reports.append(args)
    return n, cancellations, retractions, reports


def test_exact_deactivation_error_in_output_is_recognized():
    assert is_controller_deactivation(FAILURE)
    assert is_controller_deactivation('Controller is inactive')
    assert not is_controller_deactivation('Goal tolerance violated')


def test_insertion_deactivation_latches_fault_without_open_or_retract():
    n, cancellations, retractions, reports = bare_fault_manipulation()
    response = NS(status=6, result=NS(error_code=-1, error_string=FAILURE))
    n.motion_result(NS(result=lambda: response), 7)
    assert 'deactivated' in n.controller_fault
    assert n.active is None and n.phase is None
    assert n.pending_request is None and not n.phases
    assert n.arm_in_shelf is True  # preserve the operator's last known location
    assert n.holding_book == 'already_held_book'  # never automatically drop it
    assert not n.gripper_pub.messages and not retractions
    assert cancellations == [True]
    assert reports[0][0] == 'finish_grab' and reports[0][1] is False
    snapshot = json.loads(n.robot_fault_pub.messages[-1].data)
    assert snapshot['phase'] == 'motion' and snapshot['arm_inside_shelf'] is True
    # A fresh sensor message must not automatically resume the interrupted task.
    n.joint_stamp = 10.
    n.request('finish_grab')
    assert n.active is None and len(reports) == 2
    n.tick()
    assert n.holding_pub.messages[-1].data is False
    assert not retractions and not n.gripper_pub.messages


def test_late_success_from_other_controller_cannot_clear_fault():
    n, _, _, reports = bare_fault_manipulation()
    n.latch_robot_fault('arm deactivated')
    old_count = len(reports)
    n.motion_result(NS(result=lambda: NS(status=4, result=NS(error_code=0, error_string=''))), 7)
    assert n.active is None and n.controller_fault == 'arm deactivated'
    assert len(reports) == old_count


def test_joint_feedback_freeze_during_motion_does_not_send_recovery():
    n, cancellations, retractions, reports = bare_fault_manipulation()
    n.joint_stamp = 6.
    n.tick()
    assert 'Joint feedback disappeared' in n.controller_fault
    assert n.active is None and not retractions and not n.gripper_pub.messages
    assert cancellations and reports[-1][1] is False


def test_motion_timeout_is_not_treated_as_a_missed_book_retry():
    n, _, retractions, reports = bare_fault_manipulation()
    n.deadline = 9.
    n.tick()
    assert 'deadline expired' in n.controller_fault
    assert not retractions and reports[-1][1] is False


def test_failed_cancellation_does_not_hide_original_fault():
    n, _, retractions, reports = bare_fault_manipulation()
    def unavailable():
        raise RuntimeError('context unavailable')
    n.handles = [NS(cancel_goal_async=unavailable)]
    n.latch_robot_fault('arm deactivated')
    assert n.controller_fault == 'arm deactivated'
    assert reports[-1][1] is False and not retractions


def test_fault_from_navigation_inhibits_manipulation_even_with_fresh_joints():
    n, _, retractions, reports = bare_fault_manipulation()
    n.on_robot_fault(Message(data=json.dumps({'reason': 'odometry stopped', 'source': 'navigation'})))
    assert n.controller_fault == 'odometry stopped'
    assert not retractions and reports[-1][1] is False


def test_navigation_fault_path_does_not_send_normal_cancel_that_can_retract():
    n = bare_nav()
    n.state = State.WAIT_FOR_GRAB
    n.result = None
    n.on_robot_fault(Message(data=json.dumps({'reason': FAILURE})))
    assert n.state == State.FAILED and n.robot_fault == FAILURE
    assert n.last_command == (0., 0., 0.)
    assert not n.cancel_pub.messages


def test_navigation_stale_odometry_stops_then_latches_with_bounded_timeout():
    n = bare_nav()
    n.state = State.APPROACH_COLUMN
    n.robot_fault_pub = Pub()
    n.odom_stamp = 6.
    assert not n.check_feedback()
    assert n.state != State.FAILED and n.last_command == (0., 0., 0.)
    n.now = lambda: 13.1
    n.scans = {name: (13.1, value[1]) for name, value in n.scans.items()}
    assert not n.check_feedback()
    assert n.state == State.FAILED and 'odometry' in n.robot_fault
    assert len(n.robot_fault_pub.messages) == 1
    assert not n.cancel_pub.messages


def test_brief_feedback_gap_recovers_without_latching():
    n = bare_nav()
    n.state = State.APPROACH_COLUMN
    n.robot_fault_pub = Pub()
    n.odom_stamp = 6.
    assert not n.check_feedback()
    n.now = lambda: 10.5
    n.odom_stamp = 10.5
    assert n.check_feedback()
    assert n.sensor_wait_since is None and not n.robot_fault_pub.messages


@pytest.mark.parametrize('stamp,expected', [(10., True), (9.5, True), (8., False),
                                           (0., False), (10.05, True), (11., False)])
def test_transform_age_is_checked_including_future_clock_skew(stamp, expected):
    assert current_transform(10., stamp, 1.) is expected


def load_verify_script():
    spec = importlib.util.spec_from_file_location('verify_install_test', ROOT/'scripts/verify_install.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installation_check_identifies_mixed_core_modules(tmp_path):
    module = load_verify_script()
    source, installed = tmp_path/'source', tmp_path/'installed'
    source.mkdir(); installed.mkdir()
    (source/'navigation_node.py').write_text('new source')
    (installed/'navigation_node.py').write_text('old source')
    assert 'different' in module.compare_sources(source, installed, ['navigation_node.py'])[0]
    (installed/'navigation_node.py').write_text('new source')
    assert module.compare_sources(source, installed, ['navigation_node.py']) == []


def test_fingerprint_includes_all_core_sources():
    assert len(source_fingerprint()) == 16


def test_read_only_controller_diagnostics_distinguish_discovery_from_active_state():
    restore = install()
    try:
        spec = importlib.util.spec_from_file_location('preflight_test', ROOT/'scripts/preflight.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        srv = ModuleType('controller_manager_msgs.srv')
        srv.ListControllers = NS(Request=Message)
        root = ModuleType('controller_manager_msgs')
        root.srv = srv
        response = NS(controller=[NS(name='arm_right_controller', state='inactive'),
                                  NS(name='torso_controller', state='active')])
        called = []
        client = NS(wait_for_service=lambda **kwargs: True,
                    call_async=lambda request: called.append(request) or NS(done=lambda: True, result=lambda: response))
        node = NS(get_service_names_and_types=lambda: [('/robot/controller_manager/list_controllers',
                     ['controller_manager_msgs/srv/ListControllers'])],
                  create_client=lambda service, name: client, destroy_client=lambda c: None)
        results = []
        with patch.dict(sys.modules, {'controller_manager_msgs': root, 'controller_manager_msgs.srv': srv}):
            module.inspect_controller_states(node, lambda *args: results.append(args))
        assert any(level == 'FAIL' and 'arm_right_controller: inactive' in detail for level, _, detail in results)
        assert any(level == 'PASS' and 'torso_controller: active' in detail for level, _, detail in results)
        assert len(called) == 1
    finally:
        restore()
