# Library robot — Build 8.1 (0.8.1)

## Starting another run after a completed delivery

Every fresh launch now begins in `RESET_ROBOT`. The base stays stopped while
startup waits for connected arm/torso/head command subscribers, URDF limits,
fresh per-joint feedback, odometry and both LiDAR scans. It then confirms both
arms in the existing checked tuck pose, restores the torso to 0.10 m (bounded by
the loaded URDF), and centres the head at pan 0 / shelf-search tilt (also bounded
by the URDF). Each stage requires measured convergence and settling; elapsed
time cannot approve a reset. Motion deadlines include the commanded duration.
The torso lowers only after both arms are confirmed tucked.

This happens at the beginning of a new launch, including after a successful
previous run. No end-of-run reset was added. The gripper is not automatically
opened by startup. No controllers or robot safety systems are reactivated.
The base's physical location/odometry are not reset: the fresh run uses its
current measured pose as the return origin. As with the existing arm tuck,
provide clear space for the arm motion; this is not collision-planned recovery
from an arbitrary interrupted grasp or an arm wedged inside the shelf.

Stop the previous launch with Ctrl+C, then launch again. Pull the startup-reset
change, rebuild, and source the new install. The existing launch command works:

```bash
cd /opt/erc_ws
source /opt/ros/humble/setup.bash
source tiago_dds_connect/tiago_env.sh
colcon build --symlink-install --packages-select solution_4to1mux
source install/setup.bash
python3 src/solution_4to1mux/scripts/verify_install.py --source src/solution_4to1mux
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=3 book_colour:=red
```

Expect `BUILD 8.1` on all nodes, followed by `Startup reset: TUCK_ARMS`,
`RESET_TORSO`, `RESET_HEAD`, then `Startup reset complete` and `FIND_SHELF`.
`startup_torso_height:=0.10` can select another measured start height;
`startup_feedback_timeout:=45.0` controls the controller/sensor connection wait.
A rejection, stale/incomplete feedback, or missed motion deadline keeps the
mission stopped with the failed reset stage on `/solution/failure`.

Without a second-run log, the exact cause of the reported stopped run is not
proven. The previous branch only tucked arms and could send its one-shot tuck
before command subscribers connected; it did not reset torso/head as a measured
startup sequence. These code paths are now covered by offline regressions.


This release consolidates the alignment, perception, navigation and live-robot grasp repairs into the supplied package. The uploaded ZIP contained the original older code, while the supplied output contained several different builds. Replace the complete package; do not overlay selected files or leave another copy under the workspace's src directory.

The final insertion in the output failed with “Current goal cancelled during deactivate transition.” Joint and TF updates then froze. This release inhibits autonomous recovery after that failure; it does not establish or repair the underlying robot/controller/safety/communication cause. Inspect the robot's own controller and safety logs. If the arm is inside the shelf or holding a book, perform operator-controlled recovery before restarting. The application never resets safety systems or activates controllers.

## Install and check the loaded code

Use the existing ERC ROS Humble container and your normal DDS connection. Extract this archive so the package.xml is at `/opt/erc_ws/src/solution_4to1mux/package.xml`. Move any previous package copy outside src first.

```bash
cd /opt/erc_ws
source /opt/ros/humble/setup.bash
source tiago_dds_connect/tiago_env.sh
colcon build --symlink-install --packages-select solution_4to1mux
source install/setup.bash
python3 src/solution_4to1mux/scripts/verify_install.py --source src/solution_4to1mux
python3 src/solution_4to1mux/scripts/preflight.py
```

`verify_install.py` compares the loaded core Python files to this release and prints their path and source fingerprint. Do not add src to PYTHONPATH to make this check pass. All three nodes must print BUILD 8.1, version 0.8.1 and the same fingerprint. This detects mixed or stale installs, not physical readiness.

The read-only preflight checks streams, transforms, actions and advertised ListControllers services. An action server alone does not prove the controller is active. An unavailable controller-state service produces a warning, not proof of readiness. Resolve failures and check robot readiness before launching.

```bash
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=3 book_colour:=yellow
```

`book_color:=yellow` is also accepted. Conflicting colour aliases are rejected. Check the selected colour in startup output. Use `robot_profile:=sim` for the simulator; clock defaults follow the profile. This archive contains the solution package and offline URDF test fixture, not a complete simulator workspace.

## Movement, alignment and grasp changes

- Travel speed cap 0.45 m/s; carrying 0.25 m/s; docking 0.16 m/s; final grasp return 0.08 m/s. Acceleration ramps, immediate obstacle stops and scan-age braking remain enforced. These are caps, not guaranteed achieved speeds. Arm timing uses a 0.60 speed fraction and the robot's joint limits.
- Shelf headings come from distinct marker observations, reject outliers and update coherently during approach. Final docking uses a single frozen shelf reference for lateral alignment and grasp orientation. The selected book pose is handed to manipulation explicitly.
- The live marker minimum is 1.60 m, accepting the approximately 1.70 m markers seen in the output. Simulation retains 1.75 m. Override with `marker_min_height:=...` only for a measured setup.
- Startup tuck and dock stow require measured joint convergence. Right-arm joint 2 obeys the live URDF limit instead of the rejected 1.25 rad target. Both arms stay in their calibrated tucked configuration during base movement where required.
- Depth registration uses both camera transforms and timestamp checks. OCR keeps the original observation time; stale results cannot become fresh targets. Depth projection uses a nearest-depth buffer and limits repeated calibration work.
- Colour detection uses the original BGR camera pixels, without automatic white balance or brightness stretching. Book thresholds accept the live feed's olive green and pale blue, narrow spines, and pink-tinted shelf surrounds. The saved camera images also retain the original pixels apart from the detection outline and timestamp; washed-out pixels in those files originate in the camera feed.
- Bin detection requires a mostly saturated red, wide surface, plausible measured dimensions and local range, coherent depth on the red surface, and three distinct consistent camera observations. Missing depth or multiple plausible candidates keeps the search active. The centre of a bounding box is no longer used as the bin's depth sample.
- Live grasp-frame fallback, hand-camera correction and visual grasp confirmation are preserved. The fallback calibration still needs checking on the physical robot. Missing confirmation is a failed grasp, not assumed success.
- Top-row torso clearance and sampled arm-link shelf checks remain mandatory. Unsafe plans are refused. These checks are not a full mesh collision planner.
- Return and placement preserve the full navigation goal across backoffs, use carrying clearances and align laterally to the bin. Without a live bin sensor, completion means the release sequence completed, not independently proven delivery.

## Runtime faults and diagnostics

Controller deactivation, rejected/failed trajectory actions, lost action responses, motion feedback timeouts, stale grasp transforms and lost active joint feedback latch a robot fault. The attempt cancels accepted action goals where possible, stops requesting new manipulation trajectories, and does not automatically open the gripper, retract or retry. Navigation requests zero base velocity. Continued odometry/LiDAR loss stops movement immediately and latches a fault after 3 seconds. A communication failure can prevent commands reaching the robot; application commands cannot guarantee a physical stop in that situation.

Fresh feedback alone does not clear the fault. Inspect the robot, recover under operator control, then restart the solution. Inspect the recorded reason and fault snapshot with:

```bash
ros2 topic echo /solution/robot_fault
```

For a new trial, run `scripts/record.sh trial_name` in another sourced terminal before launching. It records solution diagnostics, ROS logs, TF, feedback, selected camera data and matching controller state/action-status topics when published. `--light` omits depth images; `--raw` adds raw colour and can generate very large bags. `scripts/run_trial.sh` saves launch output. No rosbag was supplied for this repair; the evidence was source code and text output.

## Validation and limits

See VALIDATION.txt and CHANGELOG_RUNTIME.md. Offline tests exercise alignment geometry, joint limits, guarded planning, state transitions, camera logic and the final deactivation/frozen-feedback failure. ROS, DDS, controller lifecycle and physical grasp/delivery have not been run here. Rebuild and verify the installed code, then validate on the robot under supervision before changing speed caps or calibrations.
