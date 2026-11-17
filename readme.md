# Library robot — Build 9.0 (0.9.0)

This release keeps the successful Build 8 grasp path and repairs the drop-box/search paths seen in the final run of `Pasted markdown(7).md`. That run visually confirmed the held book and returned, then accepted an implausible bin target (8.431 m forward, 0.277 m left) and failed the hover IK. The old centre-pixel depth sampling could read through a hollow red box to the background; without images or a bag, that particular pixel failure remains an inference.

Replace the complete package; do not overlay selected files or keep another package copy under src. The archive contains the solution, original image/template assets and offline URDF test fixtures, not a complete simulator workspace. Controller/feedback fault inhibition from Build 8 is preserved.

## Drop-box and book-search repairs

- Drop-box depth now comes from a supported cluster on coloured surface pixels, not the opening at the bounding-box centre. Dark red and oblique views are supported. Candidates must have a plausible metric width/height and world height, and confirm in three distinct, recent frames. Ambiguous, sparse or scattered depth does not produce a target. Navigation independently rejects targets beyond 4 m.
- Navigation updates the bin's lateral centre before moving the arm; stale observations stop the base. Brief dropouts wait stationary, with bounded reacquisition. Distant targets are approached with the arm in its carry pose before the placement swing. Existing LiDAR, turning and table-clearance gates remain.
- The base and arm share `bin_lateral_offset` (default -0.18 m, on the right). The exact logged y=+0.277 hover IK failure was reproduced offline. The intended y=-0.18 hover solves under the supplied URDF joint limits. This is kinematic validation, not a physical collision/delivery test.
- Navigation passes a timestamped placement snapshot to the arm through `/solution/place_target_pose`. The snapshot may be used for up to 30 seconds while the arm occludes the rim; it is never restamped as a fresh observation. Opening requires a recent completed hover, measured joint agreement, current base TF and the box inside the calibrated release envelope. Otherwise the book remains held.
- Book depth also uses coloured surface pixels. An identified column's projected image crop can contribute spines even when the surrounding white bay is clipped; 3D column, shelf-plane, row and metric-size gates still apply. One missed frame preserves recent confirmation evidence without publishing a stale pose; long gaps reset it.
- The camera first revisits the previously detected book before a broad pitch scan. A changed, freshly confirmed docking target gets up to two stationary realignments instead of an immediate 0.35 m view backout. Bounded backouts remain when the book actually cannot be reacquired. The staged backout/return used for arm clearance is intentional and remains.

Detection thresholds are conservative defaults for this competition setup: drop-box width 0.20–1.20 m, height 0.08–0.80 m, centre 0.60–1.20 m above the floor; book width 0.015–0.22 m and height 0.10–0.55 m. Annotated successful bin detections are saved as `collection_bin_*.png` in the configured images directory. Search diagnostics report candidate rejection reasons. Check these against real camera output before changing calibration or thresholds.

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

`verify_install.py` compares the loaded core Python files to this release and prints their path and source fingerprint. Do not add src to PYTHONPATH to make this check pass. All three nodes must print BUILD 9.0, version 0.9.0 and the same fingerprint. This detects mixed or stale installs, not physical readiness.

The read-only preflight checks streams, transforms, actions and advertised ListControllers services. An action server alone does not prove the controller is active. An unavailable controller-state service produces a warning, not proof of readiness. Resolve failures and check robot readiness before launching.

```bash
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=3 book_colour:=red
```

`book_color:=red` is also accepted. Conflicting colour aliases are rejected. Check the selected colour in startup output. Use `robot_profile:=sim` for the simulator; clock defaults follow the profile. This archive contains the solution package and offline URDF test fixture, not a complete simulator workspace.

## Movement, alignment and grasp changes

- Travel speed cap 0.45 m/s; carrying 0.25 m/s; docking 0.16 m/s; final grasp return 0.08 m/s. Acceleration ramps, immediate obstacle stops and scan-age braking remain enforced. These are caps, not guaranteed achieved speeds. Arm timing uses a 0.60 speed fraction and the robot's joint limits.
- Shelf headings come from distinct marker observations, reject outliers and update coherently during approach. Final docking uses a single frozen shelf reference for lateral alignment and grasp orientation. The selected book pose is handed to manipulation explicitly.
- The live marker minimum is 1.60 m, accepting the approximately 1.70 m markers seen in the output. Simulation retains 1.75 m. Override with `marker_min_height:=...` only for a measured setup.
- Startup tuck and dock stow require measured joint convergence. Right-arm joint 2 obeys the live URDF limit instead of the rejected 1.25 rad target. Both arms stay in their calibrated tucked configuration during base movement where required.
- Depth registration uses both camera transforms and timestamp checks. OCR keeps the original observation time; stale results cannot become fresh targets. Depth projection uses a nearest-depth buffer and limits repeated calibration work.
- Live grasp-frame fallback, hand-camera correction and visual grasp confirmation are preserved. The fallback calibration still needs checking on the physical robot. Missing confirmation is a failed grasp, not assumed success.
- Top-row torso clearance and sampled arm-link shelf checks remain mandatory. Unsafe plans are refused. These checks are not a full mesh collision planner.
- Return and placement preserve the full navigation goal across backoffs, use carrying clearances and align laterally to the bin. Without a live bin sensor, completion means the release sequence completed, not independently proven delivery.

## Runtime faults and diagnostics

Controller deactivation, rejected/failed trajectory actions, lost action responses, motion feedback timeouts, stale grasp transforms and lost active joint feedback latch a robot fault. The attempt cancels accepted action goals where possible, stops requesting new manipulation trajectories, and does not automatically open the gripper, retract or retry. Navigation requests zero base velocity. Continued odometry/LiDAR loss stops movement immediately and latches a fault after 3 seconds. A communication failure can prevent commands reaching the robot; application commands cannot guarantee a physical stop in that situation.

Fresh feedback alone does not clear the fault. Inspect the robot, recover under operator control, then restart the solution. Inspect the recorded reason and fault snapshot with:

```bash
ros2 topic echo /solution/robot_fault
```

For a new trial, run `scripts/record.sh trial_name` in another sourced terminal before launching. It records solution diagnostics, ROS logs, TF, feedback, selected camera data and matching controller state/action-status topics when published. `--light` omits depth images; `--raw` adds raw colour and can generate very large bags. `scripts/run_trial.sh` saves launch output. No rosbag or current camera images were supplied for this repair; the evidence was source code and text output.

## Validation and limits

See VALIDATION.txt, CHANGELOG_DROP_BOX.md and the retained CHANGELOG_RUNTIME.md (Build 8 history). Offline tests exercise alignment geometry, joint limits, guarded planning, state transitions, camera logic and the final deactivation/frozen-feedback failure. ROS, DDS, controller lifecycle and physical grasp/delivery have not been run here. Rebuild and verify the installed code, then validate on the robot under supervision before changing speed caps or calibrations.
