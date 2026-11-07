# ERC 4To1MUX solution — revision 3 (14 September 2026)
## Setting Up Solution (First Run)
Run these commands to first setup the ROS 2 environment:
```
cd ~
git clone https://github.com/dfl-rlab/erc_sim_2026
cd erc_sim_2026
./docker/up.sh --build
```
Then to setup the solution program, run these commands:
```
cd src
git clone https://github.com/mmoukayed/solution_4to1mux
cd ..
./docker/attach.sh
```
Next once you are in the docker shell, these commands will setup the environment to run ros2 commands and the gazebo simulator:
```
colcon build --symlink-install
source install/setup.bash
```
Then use this command to start up the round 1 simulation:
```
ros2 launch erc_bringup simulation_round_1.launch.py   # Round 1 environment
```
And this for round 2:
```
ros2 launch erc_bringup simulation_round_2.launch.py   # Round 2 environment
```

> [!NOTE]
> If GUIs arent supported on your OS, use the `headless:=true` argument after the `ros2 launch` command of the simulator not the solution

## Starting Up Solution
To startup the docker container and code or if you want to open more than one terminal, you don't need to build from scratch again, simply run these commands to start:
> [!WARNING]
> If your docker container is already running and you want to just open more terminals, don't run `./docker/up.sh`. Just the `attach.sh` command to connect to the shell.
> 
> Additionally, if you are experiencing errors and issues with the docker container or ros2 environment and re-running `colcon build` isn't working, then add the `--build` flag to the end of `up.sh`.

```
cd ~/erc_sim_2026
./docker/up.sh
./docker/attach.sh
```
Then similar to the previous section, once inside the docker shell:
```
colcon build --symlink-install
source install/setup.bash
ros2 launch erc_bringup solution.launch.py
```
The start command for our program should be as mentioned in the guidelines:
```
ros2 launch solution_4to1mux solution.launch.py robot_profile:=simulation shelf_column_number:=2 book_colour:=red
```

For Phase 2 Round 2, launch `erc_bringup simulation_round_2.launch.py` in the simulator. Shelf 2 uses printed labels: install Tesseract in the simulator container once (`apt install -y tesseract-ocr`), then pass the exact requested label with `book_text`. For example:

```bash
ros2 launch solution_4to1mux solution.launch.py robot_profile:=simulation shelf_column_number:=2 book_text:="TARGET LABEL"
```

Leave `book_text` empty and use `book_colour:=red|green|blue|yellow` for the colored-book shelves, including the tight shelf 4 challenge.

## ROS topics: simulation and real robot

Edit **[solution_4to1mux/ros_topics.py](solution_4to1mux/ros_topics.py)** to change ROS names for the entire program. All three nodes import this file. It includes camera and LiDAR topics, base/arm/gripper commands, controller actions, robot description, contact feedback, internal node messages, and TF frame IDs. The launch file also remaps the library-owned TF and clock topics from this configuration.

`TOPICS`, `ACTIONS`, and `FRAMES` retain the existing simulator names. `PROFILES['simulation']` and `PROFILES['real']` override only the entries that differ. For example, edit `camera_rgb` under the real profile's `topics` to change that subscription. Changing an internal topic such as `book_pose` updates both its publisher and subscribers. Frame names belong under `frames`; they are not topic names. Optional interfaces absent from a profile use `None`; required interfaces must have a valid name.

The real profile now uses the **ERC 2026 Live Phase TIAGo Pro Interface Guide**, sections 2–3:

| Interface | Real robot mapping |
| --- | --- |
| Base velocity / odometry | `/cmd_vel` / `/mobile_base_controller/odom` |
| LiDAR | `/scan`, `/scan_front_raw`, `/scan_rear_raw` |
| Colour image / intrinsics | `/head_front_camera/color/image_raw` / `/head_front_camera/color/camera_info` |
| Depth used by perception | `/head_front_camera/aligned_depth_to_color/image_raw` |
| Raw depth / point cloud | `/head_front_camera/depth/image_rect_raw` / `/head_front_camera/depth/color/points` |
| Both arms, torso, head, both grippers | `/erc/<controller>/joint_trajectory` and `/erc/<controller>/follow_joint_trajectory` |
| Controller rejection reasons | `/erc/rejections` (logged by navigation) |
| Joint feedback / URDF | `/joint_states` / `/robot_description` |
| TF | `/tf`, `/tf_static` |

Perception uses colour-aligned depth with colour intrinsics and the colour optical frame; its existing `16UC1` conversion handles millimetres. Raw depth, point clouds, merged scans, and compressed image transport endpoints are also catalogued for reuse. Compressed names expand the guide's abbreviated transport suffixes. Navigation continues using both raw LiDAR streams and transforms them through TF.

```bash
# Gazebo: select simulation explicitly.
ros2 launch solution_4to1mux solution.launch.py robot_profile:=simulation shelf_column_number:=3 book_colour:=green

# Live robot: this is also the current DEFAULT_PROFILE in ros_topics.py.
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=3 book_colour:=green
```

The launch argument `use_sim_time:=auto` is the default: simulation uses simulated time and real uses system time. An explicit `use_sim_time:=true` or `false` overrides it for all nodes. The profile selection does not add a startup waiting state.

The real profile disables simulation-only `/contacts`, `/bin_contacts`, and `/base_imu`. Manipulation skips unavailable contact subscriptions and reports the missing feedback. **Hardware grasp/delivery verification still needs an adapter or a separate implementation**; the current contact-based checks cannot confirm success on the live robot. These interface mappings have been checked against the guide, not a connected robot. They do not calibrate motion or implement hardware effort-based grasp detection.

After adding this module, build the package and source the workspace as usual. With `colcon build --symlink-install`, later edits to `ros_topics.py` take effect when you restart the nodes; a normal copied installation needs another build. Every node logs `ROS interface profile: simulation` or `real` at startup. Direct `ros2 run` also accepts `--ros-args -p robot_profile:=real`; set `use_sim_time` explicitly for direct runs and supply any custom TF/clock remappings yourself, since those are normally applied by `solution.launch.py`.

The profile and launch checks run without ROS using:

```bash
python3 -m unittest discover -s test -p test_ros_topics.py -v
```

This package targets the **ERC simulator supplied in the 11 September 2026 ZIP**, ROS 2 Humble and Gazebo Harmonic. It uses the right arm only. Launch the simulator first, then launch this package.

**Current navigation build: 5.16; perception build: 5.12; manipulation build: 5.14.** Startup must print `Navigation BUILD 5.16`, `Perception BUILD 5.12` and `Manipulation BUILD 5.14` — if it does not, the rebuild did not take and you are running older code.

- **Navigation 5.16 / Perception 5.12 / Manipulation 5.14:** ROS topics, actions, and frames now come from `ros_topics.py`, with one `robot_profile` selection passed to every node. Simulation preserves the previous endpoint names; the real profile is editable separately.

- **Navigation Build 5.15:** navigation enters and publishes `FIND_SHELF` during construction and immediately requests the first head scan pose. The startup waiting state, readiness subscription, settle delay, and startup timeout are removed. The first odometry message supplies the sweep heading and starts the search timeout, avoiding an instant timeout when the simulation clock first arrives. Base motion still requires fresh odometry and LiDAR.

- **Build 5.13:** top-row grabs keep both arms folded during the base return, then move the right arm from a front clearance point to above the book before descending to pregrasp. The grasp inserts only after the base has stopped. The IK endpoint uses `gripper_right_base_link`, which is present in the current simulator frame dump, and manipulation readiness continues publishing while simulation time is paused.

- **Manipulation Build 5.12:** removed the 1.5 cm robot-right grasp bias after repeated left-finger-only book contacts. The grasp now targets the measured book centre laterally.

- **Manipulation Build 5.11:** `prepare_grab` validates its compensated arm reach target after the planned base advance, so a safe backed-out staging pose is not rejected merely because the book is temporarily farther away.

- **Manipulation Build 5.10:** gripper close sends one position trajectory and then waits for fresh bilateral finger/book contact. It no longer republishes shorter trajectories while closing, which could restart the controller's motion. A close timeout now includes gripper joint position and current-attempt contact ages.

- **Build 5.12:** lower-row docking uses a 0.92 m book standoff and a 0.84 m hard shelf keepout to leave the folded arm more room during `ALIGN_BOOK`. The base goal is also constrained to the target column's safe lateral footprint, leaving the arm to reach sideways to books near a divider. These limits apply before and during `ALIGN_BOOK`.

- **Build 5.11:** Shelf 2 can target the exact printed book label with `book_text:=...`; OCR searches only the confirmed column, checks both vertical text directions, and runs off the perception callback so camera processing stays live. Colored-book targeting remains available with `book_colour`. Book search now completes a 1.4 s head dwell at each angle and backs out 0.35 m after an 8 s failed view, rather than waiting 24 s. Return-to-start drives directly while preserving heading and chooses a short LiDAR-clear detour when the direct path is obstructed. Bin delivery now reaches the calibrated base standoff, raises the arm to a crane-like hover, then lowers the torso by 20 cm with the arm pose fixed before releasing.

- **Build 5.10:** the local simulator's number markers spawn at 1.93 m, below the old hardcoded 1.95 m minimum. Shelf recognition now allows marker heights from 1.75 to 2.55 m and logs separate depth/height rejection counts with measured heights. The marker-to-top-active-book offset is 0.335 m for this mounting height. Launch arguments `marker_min_height`, `marker_max_height` and `top_book_marker_offset` expose these calibrations; older simulator layouts with 2.26 m markers should use `top_book_marker_offset:=0.665`. At every column scan waypoint, the base now stays stopped through all three head pitches (5.4 simulation seconds by default), with `gate=observing_columns_...` identifying that observation pause. A confirmed target immediately ends the pause.

- **Navigation build 5.9:** `APPROACH_BIN` now recovers from `turning_envelope_blocked` with a separate `BACK_OUT_FOR_BIN_TURN` state. It checks rear clearance for a 0.25 m withdrawal, reverses at up to 0.08 m/s without rotation, then retries alignment. Both LiDAR feeds, held-book contact and the virtual bin wall remain active. Recovery is limited to two retreats, with an 8 s simulation-time limit per retreat. The final forward advance compensates for the measured withdrawal so the extra clearance does not leave the book beyond arm reach. Status output now includes angular command and heading error.

- **Build 5.8:** the bin release point moves an additional 0.10 m back toward the robot: the inward bin offset is now 0.22 m (was 0.12 m).

- **Build 5.7:** the release target was offset 0.12 m beyond the bin along the camera-to-bin normal, putting the held book over the far rim. The bin inset is now applied into the opening, centring the release over the bin interior.

- **Build 5.6:** a bottom-row book can leave the elbow/gripper overlapping the shelf after the old 0.35 m withdrawal, so the high carry pose collided while its joints reshaped. `RETREAT_FROM_SHELF` now withdraws 0.80 m straight back, holding heading fixed, before it requests the carry pose. This keeps the carrying arm clear before `RETURN_TO_START` begins.

- **Build 5.5:** the final release target was unreachable because the bin remained 1.534 m ahead of the arm (`0.425 m` best IK error). After its in-place rotation, the robot now advances forward only to the 0.95 m calibrated arm standoff (maximum 0.65 m advance) and then requests the lower-and-release motion. This path never issues a reverse command.

- **Build 5.4:** direct bin lowering now tries the normal, inverted, and both quarter-turn wrist orientations, allowing a valid release IK solution from the calibrated upright carry pose. If placement still cannot plan, navigation retains the exact manipulation failure reason instead of hiding it behind the generic `Placement failed` message.

- **Build 5.3:** `APPROACH_BIN` is rotation-only. It no longer corrects target-goal x/y after reaching the calibrated pre-table/drop position, so it cannot reverse away from the bin. When heading alignment completes, navigation skips the redundant pre-place/base-advance stage and directly requests the arm's lower-and-release sequence; this removes the `WAIT_FOR_PLACE_PREP` stall.

- **Build 5.2:** after `RETREAT_FROM_SHELF`, navigation stops and requests a high carry pose before returning to the start. Manipulation preserves the measured `torso_lift_joint` value and moves only the right arm to your calibrated position: `[0.0, 1.25, 0.0, -1.0, 0.2, 1.7, 0.0]`. Only after its controller confirms completion does navigation enter `RETURN_TO_START`.

- **Build 4.9:** the table/bin stop is now 0.95 m away (was 0.80), keeping the base turning footprint away from the movable bin. After the arm reaches its high pose, the protected final advance is 0.35 m. Placement planning no longer rejects the raw farther bin point before applying this compensation; the final IK waypoints remain the reachability check. A placement-plan error now publishes a normal failure instead of letting an rcutils logger exception terminate the manipulation node.

- **Build 4.8:** bin search has its own counterclockwise default (`bin_search_direction:=left`) while retaining the full bounded sweep, so a west-facing start turns toward the south-facing bin. Bin approach now uses close-range docking rather than a travel-turn circle. Once a bin pose has set the dock goal, navigation immediately stages placement even if the close-up red detection goes stale; manipulation accepts that static bin pose for up to 30 s during the delivery sequence.

- **Build 4.7:** delivery is staged to avoid the bin table: navigation stops 0.80 m in front of the detected bin/table, manipulation raises the held book to a high pose, navigation advances only 0.18 m under a virtual bin/table wall, then manipulation lowers above the bin, opens the gripper and verifies delivery. The bin pose now supplies this virtual wall just as the shelf pose protects close docking.

- **Build 4.6:** `ALIGN_BOOK` now immediately accepts a near-goal stop caused only by the virtual shelf keepout and begins the staged grasp; it no longer waits through the old 8 s blocked interval before recovery. A genuinely blocked path now recovers after 3 s. The arm inserts 0.085 m past the detected spine, seating the book against the gripper's inner body/backplate for lift support while preserving the base keepout.

- **Build 4.5:** the gripper remains at its physical closed limit (0.0); the public simulator interface exposes no effort/force command above that. To improve retention, the staged grasp inserts 0.07 m past the perceived spine (was 0.06), repeatedly reasserts the hard-close command while seating, and requires 1.0 s of stable bilateral finger/book contact before lifting or extracting.

- **Build 4.4:** restores the declared `pregrasp_distance` parameter required by the staged planner. Its absence caused `prepare_grab` to terminate the manipulation node with `ParameterNotDeclaredException` before any arm command was sent.

- **Build 4.3:** return-for-grab now targets 0.28 m after the 0.25 m retreat, preserving a 3 cm buffer above the virtual shelf keepout. If the hard shelf gate stops only the final centimetres, navigation accepts that safe position and starts the compensated close rather than freezing in `RETURN_FOR_GRAB`. The gripper target moves 1.5 cm to robot-right and inserts 0.06 m past the measured spine (was 0.04 m), so it wraps around the book without requiring unsafe extra base motion.

- **Build 4.2:** after prealignment, the base returns 0.30 m after its 0.25 m retreat, placing it 5 cm closer than the original dock for a more secure pickup. Manipulation offsets its prealignment target by the identical 0.30 m, preserving the intended final gripper/book alignment. The gripper's closed command remains 0.0—the simulator clamp's physical closed limit, so no higher software force command exists.

- **Build 4.1:** grasping is staged around the shelf clearance. At the finished dock position the base reverses 0.25 m; manipulation opens the gripper and moves the right arm to a prealignment pose offset by the same 0.25 m. The base then returns straight to the dock, carrying the held arm into the book alignment, and only then does manipulation close, lift, extract and verify the grasp. This prevents the arm's transition from the raised stow pose from catching boards above the selected row.

- **Build 4.0:** before `ALIGN_BOOK`, navigation stops and sends the calibrated complete trajectory for each arm. Left: `[0.0, 1.25, 0.0, -1.0, -0.4, 0.0, 0.0]`; right: `[0.0, 1.25, 0.0, -1.0, 0.4, 0.0, 0.0]`. Exact fresh `/joint_states` confirmation proceeds immediately; otherwise it allows the controllers' 2 s trajectory plus a 0.5 s settle period, avoiding a false stow timeout while still refusing stale joint feedback.

- **Build 3.5:** a grasp is refused with a `redock` reason when the book sits closer than 0.45 m in the base frame (base jammed against the shelf — swinging the arm from there hits the boards). Navigation reacts by backing straight out 0.35 m and re-docking (bounded by `max_book_recoveries`) instead of failing or wedging the arm.

## Revision 3: grasp clearance, shelf retract and unstick

This update targets the two live failures: `Grasp failed: grab: close: timed out` and the robot ending up stuck at/inside the shelf. Offline regression tests (41) pass; not run in Gazebo here.

- **Gripper now opens to 0.066 instead of 0.04.** The PAL Pro gripper travel is 0–0.069 and the four-bar linkage gives roughly a 3.7 cm fingertip gap at 0.04 — only ±8 mm of clearance around the 2 cm book, so a 1 cm perception error made the fingers ram the spine or close beside the book. At 0.066 the gap is ~5.6 cm (±1.8 cm of margin).
- **Insert is deeper (0.04 m past the perceived spine, was 0.025) and pregrasp starts further back (0.10 m, was 0.06)**, so the fingertip pads overlap the book instead of pinching its front edge, and the approach starts clearly outside the shelf. IK reachability re-verified for all four row heights with the offline arm model.
- **A failed grasp now retracts the arm.** Previously an abort only cancelled the controller goals, leaving the fingers wedged between the shelf boards (your "stuck in the shelf" case). Any failure between `insert` and `extract` now opens the gripper and sends arm+torso back to the pregrasp joints — a straight pull-out along the already-solved approach line.
- **The close-timeout error now says what the fingertips touched.** If no finger/book contact was ever reported, the message says so and tells you to verify the `/contacts` bridge (`ros2 topic echo /contacts --once` while the fingers press a book). One-sided contact means lateral misalignment; no contact at all with fingers visibly on the book means a simulator bridge problem, not a solution problem.
- **The shelf is now a virtual wall — this is the forward-collision fix.** The shelf mesh has almost nothing at the base LiDAR's 0.13 m scan plane inside a column aisle: the bottom board's top edge is at ~0.13 m and the divider panels are ~0.5 m to either side, so a head-on approach measured "clear" all the way into the shelf and the robot drove its bumper into the bottom board. Navigation now synthesizes obstacle points along the shelf-front plane (marker plates are flush with the front; book spines sit on it) from the confirmed column/book poses and feeds them through the existing clearance and turning gates. A stop caused by the virtual wall shows `gate=shelf_keepout_...` in the `State=` line, so a bad goal is now visible instead of a collision.
- **Deep colour blobs can no longer create a suicidal dock goal.** A red blob measured through a shelf gap up to 0.35 m behind the front plane previously passed the column filter; docking 0.68 m in front of *that* point put the base at or beyond the shelf front. Book candidates must now lie between 0.35 m in front of and 0.20 m behind the marker plane — strict toward the shelf interior, generous toward the robot, since the confirmed column pose can carry depth bias from far away (rejections show as `plane` with the measured `plane_offsets_m` in `/perception/book_diagnostics`), and the dock goal is additionally clamped to at least `book_standoff − 0.10 m` in front of the plane, with a warning when the clamp fires.
- **A blocked docking path recovers instead of freezing in ALIGN_BOOK.** The virtual wall is now built only from the confirmed book/column poses — never the shelf-unit centroid, whose heading can be a line-of-sight bearing; a tilted wall through a distant centroid could cut across a valid docking path and silently block it. The dock goal is re-clamped against the current wall plane every tick, and if the docking path stays blocked for 8 simulation seconds the robot backs out and reacquires the book (bounded by `max_book_recoveries`, now 2) instead of sitting out the 90 s phase timeout. The `State=` line's `gate=` field always names what is blocking.
- **The frozen-at-the-shelf deadlock is fixed.** Docking zeroes translation while the heading error is large, and rotation is separately suppressed when any LiDAR point is within the 0.49 m turning envelope; near the shelf both could hold at once, freezing the robot until the phase timeout. When rotation is needed but blocked and no translation is commanded, the robot now backs straight out at 0.08 m/s (only if the rear is clear) until it can turn again. Gate shows `turn_blocked_backing_straight_out`.

Reminder on `sim/wall=0.12`: at that real-time factor every 10 s of simulated motion takes ~80 s of wall time — a trial that looks frozen may just be slow physics. All solution timeouts run on simulation time, so slowness alone does not cause the failures above, but do check the `State=`/`gate=` line before assuming a hang, and try running the Gazebo server headless (no GUI rendering) in WSL to raise the RTF.


## Revision 2: sideways travel, close-up loss and diagnostics

This update addresses faults found in revision 1's code. **The exact cause of the user's live shelf abort remains unconfirmed without its log.** This update has not been run in Gazebo here.

- Long travel turns toward the waypoint, drives forward, then sets the final shelf-facing heading. It does not command lateral velocity during travel. Small lateral movements remain intentional in the last book-alignment step; straight reverse retreats preserve the book's heading.
- Column scanning now uses fixed waypoints, each followed by a camera observation pause. Previously the goal moved sideways with the robot on every tick.
- Forward travel can reach the configurable **0.85 m/s** cap. Docking is capped at **0.25 m/s**. Obstacle clearance applies an absolute speed cap instead of multiplying an already tapered docking command by another slowdown factor. Collision stops remain active.
- A confirmed column is separate from unconfirmed marker candidates. A bad crop cannot erase the confirmed reference. A reliable shelf normal estimated from multiple signs is retained when only one sign remains visible close-up.
- Losing the close-up book view triggers a stepped head scan with time to settle at each angle. A stale cached book location no longer pins the camera indefinitely. There is one optional, scan-protected 20 cm reverse move before reacquiring; the number is bounded by `max_book_recoveries` (default 1). These recovery motions happen **before** any grasp request, never after a partially executed grasp.
- A genuine controller, IK or grasp-contact failure still stops the trial. This patch does not treat missing contact as success or blindly repeat an arm motion.
- `/solution/failure` retains the navigation failure reason and `/manipulation/failure` retains the detailed arm/gripper error. `/perception/book_diagnostics` reports rejected colour blobs, missing depth, wrong-column candidates, row mismatch, and detection confirmation progress.

Every five wall-clock seconds, navigation prints:

```text
State=...; sim/wall=...; command=...m/s; odom=...m/s; remaining=...m; book_age=...; gate=...
```

`sim/wall` estimates physics speed relative to real time; `command` is the speed currently sent to the base; `odom` is measured velocity in simulation units. The `gate` field identifies a missing scan, clearance speed limit or blocked turning envelope. An idle search/manipulation state legitimately has zero base speed. These diagnostics distinguish simulator slowness from controller waiting or blocking.

### Run with a saved log

After applying this archive, rebuilding and sourcing as described below, run **inside the existing container**, with the simulator already running:

```bash
cd /opt/erc_ws
source install/setup.bash
bash src/solution_4to1mux/scripts/run_trial.sh \
  shelf_column_number:=2 book_colour:=red
```

The helper prints the log path and saves the complete console output under `src/solution_4to1mux/logs/trial-*.log`. ROS logs also go into that package's `logs/ros` folder. These files use the existing source mount; retain the earlier D-drive storage setup. This changes no Docker settings and requires no Docker image rebuild.

If it stops, send the latest trial log, or at least its final 30–50 lines including the manipulation error and `State=...` line. You can inspect retained failure reasons while the nodes are still running:

```bash
ros2 topic echo /solution/failure --qos-durability transient_local --once
ros2 topic echo /manipulation/failure --qos-durability transient_local --once
```

A failure topic that has not been published yet will wait. Do not infer the cause from motion alone. There is no new Python dependency in revision 2.

## Install the repair on your existing WSL/Docker setup

Finish relocating Ubuntu's virtual disk and Docker Desktop's data to **D:** as discussed before rebuilding. Linux paths such as `~/erc` and `/opt/erc_ws` are then backed by those virtual disks on D:. Merely putting the download on D: does not move Docker data.

1. Stop the currently running **solution** with Ctrl+C. Keep only one solution process running. For a clean trial, restart the simulator too after installation.
2. Download the repaired `solution_4to1mux.tar.gz` to `D:\ERC`.
3. In your **Ubuntu WSL host** terminal, go to the existing simulator repository (the path below matches the earlier ZIP setup; adjust it if your repository is elsewhere):

```bash
cd ~/erc/erc_sim_2026-main
mkdir -p /mnt/d/ERC
# Backup the existing solution source. This does not remove it.
tar --exclude='.git' --exclude='__pycache__' \
  -czf /mnt/d/ERC/solution_4to1mux-before-fix.tar.gz \
  -C src solution_4to1mux
# Update source files. The archive contains the solution_4to1mux/ top folder.
tar -xzf /mnt/d/ERC/solution_4to1mux.tar.gz -C src
```

4. Enter the existing container (do not run `docker/up.sh`, which recreates it):

```bash
docker exec -it erc_sim /entrypoint.sh bash
```

5. **Inside the container**, install the numerical solver dependency if missing and rebuild just the solution:

```bash
apt-get update
apt-get install -y python3-scipy python3-opencv
cd /opt/erc_ws
colcon build --symlink-install --packages-select solution_4to1mux
source install/setup.bash
```

The provided container runs as root; use `sudo` for the apt commands only if you changed that. The existing simulator supplies the ROS dependencies. `package.xml` now declares the dependencies used by this package, including SciPy; `colcon build` itself does not install them.

6. Run the simulator in one container terminal. Disable the extra point-cloud conversion if it is slowing the simulation; this solution uses RGB/depth/CameraInfo directly:

```bash
ros2 launch erc_bringup simulation.launch.py depth_cloud:=false
```

7. Open another Ubuntu WSL terminal and attach/source again:

```bash
docker exec -it erc_sim /entrypoint.sh bash
source /opt/erc_ws/install/setup.bash
ros2 launch solution_4to1mux solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

All three solution nodes use `use_sim_time:=true` by default. Leave that enabled for Gazebo. Do not run an additional teleoperation publisher on `/cmd_vel` during the trial.

## What was wrong and what changed

| Original problem | Repair |
|---|---|
| Search started rotating as soon as odometry arrived, before the camera was ready. Sweep progress was guessed from commanded speed. | Wait for fresh camera/TF/controller inputs; check observed targets before motion; use bounded 70/155-degree search headings and odometry with wrapped angle errors. Rotation is still necessary when the shelf is outside the camera view. |
| Timers added a hardcoded `0.1` each tick using wall time. | All nodes share ROS simulation time; elapsed time comes from the ROS clock. Logs show measured simulation/wall-clock ratio. |
| Fixed/slow movement and repeated fine alignment. | Configurable 0.85 m/s forward cap and 0.25 m/s close-range docking cap, proportional approach and vector-limited diagonal motion. These are limits, not guaranteed average speeds. |
| Front laser's zero ray treated as base-forward, although its mount is yawed -45 degrees and rolled 180 degrees. | Transform both LiDAR scans into `base_footprint` using TF before directional obstacle checks. |
| Missing distance triggered blind creep; an obstacle stop was reported as successful docking; a warning could format `None` as a float. | Missing scans stop motion with a diagnostic. Obstacle stop is not arrival. Docking uses the camera-derived odometry-frame goal. |
| Largest colour blob anywhere in the image chosen as the book; row guessed from image quarters. | Require a confirmed target column, restrict colour or OCR label matches to that column in 3D, and infer row from relative marker/book heights and fixed shelf spacing. No randomized book/column world positions are read from Gazebo. |
| Empty digit-template directory led to synthetic-font matching. | Include copies of the supplied simulator's public number-marker textures 1–5 as recognition templates. |
| Code called `/gripper_right_controller/follow_joint_trajectory`, which does not exist in this simulator. | Send bounded commands to the public `/gripper_right_controller/joint_trajectory` clamp topic, using joint feedback for opening and bilateral book contact for closing. Arm/torso continue using their real actions. |
| Hardcoded placeholder arm poses did not use the book's measured position. | Read `/robot_description`, solve right-arm/torso IK for live RGB-D targets, check all waypoints before motion, and respect joint/velocity limits. Try the two symmetric wrist-roll orientations. |
| A 35 cm torso move was commanded in 2 s despite a 0.035 m/s limit. | Compute trajectory durations from the actual URDF velocity limits and joint feedback. |
| Failed grasp was hidden and eventually caused an empty return to the bin. | Publish failure immediately, cancel active motions, and show `FAILED`; never treat timeout as success. Reject cancelled/rejected/aborted action results. |
| Navigation could rotate with the book still inside the shelf. | Extract slightly, then retreat the base 35 cm at unchanged heading before returning to the start. Monitor held-book contact during carrying. |
| Moving to the bin was described as scoring even after a failed grasp. | Remove that incorrect assumption. Navigation points depend on the corresponding successful grasp/delivery in the rules. |

## Expected states and diagnostics

Normal state progression:

`FIND_SHELF -> [SCAN_COLUMNS if needed] -> APPROACH_COLUMN -> FIND_BOOK -> STOW_ARMS_FOR_DOCK -> ALIGN_BOOK -> [REACQUIRE_BOOK / BACK_OUT_FOR_VIEW if needed] -> WAIT_FOR_GRAB -> RETREAT_FROM_SHELF -> RETURN_TO_START -> FIND_BIN -> APPROACH_BIN -> APPROACH_BIN_FOR_PLACE -> WAIT_FOR_PLACE_PREP -> WAIT_FOR_PLACE -> DONE`

Each transition is printed. Failures print a cause and publish `FAILED` on `/solution/state`; the robot stops. `FAILED` is not a successful trial. Restart the solution after fixing the reported cause; restart the simulator for a clean randomized trial.

In a third container terminal:

```bash
source /opt/erc_ws/install/setup.bash
ros2 topic echo /solution/state --qos-durability transient_local
```

Useful checks (run individually):

```bash
ros2 action list
ros2 control list_controllers
ros2 topic echo /perception/book_pose --once
ros2 topic echo /manipulation/ready --once
ros2 topic echo /manipulation/holding_book --once
ros2 topic hz /head_front_camera/head_front_camera/color/image_raw
```

Expect arm and torso `follow_joint_trajectory` actions. The real gripper controller also has a `_raw` action, but this package deliberately uses the public clamped topic. There is **no public non-raw gripper action** in the supplied simulator.

The log `sim/wall=0.20` means five real seconds produce about one simulation second. Raising base velocity does not fix that. Check WSL/GPU acceleration and Gazebo rendering, and keep `depth_cloud:=false` if you do not need the point cloud. This repair does not alter Docker, the simulator, the robot model or physics settings.

## Tuning after the first live run

```bash
ros2 launch solution_4to1mux solution.launch.py \
  shelf_column_number:=2 book_colour:=red \
  linear_speed:=0.85 docking_speed:=0.25 strafe_speed:=0.25 \
  search_rotate_speed:=0.45 search_direction:=right \
  book_standoff:=0.68 book_lateral_offset:=-0.18
```

`book_standoff` is measured from **base_footprint to the observed book spine**, not from the laser or camera. The initial pose/height tests use 0.68 m and a book 0.18 m to the right of the base. Arbitrary changes can make a shelf row unreachable. Obstacle checks and proportional approach can reduce speed below the configured caps.

Annotated images go to the package source's `erc_images` directory with timestamps. The three images already in the upload are preserved as old diagnostic samples; they are not evidence of a repaired trial. New scored images must be produced by your live run.

## Validation and current limits

Offline regression tests cover the real ERC arm-chain geometry at all four active-row heights, limits and torso duration; rotated laser coordinates; shortest-angle turns; immediate target acquisition; stale/blocked scans; wrong-column colour selection; actual marker templates on the supplied camera image; grasp failure handling; bilateral contact identity; and late/cancelled action responses.

```bash
python3 -m pytest -q test/test_regressions.py
```

The callback tests use small ROS import/message stand-ins. They do **not** emulate ROS networking, controller execution, Gazebo contacts or physics. The numerical tests check waypoint reachability, not collision-free trajectories or grasp success rates.

A full Gazebo trial was **not run** in the editing environment because ROS/Gazebo are unavailable there. This is a repaired implementation to test in your simulator, not a verified competition submission. Remaining concrete limits:

- Local IK and joint waypoints are **not whole-body or mesh collision planning**. The base obstacle check covers its footprint, not extended arms. Validate arm clearance, especially the initial arm posture, with the simulator before repeated autonomous trials; a configured MoveIt planning scene is the next step if paths collide.
- Printed-label targeting needs the `tesseract-ocr` executable in the simulator container. OCR uses the confirmed column crop and matches the requested label; small, occluded, or unfamiliar print may still need a closer view or more contrast.
- RGB/depth projection assumes the supplied simulator's matching virtual-camera geometry. Real-camera deployment needs calibrated depth registration.
- Grasp insertion/lift and wrist orientation need live contact validation. The simulator README itself notes gripper/contact issues. Missing or one-sided book contact produces an explicit failed grasp rather than invented success.
- The bin placement uses a measured visual target with fixed inset/height offsets, and requires book/bin contact to report completion. It has not been tuned or validated for the rubric's gentle-placement points.
- Perception relies on the public shelf geometry (0.33 m row spacing and marker-to-top-book offset), the supplied number textures, colour segmentation, and OCR of the selected text label. Partial occlusion, odometry drift and changed simulator geometry still require testing.
- Startup/travel speeds depend on the host's actual Gazebo real-time factor. Increasing the commanded speed cannot compensate for a slow physics/rendering loop.

The fixture `test/data/erc_right_arm_chain.urdf` contains only joints/limits extracted from the supplied ERC robot description for numerical testing. Runtime kinematics always come from the live `/robot_description` topic.
