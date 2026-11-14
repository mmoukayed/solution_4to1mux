# ERC 2026: 4To1MUX solution

One ROS 2 Humble package, three nodes (perception, navigation, manipulation). It finds the
numbered shelf column, picks the book by colour (or by printed label on shelf 2) and drops it
in the bin. Right arm only.

```
ros2 launch solution_4to1mux solution.launch.py robot_profile:=<simulation|real> \
    shelf_column_number:=<1-5> book_colour:=<red|green|blue|yellow>
```

`robot_profile:=real` uses the live TIAGo Pro's topics and the organisers' `/erc/...` checked
controllers; `simulation` uses the simulator's. All topic and frame names are in
`solution_4to1mux/ros_topics.py`.

## Run it in the simulator (CPU only, no GPU needed)

`sim/` holds the competition arena in MuJoCo. It runs on the CPU in Docker, needs no GPU or
display, and has the same topics as the organisers' Gazebo simulator.

```bash
sim/sim.sh build              # once: builds the image (15-30 min, needs internet)
sim/sim.sh up 42              # start the arena (seed 42 = one random layout; any number works)
sim/sim.sh run 2 blue         # run the solution: shelf column 2, blue book. Ctrl-C stops it
sim/sim.sh down               # stop
```

- `sim/sim.sh up SEED 2` gives the round 2 arena (white-label books on shelf 2):
  `sim/sim.sh run 2 red book_text:="LABEL"` (the colour is ignored when a label is given).
- Run `up` again before every new attempt: it resets the arena.
- `sim/sim.sh shell` opens a shell in the container with ROS sourced (`ros2 topic list` etc.).
- The books have realistic friction. `BOOK_MU=10 sim/sim.sh up 42` gives Gazebo's very sticky
  books, where almost any grasp holds.

## Record a video

```bash
sim/sim.sh up 42
sim/sim.sh record start
sim/sim.sh run 2 blue         # wait until it finishes (State=DONE or FAILED), then Ctrl-C
sim/sim.sh record stop run1.mp4
```

The video shows the whole arena with the head camera in the corner. Frames are stamped with
simulation time and the video is assembled afterwards, so it always plays at the robot's real
speed, even when a slow computer ran the simulation slower than real time. Nothing needs a
screen or an internet connection once the image is built.

## On the real robot

```bash
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=3 book_colour:=green
```

What the `feat/thisas_fixes` branch changed for the live robot (each is its own commit):

- the live URDF has no `gripper_right_grasping_link`: the simulator's grasp offset is used;
- the `/erc` checker rejects goals within 0.05 of a joint limit and arm segments faster than
  0.60 / 0.75 rad/s: targets and timings now stay inside both (the torso no longer goes to 0.35,
  joint 2 no longer to 1.25);
- the robot has no `/contacts` or `/bin_contacts`: the grasp is confirmed by the gripper stopping
  part-closed on the book, the drop by the gripper opening;
- slower base and arm speeds (the simulator ran at ~0.2x real time, which hid how fast 0.85 m/s is);
- both arms tuck into the robot's resting pose before the base moves, the left arm stays tucked;
- in-place turns while carrying keep 0.85 m from anything the LiDAR sees, the return to the
  start stops short when the bin table is close, and the bin approach has a hard stop from the
  bin's measured position (the LiDAR only sees the table's legs);
- the final grab uses the last book pose when the hand hides the book from the camera.

### Slot run sheet (state of `feat/thisas_fixes`, 14 Nov)

Inside the ERC container (`source /opt/ros/humble/setup.bash && source install/setup.bash`), with the
DDS bridge up:

```bash
python3 scripts/preflight.py      # listens 8 s: every topic the nodes use, its rate and timestamp age,
                                  # the /erc controllers, the URDF, TF. Must end in READY.
```

A clock offset between the laptop and the robot shows up as a large "stamp age": fix it before running
(the nodes judge a pose's freshness by its timestamp).

Second terminal, same container - record every try, stop with Ctrl-C after it:

```bash
scripts/record.sh try1            # -> bags/<date>_<time>_try1 (mcap, ~2 MB/s; --light without depth)
```

Then:

```bash
scripts/run_trial.sh robot_profile:=real shelf_column_number:=3 book_colour:=green
# long shelf on open floor (column at its far end): add scan_half_width:=2.6
```

What to expect (Isaac twin, every measured real-robot fault on):

- **middle rows**: pick, carry and drop into the bin work (clean passes in hall and platform layouts);
  the drop is fingers-down from above the bin, the base stops 0.10 m from the table legs by LiDAR;
- **top row**: refused on purpose. Every arm path from the dock crosses the board below the target bay, so
  the plan is rejected before anything moves (`top-row path puts ... through the shelf face`): no points,
  no shelf contact;
- **bottom row**: unreliable (missed grasps, one book slipped in the carry);
- the grasp is confirmed by the camera (the right finger sensor reads 0 on the robot): up to 4 side retries;
- a failed run stops in `FAILED` with the reason on `/solution/failure`. Keep a hand on the e-stop.

## More

- `docs/HISTORY.md`: the earlier build-by-build notes and diagnostics.
- `VALIDATION.txt`: the offline regression tests.
