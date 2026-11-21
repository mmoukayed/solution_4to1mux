# Library robot — Build 10.0 / version 0.10.0

Based on `miracle_alt` commit `0dbc7016b392bec134d4611bbf7c5a8d1e78a1e7`.
This release adds depth-supported white text books, guarded edge-book alignment,
and slower release. It preserves the branch's startup reset and joint-limit fix.

## Replace, rebuild and verify

Stop the previous launch. Put the included package at
`/opt/erc_ws/src/solution_4to1mux/package.xml`, moving the old package outside src
first. Keep a backup of local changes; do not leave two copies of the ROS package
under src. Alternatively apply the included patch to the exact baseline checkout.

```bash
cd /opt/erc_ws
source /opt/ros/humble/setup.bash
source tiago_dds_connect/tiago_env.sh
colcon build --symlink-install --packages-select solution_4to1mux
source install/setup.bash
python3 src/solution_4to1mux/scripts/verify_install.py --source src/solution_4to1mux
python3 src/solution_4to1mux/scripts/preflight.py
```

Every node must print `BUILD 10.0`, version `0.10.0` and the same source fingerprint.
Do not add src to PYTHONPATH to make install verification pass. The check detects
mixed installations; it does not verify physical calibration or controller health.
The existing beginning-of-run arm/torso/head reset remains active.

## White books with black labels

Supported titles: `ROBOT`, `LIBRARY`, `AGENT`, `FUTURE`. The photo supplied for this
repair shows those on rows 1, 2, 3 and 4 respectively; the column is not inferred
from the cropped photo. Set `shelf_column_number` to the actual numbered column.
Start validation with LIBRARY on row 2 or AGENT on row 3. Existing top-row shelf
clearance refusals are retained; recognizing ROBOT does not override them.

Example **if the white books are in column 2** (replace 2 as needed):

```bash
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=2 book_colour:=white book_text:=LIBRARY
```

Change only `book_text` for the other three titles. `book_color` is also accepted.
For ordinary coloured edge books, omit `book_text` and select their colour:

```bash
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=3 book_colour:=red
```

Tesseract plus English language data must be installed in the container. OCR groups
black print into horizontal/vertical regions rather than feeding shelf grids to the
recognizer. A bounded four-title spelling correction handles one uniquely closest
glyph error at sufficient OCR score. Weak readings get an isolated-line re-read;
they are not accepted solely because spelling is close. Three independent fresh
captures and 3D bay checks must agree. A broad-side label alone does not authorize
pickup: the vertical spine must be identified.

A registered depth component anchored at the spine label determines the book's
front-surface centre, width and height. It separates the book from a deeper white
backboard and removes broad horizontal shelf-board rows. Missing/merged/incoherent
surfaces are refused, rather than treating all white pixels as a book. RGB, depth,
camera transforms and timestamps remain tied to the original capture.

Text search holds the head 6 seconds per view and permits 36 seconds before a view
backout. Colour mode keeps 3/20 seconds. Override with `head_scan_dwell:=...` and
`book_search_timeout:=...`. OCR work is bounded; stale OCR results are discarded.
The colour-mode travel speed and obstacle/sensor limits remain unchanged.

Live text grasp confirmation is now enabled. For these neutral spines it re-reads
the same title in a projected hand ROI in two distinct fresh captures, checks depth
near the hand, and requires displacement out of the frozen pre-grasp shelf position.
Reading the book still standing behind an empty gripper cannot prove a grasp.
An unreadable/occluded title remains unconfirmed and the robot refuses carry.

The supplied RGB photo passes recognition for all four titles. It contains no
depth data: depth segmentation and physical grasp success are not established by
that photo. Top/bottom rows, clipped spines, glare and low-resolution print still
need supervised robot validation. A useful read-only image check after sourcing:

```bash
python3 src/solution_4to1mux/scripts/check_text_book.py frame.png --text LIBRARY
```

`--roi X Y W H` optionally crops to a column/spine. OCR scores are not calibrated
probabilities of a successful retrieval.

## Books near a divider

Perception publishes depth-supported book dimensions, shelf normal and divider
bounds. When visible, side-facing depth clusters refine the calibrated divider
positions. An unseen divider uses the calibrated column width with a 3 cm inward
margin; it is never assumed farther away to make a plan fit.

For an edge book with checked finger clearance, the base may align laterally beyond
the old single-bay footprint clamp while its entire footprint remains behind the
shelf front. Fresh LiDAR, footprint, turning and front keepout gates remain. This
reduces excessive sideways reach by the right arm. The checked alignment snapshot
survives brief camera occlusion during stow, but the final handoff still needs fresh
matching geometry.

The frozen geometry is transformed into the actual base pose for each staged arm
plan. Adaptive samples check arm-link capsules and the open gripper envelope
against both side dividers during approach, insertion, lift and extraction.
Retry/retraction paths also need divider clearance. Impossible side gaps are
refused before opening/inserting; the robot does not shove books off a divider.

Default open gripper half-width is 0.065 m plus a 0.015 m margin. This is a
conservative configurable envelope, not a measurement of the real gripper mesh.
`gripper_half_width:=...` is shared between base and arm; measure the actual open
outer envelope before changing it. The included arm tests use the supplied URDF,
not a complete mesh/self-collision planner. The nominal column half-width remains
0.43 m; it must match the physical shelf. A book flush against a divider may be
physically ungrippable with the available fingers.

## Slower, lower release

After the measured bin hover and base approach, the arm follows four short vertical
waypoints with the torso fixed. Default lowering is up to 0.08 m at no more than
0.025 m/s, stopping above the configured bin rim. Sampled link envelopes must stay
above the configured table. If the full descent is unsafe, smaller checked descents
are tried; if none is safe, the book is held and no opening command is sent.
The gripper opens over 3 seconds and the sequence waits for the complete ramp,
including when the live finger-feedback channel is frozen. There is no automatic
fallback to the old abrupt opening. Placement requires a fresh plausible bin target
under the held book and a recently completed hover.

Launch options: `gentle_drop_distance:=0.08` (allowed 0.02–0.12 m),
`gentle_open_duration:=3.0` (2–6 seconds). Internal calibration defaults are bin rim
0.95 m and table surface 0.72 m; these must agree with the actual setup. The lowering
reduces the remaining fall height. Without bin-floor/contact/force sensing it is
not a contact-controlled set-down, and bin delivery remains unverified on live
hardware without a bin sensor. Validate with an empty bin first.

The bin detector uses surface depth instead of the hollow centre, temporal/size/range
checks, fresh lateral alignment and a frozen placement snapshot. This also resolves
inconsistencies between the branch's included drop-box tests and its earlier nodes.

## Diagnostics and validation

```bash
ros2 topic echo /perception/book_diagnostics
ros2 topic echo /perception/book_geometry
ros2 topic echo /solution/failure
ros2 topic echo /solution/robot_fault
```

`book_geometry` reports world point/dimensions, original stamp, divider bounds and
whether the bounds came from depth or calibrated margins. The recorder already
captures `/perception/*` and `/solution/*`. Use `scripts/record.sh trial_name` before
launching and preserve launch output for a supervised test.

See `VALIDATION.txt` and `CHANGELOG_TEXT_EDGE_GENTLE.md`. No real RGB-D recording,
controller session or physical execution was available for this repair. All limits,
controller-fault inhibition and guarded startup remain; no safety/controller reset
or deployment to the robot was performed.
