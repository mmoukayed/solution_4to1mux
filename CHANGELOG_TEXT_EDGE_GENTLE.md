# Build 10.0 — 2026-11-21

Baseline is the exact miracle_alt head 0dbc7016b392bec134d4611bbf7c5a8d1e78a1e7,
with every downloaded baseline blob checked against GitHub. Its corrected startup
joint bounds were retained; no older navigation branch was overlaid.

White books: isolated-print contact-sheet OCR plus bounded single-line retry reads
ROBOT, LIBRARY, AGENT and FUTURE in the actual supplied RGB photo. Three unique
capture confirmations are required. Registered depth grows a bounded front-spine
surface, separating deeper white background and horizontal board connections.
The surface centre is used for both pose and divider geometry. Text grasp checks
now use title/depth near the hand and require extraction from the original bay.

Edges: depth can refine divider side planes; unavailable measurements conservatively
shrink calibrated bounds. Fresh matching metadata gates the final grasp. The base
can align laterally with edge books while wholly behind the shelf front, preserving
LiDAR/front keepouts. Sampled arm/gripper capsules guard both dividers on every grasp
stage and recovery; impossible finger gaps are refused. Calibration-model tests
cover left/right edge grabs at row 2, 3 and 4 heights under the loaded URDF limits.
This is not a full mesh planner or proof of clearance on the actual robot.

Release: the old immediate finish_place opening was removed. A fresh, aligned bin
and completed hover precede fixed-torso Cartesian lowering, bounded table/rim checks,
and a 3-second opening ramp. An 8 cm descent from 1.08 to 1.00 m was solved/tested.
No force-controlled touchdown or bin-floor sensing is claimed. Frozen live gripper
feedback cannot shorten the opening ramp. Camera-based bin depth/confirmation and
pose handoff were consolidated so the included drop-box regressions match the nodes.

The pristine branch's offline suite had 17 failing drop-box regressions (193 passing,
3 optional lint skips). This release resolves them and adds actual-photo OCR,
synthetic RGB-D, edge path, hand verification and gentle-drop regressions. Some old
bare-node fixtures were updated to explicitly supply/mimic the now-required geometry
interface. Positive gentle-drop tests now require a planned descent before opening.
