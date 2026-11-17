# Build 9.0 — 2026-11-17

Evidence: final Build 8 run in Pasted markdown(7).md, lines 9754–9905. Earlier runs in that file use mixed older builds and were not treated as the current baseline. The final run's fingerprint matches the prior delivered Build 8 source.

Observed: book confirmed at 1794902237.862, followed by close-view column/depth rejections and a view backout. A new dock target was clamped because its depth appeared suspect; a later fresh target shift caused another view backout. The grasp then inserted, closed, lifted and extracted, and the camera confirmed the book held in 3/3 frames at 1794902312.068. Return reached FIND_BIN at 1794902327.964. The approach reported 7.80 m remaining; prepare_place received [8.431, 0.277, 0.666] in the base frame and failed IK at wrist [0.660, 0.277, 1.337] with 0.053 m best error.

Confirmed code defects: bin detection used a single centre depth with no range/metric-size/multiframe checks; navigation retained an old lateral goal; placement could ignore suspect forward range via a hardcoded release x; finish_place could open before validating a current transform/target or completed measured hover. Book colour depth also sampled its centre, and close-view candidates depended on a visible white ring. The source now addresses these paths and retains the existing guarded grasp/speed/controller handling.

Inference: the distant target may be background depth visible through the red rim or a different red object; text alone cannot identify which image pixels were selected. Synthetic hollow-rim tests exercise that failure mode. No new real RGB/depth capture was available.

The exact +0.277 m lateral IK failure was reproduced with the supplied URDF. At the intended -0.18 m lateral offset the original guarded crane waypoints solve within joint limits. Base centring is refreshed, shared with manipulation, and enforced at request time; no IK tolerance or controller limit was weakened.

Changed files: target_geometry.py (surface-depth/pose guards), perception_node.py (rim/spine measurements and recent confirmation), navigation_node.py (fresh bin centring, bounded waits, snapshots, targeted book scans/realignments), manipulation_node.py (shared target and guarded release), launch/solution.launch.py, ros_topics.py, release identity/metadata, offline regressions and instructions.

Limitations: no ROS/DDS/physical delivery validation; no camera/bag replay. Rim shape, size limits and existing fixed release calibration still need supervised validation on the actual box. Sampled arm clearance is not a full mesh collision planner. Bin delivery remains unverified without a bin sensor.
