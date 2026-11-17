# Build 8.0 — 2026-11-17

The supplied archive was the original older package, but the text output contained multiple later builds. This release consolidates the existing alignment/live-robot repairs and adds a common source fingerprint and a read-only source/install comparison so mixed installations are visible.

The final logged run reached insertion after hand correction, then the trajectory action reported status 6, error -1, “Current goal cancelled during deactivate transition.” Joint feedback and TF stopped updating. The prior recovery issued gripper/retract commands afterward. Build 8 latches the fault across navigation and manipulation, cancels accepted goals when possible, inhibits new recovery/retry/release trajectories, preserves the recorded arm/book state, and requires operator recovery and restart. It does not identify why the controller deactivated.

Active joint-feedback loss, unexpected action failures and motion deadlines use the same fault path. Grasp transforms must be current; odometry/LiDAR waits are bounded. Planning workers return retreat metadata rather than mutating live recovery state after cancellation. Controller lifecycle checks are read-only; no reset, switch or activation calls were added.

Carried forward: both colour parameter spellings, live marker-height gate, robust shelf heading fit, coherent approach updates, frozen docking reference, explicit grasp target transfer, measured tuck/stow, URDF joint limits, guarded top-row plans, moderate speed and acceleration limits, timestamp-aware registered depth/OCR, visual grasp verification, return/bin alignment, shutdown guards and trial recording. Original image/template assets are retained.

This release does not bypass rejected shelf-clearance plans or claim physical delivery confirmation without a bin sensor. The underlying controller/safety/communication fault and hardware calibration require robot-side evidence.
