# sensors

Sensor-level processing for the **ERC 2026** head RGB-D camera. Rebuilds the
depth-camera point cloud with an **Intel RealSense D435-like noise model**, so
the simulated depth stream behaves like a real sensor's rather than like a
perfect ray-traced surface.

## Node

- **`depth_to_cloud`** — the simulator publishes a depth image but no point
  cloud, so RViz's DepthCloud shows nothing. This node rebuilds the cloud the
  standard way (back-project the float32 depth image through `camera_info`,
  optionally colour it) and publishes a normal `PointCloud2`. It also applies a
  **RealSense-D435 noise model** so the cloud looks like a real sensor's.

### RealSense realism

The head camera is modelled on the **Intel RealSense D435 depth module**
(640×360, 87°×56.19° FoV — the resolution and horizontal FoV are defined in
`erc_bringup/scripts/generate_mujoco_urdf.py`). `depth_to_cloud` then models the
sensor's error:

- **Range-dependent noise** — depth → stereo disparity `d = f·B/Z`, add Gaussian
  disparity noise, back to `Z`. Error grows ~`Z²` (far surfaces get noisier, like
  a real D435), tuned by `disparity_sigma` (px) and `baseline` (m).
- **Quantisation shells** — disparity is snapped to `1/subpixel_bits` steps,
  reproducing the tell-tale depth banding.
- **Edge holes / dropouts** — points at depth discontinuities (`edge_hole_thresh`)
  and a small random fraction (`random_drop`) are removed.

Set `realsense_noise:=false` (or the individual params) for the clean cloud.

**Colouring.** A real depth camera outputs geometry, not RGB, so the cloud is
coloured by **distance** (turbo over `[color_min, color_max]` m) by default — the
RealSense-Viewer look. `color_source:=rgb` fuses the aligned colour image (the
RGBD product on a real sensor's `/depth/color/points`); `color_source:=none`
publishes plain XYZ.

## Topics

These are the node's parameter defaults, and they match what the simulator
publishes.

### Subscribed
| Topic | Type | Notes |
|---|---|---|
| `/head_front_camera/head_front_camera/depth/image_rect_raw` | `sensor_msgs/Image` (32FC1) | depth (param `depth_topic`) |
| `/head_front_camera/head_front_camera/depth/camera_info` | `sensor_msgs/CameraInfo` | depth intrinsics (param `info_topic`) |
| `/head_front_camera/head_front_camera/color/image_raw` | `sensor_msgs/Image` (rgb8) | only used when `color_source:=rgb` (param `color_topic`) |

### Published
| Topic | Type | Description |
|---|---|---|
| `/head_front_camera/head_front_camera/depth/color/points` | `sensor_msgs/PointCloud2` | **full depth-camera cloud** with the RealSense-D435 noise model, coloured by distance by default. |
| `/head_front_camera/depth/fov` | `visualization_msgs/Marker` (LINE_LIST) | cyan wireframe **frustum of the depth camera's field of view** (to `fov_range` m). |

## Build & run

`depth_to_cloud` is **brought up automatically by the sim** —
`erc_bringup/launch/mujoco_simulation.launch.py` includes this package's launch
file, so it starts with the simulator and owns the points topic. You normally do
**not** launch it by hand (a second instance means two publishers on that topic).
Just build it once so the sim can find it:

```bash
colcon build --packages-select sensors --symlink-install && source install/setup.bash
```

Turn it off, or tune it, through the sim launch:

```bash
ros2 launch erc_bringup mujoco_simulation.launch.py depth_cloud:=false
```

Standalone (only if the sim was started with `depth_cloud:=false`):

```bash
ros2 launch sensors depth_to_cloud.launch.py color_source:=rgb      # fused colour
ros2 launch sensors depth_to_cloud.launch.py realsense_noise:=false # clean cloud
```

If RViz shows the cloud flat white, set that display's **Color Transformer** to
**RGB8**. Always run RViz with `use_sim_time`.

## Notes

- **Frame.** The cloud is published in whatever frame the incoming depth
  `camera_info` carries. The simulator stamps both colour and depth with
  `head_front_camera_color_optical_frame` (Z forward) — colour and depth are
  rendered from one camera and share intrinsics, so they are already aligned.
- **Range.** `min_range`/`max_range` default to 0.2–8 m, matching the D435's
  usable range; depths outside that are dropped. This is a property of this
  node, not of the simulated camera, whose far plane is much further out.
