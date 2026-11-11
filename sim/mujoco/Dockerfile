# The ERC 2026 arena in MuJoCo: CPU only, no GPU or display needed.
# Built by sim/sim.sh build (context: this directory). The simulator is compiled
# into the image; the solution package is mounted and built when the container starts.
FROM osrf/ros:humble-desktop
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' /etc/apt/sources.list

# Build tools, the simulator's ROS dependencies, and the solution's own
# (scipy, OpenCV via cv_bridge, tesseract for book_text:=...).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions python3-pip python3-venv curl ffmpeg \
    ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-controller-manager \
    ros-humble-joint-trajectory-controller ros-humble-position-controllers \
    ros-humble-velocity-controllers ros-humble-effort-controllers \
    ros-humble-joint-state-publisher ros-humble-robot-state-publisher ros-humble-xacro \
    ros-humble-realsense2-description ros-humble-topic-tools ros-humble-rmw-cyclonedds-cpp \
    ros-humble-ros-gz-interfaces ros-humble-ros2-control-cmake ros-humble-urdfdom-py \
    ros-humble-backward-ros python3-pykdl python3-numpy python3-scipy python3-opencv \
    tesseract-ocr libglfw3-dev libegl-dev libgl1-mesa-dev libegl-mesa0 \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo 3.4.0 (mujoco_vendor 0.0.9): the version the vendored mujoco_ros2_control is
# written against. The ROS apt repo keeps only the latest build, which no longer compiles
# with it, so the exact .deb comes from the ROS snapshot, checked by hash.
ARG MUJOCO_VENDOR_DEB=ros-humble-mujoco-vendor_0.0.9-1jammy.20260722.133956_amd64.deb
ARG MUJOCO_VENDOR_SHA256=39cbb8b66183732074c1c21bea24cf300390dd21e13cacbdef2eea8b2c57af05
RUN curl -fsSL -o /tmp/$MUJOCO_VENDOR_DEB \
        http://snapshots.ros.org/humble/2026-08-07/ubuntu/pool/main/r/ros-humble-mujoco-vendor/$MUJOCO_VENDOR_DEB \
    && echo "$MUJOCO_VENDOR_SHA256  /tmp/$MUJOCO_VENDOR_DEB" | sha256sum -c - \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/$MUJOCO_VENDOR_DEB \
    && apt-mark hold ros-humble-mujoco-vendor \
    && rm -f /tmp/$MUJOCO_VENDOR_DEB && rm -rf /var/lib/apt/lists/*

# The URDF-to-MJCF converter's venv, built now so no network is needed later.
COPY src/mujoco_ros2_control/mujoco_ros2_control/scripts/requirements.txt /tmp/mjcf-requirements.txt
RUN python3 -m venv --system-site-packages /root/.ros/ros2_control/.venv \
    && source /root/.ros/ros2_control/.venv/bin/activate \
    && pip install --no-cache-dir --disable-pip-version-check -r /tmp/mjcf-requirements.txt pycollada \
    && rm /tmp/mjcf-requirements.txt

# The simulator itself, compiled once here.
COPY src /opt/erc_ws/src
RUN source /opt/ros/humble/setup.bash && cd /opt/erc_ws \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf log

ENV PYTHONDONTWRITEBYTECODE=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp MUJOCO_GL=egl
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /opt/erc_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
