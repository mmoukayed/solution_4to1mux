#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# The workspace overlay, once it has been built.
if [ -f /opt/erc_ws/install/setup.bash ]; then
    source /opt/erc_ws/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONDONTWRITEBYTECODE=1

# Defaults only. Setting ROS_LOCALHOST_ONLY=1 keeps DDS off the LAN, which
# matters on a busy network where the participant count alone can stall the
# simulation - but it also breaks a second machine, a host-side RViz outside
# this container's netns, and a real-robot bridge, so it stays opt-in and any
# value the user already exported wins.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-<CycloneDDS><Domain><Discovery><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>}"

exec "$@"
