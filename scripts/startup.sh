#!/bin/bash
selcommand=$1
shift 1
cd /opt/erc_ws
if [ "$1" = "-s" ]; then
echo "Skipping build!"
shift 1
elif [ "$1" = "-sn" ]; then
echo "Skipping build!"
else
colcon build --symlink-install
fi
if [[ "$1" = "-n" || "$1" = "-sn" ]]; then
echo "Skipping Tiago DDS install!"
shift 1
else
sudo apt update
git clone https://github.com/dfl-rlab/tiago_dds_connect.git
sudo apt install -y ros-humble-rmw-cyclonedds-cpp iproute2
fi
source tiago_dds_connect/tiago_env.sh
source install/setup.bash
while [ "$(ros2 topic list)" = *"/tf"* ]; do
echo "Waiting for robot to connect..."
done
if [ "$selcommand" = "simulation" ]; then
ros2 launch erc_bringup simulation_round_1.launch.py
elif [ "$selcommand" = "simulation2" ]; then
ros2 launch erc_bringup simulation_round_2.launch.py
elif [ "$selcommand" = "solution" ]; then
echo $@
ros2 launch solution_4to1mux solution.launch.py $@
fi
