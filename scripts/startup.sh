#!/bin/bash
selcommand=$1
shift 1
cd /opt/erc_ws
if [ "$1" = "-s" ]; then
echo "Skipping build!"
shift 1
sudo apt update
git clone https://github.com/dfl-rlab/tiago_dds_connect.git
sudo apt install -y ros-humble-rmw-cyclonedds-cpp iproute2
elif [ "$1" = "-sn" ]; then
echo "Skipping build and Tiago DDS install!"
shift 1
elif [ "$1" = "-n" ]; then
echo "Tiago DDS install!"
colcon build --symlink-install
shift 1
else
colcon build --symlink-install
sudo apt update
git clone https://github.com/dfl-rlab/tiago_dds_connect.git
sudo apt install -y ros-humble-rmw-cyclonedds-cpp iproute2
fi

if [ "$2" = "ros" ]; then
source install/setup.bash
shift 1
elif [ "$2" = "tiago" ]; then
source tiago_dds_connect/tiago_env.sh
shift 1
else
source install/setup.bash
source tiago_dds_connect/tiago_env.sh
shift 1
fi

if [ "$selcommand" = "simulation" ]; then
ros2 launch erc_bringup simulation_round_1.launch.py
elif [ "$selcommand" = "simulation2" ]; then
ros2 launch erc_bringup simulation_round_2.launch.py
elif [ "$selcommand" = "solution_custom" ]; then
echo $@
ros2 launch solution_4to1mux solution.launch.py $@
elif [ "$selcommand" = "solution_real" ]; then
ros2 launch solution_4to1mux solution.launch.py robot_profile:=real shelf_column_number:=$1 book_color:=$2
elif [ "$selcommand" = "solution_sim" ]; then
echo $@
ros2 launch solution_4to1mux solution.launch.py robot_profile:=simulation shelf_column_number:=$1 book_color:=$2

fi
