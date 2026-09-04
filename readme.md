# ERC 4To1MUX Solution
## Setting Up Solution
Run these commands to first setup the ROS 2 environment:
```
cd ~
git clone https://github.com/dfl-rlab/erc_sim_2026
cd erc_sim_2026
./docker/up.sh --build
```
Then to setup the solution program, run these commands:
```
cd src
git clone https://github.com/mmoukayed/solution_4to1mux
cd ..
./docker/attach.sh
colcon build --symlink-install
source install/setup.bash
ros2 launch erc_bringup solution.launch.py
```
These commands will download and build the whole project and then run the simulation

> [!NOTE]
> If GUIs arent supported on your OS, use the `headless:=true` argument after the `ros2 launch` command of the simulator not the solution

The start command for our program should be as mentioned in the guidelines:
```
ros2 launch solution_4to1mux solution.launch.py shelf_column_number:=2 book_colour:=red
```