# ERC 4To1MUX Solution
## Setting Up Solution (First Run)
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
```
Next once you are in the docker shell, these commands will setup the environment to run ros2 commands and the gazebo simulator:
```
colcon build --symlink-install
source install/setup.bash
ros2 launch erc_bringup simulation.launch.py
```

> [!NOTE]
> If GUIs arent supported on your OS, use the `headless:=true` argument after the `ros2 launch` command of the simulator not the solution

## Starting Up Solution
To startup the docker container and code or if you want to open more than one terminal, you don't need to build from scratch again, simply run these commands to start:
> [!WARNING]
> If your docker container is already running and you want to just open more terminals, don't run `./docker/up.sh`. Just the `attach.sh` command to connect to the shell.
> 
> Additionally, if you are experiencing errors and issues with the docker container or ros2 environment and re-running `colcon build` isn't working, then add the `--build` flag to the end of `up.sh`.

```
cd ~/erc_sim_2026
./docker/up.sh
./docker/attach.sh
```
Then similar to the previous section, once inside the docker shell:
```
colcon build --symlink-install
source install/setup.bash
```
The start command for our program should be as mentioned in the guidelines:
```
ros2 launch solution_4to1mux solution.launch.py shelf_column_number:=2 book_colour:=red
```
