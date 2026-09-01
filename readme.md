# ERC 4To1MUX Solution
## Setting Up Solution
In a regular shell, clone the repo into the `src` folder of the `erc_sim_2026` folder

`
git clone https://github.com/mmoukayed/solution_4to1mux.git
`

Then back in the main `erc_sim_2026` folder, run:
```
./docker/up.sh --build
./docker/attach.sh
colcon build --symlink-install
source install/setup.bash
```

The start command for our program should be as mentioned in the guidelines:
```
ros2 launch solution_4to1mux solution.launch.py shelf_column_number:=2 book_colour:=red
```