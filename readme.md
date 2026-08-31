# Setting Up Solution
in a regular shell, clone the repo into the `src` folder of the `erc_sim_2026` folder

```git clone https://github.com/mmoukayed/solution_4to1mux.git```

Then run 
```
../docker/attach.sh
source install/setup.bash
```

The start command for our program should be as mentioned in the guidelines:
```
ros2 launch solution_4to1mux solution.launch.py shelf_column_number:=2 book_colour:=red
```