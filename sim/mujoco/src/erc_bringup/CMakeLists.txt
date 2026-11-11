cmake_minimum_required(VERSION 3.8)
project(erc_bringup)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})
# generate_mujoco_world.py imports mesh_collision.py from its own directory,
# so the two must be installed side by side.
install(PROGRAMS
  scripts/generate_urdf.py
  scripts/gripper_command_clamp.py
  scripts/generate_mujoco_urdf.py
  scripts/generate_mujoco_world.py
  scripts/mesh_collision.py
  scripts/odom_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Contact reporting ──
# The contact sensors are a competition interface, not a MuJoCo feature, so
# they live here rather than in the vendored mujoco_ros2_control_plugins: that
# keeps the fork's diff against PAL's tree empty and keeps ros_gz_interfaces
# out of a general-purpose package. pluginlib loads them all the same.
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ros_gz_interfaces REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)

add_library(erc_mujoco_plugins SHARED src/contact_sensor_plugin.cpp)
target_link_libraries(erc_mujoco_plugins PUBLIC
  mujoco_ros2_control_plugins::mujoco_ros2_control_plugins
  pluginlib::pluginlib
  realtime_tools::realtime_tools
  ${geometry_msgs_TARGETS}
  ${ros_gz_interfaces_TARGETS}
)
target_compile_definitions(erc_mujoco_plugins PRIVATE
  REALTIME_TOOLS_VERSION_MAJOR=${realtime_tools_VERSION_MAJOR}
)
install(TARGETS erc_mujoco_plugins LIBRARY DESTINATION lib)
pluginlib_export_plugin_description_file(mujoco_ros2_control_plugins erc_mujoco_plugins.xml)

ament_package()
