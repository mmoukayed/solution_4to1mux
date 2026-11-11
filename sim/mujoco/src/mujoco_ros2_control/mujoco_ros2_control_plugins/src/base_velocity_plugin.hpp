// Copyright 2026 PAL Robotics S.L.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_ROS2_CONTROL_PLUGINS__BASE_VELOCITY_PLUGIN_HPP_
#define MUJOCO_ROS2_CONTROL_PLUGINS__BASE_VELOCITY_PLUGIN_HPP_

#include <array>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace mujoco_ros2_control_plugins
{

/**
 * @brief Drives a mobile/floating-base robot's planar DOFs, either kinematically from a
 *        commanded body velocity or dynamically from the rotation of its own wheels.
 *
 * Two drive modes, selected by the "drive_mode" parameter:
 *
 * "kinematic" (default) writes a hard override of the base's free-joint velocity into
 * data->qvel every cycle. See below.
 *
 * "traction" instead reads the four mecanum wheel joints' measured velocities, converts
 * them to a body twist with the mecanum forward kinematics, and drives the base towards
 * that twist with a force-limited velocity servo written into data->qfrc_applied. The
 * base is then moved by whatever the wheels are actually doing rather than by a
 * command, so it can be resisted: the servo force is capped at the traction the wheels
 * could develop (wheel count x wheel effort limit / wheel radius), so driving into an
 * obstacle stalls the base while the wheels keep turning -- which is what makes wheel
 * odometry drift. Nothing external is overridden, so contacts and reaction torques act
 * on the base normally, and a zero wheel speed actively damps the base to rest rather
 * than needing a pose latch. Requires "wheel_joints", "wheel_radius" and "wheel_lever".
 *
 * The kinematic mode, kept as the default so existing models are unaffected:
 *
 * Wheel-terrain friction/slip modelling is often unreliable enough to make it a poor
 * foundation for testing navigation stacks. This plugin instead subscribes to a
 * cmd_vel-style topic and, every cycle, writes the (optionally clamped) commanded planar
 * velocity directly into the base body's free-joint qvel entries in data. There is no
 * gain to tune and no convergence delay: the measured body velocity on the driven DOFs is
 * exactly the commanded velocity on the very next simulation step.
 *
 * The trade-off is that the driven DOFs cannot be influenced by anything else -- not
 * contacts, not forces from other bodies rigidly mounted on the base (e.g. a swinging
 * arm). They are simply reasserted every cycle regardless of what the physics engine
 * computed in between.
 *
 * Only the planar DOFs are driven: world-frame linear x/y (converted from the commanded
 * body-frame vx/vy via the body's current orientation) and body-local yaw-rate about z.
 * Vertical linear velocity and local roll/pitch are left untouched every cycle, so gravity
 * and contacts continue to settle the base onto the ground normally.
 *
 * Configuration parameters (declared under "mujoco_plugins.<instance_name>.")
 * -------------------------------------------------------------------------------
 *   body                (string, required) - MJCF body name of the base. Must have a
 *                        <freejoint/>; init() fails otherwise, since there is no qvel to
 *                        override without one.
 *   drive_mode          (string, default "kinematic") - "kinematic" or "traction".
 *   cmd_vel_topic       (string, default "cmd_vel")   - command topic name.
 *   use_stamped_twist   (bool,   default false)       - subscribe to
 *                        geometry_msgs/TwistStamped instead of geometry_msgs/Twist.
 *   max_linear_velocity (double, default +inf) - clamps the commanded planar speed
 *                        sqrt(vx^2+vy^2) [m/s]; direction is preserved when scaled down.
 *   max_yaw_rate        (double, default +inf) - clamps the commanded yaw-rate [rad/s].
 *   cmd_timeout         (double, default 0.5)  - seconds since the last command after
 *                        which it is treated as zero (safety stop).
 *
 * Traction-mode parameters (ignored in kinematic mode)
 * -------------------------------------------------------------------------------
 *   wheel_joints        (string[], required) - the four mecanum wheel joint names, in
 *                        the order front-left, front-right, rear-left, rear-right.
 *   wheel_radius        (double, required)   - wheel radius [m].
 *   wheel_lever         (double, required)   - lx + ly, the mecanum yaw lever arm [m]:
 *                        the same quantity mecanum_drive_controller calls
 *                        kinematics.sum_of_robot_center_projection_on_X_Y_axis.
 *   settling_time       (double, default 0.02) - servo time constant [s]. The
 *                        servo gain is the base's own effective inertia divided by this,
 *                        read fresh from the mass matrix each step, so it stays critically
 *                        tuned as the torso lifts and the arms move.
 *   max_force           (double, default +inf) - planar force cap [N].
 *   max_torque          (double, default +inf) - yaw torque cap [Nm].
 *   max_hold_offset     (double, default 0.02) - bound on the integrated position
 *                        error the servo will try to recover [m and rad]. Small
 *                        enough that the force is already at its cap well inside
 *                        it, so it holds station without winding up.
 */
class BaseVelocityPlugin : public MuJoCoROS2ControlPluginBase
{
public:
  BaseVelocityPlugin() = default;
  ~BaseVelocityPlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override;
  void pre_step(mjData* data) override;
  void cleanup() override;

private:
  /// Kinematic mode: overwrite the free joint's planar qvel with the command.
  void driveKinematic(mjData* data);
  /// Traction mode: servo the base towards the twist its wheels are turning at.
  void driveTraction(mjData* data);

  void twistCallback(const geometry_msgs::msg::Twist& msg);
  void twistStampedCallback(const geometry_msgs::msg::TwistStamped& msg);
  void storeCommand(double vx, double vy, double wz);

  // ROS interfaces
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("BaseVelocityPlugin") };
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_stamped_sub_;

  // Model/body/joint lookup
  const mjModel* model_{ nullptr };
  int body_id_{ -1 };
  int qvel_adr_{ -1 };
  int qpos_adr_{ -1 };

  // Traction mode
  bool traction_mode_{ false };
  std::array<int, 4> wheel_dof_adr_{ { -1, -1, -1, -1 } };
  double wheel_radius_{ 0.0 };
  double wheel_lever_{ 0.0 };
  double settling_time_{ 0.02 };
  double max_force_{ std::numeric_limits<double>::infinity() };
  double max_torque_{ std::numeric_limits<double>::infinity() };
  // Bounded integral of the velocity error (x, y in m, yaw in rad): the position
  // the base owes its wheels. See driveTraction.
  double hold_offset_[3]{ 0.0, 0.0, 0.0 };
  double max_hold_offset_{ 0.02 };
  // Wheel slip, the share of each wheel-derived motion (along, sideways, turn)
  // the base does not make. Odometry integrates the wheels, so it over-reports
  // by that much, as on a real floor. 0 = off. See driveTraction.
  double slip_[3]{ 0.0, 0.0, 0.0 };
  // Previous step's base position, so traction mode can tell a teleport from motion.
  bool have_last_qpos_{ false };
  mjtNum last_qpos_[3]{ 0, 0, 0 };

  // Idle pose latch (see pre_step): restore the free joint pose captured when
  // commands went stale, so articulation reaction torques cannot wander the base.
  bool hold_pose_on_idle_{ false };
  bool hold_pose_{ false };
  mjtNum held_qpos_[7]{ 0, 0, 0, 1, 0, 0, 0 };

  // Clamp parameters
  double max_linear_velocity_{ std::numeric_limits<double>::infinity() };
  double max_yaw_rate_{ std::numeric_limits<double>::infinity() };
  rclcpp::Duration cmd_timeout_{ 0, 0 };

  struct CommandState
  {
    double vx{ 0.0 };
    double vy{ 0.0 };
    double wz{ 0.0 };
    rclcpp::Time time{ 0, 0, RCL_ROS_TIME };
  };

  // Latest command, written by the subscription callback (ROS executor thread) under
  // cmd_mutex_. pre_step() (physics thread) copies it into cached_cmd_ via try_lock, so
  // it never blocks on the subscription callback.
  std::mutex cmd_mutex_;
  CommandState latest_cmd_;

  // Local copy of the command, only ever touched from pre_step() (physics thread).
  CommandState cached_cmd_;
};

}  // namespace mujoco_ros2_control_plugins

#endif  // MUJOCO_ROS2_CONTROL_PLUGINS__BASE_VELOCITY_PLUGIN_HPP_
