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

#include "base_velocity_plugin.hpp"

#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include <pluginlib/class_list_macros.hpp>

namespace mujoco_ros2_control_plugins
{

namespace
{
std::string namespacedParamName(const rclcpp::Node::SharedPtr& node, const std::string& name)
{
  const std::string sub_ns = node->get_sub_namespace();
  return sub_ns.empty() ? name : ("mujoco_plugins." + sub_ns + "." + name);
}

// Declares `name` with `default_value` only if it hasn't already been declared
// (e.g. by a test fixture), matching the pattern used by ExternalWrenchPlugin.
template <typename T>
T declareOrGetParameter(const rclcpp::Node::SharedPtr& node, const std::string& name, const T& default_value)
{
  const std::string full_name = namespacedParamName(node, name);
  if (!node->has_parameter(full_name))
  {
    node->declare_parameter(full_name, default_value);
  }
  return node->get_parameter(full_name).get_value<T>();
}
}  // namespace

bool BaseVelocityPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* /*data*/)
{
  node_ = node;
  logger_ = node_->get_logger().get_child(node->get_sub_namespace());
  model_ = model;

  // "body" has no sane default -- it must name the base's MJCF body.
  const std::string body_name = declareOrGetParameter<std::string>(node_, "body", "");
  if (body_name.empty())
  {
    RCLCPP_ERROR(logger_, "BaseVelocityPlugin requires the 'body' parameter (MJCF body name).");
    return false;
  }

  body_id_ = mj_name2id(model_, mjOBJ_BODY, body_name.c_str());
  if (body_id_ < 0)
  {
    RCLCPP_ERROR(logger_, "Body '%s' not found in MuJoCo model.", body_name.c_str());
    return false;
  }

  // A free joint's qvel is what the velocity override writes into -- without one, there is
  // nothing for this plugin to drive, so this is a hard failure rather than a warning.
  for (int j = 0; j < model_->njnt; ++j)
  {
    if (model_->jnt_bodyid[j] == body_id_ && model_->jnt_type[j] == mjJNT_FREE)
    {
      qvel_adr_ = model_->jnt_dofadr[j];
      qpos_adr_ = model_->jnt_qposadr[j];
      break;
    }
  }
  if (qvel_adr_ < 0)
  {
    RCLCPP_ERROR(logger_, "Body '%s' has no free joint; BaseVelocityPlugin cannot drive it.", body_name.c_str());
    return false;
  }

  const std::string cmd_vel_topic = declareOrGetParameter<std::string>(node_, "cmd_vel_topic", "cmd_vel");
  const bool use_stamped_twist = declareOrGetParameter<bool>(node_, "use_stamped_twist", false);
  max_linear_velocity_ = declareOrGetParameter<double>(node_, "max_linear_velocity", max_linear_velocity_);
  max_yaw_rate_ = declareOrGetParameter<double>(node_, "max_yaw_rate", max_yaw_rate_);
  const double cmd_timeout_sec = declareOrGetParameter<double>(node_, "cmd_timeout", 0.5);
  cmd_timeout_ = rclcpp::Duration::from_seconds(cmd_timeout_sec);
  // Defaults off: this is a local addition to a shared fork, so a model that
  // does not opt in behaves exactly as upstream does.
  hold_pose_on_idle_ = declareOrGetParameter<bool>(node_, "hold_pose_on_idle", false);

  const std::string drive_mode = declareOrGetParameter<std::string>(node_, "drive_mode", "kinematic");
  if (drive_mode != "kinematic" && drive_mode != "traction")
  {
    RCLCPP_ERROR(logger_, "drive_mode must be 'kinematic' or 'traction', got '%s'.", drive_mode.c_str());
    return false;
  }
  traction_mode_ = (drive_mode == "traction");
  if (traction_mode_)
  {
    // Order matters: front-left, front-right, rear-left, rear-right, matching the
    // mecanum kinematics below.
    const std::vector<std::string> wheel_joints =
        declareOrGetParameter<std::vector<std::string>>(node_, "wheel_joints", {});
    if (wheel_joints.size() != wheel_dof_adr_.size())
    {
      RCLCPP_ERROR(logger_, "drive_mode 'traction' needs exactly %zu wheel_joints (front-left, front-right, "
                            "rear-left, rear-right), got %zu.",
                   wheel_dof_adr_.size(), wheel_joints.size());
      return false;
    }
    for (size_t i = 0; i < wheel_joints.size(); ++i)
    {
      const int jid = mj_name2id(model_, mjOBJ_JOINT, wheel_joints[i].c_str());
      if (jid < 0 || model_->jnt_type[jid] != mjJNT_HINGE)
      {
        RCLCPP_ERROR(logger_, "Wheel joint '%s' is not a hinge joint in the MuJoCo model.",
                     wheel_joints[i].c_str());
        return false;
      }
      wheel_dof_adr_[i] = model_->jnt_dofadr[jid];
    }
    wheel_radius_ = declareOrGetParameter<double>(node_, "wheel_radius", 0.0);
    wheel_lever_ = declareOrGetParameter<double>(node_, "wheel_lever", 0.0);
    if (wheel_radius_ <= 0.0 || wheel_lever_ <= 0.0)
    {
      RCLCPP_ERROR(logger_, "drive_mode 'traction' needs positive wheel_radius and wheel_lever "
                            "(got %.4f, %.4f).",
                   wheel_radius_, wheel_lever_);
      return false;
    }
    settling_time_ = declareOrGetParameter<double>(node_, "settling_time", settling_time_);
    max_force_ = declareOrGetParameter<double>(node_, "max_force", max_force_);
    max_torque_ = declareOrGetParameter<double>(node_, "max_torque", max_torque_);
    max_hold_offset_ = declareOrGetParameter<double>(node_, "max_hold_offset", max_hold_offset_);
    // ERC_SLIP_ALONG / _SIDE / _TURN override the parameters, for a test run
    // without editing the model (as ERC_DEPTH_K does for the camera).
    const char* slip_names[3] = { "slip_along", "slip_side", "slip_turn" };
    const char* slip_envs[3] = { "ERC_SLIP_ALONG", "ERC_SLIP_SIDE", "ERC_SLIP_TURN" };
    for (int k = 0; k < 3; ++k)
    {
      slip_[k] = declareOrGetParameter<double>(node_, slip_names[k], 0.0);
      if (const char* env = std::getenv(slip_envs[k]); env && *env)
      {
        char* end = nullptr;
        slip_[k] = std::strtod(env, &end);
        if (end == env || *end != '\0')
        {
          RCLCPP_ERROR(logger_, "%s='%s' is not a number.", slip_envs[k], env);
          return false;
        }
      }
      if (slip_[k] < 0.0 || slip_[k] >= 1.0)
      {
        RCLCPP_ERROR(logger_, "%s must be in [0, 1), got %.3f.", slip_names[k], slip_[k]);
        return false;
      }
    }
    // settling_time divides into the servo gains, so zero produces infinite force
    // and a negative one drives the base away from its target instead of towards
    // it. A negative bound would make mju_clip's min exceed its max, which pins
    // the offset at the bound and holds the servo at full force forever.
    if (settling_time_ <= 0.0 || max_force_ <= 0.0 || max_torque_ <= 0.0 || max_hold_offset_ < 0.0)
    {
      RCLCPP_ERROR(logger_, "drive_mode 'traction' needs settling_time > 0, max_force > 0, "
                            "max_torque > 0 and max_hold_offset >= 0 (got %.4f, %.1f, %.1f, %.4f).",
                   settling_time_, max_force_, max_torque_, max_hold_offset_);
      return false;
    }
  }

  // Traction mode takes no command: the wheels are commanded through ros2_control
  // and the base only follows them, so there is nothing for a subscription to feed.
  if (traction_mode_)
  {
    // nothing to subscribe to
  }
  else if (use_stamped_twist)
  {
    twist_stamped_sub_ = node_->create_subscription<geometry_msgs::msg::TwistStamped>(
        cmd_vel_topic, rclcpp::SystemDefaultsQoS(),
        [this](const geometry_msgs::msg::TwistStamped& msg) { twistStampedCallback(msg); });
  }
  else
  {
    twist_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(cmd_vel_topic, rclcpp::SystemDefaultsQoS(),
                                                                       [this](const geometry_msgs::msg::Twist& msg) {
                                                                         twistCallback(msg);
                                                                       });
  }

  if (traction_mode_)
  {
    RCLCPP_INFO(logger_,
                "BaseVelocityPlugin initialised for body '%s' in traction mode: driven by its own "
                "wheels, radius=%.4f m lever=%.4f m settling_time=%.3f s max_force=%.1f N "
                "max_torque=%.1f Nm slip along/side/turn=%.3f/%.3f/%.3f",
                body_name.c_str(), wheel_radius_, wheel_lever_, settling_time_, max_force_, max_torque_,
                slip_[0], slip_[1], slip_[2]);
  }
  else
  {
    RCLCPP_INFO(logger_,
                "BaseVelocityPlugin initialised for body '%s' in kinematic mode. Listening for %s on "
                "'%s'. max_linear_velocity=%.2f max_yaw_rate=%.2f cmd_timeout=%.2fs",
                body_name.c_str(), use_stamped_twist ? "TwistStamped" : "Twist", cmd_vel_topic.c_str(),
                max_linear_velocity_, max_yaw_rate_, cmd_timeout_sec);
  }

  return true;
}

void BaseVelocityPlugin::twistCallback(const geometry_msgs::msg::Twist& msg)
{
  storeCommand(msg.linear.x, msg.linear.y, msg.angular.z);
}

void BaseVelocityPlugin::twistStampedCallback(const geometry_msgs::msg::TwistStamped& msg)
{
  storeCommand(msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z);
}

void BaseVelocityPlugin::storeCommand(double vx, double vy, double wz)
{
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  latest_cmd_ = { vx, vy, wz, node_->get_clock()->now() };
}

void BaseVelocityPlugin::pre_step(mjData* data)
{
  if (traction_mode_)
  {
    driveTraction(data);
    return;
  }
  driveKinematic(data);
}

// Servo the base towards the twist its own wheels are turning at, with a force cap.
//
// The wheels are commanded through ros2_control like any other joint, so what they are
// actually doing already accounts for their own torque limit and for whatever the
// controller asked of them. Converting their measured speeds back to a body twist and
// then chasing that twist with a limited force is the whole traction model: the base
// cannot exceed what the wheels could push, so an obstacle stalls it, and the wheels
// keep turning against it exactly as they would on the real robot.
void BaseVelocityPlugin::driveTraction(mjData* data)
{
  // An external teleport (reset_world, set_free_joint_state) is not something the
  // wheels did, so the stored offset it leaves behind describes a position the base
  // no longer has. Drop it, or the servo spends the next moments pushing towards
  // where the robot used to be. The base itself never moves this far in one step:
  // 5 mm at the competition's 2 ms timestep would be 2.5 m/s.
  for (int k = 0; k < 3; ++k)
  {
    if (have_last_qpos_ && std::abs(data->qpos[qpos_adr_ + k] - last_qpos_[k]) > 0.005)
    {
      hold_offset_[0] = hold_offset_[1] = hold_offset_[2] = 0.0;
      break;
    }
  }
  for (int k = 0; k < 3; ++k)
  {
    last_qpos_[k] = data->qpos[qpos_adr_ + k];
  }
  have_last_qpos_ = true;

  // Mecanum forward kinematics, the transpose of the inverse kinematics
  // mecanum_drive_controller uses, with wheels ordered FL, FR, RL, RR.
  const double fl = data->qvel[wheel_dof_adr_[0]];
  const double fr = data->qvel[wheel_dof_adr_[1]];
  const double rl = data->qvel[wheel_dof_adr_[2]];
  const double rr = data->qvel[wheel_dof_adr_[3]];
  //
  // Slip takes its share off the twist the base chases, not off the wheels: the
  // wheels (and so the odometry that integrates them) turn as commanded while
  // the base makes less of it, which is what a slipping wheel does.
  const double vx = (1.0 - slip_[0]) * (fl + fr + rl + rr) * wheel_radius_ / 4.0;
  const double vy = (1.0 - slip_[1]) * (-fl + fr + rl - rr) * wheel_radius_ / 4.0;
  const double wz = (1.0 - slip_[2]) * (-fl + fr - rl + rr) * wheel_radius_ / (4.0 * wheel_lever_);

  // A free joint's linear qvel is world-frame, so rotate the body-frame twist out;
  // its angular qvel is already body-frame, which is what wz is.
  const mjtNum* xmat = data->xmat + body_id_ * 9;
  const double vx_world = xmat[0] * vx + xmat[1] * vy;
  const double vy_world = xmat[3] * vx + xmat[4] * vy;

  // Gain = effective inertia / settling time. mjData::qM is stored sparsely with each
  // row's diagonal entry first, so qM[dof_Madr[i]] is M(i,i) -- the base's own mass and
  // yaw inertia including everything mounted on it, refreshed as the arms move.
  const double m_x = data->qM[model_->dof_Madr[qvel_adr_ + 0]];
  const double m_y = data->qM[model_->dof_Madr[qvel_adr_ + 1]];
  const double i_z = data->qM[model_->dof_Madr[qvel_adr_ + 5]];

  // A velocity servo alone leaks position: a disturbance the wheels never saw --
  // an arm accelerating -- pushes the base until the servo's damping cancels it,
  // and the displacement is never recovered. Measured, that let the idle base
  // wander 77 mm and 40 degrees over one arm cycle, which is exactly the drift
  // the old idle-pose latch existed to hide. Integrate the velocity error into a
  // bounded position offset and add a spring on it, so the base is held where its
  // wheels put it. The bound is what keeps this from winding up: against a real
  // obstacle the offset saturates, the force stays at its cap, and what is stored
  // up is limited to max_hold_offset. That much does discharge when the command
  // stops -- measured, driving into a wall and then commanding zero pushed a
  // further 21 mm over about a second before settling -- so keep the bound small.
  const double dt = model_->opt.timestep;
  const double ex = vx_world - data->qvel[qvel_adr_ + 0];
  const double ey = vy_world - data->qvel[qvel_adr_ + 1];
  const double ez = wz - data->qvel[qvel_adr_ + 5];
  hold_offset_[0] = mju_clip(hold_offset_[0] + ex * dt, -max_hold_offset_, max_hold_offset_);
  hold_offset_[1] = mju_clip(hold_offset_[1] + ey * dt, -max_hold_offset_, max_hold_offset_);
  hold_offset_[2] = mju_clip(hold_offset_[2] + ez * dt, -max_hold_offset_, max_hold_offset_);

  // Critically damped: gain kd = M/tau on the velocity error, kp = M/tau^2 on the
  // offset.
  const double kd = 1.0 / settling_time_;
  const double kp = kd / settling_time_;
  const double fx = m_x * (kd * ex + kp * hold_offset_[0]);
  const double fy = m_y * (kd * ey + kp * hold_offset_[1]);
  const double tz = i_z * (kd * ez + kp * hold_offset_[2]);

  // Cap the planar force as a vector so a diagonal push is not stronger than a
  // straight one.
  double scale = 1.0;
  const double planar = std::hypot(fx, fy);
  if (planar > max_force_)
  {
    scale = max_force_ / planar;
  }
  data->qfrc_applied[qvel_adr_ + 0] = fx * scale;
  data->qfrc_applied[qvel_adr_ + 1] = fy * scale;
  data->qfrc_applied[qvel_adr_ + 5] = mju_clip(tz, -max_torque_, max_torque_);
}

void BaseVelocityPlugin::driveKinematic(mjData* data)
{
  // Step 1 - refresh the cached command from the subscription callback without
  // blocking the real-time thread; if the lock is contended, keep using the last
  // successfully cached values.
  if (cmd_mutex_.try_lock())
  {
    cached_cmd_ = latest_cmd_;
    cmd_mutex_.unlock();
  }

  // Step 2 - a stale (or never-received) command is treated as a zero-velocity
  // command, i.e. the base is commanded to stop rather than coast on the last override.
  double vx_cmd = 0.0, vy_cmd = 0.0, wz_cmd = 0.0;
  const rclcpp::Duration age = node_->get_clock()->now() - cached_cmd_.time;
  if (age <= cmd_timeout_)
  {
    vx_cmd = cached_cmd_.vx;
    vy_cmd = cached_cmd_.vy;
    wz_cmd = cached_cmd_.wz;
    hold_pose_ = false;
  }
  else if (hold_pose_on_idle_)
  {
    // Zeroing qvel alone does not stop the base: reaction torques from the
    // articulation (an arm or gripper accelerating) integrate into the free
    // joint's pose within each step, and the base slowly wanders. Latch the
    // full free-joint pose when commands go stale and restore it every step
    // until a fresh command arrives.
    // An external teleport (reset_world, set_free_joint_state) moves the pose
    // far more in one step than contact drift ever can; yield to it by
    // re-capturing instead of restoring, so the latch never fights a reset.
    // Orientation counts as a teleport too: a reset that only re-aims the base
    // moves qpos[3..6] and barely moves qpos[0..2], and cmd_vel is always
    // stale right after a reset, so a position-only test would silently
    // revert it on the very next step.
    bool teleported = false;
    for (int k = 0; k < 3 && hold_pose_; ++k)
    {
      if (std::abs(data->qpos[qpos_adr_ + k] - held_qpos_[k]) > 0.005)
      {
        teleported = true;
      }
    }
    for (int k = 3; k < 7 && hold_pose_; ++k)
    {
      if (std::abs(data->qpos[qpos_adr_ + k] - held_qpos_[k]) > 0.002)
      {
        teleported = true;
      }
    }
    if (!hold_pose_ || teleported)
    {
      for (int k = 0; k < 7; ++k)
      {
        held_qpos_[k] = data->qpos[qpos_adr_ + k];
      }
      hold_pose_ = true;
    }
    for (int k = 0; k < 7; ++k)
    {
      data->qpos[qpos_adr_ + k] = held_qpos_[k];
    }
    for (int k = 0; k < 6; ++k)
    {
      data->qvel[qvel_adr_ + k] = 0.0;
    }
    return;
  }

  // Step 3 - clamp to the configured limits, preserving direction for the planar speed.
  const double linear_speed = std::hypot(vx_cmd, vy_cmd);
  if (linear_speed > max_linear_velocity_)
  {
    const double scale = max_linear_velocity_ / linear_speed;
    vx_cmd *= scale;
    vy_cmd *= scale;
  }
  wz_cmd = mju_clip(wz_cmd, -max_yaw_rate_, max_yaw_rate_);

  // Step 4 - rotate the commanded body-frame linear velocity into the world frame (a free
  // joint's linear qvel is world-frame)
  const mjtNum* xmat = data->xmat + body_id_ * 9;
  data->qvel[qvel_adr_ + 0] = xmat[0] * vx_cmd + xmat[1] * vy_cmd;
  data->qvel[qvel_adr_ + 1] = xmat[3] * vx_cmd + xmat[4] * vy_cmd;
  data->qvel[qvel_adr_ + 5] = wz_cmd;
}

void BaseVelocityPlugin::cleanup()
{
  RCLCPP_INFO(logger_, "BaseVelocityPlugin cleanup.");
  twist_sub_.reset();
  twist_stamped_sub_.reset();
  node_.reset();
}

}  // namespace mujoco_ros2_control_plugins

// Export the plugin
PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_plugins::BaseVelocityPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
