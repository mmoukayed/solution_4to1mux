// Copyright 2026 ERC Committee
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

#include "contact_sensor_plugin.hpp"

#include <algorithm>

#include <pluginlib/class_list_macros.hpp>
#include <ros_gz_interfaces/msg/contact.hpp>
#include <ros_gz_interfaces/msg/entity.hpp>

namespace erc_bringup
{

namespace
{
// Plugin parameters live under "mujoco_plugins.<instance_name>.", the same
// convention BaseVelocityPlugin uses, so two instances configure separately.
std::string namespacedParamName(const rclcpp::Node::SharedPtr& node, const std::string& name)
{
  const std::string sub_ns = node->get_sub_namespace();
  return sub_ns.empty() ? name : ("mujoco_plugins." + sub_ns + "." + name);
}

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

void toVector3(const mjtNum* src, geometry_msgs::msg::Vector3& out, double sign = 1.0)
{
  out.x = sign * src[0];
  out.y = sign * src[1];
  out.z = sign * src[2];
}
}  // namespace

std::size_t ContactSensorPlugin::collect_watched_geoms(const std::string& root_body)
{
  watched_.assign(model_->ngeom, 0);
  collision_names_.resize(model_->ngeom);
  body_names_.resize(model_->ngeom);

  for (int g = 0; g < model_->ngeom; ++g)
  {
    const int body_id = model_->geom_bodyid[g];
    const char* body_name = mj_id2name(model_, mjOBJ_BODY, body_id);
    body_names_[g] = body_name ? body_name : "";

    // The URDF-to-MJCF converter leaves every robot geom unnamed, so name it after its link
    // the way Gazebo does. That reproduces the collision names upstream's contact sensors
    // report ("<link>_collision"); arena geoms are named in the MJCF and keep their own name.
    const char* geom_name = mj_id2name(model_, mjOBJ_GEOM, g);
    collision_names_[g] = geom_name ? geom_name : body_names_[g] + "_collision";
  }

  const int root_id = mj_name2id(model_, mjOBJ_BODY, root_body.c_str());
  if (root_id == -1)
  {
    RCLCPP_ERROR(logger_, "Body '%s' is not in this MuJoCo model; this sensor will report nothing.",
                 root_body.c_str());
    return 0;
  }

  // MuJoCo orders bodies so that a parent always precedes its children, so one forward sweep
  // marks the whole subtree.
  std::vector<uint8_t> in_subtree(model_->nbody, 0);
  in_subtree[root_id] = 1;
  for (int b = root_id + 1; b < model_->nbody; ++b)
  {
    in_subtree[b] = in_subtree[model_->body_parentid[b]];
  }

  std::size_t count = 0;
  for (int g = 0; g < model_->ngeom; ++g)
  {
    watched_[g] = in_subtree[model_->geom_bodyid[g]];
    count += watched_[g];
  }
  return count;
}

bool ContactSensorPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* /*data*/)
{
  node_ = node;
  model_ = model;
  logger_ = node_->get_logger().get_child(node->get_sub_namespace());

  const std::string root_body = declareOrGetParameter<std::string>(node_, "body", "");
  if (root_body.empty())
  {
    RCLCPP_ERROR(logger_, "ContactSensorPlugin requires the 'body' parameter (MJCF body name).");
    return false;
  }

  const std::string topic = declareOrGetParameter<std::string>(node_, "topic", "contacts");
  const double publish_rate = declareOrGetParameter<double>(node_, "publish_rate", 30.0);
  if (publish_rate <= 0.0)
  {
    RCLCPP_ERROR(logger_, "publish_rate must be > 0, got %f.", publish_rate);
    return false;
  }

  const std::size_t watched = collect_watched_geoms(root_body);

  publish_period_ = rclcpp::Duration::from_seconds(1.0 / publish_rate);
  last_publish_time_ = node_->get_clock()->now();
  message_.header.frame_id = "world";

  publisher_raw_ = node_->create_publisher<Contacts>(topic, rclcpp::SystemDefaultsQoS());
  publisher_ = std::make_unique<realtime_tools::RealtimePublisher<Contacts>>(publisher_raw_);

  RCLCPP_INFO(logger_, "ContactSensorPlugin initialized: watching %zu geom%s under '%s', publishing to '%s' at %.1f Hz.",
              watched, watched == 1 ? "" : "s", root_body.c_str(), publisher_raw_->get_topic_name(), publish_rate);
  return true;
}

void ContactSensorPlugin::update(const mjModel* /*model*/, mjData* data)
{
  const rclcpp::Time now = node_->get_clock()->now();
  // Sim time only runs backwards if the simulator is restarted under us; without
  // this the gate would never open again and the topic would go silent for good.
  if (now < last_publish_time_)
  {
    last_publish_time_ = now;
  }
  if (now - last_publish_time_ < publish_period_)
  {
    return;
  }
  // Advance by whole periods rather than to `now`, so the rate does not round up
  // to the next control tick: at 250 Hz, resetting to `now` turns a 30 Hz period
  // into 9 ticks, i.e. 27.8 Hz. Catch up in one step if publishing fell behind.
  last_publish_time_ += publish_period_;
  if (now - last_publish_time_ > publish_period_)
  {
    last_publish_time_ = now;
  }

  message_.header.stamp = now;
  // Only the outer vector keeps its capacity; each entry's point arrays are
  // rebuilt, so a publish still allocates.
  message_.contacts.clear();

  for (int i = 0; i < data->ncon; ++i)
  {
    const mjContact& con = data->contact[i];
    const int g1 = con.geom[0];
    const int g2 = con.geom[1];
    // Flex contacts report -1 here and have no geom to name.
    if (g1 < 0 || g2 < 0 || (!watched_[g1] && !watched_[g2]))
    {
      continue;
    }

    // Gazebo emits one message entry per colliding pair, with the individual points in
    // parallel arrays; MuJoCo emits up to four separate contacts for the same pair, so they
    // are folded back together here. Entity ids carry the geom id, which makes the lookup
    // exact without a second index.
    auto it = std::find_if(message_.contacts.begin(), message_.contacts.end(),
                           [g1, g2](const ros_gz_interfaces::msg::Contact& c) {
                             return c.collision1.id == static_cast<uint64_t>(g1) &&
                                    c.collision2.id == static_cast<uint64_t>(g2);
                           });
    if (it == message_.contacts.end())
    {
      message_.contacts.emplace_back();
      it = std::prev(message_.contacts.end());
      it->collision1.id = g1;
      it->collision1.name = collision_names_[g1];
      it->collision1.type = ros_gz_interfaces::msg::Entity::COLLISION;
      it->collision2.id = g2;
      it->collision2.name = collision_names_[g2];
      it->collision2.type = ros_gz_interfaces::msg::Entity::COLLISION;
    }

    mjtNum wrench[6];
    mj_contactForce(model_, data, i, wrench);

    mjtNum force[3];
    mjtNum torque[3];
    mju_mulMatTVec3(force, con.frame, wrench);
    mju_mulMatTVec3(torque, con.frame, wrench + 3);

    it->positions.emplace_back();
    toVector3(con.pos, it->positions.back());
    it->normals.emplace_back();
    toVector3(con.frame, it->normals.back());  // frame's first row is the normal
    it->depths.push_back(-con.dist);           // dist is negative while penetrating

    // con.frame's normal points from geom[0] to geom[1] and mj_contactForce resolves to the
    // force geom[0] applies to geom[1], so body 2 takes it as-is and body 1 takes its negation.
    it->wrenches.emplace_back();
    auto& jw = it->wrenches.back();
    jw.header = message_.header;
    jw.body_1_id.data = model_->geom_bodyid[g1];
    jw.body_1_name.data = body_names_[g1];
    jw.body_2_id.data = model_->geom_bodyid[g2];
    jw.body_2_name.data = body_names_[g2];
    toVector3(force, jw.body_1_wrench.force, -1.0);
    toVector3(torque, jw.body_1_wrench.torque, -1.0);
    toVector3(force, jw.body_2_wrench.force);
    toVector3(torque, jw.body_2_wrench.torque);
  }

#if REALTIME_TOOLS_VERSION_MAJOR > 2
  publisher_->try_publish(message_);
#else
  publisher_->tryPublish(message_);
#endif
}

void ContactSensorPlugin::cleanup()
{
  RCLCPP_INFO(logger_, "ContactSensorPlugin cleanup.");
  publisher_.reset();
  publisher_raw_.reset();
  node_.reset();
}

}  // namespace erc_bringup

PLUGINLIB_EXPORT_CLASS(erc_bringup::ContactSensorPlugin, mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
