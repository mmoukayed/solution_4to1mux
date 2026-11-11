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

#ifndef ERC_BRINGUP__CONTACT_SENSOR_PLUGIN_HPP_
#define ERC_BRINGUP__CONTACT_SENSOR_PLUGIN_HPP_

#include <memory>
#include <string>
#include <vector>

#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <ros_gz_interfaces/msg/contacts.hpp>

namespace erc_bringup
{

/**
 * @brief Reports every contact on one body subtree as ros_gz_interfaces/msg/Contacts.
 *
 * MuJoCo already solves for every contact in the scene and stores it in
 * mjData::contact, complete with the pair of geoms, the world position, the
 * contact frame, the penetration depth and — via mj_contactForce — the 6D
 * constraint force. Nothing has to be declared in the MJCF for that to happen,
 * so this plugin adds no simulation state and needs no cooperation from the
 * description or arena generators. MuJoCo's own site-based `touch` sensor was
 * the alternative and cannot be used: it returns a single scalar force per
 * site, with no contact partner, position or normal, which is most of the
 * message.
 *
 * The scan runs on the control thread in update(), against the post-step
 * snapshot the hardware interface already hands every plugin, and only on the
 * cycles that actually publish. Between publishes the plugin costs one clock
 * comparison.
 *
 * Parameters (declared under "mujoco_plugins.<instance_name>.")
 * -------------------------------------------------------------
 *   body         (string, required) - MJCF body at the root of the watched
 *                 subtree. A contact is reported when either of its geoms
 *                 belongs to that body or to anything below it.
 *   topic        (string, default "contacts")   - output topic name.
 *   publish_rate (double, default 30.0)         - publish frequency in Hz.
 *
 * A `body` that the scene does not contain is an error but not a fatal one:
 * the sensor then watches nothing and reports nothing, which is the truthful
 * answer for a `scene:=` that leaves the object out. Only an unset `body`
 * fails init, because that is a configuration mistake rather than a scene.
 */
class ContactSensorPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  ContactSensorPlugin() = default;
  ~ContactSensorPlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override;
  void update(const mjModel* model, mjData* data) override;
  void cleanup() override;

private:
  using Contacts = ros_gz_interfaces::msg::Contacts;

  /// Marks the geoms of `root_body` and everything below it, and resolves the name every geom
  /// is reported under. Returns the number of geoms watched.
  std::size_t collect_watched_geoms(const std::string& root_body);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("ContactSensorPlugin") };

  const mjModel* model_{ nullptr };

  /// Per-geom, resolved once in init(): part of the watched subtree, the name the geom is
  /// published under, and its body's name and id for the wrench entries.
  std::vector<uint8_t> watched_;
  std::vector<std::string> collision_names_;
  std::vector<std::string> body_names_;

  rclcpp::Publisher<Contacts>::SharedPtr publisher_raw_;
  std::unique_ptr<realtime_tools::RealtimePublisher<Contacts>> publisher_;

  rclcpp::Duration publish_period_{ 0, 0 };
  rclcpp::Time last_publish_time_{ 0, 0, RCL_ROS_TIME };

  /// Reused across update() calls so a steady contact set stops reallocating.
  Contacts message_;
};

}  // namespace erc_bringup

#endif  // ERC_BRINGUP__CONTACT_SENSOR_PLUGIN_HPP_
