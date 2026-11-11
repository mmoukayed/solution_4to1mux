/**
 * Copyright (c) 2026, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration.
 *
 * All rights reserved.
 *
 * This software is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "camera_plugin.hpp"

namespace mujoco_ros2_control_plugins
{

bool CameraPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data)
{
  // by default use `glfwInit` to check if the GLFW is initialized.
  return this->init(node, model, data, glfwInit);
}

bool CameraPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data, GlfwInitFn glfw_init_fn)
{
  node_ = node;
  mj_model_ = model;
  mj_data_ = data;

  // Ensure the logger has a name
  logger_ = node_->get_logger().get_child(node->get_sub_namespace());
  RCLCPP_INFO(node_->get_logger(), "CameraPlugin initializing cameras...");

  // Read the mj_model_, identify the number of cameras, and populate containers for them.
  if (!register_cameras())
  {
    RCLCPP_ERROR(node_->get_logger(), "Failed to register cameras.");
    return false;
  }
  if (cameras_.empty())
  {
    return true;
  }

  // Start the rendering thread process
  // Try GLFW first, fall back to EGL for headless environments
  if (glfw_init_fn())
  {
    use_egl_ = false;
  }
  else
  {
    RCLCPP_WARN(node_->get_logger(), "Failed to initialize GLFW. Attempting EGL for headless rendering.");
    use_egl_ = true;
  }
  rendering_thread_ = std::thread(&CameraPlugin::update_loop, this);
  return true;
}

void CameraPlugin::update(const mjModel* model_arg, mjData* data)
{
  if (!publish_images_)
  {
    return;
  }

  // Streaming cameras render on a fixed-rate clock; polled cameras render as soon as a
  // trigger has been received. Both are serviced here, on the sim thread, because this is
  // the only place we can safely snapshot the live mjData without racing the simulation.
  // TODO: Support per-camera publish rates?
  // CHANGED FROM UPSTREAM: the interval is measured on the snapshot's own time,
  // not the node clock. Above real time the physics batches several control
  // periods into one snapshot while the clock runs ahead of it, so the clock
  // said a frame was due twice for the same instant and every frame was
  // rendered and published twice (at 100x: 840 camera_info in 6 s, 416 stamps).
  // A time that went backwards (reset_world) is due at once, not when it catches up.
  const rclcpp::Time now(rclcpp::Duration::from_seconds(data->time).nanoseconds(), RCL_ROS_TIME);
  const double since_last = (now - last_publish_time_).seconds();
  const bool stream_due =
      has_streaming_cameras_ && (since_last >= (1.0 / camera_publish_rate_) || since_last < 0.0);
  const bool poll_due = poll_pending_.exchange(false);

  // Nothing to do this step: avoid taking the lock or copying data.
  if (!stream_due && !poll_due)
  {
    return;
  }

  bool any_selected = false;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Flag streaming cameras when their interval is due and any polled cameras that have a
    // pending trigger (consuming the one-shot request so they render exactly once).
    // A streaming camera nobody listens to is not rendered at all: the render and the
    // mjData snapshot it needs are the plugin's whole cost, and the interval is in
    // simulated time, so at several times real time an unwatched camera would otherwise
    // eat a core for frames that are dropped. Its camera_info still goes out on every
    // tick to whoever listens for it, as a camera driver's does: the calibration costs
    // nothing and a consumer may read it before it asks for frames.
    for (auto& camera : cameras_)
    {
      if (camera.policy == CameraPolicy::STREAMING && stream_due)
      {
        if (camera.image_pub->get_subscription_count() > 0 ||
            camera.depth_image_pub->get_subscription_count() > 0)
        {
          camera.render_pending = true;
          any_selected = true;
        }
        else if (camera.camera_info_pub->get_subscription_count() > 0 ||
                 (camera.depth_info_pub && camera.depth_info_pub->get_subscription_count() > 0))
        {
          auto info = camera.camera_info;
          info.header.stamp = now;
          camera.camera_info_pub->publish(info);
          if (camera.depth_info_pub)
          {
            camera.depth_info_pub->publish(info);
          }
        }
      }
      else if (camera.policy == CameraPolicy::POLLED && camera.poll_requested)
      {
        camera.poll_requested = false;
        camera.render_pending = true;
        any_selected = true;
      }
    }

    if (stream_due)
    {
      last_publish_time_ = now;
    }

    // Only snapshot the simulation data when there is actually a camera to render.
    if (any_selected)
    {
      mjv_copyData(mj_camera_data_, model_arg, data);
      new_data_ = true;
    }
  }

  if (any_selected)
  {
    data_cv_.notify_one();
  }
}

void CameraPlugin::cleanup()
{
  close();
}

void CameraPlugin::trigger_update()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Force every streaming camera and consume any pending polls, then render synchronously.
  for (auto& camera : cameras_)
  {
    if (camera.policy == CameraPolicy::STREAMING)
    {
      camera.render_pending = true;
    }
    else if (camera.policy == CameraPolicy::POLLED && camera.poll_requested)
    {
      camera.poll_requested = false;
      camera.render_pending = true;
    }
  }
  select_cameras();
  update_cameras();
}

bool CameraPlugin::register_cameras()
{
  const std::string param_prefix = "mujoco_plugins.mujoco_camera_plugin.";

  if (!node_->has_parameter(param_prefix + "camera_publish_rate"))
  {
    node_->declare_parameter(param_prefix + "camera_publish_rate", 5.0);
  }

  camera_publish_rate_ = node_->get_parameter(param_prefix + "camera_publish_rate").as_double();
  RCLCPP_INFO(node_->get_logger(), "Publishing camera data at rate %f per second.", camera_publish_rate_);

  cameras_.resize(0);
  for (auto i = 0; i < mj_model_->ncam; ++i)
  {
    const char* cam_name = mj_model_->names + mj_model_->name_camadr[i];
    const int* cam_resolution = mj_model_->cam_resolution + 2 * i;
    const mjtNum cam_fovy = mj_model_->cam_fovy[i];

    // Construct CameraData wrapper and set defaults
    CameraData camera;
    camera.name = cam_name;
    camera.mjv_cam.type = mjCAMERA_FIXED;
    camera.mjv_cam.fixedcamid = i;
    camera.width = static_cast<uint32_t>(cam_resolution[0]);
    camera.height = static_cast<uint32_t>(cam_resolution[1]);
    camera.viewport = { 0, 0, cam_resolution[0], cam_resolution[1] };

    const std::string param_ns = param_prefix + cam_name + ".";

    const std::string policy_param = param_ns + "policy";
    if (!node_->has_parameter(policy_param))
    {
      node_->declare_parameter(policy_param, "streaming");
    }
    std::string policy_str = node_->get_parameter(policy_param).as_string();
    if (policy_str == "streaming")
    {
      camera.policy = CameraPolicy::STREAMING;
    }
    else if (policy_str == "polled")
    {
      camera.policy = CameraPolicy::POLLED;
    }
    else if (policy_str == "disabled")
    {
      camera.policy = CameraPolicy::DISABLED;
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Invalid policy for camera '%s'", cam_name);
      return false;
    }

    const std::string frame_param = param_ns + "frame_name";
    if (!node_->has_parameter(frame_param))
    {
      node_->declare_parameter(frame_param, camera.name + "_frame");
    }
    camera.frame_name = node_->get_parameter(frame_param).as_string();

    if (!node_->has_parameter(param_ns + "info_topic"))
    {
      node_->declare_parameter(param_ns + "info_topic", camera.name + "/camera_info");
    }
    camera.info_topic = node_->get_parameter(param_ns + "info_topic").as_string();

    if (!node_->has_parameter(param_ns + "image_topic"))
    {
      node_->declare_parameter(param_ns + "image_topic", camera.name + "/color");
    }
    camera.image_topic = node_->get_parameter(param_ns + "image_topic").as_string();

    if (!node_->has_parameter(param_ns + "depth_topic"))
    {
      node_->declare_parameter(param_ns + "depth_topic", camera.name + "/depth");
    }
    camera.depth_topic = node_->get_parameter(param_ns + "depth_topic").as_string();

    if (!node_->has_parameter(param_ns + "min_depth"))
    {
      node_->declare_parameter(param_ns + "min_depth", 0.0);
    }
    camera.min_depth = node_->get_parameter(param_ns + "min_depth").as_double();

    if (!node_->has_parameter(param_ns + "depth_model"))
    {
      node_->declare_parameter(param_ns + "depth_model", std::string());
    }
    camera.depth_model = node_->get_parameter(param_ns + "depth_model").as_string();
    if (!camera.depth_model.empty())
    {
      if (const char* env = std::getenv("ERC_DEPTH"); env && *env)
      {
        camera.depth_model = env;
      }
      auto& dp = camera.depth_params;
      auto num = [&](const std::string& key, double def) {
        if (!node_->has_parameter(param_ns + key))
        {
          node_->declare_parameter(param_ns + key, def);
        }
        return node_->get_parameter(param_ns + key).as_double();
      };
      dp.noise_k = num("depth_noise_k", dp.noise_k);
      dp.holes_dark = num("depth_holes_dark", dp.holes_dark);
      dp.holes_bright = num("depth_holes_bright", dp.holes_bright);
      dp.blur_px = static_cast<int>(num("depth_blur_px", dp.blur_px));
      if (const char* k = std::getenv("ERC_DEPTH_K"); k && *k)
      {
        dp.noise_k = std::atof(k);
      }
      // ERC_DEPTH_HOLES scales how much reads no return (worse than measured, for a sweep)
      if (const char* k = std::getenv("ERC_DEPTH_HOLES"); k && *k)
      {
        const double f = std::atof(k);
        dp.holes_dark = std::min(0.9, dp.holes_dark * f);
        dp.holes_bright = std::min(0.9, dp.holes_bright * f);
        dp.big_holes = std::min(0.5, dp.big_holes * f);
        dp.edge_holes = std::min(0.95, dp.edge_holes * f);
        dp.specks = std::min(0.5, dp.specks * f);
      }
      if (camera.depth_model != "d435" && camera.depth_model != "exact")
      {
        RCLCPP_ERROR(node_->get_logger(), "camera '%s': depth_model is d435 or exact, not '%s' - exact",
                     cam_name, camera.depth_model.c_str());
        camera.depth_model = "exact";
      }
    }

    if (!node_->has_parameter(param_ns + "depth_info_topic"))
    {
      node_->declare_parameter(param_ns + "depth_info_topic", std::string());
    }
    camera.depth_info_topic = node_->get_parameter(param_ns + "depth_info_topic").as_string();

    if (!node_->has_parameter(param_ns + "trigger_service_name"))
    {
      node_->declare_parameter(param_ns + "trigger_service_name", camera.name + "/trigger");
    }
    camera.trigger_service_name = node_->get_parameter(param_ns + "trigger_service_name").as_string();

    RCLCPP_INFO(node_->get_logger(), "Adding camera: '%s'", cam_name);
    RCLCPP_INFO(node_->get_logger(), "    policy: '%s'", policy_str.c_str());
    RCLCPP_INFO(node_->get_logger(), "    frame_name: '%s'", camera.frame_name.c_str());
    if (camera.policy == CameraPolicy::DISABLED)
    {
      continue;
    }
    RCLCPP_INFO(node_->get_logger(), "    info_topic: '%s'", camera.info_topic.c_str());
    RCLCPP_INFO(node_->get_logger(), "    image_topic: '%s'", camera.image_topic.c_str());
    RCLCPP_INFO(node_->get_logger(), "    depth_topic: '%s'", camera.depth_topic.c_str());
    if (camera.min_depth > 0.0)
    {
      RCLCPP_INFO(node_->get_logger(), "    min_depth: %.3f m", camera.min_depth);
    }
    if (!camera.depth_info_topic.empty())
    {
      RCLCPP_INFO(node_->get_logger(), "    depth_info_topic: '%s'", camera.depth_info_topic.c_str());
    }
    if (camera.depth_model == "d435")
    {
      const auto& dp = camera.depth_params;
      RCLCPP_INFO(node_->get_logger(),
                  "    depth_model: d435 (noise %.4f z^2 in patches, %d px matcher blur at 848 px wide, holes %.0f-%.0f %%)",
                  dp.noise_k, 2 * dp.blur_px + 1, 100 * dp.holes_dark, 100 * dp.holes_bright);
    }
    if (camera.policy == CameraPolicy::POLLED)
    {
      RCLCPP_INFO(node_->get_logger(), "    trigger_service_name: '%s'", camera.trigger_service_name.c_str());
    }

    // Configure publishers and services
    camera.camera_info_pub = node_->create_publisher<sensor_msgs::msg::CameraInfo>(camera.info_topic, 1);
    camera.image_pub = node_->create_publisher<sensor_msgs::msg::Image>(camera.image_topic, 1);
    camera.depth_image_pub = node_->create_publisher<sensor_msgs::msg::Image>(camera.depth_topic, 1);
    if (!camera.depth_info_topic.empty())
    {
      camera.depth_info_pub = node_->create_publisher<sensor_msgs::msg::CameraInfo>(camera.depth_info_topic, 1);
    }
    if (camera.policy == CameraPolicy::POLLED)
    {
      camera.trigger_service = node_->create_service<std_srvs::srv::Trigger>(
          camera.trigger_service_name, [this, i](const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                                 std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            this->handle_trigger(request, response, i);
          });
    }
    // Setup containers for color image data
    camera.image.header.frame_id = camera.frame_name;

    const auto image_size = camera.width * camera.height * 3;
    camera.image_buffer.resize(image_size);
    camera.image.data.resize(image_size);
    camera.image.width = camera.width;
    camera.image.height = camera.height;
    camera.image.step = camera.width * 3;
    camera.image.encoding = sensor_msgs::image_encodings::RGB8;

    // Depth image data
    camera.depth_image.header.frame_id = camera.frame_name;
    camera.depth_buffer.resize(camera.width * camera.height);
    camera.depth_mm = camera.depth_model == "d435";
    camera.depth_work.resize(camera.width * camera.height);
    camera.depth_image.data.resize(camera.width * camera.height * (camera.depth_mm ? sizeof(uint16_t) : sizeof(float)));
    camera.depth_image.width = camera.width;
    camera.depth_image.height = camera.height;
    camera.depth_image.step = camera.width * static_cast<uint32_t>(camera.depth_mm ? sizeof(uint16_t) : sizeof(float));
    camera.depth_image.encoding = camera.depth_mm ? sensor_msgs::image_encodings::TYPE_16UC1
                                                  : sensor_msgs::image_encodings::TYPE_32FC1;

    // Camera info
    camera.camera_info.header.frame_id = camera.frame_name;
    camera.camera_info.width = camera.width;
    camera.camera_info.height = camera.height;
    camera.camera_info.distortion_model = "plumb_bob";
    camera.camera_info.k.fill(0.0);
    camera.camera_info.r.fill(0.0);
    camera.camera_info.p.fill(0.0);
    camera.camera_info.d.resize(5, 0.0);

    double focal_scaling = (1.0 / std::tan((cam_fovy * M_PI / 180.0) / 2.0)) * camera.height / 2.0;
    camera.camera_info.k[0] = camera.camera_info.p[0] = focal_scaling;
    camera.camera_info.k[2] = camera.camera_info.p[2] = static_cast<double>(camera.width) / 2.0;
    camera.camera_info.k[4] = camera.camera_info.p[5] = focal_scaling;
    camera.camera_info.k[5] = camera.camera_info.p[6] = static_cast<double>(camera.height) / 2.0;
    camera.camera_info.k[8] = camera.camera_info.p[10] = 1.0;

    // Add to list of cameras
    cameras_.push_back(camera);
  }

  has_streaming_cameras_ = std::any_of(cameras_.begin(), cameras_.end(),
                                       [](const CameraData& cam) { return cam.policy == CameraPolicy::STREAMING; });

  // Reserve once so the per-pass render bookkeeping in `update_cameras()` never allocates.
  render_indices_.reserve(cameras_.size());

  return true;
}

void CameraPlugin::close()
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stop_requested_ = true;
    publish_images_ = false;
  }
  data_cv_.notify_one();
  if (rendering_thread_.joinable())
  {
    rendering_thread_.join();
  }
}

bool CameraPlugin::init_egl_context()
{
  // Get EGL display
  egl_display_ = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);

  // Initialize EGL
  EGLint major, minor;
  if (!eglInitialize(egl_display_, &major, &minor))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to initialize (error: 0x%x)", eglGetError());
    return false;
  }
  const char* client_apis = eglQueryString(egl_display_, EGL_CLIENT_APIS);
  const char* vendor = eglQueryString(egl_display_, EGL_VENDOR);
  RCLCPP_INFO(node_->get_logger(), "EGL: Initialized version %d.%d", major, minor);
  RCLCPP_INFO(node_->get_logger(), "EGL: Vendor: %s, APIs: %s", vendor, client_apis);
  // Choose EGL config for offscreen rendering
  const EGLint config_attribs[] = { EGL_SURFACE_TYPE,
                                    EGL_PBUFFER_BIT,
                                    EGL_RED_SIZE,
                                    8,
                                    EGL_GREEN_SIZE,
                                    8,
                                    EGL_BLUE_SIZE,
                                    8,
                                    EGL_ALPHA_SIZE,
                                    8,
                                    EGL_DEPTH_SIZE,
                                    24,
                                    EGL_RENDERABLE_TYPE,
                                    EGL_OPENGL_BIT,
                                    EGL_NONE };

  EGLConfig egl_config;
  EGLint num_configs;
  if (!eglChooseConfig(egl_display_, config_attribs, &egl_config, 1, &num_configs) || num_configs == 0)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to choose config (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Bind OpenGL API
  if (!eglBindAPI(EGL_OPENGL_API))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to bind OpenGL API (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Create EGL context
  egl_context_ = eglCreateContext(egl_display_, egl_config, EGL_NO_CONTEXT, nullptr);
  if (egl_context_ == EGL_NO_CONTEXT)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to create context (error: 0x%x)", eglGetError());
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Create PBuffer surface for offscreen rendering
  const EGLint pbuffer_attribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
  egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config, pbuffer_attribs);
  if (egl_surface_ == EGL_NO_SURFACE)
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to create PBuffer surface (error: 0x%x)", eglGetError());
    eglDestroyContext(egl_display_, egl_context_);
    eglTerminate(egl_display_);
    egl_context_ = EGL_NO_CONTEXT;
    egl_display_ = EGL_NO_DISPLAY;
    return false;
  }

  // Make the context current
  if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
  {
    RCLCPP_ERROR(node_->get_logger(), "EGL: Failed to make context current (error: 0x%x)", eglGetError());
    cleanup_egl_context();
    return false;
  }

  RCLCPP_INFO(node_->get_logger(), "EGL: Successfully initialized headless OpenGL context");
  return true;
}

void CameraPlugin::cleanup_egl_context()
{
  if (egl_display_ != EGL_NO_DISPLAY)
  {
    eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (egl_surface_ != EGL_NO_SURFACE)
    {
      eglDestroySurface(egl_display_, egl_surface_);
      egl_surface_ = EGL_NO_SURFACE;
    }
    if (egl_context_ != EGL_NO_CONTEXT)
    {
      eglDestroyContext(egl_display_, egl_context_);
      egl_context_ = EGL_NO_CONTEXT;
    }
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
  }
}

void CameraPlugin::update_loop()
{
  GLFWwindow* window = nullptr;

  if (use_egl_)
  {
    // Initialize EGL for headless rendering
    if (!init_egl_context())
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to initialize EGL context. Disabling camera publishing.");
      publish_images_ = false;
      return;
    }
  }
  else
  {
    // Use GLFW for offscreen context (display available)
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1, 1, "", NULL, NULL);
    if (!window)
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to create GLFW window. Disabling camera publishing.");
      publish_images_ = false;
      return;
    }
    glfwMakeContextCurrent(window);
  }

  // Determine rendering backend name for logging
  const char* backend = use_egl_ ? "EGL" : "GLFW";

  // Initialization of the context and data structures has to happen in the rendering thread
  RCLCPP_INFO(node_->get_logger(), "Initializing rendering for cameras (using %s)", backend);
  mjv_defaultOption(&mjv_opt_);
  mjv_defaultScene(&mjv_scn_);
  mjr_defaultContext(&mjr_con_);

  // Turn rangefinder rendering off so we don't get rays in camera images
  mjv_opt_.flags[mjtVisFlag::mjVIS_RANGEFINDER] = 0;
  // Turn off site rendering so that visualization is more realistic in cameras for testing perception.
  for (int i = 0; i < mjNGROUP; i++)
  {
    mjv_opt_.sitegroup[i] = 0;
  }

  // Initialize data for camera rendering
  mj_camera_data_ = mj_makeData(mj_model_);
  mj_render_data_ = mj_makeData(mj_model_);
  RCLCPP_INFO(node_->get_logger(), "Starting the camera rendering loop, publishing at %f Hz", camera_publish_rate_);

  // create scene and context
  // 10000, not 2000: a scene past the cap drops whatever comes last, and the
  // venue's world (a wall of ~2600 books) lost the shelf it is set in - boards,
  // books, label - from the head camera while every other view drew it
  mjv_makeScene(mj_model_, &mjv_scn_, 10000);

  mjr_makeContext(mj_model_, &mjr_con_, mjFONTSCALE_150);

  // Ensure the context will support the largest cameras
  int max_width = 1, max_height = 1;
  for (const auto& cam : cameras_)
  {
    max_width = std::max(max_width, static_cast<int>(cam.width));
    max_height = std::max(max_height, static_cast<int>(cam.height));
  }
  mjr_resizeOffscreen(max_width, max_height, &mjr_con_);
  RCLCPP_INFO(node_->get_logger(), "Resized offscreen buffer to %d x %d", max_width, max_height);

  // Only process images once all data has been initialized, and do it until told to stop.
  // Publishing is enabled under the lock so that a shutdown requested during the (possibly
  // slow) initialization above is observed here rather than being clobbered.
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    publish_images_ = !stop_requested_;
  }
  while (rclcpp::ok() && !stop_requested_)
  {
    std::unique_lock<std::mutex> lock(data_mutex_);

    // Wait for the main thread to copy the data and trigger this to process and publish
    // images. Note that condition_variables can be awoken spuriously, so the additional
    // checks are necessary to avoid doing work excepting when updated rendering data has
    // been made available.
    data_cv_.wait(lock, [this] { return new_data_ || stop_requested_; });

    // Shutdown triggered, kill the loop and clean up.
    if (stop_requested_)
    {
      break;
    }
    select_cameras();
    lock.unlock();

    update_cameras();
  }
  publish_images_ = false;

  mjv_freeScene(&mjv_scn_);
  mjr_freeContext(&mjr_con_);
  mj_deleteData(mj_camera_data_);
  mj_deleteData(mj_render_data_);

  if (use_egl_)
  {
    cleanup_egl_context();
  }
  else if (window)
  {
    glfwDestroyWindow(window);
  }
}

void CameraPlugin::select_cameras()
{
  // CHANGED FROM UPSTREAM: the pass takes the newest snapshot for its own (a
  // pointer swap) and the flags with it, both under the lock. The render used
  // to read mj_camera_data_ unlocked while the sim thread kept copying into it,
  // so with the D435 model between the depth camera's render and the colour's,
  // the colour came from up to a second later than the depth and its stamp
  // (venue world, 1 Hz passes; issue #67).
  if (new_data_)
  {
    std::swap(mj_camera_data_, mj_render_data_);
    new_data_ = false;
  }
  render_indices_.clear();
  for (size_t i = 0; i < cameras_.size(); ++i)
  {
    auto& camera = cameras_[i];
    if (camera.render_pending)
    {
      camera.render_pending = false;
      render_indices_.push_back(i);
    }
  }
}

void CameraPlugin::update_cameras()
{
  if (render_indices_.empty())
  {
    return;
  }

  // Rendering is done offscreen from the one snapshot select_cameras() took.
  mjr_setBuffer(mjFB_OFFSCREEN, &mjr_con_);

  camera_near_distance_ = static_cast<float>(mj_model_->vis.map.znear * mj_model_->stat.extent);
  const float far = static_cast<float>(mj_model_->vis.map.zfar * mj_model_->stat.extent);
  camera_depth_scale_ = 1.0f - camera_near_distance_ / far;

  // Use the snapshotted data's timestamp in the ROS header, as that is when the data was actually pulled.
  const rclcpp::Duration duration = rclcpp::Duration::from_seconds(mj_render_data_->time);
  rclcpp::Time stamp(duration.nanoseconds(), RCL_ROS_TIME);

  for (const auto idx : render_indices_)
  {
    render_and_publish_camera(cameras_[idx], stamp);
  }
}

void CameraPlugin::render_and_publish_camera(CameraData& camera, const rclcpp::Time& stamp)
{
  // Step 1: Render the scene and copy images to relevant camera data containers.
  // Render scene
  mjv_updateScene(mj_model_, mj_render_data_, &mjv_opt_, NULL, &camera.mjv_cam, mjCAT_ALL, &mjv_scn_);
  mjr_render(camera.viewport, &mjv_scn_, &mjr_con_);

  // Copy image into relevant buffers
  mjr_readPixels(camera.image_buffer.data(), camera.depth_buffer.data(), camera.viewport, &mjr_con_);

  // Step 2: Adjust the images and copy depth data.
  // Fix non-linear depth buffer and flip it vertically (OpenGL's origin is the bottom left)
  // https://github.com/google-deepmind/mujoco/blob/3.4.0/python/mujoco/renderer.py#L190
  auto* depth_out = camera.depth_mm ? camera.depth_work.data() : reinterpret_cast<float*>(camera.depth_image.data.data());
  const auto min_depth = static_cast<float>(camera.min_depth);
  for (uint32_t h = 0; h < camera.height; ++h)
  {
    const float* src_row = camera.depth_buffer.data() + static_cast<size_t>(h) * camera.width;
    float* dst_row = depth_out + static_cast<size_t>(camera.height - 1 - h) * camera.width;
    for (uint32_t w = 0; w < camera.width; ++w)
    {
      const float z = camera_near_distance_ / (1.0f - src_row[w] * camera_depth_scale_);
      dst_row[w] = z < min_depth ? 0.0f : z;
    }
  }

  // OpenGL's coordinate system's origin is in the bottom left, so we invert the images row-by-row
  const auto row_size = camera.width * 3;
  for (uint32_t h = 0; h < camera.height; ++h)
  {
    const auto src_idx = h * row_size;
    const auto dest_idx = (camera.height - 1 - h) * row_size;
    std::memcpy(&camera.image.data[dest_idx], &camera.image_buffer[src_idx], row_size);
  }

  // Step 2b: what a real depth camera makes of that (after the colour's flip:
  // its holes follow how bright each surface is).
  if (camera.depth_model == "d435")
  {
    if (!camera.depth_noise.ready())
    {
      camera.depth_params.fx = camera.camera_info.k[0];   // the stereo shadow's width
      camera.depth_noise.configure(camera.depth_params, camera.width, camera.height);
      const auto& ap = camera.depth_noise.applied();
      RCLCPP_INFO(node_->get_logger(), "camera '%s': d435 depth at %ux%u: %d px matcher blur, %d px noise patches",
                  camera.name.c_str(), camera.width, camera.height, 2 * ap.blur_px + 1, 2 * ap.noise_patch_px + 1);
    }
    camera.depth_noise.apply(depth_out, camera.image.data.data());
  }
  if (camera.depth_mm)
  {
    auto* mm = reinterpret_cast<uint16_t*>(camera.depth_image.data.data());
    const size_t n = static_cast<size_t>(camera.width) * camera.height;
    for (size_t i = 0; i < n; ++i)
    {
      const float v = depth_out[i] * 1000.0f;
      mm[i] = (v > 0.0f && v < 65535.0f) ? static_cast<uint16_t>(v + 0.5f) : 0;
    }
  }

  // Step 3: Publish the images and camera info. The info is a stamped copy: the
  // registered one is read by the sim thread too (update) and is never written.
  camera.image.header.stamp = stamp;
  camera.depth_image.header.stamp = stamp;
  auto info = camera.camera_info;
  info.header.stamp = stamp;

  camera.image_pub->publish(camera.image);
  camera.depth_image_pub->publish(camera.depth_image);
  camera.camera_info_pub->publish(info);
  if (camera.depth_info_pub)
  {
    camera.depth_info_pub->publish(info);
  }
}

void CameraPlugin::handle_trigger(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> response, const int camera_idx)
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cameras_.at(camera_idx).poll_requested = true;
  }
  // Wake the fast path in `update()` so the request is picked up on the next sim step.
  poll_pending_.store(true);
  response->success = true;
}

}  // namespace mujoco_ros2_control_plugins

// Export the plugin
PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_plugins::CameraPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
