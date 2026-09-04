-- Cartographer configuration for Waveshare WAVE ROVER
-- 2D SLAM (LD19 on /scan) WITH IMU fusion.
--
-- The IMU arrives on /imu/data from wave_rover_bridge.py, read off the same
-- serial line as the motor commands and republished at 20 Hz. With no wheel
-- odometry it is Cartographer's only source of motion between scans:
-- gravity for alignment, and yaw rate for rotation.
--
-- This file is the base for mapping AND localization --
-- wave_rover_localization.lua begins with include "wave_rover.lua", so
-- everything here is live in both unless that file overrides it afterwards.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_laser",         -- track at the IMU when fusing IMU
  published_frame = "base_link",         -- publish the robot-center pose for PPO
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  publish_tracked_pose = true,
  use_odometry = false,                 -- no wheel encoders on WAVE ROVER
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- This read `false` while the comment above it said "fuse the IMU", and the
-- scan matcher below was narrowed on the assumption the IMU was supplying a
-- rotation prior. So the matcher was searching a tight window around a prior
-- that did not exist -- with nothing to fall back on, it locked onto whatever
-- was nearest in a symmetric room. That is the state localization has been
-- running in, and no amount of fixing the scan pipeline could cure it.
TRAJECTORY_BUILDER_2D.use_imu_data = true
TRAJECTORY_BUILDER_2D.min_range = 0.1
TRAJECTORY_BUILDER_2D.max_range = 8.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.5)
-- The IMU supplies the rotation prior, so the correlative matcher only needs
-- a NARROW search around it. Was 40 deg / 0.2 m -- a legacy fix for
-- turn-warping that the scan-timestamp relay already solved. The wide window
-- let the matcher wander off the IMU prediction in open/symmetric rooms,
-- snapping the pose to wrong angles (rotating/overlapping/shaking map).
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(15.)
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1

POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.constraint_builder.max_constraint_distance = 8.0
POSE_GRAPH.constraint_builder.min_score = 0.7
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.75

return options
