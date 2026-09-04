-- Pure localization against a prebuilt house.pbstream.
--
-- Inherits the mapping config's range settings (min_range 0.1 /
-- max_range 8.0) so scans are interpreted exactly as they were when the map
-- was built. Those have to match. The IMU setting does not.
--
-- The map was built without an IMU and this localizes with one, which is
-- fine: use_imu_data changes how motion is predicted between scan matches,
-- not how the frozen submaps are read.
--
-- It is on because without it the pose extrapolator had nothing but the
-- difference between consecutive scan matches to work from, and whenever
-- scan matching was momentarily starved the pose left the map at metres per
-- second on a stationary rover -- measured repeatedly, and reproducible on
-- demand by starting Nav2 against a settled Cartographer. The IMU gives it
-- angular rates and gravity to fall back on.
--
-- /imu/data comes from wave_rover_bridge.py, which reads it off the same
-- serial line as the motor commands, at 20 Hz. There is no separate IMU
-- node: the board is on ttyS0 behind /dev/serial0 and only one process may
-- own that port. Cartographer will not process a scan without IMU data, so
-- rover-cartographer.service requires rover-bridge.service.

include "wave_rover.lua"
TRAJECTORY_BUILDER_2D.use_imu_data = true

-- Localize against the loaded map instead of extending it.
-- Keeps only the few most recent submaps so memory stays bounded.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- Optimize more often than mapping (was 35) for responsive pose updates.
POSE_GRAPH.optimize_every_n_nodes = 90

MAP_BUILDER.num_background_threads = 2
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 2
return options
