-- Pure localization against a prebuilt house.pbstream.
-- Inherits everything from the mapping config (IMU off, no odometry,
-- min_range 0.1 / max_range 8.0) so scans are interpreted exactly as
-- they were when the map was built. Only localization behavior differs.

include "wave_rover.lua"

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
