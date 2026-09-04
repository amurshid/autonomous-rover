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

-- Do not accept a match from somewhere the rover cannot be.
--
-- Cartographer builds occasional near-empty nodes -- 1, 9, 16 points where a
-- healthy scan gives 200 -- and a handful of points matches almost anywhere.
-- Measured over an evening, with the rover parked and untouched:
--
--   honest constraints    0.00 - 0.60 m away,  200-215 points
--   every harmful one     4.48, 4.88, 8.69 m,  1, 9, 16 points
--
-- Score does not separate them: the bad ones came in at 77-90%, inside the
-- range of the good ones. Distance separates them completely, with a
-- factor-of-seven gap and nothing in it. Each of those long constraints
-- dragged the whole trajectory across the house and left it there.
--
-- 2.0 sits in that gap. The seed puts us within a metre and real corrections
-- are sub-metre, so this cannot block an honest match.
POSE_GRAPH.constraint_builder.max_constraint_distance = 2.0

-- Whole-map relocalization is what lets the pose teleport rather than drift.
-- It is also what put the rover at (-21.88, -5.73) and (-311, 88) on an
-- unseeded start. seed_pose.py asserts the parking spot instead, and that
-- has been reliable.
--
-- The cost is real: the rover can no longer find itself if it is genuinely
-- lost. Move it while it is off and you re-seed by hand. That capability has
-- never once rescued this rover, and has repeatedly ruined it.
POSE_GRAPH.global_sampling_ratio = 0.0

MAP_BUILDER.num_background_threads = 2
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 2
return options
