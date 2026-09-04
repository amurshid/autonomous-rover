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

-- Stated again here even though the base config now sets it, so that a
-- stale wave_rover.lua cannot silently take the IMU away. Without it the
-- scan matcher searches a narrow window around a rotation prior that does
-- not exist -- see the note in the base config.
TRAJECTORY_BUILDER_2D.use_imu_data = true

-- Parked, the rover should barely add to its trajectory. The base config
-- creates a node at half a degree of rotation, which is right for mapping --
-- you want dense nodes while building -- and wrong here: orientation noise
-- alone pushed node ids past 246 in three minutes on a rover that had not
-- moved. Every one of those is another submap contribution and another
-- candidate for a bad constraint, and the bad constraints are what moved the
-- pose across the house. Two degrees still tracks a real turn.
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(2.0)

-- Localize against the loaded map instead of extending it.
-- Keeps only the few most recent submaps so memory stays bounded.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- Optimize more often than mapping (was 35) for responsive pose updates.
POSE_GRAPH.optimize_every_n_nodes = 90

-- Whole-map relocalization is what lets the pose teleport rather than drift,
-- and it produced (-21.88, -5.73) and (-311, 88) on unseeded starts. The
-- seed asserts the parking spot instead. The cost: the rover can no longer
-- find itself if genuinely lost -- move it while it is off and re-seed.
POSE_GRAPH.global_sampling_ratio = 0.0

MAP_BUILDER.num_background_threads = 2
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 2
return options
