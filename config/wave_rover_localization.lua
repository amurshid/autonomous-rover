-- Pure localization against a prebuilt house.pbstream.
-- Inherits everything from the mapping config so scans are interpreted exactly as
-- they were when the map was built. Only localization behavior differs.

include "wave_rover.lua"

-- Localize against the loaded map instead of extending it.
-- Keeps only the few most recent submaps so memory stays bounded.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- Optimize every 90 nodes, as the rover ran from 2026-08-26 (ffba692) until the
-- rollback on 2026-09-10. Every 20 was too much for the Pi: with a node every
-- 0.5 deg of turn that meant ~15 whole-graph optimizations a minute while
-- driving, and the constraint queue reached 749.
POSE_GRAPH.optimize_every_n_nodes = 90

-- Keep matching new nodes against the saved map for 300 s after the last
-- constraint that tied this trajectory to it, not the default 10 s. That tie is
-- only renewed when a constraint batch finishes, i.e. at an optimization, so
-- with optimizations 19 s apart (median, 90 nodes while driving) the default
-- expired between every batch: on 2026-09-10 a whole run made zero saved-map
-- matches and drifted until it overshot rooms. The same expiry ended matching
-- mid-run at 20 nodes whenever the queue backed up or the rover slowed. 300 s
-- also covers a parked rover, whose 90 nodes can take over two minutes.
POSE_GRAPH.global_constraint_search_after_n_seconds = 300.

-- Matching against the saved map is what heats the Pi: Cartographer went from
-- 88% CPU (90 nodes, not matching) to 179% (90 nodes, matching) and the Pi to
-- 84 C, and there is no room in the chassis for a cooler. So match cheaply.
-- The seeded, tracked pose is rarely more than ~1 m off, so a 7 m / 30 deg
-- search per candidate is mostly wasted; 2 m / 20 deg still covers the worst
-- drift seen. Sampling a third as many node-submap pairs, and only submaps
-- within 4 m, cuts the number of searches. Whole-map (global) searches ignore
-- these windows and are unaffected.
POSE_GRAPH.constraint_builder.sampling_ratio = 0.1
POSE_GRAPH.constraint_builder.max_constraint_distance = 4.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 2.0
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(20.)

-- Publish the pose and TF at 50 Hz rather than 200 Hz. Nav2's controller runs
-- at 10 Hz, and every TF listener -- five Nav2 servers and the pose memory --
-- deserialises each message whether it needs it or not.
options.pose_publish_period_sec = 20e-3

-- Leave two of the Pi's four cores for Nav2 and the drivers (as in ffba692).
MAP_BUILDER.num_background_threads = 2
POSE_GRAPH.optimization_problem.ceres_solver_options.num_threads = 2

return options
