-- Pure localization against a prebuilt house.pbstream.
-- Inherits everything from the mapping config so scans are interpreted exactly as
-- they were when the map was built. Only localization behavior differs.

include "wave_rover.lua"

-- Localize against the loaded map instead of extending it.
-- Keeps only the few most recent submaps so memory stays bounded.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- Optimize more often than mapping (was 35) for responsive pose updates.
POSE_GRAPH.optimize_every_n_nodes = 20

return options
