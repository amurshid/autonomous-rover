#!/bin/bash
# Relocalize the rover, from one camera frame, wherever it happens to be.
# Run AFTER the lidar and cartographer_localization launches are up.
#
# Falls back to the marked spot in the work room (x=2.276 y=8.183
# yaw=-110.1 deg) only if the camera cannot place the rover -- which needs
# someone to have physically put it there.

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

python3 ~/set_initial_pose.py &
BRIDGE_PID=$!

# Give the bridge time to connect to /finish_trajectory and /start_trajectory
sleep 5

# Work out where the rover actually is from one camera frame, instead of
# asserting the marked spot and requiring a human to have put it there.
# Measured 2026-08-24: parked at the entrance, the pose below was 5.66 m and
# 6.8 deg wrong; VPR published the correct one in 3 s and Cartographer's scan
# matcher then held it to within 4 cm over the following minute.
#
# --sim-floor rejects frames whose best match is poor, which is how the rover
# being outside recorded coverage is detected -- retrieval cannot answer such a
# query, only decline it. --timeout guarantees the script is never blocked by a
# gate that declines everything.
# ~/vpr-venv/bin/python, not python3: torch is installed in the venv, while
# rclpy comes from apt -- the venv was created with --system-site-packages so
# that one interpreter can see both. Plain python3 has no torch.
OMP_NUM_THREADS=2 ~/vpr-venv/bin/python ~/vpr_relocalise.py --bundle ~/bundle \
        --publish --timeout 60 --sim-floor 0.75 --period 1.5
VPR_STATUS=$?

if [ $VPR_STATUS -ne 0 ]; then
    # No confirmed fix: somewhere uncovered, or lighting the database does not
    # contain. Fall back to the marked spot, which needs the rover to be
    # physically on it.
    echo "VPR found no fix (exit $VPR_STATUS) — falling back to the marked spot."
    ros2 topic pub --once /initialpose geometry_msgs/PoseWithCovarianceStamped \
    '{header: {frame_id: "map"},
      pose: {pose: {position: {x: 2.276, y: 8.183, z: 0.0},
                    orientation: {x: 0.0, y: 0.0, z: -0.8196, w: 0.5730}}}}'
fi

# ROLLBACK: delete everything from "Work out where" to here and restore:
#   ros2 topic pub --once /initialpose geometry_msgs/PoseWithCovarianceStamped \
#   '{header: {frame_id: "map"},
#     pose: {pose: {position: {x: 2.276, y: 8.183, z: 0.0},
#                   orientation: {x: 0.0, y: 0.0, z: -0.8196, w: 0.5730}}}}'

echo "Pose sent. Bridge running as PID $BRIDGE_PID — Ctrl-C to stop."
wait $BRIDGE_PID
