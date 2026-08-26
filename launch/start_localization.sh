#!/bin/bash
# Relocalize to the marked spot in the work room.
# Run AFTER the lidar and cartographer_localization launches are up.
#   x=2.276  y=8.183  yaw=-110.1 deg (-1.9216 rad)

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

python3 ~/set_initial_pose.py &
BRIDGE_PID=$!

# Give the bridge time to connect to /finish_trajectory and /start_trajectory
sleep 5

ros2 topic pub --once /initialpose geometry_msgs/PoseWithCovarianceStamped \
'{header: {frame_id: "map"},
  pose: {pose: {position: {x: 2.276, y: 8.183, z: 0.0},
                orientation: {x: 0.0, y: 0.0, z: -0.8196, w: 0.5730}}}}'

echo "Pose sent. Bridge running as PID $BRIDGE_PID — Ctrl-C to stop."
wait $BRIDGE_PID
