#!/bin/bash
# Dump every fact needed to diagnose the rover, for pasting back.
# Run with a mode active. Regenerate the manifest from the Mac when
# repo files change -- the shas below are a snapshot, not live.
cat >/dev/shm/rover.manifest <<'MANIFEST'
06403e1003d14166  /home/amurshid/odom_publisher.py
81ad1aca9305fd1a  /home/amurshid/patrol.py
54be7815fb36dc99  /home/amurshid/rooms.py
52e530a0916aff7e  /home/amurshid/rover_ai.py
23488d5f079a9c99  /home/amurshid/rover_goto.py
8d1f7b32a9e9463c  /home/amurshid/rover_health.py
f4d6677016633309  /home/amurshid/rover_mode_web.py
890fe862348c13a7  /home/amurshid/rover_motions.py
853ebeae7d39678c  /home/amurshid/rover_nav.py
904e2eaebe6cb35c  /home/amurshid/rover_pose_memory.py
78f0d8e499361378  /home/amurshid/rover_serial_peek.py
e03be623fd4453ce  /home/amurshid/rover_teleop_web.py
acb9c4b3340eda9c  /home/amurshid/rover_voice.py
8d400af5d7c40288  /home/amurshid/seed_pose.py
bf363702f18ee9d2  /home/amurshid/set_initial_pose.py
38cf1ef62990d3c1  /home/amurshid/tf_watch.py
4fce68c48e652846  /home/amurshid/voice_chat.py
569a174b7405e3a2  /home/amurshid/vpr_logger.py
6046c74d442e92cd  /home/amurshid/vpr_relocalise.py
732e8e8ac9994480  /home/amurshid/wait_for_localisation.py
fc281fa9867d085e  /home/amurshid/wave_rover_bridge.py
dadcce9f86c7be08  /home/amurshid/cartographer_localization.launch.py
a0a024374f32d574  /home/amurshid/cartographer.launch.py
ee0695f0ca94c100  /home/amurshid/nav2.launch.py
ccf7f5485e2f19ea  /home/amurshid/start_localization.sh
e47eee78ee84143c  /home/amurshid/nav2_params.yaml
48fc8f42d7ebc16c  /home/amurshid/cartographer_config/wave_rover_localization.lua
dc3597710688e05d  /home/amurshid/cartographer_config/wave_rover.lua
47fe2576b3698341  /etc/systemd/system/rover-ai.service
250a68122bc3d850  /etc/systemd/system/rover-bridge.service
ee0a2fca16add8b2  /etc/systemd/system/rover-camera.service
121dd2565ba9e88b  /etc/systemd/system/rover-cartographer.service
c48a88598cfc7b3b  /etc/systemd/system/rover-initialpose.service
565b3d7e93bdd146  /etc/systemd/system/rover-lidar.service
b74a6621c0f6b3d0  /etc/systemd/system/rover-mode.service
84d39b7589aad322  /etc/systemd/system/rover-nav2.service
6de7dbf00cc127f0  /etc/systemd/system/rover-posememory.service
229ccecc16e5eb1f  /etc/systemd/system/rover-relocalise.service
ca46cf23718bcea2  /etc/systemd/system/rover-seedpose.service
d3faa7e97bddfd58  /etc/systemd/system/rover-teleop.service
126260f34b4e80e9  /etc/systemd/system/rover-common.target
18f7c3a4e8cf6a82  /etc/systemd/system/rover-teleop.target
0866c42eea287b68  /etc/systemd/system/rover.target
MANIFEST
echo "══════ FILES THAT DIFFER FROM THE REPO"
ok=0
while read -r want path; do
  b=$(basename "$path")
  if [ ! -e "$path" ]; then echo "  MISSING  $b"; continue; fi
  got=$(sha256sum "$path" 2>/dev/null | cut -c1-16)
  [ "$got" = "$want" ] && ok=$((ok+1)) || printf '  DIFFERS  %-34s pi=%s repo=%s\n' "$b" "$got" "$want"
done </dev/shm/rover.manifest
echo "  ($ok files match)"

echo "══════ BRANCH CONTAMINATION"
grep -l 'amcl\|laser_scan_matcher\|emcl' ~/nav2_params.yaml ~/*.launch.py 2>/dev/null || echo "  clean"

echo "══════ UNITS"
systemctl is-active rover-bridge rover-lidar rover-cartographer rover-initialpose \
  rover-seedpose rover-posememory rover-nav2 rover-mode 2>&1 | paste -sd' ' -

echo "══════ ROS RUNTIME"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/ros2_ws/install/setup.bash 2>/dev/null
echo "-- scan rates (want /scan and /scan_fixed both ~10Hz):"
for t in /scan /scan_fixed; do
  printf '   %-12s %s\n' "$t" "$(timeout 6 ros2 topic hz $t 2>/dev/null | grep -m1 average || echo 'SILENT')"
done
echo "-- /odom:"
printf '   %s\n' "$(timeout 6 ros2 topic hz /odom 2>/dev/null | grep -m1 average || echo 'SILENT -- nothing publishes it')"
echo "-- who publishes /cmd_vel (more than one is a problem):"
ros2 topic info /cmd_vel 2>/dev/null | grep -i publisher
echo "-- TF chain:"
timeout 4 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 -A1 'Translation' || echo "   map->base_link MISSING"
timeout 4 ros2 run tf2_ros tf2_echo odom base_link 2>/dev/null | grep -m1 -A1 'Translation' || echo "   odom->base_link MISSING"
echo "-- localisation verdict:"
cat /run/rover/pose_status.json 2>/dev/null; echo
echo "══════ CARTOGRAPHER CONFIG ACTUALLY LOADED"
ls -l ~/cartographer_config/
