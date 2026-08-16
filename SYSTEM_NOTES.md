# System changes not captured in files

- `sudo usermod -aG dialout amurshid`
- `sudo systemctl mask serial-getty@ttyS0.service`
- Removed `console=serial0,115200` from /boot/firmware/cmdline.txt
- Copy ldlidar.rules to /etc/udev/rules.d/ then `sudo udevadm control --reload`
- apt: ros-humble-cartographer, ros-humble-cartographer-ros,
  ros-humble-navigation2, ros-humble-nav2-bringup, ros-humble-foxglove-bridge
- pip: pyserial

## Rebuild
cd ~/ldlidar_ros2_ws && colcon build --symlink-install
cd ~/ros2_ws && colcon build --symlink-install

## Calibration values (measured)
- straight deadband: 0.05, spin deadband: 0.18 (bare floor; carpet higher, unmeasured)
- wheel_separation: 0.15 (spec sheet, not calibrated)
- max_wheel_speed: 1.25 (spec sheet, optimistic — real value likely ~0.85)
- ESP32: /dev/serial0 @ 115200, JSON {"T":1,"L":x,"R":y}, range ±0.5, 3s watchdog
## Nav2 rotation: the acceleration-clamp trap
RPP clamps angular commands to (current_velocity + max_angular_accel × 0.1).
With a large motor deadband, the rover doesn't move at low commands, odom
correctly reports ~0 velocity, and the clamp never releases — the rover
pulses forever at max_angular_accel/10 rad/s.
Fix: max_angular_accel: 200.0 so one cycle reaches the target rate.
Symptom: /cmd_vel angular.z stuck at exactly max_angular_accel/10.
Working values: rotate_to_heading_angular_vel 10.0, max_rotational_vel 10.0.
