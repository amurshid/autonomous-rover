# Not this repo's workspace

`ugv_imu_bridge/scan_timestamp_relay.py` lives on the rover at

    ~/ros2_ws/src/ros2_imu_bridge/ugv_imu_bridge/scan_timestamp_relay.py

and is tracked here because it turned out to be the root of a fault that took
a full day to find. Run the offline test before deploying a change -- it
simulates the LD19 driver's publish loop and replays Cartographer's drop rule
over the relay's output:

    python3 ugv_imu_bridge/test_scan_timestamp_relay.py

Editing it needs a rebuild:

    colcon build --packages-select ugv_imu_bridge
    sudo systemctl restart rover-cartographer

The relay is launched by `cartographer_localization.launch.py`, so restarting
Cartographer restarts it.
