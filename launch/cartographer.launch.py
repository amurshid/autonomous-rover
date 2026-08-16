import os

from launch import LaunchDescription
from launch_ros.actions import Node

CONFIG_DIR = os.path.expanduser('~/cartographer_config')


def generate_launch_description():
    return LaunchDescription([
        # base_link -> imu_link. Identity is fine for 2D (IMU is mounted flat,
        # Z up). args: x y z yaw pitch roll parent child
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_imu_link',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
        ),
        # NOTE: base_link -> base_laser is published by the LD19 launch
        # (z=0.18). Do NOT publish it here too — it would conflict.

        # Re-stamp the LD19 /scan with monotonic timestamps -> /scan_fixed.
        # Fixes the driver's backwards-going stamps that make Cartographer
        # drop whole scans ("time is not before" / "Dropped ... points").
        Node(
            package='ugv_imu_bridge',
            executable='scan_timestamp_relay',
            name='scan_timestamp_relay',
            output='screen',
        ),

        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            # Cartographer 'imu' -> our IMU topic; 'scan' -> re-stamped scan
            remappings=[('imu', '/imu/data'), ('scan', '/scan_fixed')],
            arguments=[
                '-configuration_directory', CONFIG_DIR,
                '-configuration_basename', 'wave_rover.lua',
            ],
        ),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{'use_sim_time': False}],
            arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
        ),
    ])

