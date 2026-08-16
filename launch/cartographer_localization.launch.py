import os
from launch import LaunchDescription
from launch_ros.actions import Node

CONFIG_DIR = os.path.expanduser('~/cartographer_config')
PBSTREAM = os.path.expanduser('~/house.pbstream')


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_imu_link',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
        ),
        # Required: Cartographer subscribes to /scan_fixed, not /scan.
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
            remappings=[('imu', '/imu/data'), ('scan', '/scan_fixed')],
            arguments=[
                '-configuration_directory', CONFIG_DIR,
                '-configuration_basename', 'wave_rover_localization.lua',
                '-load_state_filename', PBSTREAM,
            ],
        ),
    ])
