"""AMCL localisation, as a drop-in alternative to cartographer_localization.

Why this exists
---------------
Cartographer pure localisation keeps building live submaps from live scans
(pure_localization_trimmer, max_submaps_to_keep = 3). When the pose is
knocked loose, new scans get inserted at the wrong place, the frontend then
matches those wrong submaps perfectly, and the rover ends up confidently
localised in geometry it built for itself. Restarting is the only known cure,
because a restart is the only thing that deletes those submaps.

AMCL never builds a map. Every update scores particles against the static
house_map.yaml, so there is nothing self-referential to lock onto, and
augmented MCL (recovery_alpha_* in amcl_params.yaml) injects random particles
when the match degrades -- recovering without a process restart.

Odometry
--------
The WAVE ROVER bridge is open loop: it sends {"T":1,"L":..,"R":..} and gets
back only voltage and IMU. No wheel feedback exists, so AMCL's motion model
is fed by laser_scan_matcher instead.

That is a real compromise, not a free win. The scan matcher and AMCL's
update step are the same sensor, so both go blind along the same axis at the
same moment -- and this house is mostly long corridors, where lidar cannot
see along-axis translation at all. The mitigation is the IMU: feed the gyro
in as the rotation prior so the weak axis is not purely laser-derived. Left
out of this first version deliberately, to change one thing at a time.

Drift in the scan matcher does not matter. It only has to be locally smooth;
AMCL owns map->odom and corrects global position on every update.

Running it
----------
Cartographer must be stopped -- both publish odom->base_link and would
fight. map_server is NOT started here; it already runs under nav2.launch.py's
lifecycle manager, and AMCL just subscribes to its latched /map.

    systemctl --user stop rover-cartographer
    ros2 launch amcl_localization.launch.py
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node

HOME = os.path.expanduser('~')
AMCL_PARAMS = os.path.join(HOME, 'amcl_params.yaml')


def generate_launch_description():
    return LaunchDescription([
        # Same relay Cartographer used. A scan matcher is more sensitive to
        # bad stamps than a costmap is, so this is not optional here.
        Node(
            package='ugv_imu_bridge',
            executable='scan_timestamp_relay',
            name='scan_timestamp_relay',
            output='screen',
        ),

        # Odometry from consecutive scans (PLICP via Censi's csm).
        # Publishes /odom and, unlike under Cartographer, owns odom->base_link.
        #
        # Parameter names read out of the node's own source rather than the
        # README, which documents only two of them:
        #   grep -n add_parameter src/ros2_laser_scan_matcher/src/*.cpp
        Node(
            package='ros2_laser_scan_matcher',
            executable='laser_scan_matcher',
            name='laser_scan_matcher',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                # The one that must be set. Default is "laser"; this rover's
                # lidar frame is base_laser, and with the default the node
                # cannot resolve base->laser and silently skips every scan.
                'laser_frame': 'base_laser',
                'base_frame': 'base_link',
                'odom_frame': 'odom',
                # Default is "" which DISABLES odometry publication entirely.
                'publish_odom': '/odom',
                'publish_tf': True,

                # Keyframe thresholds: below these the matcher holds its
                # reference scan instead of re-matching, which is what keeps
                # a parked rover from integrating range noise into phantom
                # motion. Both at their defaults, stated explicitly because
                # they are the first thing to reach for if a stationary
                # rover drifts.
                'kf_dist_linear': 0.10,
                'kf_dist_angular': 0.174533,      # 10 degrees

                # How far one match is allowed to move the estimate -- the
                # nearest equivalent to Cartographer's search window, and
                # the bound that stops a single bad match teleporting the
                # pose. Defaults; tighten if jumps appear.
                'max_linear_correction': 0.5,
                'max_angular_correction_deg': 45.0,
                'max_iterations': 10,

                # Off by default. AMCL does not read odom covariance -- it
                # uses its own alpha* motion noise -- so this stays off
                # until something downstream (robot_localization, if the IMU
                # gets fused later) actually needs it.
                'do_compute_covariance': 0,
            }],
            # It reads base_link -> base_laser off /tf, which the LD19
            # launch already publishes.
            remappings=[('scan', '/scan_fixed')],
        ),

        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[AMCL_PARAMS],
        ),

        # A manager of its own, so nav2.launch.py's list is untouched and
        # this can be started and stopped without disturbing navigation.
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['amcl'],
            }],
        ),
    ])
