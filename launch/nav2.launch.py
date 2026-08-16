import os
from launch import LaunchDescription
from launch_ros.actions import Node

HOME = os.path.expanduser('~')
PARAMS = os.path.join(HOME, 'nav2_params.yaml')

LIFECYCLE = ['map_server', 'planner_server',
             'controller_server', 'behavior_server', 'bt_navigator']


def generate_launch_description():
    common = {'use_sim_time': False}

    return LaunchDescription([
        Node(package='nav2_map_server', executable='map_server',
             name='map_server', output='screen',
             parameters=[PARAMS, common]),

        Node(package='nav2_planner', executable='planner_server',
             name='planner_server', output='screen',
             parameters=[PARAMS, common]),

        Node(package='nav2_controller', executable='controller_server',
             name='controller_server', output='screen',
             parameters=[PARAMS, common]),

        Node(package='nav2_behaviors', executable='behavior_server',
             name='behavior_server', output='screen',
             parameters=[PARAMS, common]),

        Node(package='nav2_bt_navigator', executable='bt_navigator',
             name='bt_navigator', output='screen',
             parameters=[PARAMS, common]),

        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': False,
                          'autostart': True,
                          'node_names': LIFECYCLE}]),
    ])
