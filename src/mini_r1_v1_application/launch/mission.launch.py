"""Launch mission manager."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('mini_r1_v1_application')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='mini_r1_v1_application',
            executable='mission_manager_node',
            name='mission_manager_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'mission_manager.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
