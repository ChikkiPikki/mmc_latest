"""Top-level bringup — launches all application subsystems.

Usage:
    ros2 launch mini_r1_v1_application bringup.launch.py use_sim_time:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory('mini_r1_v1_application')
    launch_dir = os.path.join(pkg_dir, 'launch')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # ── Infrastructure (EKF + costmap) ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'infrastructure.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # ── Navigation (coverage + sweep + rrt + pursuit) ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'navigation.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # ── Perception (aruco + symbol detection) ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'perception.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # ── Mission manager ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'mission.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # ── Dynamic obstacles ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_dir, 'dynamic_obstacles.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
    ])
