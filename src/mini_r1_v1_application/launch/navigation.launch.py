"""Launch sweep planner, RRT planner, and pursuit controller."""

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

        # ── Coverage tracker ──
        Node(
            package='mini_r1_v1_application',
            executable='coverage_tracker_node',
            name='coverage_tracker_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'coverage_tracker.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),

        # ── Sweep planner ──
        Node(
            package='mini_r1_v1_application',
            executable='sweep_planner_node',
            name='sweep_planner_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'sweep_planner.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),

        # ── RRT planner ──
        Node(
            package='mini_r1_v1_application',
            executable='rrt_planner_node',
            name='rrt_planner_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'rrt_planner.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),

        # ── Pursuit controller ──
        Node(
            package='mini_r1_v1_application',
            executable='pursuit_controller_node',
            name='pursuit_controller_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'pursuit_controller.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),
    ])
