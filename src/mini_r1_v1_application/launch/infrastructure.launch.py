"""Launch EKF, costmap, and static TF publishers."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    pkg_dir = get_package_share_directory('mini_r1_v1_application')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        # ── Static TF: bridge URDF frame names → Gazebo sensor frame names ──
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_lidar_bridge',
            arguments=['0', '0', '0', '0', '0', '0',
                       'LIDAR', 'mini_r1/base_link/lidar'],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_camera_bridge',
            arguments=['0', '0', '0', '0', '0', '0',
                       'CAM', 'mini_r1/base_link/camera'],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_imu_bridge',
            arguments=['0', '0', '0', '0', '0', '0',
                       'IMU', 'mini_r1/base_link/imu'],
        ),

        # ── EKF (robot_localization) ──
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'ekf.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),

        # ── Local costmap (nav2_costmap_2d lifecycle node) ──
        LifecycleNode(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d',
            name='local_costmap',
            namespace='',
            output='screen',
            parameters=[
                os.path.join(pkg_dir, 'config', 'costmap.yaml'),
                {'use_sim_time': use_sim_time}
            ],
        ),

        # ── Lifecycle manager for costmap ──
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_costmap',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': ['local_costmap'],
            }],
        ),
    ])
