import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.substitutions import Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('mini_r1_v1_round3')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    urdf_path = os.path.join(pkg_share, 'urdf', 'mini_r1.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml')
    ekf_config = os.path.join(pkg_share, 'config', 'ekf.yaml')
    slam_config = os.path.join(pkg_share, 'config', 'slam_toolbox.yaml')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    apriltag_config = os.path.join(pkg_share, 'config', 'apriltag.yaml')
    rviz_config = os.path.join(pkg_share, 'rviz', 'round3.rviz')

    grid_world_root = os.environ.get(
        'GRID_WORLD_ROOT',
        '/home/tanay/ros2_ws/grid_world_hackathon/gaws_ws/src/grid_world',
    )
    grid_world_worlds = os.path.join(grid_world_root, 'worlds')
    grid_world_parent = os.path.dirname(grid_world_root)
    textures_dir = os.path.join(grid_world_root, 'materials', 'textures')
    world_file = os.path.join(grid_world_worlds, 'grid_world_FINAL.sdf')
    resource_path = f'{grid_world_parent}:{grid_world_root}:{grid_world_worlds}:' \
                    + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')

    # Set both Fortress (IGN) and Harmonic (GZ) resource path env vars so the
    # world SDF and its model:// references resolve regardless of which backend
    # ros_gz_sim selects at runtime.
    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH', value=resource_path)
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH', value=resource_path)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r {world_file}',
            'use_sim_time': 'true',
        }.items()
    )

    robot_description = Command(['xacro ', urdf_path])

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'mini_r1',
            '-topic', 'robot_description',
            '-x', '-1.35',
            '-y', '1.80',
            '-z', '0.1',
            '-Y', '0.0',
        ],
        parameters=[{'use_sim_time': True}],
    )

    QUIET = ['--ros-args', '--log-level', 'WARN']

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': True,
        }],
    )

    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        output='log',
        arguments=['0', '0', '0', '0', '0', '0', 'LIDAR', 'mini_r1/base_link/lidar'],
        parameters=[{'use_sim_time': True}],
    )

    static_tf_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_cam',
        output='log',
        # REP-103 body->optical rotation so the camera frame stamped on
        # /r1_mini/camera/* images matches the optical convention
        # (x-right, y-down, z-forward) that apriltag_ros and the
        # deprojection in logo_detector_node both assume. Without this,
        # points and tag poses get the wrong rotation and float in air.
        # Args: x y z yaw pitch roll parent child.
        arguments=['0', '0', '0', '-1.5707963', '0', '-1.5707963', 'CAM', 'mini_r1/base_link/camera'],
        parameters=[{'use_sim_time': True}],
    )

    static_tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_imu',
        output='log',
        arguments=['0', '0', '0', '0', '0', '0', 'IMU', 'mini_r1/base_link/imu'],
        parameters=[{'use_sim_time': True}],
    )

    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config, {'use_sim_time': True}],
    )

    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_config, {'use_sim_time': True}],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params,
            'autostart': 'true',
            'log_level': 'warn',
        }.items()
    )

    apriltag = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node',
        output='log',
        parameters=[apriltag_config, {'use_sim_time': True}],
        arguments=QUIET,
        remappings=[
            ('/image_rect', '/r1_mini/camera/image_raw'),
            ('/camera_info', '/r1_mini/camera/camera_info'),
            ('/detections', '/apriltag/detections'),
        ],
    )

    tile_detector = Node(
        package='mini_r1_v1_round3',
        executable='tile_detector_node',
        name='tile_detector_node',
        output='log',
        parameters=[{
            'use_sim_time': True,
            'textures_dir': textures_dir,
        }],
        arguments=QUIET,
    )

    tag_command = Node(
        package='mini_r1_v1_round3',
        executable='tag_command_node',
        name='tag_command_node',
        output='log',
        parameters=[{'use_sim_time': True}],
        arguments=QUIET,
    )

    mission_manager = Node(
        package='mini_r1_v1_round3',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'arena_min_x': -1.80,
            'arena_min_y': -2.25,
            'arena_max_x': 1.80,
            'arena_max_y': 2.25,
            'sweep_stride_m': 0.9,
            'sweep_margin_m': 0.0,
            'grid_mode': True,
            'grid_cell_size_m': 0.9,
            'grid_cell_dwell_s': 0.5,
        }],
    )

    nav_diagnostic = Node(
        package='mini_r1_v1_round3',
        executable='nav_diagnostic_node',
        name='nav_diagnostic_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    logo_detector = Node(
        package='mini_r1_v1_round3',
        executable='logo_detector_node',
        name='logo_detector_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'map_frame': 'odom',
        }],
        arguments=QUIET,
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config, '--ros-args', '--log-level', 'ERROR'],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        set_resource_path,
        set_gz_resource_path,
        gz_sim,
        rsp,
        static_tf_lidar,
        static_tf_cam,
        static_tf_imu,
        TimerAction(period=2.0, actions=[bridge]),
        TimerAction(period=3.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[ekf]),
        TimerAction(period=5.0, actions=[slam]),
        TimerAction(period=6.0, actions=[nav2]),
        TimerAction(period=7.0, actions=[apriltag, tile_detector, tag_command, mission_manager, nav_diagnostic, logo_detector, rviz_node]),
    ])
