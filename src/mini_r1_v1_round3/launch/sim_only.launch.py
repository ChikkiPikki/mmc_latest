import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('mini_r1_v1_round3')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    urdf_path = os.path.join(pkg_share, 'urdf', 'mini_r1.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml')
    rviz_config = os.path.join(pkg_share, 'rviz', 'round3.rviz')

    grid_world_root = os.environ.get(
        'GRID_WORLD_ROOT',
        '/home/dev/grid_world_hackathon/gaws_ws/src/grid_world',
    )
    grid_world_worlds = os.path.join(grid_world_root, 'worlds')
    grid_world_parent = os.path.dirname(grid_world_root)
    world_file = 'grid_world_FINAL.sdf'

    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock'
    )
    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='false',
        description='Launch RViz'
    )

    set_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=f'{grid_world_parent}:{grid_world_root}:{grid_world_worlds}:'
              + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    )

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
            'use_sim_time': use_sim_time,
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
        parameters=[{'use_sim_time': use_sim_time}],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': use_sim_time,
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    def maybe_rviz(context, *args, **kwargs):
        if context.launch_configurations.get('rviz', 'false').lower() in ('true', '1', 'yes'):
            return [rviz_node]
        return []

    return LaunchDescription([
        declare_use_sim_time,
        declare_rviz,
        set_resource_path,
        gz_sim,
        rsp,
        bridge,
        TimerAction(period=3.0, actions=[spawn_robot]),
        OpaqueFunction(function=maybe_rviz),
    ])
