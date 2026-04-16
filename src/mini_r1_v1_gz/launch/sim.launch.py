import math
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    """Parse active_world.building.yaml for spawn position."""
    description_package_share = get_package_share_directory("mini_r1_v1_description")
    yaml_path = os.path.join(description_package_share, 'worlds', 'active_world.building.yaml')

    spawn_x = 0.0
    spawn_y = 0.0
    spawn_yaw = 0.0

    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # Calculate scale from measurements
            level = data['levels']['floor_0']
            measurements = level['measurements'][0]
            v1 = level['vertices'][measurements[0]]
            v2 = level['vertices'][measurements[1]]
            distance_m = measurements[2]['distance'][1]
            distance_px = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
            scale = distance_m / distance_px

            # Find spawn vertex
            for v in level['vertices']:
                if len(v) >= 4 and isinstance(v[3], str) and str(v[3]).startswith("spawn"):
                    spawn_x = v[0] * scale
                    spawn_y = -v[1] * scale
                    parts = str(v[3]).split("_")
                    if len(parts) >= 2:
                        try:
                            spawn_yaw = float(parts[1]) * math.pi / 180.0
                        except ValueError:
                            pass
                    break
    except Exception as e:
        print(f"[sim.launch.py] Could not parse building YAML for spawn: {e}")

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", 'robot_description',
                   '-name', 'mini_r1',
                   '-x', str(spawn_x),
                   '-y', str(spawn_y),
                   '-z', '0.07',
                   '-Y', str(spawn_yaw)],
        output="screen"
    )
    return [spawn_entity]


def generate_launch_description():
    description_package = "mini_r1_v1_description"
    simulation_package = "mini_r1_v1_gz"

    description_package_share = get_package_share_directory(description_package)
    simulation_package_share = get_package_share_directory(simulation_package)

    # ── Gazebo Fortress resource paths ──
    fuel_cache = os.path.expanduser(
        "~/.ignition/fuel/fuel.gazebosim.org/openrobotics/models")
    world_models = os.path.join(description_package_share, "worlds", "models")
    ros_lib = os.path.join('/opt/ros', os.environ.get('ROS_DISTRO', 'humble'), 'lib')

    ign_resource = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")
    os.environ['IGN_GAZEBO_RESOURCE_PATH'] = f"{fuel_cache}:{world_models}:{ign_resource}"

    ign_plugin = os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", "")
    os.environ['IGN_GAZEBO_SYSTEM_PLUGIN_PATH'] = f"{ros_lib}:{ign_plugin}"

    # ── Default world: generated output.world (falls back to empty.sdf) ──
    output_world = os.path.join(description_package_share, 'worlds', 'output.world')
    empty_world = os.path.join(description_package_share, 'worlds', 'empty.sdf')
    default_world = output_world if os.path.exists(output_world) else empty_world
    world_path = LaunchConfiguration('world')

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(description_package_share, "launch", "rsp.launch.py")]
        ),
        launch_arguments={"use_sim_time": "true", 'use_control': "false"}.items()
    )

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory("ros_gz_sim"),
                          'launch', 'gz_sim.launch.py')]
        ),
        launch_arguments={"gz_args": ['-r ', world_path],
                          'on_exit_shutdown': 'true'}.items(),
    )

    stamper = Node(
        package="twist_stamper",
        executable="twist_stamper",
        remappings=[
            ('cmd_vel_in', 'cmd_vel'),
            ('cmd_vel_out', 'cmd_vel_stamped'),
        ],
    )

    bridge_params = os.path.join(
        get_package_share_directory(simulation_package), 'config', 'ros_gz_bridge.yaml')
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}']
    )

    rviz_config = os.path.join(simulation_package_share, 'rviz', 'sim.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    launch_args = [
        DeclareLaunchArgument(
            name="world",
            default_value=default_world,
            description="Absolute path to the world SDF file"
        )
    ]

    return LaunchDescription([
        *launch_args,
        rsp,
        stamper,
        gz,
        ros_gz_bridge,
        rviz_node,
        OpaqueFunction(function=launch_setup),
    ])
