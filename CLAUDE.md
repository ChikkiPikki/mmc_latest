# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mini R1 v1 — autonomous robot for hackathon arena exploration. ROS 2 Humble + Gazebo Fortress, running inside Docker.

## Build & Run

Everything runs in Docker. Host has no ROS installation required.

```bash
# Build image (first time / after Dockerfile changes)
./dev.sh build

# Run interactive container (mounts ./src live)
./dev.sh run

# Inside container: rebuild after code changes
source /opt/ros/humble/setup.bash
cd /home/dev/ros2_ws
colcon build --symlink-install

# Launch sim (inside container)
source install/setup.bash
ros2 launch mini_r1_v1_gz sim.launch.py

# Launch with warehouse world
ros2 launch mini_r1_v1_gz sim.launch.py world:=$(ros2 pkg prefix mini_r1_v1_description)/share/mini_r1_v1_description/worlds/warehouse.sdf
```

## Architecture

### Packages (under src/)

- **mini_r1_v1_description** — Robot URDF (xacro), STL meshes, world SDF files, joint config. The URDF includes DiffDrive, JointStatePublisher, GPU LiDAR, camera, and IMU sensor plugins.
- **mini_r1_v1_gz** — Gazebo launch file (`sim.launch.py`), ros_gz_bridge config, twist_stamper. This is the entry point for simulation.

### Legacy Reference (legacy_src/)

Contains the original Jazzy+Harmonic codebase. Has `COLCON_IGNORE` — not built. Use as reference for:
- World generation scripts: `legacy_src/mini_r1_v1_description/worlds/generate_markers_and_signs.py`
- Dynamic obstacle node: `legacy_src/mini_r1_v1_application/scripts/dynamic_obstacle_node.py`
- Navigation state machine: `legacy_src/mini_r1_v1_application/scripts/maze_navigator_node.py` + `nav_lib/`
- Behavior config: `legacy_src/mini_r1_v1_application/config/behavior_config.yaml`
- EKF fusion config: `legacy_src/mini_r1_v1_bringup/config/ekf.yaml`
- Full bringup chain: `legacy_src/mini_r1_v1_bringup/launch/bringup.launch.py`

### Gazebo Fortress vs Harmonic

This project uses **Gazebo Fortress** (Ignition). Key differences from the legacy Harmonic codebase:
- Plugin filenames: `ignition-gazebo-*-system` (not `gz-sim-*-system`)
- Plugin class names: `ignition::gazebo::systems::*` (not `gz::sim::systems::*`)
- Bridge message types: `ignition.msgs.*` (same as legacy bridge config)
- Environment vars: `IGN_GAZEBO_RESOURCE_PATH` / `IGN_GAZEBO_SYSTEM_PLUGIN_PATH` (not `GZ_SIM_*`)
- Fuel model URLs: `https://fuel.ignitionrobotics.org/...` (not `https://fuel.gazebosim.org/...`)

### Robot Specs

- 4-wheel differential drive, wheel separation 0.2885m, wheel radius 0.065m
- GPU LiDAR: 360 samples, 10Hz, 0.12–30m range, topic `/r1_mini/lidar`
- Camera: 640x480 RGB, 30Hz, topic `/r1_mini/camera`
- IMU: 100Hz, topic `/r1_mini/imu`
- Odometry: 50Hz from DiffDrive plugin, topic `/r1_mini/odom`
- Control: `/cmd_vel` (Twist) or `/cmd_vel_stamped` (TwistStamped)

### Topic Map (Gazebo ↔ ROS via ros_gz_bridge)

Configured in `src/mini_r1_v1_gz/config/ros_gz_bridge.yaml`. Sensors publish on `ignition.msgs.*` types bridged to standard ROS message types.
