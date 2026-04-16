#!/usr/bin/env python3
"""
Dynamic Obstacle Node — Fortress adaptation.

Reads the building YAML, finds dynamic_obstacle_* models and their
corresponding _high / _low waypoint vertices, then drives each obstacle
back and forth with easing motion via ign service set_pose.

Model naming convention:  dynamic_obstacle_{id}_{speed}_{turn_around}
  - id:          integer matching the _high / _low vertex suffix
  - speed:       linear speed in m/s
  - turn_around: 1 = rotate 180 deg at each endpoint, 0 = no rotation

Vertex naming convention:
  dynamic_obstacle_{id}_high   - one end of the path
  dynamic_obstacle_{id}_low    - other end of the path
"""
import math
import os
import subprocess
import yaml

import rclpy
from rclpy.node import Node


# ── Easing (smoothstep) ────────────────────────────────────────────────
def smoothstep(t: float) -> float:
    """Hermite smoothstep: ease-in-out  0->1."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# ── Helpers ─────────────────────────────────────────────────────────────
def yaw_to_quat(yaw: float):
    """Convert yaw (radians) to quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def compute_scale(vertices, measurements):
    m = measurements[0]
    v1, v2 = vertices[m[0]], vertices[m[1]]
    dist_m = m[2]["distance"][1]
    dist_px = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    return dist_px / dist_m          # px per metre


def px_to_world(px_x, px_y, scale):
    return px_x / scale, -px_y / scale


# ── Per-obstacle state machine ──────────────────────────────────────────
class ObstacleDriver:
    """State machine for one dynamic obstacle."""

    MOVING_TO_LOW = 0
    ROTATING_AT_LOW = 1
    MOVING_TO_HIGH = 2
    ROTATING_AT_HIGH = 3

    ROTATION_SPEED = 1.2   # rad/s

    def __init__(self, gz_name, high_xy, low_xy, speed, turn_around,
                 start_yaw):
        self.gz_name = gz_name
        self.high = high_xy
        self.low = low_xy
        self.speed = speed
        self.turn_around = bool(turn_around)
        self.yaw = start_yaw

        dx = self.low[0] - self.high[0]
        dy = self.low[1] - self.high[1]
        self.dist = math.hypot(dx, dy)
        self.travel_time = self.dist / self.speed if self.speed > 0 else 1.0

        self.heading_to_low = math.atan2(dy, dx)
        self.heading_to_high = self.heading_to_low + math.pi

        self.state = self.MOVING_TO_LOW
        self.yaw = self.heading_to_low
        self.elapsed = 0.0

        self.rot_target = 0.0
        self.rot_done = 0.0

    def tick(self, dt: float):
        """Advance state by dt seconds.  Returns (x, y, z, yaw)."""
        self.elapsed += dt

        if self.state == self.MOVING_TO_LOW:
            t = min(self.elapsed / self.travel_time, 1.0)
            e = smoothstep(t)
            x = self.high[0] + (self.low[0] - self.high[0]) * e
            y = self.high[1] + (self.low[1] - self.high[1]) * e
            if t >= 1.0:
                self._start_rotation(self.ROTATING_AT_LOW)
            return x, y, 0.0, self.yaw

        elif self.state == self.ROTATING_AT_LOW:
            return self._do_rotation(dt, self.MOVING_TO_HIGH,
                                     self.heading_to_high)

        elif self.state == self.MOVING_TO_HIGH:
            t = min(self.elapsed / self.travel_time, 1.0)
            e = smoothstep(t)
            x = self.low[0] + (self.high[0] - self.low[0]) * e
            y = self.low[1] + (self.high[1] - self.low[1]) * e
            if t >= 1.0:
                self._start_rotation(self.ROTATING_AT_HIGH)
            return x, y, 0.0, self.yaw

        elif self.state == self.ROTATING_AT_HIGH:
            return self._do_rotation(dt, self.MOVING_TO_LOW,
                                     self.heading_to_low)

        return self.high[0], self.high[1], 0.0, self.yaw

    def _start_rotation(self, rot_state):
        if self.turn_around:
            self.state = rot_state
            self.rot_target = math.pi
            self.rot_done = 0.0
            self.elapsed = 0.0
        else:
            next_move = (self.MOVING_TO_HIGH
                         if rot_state == self.ROTATING_AT_LOW
                         else self.MOVING_TO_LOW)
            self.state = next_move
            self.elapsed = 0.0

    def _do_rotation(self, dt, next_state, next_heading):
        step = self.ROTATION_SPEED * dt
        self.rot_done += step
        self.yaw += step

        if self.state == self.ROTATING_AT_LOW:
            pos = self.low
        else:
            pos = self.high

        if self.rot_done >= self.rot_target:
            self.yaw = next_heading
            self.state = next_state
            self.elapsed = 0.0

        return pos[0], pos[1], 0.0, self.yaw


# ── ROS 2 node ──────────────────────────────────────────────────────────
class DynamicObstacleNode(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_node')

        self.declare_parameter('world_name', 'sim_world')
        self.world_name = self.get_parameter('world_name').value

        self.drivers = []

        # Read building YAML
        from ament_index_python.packages import get_package_share_directory
        desc_share = get_package_share_directory('mini_r1_v1_description')
        yaml_path = os.path.join(desc_share, 'worlds',
                                 'active_world.building.yaml')
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            self.get_logger().info(
                f"World YAML not found: {yaml_path} -- no dynamic obstacles.")
            return
        except Exception as e:
            self.get_logger().info(
                f"Error reading world YAML: {e} -- no dynamic obstacles.")
            return

        level = data.get('levels', {}).get('floor_0', {})
        vertices = level.get('vertices', [])
        measurements = level.get('measurements', [])
        models = level.get('models', [])

        if not measurements or not vertices:
            self.get_logger().info(
                "No measurements/vertices in world YAML -- skipping.")
            return

        scale = compute_scale(vertices, measurements)

        # Collect high / low vertices per obstacle id
        waypoints = {}
        for v in vertices:
            if len(v) < 4 or not isinstance(v[3], str):
                continue
            name = v[3]
            if not name.startswith('dynamic_obstacle_'):
                continue
            parts = name.split('_')
            obs_id = parts[2]
            end = parts[3]
            wx, wy = px_to_world(v[0], v[1], scale)
            waypoints.setdefault(obs_id, {})[end] = (wx, wy)

        # Collect model definitions
        for m in models:
            name = m.get('name', '')
            if not name.startswith('dynamic_obstacle_'):
                continue
            parts = name.split('_')
            obs_id = parts[2]
            speed = float(parts[3])
            turn_around = int(parts[4])

            if obs_id not in waypoints:
                self.get_logger().warn(f"No waypoints for obstacle {obs_id}")
                continue
            wp = waypoints[obs_id]
            if 'high' not in wp or 'low' not in wp:
                self.get_logger().warn(
                    f"Missing high/low for obstacle {obs_id}")
                continue

            gz_name = name
            start_yaw = m.get('yaw', 0.0)

            driver = ObstacleDriver(
                gz_name, wp['high'], wp['low'],
                speed, turn_around, start_yaw)
            self.drivers.append(driver)
            self.get_logger().info(
                f"Obstacle '{gz_name}': speed={speed} m/s, "
                f"turn={turn_around}, dist={driver.dist:.2f} m, "
                f"travel_time={driver.travel_time:.2f} s")

        if not self.drivers:
            self.get_logger().info("No dynamic obstacles in this world.")
            return

        self.dt = 1.0 / 10.0
        self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(
            f"DynamicObstacleNode started with {len(self.drivers)} obstacle(s).")

    def timer_callback(self):
        for drv in self.drivers:
            x, y, z, yaw = drv.tick(self.dt)
            self._set_pose(drv.gz_name, x, y, z, yaw)

    def _set_pose(self, model_name, x, y, z, yaw):
        """Set model pose via ign service CLI (Fortress)."""
        qx, qy, qz, qw = yaw_to_quat(yaw)
        req = (
            f'name: "{model_name}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'
        )
        try:
            subprocess.run(
                ['ign', 'service',
                 '-s', f'/world/{self.world_name}/set_pose',
                 '--reqtype', 'ignition.msgs.Pose',
                 '--reptype', 'ignition.msgs.Boolean',
                 '--req', req,
                 '--timeout', '50'],
                capture_output=True,
                timeout=1.0
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.get_logger().warn(f"set_pose failed: {e}", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
