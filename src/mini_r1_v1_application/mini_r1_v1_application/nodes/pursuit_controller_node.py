"""Pure pursuit path-following controller with LiDAR speed scaling and stuck detection."""

import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from mini_r1_v1_application.utils.stuck_detector import StuckDetector


class PursuitControllerNode(Node):
    def __init__(self):
        super().__init__('pursuit_controller_node')

        # Parameters
        self.declare_parameter('lookahead_m', 0.20)
        self.declare_parameter('max_linear_vel', 0.56)
        self.declare_parameter('max_angular_vel', 1.5)
        self.declare_parameter('goal_tolerance_m', 0.15)
        self.declare_parameter('min_obstacle_dist_m', 0.25)
        self.declare_parameter('slowdown_dist_m', 0.60)
        self.declare_parameter('stuck_timeout_s', 5.0)
        self.declare_parameter('stuck_displacement_m', 0.05)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('path_collision_check', True)
        self.declare_parameter('obstacle_threshold', 65)

        self.lookahead = self.get_parameter('lookahead_m').value
        self.max_linear = self.get_parameter('max_linear_vel').value
        self.max_angular = self.get_parameter('max_angular_vel').value
        self.goal_tol = self.get_parameter('goal_tolerance_m').value
        self.min_obs_dist = self.get_parameter('min_obstacle_dist_m').value
        self.slowdown_dist = self.get_parameter('slowdown_dist_m').value
        self.path_collision_check = self.get_parameter('path_collision_check').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        control_rate = self.get_parameter('control_rate_hz').value

        # Stuck detector
        self.stuck = StuckDetector(
            window_s=self.get_parameter('stuck_timeout_s').value,
            displacement_m=self.get_parameter('stuck_displacement_m').value)

        # State
        self.path = None
        self.path_idx = 0
        self.odom = None
        self.min_front_range = float('inf')
        self.costmap = None
        self.costmap_info = None
        self.state = 'NO_PATH'

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # Subscribers
        self.create_subscription(Path, '/planned_path', self._path_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)
        self.create_subscription(LaserScan, '/r1_mini/lidar', self._lidar_cb, sensor_qos)
        self.create_subscription(OccupancyGrid, '/costmap/costmap', self._costmap_cb, costmap_qos)

        # Trajectory tracking
        self.trajectory = Path()
        self.trajectory.header.frame_id = 'odom'
        self._last_traj_x = None
        self._last_traj_y = None
        self._traj_min_dist = 0.05  # only add point if moved 5cm

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/controller/status', 10)
        self.traj_pub = self.create_publisher(Path, '/robot_trajectory', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/pursuit/lookahead', 10)

        # Control timer
        self.create_timer(1.0 / control_rate, self._control_tick)

        self.get_logger().info(
            f'Pursuit controller: lookahead={self.lookahead}m, '
            f'max_v={self.max_linear}, max_w={self.max_angular}')

    # ── Callbacks ──────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        if len(msg.poses) == 0:
            self.path = None
            self.state = 'NO_PATH'
            self._stop()
            return
        # Interpolate sparse RRT path into dense waypoints (every 5cm)
        self.path = self._interpolate_path(msg.poses, spacing=0.05)
        self.path_idx = 0
        self.stuck.reset()
        self.state = 'TRACKING'
        self.get_logger().info(
            f'New path received: {len(msg.poses)} raw → {len(self.path)} interpolated')

    def _odom_cb(self, msg: Odometry):
        self.odom = msg
        # Accumulate trajectory breadcrumbs
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if (self._last_traj_x is None or
                math.hypot(x - self._last_traj_x, y - self._last_traj_y) >= self._traj_min_dist):
            ps = PoseStamped()
            ps.header = msg.header
            ps.header.frame_id = 'odom'
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            self.trajectory.poses.append(ps)
            self._last_traj_x = x
            self._last_traj_y = y
            self.trajectory.header.stamp = msg.header.stamp
            self.traj_pub.publish(self.trajectory)

    def _lidar_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        n = len(ranges)
        if n == 0:
            return
        # Front cone: ~20 degree half-angle
        cone_half = max(1, n // 18)
        front_indices = list(range(0, cone_half)) + list(range(n - cone_half, n))
        front_ranges = ranges[front_indices]
        valid = front_ranges[(front_ranges > msg.range_min) &
                             (front_ranges < msg.range_max) &
                             np.isfinite(front_ranges)]
        self.min_front_range = float(np.min(valid)) if len(valid) > 0 else float('inf')

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap = np.array(msg.data, dtype=np.uint8).reshape(
            (msg.info.height, msg.info.width))
        self.costmap_info = msg.info

    # ── Control loop ──────────────────────────────────────────────────

    def _control_tick(self):
        # Always publish status
        self._publish_status()

        if self.odom is None:
            return

        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y

        if self.path is None or len(self.path) == 0:
            self.state = 'NO_PATH'
            self._stop()
            return

        # Check if goal reached
        goal = self.path[-1]
        gx = goal.pose.position.x
        gy = goal.pose.position.y
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        if dist_to_goal < self.goal_tol:
            self.state = 'REACHED'
            self.path = None
            self._stop()
            return

        # Emergency stop — obstacle too close
        if self.min_front_range < self.min_obs_dist:
            self.state = 'E_STOP'
            self._stop()
            return

        # Path collision check: test next 3 waypoints against costmap
        if self.path_collision_check and self._path_blocked():
            self.state = 'PATH_BLOCKED'
            self._stop()
            return

        # Stuck detection
        now = self.get_clock().now().nanoseconds / 1e9
        if self.stuck.update(rx, ry, now):
            self.state = 'STUCK'
            self._stop()
            return

        # ── Pure pursuit ──────────────────────────────────────────────
        # Advance path index to closest point
        self._advance_path_idx(rx, ry)

        # Find lookahead point
        lx, ly = self._find_lookahead(rx, ry)
        self._publish_lookahead(lx, ly)

        # Robot heading from quaternion
        q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Transform lookahead to robot frame
        dx = lx - rx
        dy = ly - ry
        local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)

        # Heading error to lookahead
        heading_error = math.atan2(local_y, local_x)

        # Rotate in place if heading error is large (> 45 degrees)
        if abs(heading_error) > math.pi / 4.0:
            cmd = Twist()
            cmd.angular.z = max(-self.max_angular,
                                min(self.max_angular,
                                    1.5 * heading_error))
            self.cmd_pub.publish(cmd)
            self.state = 'ROTATING'
            return

        # Curvature
        L = math.hypot(local_x, local_y)
        if L < 1e-6:
            self._stop()
            return
        curvature = 2.0 * local_y / (L * L)

        # Velocity commands — scale linear by heading alignment
        alignment = math.cos(heading_error)  # 1.0 when aligned, 0 at 45deg
        linear = self.max_linear * max(0.3, alignment)
        angular = linear * curvature

        # LiDAR speed scaling (3-tier from legacy)
        linear = self._scale_speed_by_clearance(linear)

        # Clamp
        linear = max(-self.max_linear, min(self.max_linear, linear))
        angular = max(-self.max_angular, min(self.max_angular, angular))

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)
        self.state = 'TRACKING'

    # ── Helpers ────────────────────────────────────────────────────────

    def _advance_path_idx(self, rx: float, ry: float):
        """Move path_idx forward to the closest point ahead."""
        if self.path is None:
            return
        best_dist = float('inf')
        best_idx = self.path_idx
        # Only search forward from current index
        search_end = min(self.path_idx + 20, len(self.path))
        for i in range(self.path_idx, search_end):
            px = self.path[i].pose.position.x
            py = self.path[i].pose.position.y
            d = math.hypot(px - rx, py - ry)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.path_idx = best_idx

    def _interpolate_path(self, poses, spacing=0.05):
        """Interpolate sparse waypoints into dense path with given spacing."""
        if len(poses) < 2:
            return list(poses)
        dense = [poses[0]]
        for i in range(1, len(poses)):
            x0 = dense[-1].pose.position.x
            y0 = dense[-1].pose.position.y
            x1 = poses[i].pose.position.x
            y1 = poses[i].pose.position.y
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len < 1e-6:
                continue
            n_pts = max(1, int(seg_len / spacing))
            for j in range(1, n_pts + 1):
                t = j / n_pts
                ps = PoseStamped()
                ps.header = poses[i].header
                ps.pose.position.x = x0 + t * (x1 - x0)
                ps.pose.position.y = y0 + t * (y1 - y0)
                ps.pose.orientation.w = 1.0
                dense.append(ps)
        return dense

    def _find_lookahead(self, rx: float, ry: float):
        """Find lookahead point by interpolating along path segments."""
        # Walk along path from current index, accumulating distance
        for i in range(self.path_idx, len(self.path) - 1):
            ax = self.path[i].pose.position.x
            ay = self.path[i].pose.position.y
            bx = self.path[i + 1].pose.position.x
            by = self.path[i + 1].pose.position.y

            # Check circle-segment intersection
            dx = bx - ax
            dy = by - ay
            fx = ax - rx
            fy = ay - ry

            a = dx * dx + dy * dy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - self.lookahead * self.lookahead

            disc = b * b - 4.0 * a * c
            if disc < 0 or a < 1e-12:
                continue

            disc = math.sqrt(disc)
            t1 = (-b - disc) / (2.0 * a)
            t2 = (-b + disc) / (2.0 * a)

            # Pick the farthest intersection in [0, 1]
            t = None
            if 0.0 <= t2 <= 1.0:
                t = t2
            elif 0.0 <= t1 <= 1.0:
                t = t1

            if t is not None:
                return ax + t * dx, ay + t * dy

        # Fallback: use last point
        last = self.path[-1]
        return last.pose.position.x, last.pose.position.y

    def _scale_speed_by_clearance(self, linear: float) -> float:
        """3-tier speed scaling based on front LiDAR clearance."""
        clearance = self.min_front_range
        safety = self.min_obs_dist

        if clearance < safety:
            return 0.0
        elif clearance < self.slowdown_dist:
            # Ramp from 20% to 100% between safety and slowdown_dist
            t = (clearance - safety) / (self.slowdown_dist - safety)
            return linear * (0.2 + 0.8 * t)
        else:
            return linear

    def _path_blocked(self) -> bool:
        """Check if next 3 waypoints on path are blocked in costmap."""
        if self.costmap is None or self.costmap_info is None or self.path is None:
            return False

        info = self.costmap_info
        for i in range(self.path_idx, min(self.path_idx + 3, len(self.path))):
            wx = self.path[i].pose.position.x
            wy = self.path[i].pose.position.y
            # World to costmap grid
            col = int((wx - info.origin.position.x) / info.resolution)
            row = int((wy - info.origin.position.y) / info.resolution)
            if 0 <= row < self.costmap.shape[0] and 0 <= col < self.costmap.shape[1]:
                if self.costmap[row, col] >= self.obstacle_threshold:
                    return True
        return False

    def _publish_lookahead(self, lx: float, ly: float):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'pursuit'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = Point(x=lx, y=ly, z=0.1)
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)
        m.lifetime.sec = 0
        m.lifetime.nanosec = 200_000_000  # 200ms
        self.lookahead_pub.publish(m)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _publish_status(self):
        dist = 0.0
        if self.odom and self.path and len(self.path) > 0:
            rx = self.odom.pose.pose.position.x
            ry = self.odom.pose.pose.position.y
            gx = self.path[-1].pose.position.x
            gy = self.path[-1].pose.position.y
            dist = math.hypot(gx - rx, gy - ry)

        status = {
            'state': self.state,
            'waypoint_idx': self.path_idx,
            'dist_to_goal': round(dist, 3)
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PursuitControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
