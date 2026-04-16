"""Boustrophedon sweep planner — generates waypoints, skips covered lanes, handles goal sequencing."""

import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String, Header


class SweepPlannerNode(Node):
    def __init__(self):
        super().__init__('sweep_planner_node')

        # Parameters
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('arena_width_m', 10.0)
        self.declare_parameter('arena_height_m', 10.0)
        self.declare_parameter('arena_origin_x', -5.0)
        self.declare_parameter('arena_origin_y', -5.0)
        self.declare_parameter('sweep_spacing_m', 0.30)
        self.declare_parameter('sweep_direction', 'x')
        self.declare_parameter('goal_tolerance_m', 0.25)
        self.declare_parameter('replan_interval_s', 3.0)
        self.declare_parameter('tick_rate_hz', 5.0)
        self.declare_parameter('coverage_lane_threshold', 0.7)

        self.arena_w = self.get_parameter('arena_width_m').value
        self.arena_h = self.get_parameter('arena_height_m').value
        self.origin_x = self.get_parameter('arena_origin_x').value
        self.origin_y = self.get_parameter('arena_origin_y').value
        self.spacing = self.get_parameter('sweep_spacing_m').value
        self.direction = self.get_parameter('sweep_direction').value
        self.goal_tol = self.get_parameter('goal_tolerance_m').value
        self.replan_interval = self.get_parameter('replan_interval_s').value
        self.lane_thresh = self.get_parameter('coverage_lane_threshold').value
        tick_rate = self.get_parameter('tick_rate_hz').value

        # FSM state
        self.state = 'IDLE'  # IDLE, SWEEPING, FRONTIER_SEEK, DONE
        self.current_goal = None
        self.current_row = 0

        # Generate sweep waypoints
        self.waypoints = self._generate_waypoints()
        self.wp_idx = 0

        # Tracking
        self.odom = None
        self.coverage_grid = None
        self.coverage_info = None
        self.coverage_pct = 0.0
        self.planner_ready = True
        self.controller_state = 'NO_PATH'
        self.last_plan_time = 0.0
        self.active = False  # Set by mission command

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)
        reliable_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # Subscribers
        self.create_subscription(
            OccupancyGrid, '/coverage/grid', self._coverage_cb, reliable_transient)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)
        self.create_subscription(
            String, '/controller/status', self._controller_status_cb, 10)
        self.create_subscription(
            String, '/planner/status', self._planner_status_cb, 10)
        self.create_subscription(
            String, '/mission/command', self._command_cb, 10)

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/plan_request', 10)
        self.status_pub = self.create_publisher(String, '/exploration/status', 10)

        # Tick timer
        self.create_timer(1.0 / tick_rate, self._tick)

        self.get_logger().info(
            f'Sweep planner: {len(self.waypoints)} waypoints, '
            f'spacing={self.spacing}m, direction={self.direction}')

    # ── Waypoint generation ────────────────────────────────────────────

    def _generate_waypoints(self):
        """Generate boustrophedon (lawn-mower) waypoints."""
        margin = self.spacing  # Stay away from arena edges
        x_min = self.origin_x + margin
        x_max = self.origin_x + self.arena_w - margin
        y_min = self.origin_y + margin
        y_max = self.origin_y + self.arena_h - margin

        waypoints = []

        if self.direction == 'x':
            # Sweep along X, step in Y
            y = y_min
            row = 0
            while y <= y_max:
                if row % 2 == 0:
                    waypoints.append((x_min, y, row))
                    waypoints.append((x_max, y, row))
                else:
                    waypoints.append((x_max, y, row))
                    waypoints.append((x_min, y, row))
                y += self.spacing
                row += 1
        else:
            # Sweep along Y, step in X
            x = x_min
            row = 0
            while x <= x_max:
                if row % 2 == 0:
                    waypoints.append((x, y_min, row))
                    waypoints.append((x, y_max, row))
                else:
                    waypoints.append((x, y_max, row))
                    waypoints.append((x, y_min, row))
                x += self.spacing
                row += 1

        return waypoints

    # ── Callbacks ──────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _coverage_cb(self, msg: OccupancyGrid):
        self.coverage_grid = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        self.coverage_info = msg.info
        total = msg.info.width * msg.info.height
        visited = np.count_nonzero(self.coverage_grid == 100)
        self.coverage_pct = visited / total * 100.0 if total > 0 else 0.0

    def _controller_status_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.controller_state = data.get('state', 'NO_PATH')
        except (json.JSONDecodeError, KeyError):
            pass

    def _planner_status_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.planner_ready = (data.get('state', '') == 'READY')
        except (json.JSONDecodeError, KeyError):
            pass

    def _command_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            cmd = data.get('command', '')
            if cmd == 'START':
                self.active = True
                self.state = 'SWEEPING'
                self.wp_idx = 0
                self.get_logger().info('Exploration START received')
            elif cmd == 'STOP':
                self.active = False
                self.state = 'IDLE'
                self.get_logger().info('Exploration STOP received')
        except (json.JSONDecodeError, KeyError):
            pass

    # ── Main tick ──────────────────────────────────────────────────────

    def _tick(self):
        self._publish_status()

        if not self.active or self.odom is None:
            return

        if self.state == 'DONE':
            return

        now = self.get_clock().now().nanoseconds / 1e9

        # Can we send a new goal?
        can_send = (self.planner_ready and
                    self.controller_state in ('REACHED', 'STUCK', 'NO_PATH',
                                              'PATH_BLOCKED', 'E_STOP') and
                    now - self.last_plan_time >= self.replan_interval)

        if not can_send:
            return

        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y

        if self.state == 'SWEEPING':
            goal = self._next_sweep_goal(rx, ry)
            if goal is None:
                # All sweep waypoints exhausted — try frontier
                self.state = 'FRONTIER_SEEK'
                return
            self._send_goal(goal[0], goal[1], now)

        elif self.state == 'FRONTIER_SEEK':
            goal = self._find_frontier_goal(rx, ry)
            if goal is None:
                self.state = 'DONE'
                self.get_logger().info('Exploration DONE — no frontiers remaining')
                return
            self._send_goal(goal[0], goal[1], now)

    def _next_sweep_goal(self, rx, ry):
        """Find next uncovered sweep waypoint."""
        while self.wp_idx < len(self.waypoints):
            wx, wy, row = self.waypoints[self.wp_idx]

            # Check if this waypoint is already close to current position (skip)
            if math.hypot(wx - rx, wy - ry) < self.goal_tol:
                self.wp_idx += 1
                continue

            # Check if lane is already covered using coverage grid
            if self._is_lane_covered(row):
                # Skip both waypoints of this lane (2 per lane)
                lane_start = self.wp_idx
                while (self.wp_idx < len(self.waypoints) and
                       self.waypoints[self.wp_idx][2] == row):
                    self.wp_idx += 1
                self.get_logger().debug(f'Skipping covered lane {row}')
                continue

            self.current_row = row
            self.current_goal = (wx, wy)
            self.wp_idx += 1
            return (wx, wy)
        return None

    def _is_lane_covered(self, row_idx):
        """Check if a sweep lane is sufficiently covered in the coverage grid."""
        if self.coverage_grid is None or self.coverage_info is None:
            return False

        info = self.coverage_info
        # Get the Y coordinate for this lane
        margin = self.spacing
        lane_y = self.origin_y + margin + row_idx * self.spacing

        # Sample cells along this lane
        x_min = self.origin_x + margin
        x_max = self.origin_x + self.arena_w - margin
        sample_count = 0
        covered_count = 0

        x = x_min
        while x <= x_max:
            col = int((x - info.origin.position.x) / info.resolution)
            row = int((lane_y - info.origin.position.y) / info.resolution)
            if 0 <= row < self.coverage_grid.shape[0] and 0 <= col < self.coverage_grid.shape[1]:
                sample_count += 1
                if self.coverage_grid[row, col] == 100:
                    covered_count += 1
            x += info.resolution * 2  # Sample every 2 cells

        if sample_count == 0:
            return False
        return (covered_count / sample_count) >= self.lane_thresh

    def _find_frontier_goal(self, rx, ry):
        """Find nearest unvisited cluster centroid as fallback goal."""
        if self.coverage_grid is None or self.coverage_info is None:
            return None

        info = self.coverage_info
        # Find unvisited cells
        unvisited = np.argwhere(self.coverage_grid == 0)
        if len(unvisited) == 0:
            return None

        # Convert to world coords and find nearest cluster centroid
        # Simple: just find nearest unvisited cell
        best_dist = float('inf')
        best_wx, best_wy = None, None

        # Subsample for speed
        step = max(1, len(unvisited) // 200)
        for idx in range(0, len(unvisited), step):
            row, col = unvisited[idx]
            wx = info.origin.position.x + (col + 0.5) * info.resolution
            wy = info.origin.position.y + (row + 0.5) * info.resolution
            d = math.hypot(wx - rx, wy - ry)
            # Not too close (would be already counted as visited), not too far
            if 0.5 < d < best_dist:
                best_dist = d
                best_wx = wx
                best_wy = wy

        if best_wx is None:
            return None
        return (best_wx, best_wy)

    def _send_goal(self, gx, gy, now):
        msg = PoseStamped()
        msg.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='odom')
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        self.current_goal = (gx, gy)
        self.last_plan_time = now
        self.get_logger().info(f'Goal sent: ({gx:.2f}, {gy:.2f}) state={self.state}')

    def _publish_status(self):
        status = {
            'state': self.state,
            'row': self.current_row,
            'coverage_pct': round(self.coverage_pct, 1),
            'current_goal': list(self.current_goal) if self.current_goal else None
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SweepPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
