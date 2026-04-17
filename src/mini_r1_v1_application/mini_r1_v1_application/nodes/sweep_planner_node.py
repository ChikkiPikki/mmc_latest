"""Frontier-based exploration planner — finds nearest unvisited reachable free space."""

import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String, Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class SweepPlannerNode(Node):
    def __init__(self):
        super().__init__('sweep_planner_node')

        # Parameters
        self.declare_parameter('goal_tolerance_m', 0.25)
        self.declare_parameter('replan_interval_s', 3.0)
        self.declare_parameter('tick_rate_hz', 5.0)
        self.declare_parameter('min_goal_dist_m', 0.5)
        self.declare_parameter('max_goal_dist_m', 3.0)
        self.declare_parameter('costmap_free_thresh', 50)
        self.declare_parameter('max_consecutive_failures', 5)

        self.goal_tol = self.get_parameter('goal_tolerance_m').value
        self.replan_interval = self.get_parameter('replan_interval_s').value
        self.min_goal_dist = self.get_parameter('min_goal_dist_m').value
        self.max_goal_dist = self.get_parameter('max_goal_dist_m').value
        self.free_thresh = self.get_parameter('costmap_free_thresh').value
        self.max_failures = self.get_parameter('max_consecutive_failures').value
        tick_rate = self.get_parameter('tick_rate_hz').value

        # FSM state
        self.state = 'IDLE'  # IDLE, EXPLORING, DONE
        self.current_goal = None
        self.consecutive_failures = 0
        self.tried_goals = []  # recently failed goals to avoid retrying

        # Tracking
        self.odom = None
        self.costmap = None
        self.costmap_info = None
        self.coverage_grid = None
        self.coverage_info = None
        self.coverage_pct = 0.0
        self.planner_ready = True
        self.controller_state = 'NO_PATH'
        self.last_plan_time = 0.0
        self.active = False

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
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        # Subscribers
        self.create_subscription(
            OccupancyGrid, '/coverage/grid', self._coverage_cb, reliable_transient)
        self.create_subscription(
            OccupancyGrid, '/costmap/costmap', self._costmap_cb, costmap_qos)
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
        self.wp_viz_pub = self.create_publisher(MarkerArray, '/sweep/waypoints', 10)

        # Tick timer
        self.create_timer(1.0 / tick_rate, self._tick)

        self.get_logger().info('Frontier exploration planner ready')

    # ── Callbacks ──────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap = np.array(msg.data, dtype=np.uint8).reshape(
            (msg.info.height, msg.info.width))
        self.costmap_info = msg.info

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
            self.planner_ready = (data.get('state', '') in ('READY', 'FAILED'))
        except (json.JSONDecodeError, KeyError):
            pass

    def _command_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            cmd = data.get('command', '')
            if cmd == 'START':
                self.active = True
                self.state = 'EXPLORING'
                self.consecutive_failures = 0
                self.tried_goals = []
                self.get_logger().info('Exploration START')
            elif cmd == 'STOP':
                self.active = False
                self.state = 'IDLE'
                self.get_logger().info('Exploration STOP')
        except (json.JSONDecodeError, KeyError):
            pass

    # ── Main tick ──────────────────────────────────────────────────────

    def _tick(self):
        self._publish_status()
        self._publish_waypoints_viz()

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

        # Track failures
        if self.controller_state in ('STUCK', 'PATH_BLOCKED', 'E_STOP'):
            self.consecutive_failures += 1
            if self.current_goal is not None:
                self.tried_goals.append(self.current_goal)
                # Keep list bounded
                if len(self.tried_goals) > 20:
                    self.tried_goals = self.tried_goals[-10:]
        elif self.controller_state == 'REACHED':
            self.consecutive_failures = 0

        rx = self.odom.pose.pose.position.x
        ry = self.odom.pose.pose.position.y

        goal = self._find_frontier_goal(rx, ry)
        if goal is None:
            self.state = 'DONE'
            self.get_logger().info('Exploration DONE — no reachable frontiers')
            return

        self._send_goal(goal[0], goal[1], now)

    # ── Frontier goal finding ─────────────────────────────────────────

    def _find_frontier_goal(self, rx, ry):
        """Find the best unvisited frontier goal using costmap for reachability."""
        # Use costmap to find free cells, coverage grid for unvisited
        if self.costmap is None or self.costmap_info is None:
            return self._find_simple_goal(rx, ry)

        cm_info = self.costmap_info
        cm = self.costmap

        # Collect candidate goals: free cells in costmap that are unvisited
        candidates = []

        # Sample the costmap grid
        step = max(1, cm_info.height // 40)  # ~40x40 sample points
        for row in range(0, cm_info.height, step):
            for col in range(0, cm_info.width, step):
                # Must be free in costmap
                if cm[row, col] >= self.free_thresh:
                    continue

                wx = cm_info.origin.position.x + (col + 0.5) * cm_info.resolution
                wy = cm_info.origin.position.y + (row + 0.5) * cm_info.resolution

                dist = math.hypot(wx - rx, wy - ry)
                if dist < self.min_goal_dist or dist > self.max_goal_dist:
                    continue

                # Skip recently failed goals
                too_close_to_tried = False
                for tx, ty in self.tried_goals:
                    if math.hypot(wx - tx, wy - ty) < 0.3:
                        too_close_to_tried = True
                        break
                if too_close_to_tried:
                    continue

                # Check if unvisited in coverage grid
                unvisited = True
                if self.coverage_grid is not None and self.coverage_info is not None:
                    cv_info = self.coverage_info
                    cv_col = int((wx - cv_info.origin.position.x) / cv_info.resolution)
                    cv_row = int((wy - cv_info.origin.position.y) / cv_info.resolution)
                    if (0 <= cv_row < self.coverage_grid.shape[0] and
                            0 <= cv_col < self.coverage_grid.shape[1]):
                        if self.coverage_grid[cv_row, cv_col] == 100:
                            unvisited = False

                if unvisited:
                    candidates.append((wx, wy, dist))

        if not candidates:
            # If no unvisited candidates, try any free cell (even visited)
            # to escape dead ends
            return self._find_any_free_goal(rx, ry)

        # Sort by distance — prefer closer goals for reliability
        candidates.sort(key=lambda c: c[2])

        # Pick from the closest few with some randomness to avoid getting stuck
        pick_range = min(5, len(candidates))
        idx = np.random.randint(0, pick_range)
        wx, wy, _ = candidates[idx]
        return (wx, wy)

    def _find_any_free_goal(self, rx, ry):
        """Fallback: find any free cell in costmap at moderate distance."""
        if self.costmap is None or self.costmap_info is None:
            return None

        cm_info = self.costmap_info
        cm = self.costmap
        best = None
        best_dist = float('inf')

        step = max(1, cm_info.height // 30)
        for row in range(0, cm_info.height, step):
            for col in range(0, cm_info.width, step):
                if cm[row, col] >= self.free_thresh:
                    continue

                wx = cm_info.origin.position.x + (col + 0.5) * cm_info.resolution
                wy = cm_info.origin.position.y + (row + 0.5) * cm_info.resolution

                dist = math.hypot(wx - rx, wy - ry)
                if dist < self.min_goal_dist:
                    continue

                too_close = False
                for tx, ty in self.tried_goals:
                    if math.hypot(wx - tx, wy - ty) < 0.3:
                        too_close = True
                        break
                if too_close:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best = (wx, wy)

        return best

    def _find_simple_goal(self, rx, ry):
        """Fallback when no costmap: use coverage grid only."""
        if self.coverage_grid is None or self.coverage_info is None:
            return None

        info = self.coverage_info
        unvisited = np.argwhere(self.coverage_grid == 0)
        if len(unvisited) == 0:
            return None

        best_dist = float('inf')
        best = None
        step = max(1, len(unvisited) // 200)
        for idx in range(0, len(unvisited), step):
            row, col = unvisited[idx]
            wx = info.origin.position.x + (col + 0.5) * info.resolution
            wy = info.origin.position.y + (row + 0.5) * info.resolution
            d = math.hypot(wx - rx, wy - ry)
            if self.min_goal_dist < d < best_dist:
                best_dist = d
                best = (wx, wy)

        return best

    # ── Publishing ────────────────────────────────────────────────────

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

    def _publish_waypoints_viz(self):
        """Publish current goal as a yellow sphere."""
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        if self.current_goal is not None:
            m = Marker()
            m.header = Header(stamp=stamp, frame_id='odom')
            m.ns = 'exploration_goal'
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(
                x=self.current_goal[0], y=self.current_goal[1], z=0.15)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)
            ma.markers.append(m)

        # Show tried/failed goals as small red dots
        for i, (tx, ty) in enumerate(self.tried_goals[-10:]):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id='odom')
            m.ns = 'tried_goals'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=tx, y=ty, z=0.05)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.06
            m.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.6)
            ma.markers.append(m)

        if ma.markers:
            self.wp_viz_pub.publish(ma)

    def _publish_status(self):
        status = {
            'state': self.state,
            'coverage_pct': round(self.coverage_pct, 1),
            'failures': self.consecutive_failures,
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
