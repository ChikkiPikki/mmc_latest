"""RRT path planner on local costmap — event-driven, publishes Path on /planned_path."""

import json
import math
import random
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from std_msgs.msg import String, Header


class RRTNode:
    """A single node in the RRT tree."""
    __slots__ = ('x', 'y', 'parent')

    def __init__(self, x: float, y: float, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


class RRTPlannerNode(Node):
    def __init__(self):
        super().__init__('rrt_planner_node')

        # Parameters
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('max_iterations', 1500)
        self.declare_parameter('step_size_m', 0.20)
        self.declare_parameter('goal_bias', 0.20)
        self.declare_parameter('obstacle_threshold', 65)
        self.declare_parameter('robot_radius_m', 0.18)
        self.declare_parameter('path_simplify', True)
        self.declare_parameter('timeout_s', 0.5)
        self.declare_parameter('use_rrt_star', False)

        self.max_iter = self.get_parameter('max_iterations').value
        self.step_size = self.get_parameter('step_size_m').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.obs_thresh = self.get_parameter('obstacle_threshold').value
        self.robot_radius = self.get_parameter('robot_radius_m').value
        self.simplify = self.get_parameter('path_simplify').value
        self.timeout = self.get_parameter('timeout_s').value
        self.use_rrt_star = self.get_parameter('use_rrt_star').value

        # State
        self.costmap = None
        self.costmap_info = None
        self.odom = None
        self.state = 'READY'

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

        # Subscribers
        self.create_subscription(PoseStamped, '/plan_request', self._plan_request_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.status_pub = self.create_publisher(String, '/planner/status', 10)

        # Heartbeat timer (1 Hz status)
        self.create_timer(1.0, self._status_tick)

        self.get_logger().info(
            f'RRT planner: max_iter={self.max_iter}, step={self.step_size}m, '
            f'timeout={self.timeout}s, rrt_star={self.use_rrt_star}')

    # ── Callbacks ──────────────────────────────────────────────────────

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))
        self.costmap_info = msg.info

    def _odom_cb(self, msg: Odometry):
        self.odom = msg

    def _plan_request_cb(self, msg: PoseStamped):
        if self.odom is None:
            self.get_logger().warn('Plan request received but no odom available')
            self._publish_empty_path(msg.header.stamp)
            return
        if self.costmap is None:
            self.get_logger().warn('Plan request received but no costmap available')
            self._publish_empty_path(msg.header.stamp)
            return

        self.state = 'PLANNING'
        self._publish_status()

        sx = self.odom.pose.pose.position.x
        sy = self.odom.pose.pose.position.y
        gx = msg.pose.position.x
        gy = msg.pose.position.y

        self.get_logger().info(f'Planning: ({sx:.2f},{sy:.2f}) -> ({gx:.2f},{gy:.2f})')

        path = self._run_rrt(sx, sy, gx, gy)

        if path is None:
            self.get_logger().warn('RRT failed to find path')
            self.state = 'FAILED'
            self._publish_empty_path(msg.header.stamp)
        else:
            if self.simplify:
                path = self._simplify_path(path)
            self.state = 'READY'
            self._publish_path(path, msg.header.stamp)
            self.get_logger().info(f'Path published: {len(path)} waypoints')

        self._publish_status()

    # ── RRT Core ───────────────────────────────────────────────────────

    def _run_rrt(self, sx, sy, gx, gy):
        start_time = time.monotonic()
        start = RRTNode(sx, sy)
        tree = [start]
        goal_tol = self.step_size * 1.5

        for _ in range(self.max_iter):
            if time.monotonic() - start_time > self.timeout:
                break

            # Sample (with goal bias)
            if random.random() < self.goal_bias:
                rx, ry = gx, gy
            else:
                rx, ry = self._random_sample()

            # Find nearest node
            nearest = self._nearest(tree, rx, ry)

            # Steer
            nx, ny = self._steer(nearest.x, nearest.y, rx, ry)

            # Collision check
            if not self._collision_free(nearest.x, nearest.y, nx, ny):
                continue

            new_node = RRTNode(nx, ny, parent=nearest)
            tree.append(new_node)

            # Check goal
            if math.hypot(nx - gx, ny - gy) < goal_tol:
                # Add goal node
                if self._collision_free(nx, ny, gx, gy):
                    goal_node = RRTNode(gx, gy, parent=new_node)
                    return self._extract_path(goal_node)

        return None

    def _random_sample(self):
        """Sample a random point within the costmap bounds."""
        info = self.costmap_info
        x_min = info.origin.position.x
        y_min = info.origin.position.y
        x_max = x_min + info.width * info.resolution
        y_max = y_min + info.height * info.resolution
        return (random.uniform(x_min, x_max),
                random.uniform(y_min, y_max))

    def _nearest(self, tree, x, y):
        best = None
        best_dist = float('inf')
        for node in tree:
            d = (node.x - x) ** 2 + (node.y - y) ** 2
            if d < best_dist:
                best_dist = d
                best = node
        return best

    def _steer(self, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y
        dist = math.hypot(dx, dy)
        if dist <= self.step_size:
            return to_x, to_y
        ratio = self.step_size / dist
        return from_x + dx * ratio, from_y + dy * ratio

    def _collision_free(self, x1, y1, x2, y2):
        """Walk the edge at costmap resolution, checking for obstacles."""
        info = self.costmap_info
        res = info.resolution
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(2, int(dist / res))

        # Also check with robot radius offset
        radius_cells = int(self.robot_radius / res)

        for i in range(steps + 1):
            t = i / steps
            wx = x1 + t * (x2 - x1)
            wy = y1 + t * (y2 - y1)

            col = int((wx - info.origin.position.x) / res)
            row = int((wy - info.origin.position.y) / res)

            # Check cells within robot radius
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if dr * dr + dc * dc > radius_cells * radius_cells:
                        continue
                    r = row + dr
                    c = col + dc
                    if 0 <= r < self.costmap.shape[0] and 0 <= c < self.costmap.shape[1]:
                        if self.costmap[r, c] >= self.obs_thresh:
                            return False
                    else:
                        # Out of bounds treated as obstacle
                        return False
        return True

    def _extract_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()
        return path

    def _simplify_path(self, path):
        """Line-of-sight path simplification."""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Find farthest visible point
            farthest = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._collision_free(path[i][0], path[i][1],
                                        path[j][0], path[j][1]):
                    farthest = j
                    break
            simplified.append(path[farthest])
            i = farthest
        return simplified

    # ── Publishing ─────────────────────────────────────────────────────

    def _publish_path(self, waypoints, stamp):
        msg = Path()
        msg.header = Header(stamp=stamp, frame_id='odom')
        for wx, wy in waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_empty_path(self, stamp):
        msg = Path()
        msg.header = Header(stamp=stamp, frame_id='odom')
        self.path_pub.publish(msg)

    def _publish_status(self):
        status = {'state': self.state}
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _status_tick(self):
        self._publish_status()


def main(args=None):
    rclpy.init(args=args)
    node = RRTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
