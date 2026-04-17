#!/usr/bin/env python3
"""MPC Tracker — Bezier-smoothed path tracker with reactive obstacle avoidance.

Ported from nav_assignment/nav/src/mpc_tracker.py + smoothing.cpp.

Architecture:
- Subscribes to `/mpc/path` (latched Path in map frame) — published by mission_manager.
- Applies Bezier corner rounding + densification to the incoming path.
- Looks up robot pose in the map frame via TF at each tick.
- Runs a CasADi/IPOPT MPC at HZ to track the smoothed reference.
- Publishes `/cmd_vel` (Twist) directly.
- Publishes `/mpc/goal_reached` (Empty) once within GOAL_TOL of final waypoint.
- Reactive obstacle avoidance using `/r1_mini/lidar`: turn toward free side,
  creep forward, resume MPC when clear.
"""
import math
import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
import tf2_ros

from mini_r1_v1_round3.utils.path_smoother import smooth_path


# ── MPC params ─────────────────────────────────────────────────────────────
N = 10
DT = 0.1
MAX_V = 0.25
MIN_V = 0.0
MAX_W = 1.5
MAX_DV = 0.10
MAX_DW = 0.25
Q = np.diag([8.0, 8.0, 2.0])
QN = np.diag([12.0, 12.0, 4.0])
R = np.diag([0.5, 0.4])

# ── Obstacle params ────────────────────────────────────────────────────────
OBS_FRONT_DEG = 60.0
OBS_SIDE_DEG = 100.0
OBS_AVOID_D = 0.40
OBS_CLEAR_D = 0.55
OBS_STOP_D = 0.25
TURN_SPEED = 1.0
CREEP_SPEED = 0.08

# ── Path / goal params ─────────────────────────────────────────────────────
LOOKAHEAD = 0.5
GOAL_TOL = 0.25
HZ = 10.0

# ── Smoother params ────────────────────────────────────────────────────────
SMOOTH_RES = 0.08
CORNER_RADIUS = 0.4


def angle_wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def build_mpc():
    op = ca.Opti()
    U = op.variable(2, N)
    X = op.variable(3, N + 1)
    P = op.parameter(3 + 3 * (N + 1))
    op.subject_to(X[:, 0] == P[:3])
    cost = 0.0
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        vk = uk[0]
        wk = uk[1]
        op.subject_to(X[:, k + 1] == ca.vertcat(
            xk[0] + DT * vk * ca.cos(xk[2]),
            xk[1] + DT * vk * ca.sin(xk[2]),
            xk[2] + DT * wk))
        rk = P[3 + 3 * k:3 + 3 * k + 3]
        e = ca.vertcat(
            xk[0] - rk[0],
            xk[1] - rk[1],
            ca.atan2(ca.sin(xk[2] - rk[2]), ca.cos(xk[2] - rk[2])))
        cost += e.T @ Q @ e + uk.T @ R @ uk
        op.subject_to(op.bounded(MIN_V, vk, MAX_V))
        op.subject_to(op.bounded(-MAX_W, wk, MAX_W))
        if k > 0:
            op.subject_to(op.bounded(-MAX_DV, vk - U[0, k - 1], MAX_DV))
            op.subject_to(op.bounded(-MAX_DW, wk - U[1, k - 1], MAX_DW))
    rN = P[3 + 3 * N:3 + 3 * N + 3]
    eN = ca.vertcat(
        X[0, N] - rN[0],
        X[1, N] - rN[1],
        ca.atan2(ca.sin(X[2, N] - rN[2]), ca.cos(X[2, N] - rN[2])))
    cost += eN.T @ QN @ eN
    op.minimize(cost)
    op.solver('ipopt', {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 80,
        'ipopt.tol': 1e-3,
        'ipopt.acceptable_tol': 1e-2,
        'print_time': 0,
        'ipopt.sb': 'yes',
        'ipopt.warm_start_init_point': 'yes',
    })
    return op, X, U, P


class MPCTrackerNode(Node):
    def __init__(self):
        super().__init__('mpc_tracker_node')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.x = self.y = self.yaw = 0.0
        self.have_pose = False

        self.path = []
        self.path_idx = 0
        self.goal_reached = False

        self.scan_ranges = []
        self.scan_amin = 0.0
        self.scan_ainc = 0.0
        self.front_dist = float('inf')
        self.obs_angle = 0.0
        self.avoiding = False
        self.avoid_dir = 0.0

        self.get_logger().info('Building MPC (CasADi/IPOPT)…')
        self.opti, self.Xv, self.Uv, self.Pv = build_mpc()
        self.u_prev = np.zeros((2, N))
        self.get_logger().info('MPC ready ✓')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(Path, '/mpc/path', self._path_cb, latched)
        self.create_subscription(LaserScan, '/r1_mini/lidar', self._scan_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lh_pub = self.create_publisher(PointStamped, '/mpc/lookahead_point', 10)
        self.reached_pub = self.create_publisher(Empty, '/mpc/goal_reached', latched)
        self.smoothed_pub = self.create_publisher(Path, '/mpc/smoothed_path', latched)

        self.create_timer(1.0 / HZ, self._loop)
        self.get_logger().info('MPCTrackerNode started ✓')

    # ── callbacks ──────────────────────────────────────────────────────────
    def _path_cb(self, msg: Path):
        raw = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(raw) < 2:
            self.get_logger().warn(f'Path too short ({len(raw)}), ignoring')
            return
        smoothed = smooth_path(raw, resolution=SMOOTH_RES, radius=CORNER_RADIUS)
        self.path = smoothed
        self.path_idx = 0
        self.goal_reached = False
        self.u_prev = np.zeros((2, N))
        self.avoiding = False
        self.get_logger().info(
            f'Path received: raw={len(raw)} → smoothed={len(smoothed)} pts '
            f'(first={smoothed[0]}, last={smoothed[-1]})')

        sm = Path()
        sm.header.stamp = self.get_clock().now().to_msg()
        sm.header.frame_id = self.map_frame
        from geometry_msgs.msg import PoseStamped as PS
        for (x, y) in smoothed:
            ps = PS()
            ps.header = sm.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            sm.poses.append(ps)
        self.smoothed_pub.publish(sm)

    def _scan_cb(self, msg: LaserScan):
        self.scan_ranges = list(msg.ranges)
        self.scan_amin = msg.angle_min
        self.scan_ainc = msg.angle_increment
        arc = math.radians(OBS_FRONT_DEG)
        md = float('inf')
        ma = 0.0
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            if abs(a) <= arc and math.isfinite(r) and msg.range_min < r < msg.range_max:
                if r < md:
                    md = r
                    ma = a
        self.front_dist = md
        self.obs_angle = ma

    # ── helpers ────────────────────────────────────────────────────────────
    def _update_pose_from_tf(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return False
        self.x = t.transform.translation.x
        self.y = t.transform.translation.y
        q = t.transform.rotation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.have_pose = True
        return True

    def _side_range(self, left=True):
        if not self.scan_ranges:
            return 0.0
        arc = math.radians(OBS_SIDE_DEG)
        vals = []
        for i, r in enumerate(self.scan_ranges):
            a = self.scan_amin + i * self.scan_ainc
            if math.isfinite(r):
                if left and 0.1 < a <= arc:
                    vals.append(r)
                if not left and -arc <= a < -0.1:
                    vals.append(r)
        return sum(vals) / len(vals) if vals else 0.0

    def _send(self, v, w):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.cmd_pub.publish(t)

    def _advance_idx(self):
        best = self.path_idx
        bd = float('inf')
        stop = min(self.path_idx + 40, len(self.path))
        for i in range(self.path_idx, stop):
            d = math.hypot(self.x - self.path[i][0], self.y - self.path[i][1])
            if d < bd:
                bd = d
                best = i
        self.path_idx = max(self.path_idx, best)

    def _closest_idx(self):
        best = 0
        bd = float('inf')
        for i in range(len(self.path)):
            d = math.hypot(self.x - self.path[i][0], self.y - self.path[i][1])
            if d < bd:
                bd = d
                best = i
        return best

    def _refs(self):
        step = max(1, int(LOOKAHEAD / (N * DT * MAX_V)))
        refs = []
        for k in range(N + 1):
            i = min(self.path_idx + k * step, len(self.path) - 1)
            px, py = self.path[i]
            if i + 1 < len(self.path):
                yr = math.atan2(self.path[i + 1][1] - py, self.path[i + 1][0] - px)
            elif i > 0:
                yr = math.atan2(py - self.path[i - 1][1], px - self.path[i - 1][0])
            else:
                yr = self.yaw
            refs.append((px, py, yr))
        return refs

    def _run_mpc(self, refs):
        pv = np.array([self.x, self.y, self.yaw])
        for rx, ry, ry2 in refs:
            pv = np.append(pv, [rx, ry, ry2])
        ui = np.hstack([self.u_prev[:, 1:], self.u_prev[:, -1:]])
        xi = np.zeros((3, N + 1))
        xi[:, 0] = [self.x, self.y, self.yaw]
        for k in range(N):
            vk = float(ui[0, k])
            wk = float(ui[1, k])
            xi[0, k + 1] = xi[0, k] + DT * vk * math.cos(xi[2, k])
            xi[1, k + 1] = xi[1, k] + DT * vk * math.sin(xi[2, k])
            xi[2, k + 1] = xi[2, k] + DT * wk
        self.opti.set_value(self.Pv, pv)
        self.opti.set_initial(self.Uv, ui)
        self.opti.set_initial(self.Xv, xi)
        try:
            sol = self.opti.solve()
            u = sol.value(self.Uv)
            self.u_prev = u
            return float(u[0, 0]), float(u[1, 0])
        except Exception:
            rx, ry, _ = refs[1]
            he = angle_wrap(math.atan2(ry - self.y, rx - self.x) - self.yaw)
            self.u_prev = np.zeros((2, N))
            return MAX_V * max(0.0, 1.0 - abs(he)), 1.5 * he

    # ── main loop ──────────────────────────────────────────────────────────
    def _loop(self):
        if not self.path or self.goal_reached:
            return
        if not self._update_pose_from_tf():
            return

        gx, gy = self.path[-1]
        if math.hypot(self.x - gx, self.y - gy) < GOAL_TOL:
            self._send(0.0, 0.0)
            self.goal_reached = True
            self.reached_pub.publish(Empty())
            self.get_logger().info('Goal reached ✓')
            return

        fd = self.front_dist

        if not self.avoiding and fd < OBS_AVOID_D:
            l = self._side_range(left=True)
            r = self._side_range(left=False)
            self.avoid_dir = 1.0 if l >= r else -1.0
            self.avoiding = True
            self.u_prev = np.zeros((2, N))
            side = 'left' if self.avoid_dir > 0 else 'right'
            self.get_logger().warn(f'Obstacle {fd:.2f}m — avoiding {side}')

        if self.avoiding:
            if fd > OBS_CLEAR_D:
                self.path_idx = self._closest_idx()
                self.avoiding = False
                self.u_prev = np.zeros((2, N))
                self.get_logger().info('Obstacle cleared — resuming MPC ✓')
            else:
                turn = TURN_SPEED * self.avoid_dir
                if abs(self.obs_angle) > math.radians(20) or fd > OBS_STOP_D + 0.15:
                    creep = CREEP_SPEED
                else:
                    creep = 0.0
                self._send(creep, turn)
                self.get_logger().warn(
                    f'[AVOID] v={creep:.2f} w={turn:.2f} obs={fd:.2f}m '
                    f'angle={math.degrees(self.obs_angle):.1f}°',
                    throttle_duration_sec=0.5)
                return

        self._advance_idx()
        refs = self._refs()
        v, w = self._run_mpc(refs)
        v = float(np.clip(v, MIN_V, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        self._send(v, w)

        lx, ly, _ = refs[min(3, N)]
        lh = PointStamped()
        lh.header.stamp = self.get_clock().now().to_msg()
        lh.header.frame_id = self.map_frame
        lh.point.x = lx
        lh.point.y = ly
        self.lh_pub.publish(lh)
        self.get_logger().info(
            f'[TRACK] v={v:.3f} w={w:.3f} obs={fd:.2f}m '
            f'idx={self.path_idx}/{len(self.path)}',
            throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = MPCTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
