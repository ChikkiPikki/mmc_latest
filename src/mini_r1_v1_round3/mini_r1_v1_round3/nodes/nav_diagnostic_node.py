import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from action_msgs.msg import GoalStatusArray


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class NavDiagnostic(Node):
    def __init__(self):
        super().__init__('nav_diagnostic_node')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.last_plan = None
        self.last_plan_time = 0.0
        self.plan_count = 0
        self.plan_endpoint_changes = 0
        self.prev_plan_endpoint = None

        self.last_cmd = Twist()
        self.cmd_count = 0
        self.cmd_sat_theta = 0

        self.last_odom = None
        self.odom_count = 0

        self.goal_status_counts = {}
        self.active_goal = False

        self.create_subscription(Path, '/plan', self.on_plan, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(Odometry, '/r1_mini/odom', self.on_odom, sensor_qos)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom_ekf, 10)
        self.create_subscription(
            GoalStatusArray,
            '/navigate_to_pose/_action/status',
            self.on_goal_status,
            10,
        )

        self.last_ekf = None

        self.window_start = time.monotonic()
        self.create_timer(1.0, self.tick)

        self.get_logger().info('nav_diagnostic_node up — 1Hz compact status')

    def on_plan(self, msg: Path):
        self.plan_count += 1
        self.last_plan = msg
        self.last_plan_time = time.monotonic()
        if msg.poses:
            ep = msg.poses[-1].pose.position
            key = (round(ep.x, 2), round(ep.y, 2))
            if self.prev_plan_endpoint is not None and key != self.prev_plan_endpoint:
                self.plan_endpoint_changes += 1
            self.prev_plan_endpoint = key

    def on_cmd(self, msg: Twist):
        self.last_cmd = msg
        self.cmd_count += 1
        if abs(msg.angular.z) >= 0.95:
            self.cmd_sat_theta += 1

    def on_odom(self, msg: Odometry):
        self.last_odom = msg
        self.odom_count += 1

    def on_odom_ekf(self, msg: Odometry):
        self.last_ekf = msg

    def on_goal_status(self, msg: GoalStatusArray):
        for s in msg.status_list:
            self.goal_status_counts[s.status] = self.goal_status_counts.get(s.status, 0) + 1
            self.active_goal = s.status in (1, 2)

    def tick(self):
        now = time.monotonic()
        dt = now - self.window_start
        self.window_start = now

        plan_hz = self.plan_count / dt if dt > 0 else 0.0
        cmd_hz = self.cmd_count / dt if dt > 0 else 0.0
        odom_hz = self.odom_count / dt if dt > 0 else 0.0
        sat_pct = (100.0 * self.cmd_sat_theta / self.cmd_count) if self.cmd_count else 0.0

        pose_str = 'pose=?'
        track_str = 'track=?'
        if self.last_odom:
            p = self.last_odom.pose.pose.position
            y = math.degrees(yaw_from_quat(self.last_odom.pose.pose.orientation))
            pose_str = f'pose=({p.x:+.2f},{p.y:+.2f},{y:+.0f}°)'
            vx_odom = self.last_odom.twist.twist.linear.x
            wz_odom = self.last_odom.twist.twist.angular.z
            track_str = f'odom_v=({vx_odom:+.2f},{wz_odom:+.2f})'

        ekf_str = 'ekf=?'
        if self.last_ekf:
            vx_e = self.last_ekf.twist.twist.linear.x
            wz_e = self.last_ekf.twist.twist.angular.z
            ekf_str = f'ekf_v=({vx_e:+.2f},{wz_e:+.2f})'

        cmd_str = f'cmd=({self.last_cmd.linear.x:+.2f},{self.last_cmd.angular.z:+.2f})'

        plan_str = 'plan=?'
        if self.last_plan is not None and self.last_plan.poses and self.last_odom:
            ep = self.last_plan.poses[-1].pose.position
            rp = self.last_odom.pose.pose.position
            dist = math.hypot(ep.x - rp.x, ep.y - rp.y)
            age = now - self.last_plan_time
            plan_str = (
                f'plan(len={len(self.last_plan.poses)},'
                f'goal_d={dist:.2f}m,age={age:.1f}s,'
                f'ep_changes={self.plan_endpoint_changes})'
            )

        goal_str = f'goals={dict(sorted(self.goal_status_counts.items()))}'

        self.get_logger().info(
            f'[{plan_hz:.1f}|{cmd_hz:.0f}|{odom_hz:.0f}Hz] '
            f'{pose_str} {cmd_str} {track_str} {ekf_str} '
            f'sat_θ={sat_pct:.0f}% {plan_str} {goal_str}'
        )

        self.plan_count = 0
        self.cmd_count = 0
        self.odom_count = 0
        self.cmd_sat_theta = 0
        self.plan_endpoint_changes = 0


def main():
    rclpy.init()
    node = NavDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
