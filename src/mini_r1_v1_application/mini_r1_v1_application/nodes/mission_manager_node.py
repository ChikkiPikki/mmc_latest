"""Mission manager — top-level orchestrator for autonomous exploration."""

import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        # Parameters
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('coverage_target_pct', 90.0)
        self.declare_parameter('max_mission_time_s', 300.0)
        self.declare_parameter('report_interval_s', 5.0)
        self.declare_parameter('node_timeout_s', 3.0)

        self.coverage_target = self.get_parameter('coverage_target_pct').value
        self.max_time = self.get_parameter('max_mission_time_s').value
        self.report_interval = self.get_parameter('report_interval_s').value
        self.node_timeout = self.get_parameter('node_timeout_s').value

        # FSM
        self.state = 'STARTING'  # STARTING, EXPLORING, COMPLETE
        self.mission_start_time = None
        self.coverage_pct = 0.0
        self.symbols = []
        self.exploration_state = 'IDLE'
        self.controller_state = 'NO_PATH'

        # Heartbeat tracking
        self.last_exploration_time = 0.0
        self.last_controller_time = 0.0

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
            String, '/exploration/status', self._exploration_cb, 10)
        self.create_subscription(
            String, '/controller/status', self._controller_cb, 10)
        self.create_subscription(
            String, '/symbols/report', self._symbols_cb, reliable_transient)
        self.create_subscription(
            Float32, '/coverage/percent', self._coverage_cb, 10)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)

        # Publishers
        self.status_pub = self.create_publisher(String, '/mission/status', 10)
        self.command_pub = self.create_publisher(String, '/mission/command', 10)

        # Timers
        self.create_timer(1.0, self._tick)
        self.create_timer(self.report_interval, self._report_tick)

        self.get_logger().info(
            f'Mission manager: target={self.coverage_target}%, '
            f'max_time={self.max_time}s')

    # ── Callbacks ──────────────────────────────────────────────────────

    def _exploration_cb(self, msg: String):
        now = self.get_clock().now().nanoseconds / 1e9
        self.last_exploration_time = now
        try:
            data = json.loads(msg.data)
            self.exploration_state = data.get('state', 'IDLE')
        except (json.JSONDecodeError, KeyError):
            pass

    def _controller_cb(self, msg: String):
        now = self.get_clock().now().nanoseconds / 1e9
        self.last_controller_time = now
        try:
            data = json.loads(msg.data)
            self.controller_state = data.get('state', 'NO_PATH')
        except (json.JSONDecodeError, KeyError):
            pass

    def _symbols_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.symbols = data.get('symbols', [])
        except (json.JSONDecodeError, KeyError):
            pass

    def _coverage_cb(self, msg: Float32):
        self.coverage_pct = msg.data

    def _odom_cb(self, msg: Odometry):
        # Just used to confirm odom is alive; start mission when first odom arrives
        if self.state == 'STARTING' and self.mission_start_time is None:
            self.mission_start_time = self.get_clock().now().nanoseconds / 1e9
            self._send_command('START')
            self.state = 'EXPLORING'
            self.get_logger().info('Mission STARTED — odom online, sending START to sweep planner')

    # ── Main tick ──────────────────────────────────────────────────────

    def _tick(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == 'COMPLETE':
            self._publish_status()
            return

        if self.state == 'STARTING':
            self._publish_status()
            return

        # Check mission completion conditions
        elapsed = now - self.mission_start_time if self.mission_start_time else 0.0

        if self.coverage_pct >= self.coverage_target:
            self.state = 'COMPLETE'
            self._send_command('STOP')
            self.get_logger().info(
                f'Mission COMPLETE — coverage {self.coverage_pct:.1f}% '
                f'>= target {self.coverage_target}%')
        elif elapsed >= self.max_time:
            self.state = 'COMPLETE'
            self._send_command('STOP')
            self.get_logger().info(
                f'Mission COMPLETE — time limit {self.max_time}s reached, '
                f'coverage={self.coverage_pct:.1f}%')
        elif self.exploration_state == 'DONE':
            self.state = 'COMPLETE'
            self.get_logger().info(
                f'Mission COMPLETE — sweep planner reports DONE, '
                f'coverage={self.coverage_pct:.1f}%')

        # Heartbeat monitoring
        if self.last_exploration_time > 0 and now - self.last_exploration_time > self.node_timeout:
            self.get_logger().warn(
                f'No heartbeat from exploration for '
                f'{now - self.last_exploration_time:.1f}s',
                throttle_duration_sec=5.0)
        if self.last_controller_time > 0 and now - self.last_controller_time > self.node_timeout:
            self.get_logger().warn(
                f'No heartbeat from controller for '
                f'{now - self.last_controller_time:.1f}s',
                throttle_duration_sec=5.0)

        self._publish_status()

    def _report_tick(self):
        """Periodic status report to logger."""
        if self.state != 'EXPLORING':
            return
        elapsed = (self.get_clock().now().nanoseconds / 1e9 - self.mission_start_time
                   if self.mission_start_time else 0.0)
        self.get_logger().info(
            f'[Report] t={elapsed:.0f}s coverage={self.coverage_pct:.1f}% '
            f'symbols={len(self.symbols)} '
            f'explore={self.exploration_state} ctrl={self.controller_state}')

    # ── Helpers ────────────────────────────────────────────────────────

    def _send_command(self, command: str):
        msg = String()
        msg.data = json.dumps({'command': command})
        self.command_pub.publish(msg)

    def _publish_status(self):
        elapsed = (self.get_clock().now().nanoseconds / 1e9 - self.mission_start_time
                   if self.mission_start_time else 0.0)
        status = {
            'state': self.state,
            'elapsed_s': round(elapsed, 1),
            'coverage_pct': round(self.coverage_pct, 1),
            'symbols_found': len(self.symbols),
            'symbols': self.symbols,
            'exploration': self.exploration_state,
            'controller': self.controller_state
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
