#!/usr/bin/env python3
"""Mission Manager Node — Round 3 hackathon FSM orchestrator.

Drives Nav2 via nav2_simple_commander. Overrides planner goals based on
AprilTag commands and detected tile positions.
"""

import json
import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Empty, Int32MultiArray
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, Twist
import tf2_ros
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from mini_r1_v1_round3.utils.sweep_planner import (
    boustrophedon_waypoints, grid_cell_waypoints, nearest_remaining,
)


EXPLORING = 'EXPLORING'
STOP_AT_TAG = 'STOP_AT_TAG'
EXECUTE_TAG_TURN = 'EXECUTE_TAG_TURN'
FOLLOW_GREEN = 'FOLLOW_GREEN'
FOLLOW_ORANGE = 'FOLLOW_ORANGE'
GOING_TO_STOP_ZONE = 'GOING_TO_STOP_ZONE'
HALTED = 'HALTED'


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class MissionManagerNode(Node):
    def __init__(self):
        super().__init__('mission_manager_node')

        self.declare_parameter('arena_min_x', 0.0)
        self.declare_parameter('arena_min_y', 0.0)
        self.declare_parameter('arena_max_x', 5.0)
        self.declare_parameter('arena_max_y', 5.0)
        self.declare_parameter('sweep_stride_m', 0.5)
        self.declare_parameter('sweep_margin_m', 0.35)
        self.declare_parameter('grid_cell_size_m', 0.9)
        self.declare_parameter('grid_mode', True)
        self.declare_parameter('grid_cell_dwell_s', 0.4)
        self.declare_parameter('follow_color_giveup_radius_m', 2.0)
        self.declare_parameter('tag_turn_distance_m', 1.0)
        self.declare_parameter('stop_at_tag_dwell_s', 1.0)
        self.declare_parameter('map_warp_threshold_m', 0.3)
        self.declare_parameter('idle_watchdog_s', 4.0)

        self.arena_min_x = self.get_parameter('arena_min_x').value
        self.arena_min_y = self.get_parameter('arena_min_y').value
        self.arena_max_x = self.get_parameter('arena_max_x').value
        self.arena_max_y = self.get_parameter('arena_max_y').value
        self.sweep_stride_m = self.get_parameter('sweep_stride_m').value
        self.sweep_margin_m = self.get_parameter('sweep_margin_m').value
        self.grid_cell_size_m = self.get_parameter('grid_cell_size_m').value
        self.grid_mode = bool(self.get_parameter('grid_mode').value)
        self.grid_cell_dwell_s = self.get_parameter('grid_cell_dwell_s').value
        self.follow_color_giveup_radius_m = self.get_parameter('follow_color_giveup_radius_m').value
        self.tag_turn_distance_m = self.get_parameter('tag_turn_distance_m').value
        self.stop_at_tag_dwell_s = self.get_parameter('stop_at_tag_dwell_s').value
        self.map_warp_threshold_m = self.get_parameter('map_warp_threshold_m').value
        self.idle_watchdog_s = self.get_parameter('idle_watchdog_s').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav = BasicNavigator()
        self._nav_ready = False

        self.create_subscription(String, '/tag_commands', self._on_tag_command, 10)
        self.create_subscription(PoseArray, '/detected_tiles', self._on_detected_tiles, 10)
        self.create_subscription(String, '/tile_metadata', self._on_tile_metadata, 10)
        self.create_subscription(Int32MultiArray, '/visited_tiles', self._on_visited_tiles, 10)
        self.create_subscription(PoseStamped, '/stop_zone', self._on_stop_zone, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._on_odom, 20)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, 1)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._on_costmap, 1)
        self._have_map = False
        self._have_costmap = False
        self._map_received_at = 0.0
        self._costmap_warmup_s = 3.0
        self.create_subscription(String, '/logged_tags', self._on_logged_tags, 10)
        latched_sub = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, '/apriltag/map_positions',
            self._on_tag_positions, latched_sub,
        )
        self._tag_positions = {}

        self.create_subscription(
            PoseStamped, '/logo/pose',
            self._on_logo_pose, latched_sub,
        )
        # List of (yaw_rad, x, y) — one entry per detected logo, in detection order.
        self._logo_orientations = []

        self.state_pub = self.create_publisher(String, '/mission/state', 10)
        self.ended_pub = self.create_publisher(Empty, '/mission_ended', 1)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._nav_goal_active = False
        self._nav_goal_submitted_at = 0.0
        self._nav_last_reached = False

        self.state = EXPLORING
        self._prev_state = None
        self._state_entered_at = time.time()
        self._mission_ended_fired = False

        if self.grid_mode:
            self.sweep_waypoints = grid_cell_waypoints(
                self.arena_min_x, self.arena_min_y,
                self.arena_max_x, self.arena_max_y,
                self.grid_cell_size_m, self.sweep_margin_m,
            )
            self.get_logger().info(
                f'Grid mode: {len(self.sweep_waypoints)} cells @ '
                f'{self.grid_cell_size_m}m stride'
            )
        else:
            self.sweep_waypoints = boustrophedon_waypoints(
                self.arena_min_x, self.arena_min_y,
                self.arena_max_x, self.arena_max_y,
                self.sweep_stride_m, self.sweep_margin_m,
            )
        self.sweep_idx = 0
        self._current_goal = None
        self._cell_reached_at = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self._have_odom = False

        self.detected_tiles = []
        self.tile_metadata = []
        self.visited_tile_ids = set()

        self.stop_zone_pose = None
        self.logged_tags_count = 0

        self._pending_tag_cmd = None

        self._last_map_odom = None
        self._warp_detected = False

        self._odom_history = deque(maxlen=200)
        self._spin_triggers = deque()

        self.create_timer(0.1, self._tick)
        self.create_timer(0.2, self._warp_watchdog)
        self.create_timer(1.0, self._idle_watchdog)
        self.create_timer(0.5, self._publish_state)

        self._initial_goal_sent = False
        self.get_logger().info('Mission manager initialized. Starting EXPLORING.')

    def _on_tag_command(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Bad tag_commands JSON: {e}')
            return
        cmd = data.get('command', '').lower()
        self.get_logger().info(f'Tag command received: {cmd} (id={data.get("tag_id")})')
        if self.state in (HALTED,):
            return
        self._pending_tag_cmd = {
            'command': cmd,
            'tag_id': data.get('tag_id'),
            'stamp': data.get('stamp', time.time()),
        }
        self._transition(STOP_AT_TAG)

    def _on_detected_tiles(self, msg: PoseArray):
        tiles = []
        for p in msg.poses:
            tiles.append((p.position.x, p.position.y))
        self.detected_tiles = tiles

    def _on_tile_metadata(self, msg: String):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.tile_metadata = data
            elif isinstance(data, dict) and 'tiles' in data:
                self.tile_metadata = data['tiles']
        except Exception as e:
            self.get_logger().warn(f'Bad tile_metadata JSON: {e}')

    def _on_visited_tiles(self, msg: Int32MultiArray):
        self.visited_tile_ids = set(msg.data)

    def _on_stop_zone(self, msg: PoseStamped):
        self.stop_zone_pose = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'Stop zone received at {self.stop_zone_pose}')

    def _on_odom(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        if not self._have_odom:
            self.get_logger().info(
                f'[ODOM] First odom received at ({self.robot_x:.2f}, {self.robot_y:.2f})')
        self._have_odom = True
        self._odom_history.append((time.time(), self.robot_x, self.robot_y))

    def _on_map(self, msg: OccupancyGrid):
        if not self._have_map:
            self._have_map = True
            self._map_received_at = time.time()
            self.get_logger().info('[MAP] First map received — waiting for costmap...')

    def _on_costmap(self, msg: OccupancyGrid):
        if not self._have_costmap:
            self._have_costmap = True
            self.get_logger().info('[MAP] Global costmap ready — goals can now be sent.')

    def _on_tag_positions(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Bad tag_positions JSON: {e}')
            return
        for k, v in data.items():
            try:
                self._tag_positions[int(k)] = (float(v[0]), float(v[1]))
            except (ValueError, IndexError, TypeError):
                continue

    def _on_logo_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        lx = msg.pose.position.x
        ly = msg.pose.position.y
        self._logo_orientations.append((yaw, lx, ly))
        self.get_logger().info(
            f'[LOGO] #{len(self._logo_orientations)} orientation received: '
            f'{math.degrees(yaw):.1f}° at ({lx:.2f},{ly:.2f})'
        )

    def _on_logged_tags(self, msg: String):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.logged_tags_count = len(data)
            elif isinstance(data, dict):
                self.logged_tags_count = int(data.get('count', len(data.get('logged', []))))
        except Exception as e:
            self.get_logger().warn(f'Bad logged_tags JSON: {e}')

    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    def _transition(self, new_state):
        if new_state == self.state:
            return
        self.get_logger().info(f'FSM: {self.state} -> {new_state}')
        self._prev_state = self.state
        self.state = new_state
        self._state_entered_at = time.time()
        self._on_state_enter(new_state)

    def _make_pose_stamped(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def _cancel_mpc(self):
        if not self._nav_goal_active:
            return
        try:
            self.nav.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'[NAV] cancelTask raised: {e!r}')
        self._nav_goal_active = False
        self.cmd_pub.publish(Twist())

    def _send_goal(self, x, y, yaw):
        self._current_goal = (x, y, yaw)
        self.get_logger().info(
            f'[GOAL] goToPose ({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.1f}°) '
            f'from robot=({self.robot_x:.2f}, {self.robot_y:.2f})')
        goal_ps = self._make_pose_stamped(x, y, yaw)
        try:
            self.nav.goToPose(goal_ps)
        except Exception as e:
            self.get_logger().error(f'[GOAL] goToPose raised: {e!r}')
            self._nav_goal_active = False
            return
        self._nav_goal_active = True
        self._nav_last_reached = False
        self._nav_goal_submitted_at = time.time()

    def _mpc_is_complete(self) -> bool:
        if not self._nav_goal_active:
            return True
        try:
            done = self.nav.isTaskComplete()
        except Exception as e:
            self.get_logger().warn(f'[NAV] isTaskComplete raised: {e!r}')
            return False
        if not done:
            return False
        self._nav_goal_active = False
        try:
            result = self.nav.getResult()
        except Exception:
            result = None
        self._nav_last_reached = (result == TaskResult.SUCCEEDED)
        if self._nav_last_reached:
            self.get_logger().info('[NAV] Goal reached')
        else:
            self.get_logger().warn(f'[NAV] Goal ended: {result}')
        return True

    def _tiles_of_color(self, color):
        out = []
        n = min(len(self.detected_tiles), len(self.tile_metadata))
        for i in range(n):
            meta = self.tile_metadata[i] if isinstance(self.tile_metadata[i], dict) else {}
            if meta.get('color', '').lower() == color:
                tid = meta.get('id', i)
                if tid in self.visited_tile_ids:
                    continue
                out.append((self.detected_tiles[i], tid))
        return out

    def _nearest_tile(self, tiles):
        best = None
        best_d = float('inf')
        for (pos, tid) in tiles:
            d = math.hypot(pos[0] - self.robot_x, pos[1] - self.robot_y)
            if d < best_d:
                best_d = d
                best = (pos, tid, d)
        return best

    def _stop_zone_gate_check(self):
        if self.state in (GOING_TO_STOP_ZONE, HALTED):
            return False
        if self.stop_zone_pose is not None and self.logged_tags_count >= 5:
            return True
        arena_tile_count = 25
        if arena_tile_count > 0:
            ratio = len(self.visited_tile_ids) / float(arena_tile_count)
            if ratio >= 0.8 and self.stop_zone_pose is not None:
                return True
        return False

    def _warp_watchdog(self):
        try:
            tf = self.tf_buffer.lookup_transform('odom', 'map', rclpy.time.Time())
        except Exception:
            return
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        if self._last_map_odom is None:
            self._last_map_odom = (tx, ty, yaw)
            return
        dx = tx - self._last_map_odom[0]
        dy = ty - self._last_map_odom[1]
        dyaw = abs(math.atan2(math.sin(yaw - self._last_map_odom[2]),
                              math.cos(yaw - self._last_map_odom[2])))
        trans_delta = math.hypot(dx, dy)
        if trans_delta > self.map_warp_threshold_m or dyaw > math.radians(15.0):
            self.get_logger().warn(
                f'Map->odom warp detected: trans={trans_delta:.2f}m, yaw={math.degrees(dyaw):.1f}deg')
            self._warp_detected = True
        self._last_map_odom = (tx, ty, yaw)

    def _idle_watchdog(self):
        if self.state in (STOP_AT_TAG, HALTED):
            return
        if not self._have_odom or len(self._odom_history) < 2:
            return
        now = time.time()
        oldest = self._odom_history[0]
        if now - oldest[0] < self.idle_watchdog_s:
            return
        ref = oldest
        dx = self.robot_x - ref[1]
        dy = self.robot_y - ref[2]
        if math.hypot(dx, dy) < 0.02:
            self.get_logger().warn(
                f'Idle watchdog: no motion in {self.idle_watchdog_s:.0f}s '
                f'(dx={dx:.3f}, dy={dy:.3f}). Letting Nav2 BT handle recovery.'
            )

    def _on_state_enter(self, state):
        if state == EXPLORING:
            self._enter_exploring()
        elif state == STOP_AT_TAG:
            self._cancel_mpc()
        elif state == EXECUTE_TAG_TURN:
            self._enter_execute_tag_turn()
        elif state == FOLLOW_GREEN:
            self._enter_follow_color('green')
        elif state == FOLLOW_ORANGE:
            self._enter_follow_color('orange')
        elif state == GOING_TO_STOP_ZONE:
            if self.stop_zone_pose is not None:
                x, y = self.stop_zone_pose
                self._send_goal(x, y, self.robot_yaw)
        elif state == HALTED:
            self._cancel_mpc()
            if not self._mission_ended_fired:
                self.ended_pub.publish(Empty())
                self._mission_ended_fired = True
            self.get_logger().info('Mission HALTED. Done.')

    def _enter_exploring(self):
        self.get_logger().info(
            f'[ENTER_EXPLORING] sweep_idx={self.sweep_idx}/{len(self.sweep_waypoints)}, '
            f'have_odom={self._have_odom}')
        if self.sweep_idx >= len(self.sweep_waypoints):
            self.get_logger().info('[ENTER_EXPLORING] Sweep complete.')
            return
        try:
            visited = set(range(self.sweep_idx))
            idx = nearest_remaining(
                (self.robot_x, self.robot_y), self.sweep_waypoints, visited)
            if idx is not None and idx != self.sweep_idx:
                self.sweep_waypoints[self.sweep_idx], self.sweep_waypoints[idx] = \
                    self.sweep_waypoints[idx], self.sweep_waypoints[self.sweep_idx]
            wp = self.sweep_waypoints[self.sweep_idx]
        except Exception as e:
            self.get_logger().warn(f'[ENTER_EXPLORING] nearest_remaining failed: {e}')
            wp = self.sweep_waypoints[self.sweep_idx]
        x, y = wp[0], wp[1]
        # On the first waypoint, align to the first detected logo orientation so
        # the robot heads in the direction the logo points (green → orange axis).
        if self.sweep_idx == 0 and self._logo_orientations:
            yaw = self._logo_orientations[0][0]
            self.get_logger().info(
                f'[ENTER_EXPLORING] Using logo heading {math.degrees(yaw):.1f}° '
                f'for initial waypoint'
            )
        else:
            yaw = math.atan2(y - self.robot_y, x - self.robot_x)
        self._send_goal(x, y, yaw)

    def _enter_execute_tag_turn(self):
        pending = self._pending_tag_cmd or {}
        cmd = pending.get('command', '')
        tag_id = pending.get('tag_id')

        if cmd == 'left':
            delta = math.pi / 2.0
        elif cmd == 'right':
            delta = -math.pi / 2.0
        elif cmd == 'u-turn':
            delta = math.pi
        else:
            delta = 0.0

        tag_pos = self._tag_positions.get(tag_id) if tag_id is not None else None
        if tag_pos is not None:
            tx, ty = tag_pos
            base_yaw = math.atan2(ty - self.robot_y, tx - self.robot_x)
            ref = f'head-on to tag#{tag_id}@({tx:.2f},{ty:.2f})'
        else:
            base_yaw = self.robot_yaw
            ref = 'current robot yaw (tag pos unknown)'

        new_yaw = base_yaw + delta
        gx = self.robot_x + self.tag_turn_distance_m * math.cos(new_yaw)
        gy = self.robot_y + self.tag_turn_distance_m * math.sin(new_yaw)
        self.get_logger().info(
            f'Tag turn {cmd}: ref={ref} base_yaw={math.degrees(base_yaw):.1f}° '
            f'-> new_yaw={math.degrees(new_yaw):.1f}° goal=({gx:.2f},{gy:.2f})'
        )
        self._send_goal(gx, gy, new_yaw)

    def _enter_follow_color(self, color):
        tiles = self._tiles_of_color(color)
        best = self._nearest_tile(tiles)
        if best is None or best[2] > self.follow_color_giveup_radius_m:
            self.get_logger().info(f'No {color} tile within radius, back to EXPLORING.')
            self._transition(EXPLORING)
            return
        (pos, tid, d) = best
        yaw = math.atan2(pos[1] - self.robot_y, pos[0] - self.robot_x)
        self.get_logger().info(f'Following {color} tile id={tid} at {pos} (d={d:.2f}m)')
        self._send_goal(pos[0], pos[1], yaw)

    def _tick(self):
        if self._warp_detected:
            self._warp_detected = False
            self.get_logger().info(f'Warp detected, re-entering state {self.state}.')
            self._on_state_enter(self.state)

        if self.state not in (STOP_AT_TAG, EXECUTE_TAG_TURN, GOING_TO_STOP_ZONE, HALTED):
            if self._stop_zone_gate_check():
                self._transition(GOING_TO_STOP_ZONE)
                return

        if self.state == EXPLORING:
            self._tick_exploring()
        elif self.state == STOP_AT_TAG:
            self._tick_stop_at_tag()
        elif self.state == EXECUTE_TAG_TURN:
            self._tick_execute_tag_turn()
        elif self.state == FOLLOW_GREEN:
            self._tick_follow_color('green')
        elif self.state == FOLLOW_ORANGE:
            self._tick_follow_color('orange')
        elif self.state == GOING_TO_STOP_ZONE:
            self._tick_going_to_stop_zone()
        elif self.state == HALTED:
            pass

    def _tick_exploring(self):
        if self.sweep_idx >= len(self.sweep_waypoints):
            return
        if not self._nav_ready:
            try:
                ready = self.nav.nav_to_pose_client.wait_for_server(timeout_sec=0.0)
            except Exception:
                ready = False
            if not ready:
                self.get_logger().info(
                    '[TICK] Waiting for Nav2 action server...',
                    throttle_duration_sec=2.0)
                return
            self._nav_ready = True
            self.get_logger().info('Nav2 action server up.')
        if not self._initial_goal_sent:
            if not self._have_odom:
                self.get_logger().info(
                    '[TICK] Waiting for odometry before sending first goal...',
                    throttle_duration_sec=2.0)
                return
            if not self._have_map:
                self.get_logger().info(
                    '[TICK] Waiting for SLAM map before sending first goal...',
                    throttle_duration_sec=2.0)
                return
            costmap_ready = self._have_costmap or (
                self._map_received_at > 0
                and (time.time() - self._map_received_at) >= self._costmap_warmup_s
            )
            if not costmap_ready:
                self.get_logger().info(
                    '[TICK] Waiting for Nav2 costmap to initialize...',
                    throttle_duration_sec=2.0)
                return
            self.get_logger().info(
                f'[TICK] Sending INITIAL exploration goal from '
                f'({self.robot_x:.2f},{self.robot_y:.2f}). '
                f'Total waypoints={len(self.sweep_waypoints)}')
            self._initial_goal_sent = True
            self._enter_exploring()
            return
        if self._mpc_is_complete():
            if self._nav_last_reached:
                self.get_logger().info(f'[TICK] Sweep waypoint {self.sweep_idx} reached.')
                if self.grid_mode and self._cell_reached_at is None:
                    self._cell_reached_at = time.time()
                    self.cmd_pub.publish(Twist())
                    return
                if self.grid_mode and (time.time() - self._cell_reached_at) < self.grid_cell_dwell_s:
                    return
            else:
                self.get_logger().warn(f'[TICK] MPC inactive without reaching goal, advancing.')
            self._cell_reached_at = None
            self.sweep_idx += 1
            if self.sweep_idx < len(self.sweep_waypoints):
                self._enter_exploring()

    def _tick_stop_at_tag(self):
        if (time.time() - self._state_entered_at) < self.stop_at_tag_dwell_s:
            return
        cmd = (self._pending_tag_cmd or {}).get('command', '')
        if cmd in ('left', 'right', 'u-turn'):
            self._transition(EXECUTE_TAG_TURN)
        elif cmd == 'follow-green':
            self._transition(FOLLOW_GREEN)
        elif cmd == 'follow-orange':
            self._transition(FOLLOW_ORANGE)
        else:
            self.get_logger().warn(f'Unknown tag cmd "{cmd}", resuming EXPLORING.')
            self._transition(EXPLORING)

    def _tick_execute_tag_turn(self):
        if self._mpc_is_complete():
            self.get_logger().info(f'Tag turn done (reached={self._nav_last_reached}), resuming EXPLORING.')
            self._pending_tag_cmd = None
            self._transition(EXPLORING)

    def _tick_follow_color(self, color):
        if self._mpc_is_complete():
            self.get_logger().info(f'Follow-{color} done (reached={self._nav_last_reached}), resuming EXPLORING.')
            self._transition(EXPLORING)

    def _tick_going_to_stop_zone(self):
        if self._mpc_is_complete():
            self.get_logger().info(f'Stop-zone arrival (reached={self._nav_last_reached}).')
            self._transition(HALTED)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
