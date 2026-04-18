#!/usr/bin/env python3
"""Mission Manager Node — Round 3 hackathon FSM orchestrator.

Drives Nav2 via nav2_simple_commander. Overrides planner goals based on
AprilTag commands and detected tile positions.
"""

import json
import math
import time
from collections import deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Empty, Int32MultiArray, Bool
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
# New gated-tag-action states (replace the old STOP_AT_TAG/EXECUTE_TAG_TURN
# pair when the logical tag is 1 or 2):
TAG1_GOAL_A = 'TAG1_GOAL_A'        # drive to (tag1_x-0.5, tag1_y, yaw=90°)
TAG1_GOAL_B = 'TAG1_GOAL_B'        # drive forward to (tag1_x-0.5, tag1_y+1.5, yaw=90°)
LOOKING_FOR_TAG2 = 'LOOKING_FOR_TAG2'  # rotate in place until tag 2 is centered in image
TAG2_GOAL   = 'TAG2_GOAL'          # drive to (tag2_x, tag2_y-1.5, yaw=0°)


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
        self.declare_parameter('idle_watchdog_s', 3.5)

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
        # Also listen to /apriltag/map_positions so we know tag coords
        # independent of the command stream ordering.
        self.create_subscription(String, '/apriltag/map_positions',
                                 self._on_tag_positions_msg, 10)
        # Live apriltag detections (raw IDs) — used in LOOKING_FOR_TAG2 to
        # detect when tag 2 is centered in the camera frame.
        from apriltag_msgs.msg import AprilTagDetectionArray as _AtaArray
        self._AtaArray = _AtaArray
        self.create_subscription(
            _AtaArray, '/apriltag/detections',
            self._on_apriltag_detections_raw, 10)
        self.create_subscription(PoseArray, '/detected_tiles', self._on_detected_tiles, 10)
        self.create_subscription(String, '/tile_metadata', self._on_tile_metadata, 10)
        self.create_subscription(Int32MultiArray, '/visited_tiles', self._on_visited_tiles, 10)
        self.create_subscription(PoseStamped, '/stop_zone', self._on_stop_zone, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._on_odom, 20)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map: OccupancyGrid | None = None
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
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

        # Recovery state: when the idle watchdog fires, we cancel the active
        # goal, run nav.backup(), and on completion re-send _current_goal.
        self._recovery_active = False
        self._recovery_started_at = 0.0

        # External pause gate (e.g., AprilTag command handler calls this to
        # freeze navigation while it re-plans the override).
        self._paused = False
        self.create_subscription(Bool, '/mission/pause', self._on_pause_msg, 10)

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

        # Robot pose in the SLAM `map` frame (via TF map→base_link).
        # Frontier centroids, sweep waypoints, and Nav2 goal PoseStamped
        # are all in map frame. Using odom-frame robot_x/y against them
        # yields garbage distance scores once odom drifts from map.
        self._map_pose: tuple[float, float, float] | None = None
        self.create_timer(0.1, self._update_map_pose)

        self.detected_tiles = []
        self.tile_metadata = []
        self.visited_tile_ids = set()

        self.stop_zone_pose = None
        self.logged_tags_count = 0

        self._pending_tag_cmd = None
        # Locked tag IDs: once an AprilTag's command has been dispatched to
        # the FSM, we never re-act on the same ID — even if apriltag_ros
        # keeps re-publishing detections as the robot passes the marker.
        # Mirrors legacy marker_detector_node.locked_viz_markers pattern.
        self._executed_tag_ids: set[int] = set()
        # Logical-tag bookkeeping for the new gated flow.
        # _tag_positions_map: logical_id -> (x, y, z) in odom/map frame.
        # _last_raw_detection: most recent raw-id apriltag pixel centre
        #   (raw_id -> (px_u, px_v, frame_w)) used by LOOKING_FOR_TAG2.
        # _action_next_goal: tuple queued to dispatch inside _tick as
        #   soon as we enter a TAGn_GOAL state (set by _on_state_enter).
        self._tag_positions_map: dict[int, tuple[float, float, float]] = {}
        self._last_raw_detection: dict[int, tuple[int, int, int]] = {}
        self._action_next_goal: tuple[float, float, float] | None = None
        # Image-center thresholds used to decide when tag 2 is "perfectly
        # in view" during LOOKING_FOR_TAG2. Matches a 640-wide camera.
        self.TAG_CENTERED_PIXEL_TOL = 35
        self.LOOK_ROTATE_OMEGA = 0.45   # rad/s while scanning for tag 2
        self._tag1_pos: tuple[float, float] | None = None
        self._tag2_pos: tuple[float, float] | None = None

        self._last_map_odom = None
        self._warp_detected = False

        self._odom_history = deque(maxlen=200)
        # Goal-progress watchdog: rolling window of (timestamp, dist_to_goal).
        # If dist_to_goal isn't shrinking by GOAL_PROGRESS_MIN_M over
        # GOAL_PROGRESS_WINDOW_S, we abandon the waypoint (it's either
        # unreachable given the current map, or Nav2 is looping).
        self._goal_dist_history: deque[tuple[float, float]] = deque(maxlen=200)
        self.GOAL_PROGRESS_WINDOW_S = 5.0
        self.GOAL_PROGRESS_MIN_M = 0.10
        self._spin_triggers = deque()

        # ── Visited-waypoint tracking (anti-backtrack) ──────────────────
        # Positions (quantized to 5cm) of waypoints we've either completed or
        # physically driven within WAYPOINT_VISIT_RADIUS_M of en route to
        # something else. Keyed by position rather than list index because
        # _enter_exploring() swaps sweep_waypoints entries during selection,
        # which would invalidate index-based tracking. Any waypoint whose
        # position matches is excluded from future goal selection.
        self._visited_positions: set[tuple[int, int]] = set()
        self.WAYPOINT_VISIT_RADIUS_M = 0.35  # ~half a 0.9m tile

        # ── Frontier exploration ───────────────────────────────────────
        # Replaces boustrophedon sweep as primary goal source. We detect
        # free-cells-adjacent-to-unknown in the SLAM map, cluster via BFS,
        # and score clusters by size / (distance + eps). The sweep list
        # remains as a fallback for pre-map startup and post-exhaustion.
        self.FRONTIER_FREE_MAX = 20            # occupancy ≤ → "free"
        self.FRONTIER_OCC_MIN = 65             # occupancy ≥ → "occupied"
        self.FRONTIER_MIN_CLUSTER = 4          # cells, filters noise
        self.FRONTIER_BLACKLIST_RADIUS_M = 0.5 # centroid skip radius
        self.FRONTIER_NEAR_OBSTACLE_CELLS = 1  # inflate away this many cells
        # Reachable-waypoint tuning. We pick cells that sit in already-mapped
        # free space with enough clearance for the robot (so Nav2 has a
        # navigable path) and that are close to an unknown-region boundary
        # (so getting there actually reveals new map).
        # Clearance + distance tuning: give Nav2's RegulatedPurePursuit a
        # generously-clear, non-trivial goal each time. The previous values
        # picked cells too close to walls (stall inside narrow passages)
        # and too close to the robot (bot "reached" instantly, burned time
        # on re-planning instead of driving).
        self.REACH_MIN_CLEARANCE_M = 0.28      # free-cell must be ≥ this from nearest wall
        self.REACH_MAX_UNKNOWN_DIST_M = 1.20   # and ≤ this from nearest unknown cell
        self.REACH_STRIDE_M = 0.25             # down-sample candidate grid to this spacing
        self.REACH_MIN_GOAL_DIST_M = 0.60      # skip candidates closer than this to robot
        self.VISITED_RADIUS_M = 0.40           # cells within this of trajectory = "been there"
        # Radius over which we sum unknown cells. Matches the lidar sensor
        # horizon (~2 m useful revealing radius) — bigger = bias waypoints
        # into the largest unknown pockets.
        self.UNKNOWN_DENSITY_RADIUS_M = 2.0
        # Continuous goal re-selection: every tick while a goal is live, we
        # re-score candidate waypoints and preempt the current Nav2 goal if a
        # meaningfully better one has appeared (the robot revealed new map, so
        # a waypoint closer to the largest unknown region is now available).
        # Patience knobs: goals stick until progress fails or we arrive.
        # We only preempt if a LOT of time has passed AND the new candidate
        # is very different. With the new unknown-density scorer this rarely
        # needs to fire — the goal we picked will still be the best goal.
        self.GOAL_UPDATE_INTERVAL_S = 12.0
        self.GOAL_UPDATE_DIST_M = 1.20
        self._last_goal_update_at = 0.0
        self._current_frontier: tuple[float, float] | None = None
        self._frontier_last_logged = 0.0
        # Cooldown timestamp: when _enter_exploring can't find anything,
        # we wait FRONTIER_IDLE_RETRY_S before trying again so the map has
        # time to grow (slam_toolbox scan-matching, bot drifting slightly).
        self._next_frontier_retry_at = 0.0
        self.FRONTIER_IDLE_RETRY_S = 2.0

        # ── cmd_vel tap (for diagnostics only) ──────────────────────────
        self._last_cmd = (0.0, 0.0)
        self._last_cmd_stamp = 0.0
        self._cmd_count = 0
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel_tap, 20)

        self.create_timer(0.1, self._tick)
        self.create_timer(0.2, self._warp_watchdog)
        self.create_timer(0.2, self._passthrough_visit_check)
        self.create_timer(1.0, self._idle_watchdog)
        self.create_timer(1.0, self._nav_diag)
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
        tag_id = data.get('tag_id')
        self.get_logger().info(f'Tag command received: {cmd} (logical_id={tag_id})')
        if self.state in (HALTED,):
            return
        if tag_id is not None and tag_id in self._executed_tag_ids:
            self.get_logger().info(
                f'Tag#{tag_id} already executed — ignoring re-detection.')
            return

        # New gated tag flow: tags 1 and 2 drive a two-stage FSM. Tags 3+
        # fall back to the legacy STOP_AT_TAG path until their actions
        # are wired up.
        if tag_id == 1:
            pos = self._tag_positions_map.get(1)
            if pos is None:
                self.get_logger().warn(
                    '[TAG1] command arrived but no /apriltag/map_positions '
                    'entry yet — deferring one cycle.')
                return
            self._executed_tag_ids.add(1)
            self._enter_tag1_flow(pos)
            return
        if tag_id == 2:
            # Tag 2 action only runs after tag 1. If tag 1 hasn't run,
            # buffer: record its position for later rotate-to-face and
            # take no action now.
            if 1 not in self._executed_tag_ids:
                self.get_logger().info(
                    '[TAG2] seen BEFORE tag 1 — buffering sighting, '
                    'will act on it after tag 1 action completes.')
                return
            pos = self._tag_positions_map.get(2)
            if pos is None:
                self.get_logger().warn(
                    '[TAG2] command arrived but no /apriltag/map_positions '
                    'entry yet — deferring.')
                return
            self._executed_tag_ids.add(2)
            self._enter_tag2_flow(pos)
            return

        # Legacy path for tags 3-5 (not yet wired to new gated flow).
        self._pending_tag_cmd = {
            'command': cmd,
            'tag_id': tag_id,
            'stamp': data.get('stamp', time.time()),
        }
        if tag_id is not None:
            self._executed_tag_ids.add(tag_id)
        self._transition(STOP_AT_TAG)

    def _on_tag_positions_msg(self, msg: String):
        """Logo-detector-published JSON of logical_id -> [x, y, z]. Also
        used as a trigger: when logical tag 1 first becomes known AND we
        haven't executed it AND we're in EXPLORING, fire the tag-1 flow
        directly. This makes the gated flow robust to /tag_commands and
        /apriltag/map_positions arriving in either order."""
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Bad map_positions JSON: {e}')
            return
        newly_known: list[int] = []
        for k, v in data.items():
            try:
                logical_id = int(k)
                pos = (float(v[0]), float(v[1]), float(v[2]))
            except (ValueError, IndexError, TypeError):
                continue
            if logical_id not in self._tag_positions_map:
                newly_known.append(logical_id)
            self._tag_positions_map[logical_id] = pos
        for lid in newly_known:
            self._maybe_trigger_gated_tag(lid)

    def _maybe_trigger_gated_tag(self, logical_id: int):
        """Fire tag 1 or tag 2 action as soon as we have its position AND
        the predecessor has run AND we're in a state that can accept an
        action transition."""
        if logical_id in self._executed_tag_ids:
            return
        if self.state == HALTED:
            return
        pos = self._tag_positions_map.get(logical_id)
        if pos is None:
            return
        if logical_id == 1:
            if self.state in (TAG1_GOAL_A, TAG1_GOAL_B,
                              LOOKING_FOR_TAG2, TAG2_GOAL):
                return
            self._executed_tag_ids.add(1)
            self._enter_tag1_flow(pos)
        elif logical_id == 2:
            # Logo-detector's predecessor gate means tag 2 can only ever
            # appear in map_positions AFTER tag 1 is locked, so we're
            # clear to act. Any ongoing action keeps owning control —
            # LOOKING_FOR_TAG2 will consume the position itself.
            if self.state in (TAG1_GOAL_A, TAG1_GOAL_B,
                              LOOKING_FOR_TAG2, TAG2_GOAL):
                return
            self._executed_tag_ids.add(2)
            self._enter_tag2_flow(pos)

    def _on_apriltag_detections_raw(self, msg):
        """Capture raw apriltag pixel centres for the LOOKING_FOR_TAG2
        rotation-to-center test."""
        for det in msg.detections:
            try:
                raw_id = int(det.id)
            except (TypeError, ValueError):
                try:
                    raw_id = int(det.id[0])
                except Exception:
                    continue
            self._last_raw_detection[raw_id] = (
                int(det.centre.x), int(det.centre.y), 640)

    # ── Tag 1 and Tag 2 action entry points ──────────────────────────────
    def _enter_tag1_flow(self, tag_pos: tuple[float, float, float]):
        """Start the two-step Tag-1 action: goal A = (tx-0.5, ty, yaw=90°)."""
        tx, ty, _tz = tag_pos
        self._tag1_pos = (tx, ty)
        self._pause_exploration_for_action('TAG1')
        self._action_next_goal = (tx - 0.5, ty, math.radians(90.0))
        self._transition(TAG1_GOAL_A)

    def _enter_tag2_flow(self, tag_pos: tuple[float, float, float]):
        tx, ty, _tz = tag_pos
        self._tag2_pos = (tx, ty)
        self._pause_exploration_for_action('TAG2')
        self._action_next_goal = (tx, ty - 1.5, 0.0)
        self._transition(TAG2_GOAL)

    def _pause_exploration_for_action(self, tag_label: str):
        """Kill any in-flight exploration goal so the action is clean."""
        try:
            self.nav.cancelTask()
        except Exception:
            pass
        self._nav_goal_active = False
        self._nav_last_reached = False
        self._current_frontier = None
        self._goal_dist_history.clear()
        self._odom_history.clear()
        self.cmd_pub.publish(Twist())
        self.get_logger().warn(
            f'[{tag_label}] exploration paused — starting scripted action.')

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
        now_t = time.time()
        # Track yaw too so the stall watchdog can distinguish
        # "rotating-in-place to align with path" from "actually stuck".
        self._odom_history.append((now_t, self.robot_x, self.robot_y, self.robot_yaw))
        # Goal-progress telemetry — map frame for goal comparison when
        # available, otherwise fall back to odom.
        if self._nav_goal_active and self._current_goal is not None:
            rx, ry = self._map_xy()
            gx, gy, _gyaw = self._current_goal
            self._goal_dist_history.append((now_t, math.hypot(gx - rx, gy - ry)))

    def _on_map(self, msg: OccupancyGrid):
        self._map = msg
        if not self._have_map:
            self._have_map = True
            self._map_received_at = time.time()
            self.get_logger().info('[MAP] First map received — waiting for costmap...')

    def _on_costmap(self, msg: OccupancyGrid):
        if not self._have_costmap:
            self._have_costmap = True
            self.get_logger().info('[MAP] Global costmap ready — goals can now be sent.')

    def _update_map_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception:
            return
        q = t.transform.rotation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        x = float(t.transform.translation.x)
        y = float(t.transform.translation.y)
        self._map_pose = (x, y, yaw)
        # Mark the robot's trajectory itself as "visited" so frontier
        # centroids near where we already drove get excluded (via the
        # FRONTIER_BLACKLIST_RADIUS_M halo around each trajectory cell).
        # Without this, frontiers regenerated behind us stay eligible
        # and the bot oscillates back into explored areas.
        self._mark_position_visited(x, y)

    def _map_xy(self) -> tuple[float, float]:
        """Best-available robot (x, y) in map frame. Falls back to odom if
        the map→base_link transform is not yet published."""
        if self._map_pose is not None:
            return self._map_pose[0], self._map_pose[1]
        return self.robot_x, self.robot_y

    def _map_yaw(self) -> float:
        if self._map_pose is not None:
            return self._map_pose[2]
        return self.robot_yaw

    # ── Frontier exploration helpers ────────────────────────────────────
    def _frontier_blacklisted(self, x: float, y: float) -> bool:
        r = self.FRONTIER_BLACKLIST_RADIUS_M
        r2 = r * r
        for kx, ky in self._visited_positions:
            wx, wy = kx / 20.0, ky / 20.0
            if (wx - x) ** 2 + (wy - y) ** 2 <= r2:
                return True
        return False

    def _detect_frontiers(self) -> list[tuple[float, float, float, float, float]]:
        """Return exploration-candidate waypoints derived purely from the SLAM
        occupancy grid + the robot's trajectory. No arena bounds, no tile
        grid, nothing hardcoded. A candidate is a cell that is:

          - free (not an obstacle, not unknown),
          - at least REACH_MIN_CLEARANCE_M from the nearest wall (reachable),
          - more than VISITED_RADIUS_M from any cell the robot has been in
            (never physically occupied),
          - within REACH_MAX_UNKNOWN_DIST_M of an unknown cell (so visiting
            it will reveal new map — keeps the behavior exploration-based).

        Returns tuples (wx, wy, clearance_m, dist_to_unknown_m, dist_from_traj_m).
        """
        m = self._map
        if m is None:
            return []
        w = m.info.width
        h = m.info.height
        if w == 0 or h == 0:
            return []
        res = float(m.info.resolution)
        ox = float(m.info.origin.position.x)
        oy = float(m.info.origin.position.y)

        grid = np.asarray(m.data, dtype=np.int16).reshape(h, w)
        free_mask    = ((grid >= 0) & (grid <= self.FRONTIER_FREE_MAX)).astype(np.uint8)
        occ_mask     = (grid >= self.FRONTIER_OCC_MIN).astype(np.uint8)
        unknown_mask = (grid < 0).astype(np.uint8)
        if free_mask.sum() == 0:
            return []

        dist_obs_m = cv2.distanceTransform(1 - occ_mask, cv2.DIST_L2, 3) * res
        if unknown_mask.sum() > 0:
            dist_unk_m = cv2.distanceTransform(1 - unknown_mask, cv2.DIST_L2, 3) * res
        else:
            dist_unk_m = np.full((h, w), 1e6, dtype=np.float32)

        # Unknown density: count of unknown cells in a sensor-radius window
        # around each cell. Waypoints with a LOT of unknown around them are
        # worth much more than waypoints next to a tiny isolated pocket.
        density_win = max(1, int(round(self.UNKNOWN_DENSITY_RADIUS_M / res)))
        k = 2 * density_win + 1
        # cv2.boxFilter sums over the window; scale back to "count".
        unknown_density = cv2.boxFilter(
            unknown_mask.astype(np.float32), ddepth=-1,
            ksize=(k, k), normalize=False,
        )

        # Rasterise the robot's trajectory into this grid.
        traj_mask = np.zeros((h, w), dtype=np.uint8)
        for kx, ky in self._visited_positions:
            wx = kx / 20.0
            wy = ky / 20.0
            cx = int((wx - ox) / res)
            cy = int((wy - oy) / res)
            if 0 <= cx < w and 0 <= cy < h:
                traj_mask[cy, cx] = 1
        if traj_mask.sum() > 0:
            dist_traj_m = cv2.distanceTransform(1 - traj_mask, cv2.DIST_L2, 3) * res
        else:
            dist_traj_m = np.full((h, w), 1e6, dtype=np.float32)

        min_clear = self.REACH_MIN_CLEARANCE_M
        max_unk   = self.REACH_MAX_UNKNOWN_DIST_M
        visited_r = self.VISITED_RADIUS_M

        cand_mask = (
            (free_mask > 0) &
            (dist_obs_m >= min_clear) &
            (dist_traj_m > visited_r) &
            (dist_unk_m <= max_unk)
        )
        if not cand_mask.any():
            return []

        stride = max(1, int(round(self.REACH_STRIDE_M / res)))
        ys, xs = np.where(cand_mask)
        keys = (ys // stride) * (w // max(1, stride) + 1) + (xs // stride)
        _, uniq = np.unique(keys, return_index=True)
        xs = xs[uniq]; ys = ys[uniq]

        cands: list[tuple[float, float, float, float, float, float]] = []
        for cx, cy in zip(xs.tolist(), ys.tolist()):
            wx = ox + (cx + 0.5) * res
            wy = oy + (cy + 0.5) * res
            cands.append((
                wx, wy,
                float(dist_obs_m[cy, cx]),
                float(dist_unk_m[cy, cx]),
                float(dist_traj_m[cy, cx]),
                float(unknown_density[cy, cx]),
            ))
        return cands

    def _select_frontier(self) -> tuple[float, float, float] | None:
        """Pick the lowest-cost exploration waypoint. Strong preference for
        waypoints with HIGH local unknown density (lots of new map to reveal
        if we go there) and high clearance (so RPP can sprint through).

        Cost (lower is better):
          0.5 * d_rob           — still discourage far drives so we don't
                                  constantly reach for the edge of the arena.
          -2.0 * unknown_density_norm
                                 — DOMINANT term: where is the most unknown?
          -0.4 * d_traj         — novelty bonus.
          -0.6 * clearance      — prefer wide-open cells.
        """
        cands = self._detect_frontiers()
        if not cands:
            return None
        rx, ry = self._map_xy()
        # Normalize unknown_density by the max we saw so it's on the same
        # scale as the other metrics (0..1-ish).
        max_density = max((c[5] for c in cands), default=1.0)
        if max_density <= 0.0:
            max_density = 1.0

        best = None
        best_cost = math.inf
        min_d = self.REACH_MIN_GOAL_DIST_M
        for wx, wy, clearance, _d_unk, d_traj, density in cands:
            d_rob = math.hypot(wx - rx, wy - ry)
            if d_rob < min_d:
                continue
            density_norm = density / max_density
            cost = (0.5 * d_rob
                    - 2.0 * density_norm * 5.0   # scale normalised density up
                    - 0.4 * d_traj
                    - 0.6 * clearance)
            if cost < best_cost:
                best_cost = cost
                best = (wx, wy, clearance)
        return best

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
        # Reset both watchdog histories — this is a fresh goal. Without
        # this the stall watchdog instantly fires on the new goal using
        # odometry from BEFORE dispatch (while the robot was sitting
        # between goals) and cancels before RPP even runs.
        self._goal_dist_history.clear()
        self._odom_history.clear()
        if self._paused:
            self.get_logger().info(
                f'[GOAL] paused — storing ({x:.2f},{y:.2f}) for resume, not dispatching')
            return
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
        self._odom_history.clear()

    def _mpc_is_complete(self) -> bool:
        # Returns True exactly once per goal completion. When no goal is
        # active (idle), returns False so tick loops don't repeatedly
        # process stale "completion" state.
        if not self._nav_goal_active:
            return False
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
        # Recovery backup just finished — re-send the saved waypoint.
        if self._recovery_active:
            self._recovery_active = False
            self.get_logger().info(f'[RECOVERY] backup done ({result}); re-sending goal')
            if self._current_goal is not None and not self._paused:
                x, y, yaw = self._current_goal
                self._send_goal(x, y, yaw)
            return False
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
        # Verbose: log the exact reason each tick so you can see why
        # recovery did or did not fire.
        reason = None
        if self.state in (STOP_AT_TAG, HALTED, LOOKING_FOR_TAG2):
            reason = f'state={self.state}'
        elif self._paused:
            reason = 'paused'
        elif self._recovery_active:
            reason = 'recovery in progress'
        elif not self._nav_goal_active:
            reason = 'no active nav goal'
        elif not self._have_odom:
            reason = 'no odom yet'
        if reason is not None:
            self.get_logger().info(
                f'[WATCHDOG] skip ({reason})', throttle_duration_sec=5.0)
            return

        now = time.time()
        cutoff = now - self.idle_watchdog_s
        while self._odom_history and self._odom_history[0][0] < cutoff:
            self._odom_history.popleft()
        if len(self._odom_history) < 2:
            self.get_logger().info(
                '[WATCHDOG] skip (window too short — just dispatched)',
                throttle_duration_sec=5.0)
            return
        oldest = self._odom_history[0]
        dx = self.robot_x - oldest[1]
        dy = self.robot_y - oldest[2]
        moved = math.hypot(dx, dy)
        # Yaw-change over the window. Short-circuits the stall check
        # while the controller is intentionally rotating in place
        # (RPP's rotate-to-heading phase before forward motion).
        oldest_yaw = oldest[3] if len(oldest) > 3 else self.robot_yaw
        dyaw = abs(math.atan2(math.sin(self.robot_yaw - oldest_yaw),
                              math.cos(self.robot_yaw - oldest_yaw)))
        window = now - oldest[0]
        self.get_logger().info(
            f'[WATCHDOG] window={window:.1f}s moved={moved*100:.1f}cm '
            f'|dyaw|={math.degrees(dyaw):.1f}° samples={len(self._odom_history)}',
            throttle_duration_sec=2.0)
        # If robot has rotated ≥ ~17° in the window, count as progress:
        # it's aligning to heading, not stuck.
        if dyaw > 0.30 and moved < 0.05:
            return
        if moved < 0.05 and dyaw < 0.10:
            self.get_logger().warn(
                f'[WATCHDOG] STALLED — moved {moved*100:.1f}cm, '
                f'|dyaw|={math.degrees(dyaw):.1f}° in {window:.1f}s. '
                f'Triggering escape recovery.')
            self._trigger_backup_recovery()
            return

        # Goal-progress watchdog. Only abandon if the robot is driving
        # forward but not closing distance — not while it's aligning.
        if self._current_goal is None or self._recovery_active:
            return
        if dyaw > 0.20:
            # Still turning to face the goal — give it time.
            return
        gdc = now - self.GOAL_PROGRESS_WINDOW_S
        while self._goal_dist_history and self._goal_dist_history[0][0] < gdc:
            self._goal_dist_history.popleft()
        if len(self._goal_dist_history) >= 2:
            d_start = self._goal_dist_history[0][1]
            d_now   = self._goal_dist_history[-1][1]
            progress = d_start - d_now
            if progress < self.GOAL_PROGRESS_MIN_M:
                gx, gy, _ = self._current_goal
                self.get_logger().warn(
                    f'[WATCHDOG] NO-PROGRESS — dist_to_goal '
                    f'{d_start:.2f}m → {d_now:.2f}m (Δ={progress*100:+.1f}cm) '
                    f'over {self.GOAL_PROGRESS_WINDOW_S:.1f}s; abandoning '
                    f'waypoint ({gx:+.2f},{gy:+.2f}) and re-selecting')
                self._abandon_current_goal()

    def _abandon_current_goal(self):
        """Cancel Nav2, mark the current waypoint as visited (so the
        exploration scorer doesn't re-pick it immediately), and let the
        next _tick_exploring fire _enter_exploring()."""
        try:
            self.nav.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'[ABANDON] cancelTask: {e!r}')
        if self._current_goal is not None:
            gx, gy, _ = self._current_goal
            # Mark the goal position + its immediate neighborhood so the
            # trajectory-halo test in _detect_frontiers skips it.
            self._mark_position_visited(gx, gy)
        if self._current_frontier is not None:
            fx, fy = self._current_frontier
            self._mark_position_visited(fx, fy)
        self._nav_goal_active = False
        self._nav_last_reached = False
        self._current_frontier = None
        self._goal_dist_history.clear()
        self._odom_history.clear()
        self.cmd_pub.publish(Twist())

    # ── cmd_vel tap (diagnostic only) ───────────────────────────────────
    def _on_cmd_vel_tap(self, msg: Twist):
        self._last_cmd = (msg.linear.x, msg.angular.z)
        self._last_cmd_stamp = time.time()
        self._cmd_count += 1

    # ── Periodic nav status summary ─────────────────────────────────────
    def _nav_diag(self):
        if not self._nav_goal_active and not self._paused:
            return
        gx, gy = (self._current_goal[0], self._current_goal[1]) \
            if self._current_goal else (float('nan'), float('nan'))
        dist = math.hypot(gx - self.robot_x, gy - self.robot_y) \
            if self._current_goal else float('nan')
        now = time.time()
        cmd_age = now - self._last_cmd_stamp if self._last_cmd_stamp else float('inf')
        # recent displacement (last 4s)
        cutoff = now - self.idle_watchdog_s
        recent_samples = [s for s in self._odom_history if s[0] >= cutoff]
        if len(recent_samples) >= 2:
            dx = self.robot_x - recent_samples[0][1]
            dy = self.robot_y - recent_samples[0][2]
            recent_moved = math.hypot(dx, dy)
            window = now - recent_samples[0][0]
        else:
            recent_moved = 0.0
            window = 0.0
        flags = []
        if self._paused: flags.append('PAUSED')
        if self._recovery_active: flags.append('RECOVERING')
        if self._nav_goal_active: flags.append('ACTIVE')
        tag = ','.join(flags) if flags else 'idle'
        self.get_logger().info(
            f'[NAV] pose=({self.robot_x:+.2f},{self.robot_y:+.2f}@'
            f'{math.degrees(self.robot_yaw):+.0f}°) '
            f'goal=({gx:+.2f},{gy:+.2f}) d={dist:.2f}m '
            f'cmd=({self._last_cmd[0]:+.2f},{self._last_cmd[1]:+.2f}) age={cmd_age:.1f}s '
            f'moved_{window:.0f}s={recent_moved*100:.1f}cm '
            f'sweep={self.sweep_idx}/{len(self.sweep_waypoints)} '
            f'visited={len(self._visited_positions)} [{tag}]')

    # ── Anti-backtrack: mark passed-through waypoints as visited ────────
    @staticmethod
    def _pos_key(x: float, y: float) -> tuple[int, int]:
        # 5cm quantization — stable across sweep_waypoints list swaps
        return (int(round(x * 20.0)), int(round(y * 20.0)))

    def _mark_position_visited(self, x: float, y: float) -> bool:
        k = self._pos_key(x, y)
        if k in self._visited_positions:
            return False
        self._visited_positions.add(k)
        return True

    def _passthrough_visit_check(self):
        if not self._have_odom or self._map_pose is None:
            return
        rx, ry = self._map_xy()  # sweep_waypoints are in map frame
        r = self.WAYPOINT_VISIT_RADIUS_M
        newly = []
        for wp in self.sweep_waypoints:
            if math.hypot(wp[0] - rx, wp[1] - ry) >= r:
                continue
            if self._mark_position_visited(wp[0], wp[1]):
                newly.append((round(wp[0], 2), round(wp[1], 2)))
        if newly:
            self.get_logger().info(
                f'[SWEEP] passed-through waypoints {newly} '
                f'(visited={len(self._visited_positions)}/{len(self.sweep_waypoints)})')

    def _trigger_backup_recovery(self):
        """Unstall: cancel the current goal and pick a nearby reachable cell
        to escape to. Replaces nav.backup() — BasicNavigator.backup() was
        raising AssertionError('sec field must be int') and never ran the
        actual recovery behavior."""
        if self._recovery_active:
            return
        try:
            self.nav.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'[RECOVERY] cancelTask: {e!r}')

        escape = self._pick_escape_waypoint()
        if escape is None:
            self.get_logger().warn(
                '[RECOVERY] no nearby reachable waypoint found; clearing state')
            self._nav_goal_active = False
            self._odom_history.clear()
            self.cmd_pub.publish(Twist())
            return

        ex, ey = escape
        rx, ry = self._map_xy()
        yaw = math.atan2(ey - ry, ex - rx)
        self.get_logger().warn(
            f'[RECOVERY] stalled — escaping to nearby waypoint '
            f'({ex:+.2f},{ey:+.2f}) from ({rx:+.2f},{ry:+.2f})')
        self._recovery_active = True
        self._recovery_started_at = time.time()
        self._odom_history.clear()
        self._send_goal(ex, ey, yaw)

    def _pick_escape_waypoint(self,
                              radius_min_m: float = 0.25,
                              radius_max_m: float = 1.20,
                              clearance_m: float = None) -> tuple[float, float] | None:
        """Pick any free + reachable cell in an annulus around the robot.
        Prefers cells behind / to the side of the current heading (so we
        don't drive right back into whatever got us stuck). Returns
        (wx, wy) in map frame, or None if no escape cell exists."""
        m = self._map
        if m is None:
            return None
        if clearance_m is None:
            clearance_m = self.REACH_MIN_CLEARANCE_M
        w = m.info.width; h = m.info.height
        if w == 0 or h == 0:
            return None
        res = float(m.info.resolution)
        ox = float(m.info.origin.position.x)
        oy = float(m.info.origin.position.y)
        grid = np.asarray(m.data, dtype=np.int16).reshape(h, w)
        free = ((grid >= 0) & (grid <= self.FRONTIER_FREE_MAX)).astype(np.uint8)
        occ  = (grid >= self.FRONTIER_OCC_MIN).astype(np.uint8)
        if free.sum() == 0:
            return None
        dist_obs_m = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3) * res

        rx, ry = self._map_xy()
        ryaw = self._map_yaw()

        # Cell-space annulus around robot.
        r_min_px = max(1, int(radius_min_m / res))
        r_max_px = max(r_min_px + 1, int(radius_max_m / res))
        rcx = int((rx - ox) / res)
        rcy = int((ry - oy) / res)

        best = None
        best_cost = math.inf
        r_max_sq = r_max_px * r_max_px
        r_min_sq = r_min_px * r_min_px
        y_lo = max(0, rcy - r_max_px); y_hi = min(h, rcy + r_max_px + 1)
        x_lo = max(0, rcx - r_max_px); x_hi = min(w, rcx + r_max_px + 1)
        for cy in range(y_lo, y_hi):
            for cx in range(x_lo, x_hi):
                dcx = cx - rcx; dcy = cy - rcy
                d_sq = dcx * dcx + dcy * dcy
                if d_sq < r_min_sq or d_sq > r_max_sq:
                    continue
                if free[cy, cx] == 0:
                    continue
                if dist_obs_m[cy, cx] < clearance_m:
                    continue
                wx = ox + (cx + 0.5) * res
                wy = oy + (cy + 0.5) * res
                # Prefer cells not in front of current heading (penalize
                # dot-product with heading direction).
                heading_dot = (dcx * math.cos(ryaw) + dcy * math.sin(ryaw))
                cost = heading_dot * 0.5 - dist_obs_m[cy, cx]
                if cost < best_cost:
                    best_cost = cost
                    best = (wx, wy)
        return best

    # ── Pause / resume (external override hook) ─────────────────────────
    def _on_pause_msg(self, msg: Bool):
        if msg.data:
            self._pause_nav('pause topic')
        else:
            self._resume_nav('pause topic')

    def _pause_nav(self, reason: str = ''):
        if self._paused:
            return
        self._paused = True
        try:
            self.nav.cancelTask()
        except Exception as e:
            self.get_logger().warn(f'[PAUSE] cancelTask: {e!r}')
        self._nav_goal_active = False
        self._recovery_active = False
        self.cmd_pub.publish(Twist())
        self.get_logger().info(f'[PAUSE] nav paused ({reason})')

    def _resume_nav(self, reason: str = ''):
        if not self._paused:
            return
        self._paused = False
        self.get_logger().info(f'[PAUSE] nav resumed ({reason})')
        # Re-dispatch the saved goal so the mission continues.
        if self._current_goal is not None:
            x, y, yaw = self._current_goal
            self._send_goal(x, y, yaw)

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
        # Primary: pick the best-scoring frontier from the SLAM map.
        # Fallback: if no map yet (startup) or no frontier candidates (arena
        # fully explored), use the pre-computed boustrophedon/grid sweep.
        # All goal math is in map frame (via TF map→base_link), since
        # PoseStamped goals, sweep waypoints, and frontier centroids are all
        # map-frame. Odom frame drifts from map once SLAM loop-closes.
        if not self._have_odom:
            return
        rx, ry = self._map_xy()
        frontier = self._select_frontier()
        if frontier is not None:
            x, y, score = frontier
            yaw = math.atan2(y - ry, x - rx)
            self._current_frontier = (x, y)
            # Blacklist this centroid so we don't re-pick it while approaching.
            self._mark_position_visited(x, y)
            self.get_logger().info(
                f'[FRONTIER] goal=({x:+.2f},{y:+.2f}) score={score:.2f} '
                f'from_map_pose=({rx:+.2f},{ry:+.2f})')
            self._send_goal(x, y, yaw)
            return

        # Fallback to sweep
        self._current_frontier = None
        have_map = self._map is not None
        self.get_logger().info(
            f'[ENTER_EXPLORING] no frontier (have_map={have_map}); '
            f'falling back to sweep idx={self.sweep_idx}/{len(self.sweep_waypoints)}')
        if self.sweep_idx >= len(self.sweep_waypoints):
            self.get_logger().info('[ENTER_EXPLORING] sweep exhausted too.')
            return
        try:
            visited = set(range(self.sweep_idx))
            for i, wp in enumerate(self.sweep_waypoints):
                if self._pos_key(wp[0], wp[1]) in self._visited_positions:
                    visited.add(i)
            idx = nearest_remaining(
                (rx, ry), self.sweep_waypoints, visited)
            if idx is None:
                wp = self.sweep_waypoints[self.sweep_idx]
            else:
                if idx != self.sweep_idx:
                    self.sweep_waypoints[self.sweep_idx], self.sweep_waypoints[idx] = \
                        self.sweep_waypoints[idx], self.sweep_waypoints[self.sweep_idx]
                wp = self.sweep_waypoints[self.sweep_idx]
        except Exception as e:
            self.get_logger().warn(f'[ENTER_EXPLORING] sweep select failed: {e}')
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
            yaw = math.atan2(y - ry, x - rx)
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
        elif self.state == TAG1_GOAL_A:
            self._tick_tag1_goal_a()
        elif self.state == TAG1_GOAL_B:
            self._tick_tag1_goal_b()
        elif self.state == LOOKING_FOR_TAG2:
            self._tick_looking_for_tag2()
        elif self.state == TAG2_GOAL:
            self._tick_tag2_goal()

    def _tick_exploring(self):
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
                f'({self.robot_x:.2f},{self.robot_y:.2f}).')
            self._initial_goal_sent = True
            self._enter_exploring()
            return
        if self._mpc_is_complete():
            if self._nav_last_reached:
                tgt = self._current_frontier or self._current_goal
                self.get_logger().info(f'[TICK] exploration goal reached: {tgt}')
                if self.grid_mode and self._cell_reached_at is None:
                    self._cell_reached_at = time.time()
                    self.cmd_pub.publish(Twist())
                    return
                if self.grid_mode and (time.time() - self._cell_reached_at) < self.grid_cell_dwell_s:
                    return
            else:
                self.get_logger().warn('[TICK] goal ended without reach, selecting next.')
            self._cell_reached_at = None
            # Ensure current goal position is blacklisted before re-selecting.
            if self._current_frontier is not None:
                fx, fy = self._current_frontier
                self._mark_position_visited(fx, fy)
            elif self._current_goal is not None:
                gx, gy = self._current_goal[0], self._current_goal[1]
                self._mark_position_visited(gx, gy)
                # sweep fallback progress bookkeeping
                if self.sweep_idx < len(self.sweep_waypoints):
                    self.sweep_idx += 1
            self._nav_last_reached = False
            self._enter_exploring()
            return

        # No active goal AND no completion transition to process → idle.
        # Periodically retry frontier selection so that a freshly-grown map
        # can unblock exploration without waiting for an external trigger.
        if not self._nav_goal_active and not self._paused:
            now = time.time()
            if now >= self._next_frontier_retry_at:
                self._next_frontier_retry_at = now + self.FRONTIER_IDLE_RETRY_S
                self.get_logger().info(
                    '[TICK] idle retry of _enter_exploring',
                    throttle_duration_sec=5.0)
                self._enter_exploring()
            return

        # Goal is active: re-score candidate waypoints periodically. As the
        # robot advances, new map is revealed and the best reachable-near-
        # unknown waypoint shifts. Preempt the current goal if the new best
        # differs enough to be worth re-planning for.
        now = time.time()
        if (now - self._last_goal_update_at) >= self.GOAL_UPDATE_INTERVAL_S:
            self._last_goal_update_at = now
            if self._current_goal is not None and not self._paused:
                cand = self._select_frontier()
                if cand is not None:
                    nx, ny, _ = cand
                    gx, gy, _gyaw = self._current_goal
                    if math.hypot(nx - gx, ny - gy) >= self.GOAL_UPDATE_DIST_M:
                        rx, ry = self._map_xy()
                        yaw = math.atan2(ny - ry, nx - rx)
                        self.get_logger().info(
                            f'[FRONTIER↻] updating goal ({gx:+.2f},{gy:+.2f}) '
                            f'→ ({nx:+.2f},{ny:+.2f}) as bot advanced')
                        self._current_frontier = (nx, ny)
                        self._mark_position_visited(nx, ny)
                        try:
                            self.nav.cancelTask()
                        except Exception:
                            pass
                        self._send_goal(nx, ny, yaw)

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

    # ── Tag 1 / 2 gated action ticks ─────────────────────────────────────
    def _dispatch_pending_action_goal(self):
        if self._action_next_goal is None:
            return
        x, y, yaw = self._action_next_goal
        self._action_next_goal = None
        self._send_goal(x, y, yaw)

    def _tick_tag1_goal_a(self):
        # Dispatch as soon as we're in this state (action_next_goal holds
        # (tag1_x-0.5, tag1_y, yaw=90°) from _enter_tag1_flow).
        if not self._nav_goal_active and self._action_next_goal is not None:
            self._dispatch_pending_action_goal()
            return
        if self._mpc_is_complete():
            if not self._nav_last_reached:
                self.get_logger().warn(
                    '[TAG1A] goal ended without reach — retrying same goal.')
                tx, ty = self._tag1_pos
                self._action_next_goal = (tx - 0.5, ty, math.radians(90.0))
                self._nav_last_reached = False
                return
            self.get_logger().warn('[TAG1A] reached A — sending B (forward 1.5 m).')
            self._nav_last_reached = False
            tx, ty = self._tag1_pos
            self._action_next_goal = (tx - 0.5, ty + 1.5, math.radians(90.0))
            self._transition(TAG1_GOAL_B)

    def _tick_tag1_goal_b(self):
        if not self._nav_goal_active and self._action_next_goal is not None:
            self._dispatch_pending_action_goal()
            return
        if self._mpc_is_complete():
            if not self._nav_last_reached:
                self.get_logger().warn(
                    '[TAG1B] goal ended without reach — retrying.')
                tx, ty = self._tag1_pos
                self._action_next_goal = (tx - 0.5, ty + 1.5, math.radians(90.0))
                self._nav_last_reached = False
                return
            self.get_logger().warn(
                '[TAG1B] reached — entering LOOKING_FOR_TAG2.')
            self._nav_last_reached = False
            self._transition(LOOKING_FOR_TAG2)

    def _tick_looking_for_tag2(self):
        """Rotate in place until raw tag id 0 (= logical 2) is centred in
        the image. Stop the rotation as soon as the pixel x-centre is
        within TAG_CENTERED_PIXEL_TOL of the image centre."""
        # Raw id 0 corresponds to logical tag 2 (see RAW_TO_LOGICAL_ID).
        det = self._last_raw_detection.get(0)
        img_center_px = 320  # camera x-centre for 640-wide image
        if det is not None:
            px_u, _px_v, _ = det
            # Stale guard: detection is only fresh if it was reported in
            # the past second. If the detection dict is older, treat as
            # if tag 2 not visible.
            # (We don't currently stamp detections; rely on the raw
            # subscription clearing only when new detections come in.)
            if abs(px_u - img_center_px) < self.TAG_CENTERED_PIXEL_TOL:
                self.cmd_pub.publish(Twist())
                self.get_logger().warn(
                    f'[LOOK2] tag 2 centred at u={px_u} (|Δ|='
                    f'{abs(px_u - img_center_px)} px) — firing action.')
                # Mark executed so the tag-command callback doesn't
                # re-trigger it, then enter TAG2_GOAL.
                self._executed_tag_ids.add(2)
                pos = self._tag_positions_map.get(2)
                if pos is None:
                    self.get_logger().error(
                        '[LOOK2] centred but no tag 2 position known!')
                    self._transition(EXPLORING)
                    return
                self._tag2_pos = (pos[0], pos[1])
                self._action_next_goal = (pos[0], pos[1] - 1.5, 0.0)
                self._transition(TAG2_GOAL)
                return
            # Seen but off-centre: rotate toward it. Positive dx means
            # tag is right of centre → rotate clockwise (negative omega).
            dx = px_u - img_center_px
            omega = -self.LOOK_ROTATE_OMEGA * (1.0 if dx > 0 else -1.0)
            t = Twist()
            t.angular.z = omega
            self.cmd_pub.publish(t)
            return

        # Tag 2 not in sight yet: rotate slowly CCW while searching. If
        # we have a buffered tag 2 position, bias direction toward it.
        t = Twist()
        t.angular.z = self.LOOK_ROTATE_OMEGA
        pos = self._tag_positions_map.get(2)
        if pos is not None:
            rx, ry = self._map_xy()
            target_yaw = math.atan2(pos[1] - ry, pos[0] - rx)
            yaw_err = math.atan2(
                math.sin(target_yaw - self.robot_yaw),
                math.cos(target_yaw - self.robot_yaw))
            t.angular.z = self.LOOK_ROTATE_OMEGA * (1.0 if yaw_err >= 0 else -1.0)
        self.cmd_pub.publish(t)

    def _tick_tag2_goal(self):
        if not self._nav_goal_active and self._action_next_goal is not None:
            self._dispatch_pending_action_goal()
            return
        if self._mpc_is_complete():
            if not self._nav_last_reached:
                self.get_logger().warn(
                    '[TAG2] goal ended without reach — retrying.')
                if self._tag2_pos is not None:
                    tx, ty = self._tag2_pos
                    self._action_next_goal = (tx, ty - 1.5, 0.0)
                self._nav_last_reached = False
                return
            self.get_logger().warn(
                '[TAG2] reached — tag-1/2 demo sequence done, resuming EXPLORING.')
            self._nav_last_reached = False
            self._transition(EXPLORING)


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
