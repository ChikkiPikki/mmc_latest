#!/usr/bin/env python3
"""Mission Manager Node — Round 3 hackathon FSM orchestrator.

Drives Nav2 via nav2_simple_commander. Overrides planner goals based on
AprilTag commands and detected tile positions.
"""

import csv
import datetime
import json
import math
import os
import time
from collections import deque

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Empty, Int32MultiArray, Bool
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
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
TAG4_UTURN  = 'TAG4_UTURN'         # rotate 180° in place, stop following
TAG5_APPROACH = 'TAG5_APPROACH'    # drive to within 0.5 m of tag 5 then start FOLLOW_ORANGE


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
        # Diagnostic overlay: POI spheres for every action waypoint we
        # compute (tag 1 A/B, tag 2), buffered tag sightings, and the
        # world/odom axes for orientation. Transient-local so late RViz
        # subscribers pick up the full overlay immediately.
        poi_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.poi_pub = self.create_publisher(
            MarkerArray, '/mission/poi', poi_qos)
        # logical_id -> list of ('A'|'B'|'GOAL', x, y, yaw, color_rgb)
        self._action_poi: dict[int, list[tuple]] = {}
        # (x, y, z) of any buffered tag sightings we've stored in the
        # logo_detector but not acted on yet. Keyed by logical_id.
        self._buffered_tag_poi: dict[int, tuple[float, float, float]] = {}
        # Transient-local QoS means a fresh RViz instantly gets the full
        # overlay; we only need to republish on state change, not on a
        # timer. (Old 1 Hz timer was contributing to the RTF drop.)

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
        self.TAG_CENTERED_PIXEL_TOL = 60
        self.LOOK_ROTATE_OMEGA = 0.25   # slower so we don't overshoot
                                        # between apriltag frames (~1 Hz)
        self.LOOK_DETECTION_MAX_AGE_S = 0.7  # reject stale apriltag centroids
        self.LOOK_TIMEOUT_S = 25.0      # force-transition after this long
        self._look_started_at: float = 0.0
        self._tag1_pos: tuple[float, float] | None = None
        self._tag2_pos: tuple[float, float] | None = None
        self._tag5_pos: tuple[float, float] | None = None

        # ── CSV mission log (workspace root /home/.../ros2_ws/mission.csv) ──
        # Columns: sim_sec, wall_iso, event_type, logical_id, raw_id,
        #          tag_x, tag_y, tag_z, tag_yaw_deg,
        #          bot_x, bot_y, bot_z, bot_yaw_deg
        self._csv_path = self._resolve_workspace_csv_path('mission.csv')
        self._csv_file = None
        self._csv_writer = None
        self._init_csv_log()
        self._logged_tile_keys: set[tuple[int, int]] = set()

        # ── Photo capture for each tag event ────────────────────────────
        try:
            from cv_bridge import CvBridge as _CvBridge
            self._cv_bridge = _CvBridge()
        except Exception as e:
            self._cv_bridge = None
            self.get_logger().warn(f'[CSV] cv_bridge unavailable: {e}')
        self._latest_frame_rgb = None   # np.ndarray HxWx3 uint8
        self._photo_dir = self._resolve_workspace_csv_path('mission_photos')
        try:
            os.makedirs(self._photo_dir, exist_ok=True)
        except Exception as e:
            self.get_logger().warn(f'[CSV] mkdir {self._photo_dir}: {e}')
        from sensor_msgs.msg import Image as _Image
        best_effort_img_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            _Image, '/r1_mini/camera/image_raw',
            self._on_camera_image, best_effort_img_qos)

        # ── Tile-pairs from logo_detector ───────────────────────────────
        # Each entry: {'green': [gx,gy], 'orange': [ox,oy], 'yaw_snapped': φ}
        self._tile_pairs: list[dict] = []
        self.create_subscription(
            String, '/logo/tile_pairs',
            self._on_tile_pairs, 5)
        # Consumed tile-pair keys (quantised to 0.3 m) so follow states
        # don't re-target the same tile.
        self._consumed_tile_keys: set[tuple[int, int]] = set()
        self._follow_active_color: str | None = None   # 'green' / 'orange' / None
        self._current_follow_tile_key: tuple[int, int] | None = None

        self._last_map_odom = None
        self._warp_detected = False

        self._odom_history = deque(maxlen=200)
        # Goal-progress watchdog: rolling window of (timestamp, dist_to_goal).
        # If dist_to_goal isn't shrinking by GOAL_PROGRESS_MIN_M over
        # GOAL_PROGRESS_WINDOW_S, we abandon the waypoint (it's either
        # unreachable given the current map, or Nav2 is looping).
        self._goal_dist_history: deque[tuple[float, float]] = deque(maxlen=200)
        self.GOAL_PROGRESS_WINDOW_S = 8.0
        self.GOAL_PROGRESS_MIN_M = 0.05
        self._spin_triggers = deque()

        # ── Visited-waypoint tracking (anti-backtrack) ──────────────────
        # Positions (quantized to 5cm) of waypoints we've either completed or
        # physically driven within WAYPOINT_VISIT_RADIUS_M of en route to
        # something else. Keyed by position rather than list index because
        # _enter_exploring() swaps sweep_waypoints entries during selection,
        # which would invalidate index-based tracking. Any waypoint whose
        # position matches is excluded from future goal selection.
        self._visited_positions: set[tuple[int, int]] = set()
        # Every goal we've ever dispatched to Nav2 (quantised to 20 cm
        # so near-duplicates count as the same point). Frontier picker
        # skips anything in here so we never re-target a spot we've
        # already committed to, even if the bot failed to actually reach
        # it. Reset only explicitly.
        self._tried_waypoints: set[tuple[int, int]] = set()
        self.WAYPOINT_VISIT_RADIUS_M = 0.35  # ~half a 0.9m tile

        # ── Frontier exploration ───────────────────────────────────────
        # Replaces boustrophedon sweep as primary goal source. We detect
        # free-cells-adjacent-to-unknown in the SLAM map, cluster via BFS,
        # and score clusters by size / (distance + eps). The sweep list
        # remains as a fallback for pre-map startup and post-exhaustion.
        self.FRONTIER_FREE_MAX = 20            # occupancy ≤ → "free"
        self.FRONTIER_OCC_MIN = 65             # occupancy ≥ → "occupied"
        self.FRONTIER_MIN_CLUSTER = 4          # cells, filters noise
        self.FRONTIER_BLACKLIST_RADIUS_M = 0.30 # centroid skip radius
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
        # Waypoints must sit inside open space but not too deep — the
        # tighter this, the fewer candidates. Footprint half-diag ~0.23 m,
        # Nav2 inflation 0.06 m, plus small safety margin.
        self.REACH_MIN_CLEARANCE_M = 0.32      # free-cell must be ≥ this from nearest wall
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
        # Filled by _detect_frontiers each call. Printed on idle so the
        # user can see WHY no frontier was returned.
        self._last_frontier_stats: dict = {}

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
        """Fire tag N action as soon as we have its position AND the
        predecessor has run AND we're in a state that can accept an
        action transition."""
        if logical_id in self._executed_tag_ids:
            return
        if self.state == HALTED:
            return
        pos = self._tag_positions_map.get(logical_id)
        if pos is None:
            return
        busy_states = (TAG1_GOAL_A, TAG1_GOAL_B, LOOKING_FOR_TAG2,
                       TAG2_GOAL, TAG4_UTURN, TAG5_APPROACH,
                       FOLLOW_GREEN, FOLLOW_ORANGE)
        if logical_id == 1:
            if self.state in busy_states:
                return
            self._executed_tag_ids.add(1)
            self._enter_tag1_flow(pos)
        elif logical_id == 2:
            if self.state in busy_states:
                return
            self._executed_tag_ids.add(2)
            self._enter_tag2_flow(pos)
        elif logical_id == 3:
            if self.state in busy_states:
                return
            # Tag 3 = START FOLLOWING GREEN (only after tag 2 completed).
            if 2 not in self._executed_tag_ids:
                self.get_logger().info(
                    '[TAG3] seen before tag 2 — deferring.')
                return
            self._executed_tag_ids.add(3)
            self._log_tag_event(3, pos)
            self._enter_follow_flow('green', pos)
        elif logical_id == 4:
            # Tag 4 = U-TURN + STOP FOLLOWING.
            if 3 not in self._executed_tag_ids:
                self.get_logger().info(
                    '[TAG4] seen before tag 3 — deferring.')
                return
            self._executed_tag_ids.add(4)
            self._log_tag_event(4, pos)
            self._enter_tag4_uturn(pos)
        elif logical_id == 5:
            # Tag 5 = APPROACH to 0.5 m then FOLLOW ORANGE.
            if 4 not in self._executed_tag_ids:
                self.get_logger().info(
                    '[TAG5] seen before tag 4 — deferring.')
                return
            self._executed_tag_ids.add(5)
            self._log_tag_event(5, pos)
            self._enter_tag5_approach(pos)

    def _on_apriltag_detections_raw(self, msg):
        """Capture raw apriltag pixel centres for the LOOKING_FOR_TAG2
        rotation-to-center test. Timestamp is attached so the look
        logic can ignore stale detections (apriltag_node updates at
        <1 Hz under sim load — a stale centroid could falsely trigger
        the 'tag centered' branch)."""
        now_t = time.time()
        for det in msg.detections:
            try:
                raw_id = int(det.id)
            except (TypeError, ValueError):
                try:
                    raw_id = int(det.id[0])
                except Exception:
                    continue
            self._last_raw_detection[raw_id] = (
                int(det.centre.x), int(det.centre.y), now_t)

    # ── Workspace-relative path resolution ──────────────────────────────
    def _resolve_workspace_csv_path(self, name: str) -> str:
        """Find the ros2_ws root by walking up from this node's module dir.
        Falls back to $HOME if the walk doesn't find 'src' next to it."""
        # __file__ is .../ros2_ws/src/mini_r1_v1_round3/mini_r1_v1_round3/nodes/...
        here = os.path.dirname(os.path.abspath(__file__))
        cur = here
        for _ in range(8):
            parent = os.path.dirname(cur)
            if os.path.isdir(os.path.join(parent, 'src')):
                return os.path.join(parent, name)
            cur = parent
        return os.path.join(os.path.expanduser('~'), name)

    # ── CSV + photo + tile-pair plumbing ────────────────────────────────
    def _init_csv_log(self):
        try:
            # Append so we don't lose previous runs; write header only
            # if the file didn't already exist.
            write_header = not os.path.exists(self._csv_path)
            self._csv_file = open(self._csv_path, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow([
                    'sim_sec', 'wall_iso', 'event_type',
                    'logical_id', 'raw_id',
                    'tag_x', 'tag_y', 'tag_z', 'tag_yaw_deg',
                    'bot_x', 'bot_y', 'bot_z', 'bot_yaw_deg',
                ])
                self._csv_file.flush()
            self.get_logger().warn(f'[CSV] logging to {self._csv_path}')
        except Exception as e:
            self.get_logger().error(f'[CSV] open failed: {e}')

    def _csv_log(self, event_type: str, logical_id=None, raw_id=None,
                 tag_pos=None, tag_yaw_deg=None):
        if self._csv_writer is None:
            return
        try:
            sim_sec = float(self.get_clock().now().nanoseconds) * 1e-9
            wall_iso = datetime.datetime.now().isoformat(timespec='seconds')
            rx, ry = self._map_xy()
            ryaw = self._map_yaw()
            tx = ty = tz = tyw = ''
            if tag_pos is not None:
                tx, ty = tag_pos[0], tag_pos[1]
                tz = tag_pos[2] if len(tag_pos) > 2 else ''
                if tag_yaw_deg is not None:
                    tyw = f'{tag_yaw_deg:.1f}'
            self._csv_writer.writerow([
                f'{sim_sec:.3f}', wall_iso, event_type,
                logical_id if logical_id is not None else '',
                raw_id if raw_id is not None else '',
                tx if tx == '' else f'{tx:+.3f}',
                ty if ty == '' else f'{ty:+.3f}',
                tz if tz == '' else f'{tz:+.3f}',
                tyw,
                f'{rx:+.3f}', f'{ry:+.3f}', '0.000',
                f'{math.degrees(ryaw):+.1f}',
            ])
            self._csv_file.flush()
        except Exception as e:
            self.get_logger().warn(f'[CSV] write failed: {e}')

    def _on_camera_image(self, msg):
        if self._cv_bridge is None:
            return
        try:
            self._latest_frame_rgb = self._cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding='rgb8')
        except Exception:
            pass

    def _save_photo(self, logical_id: int) -> str | None:
        """Save current camera frame with a corner overlay. Returns path
        or None if no frame available."""
        img = self._latest_frame_rgb
        if img is None:
            self.get_logger().warn(
                f'[PHOTO] no camera frame yet for tag {logical_id}')
            return None
        try:
            sim_sec = float(self.get_clock().now().nanoseconds) * 1e-9
            fname = f'tag{logical_id}_{sim_sec:.2f}.png'
            fpath = os.path.join(self._photo_dir, fname)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            overlay_txt = f'TAG {logical_id} @ {sim_sec:.2f}s'
            cv2.rectangle(bgr, (5, 5), (270, 35), (0, 0, 0), -1)
            cv2.putText(bgr, overlay_txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(fpath, bgr)
            self.get_logger().warn(f'[PHOTO] saved {fpath}')
            return fpath
        except Exception as e:
            self.get_logger().warn(f'[PHOTO] save failed: {e}')
            return None

    def _log_tag_event(self, logical_id: int, tag_pos):
        """Unified tag log step: save photo + append CSV row."""
        self._save_photo(logical_id)
        # tag yaw captured at detection time — we only have position
        # in mission_manager, so skip the yaw_deg column here.
        self._csv_log('tag', logical_id=logical_id,
                      tag_pos=tag_pos)

    def _on_tile_pairs(self, msg):
        """Parse logo_detector's /logo/tile_pairs JSON into self._tile_pairs."""
        try:
            data = json.loads(msg.data)
            if not isinstance(data, list):
                return
            pairs = []
            for p in data:
                g = p.get('green'); o = p.get('orange')
                if g is None or o is None:
                    continue
                pairs.append({
                    'green': (float(g[0]), float(g[1])),
                    'orange': (float(o[0]), float(o[1])),
                    'yaw_snapped': float(p.get('yaw_snapped', 0.0)),
                })
            # Also CSV-log any newly-observed tile centroids (once each).
            for p in pairs:
                self._maybe_log_tile(p)
            self._tile_pairs = pairs
        except Exception as e:
            self.get_logger().warn(f'[TILES] bad tile_pairs JSON: {e}')

    def _maybe_log_tile(self, pair: dict):
        gx, gy = pair['green']
        ox, oy = pair['orange']
        cx, cy = (gx + ox) / 2.0, (gy + oy) / 2.0
        key = (int(round(cx * 5.0)), int(round(cy * 5.0)))   # 20 cm
        if key in self._logged_tile_keys:
            return
        self._logged_tile_keys.add(key)
        self._csv_log('tile', tag_pos=(cx, cy, 0.0))

    def _tile_key(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(x * 5.0)), int(round(y * 5.0)))   # 20 cm

    def _pick_follow_target(self, color: str) -> tuple[float, float, float] | None:
        """Choose the nearest non-consumed tile within 1.7 m and return
        (wx, wy, yaw) for the waypoint — 0.3 m past the ORANGE centroid
        when color=='orange', or 0.3 m past the GREEN centroid when
        color=='green' (i.e. start of arrow + 0.3 m in reverse
        direction)."""
        if not self._tile_pairs:
            return None
        rx, ry = self._map_xy()
        best = None
        best_d = math.inf
        for p in self._tile_pairs:
            gx, gy = p['green']
            ox, oy = p['orange']
            cx, cy = (gx + ox) / 2.0, (gy + oy) / 2.0
            key = self._tile_key(cx, cy)
            if key in self._consumed_tile_keys:
                continue
            d = math.hypot(cx - rx, cy - ry)
            if d > 1.7:
                continue
            if d < best_d:
                best_d = d
                best = (p, key)
        if best is None:
            return None
        p, key = best
        gx, gy = p['green']
        ox, oy = p['orange']
        yaw = p['yaw_snapped']
        if color == 'orange':
            # waypoint = orange + 0.3 m along (orange - green)
            wx = ox + 0.30 * math.cos(yaw)
            wy = oy + 0.30 * math.sin(yaw)
        else:   # green — vice versa: 0.3 m past the green side
            wx = gx - 0.30 * math.cos(yaw)
            wy = gy - 0.30 * math.sin(yaw)
        self._current_follow_tile_key = key
        return (wx, wy, yaw)

    # ── Tag 1 and Tag 2 action entry points ──────────────────────────────
    def _enter_tag1_flow(self, tag_pos: tuple[float, float, float]):
        """Start the two-step Tag-1 action: goal A = (tx-0.5, ty, yaw=90°)."""
        tx, ty, _tz = tag_pos
        self._tag1_pos = (tx, ty)
        self._log_tag_event(1, tag_pos)
        self._pause_exploration_for_action('TAG1')
        goal_a = (tx - 0.5, ty, math.radians(90.0))
        goal_b = (tx - 0.5, ty + 1.5, math.radians(90.0))
        self._action_poi[1] = [
            ('A',    goal_a[0], goal_a[1], goal_a[2], (0.15, 0.60, 1.0)),
            ('B',    goal_b[0], goal_b[1], goal_b[2], (0.15, 0.60, 1.0)),
        ]
        self._publish_poi_markers()
        self._action_next_goal = goal_a
        self._transition(TAG1_GOAL_A)

    def _enter_tag2_flow(self, tag_pos: tuple[float, float, float]):
        tx, ty, _tz = tag_pos
        self._tag2_pos = (tx, ty)
        self._log_tag_event(2, tag_pos)
        self._pause_exploration_for_action('TAG2')
        goal = (tx, ty - 1.5, 0.0)
        self._action_poi[2] = [
            ('GOAL', goal[0], goal[1], goal[2], (1.0, 0.7, 0.1)),
        ]
        self._publish_poi_markers()
        self._action_next_goal = goal
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

    def _detect_frontiers(self) -> list[tuple[float, float]]:
        """Classic frontier picker: return every free cell that is adjacent
        to an unknown cell, has at least REACH_MIN_CLEARANCE_M of wall
        clearance, hasn't already been driven near, and hasn't already
        been dispatched as a goal. Kept deliberately simple — one mask
        op, one distance transform, one roll-shift to find the boundary.
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
        if free_mask.sum() == 0 or unknown_mask.sum() == 0:
            return []

        # Frontier = free cell with any 4-neighbour unknown. Compute the
        # "unknown-adjacent" mask by OR-ing the unknown mask shifted in
        # each cardinal direction.
        u_up    = np.zeros_like(unknown_mask); u_up[:-1, :]  = unknown_mask[1:, :]
        u_down  = np.zeros_like(unknown_mask); u_down[1:, :]  = unknown_mask[:-1, :]
        u_left  = np.zeros_like(unknown_mask); u_left[:, :-1] = unknown_mask[:, 1:]
        u_right = np.zeros_like(unknown_mask); u_right[:, 1:] = unknown_mask[:, :-1]
        adj_unknown = (u_up | u_down | u_left | u_right).astype(np.uint8)

        # Clearance check keeps the goal navigable after Nav2 inflation.
        dist_obs_m = cv2.distanceTransform(1 - occ_mask, cv2.DIST_L2, 3) * res

        frontier_mask = (free_mask > 0) & (adj_unknown > 0) & (dist_obs_m >= self.REACH_MIN_CLEARANCE_M)
        if not frontier_mask.any():
            return []

        ys, xs = np.where(frontier_mask)
        raw_cells = len(ys)
        cands: list[tuple[float, float]] = []
        rx_now, ry_now = self._map_xy()
        min_d2 = self.REACH_MIN_GOAL_DIST_M ** 2
        rej_near = rej_halo = rej_tried = 0
        for cx, cy in zip(xs.tolist(), ys.tolist()):
            wx = ox + (cx + 0.5) * res
            wy = oy + (cy + 0.5) * res
            if (wx - rx_now) ** 2 + (wy - ry_now) ** 2 < min_d2:
                rej_near += 1
                continue
            if self._frontier_blacklisted(wx, wy):
                rej_halo += 1
                continue
            qkey = (int(round(wx * 5.0)), int(round(wy * 5.0)))
            if qkey in self._tried_waypoints:
                rej_tried += 1
                continue
            cands.append((wx, wy))
        # Loud diagnostic so when the bot sits idle we can see exactly
        # which filter is eating the candidates.
        self._last_frontier_stats = {
            'raw_cells': raw_cells,
            'rej_near': rej_near,
            'rej_halo': rej_halo,
            'rej_tried': rej_tried,
            'kept': len(cands),
        }
        return cands

    def _select_frontier(self) -> tuple[float, float, float] | None:
        """Pick a frontier cell using a stable normalized scoring:
          fwd_align  = cos(angle between heading and goal) ∈ [-1, +1]
          cost       = d − 2.5 · fwd_align · max(d, 1.0)

        Multiplying fwd_align by max(d, 1) prevents small-distance noise
        from flipping the winner between ticks (the raw-projection
        version swung cost by only ±1.2 m at d≈0.6, so map updates
        could change the pick). Large-d forward cells still win by a
        big margin, small-d near-behind cells still lose."""
        cands = self._detect_frontiers()
        if not cands:
            return None
        rx, ry = self._map_xy()
        ryaw = self._map_yaw()
        hx = math.cos(ryaw)
        hy = math.sin(ryaw)

        best = None
        best_cost = math.inf
        for wx, wy in cands:
            dx = wx - rx
            dy = wy - ry
            d = math.hypot(dx, dy)
            if d < 1e-3:
                continue
            fwd_align = (dx * hx + dy * hy) / d   # normalized projection
            cost = d - 2.5 * fwd_align * max(d, 1.0)
            if cost < best_cost:
                best_cost = cost
                best = (wx, wy, fwd_align)
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
        # Burn the target into the "tried" set so the frontier picker
        # never re-offers it. Even if the bot fails to reach it we move
        # on to a different boundary cell next time.
        self._tried_waypoints.add((int(round(x * 5.0)), int(round(y * 5.0))))
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
        rx, ry = self._map_xy()
        d = math.hypot(x - rx, y - ry)
        self.get_logger().info(
            f'[FLOW] DISPATCH goal=({x:+.2f},{y:+.2f},{math.degrees(yaw):+.0f}°) '
            f'd={d:.2f}m')
        goal_ps = self._make_pose_stamped(x, y, yaw)
        try:
            self.nav.goToPose(goal_ps)
        except Exception as e:
            self.get_logger().error(f'[FLOW] goToPose raised: {e!r}')
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
        rx, ry = self._map_xy()
        if self._nav_last_reached:
            self.get_logger().info(
                f'[FLOW] ARRIVED at=({rx:+.2f},{ry:+.2f})')
        else:
            self.get_logger().warn(f'[FLOW] FAILED result={result}')
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
        # Silent skip for uninteresting cases — state logs already cover
        # STOP_AT_TAG/HALTED/etc and a noisy [WATCHDOG] skip on every
        # tick was drowning the log.
        if self.state in (STOP_AT_TAG, HALTED, LOOKING_FOR_TAG2):
            return
        if self._paused or self._recovery_active or not self._have_odom:
            return
        if not self._nav_goal_active:
            return
        now = time.time()
        cutoff = now - self.idle_watchdog_s
        while self._odom_history and self._odom_history[0][0] < cutoff:
            self._odom_history.popleft()
        if len(self._odom_history) < 2:
            return
        oldest = self._odom_history[0]
        dx = self.robot_x - oldest[1]
        dy = self.robot_y - oldest[2]
        moved = math.hypot(dx, dy)
        oldest_yaw = oldest[3] if len(oldest) > 3 else self.robot_yaw
        dyaw = abs(math.atan2(math.sin(self.robot_yaw - oldest_yaw),
                              math.cos(self.robot_yaw - oldest_yaw)))
        window = now - oldest[0]
        if dyaw > 0.30 and moved < 0.05:
            return  # actively rotating — not stuck
        if moved < 0.05 and dyaw < 0.10:
            # Don't trigger recovery while we're within the final-approach
            # zone (~2× xy_goal_tolerance). Nav2 is allowed to jiggle the
            # last 30 cm; abandoning here yanks the controller off a
            # goal it's almost reached and sends us to a nonsense escape
            # waypoint behind the bot.
            if self._current_goal is not None:
                gx, gy, _ = self._current_goal
                rx, ry = self._map_xy()
                if math.hypot(gx - rx, gy - ry) < 0.35:
                    return
            self.get_logger().warn(
                f'[FLOW] STALLED — moved {moved*100:.1f}cm, '
                f'|dyaw|={math.degrees(dyaw):.1f}° in {window:.1f}s → escape')
            self._trigger_backup_recovery()
            return

        # Goal-progress watchdog DISABLED. It was firing when the bot was
        # actually making forward progress (7 cm in 5 s while navigating
        # a tight turn), yanking goals out from under a CPU-overloaded
        # controller and leaving the bot spinning at an unreachable
        # escape point. Nav2's own timeout + the full-stall check above
        # is enough to recover from genuinely bad goals.

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

    # ── Periodic flow summary (one concise line per 3 s) ────────────────
    def _nav_diag(self):
        # Compact single-line state view. Only prints when something
        # is actually happening; silent when paused / fully idle.
        if not (self._nav_goal_active or self._paused or self._recovery_active):
            return
        if self._current_goal is None:
            return
        gx, gy, _gyaw = self._current_goal
        rx, ry = self._map_xy()
        dist = math.hypot(gx - rx, gy - ry)
        flags = []
        if self._paused: flags.append('PAUSE')
        if self._recovery_active: flags.append('RECOV')
        if self._nav_goal_active: flags.append('ACTIVE')
        self.get_logger().info(
            f'[FLOW] {self.state} '
            f'bot=({rx:+.2f},{ry:+.2f}@{math.degrees(self._map_yaw()):+.0f}°) '
            f'goal=({gx:+.2f},{gy:+.2f}) d={dist:.2f}m '
            f'cmd=({self._last_cmd[0]:+.2f},{self._last_cmd[1]:+.2f}) '
            f'[{",".join(flags)}]',
            throttle_duration_sec=10.0)

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
                              radius_min_m: float = 0.80,
                              radius_max_m: float = 1.80,
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
                d_m = math.hypot(dcx, dcy) * res
                # Reject cells whose direction would require a >120° turn
                # from current heading. The stuck controller CAN'T
                # execute a u-turn — better to return None and let the
                # caller fall back to just stopping.
                if d_m > 1e-3:
                    heading_cos = (dcx * math.cos(ryaw) + dcy * math.sin(ryaw)) / (d_m / res)
                    if heading_cos < -0.5:   # >120° from forward
                        continue
                cost = d_m - 0.5 * dist_obs_m[cy, cx]
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
        if frontier is None and self._tried_waypoints:
            # Second chance: every frontier cell has been blacklisted at
            # least once. Reset the dispatched-goals memory so the bot can
            # re-attempt goals it previously failed to reach. Trajectory
            # halo blacklist stays — we still won't re-pick cells the bot
            # physically drove through.
            self.get_logger().warn(
                f'[FRONTIER] all {len(self._tried_waypoints)} dispatched '
                f'waypoints blacklisted — clearing tried-set and retrying')
            self._tried_waypoints.clear()
            frontier = self._select_frontier()
        if frontier is not None:
            x, y, score = frontier
            yaw = math.atan2(y - ry, x - rx)
            self._current_frontier = (x, y)
            stats = self._last_frontier_stats
            self.get_logger().info(
                f'[FLOW] PICK frontier=({x:+.2f},{y:+.2f}) '
                f'align={score:+.2f} bot=({rx:+.2f},{ry:+.2f}) '
                f'kept={stats.get("kept", "?")}/{stats.get("raw_cells", "?")} '
                f'rej(near={stats.get("rej_near", 0)},'
                f'halo={stats.get("rej_halo", 0)},'
                f'tried={stats.get("rej_tried", 0)})')
            self._send_goal(x, y, yaw)
            return

        # Fallback to sweep
        self._current_frontier = None
        have_map = self._map is not None
        stats = self._last_frontier_stats
        self.get_logger().warn(
            f'[FLOW] NO FRONTIER have_map={have_map} '
            f'raw_cells={stats.get("raw_cells", 0)} '
            f'rej(near={stats.get("rej_near", 0)},'
            f'halo={stats.get("rej_halo", 0)},'
            f'tried={stats.get("rej_tried", 0)}) '
            f'→ sweep idx={self.sweep_idx}/{len(self.sweep_waypoints)}')
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
            self._tick_follow_tiles('green')
        elif self.state == FOLLOW_ORANGE:
            self._tick_follow_tiles('orange')
        elif self.state == TAG4_UTURN:
            self._tick_tag4_uturn()
        elif self.state == TAG5_APPROACH:
            self._tick_tag5_approach()
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

        # No active goal AND no completion transition → idle. Retry the
        # frontier pick every FRONTIER_IDLE_RETRY_S seconds — the map
        # is still growing, previously-impossible cells may become
        # reachable.
        if not self._nav_goal_active and not self._paused:
            now = time.time()
            if now >= self._next_frontier_retry_at:
                self._next_frontier_retry_at = now + self.FRONTIER_IDLE_RETRY_S
                self._enter_exploring()
            return

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

    # ── POI + world-axes RViz overlay ────────────────────────────────────
    def _publish_poi_markers(self):
        """Render every point-of-interest in the mission onto /mission/poi
        as a MarkerArray. Includes the world/odom axes at the origin,
        all current action waypoints (tag 1 A/B, tag 2 GOAL), known tag
        positions from logo_detector, and any buffered tag sightings."""
        arr = MarkerArray()
        # Clear-all sentinel so old markers from previous action flows
        # don't linger when we enter a new phase.
        clear = Marker()
        clear.header.frame_id = 'odom'
        clear.ns = 'mission_poi'
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        now = self.get_clock().now().to_msg()

        # 1) World / odom axes at the origin — X red, Y green, Z blue.
        for i, (vx, vy, vz, rr, gg, bb, label) in enumerate([
            (1.0, 0.0, 0.0, 1.0, 0.1, 0.1, '+X'),
            (0.0, 1.0, 0.0, 0.1, 1.0, 0.1, '+Y'),
            (0.0, 0.0, 1.0, 0.1, 0.3, 1.0, '+Z'),
        ]):
            a = Marker()
            a.header.frame_id = 'odom'
            a.header.stamp = now
            a.ns = 'mission_poi_axes'
            a.id = i
            a.type = Marker.ARROW
            a.action = Marker.ADD
            a.points = [Pose().position, Pose().position]  # placeholder
            p0 = Point(); p0.x = 0.0; p0.y = 0.0; p0.z = 0.0
            p1 = Point(); p1.x = vx * 0.8; p1.y = vy * 0.8; p1.z = vz * 0.8
            a.points = [p0, p1]
            a.scale.x = 0.035         # shaft diameter
            a.scale.y = 0.07          # head diameter
            a.scale.z = 0.10          # head length
            a.color.r = rr; a.color.g = gg; a.color.b = bb; a.color.a = 1.0
            a.pose.orientation.w = 1.0
            arr.markers.append(a)

            t = Marker()
            t.header.frame_id = 'odom'
            t.header.stamp = now
            t.ns = 'mission_poi_axes_label'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = vx * 0.9
            t.pose.position.y = vy * 0.9
            t.pose.position.z = vz * 0.9 + 0.05
            t.pose.orientation.w = 1.0
            t.scale.z = 0.12
            t.color.r = rr; t.color.g = gg; t.color.b = bb; t.color.a = 1.0
            t.text = label
            arr.markers.append(t)

        # 2) Known logical-tag positions — small white spheres + label.
        for logical_id, (x, y, z) in self._tag_positions_map.items():
            arr.markers.append(self._poi_sphere(
                f'tag_pos', logical_id * 10, x, y, z + 0.05,
                0.10, (0.9, 0.9, 0.9), 0.6))
            arr.markers.append(self._poi_text(
                f'tag_pos_label', logical_id * 10, x, y, z + 0.30,
                f'TAG{logical_id}', (0.9, 0.9, 0.9)))

        # 3) Action waypoints — colored spheres + labels + a thin arrow
        #    pointing in the commanded yaw direction.
        for logical_id, pois in self._action_poi.items():
            for i, (label, x, y, yaw, col) in enumerate(pois):
                mid = logical_id * 100 + i
                arr.markers.append(self._poi_sphere(
                    'action_waypoint', mid, x, y, 0.05, 0.22, col, 0.9))
                arr.markers.append(self._poi_text(
                    'action_waypoint_label', mid, x, y, 0.35,
                    f'T{logical_id}·{label}', col))
                # Direction arrow to show commanded yaw.
                arrow = Marker()
                arrow.header.frame_id = 'odom'
                arrow.header.stamp = now
                arrow.ns = 'action_waypoint_yaw'
                arrow.id = mid
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                p0 = Point(); p0.x = x; p0.y = y; p0.z = 0.05
                p1 = Point()
                p1.x = x + 0.35 * math.cos(yaw)
                p1.y = y + 0.35 * math.sin(yaw)
                p1.z = 0.05
                arrow.points = [p0, p1]
                arrow.scale.x = 0.04; arrow.scale.y = 0.08; arrow.scale.z = 0.12
                arrow.color.r = col[0]; arrow.color.g = col[1]; arrow.color.b = col[2]
                arrow.color.a = 0.9
                arrow.pose.orientation.w = 1.0
                arr.markers.append(arrow)

        # 4) Buffered (pre-predecessor) tag sightings — pulsing magenta
        #    sphere so the user can see WHERE we think the tag is even
        #    before we're allowed to act on it.
        for logical_id, (x, y, z) in self._buffered_tag_poi.items():
            arr.markers.append(self._poi_sphere(
                'buffered_tag', logical_id, x, y, z + 0.05,
                0.14, (1.0, 0.2, 0.9), 0.7))
            arr.markers.append(self._poi_text(
                'buffered_tag_label', logical_id, x, y, z + 0.25,
                f'TAG{logical_id}?', (1.0, 0.2, 0.9)))

        self.poi_pub.publish(arr)

    def _poi_sphere(self, ns, mid, x, y, z, diameter, rgb, alpha):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = diameter; m.scale.y = diameter; m.scale.z = diameter
        m.color.r = float(rgb[0]); m.color.g = float(rgb[1]); m.color.b = float(rgb[2])
        m.color.a = float(alpha)
        return m

    def _poi_text(self, ns, mid, x, y, z, text, rgb):
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(mid)
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.14
        m.color.r = float(rgb[0]); m.color.g = float(rgb[1]); m.color.b = float(rgb[2])
        m.color.a = 1.0
        m.text = text
        return m

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
        the image. Only trusts apriltag centroids seen within the last
        LOOK_DETECTION_MAX_AGE_S seconds so stale dict entries can't
        spoof the centering test."""
        now = time.time()
        if self._look_started_at == 0.0:
            self._look_started_at = now
            self.get_logger().warn(
                f'[LOOK2] ENTER — rotating to centre tag 2 '
                f'(buffered_pos={self._tag_positions_map.get(2)})')

        img_center_px = 320  # 640-wide image
        det = self._last_raw_detection.get(0)
        fresh = (det is not None
                 and (now - det[2]) <= self.LOOK_DETECTION_MAX_AGE_S)

        # Timeout escape: if we've been rotating this long without
        # centring, just commit to the buffered position and fire.
        if now - self._look_started_at > self.LOOK_TIMEOUT_S:
            self.get_logger().warn(
                f'[LOOK2] TIMEOUT after {self.LOOK_TIMEOUT_S:.0f}s — '
                f'firing tag 2 action using buffered position')
            self.cmd_pub.publish(Twist())
            self._enter_tag2_from_buffered()
            return

        if fresh:
            # Per user: "as soon as it is detected, go to the waypoint
            # without fail". No pixel-centering wait — ANY fresh raw-0
            # detection fires the action immediately.
            px_u = det[0]
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(
                f'[LOOK2] DETECTED at u={px_u} age={now-det[2]:.2f}s '
                f'— firing tag 2 action immediately')
            self._enter_tag2_from_buffered()
            return

        # No fresh detection. Rotate in the direction of the buffered
        # tag-2 position (in map frame) so we search the right way.
        t = Twist()
        pos = self._tag_positions_map.get(2)
        if pos is not None:
            rx, ry = self._map_xy()
            target_yaw = math.atan2(pos[1] - ry, pos[0] - rx)
            yaw_err = math.atan2(
                math.sin(target_yaw - self.robot_yaw),
                math.cos(target_yaw - self.robot_yaw))
            t.angular.z = self.LOOK_ROTATE_OMEGA * (1.0 if yaw_err >= 0 else -1.0)
            self.get_logger().info(
                f'[LOOK2] searching: target_yaw={math.degrees(target_yaw):+.0f}° '
                f'err={math.degrees(yaw_err):+.0f}° ω={t.angular.z:+.2f}',
                throttle_duration_sec=1.0)
        else:
            t.angular.z = self.LOOK_ROTATE_OMEGA
            self.get_logger().info(
                '[LOOK2] no buffered position — default CCW sweep',
                throttle_duration_sec=2.0)
        self.cmd_pub.publish(t)

    def _enter_tag2_from_buffered(self):
        """Shared exit path from LOOKING_FOR_TAG2 into TAG2_GOAL.
        Uses the stored _tag_positions_map[2] (captured the moment tag 2
        was first sighted, well before LOOK2 started)."""
        self._executed_tag_ids.add(2)
        pos = self._tag_positions_map.get(2)
        if pos is None:
            self.get_logger().error(
                '[LOOK2] no tag 2 position available at exit — back to EXPLORING')
            self._look_started_at = 0.0
            self._transition(EXPLORING)
            return
        self._tag2_pos = (pos[0], pos[1])
        self._action_next_goal = (pos[0], pos[1] - 1.5, 0.0)
        self.get_logger().warn(
            f'[TAG2] action queued: goal=({pos[0]:+.2f},{pos[1] - 1.5:+.2f}, yaw=0°)')
        self._look_started_at = 0.0
        self._transition(TAG2_GOAL)

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
            self.get_logger().warn('[TAG2] reached — logging + resuming EXPLORING.')
            if self._tag2_pos is not None:
                self._log_tag_event(2, (self._tag2_pos[0], self._tag2_pos[1], 0.0))
            self._nav_last_reached = False
            self._transition(EXPLORING)

    # ── Tag 3/4/5 entry points ──────────────────────────────────────────
    def _enter_follow_flow(self, color: str, tag_pos):
        """Start follow-green (color='green') or follow-orange
        ('orange'). The follow logic picks a tile pair each cycle
        until no more unused tiles remain within 1.7 m, then falls
        back to EXPLORING."""
        self._pause_exploration_for_action(f'FOLLOW_{color.upper()}')
        self._follow_active_color = color
        target_state = FOLLOW_GREEN if color == 'green' else FOLLOW_ORANGE
        self.get_logger().warn(
            f'[FOLLOW] color={color} (tag fired at '
            f'{tag_pos[0]:+.2f},{tag_pos[1]:+.2f})')
        self._transition(target_state)

    def _enter_tag4_uturn(self, tag_pos):
        """Tag 4: stop following + 180° rotate in place."""
        self._follow_active_color = None
        self._current_follow_tile_key = None
        self._pause_exploration_for_action('TAG4_UTURN')
        self._tag4_start_yaw = self._map_yaw()
        self._tag4_target_yaw = math.atan2(
            math.sin(self._tag4_start_yaw + math.pi),
            math.cos(self._tag4_start_yaw + math.pi))
        self.get_logger().warn(
            f'[TAG4] U-turn: {math.degrees(self._tag4_start_yaw):+.0f}° '
            f'→ {math.degrees(self._tag4_target_yaw):+.0f}°')
        self._transition(TAG4_UTURN)

    def _enter_tag5_approach(self, tag_pos):
        """Tag 5: approach to within 0.5 m along bot→tag line, then
        transition to FOLLOW_ORANGE."""
        tx, ty = tag_pos[0], tag_pos[1]
        self._tag5_pos = (tx, ty)
        rx, ry = self._map_xy()
        dx, dy = tx - rx, ty - ry
        d = math.hypot(dx, dy)
        if d < 1e-3:
            gx, gy = rx, ry
        else:
            # Stop 0.5 m before the tag along the bot→tag line.
            ux, uy = dx / d, dy / d
            stop_d = max(0.0, d - 0.5)
            gx = rx + ux * stop_d
            gy = ry + uy * stop_d
        yaw = math.atan2(ty - gy, tx - gx)  # face the tag
        self._pause_exploration_for_action('TAG5_APPROACH')
        self._action_next_goal = (gx, gy, yaw)
        self.get_logger().warn(
            f'[TAG5] approach: ({gx:+.2f},{gy:+.2f},yaw={math.degrees(yaw):+.0f}°)')
        self._transition(TAG5_APPROACH)

    # ── Tag 3/4/5 tick handlers ─────────────────────────────────────────
    def _tick_follow_tiles(self, color: str):
        """Drive from one tile waypoint to the next. Picks the nearest
        unused tile pair within 1.7 m; when it's reached, mark consumed
        and pick another. No more tiles → back to EXPLORING."""
        if not self._nav_goal_active:
            cand = self._pick_follow_target(color)
            if cand is None:
                self.get_logger().info(
                    f'[FOLLOW] no unused {color} tiles within 1.7 m — '
                    f'resuming EXPLORING')
                self._follow_active_color = None
                self._transition(EXPLORING)
                return
            wx, wy, yaw = cand
            self.get_logger().warn(
                f'[FOLLOW] {color} → waypoint=({wx:+.2f},{wy:+.2f},'
                f'yaw={math.degrees(yaw):+.0f}°)')
            self._send_goal(wx, wy, yaw)
            return
        if self._mpc_is_complete():
            if self._nav_last_reached and self._current_follow_tile_key is not None:
                self._consumed_tile_keys.add(self._current_follow_tile_key)
                rx, ry = self._map_xy()
                # CSV log: tile crossed during follow flow.
                self._csv_log(
                    f'tile_crossed_{color}',
                    tag_pos=(rx, ry, 0.0))
                self.get_logger().warn(
                    f'[FOLLOW] consumed tile {self._current_follow_tile_key} '
                    f'(total consumed={len(self._consumed_tile_keys)})')
            self._current_follow_tile_key = None
            self._nav_last_reached = False

    def _tick_tag4_uturn(self):
        """Pure rotation in place to the target yaw. Publishes direct
        cmd_vel (Nav2 can't do zero-translation rotations cleanly here)."""
        yaw = self._map_yaw()
        err = math.atan2(
            math.sin(self._tag4_target_yaw - yaw),
            math.cos(self._tag4_target_yaw - yaw))
        if abs(err) < math.radians(8):
            self.cmd_pub.publish(Twist())
            self.get_logger().warn('[TAG4] U-turn complete — EXPLORING')
            self._transition(EXPLORING)
            return
        t = Twist()
        t.angular.z = 1.6 * (1.0 if err > 0 else -1.0)
        self.cmd_pub.publish(t)

    def _tick_tag5_approach(self):
        """Drive to the pre-computed 0.5-m-from-tag5 waypoint; on
        arrival, transition to FOLLOW_ORANGE."""
        if not self._nav_goal_active and self._action_next_goal is not None:
            self._dispatch_pending_action_goal()
            return
        if self._mpc_is_complete():
            if not self._nav_last_reached:
                self.get_logger().warn(
                    '[TAG5] approach goal ended without reach — retrying.')
                self._nav_last_reached = False
                if self._tag5_pos is not None:
                    self._enter_tag5_approach((self._tag5_pos[0],
                                               self._tag5_pos[1], 0.0))
                return
            self.get_logger().warn(
                '[TAG5] within 0.5 m — starting FOLLOW_ORANGE')
            self._nav_last_reached = False
            self._follow_active_color = 'orange'
            self._transition(FOLLOW_ORANGE)


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
