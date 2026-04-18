"""Logo + AprilTag deprojection node.

Detects three logo colors (#3b12b6, #e77f33, #2b9e27) and AprilTag centers in
the RGB image, deprojects pixel + depth to 3D in the camera frame, transforms
into `map`, and publishes plane markers.

Logo orientation:
  Each tile logo has orange arcs at the top and green arcs at the bottom.
  Orientation (yaw) = atan2(orange_y - green_y, orange_x - green_x) in map frame.
  Published once per unique logo location on /logo/pose (PoseStamped, latched).

Diagnostics:
  /logo/depth_viz  — MONO8 normalized depth (view in RViz Image display)
  /logo/mask_viz   — BGR overlay of HSV masks on RGB (debug color thresholds)
  /logo/markers    — MarkerArray (thin CUBE planes + text labels + orientation arrows)

Log lines (throttled to ~3s):
  [DEPTH] shape, min, max, % in valid range
  [HSV]   pixel count per color
  [TF]    camera → map translation + rotation euler
  [LOGO]  orientation locked: center, yaw, orange/green positions
"""

import json
import math
import struct
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from apriltag_msgs.msg import AprilTagDetectionArray

import tf2_ros
from tf2_ros import TransformException


# Raw apriltag IDs (printed on SDF models) -> "logical" mission IDs that
# the mission_manager and CSV speak in. Kept in sync with the mapping in
# tag_command_node — both nodes must remap identically.
RAW_TO_LOGICAL_ID = {
    0: 2,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
}


# Target pixel hexes (RGB): orange #e77f33, blue #3b12b6, green #2baa27.
# Expected OpenCV HSV centers (OpenCV H is 0-180):
#   orange H≈13  S≈199 V≈231
#   blue   H≈127 S≈230 V≈182
#   green  H≈59  S≈196 V≈170   (rgba(43, 170, 39))
# Bands are ±10% of each channel's range around the center
# (±18 on H, ±25 on S/V), widened slightly on the lower S/V side
# to absorb Gazebo material shading.
COLOR_TARGETS = [
    ("orange", 0xe77f33,
     np.array([  0, 140, 150], dtype=np.uint8),
     np.array([ 31, 255, 255], dtype=np.uint8),
     (0.91, 0.50, 0.20), (51, 127, 231)),
    ("blue",   0x3b12b6,
     np.array([109, 170, 130], dtype=np.uint8),
     np.array([145, 255, 210], dtype=np.uint8),
     (0.23, 0.07, 0.71), (182, 18, 59)),
    ("green",  0x2baa27,
     np.array([ 41, 171, 145], dtype=np.uint8),
     np.array([ 77, 255, 195], dtype=np.uint8),
     (0.17, 0.67, 0.15), (39, 170, 43)),
    # Red (for stop-zone detection). OpenCV-HSV H≈0/180 (wraps), so
    # we use a low-H band; for highly saturated red this matches.
    # A second wrap-around range (170-180) is folded in via masking
    # in _red_mask() below.
    ("red",    0xc71c22,
     np.array([  0, 140, 120], dtype=np.uint8),
     np.array([ 10, 255, 255], dtype=np.uint8),
     (0.78, 0.11, 0.13), (34, 28, 199)),
]


def _white_near_mask(hsv: np.ndarray, radius_px: int) -> np.ndarray:
    """Return a uint8 mask where each pixel is 1 if a white pixel lies within
    `radius_px` in HSV. White is low saturation (S<40) + high value (V>200)."""
    white = cv2.inRange(hsv, np.array([0, 0, 200], dtype=np.uint8),
                             np.array([180, 40, 255], dtype=np.uint8))
    if radius_px <= 0:
        return white
    k = max(1, int(radius_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(white, kernel)


class LogoDetectorNode(Node):
    def __init__(self):
        super().__init__("logo_detector_node")

        self.bridge = CvBridge()

        self.declare_parameter("map_frame", "odom")
        self.declare_parameter("min_cluster_px", 60)
        self.declare_parameter("min_depth", 0.15)
        self.declare_parameter("max_depth", 4.0)
        self.declare_parameter("dedup_radius_m", 0.35)
        self.declare_parameter("apriltag_size_m", 0.12)
        self.declare_parameter("plane_thickness_m", 0.01)
        self.declare_parameter("logo_plane_size_m", 0.20)
        self.declare_parameter("depth_is_optical", True)
        self.declare_parameter("logo_cluster_radius_m", 0.7)
        self.declare_parameter("max_cloud_pts_per_color", 400)
        # Only count colors that land on the ground as logo symbols. Pixels
        # whose deprojected map-frame z is outside [ground_min_z, ground_max_z]
        # are rejected (e.g., colored walls, floating décor, lighting artifacts).
        self.declare_parameter("ground_min_z_m", -0.10)
        self.declare_parameter("ground_max_z_m", 0.12)
        # Only accept color pixels that have a white pixel within this many
        # pixels (dilated white mask). Ignores painted walls / décor that
        # aren't surrounded by the white tile background.
        self.declare_parameter("white_halo_px", 6)
        # If true, every published logo-cloud point is flattened to z = this
        # value (the nominal ground height in map frame), so the symbol
        # layer appears as a perfectly flat decal in RViz.
        self.declare_parameter("flatten_to_ground", True)
        self.declare_parameter("ground_plane_z_m", 0.0)

        self.map_frame            = self.get_parameter("map_frame").value
        self.min_cluster_px       = int(self.get_parameter("min_cluster_px").value)
        self.min_depth            = float(self.get_parameter("min_depth").value)
        self.max_depth            = float(self.get_parameter("max_depth").value)
        self.dedup_radius_sq      = float(self.get_parameter("dedup_radius_m").value) ** 2
        self.tag_size             = float(self.get_parameter("apriltag_size_m").value)
        self.plane_thickness      = float(self.get_parameter("plane_thickness_m").value)
        self.logo_size            = float(self.get_parameter("logo_plane_size_m").value)
        self.depth_is_optical     = bool(self.get_parameter("depth_is_optical").value)
        self._logo_cluster_r_sq   = float(self.get_parameter("logo_cluster_radius_m").value) ** 2
        self._max_cloud_pts       = int(self.get_parameter("max_cloud_pts_per_color").value)
        self.ground_min_z         = float(self.get_parameter("ground_min_z_m").value)
        self.ground_max_z         = float(self.get_parameter("ground_max_z_m").value)
        self.white_halo_px        = int(self.get_parameter("white_halo_px").value)
        self.flatten_to_ground    = bool(self.get_parameter("flatten_to_ground").value)
        self.ground_plane_z       = float(self.get_parameter("ground_plane_z_m").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Match the best_effort QoS we set on the ros_gz_bridge camera
        # topics — otherwise the reliable-default subscriber silently
        # doesn't connect and the RGBD callback never fires.
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        sub_rgb   = message_filters.Subscriber(self, Image,      "/r1_mini/camera/image_raw",  qos_profile=sensor_qos)
        sub_depth = message_filters.Subscriber(self, Image,      "/r1_mini/camera/depth_image", qos_profile=sensor_qos)
        sub_info  = message_filters.Subscriber(self, CameraInfo, "/r1_mini/camera/camera_info", qos_profile=sensor_qos)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info], queue_size=5, slop=0.15
        )
        self.ts.registerCallback(self._rgbd_cb)

        self._last_info = None
        self._last_depth = None
        self._last_depth_stamp = None

        self.create_subscription(
            AprilTagDetectionArray, "/apriltag/detections",
            self._apriltag_cb, 10,
        )

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_pub    = self.create_publisher(MarkerArray,  "/logo/markers",          latched)
        self.tag_pos_pub   = self.create_publisher(String,       "/apriltag/map_positions", latched)
        self.logo_pose_pub = self.create_publisher(PoseStamped,  "/logo/pose",             latched)
        # Persistent ground-paint cloud so the user can SEE the detected
        # orange / blue / green pixels on the floor in RViz. Accumulated
        # in self._ground_paint and republished on a timer whenever new
        # pixels have been added. Transient-local QoS = late RViz wins.
        persistent_cloud_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.logo_cloud_pub = self.create_publisher(
            PointCloud2, "/logo/points", persistent_cloud_qos)
        self.depth_viz_pub = self.create_publisher(Image,        "/logo/depth_viz",        10)
        self.mask_viz_pub  = self.create_publisher(Image,        "/logo/mask_viz",         10)

        self._locked_logos = []   # (name, x, y, z, rviz_rgb)
        self._locked_tags  = {}   # logical_id -> (x, y, z) — gated by order
        # Tag yaw (in map frame) at the moment of detection. Used to draw
        # the RViz plate upright facing the camera that saw it.
        self._locked_tag_yaws: dict[int, float] = {}
        # Buffered sightings of logical tags whose predecessor hasn't been
        # locked yet. Position is known so callers (mission_manager) can
        # turn/approach the tag, but it's not yet plated or CSV-logged.
        self._pending_tag_positions: dict[int, tuple[float, float, float]] = {}
        # Each entry: (cx, cy, cz, yaw) — one per unique logo location, locked once.
        self._locked_logo_poses = []

        # ── Persistent ground-paint memory ────────────────────────────────
        # Every ground-level color-band pixel that survived the white-halo
        # filter is deprojected into the map frame, quantized to a fixed
        # grid, and stored here. The store is keyed by the quantized cell
        # so we don't blow up memory when the same tile is re-seen many
        # times. Values are (rgb_int_packed, color_name, wx, wy) so we can
        # both visualize and query by color.
        self.GROUND_PIXEL_QUANTIZE_M = 0.02
        self._ground_paint: dict[tuple[int, int], tuple[int, str, float, float]] = {}
        self._ground_paint_dirty = False
        # 2 Hz re-publish is plenty for a growing-set visual.
        self.create_timer(0.5, self._publish_ground_paint)

        # ── Tile-pair publisher (blue-clustered tiles + paired orange/green) ──
        self.tile_pairs_pub = self.create_publisher(
            String, '/logo/tile_pairs', latched)
        # Red stop-zone publisher — fires when the red cluster centroid
        # is stable. Latched so mission_manager picks up a single value.
        self.stop_zone_pub = self.create_publisher(
            PoseStamped, '/stop_zone', latched)
        self._stop_zone_fired = False
        # Run clustering every 1 s — not every RGBD frame.
        self.create_timer(1.0, self._run_tile_clustering)
        # Tile-clustering parameters.
        self.TILE_CLUSTER_BIN_M = 0.05    # 5 cm cells in the binary grid
        self.TILE_CLUSTER_MIN_CELLS = 4   # ≥ 4 cells (~100 cm²) to count
        self.TILE_PAIR_MAX_RADIUS_M = 0.45  # orange/green ≤ this from blue centre

        self._frame_count = 0

        self.get_logger().info("logo_detector_node up")

    # ──────────────────────────────────────────────────────────────
    def _lookup_cam_to_map(self, camera_frame, stamp=None):
        try:
            lookup_t = stamp if stamp is not None else rclpy.time.Time()
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, camera_frame, lookup_t,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame, camera_frame, rclpy.time.Time()
                )
            except TransformException as e:
                self.get_logger().warn(
                    f"[TF] {self.map_frame}<-{camera_frame} failed: {e}",
                    throttle_duration_sec=3.0,
                )
                return None, None
        q = tf.transform.rotation
        t = tf.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([
            [1-2*y*y-2*z*z,  2*x*y-2*z*w,    2*x*z+2*y*w],
            [2*x*y+2*z*w,    1-2*x*x-2*z*z,  2*y*z-2*x*w],
            [2*x*z-2*y*w,    2*y*z+2*x*w,    1-2*x*x-2*y*y],
        ])
        return rot, np.array([t.x, t.y, t.z])

    def _deproject(self, u, v, d, fx, fy, cx, cy):
        """Deproject pixel+depth to 3D in the image's frame.

        If depth_is_optical: output is in optical frame (x-right, y-down, z-fwd).
        Else: convert to link frame (x-fwd, y-left, z-up).
        """
        opt_x = (u - cx) * d / fx
        opt_y = (v - cy) * d / fy
        opt_z = d
        if self.depth_is_optical:
            return np.array([opt_x, opt_y, opt_z])
        return np.array([opt_z, -opt_x, -opt_y])

    def _sample_depth(self, depth_img, u, v, patch_r=3):
        h, w = depth_img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        u_lo = max(0, u - patch_r); u_hi = min(w, u + patch_r + 1)
        v_lo = max(0, v - patch_r); v_hi = min(h, v + patch_r + 1)
        patch = depth_img[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
        valid = patch[(patch > self.min_depth) &
                      (patch < self.max_depth) &
                      np.isfinite(patch)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def _dedup_logo(self, name, x, y, z):
        # Dedup only within the same color — cross-color blobs belong to the same logo
        # and must all be locked independently so orientation can be computed.
        for lname, lx, ly, lz, _ in self._locked_logos:
            if lname != name:
                continue
            if (lx-x)**2 + (ly-y)**2 + (lz-z)**2 < self.dedup_radius_sq:
                return True
        return False

    # ──────────────────────────────────────────────────────────────
    def _publish_depth_viz(self, depth, stamp):
        finite = np.isfinite(depth) & (depth > 0)
        viz = np.zeros(depth.shape, dtype=np.uint8)
        if finite.any():
            d = np.where(finite, depth, 0.0)
            dmax = min(self.max_depth, float(np.percentile(d[finite], 95)))
            dmin = max(self.min_depth, float(d[finite].min()))
            if dmax - dmin > 1e-3:
                norm = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.0)
                norm[~finite] = 0.0
                # Invert so close = bright, far = dark (more intuitive).
                viz = ((1.0 - norm) * 255).astype(np.uint8)
                viz[~finite] = 0
        try:
            msg = self.bridge.cv2_to_imgmsg(viz, encoding="mono8")
            msg.header.stamp = stamp
            msg.header.frame_id = self._last_info[4] if self._last_info else "camera"
            self.depth_viz_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"depth_viz publish failed: {e}", throttle_duration_sec=5.0)

    def _publish_mask_viz(self, rgb, hsv, stamp):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = bgr.copy()
        for name, _, lo, hi, _, bgr_color in COLOR_TARGETS:
            mask = cv2.inRange(hsv, lo, hi)
            overlay[mask > 0] = bgr_color
        out = cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0)
        try:
            msg = self.bridge.cv2_to_imgmsg(out, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = self._last_info[4] if self._last_info else "camera"
            self.mask_viz_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"mask_viz publish failed: {e}", throttle_duration_sec=5.0)

    # ──────────────────────────────────────────────────────────────
    def _rgbd_cb(self, msg_rgb, msg_depth, msg_info):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="rgb8")
            depth_raw = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Normalize depth to meters float. Accept 32FC1 (meters) or 16UC1 (mm).
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) / 1000.0
        else:
            depth = depth_raw.astype(np.float32)

        fx, fy = float(msg_info.k[0]), float(msg_info.k[4])
        cx, cy = float(msg_info.k[2]), float(msg_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return

        # ── Resolution-mismatch fix (ros_gz_bridge / gz Fortress RGBD) ──
        # Gazebo Fortress publishes CameraInfo with `width/height` for the
        # RGB image, but sometimes stuffs the `K` matrix with intrinsics
        # computed for a lower-res internal render (cx=160, cy=120, fx=277
        # for what's actually a 640×480 image). We detect this by
        # comparing 2·cx to the actual image width — a correctly-built K
        # has cx ≈ img_w/2. Any deviation >10% triggers a rescale.
        img_h_actual, img_w_actual = depth.shape[:2]
        inferred_w = max(1.0, 2.0 * cx)
        inferred_h = max(1.0, 2.0 * cy)
        sx = img_w_actual / inferred_w
        sy = img_h_actual / inferred_h
        if abs(sx - 1.0) > 0.1 or abs(sy - 1.0) > 0.1:
            fx *= sx; cx *= sx
            fy *= sy; cy *= sy
            self.get_logger().warn(
                f"[K] K-matrix implies {inferred_w:.0f}x{inferred_h:.0f} "
                f"but image is {img_w_actual}x{img_h_actual} "
                f"— rescaled intrinsics by ({sx:.2f},{sy:.2f}) → "
                f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                throttle_duration_sec=10.0,
            )

        self._last_info = (fx, fy, cx, cy, msg_rgb.header.frame_id)
        self._last_depth = depth
        self._last_depth_stamp = msg_depth.header.stamp
        self._frame_count += 1

        self._publish_depth_viz(depth, msg_depth.header.stamp)

        rot_t = self._lookup_cam_to_map(msg_rgb.header.frame_id,
                                        rclpy.time.Time.from_msg(msg_rgb.header.stamp))
        if rot_t[0] is None:
            return
        rot, t_vec = rot_t

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        near_white = _white_near_mask(hsv, self.white_halo_px)

        new_detections = 0
        for name, _hx, lo, hi, rviz_rgb, _bgr in COLOR_TARGETS:
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.bitwise_and(mask, near_white)
            if cv2.countNonZero(mask) < self.min_cluster_px:
                continue

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_cluster_px:
                    continue
                u = int(cents[i, 0]); v = int(cents[i, 1])
                d = self._sample_depth(depth, u, v)
                if d is None:
                    self.get_logger().info(
                        f"[{name}] cluster at px=({u},{v}) area={area} "
                        f"but no valid depth (checked ±3px patch)"
                    )
                    continue
                p_cam = self._deproject(u, v, d, fx, fy, cx, cy)
                p_map = rot @ p_cam + t_vec
                x, y, z = float(p_map[0]), float(p_map[1]), float(p_map[2])
                on_ground = self.ground_min_z <= z <= self.ground_max_z
                self.get_logger().info(
                    f"[{name}] px=({u},{v}) d={d:.2f}m  "
                    f"p_cam=({p_cam[0]:.2f},{p_cam[1]:.2f},{p_cam[2]:.2f}) "
                    f"p_map=({x:.2f},{y:.2f},{z:.2f}) area={area} "
                    f"on_ground={on_ground}"
                )
                if not on_ground:
                    continue
                if self._dedup_logo(name, x, y, z):
                    continue
                self._locked_logos.append((name, x, y, z, rviz_rgb))
                new_detections += 1

        self._publish_mask_viz(rgb, hsv, msg_rgb.header.stamp)
        self._publish_logo_cloud(rgb, hsv, depth, rot, t_vec, fx, fy, cx, cy,
                                 msg_rgb.header.stamp)

        # _try_lock_logo (centroid-based orientation arrow) disabled —
        # the user will wire up sign-marking from the pixel store later.
        # Always republish markers so the progress strip stays alive even
        # before the first tag is detected.
        self._publish_markers()

    # ──────────────────────────────────────────────────────────────
    def _deproject_mask(self, mask, depth, fx, fy, cx, cy):
        """Vectorized deprojection of all non-zero mask pixels to 3D camera-frame points."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.empty((0, 3), dtype=np.float32)
        if len(xs) > self._max_cloud_pts:
            idx = np.random.choice(len(xs), self._max_cloud_pts, replace=False)
            xs, ys = xs[idx], ys[idx]
        h, w = depth.shape[:2]
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[valid], ys[valid]
        ds = depth[ys, xs].astype(np.float64)
        valid = (ds > self.min_depth) & (ds < self.max_depth) & np.isfinite(ds)
        xs, ys, ds = xs[valid], ys[valid], ds[valid]
        if len(xs) == 0:
            return np.empty((0, 3), dtype=np.float32)
        opt_x = (xs - cx) * ds / fx
        opt_y = (ys - cy) * ds / fy
        if self.depth_is_optical:
            pts = np.stack([opt_x, opt_y, ds], axis=1)
        else:
            pts = np.stack([ds, -opt_x, -opt_y], axis=1)
        return pts.astype(np.float32)

    def _publish_logo_cloud(self, rgb, hsv, depth, rot, t_vec, fx, fy, cx, cy, stamp):
        """Deproject every ground-level pixel that falls in any of the three
        target color bands, tag it with its color name, and store it in the
        persistent ground-paint map. Publication is handled on a timer by
        `_publish_ground_paint`, not per-frame, so the RViz decal accumulates
        over the whole run and is available as future direction input."""
        near_white = _white_near_mask(hsv, self.white_halo_px)
        h_img, w_img = depth.shape[:2]
        q = self.GROUND_PIXEL_QUANTIZE_M
        added = 0

        # Always accumulate ALL colours — blue is used later for tile
        # clustering, red for stop-zone detection. First-write-wins on
        # each quantised cell keeps memory bounded.
        for name, _, lo, hi, _, _ in COLOR_TARGETS:
            band = cv2.inRange(hsv, lo, hi)
            mask = cv2.bitwise_and(band, near_white)
            if cv2.countNonZero(mask) < self.min_cluster_px:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            cap = int(self._max_cloud_pts)
            if len(xs) > cap:
                idx = np.random.choice(len(xs), cap, replace=False)
                xs, ys = xs[idx], ys[idx]

            valid = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)
            xs, ys = xs[valid], ys[valid]
            ds = depth[ys, xs].astype(np.float64)
            valid = (ds > self.min_depth) & (ds < self.max_depth) & np.isfinite(ds)
            xs, ys, ds = xs[valid], ys[valid], ds[valid]
            if len(xs) == 0:
                continue

            opt_x = (xs - cx) * ds / fx
            opt_y = (ys - cy) * ds / fy
            if self.depth_is_optical:
                pts_cam = np.stack([opt_x, opt_y, ds], axis=1)
            else:
                pts_cam = np.stack([ds, -opt_x, -opt_y], axis=1)
            pts_cam = pts_cam.astype(np.float32)
            pts_map = (rot @ pts_cam.T).T + t_vec

            zs = pts_map[:, 2]
            ground_mask = (zs >= self.ground_min_z) & (zs <= self.ground_max_z)
            pts_map = pts_map[ground_mask]
            xs_ok = xs[ground_mask]
            ys_ok = ys[ground_mask]
            if len(pts_map) == 0:
                continue

            # Quantize to the persistent-paint grid and store per-cell.
            px_rgb = rgb[ys_ok, xs_ok]  # (N, 3) uint8 RGB
            qxs = np.round(pts_map[:, 0] / q).astype(np.int32)
            qys = np.round(pts_map[:, 1] / q).astype(np.int32)
            for qx, qy, pr, pg, pb, wx, wy in zip(
                    qxs.tolist(), qys.tolist(),
                    px_rgb[:, 0].tolist(), px_rgb[:, 1].tolist(), px_rgb[:, 2].tolist(),
                    pts_map[:, 0].tolist(), pts_map[:, 1].tolist()):
                key = (qx, qy)
                if key in self._ground_paint:
                    continue  # first-write wins, keeps it stable
                rgb_int = (int(pr) << 16) | (int(pg) << 8) | int(pb)
                self._ground_paint[key] = (rgb_int, name, float(wx), float(wy))
                added += 1

        if added > 0:
            self._ground_paint_dirty = True
            self.get_logger().info(
                f"[GROUND-PAINT] +{added} pixels (total={len(self._ground_paint)})",
                throttle_duration_sec=3.0,
            )

    def _publish_ground_paint(self):
        """Republish the accumulated ground-paint cloud on /logo/points."""
        if not self._ground_paint_dirty:
            return
        self._ground_paint_dirty = False
        n = len(self._ground_paint)
        if n == 0:
            return
        data = np.empty((n, 4), dtype=np.float32)
        z_plane = self.ground_plane_z if self.flatten_to_ground else 0.0
        for i, (rgb_int, _name, wx, wy) in enumerate(self._ground_paint.values()):
            data[i, 0] = wx
            data[i, 1] = wy
            data[i, 2] = z_plane
            data[i, 3] = np.array([rgb_int], dtype=np.uint32).view(np.float32)[0]
        cloud = PointCloud2()
        cloud.header.frame_id = self.map_frame
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.height = 1
        cloud.width = n
        cloud.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = 16 * n
        cloud.is_dense = True
        cloud.data = data.tobytes()
        self.logo_cloud_pub.publish(cloud)

    # ── Tile clustering + stop-zone ─────────────────────────────────────
    def _cluster_color_to_centroids(self, color_name: str,
                                    cells_per_m: float) -> list[tuple[float, float, int]]:
        """Build a binary 2-D grid of all accumulated pixels whose colour
        matches `color_name`, run connected-component labelling, return
        [(wx, wy, pixel_count), …] for each cluster that has at least
        TILE_CLUSTER_MIN_CELLS cells."""
        pts = [(wx, wy) for _rgb, n, wx, wy in self._ground_paint.values()
               if n == color_name]
        if len(pts) < self.TILE_CLUSTER_MIN_CELLS:
            return []
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        # Bounding box of this colour's points, padded by 2 cells.
        xmin, xmax = xs.min() - 0.1, xs.max() + 0.1
        ymin, ymax = ys.min() - 0.1, ys.max() + 0.1
        W = max(3, int(math.ceil((xmax - xmin) * cells_per_m)))
        H = max(3, int(math.ceil((ymax - ymin) * cells_per_m)))
        grid = np.zeros((H, W), dtype=np.uint8)
        cxs = np.clip(((xs - xmin) * cells_per_m).astype(np.int32), 0, W - 1)
        cys = np.clip(((ys - ymin) * cells_per_m).astype(np.int32), 0, H - 1)
        grid[cys, cxs] = 1
        # 1-cell dilation joins near-neighbour pixels into one blob.
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(grid, connectivity=8)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.TILE_CLUSTER_MIN_CELLS:
                continue
            gx, gy = float(cents[i, 0]), float(cents[i, 1])
            wx = xmin + gx / cells_per_m
            wy = ymin + gy / cells_per_m
            out.append((wx, wy, area))
        return out

    def _run_tile_clustering(self):
        """Cluster accumulated blue pixels into tile centres, pair each
        with the nearest orange + green cluster (within
        TILE_PAIR_MAX_RADIUS_M), snap the arrow direction to the nearest
        90°, and publish JSON on /logo/tile_pairs. Also publishes the
        red cluster (if big enough + stable) as /stop_zone."""
        if not self._ground_paint:
            return
        cells_per_m = 1.0 / self.TILE_CLUSTER_BIN_M
        blue_clusters = self._cluster_color_to_centroids('blue', cells_per_m)
        orange_clusters = self._cluster_color_to_centroids('orange', cells_per_m)
        green_clusters = self._cluster_color_to_centroids('green', cells_per_m)
        red_clusters = self._cluster_color_to_centroids('red', cells_per_m)

        # Pair each blue centre with the nearest orange + green within
        # TILE_PAIR_MAX_RADIUS_M. Snap the green→orange vector direction
        # to the nearest multiple of π/2.
        pairs = []
        R = self.TILE_PAIR_MAX_RADIUS_M
        for bx, by, _ba in blue_clusters:
            def nearest(xys):
                best = None; best_d = math.inf
                for x, y, a in xys:
                    d = math.hypot(x - bx, y - by)
                    if d <= R and d < best_d:
                        best_d = d; best = (x, y, a)
                return best
            o = nearest(orange_clusters)
            g = nearest(green_clusters)
            if o is None or g is None:
                continue
            ox, oy, _ = o; gx, gy, _ = g
            yaw_raw = math.atan2(oy - gy, ox - gx)
            # Snap to nearest π/2.
            yaw = round(yaw_raw / (math.pi / 2.0)) * (math.pi / 2.0)
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            pairs.append({
                'blue': [bx, by],
                'orange': [ox, oy],
                'green': [gx, gy],
                'yaw_snapped': yaw,
            })

        msg = String()
        msg.data = json.dumps(pairs)
        self.tile_pairs_pub.publish(msg)

        # Stop zone: publish the biggest red cluster once.
        if not self._stop_zone_fired and red_clusters:
            red_clusters.sort(key=lambda c: -c[2])
            rx, ry, area = red_clusters[0]
            if area >= 8:
                ps = PoseStamped()
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.header.frame_id = self.map_frame
                ps.pose.position.x = rx
                ps.pose.position.y = ry
                ps.pose.orientation.w = 1.0
                self.stop_zone_pub.publish(ps)
                self._stop_zone_fired = True
                self.get_logger().warn(
                    f'[STOP_ZONE] red cluster at ({rx:+.2f},{ry:+.2f}) '
                    f'area={area} cells')

    # ── Public API for future direction-decision logic ─────────────────
    def get_ground_pixels(self, color_name: str | None = None
                          ) -> list[tuple[float, float]]:
        """Return all accumulated ground-paint (x, y) positions in the map
        frame. If `color_name` is given ('orange' | 'blue' | 'green'),
        only pixels tagged with that band are returned."""
        if color_name is None:
            return [(wx, wy) for _rgb, _n, wx, wy in self._ground_paint.values()]
        return [(wx, wy) for _rgb, n, wx, wy in self._ground_paint.values()
                if n == color_name]

    def get_ground_pixel_count(self, color_name: str | None = None) -> int:
        if color_name is None:
            return len(self._ground_paint)
        return sum(1 for _r, n, _x, _y in self._ground_paint.values()
                   if n == color_name)

    # ──────────────────────────────────────────────────────────────
    def _try_lock_logo(self):
        """Group spatially correlated orange+green blobs into a logo instance.

        The logo's orientation is the vector from the green centroid to the orange
        centroid projected onto the XY (ground) plane: yaw = atan2(oy-gy, ox-gx).
        Each unique logo location is locked exactly once.
        """
        by_color = {}
        for lname, x, y, z, _ in self._locked_logos:
            by_color.setdefault(lname, []).append((x, y, z))

        oranges = by_color.get('orange', [])
        greens  = by_color.get('green',  [])
        if not oranges or not greens:
            return

        for ox, oy, oz in oranges:
            for gx, gy, gz in greens:
                d_sq = (ox - gx) ** 2 + (oy - gy) ** 2
                if d_sq > self._logo_cluster_r_sq:
                    continue

                # Candidate logo center (midpoint of orange/green, or centroid with blue)
                cx, cy, cz = (ox + gx) / 2.0, (oy + gy) / 2.0, (oz + gz) / 2.0
                for px, py, pz in by_color.get('blue', []):
                    if (px - cx) ** 2 + (py - cy) ** 2 < self._logo_cluster_r_sq:
                        cx = (ox + gx + px) / 3.0
                        cy = (oy + gy + py) / 3.0
                        cz = (oz + gz + pz) / 3.0
                        break

                # Skip if we already locked a logo at this location
                already = any(
                    (lx - cx) ** 2 + (ly - cy) ** 2 < self._logo_cluster_r_sq
                    for lx, ly, _lz, _yaw in self._locked_logo_poses
                )
                if already:
                    continue

                yaw = math.atan2(oy - gy, ox - gx)
                self._locked_logo_poses.append((cx, cy, cz, yaw))
                self._publish_logo_pose(cx, cy, cz, yaw)
                self.get_logger().info(
                    f'[LOGO] Orientation locked #{len(self._locked_logo_poses)}: '
                    f'center=({cx:.2f},{cy:.2f},{cz:.2f}) '
                    f'yaw={math.degrees(yaw):.1f}° '
                    f'orange=({ox:.2f},{oy:.2f}) green=({gx:.2f},{gy:.2f})'
                )

    def _publish_logo_pose(self, x, y, z, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        self.logo_pose_pub.publish(ps)

    # ──────────────────────────────────────────────────────────────
    def _apriltag_cb(self, msg):
        if not msg.detections:
            return
        if self._last_info is None or self._last_depth is None:
            self.get_logger().warn(
                "[APRIL] detection arrived before first RGBD frame — dropping",
                throttle_duration_sec=3.0,
            )
            return
        fx, fy, cx, cy, cam_frame = self._last_info
        depth = self._last_depth

        # Use LATEST TF (matches legacy aruco_localizer: it also uses
        # rclpy.time.Time()). Using the detection's header stamp caused
        # extrapolation failures + stale TF lookups.
        rot_t = self._lookup_cam_to_map(cam_frame)
        if rot_t[0] is None:
            self.get_logger().error(
                f"[APRIL] TF lookup {self.map_frame}<-{cam_frame} FAILED — "
                f"all tag positions will be dropped until TF tree is up")
            return
        rot, t_vec = rot_t

        # Loud TF sanity print, throttled. If static_tf_cam has the correct
        # body->optical rotation baked in, rot @ [0,0,1] (optical-forward)
        # should be in the direction the robot is currently facing in map.
        # Logged at WARN level so it survives --log-level WARN on this node.
        forward_in_map = rot @ np.array([0.0, 0.0, 1.0])
        self.get_logger().info(
            f"[APRIL][TF] optical_fwd_in_map=({forward_in_map[0]:+.2f},"
            f"{forward_in_map[1]:+.2f},{forward_in_map[2]:+.2f})",
            throttle_duration_sec=30.0,
        )
        # The z-component of optical_fwd_in_map should be ≈ 0 (camera points
        # horizontally, not at the sky or the floor). If it's close to ±1
        # the static_tf_cam rotation is still identity / wrong — loud warn.
        if abs(forward_in_map[2]) > 0.5:
            self.get_logger().error(
                f"[APRIL][TF] ⚠ optical_forward in map has |z|={abs(forward_in_map[2]):.2f} "
                f">0.5 — camera TF rotation is probably wrong. Check static_tf_cam "
                f"in round3.launch.py; body→optical rotation must be yaw=-π/2 roll=-π/2"
            )

        changed = False
        for det in msg.detections:
            try:
                raw_id = int(det.id)
            except (TypeError, ValueError):
                raw_id = int(det.id[0])
            logical_id = RAW_TO_LOGICAL_ID.get(raw_id)
            if logical_id is None:
                continue
            u = int(det.centre.x); v = int(det.centre.y)
            d = self._sample_depth(depth, u, v, patch_r=5)
            if d is None:
                self.get_logger().warn(
                    f"[APRIL#L{logical_id}/R{raw_id}] centre=({u},{v}) but no valid depth — skipping",
                    throttle_duration_sec=3.0,
                )
                continue
            p_cam = self._deproject(u, v, d, fx, fy, cx, cy)
            p_map = rot @ p_cam + t_vec
            x, y, z = float(p_map[0]), float(p_map[1]), float(p_map[2])
            # Only log on first lock of a new tag (or heavily throttled
            # for already-locked re-sightings). Per-frame WARN output at
            # 10-15 Hz was eating real-time factor.
            is_new = logical_id not in self._locked_tags
            if is_new:
                self.get_logger().warn(
                    f"[APRIL#L{logical_id}/R{raw_id}][NEW] "
                    f"px=({u},{v}) d={d:.3f}m "
                    f"p_map=({x:+.3f},{y:+.3f},{z:+.3f})"
                )
            if logical_id in self._locked_tags:
                continue
            # Gate plate visibility: never draw a plate for logical N if
            # logical N-1 hasn't been locked yet (tag 1 is exempt). The
            # live position/TF detection math still runs and the position
            # is stored so actions can later use it.
            if logical_id > 1 and (logical_id - 1) not in self._locked_tags:
                self.get_logger().info(
                    f"[APRIL#L{logical_id}] sighting BUFFERED — waiting for "
                    f"tag {logical_id - 1} to be detected first",
                    throttle_duration_sec=3.0,
                )
                # Still record the position so mission_manager can use it
                # (buffered-sighting semantics: we know *where* tag N is;
                # we just don't render or log it yet).
                self._pending_tag_positions[logical_id] = (x, y, z)
                continue
            self._locked_tags[logical_id] = (x, y, z)
            # Tag's outward normal points back at the camera that saw it:
            # yaw = atan2(cam_y − tag_y, cam_x − tag_x). The plate's face
            # will be drawn pointing in this direction (i.e., facing us).
            tag_yaw = math.atan2(t_vec[1] - y, t_vec[0] - x)
            self._locked_tag_yaws[logical_id] = float(tag_yaw)
            self._pending_tag_positions.pop(logical_id, None)
            changed = True

        # Promotion sweep: any buffered sighting whose predecessor is now
        # locked gets promoted to _locked_tags (and published) without
        # needing to re-see the tag. Critical for tag 2 — it's typically
        # seen BEFORE tag 1, and LOOKING_FOR_TAG2 needs the position to
        # dispatch the approach goal.
        for pend_id in sorted(list(self._pending_tag_positions.keys())):
            if (pend_id - 1) in self._locked_tags and pend_id not in self._locked_tags:
                px, py, pz = self._pending_tag_positions.pop(pend_id)
                self._locked_tags[pend_id] = (px, py, pz)
                # We don't have a fresh camera origin for the buffered
                # detection, so use the current cam origin as a proxy
                # for plate orientation. Good enough for a visualisation.
                self._locked_tag_yaws[pend_id] = float(
                    math.atan2(t_vec[1] - py, t_vec[0] - px))
                self.get_logger().warn(
                    f'[APRIL#L{pend_id}][PROMOTED] buffered sighting '
                    f'→ _locked_tags (predecessor now locked)')
                changed = True

        if changed:
            self._publish_markers()
            self._publish_tag_positions()

    def _publish_tag_positions(self):
        payload = {str(tid): [round(x, 3), round(y, 3), round(z, 3)]
                   for tid, (x, y, z) in self._locked_tags.items()}
        msg = String()
        msg.data = json.dumps(payload)
        self.tag_pos_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────
    def _plane_marker(self, mid, ns, x, y, z, r, g, b, size):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = size
        m.scale.y = size
        m.scale.z = self.plane_thickness
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b)
        m.color.a = 0.9
        m.lifetime.sec = 0
        return m

    def _progress_strip_marker(self) -> Marker:
        """TEXT_VIEW_FACING strip anchored in the map frame that shows the
        live tag-detection progress, e.g.  'TAGS SEEN:  1✓  2_  3_  4_  5_'.
        Locked tags are ticked; unseen are underscored. Colour toggles
        to green once all five are locked."""
        pieces = []
        for i in range(1, 6):
            mark = '✓' if i in self._locked_tags else '_'
            pieces.append(f'{i}{mark}')
        text = 'TAGS SEEN: ' + '  '.join(pieces)
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'tag_progress_strip'
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = 0.0
        m.pose.position.y = 0.0
        m.pose.position.z = 1.8
        m.pose.orientation.w = 1.0
        m.scale.z = 0.22
        if len(self._locked_tags) >= 5:
            m.color.r = 0.2; m.color.g = 1.0; m.color.b = 0.3
        else:
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        m.lifetime.sec = 0
        return m

    def _vertical_plate_marker(self, mid, ns, x, y, z, yaw, r, g, b, size):
        """Vertical square whose outward normal points along the given yaw
        in the map frame. Used so apriltag plates in RViz stand upright
        matching the physical tag orientation (tags are mounted on walls).

        Construction: a CUBE of size×size×thickness. We orient it so:
          - the tag's thin axis (z_cube) points along the tag's normal
          - the tag's x_cube (first face-axis) points in the world horizontal
          - the tag's y_cube (second face-axis) points world-up

        That's a rotation: first roll=−π/2 (flip flat CUBE onto its edge so
        thin axis is horizontal), then yaw about world-z so the thin axis
        aligns with the tag's outward normal.
        """
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        # Compose q = Rz(yaw) * Rx(-π/2) as a single quaternion.
        # Rx(-π/2): qx = sin(-π/4), qw = cos(-π/4)  → (-0.7071, 0, 0, 0.7071)
        # Rz(yaw):  qz = sin(yaw/2), qw = cos(yaw/2)
        # Product (rotation Rz then Rx applied to the frame):
        cy_h = math.cos(yaw / 2.0); sy_h = math.sin(yaw / 2.0)
        cx_h = math.cos(-math.pi / 4.0); sx_h = math.sin(-math.pi / 4.0)
        # q_z * q_x  (quaternion multiplication, scalar-last: (x,y,z,w))
        qx = cy_h * sx_h
        qy = sy_h * sx_h
        qz = sy_h * cx_h
        qw = cy_h * cx_h
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = size                     # face width
        m.scale.y = size                     # face height
        m.scale.z = self.plane_thickness     # along tag normal
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b)
        m.color.a = 0.9
        m.lifetime.sec = 0
        return m

    def _text_marker(self, mid, ns, x, y, z, text):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + 0.15
        m.pose.orientation.w = 1.0
        m.scale.z = 0.12
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0
        m.text = text
        m.lifetime.sec = 0
        return m

    def _arrow_marker(self, mid, ns, x, y, z, yaw, length=0.4):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + 0.05
        m.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.orientation.w = math.cos(yaw / 2.0)
        m.scale.x = length
        m.scale.y = 0.05
        m.scale.z = 0.05
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
        m.lifetime.sec = 0
        return m

    def _publish_markers(self):
        # Logo markers (plate / arrow / text) are removed. The per-pixel
        # ground-paint accumulation in self._ground_paint will be used to
        # mark signs when the user wires up that logic explicitly. For
        # now the only visual is apriltag plates + their IDs.
        arr = MarkerArray()
        for logical_id, (x, y, z) in self._locked_tags.items():
            yaw = self._locked_tag_yaws.get(logical_id, 0.0)
            arr.markers.append(self._vertical_plate_marker(
                logical_id, "apriltag", x, y, z, yaw, 1.0, 1.0, 0.0, self.tag_size))
            arr.markers.append(self._text_marker(
                logical_id, "apriltag_text", x, y, z, f"tag{logical_id}"))
        # Demo progress strip: shows the detection sequence 1..5 at a fixed
        # spot in the map frame. Tags already locked render in green, the
        # rest in dim grey. Floats at z=1.8 m, screen-facing so it's
        # always readable.
        arr.markers.append(self._progress_strip_marker())
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = LogoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
