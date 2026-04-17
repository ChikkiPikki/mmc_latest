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


# Target pixel hexes (RGB): purple #3b12b6, orange #e77f33, green #2b9e27.
# Expected OpenCV HSV centers: purple H≈127 S≈230 V≈182,
# orange H≈13 S≈199 V≈231, green H≈59 S≈192 V≈158.
# Bands are wide because Gazebo lighting / material shading shifts hue+sat.
COLOR_TARGETS = [
    ("purple", 0x3b12b6,
     np.array([105,  50,  30], dtype=np.uint8),
     np.array([150, 255, 255], dtype=np.uint8),
     (0.23, 0.07, 0.71), (182, 18, 59)),
    ("orange", 0xe77f33,
     np.array([  0,  90,  80], dtype=np.uint8),
     np.array([ 28, 255, 255], dtype=np.uint8),
     (0.91, 0.50, 0.20), (51, 127, 231)),
    ("green",  0x2b9e27,
     np.array([ 35,  70,  40], dtype=np.uint8),
     np.array([ 85, 255, 255], dtype=np.uint8),
     (0.17, 0.62, 0.15), (39, 158, 43)),
]


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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sub_rgb   = message_filters.Subscriber(self, Image,      "/r1_mini/camera/image_raw")
        sub_depth = message_filters.Subscriber(self, Image,      "/r1_mini/camera/depth_image")
        sub_info  = message_filters.Subscriber(self, CameraInfo, "/r1_mini/camera/camera_info")
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
        self.logo_cloud_pub = self.create_publisher(PointCloud2, "/logo/points",           10)
        self.depth_viz_pub = self.create_publisher(Image,        "/logo/depth_viz",        10)
        self.mask_viz_pub  = self.create_publisher(Image,        "/logo/mask_viz",         10)

        self._locked_logos = []   # (name, x, y, z, rviz_rgb)
        self._locked_tags  = {}   # tag_id -> (x, y, z)
        # Each entry: (cx, cy, cz, yaw) — one per unique logo location, locked once.
        self._locked_logo_poses = []

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
            dmax = min(self.max_depth, float(np.percentile(depth[finite], 95)))
            dmin = max(self.min_depth, float(depth[finite].min()))
            if dmax - dmin > 1e-3:
                norm = np.clip((depth - dmin) / (dmax - dmin), 0.0, 1.0)
                viz = (norm * 255).astype(np.uint8)
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

        fx, fy = msg_info.k[0], msg_info.k[4]
        cx, cy = msg_info.k[2], msg_info.k[5]
        if fx == 0.0 or fy == 0.0:
            return

        self._last_info = (fx, fy, cx, cy, msg_rgb.header.frame_id)
        self._last_depth = depth
        self._last_depth_stamp = msg_depth.header.stamp
        self._frame_count += 1

        # ── Depth diagnostics ──
        finite = np.isfinite(depth) & (depth > 0)
        in_range = finite & (depth >= self.min_depth) & (depth <= self.max_depth)
        if self._frame_count <= 3 or self._frame_count % 30 == 0:
            if finite.any():
                dmin = float(depth[finite].min()); dmax = float(depth[finite].max())
                dmed = float(np.median(depth[finite]))
            else:
                dmin = dmax = dmed = float('nan')
            self.get_logger().info(
                f"[DEPTH] dtype={depth_raw.dtype} shape={depth.shape} "
                f"finite={finite.mean()*100:.1f}% in_range={in_range.mean()*100:.1f}% "
                f"min={dmin:.2f} med={dmed:.2f} max={dmax:.2f} "
                f"K=[fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}] "
                f"frame_id='{msg_rgb.header.frame_id}'"
            )

        rot_t = self._lookup_cam_to_map(msg_rgb.header.frame_id,
                                        rclpy.time.Time.from_msg(msg_rgb.header.stamp))
        if rot_t[0] is None:
            return
        rot, t_vec = rot_t

        if self._frame_count <= 3 or self._frame_count % 30 == 0:
            self.get_logger().info(
                f"[TF] map<-cam t=({t_vec[0]:.2f},{t_vec[1]:.2f},{t_vec[2]:.2f}) "
                f"Rx=({rot[0,0]:.2f},{rot[0,1]:.2f},{rot[0,2]:.2f})"
            )

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # ── HSV diagnostics ──
        if self._frame_count % 15 == 0:
            counts = {name: int(cv2.countNonZero(cv2.inRange(hsv, lo, hi)))
                      for name, _, lo, hi, _, _ in COLOR_TARGETS}
            sat_mask = hsv[:, :, 1] > 80
            if sat_mask.any():
                h_sat = hsv[:, :, 0][sat_mask]
                hist, _ = np.histogram(h_sat, bins=18, range=(0, 180))
                top = np.argsort(hist)[-5:][::-1]
                peaks = [(int(b) * 10, int(hist[b])) for b in top if hist[b] > 50]
                self.get_logger().info(
                    f"[HSV] band_counts={counts} top_hue_peaks(saturated)={peaks}"
                )
            else:
                self.get_logger().info(f"[HSV] band_counts={counts} (no saturated px)")

        new_detections = 0
        for name, _hx, lo, hi, rviz_rgb, _bgr in COLOR_TARGETS:
            mask = cv2.inRange(hsv, lo, hi)
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
                self.get_logger().info(
                    f"[{name}] px=({u},{v}) d={d:.2f}m  "
                    f"p_cam=({p_cam[0]:.2f},{p_cam[1]:.2f},{p_cam[2]:.2f}) "
                    f"p_map=({x:.2f},{y:.2f},{z:.2f}) area={area}"
                )
                if self._dedup_logo(name, x, y, z):
                    continue
                self._locked_logos.append((name, x, y, z, rviz_rgb))
                new_detections += 1

        self._publish_depth_viz(depth, msg_depth.header.stamp)
        self._publish_mask_viz(rgb, hsv, msg_rgb.header.stamp)
        self._publish_logo_cloud(hsv, depth, rot, t_vec, fx, fy, cx, cy,
                                 msg_rgb.header.stamp)

        if new_detections:
            self._try_lock_logo()

        if new_detections or self._locked_tags:
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

    def _publish_logo_cloud(self, hsv, depth, rot, t_vec, fx, fy, cx, cy, stamp):
        """Deproject all detected logo color pixels to 3D map coords and publish PointCloud2."""
        segments = []
        for name, _, lo, hi, rviz_rgb, _ in COLOR_TARGETS:
            mask = cv2.inRange(hsv, lo, hi)
            if cv2.countNonZero(mask) < self.min_cluster_px:
                continue
            pts_cam = self._deproject_mask(mask, depth, fx, fy, cx, cy)
            if len(pts_cam) == 0:
                continue
            pts_map = (rot @ pts_cam.T).T + t_vec
            r, g, b = rviz_rgb
            rgb_int = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
            segments.append((pts_map.astype(np.float32), rgb_float))

        if not segments:
            return

        rows = []
        for pts, rgb_f in segments:
            col = np.full((len(pts), 1), rgb_f, dtype=np.float32)
            rows.append(np.hstack([pts, col]))
        data = np.vstack(rows)

        cloud = PointCloud2()
        cloud.header.frame_id = self.map_frame
        cloud.header.stamp = stamp
        cloud.height = 1
        cloud.width = len(data)
        cloud.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = 16 * len(data)
        cloud.is_dense = False
        cloud.data = data.tobytes()
        self.logo_cloud_pub.publish(cloud)

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

                # Candidate logo center (midpoint of orange/green, or centroid with purple)
                cx, cy, cz = (ox + gx) / 2.0, (oy + gy) / 2.0, (oz + gz) / 2.0
                for px, py, pz in by_color.get('purple', []):
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

        rot_t = self._lookup_cam_to_map(cam_frame,
                                        rclpy.time.Time.from_msg(msg.header.stamp))
        if rot_t[0] is None:
            return
        rot, t_vec = rot_t

        changed = False
        for det in msg.detections:
            try:
                tag_id = int(det.id)
            except (TypeError, ValueError):
                tag_id = int(det.id[0])
            if tag_id in self._locked_tags:
                continue
            u = int(det.centre.x); v = int(det.centre.y)
            d = self._sample_depth(depth, u, v, patch_r=5)
            if d is None:
                self.get_logger().warn(
                    f"[APRIL#{tag_id}] centre=({u},{v}) but no valid depth — skipping",
                    throttle_duration_sec=3.0,
                )
                continue
            p_cam = self._deproject(u, v, d, fx, fy, cx, cy)
            p_map = rot @ p_cam + t_vec
            x, y, z = float(p_map[0]), float(p_map[1]), float(p_map[2])
            self.get_logger().info(
                f"[APRIL#{tag_id}] px=({u},{v}) d={d:.2f}m "
                f"p_cam=({p_cam[0]:.2f},{p_cam[1]:.2f},{p_cam[2]:.2f}) "
                f"p_map=({x:.2f},{y:.2f},{z:.2f}) t_cam=({t_vec[0]:.2f},{t_vec[1]:.2f},{t_vec[2]:.2f})"
            )
            self._locked_tags[tag_id] = (x, y, z)
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
        arr = MarkerArray()
        for idx, (name, x, y, z, rgb) in enumerate(self._locked_logos):
            r, g, b = rgb
            arr.markers.append(self._plane_marker(idx, f"logo_{name}", x, y, z, r, g, b, self.logo_size))
            arr.markers.append(self._text_marker(idx, f"logo_{name}_text", x, y, z, name))
        for tag_id, (x, y, z) in self._locked_tags.items():
            arr.markers.append(self._plane_marker(tag_id, "apriltag", x, y, z, 1.0, 1.0, 0.0, self.tag_size))
            arr.markers.append(self._text_marker(tag_id, "apriltag_text", x, y, z, f"tag{tag_id}"))
        for idx, (cx, cy, cz, yaw) in enumerate(self._locked_logo_poses):
            arr.markers.append(self._arrow_marker(idx, "logo_orientation", cx, cy, cz, yaw))
            arr.markers.append(self._text_marker(
                idx, "logo_orient_text", cx, cy, cz,
                f"logo#{idx+1} {math.degrees(yaw):.0f}°",
            ))
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
