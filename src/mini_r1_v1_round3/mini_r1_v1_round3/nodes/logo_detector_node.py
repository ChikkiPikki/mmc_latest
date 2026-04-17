"""Logo + AprilTag deprojection node.

Detects three logo colors (#3b12b6, #e77f33, #2b9e27) and AprilTag centers in
the RGB image, deprojects pixel + depth to 3D in the camera frame, transforms
into `map`, and publishes plane markers.

Diagnostics:
  /logo/depth_viz  — MONO8 normalized depth (view in RViz Image display)
  /logo/mask_viz   — BGR overlay of HSV masks on RGB (debug color thresholds)
  /logo/markers    — MarkerArray (thin CUBE planes + text labels)

Log lines (throttled to ~3s):
  [DEPTH] shape, min, max, % in valid range
  [HSV]   pixel count per color
  [TF]    camera → map translation + rotation euler
"""

import json
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
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

        self.map_frame       = self.get_parameter("map_frame").value
        self.min_cluster_px  = int(self.get_parameter("min_cluster_px").value)
        self.min_depth       = float(self.get_parameter("min_depth").value)
        self.max_depth       = float(self.get_parameter("max_depth").value)
        self.dedup_radius_sq = float(self.get_parameter("dedup_radius_m").value) ** 2
        self.tag_size        = float(self.get_parameter("apriltag_size_m").value)
        self.plane_thickness = float(self.get_parameter("plane_thickness_m").value)
        self.logo_size       = float(self.get_parameter("logo_plane_size_m").value)
        self.depth_is_optical = bool(self.get_parameter("depth_is_optical").value)

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
        self.marker_pub    = self.create_publisher(MarkerArray, "/logo/markers",          latched)
        self.tag_pos_pub   = self.create_publisher(String,      "/apriltag/map_positions", latched)
        self.depth_viz_pub = self.create_publisher(Image,       "/logo/depth_viz",        10)
        self.mask_viz_pub  = self.create_publisher(Image,       "/logo/mask_viz",         10)

        self._locked_logos = []   # (name, x, y, z, rviz_rgb)
        self._locked_tags  = {}   # tag_id -> (x, y, z)

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

    def _dedup_logo(self, x, y, z):
        for _, lx, ly, lz, _ in self._locked_logos:
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
                if self._dedup_logo(x, y, z):
                    continue
                self._locked_logos.append((name, x, y, z, rviz_rgb))
                new_detections += 1

        self._publish_depth_viz(depth, msg_depth.header.stamp)
        self._publish_mask_viz(rgb, hsv, msg_rgb.header.stamp)

        if new_detections or self._locked_tags:
            self._publish_markers()

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

    def _publish_markers(self):
        arr = MarkerArray()
        for idx, (name, x, y, z, rgb) in enumerate(self._locked_logos):
            r, g, b = rgb
            arr.markers.append(self._plane_marker(idx, f"logo_{name}", x, y, z, r, g, b, self.logo_size))
            arr.markers.append(self._text_marker(idx, f"logo_{name}_text", x, y, z, name))
        for tag_id, (x, y, z) in self._locked_tags.items():
            arr.markers.append(self._plane_marker(tag_id, "apriltag", x, y, z, 1.0, 1.0, 0.0, self.tag_size))
            arr.markers.append(self._text_marker(tag_id, "apriltag_text", x, y, z, f"tag{tag_id}"))
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
