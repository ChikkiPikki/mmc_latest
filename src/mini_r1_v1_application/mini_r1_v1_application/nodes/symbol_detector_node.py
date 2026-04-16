"""Symbol (arrow sign) detector — FastSAM + HSV filtering, ported from legacy marker_detector_node.

Pipeline: FastSAM segmentation → orange HSV filter → arrow direction classification
        → depth deprojection → spatial dedup → publish.
"""

import json
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
import tf2_ros

from mini_r1_v1_application.utils.deprojection import (
    depth_to_meters, deproject_pixel, depth_patch_median, tf_to_rotation_matrix)


class SymbolDetectorNode(Node):
    def __init__(self):
        super().__init__('symbol_detector_node')

        # Parameters
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('max_detection_dist_m', 3.5)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('dedup_radius_m', 0.5)
        self.declare_parameter('process_every_n', 3)
        self.declare_parameter('sync_slop_s', 0.1)
        self.declare_parameter('sam_imgsz', 480)
        self.declare_parameter('sam_conf', 0.3)

        # HSV orange arrow filter params (from legacy)
        self.declare_parameter('sign_arrow_h_low', 0)
        self.declare_parameter('sign_arrow_h_high', 25)
        self.declare_parameter('sign_arrow_s_min', 150)
        self.declare_parameter('sign_arrow_v_min', 100)
        self.declare_parameter('sign_min_mask_area', 300)
        self.declare_parameter('sign_arrow_ratio_min', 0.30)
        self.declare_parameter('sign_curved_max_convexity', 0.65)

        self.max_det_dist = self.get_parameter('max_detection_dist_m').value
        self.min_depth = self.get_parameter('min_depth').value
        self.dedup_radius = self.get_parameter('dedup_radius_m').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self.sync_slop = self.get_parameter('sync_slop_s').value
        self.sam_imgsz = self.get_parameter('sam_imgsz').value
        self.sam_conf = self.get_parameter('sam_conf').value

        self.arrow_h_low = self.get_parameter('sign_arrow_h_low').value
        self.arrow_h_high = self.get_parameter('sign_arrow_h_high').value
        self.arrow_s_min = self.get_parameter('sign_arrow_s_min').value
        self.arrow_v_min = self.get_parameter('sign_arrow_v_min').value
        self.min_mask_area = self.get_parameter('sign_min_mask_area').value
        self.arrow_ratio_min = self.get_parameter('sign_arrow_ratio_min').value
        self.curved_max_convexity = self.get_parameter('sign_curved_max_convexity').value

        self.bridge = CvBridge()
        self.frame_count = 0
        self.depth_encoding_logged = False
        self.sam_model = None

        # Locked detections: {id: {'direction': str, 'x': float, 'y': float, ...}}
        self.locked_detections = {}
        self.next_det_id = 0

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        # Synchronized subscribers
        rgb_sub = message_filters.Subscriber(
            self, Image, '/r1_mini/camera/image_raw', qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, '/r1_mini/camera/depth_image', qos_profile=sensor_qos)
        info_sub = message_filters.Subscriber(
            self, CameraInfo, '/r1_mini/camera/camera_info', qos_profile=sensor_qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub],
            queue_size=5,
            slop=self.sync_slop)
        self.sync.registerCallback(self._sync_cb)

        # Publishers
        self.markers_pub = self.create_publisher(MarkerArray, '/symbols/detections', 10)
        self.report_pub = self.create_publisher(String, '/symbols/report', reliable_transient)
        self.annotated_pub = self.create_publisher(Image, '/symbols/annotated_image',
                                                   QoSProfile(
                                                       reliability=ReliabilityPolicy.BEST_EFFORT,
                                                       depth=1))

        # Load FastSAM model
        self._load_model()

        self.get_logger().info(
            f'Symbol detector (FastSAM): dedup_r={self.dedup_radius}m, '
            f'process_every={self.process_every_n}')

    def _load_model(self):
        try:
            from ultralytics import FastSAM
            self.sam_model = FastSAM('FastSAM-s.pt')
            self.get_logger().info('FastSAM model loaded')
        except Exception as e:
            self.get_logger().error(f'Failed to load FastSAM model: {e}')
            self.sam_model = None

    # ── Sync callback ──────────────────────────────────────────────────

    def _sync_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return
        if self.sam_model is None:
            return

        # Log depth encoding once
        if not self.depth_encoding_logged:
            self.get_logger().info(f'Depth encoding: {depth_msg.encoding}')
            self.depth_encoding_logged = True

        # Convert images
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
            cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            cv_depth = depth_to_meters(cv_depth_raw, depth_msg.encoding)
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')
            return

        # Camera intrinsics
        fx, fy = info_msg.k[0], info_msg.k[4]
        cx, cy = info_msg.k[2], info_msg.k[5]
        if fx == 0.0 or fy == 0.0:
            return

        # Get TF: camera frame -> odom
        try:
            trans = self.tf_buffer.lookup_transform(
                'odom', rgb_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return

        q = trans.transform.rotation
        t = trans.transform.translation
        rot = tf_to_rotation_matrix(q.x, q.y, q.z, q.w)
        t_vec = np.array([t.x, t.y, t.z])

        # Run FastSAM + sign detection pipeline
        self._detect_signs(cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec,
                           rgb_msg.header.frame_id)

    # ── FastSAM sign detection (ported from legacy) ────────────────────

    def _detect_signs(self, cv_rgb, cv_depth, fx, fy, cx, cy, rot, t_vec, frame_id):
        h_img, w_img = cv_rgb.shape[:2]
        h_depth, w_depth = cv_depth.shape[:2]
        stamp = self.get_clock().now().to_msg()

        # Legacy used RGB->BGR for SAM, and RGB->HSV for filtering
        if len(cv_rgb.shape) == 3 and cv_rgb.shape[2] == 3:
            cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2HSV)
        else:
            cv_bgr = cv_rgb
            hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)

        try:
            results = self.sam_model(cv_bgr, imgsz=self.sam_imgsz,
                                     conf=self.sam_conf, verbose=False)
        except Exception as e:
            self.get_logger().error(f'SAM inference error: {e}',
                                    throttle_duration_sec=10.0)
            return

        annotated = cv_bgr.copy()
        new_detections = False

        if results and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()

            for idx in range(masks_data.shape[0]):
                raw_mask = masks_data[idx].astype(np.uint8)
                if raw_mask.shape[:2] != (h_img, w_img):
                    mask = cv2.resize(raw_mask, (w_img, h_img),
                                      interpolation=cv2.INTER_NEAREST)
                else:
                    mask = raw_mask
                mask_u8 = (mask > 0.5).astype(np.uint8) * 255

                mask_area = np.count_nonzero(mask_u8)
                if mask_area < self.min_mask_area:
                    continue
                # Reject very large masks (floor, walls)
                if mask_area > (h_img * w_img * 0.1):
                    continue

                # Filter: must be orange (arrow color)
                if not self._is_arrow_color_mask(hsv, mask_u8):
                    continue

                # Reject if mask is too large (dynamic obstacles, large orange objects)
                if mask_area > (h_img * w_img * 0.03):
                    continue

                # Filter: orange region must have roughly square-ish bbox
                ys_m, xs_m = np.where(mask_u8 > 0)
                mw = xs_m.max() - xs_m.min() + 1
                mh = ys_m.max() - ys_m.min() + 1
                aspect = max(mw, mh) / (min(mw, mh) + 1)
                if aspect > 4.0:
                    continue  # too elongated, not an arrow sign

                # Extract ONLY orange pixels within this SAM mask, classify direction
                lower = np.array([self.arrow_h_low, self.arrow_s_min, self.arrow_v_min])
                upper = np.array([self.arrow_h_high, 255, 255])
                orange_in_mask = cv2.inRange(hsv, lower, upper) & mask_u8
                direction, cnt = self._classify_direction_from_mask(orange_in_mask)
                if direction is None or cnt is None:
                    continue

                # Get centroid
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                centroid_u = int(M['m10'] / M['m00'])
                centroid_v = int(M['m01'] / M['m00'])

                if centroid_u < 0 or centroid_u >= w_depth or centroid_v < 0 or centroid_v >= h_depth:
                    continue

                # Depth at centroid (patch median)
                depth_m = depth_patch_median(cv_depth, centroid_u, centroid_v,
                                             patch_radius=5,
                                             min_depth=self.min_depth,
                                             max_depth=self.max_det_dist)
                if math.isnan(depth_m):
                    continue

                # 3D projection: camera -> odom
                p_cam = deproject_pixel(centroid_u, centroid_v, depth_m, fx, fy, cx, cy)
                p_odom = rot @ p_cam + t_vec
                wx, wy, wz = float(p_odom[0]), float(p_odom[1]), float(p_odom[2])

                # Dedup: skip if within dedup_radius of existing detection
                dup = False
                for det in self.locked_detections.values():
                    ddx = det['x'] - wx
                    ddy = det['y'] - wy
                    if ddx * ddx + ddy * ddy < self.dedup_radius * self.dedup_radius:
                        dup = True
                        break
                if dup:
                    continue

                # Lock new detection
                det_id = self.next_det_id
                self.next_det_id += 1
                self.locked_detections[det_id] = {
                    'direction': direction,
                    'x': wx, 'y': wy, 'z': wz,
                    'depth': depth_m
                }
                new_detections = True
                self.get_logger().info(
                    f'Sign #{det_id}: {direction} arrow at ({wx:.2f}, {wy:.2f}) '
                    f'depth={depth_m:.2f}m')

                # Draw on annotated image
                color_map = {
                    'forward': (0, 255, 0),
                    'left': (0, 255, 255),
                    'right': (255, 255, 0),
                    'rotate_180': (255, 0, 255),
                }
                color = color_map.get(direction, (255, 255, 255))
                cv2.drawContours(annotated, [cnt], -1, color, 2)
                cv2.putText(annotated, direction, (centroid_u, centroid_v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Publish markers (always — includes locked detections)
        self._publish_markers(stamp)

        # Publish report if new detections
        if new_detections:
            self._publish_report()

        # Publish annotated image
        self._publish_annotated(annotated, stamp, frame_id)

    # ── HSV arrow filtering (from legacy) ──────────────────────────────

    def _is_arrow_color_mask(self, hsv_img, mask):
        """Check if >arrow_ratio_min of mask pixels are orange (arrow color)."""
        lower = np.array([self.arrow_h_low, self.arrow_s_min, self.arrow_v_min])
        upper = np.array([self.arrow_h_high, 255, 255])
        color_in_hsv = cv2.inRange(hsv_img, lower, upper)
        color_pixels = np.count_nonzero(color_in_hsv & mask)
        mask_pixels = np.count_nonzero(mask)
        if mask_pixels == 0:
            return False
        return (color_pixels / mask_pixels) > self.arrow_ratio_min

    # ── Arrow direction classification (from legacy) ───────────────────

    def _classify_direction_from_mask(self, mask):
        """Classify arrow direction from orange-only binary mask.

        Uses bbox-center vs centroid offset: arrows are asymmetric
        (wide head, narrow tail) so centroid sits toward the tail.
        Tip direction = bbox_center - centroid.
        Returns ('forward', 'left', 'right', 'rotate_180') or (None, None).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.min_mask_area:
            return None, None

        x, y, w, h = cv2.boundingRect(cnt)
        bbox_cx = x + w / 2.0
        bbox_cy = y + h / 2.0

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None, None
        cx_m = M['m10'] / M['m00']
        cy_m = M['m01'] / M['m00']

        # Tip direction = bbox_center - centroid (centroid is toward tail)
        tip_dx = bbox_cx - cx_m
        tip_dy = bbox_cy - cy_m
        offset_mag = np.sqrt(tip_dx ** 2 + tip_dy ** 2)
        if offset_mag < 1.5:
            # Symmetric shape — could be rotate_180 or not an arrow
            hull_pts = cv2.convexHull(cnt, returnPoints=True)
            hull_area = cv2.contourArea(hull_pts)
            convexity = area / hull_area if hull_area > 0 else 1.0
            if convexity < self.curved_max_convexity:
                return 'rotate_180', cnt
            return None, None

        angle = np.degrees(np.arctan2(tip_dy, tip_dx))

        # Camera sees sign face-on: image coords
        if -135 < angle <= -45:
            return 'forward', cnt    # tip up in image = forward
        elif -45 < angle <= 45:
            return 'left', cnt       # tip right in image = sign's left
        elif 45 < angle <= 135:
            return None, None        # tip down = back of sign
        else:
            return 'right', cnt      # tip left in image = sign's right

    # ── Publishing ─────────────────────────────────────────────────────

    def _publish_markers(self, stamp):
        ma = MarkerArray()
        color_map = {
            'forward': (0.0, 1.0, 0.0),
            'left': (1.0, 1.0, 0.0),
            'right': (0.0, 1.0, 1.0),
            'rotate_180': (1.0, 0.0, 1.0),
        }

        for det_id, det in self.locked_detections.items():
            cr, cg, cb = color_map.get(det['direction'], (1.0, 1.0, 1.0))

            m = Marker()
            m.header = Header(stamp=stamp, frame_id='odom')
            m.ns = 'symbols'
            m.id = det_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = det['x']
            m.pose.position.y = det['y']
            m.pose.position.z = det.get('z', 0.0)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.05
            m.color.r = cr
            m.color.g = cg
            m.color.b = cb
            m.color.a = 0.8
            m.lifetime.sec = 0
            ma.markers.append(m)

            # Text label
            mt = Marker()
            mt.header = m.header
            mt.ns = 'symbol_labels'
            mt.id = det_id
            mt.type = Marker.TEXT_VIEW_FACING
            mt.action = Marker.ADD
            mt.pose.position.x = det['x']
            mt.pose.position.y = det['y']
            mt.pose.position.z = det.get('z', 0.0) + 0.3
            mt.pose.orientation.w = 1.0
            mt.scale.z = 0.15
            mt.color.r = 1.0
            mt.color.g = 1.0
            mt.color.b = 1.0
            mt.color.a = 1.0
            mt.text = f"{det['direction']}"
            mt.lifetime.sec = 0
            ma.markers.append(mt)

        self.markers_pub.publish(ma)

    def _publish_report(self):
        symbols = []
        for det in self.locked_detections.values():
            symbols.append({
                'direction': det['direction'],
                'x': round(det['x'], 2),
                'y': round(det['y'], 2)
            })
        report = {'symbols': symbols}
        msg = String()
        msg.data = json.dumps(report)
        self.report_pub.publish(msg)

    def _publish_annotated(self, cv_bgr, stamp, frame_id):
        if self.annotated_pub.get_subscription_count() == 0:
            return
        try:
            img_msg = self.bridge.cv2_to_imgmsg(cv_bgr, 'bgr8')
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = frame_id
            self.annotated_pub.publish(img_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = SymbolDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
