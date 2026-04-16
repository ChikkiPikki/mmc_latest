"""ArUco localizer node — detects markers, publishes poses for EKF correction."""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
import tf2_ros
from scipy.spatial.transform import Rotation as ScipyRotation

from mini_r1_v1_application.utils.deprojection import (
    depth_to_meters, deproject_pixel, depth_patch_median,
    tf_to_rotation_matrix, R_OPTICAL_TO_BODY)


class ArucoLocalizerNode(Node):
    def __init__(self):
        super().__init__('aruco_localizer_node')

        # Parameters
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('marker_length_m', 0.4)
        self.declare_parameter('max_detection_dist_m', 3.5)
        self.declare_parameter('min_marker_area_px', 200)
        self.declare_parameter('known_markers_file', '')
        self.declare_parameter('process_every_n', 3)
        self.declare_parameter('sync_slop_s', 0.1)
        self.declare_parameter('min_depth', 0.12)
        self.declare_parameter('max_depth', 10.0)

        self.marker_length = self.get_parameter('marker_length_m').value
        self.max_det_dist = self.get_parameter('max_detection_dist_m').value
        self.min_area = self.get_parameter('min_marker_area_px').value
        self.known_markers_file = self.get_parameter('known_markers_file').value
        self.process_every_n = self.get_parameter('process_every_n').value
        self.sync_slop = self.get_parameter('sync_slop_s').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value

        self.bridge = CvBridge()
        self.frame_count = 0
        self.depth_encoding_logged = False

        # ArUco setup (tuned params from legacy marker_detector_node)
        dict_name = self.get_parameter('aruco_dict').value
        aruco_dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.errorCorrectionRate = 0.6

        # OpenCV 4.8+ uses ArucoDetector class
        self._use_new_api = hasattr(cv2.aruco, 'ArucoDetector')
        if self._use_new_api:
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.aruco_params)

        # solvePnP object points for marker
        half = self.marker_length / 2.0
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        # Locked markers (already detected)
        self.locked_markers = {}  # marker_id -> {'x', 'y', 'z', 'q'}

        # Known marker positions (for EKF pose correction)
        self.known_markers = {}
        if self.known_markers_file:
            self._load_known_markers()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5)

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
        self.markers_pub = self.create_publisher(MarkerArray, '/aruco/markers', 10)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/aruco/pose', 10)

        self.get_logger().info(
            f'ArUco localizer: dict={dict_name}, marker_len={self.marker_length}m, '
            f'new_api={self._use_new_api}')

    def _load_known_markers(self):
        """Load known marker positions from YAML file for EKF correction."""
        try:
            import yaml
            with open(self.known_markers_file, 'r') as f:
                data = yaml.safe_load(f)
            for entry in data.get('markers', []):
                mid = entry['id']
                self.known_markers[mid] = {
                    'x': entry['x'], 'y': entry['y'], 'z': entry.get('z', 0.0),
                    'yaw': entry.get('yaw', 0.0)
                }
            self.get_logger().info(
                f'Loaded {len(self.known_markers)} known markers from {self.known_markers_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to load known markers: {e}')

    def _sync_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        if not self.depth_encoding_logged:
            self.get_logger().info(f'Depth encoding: {depth_msg.encoding}')
            self.depth_encoding_logged = True

        # Convert images
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            cv_gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
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

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

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

        q_tf = trans.transform.rotation
        t_tf = trans.transform.translation
        rot = tf_to_rotation_matrix(q_tf.x, q_tf.y, q_tf.z, q_tf.w)
        t_vec = np.array([t_tf.x, t_tf.y, t_tf.z])

        # Detect ArUco markers
        if self._use_new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(cv_gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                cv_gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        for i, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])

            # Skip already-locked markers
            if marker_id in self.locked_markers:
                # Still publish visualization for locked markers
                det = self.locked_markers[marker_id]
                marker_array.markers.extend(
                    self._make_viz_markers(marker_id, det, stamp))
                continue

            c = corners[i][0]
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            # solvePnP for marker pose
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points, c, camera_matrix, dist_coeffs)
            if not success:
                continue

            # Depth at marker centroid
            center_u = int(c[:, 0].mean())
            center_v = int(c[:, 1].mean())

            depth_m = depth_patch_median(
                cv_depth, center_u, center_v,
                patch_radius=3, min_depth=self.min_depth, max_depth=self.max_depth)
            if np.isnan(depth_m) or depth_m > self.max_det_dist:
                continue

            # Deproject to camera frame, then to odom
            p_cam = deproject_pixel(center_u, center_v, depth_m, fx, fy, cx, cy)

            # Marker orientation in odom frame
            rmat, _ = cv2.Rodrigues(rvec.flatten())
            R_marker2cam = R_OPTICAL_TO_BODY @ rmat
            R_marker2odom = rot @ R_marker2cam
            q_marker = ScipyRotation.from_matrix(R_marker2odom).as_quat()  # [x,y,z,w]

            p_odom = rot @ p_cam + t_vec
            wx, wy, wz = float(p_odom[0]), float(p_odom[1]), float(p_odom[2])

            # Lock marker
            self.locked_markers[marker_id] = {
                'x': wx, 'y': wy, 'z': wz,
                'q': q_marker, 'depth': depth_m
            }
            self.get_logger().info(
                f'ArUco #{marker_id} LOCKED at ({wx:.2f}, {wy:.2f}, {wz:.2f}) '
                f'depth={depth_m:.2f}m')

            # Visualization
            marker_array.markers.extend(
                self._make_viz_markers(marker_id, self.locked_markers[marker_id], stamp))

            # Publish pose for EKF if known markers file is loaded
            if self.known_markers:
                self._publish_pose_correction(marker_id, wx, wy, wz, depth_m, stamp)

        if marker_array.markers:
            self.markers_pub.publish(marker_array)

    def _publish_pose_correction(self, marker_id, observed_x, observed_y, observed_z,
                                 depth_m, stamp):
        """Publish robot pose correction based on known marker position vs observed."""
        if marker_id not in self.known_markers:
            return

        # This is a simplified correction — in production you'd compute
        # the full robot pose from the marker observation + known position.
        # For now, publish the observation as a pose for the EKF.
        msg = PoseWithCovarianceStamped()
        msg.header = Header(stamp=stamp, frame_id='odom')
        msg.pose.pose.position.x = observed_x
        msg.pose.pose.position.y = observed_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        # Covariance scales with distance (closer = tighter)
        base_cov = 0.1
        dist_factor = max(1.0, depth_m / 2.0)
        cov = base_cov * dist_factor

        # 6x6 covariance (x, y, z, roll, pitch, yaw)
        msg.pose.covariance[0] = cov       # x
        msg.pose.covariance[7] = cov       # y
        msg.pose.covariance[14] = 999.0    # z (ignore)
        msg.pose.covariance[21] = 999.0    # roll (ignore)
        msg.pose.covariance[28] = 999.0    # pitch (ignore)
        msg.pose.covariance[35] = cov * 2  # yaw

        self.pose_pub.publish(msg)

    def _make_viz_markers(self, marker_id, det, stamp):
        """Create visualization markers for a detected ArUco."""
        markers = []

        # Cube marker
        m = Marker()
        m.header = Header(stamp=stamp, frame_id='odom')
        m.ns = 'aruco'
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = det['x']
        m.pose.position.y = det['y']
        m.pose.position.z = det['z']
        m.pose.orientation.w = 1.0
        m.scale.x = self.marker_length
        m.scale.y = self.marker_length
        m.scale.z = 0.02
        # Color by marker ID
        np.random.seed(marker_id)
        m.color.r, m.color.g, m.color.b = np.random.rand(3).astype(float)
        m.color.a = 0.8
        m.lifetime.sec = 0
        markers.append(m)

        # Text label
        mt = Marker()
        mt.header = m.header
        mt.ns = 'aruco_text'
        mt.id = marker_id
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = Marker.ADD
        mt.pose.position.x = det['x']
        mt.pose.position.y = det['y']
        mt.pose.position.z = det['z'] + 0.4
        mt.pose.orientation.w = 1.0
        mt.scale.z = 0.15
        mt.color.r = 1.0
        mt.color.g = 1.0
        mt.color.b = 1.0
        mt.color.a = 1.0
        mt.text = f'ArUco #{marker_id}'
        mt.lifetime.sec = 0
        markers.append(mt)

        return markers


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
