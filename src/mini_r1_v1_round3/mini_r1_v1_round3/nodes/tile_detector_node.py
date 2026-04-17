import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int32MultiArray, String
import tf2_ros
from mini_r1_v1_round3.utils.palette import load_palette, inrange_wrapped
from mini_r1_v1_round3.utils.deprojection import deproject_pixel, transform_point_to_odom
import json
import os


COLOR_RGB = {
    'green': (0.0, 1.0, 0.0),
    'orange': (1.0, 0.5, 0.0),
    'red': (1.0, 0.0, 0.0),
}


class TileDetectorNode(Node):
    def __init__(self):
        super().__init__('tile_detector_node')

        self.declare_parameter('camera_frame', 'mini_r1/base_link/camera')
        self.declare_parameter('min_blob_area', 300)
        self.declare_parameter('dedup_radius_m', 0.3)
        self.declare_parameter('confirmation_count', 3)
        self.declare_parameter('visited_radius_m', 0.4)
        self.declare_parameter('spawn_ignore_radius_m', 0.5)
        self.declare_parameter('spawn_ignore_duration_s', 10.0)
        self.declare_parameter('textures_dir', '')

        self.camera_frame = self.get_parameter('camera_frame').value
        self.min_blob_area = int(self.get_parameter('min_blob_area').value)
        self.dedup_radius_m = float(self.get_parameter('dedup_radius_m').value)
        self.confirmation_count = int(self.get_parameter('confirmation_count').value)
        self.visited_radius_m = float(self.get_parameter('visited_radius_m').value)
        self.spawn_ignore_radius_m = float(self.get_parameter('spawn_ignore_radius_m').value)
        self.spawn_ignore_duration_s = float(self.get_parameter('spawn_ignore_duration_s').value)
        textures_dir = self.get_parameter('textures_dir').value

        if not textures_dir:
            raise RuntimeError('textures_dir parameter is required')

        self.palettes = {
            'green': load_palette(os.path.join(textures_dir, 'green.png'), 'green'),
            'orange': load_palette(os.path.join(textures_dir, 'orange.png'), 'orange'),
            'red': load_palette(os.path.join(textures_dir, 'stop.png'), 'red'),
        }
        for name, (lo, hi) in self.palettes.items():
            self.get_logger().info(f'Palette {name}: HSV low={lo.tolist()} high={hi.tolist()}')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tiles = []
        self.next_id = 0
        self.visited_ids = set()
        self.spawn_pose = None
        self.start_time = self.get_clock().now()
        self.stop_zone_published = False

        self.pose_pub = self.create_publisher(PoseArray, '/detected_tiles', 10)
        self.meta_pub = self.create_publisher(String, '/tile_metadata', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tile_markers', 10)
        self.stop_pub = self.create_publisher(PoseStamped, '/stop_zone', 1)
        self.visited_pub = self.create_publisher(Int32MultiArray, '/visited_tiles', 10)

        rgb_sub = message_filters.Subscriber(self, Image, '/r1_mini/camera/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/r1_mini/camera/depth_image')
        info_sub = message_filters.Subscriber(self, CameraInfo, '/r1_mini/camera/camera_info')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.on_frame)
        self._frame_count = 0

        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 10)
        self.create_subscription(Odometry, '/r1_mini/odom', self.on_odom, 10)

        self.create_timer(0.5, self.on_timer)

        self.get_logger().info('TileDetectorNode ready')

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        if self.spawn_pose is None:
            self.spawn_pose = np.array([p.x, p.y])
            self.get_logger().info(f'Spawn pose set to ({p.x:.2f}, {p.y:.2f})')
        rx, ry = p.x, p.y
        for t in self.tiles:
            if not t['confirmed']:
                continue
            d = np.hypot(rx - t['x'], ry - t['y'])
            if d <= self.visited_radius_m:
                self.visited_ids.add(t['id'])

    def on_frame(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')
            return

        k = info_msg.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        try:
            self.tf_buffer.lookup_transform('odom', self.camera_frame, rgb_msg.header.stamp,
                                            timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        h_img, w_img = depth.shape[:2]

        for color, (lo, hi) in self.palettes.items():
            mask = inrange_wrapped(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.min_blob_area:
                    continue
                u, v = centroids[i]
                iu, iv = int(u), int(v)
                if iv < 0 or iv >= h_img or iu < 0 or iu >= w_img:
                    continue
                d = float(depth[iv, iu])
                if not np.isfinite(d) or d <= 0.0:
                    continue
                p_cam = deproject_pixel(u, v, d, fx, fy, cx, cy)
                p_odom = transform_point_to_odom(p_cam, self.tf_buffer, self.camera_frame,
                                                 rgb_msg.header.stamp)
                if p_odom is None:
                    continue

                if color == 'green' and self.spawn_pose is not None:
                    elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
                    if elapsed < self.spawn_ignore_duration_s:
                        d_spawn = np.hypot(p_odom[0] - self.spawn_pose[0],
                                           p_odom[1] - self.spawn_pose[1])
                        if d_spawn < self.spawn_ignore_radius_m:
                            continue

                self._add_or_merge(color, float(p_odom[0]), float(p_odom[1]))

        for t in self.tiles:
            if not t['confirmed'] and t['obs_count'] >= self.confirmation_count:
                t['confirmed'] = True
                self.get_logger().info(
                    f"Confirmed tile T{t['id']} {t['color']} @ ({t['x']:.2f},{t['y']:.2f})")

    def _add_or_merge(self, color, x, y):
        for t in self.tiles:
            if t['color'] != color:
                continue
            if np.hypot(t['x'] - x, t['y'] - y) <= self.dedup_radius_m:
                n = t['obs_count']
                t['x'] = (t['x'] * n + x) / (n + 1)
                t['y'] = (t['y'] * n + y) / (n + 1)
                t['obs_count'] = n + 1
                return
        self.tiles.append({
            'id': self.next_id,
            'color': color,
            'x': x,
            'y': y,
            'obs_count': 1,
            'confirmed': False,
        })
        self.next_id += 1

    def on_timer(self):
        now = self.get_clock().now().to_msg()
        confirmed = [t for t in self.tiles if t['confirmed']]

        pa = PoseArray()
        pa.header.stamp = now
        pa.header.frame_id = 'odom'
        meta = []
        for t in confirmed:
            p = Pose()
            p.position.x = t['x']
            p.position.y = t['y']
            p.position.z = 0.0
            p.orientation.w = 1.0
            pa.poses.append(p)
            meta.append({'id': t['id'], 'color': t['color'], 'x': t['x'], 'y': t['y']})
        self.pose_pub.publish(pa)

        meta_msg = String()
        meta_msg.data = json.dumps(meta)
        self.meta_pub.publish(meta_msg)

        ma = MarkerArray()
        for t in confirmed:
            cube = Marker()
            cube.header.frame_id = 'odom'
            cube.header.stamp = now
            cube.ns = 'tile_cubes'
            cube.id = t['id']
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position.x = t['x']
            cube.pose.position.y = t['y']
            cube.pose.position.z = 0.025
            cube.pose.orientation.w = 1.0
            cube.scale.x = 0.3
            cube.scale.y = 0.3
            cube.scale.z = 0.05
            r, g, b = COLOR_RGB[t['color']]
            cube.color.r = r
            cube.color.g = g
            cube.color.b = b
            cube.color.a = 0.7
            ma.markers.append(cube)

            txt = Marker()
            txt.header.frame_id = 'odom'
            txt.header.stamp = now
            txt.ns = 'tile_labels'
            txt.id = t['id']
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = t['x']
            txt.pose.position.y = t['y']
            txt.pose.position.z = 0.3
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.15
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 1.0
            txt.text = f"T{t['id']}:{t['color'][0]}"
            ma.markers.append(txt)
        self.marker_pub.publish(ma)

        vis = Int32MultiArray()
        vis.data = sorted(self.visited_ids)
        self.visited_pub.publish(vis)

        if not self.stop_zone_published:
            for t in confirmed:
                if t['color'] == 'red':
                    ps = PoseStamped()
                    ps.header.stamp = now
                    ps.header.frame_id = 'odom'
                    ps.pose.position.x = t['x']
                    ps.pose.position.y = t['y']
                    ps.pose.position.z = 0.0
                    ps.pose.orientation.w = 1.0
                    self.stop_pub.publish(ps)
                    self.stop_zone_published = True
                    self.get_logger().info(
                        f"Published /stop_zone at ({t['x']:.2f},{t['y']:.2f})")
                    break


def main(args=None):
    rclpy.init(args=args)
    node = TileDetectorNode()
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
