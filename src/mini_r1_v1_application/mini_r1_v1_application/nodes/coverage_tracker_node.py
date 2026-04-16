"""Coverage tracker node — maintains visited-cell bitmap in odom frame."""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32, Header


class CoverageTrackerNode(Node):
    def __init__(self):
        super().__init__('coverage_tracker_node')

        # Parameters
        self.declare_parameter('arena_width_m', 10.0)
        self.declare_parameter('arena_height_m', 10.0)
        self.declare_parameter('arena_origin_x', -5.0)
        self.declare_parameter('arena_origin_y', -5.0)
        self.declare_parameter('cell_size_m', 0.15)
        self.declare_parameter('visit_radius_m', 0.20)
        self.declare_parameter('publish_rate_hz', 2.0)

        self.arena_w = self.get_parameter('arena_width_m').value
        self.arena_h = self.get_parameter('arena_height_m').value
        self.origin_x = self.get_parameter('arena_origin_x').value
        self.origin_y = self.get_parameter('arena_origin_y').value
        self.cell_size = self.get_parameter('cell_size_m').value
        self.visit_radius = self.get_parameter('visit_radius_m').value
        publish_rate = self.get_parameter('publish_rate_hz').value

        # Grid dimensions
        self.grid_w = int(self.arena_w / self.cell_size)
        self.grid_h = int(self.arena_h / self.cell_size)
        self.total_cells = self.grid_w * self.grid_h

        # Visited bitmap (0 = unvisited, 100 = visited)
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)

        # Pre-compute visit radius in cells
        self.visit_radius_cells = int(np.ceil(self.visit_radius / self.cell_size))

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, sensor_qos)

        # Publishers
        self.grid_pub = self.create_publisher(
            OccupancyGrid, '/coverage/grid', reliable_transient)
        self.pct_pub = self.create_publisher(
            Float32, '/coverage/percent', 10)

        # Timer
        self.create_timer(1.0 / publish_rate, self._publish_cb)

        self.visited_count = 0
        self.get_logger().info(
            f'Coverage tracker: {self.grid_w}x{self.grid_h} grid, '
            f'cell={self.cell_size}m, visit_r={self.visit_radius}m')

    def _world_to_grid(self, wx: float, wy: float):
        """Convert world coords to grid indices. Returns (row, col) or None if out of bounds."""
        col = int((wx - self.origin_x) / self.cell_size)
        row = int((wy - self.origin_y) / self.cell_size)
        if 0 <= col < self.grid_w and 0 <= row < self.grid_h:
            return row, col
        return None

    def _odom_cb(self, msg: Odometry):
        wx = msg.pose.pose.position.x
        wy = msg.pose.pose.position.y
        center = self._world_to_grid(wx, wy)
        if center is None:
            return

        cr, cc = center
        r = self.visit_radius_cells

        # Mark cells within visit radius
        r_lo = max(0, cr - r)
        r_hi = min(self.grid_h, cr + r + 1)
        c_lo = max(0, cc - r)
        c_hi = min(self.grid_w, cc + r + 1)

        for row in range(r_lo, r_hi):
            for col in range(c_lo, c_hi):
                dr = row - cr
                dc = col - cc
                if dr * dr + dc * dc <= r * r:
                    if self.grid[row, col] == 0:
                        self.grid[row, col] = 100
                        self.visited_count += 1

    def _publish_cb(self):
        now = self.get_clock().now().to_msg()

        # OccupancyGrid
        grid_msg = OccupancyGrid()
        grid_msg.header = Header(stamp=now, frame_id='odom')
        grid_msg.info.resolution = self.cell_size
        grid_msg.info.width = self.grid_w
        grid_msg.info.height = self.grid_h
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        # OccupancyGrid data is row-major, int8[]
        grid_msg.data = self.grid.flatten().tolist()
        self.grid_pub.publish(grid_msg)

        # Coverage percentage
        pct_msg = Float32()
        pct_msg.data = (self.visited_count / self.total_cells * 100.0) if self.total_cells > 0 else 0.0
        self.pct_pub.publish(pct_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
