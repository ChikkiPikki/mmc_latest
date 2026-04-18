"""Tag command bridge node.

Subscribes AprilTag detections and emits:
  * /tag_commands  — immediate, once per tag, with mapped command string
  * /logged_tags   — strict ascending order 1..tags_total, deferring out-of-order ids
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty
from apriltag_msgs.msg import AprilTagDetectionArray


# Raw apriltag IDs as printed on the physical/SDF tags are off-by-one from
# the "logical" mission IDs. We remap once here; every downstream consumer
# (logo_detector, mission_manager, CSV logger, RViz plates) only ever
# sees the logical IDs and never has to worry about the mapping.
RAW_TO_LOGICAL_ID = {
    0: 2,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
}


TAG_COMMANDS = {
    1: "left",
    2: "right",
    3: "follow-green",
    4: "u-turn",
    5: "follow-orange",
}


class TagCommandNode(Node):
    def __init__(self):
        super().__init__("tag_command_node")

        self.declare_parameter("tags_total", 5)
        self.tags_total = int(self.get_parameter("tags_total").value)

        self.seen_set: set[int] = set()
        self.next_expected_log_id: int = 1
        self.defer_queue: set[int] = set()

        self.cmd_pub = self.create_publisher(String, "/tag_commands", 10)
        self.log_pub = self.create_publisher(String, "/logged_tags", 10)

        self.create_subscription(
            AprilTagDetectionArray,
            "/apriltag/detections",
            self._on_detections,
            10,
        )
        self.create_subscription(
            Empty,
            "/mission_ended",
            self._on_mission_ended,
            10,
        )

        self.get_logger().info(
            f"tag_command_node up — tags_total={self.tags_total}, "
            f"commands={TAG_COMMANDS}"
        )

    def _now_float(self) -> float:
        t = self.get_clock().now().to_msg()
        return float(t.sec) + float(t.nanosec) * 1e-9

    @staticmethod
    def _extract_id(det) -> int:
        raw = det.id
        if isinstance(raw, int):
            return raw
        try:
            return int(raw[0])
        except (TypeError, IndexError):
            return int(raw)

    def _publish_command(self, tag_id: int) -> None:
        cmd = TAG_COMMANDS.get(tag_id)
        if cmd is None:
            self.get_logger().warn(f"Tag {tag_id} has no command mapping — skipping")
            return
        msg = String()
        msg.data = json.dumps(
            {"tag_id": tag_id, "command": cmd, "stamp": self._now_float()}
        )
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"/tag_commands  tag={tag_id} command={cmd}")

    def _publish_log(self, tag_id: int) -> None:
        msg = String()
        msg.data = json.dumps({"tag_id": tag_id, "stamp": self._now_float()})
        self.log_pub.publish(msg)
        self.get_logger().info(f"/logged_tags   tag={tag_id}")

    def _log_state(self, detected_id: int) -> None:
        self.get_logger().info(
            f"detect={detected_id} seen={sorted(self.seen_set)} "
            f"N={self.next_expected_log_id} defer={sorted(self.defer_queue)}"
        )

    def _on_detections(self, msg: AprilTagDetectionArray) -> None:
        for det in msg.detections:
            try:
                raw_id = self._extract_id(det)
            except Exception as e:
                self.get_logger().warn(f"Could not parse detection id: {e}")
                continue
            logical_id = RAW_TO_LOGICAL_ID.get(raw_id)
            if logical_id is None:
                self.get_logger().warn(
                    f"Raw tag id {raw_id} has no logical mapping — skipping")
                continue
            self._process_tag(logical_id)

    def _process_tag(self, tag_id: int) -> None:
        if tag_id in self.seen_set:
            return

        self.seen_set.add(tag_id)
        self._publish_command(tag_id)

        n = self.next_expected_log_id
        if tag_id == n:
            self._publish_log(tag_id)
            n += 1
            while n in self.defer_queue:
                self._publish_log(n)
                self.defer_queue.remove(n)
                n += 1
            self.next_expected_log_id = n
        elif tag_id > n:
            self.defer_queue.add(tag_id)
        else:
            self.get_logger().warn(
                f"Tag {tag_id} < N={n} but not in seen_set — truth-table bug?"
            )

        self._log_state(tag_id)

    def _on_mission_ended(self, _msg: Empty) -> None:
        self.get_logger().info(
            f"/mission_ended received — flushing defer={sorted(self.defer_queue)} "
            f"from N={self.next_expected_log_id} up to {self.tags_total}"
        )
        n = self.next_expected_log_id
        for tag_id in sorted(self.defer_queue):
            if tag_id < n:
                continue
            while n < tag_id and n <= self.tags_total:
                n += 1
            self._publish_log(tag_id)
            n = tag_id + 1
        self.defer_queue.clear()
        self.next_expected_log_id = n
        self.get_logger().info(
            f"Flush complete — N={self.next_expected_log_id} seen={sorted(self.seen_set)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TagCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
