#!/usr/bin/env python3
"""
generate_dataset.py

Captures a synthetic training dataset of floor symbols by moving the robot
(via ign service set_pose) to many viewpoints around each symbol vertex
in the active building YAML, then saving the robot camera frame plus a
few augmented variants.

Prerequisites (in another terminal):
  ros2 launch mini_r1_v1_gz sim.launch.py

Usage (positives — capture around symbol vertices):
  python3 generate_dataset.py \
      --yaml  <active_world.building.yaml> \
      --output <dataset_dir> \
      [--symbol-id 1] [--world-name generated_world] \
      [--combo-label floor-beige_wall-white]

Usage (negatives — robot in empty map, no symbol visible):
  python3 generate_dataset.py \
      --yaml  <active_world.building.yaml> \
      --output <dataset_dir> \
      --negatives [--negative-samples 40]

Output layout:
  <output>/symbol_<id>/img_00000.png
  <output>/negatives/img_00000.png

Numbering resumes from max existing index + 1, so running this multiple
times (e.g. across texture combos) just extends the dataset.

Notes:
  * No image flipping (symbols are directional).
  * Augmentations: raw + gaussian noise + brightness jitter + slight blur.
  * This script does NOT launch Gazebo — the sim must already be running.
"""
import argparse
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ── Sampling grid ──────────────────────────────────────────────────────
DISTANCES_M = [0.35, 0.55, 0.8, 1.1, 1.5]
YAW_OFFSETS_DEG = [-90, -60, -30, -10, 0, 10, 30, 60, 90]
ROBOT_Z = 0.07
ROBOT_NAME = "mini_r1"

# Approximate horizontal half-FOV of the onboard camera. Slightly padded
# (camera HFOV ≈ 60°, we use ±40°) so a nearby symbol that's only partially
# in the frame still trips the filter.
CAMERA_HFOV_HALF_RAD = math.radians(40.0)
# How much closer than the target another symbol must be to disqualify the
# viewpoint (in metres). Set to 0 to reject *any* closer other symbol; a
# small positive margin tolerates near-ties so we don't throw out too many.
OCCLUSION_REJECT_MARGIN_M = 0.2

# Camera-sits-inside-wall guard. If the viewpoint is closer than this to
# any wall segment, the frame clips through the wall and we get a big
# black region.
VIEWPOINT_WALL_CLEARANCE_M = 0.22

# Post-capture frame sanity check: reject frames where a large fraction of
# pixels are near-black (camera buried in geometry). Dark = max(r,g,b)<15.
DARK_PIXEL_MAX = 15
DARK_FRAME_REJECT_FRAC = 0.18

# ── Augmentation knobs ─────────────────────────────────────────────────
AUG_NOISE_STD = 6
AUG_BRIGHTNESS_JITTER = 20
AUG_BLUR_KSIZE = 3
AUG_COUNT_PER_RAW = 3

# ── Speed knobs ───────────────────────────────────────────────────────
# PNG compression: 0-9. 1 ≈ 3-4× faster encode than default 3, ~15% larger files.
PNG_COMPRESS_LEVEL = 1
# Background encoder thread count. PNG encode is CPU-bound; 4 lets the
# capture loop run ahead of the writer.
SAVE_WORKERS = 4

# ── Negatives sampling ────────────────────────────────────────────────
NEG_WALL_CLEARANCE_M = 0.5
# Keep negatives well away from symbols. Larger than a symbol's visible
# range so a symbol almost never ends up in a negative frame.
NEG_SYMBOL_CLEARANCE_M = 3.5
# Camera half-cone used to reject negative viewpoints looking at a symbol
# (even if the robot is outside the NEG_SYMBOL_CLEARANCE_M ring).
NEG_VIEW_CONE_RANGE_M = 5.0


# ── Geometry helpers ──────────────────────────────────────────────────
def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def compute_scale(vertices, measurements):
    m = measurements[0]
    v1, v2 = vertices[m[0]], vertices[m[1]]
    dist_m = m[2]["distance"][1]
    dist_px = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    return dist_px / dist_m  # px per metre


def px_to_world(px_x, px_y, scale):
    return px_x / scale, -px_y / scale


def dist_point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def load_yaml_geometry(yaml_path):
    """Return (symbols, walls_world, bounds_world, scale).

    symbols: list of (type_id, wx, wy, yaw_rad)
    walls_world: list of (ax, ay, bx, by) in world metres
    bounds_world: (x_min, x_max, y_min, y_max) in world metres
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    level_name = next(iter(data["levels"]))
    level = data["levels"][level_name]
    vertices = level["vertices"]
    walls = level.get("walls", [])
    measurements = level.get("measurements", [])
    if not measurements:
        raise RuntimeError("YAML has no measurements — cannot compute scale.")
    scale = compute_scale(vertices, measurements)

    symbols = []
    for v in vertices:
        if len(v) < 4 or not isinstance(v[3], str):
            continue
        name = v[3]
        if not name.startswith("symbol_"):
            continue
        parts = name.split("_")
        type_id = parts[1]
        try:
            yaw_deg = float(parts[-1])
        except ValueError:
            yaw_deg = 0.0
        wx, wy = px_to_world(v[0], v[1], scale)
        symbols.append((type_id, wx, wy, math.radians(yaw_deg)))

    walls_world = []
    for w in walls:
        a, b = vertices[w[0]], vertices[w[1]]
        ax, ay = px_to_world(a[0], a[1], scale)
        bx, by = px_to_world(b[0], b[1], scale)
        walls_world.append((ax, ay, bx, by))

    xs = [v[0] / scale for v in vertices]
    ys = [-v[1] / scale for v in vertices]
    bounds = (min(xs), max(xs), min(ys), max(ys))
    return symbols, walls_world, bounds, scale


# ── Augmentation ──────────────────────────────────────────────────────
def augment(img):
    """Produce a few mild distortion variants (no flip)."""
    out = [img]

    noise = np.random.normal(0, AUG_NOISE_STD, img.shape).astype(np.int16)
    out.append(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))

    delta = int(np.random.randint(-AUG_BRIGHTNESS_JITTER, AUG_BRIGHTNESS_JITTER + 1))
    out.append(np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8))

    out.append(cv2.blur(img, (AUG_BLUR_KSIZE, AUG_BLUR_KSIZE)))

    return out[: 1 + AUG_COUNT_PER_RAW]


# ── Wall occlusion / dark-frame filters ──────────────────────────────
def _segments_intersect(p1, p2, p3, p4):
    """True if line segment p1-p2 crosses p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)
            and ccw(p1, p2, p3) != ccw(p1, p2, p4))


def _wall_between(rx, ry, sx, sy, walls):
    """True if any wall segment separates the camera from the target."""
    p1, p2 = (rx, ry), (sx, sy)
    for (ax, ay, bx, by) in walls:
        if _segments_intersect(p1, p2, (ax, ay), (bx, by)):
            return True
    return False


def _viewpoint_near_wall(rx, ry, walls, clearance=VIEWPOINT_WALL_CLEARANCE_M):
    for (ax, ay, bx, by) in walls:
        if dist_point_to_segment(rx, ry, ax, ay, bx, by) < clearance:
            return True
    return False


def _frame_mostly_dark(frame):
    """True when too much of the frame is nearly black (camera buried)."""
    if frame is None:
        return True
    mx = frame.max(axis=2) if frame.ndim == 3 else frame
    dark = (mx < DARK_PIXEL_MAX).sum()
    return dark / mx.size > DARK_FRAME_REJECT_FRAC


# ── Occlusion / frame-cleanliness filter ─────────────────────────────
def _other_symbol_in_view(rx, ry, robot_yaw, target_xy, all_symbols, target_dist):
    """Return True if a non-target symbol lies inside the camera cone and
    close enough to compete with (or dominate) the target.
    """
    tx, ty = target_xy
    for (_, sx, sy, _) in all_symbols:
        if abs(sx - tx) < 1e-6 and abs(sy - ty) < 1e-6:
            continue
        dx, dy = sx - rx, sy - ry
        d = math.hypot(dx, dy)
        if d > target_dist + 0.8:
            continue  # behind target, harmless
        bearing = math.atan2(dy, dx)
        ang = (bearing - robot_yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(ang) < CAMERA_HFOV_HALF_RAD and d < target_dist + OCCLUSION_REJECT_MARGIN_M:
            return True
    return False


# ── Filename / resume numbering ──────────────────────────────────────
INDEX_RE = re.compile(r"img_(\d+)")


def next_index(subdir):
    """Highest existing img_<N>.png index in subdir, +1. 0 if empty."""
    if not os.path.isdir(subdir):
        return 0
    mx = -1
    for f in os.listdir(subdir):
        if not f.endswith(".png"):
            continue
        m = INDEX_RE.match(f)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s) if s else ""


def _fmt_dur(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Capture node ──────────────────────────────────────────────────────
class DatasetCaptureNode(Node):
    def __init__(self, output_dir, world_name, combo_label=""):
        super().__init__("dataset_capture")
        self.output_dir = output_dir
        self.world_name = world_name
        self.combo_label = _sanitize(combo_label)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_stamp = 0.0
        self.frame_seq = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.create_subscription(
            Image, "/r1_mini/camera/image_raw", self._image_cb, qos
        )

        # Async PNG encoder/writer pool — offloads ~100-200ms of CPU work
        # per viewpoint from the capture loop. Multiple workers so encode
        # parallelizes across cores.
        self._save_q = queue.Queue(maxsize=256)
        self._save_threads = []
        for _ in range(SAVE_WORKERS):
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self._save_threads.append(t)
        # Per-subdir next-index cache so we don't collide with unflushed
        # writes (queued but not yet on disk).
        self._next_idx = {}

    def _save_worker(self):
        params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESS_LEVEL]
        while True:
            item = self._save_q.get()
            try:
                if item is None:
                    return
                path, img = item
                try:
                    cv2.imwrite(path, img, params)
                except Exception as e:
                    print(f"[save_worker] {path}: {e}", file=sys.stderr)
            finally:
                self._save_q.task_done()

    def flush_saves(self):
        self._save_q.join()

    def _image_cb(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.frame_seq += 1

    def _set_pose(self, x, y, z, yaw):
        qx, qy, qz, qw = yaw_to_quat(yaw)
        req = (
            f'name: "{ROBOT_NAME}", '
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}"
        )
        try:
            subprocess.run(
                ["ign", "service",
                 "-s", f"/world/{self.world_name}/set_pose",
                 "--reqtype", "ignition.msgs.Pose",
                 "--reptype", "ignition.msgs.Boolean",
                 "--timeout", "500",
                 "--req", req],
                capture_output=True, timeout=3.0,
            )
        except subprocess.TimeoutExpired:
            self.get_logger().warn("set_pose timed out")
        # Drain anything in flight pre-teleport, then require N new frames
        # after the service returns before _wait_fresh_frame accepts one.
        # Frame-counter gating is robust against clock jitter and middleware
        # backlog — we know for certain each frame is post-teleport.
        deadline = time.monotonic() + 0.2
        drained = 0
        while time.monotonic() < deadline and drained < 10:
            before = self.frame_seq
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.frame_seq == before:
                break
            drained += 1
        self._require_seq = self.frame_seq + 2

    def _wait_fresh_frame(self, timeout_s=2.0):
        """Block until two new camera frames have arrived after the last
        set_pose call, so we know we're not looking at pre-teleport geometry."""
        deadline = time.monotonic() + timeout_s
        target = getattr(self, "_require_seq", 0)
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.latest_frame is not None and self.frame_seq >= target:
                return True
        return False

    def _save_variants(self, subdir, frame):
        os.makedirs(subdir, exist_ok=True)
        counter = self._next_idx.get(subdir)
        if counter is None:
            counter = next_index(subdir)
        suffix = f"_{self.combo_label}" if self.combo_label else ""
        saved = 0
        for variant in augment(frame):
            path = os.path.join(subdir, f"img_{counter:05d}{suffix}.png")
            self._save_q.put((path, variant))
            counter += 1
            saved += 1
        self._next_idx[subdir] = counter
        return saved

    def capture_symbol(self, type_id, sx, sy, s_yaw, all_symbols=(), walls=(),
                       progress_prefix=""):
        sym_dir = os.path.join(self.output_dir, f"symbol_{type_id}")
        total_vp = len(DISTANCES_M) * len(YAW_OFFSETS_DEG)
        captured = 0
        skipped = 0
        done_vp = 0
        t0 = time.monotonic()

        for dist in DISTANCES_M:
            for yaw_off_deg in YAW_OFFSETS_DEG:
                done_vp += 1
                yaw_off = math.radians(yaw_off_deg)
                approach_heading = s_yaw + math.pi + yaw_off
                rx = sx + dist * math.cos(approach_heading)
                ry = sy + dist * math.sin(approach_heading)
                robot_yaw = approach_heading + math.pi

                reject = False
                if _viewpoint_near_wall(rx, ry, walls):
                    reject = True
                elif _wall_between(rx, ry, sx, sy, walls):
                    reject = True
                elif _other_symbol_in_view(rx, ry, robot_yaw, (sx, sy),
                                           all_symbols, dist):
                    reject = True

                if reject:
                    skipped += 1
                else:
                    self._set_pose(rx, ry, ROBOT_Z, robot_yaw)
                    if self._wait_fresh_frame():
                        if _frame_mostly_dark(self.latest_frame):
                            skipped += 1
                        else:
                            captured += self._save_variants(sym_dir,
                                                            self.latest_frame)

                if done_vp % 10 == 0 or done_vp == total_vp:
                    elapsed = time.monotonic() - t0
                    rate = captured / elapsed if elapsed > 0 else 0.0
                    pct = int(100 * done_vp / total_vp)
                    print(f"    {progress_prefix}viewpoint {done_vp:2d}/{total_vp} "
                          f"({pct:3d}%)  saved={captured:4d}  skipped={skipped:2d}  "
                          f"{rate:.1f} img/s",
                          flush=True)
        return captured

    def capture_negatives(self, walls, bounds, symbols, n_samples):
        """Teleport to random wall-clear, symbol-free positions & headings."""
        neg_dir = os.path.join(self.output_dir, "negatives")
        x_min, x_max, y_min, y_max = bounds
        wall_clear = NEG_WALL_CLEARANCE_M
        sym_clear = NEG_SYMBOL_CLEARANCE_M
        captured = 0
        attempts = 0
        rng = random.Random(0xD175E7)
        target_saves = n_samples * (1 + AUG_COUNT_PER_RAW)
        max_attempts = n_samples * 80

        while captured < target_saves and attempts < max_attempts:
            attempts += 1
            rx = rng.uniform(x_min + wall_clear, x_max - wall_clear)
            ry = rng.uniform(y_min + wall_clear, y_max - wall_clear)

            # 1. Not inside a wall
            near_wall = False
            for (ax, ay, bx, by) in walls:
                if dist_point_to_segment(rx, ry, ax, ay, bx, by) < wall_clear:
                    near_wall = True
                    break
            if near_wall:
                continue

            # 2. Not too close to any symbol
            too_close = False
            for (_, sx, sy, _) in symbols:
                if math.hypot(rx - sx, ry - sy) < sym_clear:
                    too_close = True
                    break
            if too_close:
                continue

            # 3. Pick a heading that does NOT point at any symbol within
            #    NEG_VIEW_CONE_RANGE_M, and that isn't blocked by a wall.
            chosen_yaw = None
            for _ in range(12):
                yaw = rng.uniform(-math.pi, math.pi)
                bad = False
                for (_, sx, sy, _) in symbols:
                    dx, dy = sx - rx, sy - ry
                    d = math.hypot(dx, dy)
                    if d > NEG_VIEW_CONE_RANGE_M:
                        continue
                    bearing = math.atan2(dy, dx)
                    ang = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(ang) < CAMERA_HFOV_HALF_RAD:
                        bad = True
                        break
                if not bad:
                    chosen_yaw = yaw
                    break
            if chosen_yaw is None:
                continue

            self._set_pose(rx, ry, ROBOT_Z, chosen_yaw)
            if not self._wait_fresh_frame():
                continue
            if _frame_mostly_dark(self.latest_frame):
                continue
            captured += self._save_variants(neg_dir, self.latest_frame)

        return captured


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--world-name", default="generated_world")
    ap.add_argument("--symbol-id", default=None,
                    help="Filter positives — single id or comma-separated list")
    ap.add_argument("--combo-label", default="",
                    help="Texture-combo tag, appended to filenames (e.g. floor-beige_wall-default)")
    ap.add_argument("--negatives", action="store_true",
                    help="Capture wall-only images instead of symbol viewpoints")
    ap.add_argument("--negative-samples", type=int, default=40)
    ap.add_argument("--progress-prefix", default="")
    ap.add_argument("--target-images", type=int, default=0,
                    help="Early-exit per symbol type once this many images are on disk (0 = no cap)")
    args = ap.parse_args()

    all_symbols, walls, bounds, _ = load_yaml_geometry(args.yaml)
    if args.symbol_id and not args.negatives:
        wanted_ids = {s.strip() for s in args.symbol_id.split(",") if s.strip()}
        symbols = [s for s in all_symbols if s[0] in wanted_ids]
    else:
        symbols = list(all_symbols)

    if not args.negatives and not symbols:
        print("No symbol vertices found in YAML.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    rclpy.init()
    node = DatasetCaptureNode(args.output, args.world_name, args.combo_label)
    try:
        print("Waiting for first camera frame...")
        t0 = time.monotonic()
        while node.latest_frame is None and time.monotonic() - t0 < 30.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_frame is None:
            print("ERROR: no camera frame in 30 s. Is sim.launch.py running?",
                  file=sys.stderr)
            sys.exit(1)

        total = 0
        run_t0 = time.monotonic()
        if args.negatives:
            n = node.capture_negatives(walls, bounds, all_symbols, args.negative_samples)
            total += n
            print(f"Negatives captured: {n} images")
        else:
            print(f"Found {len(symbols)} symbol instance(s).")
            for i, (type_id, wx, wy, yaw) in enumerate(symbols):
                if args.target_images > 0:
                    sym_dir = os.path.join(args.output, f"symbol_{type_id}")
                    existing = (len([f for f in os.listdir(sym_dir)
                                     if f.endswith('.png')])
                                if os.path.isdir(sym_dir) else 0)
                    if existing >= args.target_images:
                        print(f"[{i+1}/{len(symbols)}] symbol_{type_id}: "
                              f"{existing} ≥ target {args.target_images}, skipping")
                        continue
                prefix = f"{args.progress_prefix}sym {i+1}/{len(symbols)} "
                n = node.capture_symbol(type_id, wx, wy, yaw, all_symbols,
                                        walls=walls, progress_prefix=prefix)
                total += n
                elapsed = time.monotonic() - run_t0
                remaining = len(symbols) - (i + 1)
                avg = elapsed / (i + 1)
                eta = remaining * avg
                print(f"[{i+1}/{len(symbols)}] symbol_{type_id} @ "
                      f"({wx:.2f}, {wy:.2f}) yaw={math.degrees(yaw):.0f} "
                      f"-> {n} images  "
                      f"(elapsed {_fmt_dur(elapsed)}, ETA {_fmt_dur(eta)})")

        print("Flushing pending image writes…")
        node.flush_saves()

        print(f"\n== Dataset summary ({total} new images this run) ==")
        for entry in sorted(os.listdir(args.output)):
            p = os.path.join(args.output, entry)
            if os.path.isdir(p):
                n = len([f for f in os.listdir(p) if f.endswith(".png")])
                print(f"  {entry}: {n} total images")
    finally:
        try:
            node.flush_saves()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
