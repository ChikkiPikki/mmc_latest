#!/usr/bin/env python3
"""
build_dataset_world.py

Generate a dataset-specific building YAML by copying a base world and
appending `symbol_<id>_<yaw>` vertex entries on a grid, skipping
positions too close to wall segments.

The base YAML is copied **byte-for-byte** and new vertex lines are
inserted into the vertices list via text substitution, so the file
format remains identical to what the RMF building_map_generator expects.

Usage:
  python3 build_dataset_world.py \
      --base <source/simple/simple.building.yaml> \
      --symbol-id 1 \
      --out  <source/dataset_symbol_1/dataset_symbol_1.building.yaml>
"""
import argparse
import math
import os
import re
import shutil
import yaml


GRID_STEP_M = 4.0
WALL_CLEARANCE_M = 1.3
SYMBOL_BORDER_M = 0.4
YAW_CYCLE_DEG = [0, 90, 180, 270, 45, 135, 225, 315]


def compute_scale(vertices, measurements):
    m = measurements[0]
    v1, v2 = vertices[m[0]], vertices[m[1]]
    dist_m = m[2]["distance"][1]
    dist_px = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    return dist_px / dist_m  # px per metre


def dist_point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def compute_candidates(vertices, walls, scale, grid_step_m, clearance_m):
    clearance_px = clearance_m * scale
    step_px = grid_step_m * scale
    margin_px = (SYMBOL_BORDER_M + 0.1) * scale

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    wall_segs = []
    for w in walls:
        a, b = vertices[w[0]], vertices[w[1]]
        wall_segs.append((a[0], a[1], b[0], b[1]))

    candidates = []
    y = y_min + margin_px
    while y <= y_max - margin_px:
        x = x_min + margin_px
        while x <= x_max - margin_px:
            ok = True
            for (ax, ay, bx, by) in wall_segs:
                if dist_point_to_segment(x, y, ax, ay, bx, by) < clearance_px:
                    ok = False
                    break
            if ok:
                candidates.append((x, y))
            x += step_px
        y += step_px
    return candidates


def inject_vertices(text, new_lines):
    """Insert `new_lines` (each a pre-formatted YAML vertex line with 6-space
    indent) at the end of the top-level vertices list under levels/floor_*/.

    Strategy: find `    vertices:\n`, then find the first line whose indent
    is less than 6 spaces and precede it with our new lines.
    """
    # Match "    vertices:" followed by list items indented with 6 spaces
    m = re.search(r"(?m)^    vertices:\s*\n", text)
    if not m:
        raise RuntimeError("Could not find 'vertices:' section in base YAML")
    start = m.end()

    # Walk forward until we hit a line with indent < 6 (end of the list)
    i = start
    while i < len(text):
        line_end = text.find("\n", i)
        if line_end == -1:
            line_end = len(text)
        line = text[i:line_end]
        if line.strip() == "":
            i = line_end + 1
            continue
        # Count leading spaces
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent < 6:
            break  # end of vertices list
        i = line_end + 1

    insertion = "".join(ln.rstrip() + "\n" for ln in new_lines)
    return text[:i] + insertion + text[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base YAML to extend")
    ap.add_argument("--symbol-id", default="1",
                    help="Single id or comma-separated list, e.g. '1,2,3'")
    ap.add_argument("--out", required=True, help="Output YAML path")
    ap.add_argument("--grid-step", type=float, default=GRID_STEP_M)
    ap.add_argument("--clearance", type=float, default=WALL_CLEARANCE_M)
    ap.add_argument("--per-id", type=int, default=1,
                    help="Number of instances per symbol id (default 1)")
    args = ap.parse_args()

    symbol_ids = [s.strip() for s in args.symbol_id.split(",") if s.strip()]
    want_total = args.per_id * len(symbol_ids)

    with open(args.base) as f:
        base_text = f.read()
    data = yaml.safe_load(base_text)

    level_name = next(iter(data["levels"]))
    level = data["levels"][level_name]
    vertices = level["vertices"]
    walls = level.get("walls", [])
    measurements = level["measurements"]
    scale = compute_scale(vertices, measurements)

    candidates = compute_candidates(
        vertices, walls, scale, args.grid_step, args.clearance)

    print(f"Base '{args.base}': {len(vertices)} vertices, {len(walls)} walls, "
          f"scale={scale:.2f} px/m")
    print(f"Candidate positions after wall-clearance: {len(candidates)}")
    print(f"Placing {want_total} instance(s) total ({args.per_id} per id × {len(symbol_ids)} ids)")

    if want_total > len(candidates):
        raise RuntimeError(
            f"Not enough clear positions ({len(candidates)}) to place {want_total} "
            f"instances. Reduce --per-id or --clearance, or use a larger base world.")

    # Pick positions spread across the candidate set (not clustered)
    step = max(1, len(candidates) // want_total)
    chosen = [candidates[(i * step) % len(candidates)] for i in range(want_total)]

    new_lines = []
    for i, (px, py) in enumerate(chosen):
        sid = symbol_ids[i % len(symbol_ids)]
        yaw = YAW_CYCLE_DEG[i % len(YAW_CYCLE_DEG)]
        name = f"symbol_{sid}_{yaw}"
        new_lines.append(f"      - [{px}, {py}, 0, {name}]")

    new_text = inject_vertices(base_text, new_lines)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(new_text)

    # Copy map image alongside
    base_dir = os.path.dirname(args.base)
    out_dir = os.path.dirname(args.out)
    for ext in ("png", "jpeg", "jpg"):
        src = os.path.join(base_dir, f"map.{ext}")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, f"map.{ext}"))
            break

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
