#!/usr/bin/env python3
"""
inject_obstacles.py  --world <output.world>  --yaml <active.building.yaml>
                     --count N [--seed S]

Insert N random static obstacle models (mixed boxes and cylinders) into
output.world, just before the closing </world> tag. Obstacles are placed
in wall- and symbol-clear positions derived from the active building
YAML.

Each run with a fresh --seed (or no seed) yields a different obstacle
layout, adding visual diversity so the model doesn't overfit to a
symbol-only scene.

Usage:
  python3 inject_obstacles.py --world output.world \
      --yaml active_world.building.yaml --count 6
"""
import argparse
import math
import os
import random
import sys

import yaml


WALL_CLEARANCE_M = 0.6
SYMBOL_CLEARANCE_M = 1.2     # keep obstacles away from symbols so they don't occlude
OBSTACLE_MIN_SPACING_M = 0.8

COLORS = [
    ("0.85 0.2 0.2 1", "0.6 0.1 0.1 1"),     # red
    ("0.2 0.6 0.85 1", "0.1 0.4 0.6 1"),     # blue
    ("0.95 0.75 0.2 1", "0.65 0.5 0.1 1"),   # yellow
    ("0.3 0.75 0.3 1", "0.1 0.5 0.1 1"),     # green
    ("0.55 0.35 0.18 1", "0.35 0.2 0.08 1"), # brown
    ("0.4 0.4 0.4 1", "0.25 0.25 0.25 1"),   # grey
]


def compute_scale(vertices, measurements):
    m = measurements[0]
    v1, v2 = vertices[m[0]], vertices[m[1]]
    dist_m = m[2]["distance"][1]
    dist_px = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
    return dist_px / dist_m


def dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def box_sdf(name, x, y, sx, sy, sz, ambient, diffuse):
    z = sz / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material>
            <ambient>{ambient}</ambient>
            <diffuse>{diffuse}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def cyl_sdf(name, x, y, r, h, ambient, diffuse):
    z = h / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
        <visual name="v">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          <material>
            <ambient>{ambient}</ambient>
            <diffuse>{diffuse}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def load_layout(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    level = data["levels"][next(iter(data["levels"]))]
    vertices = level["vertices"]
    walls_px = level.get("walls", [])
    measurements = level["measurements"]
    scale = compute_scale(vertices, measurements)

    walls_world = []
    for w in walls_px:
        a, b = vertices[w[0]], vertices[w[1]]
        walls_world.append((a[0] / scale, -a[1] / scale,
                            b[0] / scale, -b[1] / scale))

    symbols_world = []
    for v in vertices:
        if len(v) >= 4 and isinstance(v[3], str) and v[3].startswith("symbol_"):
            symbols_world.append((v[0] / scale, -v[1] / scale))

    xs = [v[0] / scale for v in vertices]
    ys = [-v[1] / scale for v in vertices]
    return walls_world, symbols_world, (min(xs), max(xs), min(ys), max(ys))


def strip_existing(text):
    """Remove any obstacles we injected in a previous run (tagged by name prefix)."""
    import re
    return re.sub(
        r"\n\s*<model name=\"obstacle_\d+\">.*?</model>",
        "",
        text,
        flags=re.DOTALL,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    walls, symbols, bounds = load_layout(args.yaml)
    x_min, x_max, y_min, y_max = bounds

    placed = []
    attempts = 0
    while len(placed) < args.count and attempts < args.count * 60:
        attempts += 1
        x = rng.uniform(x_min + WALL_CLEARANCE_M, x_max - WALL_CLEARANCE_M)
        y = rng.uniform(y_min + WALL_CLEARANCE_M, y_max - WALL_CLEARANCE_M)

        too_close = False
        for (ax, ay, bx, by) in walls:
            if dist_to_segment(x, y, ax, ay, bx, by) < WALL_CLEARANCE_M:
                too_close = True; break
        if too_close:
            continue
        for (sx, sy) in symbols:
            if math.hypot(x - sx, y - sy) < SYMBOL_CLEARANCE_M:
                too_close = True; break
        if too_close:
            continue
        for (px, py) in placed:
            if math.hypot(x - px, y - py) < OBSTACLE_MIN_SPACING_M:
                too_close = True; break
        if too_close:
            continue
        placed.append((x, y))

    if not placed:
        print("[inject_obstacles] no valid positions — leaving world unchanged",
              file=sys.stderr)
        return

    snippets = []
    for i, (x, y) in enumerate(placed):
        ambient, diffuse = rng.choice(COLORS)
        if rng.random() < 0.5:
            sx = rng.uniform(0.2, 0.45)
            sy = rng.uniform(0.2, 0.45)
            sz = rng.uniform(0.25, 0.75)
            snippets.append(box_sdf(f"obstacle_{i}", x, y, sx, sy, sz,
                                    ambient, diffuse))
        else:
            r = rng.uniform(0.12, 0.25)
            h = rng.uniform(0.3, 0.9)
            snippets.append(cyl_sdf(f"obstacle_{i}", x, y, r, h, ambient, diffuse))

    with open(args.world) as f:
        text = f.read()
    text = strip_existing(text)

    marker = "</world>"
    idx = text.rfind(marker)
    if idx < 0:
        print("[inject_obstacles] could not find </world>", file=sys.stderr)
        sys.exit(2)

    new_text = text[:idx] + "\n".join(snippets) + "\n  " + text[idx:]
    with open(args.world, "w") as f:
        f.write(new_text)
    print(f"[inject_obstacles] injected {len(placed)} obstacles (seed={args.seed})")


if __name__ == "__main__":
    main()
