#!/usr/bin/env python3
"""
make_big_world.py  --symbol-id <id>  --out <out.building.yaml>
                  [--size-m 15] [--spacing-m 4] [--wall-clearance-m 2]
                  [--yaw-cycle 0,45,90,135,180,225,270,315]

Synthesize a clean RMF building YAML from scratch — one big square room
with a single symbol type placed on a generous grid. Use this instead of
`build_dataset_world.py` when you want more spacing than any base world
can offer and no interior obstacles interfering with capture viewpoints.

Also writes a blank map.png next to the YAML so the RMF building_map
generator has something to reference.

Usage:
  python3 make_big_world.py --symbol-id 3 \
      --out source/dataset_big_symbol_3/dataset_big_symbol_3.building.yaml
"""
import argparse
import os
import textwrap

import numpy as np
import cv2


SCALE_PX_PER_M = 100  # YAML vertices are pixel coords; this sets them.


def yaml_dump(size_m, vertices, walls, measurement):
    """Hand-roll the YAML so we match the exact formatting RMF expects —
    inline dicts for parameters, one vertex per line, etc.
    """
    px = size_m * SCALE_PX_PER_M
    head = textwrap.dedent(f"""\
        coordinate_system: reference_image
        crowd_sim:
          agent_groups:
            - {{agents_name: [], agents_number: 0, group_id: 0, profile_selector: external_agent, state_selector: external_static, x: 0, y: 0}}
          agent_profiles:
            - {{ORCA_tau: 1, ORCA_tauObst: 0.40000000000000002, class: 1, max_accel: 0, max_angle_vel: 0, max_neighbors: 10, max_speed: 0, name: external_agent, neighbor_dist: 5, obstacle_set: 1, pref_speed: 0, r: 0.25}}
          enable: 0
          goal_sets: []
          model_types: []
          obstacle_set: {{class: 1, file_name: floor_0_navmesh.nav, type: nav_mesh}}
          states:
            - {{final: 1, goal_set: -1, name: external_static, navmesh_file_name: ""}}
          transitions: []
          update_time_step: 0.10000000000000001
        graphs:
          {{}}
        levels:
          floor_0:
            drawing:
              filename: map.png
            elevation: 0
            floors:
              - parameters: {{ceiling_scale: [3, 1], ceiling_texture: [1, blue_linoleum], indoor: [2, 0], texture_name: [1, blue_linoleum], texture_rotation: [3, 0], texture_scale: [3, 1]}}
                vertices: [0, 1, 2, 3]
            layers:
              {{}}
            measurements:
              - [{measurement[0]}, {measurement[1]}, {{distance: [3, {measurement[2]}]}}]
            vertices:
        """)

    vert_lines = []
    for (x, y, tag) in vertices:
        name = f'"{tag}"' if tag else '""'
        vert_lines.append(f"      - [{x}, {y}, 0, {name}]")

    wall_params = ("{alpha: [3, 1], texture_height: [3, 2.5], "
                   "texture_name: [1, default], texture_scale: [3, 1], "
                   "texture_width: [3, 1]}")
    wall_lines = []
    for (a, b) in walls:
        wall_lines.append(f"      - [{a}, {b}, {wall_params}]")

    tail = f"lifts: {{}}\nname: big_world\n"
    return (head
            + "\n".join(vert_lines) + "\n"
            + "    walls:\n"
            + "\n".join(wall_lines) + "\n"
            + tail)


def compute_grid(size_m, spacing_m, wall_clearance_m):
    inner = size_m - 2 * wall_clearance_m
    if inner <= 0:
        raise ValueError("size-m too small for wall-clearance")
    n = int(inner // spacing_m) + 1
    if n < 1:
        raise ValueError("spacing-m too large for room size")
    # Centre the grid inside the usable area
    used = (n - 1) * spacing_m
    start = wall_clearance_m + (inner - used) / 2.0
    return [start + i * spacing_m for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol-id", required=True,
                    help="single id or comma list (each gets its own world "
                         "only if --per-symbol worlds are made externally)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-m", type=float, default=10.0)
    ap.add_argument("--count", type=int, default=1,
                    help="number of symbol instances (default 1 — one "
                         "symbol per world so the camera never sees two at "
                         "once). >1 uses a spaced grid.")
    ap.add_argument("--spacing-m", type=float, default=4.0,
                    help="distance between grid positions when count>1")
    ap.add_argument("--wall-clearance-m", type=float, default=2.0)
    ap.add_argument("--yaw-cycle", default="0,45,90,135,180,225,270,315")
    args = ap.parse_args()

    yaws = [int(y.strip()) for y in args.yaw_cycle.split(",") if y.strip()]

    size_px = args.size_m * SCALE_PX_PER_M

    # 4 corner vertices (image-coord, Y down)
    corners = [
        (0.0,        0.0,        ""),
        (size_px,    0.0,        ""),
        (size_px,    size_px,    ""),
        (0.0,        size_px,    ""),
    ]

    # 1 measurement — bottom edge is `size_m` metres.
    measurement = (0, 1, args.size_m)

    # Robot spawn vertex — offset from corner so sim.launch.py can parse it
    # and the robot doesn't land in the wall. Single symbol goes at centre;
    # spawn goes at (1 m, 1 m) so the robot starts 1 m from both walls.
    spawn_px = 1.0 * SCALE_PX_PER_M
    extra_vertices = [(spawn_px, spawn_px, "spawn_0")]

    symbol_vertices = []
    if args.count <= 1:
        # One symbol, dead centre. Cleanest option — camera can never see two.
        xm = args.size_m / 2.0
        ym = args.size_m / 2.0
        yaw = yaws[0]
        tag = f"symbol_{args.symbol_id}_{yaw}"
        symbol_vertices.append((xm * SCALE_PX_PER_M,
                                ym * SCALE_PX_PER_M, tag))
    else:
        xs_m = compute_grid(args.size_m, args.spacing_m, args.wall_clearance_m)
        ys_m = compute_grid(args.size_m, args.spacing_m, args.wall_clearance_m)
        idx = 0
        for ym in ys_m:
            for xm in xs_m:
                if idx >= args.count:
                    break
                yaw = yaws[idx % len(yaws)]
                tag = f"symbol_{args.symbol_id}_{yaw}"
                symbol_vertices.append((xm * SCALE_PX_PER_M,
                                        ym * SCALE_PX_PER_M, tag))
                idx += 1
            if idx >= args.count:
                break

    vertices = corners + extra_vertices + symbol_vertices
    walls = [(0, 1), (1, 2), (2, 3), (3, 0)]

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(yaml_dump(args.size_m, vertices, walls, measurement))

    # Blank map.png — required by RMF generator. Keep small; it's just a
    # reference image for the traffic editor, not used at sim time.
    map_path = os.path.join(os.path.dirname(out_path), "map.png")
    img = np.full((int(size_px), int(size_px), 3), 255, dtype=np.uint8)
    cv2.imwrite(map_path, img)

    print(f"[make_big_world] room {args.size_m}x{args.size_m} m  "
          f"({len(symbol_vertices)} symbol instances, spacing "
          f"{args.spacing_m} m)")
    print(f"  wrote {out_path}")
    print(f"  wrote {map_path}")


if __name__ == "__main__":
    main()
