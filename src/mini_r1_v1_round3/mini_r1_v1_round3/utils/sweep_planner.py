"""Pure-geometry sweep planning utilities. No ROS dependencies."""
import os
import re
from typing import Optional


def parse_grid_tiles_from_sdf(world_file_path: str) -> list[tuple[int, int, float, float]]:
    """Scan an SDF world for `tile_rR_cC` models and return [(r, c, x, y)].

    Expects blocks like:
        <model name="tile_r2_c3">
          ...
          <pose>1.350 0.000 0 0 0 0</pose>
    """
    if not world_file_path or not os.path.isfile(world_file_path):
        return []
    try:
        with open(world_file_path, "r") as f:
            text = f.read()
    except OSError:
        return []

    pattern = re.compile(
        r'<model\s+name="tile_r(\d+)_c(\d+)">.*?<pose>\s*([-\d.eE]+)\s+([-\d.eE]+)',
        re.DOTALL,
    )
    out: list[tuple[int, int, float, float]] = []
    for m in pattern.finditer(text):
        r = int(m.group(1)); c = int(m.group(2))
        x = float(m.group(3)); y = float(m.group(4))
        out.append((r, c, x, y))
    return out


def waypoints_from_sdf_grid(world_file_path: str) -> tuple[list[tuple[float, float]], float]:
    """Return (waypoints, cell_size) visiting every tile center in snake order.

    Infers cell_size from inter-column spacing. Returns ([], 0.0) on failure.
    """
    tiles = parse_grid_tiles_from_sdf(world_file_path)
    if not tiles:
        return [], 0.0

    by_row: dict[int, list[tuple[int, float, float]]] = {}
    for (r, c, x, y) in tiles:
        by_row.setdefault(r, []).append((c, x, y))

    xs_sorted = sorted({x for _, _, x, _ in tiles})
    ys_sorted = sorted({y for _, _, _, y in tiles})
    cell_size = 0.9
    if len(xs_sorted) >= 2:
        cell_size = abs(xs_sorted[1] - xs_sorted[0])
    elif len(ys_sorted) >= 2:
        cell_size = abs(ys_sorted[1] - ys_sorted[0])

    waypoints: list[tuple[float, float]] = []
    for j, r in enumerate(sorted(by_row.keys(), reverse=True)):
        row_cells = sorted(by_row[r], key=lambda t: t[0])
        if j % 2 == 1:
            row_cells = list(reversed(row_cells))
        for (_c, x, y) in row_cells:
            waypoints.append((x, y))
    return waypoints, cell_size


def boustrophedon_waypoints(
    arena_min_x: float, arena_min_y: float,
    arena_max_x: float, arena_max_y: float,
    stride_m: float = 0.5,
    margin_m: float = 0.35,
) -> list[tuple[float, float]]:
    """Return (x, y) waypoints in odom frame covering the arena in a snake pattern.

    - margin_m keeps waypoints away from walls (robot footprint ~0.3m, inflation ~0.25m).
    - stride_m is row spacing; default 0.5 means 1m tile has 2 rows through it.
    - Start at (arena_min_x+margin, arena_min_y+margin), snake back and forth.
    """
    x_lo = arena_min_x + margin_m
    x_hi = arena_max_x - margin_m
    y_lo = arena_min_y + margin_m
    y_hi = arena_max_y - margin_m
    if x_hi <= x_lo or y_hi <= y_lo or stride_m <= 0.0:
        return []

    waypoints: list[tuple[float, float]] = []
    y = y_lo
    row = 0
    while y <= y_hi + 1e-9:
        yc = min(y, y_hi)
        if row % 2 == 0:
            waypoints.append((x_lo, yc))
            waypoints.append((x_hi, yc))
        else:
            waypoints.append((x_hi, yc))
            waypoints.append((x_lo, yc))
        row += 1
        y += stride_m
    return waypoints


def grid_cell_waypoints(
    arena_min_x: float, arena_min_y: float,
    arena_max_x: float, arena_max_y: float,
    cell_size_m: float = 0.9,
    margin_m: float = 0.0,
) -> list[tuple[float, float]]:
    """Centers of each grid cell in arena, visited in snake order.

    Each cell_size_m x cell_size_m cell yields one waypoint at its center.
    Rows alternate direction (boustrophedon over cells).
    """
    x_lo = arena_min_x + margin_m
    x_hi = arena_max_x - margin_m
    y_lo = arena_min_y + margin_m
    y_hi = arena_max_y - margin_m
    if x_hi <= x_lo or y_hi <= y_lo or cell_size_m <= 0.0:
        return []

    nx = max(1, int(round((x_hi - x_lo) / cell_size_m)))
    ny = max(1, int(round((y_hi - y_lo) / cell_size_m)))
    dx = (x_hi - x_lo) / nx
    dy = (y_hi - y_lo) / ny

    waypoints: list[tuple[float, float]] = []
    for j in range(ny):
        ys = y_lo + (j + 0.5) * dy
        cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
        for i in cols:
            xs = x_lo + (i + 0.5) * dx
            waypoints.append((xs, ys))
    return waypoints


def nearest_remaining(current_xy: tuple[float, float],
                      waypoints: list[tuple[float, float]],
                      visited_indices: set[int]) -> Optional[int]:
    """Index of closest unvisited waypoint, or None if all visited. Used to resume
    sweep after a tag-command detour."""
    best_idx: Optional[int] = None
    best_d2 = float("inf")
    cx, cy = current_xy
    for i, (wx, wy) in enumerate(waypoints):
        if i in visited_indices:
            continue
        dx = wx - cx
        dy = wy - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx
