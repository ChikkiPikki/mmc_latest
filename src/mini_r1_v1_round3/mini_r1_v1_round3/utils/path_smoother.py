"""Bezier corner-rounding path smoother.

Ported from nav_assignment/nav/src/smoothing.cpp (PathSmoother). Rounds sharp
corners with a quadratic Bezier and densifies the remaining segments to a
fixed resolution.
"""
import math


def _round_corners(points, radius):
    if len(points) < 3:
        return list(points)
    rounded = [points[0]]
    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        nxt = points[i + 1]
        dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
        dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]
        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 1e-6 or len2 < 1e-6:
            rounded.append(curr)
            continue
        dx1 /= len1; dy1 /= len1
        dx2 /= len2; dy2 /= len2
        dot = max(-1.0, min(1.0, dx1 * dx2 + dy1 * dy2))
        angle = math.acos(dot)
        if angle > 0.1:
            offset = min(len1 * 0.3, len2 * 0.3, radius)
            p1 = (curr[0] - dx1 * offset, curr[1] - dy1 * offset)
            rounded.append(p1)
            p2 = (curr[0] + dx2 * offset, curr[1] + dy2 * offset)
            for j in range(1, 5):
                t = j / 5.0
                s = 1.0 - t
                bx = s * s * p1[0] + 2 * s * t * curr[0] + t * t * p2[0]
                by = s * s * p1[1] + 2 * s * t * curr[1] + t * t * p2[1]
                rounded.append((bx, by))
            rounded.append(p2)
        else:
            rounded.append(curr)
    rounded.append(points[-1])
    return rounded


def smooth_path(points, resolution=0.08, radius=0.5):
    """Smooth and densify a polyline. Input and output: list of (x,y) tuples."""
    if len(points) < 2:
        return list(points)
    rounded = _round_corners(points, radius)
    out = []
    for i in range(len(rounded) - 1):
        a = rounded[i]
        b = rounded[i + 1]
        d = math.hypot(a[0] - b[0], a[1] - b[1])
        if d < 1e-6:
            continue
        steps = max(1, int(math.ceil(d / resolution)))
        for s in range(steps):
            t = s / steps
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    out.append(rounded[-1])
    return out
