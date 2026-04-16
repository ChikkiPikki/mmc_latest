"""Shared depth deprojection and encoding utilities."""

import numpy as np


def depth_to_meters(depth_img: np.ndarray, encoding: str) -> np.ndarray:
    """Convert depth image to float32 meters based on encoding."""
    if encoding == '32FC1':
        return depth_img.astype(np.float32)
    elif encoding == '16UC1':
        return depth_img.astype(np.float32) / 1000.0
    else:
        raise ValueError(f"Unexpected depth encoding: {encoding}")


def deproject_pixel(u: int, v: int, depth_m: float,
                    fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Deproject a single pixel (u, v) + depth to a 3D point in camera optical frame."""
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


def depth_patch_median(depth_img: np.ndarray, u: int, v: int,
                       patch_radius: int = 3,
                       min_depth: float = 0.12,
                       max_depth: float = 10.0) -> float:
    """Extract median depth from a patch around (u, v), filtering invalid values.

    Returns NaN if no valid depths found.
    """
    h, w = depth_img.shape[:2]
    u_lo = max(0, u - patch_radius)
    u_hi = min(w, u + patch_radius + 1)
    v_lo = max(0, v - patch_radius)
    v_hi = min(h, v + patch_radius + 1)

    patch = depth_img[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
    valid = patch[(patch > min_depth) & (patch < max_depth) & np.isfinite(patch)]

    if len(valid) == 0:
        return float('nan')
    return float(np.median(valid))


def tf_to_rotation_matrix(q_x: float, q_y: float, q_z: float, q_w: float):
    """Convert quaternion to 3x3 rotation matrix (no scipy dependency).

    Returns (rotation_matrix, ).
    """
    x, y, z, w = q_x, q_y, q_z, q_w
    rot = np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)
    return rot


# Optical frame to ROS body frame rotation
# Camera optical: Z forward, X right, Y down
# ROS body: X forward, Y left, Z up
R_OPTICAL_TO_BODY = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
], dtype=np.float64)
