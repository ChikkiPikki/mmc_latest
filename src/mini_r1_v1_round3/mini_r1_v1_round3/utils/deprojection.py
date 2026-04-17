from typing import Optional

import numpy as np
import rclpy
from tf2_ros import TransformException


def deproject_pixel(u, v, depth_m, fx, fy, cx, cy) -> np.ndarray:
    # Legacy convention (marker_detector_node.py:233-237): returns [z, -x, -y],
    # which is the ROS camera_link frame (x forward, y left, z up), NOT the
    # optical frame. Callers should treat the result as already in camera_link.
    opt_x = (u - cx) * depth_m / fx
    opt_y = (v - cy) * depth_m / fy
    opt_z = depth_m
    return np.array([opt_z, -opt_x, -opt_y])


def transform_point_to_odom(p_cam, tf_buffer, camera_frame, stamp) -> Optional[np.ndarray]:
    try:
        trans = tf_buffer.lookup_transform("odom", camera_frame, rclpy.time.Time())
    except TransformException:
        return None
    q = trans.transform.rotation
    t = trans.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y],
    ])
    t_vec = np.array([t.x, t.y, t.z])
    return rot @ np.asarray(p_cam, dtype=float) + t_vec
