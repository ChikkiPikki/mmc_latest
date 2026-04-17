import cv2
import numpy as np


HUE_SEEDS = {"green": 60, "orange": 12, "red": 0}


def load_palette(png_path, color_name):
    bgr = cv2.imread(png_path)
    if bgr is None:
        raise FileNotFoundError(png_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s >= 50) & (v >= 50)
    hues = h[valid].astype(np.int32)
    if hues.size == 0:
        raise ValueError(f"no saturated pixels in {png_path}")

    # green.png and orange.png are byte-identical multi-stripe tiles, so the GLOBAL
    # hue peak is not necessarily the requested color. Seed-search near expected hue.
    seed = HUE_SEEDS[color_name]
    search = 25
    dist_to_seed = np.minimum(np.abs(hues - seed), 180 - np.abs(hues - seed))
    near_seed = dist_to_seed <= search
    if not np.any(near_seed):
        raise ValueError(f"no {color_name} pixels near hue {seed} in {png_path}")

    hist = np.bincount(hues[near_seed] % 180, minlength=180)
    peak = int(np.argmax(hist))

    width = 15
    dist = np.minimum(np.abs(hues - peak), 180 - np.abs(hues - peak))
    cluster_mask = dist <= width
    cluster_s = s[valid][cluster_mask]
    cluster_v = v[valid][cluster_mask]

    s_low = max(40, int(np.percentile(cluster_s, 5)))
    v_low = max(40, int(np.percentile(cluster_v, 5)))
    h_low = (peak - width) % 180
    h_high = (peak + width) % 180

    return (np.array([h_low, s_low, v_low], dtype=np.uint8),
            np.array([h_high, 255, 255], dtype=np.uint8))


def inrange_wrapped(hsv, lo, hi):
    # When hue wraps past 0 (lo > hi on H channel), OR two sub-ranges together.
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    lo_a = np.array([0, lo[1], lo[2]], dtype=np.uint8)
    hi_a = np.array([hi[0], hi[1], hi[2]], dtype=np.uint8)
    lo_b = np.array([lo[0], lo[1], lo[2]], dtype=np.uint8)
    hi_b = np.array([179, hi[1], hi[2]], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lo_a, hi_a),
                          cv2.inRange(hsv, lo_b, hi_b))


def validate_against_sim(cv_rgb_frame, palette, min_pixels=500):
    hsv = cv2.cvtColor(cv_rgb_frame, cv2.COLOR_RGB2HSV)
    lo, hi = palette
    mask = inrange_wrapped(hsv, lo, hi)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return False
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest >= min_pixels
