#!/usr/bin/env python3
"""
generate_markers_and_signs.py

Reads the building YAML, finds aruco_* and symbol_* vertices,
generates textures, and injects inline SDF <model> blocks into
the output.world file.

Symbols are flat floor decals (the PNG is the top-face albedo).
ArUco markers are wall-mounted cubes.

Usage:
  python3 generate_markers_and_signs.py <building.yaml> <output.world> <worlds_dir>
"""

import sys
import os
import math
import yaml
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IMG_PX = 512
ARUCO_BOX_SIZE = 0.4
ARUCO_BOX_COLOR = "0.8 0.8 0.2 1"

ZONE_RADIUS = 0.85
ZONE_HEIGHT = 2.0
ZONE_TRANSPARENCY = 0.4
ZONE_START_COLOR = "0.9 0.1 0.1 1.0"
ZONE_GOAL_COLOR = "0.0 1.0 1.0 1.0"

SYMBOL_SIZE = 0.45            # metres (square decal on ground)
SYMBOL_THICKNESS = 0.002      # very thin — visual only
SYMBOL_Z_OFFSET = 0.001       # lifted just off floor to avoid z-fighting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_scale(vertices, measurements):
    m = measurements[0]
    idx_a, idx_b = m[0], m[1]
    dist_m = m[2]["distance"][1]
    va = vertices[idx_a]
    vb = vertices[idx_b]
    px_dist = math.hypot(vb[0] - va[0], vb[1] - va[1])
    return px_dist / dist_m


def px_to_world(px_x, px_y, scale):
    return px_x / scale, -px_y / scale


def deg_to_rad(d):
    return d * math.pi / 180.0


# ---------------------------------------------------------------------------
# ArUco generation
# ---------------------------------------------------------------------------
def generate_aruco_image(marker_id, out_path):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        marker_core = cv2.aruco.generateImageMarker(aruco_dict, marker_id, ARUCO_IMG_PX - 80)
    except AttributeError:
        marker_core = cv2.aruco.drawMarker(aruco_dict, marker_id, ARUCO_IMG_PX - 80)
    img = 255 * np.ones((ARUCO_IMG_PX, ARUCO_IMG_PX), dtype=np.uint8)
    off = 40
    img[off:off + marker_core.shape[0], off:off + marker_core.shape[1]] = marker_core
    cv2.imwrite(out_path, img)


def aruco_model_sdf(name, wx, wy, yaw_rad, texture_path):
    half = ARUCO_BOX_SIZE / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{wx} {wy} {half} 0 0 {yaw_rad}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE}</size></box></geometry>
          <material>
            <ambient>{ARUCO_BOX_COLOR}</ambient>
            <diffuse>{ARUCO_BOX_COLOR}</diffuse>
            <pbr>
              <metal>
                <albedo_map>{texture_path}</albedo_map>
              </metal>
            </pbr>
          </material>
        </visual>
        <collision name="collision">
          <geometry><box><size>{ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE} {ARUCO_BOX_SIZE}</size></box></geometry>
        </collision>
      </link>
    </model>"""


# ---------------------------------------------------------------------------
# Floor symbol decal
# ---------------------------------------------------------------------------
def symbol_model_sdf(name, wx, wy, yaw_rad, texture_path):
    """Flat square decal, face-up, with the symbol PNG as albedo.

    The box is extremely thin; only the top face is meaningfully visible.
    No collision — it's purely visual (robot can drive over it).
    """
    z = SYMBOL_Z_OFFSET + SYMBOL_THICKNESS / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{wx} {wy} {z} 0 0 {yaw_rad}</pose>
      <link name="link">
        <visual name="decal">
          <geometry><box><size>{SYMBOL_SIZE} {SYMBOL_SIZE} {SYMBOL_THICKNESS}</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
            <pbr>
              <metal>
                <albedo_map>{texture_path}</albedo_map>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>"""


def zone_model_sdf(name, wx, wy, color, label):
    center_z = ZONE_HEIGHT / 2.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{wx} {wy} {center_z} 0 0 0</pose>
      <link name="link">
        <visual name="cylinder">
          <visibility_flags>1</visibility_flags>
          <transparency>{ZONE_TRANSPARENCY}</transparency>
          <geometry>
            <cylinder>
              <radius>{ZONE_RADIUS}</radius>
              <length>{ZONE_HEIGHT}</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


# ---------------------------------------------------------------------------
# Vertex name parsers
# ---------------------------------------------------------------------------
def parse_aruco(name):
    """Parse 'aruco_{id}_{yaw}' -> (marker_id, yaw_deg)."""
    parts = name.split("_")
    marker_id = int(parts[1])
    yaw_deg = float(parts[-1])
    return marker_id, yaw_deg


def parse_symbol(name):
    """Parse 'symbol_{type_id}_{yaw}' -> (type_id, yaw_deg)."""
    parts = name.split("_")
    type_id = parts[1]
    yaw_deg = float(parts[-1])
    return type_id, yaw_deg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <building.yaml> <output.world> <worlds_dir>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    world_path = sys.argv[2]
    worlds_dir = sys.argv[3]

    gen_dir = os.path.join(worlds_dir, "generated")
    aruco_dir = os.path.join(gen_dir, "aruco")
    os.makedirs(aruco_dir, exist_ok=True)

    symbols_dir = os.path.join(worlds_dir, "symbols")

    with open(yaml_path, "r") as f:
        building = yaml.safe_load(f)

    level_name = list(building["levels"].keys())[0]
    level = building["levels"][level_name]
    vertices = level["vertices"]
    measurements = level.get("measurements", [])

    if not measurements:
        print("ERROR: No measurements found — cannot compute scale.", file=sys.stderr)
        sys.exit(1)

    scale = compute_scale(vertices, measurements)
    print(f"[markers_signs] Scale: {scale:.4f} px/m")

    sdf_blocks = []

    # --- ArUco markers (wall-mounted cubes) ---
    for idx, v in enumerate(vertices):
        name = v[3] if len(v) >= 4 else ""
        if not isinstance(name, str) or not name.startswith("aruco_"):
            continue

        marker_id, yaw_deg = parse_aruco(name)
        wx, wy = px_to_world(v[0], v[1], scale)
        yaw_rad = deg_to_rad(yaw_deg)

        tex_path = os.path.join(aruco_dir, f"aruco_{marker_id}.png")
        generate_aruco_image(marker_id, tex_path)

        model_name = f"aruco_{marker_id}_v{idx}"
        sdf = aruco_model_sdf(model_name, wx, wy, yaw_rad, tex_path)
        sdf_blocks.append(sdf)
        print(f"  [aruco] id={marker_id} yaw={yaw_deg} deg @ ({wx:.2f}, {wy:.2f})")

    # --- Floor symbols (flat decals) ---
    for idx, v in enumerate(vertices):
        name = v[3] if len(v) >= 4 else ""
        if not isinstance(name, str) or not name.startswith("symbol_"):
            continue

        type_id, yaw_deg = parse_symbol(name)
        wx, wy = px_to_world(v[0], v[1], scale)
        yaw_rad = deg_to_rad(yaw_deg)

        tex_path = os.path.join(symbols_dir, f"symbol_{type_id}.png")
        if not os.path.exists(tex_path):
            print(f"  [symbol] WARNING: missing texture {tex_path}, skipping", file=sys.stderr)
            continue

        model_name = f"symbol_{type_id}_v{idx}"
        sdf = symbol_model_sdf(model_name, wx, wy, yaw_rad, tex_path)
        sdf_blocks.append(sdf)
        print(f"  [symbol] type={type_id} yaw={yaw_deg} deg @ ({wx:.2f}, {wy:.2f})")

    if not sdf_blocks:
        print("[markers_signs] No markers or symbols found.")
        return

    # --- Inject into output.world ---
    with open(world_path, "r") as f:
        world_xml = f.read()

    injection = "\n".join(sdf_blocks)
    marker_tag = "</world>"
    if marker_tag not in world_xml:
        print("ERROR: Could not find </world> in output.world", file=sys.stderr)
        sys.exit(1)

    world_xml = world_xml.replace(marker_tag, injection + "\n  " + marker_tag)

    with open(world_path, "w") as f:
        f.write(world_xml)

    print(f"[markers_signs] Injected {len(sdf_blocks)} models into {world_path}")


if __name__ == "__main__":
    main()
