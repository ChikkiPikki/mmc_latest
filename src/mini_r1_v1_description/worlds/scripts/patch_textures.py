#!/usr/bin/env python3
"""
patch_textures.py  <floor_texture>  <wall_texture>

Swap the floor and wall textures of the generated RMF floor model.

Strategy (filename-swap, not byte-overwrite):
  * Copy  <floor>.png  from the RMF texture library into the model's
    meshes/ dir (fresh filename).
  * Copy  <wall>.png   into meshes/ as well.
  * Rewrite the references in model.sdf, floor_1.mtl, wall_1.mtl, and
    model.material so they point at the new filenames.

Why filename-swap instead of byte overwrite: Gazebo Fortress / ogre2
caches compiled GPU textures keyed on the resource path. Overwriting
the bytes at the same path often yields the cached (old) texture on
the next launch. Using a new filename bypasses the cache entirely.

Usage:
  python3 patch_textures.py blue_linoleum default
  python3 patch_textures.py --list
"""
import argparse
import glob
import os
import re
import shutil
import sys


RMF_TEXTURES_REL = "src/rmf_traffic_editor/rmf_building_map_tools/building_map_generator/textures"
ALBEDO_RE = re.compile(r"<albedo_map>([^<]+)</albedo_map>")
MODEL_URI_RE = re.compile(r"^model://([^/]+)/(.+)$")


def find_rmf_texture_dir(start):
    cur = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(cur, RMF_TEXTURES_REL)
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def find_floor_model_dir(worlds_dir):
    models_dir = os.path.join(worlds_dir, "models")
    sdfs = glob.glob(os.path.join(models_dir, "*_floor_*", "model.sdf"))
    if not sdfs:
        raise RuntimeError(f"No *_floor_*/model.sdf under {models_dir}")

    output_world = os.path.join(worlds_dir, "output.world")
    if os.path.exists(output_world):
        with open(output_world) as f:
            txt = f.read()
        for sdf in sdfs:
            name = os.path.basename(os.path.dirname(sdf))
            if f"model://{name}" in txt:
                return os.path.dirname(sdf)
    sdfs.sort(key=os.path.getmtime, reverse=True)
    return os.path.dirname(sdfs[0])


def current_texture_filenames(sdf_path):
    """Return (floor_png, wall_png) — the *filenames* referenced in the SDF."""
    with open(sdf_path) as f:
        text = f.read()
    uris = ALBEDO_RE.findall(text)
    if len(uris) < 2:
        raise RuntimeError(f"expected ≥2 <albedo_map> in {sdf_path}, found {len(uris)}")
    def to_name(uri):
        m = MODEL_URI_RE.match(uri.strip())
        return os.path.basename(m.group(2)) if m else os.path.basename(uri.strip())
    return to_name(uris[0]), to_name(uris[1])


def replace_in_file(path, old, new):
    if not os.path.isfile(path):
        return 0
    with open(path) as f:
        text = f.read()
    if old not in text:
        return 0
    new_text = text.replace(old, new)
    with open(path, "w") as f:
        f.write(new_text)
    return text.count(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("floor", nargs="?")
    ap.add_argument("wall", nargs="?")
    ap.add_argument("--worlds-dir", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    worlds_dir = args.worlds_dir or os.path.abspath(os.path.join(here, ".."))

    tex_dir = find_rmf_texture_dir(worlds_dir)
    if not tex_dir:
        print(f"ERROR: {RMF_TEXTURES_REL} not found above {worlds_dir}",
              file=sys.stderr)
        sys.exit(2)

    available = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(os.path.join(tex_dir, "*.png")))

    if args.list:
        print("Available RMF textures:")
        for n in available:
            print(f"  {n}")
        return

    if not args.floor or not args.wall:
        ap.error("floor and wall texture names required (or use --list)")
    for name in (args.floor, args.wall):
        if name not in available:
            print(f"ERROR: texture '{name}' not in RMF library at {tex_dir}",
                  file=sys.stderr)
            sys.exit(2)

    model_dir = find_floor_model_dir(worlds_dir)
    meshes_dir = os.path.join(model_dir, "meshes")
    sdf_path = os.path.join(model_dir, "model.sdf")

    old_floor, old_wall = current_texture_filenames(sdf_path)
    new_floor = f"{args.floor}.png"
    new_wall = f"{args.wall}.png"

    # Guard: if floor==wall source name they'd collide in the model dir.
    if new_floor == new_wall:
        new_wall = f"{args.wall}__wall.png"

    # 1. Copy the RMF textures into meshes/ under their real filenames.
    shutil.copy2(os.path.join(tex_dir, f"{args.floor}.png"),
                 os.path.join(meshes_dir, new_floor))
    shutil.copy2(os.path.join(tex_dir, f"{args.wall}.png"),
                 os.path.join(meshes_dir, new_wall))

    # 2. Rewrite references in sdf + both mtl + ogre material. The floor
    #    filename occurs first in each file, so replace floor first.
    touched = 0
    for p in (sdf_path,
              os.path.join(meshes_dir, "floor_1.mtl"),
              os.path.join(meshes_dir, "wall_1.mtl"),
              os.path.join(meshes_dir, "model.material")):
        if old_floor != new_floor:
            touched += replace_in_file(p, old_floor, new_floor)
        if old_wall != new_wall:
            touched += replace_in_file(p, old_wall, new_wall)

    print(f"[patch_textures] floor: {old_floor} -> {new_floor}")
    print(f"[patch_textures] wall:  {old_wall} -> {new_wall}")
    print(f"[patch_textures] patched {touched} occurrences across sdf/mtl/material")


if __name__ == "__main__":
    main()
