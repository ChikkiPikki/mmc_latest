#!/usr/bin/env python3
"""
Run this once before launching your Gazebo Fortress world.
It will:
  1. Parse all model:// URIs from your .world or .yaml file
  2. Download any missing models from Gazebo Fuel
  3. Create correctly-cased absolute symlinks for each model
  4. Add IGN_GAZEBO_RESOURCE_PATH and IGN_GAZEBO_SYSTEM_PLUGIN_PATH to ~/.bashrc
  5. Apply those env vars to the current shell via a sourced script

Usage:
    python3 download_models.py <path_to_world.sdf_or_building.yaml>

Ported from legacy (Harmonic) to Gazebo Fortress:
  - gz CLI -> ign CLI
  - ~/.gz/fuel -> ~/.ignition/fuel
  - GZ_SIM_* env vars -> IGN_GAZEBO_* env vars
"""

import os
import re
import sys
import subprocess
import yaml
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

FUEL_OWNER     = "OpenRobotics"
FUEL_BASE_URL  = f"https://fuel.gazebosim.org/1.0/{FUEL_OWNER}/models"
FUEL_CACHE     = (Path.home() / ".ignition/fuel/fuel.gazebosim.org" / FUEL_OWNER.lower() / "models").resolve()
ROS_DISTRO     = os.environ.get("ROS_DISTRO", "humble")
ROS_LIB        = Path(f"/opt/ros/{ROS_DISTRO}/lib").resolve()
BASHRC         = Path.home() / ".bashrc"

# ── Parsing ───────────────────────────────────────────────────────────────────

def get_models_from_sdf(path: Path) -> list:
    raw = re.findall(r"model://([^<\"']+)", path.read_text())
    models = [m.strip() for m in raw if m.strip()]
    return sorted(set(models))

def get_models_from_yaml(path: Path) -> list:
    data = yaml.safe_load(path.read_text())
    names = set()
    for level in data.get("levels", {}).values():
        for m in level.get("models", []):
            if isinstance(m, dict):
                name = m.get("model_name") or m.get("type")
                if name:
                    names.add(name)
    return sorted(names)

def get_model_names(path: Path) -> list:
    if path.suffix in (".yaml", ".yml"):
        return get_models_from_yaml(path)
    return get_models_from_sdf(path)

# ── Fuel cache helpers ────────────────────────────────────────────────────────

def find_versioned_dir(model_name: str):
    """Return the latest versioned content dir (e.g. .../chair/3/) or None."""
    name = model_name.split("/")[-1]
    lower_dir = FUEL_CACHE / name.lower()
    if not lower_dir.exists():
        return None
    versions = sorted(
        (p for p in lower_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name)
    )
    return versions[-1].resolve() if versions else None

def is_downloaded(model_name: str) -> bool:
    return find_versioned_dir(model_name) is not None

def ensure_symlink(model_name: str) -> bool:
    """Create correctly-cased absolute symlink. Remove broken ones first."""
    name = model_name.split("/")[-1]
    versioned = find_versioned_dir(model_name)
    if not versioned:
        print(f"  x No versioned dir found for '{model_name}' -- download may have failed")
        return False

    symlink = FUEL_CACHE / name

    # Remove broken symlink
    if symlink.is_symlink() and not symlink.exists():
        symlink.unlink()
        print(f"  -> Removed broken symlink: {symlink.name}")

    if symlink.exists():
        return True  # already good

    try:
        symlink.symlink_to(versioned)
        print(f"  -> Symlink: {symlink.name} -> {versioned}")
        return True
    except Exception as e:
        print(f"  x Symlink failed: {e}")
        return False

# ── Download ──────────────────────────────────────────────────────────────────

def download_model(model_name: str) -> bool:
    name = model_name.split("/")[-1]  # strip any owner prefix
    url  = f"{FUEL_BASE_URL}/{name}"
    print(f"  ign fuel download --url {url}")
    r = subprocess.run(
        ["ign", "fuel", "download", "--url", url],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"  OK Downloaded '{name}'")
        return True
    print(f"  FAIL: {(r.stderr or r.stdout).strip()}")
    return False

# ── ~/.bashrc management ──────────────────────────────────────────────────────

def ensure_in_bashrc(var: str, value: str):
    text = BASHRC.read_text() if BASHRC.exists() else ""
    export_line = f"export {var}=${var}:{value}"
    if value in text:
        return  # already present
    with open(BASHRC, "a") as f:
        f.write(f"\n{export_line}")
    print(f"  OK ~/.bashrc: {export_line}")

def apply_to_env(var: str, value: str):
    current = os.environ.get(var, "")
    if value not in current:
        os.environ[var] = f"{current}:{value}"

def setup_env_var(var: str, value: str):
    path = Path(value)
    if not path.exists():
        print(f"  [skip] {value} does not exist")
        return
    ensure_in_bashrc(var, value)
    apply_to_env(var, value)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_world.sdf | path_to_building.yaml>")
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Gazebo Fortress Model Setup")
    print(f"{'='*55}")
    print(f"  Input : {input_path}")
    print(f"  Cache : {FUEL_CACHE}")
    print(f"{'='*55}\n")

    models = get_model_names(input_path)
    if not models:
        print("No models found in file.")
    else:
        print(f"Found {len(models)} model(s): {', '.join(models)}\n")

    already_cached, downloaded, failed = [], [], []

    for model in models:
        print(f"-- {model}")
        if is_downloaded(model):
            print(f"  OK Already downloaded")
            already_cached.append(model)
        else:
            if download_model(model):
                downloaded.append(model)
            else:
                failed.append(model)
        ensure_symlink(model)
        print()

    # ── Environment setup ─────────────────────────────────────────────────────
    print(f"{'='*55}")
    print("  Setting up environment paths")
    print(f"{'='*55}\n")

    env_paths = {
        "IGN_GAZEBO_RESOURCE_PATH": [
            str(FUEL_CACHE),
            str(input_path.parent / "models"),  # generated world models
        ],
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": [
            str(ROS_LIB),
        ],
    }

    for var, paths in env_paths.items():
        print(f"{var}:")
        for p in paths:
            setup_env_var(var, p)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{'='*55}")
    print(f"  Summary")
    print(f"{'='*55}")
    print(f"  Already cached : {len(already_cached)}")
    print(f"  Downloaded     : {len(downloaded)}")
    print(f"  Failed         : {len(failed)}")
    if failed:
        print(f"\n  Failed models (search manually):")
        for m in failed:
            print(f"    - {m}")
            print(f"      https://app.gazebosim.org/fuel/models?q={m}")

    print(f"\n  ~/.bashrc has been updated.")
    print(f"  Run the following to apply in this terminal, then launch your world:\n")
    print(f"    source ~/.bashrc && ign gazebo {input_path}\n")

if __name__ == "__main__":
    main()
