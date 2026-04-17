#!/usr/bin/env python3
"""
World Selector GUI — Tkinter-based picker for available worlds.

Scans the 'source/' subdirectory for world folders containing a
.building.yaml file and a map image. Copies the selected world's
files to the worlds/ root as 'active_world.building.yaml' and 'map.jpeg'
so that the build pipeline can consume them.

Usage:
    python3 world_selector.py            (from worlds/ directory, or)
    python3 world_selector.py /path/to/worlds/
"""
import os
import sys
import glob
import shutil
import tkinter as tk
from tkinter import ttk, messagebox


def find_worlds(source_dir):
    """Return list of (name, yaml_path, map_path) for each valid world."""
    worlds = []
    if not os.path.isdir(source_dir):
        return worlds

    for entry in sorted(os.listdir(source_dir)):
        subdir = os.path.join(source_dir, entry)
        if not os.path.isdir(subdir):
            continue

        # Find the .building.yaml
        yamls = glob.glob(os.path.join(subdir, "*.building.yaml"))
        if not yamls:
            continue
        yaml_path = yamls[0]

        # Find the map image (jpeg or png)
        map_path = None
        for ext in ["map.jpeg", "map.jpg", "map.png"]:
            candidate = os.path.join(subdir, ext)
            if os.path.exists(candidate):
                map_path = candidate
                break
        # Fallback: any image
        if map_path is None:
            for ext in ["*.jpeg", "*.jpg", "*.png"]:
                imgs = glob.glob(os.path.join(subdir, ext))
                if imgs:
                    map_path = imgs[0]
                    break

        worlds.append((entry, yaml_path, map_path))

    return worlds


def select_world(worlds_dir, source_dir):
    """Show a Tkinter GUI to select and activate a world."""
    worlds = find_worlds(source_dir)
    if not worlds:
        print(f"ERROR: No valid worlds found in {source_dir}")
        print("Each world should be a subdirectory containing a .building.yaml and a map image.")
        sys.exit(1)

    selected = [None]

    def on_select():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a world.")
            return
        idx = sel[0]
        name, yaml_path, map_path = worlds[idx]

        # Copy YAML → active_world.building.yaml
        dest_yaml = os.path.join(worlds_dir, "active_world.building.yaml")
        shutil.copy2(yaml_path, dest_yaml)
        print(f"Copied {yaml_path} → {dest_yaml}")

        # Copy map image → map.jpeg (or map.png)
        if map_path:
            ext = os.path.splitext(map_path)[1]
            dest_map = os.path.join(worlds_dir, f"map{ext}")
            # Remove old map files first
            for old in glob.glob(os.path.join(worlds_dir, "map.*")):
                if os.path.basename(old) in ["map.jpeg", "map.jpg", "map.png"]:
                    os.remove(old)
            shutil.copy2(map_path, dest_map)
            print(f"Copied {map_path} → {dest_map}")

        # Write marker file
        marker_path = os.path.join(worlds_dir, "selected_world.txt")
        with open(marker_path, "w") as f:
            f.write(name + "\n")
        print(f"Written marker: {marker_path} → {name}")

        selected[0] = name
        root.destroy()

    # ── Build GUI ──
    root = tk.Tk()
    root.title("World Selector — Mini R1 V1")
    root.geometry("480x360")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Select a world to load:",
              font=("Helvetica", 14, "bold")).pack(anchor=tk.W, pady=(0, 10))

    listbox = tk.Listbox(frame, font=("Helvetica", 12), selectmode=tk.SINGLE, height=10)
    for name, yaml_path, map_path in worlds:
        label = name
        if map_path is None:
            label += "  (⚠ no map image)"
        listbox.insert(tk.END, label)
    listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    # Pre-select first
    if worlds:
        listbox.selection_set(0)

    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill=tk.X)

    ttk.Button(btn_frame, text="Select & Activate", command=on_select).pack(side=tk.RIGHT)
    ttk.Button(btn_frame, text="Cancel", command=root.destroy).pack(side=tk.RIGHT, padx=(0, 10))

    # Current active world indicator
    marker_path = os.path.join(worlds_dir, "selected_world.txt")
    if os.path.exists(marker_path):
        with open(marker_path) as f:
            current = f.read().strip()
        ttk.Label(frame, text=f"Currently active: {current}",
                  font=("Helvetica", 10), foreground="gray").pack(anchor=tk.W, pady=(5, 0))

    root.mainloop()

    if selected[0]:
        print(f"\n✅ World '{selected[0]}' activated. Run 'colcon build' to regenerate.")
    else:
        print("\nCancelled — no world selected.")
        sys.exit(0)


def main():
    if len(sys.argv) > 1:
        worlds_dir = sys.argv[1]
    else:
        worlds_dir = os.path.dirname(os.path.abspath(__file__))

    source_dir = os.path.join(worlds_dir, "source")
    select_world(worlds_dir, source_dir)


if __name__ == "__main__":
    main()
