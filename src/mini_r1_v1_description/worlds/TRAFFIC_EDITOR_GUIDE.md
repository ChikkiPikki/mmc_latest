# Traffic Editor Vertex Naming Guide

## How the World Generation Pipeline Works

The `generate_markers_and_signs.py` script reads vertex names from the `.building.yaml`
file and generates SDF models at those positions. Each named vertex becomes a model
in the Gazebo world.

## Vertex Naming Conventions

### Floor Symbols (flat decals on the ground)

```
symbol_{type_id}_{yaw_degrees}
```

| Part | Meaning |
|------|---------|
| `symbol` | Prefix — tells the generator this is a floor symbol |
| `{type_id}` | Which symbol image to use (matches `symbol_{type_id}.png` in `worlds/symbols/`) |
| `{yaw_degrees}` | Rotation of the symbol on the floor (0 = facing +X, 90 = facing +Y, etc.) |

**Examples:**
- `symbol_1_0` — places `symbol_1.png` at this vertex, no rotation
- `symbol_1_90` — places `symbol_1.png` rotated 90 degrees
- `symbol_2_0` — places `symbol_2.png` at this vertex, no rotation
- `symbol_3_270` — places `symbol_3.png` rotated 270 degrees

You can place the same symbol type at multiple vertices. Each vertex gets a unique
model instance in the world.

### Wall-Mounted ArUco Markers

```
aruco_{marker_id}_{yaw_degrees}
```

| Part | Meaning |
|------|---------|
| `aruco` | Prefix — tells the generator this is an ArUco marker |
| `{marker_id}` | ArUco dictionary ID (0-49, using DICT_4X4_50) |
| `{yaw_degrees}` | Which direction the marker faces (must face away from the wall) |

**Examples:**
- `aruco_1_0` — ArUco marker ID 1, facing +X direction
- `aruco_2_180` — ArUco marker ID 2, facing -X direction
- `aruco_3_90` — ArUco marker ID 3, facing +Y direction

### Zone Markers (spawn/goal)

```
spawn_{yaw_degrees}    — Robot spawn position (red cylinder)
goal                   — Goal position (cyan cylinder)
```

**Examples:**
- `spawn_0` — Spawn point, robot facing +X
- `spawn_90` — Spawn point, robot facing +Y
- `goal` — Goal zone

## Step-by-Step: Adding Symbols in the Traffic Editor

1. Open the traffic editor: `traffic-editor`
2. Load your `.building.yaml` file
3. Switch to vertex mode (V key)
4. Click to place a vertex where you want a symbol on the floor
5. In the vertex properties, set the **name** field using the convention above
   - e.g., `symbol_1_0` for symbol_1.png with no rotation
6. Save the YAML
7. Run `colcon build` — the generate_world target will:
   - Read the YAML
   - Find all `symbol_*` vertices
   - Generate flat floor decal models in the SDF world
   - Place them at the correct world coordinates

## For Dataset Generation Worlds

Dataset worlds live under `source/dataset_symbol_<id>/`. To create one, either:

1. **Auto-generate** from an existing base world:
   ```bash
   python3 scripts/build_dataset_world.py \
     --base source/simple/simple.building.yaml \
     --symbol-id 1 \
     --out  source/dataset_symbol_1/dataset_symbol_1.building.yaml
   ```
   This injects `symbol_1_*` vertices on a 1.2 m grid, skipping positions too close to walls.

2. **Hand-author** via traffic-editor, placing many `symbol_<id>_<yaw>` vertices.

Then capture the dataset:
```bash
# Terminal 1 — start sim with the dataset world selected
echo "dataset_symbol_1" > selected_world.txt
colcon build --packages-select mini_r1_v1_description
ros2 launch mini_r1_v1_gz sim.launch.py

# Terminal 2 — run the capture script
python3 scripts/generate_dataset.py \
  --yaml  active_world.building.yaml \
  --output ~/dataset
```

## Quick Reference

| Vertex Name | What It Creates |
|-------------|----------------|
| `symbol_1_0` | Floor decal of symbol_1.png, 0 deg rotation |
| `symbol_2_90` | Floor decal of symbol_2.png, 90 deg rotation |
| `aruco_1_0` | Wall ArUco marker ID 1, facing 0 deg |
| `aruco_3_180` | Wall ArUco marker ID 3, facing 180 deg |
| `spawn_0` | Robot spawn (red zone), facing 0 deg |
| `goal` | Goal zone (cyan) |
| `dynamic_obstacle_1_high` | Dynamic obstacle waypoint (high end) |
| `dynamic_obstacle_1_low` | Dynamic obstacle waypoint (low end) |
