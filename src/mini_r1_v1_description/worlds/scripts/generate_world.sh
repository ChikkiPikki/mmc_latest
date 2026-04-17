#!/bin/bash
# World generation pipeline — called by CMake generate_world target.
# Usage: bash generate_world.sh <worlds_dir>
set -eo pipefail   # NOTE: no `-u` because sourcing ROS setup hits unset vars

W="$1"

echo "[generate_world] Worlds dir: $W"

# 1. Copy from source/<selected_world>/ → active_world.building.yaml + map
echo "[generate_world] Step 1/6: copy selected world"
python3 -c "
import os, glob, shutil
w = '$W'
sel_file = os.path.join(w, 'selected_world.txt')
s = open(sel_file).read().strip() if os.path.exists(sel_file) else 'multi_floor_college'
d = os.path.join(w, 'source', s)
yamls = glob.glob(os.path.join(d, '*.building.yaml'))
if yamls:
    shutil.copy2(yamls[0], os.path.join(w, 'active_world.building.yaml'))
    print(f'[generate_world] Copied {yamls[0]}')
for e in ['jpeg', 'jpg', 'png']:
    src = os.path.join(d, f'map.{e}')
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(w, f'map.{e}'))
"

# 2. Download Fuel models (non-fatal — worlds without model:// refs just print "No models")
echo "[generate_world] Step 2/6: download_models.py"
python3 "$W/scripts/download_models.py" "$W/active_world.building.yaml" || {
    echo "[generate_world] download_models.py failed (non-fatal, continuing)"
}

# 3. Source ROS and run RMF building map generator
echo "[generate_world] Step 3/6: source ROS"
set +u
source /opt/ros/humble/setup.bash 2>/dev/null || true
for setup in "$W/../../../install/setup.bash" "$W/../../../../install/setup.bash"; do
    if [ -f "$setup" ]; then
        echo "[generate_world] sourcing $setup"
        source "$setup" 2>/dev/null || true
        break
    fi
done
set -u

echo "[generate_world] Step 4/6: building_map_generator"
if ! command -v ros2 >/dev/null 2>&1; then
    echo "[generate_world] ERROR: ros2 command not found after sourcing setup.bash" >&2
    exit 1
fi
ros2 run rmf_building_map_tools building_map_generator gazebo \
    "$W/active_world.building.yaml" "$W/output.world" "$W/models"

# 4. Fix plugin names for Fortress (gz-sim-* → ignition-gazebo-*)
echo "[generate_world] Step 5/6: patch plugins + sun + inject system plugins"
sed -i 's/libgz-sim-/libignition-gazebo-/g; s/gz::sim::systems/ignition::gazebo::systems/g' "$W/output.world"
sed -i 's/<world name="world">/<world name="generated_world">/' "$W/output.world"

python3 -c "
import re
with open('$W/output.world', 'r') as f:
    c = f.read()
sun = re.compile(r'<include>\s*<uri>model://sun</uri>\s*</include>', re.DOTALL)
light = '''<light type=\"directional\" name=\"sun\">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>0.5 0.1 -0.9</direction>
    </light>'''
c = sun.sub(light, c)
with open('$W/output.world', 'w') as f:
    f.write(c)
"

if ! grep -q 'Physics' "$W/output.world"; then
    sed -i '/<\/scene>/a \
    <physics name="1ms" type="ignored">\
      <max_step_size>0.001</max_step_size>\
      <real_time_factor>1.0</real_time_factor>\
    </physics>\
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"></plugin>\
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"></plugin>\
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"></plugin>\
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors"><render_engine>ogre2</render_engine></plugin>\
    <plugin filename="ignition-gazebo-contact-system" name="ignition::gazebo::systems::Contact"></plugin>\
    <plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"></plugin>' "$W/output.world"
    echo "[generate_world] Injected Fortress system plugins"
fi

# 6. Generate ArUco markers and floor symbols
echo "[generate_world] Step 6/6: generate_markers_and_signs.py"
python3 "$W/scripts/generate_markers_and_signs.py" "$W/active_world.building.yaml" "$W/output.world" "$W"

echo "[generate_world] Done!"
