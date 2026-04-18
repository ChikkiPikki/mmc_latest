#!/usr/bin/env bash
# Wrapper that guarantees the Gazebo resource path reaches the gz-sim
# GUI subprocess (shell exports propagate through fork/exec, unlike
# launch's SetEnvironmentVariable which gets lost on some setups).
#
# Usage:
#   source install/setup.bash
#   ./src/mini_r1_v1_round3/run_round3.sh
#
# Any args are forwarded to ros2 launch.

set -e

# Pick the grid_world root that actually exists on this box.
for p in \
    /home/dev/grid_world_hackathon/gaws_ws/src/grid_world \
    "$HOME/ros2_ws/grid_world_hackathon/gaws_ws/src/grid_world" \
    /home/tanay/ros2_ws/grid_world_hackathon/gaws_ws/src/grid_world ; do
    if [ -d "$p" ]; then
        GRID_WORLD_ROOT="$p"
        break
    fi
done

if [ -z "$GRID_WORLD_ROOT" ]; then
    echo "[run_round3] ERROR: no grid_world directory found on this machine."
    echo "             Searched /home/dev/... and ~/ros2_ws/..."
    exit 1
fi

GRID_WORLD_PARENT="$(dirname "$GRID_WORLD_ROOT")"
GRID_WORLD_WORLDS="$GRID_WORLD_ROOT/worlds"

# Prepend to both legacy (IGN) and new (GZ) env vars so either backend
# works. Parent dir is what resolves `model://grid_world/...` URIs.
export IGN_GAZEBO_RESOURCE_PATH="$GRID_WORLD_PARENT:$GRID_WORLD_ROOT:$GRID_WORLD_WORLDS${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
export GZ_SIM_RESOURCE_PATH="$GRID_WORLD_PARENT:$GRID_WORLD_ROOT:$GRID_WORLD_WORLDS${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GRID_WORLD_ROOT

echo "[run_round3] GRID_WORLD_ROOT=$GRID_WORLD_ROOT"
echo "[run_round3] IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH"

exec ros2 launch mini_r1_v1_round3 round3.launch.py "$@"
