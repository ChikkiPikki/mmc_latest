#!/bin/bash
# run_dataset_capture.sh
#
# Dataset generation — one symbol type per world. For each requested
# symbol id we synthesize a fresh big square room containing only that
# symbol (generously spaced on a grid), rebuild output.world, then loop
# N random texture combos. Optional wall-only "negatives" pass at end.
# Must run inside the Docker container.
#
# Usage:
#   run_dataset_capture.sh SYMBOL [OPTIONS]
#
# SYMBOL:
#   <id>            single symbol id (1..7 or imported)
#   <id>,<id>,...   multiple ids — each gets its own world
#   all             every symbol PNG under worlds/symbols/
#
# Options:
#   --output DIR           Output directory (default: $HOME/dataset)
#   --combos N             Number of random texture combos (default: 5)
#   --negatives[:N]        Also capture N wall-only negative positions
#                          (default N=40)
#   --obstacles N          Inject N random static obstacles per combo
#                          (default 0 = disabled)
#   --target-images N      Stop capturing a symbol once it has N images on
#                          disk (default 0 = no cap, let combos stack)
#   --no-headless          Run Gazebo with GUI (slower, for debugging)
#   --clear                Delete the output dir before starting
#   --skip-build           Assume output.world is already fresh (dev flag)
set -eo pipefail

usage() {
    grep -E '^# ' "$0" | sed 's/^# \?//'
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

SYMBOL="$1"; shift

OUT="$HOME/dataset"
COMBOS=5
NEGATIVES=0
NEG_COUNT=40
OBSTACLES=0
TARGET_IMAGES=0
HEADLESS=true
CLEAR=false
SKIP_BUILD=false

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUT="$2"; shift 2;;
        --combos) COMBOS="$2"; shift 2;;
        --negatives) NEGATIVES=1; shift;;
        --negatives:*) NEGATIVES=1; NEG_COUNT="${1#--negatives:}"; shift;;
        --obstacles) OBSTACLES="$2"; shift 2;;
        --target-images) TARGET_IMAGES="$2"; shift 2;;
        --no-headless) HEADLESS=false; shift;;
        --clear) CLEAR=true; shift;;
        --skip-build) SKIP_BUILD=true; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WS_DIR="$(cd "$WORLDS_DIR/../../../.." && pwd)"
LOG="$WORLDS_DIR/dataset_sim.log"

# Expand SYMBOL argument to an explicit list of ids. "all" → every
# symbol_<id>.png under worlds/symbols/.
if [ "$SYMBOL" = "all" ]; then
    mapfile -t SYMBOL_LIST < <(
        ls "$WORLDS_DIR/symbols"/symbol_*.png 2>/dev/null \
          | sed -n 's#.*/symbol_\([0-9]\+\)\.png#\1#p' | sort -n -u
    )
elif [[ "$SYMBOL" == *,* ]]; then
    IFS=',' read -ra SYMBOL_LIST <<< "$SYMBOL"
else
    SYMBOL_LIST=("$SYMBOL")
fi

if [ "${#SYMBOL_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no symbol ids resolved from '$SYMBOL'" >&2
    exit 1
fi

echo "[run_dataset] Symbols: ${SYMBOL_LIST[*]}"
echo "[run_dataset] Output=$OUT  Combos=$COMBOS  Negatives=$NEGATIVES ($NEG_COUNT)"
echo "[run_dataset] Obstacles=$OBSTACLES  Target=$TARGET_IMAGES  Headless=$HEADLESS  Clear=$CLEAR"

mkdir -p "$OUT"
if [ "$CLEAR" = "true" ]; then
    echo "[run_dataset] Clearing existing contents of $OUT"
    find "$OUT" -mindepth 1 -delete
fi

cd "$WS_DIR"
source /opt/ros/humble/setup.bash
if [ "$SKIP_BUILD" = "false" ]; then
    colcon build --packages-select mini_r1_v1_description --symlink-install
fi
source install/setup.bash

ORIG_WORLD="$WORLDS_DIR/output.world.orig"

# ── Texture combos ────────────────────────────────────────────────────
FLOOR_POOL=(blue_linoleum beige gray_flooring white_marble red_tiles beige_tiles concrete3 white_beige_tiles brown_marble)
WALL_POOL=(default wall wall_white white_brick concrete concrete4 black_concrete beige)

ALL_PAIRS=()
for f in "${FLOOR_POOL[@]}"; do
    for w in "${WALL_POOL[@]}"; do
        ALL_PAIRS+=("${f}|${w}")
    done
done

mapfile -t COMBO_PAIRS < <(printf "%s\n" "${ALL_PAIRS[@]}" | shuf | head -n "$COMBOS")

# Hard-truncate in case mapfile over-read (shouldn't, but defensive).
COMBO_PAIRS=("${COMBO_PAIRS[@]:0:$COMBOS}")

echo "[run_dataset] Texture combos (${#COMBO_PAIRS[@]} of requested $COMBOS):"
for pair in "${COMBO_PAIRS[@]}"; do
    echo "    ${pair/|/ + }"
done

# ── Progress helpers ──────────────────────────────────────────────────
RUN_T0=$(date +%s)

fmt_dur() {
    local s=$1 h m
    h=$(( s / 3600 )); s=$(( s % 3600 ))
    m=$(( s / 60 )); s=$(( s % 60 ))
    if [ "$h" -gt 0 ]; then printf "%dh%02dm" "$h" "$m"
    elif [ "$m" -gt 0 ]; then printf "%dm%02ds" "$m" "$s"
    else printf "%ds" "$s"
    fi
}

count_pngs() {
    find "$OUT" -maxdepth 2 -name '*.png' -type f 2>/dev/null | wc -l
}

print_progress() {
    local stage="$1" done="$2" total="$3"
    local elapsed=$(( $(date +%s) - RUN_T0 ))
    local pngs; pngs=$(count_pngs)
    local pct=$(( done * 100 / total ))
    local eta=0
    if [ "$done" -gt 0 ]; then
        eta=$(( elapsed * (total - done) / done ))
    fi
    printf "[progress] %s %d/%d (%d%%)  images=%d  elapsed=%s  ETA=%s\n" \
        "$stage" "$done" "$total" "$pct" "$pngs" \
        "$(fmt_dur "$elapsed")" "$(fmt_dur "$eta")"
}

LAUNCH_ARGS=()
if [ "$HEADLESS" = "true" ]; then
    LAUNCH_ARGS+=(headless:=true)
fi

SIM_PID=""
launch_sim() {
    : > "$LOG"
    # setsid so the sim is its own session — kill -- -PGID wipes the whole tree.
    setsid ros2 launch mini_r1_v1_gz sim.launch.py "${LAUNCH_ARGS[@]}" \
        > "$LOG" 2>&1 &
    SIM_PID=$!
    echo "[run_dataset] Launched sim (PID $SIM_PID)"
}

kill_sim() {
    echo "[run_dataset] Stopping sim..."
    if [ -n "$SIM_PID" ]; then
        # Signal the whole session group (setsid → PID == PGID)
        kill -INT  -- -"$SIM_PID" 2>/dev/null || kill -INT  "$SIM_PID" 2>/dev/null || true
        sleep 2
        kill -TERM -- -"$SIM_PID" 2>/dev/null || kill -TERM "$SIM_PID" 2>/dev/null || true
        sleep 1
        kill -KILL -- -"$SIM_PID" 2>/dev/null || kill -KILL "$SIM_PID" 2>/dev/null || true
        # Do NOT `wait` — the proc may be in an uninterruptible state;
        # blocking here is exactly what caused the hang.
        SIM_PID=""
    fi
    # Catch orphans (gz-sim server keeps running if the launch supervisor died).
    # NOTE: do NOT match on 'dataset_capture' — the shell script's own cmdline
    # contains that substring (run_dataset_capture.sh), and pkill does not
    # exclude parent shells, so it would SIGKILL this very script.
    pkill -KILL -f 'ign-gazebo-server'     2>/dev/null || true
    pkill -KILL -f 'ign-gazebo-gui'        2>/dev/null || true
    pkill -KILL -f 'ignition-gazebo'       2>/dev/null || true
    pkill -KILL -f 'parameter_bridge'      2>/dev/null || true
    pkill -KILL -f 'robot_state_publisher' 2>/dev/null || true
    pkill -KILL -f 'twist_stamper'         2>/dev/null || true
    sleep 2
    echo "[run_dataset] Sim stopped"
}

cleanup() { kill_sim; }
trap cleanup EXIT

wait_camera() {
    echo "[run_dataset] Waiting up to 50 s for /r1_mini/camera/image_raw..."
    for i in $(seq 1 50); do
        if ros2 topic list 2>/dev/null | grep -q '^/r1_mini/camera/image_raw$'; then
            echo "[run_dataset] Camera topic up after ${i}s"
            sleep 3
            return 0
        fi
        sleep 1
    done
    echo "[run_dataset] Camera topic never appeared. Last 60 log lines:" >&2
    tail -60 "$LOG" >&2
    return 1
}

# ── Per-symbol outer loop: fresh big world per symbol, one type only ──
SYM_TOTAL=${#SYMBOL_LIST[@]}
for sidx in "${!SYMBOL_LIST[@]}"; do
    sid="${SYMBOL_LIST[$sidx]}"
    WORLD_NAME="dataset_big_symbol_${sid}"
    WORLD_DIR="$WORLDS_DIR/source/$WORLD_NAME"
    WORLD_YAML="$WORLD_DIR/${WORLD_NAME}.building.yaml"

    echo
    echo "############################################################"
    echo "[run_dataset] Symbol $((sidx+1))/$SYM_TOTAL: symbol_${sid}"
    echo "############################################################"

    # 1. Fresh big world YAML with ONLY this symbol type, widely spaced.
    python3 "$SCRIPT_DIR/make_big_world.py" \
        --symbol-id "$sid" --out "$WORLD_YAML"

    # 2. Rebuild output.world via the standard pipeline.
    echo "$WORLD_NAME" > "$WORLDS_DIR/selected_world.txt"
    rm -f "$WORLDS_DIR/output.world"
    bash "$SCRIPT_DIR/generate_world.sh" "$WORLDS_DIR" || {
        echo "[run_dataset] World generation failed for symbol $sid." >&2
        exit 1
    }
    if ! grep -q '<model name="symbol_' "$WORLDS_DIR/output.world"; then
        echo "[run_dataset] ERROR: no symbol models in output.world for $sid" >&2
        exit 1
    fi
    cp "$WORLDS_DIR/output.world" "$ORIG_WORLD"

    # 3. Texture combos for this symbol.
    for (( i=0; i < COMBOS && i < ${#COMBO_PAIRS[@]}; i++ )); do
        pair="${COMBO_PAIRS[$i]}"
        floor="${pair%|*}"
        wall="${pair#*|}"
        label="s${sid}_floor-${floor}_wall-${wall}"

        echo
        echo "------------------------------------------------------------"
        echo "[run_dataset] Symbol $sid  Combo $((i+1))/$COMBOS: $label"
        echo "------------------------------------------------------------"

        cp "$ORIG_WORLD" "$WORLDS_DIR/output.world"
        if [ "$OBSTACLES" -gt 0 ]; then
            python3 "$SCRIPT_DIR/inject_obstacles.py" \
                --world "$WORLDS_DIR/output.world" \
                --yaml  "$WORLDS_DIR/active_world.building.yaml" \
                --count "$OBSTACLES" \
                --seed  "$((RANDOM * 32768 + RANDOM))"
        fi
        python3 "$SCRIPT_DIR/patch_textures.py" "$floor" "$wall"

        kill_sim
        launch_sim
        wait_camera || exit 1

        TARGET_ARG=()
        if [ "$TARGET_IMAGES" -gt 0 ]; then
            TARGET_ARG=(--target-images "$TARGET_IMAGES")
        fi

        python3 "$SCRIPT_DIR/generate_dataset.py" \
            --yaml "$WORLDS_DIR/active_world.building.yaml" \
            --output "$OUT" \
            --combo-label "$label" \
            --progress-prefix "sym${sid} combo $((i+1))/$COMBOS " \
            "${TARGET_ARG[@]}"

        kill_sim
        STEP=$(( sidx * COMBOS + i + 1 ))
        TOTAL=$(( SYM_TOTAL * COMBOS ))
        print_progress "overall" "$STEP" "$TOTAL"
    done
done

# ── Negatives pass ────────────────────────────────────────────────────
if [ "$NEGATIVES" = "1" ]; then
    echo
    echo "============================================================"
    echo "[run_dataset] Negatives: $NEG_COUNT positions"
    echo "============================================================"

    pair="${COMBO_PAIRS[$((${#COMBO_PAIRS[@]}-1))]}"
    floor="${pair%|*}"
    wall="${pair#*|}"
    label="negatives_floor-${floor}_wall-${wall}"
    # Pristine world for negatives — no obstacles (cleaner wall samples)
    cp "$ORIG_WORLD" "$WORLDS_DIR/output.world"
    python3 "$SCRIPT_DIR/patch_textures.py" "$floor" "$wall"

    kill_sim
    launch_sim
    wait_camera || exit 1

    python3 "$SCRIPT_DIR/generate_dataset.py" \
        --yaml "$WORLDS_DIR/active_world.building.yaml" \
        --output "$OUT" \
        --combo-label "$label" \
        --progress-prefix "negatives " \
        --negatives --negative-samples "$NEG_COUNT"

    kill_sim
fi

FINAL_ELAPSED=$(( $(date +%s) - RUN_T0 ))
FINAL_PNGS=$(count_pngs)
echo
echo "[run_dataset] Done in $(fmt_dur "$FINAL_ELAPSED"). ${FINAL_PNGS} images in $OUT"
for d in "$OUT"/*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    n=$(find "$d" -maxdepth 1 -name '*.png' | wc -l)
    printf "  %-30s %5d images\n" "$name" "$n"
done
