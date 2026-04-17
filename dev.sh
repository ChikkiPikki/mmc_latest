#!/usr/bin/env bash
###############################################################################
# dev.sh — Build & run the Mini R1 Humble+Fortress Docker container
#
# Usage:
#   ./dev.sh build          # Build the Docker image
#   ./dev.sh run             # Run interactive container (mounts ./src)
#   ./dev.sh run-detached    # Run in background
#   ./dev.sh exec            # Attach a new shell to running container
#   ./dev.sh stop            # Stop the running container
#   ./dev.sh rebuild         # Rebuild inside running container (colcon)
###############################################################################

set -euo pipefail

IMAGE_NAME="mini_r1_humble"
CONTAINER_NAME="mini_r1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build() {
    echo "==> Building Docker image: ${IMAGE_NAME}"
    docker build --network=host -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
}

run() {
    xhost +local:docker 2>/dev/null || true

    # Match the host render device group so EGL/GPU sensors work
    local render_gid=""
    if [ -e /dev/dri/renderD128 ]; then
        render_gid="--group-add $(stat -c '%g' /dev/dri/renderD128)"
    fi

    mkdir -p "${SCRIPT_DIR}/dataset"

    echo "==> Starting container: ${CONTAINER_NAME}"
    docker run -it --rm \
        --name "${CONTAINER_NAME}" \
        --hostname openbot \
        --network=host \
        --env DISPLAY="${DISPLAY}" \
        --env QT_X11_NO_MITSHM=1 \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume "${SCRIPT_DIR}/src:/home/dev/ros2_ws/src" \
        --volume "${SCRIPT_DIR}/dataset:/home/dev/dataset" \
        --volume "${SCRIPT_DIR}/grid_world_hackathon:/home/dev/grid_world_hackathon" \
        --device /dev/dri \
        ${render_gid} \
        "${IMAGE_NAME}"
}

run_detached() {
    xhost +local:docker 2>/dev/null || true

    mkdir -p "${SCRIPT_DIR}/dataset"

    echo "==> Starting container (detached): ${CONTAINER_NAME}"
    docker run -d --rm \
        --name "${CONTAINER_NAME}" \
        --hostname openbot \
        --network=host \
        --runtime=nvidia \
        --env DISPLAY="${DISPLAY}" \
        --env QT_X11_NO_MITSHM=1 \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume "${SCRIPT_DIR}/src:/home/dev/ros2_ws/src" \
        --volume "${SCRIPT_DIR}/dataset:/home/dev/dataset" \
        --device /dev/dri \
        "${IMAGE_NAME}" \
        bash -c "sleep infinity"
    echo "==> Container running. Use './dev.sh exec' to attach."
}

exec_shell() {
    echo "==> Attaching to: ${CONTAINER_NAME}"
    docker exec -it "${CONTAINER_NAME}" bash
}

stop_container() {
    echo "==> Stopping: ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || echo "Container not running."
}

rebuild() {
    echo "==> Rebuilding workspace inside container"
    docker exec -it "${CONTAINER_NAME}" bash -c \
        "source /opt/ros/humble/setup.bash && cd /home/dev/ros2_ws && colcon build --symlink-install"
}

case "${1:-run}" in
    build)         build ;;
    run)           run ;;
    run-detached)  run_detached ;;
    exec)          exec_shell ;;
    stop)          stop_container ;;
    rebuild)       rebuild ;;
    *)
        echo "Usage: $0 {build|run|run-detached|exec|stop|rebuild}"
        exit 1
        ;;
esac
