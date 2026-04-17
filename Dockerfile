# syntax=docker/dockerfile:1.4
###############################################################################
# Dockerfile — Mini R1 v1  (ROS 2 Humble · Ubuntu 22.04 · Gazebo Fortress)
#
# Build (BuildKit required for cache mounts):
#   DOCKER_BUILDKIT=1 docker build -t mini_r1_humble .
#
# Run (GPU + display):
#   xhost +local:docker
#   docker run -it --rm \
#     --name mini_r1 \
#     --network=host \
#     --runtime=nvidia \
#     --env DISPLAY=$DISPLAY \
#     --env QT_X11_NO_MITSHM=1 \
#     --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
#     --volume ./src:/home/dev/ros2_ws/src \
#     --device /dev/dri \
#     mini_r1_humble
###############################################################################

FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# ── NVIDIA Container Runtime ────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

LABEL hostname="openbot"

# ── Faster Ubuntu mirror (ROS mirror untouched — already fast) ──────────
# mirror.sg.gs (Singapore) benchmarked ~4x faster than archive.ubuntu.com
# from this location. To revert, remove this RUN block.
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirror.sg.gs/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirror.sg.gs/ubuntu|g' /etc/apt/sources.list

# ── Apt reliability: retries, longer timeouts, pipelining ───────────────
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "60";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::https::Timeout "60";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Pipeline-Depth "5";' >> /etc/apt/apt.conf.d/80-retries && \
    echo 'APT::Acquire::Queue-Mode "host";' >> /etc/apt/apt.conf.d/80-retries

# ── Group 1: Build tools (rarely changes — stays cached longest) ────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
        git \
        wget \
        mesa-utils

# ── Group 2: Gazebo Fortress + ros-gz bridge ────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-gz \
        ros-humble-ros-gz-bridge \
        ros-humble-ros-gz-sim \
        ros-humble-ros-gz-image

# ── Group 3: ROS 2 core packages ────────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-twist-stamper \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-cv-bridge \
        ros-humble-message-filters

# ── Group 4: Nav2 + RMF + localization + rviz ───────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-nav2-costmap-2d \
        ros-humble-nav2-lifecycle-manager \
        ros-humble-rmf-building-map-msgs \
        ros-humble-rmf-building-map-tools \
        ros-humble-rmf-site-map-msgs \
        ros-humble-rmf-utils \
        ros-humble-robot-localization \
        ros-humble-rviz2 \
        ros-humble-rviz-common \
        ros-humble-rviz-default-plugins

# ── Group 5: Python system packages ─────────────────────────────────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3-opencv \
        python3-numpy \
        python3-yaml \
        python3-shapely \
        python3-pyproj \
        python3-requests \
        python3-rtree \
        python3-fiona \
        python3-ignition-math6

# ── Group 6: Round 3 deps (SLAM, Nav2, AprilTag, localization) ──────────
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-slam-toolbox \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-nav2-simple-commander \
        ros-humble-apriltag-ros \
        ros-humble-apriltag-msgs \
        ros-humble-robot-localization

# ── Python pip packages (cached between builds) ─────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.txt

# ── Initialise rosdep (with retries for flaky networks) ─────────────────
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi && \
    for i in 1 2 3 4 5; do \
        rosdep update --rosdistro=humble && break || \
        { echo "rosdep update failed, retry $i/5"; sleep 5; }; \
    done

# ── Create non-root user ───────────────────────────────────────────────
RUN groupadd -g 109 render 2>/dev/null || true && \
    useradd -m -s /bin/bash -G video,render,dialout dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ── Workspace setup ────────────────────────────────────────────────────
USER dev
WORKDIR /home/dev/ros2_ws

# Copy only src (legacy_src excluded via .dockerignore)
COPY --chown=dev:dev src/ src/

# ── rosdep install for workspace deps ──────────────────────────────────
# Needs root for apt. Kept in its own layer after COPY src/ so src/ changes
# don't re-trigger the big apt groups above.
USER root
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    . /opt/ros/humble/setup.sh && \
    apt-get update && \
    rosdep install --from-paths /home/dev/ros2_ws/src --ignore-src -r -y
USER dev

# ── Build workspace (errors now surface — no silent || true) ───────────
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install \
        --parallel-workers $(nproc) \
        --event-handlers console_direct+ \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# ── Shell setup ────────────────────────────────────────────────────────
RUN echo 'source /opt/ros/humble/setup.bash' >> /home/dev/.bashrc && \
    echo '[ -f /home/dev/ros2_ws/install/setup.bash ] && source /home/dev/ros2_ws/install/setup.bash' >> /home/dev/.bashrc && \
    echo 'export ROS_DOMAIN_ID=1' >> /home/dev/.bashrc && \
    echo 'export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST' >> /home/dev/.bashrc && \
    echo 'export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$HOME/.ignition/fuel/fuel.gazebosim.org/openrobotics/models' >> /home/dev/.bashrc && \
    echo 'export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:/opt/ros/humble/lib' >> /home/dev/.bashrc

CMD ["/bin/bash"]