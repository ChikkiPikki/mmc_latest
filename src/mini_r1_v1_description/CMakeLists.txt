cmake_minimum_required(VERSION 3.8)
project(mini_r1_v1_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_sim REQUIRED)

# ── World selector GUI (run manually: colcon build --packages-select mini_r1_v1_description --cmake-target select_world) ──
add_custom_target(select_world
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/worlds/scripts/world_selector.py ${CMAKE_CURRENT_SOURCE_DIR}/worlds
  COMMENT "Launching world selector GUI..."
  VERBATIM
)

# ── World generation pipeline (runs on every build, non-fatal) ──
# Uses a separate shell script to avoid CMake quoting issues.
# Run manually: bash src/mini_r1_v1_description/worlds/scripts/generate_world.sh <worlds_dir>
add_custom_target(generate_world ALL
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/worlds/scripts/generate_world.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/worlds
          || echo "[generate_world] WARNING: World generation failed"
  COMMENT "Generating Gazebo world from active_world.building.yaml..."
)

install(
  DIRECTORY config launch meshes urdf worlds
  DESTINATION share/${PROJECT_NAME}
)
  
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()
