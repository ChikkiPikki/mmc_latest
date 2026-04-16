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
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/worlds/world_selector.py ${CMAKE_CURRENT_SOURCE_DIR}/worlds
  COMMENT "Launching world selector GUI..."
  VERBATIM
)

# ── World generation pipeline (runs on every build, non-fatal) ──
# Wrapped in a single bash script so failure doesn't block colcon build.
# Run manually if needed: colcon build --packages-select mini_r1_v1_description --cmake-target generate_world
add_custom_target(generate_world ALL
  COMMAND bash -c "\
    set -x $<SEMICOLON> \
    W='${CMAKE_CURRENT_SOURCE_DIR}/worlds' $<SEMICOLON> \
    python3 -c \"import os,glob,shutil; w='$$W'; s=open(os.path.join(w,'selected_world.txt')).read().strip() if os.path.exists(os.path.join(w,'selected_world.txt')) else 'multi_floor_college'; d=os.path.join(w,'source',s); yamls=glob.glob(os.path.join(d,'*.building.yaml')); shutil.copy2(yamls[0],os.path.join(w,'active_world.building.yaml')) if yamls else None; [shutil.copy2(os.path.join(d,'map.'+e),os.path.join(w,'map.'+e)) for e in ['jpeg','jpg','png'] if os.path.exists(os.path.join(d,'map.'+e))]\" $<SEMICOLON> \
    python3 $$W/download_models.py $$W/active_world.building.yaml $<SEMICOLON> \
    source /opt/ros/humble/setup.bash 2>/dev/null || true $<SEMICOLON> \
    source ${CMAKE_INSTALL_PREFIX}/../../install/setup.bash 2>/dev/null || source ${CMAKE_INSTALL_PREFIX}/../setup.bash 2>/dev/null || true $<SEMICOLON> \
    ros2 run rmf_building_map_tools building_map_generator gazebo $$W/active_world.building.yaml $$W/output.world $$W/models $<SEMICOLON> \
    sed -i 's/libgz-sim-/libignition-gazebo-/g; s/gz::sim::systems/ignition::gazebo::systems/g' $$W/output.world $<SEMICOLON> \
    if ! grep -q 'sensors-system' $$W/output.world; then sed -i '/<plugin filename=\"libignition-gazebo-physics-system.so\"/i \\    <plugin filename=\"libignition-gazebo-sensors-system.so\" name=\"ignition::gazebo::systems::Sensors\">\\n      <render_engine>ogre2</render_engine>\\n    </plugin>' $$W/output.world; fi $<SEMICOLON> \
    if ! grep -q 'imu-system' $$W/output.world; then sed -i '/<plugin filename=\"libignition-gazebo-physics-system.so\"/i \\    <plugin filename=\"libignition-gazebo-imu-system.so\" name=\"ignition::gazebo::systems::Imu\">\\n    </plugin>' $$W/output.world; fi $<SEMICOLON> \
    python3 $$W/generate_markers_and_signs.py $$W/active_world.building.yaml $$W/output.world $$W \
    || echo '[generate_world] WARNING: World generation failed — run manually inside container' \
  "
  COMMENT "Generating Gazebo world from active_world.building.yaml..."
  VERBATIM
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
