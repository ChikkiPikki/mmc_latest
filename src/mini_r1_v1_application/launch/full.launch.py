"""Full stack launch — world selection + Gazebo sim + all application nodes.

Usage:
    ros2 launch mini_r1_v1_application full.launch.py
    ros2 launch mini_r1_v1_application full.launch.py select_world:=false
"""

import glob
import math
import os
import re
import shutil
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _find_src_worlds_dir():
    """Locate the source worlds/ directory from the installed share path."""
    pkg_share = get_package_share_directory("mini_r1_v1_description")
    ws_root = os.path.abspath(os.path.join(pkg_share, '..', '..', '..', '..'))
    src_worlds = os.path.join(ws_root, 'src', 'mini_r1_v1_description', 'worlds')
    if os.path.isdir(src_worlds):
        return src_worlds
    return None


def _copy_selected_world(src_worlds):
    """Copy the selected world's YAML + map into the active position."""
    sel_file = os.path.join(src_worlds, 'selected_world.txt')
    selected = 'multi_floor_college'
    if os.path.exists(sel_file):
        selected = open(sel_file).read().strip()

    source_dir = os.path.join(src_worlds, 'source', selected)
    if not os.path.isdir(source_dir):
        print(f"[full.launch] Source world dir not found: {source_dir}")
        return

    yamls = glob.glob(os.path.join(source_dir, '*.building.yaml'))
    if yamls:
        shutil.copy2(yamls[0], os.path.join(src_worlds, 'active_world.building.yaml'))
        print(f"[full.launch] Copied {yamls[0]}")

    for ext in ['jpeg', 'jpg', 'png']:
        src = os.path.join(source_dir, f'map.{ext}')
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(src_worlds, f'map.{ext}'))


def _regenerate_world(src_worlds):
    """Run the world generation pipeline entirely in Python."""
    yaml_path = os.path.join(src_worlds, 'active_world.building.yaml')
    output_world = os.path.join(src_worlds, 'output.world')
    models_dir = os.path.join(src_worlds, 'models')

    # 1. Copy selected world YAML + map
    _copy_selected_world(src_worlds)

    if not os.path.exists(yaml_path):
        print("[full.launch] No active_world.building.yaml found, skipping world gen")
        return

    # 2. Download Fuel models
    download_script = os.path.join(src_worlds, 'download_models.py')
    if os.path.exists(download_script):
        r = subprocess.run(['python3', download_script, yaml_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[full.launch] download_models warning: {r.stderr}")

    # 3. Run RMF building map generator
    print("[full.launch] Running building_map_generator...")
    r = subprocess.run([
        'ros2', 'run', 'rmf_building_map_tools', 'building_map_generator',
        'gazebo', yaml_path, output_world, models_dir
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[full.launch] building_map_generator FAILED (exit {r.returncode})")
        print(f"[full.launch]   stdout: {r.stdout[-500:] if r.stdout else '(empty)'}")
        print(f"[full.launch]   stderr: {r.stderr[-500:] if r.stderr else '(empty)'}")
        raise RuntimeError("building_map_generator failed")

    print("[full.launch] building_map_generator succeeded")

    # 4. Read output and fix plugin names for Fortress
    with open(output_world, 'r') as f:
        content = f.read()

    content = content.replace('libgz-sim-', 'libignition-gazebo-')
    content = content.replace('gz::sim::systems', 'ignition::gazebo::systems')

    # Fix reserved world name "world" → "generated_world" (Fortress rejects "world")
    content = content.replace('<world name="world">', '<world name="generated_world">')

    # Replace model://sun include with inline directional light (Fortress compatible)
    sun_pattern = re.compile(
        r'<include>\s*<uri>model://sun</uri>\s*</include>', re.DOTALL)
    sun_light = (
        '<light type="directional" name="sun">\n'
        '      <cast_shadows>true</cast_shadows>\n'
        '      <pose>0 0 10 0 0 0</pose>\n'
        '      <diffuse>0.8 0.8 0.8 1</diffuse>\n'
        '      <specular>0.2 0.2 0.2 1</specular>\n'
        '      <direction>0.5 0.1 -0.9</direction>\n'
        '    </light>'
    )
    content = sun_pattern.sub(sun_light, content)

    # 5. Inject Sensors plugin if missing
    if 'sensors-system' not in content:
        content = content.replace(
            '<plugin filename="libignition-gazebo-physics-system.so"',
            '    <plugin filename="libignition-gazebo-sensors-system.so" '
            'name="ignition::gazebo::systems::Sensors">\n'
            '      <render_engine>ogre2</render_engine>\n'
            '    </plugin>\n'
            '    <plugin filename="libignition-gazebo-physics-system.so"'
        )

    # 6. Inject IMU plugin if missing
    if 'imu-system' not in content:
        content = content.replace(
            '<plugin filename="libignition-gazebo-physics-system.so"',
            '    <plugin filename="libignition-gazebo-imu-system.so" '
            'name="ignition::gazebo::systems::Imu">\n'
            '    </plugin>\n'
            '    <plugin filename="libignition-gazebo-physics-system.so"'
        )

    with open(output_world, 'w') as f:
        f.write(content)

    # 7. Generate ArUco markers and floor symbols
    gen_script = os.path.join(src_worlds, 'generate_markers_and_signs.py')
    if os.path.exists(gen_script):
        r = subprocess.run(
            ['python3', gen_script, yaml_path, output_world, src_worlds],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[full.launch] generate_markers warning: {r.stderr}")

    print(f"[full.launch] World generated: {output_world}")


def _patch_world_for_fortress(world_path):
    """Apply Fortress-specific fixes to an existing output.world file.

    The RMF building_map_generator produces a world with NO system plugins.
    We must inject physics, sensors, IMU, scene broadcaster, user commands,
    and contact plugins. Also fix world name and sun model.
    """
    with open(world_path, 'r') as f:
        content = f.read()

    changed = False

    # Fix plugin names: gz-sim → ignition-gazebo
    if 'libgz-sim-' in content:
        content = content.replace('libgz-sim-', 'libignition-gazebo-')
        content = content.replace('gz::sim::systems', 'ignition::gazebo::systems')
        changed = True

    # Fix reserved world name
    if '<world name="world">' in content:
        content = content.replace('<world name="world">', '<world name="generated_world">')
        changed = True

    # Replace model://sun with inline directional light
    if 'model://sun' in content:
        sun_pattern = re.compile(
            r'<include>\s*<uri>model://sun</uri>\s*</include>', re.DOTALL)
        sun_light = (
            '<light type="directional" name="sun">\n'
            '      <cast_shadows>true</cast_shadows>\n'
            '      <pose>0 0 10 0 0 0</pose>\n'
            '      <diffuse>0.8 0.8 0.8 1</diffuse>\n'
            '      <specular>0.2 0.2 0.2 1</specular>\n'
            '      <direction>0.5 0.1 -0.9</direction>\n'
            '    </light>'
        )
        content = sun_pattern.sub(sun_light, content)
        changed = True

    # Inject ALL required system plugins if physics plugin is missing
    # (building_map_generator doesn't add any system plugins)
    if 'physics-system' not in content and 'Physics' not in content:
        system_plugins = (
            '\n'
            '    <physics name="1ms" type="ignored">\n'
            '      <max_step_size>0.001</max_step_size>\n'
            '      <real_time_factor>1.0</real_time_factor>\n'
            '    </physics>\n'
            '    <plugin\n'
            '      filename="ignition-gazebo-physics-system"\n'
            '      name="ignition::gazebo::systems::Physics">\n'
            '    </plugin>\n'
            '    <plugin\n'
            '      filename="ignition-gazebo-user-commands-system"\n'
            '      name="ignition::gazebo::systems::UserCommands">\n'
            '    </plugin>\n'
            '    <plugin\n'
            '      filename="ignition-gazebo-scene-broadcaster-system"\n'
            '      name="ignition::gazebo::systems::SceneBroadcaster">\n'
            '    </plugin>\n'
            '    <plugin\n'
            '      filename="ignition-gazebo-sensors-system"\n'
            '      name="ignition::gazebo::systems::Sensors">\n'
            '      <render_engine>ogre2</render_engine>\n'
            '    </plugin>\n'
            '    <plugin\n'
            '      filename="ignition-gazebo-contact-system"\n'
            '      name="ignition::gazebo::systems::Contact">\n'
            '    </plugin>\n'
            '    <plugin\n'
            '      filename="ignition-gazebo-imu-system"\n'
            '      name="ignition::gazebo::systems::Imu">\n'
            '    </plugin>\n'
        )
        # Insert after </scene> tag
        content = content.replace('</scene>', '</scene>' + system_plugins)
        changed = True
        print("[full.launch] Injected all Fortress system plugins")
    else:
        # Individual plugin checks for worlds that have some but not all
        if 'sensors-system' not in content and 'Sensors' not in content:
            content = content.replace(
                '</scene>',
                '</scene>\n'
                '    <plugin filename="ignition-gazebo-sensors-system"\n'
                '      name="ignition::gazebo::systems::Sensors">\n'
                '      <render_engine>ogre2</render_engine>\n'
                '    </plugin>\n'
            )
            changed = True
        if 'imu-system' not in content and '::Imu' not in content:
            content = content.replace(
                '</scene>',
                '</scene>\n'
                '    <plugin filename="ignition-gazebo-imu-system"\n'
                '      name="ignition::gazebo::systems::Imu">\n'
                '    </plugin>\n'
            )
            changed = True

    if changed:
        with open(world_path, 'w') as f:
            f.write(content)
        print(f"[full.launch] Patched world file: {world_path}")


def _sync_to_install(src_worlds):
    """Copy generated world files from source to install directory.

    With --symlink-install, src and install may be the same — skip if so.
    """
    install_worlds = os.path.join(
        get_package_share_directory("mini_r1_v1_description"), 'worlds')

    if os.path.realpath(install_worlds) == os.path.realpath(src_worlds):
        return

    for fname in ['output.world', 'active_world.building.yaml', 'selected_world.txt']:
        src = os.path.join(src_worlds, fname)
        dst = os.path.join(install_worlds, fname)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) and os.path.samefile(src, dst):
            continue
        shutil.copy2(src, dst)

    for subdir in ['generated', 'models']:
        src_sub = os.path.join(src_worlds, subdir)
        dst_sub = os.path.join(install_worlds, subdir)
        if not os.path.isdir(src_sub):
            continue
        if os.path.isdir(dst_sub) and os.path.samefile(src_sub, dst_sub):
            continue
        if os.path.isdir(dst_sub):
            shutil.rmtree(dst_sub)
        shutil.copytree(src_sub, dst_sub)


def _select_prepare_and_launch(context, *args, **kwargs):
    """Pre-launch: world selection + generation, then start sim + app."""
    select = LaunchConfiguration('select_world').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    src_worlds = _find_src_worlds_dir()

    if src_worlds is not None:
        marker_file = os.path.join(src_worlds, 'selected_world.txt')
        prev_world = ''
        if os.path.exists(marker_file):
            with open(marker_file) as f:
                prev_world = f.read().strip()

        # Run world selector GUI
        if select.lower() == 'true':
            selector_script = os.path.join(src_worlds, 'world_selector.py')
            if os.path.exists(selector_script):
                try:
                    subprocess.run(['python3', selector_script, src_worlds],
                                   check=True)
                except subprocess.CalledProcessError:
                    print("[full.launch] World selector cancelled, using current world")
                except Exception as e:
                    print(f"[full.launch] Could not run world selector: {e}")

        new_world = ''
        if os.path.exists(marker_file):
            with open(marker_file) as f:
                new_world = f.read().strip()

        output_world_path = os.path.join(src_worlds, 'output.world')
        needs_regen = (new_world != prev_world) or not os.path.exists(output_world_path)

        if needs_regen:
            print(f"[full.launch] Generating world '{new_world}'...")
            try:
                _regenerate_world(src_worlds)
            except Exception as e:
                print(f"[full.launch] World generation failed: {e}")
                print("[full.launch] Falling back to empty world")

        # Always apply Fortress patches to existing output.world
        if os.path.exists(output_world_path):
            _patch_world_for_fortress(output_world_path)

        # Sync to install
        try:
            _sync_to_install(src_worlds)
        except Exception as e:
            print(f"[full.launch] Sync warning: {e}")
    else:
        print("[full.launch] Source worlds dir not found, skipping world selection")

    # Resolve world path AFTER generation
    desc_pkg = get_package_share_directory('mini_r1_v1_description')
    output_world = os.path.join(desc_pkg, 'worlds', 'output.world')
    empty_world = os.path.join(desc_pkg, 'worlds', 'empty.sdf')
    world_path = output_world if os.path.exists(output_world) else empty_world
    print(f"[full.launch] Using world: {world_path}")

    gz_pkg = get_package_share_directory('mini_r1_v1_gz')
    app_pkg = get_package_share_directory('mini_r1_v1_application')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gz_pkg, 'launch', 'sim.launch.py')),
            launch_arguments={'world': world_path}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(app_pkg, 'launch', 'bringup.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('select_world', default_value='true',
                              description='Show world selector GUI before launching'),
        OpaqueFunction(function=_select_prepare_and_launch),
    ])
