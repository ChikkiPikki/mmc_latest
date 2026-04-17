from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mini_r1_v1_round3'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.STL')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tanay',
    maintainer_email='minidevfun@gmail.com',
    description='Mini R1 Round 3 hackathon: online SLAM + Nav2 + AprilTag command fiducials',
    license='MIT',
    entry_points={
        'console_scripts': [
            'tag_command_node = mini_r1_v1_round3.nodes.tag_command_node:main',
            'tile_detector_node = mini_r1_v1_round3.nodes.tile_detector_node:main',
            'mission_manager_node = mini_r1_v1_round3.nodes.mission_manager_node:main',
            'nav_diagnostic_node = mini_r1_v1_round3.nodes.nav_diagnostic_node:main',
            'mpc_tracker_node = mini_r1_v1_round3.nodes.mpc_tracker_node:main',
            'logo_detector_node = mini_r1_v1_round3.nodes.logo_detector_node:main',
        ],
    },
)
