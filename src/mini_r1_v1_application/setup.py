from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mini_r1_v1_application'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tanay',
    maintainer_email='minidevfun@gmail.com',
    description='Autonomous exploration and symbol detection for Mini R1 v1',
    license='MIT',
    entry_points={
        'console_scripts': [
            'coverage_tracker_node = mini_r1_v1_application.nodes.coverage_tracker_node:main',
            'pursuit_controller_node = mini_r1_v1_application.nodes.pursuit_controller_node:main',
            'rrt_planner_node = mini_r1_v1_application.nodes.rrt_planner_node:main',
            'sweep_planner_node = mini_r1_v1_application.nodes.sweep_planner_node:main',
            'symbol_detector_node = mini_r1_v1_application.nodes.symbol_detector_node:main',
            'aruco_localizer_node = mini_r1_v1_application.nodes.aruco_localizer_node:main',
            'mission_manager_node = mini_r1_v1_application.nodes.mission_manager_node:main',
            'dynamic_obstacle_node = mini_r1_v1_application.nodes.dynamic_obstacle_node:main',
        ],
    },
)
