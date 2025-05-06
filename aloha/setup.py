from glob import glob
import os

from setuptools import (
    find_packages,
    setup,
)

package_name = 'aloha'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude='test'),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch/*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config', 'moveit'), glob('config/moveit/*.yaml')),
        (os.path.join('share', package_name, 'config', 'moveit', 'controllers'), glob('config/moveit/controllers/*.yaml')),
        (os.path.join('share', package_name, 'config', 'moveit', 'joint_limits'), glob('config/moveit/joint_limits/*.yaml')),
        (os.path.join('share', package_name, 'config', 'moveit', 'srdf'), glob('config/moveit/srdf/*.yaml')),
        (os.path.join('share', package_name, 'config', 'moveit', 'srdf'), glob('config/moveit/srdf/*.srdf')),
        (os.path.join('share', package_name, 'config', 'moveit', 'srdf'), glob('config/moveit/srdf/*.xacro')),
        (os.path.join('share', package_name, 'config', 'moveit', 'urdf'), glob('config/moveit/urdf/*.yaml')),
        (os.path.join('share', package_name, 'config', 'moveit', 'urdf'), glob('config/moveit/urdf/*.urdf')),
        (os.path.join('share', package_name, 'config', 'moveit', 'urdf'), glob('config/moveit/urdf/*.xacro')),
        (os.path.join('share', package_name, 'config', 'apriltag', 'patterns'), glob('config/apriltag/patterns/*.yaml')),
        (os.path.join('share', package_name, 'config', 'apriltag', 'tags'), glob('config/apriltag/tags/*.yaml')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author_email='tonyzhao@stanford.edu',
    author='Tony Zhao',
    maintainer='Trossen Robotics',
    maintainer_email='trsupport@trossenrobotics.com',
    description='ALOHA: A Low-cost Open-source Hardware System for Bimanual Teleoperation',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
