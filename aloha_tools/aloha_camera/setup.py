from setuptools import find_packages, setup

package_name = 'aloha_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aloha',
    maintainer_email='aloha@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apriltag_localization_node = aloha_camera.apriltag_localization_node:main',
            'single_stream_node = aloha_camera.single_stream_node:main',
            'apriltag_frame_plot_node = aloha_camera.apriltag_frame_plot_node:main',
        ],
    },
)
