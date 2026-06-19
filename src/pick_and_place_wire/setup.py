from setuptools import find_packages, setup
import glob
import os

package_name = 'pick_and_place_wire'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ji-hyeon',
    maintainer_email='ji-hyeon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_move = pick_and_place_wire.robot_move:main',
            'detection = pick_and_place_wire.detection:main',
            'tracking = pick_and_place_wire.tracking:main',
            'jog_tracking = pick_and_place_wire.jog_tracking:main',
            'wire_pick = pick_and_place_wire.wire_pick40:main',
            'mode_manager = pick_and_place_wire.mode_manager:main',
        ],
    },
)
