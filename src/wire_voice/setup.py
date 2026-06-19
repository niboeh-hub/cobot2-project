import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'wire_voice'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # [추가] 빌드 시 resource 폴더 안의 모든 파이썬(.py)과 데이터(.pt, .npy) 파일들을 패키지 공유 공간으로 복사합니다.
        (os.path.join('share', package_name, 'resource'), glob('resource/*.py')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.pt')),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.npy')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='heobin',
    maintainer_email='heobin8128@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'get_wire_keyword = wire_voice.get_wire_keyword:main',
            'voice_yolo_seg = wire_voice.voice_yolo_seg:main',
            'wire_arrangement_checker = wire_voice.wire_arrangement_checker:main',
            'arrangement_judge = wire_voice.arrangement_judge:main',
            'wire_pick = wire_voice.wire_pick:main',
            'mission_bridge = wire_voice.mission_bridge:main',
            'get_hand_command = wire_voice.get_hand_command:main',
            'hand_verify_bridge = wire_voice.hand_verify_bridge:main',
            'webcam_publisher = wire_voice.webcam_publisher:main',
            'wire_guide_node = wire_voice.wire_guide_node:main',
        ],
    },
)