"""ROKEY 시스템 — 모든 노드 한 번에 띄우는 런치 파일.

사용:
  ros2 launch wire_voice all.launch.py

별도 띄울 거 (런치 외):
  - Flask:  cd ~/rokey_web && python3 app.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    # ROS 도메인 84 박힘 (.bashrc에서 export됨)

    return LaunchDescription([

        # ── LLM 모드 음성 ──
        Node(
            package='wire_voice',
            executable='get_wire_keyword',
            name='wire_voice_node',
            output='screen',
        ),

        # ── 핸드/Shadow 모드 음성 ──
        Node(
            package='wire_voice',
            executable='get_hand_command',
            name='hand_command_node',
            output='screen',
        ),

        # ── 정렬 검사 노드 (AI팀 거 — 이제 사용자분이 띄움) ──
        Node(
            package='wire_voice',
            executable='wire_arrangement_checker',
            name='wire_arrangement_checker_node',
            output='screen',
        ),

        # ── LLM 모드 자동 검증 다리 ──
        Node(
            package='wire_voice',
            executable='arrangement_judge',
            name='arrangement_judge_node',
            output='screen',
        ),

        # ── LLM 모드 진행 다리 ──
        Node(
            package='wire_voice',
            executable='mission_bridge',
            name='mission_bridge_node',
            output='screen',
        ),

        # ── 핸드/Shadow 모드 검증 다리 ──
        Node(
            package='wire_voice',
            executable='hand_verify_bridge',
            name='hand_verify_bridge_node',
            output='screen',
        ),

        # ── 영상 → 웹 스트림 ──
        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            output='screen',
        ),

        # ── 브라우저 ↔ ROS 다리 (핸드 모드 스켈레톤용) ──
        ExecuteProcess(
            cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'],
            output='screen',
        ),
    ])