"""
[B안] ROS2 모드 매니저 노드

- 토픽으로 모드 전환 명령을 받아서, 해당 모드 스크립트를 subprocess로 띄움
- 로봇 제어 코드(tracking.py, jog_tracking.py, wire_pick37.py)는 수정 불필요
- 웹 백엔드는 ROS2 토픽만 발행하면 됨 (rosbridge_websocket 등으로 연동)

[사용법]
  # 매니저 실행
  python3 mode_manager.py

  # 다른 터미널에서 모드 시작
  ros2 topic pub --once /mode_select std_msgs/String "data: 'mode1_tracking'"
  ros2 topic pub --once /mode_select std_msgs/String "data: 'mode2_jog'"
  ros2 topic pub --once /mode_select std_msgs/String "data: 'mode3_wire_pick'"

  # 현재 모드 종료 (다른 모드로 안 옮기고 그냥 끄기)
  ros2 topic pub --once /mode_select std_msgs/String "data: 'stop'"

  # 상태 확인 토픽 (매니저가 1초마다 발행)
  ros2 topic echo /mode_status
"""

import os
import signal
import subprocess
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ─── 모드 이름 → 실행할 스크립트 경로 ───
# 실제 경로에 맞게 수정해주세요
MODE_SCRIPTS = {
    "mode1_tracking":  "/home/soyoung/cobot_ws/src/pick_and_place_wire/pick_and_place_wire/tracking.py",
    "mode2_jog":       "/home/soyoung/cobot_ws/src/pick_and_place_wire/pick_and_place_wire/jog_tracking.py",
    "mode3_wire_pick": "/home/soyoung/cobot_ws/src/pick_and_place_wire/pick_and_place_wire/wire_pick40.py",
}

# SIGINT 보낸 뒤 정상 종료까지 기다리는 시간
SHUTDOWN_TIMEOUT = 5.0


class ModeManagerNode(Node):
    def __init__(self):
        super().__init__('mode_manager')

        self._proc: subprocess.Popen | None = None
        self._current_mode: str | None = None
        self._lock = threading.Lock()

        # ─── 모드 전환 명령 구독 ───
        self.create_subscription(
            String, '/mode_select', self._on_mode_select, 10
        )

        # ─── 현재 상태 발행 (1초 주기) ───
        self._status_pub = self.create_publisher(String, '/mode_status', 10)
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info("Mode Manager Node 시작 — /mode_select 토픽 대기 중")

    def _on_mode_select(self, msg: String):
        cmd = msg.data.strip()
        self.get_logger().info(f"[/mode_select] 수신: '{cmd}'")

        # 방어 로직: 이미 실행 중인 모드와 동일한 명령이 오면 무시
        if self._current_mode == cmd and self._is_alive():
            return
            
        # 방어 로직 2: 프로세스가 죽었는데, 똑같은 모드 명령이 들어온 경우 (중복 퍼블리시 방어)
        # 만약 진짜로 재실행하고 싶다면 웹에서 'stop'을 먼저 보내거나 다른 모드를 보냈다가 와야 함
        if self._current_mode == cmd and not self._is_alive():
            self.get_logger().info(f"'{cmd}' 모드는 이미 종료(또는 크래시) 상태입니다. 중복 실행을 막습니다.")
            return
        
        self.get_logger().info(f"[/mode_select] 수신: '{cmd}'")

        if cmd == "stop":
            self._stop_current()
            return

        if cmd not in MODE_SCRIPTS:
            self.get_logger().warn(f"알 수 없는 모드: '{cmd}'")
            return

        self._start_mode(cmd)

    def _start_mode(self, mode: str):
        with self._lock:
            # 같은 모드 이미 살아있으면 무시
            if self._current_mode == mode and self._is_alive():
                self.get_logger().info(f"'{mode}' 이미 실행 중 — 무시")
                return

            # 다른 모드 켜져 있으면 종료
            if self._is_alive():
                self.get_logger().info(
                    f"기존 모드 '{self._current_mode}' 종료 → '{mode}' 시작 준비"
                )
                self._stop_internal()

            script = MODE_SCRIPTS[mode]
            if not os.path.exists(script):
                self.get_logger().error(f"스크립트 없음: {script}")
                return

            env = os.environ.copy()
            try:
                self._proc = subprocess.Popen(["python3", script], env=env)
                self._current_mode = mode
                self.get_logger().info(
                    f"'{mode}' 시작됨 (pid={self._proc.pid})"
                )
            except Exception as e:
                self.get_logger().error(f"실행 실패: {e}")
                self._proc = None
                self._current_mode = None

    def _stop_current(self):
        with self._lock:
            if not self._is_alive():
                self.get_logger().info("실행 중인 모드 없음")
                self._proc = None
                self._current_mode = None
                return
            mode = self._current_mode
            self._stop_internal()
            self.get_logger().info(f"'{mode}' 종료됨")

    def _stop_internal(self):
        """SIGINT 보내고 정리. 락 잡힌 상태에서 호출."""
        if self._proc is None:
            return

        try:
            self._proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            self._proc = None
            self._current_mode = None
            return

        try:
            self._proc.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.get_logger().warn("SIGINT 타임아웃 — SIGKILL 전송")
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        self._proc = None
        self._current_mode = None

    def _is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _publish_status(self):
        alive = self._is_alive()
        msg = String()
        if alive:
            msg.data = f"running:{self._current_mode}:pid{self._proc.pid}"
        else:
            # 프로세스가 죽었으면 정리
            if self._proc is not None:
                self._proc = None
                self._current_mode = None
            msg.data = "idle"
        self._status_pub.publish(msg)

    def shutdown(self):
        """노드 종료 시 자식 프로세스도 정리."""
        self._stop_current()


def main():
    rclpy.init()
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
