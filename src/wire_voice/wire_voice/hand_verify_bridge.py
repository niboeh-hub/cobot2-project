"""
hand_verify_bridge.py
======================

핸드 모드 전용 검증 다리 노드.

[하는 일]
  - /hand_verify 토픽 구독
  - 받으면 -> 1초 대기 (영상 안정) -> wire_arrangement_checker 서비스 호출
  - 결과(PASS/FAIL)에 따라 -> end_operation + update_state(phase='success'/'aborted')

[왜 별도 노드가 필요한가]
  - LLM 모드에선 arrangement_judge가 Firebase connected_sequence 감시로 자동 트리거.
  - 핸드 모드엔 그 신호가 없음. 대신 사용자가 "검증 시작해" 음성으로 트리거.
  - 그 토픽을 받아서 검증 흐름을 시작시킬 다리가 필요함.

[실행]
  ros2 run wire_voice hand_verify_bridge
"""

import sys
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

# db_logger 임포트
sys.path.insert(0, '/home/soyoung/rokey_web')
import db_logger


class HandVerifyBridgeNode(Node):
    """핸드 모드 검증 토픽 → 정렬 검사 서비스 호출."""

    SETTLE_SECONDS = 3.0   # 검증 호출 전 영상 안정용 대기

    def __init__(self):
        super().__init__('hand_verify_bridge_node')

        # ─── 정렬 검사 서비스 클라이언트 ───
        # wire_arrangement_checker가 제공하는 서비스를 호출할 거.
        self.cli = self.create_client(Trigger, '/verify_wire_arrangement')

        # 서비스가 떠 있나 한 번 확인 (없으면 경고만 하고 계속)
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                "verify_wire_arrangement 서비스가 아직 안 떴습니다. "
                "그래도 listen은 계속합니다."
            )

        # ─── /hand_verify 토픽 구독 ───
        self.create_subscription(
            String, '/hand_verify',
            self._on_hand_verify, 10
        )

        # ─── 한 번에 한 검증만 (중복 방지) ───
        self._verifying = False
        self._lock = threading.Lock()
        self._verified_op_ids = set()   # 이미 검증한 임무

        self.get_logger().info("HandVerifyBridgeNode ready.")
        self.get_logger().info("  - 구독: /hand_verify (핸드 모드 검증 신호)")

    # ============================================================
    # 토픽 콜백
    # ============================================================
    def _on_hand_verify(self, msg):
        """/hand_verify 받으면 검증 흐름 시작."""
        self.get_logger().info(f"[수신] /hand_verify: {msg.data}")

        # 한 번에 한 검증만
        with self._lock:
            if self._verifying:
                self.get_logger().warn("이미 검증 중. 중복 무시.")
                return
            self._verifying = True

        # 별 스레드에서 진행 (콜백 막지 않으려고)
        threading.Thread(target=self._run_verify, daemon=True).start()

    # ============================================================
    # 검증
    # ============================================================
    def _run_verify(self):
        try:
            self._do_verify()
        except Exception as e:
            self.get_logger().error(f"verify error: {e}")
            self._safe_set_error(f"검증 중 오류: {e}")
        finally:
            with self._lock:
                self._verifying = False

    def _do_verify(self):
        # 현재 진행 중인 임무 찾기
        ops = db_logger.get_all_operations()
        if not ops:
            self.get_logger().warn("진행 중인 임무를 찾을 수 없음.")
            return

        op = ops[0]
        op_id = op['operation_id']
        if op['result'] != 'running':
            self.get_logger().info(
                f"임무 {op_id}는 이미 종료됨 ({op['result']}). 검증 생략."
            )
            return
        if op_id in self._verified_op_ids:
            self.get_logger().info(f"임무 {op_id}는 이미 검증함. 생략.")
            return

        self._verified_op_ids.add(op_id)

        # 영상 안정 대기 - phase를 verifying으로 박아서 웹이 영상 토픽 바꿀 수 있게
        db_logger.update_state(
            phase='verifying',
            message='정렬 검사 중...',
        )
        self.get_logger().info(
            f"임무 {op_id} 검증 시작. {self.SETTLE_SECONDS}초 대기 ..."
        )
        time.sleep(self.SETTLE_SECONDS)

        # 서비스 호출
        if not self.cli.service_is_ready():
            self.get_logger().error("verify_wire_arrangement 서비스 미준비.")
            self._safe_set_error("정렬 검사 서비스 연결 실패")
            return

        future = self.cli.call_async(Trigger.Request())
        # 응답 대기 (간단 폴링, 10초 타임아웃)
        timeout = 10.0
        t0 = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - t0 > timeout:
                self.get_logger().error("정렬 검사 응답 타임아웃.")
                self._safe_set_error("정렬 검사 응답 없음")
                return
            time.sleep(0.05)

        resp = future.result()
        if resp is None:
            self.get_logger().error("정렬 검사 응답 None.")
            self._safe_set_error("정렬 검사 응답 None")
            return

        self.get_logger().info(
            f"검사 결과: success={resp.success}, msg={resp.message}"
        )

        # 검증 영상을 사용자가 충분히 볼 수 있게 대기
        time.sleep(3.0)

        # 임무 종료 처리
        if resp.success:
            db_logger.end_operation(op_id, result='success')
            db_logger.update_state(
                phase='success',
                message='✓ 정렬 성공! 임무 완료',
            )
            self.get_logger().info(f"임무 {op_id} 성공으로 종료.")
        else:
            db_logger.end_operation(
                op_id, result='aborted',
                reason='wrong_arrangement',
            )
            db_logger.update_state(
                phase='aborted',
                message=f'✗ 정렬 실패: {resp.message}',
            )
            self.get_logger().info(f"임무 {op_id} 정렬 실패로 중단.")

    def _safe_set_error(self, msg):
        try:
            db_logger.set_error(msg, source='hand_verify_bridge')
        except Exception:
            pass


def main():
    rclpy.init()
    node = HandVerifyBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()