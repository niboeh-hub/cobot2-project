"""정렬 심판 노드 (polling 버전).

흐름:
  1. 0.5초마다 Firebase operation_state 폴링
  2. phase가 'running'이고 connected_sequence 길이가 requested_sequence와 같아지면
     -> '다 꽂힘' 판정
  3. 1초 대기 (영상 안정)
  4. /verify_wire_arrangement 서비스 호출
  5. PASS  -> end_operation(success)
     FAIL  -> end_operation(aborted, reason='wrong_arrangement')

[변경 이력]
  - Firebase state_ref.listen() 이 가끔 콜백 발동 안 하는 문제 때문에
    polling 방식으로 변경 (0.5초 주기로 직접 읽음).
"""

import sys
import time
import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# rokey_web의 db_logger 사용
sys.path.insert(0, '/home/soyoung/rokey_web')
import db_logger

from firebase_admin import db as fb_db


class ArrangementJudgeNode(Node):
    """임무 끝나는 시점에 자동으로 정렬을 검사하고 결과를 기록한다."""

    SETTLE_SECONDS = 3.0   # 다 꽂힌 뒤 영상 안정용 대기
    POLL_PERIOD = 0.5      # Firebase 폴링 주기

    def __init__(self):
        super().__init__('arrangement_judge_node')

        # 서비스 클라이언트
        self.cli = self.create_client(Trigger, '/verify_wire_arrangement')
        self.get_logger().info("Waiting for /verify_wire_arrangement service ...")
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                "verify_wire_arrangement 서비스가 아직 안 떴습니다. "
                "그래도 listen은 계속합니다."
            )

        # 한 번에 한 임무만 판정 (중복 방지)
        self._judging = False
        self._lock = threading.Lock()
        self._judged_op_ids = set()   # 이미 판정한 임무

        # Firebase 폴링
        self.state_ref = fb_db.reference('operation_state')
        self._poll_timer = self.create_timer(self.POLL_PERIOD, self._poll_state)

        self.get_logger().info("ArrangementJudgeNode ready (polling 모드).")

    # ---------------------------------------------------------------
    # Firebase 폴링 — 0.5초마다
    # ---------------------------------------------------------------
    def _poll_state(self):
        try:
            data = self.state_ref.get()
        except Exception as e:
            self.get_logger().warn(f"Firebase 읽기 실패: {e}")
            return

        if not isinstance(data, dict):
            return

        phase = data.get('phase')
        if phase != 'running':
            return

        req = data.get('requested_sequence') or []
        conn = data.get('connected_sequence') or []
        if not req or len(conn) < len(req):
            return   # 아직 다 꽂히지 않음

        # 다 꽂혔다 -> 판정 시작 (한 번만)
        with self._lock:
            if self._judging:
                return
            self._judging = True

        threading.Thread(target=self._run_judgement, daemon=True).start()

    # ---------------------------------------------------------------
    # 판정
    # ---------------------------------------------------------------
    def _run_judgement(self):
        try:
            self._do_judgement()
        except Exception as e:
            self.get_logger().error(f"judgement error: {e}")
            self._safe_set_error(f"정렬 판정 중 오류: {e}")
        finally:
            with self._lock:
                self._judging = False

    def _do_judgement(self):
        # 현재 상태에서 op_id 알아내기 (가장 최근 임무)
        ops = db_logger.get_all_operations()
        if not ops:
            self.get_logger().warn("진행 중인 임무를 찾을 수 없음.")
            return

        op = ops[0]
        op_id = op['operation_id']
        if op['result'] != 'running':
            self.get_logger().info(
                f"임무 {op_id}는 이미 종료됨 ({op['result']}). 판정 생략."
            )
            return
        if op_id in self._judged_op_ids:
            self.get_logger().info(f"임무 {op_id}는 이미 판정함. 생략.")
            return

        self._judged_op_ids.add(op_id)

        # phase를 'verifying'으로 — 웹이 영상 토픽 바꾸도록 신호
        db_logger.update_state(
            phase='verifying',
            message='정렬 검사 중...',
        )
        self.get_logger().info(f"임무 {op_id} 판정 시작. {self.SETTLE_SECONDS}초 대기 ...")
        time.sleep(self.SETTLE_SECONDS)

        # 서비스 호출
        if not self.cli.service_is_ready():
            self.get_logger().error("verify_wire_arrangement 서비스 미준비.")
            self._safe_set_error("정렬 검사 서비스 연결 실패")
            return

        future = self.cli.call_async(Trigger.Request())
        # 서비스 응답 대기 (간단히 폴링)
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

        self.get_logger().info(f"검사 결과: success={resp.success}, msg={resp.message}")

        # 검증 영상을 사용자가 충분히 볼 수 있게 대기
        time.sleep(2.0)

        # 임무 종료 처리
        connected = (db_logger.get_state() or {}).get('connected_sequence', []) or []

        if resp.success:
            # 성공
            db_logger.end_operation(
                op_id, result='success', connected_sequence=connected
            )
            db_logger.update_state(
                phase='success',
                message='✓ 정렬 성공! 임무 완료',
            )
            self.get_logger().info(f"임무 {op_id} 성공으로 종료.")
        else:
            # 실패
            db_logger.end_operation(
                op_id, result='aborted',
                reason='wrong_arrangement',
                connected_sequence=connected,
            )
            db_logger.update_state(
                phase='aborted',
                message=f'✗ 정렬 실패: {resp.message}',
            )
            self.get_logger().info(f"임무 {op_id} 정렬 실패로 중단.")

    def _safe_set_error(self, msg):
        try:
            db_logger.set_error(msg, source='arrangement_judge')
        except Exception:
            pass


def main():
    rclpy.init()
    node = ArrangementJudgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()