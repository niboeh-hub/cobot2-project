"""핸드 모드 음성 명령 노드.

흐름:
  1. Firebase 'operation_state.hand_active' 감시
  2. true가 되면 -> 마이크 열고 계속 듣는 루프 시작
  3. "헬로 로키" 들으면 -> STT -> LLM이 명령 추출
  4. "start" 명령 -> /hand_start 토픽 발행 -> 계속 듣기
  5. "verify" 명령 -> /hand_verify 토픽 발행 -> 루프 종료
  6. hand_active=false 되면 -> 즉시 종료

[LLM 모드와 다른 점]
  - 한 사이클 끝나도 자동으로 안 끔. "verify"까지 두 명령 다 받아야 끝.
  - 마이크가 계속 켜져 있음. wakeup 다시 듣고 또 받는 루프.

[토픽]
  - /hand_start  (std_msgs/String, data="start") - 로봇팀 손 추적 노드 켜는 신호
  - /hand_verify (std_msgs/String, data="verify") - 검증 시작 신호
"""

import os
import sys
import time
import threading

import rclpy
import pyaudio
from rclpy.node import Node
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from voice_processing.MicController import MicController, MicConfig
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

# rokey_web의 db_logger 쓸 수 있게 경로 추가
sys.path.insert(0, '/home/soyoung/rokey_web')
import db_logger

# Firebase 직접 감시
import firebase_admin
from firebase_admin import db as fb_db

# ====== 환경 ======
PACKAGE_NAME = "voice_processing"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
RESOURCE_PATH = os.path.join(PACKAGE_PATH, "resource")
ENV_PATH = os.path.join(RESOURCE_PATH, ".env")
load_dotenv(dotenv_path=ENV_PATH)
openai_api_key = os.getenv("OPENAI_API_KEY")


class HandCommandNode(Node):
    """핸드 모드 음성 명령 감지 + 토픽 발행."""

    def __init__(self):
        super().__init__("hand_command_node")
        self.get_logger().info(f"Loaded ENV from: {ENV_PATH}")

        # ─── LLM 설정 ───
        # GPT가 사용자 발화에서 "start" / "verify" / "unknown" 셋 중 하나 뽑음
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.2,
            openai_api_key=openai_api_key,
        )
        prompt_content = """
            당신은 사용자의 음성 명령을 분류해야 합니다.

            <분류 카테고리>
            - "start" : 사용자가 시작/조종/움직임 시작 등을 요청
              예: "시작해", "조종 시작", "움직여", "스타트"
            - "verify" : 사용자가 검증/확인/검사를 요청
              예: "검증 시작해", "검사해", "확인해", "정렬 확인"
            - "unknown" : 위 둘 어디에도 안 맞음

            <출력 형식>
            - 단어 하나만 출력: start, verify, 또는 unknown
            - 다른 설명, 따옴표, 줄바꿈 없이.

            <사용자 입력>
            "{user_input}"

            출력:
        """
        self.prompt = PromptTemplate(
            input_variables=["user_input"], template=prompt_content,
        )
        self.chain = self.prompt | self.llm

        # ─── 마이크 설정 ───
        self.mic_config = MicConfig(
            chunk=12000, rate=48000, channels=1,
            record_seconds=10, fmt=pyaudio.paInt16,
            device_index=10, buffer_size=24000,
        )
        self.mic = MicController(config=self.mic_config)
        self.stt = STT(openai_api_key=openai_api_key)
        self.wakeup = WakeupWord(self.mic_config.buffer_size)

        # ─── ROS 토픽 publisher ───
        # /hand_start: 손 추적 시작 신호. 로봇팀이 구독.
        self.start_pub = self.create_publisher(String, '/hand_start', 10)
        # /hand_verify: 검증 시작 신호. 사용자분 다른 노드(또는 직접 검증)가 구독.
        self.verify_pub = self.create_publisher(String, '/hand_verify', 10)

        # ─── 상태 ───
        self._processing = False
        self._lock = threading.Lock()
        # 한 사이클(hand_active=true) 안에서 "start"를 이미 받았는지.
        # 같은 사이클에서 "start" 두 번 받지 않게.
        self._started = False

        # ─── Firebase 감시 ───
        self.state_ref = fb_db.reference('operation_state')
        self.state_ref.listen(self._on_state_change)

        self.get_logger().info(
            "HandCommandNode ready. Waiting for hand_active=true ..."
        )

    # ============================================================
    # Firebase 콜백
    # ============================================================
    def _on_state_change(self, event):
        """operation_state 변할 때마다 호출됨. hand_active를 본다."""
        data = event.data
        if not isinstance(data, dict):
            return
        hand_active = bool(data.get('hand_active', False))
        if not hand_active:
            return

        # 이미 처리 중이면 무시
        with self._lock:
            if self._processing:
                return
            self._processing = True
            self._started = False  # 새 사이클 시작이니 초기화

        # 별 스레드에서 루프 처리 (Firebase 콜백 막지 않으려고)
        threading.Thread(target=self._run_loop, daemon=True).start()

    # ============================================================
    # 메인 루프: hand_active 동안 계속 듣기
    # ============================================================
    def _run_loop(self):
        """hand_active가 true인 동안 — 명령 듣기를 계속 반복."""
        try:
            self._do_loop()
        except Exception as e:
            self.get_logger().error(f"loop error: {e}")
            self._safe_set_error(f"핸드 음성 처리 중 오류: {e}")
        finally:
            # 어떤 결과든 — hand_active 끄고 종료
            try:
                db_logger.update_state(hand_active=False, voice_status='')
            except Exception:
                pass
            with self._lock:
                self._processing = False

    def _do_loop(self):
        """진짜 루프. start 받고 → 작업 → verify 받으면 종료."""
        # 1) 마이크 열기 (한 번만)
        try:
            self.mic.open_stream()
            self.wakeup.set_stream(self.mic.stream)
        except OSError as e:
            self.get_logger().error(f"mic open failed: {e}")
            self._safe_set_error("마이크 열기 실패")
            return

        # 2) 반복: hand_active가 false 되거나 verify 받을 때까지
        while True:
            # 현재 상태 확인 — hand_active가 false면 즉시 종료
            cur = self.state_ref.get() or {}
            if not cur.get('hand_active', False):
                self.get_logger().info("hand_active=false 감지. 루프 종료.")
                return

            # 안내 메시지
            if not self._started:
                db_logger.update_state(
                    voice_status='헬로 로키 시작 명령 대기 중...'
                )
            else:
                db_logger.update_state(
                    voice_status='헬로 로키 검증 명령 대기 중...'
                )

            # ─── 헬로 로키 대기 (취소 체크하며) ───
            self.get_logger().info("Waiting for wakeword ...")
            woke = self._wait_wakeup_or_cancel()
            if not woke:
                self.get_logger().info("Wakeword 대기 중 취소됨.")
                return

            # ─── 깨어남 안내 ───
            self.get_logger().info("Wakeword 감지. 명령 받는 중 ...")
            db_logger.update_state(voice_status='✓ 헬로 로키 감지! 명령을 말하세요')
            time.sleep(1.5)
            db_logger.update_state(voice_status='🎙 듣는 중...')

            # ─── STT ───
            try:
                user_text = self.stt.speech2text()
            except Exception as e:
                self.get_logger().error(f"STT error: {e}")
                db_logger.update_state(voice_status='음성 인식 실패. 다시 말해주세요')
                time.sleep(2)
                continue   # 다시 듣기

            self.get_logger().info(f"STT: {user_text}")
            db_logger.update_state(voice_status='명령 분류 중...')

            # ─── LLM 명령 분류 ───
            cmd = self._classify(user_text)
            self.get_logger().info(f"분류 결과: {cmd}")

            # ─── 결과 처리 ───
            if cmd == "start":
                if self._started:
                    # 이미 시작했는데 또 "start" — 무시
                    db_logger.update_state(
                        last_voice_text=user_text,
                        voice_status='이미 시작됨. "검증 시작해"라고 말하세요',
                    )
                    time.sleep(2)
                    db_logger.update_state(voice_status='')
                    continue

                # 시작 토픽 발행
                self.start_pub.publish(String(data='start'))
                self._started = True
                db_logger.update_state(
                    phase='running',
                    last_voice_text=user_text,
                    voice_status='✓ 시작 신호 발행',
                    message='손 제스처로 로봇 조종 중',
                )
                self.get_logger().info("[발행] /hand_start: start")

                # 임무 시작 — SQLite에도 기록
                try:
                    op_id = db_logger.start_operation(['hand_mode'])
                    db_logger.log_voice(
                        operation_id=op_id, raw_text=user_text,
                        stt_result=user_text,
                        requested_sequence=['hand_mode'],
                    )
                except Exception as e:
                    self.get_logger().warn(f"임무 기록 실패: {e}")

                time.sleep(2)
                db_logger.update_state(voice_status='')
                # 루프 계속 — 이제 verify 대기

            elif cmd == "verify":
                if not self._started:
                    # 시작 안 했는데 verify — 무시
                    db_logger.update_state(
                        last_voice_text=user_text,
                        voice_status='먼저 "시작해"라고 말하세요',
                    )
                    time.sleep(2)
                    db_logger.update_state(voice_status='')
                    continue

                # 검증 토픽 발행
                self.verify_pub.publish(String(data='verify'))
                db_logger.update_state(
                    last_voice_text=user_text,
                    voice_status='✓ 검증 신호 발행',
                    message='정렬 검사 중...',
                )
                self.get_logger().info("[발행] /hand_verify: verify")
                time.sleep(1.5)
                # 루프 종료 — finally에서 hand_active 끄임
                return

            else:
                # unknown
                db_logger.update_state(
                    last_voice_text=user_text,
                    voice_status='⚠ 명령을 이해 못함. 다시 말해주세요',
                )
                self.get_logger().info("명령 분류 실패. 다시 듣기.")
                time.sleep(3)
                db_logger.update_state(voice_status='')
                # 루프 계속

    # ============================================================
    # 헬로 로키 대기 (취소 체크)
    # ============================================================
    def _wait_wakeup_or_cancel(self):
        """헬로 로키 대기. hand_active=false 되면 중단."""
        while True:
            cur = self.state_ref.get() or {}
            if not cur.get('hand_active', False):
                return False
            if self.wakeup.is_wakeup():
                return True

    # ============================================================
    # LLM 분류
    # ============================================================
    def _classify(self, user_text):
        """LLM으로 'start' / 'verify' / 'unknown' 셋 중 하나."""
        try:
            response = self.chain.invoke({"user_input": user_text})
            out = response.content.strip().lower()
        except Exception as e:
            self.get_logger().error(f"LLM error: {e}")
            return "unknown"

        # 출력에서 키워드 찾기 (LLM이 다른 글자도 같이 뱉을 수 있음)
        if "start" in out:
            return "start"
        if "verify" in out:
            return "verify"
        return "unknown"

    # ============================================================
    # 에러는 active_error에 띄움
    # ============================================================
    def _safe_set_error(self, msg):
        try:
            db_logger.set_error(msg, source='hand_command')
        except Exception:
            pass


def main():
    rclpy.init()
    node = HandCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()