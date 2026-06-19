"""음성 명령 노드 (새 시나리오).

흐름:
  1. Firebase 'operation_state.mic_active' 를 감시
  2. true 가 되면 -> 마이크 열고 "헬로 로키" 기다림
  3. 깨어나면 -> 10초 안에 색 명령(예: "빨강 초록 연결해") 듣기
  4. LLM 이 여러 색을 '순서대로' 추출
  5. 색 받으면 -> db_logger.start_operation([...]) + update_state
  6. 실패(매칭 없음/타임아웃) -> mic_active 끄고 종료
  7. 다시 mic_active 가 true 가 될 때까지 대기
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

# rokey_web의 db_logger를 쓸 수 있게 경로 추가
sys.path.insert(0, '/home/soyoung/rokey_web/')
import db_logger

# Firebase 직접 감시 (mic_active 변화 감지)
import firebase_admin
from firebase_admin import db as fb_db

# ====== 환경 ======
PACKAGE_NAME = "voice_processing"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
RESOURCE_PATH = os.path.join(PACKAGE_PATH, "resource")
ENV_PATH = os.path.join(RESOURCE_PATH, ".env")
load_dotenv(dotenv_path=ENV_PATH)
openai_api_key = os.getenv("OPENAI_API_KEY")

# ====== 설정 ======
LISTEN_TIMEOUT_SEC = 10   # 헬로 로키 후 색 명령 듣는 제한 시간

VALID_COLORS = {"red", "green", "blue"}


class WireVoiceNode(Node):
    """Firebase mic_active 감시 + 음성 처리 + LLM."""

    def __init__(self):
        super().__init__("wire_voice_node")
        self.get_logger().info(f"Loaded ENV from: {ENV_PATH}")

        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.2,
            openai_api_key=openai_api_key,
        )
        prompt_content = """
            당신은 사용자의 문장에서 '연결할 선의 색상 순서'를 추출합니다.

            <대상 색상>
            - 빨강(빨간선/빨강선/red), 초록(초록선/녹색선/green), 파랑(파란선/파랑선/blue)

            <추출 규칙>
            - 사용자가 말한 '순서대로' 추출하세요.
            - 같은 색이 여러 번 나오면 중복 그대로 유지.
            - 대상 색이 하나도 없으면 빈 결과.

            <출력 형식>
            - 영문 색상 키워드만, 콤마로 구분된 한 줄.
              가능한 값: red, green, blue
            - 예시:
              "빨강 초록 파랑 연결해"  -> red,green,blue
              "초록부터 뽑아"          -> green
              "파란선 빨간선 순서로"    -> blue,red
              "안녕하세요"             -> (빈 줄)

            <사용자 입력>
            "{user_input}"

            출력:
        """
        self.prompt = PromptTemplate(
            input_variables=["user_input"], template=prompt_content,
        )
        self.chain = self.prompt | self.llm

        # 마이크
        self.mic_config = MicConfig(
            chunk=12000, rate=48000, channels=1,
            record_seconds=10, fmt=pyaudio.paInt16,
            device_index=10, buffer_size=24000,
        )
        self.mic = MicController(config=self.mic_config)
        self.stt = STT(openai_api_key=openai_api_key)
        self.wakeup = WakeupWord(self.mic_config.buffer_size)

        # 상태
        self._processing = False   # 현재 한 사이클 처리 중인지
        self._lock = threading.Lock()

        # Firebase 감시 (db_logger가 이미 초기화해놨음)
        self.state_ref = fb_db.reference('operation_state')
        self.state_ref.listen(self._on_state_change)

        # 추출된 키워드를 보내기 위한 토픽 퍼블리셔
        self.keyword_pub = self.create_publisher(String, '/extracted_wire_keywords', 10)

        self.get_logger().info("WireVoiceNode ready. Waiting for mic_active=true ...")

    # ---------------------------------------------------------------
    # Firebase 콜백
    # ---------------------------------------------------------------
    def _on_state_change(self, event):
        """operation_state가 바뀔 때마다 호출됨. mic_active를 본다."""
        data = event.data
        if not isinstance(data, dict):
            return
        mic_active = bool(data.get('mic_active', False))
        if not mic_active:
            return

        with self._lock:
            if self._processing:
                return    # 이미 한 사이클 처리 중이면 무시
            self._processing = True

        # 새 스레드에서 처리 (Firebase 콜백을 막지 않기 위해)
        threading.Thread(target=self._run_cycle, daemon=True).start()

    # ---------------------------------------------------------------
    # 한 사이클: 헬로 로키 -> 색 명령 -> 임무 시작 or 종료
    # ---------------------------------------------------------------
    def _run_cycle(self):
        try:
            self._do_cycle()
        except Exception as e:
            self.get_logger().error(f"cycle error: {e}")
            self._safe_set_error(f"음성 처리 중 오류: {e}")
        finally:
            # 어떤 결과든 마이크는 끈다
            db_logger.update_state(mic_active=False)
            with self._lock:
                self._processing = False

    def _do_cycle(self):
        # 1) 마이크 열기
        try:
            self.mic.open_stream()
            self.wakeup.set_stream(self.mic.stream)
        except OSError as e:
            self.get_logger().error(f"mic open failed: {e}")
            self._safe_set_error("마이크 열기 실패")
            return

        # 2) 헬로 로키 대기
        self.get_logger().info("Waiting for wakeword 'hello rokey' ...")
        db_logger.update_state(voice_status='헬로 로키 대기 중...')

        woke_up = self._wait_wakeup_or_cancel()
        if not woke_up:
            self.get_logger().info("Wakeword 대기 중 취소됨.")
            db_logger.update_state(voice_status='')
            return

        # 3) 깨어남! 사용자가 알 수 있게 메시지 띄움
        self.get_logger().info("Wakeword 감지. 명령 받는 중 ...")
        db_logger.update_state(voice_status='✓ 헬로 로키 감지! 명령을 말하세요')
        time.sleep(1.5)   # 사용자가 메시지 볼 시간

        # 듣기 시작 안내
        db_logger.update_state(voice_status='🎙 듣는 중... (말씀하세요)')
        try:
            user_text = self.stt.speech2text()
        except Exception as e:
            self.get_logger().error(f"STT error: {e}")
            self._safe_set_error("음성 인식 실패")
            db_logger.update_state(voice_status='')
            return

        self.get_logger().info(f"STT: {user_text}")
        db_logger.update_state(voice_status='색 추출 중...')

        # 4) LLM으로 색 순서 추출
        sequence = self._extract_sequence(user_text)
        self.get_logger().info(f"추출된 색 순서: {sequence}")

        # 5) 결과 처리
        if not sequence:
            db_logger.update_state(
                last_voice_text=user_text,
                voice_status='색 명령을 인식하지 못했습니다',
            )
            self.get_logger().info("색 매칭 실패.")
            time.sleep(2)
            db_logger.update_state(voice_status='')
            return
        
        # 5-1) 중복 색상 검사
        if len(sequence) != len(set(sequence)):
            db_logger.update_state(
                last_voice_text=user_text,
                voice_status='중복된 색상이 있습니다. 다시 말씀해주세요.',
            )
            self.get_logger().info(f"중복 색상 감지됨: {sequence}")
            time.sleep(4)
            db_logger.update_state(voice_status='')
            return

        # 6) 임무 시작
        op_id = db_logger.start_operation(sequence)
        db_logger.update_state(
            phase='running',
            requested_sequence=sequence,
            connected_sequence=[],
            current_index=0,
            current_target=sequence[0],
            last_voice_text=user_text,
            message=f'{sequence[0].upper()} 잡으러 가는 중',
            voice_status='',   # 임무 시작했으니 음성 상태 비움
        )
        db_logger.log_voice(
            operation_id=op_id,
            raw_text=user_text,
            stt_result=user_text,
            requested_sequence=sequence,
        )
        self.get_logger().info(
            f"임무 시작: op_id={op_id}, sequence={sequence}"
        )
        
        # 토픽으로 추출된 색상 순서 퍼블리시
        msg = String()
        msg.data = ",".join(sequence)
        self.keyword_pub.publish(msg)
        self.get_logger().info(f"Published extracted keywords to topic: {msg.data}")
       
    # ---------------------------------------------------------------
    # 헬로 로키 대기: mic_active가 false로 바뀌면 중단
    # ---------------------------------------------------------------
    def _wait_wakeup_or_cancel(self):
        """헬로 로키를 기다린다. 도중에 mic_active가 false면 중단.

        반환: 깨어났으면 True, 취소되면 False.
        """
        while True:
            # 취소 체크: Firebase 상태 다시 읽기
            cur = self.state_ref.get() or {}
            if not cur.get('mic_active', False):
                return False
            # 깨우기 시도 (한 청크 분량)
            if self.wakeup.is_wakeup():
                return True

    # ---------------------------------------------------------------
    # LLM 색 순서 추출
    # ---------------------------------------------------------------
    def _extract_sequence(self, user_text):
        """LLM이 한 줄로 반환. 'red,green,blue' 같은 거. 빈 줄이면 []."""
        try:
            response = self.chain.invoke({"user_input": user_text})
            out = response.content.strip()
        except Exception as e:
            self.get_logger().error(f"LLM error: {e}")
            return []

        self.get_logger().info(f"LLM raw: {out!r}")
        if not out:
            return []

        # 콤마/공백 둘 다 허용해서 쪼개기
        out = out.replace('\n', ' ').replace(' ', ',')
        items = [s.strip().lower() for s in out.split(',') if s.strip()]
        # 유효한 색만 유지, 순서 유지
        cleaned = [c for c in items if c in VALID_COLORS]
        return cleaned

    # ---------------------------------------------------------------
    # 에러는 active_error 에 띄움
    # ---------------------------------------------------------------
    def _safe_set_error(self, msg):
        try:
            db_logger.set_error(msg, source='wire_voice')
        except Exception:
            pass


def main():
    rclpy.init()
    node = WireVoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()