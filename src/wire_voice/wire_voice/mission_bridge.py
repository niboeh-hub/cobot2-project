"""
mission_bridge.py
=================

로봇 노드가 발행하는 /wire_connected 토픽을 듣고,
Firebase의 진행 상황을 갱신해주는 다리 노드.

[배경]
  - 로봇 노드(wire_pick)는 Firebase 안 만지고, ROS 토픽만 다룸.
  - Firebase에 진행 상황 박는 책임은 이 노드가 들고 있음.
  - 그래야 분담 깔끔.

[하는 일]
  - /wire_connected 토픽 구독.
  - 받을 때마다 — Firebase의 connected_sequence 에 색 추가,
    current_index, current_target, message 갱신.
  - 웹 임무 페이지가 자동으로 진행도 갱신.

[실행]
  ros2 run wire_voice mission_bridge
"""

# ─── 기본 모듈 ───
import sys                       # sys.path 조작용
import rclpy                     # ROS 2 파이썬 라이브러리
from rclpy.node import Node      # 노드 만들 때 상속할 부모 클래스
from std_msgs.msg import String  # 토픽 메시지 타입 (간단한 문자열)

# ─── db_logger 임포트 ───
# db_logger.py는 rokey_web 폴더에 있음.
# 그쪽 경로를 파이썬 import 검색 경로에 추가해서 import 가능하게 함.
# 이 한 줄로 — 다른 폴더에 있는 파일도 import할 수 있게 됨.
sys.path.insert(0, '/home/soyoung/rokey_web')
import db_logger


class MissionBridgeNode(Node):
    """로봇의 진행 토픽을 받아서 Firebase에 박는 다리 노드."""

    def __init__(self):
        # 부모 클래스(Node) 초기화. 노드 이름을 ROS에 등록.
        # 이 이름은 'ros2 node list'에서 보임.
        super().__init__('mission_bridge_node')

        # ─── 토픽 구독 ───
        # /wire_connected 토픽을 듣기 시작.
        # 메시지가 도착할 때마다 self._on_wire_connected 가 자동으로 호출됨.
        #
        # 인자 4개:
        #   - String: 메시지 타입. 위에서 import한 거.
        #   - '/wire_connected': 토픽 이름. 로봇팀이랑 합의한 이름.
        #   - self._on_wire_connected: 메시지 올 때 호출될 함수(콜백).
        #   - 10: 큐 크기. 한 번에 10개까지 버퍼링.
        self.create_subscription(
            String, '/wire_connected', self._on_wire_connected, 10
        )

        # 노드 떴다는 로그. 터미널에 한 줄 표시됨.
        self.get_logger().info("MissionBridgeNode ready. 구독: /wire_connected")

    # ─────────────────────────────────────────────────
    # 토픽 콜백: 로봇이 한 색 끝낼 때마다 자동으로 호출됨
    # ─────────────────────────────────────────────────
    def _on_wire_connected(self, msg):
        """
        로봇이 '/wire_connected' 토픽으로 보낸 메시지를 받아 처리.
        msg.data 에 색 이름이 문자열로 들어옴. 예: "red"

        전체를 try/except로 감싸서 — 안에서 뭐가 터져도 노드가 안 죽게.
        예외가 나면 그것만 로그로 찍고, 다음 메시지를 받을 준비를 함.
        """
        try:
            # ─── 색 이름 정리 ───
            # 받은 데이터에서 양 끝 공백 제거(strip), 소문자로 통일(lower).
            # 혹시 "Red", "  red " 같이 와도 모두 "red"로 처리됨.
            color = msg.data.strip().lower()

            # 받은 거 로그로 찍기. 디버깅할 때 도움됨.
            self.get_logger().info(f"수신: {color}")

            # ─── Firebase 현재 상태 읽기 ───
            # db_logger.get_state()는 operation_state 전체를 dict로 돌려줌.
            # 만약 못 읽으면 빈 dict {} 로 대체 (or {} 부분).
            state = db_logger.get_state() or {}

            # ─── 임무 진행 중인지 체크 ───
            # phase가 'running'이 아니면 — 이미 끝난 임무거나 시작 전.
            # 그땐 무시하고 끝.
            if state.get('phase') != 'running':
                return

            # ─── 지금까지 연결한 색 목록 가져와서 새 색 추가 ───
            # connected_sequence가 없으면 빈 리스트로 시작.
            # list(...)로 감싼 이유: 원본 안 건드리려고 복사본 만든 거.
            connected = list(state.get('connected_sequence') or [])
            connected.append(color)

            # ─── 다음 타겟 색이 뭔지 계산 ───
            # requested_sequence: 사용자가 요청한 전체 색 순서.
            # next_index: 다음에 처리할 색의 인덱스 (지금까지 한 개수와 같음).
            requested = state.get('requested_sequence') or []
            next_index = len(connected)

            # 아직 남은 색이 있으면 — 다음 타겟 설정.
            # 다 끝났으면 — 빈 문자열 + 완료 메시지.
            if next_index < len(requested):
                next_target = requested[next_index]
                next_message = f'{next_target.upper()} 잡으러 가는 중'
            else:
                # 다 끝남. 심판 노드(arrangement_judge)가 이걸 감지하고
                # 곧 정렬 검사 호출할 것.
                next_target = ''
                next_message = '모든 와이어 연결 완료'

            # ─── Firebase 갱신 ───
            # 한 번의 update_state 호출로 여러 필드를 동시에 박음.
            # 이게 박히는 순간 — 임무 페이지가 자동으로 화면 갱신함.
            # (웹이 Firebase listen으로 듣고 있어서.)
            db_logger.update_state(
                connected_sequence=connected,
                current_index=next_index,
                current_target=next_target,
                message=next_message,
            )

            # ─── 로그 페이지 타임라인용 이벤트 기록 ───
            # robot_events 컬렉션에 한 줄 추가.
            # 로그 페이지에서 임무 클릭하면 시간순으로 보임.
            # (이 블록 없어도 동작은 굴러감. 로그 페이지에서만 안 보임.)
            ops = db_logger.get_all_operations()
            if ops and ops[0].get('result') == 'running':
                db_logger.log_robot_event(
                    ops[0]['operation_id'], 'connect', {'color': color}
                )

            # 처리 결과 로그.
            self.get_logger().info(
                f"→ {connected} / 진행 {next_index}/{len(requested)}"
            )

        except Exception as e:
            # 위 try 블록 안에서 뭐가 터져도 여기로 떨어짐.
            # 노드가 죽지 않고, 에러만 로그로 찍고 끝남.
            # 다음 토픽 메시지가 오면 다시 처리 시도.
            self.get_logger().error(f"처리 실패: {e}")


def main():
    """
    노드 띄우기 — ros2 run wire_voice mission_bridge 명령으로 호출됨.
    setup.py의 entry_points에 등록되어 있어서 그 명령이 굴러감.
    """
    # ROS 2 초기화. 모든 ROS 2 프로그램 시작 시 한 번 필요.
    rclpy.init()

    # 우리가 만든 노드 객체 생성.
    node = MissionBridgeNode()

    try:
        # spin: 노드를 계속 살려두면서 콜백들이 호출될 수 있게 함.
        # Ctrl+C 누르기 전까지 여기서 무한 대기.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C 누르면 깔끔하게 종료.
        pass
    finally:
        # 어떻게 종료되든 — 노드 정리하고 ROS 2 종료.
        node.destroy_node()
        rclpy.shutdown()


# 이 파일을 직접 실행하면 main() 호출.
# (ros2 run 도 결국 main()을 부름.)
if __name__ == '__main__':
    main()