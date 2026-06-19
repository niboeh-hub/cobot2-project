import os
import time
import sys
import threading
import math
import numpy as np
import cv2
import mediapipe as mp
from collections import deque  

import rclpy
from rclpy.node import Node
import DR_init

from pick_and_place_wire.onrobot import RG
from dsr_msgs2.msg import ServolStream 

# ─── 핸드/섀도우 모드 통합: 시작/검증 토픽용 메시지 타입 ───
from std_msgs.msg import String

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

rclpy.init()
dsr_node = rclpy.create_node("rokey_shadow_sync_stream", namespace=ROBOT_ID)
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, get_current_posx, mwait
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

ROBOT_BASE_X = 367.34  
ROBOT_BASE_Y = 3.78    
ROBOT_BASE_Z = 191.64  

# ─── 그리퍼 설정 (1/10mm 단위) ───
GRIPPER_OPEN_WIDTH = 500   # 5cm = 50mm = 500
GRIPPER_FORCE = 400        # 40N

SCALE_X = 600.0  
SCALE_Z = 400.0  
SCALE_Y = 3500.0  # [추가] Y축(원근감) 민감도 (테스트하며 조절하세요!)

calibration_triggered = False

# ==========================================
# 1Euro Filter 클래스 정의 (시간적 떨림 제거)
# ==========================================
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.1, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def alpha_calc(self, t_e, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / t_e
        alpha_d = self.alpha_calc(t_e, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha_calc(t_e, cutoff)
        
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
# ==========================================


def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def get_finger_state(hand_landmarks):
    points = hand_landmarks.landmark
    open_count = 0
    if distance(points[4], points[9]) > distance(points[3], points[9]): open_count += 1
    for i in range(8, 21, 4):
        if distance(points[i], points[0]) > distance(points[i - 1], points[0]): open_count += 1
            
    if open_count >= 4: return "OPEN"
    elif open_count <= 1: return "CLOSE"
    return "HOLD"


class ShadowSyncController(Node):
    def __init__(self):
        super().__init__('shadow_sync_controller')
        self.get_logger().info("Shadow Sync 50Hz (ServolStream) Node Initialized.")
        
        topic_name = f'/{ROBOT_ID}/servol_stream'
        self.servol_pub = self.create_publisher(ServolStream, topic_name, 10)

        # [수정] Y축 스무딩 큐 추가
        self.buffer_size = 10  
        self.path_queue_x = deque(maxlen=self.buffer_size)
        self.path_queue_y = deque(maxlen=self.buffer_size) 
        self.path_queue_z = deque(maxlen=self.buffer_size)
        
        self.clean_target_x = ROBOT_BASE_X
        self.clean_target_y = ROBOT_BASE_Y # [추가]
        self.clean_target_z = ROBOT_BASE_Z
        self.cam_finger_state = "OPEN"
        
        self.current_gripper_state = "OPEN"
        self.last_gripper_cmd_time = time.time()
        self.is_ready = False
        
        self.init_robot()
        
        self.control_period = 0.02  
        self.timer = self.create_timer(self.control_period, self.control_loop)

        # ─── 핸드/섀도우 모드 통합: 시작/검증 토픽 구독 ───
        self.create_subscription(
            String, '/hand_start',
            self._on_hand_start, 10
        )
        self.create_subscription(
            String, '/hand_verify',
            self._on_hand_verify, 10
        )

    def _on_hand_start(self, msg):
        """헬로 로키 시작해 — 캘리브레이션 + 동작 시작."""
        global calibration_triggered
        calibration_triggered = True
        self.get_logger().info("[섀도우 통합] /hand_start 수신 — 캘리브레이션 시작")

    def _on_hand_verify(self, msg):
        """헬로 로키 검증 시작해 — 동작 멈추고 대기 모드로."""
        global calibration_triggered
        calibration_triggered = False
        # 안전을 위해 타겟 좌표를 현재 ROBOT_BASE로 리셋
        # (다음 캘리브레이션 전까지 로봇이 더 안 움직이게)
        self.clean_target_x = ROBOT_BASE_X
        self.clean_target_y = ROBOT_BASE_Y
        self.clean_target_z = ROBOT_BASE_Z
        self.get_logger().info("[섀도우 통합] /hand_verify 수신 — 동작 멈춤")

    def init_robot(self):
        self.get_logger().info("로봇 초기 자세(JReady)로 이동합니다.")
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=60, acc=60)
        gripper.move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_FORCE)
        mwait()
        
        current_pos = get_current_posx()[0]
        self.locked_rx = current_pos[3]
        self.locked_rz = current_pos[5]
        
        self.get_logger().info(f"초기화 완료 (rx:{self.locked_rx:.1f}, rz:{self.locked_rz:.1f})")
        self.is_ready = True

    # [수정] clean_y 파라미터 추가
    def update_camera_target(self, clean_x, clean_y, clean_z, finger_state):
        self.clean_target_x = clean_x
        self.clean_target_y = clean_y
        self.clean_target_z = clean_z
        self.cam_finger_state = finger_state

    def control_loop(self):
        if not self.is_ready:
            return

        # 1. 1유로 필터를 통과한 정제된 좌표를 큐 버퍼에 저장 [Y축 추가]
        self.path_queue_x.append(self.clean_target_x)
        self.path_queue_y.append(self.clean_target_y)
        self.path_queue_z.append(self.clean_target_z)

        # 2. [공간 스무딩] 큐에 쌓인 과거 좌표들의 평균 계산 [Y축 추가]
        if len(self.path_queue_x) > 0:
            smooth_x = np.mean(self.path_queue_x)
            smooth_y = np.mean(self.path_queue_y)
            smooth_z = np.mean(self.path_queue_z)
        else:
            smooth_x = self.clean_target_x
            smooth_y = self.clean_target_y
            smooth_z = self.clean_target_z

        # 3. 안전 범위 클리핑 [Y축 추가]
        safe_x = np.clip(smooth_x, ROBOT_BASE_X - 300, ROBOT_BASE_X + 300)
        safe_y = np.clip(smooth_y, ROBOT_BASE_Y - 200, ROBOT_BASE_Y + 200) # 로봇 작업 반경에 맞춰 조절하세요
        safe_z = np.clip(smooth_z, ROBOT_BASE_Z - 200, ROBOT_BASE_Z + 300)

        # 4. 로봇 제어 메시지 전송 [Y축 적용]
        msg = ServolStream()
        msg.pos = [
            float(safe_x), 
            float(safe_y), # 기존 ROBOT_BASE_Y에서 safe_y로 변경
            float(safe_z), 
            float(self.locked_rx),  
            179.9,                  
            float(self.locked_rz)   
        ]
        
        msg.vel = [250.0, 100.0] 
        msg.acc = [250.0, 100.0] 
        msg.time = 0.03  
        
        self.servol_pub.publish(msg)

        current_time = time.time()
        if (current_time - self.last_gripper_cmd_time) > 0.5: 
            if self.cam_finger_state == "OPEN" and self.current_gripper_state != "OPEN":
                gripper.move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_FORCE)
                self.current_gripper_state = "OPEN"
                self.last_gripper_cmd_time = current_time
            elif self.cam_finger_state == "CLOSE" and self.current_gripper_state != "CLOSE":
                gripper.close_gripper()
                self.current_gripper_state = "CLOSE"
                self.last_gripper_cmd_time = current_time


def wait_for_enter():
    global calibration_triggered
    input("\n[동기화 준비] 로봇이 대기 중입니다. 가슴 중앙에 손을 얹고 [Enter] 키를 누르세요!\n")
    calibration_triggered = True


def main(args=None):
    controller = ShadowSyncController()
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
    ros_thread.start()

    # ─── 핸드/섀도우 모드 통합: input 대신 /hand_start 토픽으로 받음 ───
    # threading.Thread(target=wait_for_enter, daemon=True).start()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    is_calibrated = False
    base_hand_x, base_hand_y = 0.0, 0.0
    base_hand_dist = 0.0 # [추가] 캘리브레이션 시점의 손 너비 저장용

    current_time = time.time()
    filter_x = OneEuroFilter(current_time, ROBOT_BASE_X, min_cutoff=0.1, beta=0.01)
    filter_y = OneEuroFilter(current_time, ROBOT_BASE_Y, min_cutoff=0.1, beta=0.005) # [추가] Y축 1유로 필터
    filter_z = OneEuroFilter(current_time, ROBOT_BASE_Z, min_cutoff=0.1, beta=0.01)

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            global calibration_triggered

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_state = get_finger_state(hand_landmarks)

                current_hx = hand_landmarks.landmark[9].x
                current_hy = hand_landmarks.landmark[9].y
                
                # [추가] 5번 관절(검지 밑동)과 17번 관절(새끼 밑동) 사이의 거리 계산
                p5 = hand_landmarks.landmark[5]
                p17 = hand_landmarks.landmark[17]
                current_dist = distance(p5, p17)

                if calibration_triggered and not is_calibrated:
                    base_hand_x = current_hx
                    base_hand_y = current_hy
                    base_hand_dist = current_dist # [추가] 기준 너비 저장
                    is_calibrated = True

                if is_calibrated:
                    delta_x = current_hx - base_hand_x
                    delta_y = current_hy - base_hand_y
                    
                    # [추가] 손이 카메라에 가까워지면(거리가 멀어지면) 양수, 멀어지면 음수
                    delta_dist = current_dist - base_hand_dist

                    raw_target_x = ROBOT_BASE_X + (delta_x * SCALE_X)
                    
                    # 손을 카메라로 밀 때(가까워질 때) 로봇이 어느 방향으로 가야 하는지에 따라 부호(+,-)를 바꾸세요.
                    raw_target_y = ROBOT_BASE_Y + (delta_dist * SCALE_Y) # [추가]
                    
                    raw_target_z = ROBOT_BASE_Z - (delta_y * SCALE_Z) 

                    timestamp = time.time()
                    clean_x = filter_x(timestamp, raw_target_x)
                    clean_y = filter_y(timestamp, raw_target_y) # [추가]
                    clean_z = filter_z(timestamp, raw_target_z)

                    # [수정] 컨트롤러에 Y좌표 전달
                    controller.update_camera_target(clean_x, clean_y, clean_z, finger_state)

                    cv2.putText(frame, f"STREAM 50Hz | Grip: {finger_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Real-time Shadow Sync (Stream 50Hz)", frame)
            
            if cv2.waitKey(5) & 0xFF == 27: break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()