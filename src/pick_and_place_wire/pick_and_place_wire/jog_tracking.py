import os
import time
import sys
import threading
import math
import json  # ─── 오버레이 데이터 JSON 직렬화용 ───
import numpy as np
import cv2
import mediapipe as mp

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import DR_init

# ─── RealSense ImgNode 제거: 영상은 웹이 직접 구독 ───
# from pick_and_place_wire.realsense import ImgNode 
from pick_and_place_wire.onrobot import RG
from dsr_msgs2.msg import ServolStream 

# ─── 핸드 모드 웹 통합: 시작/검증/오버레이 토픽 ───
from std_msgs.msg import String

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

rclpy.init()
dsr_node = rclpy.create_node("jog_sync_stream", namespace=ROBOT_ID)
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

# ==========================================
# ⚙️ 핵심 조작 세팅
# ==========================================
INVERT_X = 1.0  
INVERT_Y = -1.0  
INVERT_Z = 1.0  
Z_FLOOR_LIMIT = 10.0 

ROBOT_BASE_X = 367.32  
ROBOT_BASE_Y = 3.69    
ROBOT_BASE_Z = 422.92  

# ─── 그리퍼 설정 (1/10mm 단위) ───
GRIPPER_OPEN_WIDTH = 500   # 5cm = 50mm = 500
GRIPPER_FORCE = 400        # 40N

# ─── 핸드 모드 웹 통합: 전역 변수 설정 (콜백 제어용) ───
calibration_triggered = False
is_calibrated = False
base_hand_dist = 0.0

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

def calculate_jog_velocity(center, current_pos, deadzone, max_dist, v_max, rs_width, rs_height):
    nx = (current_pos[0] - center[0]) / rs_height
    ny = (current_pos[1] - center[1]) / rs_height
    
    norm_dist = math.hypot(nx, ny)
    norm_deadzone = deadzone / rs_height
    norm_max_dist = max_dist / rs_height

    if norm_dist < norm_deadzone:
        return 0.0, 0.0
    
    clamped_dist = min(norm_dist, norm_max_dist)
    speed_ratio = (clamped_dist - norm_deadzone) / (norm_max_dist - norm_deadzone)
    current_speed = (speed_ratio ** 1.5) * v_max

    dir_x = nx / norm_dist
    dir_y = ny / norm_dist
    return current_speed * dir_x, current_speed * dir_y


class JogSyncController(Node):
    def __init__(self):
        super().__init__('jog_sync_controller')
        self.servol_pub = self.create_publisher(ServolStream, f'/{ROBOT_ID}/servol_stream', 10)
        
        # ─── 웹 오버레이용 JSON 퍼블리셔 ───
        # 영상은 안 보내고, 그 위에 그릴 정보만 정규화 좌표(0.0~1.0)로 보냄
        # 웹은 자기가 받은 RealSense 영상 크기에 맞춰 곱해서 그리면 됨
        self.overlay_pub = self.create_publisher(String, '/hand_overlay_data', 10)
        
        # ─── 핸드 모드 웹 통합: 시작/검증 토픽 구독 ───
        self.create_subscription(String, '/hand_start', self._on_hand_start, 10)
        self.create_subscription(String, '/hand_verify', self._on_hand_verify, 10)
        
        self.target_x = ROBOT_BASE_X
        self.target_y = ROBOT_BASE_Y
        self.target_z = ROBOT_BASE_Z
        
        self.locked_rx = 0.0
        self.locked_rz = 0.0
        
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        self.cam_finger_state = "OPEN"
        self.current_gripper_state = "OPEN"
        self.last_gripper_cmd_time = time.time()
        
        self.is_ready = False
        self.init_robot()
        
        self.control_period = 0.02 
        self.timer = self.create_timer(self.control_period, self.control_loop)

    # ─── 핸드 모드 웹 통합: 콜백 메서드 ───
    def _on_hand_start(self, msg):
        global calibration_triggered, is_calibrated
        calibration_triggered = True
        is_calibrated = False # 새롭게 캘리브레이션 갱신
        self.get_logger().info("[핸드 통합] /hand_start 수신 — 캘리브레이션 시작")

    def _on_hand_verify(self, msg):
        global calibration_triggered, is_calibrated
        calibration_triggered = False
        is_calibrated = False # 플래그를 꺼서 좌표 갱신 멈춤
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.get_logger().info("[핸드 통합] /hand_verify 수신 — 동작 멈춤")

    def init_robot(self):
        self.get_logger().info("로봇 초기 자세(JReady)로 이동합니다.")
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=60, acc=60)
        gripper.move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_FORCE)
        mwait()
        
        current_pos = get_current_posx()[0]
        self.locked_rx = current_pos[3]
        self.locked_rz = current_pos[5]
        
        self.target_x = current_pos[0]
        self.target_y = current_pos[1]
        self.target_z = current_pos[2]
        
        self.get_logger().info("초기화 완료. 조그 제어 대기 중...")
        self.is_ready = True

    def update_state(self, vx, vy, vz, finger_state):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.cam_finger_state = finger_state

    def control_loop(self):
        if not self.is_ready: return

        self.target_x += self.vx * self.control_period
        self.target_y += self.vy * self.control_period
        self.target_z += self.vz * self.control_period

        self.target_x = np.clip(self.target_x, ROBOT_BASE_X - 400, ROBOT_BASE_X + 400)
        self.target_y = np.clip(self.target_y, ROBOT_BASE_Y - 400, ROBOT_BASE_Y + 400)
        self.target_z = np.clip(self.target_z, Z_FLOOR_LIMIT, ROBOT_BASE_Z + 400)

        msg = ServolStream()
        msg.pos = [
            float(self.target_x), 
            float(self.target_y), 
            float(self.target_z), 
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

def main(args=None):
    controller = JogSyncController()
    # ─── ImgNode 제거: RealSense 영상은 웹이 직접 구독 ───
    # img_node = ImgNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    # executor.add_node(img_node)
    
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # ─── 핸드 모드 웹 통합: input("Enter") 대기 스레드 제거 ───
    # threading.Thread(target=wait_for_enter, daemon=True).start()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.7
    )
    cap_webcam = cv2.VideoCapture(0)

    # ─── 정규화 좌표 기준 파라미터 (0.0~1.0, height 기준) ───
    # 기존 픽셀값은 RealSense 480 높이 기준이었음
    DEADZONE_NORM = 40.0 / 480.0   # 0.0833
    MAX_DIST_NORM = 300.0 / 480.0  # 0.625
    V_MAX = 80.0

    # 🎯 타겟 크로스헤어 — 정규화 좌표로 (RealSense 640x480 기준)
    TARGET_X_NORM = 429.0 / 640.0  # 0.670
    TARGET_Y_NORM = 450.0 / 480.0  # 0.938

    # Z축 deadzone — 손 너비(p5-p17) 변화량, 픽셀 비교용이라 정규화 안 함
    # 단, 손 너비 자체는 정규화 좌표(0~1)로 계산하므로 deadzone도 정규화 단위
    Z_DEADZONE_NORM = 10.0 / 480.0  # 0.0208
    Z_SENSITIVITY = 1.5 * 480.0     # 정규화 단위로 변환 보정

    global calibration_triggered, is_calibrated, base_hand_dist

    try:
        while rclpy.ok():
            ret_w, frame_webcam = cap_webcam.read()
            if not ret_w: break

            frame_webcam = cv2.flip(frame_webcam, 1)
            rgb_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_webcam)

            # 비상 브레이크 기본값
            vx, vy, vz = 0.0, 0.0, 0.0
            finger_state = "HOLD"

            # ─── 웹으로 보낼 오버레이 데이터 (정규화 좌표) ───
            overlay = {
                "landmarks": [],            # [{x, y}, ...] 21개, 0~1 정규화
                "deadzone": {"cx": 0.5, "cy": 0.5, "r": DEADZONE_NORM},
                "target":   {"x": TARGET_X_NORM, "y": TARGET_Y_NORM},
                "status": {
                    "calibrated": is_calibrated,
                    "finger": finger_state,
                    "vx": 0.0, "vy": 0.0, "vz": 0.0,
                    "base_w": 0.0, "cur_w": 0.0,
                    "z_limit": False,
                },
            }

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                finger_state = get_finger_state(hand_landmarks)

                # 21개 랜드마크 정규화 좌표 그대로 (mediapipe 출력 자체가 0~1)
                overlay["landmarks"] = [
                    {"x": float(lm.x), "y": float(lm.y)} for lm in hand_landmarks.landmark
                ]

                # 9번 관절 (손바닥 중심) — 정규화 좌표
                p9_x_n = hand_landmarks.landmark[9].x
                p9_y_n = hand_landmarks.landmark[9].y

                # 손 너비 — 정규화 좌표 거리
                p5 = hand_landmarks.landmark[5]
                p17 = hand_landmarks.landmark[17]
                current_dist_n = math.hypot(p17.x - p5.x, p17.y - p5.y)

                # 캘리브레이션
                if calibration_triggered and not is_calibrated:
                    base_hand_dist = current_dist_n
                    is_calibrated = True

                if is_calibrated:
                    # 정규화 좌표 기준 jog 속도 계산
                    # calculate_jog_velocity는 픽셀 입력을 받으니, 가짜로 height=1로 넣어
                    # 정규화 단위 그대로 통과시킴
                    raw_vx, raw_vy = calculate_jog_velocity(
                        (0.5, 0.5), (p9_x_n, p9_y_n),
                        DEADZONE_NORM, MAX_DIST_NORM,
                        V_MAX, 1.0, 1.0
                    )

                    vx = raw_vx * INVERT_X
                    vy = raw_vy * INVERT_Y

                    dist_diff = current_dist_n - base_hand_dist
                    if abs(dist_diff) > Z_DEADZONE_NORM:
                        vz_raw = min(abs(dist_diff) * Z_SENSITIVITY, V_MAX)
                        vz = -np.sign(dist_diff) * vz_raw * INVERT_Z

                overlay["status"].update({
                    "calibrated": is_calibrated,
                    "finger": finger_state,
                    "vx": float(vx), "vy": float(vy), "vz": float(vz),
                    "base_w": float(base_hand_dist),
                    "cur_w": float(current_dist_n),
                })

            # Z-LIMIT 경고
            if controller.target_z <= Z_FLOOR_LIMIT + 2.0:
                overlay["status"]["z_limit"] = True

            controller.update_state(vx, vy, vz, finger_state)

            # ─── 오버레이 JSON 발행 ───
            try:
                msg = String()
                msg.data = json.dumps(overlay)
                controller.overlay_pub.publish(msg)
            except Exception:
                pass

            # ─── 디버그용 웹캠 창 (선택, 끄려면 주석 처리) ───
            cv2.imshow("Webcam (debug)", frame_webcam)
            if cv2.waitKey(1) & 0xFF == 27: break

    finally:
        cap_webcam.release()
        cv2.destroyAllWindows()
        controller.destroy_node()
        # img_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()