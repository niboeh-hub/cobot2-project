#제미나이 치우기 수정
import os
import sys
import time
import threading
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge   
from rclpy.qos import QoSProfile 

from od_msg.srv import SrvDepthPosition
from resource.realsense import ImgNode
from resource.onrobot import RG
from ultralytics import YOLO
import DR_init

from dsr_msgs2.srv import SetRobotControl, GetRobotState

# ──────────────────────────────────────────────
# 1. 전역 로봇 및 하드웨어 설정
# ──────────────────────────────────────────────
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 120, 120
MIN_SAFE_Z = 3.0 #Z 마지노선

try:
    rclpy.init()
except Exception:
    pass

dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

YOLO_MODEL_PATH  = "/home/soyoung/cobot_ws/src/pick_and_place_wire/resource/best.pt"
GRIPPER2CAM_PATH = "/home/soyoung/cobot_ws/src/pick_and_place_wire/resource/T_gripper2camera.npy"
#YOLO_MODEL_PATH  = "/home/rokey/cobot_ws/src/bomb_defuse/resource/best.pt"
#GRIPPER2CAM_PATH = "/home/rokey/cobot_ws/src/bomb_defuse/T_gripper2camera.npy"
try:
    from DSR_ROBOT2 import movej, movel, movejx, get_current_posx, get_current_posj, mwait, task_compliance_ctrl, release_compliance_ctrl
except ImportError as e:
    print(f"[ERROR] DSR_ROBOT2 임포트 실패: {e}")
    sys.exit()

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)

APPROACH_Z = 200.0
Z_OFFSET = -29.0
MIN_DEPTH = 8.5

# ──────────────────────────────────────────────
# 2. 비전 & YOLO 세그멘테이션 노드
# ──────────────────────────────────────────────
class ObjectDetectionNode(Node):
    def __init__(self, img_node):
        super().__init__('object_detection_node')
        self.img_node = img_node

        self.get_logger().info("[비전 초기화] 카메라 내부 파라미터 대기 중...")
        self.intrinsics = self.img_node.get_camera_intrinsic()
        while self.intrinsics is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            self.intrinsics = self.img_node.get_camera_intrinsic()
        self.get_logger().info("[비전 초기화] 카메라 파라미터 수신 완료!")

        self.create_service(SrvDepthPosition, 'get_3d_position', self.handle_get_depth)

        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.target_point = None
        self.latest_angle_deg = 0.0
        self.skip_detection = False
        
        # 🔥 [추가] 방해물 감지용 변수
        self.needs_sweep = False
        self.obs_pixel = None
        
        self.target_class_name = 'green'
        self.target_part = "TOP"
        self._lock = threading.Lock() 
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/yolo/image_with_boxes', 10)

        cv2.namedWindow("YOLO Wire Auto-Pick")
        self.get_logger().info(f"🚨 현재 모델이 인식하는 클래스 목록: {self.yolo_model.names}")

    def update_gui(self):
        frame = self.img_node.get_color_frame()
        if frame is None or (isinstance(frame, np.ndarray) and not frame.any()):
            cv2.waitKey(10)
            return

        h, w = frame.shape[:2]
        cam_cx, cam_cy = w // 2, h // 2

        if self.skip_detection:
            display = frame.copy()
            cv2.putText(display, f"ROBOT MOVING... TASK: {self.target_part}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(display, 'bgr8'))
            except Exception:
                pass
            cv2.imshow("YOLO Wire Auto-Pick", display)
            cv2.waitKey(10)
            return

        # conf(신뢰도)는 0.5 이상, iou(중복 박스 제거 기준)는 0.4 ~ 0.45 정도로 꽉 조여줍니다.
        results = self.yolo_model(frame, conf=0.50, iou=0.40, verbose=False)
        annotated_frame = results[0].plot()

        with self._lock:
            self.target_point = None
            self.needs_sweep = False
            self.obs_pixel = None

        if results[0].masks is not None and len(results[0].masks) > 0:
            classes = results[0].boxes.cls.cpu().numpy()
            masks_xy = results[0].masks.xy

            target_id = -1
            for k, v in self.yolo_model.names.items():
                if self.target_class_name in v.lower():
                    target_id = k
                    break

            target_mask_idx = -1
            min_center_dist = float('inf')
            target_pts = None

            for i, polygon_points in enumerate(masks_xy):
                if int(classes[i]) != target_id or len(polygon_points) < 5:
                    continue

                pts = np.array(polygon_points, dtype=np.int32)
                distances_sq = np.square(pts[:, 0] - cam_cx) + np.square(pts[:, 1] - cam_cy)
                dist = distances_sq[np.argmin(distances_sq)]

                if dist < min_center_dist:
                    min_center_dist = dist
                    target_mask_idx = i
                    target_pts = pts

            if target_mask_idx != -1 and target_pts is not None:
                if self.target_part == "TOP":
                    target_tip_pt = target_pts[np.argmin(target_pts[:, 1])]
                else:
                    target_tip_pt = target_pts[np.argmax(target_pts[:, 1])]

                center_pt = np.mean(target_pts, axis=0)
                vec = center_pt - target_tip_pt
                norm = np.linalg.norm(vec)

                if norm > 0:
                    if self.target_part == "TOP":
                        INWARD_PIXELS = 70
                    else:
                        INWARD_PIXELS = 40
                    
                    inward_pt = target_tip_pt + (vec / norm) * INWARD_PIXELS if norm > INWARD_PIXELS else center_pt
                else:
                    inward_pt = target_tip_pt

                with self._lock:
                    self.target_point = (int(inward_pt[0]), int(inward_pt[1]))

                rect = cv2.minAreaRect(target_pts)
                box = np.int0(cv2.boxPoints(rect))

                dist01 = np.linalg.norm(box[0] - box[1])
                dist12 = np.linalg.norm(box[1] - box[2])

                if dist01 > dist12:
                    pt1 = np.mean([box[1], box[2]], axis=0).astype(int)
                    pt2 = np.mean([box[3], box[0]], axis=0).astype(int)
                else:
                    pt1 = np.mean([box[0], box[1]], axis=0).astype(int)
                    pt2 = np.mean([box[2], box[3]], axis=0).astype(int)

                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                angle_deg = np.degrees(np.arctan2(dy, dx)) + 90.0
                if angle_deg > 180.0: angle_deg -= 360.0
                elif angle_deg < -180.0: angle_deg += 360.0

                self.latest_angle_deg = angle_deg
                
                # 🔥 [핵심 추가] 3cm 이내 방해물 감지 로직
                min_obs_dist = float('inf')
                closest_obs_pt = None
                
                for i, polygon_points in enumerate(masks_xy):
                    if i == target_mask_idx or len(polygon_points) < 5: continue
                    
                    obs_pts = np.array(polygon_points, dtype=np.int32)
                    dists = np.linalg.norm(obs_pts - self.target_point, axis=1)
                    min_idx = np.argmin(dists)
                    
                    if dists[min_idx] < min_obs_dist:
                        min_obs_dist = dists[min_idx]
                        closest_obs_pt = obs_pts[min_idx]
                
                PIXEL_THRESHOLD = 60 # 약 3cm
                if min_obs_dist < PIXEL_THRESHOLD and closest_obs_pt is not None:
                    with self._lock:
                        self.needs_sweep = True
                        self.obs_pixel = tuple(closest_obs_pt)
                    cv2.arrowedLine(annotated_frame, self.target_point, tuple(closest_obs_pt), (0, 255, 255), 3)
                    cv2.putText(annotated_frame, "SWEEP NEEDED!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                cv2.circle(annotated_frame, (int(target_tip_pt[0]), int(target_tip_pt[1])), 4, (0, 0, 255), -1)
                cv2.circle(annotated_frame, self.target_point, 9, (0, 255, 255), -1)
                cv2.putText(annotated_frame, f"Target ({self.target_part}): {angle_deg:.1f} deg", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8'))
        except Exception:
            pass
        cv2.imshow("YOLO Wire Auto-Pick", annotated_frame)
        cv2.waitKey(10)

    def handle_get_depth(self, request, response):
        with self._lock:
            tp = self.target_point

        if tp is None:
            response.depth_position = [0.0, 0.0, 0.0, 0.0]
            return response

        cx, cy = tp
        cz = self._get_depth(cx, cy)

        if cz is None or cz <= 0:
            self.get_logger().warn(f"⚠️ 픽셀 ({cx}, {cy})의 Depth 데이터가 0입니다.")
            response.depth_position = [0.0, 0.0, 0.0, 0.0]
        else:
            coords = self._pixel_to_camera_coords(cx, cy, cz)
            response.depth_position = [float(coords[0]), float(coords[1]), float(coords[2]), float(self.latest_angle_deg)]
        return response

    def _get_depth(self, x, y):
        frame = self.img_node.get_depth_frame()
        if frame is None: return None
        try:
            val = frame[y, x]
            if val > 0:
                return val
            h, w = frame.shape[:2]
            patch = frame[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
            valid_vals = patch[patch > 0]
            return int(np.median(valid_vals)) if len(valid_vals) > 0 else None
        except IndexError:
            return None

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        ppx, ppy = self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)


# ──────────────────────────────────────────────
# 3. 로봇 제어 오케스트레이터 노드
# ──────────────────────────────────────────────
class RobotController(Node):
    def __init__(self, detection_node):
        super().__init__("robot_control_node")
        self.detection_node = detection_node
        self.cb_group = ReentrantCallbackGroup()

        self.current_step = 1
        self.current_target_color = "green"
        self.color_queue = []
        self.global_slot_idx = 0
        self.is_recovering = False  

        #posx([592.315, -4.129, 272.13, 106.331, -178.983, 105.789]) scan_joint_pose posx
        self.scan_joint_pose = [-0.325, 41.265, 19.095, -0.842, 119.34, -1.554]

        self.up_slots = [
            [409.85, 167.14, 62.246, 37.681, -179.428, 38.099], 
            [501.653, 165.105, 62.13, 44.407, -179.307, 44.695],
            [593.686, 162.256, 61.11, 66.324, -179.252, 66.526],
        ]
        self.down_slots = [
            [409.119, -22.944, 63.567, 24.554, -178.606, 24.437],
            [497.689, -25.488, 62.333, 39.865, -179.038, 39.778],
            [588.724, -25.543, 64.829, 41.567, -178.951, 41.426],
        ]

        self.depth_client = self.create_client(SrvDepthPosition, "get_3d_position", callback_group=self.cb_group)
        self.control_cli = self.create_client(SetRobotControl, '/dsr01/system/set_robot_control', callback_group=self.cb_group)
        self.state_cli = self.create_client(GetRobotState, '/dsr01/system/get_robot_state', callback_group=self.cb_group)
        
        qos = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(String, '/extracted_wire_keywords', qos)

        self.manual_sub = self.create_subscription(String, '/manual_wire_cmd', self.manual_wire_callback, 10, callback_group=self.cb_group)
        self.voice_topic_sub = self.create_subscription(String, '/extracted_wire_keywords', self.voice_wire_topic_callback, 10, callback_group=self.cb_group)
        self.depth_request = SrvDepthPosition.Request()
        self.connected_pub = self.create_publisher(String, '/wire_connected', 10)

        self.create_timer(0.5, self._safety_poll_timer, callback_group=self.cb_group)

    def _safety_poll_timer(self):
        if self.is_recovering or not self.state_cli.wait_for_service(timeout_sec=0.1):
            return
        req = GetRobotState.Request()
        future = self.state_cli.call_async(req)
        future.add_done_callback(self._on_state_received)

    def _on_state_received(self, future):
        try:
            response = future.result()
            state_code = response.robot_state
        except Exception:
            return

        if self.is_recovering:
            return

        if state_code in [5, 6]:
            self.is_recovering = True
            if state_code == 5:
                self.get_logger().warn("🚨 [하드웨어 에러 감지] 노란불(SAFE_STOP) 발생! 복구 프로세스를 시작합니다.")
            else:
                self.get_logger().error("🚨 [하드웨어 에러 감지] 빨간불(EMERGENCY_STOP) 발생! 비상버튼을 해제해주세요.")
            
            threading.Thread(target=self.handle_robot_recovery, daemon=True).start()

    def handle_robot_recovery(self):
        try:
            self.get_logger().info("🔄 제어기 상태 복구를 시작합니다. (비상버튼 해제 대기)")
            
            while rclpy.ok():
                req_state = GetRobotState.Request()
                future_state = self.state_cli.call_async(req_state)
                
                while not future_state.done():
                    time.sleep(0.1)
                
                state_code = future_state.result().robot_state
                if state_code in [1, 2]:
                    self.get_logger().info("✅ 로봇 제어기가 정상(STANDBY) 상태로 확인되었습니다.")
                    break
                
                self.get_logger().info(f"⏳ 현재 상태코드 [{state_code}]. 알람 리셋 및 서보온을 재시도합니다...")
                
                req_ctrl = SetRobotControl.Request()
                req_ctrl.robot_control = 2
                self.control_cli.call_async(req_ctrl)
                time.sleep(2.0)
                
                req_ctrl.robot_control = 3
                self.control_cli.call_async(req_ctrl)
                time.sleep(3.0)

            self.get_logger().info("⬆️ [안전 복구] 충돌 회피를 위해 제자리 20cm 수직 상승합니다.")
            try:
                current_posx = get_current_posx()[0]
                safe_up_pose = list(current_posx)
                safe_up_pose[2] += 200.0
                movel(safe_up_pose, vel=60, acc=60)
                mwait()
            except Exception as e:
                self.get_logger().error(f"⚠️ 수직 상승 모션 실패: {e}")

            self.get_logger().info("🏠 [안전 복구] 안전 구역 확보 완료. 홈 위치(JReady)로 복귀합니다.")
            try:
                JReady = [0, 0, 90, 0, 90, 0]
                movej(JReady, vel=60, acc=60)
                mwait()
                self.open_gripper_5cm()
            except Exception as e:
                self.get_logger().error(f"⚠️ 홈 위치 복귀 모션 실패: {e}")

            self.current_step = 1
            self.detection_node.target_part = "TOP"
            self.detection_node.skip_detection = False
            self.get_logger().info("🟢 시스템 안전 복구 시퀀 종료. 메인 제어 루프를 재개합니다.")

        except Exception as e:
            self.get_logger().error(f"❌ 복구 프로세스 실패: {e}")
        finally:
            self.is_recovering = False  

    def execute_move(self, move_func, *args, **kwargs):
        try:
            move_func(*args, **kwargs)
            mwait()
            return True
        except Exception as e:
            self.get_logger().warn(f"⚠️ [모션 중지 감지]: {e}. 하드웨어 감시기가 복구를 시작할 때까지 대기합니다.")
            self.detection_node.skip_detection = False
            return False

    def manual_wire_callback(self, msg):
        color_str = msg.data.strip().lower()
        self.get_logger().info(f"📥 [토픽 수신] 수동 명령: {color_str}")
        self.color_queue.extend(color_str.split())

    def voice_wire_topic_callback(self, msg):
        raw_data = msg.data.strip().lower()
        if hasattr(self, '_last_msg') and self._last_msg == raw_data:
            return 
        self._last_msg = raw_data
        
        self.get_logger().info(f"🎙️ [음성 토픽 수신] {raw_data}")
        import re
        parsed_colors = [c.strip() for c in re.split(r'[\s,]+', raw_data) if c.strip()]

        new_colors = []
        for color in parsed_colors:
            if color not in self.color_queue:
                new_colors.append(color)
        
        if not new_colors:
            return

        available_space = len(self.up_slots) - len(self.color_queue)
        if len(new_colors) > available_space:
            new_colors = new_colors[:available_space]

        self.color_queue.extend(new_colors)
        self.get_logger().info(f"📊 큐 업데이트 완료: {self.color_queue}")

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, robot_pos):
        gripper2cam = np.load(GRIPPER2CAM_PATH)
        coord = np.append(np.array(camera_coords), 1)
        base2gripper = self.get_robot_pose_matrix(*robot_pos)
        base2cam = base2gripper @ gripper2cam
        return np.dot(base2cam, coord)[:3]

    def open_gripper_5cm(self):
        self.get_logger().info("[그리퍼] 5cm 개방")
        try:
            gripper.move_gripper(400)
        except Exception as e:
            self.get_logger().error(f"❌ [그리퍼 오류] {e}")
        mwait()

    def init_robot(self):
        self.get_logger().info("[로봇] JReady 복귀 및 그리퍼 세팅")
        JReady = [0, 0, 90, 0, 90, 0]
        self.execute_move(movej, JReady, vel=VELOCITY, acc=ACC)
        self.open_gripper_5cm()
        time.sleep(1.5)
        self.get_logger().info("[로봇] 준비 완료.")

    # ──────────────────────────────────────────────
    # 메인 파이프라인
    # ──────────────────────────────────────────────
    def run_pipeline(self):
        if self.is_recovering:
            time.sleep(0.5)
            return True

        print("\n" + "="*50)

        if self.current_step == 1:
            if not self.color_queue:
                self.get_logger().info("💤 [대기 중] 비동기 토픽 명령(/manual_wire_cmd 등)을 기다리는 중...", throttle_duration_sec=3.0)
                return True

            self.current_target_color = self.color_queue[0]
            self.get_logger().info(f"▶ 작업 대상: {self.current_target_color.upper()} (슬롯: {self.global_slot_idx + 1}번) - TOP 탐색 시작")
            self.detection_node.target_part = "TOP"

        self.detection_node.target_class_name = self.current_target_color

        for _ in range(5):
            self.detection_node.update_gui()
            time.sleep(0.03)

        with self.detection_node._lock:
            tp = self.detection_node.target_point
            needs_sweep = self.detection_node.needs_sweep
            obs_pixel = self.detection_node.obs_pixel

        if tp is None:
            self.get_logger().error("[오류] 타겟 미감지. 카메라 갱신 대기...")
            return True

        self.depth_request.target = "auto_wire"
        future = self.depth_client.call_async(self.depth_request)

        start_time = time.time()
        while rclpy.ok() and not future.done():
            if self.is_recovering:
                return True
            if time.time() - start_time > 5.0:
                self.get_logger().error("❌ 비전 서비스 응답 지연 (타임아웃). 재시도합니다.")
                self.detection_node.skip_detection = False
                return True
            time.sleep(0.05)

        if not future.result():
            self.get_logger().error("❌ 비전 서비스 응답 처리 실패!")
            return True

        result_list = future.result().depth_position.tolist()
        if sum(result_list[:3]) == 0:
            self.get_logger().error("❌ Depth 값 추출 실패! 다음 프레임 대기...")
            return True

        cam_coords = result_list[:3]
        wire_angle_deg = result_list[3]
        robot_posx = get_current_posx()[0]
        base_coords = self.transform_to_base(cam_coords, robot_posx)
        
        target_x, target_y = base_coords[0], base_coords[1]
        # 🚀 [핵심 수정] TOP과 BOTTOM 파지 시 서로 다른 Z축 고도 강제 고정
        if self.detection_node.target_part == "TOP":
            target_z = 9.0
            self.get_logger().info("📏 [파지 고도 설정] 'TOP' 전선 파지: Z축을 10.0으로 강제 고정합니다.")
        else: # "BOTTOM"인 경우
            target_z = 62.0
            self.get_logger().info("📏 [파지 고도 설정] 'BOTTOM' 전선 파지: Z축을 62.0으로 강제 고정합니다.")
        
        self.get_logger().info(f"DEBUG [3.Final] Target X:{target_x:.2f} Y:{target_y:.2f} Final_Z:{target_z:.2f}")
        self.detection_node.skip_detection = True
        # ──────────────────────────────────────────────────────────
        # 🔥 [핵심 추가] 다중 방해물 자동 Sweeping 시퀀스 (37번에 통합)
        # ──────────────────────────────────────────────────────────
        if needs_sweep and obs_pixel is not None and self.current_step == 1:
            self.get_logger().warn("[로봇 모션] ⚠️ 3cm 이내에 다른 전선 감지! Sweeping(물리적 분리) 시퀀스 돌입!")
            
            obs_cx, obs_cy = obs_pixel
            obs_cz = self.detection_node._get_depth(obs_cx, obs_cy)
            if obs_cz is None or obs_cz <= 0: obs_cz = cam_coords[2] 
            
            obs_cam_coords = self.detection_node._pixel_to_camera_coords(obs_cx, obs_cy, obs_cz)
            obs_base_coords = self.transform_to_base(obs_cam_coords, robot_posx)

            try:
                gripper.close_gripper()
                time.sleep(1.0)
            except Exception:
                pass

            # 1. 타겟과 방해물 사이의 벡터 및 거리 미리 계산
            vec_x = obs_base_coords[0] - target_x
            vec_y = obs_base_coords[1] - target_y
            mag = np.hypot(vec_x, vec_y)

            # 2. 타겟과 방해물의 정확히 중간 지점(50%) 좌표 계산
            mid_x = target_x + (vec_x * 0.5)
            mid_y = target_y + (vec_y * 0.5)

            # 3. 타겟 위가 아닌 '중간 지점' 위로 접근
            approach_pose = [mid_x, mid_y, APPROACH_Z, robot_posx[3], robot_posx[4], robot_posx[5]]
            if not self.execute_move(movel, approach_pose, vel=VELOCITY, acc=ACC): return True

            # 4. '중간 지점'에서 이전 요청대로 Z=10.0 높이로 하강
            sweep_down_pose = approach_pose.copy()
            sweep_down_pose[2] = 10.0 
            if not self.execute_move(movel, sweep_down_pose, vel=VELOCITY, acc=ACC): return True
            
            SWEEP_DIST = 50.0 
            sweep_pose = sweep_down_pose.copy()
            
            if mag > 0.001:
                sweep_pose[0] += (vec_x / mag) * SWEEP_DIST
                sweep_pose[1] += (vec_y / mag) * SWEEP_DIST
                if not self.execute_move(movel, sweep_pose, vel=VELOCITY, acc=ACC): return True

            sweep_up_pose = sweep_pose.copy()
            sweep_up_pose[2] = APPROACH_Z
            if not self.execute_move(movel, sweep_up_pose, vel=VELOCITY, acc=ACC): return True

            self.get_logger().info("[로봇 모션] Sweeping 완료! 전선 안정화를 위해 홈으로 복귀 후 자동으로 이어서 작업합니다.")
            
            self.init_robot()
            self.detection_node.skip_detection = False
            time.sleep(2.0) 
            return True 
        # ──────────────────────────────────────────────────────────
        HOVER_HEIGHT = 150.0
        app_pose = [target_x, target_y, target_z + HOVER_HEIGHT,
                    robot_posx[3], robot_posx[4], robot_posx[5]]
        app_pose[2] = max(app_pose[2], MIN_SAFE_Z) 
        if not self.execute_move(movel, app_pose, vel=VELOCITY, acc=ACC): return True

        current_j = np.array(get_current_posj()).flatten().tolist()
        diff = (float(wire_angle_deg)) % 180
        if diff > 90: diff -= 180
        elif diff < -90: diff += 180

        applied_j6_diff = 0.0

        if abs(diff) > 10.0:
            target_j = current_j.copy()
            target_j[5] = current_j[5] + diff
            self.get_logger().info(f"[모션] 각도 정렬 ({diff:.1f}도 회전)")
            if not self.execute_move(movej, target_j, vel=VELOCITY, acc=ACC): return True
            applied_j6_diff = diff  
        else:
            self.get_logger().info(f"[모션] 각도 오차 작음 ({diff:.1f}도) — 회전 생략")

        updated_posx = get_current_posx()[0]
        # 🚀 [수정] 기존의 target_z - 1.0 대신 고정된 target_z(1.25)를 그대로 매핑
        actual_drop_pose = [target_x, target_y, target_z,
                            updated_posx[3], updated_posx[4], updated_posx[5]]
        actual_drop_pose[2] = max(actual_drop_pose[2], MIN_SAFE_Z) 
        self.get_logger().info(f"👇 최종 하강 목표 Z 높이 (Drop Z): {actual_drop_pose[2]:.3f} mm")

        if not self.execute_move(movel, actual_drop_pose, vel=VELOCITY, acc=ACC): 
            self.get_logger().error("❌ 하강 모션(movel) 실패! 리미트 확인 필요.")
            self.detection_node.skip_detection = False
            return True
        try:
            gripper.close_gripper()
            time.sleep(1.5)
            mwait()
        except Exception as e:
            self.get_logger().error(f"❌ [그리퍼 닫기 실패] {e}")
            self.detection_node.skip_detection = False
            return True

        lift_pose = actual_drop_pose.copy()
        LIFT_HEIGHT = 150.0 if self.current_step == 1 else 10.0
        lift_pose[2] += LIFT_HEIGHT
        lift_pose[2] = max(lift_pose[2], MIN_SAFE_Z) 
        if not self.execute_move(movel, lift_pose, vel=VELOCITY, acc=ACC): return True

        if abs(applied_j6_diff) > 0.0:
            self.get_logger().info(f"🔄 [모션] 슬롯 이동 전 J6 관절 원위치 복구 ({-applied_j6_diff:.1f}도 반대로 회전)")
            current_j_lift = np.array(get_current_posj()).flatten().tolist()
            target_j_lift = current_j_lift.copy()
            target_j_lift[5] -= applied_j6_diff  
            if not self.execute_move(movej, target_j_lift, vel=VELOCITY, acc=ACC): return True

        # ── STEP 1: 위쪽 끝단 → up_N 슬롯 ──
        if self.current_step == 1:
            safe_slot_index = min(self.global_slot_idx, len(self.up_slots) - 1)
            current_up_slot = self.up_slots[safe_slot_index]
            
            up_above = list(current_up_slot)
            up_above[2] += 30.0
            if not self.execute_move(movel, up_above, vel=VELOCITY, acc=ACC): return True
            mwait()
            
            self.get_logger().info(f"📍 [삽입] {self.global_slot_idx+1}번 상단 슬롯에 전선 삽입 중...")
            if not self.execute_move(movel, current_up_slot, vel=60, acc=60): return True 
            mwait() 
            time.sleep(1.0) 

            self.open_gripper_5cm()
            time.sleep(1.5)

            if not self.execute_move(movel, up_above, vel=VELOCITY, acc=ACC): return True
            mwait()

            self.get_logger().info("📸 [스캔] 특이점을 피해 관절 이동(movej)으로 BOTTOM 스캔 위치로 부드럽게 이동합니다.")
            if not self.execute_move(movej, self.scan_joint_pose, vel=60, acc=60): return True
            mwait()
            time.sleep(0.5) 
            
            self.get_logger().info(f"🔄 [Step 2] 로봇 정지 확인 완료. 이제 '{self.current_target_color.upper()}' BOTTOM 탐색을 시작합니다.")
            self.current_step = 2
            self.detection_node.target_part = "BOTTOM"
            self.detection_node.skip_detection = False 
            return True

        # ── STEP 2: 아래쪽 끝단 → down_N 슬롯 ──
        elif self.current_step == 2:
            safe_slot_index = min(self.global_slot_idx, len(self.down_slots) - 1)
            current_down_slot = self.down_slots[safe_slot_index]

            down_above = list(current_down_slot)
            down_above[2] += 20.0

            if not self.execute_move(movel, down_above, vel=VELOCITY, acc=ACC): return True
            mwait()
            
            self.get_logger().info(f"📍 [삽입] {self.global_slot_idx+1}번 하단 슬롯에 전선 삽입 중...")
            if not self.execute_move(movel, current_down_slot, vel=60, acc=60): return True
            mwait()
            time.sleep(1.0)

            self.open_gripper_5cm()
            time.sleep(1.5)

            safe_lift_pose = list(current_down_slot)
            safe_lift_pose[2] += 20.0
            if not self.execute_move(movel, safe_lift_pose, vel=VELOCITY, acc=ACC): return True
            mwait()

            self.get_logger().info(f"✅ {self.current_target_color.upper()} 전선 장착 완료.")
            
            completed_color = self.current_target_color
            try:
                self.connected_pub.publish(String(data=completed_color))
                self.get_logger().info(f"[웹 통합] /wire_connected 발행: {completed_color}")
            except Exception as e:
                self.get_logger().warn(f"[웹 통합] 발행 실패: {e}")

            self.global_slot_idx += 1
            if self.global_slot_idx >= len(self.up_slots):
                is_all_done = True
                self.global_slot_idx = 0
            else:
                is_all_done = False

            if self.color_queue:
                self.color_queue.pop(0)

            if not is_all_done:
                self.get_logger().info(f"🏠 [중간 복귀] {self.global_slot_idx}번째 사이클 종료 — 홈(JReady) 위치로 복귀합니다.")
                self.init_robot() 
            else:
                self.get_logger().info("📸 [최종 복귀] 모든 전선 장착 완료! 관절 이동(movej)으로 스캔 대기 위치로 이동합니다.")
                if not self.execute_move(movej, self.scan_joint_pose, vel=60, acc=60): return True
                mwait()
                
                self.get_logger().info("🎉 [작업 완료] 3개의 전선 장착이 모두 끝났습니다! 프로그램을 안전하게 자동 종료합니다.")
                os._exit(0) 

            self.current_step = 1
            self.detection_node.skip_detection = False
            return True

        return True

# ──────────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────────
def main(args=None):
    print("[시스템 메인] ROS2 통합 노드 시작...")

    img_node = ImgNode()
    detection_node = ObjectDetectionNode(img_node)
    robot_node = RobotController(detection_node)

    bg_executor = MultiThreadedExecutor()
    bg_executor.add_node(img_node)
    bg_executor.add_node(detection_node)
    bg_executor.add_node(robot_node)
    bg_executor.add_node(dsr_node)

    t1 = threading.Thread(target=bg_executor.spin, daemon=True)
    t1.start()

    time.sleep(0.5)
    robot_node.init_robot()

    try:
        while rclpy.ok():
            detection_node.update_gui()
            if not robot_node.detection_node.skip_detection and not robot_node.is_recovering:
                robot_node.run_pipeline()
            elif robot_node.is_recovering:
                time.sleep(0.1) 
    except KeyboardInterrupt:
        print("\n[시스템 메인] 강제 종료.")
    finally:
        nodes = [detection_node, robot_node, img_node, dsr_node]
        for n in nodes:
            try: n.destroy_node()
            except Exception: pass
        cv2.destroyAllWindows()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()