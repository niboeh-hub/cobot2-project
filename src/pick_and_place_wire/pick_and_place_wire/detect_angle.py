import os
import sys

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

import DR_init

# 구독할 카메라 컬러 토픽
COLOR_TOPIC = '/camera/camera/color/image_raw'

# 학습한 와이어 모델 경로. 실제 best.pt 위치로 바꿀 것.
MODEL_PATH = os.path.expanduser('/home/ji-hyeon/cobot2_ws/src/pick_and_place_wire/resource/best_bb.pt')
CONF_THRESHOLD = 0.5

# 진한 초록 HSV 범위
LOWER_GREEN = np.array([40, 80, 40])
UPPER_GREEN = np.array([85, 255, 255])

# 가까운 지점 주변에서 각도를 구할 영역의 반쪽 크기 (픽셀)
LOCAL_HALF = 40

# 6번 조인트 보정값. 와이어 방향에서 이만큼 틀어 가로질러 잡음.
JOINT6_OFFSET = -90.0

# --- 로봇 설정 ---
ROBOT_ID = 'dsr01'
ROBOT_MODEL = 'm0609'
# 처음엔 느리게! 검증되면 올릴 것.
JOINT_VEL = 20
JOINT_ACC = 20

# 6번 조인트 가동 범위 (도). 계산값이 이 범위를 벗어나면 회전 안 함.
JOINT6_MIN = -360.0
JOINT6_MAX = 360.0


# ===================================================================
# 두산 로봇 라이브러리 초기화 (DSR_ROBOT2 import 전에 먼저)
# ===================================================================
rclpy.init()
dsr_node = rclpy.create_node('detect_angle', namespace=ROBOT_ID)
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, get_current_posj
except ImportError as e:
    print(f'Error importing DSR_ROBOT2: {e}')
    sys.exit()


class DetectAngle:
    """YOLO로 와이어를 찾고 각도를 추정하며,
    'g' 키를 누르면 6번 조인트를 그 각도에 맞춰 회전한다.
    """

    def __init__(self, node):
        self.node = node
        self.bridge = CvBridge()
        self.frame = None
        self.latest_joint6 = None    # 가장 최근에 계산된 6번 목표값

        self.model = YOLO(MODEL_PATH)
        self.node.get_logger().info(f'Model loaded: {MODEL_PATH}')

        self.node.create_subscription(
            Image, COLOR_TOPIC, self._callback, 10)
        self.node.get_logger().info(
            "DetectAngle ready. Press 'g' to rotate joint 6, 'q' to quit.")

    def _callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def pick_target_box(self, image):
        """YOLO를 돌려 가장 확신도 높은 와이어 박스를 고른다."""
        results = self.model(image, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
        best = max(boxes, key=lambda b: float(b.conf[0]))
        return [int(v) for v in best.xyxy[0]]

    def angle_in_box(self, image, box):
        """박스 안에서 화면 중앙에 가장 가까운 와이어 부분의 각도를 구한다."""
        h, w = image.shape[:2]
        center = np.array([w // 2, h // 2])

        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        points = cv2.findNonZero(mask)
        if points is None or len(points) < 50:
            return None, None, None

        # roi 기준 좌표 -> 전체 이미지 기준
        pts = points.reshape(-1, 2) + np.array([x1, y1])
        dists = np.linalg.norm(pts - center, axis=1)
        nearest = pts[np.argmin(dists)]

        nx, ny = nearest
        lx1 = max(nx - LOCAL_HALF, 0)
        ly1 = max(ny - LOCAL_HALF, 0)
        lx2 = min(nx + LOCAL_HALF, w)
        ly2 = min(ny + LOCAL_HALF, h)

        local_roi = image[ly1:ly2, lx1:lx2]
        local_hsv = cv2.cvtColor(local_roi, cv2.COLOR_BGR2HSV)
        local_mask = cv2.inRange(local_hsv, LOWER_GREEN, UPPER_GREEN)

        local_points = cv2.findNonZero(local_mask)
        if local_points is None or len(local_points) < 10:
            return None, None, None

        rect = cv2.minAreaRect(local_points)
        (_, _), (rw, rh), angle = rect
        if rw < rh:
            angle = angle + 90

        return angle, nearest, (lx1, ly1, lx2, ly2)

    def process(self, image):
        """박스 찾기 -> 각도 -> 화면 표시. latest_joint6를 갱신한다."""
        display = image.copy()
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        cv2.circle(display, center, 6, (255, 0, 0), -1)

        # 이번 프레임 처리 시작 시 일단 초기화
        self.latest_joint6 = None

        box = self.pick_target_box(image)
        if box is None:
            cv2.putText(display, 'no wire', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return display

        x1, y1, x2, y2 = box
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        angle, nearest, roi_range = self.angle_in_box(image, box)
        if angle is None:
            cv2.putText(display, 'angle failed', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return display

        cv2.circle(display, tuple(nearest), 6, (0, 255, 0), -1)
        cv2.line(display, center, tuple(nearest), (255, 255, 0), 1)
        lx1, ly1, lx2, ly2 = roi_range
        cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (0, 255, 0), 1)

        joint6 = angle + JOINT6_OFFSET
        self.latest_joint6 = joint6    # g 키 눌렀을 때 쓸 값 저장

        cv2.putText(display, f'angle: {angle:.1f}',
                    (nearest[0] - 40, nearest[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, f'joint6: {joint6:.1f}',
                    (nearest[0] - 40, nearest[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, "press 'g' to rotate", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return display

    def rotate_joint6(self):
        """현재 6번 조인트 값에 화면 joint6 값을 더해 회전한다.

        더한 값이 6번 범위를 벗어나면, angle에 90을 더한 대안값으로 재시도한다.
        """
        if self.latest_joint6 is None:
            self.node.get_logger().warn('No angle available. Cannot rotate.')
            return

        current = get_current_posj()
        current6 = current[5]

        # 1순위: 화면에 뜬 joint6 값 그대로 더하기
        delta = self.latest_joint6
        target6 = current6 + delta

        # 범위를 벗어나면 2순위: angle에 90을 더한 값으로 다시 계산
        if not (JOINT6_MIN <= target6 <= JOINT6_MAX):
            self.node.get_logger().warn(
                f'joint6 target {target6:.1f} out of range. '
                f'Trying alternative (+180)...')
            # delta = (angle + JOINT6_OFFSET) 였으므로, angle에 90을 더하면
            # delta 도 90 만큼 커진다
            alt_delta = delta + 180.0
            target6 = current6 + alt_delta

            # 그래도 범위를 벗어나면 -90 쪽도 시도
            if not (JOINT6_MIN <= target6 <= JOINT6_MAX):
                alt_delta = delta - 180.0
                target6 = current6 + alt_delta

            # 둘 다 안 되면 포기
            if not (JOINT6_MIN <= target6 <= JOINT6_MAX):
                self.node.get_logger().warn(
                    'No valid joint6 target within range. Skipping.')
                return

            self.node.get_logger().info(
                f'Using alternative delta: {alt_delta:.1f}')
            delta = alt_delta

        # 1~5번은 그대로, 6번만 교체
        target_pose = list(current)
        target_pose[5] = target6

        self.node.get_logger().info(
            f'Rotating joint6: {current6:.1f} + {delta:.1f} = {target6:.1f}')
        movej(target_pose, vel=JOINT_VEL, acc=JOINT_ACC)
        self.node.get_logger().info('Rotation done.')


def main():
    handler = DetectAngle(dsr_node)

    try:
        while rclpy.ok():
            rclpy.spin_once(dsr_node, timeout_sec=0.1)
            if handler.frame is None:
                continue

            display = handler.process(handler.frame)
            cv2.imshow('detect + angle', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                # g 키: 지금 화면의 각도로 6번 조인트 회전
                handler.rotate_joint6()

    finally:
        cv2.destroyAllWindows()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()