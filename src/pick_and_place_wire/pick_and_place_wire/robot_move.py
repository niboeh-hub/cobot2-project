import os
import time
import sys
from scipy.spatial.transform import Rotation
import numpy as np
import rclpy
from rclpy.node import Node
import DR_init

from od_msg.srv import SrvDepthPosition
from ament_index_python.packages import get_package_share_directory
from pick_and_place_wire.onrobot import RG

PACKAGE_NAME = "pick_and_place_wire"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

tool_dict = {1: "drill", 2: "hammer", 3: "pliers", 4: "screwdriver", 5: "wrench"}

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
BUCKET_POS = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
dsr_node = rclpy.create_node("rokey_simple_move", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node

try:
    from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, trans
except ImportError as e:
    print(f"Error importing DSR_ROBOT2: {e}")
    sys.exit()

########### Gripper Setup. Do not modify this area ############

GRIPPER_NAME = "rg2"
TOOLCHANGER_IP = "192.168.1.1"
TOOLCHANGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHANGER_IP, TOOLCHANGER_PORT)


########### Robot Controller ############


class RobotController(Node):
    def __init__(self):
        super().__init__("pick_and_place")
        self.init_robot()
        self.depth_client = self.create_client(SrvDepthPosition, "/get_3d_position")
        while not self.depth_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("Waiting for depth position service...")
        self.depth_request = SrvDepthPosition.Request()
        self.robot_control()

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def transform_to_base(self, camera_coords, gripper2cam_path, robot_pos):
        """
        Converts 3D coordinates from the camera coordinate system
        to the robot's base coordinate system.
        """
        gripper2cam = np.load(gripper2cam_path)
        coord = np.append(np.array(camera_coords), 1)  # Homogeneous coordinate

        x, y, z, rx, ry, rz = robot_pos
        base2gripper = self.get_robot_pose_matrix(x, y, z, rx, ry, rz)

        # 좌표 변환 (그리퍼 → 베이스)
        base2cam = base2gripper @ gripper2cam
        td_coord = np.dot(base2cam, coord)

        return td_coord[:3]

    def robot_control(self):
        print("====================================")
        user_input = input("Press Enter to start clicking, or 'q' to quit: ")
        if user_input.lower() == "q":
            self.get_logger().info("Quit the program...")
            sys.exit()

        # Target 문자열은 더 이상 중요하지 않으므로 'click'으로 전송
        self.depth_request.target = "click"
        self.get_logger().info("Call depth position service. Please click the image window.")
        
        depth_future = self.depth_client.call_async(self.depth_request)
        rclpy.spin_until_future_complete(self, depth_future)

        if depth_future.result():
            result = depth_future.result().depth_position.tolist()
            if sum(result) == 0:
                print("Failed to get target position (Invalid depth).")
                return

            gripper2cam_path = os.path.join(
                PACKAGE_PATH, "resource", "T_gripper2camera.npy"
            )
            robot_posx = get_current_posx()[0]
            
            # 카메라 좌표 -> 로봇 베이스 좌표 변환 (행렬 연산으로 완벽한 Z값 도출)
            td_coord = self.transform_to_base(result, gripper2cam_path, robot_posx)

            if td_coord[2] and sum(td_coord) != 0:
                # 전선은 바닥에 붙어있거나 얇으므로 그리퍼 손가락 길이를 고려해 미세 조정 필요
                Z_OFFSET = -25  # 기존 코드 유지. 전선을 더 깊게 잡으려면 이 값을 더 빼주세요 (예: -10)
                td_coord[2] += Z_OFFSET
                td_coord[2] = max(td_coord[2], 2)  # MIN_DEPTH 방어코드

            target_pos = list(td_coord[:3]) + robot_posx[3:]

            self.get_logger().info(f"Target Base Position: {target_pos}")
            self.pick_and_place_target(target_pos)
            self.init_robot()

    def init_robot(self):
        JReady = [0, 0, 90, 0, 90, 0]
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()

    def pick_and_place_target(self, target_pos):
        # delete
        target_pos[3] += 0

        # 1. 타겟 위치로 이동
        movel(target_pos, vel=VELOCITY, acc=ACC)
        mwait()
        
        # 2. 그리퍼 닫기 (전선 파지)
        gripper.close_gripper()
        time.sleep(1.5)

        # 3. Base(바닥) 기준으로 3cm (30mm) 수직 상승
        # trans의 3번째 인자 기본값이 0(DR_BASE)이므로, 절대적인 위쪽(하늘)으로 30mm 올라갑니다.
        # target_pos_up = trans(target_pos, [0, 0, -30, 0, 0, 0]).tolist()
        target_pos_up = list(target_pos)
        target_pos_up[2] += 100
        movel(target_pos_up, vel=VELOCITY, acc=ACC)
        mwait()

        # 4. Tool(그리퍼) 기준으로 수평 방향으로 빼기
        # 그리퍼 방향을 기준으로 움직이려면 trans 함수의 3번째 인자에 1(DR_TOOL)을 넣어줍니다.
        PULL_DISTANCE = 150  # 전선을 옆으로 빼낼 거리 (단위: mm, 현재 10cm)

        # 우리가 정의해둔 함수를 이용해 로봇 그리퍼의 현재 3D 기울기(Matrix)를 가져옵니다.
        base2gripper = self.get_robot_pose_matrix(*target_pos_up)
        
        # 그리퍼(Tool) 입장에서의 이동 벡터를 만듭니다. (X축 방향으로 100mm 이동)
        # ※ 배열 끝의 1.0은 행렬 연산을 위한 동차좌표계용 필수 상수입니다.
        tool_movement = np.array([0.0, -PULL_DISTANCE, 0.0, 1.0]) 
        
        # 행렬 곱셈(@)을 통해 로봇이 거부할 수 없는 완벽한 Base(바닥) 절대 좌표를 산출합니다!
        new_base_coords = base2gripper @ tool_movement
        
        # 새로운 3D 위치(x, y, z)와 기존 로봇 각도(rx, ry, rz)를 합쳐 목표점 완성
        target_pos_pull = list(new_base_coords[:3]) + target_pos_up[3:]
        
        # # 주의: 온로봇 그리퍼가 로봇에 체결된 방향에 따라 '앞뒤' 방향이 X축일 수도, Y축일 수도 있습니다.
        # # 만약 로봇이 정면이 아니라 옆으로(게걸음처럼) 움직인다면, 아래 리스트를 
        # # [0, PULL_DISTANCE, 0, 0, 0, 0] (Y축 이동) 으로 변경해 주세요.
        # target_pos_pull = trans(target_pos_up, [0, 0, -PULL_DISTANCE, 0, 0, 0], 1)

        # if hasattr(target_pos_pull, 'tolist'):
        #     target_pos_pull = target_pos_pull.tolist()
        
        movel(target_pos_pull, vel=VELOCITY, acc=ACC)
        mwait()

        # 5. 그리퍼 열기 (전선 놓기)
        gripper.open_gripper()
        while gripper.get_status()[0]:
            time.sleep(0.5)


def main(args=None):
    node = RobotController()
    while rclpy.ok():
        node.robot_control()
    rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
    main()
