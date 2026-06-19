import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from od_msg.srv import SrvDepthPosition
from pick_and_place_wire.realsense import ImgNode
import threading
import time
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        
        # 1. 초기화 단계: 아직 Executor(백그라운드 통신)가 안 켜졌으니 수동으로 spin을 돌려 데이터를 받습니다.
        self.get_logger().info("Waiting for camera intrinsics...")
        self.intrinsics = self.img_node.get_camera_intrinsic()
        while self.intrinsics is None:
            rclpy.spin_once(self.img_node, timeout_sec=0.1)
            self.intrinsics = self.img_node.get_camera_intrinsic()
        self.get_logger().info("Camera intrinsics received!")
        
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        
        self.get_logger().info("Loading YOLO Segmentation Model...")
        self.yolo_model = YOLO("/home/ji-hyeon/cobot2_ws/src/pick_and_place_wire/resource/best.pt") # 실제 파일 경로 확인
        
        cv2.namedWindow("YOLO Wire Auto-Pick")
        self.target_point = None
        self.display_frame = None
        
        self.get_logger().info("YOLO Auto-Detection Node initialized.")

    def update_gui(self):
        """메인 스레드: 화면 갱신만 전담. (매우 안전함)"""
        frame = self.img_node.get_color_frame()
        
        if frame is not None:
            self.display_frame = frame.copy()
            
            if self.target_point:
                cv2.circle(self.display_frame, self.target_point, 5, (0, 0, 255), -1)
                cv2.putText(self.display_frame, "Target", (self.target_point[0]+10, self.target_point[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            h, w = self.display_frame.shape[:2]
            cv2.line(self.display_frame, (w//2 - 10, h//2), (w//2 + 10, h//2), (0, 255, 0), 2)
            cv2.line(self.display_frame, (w//2, h//2 - 10), (w//2, h//2 + 10), (0, 255, 0), 2)
                
            cv2.imshow("YOLO Wire Auto-Pick", self.display_frame)
            
        cv2.waitKey(10)

    def handle_get_depth(self, request, response):
        """백그라운드 스레드: ROS 서비스 및 YOLO 추론 전담"""
        self.get_logger().info("Running YOLO Inference...")
        self.target_point = None
        
        # 2. 런타임 단계: 이미 Executor가 통신을 처리하고 있으므로 time.sleep으로 조용히 기다립니다.
        frame = self.img_node.get_color_frame()
        while frame is None or (isinstance(frame, np.ndarray) and not frame.any()):
            time.sleep(0.05)
            frame = self.img_node.get_color_frame()
        
        results = self.yolo_model(frame, verbose=False)
        
        if results[0].masks is None:
            self.get_logger().warn("No wire detected in the frame!")
            response.depth_position = [0.0, 0.0, 0.0]
            return response
            
        polygon_points = results[0].masks.xy[0] 
        
        if len(polygon_points) == 0:
            self.get_logger().warn("Mask points are empty!")
            response.depth_position = [0.0, 0.0, 0.0]
            return response

        h, w = frame.shape[:2]
        center_point = np.array([w / 2, h / 2])
        
        distances = np.linalg.norm(polygon_points - center_point, axis=1)
        closest_idx = np.argmin(distances)
        
        cx = int(polygon_points[closest_idx][0])
        cy = int(polygon_points[closest_idx][1])
        
        self.target_point = (cx, cy)
        self.get_logger().info(f"YOLO selected closest point: ({cx}, {cy})")
        
        cz = self._get_depth(cx, cy)
        
        if cz is None or cz == 0:
            self.get_logger().warn("Invalid depth at YOLO target point.")
            response.depth_position = [0.0, 0.0, 0.0]
        else:
            coords = self._pixel_to_camera_coords(cx, cy, cz)
            response.depth_position = [float(x) for x in coords]
            
        return response

    def _get_depth(self, x, y):
        frame = self.img_node.get_depth_frame()
        if frame is None: return None
        try: return frame[y, x]
        except IndexError: return None

    def _pixel_to_camera_coords(self, x, y, z):
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        ppx, ppy = self.intrinsics['ppx'], self.intrinsics['ppy']
        return ((x - ppx) * z / fx, (y - ppy) * z / fy, z)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.img_node)
    
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        while rclpy.ok():
            node.update_gui()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        node.img_node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()