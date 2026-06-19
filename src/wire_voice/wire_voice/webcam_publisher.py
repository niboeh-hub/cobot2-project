"""사용자 노트북 웹캠을 ROS 토픽으로 발행 (웹 최적화 버전)"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        
        # 💡 [팁] 만약 v4l2-ctl로 확인한 인덱스가 2번이라면 아래 숫자를 2로 고쳐주세요!
        camera_index = 8
        
        self.cap = cv2.VideoCapture(camera_index)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/user_webcam', 10)
        self.timer = self.create_timer(0.05, self.tick)  # 20Hz

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ 카메라(인덱스 {camera_index}번)를 열 수 없습니다! 번호를 확인하세요.")
        else:
            self.get_logger().info(f"✅ 카메라(인덱스 {camera_index}번) 연결 성공!")

    def tick(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # OpenCV는 BGR로 받으므로 bgr8로 발행 (표준)
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()