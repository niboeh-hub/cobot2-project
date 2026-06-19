import os
os.environ["ROS_DOMAIN_ID"] = "84"

import argparse
import sys
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import DR_init  # 두산 로봇 전역 설정 모듈
from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ultralytics import YOLO

# rokey_web의 db_logger 사용
sys.path.insert(0, os.path.expanduser('~/rokey_web'))
try:
    import db_logger
except ImportError:
    pass

# ==========================================
# 두산 로봇 글로벌 초기화 (가장 먼저 실행되어야 함)
# ==========================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


PACKAGE_NAME = "wire_voice"
DEFAULT_TARGET_ORDER = (0, 1, 2)
DEFAULT_COLOR_CLASS_IDS = {
    "red": 0,
    "green": 1,
    "blue": 2,
}

COLOR_RANGES = {
    "red": [
        ((0, 100, 50), (10, 255, 255)),
        ((170, 100, 50), (180, 255, 255)),
    ],
    "green": [
        ((35, 80, 50), (85, 255, 255)),
    ],
    "blue": [
        ((95, 80, 50), (130, 255, 255)),
    ],
}


@dataclass
class WireDetection:
    class_id: int
    center_x: float
    confidence: float
    name: str


def format_detections(detections: Sequence[WireDetection]) -> str:
    if not detections:
        return "none"

    return ", ".join(
        f"{wire.name}(id={wire.class_id}, conf={wire.confidence:.2f}, "
        f"x={wire.center_x:.1f})"
        for wire in detections
    )


def get_default_model_path() -> str:
    try:
        package_share = get_package_share_directory(PACKAGE_NAME)
        installed_path = os.path.join(package_share, "resource", "best.pt")
        if os.path.exists(installed_path):
            return installed_path
    except PackageNotFoundError:
        pass

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_dir, "resource", "best.pt")


def parse_order(value: str) -> List[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("target_order는 예: '0,1,2' 형식이어야 합니다.") from exc


def class_name_map(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {idx: str(name) for idx, name in enumerate(names or [])}


def class_ids_by_color(names) -> Dict[str, int]:
    class_names = class_name_map(names)
    color_ids = DEFAULT_COLOR_CLASS_IDS.copy()

    for class_id, name in class_names.items():
        normalized = name.lower()
        for color_name in DEFAULT_COLOR_CLASS_IDS:
            if color_name in normalized:
                color_ids[color_name] = class_id

    return color_ids


def mask_center_x(mask_points: Sequence[Sequence[float]]) -> Optional[float]:
    points = np.asarray(mask_points, dtype=np.float32)
    if points.size == 0:
        return None

    moments = cv2.moments(points)
    if moments["m00"] == 0:
        return float(np.mean(points[:, 0]))
    return float(moments["m10"] / moments["m00"])


def largest_color_component(mask, min_area: int):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None, area

    return contour, area


def detect_wires_by_color(
    image,
    names,
    min_area: int = 250,
) -> List[WireDetection]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_ids = class_ids_by_color(names)
    detections: List[WireDetection] = []

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                ),
            )

        contour, area = largest_color_component(mask, min_area=min_area)
        if contour is None:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            points = contour.reshape(-1, 2)
            center_x = float(np.mean(points[:, 0]))
        else:
            center_x = float(moments["m10"] / moments["m00"])

        confidence = min(1.0, area / float(image.shape[0] * image.shape[1]))
        detections.append(
            WireDetection(
                class_id=color_ids[color_name],
                center_x=center_x,
                confidence=confidence,
                name=color_name,
            )
        )

    return detections


def collect_wire_detections(
    result,
    confidence_threshold: float,
    use_mask_centroid: bool = True,
) -> List[WireDetection]:
    boxes = result.boxes
    if boxes is None:
        return []

    names = class_name_map(result.names)
    mask_polygons = result.masks.xy if result.masks is not None else []
    detections: List[WireDetection] = []

    for index, box in enumerate(boxes):
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        center_x = (float(xyxy[0]) + float(xyxy[2])) / 2.0

        if use_mask_centroid and index < len(mask_polygons):
            mask_x = mask_center_x(mask_polygons[index])
            if mask_x is not None:
                center_x = mask_x

        detections.append(
            WireDetection(
                class_id=class_id,
                center_x=center_x,
                confidence=confidence,
                name=names.get(class_id, f"class_{class_id}"),
            )
        )

    return detections


def best_detection_per_target_class(
    detections: Sequence[WireDetection],
    target_order: Sequence[int],
) -> List[WireDetection]:
    target_ids = set(target_order)
    best_by_class: Dict[int, WireDetection] = {}

    for detection in detections:
        if detection.class_id not in target_ids:
            continue

        current_best = best_by_class.get(detection.class_id)
        if (
            current_best is None
            or detection.confidence > current_best.confidence
        ):
            best_by_class[detection.class_id] = detection

    if set(best_by_class) != target_ids:
        return []

    return list(best_by_class.values())


def check_order(
    detections: List[WireDetection],
    target_order: Sequence[int],
    source: str,
) -> Tuple[bool, List[WireDetection], str]:
    unique_detections = best_detection_per_target_class(
        detections,
        target_order,
    )

    if len(unique_detections) != 3:
        message = (
            f"FAIL({source}): detected {len(detections)} wires, "
            "expected one each for target classes; "
            f"detected={format_detections(detections)}"
        )
        return False, detections, message

    unique_detections.sort(key=lambda wire: wire.center_x)
    final_order = [wire.class_id for wire in unique_detections]
    final_names = [wire.name for wire in unique_detections]

    if final_order == list(target_order):
        return (
            True,
            unique_detections,
            f"PASS({source}): left-to-right order is {final_names}",
        )

    expected = [str(class_id) for class_id in target_order]
    return (
        False,
        unique_detections,
        f"FAIL({source}): left-to-right order is {final_order} "
        f"({final_names}), expected [{', '.join(expected)}]",
    )


def verify_wire_arrangement(
    model: YOLO,
    image,
    target_order: Sequence[int] = DEFAULT_TARGET_ORDER,
    confidence_threshold: float = 0.5,
    use_mask_centroid: bool = True,
    use_color_fallback: bool = True,
    color_min_area: int = 250,
) -> Tuple[bool, List[WireDetection], str]:
    results = model(image, conf=confidence_threshold, verbose=False)
    detections = collect_wire_detections(
        results[0],
        confidence_threshold=confidence_threshold,
        use_mask_centroid=use_mask_centroid,
    )

    yolo_ok, yolo_checked, yolo_message = check_order(
        detections,
        target_order=target_order,
        source="yolo",
    )
    if yolo_ok or not use_color_fallback:
        return yolo_ok, yolo_checked, yolo_message

    color_detections = detect_wires_by_color(
        image,
        results[0].names,
        min_area=color_min_area,
    )
    ok, checked_detections, message = check_order(
        color_detections,
        target_order=target_order,
        source="color",
    )

    yolo_summary = format_detections(detections)
    return ok, checked_detections, f"{message}; yolo_detected={yolo_summary}"


def draw_arrangement_result(
    image,
    ok: bool,
    detections: Sequence[WireDetection],
    target_order: Sequence[int] = DEFAULT_TARGET_ORDER,
):
    annotated = image.copy()
    label = "PASS" if ok else "FAIL"
    label_color = (0, 180, 0) if ok else (0, 0, 255)

    sorted_detections = sorted(detections, key=lambda w: w.center_x)

    for rank, wire in enumerate(sorted_detections, start=1):
        if ok:
            wire_color = (0, 180, 0)
        else:
            if len(sorted_detections) == len(target_order) and wire.class_id == target_order[rank - 1]:
                wire_color = (0, 180, 0)
            else:
                wire_color = (0, 0, 255)

        x = int(wire.center_x)
        cv2.line(annotated, (x, 0), (x, annotated.shape[0]), wire_color, 2)
        cv2.putText(
            annotated,
            f"{rank}: {wire.name} ({wire.confidence:.2f})",
            (max(10, x - 80), 40 + rank * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            wire_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        label_color,
        3,
        cv2.LINE_AA,
    )
    return annotated


class WireArrangementCheckerNode(Node):
    def __init__(self):
        super().__init__("wire_arrangement_checker_node")

        self.declare_parameter("model_path", get_default_model_path())
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("target_order", "0,1,2")
        self.declare_parameter("use_mask_centroid", True)
        self.declare_parameter("use_color_fallback", True)
        self.declare_parameter("color_min_area", 250)
        self.declare_parameter("show_window", True)

        self._robot_motion_ready = False
        self._movel = None
        self._movejx = None # 안전한 이동 명령
        self._mwait = None
        self._posx = None
        
        model_path = self.get_parameter("model_path").value
        image_topic = self.get_parameter("image_topic").value

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_annotated = None

        self.result_pub = self.create_publisher(Image, '/arrangement_result', 10)
        self.guide_pub = self.create_publisher(String, '/arrangement_guide_request', 10)

        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.create_service(
            Trigger,
            "verify_wire_arrangement",
            self.verify_callback,
        )

        self.get_logger().info(f"YOLO model loaded: {model_path}")
        self.get_logger().info(f"YOLO classes: {self.model.names}")
        self.get_logger().info(f"Subscribing image topic: {image_topic}")
        self.get_logger().info("Service ready: /verify_wire_arrangement")

    def setup_robot_motion(self):
        try:
            import sys
            import DR_init

            # 1. 모든 버전의 DR_init 모듈에 설정을 주입하여 임포트 시점 문제를 완벽히 예방합니다.
            for name, module in list(sys.modules.items()):
                if 'DR_init' in name:
                    try:
                        module.__dsr__node = self
                        module.__dsr__id = ROBOT_ID
                        module.__dsr__model = ROBOT_MODEL
                    except Exception:
                        pass
            
            DR_init.__dsr__node = self
            DR_init.__dsr__id = ROBOT_ID
            DR_init.__dsr__model = ROBOT_MODEL

            # 2. 혹시 이전에 불완전하게 캐싱되었을 수 있으므로 DSR_ROBOT2를 캐시에서 지우고 새로 로드합니다.
            if 'DSR_ROBOT2' in sys.modules:
                del sys.modules['DSR_ROBOT2']

            import DSR_ROBOT2

            # 로봇 제어 함수 맵핑
            self._movel = DSR_ROBOT2.movel
            self._mwait = DSR_ROBOT2.mwait
            self._posx = DSR_ROBOT2.posx
            
            # 💡 직선 경로(movel)가 꼬일 때를 대비한 관절 곡선 이동(movejx) 추가
            if hasattr(DSR_ROBOT2, 'movejx'):
                self._movejx = DSR_ROBOT2.movejx
            
            DSR_ROBOT2.set_robot_mode(DSR_ROBOT2.ROBOT_MODE_MANUAL)
            DSR_ROBOT2.set_tool("Tool Weight") 
            DSR_ROBOT2.set_tcp("GripperDA_v1")

            DSR_ROBOT2.set_robot_mode(DSR_ROBOT2.ROBOT_MODE_AUTONOMOUS)
            
            # 사용자 정상 코드처럼 2초 대기하여 Servo ON 및 모드 전환 확실히 적용
            time.sleep(2.0) 

            self._robot_motion_ready = True
            self.get_logger().info("✅ Doosan robot API successfully loaded & Autonomous Mode ON.")
        except Exception as exc:
            self.get_logger().error(f"❌ Failed to load Doosan API: {exc}")

    def startup_move(self):
        if not self._robot_motion_ready:
            self.get_logger().error("Cannot move robot: API not ready.")
            return

        scan_pose = [565.11, -78.51, 487.64, 155.73, -180.0, 155.66]

        self.get_logger().info(f"🚀 노드 구동 완료. 로봇을 scan_pose로 안전 이동합니다: {scan_pose}")
        
        try:
            target_pose = self._posx(scan_pose)
            
            # 💡 movel(직선 이동)은 현재 위치에 따라 특이점에 걸려 캔슬될 수 있으므로,
            # 도착지는 XYZ로 유지하되 관절 기반으로 안전하게 꺾어서 이동하는 movejx를 우선 사용합니다.
            ret = -1
            if self._movejx:
                ret = self._movejx(target_pose, vel=60.0, acc=60.0, radius=0.0)
            else:
                ret = self._movel(target_pose, vel=60.0, acc=60.0)
                
            mwait_ret = self._mwait()  # 여기서 확실히 이동을 마칠 때까지 대기합니다.
            
            if ret == -1 or mwait_ret == -1:
                self.get_logger().error(
                    "❌ 로봇 이동 명령이 실패했습니다(리턴값 -1). "
                    "1) 로봇의 비상정지(E-stop)가 풀려있고 Servo ON 상태인지 확인하세요. "
                    "2) 로봇 펜던트 키 스위치가 AUTO 모드이며 화면 상에서 AUTONOMOUS 모드가 켜져있는지 확인하세요. "
                    "3) 목표 좌표 [565.11, -78.51, 487.64]가 작업 영역(Reach) 내에 있는지 확인하세요."
                )
            else:
                self.get_logger().info("✅ 스캔 위치로 로봇 이동 완료! 비전 검사 준비됨.")
        except Exception as exc:
            self.get_logger().error(f"❌ 로봇 이동 중 예외 발생: {exc}")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            # 항상 발행 — 검증 결과 있으면 그거, 없으면 원본
            out_frame = self.latest_annotated if self.latest_annotated is not None else self.latest_frame
            try:
                msg_out = self.bridge.cv2_to_imgmsg(out_frame, 'bgr8')
                self.result_pub.publish(msg_out)
            except Exception:
                pass
        except Exception as exc:
            pass

    def verify_callback(self, request, response):
        if self.latest_frame is None:
            response.success = False
            response.message = "FAIL: no camera frame received yet"
            return response

        try:
            target_order = parse_order(self.get_parameter("target_order").value)
            confidence = float(self.get_parameter("confidence").value)
            use_mask_centroid = bool(self.get_parameter("use_mask_centroid").value)
            use_color_fallback = bool(self.get_parameter("use_color_fallback").value)
            color_min_area = int(self.get_parameter("color_min_area").value)

            ok, detections, message = verify_wire_arrangement(
                self.model,
                self.latest_frame,
                target_order=target_order,
                confidence_threshold=confidence,
                use_mask_centroid=use_mask_centroid,
                use_color_fallback=use_color_fallback,
                color_min_area=color_min_area,
            )

            try:
                current_names = [wire.name for wire in sorted(detections, key=lambda w: w.center_x)]
                target_names = []
                for class_id in target_order:
                    for name, cid in DEFAULT_COLOR_CLASS_IDS.items():
                        if cid == class_id:
                            target_names.append(name)
                            break

                req_msg = String()
                req_msg.data = json.dumps({
                    "current_order": ", ".join(current_names),
                    "target_order": ", ".join(target_names)
                })
                self.guide_pub.publish(req_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish arrangement guide request: {e}")

            self.latest_annotated = draw_arrangement_result(
                self.latest_frame,
                ok,
                detections,
                target_order=target_order,
            )
            self.get_logger().info(message)
            response.success = ok
            response.message = message
            if self.latest_annotated is not None:
                try:
                    msg_out = self.bridge.cv2_to_imgmsg(self.latest_annotated, 'bgr8')
                    self.result_pub.publish(msg_out)
                except Exception:
                    pass
        except Exception as e:
            self.get_logger().error(f"Error verifying arrangement: {e}")
            try:
                db_logger.set_error(f"비전 검증 오류: {e}", source="vision_node")
            except Exception:
                pass
            response.success = False
            response.message = f"FAIL: Exception occurred: {e}"

        return response


def run_image_check(args) -> int:
    model = YOLO(args.model_path)
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    ok, detections, message = verify_wire_arrangement(
        model,
        image,
        target_order=parse_order(args.target_order),
        confidence_threshold=args.confidence,
        use_mask_centroid=not args.box_center,
        use_color_fallback=not args.no_color_fallback,
        color_min_area=args.color_min_area,
    )

    print(message)
    print("left_to_right:", [wire.class_id for wire in detections])

    if args.output:
        annotated = draw_arrangement_result(
            image,
            ok,
            detections,
            target_order=parse_order(args.target_order),
        )
        cv2.imwrite(args.output, annotated)
        print(f"saved annotated image: {args.output}")

    return 0 if ok else 1


def run_ros_node() -> None:
    rclpy.init()
    node = WireArrangementCheckerNode()

    # 1. 노드 생성 후 최우선으로 DR_init과 바인딩
    DR_init.__dsr__node = node
    
    # 2. 로봇 제어 API 로드 및 세팅
    # node.setup_robot_motion()

    # 3. ROS2 콜백 루프에 들어가기 전에, 확실하게 로봇을 먼저 이동
    # node.startup_move()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            
            if bool(node.get_parameter("show_window").value):
                frame = node.latest_annotated
                if frame is None:
                    frame = node.latest_frame
                if frame is not None:
                    cv2.imshow("Wire Arrangement Checker", frame)
                
                # 항상 cv2.waitKey(1)을 호출하여 GUI 이벤트를 강제 갱신하고 창이 얼거나 안 뜨는 현상을 완벽히 방지합니다.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8 segmentation based 3-wire arrangement checker."
    )
    parser.add_argument("--image", dest="image_path")
    parser.add_argument("--model-path", default=get_default_model_path())
    parser.add_argument("--target-order", default="0,1,2")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--box-center", action="store_true")
    parser.add_argument("--no-color-fallback", action="store_true")
    parser.add_argument("--color-min-area", type=int, default=250)
    parser.add_argument("--output")
    return parser


def main(args=None):
    parser = build_parser()
    parsed_args, _ = parser.parse_known_args(args=args)

    if parsed_args.image_path:
        return run_image_check(parsed_args)

    run_ros_node()
    return None


if __name__ == "__main__":
    main()