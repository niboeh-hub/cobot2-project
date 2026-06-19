# 🤖 ROKEY — Wire Pick-and-Place Robot System

> ROS 2 기반 협동로봇(Doosan M0609) 와이어 정렬 자동화 시스템  
> 음성 명령 · 손 제스처 · 비전 AI를 통합한 멀티모달 로봇 제어 플랫폼

---

## 📌 프로젝트 개요

ROKEY는 산업 현장의 와이어(전선) 정렬 작업을 자동화하는 시스템입니다.  
사용자는 세 가지 방식 중 하나를 선택해 로봇을 제어할 수 있습니다.

| 모드 | 설명 |
|------|------|
| **LLM Mode** | "빨강 초록 파랑 연결해"처럼 음성으로 색 순서를 지정하면 로봇이 자동으로 와이어를 집어 꽂음 |
| **Hand Mode** | 웹캠으로 손 위치를 인식해 로봇팔을 속도 기반으로 조그(jog) 제어 |
| **Shadow Mode** | 손의 절대 위치를 로봇 좌표에 1:1 미러링 (섀도우 동기) |

작업 완료 후 카메라로 와이어 정렬 상태를 자동 검증하고, 결과를 웹 대시보드와 DB에 기록합니다.

---

## 🗂️ 전체 아키텍처

```
[웹 대시보드 (Flask + Firebase)]
        │
        ├─ 음성 명령 ──→ [get_wire_keyword.py / get_hand_command.py]
        │                      │ Wakeword → STT → GPT-4o → 색 순서 추출
        │                      │
        ├─ 모드 선택 ──→ [mode_manager.py]
        │                      │ subprocess로 모드 스크립트 실행/종료
        │                      │
        ├─ 로봇 제어 ──→ [wire_pick / tracking.py / jog_tracking.py]
        │                      │ DSR_ROBOT2 API → Doosan M0609
        │                      │
        ├─ 비전 ───────→ [detection.py / detect_angle.py / wire_arrangement_checker.py]
        │                      │ YOLOv8 + RealSense D435
        │                      │
        └─ 상태 동기 ──→ [mission_bridge.py / arrangement_judge.py]
                               │ Firebase Realtime DB ↔ SQLite
```

---

## 📁 파일 구조

```
rokey_ws/
├── pick_and_place_wire/          # 로봇 제어 패키지
│   ├── wire_pick40.py            # LLM 모드 메인 픽앤플레이스
│   ├── robot_move.py             # 카메라→로봇 좌표 변환 및 이동
│   ├── tracking.py               # Shadow 모드 (1Euro 필터 + 이동평균)
│   ├── jog_tracking.py           # Hand 모드 (속도 기반 jog 제어)
│   ├── detection.py              # YOLO 세그멘테이션 + 3D 좌표 서비스
│   ├── detect_angle.py           # 와이어 각도 추정 + 6번 조인트 회전
│   ├── yolo.py                   # YOLOv8 다중 프레임 집계 / IoU 융합
│   ├── realsense.py              # RealSense 카메라 ROS 구독 노드
│   ├── onrobot.py                # OnRobot RG2/RG6 그리퍼 Modbus TCP 드라이버
│   └── mode_manager.py           # 모드 전환 매니저 (subprocess 관리)
│
├── voice_processing/             # 음성 처리 패키지
│   ├── get_wire_keyword.py       # LLM 모드 음성 명령 처리 (색 순서 추출)
│   ├── get_hand_command.py       # Hand 모드 음성 명령 처리 (start/verify)
│   ├── wire_guide_node.py        # 정렬 가이드 TTS 피드백 (장난기 넘치는 AI 코치)
│   ├── MicController.py          # PyAudio 마이크 스트림 관리
│   ├── stt.py                    # OpenAI Whisper STT
│   ├── wakeup_word.py            # "헬로 로키" 웨이크워드 감지
│   ├── mission_bridge.py         # /wire_connected 토픽 → Firebase 진행 상황 갱신
│   ├── hand_verify_bridge.py     # /hand_verify 토픽 → 정렬 검증 서비스 호출
│   └── arrangement_judge.py      # 와이어 정렬 자동 판정 노드
│
├── wire_arrangement_checker/     # 정렬 검증 패키지
│   └── wire_arrangement_checker.py  # YOLO + HSV로 정렬 순서 판별
│
├── rokey_web/                    # 웹 백엔드
│   ├── app.py                    # Flask 서버
│   ├── db_logger.py              # SQLite + Firebase 통합 DB 모듈
│   ├── init_db.py                # SQLite DB 초기화 스크립트
│   ├── webcam_publisher.py       # 웹캠 → ROS 토픽 퍼블리셔
│   └── templates/
│       ├── index.html            # 메인 허브 페이지
│       ├── operation.html        # LLM 모드 임무 화면
│       ├── hand_operation.html   # Hand 모드 임무 화면
│       ├── shadow_operation.html # Shadow 모드 임무 화면
│       ├── logs.html             # 작업 이력 조회
│       └── _error_bar.html       # 공통 에러 알림 컴포넌트
```

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **로봇** | Doosan M0609 협동로봇, DSR_ROBOT2 API |
| **그리퍼** | OnRobot RG2 (Modbus TCP) |
| **카메라** | Intel RealSense D435 (RGB-D) |
| **미들웨어** | ROS 2 (Humble) |
| **비전 AI** | YOLOv8 (Ultralytics), OpenCV, MediaPipe |
| **음성 AI** | OpenAI Whisper (STT), GPT-4o (LLM), OpenAI TTS |
| **웨이크워드** | openWakeWord (커스텀 모델: "헬로 로키") |
| **웹** | Flask, Firebase Realtime DB, rosbridge_websocket |
| **DB** | SQLite (임무 기록), Firebase (실시간 상태) |
| **언어** | Python 3.10+ |

---

## ⚙️ 설치 및 환경 설정

### 사전 요구사항

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- Intel RealSense SDK 2.0
- Doosan Robot SDK (DR_init, DSR_ROBOT2)

### 패키지 설치

```bash
# ROS 2 의존성
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs

# Python 패키지
pip install ultralytics opencv-python mediapipe pymodbus \
            openai langchain langchain-openai openwakeword \
            pyaudio sounddevice scipy firebase-admin flask \
            python-dotenv
```

### 환경 변수 설정

`voice_processing/resource/.env` 파일을 생성합니다.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Firebase 설정

1. Firebase 콘솔에서 Realtime Database 생성
2. 서비스 계정 키(JSON)를 `rokey_web/db/` 에 저장
3. `db_logger.py` 의 `FIREBASE_KEY_PATH`, `FIREBASE_DB_URL` 수정

### DB 초기화

```bash
cd rokey_web
python3 init_db.py
```

---

## 🚀 실행 방법

### 1. ROS 2 워크스페이스 빌드

```bash
cd ~/rokey_ws
colcon build
source install/setup.bash
```

### 2. 노드 실행 (각 터미널에서)

```bash
# 모드 매니저 (항상 먼저 실행)
ros2 run pick_and_place_wire mode_manager

# 비전 노드 (와이어 감지)
ros2 run pick_and_place_wire detection

# 정렬 검증 노드
ros2 run wire_arrangement_checker wire_arrangement_checker

# LLM 모드 음성 처리
ros2 run voice_processing get_wire_keyword

# Hand 모드 음성 처리
ros2 run voice_processing get_hand_command

# 임무 진행 상황 브릿지
ros2 run voice_processing mission_bridge

# 정렬 자동 판정
ros2 run voice_processing arrangement_judge

# 웹캠 퍼블리셔
ros2 run voice_processing webcam_publisher

# Flask 웹 서버
cd rokey_web && python3 app.py
```

### 3. 웹 대시보드 접속

브라우저에서 `http://localhost:5000` 접속

---

## 🎮 사용 방법

### LLM Mode (음성 와이어 정렬)

1. 웹에서 **LLM Mode** 선택 → 마이크 활성화 버튼 클릭
2. **"헬로 로키"** 웨이크워드 발화
3. **"빨강 초록 파랑 연결해"** 와 같이 색 순서 명령
4. 로봇이 자동으로 와이어를 순서대로 집어 꽂음
5. 완료 후 카메라가 정렬 상태를 자동 검증하고 결과 표시

지원 색상: `red (빨강)`, `green (초록)`, `blue (파랑)`

### Hand Mode (손 제스처 조종)

1. 웹에서 **Hand Mode** 선택
2. **"헬로 로키 시작해"** 발화 → 손 제스처로 로봇 조종 시작
3. 손 위치로 X/Y 이동, 손 크기(원근)로 Z 이동, 주먹/펼침으로 그리퍼 제어
4. **"헬로 로키 검증 시작해"** 발화 → 정렬 검증 실행

### Shadow Mode (손 미러링)

1. 웹에서 **Shadow Mode** 선택
2. 손을 가슴 중앙에 위치시키고 캘리브레이션 시작
3. 손의 절대 위치가 로봇에 실시간 미러링됨

---

## 📡 주요 ROS 2 토픽 / 서비스

| 이름 | 타입 | 설명 |
|------|------|------|
| `/mode_select` | `std_msgs/String` | 모드 전환 명령 (`mode1_tracking`, `mode2_jog`, `mode3_wire_pick`, `stop`) |
| `/mode_status` | `std_msgs/String` | 현재 모드 상태 (1초 주기) |
| `/wire_connected` | `std_msgs/String` | 로봇이 와이어 하나 연결 완료 시 색 이름 발행 |
| `/extracted_wire_keywords` | `std_msgs/String` | LLM이 추출한 색 순서 (`red,green,blue`) |
| `/hand_start` | `std_msgs/String` | Hand 모드 시작 신호 |
| `/hand_verify` | `std_msgs/String` | Hand 모드 검증 신호 |
| `/hand_overlay_data` | `std_msgs/String` | 웹 오버레이용 손 랜드마크 JSON |
| `/arrangement_guide_request` | `std_msgs/String` | 정렬 가이드 AI 코치 요청 |
| `/arrangement_guide_response` | `std_msgs/String` | 가이드 텍스트 + TTS 재생 |
| `/get_3d_position` | `od_msg/SrvDepthPosition` | 픽셀 좌표 → 3D 카메라 좌표 변환 서비스 |
| `/verify_wire_arrangement` | `std_srvs/Trigger` | 와이어 정렬 검증 서비스 |
| `/{ROBOT_ID}/servol_stream` | `dsr_msgs2/ServolStream` | 50Hz 실시간 로봇 위치 스트리밍 |

---

## 🗄️ 데이터 흐름

```
음성 명령
   │
   ▼
Wakeword ("헬로 로키")
   │
   ▼
STT (Whisper) → 텍스트
   │
   ▼
GPT-4o → 색 순서 추출 [red, green, blue]
   │
   ▼
db_logger.start_operation()
Firebase: phase=running, requested_sequence=[...]
   │
   ▼
로봇 실행 → 와이어 순서대로 픽앤플레이스
   │
   ▼
/wire_connected 토픽 발행 (색 완료마다)
   │
   ▼
mission_bridge → Firebase connected_sequence 갱신
   │
   ▼
모든 와이어 완료 감지
   │
   ▼
arrangement_judge → /verify_wire_arrangement 서비스 호출
   │
   ├─ PASS → db_logger.end_operation(success)
   └─ FAIL → db_logger.end_operation(aborted, wrong_arrangement)
```

---

## 🔧 주요 설정값

### 로봇 설정 (`tracking.py`, `jog_tracking.py`)

```python
ROBOT_ID    = "dsr01"
ROBOT_MODEL = "m0609"

# 로봇 기본 위치 (mm)
ROBOT_BASE_X = 367.32
ROBOT_BASE_Y = 3.69
ROBOT_BASE_Z = 422.92

# 그리퍼 (1/10mm 단위)
GRIPPER_OPEN_WIDTH = 500   # 50mm
GRIPPER_FORCE      = 400   # 40N
```

### 카메라 토픽 (`realsense.py`)

```python
COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC  = '/camera/camera/color/camera_info'
```

### YOLO 모델 경로

```
pick_and_place_wire/resource/yolov8n_tools_0122.pt   # 도구 검출
pick_and_place_wire/resource/best_bb.pt               # 와이어 검출 (바운딩박스)
pick_and_place_wire/resource/best.pt                  # 와이어 검출 (세그멘테이션)
```

---

## 👥 팀원 역할 분담

| 역할 | 담당 모듈 |
|------|-----------|
| 로봇 제어 | `wire_pick40.py`, `robot_move.py`, `tracking.py`, `jog_tracking.py`, `onrobot.py` |
| 비전 AI | `detection.py`, `detect_angle.py`, `yolo.py`, `wire_arrangement_checker.py` |
| 음성 AI | `get_wire_keyword.py`, `get_hand_command.py`, `wire_guide_node.py`, `stt.py`, `wakeup_word.py` |
| 웹 / 백엔드 | `app.py`, `db_logger.py`, `mission_bridge.py`, `arrangement_judge.py`, HTML 템플릿 |

---

## 📝 라이선스

본 프로젝트는 교육 목적으로 개발되었습니다.

---

## 🙏 참고

- [Doosan Robotics DSR SDK](https://github.com/doosan-robotics/doosan-robot2)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [OnRobot RG Gripper](https://onrobot.com/en/products/rg2-gripper)