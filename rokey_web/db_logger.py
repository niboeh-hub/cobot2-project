"""DB 기록 모듈.

팀원들은 이 파일의 함수만 호출하면 된다.
DB(SQLite/Firebase)를 직접 다룰 필요 없음.
"""
import sqlite3
import os
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db as fb_db

DB_PATH = os.path.join(os.path.dirname(__file__), 'db', 'rokey.db')

# --- Firebase 설정 ---
FIREBASE_KEY_PATH = os.path.join(
    os.path.dirname(__file__), 'db',
    'rokey-d-2-1f6ab-firebase-adminsdk-fbsvc-17b02ff1a6.json')
FIREBASE_DB_URL = (
    'https://rokey-d-2-1f6ab-default-rtdb.asia-southeast1.firebasedatabase.app')

# Firebase 앱 초기화 (모듈이 처음 import될 때 한 번만)
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})


def _connect():
    """SQLite 연결을 연다. (내부용)"""
    return sqlite3.connect(DB_PATH)


# ===================================================================
# SQLite: 임무 기록
# ===================================================================
def start_operation(requested_sequence):
    """새 임무를 시작한다. operation_id를 반환.

    requested_sequence: 사람이 요청한 색 순서 리스트, 예: ['red','green','blue'].
    첫 음성 명령("~색 연결해")을 받은 시점에 호출.
    """
    conn = _connect()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    req_str = ','.join(requested_sequence)
    cur.execute(
        "INSERT INTO operation_log "
        "(start_time, result, requested_sequence) VALUES (?, ?, ?)",
        (now, 'running', req_str)
    )
    operation_id = cur.lastrowid
    conn.commit()
    conn.close()
    print(f'[db] 임무 시작: operation_id={operation_id}, '
          f'요청={requested_sequence}')
    return operation_id


def end_operation(operation_id, result, connected_sequence=None, reason=''):
    """임무를 종료하고 결과를 기록한다.

    result             : 'success' 또는 'aborted'
    connected_sequence : 실제로 연결한 색 리스트, 예: ['red','green']
    reason             : 중단 사유 (성공이면 빈 문자열)
    """
    if connected_sequence is None:
        connected_sequence = []

    conn = _connect()
    cur = conn.cursor()

    # 시작 시각을 읽어 소요 시간 계산
    cur.execute(
        "SELECT start_time FROM operation_log WHERE operation_id = ?",
        (operation_id,)
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        print(f'[db] 경고: operation_id {operation_id} 없음.')
        return

    start = datetime.fromisoformat(row[0])
    now = datetime.now()
    elapsed = (now - start).total_seconds()

    conn_str = ','.join(connected_sequence)
    cur.execute(
        "UPDATE operation_log "
        "SET end_time = ?, result = ?, reason = ?, "
        "    connected_sequence = ?, elapsed_sec = ? "
        "WHERE operation_id = ?",
        (now.isoformat(timespec='seconds'), result, reason,
         conn_str, elapsed, operation_id)
    )
    conn.commit()
    conn.close()
    print(f'[db] 임무 종료: operation_id={operation_id}, '
          f'result={result}, elapsed={elapsed:.1f}s')


def get_operation(operation_id):
    """임무 하나의 기록을 딕셔너리로 반환. 없으면 None."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT operation_id, start_time, end_time, result, reason, "
        "       requested_sequence, connected_sequence, elapsed_sec "
        "FROM operation_log WHERE operation_id = ?",
        (operation_id,)
    )
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_all_operations():
    """모든 임무 기록을 최신순으로 반환. (GUI 기록 화면용)"""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT operation_id, start_time, end_time, result, reason, "
        "       requested_sequence, connected_sequence, elapsed_sec "
        "FROM operation_log ORDER BY operation_id DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row):
    """DB 한 행을 딕셔너리로 변환. (내부용)"""
    # 콤마 문자열 -> 리스트 (비었으면 빈 리스트)
    requested = row[5].split(',') if row[5] else []
    connected = row[6].split(',') if row[6] else []
    return {
        'operation_id': row[0],
        'start_time': row[1],
        'end_time': row[2],
        'result': row[3],
        'reason': row[4],
        'requested_sequence': requested,
        'connected_sequence': connected,
        'elapsed_sec': row[7],
    }


# ===================================================================
# Firebase: 실시간 상태
# ===================================================================
def update_state(**kwargs):
    """operation_state의 일부를 갱신한다.

    예: update_state(phase='running', current_target='green')
    넘긴 키만 바뀌고, 나머지는 그대로 유지.
    로봇/비전/음성 노드가 진행에 따라 수시로 호출.
    """
    ref = fb_db.reference('operation_state')
    ref.update(kwargs)
    print(f'[fb] state 갱신: {kwargs}')


def reset_state():
    """operation_state를 임무 시작 기본값으로 초기화한다."""
    ref = fb_db.reference('operation_state')
    ref.set({
        'phase': 'idle',
        'elapsed_sec': 0,
        'requested_sequence': [],
        'connected_sequence': [],
        'current_index': 0,
        'current_target': '',
        'message': '대기 중',
    })
    print('[fb] state 초기화')


def get_state():
    """현재 operation_state 전체를 딕셔너리로 반환."""
    ref = fb_db.reference('operation_state')
    return ref.get()


# ===================================================================
# Firebase: 로그 (쌓임)
# ===================================================================
def log_voice(operation_id, raw_text, stt_result, requested_sequence):
    """음성 명령을 voice_logs에 기록한다.

    raw_text           : 사용자가 한 말 원문
    stt_result         : STT 변환 결과
    requested_sequence : LLM이 뽑은 색 순서 리스트
    """
    ref = fb_db.reference('voice_logs')
    ref.push({
        'operation_id': operation_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'raw_text': raw_text,
        'stt_result': stt_result,
        'requested_sequence': requested_sequence,
    })
    print(f'[fb] voice 로그: {requested_sequence}')


def log_robot_event(operation_id, event_type, detail=None):
    """로봇 동작 이벤트를 robot_events에 기록한다.

    event_type : 'detection' / 'pick' / 'connect' / 'error' 등
    detail     : 이벤트 상세 (이벤트마다 다른 딕셔너리)
    """
    if detail is None:
        detail = {}
    ref = fb_db.reference('robot_events')
    ref.push({
        'operation_id': operation_id,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'event_type': event_type,
        'detail': detail,
    })
    print(f'[fb] robot 이벤트: {event_type}')

    # ===================================================================
# Firebase: 현재 활성 에러 (메인 화면 워닝용)
# ===================================================================
def set_error(message, source=''):
    """현재 활성 에러를 기록한다. 메인 화면 왼쪽 아래 워닝에 뜬다.

    노드가 큰 예외로 종료하기 직전에 호출.
    message : 화면에 띄울 한 줄 메시지
    source  : 어느 노드에서 발생했는지 (예: 'vision_node')
    """
    ref = fb_db.reference('active_error')
    ref.set({
        'message': message,
        'source': source,
        'occurred_at': datetime.now().isoformat(timespec='seconds'),
    })
    print(f'[fb] 활성 에러: {message}')


def clear_error():
    """현재 활성 에러를 지운다. 메인 화면 reset 버튼이 호출."""
    ref = fb_db.reference('active_error')
    ref.delete()
    print('[fb] 활성 에러 해제')