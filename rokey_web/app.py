import subprocess
from flask import Flask, render_template, jsonify
import db_logger

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logs')
def logs():
    """로그 페이지를 보여준다."""
    return render_template('logs.html')

@app.route('/operation')
def operation():
    """임무 수행 페이지를 보여준다."""
    return render_template('operation.html')

@app.route('/hand_operation')
def hand_operation():
    """핸드 모드 임무 수행 페이지."""
    return render_template('hand_operation.html')

@app.route('/shadow_operation')
def shadow_operation():
    return render_template('shadow_operation.html')

@app.route('/api/operations')
def api_operations():
    """모든 임무 기록을 JSON으로 반환."""
    return jsonify(db_logger.get_all_operations())

@app.route('/api/abort_operation', methods=['POST'])
def api_abort_operation():
    """현재 진행 중인 임무를 강제 중단한다."""
    try:
        state = db_logger.get_state() or {}
        ops = db_logger.get_all_operations()

        # 진행 중 임무가 있으면 SQLite에 aborted로 마무리
        if ops and ops[0].get('result') == 'running':
            op_id = ops[0]['operation_id']
            connected = state.get('connected_sequence', []) or []
            db_logger.end_operation(
                op_id, result='aborted',
                reason='user_stopped',
                connected_sequence=connected,
            )

        # Firebase 상태를 idle로 강제 리셋 (각 필드 명시적으로)
        db_logger.update_state(
            phase='idle',
            requested_sequence=[],
            connected_sequence=[],
            current_index=0,
            current_target='',
            message='임무 중단됨',
            voice_status='',
            mic_active=False,
            hand_active=False,
            shadow_active=False,
            last_voice_text='',
        )
        return jsonify({'ok': True})
    except Exception as e:
        print(f"abort error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/stop_mode', methods=['POST'])
def api_stop_mode():
    """로봇 PC의 mode_manager에 stop 신호 발행 (현재 모드 노드 중단)."""
    try:
        subprocess.Popen([
            'ros2', 'topic', 'pub', '--once',
            '/mode_select', 'std_msgs/String',
            'data: stop'
        ])
        return jsonify({'ok': True})
    except Exception as e:
        print(f"stop_mode error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)