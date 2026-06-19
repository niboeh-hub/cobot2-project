"""SQLite DB와 테이블을 만드는 스크립트. 처음 한 번만 실행."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'db', 'rokey.db')


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 임무(operation) 기록 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_log (
            operation_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time          TEXT,
            end_time            TEXT,
            result              TEXT,
            reason              TEXT,
            requested_sequence  TEXT,
            connected_sequence  TEXT,
            elapsed_sec         REAL
        )
    """)

    conn.commit()
    conn.close()
    print(f'DB 준비 완료: {DB_PATH}')


if __name__ == '__main__':
    init_db()