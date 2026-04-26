import json
import os
import shutil
import sqlite3
from datetime import datetime


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "authentix_history.db")
LEGACY_DB_PATH = os.path.join(BASE_DIR, "authentix_history.db")


def _has_scans_table(db_path):
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'scans'"
        ).fetchone()
        conn.close()
        return bool(row)
    except sqlite3.Error:
        return False


def _ensure_schema(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            file_name TEXT,
            file_type TEXT,
            upload_date TEXT,
            confidence REAL,
            verdict TEXT,
            heatmap_url TEXT,
            explanation TEXT,
            deepfake_type TEXT,
            faceswap_score REAL,
            strongest_frame_url TEXT,
            faceswap_payload TEXT
        )
        """
    )
    conn.commit()

    existing_columns = {
        row["name"] for row in cursor.execute("PRAGMA table_info(scans)").fetchall()
    }
    required = {
        "faceswap_score": "REAL",
        "strongest_frame_url": "TEXT",
        "faceswap_payload": "TEXT",
    }
    for column, column_type in required.items():
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE scans ADD COLUMN {column} {column_type}")
    conn.commit()
    conn.close()


def _sync_legacy_copy():
    if not os.path.exists(DB_PATH):
        return
    try:
        if os.path.exists(LEGACY_DB_PATH):
            legacy_has_scans = _has_scans_table(LEGACY_DB_PATH)
            data_has_scans = _has_scans_table(DB_PATH)
            if data_has_scans and not legacy_has_scans:
                shutil.copyfile(DB_PATH, LEGACY_DB_PATH)
        else:
            shutil.copyfile(DB_PATH, LEGACY_DB_PATH)
    except OSError:
        # Legacy copy is best-effort only. The app always uses DB_PATH directly.
        pass


def _connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    _ensure_schema(DB_PATH)
    try:
        _ensure_schema(LEGACY_DB_PATH)
    except sqlite3.Error:
        pass
    _sync_legacy_copy()


def save_scan_result(
    job_id,
    file_name,
    file_type,
    confidence,
    verdict,
    heatmap_url,
    explanation,
    df_type="Unknown",
    faceswap_score=0.0,
    strongest_frame_url="",
    faceswap_payload=None,
):
    conn = _connect()
    cursor = conn.cursor()
    payload_text = json.dumps(faceswap_payload or {}, ensure_ascii=True)
    cursor.execute(
        """
        INSERT OR REPLACE INTO scans (
            id, file_name, file_type, upload_date, confidence, verdict,
            heatmap_url, explanation, deepfake_type, faceswap_score,
            strongest_frame_url, faceswap_payload
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            file_name,
            file_type,
            datetime.now().isoformat(),
            confidence,
            verdict,
            heatmap_url,
            explanation,
            df_type,
            faceswap_score,
            strongest_frame_url,
            payload_text,
        ),
    )
    conn.commit()
    conn.close()
    _sync_legacy_copy()


def get_history():
    conn = _connect()
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM scans ORDER BY upload_date DESC").fetchall()
    conn.close()

    history = []
    for row in rows:
        try:
            faceswap_payload = json.loads(row["faceswap_payload"]) if row["faceswap_payload"] else {}
        except json.JSONDecodeError:
            faceswap_payload = {}
        history.append(
            {
                "id": row["id"],
                "file_name": row["file_name"],
                "file_type": row["file_type"],
                "upload_date": row["upload_date"],
                "confidence": row["confidence"],
                "verdict": row["verdict"],
                "heatmap_url": row["heatmap_url"],
                "explanation": row["explanation"],
                "deepfake_type": row["deepfake_type"],
                "faceswap_score": row["faceswap_score"],
                "strongest_frame_url": row["strongest_frame_url"],
                "faceswap_payload": faceswap_payload,
            }
        )
    return history


def clear_history():
    conn = _connect()
    conn.execute("DELETE FROM scans")
    conn.commit()
    conn.close()
    _sync_legacy_copy()


def delete_entry_by_id(job_id):
    conn = _connect()
    conn.execute("DELETE FROM scans WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()
    _sync_legacy_copy()
