"""SQLite persistence layer for AUTHENTIX scan history."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "authentix.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                verdict TEXT NOT NULL,
                processing_time_ms REAL NOT NULL,
                explanation TEXT NOT NULL,
                indicators TEXT NOT NULL,
                thumbnail_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_scan(record: Dict[str, object]) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO scans (
                file_name, stored_name, file_type, confidence, verdict,
                processing_time_ms, explanation, indicators, thumbnail_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["file_name"],
                record["stored_name"],
                record["file_type"],
                float(record["confidence"]),
                record["verdict"],
                float(record["processing_time_ms"]),
                record["explanation"],
                json.dumps(record["indicators"]),
                record.get("thumbnail_path"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_all_scans() -> List[Dict[str, object]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()

    scans: List[Dict[str, object]] = []
    for row in rows:
        scan = dict(row)
        scan["indicators"] = json.loads(scan["indicators"])
        scans.append(scan)
    return scans


def delete_scan(scan_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.commit()
        return cur.rowcount > 0


def clear_scans() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM scans")
        conn.commit()
