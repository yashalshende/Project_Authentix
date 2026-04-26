import json
import sqlite3

from flask import render_template

from web_backend.database import DB_PATH, init_db


def generate_html_report(job_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM scans WHERE id = ?", (job_id,)).fetchone()
    conn.close()

    if not row:
        return None, None

    try:
        faceswap_payload = json.loads(row["faceswap_payload"]) if row["faceswap_payload"] else {}
    except json.JSONDecodeError:
        faceswap_payload = {}

    artifacts = faceswap_payload.get("artifacts", {}) if isinstance(faceswap_payload, dict) else {}
    data = {
        "job_id": row["id"],
        "file_name": row["file_name"],
        "file_type": row["file_type"],
        "upload_date": row["upload_date"],
        "confidence": row["confidence"],
        "verdict": row["verdict"],
        "heatmap_url": row["heatmap_url"],
        "explanation": row["explanation"],
        "deepfake_type": row["deepfake_type"],
        "faceswap_score": row["faceswap_score"] or 0.0,
        "strongest_frame_url": row["strongest_frame_url"] or "",
        "faceswap_analysis": faceswap_payload,
        "faceswap_artifacts": artifacts,
    }

    html_content = render_template("report_template.html", report=data)
    return html_content, data["file_name"]
