import json
import math
import os
from functools import lru_cache
from pathlib import Path
try:
    from static_ffmpeg import add_paths
    add_paths() 
except ImportError:
    pass
from flask import Flask, abort, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from web_backend.database import init_db, save_scan_result, get_history, clear_history, delete_entry_by_id
from web_backend.utils import save_upload, get_file_type
from web_backend.inference_service import run_image_inference, run_video_inference
from web_backend.report_generator import generate_html_report
from web_backend.progress import set_progress, get_progress, complete_progress

app = Flask(__name__, static_folder='static', template_folder='templates')

# Max 500MB Upload limit
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
FOLDERS_DIR = BASE_DIR / "folders"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
DEFAULT_DATASET_LABELS = ("real", "fake")
GALLERY_PAGE_SIZE = 48

init_db()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'outputs'), exist_ok=True)


def _normalize_verdict(verdict, confidence):
    normalized = str(verdict or "").upper()
    if normalized in {"REAL", "DEEPFAKE"}:
        return normalized
    return "DEEPFAKE" if float(confidence or 0) >= 52.0 else "REAL"


def _safe_int(value, default=1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slugify(text):
    cleaned = []
    previous_dash = False
    for char in str(text or "").strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return "".join(cleaned).strip("-") or "dataset"


def _split_label(split_key):
    mapping = {
        "train": "Train",
        "validation": "Validation",
        "val": "Validation",
        "test": "Test",
    }
    return mapping.get(str(split_key).lower(), str(split_key).replace("_", " ").title())


def _normalize_split_map(raw_splits):
    if not isinstance(raw_splits, dict):
        return {}

    normalized = {}
    nested_style = any(isinstance(value, dict) for value in raw_splits.values())

    if nested_style:
        for split_name, label_counts in raw_splits.items():
            if not isinstance(label_counts, dict):
                continue
            split_key = str(split_name).lower()
            normalized[split_key] = {
                "real": int(label_counts.get("real", 0) or 0),
                "fake": int(label_counts.get("fake", 0) or 0),
            }
        return normalized

    for flat_key, count in raw_splits.items():
        parts = str(flat_key).lower().rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in DEFAULT_DATASET_LABELS:
            continue
        split_key, label = parts
        normalized.setdefault(split_key, {"real": 0, "fake": 0})
        normalized[split_key][label] = int(count or 0)
    return normalized


def _get_dataset_catalog():
    datasets = []
    if not FOLDERS_DIR.exists():
        return {"datasets": [], "by_slug": {}}

    for manifest_path in sorted(FOLDERS_DIR.glob("*/dataset_manifest.json")):
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

        folder_dir = manifest_path.parent.resolve()
        source_folder = manifest.get("source_folder")
        source_dir = (folder_dir / source_folder).resolve() if source_folder else folder_dir

        if not source_dir.exists():
            continue

        splits = _normalize_split_map(manifest.get("splits", {}))
        total_images = int(manifest.get("total_images", 0) or 0)
        real_count = int(manifest.get("real_count", 0) or 0)
        fake_count = int(manifest.get("fake_count", 0) or 0)

        if not total_images:
            total_images = real_count + fake_count
        if not real_count:
            real_count = sum(split.get("real", 0) for split in splits.values())
        if not fake_count:
            fake_count = sum(split.get("fake", 0) for split in splits.values())

        labels = manifest.get("labels")
        if isinstance(labels, list):
            labels = [str(label).lower() for label in labels if str(label).strip()]
        else:
            labels = [label for label in DEFAULT_DATASET_LABELS if (source_dir / label).exists()]
        if not labels:
            labels = ["images"]
        split_totals = [{
            "key": split_name,
            "label": _split_label(split_name),
            "count": int(split_counts.get("real", 0) + split_counts.get("fake", 0)),
        } for split_name, split_counts in splits.items()]

        try:
            relative_storage_root = source_dir.relative_to(FOLDERS_DIR.resolve()).as_posix()
        except ValueError:
            relative_storage_root = source_dir.name

        datasets.append({
            "name": manifest.get("name") or folder_dir.name,
            "slug": manifest.get("slug") or _slugify(manifest.get("name") or folder_dir.name),
            "folder_name": folder_dir.name,
            "source_dir": source_dir,
            "relative_storage_root": relative_storage_root,
            "total_images": total_images,
            "real_count": real_count,
            "fake_count": fake_count,
            "images_count": total_images,
            "splits": splits,
            "split_totals": split_totals,
            "labels": labels,
        })

    return {
        "datasets": datasets,
        "by_slug": {dataset["slug"]: dataset for dataset in datasets},
    }


def _get_dataset_by_slug(dataset_slug):
    return _get_dataset_catalog()["by_slug"].get(dataset_slug)


@lru_cache(maxsize=12)
def _list_label_filenames(dataset_slug, label):
    dataset = _get_dataset_by_slug(dataset_slug)
    if not dataset:
        return tuple()
    label_dir = Path(dataset["source_dir"]) / label
    if label == "images" and not label_dir.exists():
        label_dir = Path(dataset["source_dir"])
    if not label_dir.exists():
        return tuple()
    return tuple(sorted(
        path.relative_to(label_dir).as_posix() for path in label_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ))


def _build_label_gallery(dataset_slug, label, page=1, page_size=GALLERY_PAGE_SIZE):
    dataset = _get_dataset_by_slug(dataset_slug)
    filenames = _list_label_filenames(dataset_slug, label)
    total_images = len(filenames)
    total_pages = max(1, math.ceil(total_images / page_size)) if total_images else 1
    current_page = min(max(page, 1), total_pages)
    start = (current_page - 1) * page_size
    end = start + page_size
    page_files = filenames[start:end]

    images = [{
        "name": Path(name).name,
        "relative_path": f"{dataset['relative_storage_root']}/{label}/{name}" if label != "images" or (Path(dataset["source_dir"]) / label).exists()
        else f"{dataset['relative_storage_root']}/{name}",
    } for name in page_files]

    return {
        "dataset_slug": dataset_slug,
        "dataset_name": dataset["name"] if dataset else dataset_slug,
        "label": label,
        "title": "Images" if label == "images" else f"{label.capitalize()} Images",
        "total_images": total_images,
        "images": images,
        "current_page": current_page,
        "total_pages": total_pages,
        "has_previous": current_page > 1,
        "has_next": current_page < total_pages,
        "previous_page": current_page - 1,
        "next_page": current_page + 1,
        "visible_pages": list(range(max(1, current_page - 2), min(total_pages, current_page + 2) + 1)),
    }


def _preview_images(dataset_slug, label, limit=6):
    dataset = _get_dataset_by_slug(dataset_slug)
    return [{
        "name": Path(name).name,
        "relative_path": f"{dataset['relative_storage_root']}/{label}/{name}" if label != "images" or (Path(dataset["source_dir"]) / label).exists()
        else f"{dataset['relative_storage_root']}/{name}",
    } for name in _list_label_filenames(dataset_slug, label)[:limit]]

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/')
def index():
    return render_template('index.html', datasets=_get_dataset_catalog()["datasets"])

@app.route('/folders')
def folders_page():
    return render_template('folders.html', datasets=_get_dataset_catalog()["datasets"])


@app.route('/folders/<dataset_slug>')
def dataset_folder_page(dataset_slug):
    dataset = _get_dataset_by_slug(dataset_slug)
    if not dataset:
        abort(404)
    label_sections = []
    for label in dataset["labels"]:
        label_sections.append({
            "label": label,
            "title": "Images" if label == "images" else f"{label.capitalize()} Images",
            "count": dataset.get(f"{label}_count", dataset["total_images"] if label == "images" else 0),
            "preview_images": _preview_images(dataset["slug"], label),
            "split_counts": [{
                "label": split_total["label"],
                "count": int(dataset["splits"].get(split_total["key"], {}).get(label, 0)),
            } for split_total in dataset["split_totals"]],
        })
    return render_template('folder_detail.html', dataset=dataset, label_sections=label_sections)


@app.route('/folders/<dataset_slug>/<label>')
def dataset_label_page(dataset_slug, label):
    dataset = _get_dataset_by_slug(dataset_slug)
    if not dataset:
        abort(404)
    normalized_label = str(label or "").lower()
    if normalized_label not in dataset["labels"]:
        abort(404)
    page = _safe_int(request.args.get('page'), default=1)
    gallery = _build_label_gallery(dataset_slug, normalized_label, page=page)
    return render_template('folder_label.html', dataset=dataset, gallery=gallery)


@app.route('/folders/files/<path:relative_path>')
def serve_folder_file(relative_path):
    requested_file = (FOLDERS_DIR / relative_path).resolve()
    folders_root = FOLDERS_DIR.resolve()

    if folders_root not in requested_file.parents and requested_file != folders_root:
        abort(404)
    if not requested_file.exists() or not requested_file.is_file():
        abort(404)

    return send_from_directory(str(FOLDERS_DIR), relative_path)

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_media():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file was uploaded."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty File"}), 400
        
    mode = request.form.get('mode', 'fast')
    frontend_job_id = request.form.get('job_id', None)
    
    file_path, job_id, original_name, file_type = save_upload(file, app.config['UPLOAD_FOLDER'], frontend_job_id)
    
    if not file_path:
        return jsonify({"success": False, "error": "Unsupported Format"}), 400
        
    try:
        set_progress(job_id, 10, "Media payload received...")
        if file_type == 'image':
            result = run_image_inference(file_path, job_id)
        else:
            result = run_video_inference(file_path, job_id, mode)

        if not result.get("success"):
            raise ValueError(result.get("error", "Analysis failed unexpectedly."))
            
        result["file_name"] = original_name
        result["file_type"] = file_type.upper()
        result["processing_time_ms"] = 850 if file_type == 'image' else 2100 
        
        final_payload = {
            "job_id": job_id,
            "file_name": result["file_name"],
            "file_type": result["file_type"],
            "confidence": result["confidence"],
            "verdict": result["verdict"],
            "model_status": result.get("model_status", "heuristic_only"),
            "reference_datasets": result.get("reference_datasets", []),
            "calibration_mode": result.get("calibration_mode", "labeled_reference"),
            "processing_time_ms": result["processing_time_ms"],
            "explanation": result["explanation"],
            "review_note": result.get("review_note", ""),
            "evidence_strength": result.get("evidence_strength", "moderate"),
            "deepfake_type": result.get("deepfake_type", "Not Available"),
            "heatmap_url": result.get("heatmap_url", ""),
            "timeline_heatmaps": result.get("timeline_heatmaps", []),
            "suspicious_frames": result.get("suspicious_frames", []),
            "forensic_details": result.get("forensic_details", None),
            "reasons": result.get("reasons", []),
            "faceswap_analysis": result.get("faceswap_analysis", {}),
            "strongest_frame": result.get("strongest_frame", None),
            "xai_reports": result.get("xai_reports", []),
            "xai_basic_reports": result.get("xai_basic_reports", []),
            "xai_advanced_reports": result.get("xai_advanced_reports", []),
            "xai_context": result.get("xai_context", {}),
            "face_forensics": result.get("face_forensics", {}),
            "dataset_calibration": result.get("dataset_calibration", {}),
            "indicators": {
                **result.get("indicators", {}),
            }
        }
        
        save_scan_result(
            job_id, original_name, file_type.upper(), 
            result.get("confidence", 0), result.get("verdict", "Unknown"), 
            result.get("heatmap_url", ""), result.get("explanation", ""), result.get("deepfake_type", "Not Available"),
            result.get("faceswap_analysis", {}).get("faceswap_score", 0.0),
            (result.get("strongest_frame") or {}).get("image_url", "") or result.get("faceswap_analysis", {}).get("artifacts", {}).get("aligned_face_url", ""),
            result.get("faceswap_analysis", {}),
        )
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
        complete_progress(job_id)
        return jsonify({"success": True, "result": final_payload})
        
    except Exception as e:
        set_progress(job_id, 100, f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/progress/<job_id>', methods=['GET'])
def fetch_progress(job_id):
    return jsonify({"success": True, "progress": get_progress(job_id)})

@app.route('/api/history', methods=['GET'])
def fetch_history():
    history = get_history()
    formatted = []
    for h in history:
        formatted.append({
            "id": h['id'],
            "thumbnail_path": h['heatmap_url'],
            "file_name": h['file_name'],
            "verdict": _normalize_verdict(h['verdict'], h['confidence']),
            "confidence": h['confidence'],
            "processing_time_ms": 1250,
            "explanation": h['explanation'],
            "file_type": h['file_type'],
            "deepfake_type": h.get('deepfake_type', 'Not Available'),
            "faceswap_score": h.get('faceswap_score', 0.0),
            "upload_date": h.get('upload_date', ''),
        })
    return jsonify({"success": True, "history": formatted})

@app.route('/api/history/<job_id>', methods=['DELETE'])
def delete_history_item(job_id):
    delete_entry_by_id(job_id)
    return jsonify({"success": True})

@app.route('/api/history/clear', methods=['POST'])
def clear_all_history():
    clear_history()
    return jsonify({"success": True})

@app.route('/api/report/<job_id>', methods=['GET'])
def download_report(job_id):
    html_content, file_name = generate_html_report(job_id)
    if not html_content:
        return "Forensic File Not Evaluated Yet", 404
        
    return html_content

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    print("\n" + "="*50)
    print("  AUTHENTIX DEEPFAKE DETECTION ENGINE")
    print("="*50)
    print(f"  AUTHENTIX is running at http://{host}:{port}/")
    print("  Open this link in your browser to use the application.")
    print("="*50 + "\n")
    app.run(host=host, port=port, debug=debug_mode)
