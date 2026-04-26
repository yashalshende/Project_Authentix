import io
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core_engine.config import ModelConfig

ModelConfig.DATASET_REFERENCE_SAMPLE_SIZE = 16

import app as app_module


app_module.app.testing = True
app_module.save_scan_result = lambda *args, **kwargs: None

client = app_module.app.test_client()


def pick_first_in(directory: Path) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    matches = sorted(path for path in directory.iterdir() if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No files found in {directory}")
    return matches[0]


def pick_nth_in(directory: Path, index: int) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    matches = sorted(path for path in directory.iterdir() if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No files found in {directory}")
    return matches[min(index, len(matches) - 1)]


def assert_status(route: str, expected: int = 200) -> None:
    response = client.get(route)
    if response.status_code != expected:
        raise AssertionError(f"{route} returned {response.status_code}, expected {expected}")
    print(f"[route-ok] {route} -> {response.status_code}")


def upload_sample(sample_path: Path, expected_verdict: str | None) -> tuple[bool, dict]:
    with sample_path.open("rb") as file_handle:
        payload = {
            "file": (io.BytesIO(file_handle.read()), sample_path.name),
            "mode": "fast",
            "job_id": f"smoke_{sample_path.stem}",
        }
    response = client.post("/api/analyze", data=payload, content_type="multipart/form-data")
    if response.status_code != 200:
        raise AssertionError(f"/api/analyze failed for {sample_path.name}: {response.status_code} {response.get_data(as_text=True)}")

    body = response.get_json() or {}
    if not body.get("success"):
        raise AssertionError(f"/api/analyze returned failure for {sample_path.name}: {body}")

    result = body.get("result") or {}
    verdict = str(result.get("verdict", "")).upper()
    confidence = float(result.get("confidence", 0.0))
    if str(result.get("model_status", "")) not in {"checkpoint_loaded", "heuristic_only"}:
        raise AssertionError(f"Unexpected model_status for {sample_path.name}: {result.get('model_status')}")
    if str(result.get("calibration_mode", "")) != "labeled_reference":
        raise AssertionError(f"Unexpected calibration_mode for {sample_path.name}: {result.get('calibration_mode')}")
    reference_datasets = result.get("reference_datasets")
    if reference_datasets != ["CelebDF", "DFDC Challenge"]:
        raise AssertionError(f"Unexpected reference_datasets for {sample_path.name}: {reference_datasets}")
    if expected_verdict is None:
        print(f"[analyze-ok] {sample_path.name} -> {verdict} ({confidence:.2f}%)")
        return True, result

    matched = verdict == expected_verdict
    status = "match" if matched else "mismatch"
    print(f"[analyze-{status}] {sample_path.name} -> {verdict} ({confidence:.2f}%), expected {expected_verdict}")
    return matched, result


def main() -> int:
    routes = [
        "/",
        "/folders",
        "/folders/dfdc-challenge",
        "/folders/celebdf",
        "/folders/face-forencics",
        "/folders/face-forencics/images",
        "/api/history",
    ]
    for route in routes:
        assert_status(route)

    samples = [
        (pick_first_in(PROJECT_ROOT / "folders" / "DFDC Challenge" / "real"), "REAL"),
        (pick_first_in(PROJECT_ROOT / "folders" / "DFDC Challenge" / "fake"), "DEEPFAKE"),
        (pick_first_in(PROJECT_ROOT / "folders" / "celebDF" / "real"), "REAL"),
        (pick_first_in(PROJECT_ROOT / "folders" / "celebDF" / "fake"), "DEEPFAKE"),
        (pick_first_in(PROJECT_ROOT / "folders" / "Face Forencics" / "cropped_images" / "000_003"), None),
    ]

    matches = 0
    graded_total = 0
    for sample_path, expected in samples:
        matched, _ = upload_sample(sample_path, expected)
        if expected is not None:
            graded_total += 1
        if expected is not None and matched:
            matches += 1

    accuracy = matches / graded_total if graded_total else 0.0
    print(f"[summary] graded sample verdict agreement: {matches}/{graded_total} ({accuracy * 100:.1f}%)")

    mini_batch = [
        (pick_nth_in(PROJECT_ROOT / "folders" / "DFDC Challenge" / "real", 0), "REAL"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "DFDC Challenge" / "real", 8), "REAL"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "DFDC Challenge" / "fake", 0), "DEEPFAKE"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "DFDC Challenge" / "fake", 8), "DEEPFAKE"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "celebDF" / "real", 0), "REAL"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "celebDF" / "real", 8), "REAL"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "celebDF" / "fake", 0), "DEEPFAKE"),
        (pick_nth_in(PROJECT_ROOT / "folders" / "celebDF" / "fake", 8), "DEEPFAKE"),
    ]
    mini_matches = 0
    fake_total = 0
    fake_hits = 0
    for sample_path, expected in mini_batch:
        matched, result = upload_sample(sample_path, expected)
        if matched:
            mini_matches += 1
        if expected == "DEEPFAKE":
            fake_total += 1
            if str(result.get("verdict", "")).upper() == "DEEPFAKE":
                fake_hits += 1

    mini_accuracy = mini_matches / len(mini_batch)
    fake_recall = fake_hits / fake_total if fake_total else 0.0
    baseline_fake_recall = 0.0
    print(f"[mini-batch] accuracy={mini_matches}/{len(mini_batch)} ({mini_accuracy * 100:.1f}%), fake_recall={fake_recall:.2f}, baseline_fake_recall={baseline_fake_recall:.2f}")

    return 0 if accuracy >= 1.0 and mini_accuracy >= 0.70 and fake_recall > baseline_fake_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
