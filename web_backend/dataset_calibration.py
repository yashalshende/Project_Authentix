from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from web_backend.face_alignment import get_face_alignment_service


DATASET_LABELS = ("real", "fake")
PROFILE_VERSION = 3
CALIBRATION_MODE = "labeled_reference"
FEATURE_WEIGHTS = {
    "global_score": 0.28,
    "freq_score": 0.22,
    "blocking_score": 0.18,
    "edge_energy": 0.14,
    "noise_score": 0.10,
    "color_divergence": 0.08,
}


def _project_root(project_root: Optional[Path]) -> Path:
    return Path(project_root or Path(__file__).resolve().parent.parent).resolve()


def _cache_path(project_root: Path) -> Path:
    cache_dir = project_root / "model_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "dataset_reference_profile.json"


def _manifest_paths(project_root: Path) -> List[Path]:
    folders_dir = project_root / "folders"
    return sorted(folders_dir.glob("*/dataset_manifest.json"))


def _manifest_signature(manifest_paths: Iterable[Path]) -> List[Dict[str, object]]:
    signature = []
    for manifest_path in manifest_paths:
        folder_dir = manifest_path.parent
        label_state = {}
        for label in DATASET_LABELS:
            label_dir = folder_dir / label
            if not label_dir.exists():
                continue
            file_count = 0
            latest_mtime = 0
            for file_path in label_dir.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    latest_mtime = max(latest_mtime, int(file_path.stat().st_mtime_ns))
            label_state[label] = {
                "count": file_count,
                "mtime_ns": latest_mtime,
            }
        signature.append(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_mtime_ns": int(manifest_path.stat().st_mtime_ns),
                "labels": label_state,
            }
        )
    return signature


def _load_cache(cache_path: Path) -> Optional[Dict[str, object]]:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception:
        return None


def _save_cache(cache_path: Path, payload: Dict[str, object]) -> None:
    try:
        with cache_path.open("w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, indent=2)
    except Exception:
        return


def _sample_evenly(file_paths: List[Path], limit: int) -> List[Path]:
    if len(file_paths) <= limit:
        return file_paths
    positions = np.linspace(0, len(file_paths) - 1, limit, dtype=int)
    return [file_paths[int(index)] for index in positions]


def _iter_image_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )


def _split_feature_samples(sampled_features: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    if len(sampled_features) <= 3:
        return sampled_features, sampled_features

    holdout = max(1, int(round(len(sampled_features) * 0.25)))
    calibration = sampled_features[:-holdout] or sampled_features
    validation = sampled_features[-holdout:] or sampled_features[:1]
    return calibration, validation


def _normalize(value: float, low: float, high: float) -> float:
    clipped = min(max(float(value), low), high)
    if high - low == 0:
        return 0.0
    return float((clipped - low) / (high - low))


def _choose_focus_region(bgr_image: np.ndarray) -> np.ndarray:
    alignment = get_face_alignment_service().align_primary_face(bgr_image)
    if alignment.aligned_face is not None and alignment.aligned_face.size > 0:
        return alignment.aligned_face

    h, w = bgr_image.shape[:2]
    side = min(h, w)
    x0 = max(0, (w - side) // 2)
    y0 = max(0, (h - side) // 2)
    return bgr_image[y0 : y0 + side, x0 : x0 + side]


def _artifact_metrics(bgr_img: np.ndarray) -> Dict[str, float]:
    bgr = bgr_img.astype(np.float32) / 255.0
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt((sobel_x ** 2) + (sobel_y ** 2))))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = float(np.std(gray - blurred))

    diff_x = np.abs(np.diff(gray, axis=1))
    diff_y = np.abs(np.diff(gray, axis=0))
    block_x = float(np.mean(diff_x[:, 7::8])) if diff_x.shape[1] > 8 else 0.0
    block_y = float(np.mean(diff_y[7::8, :])) if diff_y.shape[0] > 8 else 0.0
    blocking_score = (block_x + block_y) / 2.0

    b_chan, g_chan, r_chan = cv2.split(bgr)
    color_divergence = float(
        (np.mean(np.abs(r_chan - g_chan)) + np.mean(np.abs(g_chan - b_chan)) + np.mean(np.abs(r_chan - b_chan))) / 3.0
    )

    return {
        "edge_energy": edge_energy,
        "noise_score": noise_score,
        "blocking_score": blocking_score,
        "color_divergence": color_divergence,
    }


def _score_from_metrics(metrics: Dict[str, float]) -> float:
    edge_component = 1.0 - _normalize(metrics["edge_energy"], 0.03, 0.22)
    noise_component = _normalize(metrics["noise_score"], 0.02, 0.20)
    block_component = _normalize(metrics["blocking_score"], 0.01, 0.22)
    color_component = _normalize(metrics["color_divergence"], 0.02, 0.25)
    score = (
        (0.36 * edge_component)
        + (0.22 * noise_component)
        + (0.27 * block_component)
        + (0.15 * color_component)
    )
    return float(max(0.0, min(1.0, score))) * 100.0


def _frequency_score(bgr_img: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (256, 256))
        height, width = gray.shape
        window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
        fft_shifted = np.fft.fftshift(np.fft.fft2(gray * window))
        magnitude = np.log1p(np.abs(fft_shifted))
        cy, cx = height // 2, width // 2
        magnitude[cy - 8 : cy + 8, cx - 8 : cx + 8] = 0
        non_zero = magnitude[magnitude > 0]
        if non_zero.size == 0:
            return 0.0
        mean_mag = float(np.mean(non_zero))
        mid_band = magnitude[cy // 2 : cy + cy // 2, cx // 2 : cx + cx // 2]
        peak_ratio = float(np.percentile(mid_band, 99)) / (mean_mag + 1e-6)
        high_band = np.concatenate([magnitude[: cy // 3, :], magnitude[cy + cy // 3 :, :]], axis=0)
        high_ratio = float(np.mean(high_band)) / (mean_mag + 1e-6)
        score = min(
            100.0,
            max(
                0.0,
                (_normalize(peak_ratio, 1.5, 6.0) * 60.0)
                + (_normalize(high_ratio, 0.8, 2.5) * 40.0),
            ),
        )
        return round(float(score), 2)
    except Exception:
        return 0.0


def _compute_score_from_stats(observed_features: Dict[str, float], feature_stats: Dict[str, object]) -> Tuple[Optional[float], Optional[float], List[Dict[str, object]]]:
    real_feature_stats = feature_stats.get("real", {})
    fake_feature_stats = feature_stats.get("fake", {})
    if not real_feature_stats or not fake_feature_stats:
        return None, None, []

    contributions = []
    total_weight = 0.0
    weighted_score = 0.0

    for feature_name, base_weight in FEATURE_WEIGHTS.items():
        if feature_name not in observed_features:
            continue
        real_stats = real_feature_stats.get(feature_name)
        fake_stats = fake_feature_stats.get(feature_name)
        if not real_stats or not fake_stats:
            continue

        real_mean = float(real_stats["mean"])
        fake_mean = float(fake_stats["mean"])
        real_std = max(float(real_stats["std"]), 1e-6)
        fake_std = max(float(fake_stats["std"]), 1e-6)
        gap = fake_mean - real_mean
        if abs(gap) < 1e-6:
            continue

        observed_value = float(observed_features[feature_name])
        directional = (observed_value - real_mean) / gap if gap > 0 else (real_mean - observed_value) / abs(gap)
        directional = float(max(0.0, min(1.0, directional)))

        real_distance = abs(observed_value - real_mean)
        fake_distance = abs(observed_value - fake_mean)
        closeness = 1.0 - (fake_distance / (fake_distance + real_distance + 1e-6))
        closeness = float(max(0.0, min(1.0, closeness)))

        separability = min(1.35, abs(gap) / (real_std + fake_std + 1e-6))
        effective_weight = base_weight * max(0.35, separability)
        feature_score = (directional * 0.62) + (closeness * 0.38)

        weighted_score += feature_score * effective_weight
        total_weight += effective_weight
        contributions.append(
            {
                "feature": feature_name,
                "score": round(feature_score * 100.0, 2),
                "observed": round(observed_value, 6),
                "real_mean": round(real_mean, 6),
                "fake_mean": round(fake_mean, 6),
                "weight": round(effective_weight, 4),
            }
        )

    if total_weight <= 0:
        return None, None, []

    contributions.sort(key=lambda item: item["score"] * item["weight"], reverse=True)
    final_score = (weighted_score / total_weight) * 100.0
    return final_score, total_weight, contributions


def _tune_thresholds(feature_stats: Dict[str, object], validation_rows: List[Dict[str, object]]) -> Tuple[Dict[str, float], Dict[str, object]]:
    scored_rows = []
    for row in validation_rows:
        score, _, _ = _compute_score_from_stats(row["features"], feature_stats)
        if score is None:
            continue
        scored_rows.append(
            {
                "score": float(score),
                "label": str(row["label"]).lower(),
            }
        )

    default_thresholds = {
        "image": 50.0,
        "video": 45.0,
    }
    if not scored_rows or len({row["label"] for row in scored_rows}) < 2:
        return default_thresholds, {
            "available": False,
            "validation_samples": len(scored_rows),
            "selected_threshold": default_thresholds["image"],
        }

    best = None
    for threshold in np.linspace(34.0, 68.0, 69):
        tp = fp = tn = fn = 0
        for row in scored_rows:
            predicted_fake = row["score"] >= threshold
            is_fake = row["label"] == "fake"
            if predicted_fake and is_fake:
                tp += 1
            elif predicted_fake and not is_fake:
                fp += 1
            elif not predicted_fake and is_fake:
                fn += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        objective = (recall * 0.45) + (accuracy * 0.25) + (precision * 0.15) + (f1 * 0.15)
        candidate = {
            "threshold": round(float(threshold), 2),
            "accuracy": round(float(accuracy), 4),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4),
            "objective": round(float(objective), 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if best is None or candidate["objective"] > best["objective"] or (
            candidate["objective"] == best["objective"] and candidate["recall"] > best["recall"]
        ):
            best = candidate

    image_threshold = float(best["threshold"]) if best else default_thresholds["image"]
    video_threshold = max(36.0, round(image_threshold - 4.0, 2))
    return {
        "image": round(image_threshold, 2),
        "video": round(video_threshold, 2),
    }, {
        "available": best is not None,
        "validation_samples": len(scored_rows),
        "selected_threshold": round(image_threshold, 2),
        "video_threshold": round(video_threshold, 2),
        "accuracy": best["accuracy"] if best else None,
        "recall": best["recall"] if best else None,
        "precision": best["precision"] if best else None,
        "f1": best["f1"] if best else None,
    }


def _extract_features(image_path: Path) -> Optional[Dict[str, float]]:
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        return None

    focus_region = _choose_focus_region(bgr_image)
    metrics = _artifact_metrics(focus_region)
    return {
        **metrics,
        "global_score": round(_score_from_metrics(metrics), 4),
        "freq_score": round(_frequency_score(focus_region), 4),
    }


def _compute_profile(project_root: Path, max_files_per_label: int) -> Dict[str, object]:
    manifest_paths = _manifest_paths(project_root)
    datasets = []
    feature_bank = {label: {key: [] for key in FEATURE_WEIGHTS} for label in DATASET_LABELS}
    validation_rows = []

    for manifest_path in manifest_paths:
        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except Exception:
            continue

        folder_dir = manifest_path.parent
        label_dirs = {label: folder_dir / label for label in DATASET_LABELS}
        if not all(label_dir.exists() for label_dir in label_dirs.values()):
            continue

        dataset_entry = {
            "name": manifest.get("name") or folder_dir.name,
            "slug": manifest.get("slug") or folder_dir.name.lower().replace(" ", "-"),
            "labels": {},
        }

        for label in DATASET_LABELS:
            label_dir = label_dirs[label]
            all_files = _iter_image_files(label_dir)
            sampled_files = _sample_evenly(all_files, max_files_per_label)
            sampled_features = []
            for image_path in sampled_files:
                feature_map = _extract_features(image_path)
                if not feature_map:
                    continue
                sampled_features.append(feature_map)
            calibration_features, validation_features = _split_feature_samples(sampled_features)
            for feature_map in calibration_features:
                for key in FEATURE_WEIGHTS:
                    feature_bank[label][key].append(float(feature_map[key]))
            for feature_map in validation_features:
                validation_rows.append({"label": label, "features": feature_map})

            dataset_entry["labels"][label] = {
                "sample_count": len(calibration_features),
                "validation_count": len(validation_features),
                "file_count": len(all_files),
            }

        if dataset_entry["labels"]:
            datasets.append(dataset_entry)

    feature_stats = {}
    for label in DATASET_LABELS:
        feature_stats[label] = {}
        for key in FEATURE_WEIGHTS:
            values = feature_bank[label][key]
            if not values:
                continue
            feature_stats[label][key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "count": len(values),
            }

    calibrated_thresholds, validation_summary = _tune_thresholds(feature_stats, validation_rows)

    return {
        "available": bool(datasets and feature_stats.get("real") and feature_stats.get("fake")),
        "datasets": datasets,
        "reference_datasets": [dataset["name"] for dataset in datasets],
        "calibration_mode": CALIBRATION_MODE,
        "max_files_per_label": int(max_files_per_label),
        "feature_stats": feature_stats,
        "calibrated_thresholds": calibrated_thresholds,
        "validation_summary": validation_summary,
    }


def build_dataset_reference_profile(project_root: Optional[Path] = None, max_files_per_label: int = 18) -> Dict[str, object]:
    root = _project_root(project_root)
    cache_path = _cache_path(root)
    manifest_paths = _manifest_paths(root)
    signature = _manifest_signature(manifest_paths)
    cached = _load_cache(cache_path)

    if (
        cached
        and cached.get("version") == PROFILE_VERSION
        and cached.get("signature") == signature
        and cached.get("profile", {}).get("max_files_per_label") == int(max_files_per_label)
    ):
        return cached["profile"]

    profile = _compute_profile(root, max_files_per_label=max_files_per_label)
    payload = {
        "version": PROFILE_VERSION,
        "signature": signature,
        "profile": profile,
    }
    _save_cache(cache_path, payload)
    return profile


def score_observation(observed_features: Dict[str, float], profile: Dict[str, object]) -> Tuple[Optional[float], Dict[str, object]]:
    feature_stats = profile.get("feature_stats", {}) if isinstance(profile, dict) else {}
    if not profile or not profile.get("available") or not feature_stats.get("real") or not feature_stats.get("fake"):
        return None, {"available": False, "supporting_features": [], "profile_datasets": 0, "reference_datasets": []}

    final_score, total_weight, contributions = _compute_score_from_stats(observed_features, feature_stats)
    if final_score is None or total_weight is None:
        return None, {"available": False, "supporting_features": [], "profile_datasets": len(profile.get("datasets", []))}

    final_score = round(final_score, 2)
    diagnostics = {
        "available": True,
        "profile_datasets": len(profile.get("datasets", [])),
        "reference_datasets": profile.get("reference_datasets", []),
        "supporting_features": contributions[:3],
        "reference_sample_size": sum(
            int(dataset.get("labels", {}).get("real", {}).get("sample_count", 0))
            + int(dataset.get("labels", {}).get("fake", {}).get("sample_count", 0))
            for dataset in profile.get("datasets", [])
        ),
        "calibration_mode": profile.get("calibration_mode", CALIBRATION_MODE),
        "calibrated_thresholds": profile.get("calibrated_thresholds", {}),
        "validation_summary": profile.get("validation_summary", {}),
    }
    return final_score, diagnostics
