import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_engine.config import ModelConfig
from utils.face_regions import FaceRegionAnalyzer
from web_backend.dataset_calibration import build_dataset_reference_profile, score_observation
from web_backend.face_alignment import FaceAlignmentService
from web_backend.faceswap_detector import FaceSwapDetector
from web_backend.progress import set_progress
from web_backend.xai_pipeline import XAIPipelineManager

if not getattr(ModelConfig, "DEMO_MODE_ACTIVE", False):
    import torch
    from facenet_pytorch import MTCNN

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch = None
    MTCNN = None
    DEVICE = "cpu"

print("Establishing AUTHENTIX AI Neural Models locally cleanly efficiently precisely...")
xai_manager = XAIPipelineManager()
face_region_analyzer = FaceRegionAnalyzer()
faceswap_detector = FaceSwapDetector()
face_alignment_service = FaceAlignmentService()
mtcnn = None
mtcnn_attempted = False

if not getattr(ModelConfig, "DEMO_MODE_ACTIVE", False):
    from core_engine.fusion_model import AuthentixHybridModel
    from core_engine.temporal_net import AuthentixTemporalLSTM

    image_model = None
    video_model = None
else:
    print("AUTHENTIX DEMO MODE ACTIVE - Bypassing heavy PyTorch initialization for lab laptop compatibility.")
    image_model = None
    video_model = None

audio_suite = None
audio_suite_attempted = False


def _project_path(*parts):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *parts))


@lru_cache(maxsize=4)
def _get_reference_profile_cached(sample_size):
    return build_dataset_reference_profile(
        project_root=_project_path(),
        max_files_per_label=int(sample_size),
    )


def _get_dataset_reference_profile():
    sample_size = int(getattr(ModelConfig, "DATASET_REFERENCE_SAMPLE_SIZE", 24))
    return _get_reference_profile_cached(sample_size)


def _reference_context(profile):
    return {
        "reference_datasets": list(profile.get("reference_datasets", [])) if isinstance(profile, dict) else [],
        "calibration_mode": str((profile or {}).get("calibration_mode", "labeled_reference")),
    }


def _model_status_for_media(media_type):
    normalized = str(media_type or "image").lower()
    if normalized == "video":
        has_checkpoint = video_model is not None or image_model is not None
    else:
        has_checkpoint = image_model is not None
    return "checkpoint_loaded" if has_checkpoint else "heuristic_only"


def _should_run_audio_scan(mode, video_duration_seconds):
    if getattr(ModelConfig, "DEMO_MODE_ACTIVE", False):
        return False, "demo_mode"
    if not bool(getattr(ModelConfig, "AUDIO_ANALYSIS_ENABLED", True)):
        return False, "disabled"
    if str(mode).lower() == "fast" and not bool(getattr(ModelConfig, "AUDIO_ANALYSIS_IN_FAST_MODE", False)):
        return False, "fast_mode"
    max_duration = float(getattr(ModelConfig, "AUDIO_SKIP_FOR_LONG_VIDEOS_SECONDS", 45))
    if float(video_duration_seconds or 0.0) > max_duration:
        return False, "video_too_long"
    return True, "enabled"


def _load_optional_models():
    global image_model
    global video_model

    if getattr(ModelConfig, "DEMO_MODE_ACTIVE", False) or torch is None:
        return

    try:
        image_ckpt = _project_path(getattr(ModelConfig, "BEST_MODEL_PATH", os.path.join("models", "checkpoints", "authentix_best_model.pth")))
        if os.path.exists(image_ckpt):
            checkpoint = torch.load(image_ckpt, map_location=DEVICE)
            image_model = AuthentixHybridModel().to(DEVICE)
            image_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            image_model.eval()
    except Exception as image_exc:
        print(f"AUTHENTIX image checkpoint unavailable: {image_exc}")
        image_model = None

    try:
        temporal_ckpt = _project_path(getattr(ModelConfig, "BEST_TEMPORAL_MODEL_PATH", os.path.join("models", "checkpoints", "authentix_temporal_best_model.pth")))
        if os.path.exists(temporal_ckpt):
            checkpoint = torch.load(temporal_ckpt, map_location=DEVICE)
            video_model = AuthentixTemporalLSTM().to(DEVICE)
            video_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            video_model.eval()
    except Exception as video_exc:
        print(f"AUTHENTIX temporal checkpoint unavailable: {video_exc}")
        video_model = None


_load_optional_models()


def sanitize_numpy(data):
    if isinstance(data, dict):
        return {k: sanitize_numpy(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_numpy(v) for v in data]
    if isinstance(data, tuple):
        return [sanitize_numpy(v) for v in data]
    if isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    if isinstance(data, (np.float64, np.float32, np.float16)):
        return float(data)
    if isinstance(data, np.ndarray):
        return sanitize_numpy(data.tolist())
    return data


def _to_model_tensor(bgr_image):
    if torch is None:
        return None
    resized = cv2.resize(bgr_image, (ModelConfig.IMG_SIZE, ModelConfig.IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def _to_region_tensor(bgr_image):
    if torch is None:
        return None
    rgb_face = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    region_stack = face_region_analyzer.extract_region_tensor_stack(rgb_face, output_size=ModelConfig.REGION_SIZE).astype(np.float32) / 255.0
    region_stack = (region_stack - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    region_stack = np.transpose(region_stack, (0, 3, 1, 2))
    return torch.from_numpy(region_stack).unsqueeze(0).to(DEVICE)


def _build_aux_feature_vector(face_forensics, faceswap_analysis):
    values = np.asarray(
        [
            float(faceswap_analysis.get("identity_inconsistency_score", 0.0)),
            float(faceswap_analysis.get("boundary_anomaly_score", 0.0)),
            float(faceswap_analysis.get("texture_mismatch_score", 0.0)),
            float(face_forensics.get("landmark_integrity", {}).get("eye_alignment", 0.0)),
            float(face_forensics.get("landmark_integrity", {}).get("mouth_geometry", 0.0)),
            float(face_forensics.get("landmark_integrity", {}).get("face_symmetry", 0.0)),
            float(face_forensics.get("landmark_integrity", {}).get("contour_consistency", 0.0)),
            float(faceswap_analysis.get("region_consensus_score", 0.0)),
            float(face_forensics.get("face_quality", {}).get("quality_score", 0.0)),
            float(face_forensics.get("face_score", 0.0)),
        ],
        dtype=np.float32,
    )
    values /= 100.0
    return values[: int(getattr(ModelConfig, "FACE_SWAP_AUX_DIM", 10))]


def _predict_model_score(aligned_face_bgr, face_forensics, faceswap_analysis):
    if image_model is None or torch is None:
        return None
    try:
        image_tensor = _to_model_tensor(aligned_face_bgr)
        region_tensor = _to_region_tensor(aligned_face_bgr)
        aux_vector = torch.from_numpy(_build_aux_feature_vector(face_forensics, faceswap_analysis)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _, _, faceswap_logits = image_model(image_tensor, region_tensor, aux_vector)
            main_prob = float(torch.sigmoid(logits).squeeze().item() * 100.0)
            branch_prob = float(torch.sigmoid(faceswap_logits).squeeze().item() * 100.0)
        return round((0.72 * main_prob) + (0.28 * branch_prob), 2)
    except Exception as model_exc:
        print(f"AUTHENTIX model inference fallback: {model_exc}")
        return None


def _predict_temporal_model_score(frame_payloads):
    if video_model is None or torch is None or not frame_payloads:
        return None
    try:
        seq_len = min(len(frame_payloads), int(getattr(ModelConfig, "SEQ_LENGTH", 10)))
        selected = frame_payloads[:seq_len]
        if len(selected) < seq_len:
            selected.extend([selected[-1]] * (seq_len - len(selected)))

        image_seq = torch.cat([_to_model_tensor(item["aligned_face"]) for item in selected], dim=0).unsqueeze(0)
        region_seq = torch.cat([_to_region_tensor(item["aligned_face"]) for item in selected], dim=0).unsqueeze(0)
        aux_seq = torch.from_numpy(np.stack([_build_aux_feature_vector(item["face_forensics"], item["faceswap_analysis"]) for item in selected], axis=0)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _, faceswap_logits, _ = video_model(image_seq, region_seq, aux_seq)
            main_prob = float(torch.sigmoid(logits).squeeze().item() * 100.0)
            branch_prob = float(torch.sigmoid(faceswap_logits).squeeze().item() * 100.0)
        return round((0.68 * main_prob) + (0.32 * branch_prob), 2)
    except Exception as temporal_exc:
        print(f"AUTHENTIX temporal model fallback: {temporal_exc}")
        return None


def _get_audio_suite():
    global audio_suite
    global audio_suite_attempted

    if audio_suite_attempted:
        return audio_suite

    audio_suite_attempted = True
    try:
        from web_backend.audio_forensics import AuthentixAudioSuite

        audio_suite = AuthentixAudioSuite(device=DEVICE)
    except Exception as audio_exc:
        print(f"AUTHENTIX audio module disabled: {audio_exc}")
        audio_suite = None

    return audio_suite


def _get_mtcnn():
    global mtcnn
    global mtcnn_attempted

    if getattr(ModelConfig, "DEMO_MODE_ACTIVE", False):
        return None

    if mtcnn_attempted:
        return mtcnn

    mtcnn_attempted = True
    try:
        mtcnn = MTCNN(keep_all=False, device=DEVICE)
    except Exception as detector_exc:
        print(f"AUTHENTIX face detector unavailable: {detector_exc}")
        mtcnn = None

    return mtcnn


def add_forensic_overlays_to_path(image_path, mtcnn_model, conf_percent):
    if not os.path.exists(image_path):
        return

    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        return

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    try:
        boxes, probs, landmarks = mtcnn_model.detect(pil_img, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return

        best_idx = int(np.argmax(probs))
        box = boxes[best_idx]
        pts = landmarks[best_idx]

        is_fake = conf_percent >= 50
        color = (0, 0, 255) if is_fake else (0, 255, 0)
        label = f"ANOMALY {conf_percent}%" if is_fake else f"AUTHENTIC {conf_percent}%"

        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, 1)

        length = min(20, int((x2 - x1) * 0.2))
        thickness = 3
        cv2.line(bgr_img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(bgr_img, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(bgr_img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(bgr_img, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(bgr_img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(bgr_img, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(bgr_img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(bgr_img, (x2, y2), (x2, y2 - length), color, thickness)

        if pts is not None:
            for pt in pts:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(bgr_img, (px, py), 2, (0, 255, 255), -1)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(bgr_img, (x1, max(0, y1 - 25)), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(
            bgr_img,
            label,
            (x1 + 5, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0) if not is_fake else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(image_path, bgr_img)
    except Exception:
        return


def _normalize(value, low, high):
    clipped = min(max(value, low), high)
    if high - low == 0:
        return 0.0
    return float((clipped - low) / (high - low))


def _artifact_metrics(bgr_img):
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

    b, g, r = cv2.split(bgr)
    color_divergence = float(
        (np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0
    )

    return {
        "edge_energy": edge_energy,
        "noise_score": noise_score,
        "blocking_score": blocking_score,
        "color_divergence": color_divergence,
    }


def _frequency_heuristic_score(bgr_img):
    """
    FFT-based periodic grid artifact detector.
    GAN-generated images often have spectrally periodic noise at specific frequencies
    (the 'GAN fingerprint'). This scores 0-100 based on how prominent those artifacts are.
    Works in demo mode without PyTorch.
    """
    try:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Resize to fixed resolution for consistent FFT
        gray = cv2.resize(gray, (256, 256))
        # Apply window to reduce edge effects
        h, w = gray.shape
        window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        windowed = gray * window
        # 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shifted))
        # Center region (DC component) — exclude it
        cy, cx = h // 2, w // 2
        magnitude[cy-8:cy+8, cx-8:cx+8] = 0
        # Look for high-energy spikes in mid-frequency bands (GAN artifacts)
        mid_band = magnitude[cy//2:cy+cy//2, cx//2:cx+cx//2]
        mean_mag = float(np.mean(magnitude[magnitude > 0]))
        if mean_mag < 1e-6:
            return 0.0
        # Spiky peaks relative to mean indicate periodic GAN noise
        peak_ratio = float(np.percentile(mid_band, 99)) / (mean_mag + 1e-6)
        # Also check for high-frequency residuals (blending seams)
        high_band = magnitude[:cy//3, :]
        high_band = np.concatenate([high_band, magnitude[cy+cy//3:, :]], axis=0)
        high_ratio = float(np.mean(high_band)) / (mean_mag + 1e-6)
        # Score: high peak_ratio + high_ratio = suspicious
        score = min(100.0, max(0.0,
            (_normalize(peak_ratio, 1.5, 6.0) * 60.0) +
            (_normalize(high_ratio, 0.8, 2.5) * 40.0)
        ))
        return round(score, 2)
    except Exception:
        return 0.0


def _reenactment_signal_score(face_forensics, faceswap_analysis):
    """
    Detects reenactment/lip-sync fakes where identity is preserved but landmark
    geometry is distorted. These fakes pass identity checks but fail landmark/boundary.
    """
    landmark_mismatch = float(faceswap_analysis.get("landmark_mismatch_score", 0.0))
    boundary_anomaly = float(faceswap_analysis.get("boundary_anomaly_score", 0.0))
    texture_mismatch = float(faceswap_analysis.get("texture_mismatch_score", 0.0))
    # Check mouth geometry specifically (lip-sync fakes distort mouth most)
    mouth_geometry = float(face_forensics.get("landmark_integrity", {}).get("mouth_geometry", 100.0))
    contour = float(face_forensics.get("landmark_integrity", {}).get("contour_consistency", 100.0))
    # Bottom-centre region is mouth — elevated score indicates lip-sync
    bottom_regions = [r for r in face_forensics.get("region_grid_ordered", []) if r.get("key") in {"bottom_centre", "middle_centre"}]
    lip_score = float(np.mean([float(r.get("score", 0.0)) for r in bottom_regions])) if bottom_regions else 0.0
    # Reenactment signal: high landmark mismatch + boundary + lip region
    threshold = float(getattr(ModelConfig, "REENACTMENT_SIGNAL_THRESHOLD", 38.0))
    reenactment_score = (
        landmark_mismatch * 0.30
        + boundary_anomaly * 0.25
        + texture_mismatch * 0.15
        + max(0.0, 100.0 - mouth_geometry) * 0.15   # low mouth_geometry = bad = suspicious
        + max(0.0, 100.0 - contour) * 0.10
        + lip_score * 0.05
    )
    return round(float(min(100.0, max(0.0, reenactment_score))), 2)


def _score_from_metrics(metrics):
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
    return max(0.0, min(1.0, float(score)))


def _decision_threshold(media_type="image"):
    reference_profile = _get_dataset_reference_profile()
    thresholds = reference_profile.get("calibrated_thresholds", {}) if isinstance(reference_profile, dict) else {}
    if str(media_type).lower() == "video":
        return float(
            thresholds.get(
                "video",
                getattr(ModelConfig, "VIDEO_FINAL_DEEPFAKE_THRESHOLD", getattr(ModelConfig, "FINAL_DEEPFAKE_THRESHOLD", 50.0)),
            )
        )
    return float(thresholds.get("image", getattr(ModelConfig, "FINAL_DEEPFAKE_THRESHOLD", 50.0)))


def _verdict_from_confidence(conf_percent, media_type="image"):
    threshold = _decision_threshold(media_type)
    return "DEEPFAKE" if conf_percent >= threshold else "REAL"


def _review_note(conf_percent, media_type="image"):
    threshold = _decision_threshold(media_type)
    low_floor = float(getattr(ModelConfig, "LOW_CONFIDENCE_FLOOR", 40.0))
    high_floor = float(getattr(ModelConfig, "HIGH_CONFIDENCE_FLOOR", 78.0))
    margin = abs(conf_percent - threshold)

    if margin <= 6:
        return "Confidence is moderate, further review suggested."
    if conf_percent < low_floor:
        return "Low confidence result."
    if conf_percent >= high_floor:
        return "High confidence result."
    return ""


def _evidence_strength(conf_percent, media_type="image"):
    threshold = _decision_threshold(media_type)
    low_floor = float(getattr(ModelConfig, "LOW_CONFIDENCE_FLOOR", 40.0))
    high_floor = float(getattr(ModelConfig, "HIGH_CONFIDENCE_FLOOR", 78.0))
    margin = abs(conf_percent - threshold)
    if conf_percent < low_floor:
        return "moderate"
    if conf_percent >= high_floor:
        return "strong"
    if margin <= 6:
        return "moderate"
    if margin <= 16:
        return "good"
    return "strong"


def _faceswap_signal_count(faceswap_analysis):
    minimum = float(getattr(ModelConfig, "FACE_SWAP_SIGNAL_MIN", 55.0))
    signals = [
        float(faceswap_analysis.get("identity_inconsistency_score", 0.0)),
        float(faceswap_analysis.get("boundary_anomaly_score", 0.0)),
        float(faceswap_analysis.get("landmark_mismatch_score", 0.0)),
        float(faceswap_analysis.get("texture_mismatch_score", 0.0)),
    ]
    return sum(1 for signal in signals if signal >= minimum)


def _classify_deepfake_type(verdict, faceswap_analysis, face_forensics=None):
    if verdict != "DEEPFAKE":
        return "Not Available"
    if not faceswap_analysis or not faceswap_analysis.get("available"):
        return "Not Available"

    faceswap_score = float(faceswap_analysis.get("faceswap_score", 0.0))
    identity_inconsistency = float(faceswap_analysis.get("identity_inconsistency_score", 0.0))
    landmark_mismatch = float(faceswap_analysis.get("landmark_mismatch_score", 0.0))
    boundary_anomaly = float(faceswap_analysis.get("boundary_anomaly_score", 0.0))

    # Face swap: high identity inconsistency + faceswap_score
    if (
        faceswap_score >= float(getattr(ModelConfig, "FACE_SWAP_TYPE_THRESHOLD", 58.0))
        and _faceswap_signal_count(faceswap_analysis) >= 2
        and identity_inconsistency >= 40.0
    ):
        return "Face Swap"

    # Reenactment/lip-sync: landmark mismatch + boundary elevated, but identity may be intact
    if face_forensics is not None:
        mouth_geometry = float(face_forensics.get("landmark_integrity", {}).get("mouth_geometry", 100.0))
        contour = float(face_forensics.get("landmark_integrity", {}).get("contour_consistency", 100.0))
        reenactment_signal = (
            landmark_mismatch * 0.40
            + boundary_anomaly * 0.30
            + max(0.0, 100.0 - mouth_geometry) * 0.20
            + max(0.0, 100.0 - contour) * 0.10
        )
        if reenactment_signal >= 35.0:
            return "Reenactment / Lip-Sync"

    if faceswap_score >= 40.0 and _faceswap_signal_count(faceswap_analysis) >= 1:
        return "Face Swap"

    return "Other Manipulation"


def _blend_faceswap_consensus(base_confidence, faceswap_analysis):
    if not faceswap_analysis or not faceswap_analysis.get("available"):
        return round(float(base_confidence), 2)

    faceswap_score = float(faceswap_analysis.get("faceswap_score", 0.0))
    combined = (
        float(base_confidence) * float(getattr(ModelConfig, "HEURISTIC_BLEND_WEIGHT", 0.50))
        + faceswap_score * float(getattr(ModelConfig, "MODEL_BLEND_WEIGHT", 0.50))
    )
    video_threshold = float(getattr(ModelConfig, "VIDEO_FINAL_DEEPFAKE_THRESHOLD", 28.0))
    # Boost if faceswap_score is significant AND at least one signal is elevated
    if (
        faceswap_score >= float(getattr(ModelConfig, "FACE_SWAP_THRESHOLD", 48.0))
        and _faceswap_signal_count(faceswap_analysis) >= 1  # Lowered from 2
        and combined < video_threshold
    ):
        combined = max(
            combined,
            video_threshold + min(12.0, (faceswap_score - 48.0) * 0.25),
        )
    return round(float(max(0.0, min(100.0, combined))), 2)


def _apply_labeled_reference_adjustment(
    conf_percent,
    dataset_score=None,
    faceswap_score=0.0,
    freq_score=0.0,
    reenactment_signal=0.0,
    media_type="image",
    model_score=None,
):
    adjusted = float(conf_percent)
    threshold = _decision_threshold(media_type)
    heuristic_only = model_score is None
    reference_profile = _get_dataset_reference_profile()
    validation_accuracy = float((reference_profile.get("validation_summary", {}) or {}).get("accuracy", 0.0))
    dataset_reliable = validation_accuracy >= 0.62

    if dataset_score is not None:
        dataset_score = float(dataset_score)
        if dataset_reliable and dataset_score >= threshold:
            adjusted += min(9.0 if heuristic_only else 5.0, ((dataset_score - threshold) * 0.22) + max(0.0, freq_score - 35.0) * 0.04)
        elif dataset_reliable and dataset_score <= threshold - 10.0 and faceswap_score < threshold - 4.0 and reenactment_signal < 32.0:
            adjusted -= min(4.5, (threshold - dataset_score) * 0.12)

    if faceswap_score >= max(42.0, threshold - 8.0):
        adjusted += min(8.0, (faceswap_score - max(42.0, threshold - 8.0)) * 0.25)
    if reenactment_signal >= float(getattr(ModelConfig, "REENACTMENT_SIGNAL_THRESHOLD", 38.0)):
        adjusted += min(5.0, (float(reenactment_signal) - float(getattr(ModelConfig, "REENACTMENT_SIGNAL_THRESHOLD", 38.0))) * 0.18 + 1.2)
    if freq_score >= 48.0 and dataset_score is not None and float(dataset_score) >= threshold - 2.0:
        adjusted += min(3.5, (float(freq_score) - 48.0) * 0.08)

    return round(float(max(0.0, min(100.0, adjusted))), 2)


def _stabilize_image_confidence(conf_percent, base_confidence, face_forensics, faceswap_analysis, dataset_score=None):
    adjusted = float(conf_percent)
    threshold = _decision_threshold("image")
    top_region_mean = _top_region_mean(face_forensics, top_n=3)
    face_score = float(face_forensics.get("face_score", 0.0))
    face_detected = bool(face_forensics.get("face_detected"))
    face_authenticity = float(face_forensics.get("face_authenticity_score", 0.0))
    quality_score = float(face_forensics.get("face_quality", {}).get("quality_score", 0.0))
    faceswap_available = bool((faceswap_analysis or {}).get("available"))
    region_consensus = float((faceswap_analysis or {}).get("region_consensus_score", 0.0))

    if face_detected and (top_region_mean + face_score) >= 76.0 and region_consensus >= 36.0:
        adjusted = max(adjusted, threshold + min(6.0, ((top_region_mean + face_score) - 76.0) * 0.18 + 1.2))

    if not face_detected and not faceswap_available and quality_score < 50.0 and face_authenticity >= 66.0:
        adjusted = min(adjusted, max(float(base_confidence) + 1.0, threshold - 2.0))

    if dataset_score is not None and float(dataset_score) >= threshold + 10.0 and not face_detected and not faceswap_available:
        adjusted = min(adjusted, max(float(base_confidence) + 1.5, threshold - 1.0))

    return round(float(max(0.0, min(100.0, adjusted))), 2)


def _blend_video_frame_consensus(
    base_confidence,
    faceswap_analysis,
    dataset_score=None,
    freq_score=0.0,
    reenactment_signal=0.0,
    model_score=None,
):
    reference_profile = _get_dataset_reference_profile()
    validation_accuracy = float((reference_profile.get("validation_summary", {}) or {}).get("accuracy", 0.0))
    dataset_reliable = validation_accuracy >= 0.62
    components = [
        (float(base_confidence), 0.34),
        (float((faceswap_analysis or {}).get("faceswap_score", 0.0)), 0.34),
        (float(freq_score), 0.10),
        (float(reenactment_signal), 0.10),
    ]
    if dataset_score is not None and dataset_reliable:
        components.append((float(dataset_score), 0.12))

    total_weight = sum(weight for _, weight in components)
    combined = sum(score * weight for score, weight in components) / max(total_weight, 1e-6)
    combined = _apply_labeled_reference_adjustment(
        combined,
        dataset_score=dataset_score,
        faceswap_score=float((faceswap_analysis or {}).get("faceswap_score", 0.0)),
        freq_score=freq_score,
        reenactment_signal=reenactment_signal,
        media_type="video",
        model_score=model_score,
    )
    return round(float(max(0.0, min(100.0, combined))), 2)


def _blend_image_consensus(
    base_confidence,
    faceswap_analysis,
    dataset_score=None,
    freq_score=0.0,
    reenactment_signal=0.0,
):
    reference_profile = _get_dataset_reference_profile()
    validation_accuracy = float((reference_profile.get("validation_summary", {}) or {}).get("accuracy", 0.0))
    dataset_reliable = validation_accuracy >= 0.62
    components = [
        (
            float(base_confidence),
            float(getattr(ModelConfig, "IMAGE_BASE_BLEND_WEIGHT", 0.42)),
        ),
        (
            float(freq_score),
            float(getattr(ModelConfig, "IMAGE_FREQUENCY_BLEND_WEIGHT", 0.10)),
        ),
    ]

    if faceswap_analysis and faceswap_analysis.get("available"):
        components.append(
            (
                float(faceswap_analysis.get("faceswap_score", 0.0)),
                float(getattr(ModelConfig, "IMAGE_FACESWAP_BLEND_WEIGHT", 0.24)),
            )
        )
    if dataset_score is not None and dataset_reliable:
        components.append(
            (
                float(dataset_score),
                float(getattr(ModelConfig, "IMAGE_DATASET_BLEND_WEIGHT", 0.18)),
            )
        )
    if reenactment_signal > 0:
        components.append(
            (
                float(reenactment_signal),
                float(getattr(ModelConfig, "IMAGE_REENACTMENT_BLEND_WEIGHT", 0.06)),
            )
        )

    total_weight = sum(weight for _, weight in components if weight > 0)
    if total_weight <= 0:
        return round(float(base_confidence), 2)

    combined = sum(score * weight for score, weight in components if weight > 0) / total_weight
    faceswap_score = float((faceswap_analysis or {}).get("faceswap_score", 0.0))
    threshold = _decision_threshold("image")

    if faceswap_analysis and faceswap_analysis.get("available"):
        if (
            faceswap_score >= float(getattr(ModelConfig, "FACE_SWAP_THRESHOLD", 48.0))
            and _faceswap_signal_count(faceswap_analysis) >= 1
        ):
            combined = max(combined, (combined * 0.70) + (faceswap_score * 0.30))

    if reenactment_signal >= float(getattr(ModelConfig, "REENACTMENT_SIGNAL_THRESHOLD", 38.0)):
        combined = max(combined, (combined * 0.78) + (float(reenactment_signal) * 0.22))

    if dataset_score is not None and dataset_reliable:
        if float(dataset_score) >= threshold + 8.0 and float(freq_score) >= 40.0:
            combined += min(5.5, ((float(dataset_score) - threshold) * 0.10) + ((float(freq_score) - 40.0) * 0.04))
        elif (
            float(dataset_score) <= threshold - 10.0
            and faceswap_score < threshold
            and float(base_confidence) < threshold
        ):
            combined -= min(4.0, (threshold - float(dataset_score)) * 0.10)

    combined = _apply_labeled_reference_adjustment(
        combined,
        dataset_score=dataset_score,
        faceswap_score=faceswap_score,
        freq_score=freq_score,
        reenactment_signal=reenactment_signal,
        media_type="image",
    )

    return round(float(max(0.0, min(100.0, combined))), 2)


def _public_faceswap_payload(faceswap_analysis):
    if not isinstance(faceswap_analysis, dict):
        return {}
    return {key: value for key, value in faceswap_analysis.items() if not str(key).startswith("_")}


def _build_indicators(metrics, analyzed_frames, total_frames=None):
    indicators = {
        "edge_energy": round(metrics["edge_energy"], 5),
        "noise_score": round(metrics["noise_score"], 5),
        "blocking_score": round(metrics["blocking_score"], 5),
        "color_divergence": round(metrics["color_divergence"], 5),
        "analyzed_frames": analyzed_frames,
    }
    if total_frames is not None:
        indicators["total_frames"] = total_frames
    return indicators


def _top_region_mean(face_forensics, top_n=3):
    regions = face_forensics.get("top_regions", []) or []
    if not regions:
        return 0.0
    return float(np.mean([float(region.get("score", 0.0)) for region in regions[:top_n]]))


def _face_peak_score(face_forensics, faceswap_analysis=None):
    top_region = _top_region_mean(face_forensics, top_n=3)
    face_score = float(face_forensics.get("face_score", 0.0))
    faceswap_score = float((faceswap_analysis or {}).get("faceswap_score", 0.0))
    return round(max(top_region, face_score, faceswap_score), 2)


def _blend_face_scores(global_score, face_forensics):
    face_score = float(face_forensics.get("face_score", 0.0))
    top_region_mean = _top_region_mean(face_forensics, top_n=3)
    top_region_peak = float(max((float(region.get("score", 0.0)) for region in face_forensics.get("top_regions", [])[:3]), default=0.0))
    quality_score = float(face_forensics.get("face_quality", {}).get("quality_score", 60.0))
    landmark_integrity = face_forensics.get("landmark_integrity", {})

    consensus = (
        float(global_score) * 0.05
        + face_score * 0.30
        + top_region_mean * 0.65
    )
    if top_region_peak >= 45.0:
        consensus += min(8.0, (top_region_peak - 45.0) * 0.45)
    # Quality gate: ONLY suppress if quality is low AND no strong anomaly signal is present.
    # Compressed deepfakes often have low quality — we must not suppress suspicious frames.
    if quality_score < 45 and face_score < 30 and top_region_mean < 30:
        consensus *= 0.88  # Mild suppression only when all signals are genuinely low
    if landmark_integrity.get("available"):
        eye_alignment = float(landmark_integrity.get("eye_alignment", 60.0))
        symmetry = float(landmark_integrity.get("face_symmetry", 60.0))
        mouth_geometry = float(landmark_integrity.get("mouth_geometry", 60.0))
        contour = float(landmark_integrity.get("contour_consistency", 60.0))
        support = (eye_alignment + symmetry + mouth_geometry + contour) / 4.0
        if support > 80 and consensus < 60:
            consensus *= 0.90  # Only suppress on very clear authentic landmarks
        elif support < 45:
            consensus *= 1.12  # Stronger boost for poor landmark geometry
    else:
        consensus *= 1.06
    return round(float(max(0.0, min(100.0, consensus))), 2)


def extract_timeline_frames(video_path, num_frames=10, mode="full"):
    """Extract frames using a smarter strategy:
    - Uniform sampling across the whole video
    - Extra keyframes from first and last 20% (intro/outro often expose artifacts)
    - Avoid pure duplicate frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open or decode video block.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps

    if mode == "fast":
        base_fast = int(getattr(ModelConfig, "VIDEO_FAST_FRAME_COUNT", 14))
        if duration < 3:
            sample_n = max(8, base_fast - 2)
        elif duration < 10:
            sample_n = base_fast
        else:
            sample_n = base_fast + 4
    else:
        sample_n = max(num_frames, int(getattr(ModelConfig, "VIDEO_DEEP_FRAME_COUNT", 24)))

    sample_n = min(max(6, sample_n), total_frames)

    # Smart sampling: 70% uniform + 15% from first quarter + 15% from last quarter
    uniform_n = max(4, int(sample_n * 0.70))
    early_n = max(1, int(sample_n * 0.15))
    late_n = sample_n - uniform_n - early_n

    uniform_indices = np.linspace(0, total_frames - 1, uniform_n, dtype=int).tolist()
    early_indices = np.linspace(0, max(1, int(total_frames * 0.25)), early_n, dtype=int).tolist()
    late_indices = np.linspace(max(1, int(total_frames * 0.75)), total_frames - 1, late_n, dtype=int).tolist()

    # Merge and deduplicate while preserving order
    all_indices = sorted(set(uniform_indices + early_indices + late_indices))
    extracted = []
    prev_frame = None
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            # Skip near-duplicate frames (scene hasn't changed)
            if prev_frame is not None:
                diff = float(np.mean(np.abs(frame.astype(float) - prev_frame.astype(float))))
                if diff < 2.0:  # Very similar frames — skip
                    continue
            extracted.append((int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            prev_frame = frame.copy()

    cap.release()
    return extracted


def crop_and_align_face(frame_img, detector, margin_ratio=1.3):
    if detector is None:
        return None

    pil_img = Image.fromarray(frame_img)
    boxes, probs = detector.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(np.argmax(probs))
    confidence = probs[best_idx]
    if confidence < 0.85:
        return None

    x1, y1, x2, y2 = boxes[best_idx]
    w, h = x2 - x1, y2 - y1
    x_margin = (w * margin_ratio - w) / 2
    y_margin = (h * margin_ratio - h) / 2

    x1 = max(0, int(x1 - x_margin))
    y1 = max(0, int(y1 - y_margin))
    x2 = min(frame_img.shape[1], int(x2 + x_margin))
    y2 = min(frame_img.shape[0], int(y2 + y_margin))

    return frame_img[y1:y2, x1:x2]


def _choose_image_focus_region(original_bgr):
    aligned = face_alignment_service.align_primary_face(original_bgr)
    if aligned.aligned_face is not None and aligned.aligned_face.size > 0:
        return aligned.aligned_face

    detector = _get_mtcnn()
    rgb_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    face = crop_and_align_face(rgb_img, detector, margin_ratio=1.3) if detector is not None else None
    if face is not None and face.size > 0:
        return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    h, w = original_bgr.shape[:2]
    side = min(h, w)
    return original_bgr[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]


def generate_structured_reasons(is_video, conf_percent):
    threshold = _decision_threshold("video" if is_video else "image")
    strong_cutoff = threshold + 20
    moderate_cutoff = threshold + 8
    watch_cutoff = threshold - 4
    reasons = []
    if conf_percent >= strong_cutoff:
        reasons.append("Multi-region face evidence strongly supports synthetic manipulation.")
        reasons.append("Facial blending boundaries and feature coherence are materially degraded.")
        if is_video:
            reasons.append("Frame-to-frame facial consistency remains unstable across the sampled timeline.")
    elif conf_percent >= moderate_cutoff:
        reasons.append("Several facial regions show review-grade boundary or texture anomalies.")
        reasons.append("Face-led evidence is stronger than generic scene texture in this run.")
        if is_video:
            reasons.append("Temporal tracking around key face regions shows mild instability.")
    elif conf_percent >= watch_cutoff:
        reasons.append("Regional facial evidence is mixed and warrants manual review.")
        reasons.append("Face-focused cues are present but not strong enough for a hard fake verdict.")
        if is_video:
            reasons.append("Temporal evidence is weak and may be influenced by source compression.")
    else:
        reasons.append("Facial structure, texture continuity, and landmark geometry remain largely coherent.")
        reasons.append("No dominant face-region anomaly cluster is driving the verdict.")
        if is_video:
            reasons.append("Temporal face evidence remains stable across sampled frames.")
    return reasons


def _video_consensus_confidence(scored_frames, temporal_model_score=None):
    if not scored_frames:
        return 0.0, {}
    reference_profile = _get_dataset_reference_profile()
    validation_accuracy = float((reference_profile.get("validation_summary", {}) or {}).get("accuracy", 0.0))
    dataset_reliable = validation_accuracy >= 0.62

    ranked = sorted(scored_frames, key=lambda item: item["combined_confidence"], reverse=True)
    top_k = min(len(ranked), int(getattr(ModelConfig, "VIDEO_TOPK_FRAMES", 6)))
    top_window = ranked[:top_k]

    base_mean = float(np.mean([frame["combined_confidence"] for frame in top_window]))
    peak_mean = float(np.mean([frame["forensic_peak"] for frame in top_window]))
    critical_regions = {"middle_left", "middle_centre", "middle_right", "bottom_centre", "bottom_left", "bottom_right"}
    critical_scores = [
        float(frame["face_forensics"].get("top_regions", [{}])[0].get("score", 0.0))
        for frame in scored_frames
        if str(frame["face_forensics"].get("top_regions", [{}])[0].get("key", "")) in critical_regions
    ]
    critical_peak_mean = float(np.mean(sorted(critical_scores, reverse=True)[:top_k])) if critical_scores else peak_mean
    critical_hit_ratio = sum(1 for score in critical_scores if score >= 38.0) / max(len(scored_frames), 1)  # Lowered from 45
    persistence_hits = [
        frame
        for frame in scored_frames
        if frame["forensic_peak"] >= float(getattr(ModelConfig, "VIDEO_PERSISTENCE_THRESHOLD", 38.0))
        or float(frame["faceswap_analysis"].get("faceswap_score", 0.0)) >= 25.0  # Lowered from 30
    ]
    persistence_ratio = len(persistence_hits) / max(len(scored_frames), 1)
    persistence_boost = persistence_ratio * float(getattr(ModelConfig, "TEMPORAL_CONSENSUS_BOOST", 20.0))
    faceswap_peak = float(max(frame["faceswap_analysis"].get("faceswap_score", 0.0) for frame in top_window))

    # Reenactment signal: detect lip-sync / face reenactment across sampled frames
    reenactment_scores = [frame.get("reenactment_signal", 0.0) for frame in scored_frames]
    reenactment_peak = float(max(reenactment_scores)) if reenactment_scores else 0.0
    reenactment_mean = float(np.mean(reenactment_scores)) if reenactment_scores else 0.0
    reenactment_boost = 0.0
    reenact_threshold = float(getattr(ModelConfig, "REENACTMENT_SIGNAL_THRESHOLD", 38.0))
    if reenactment_peak >= reenact_threshold:
        reenactment_boost = min(15.0, (reenactment_peak - reenact_threshold) * 0.35 + reenactment_mean * 0.12)

    # Frequency heuristic: FFT-based GAN fingerprint signal
    freq_scores = [frame.get("freq_score", 0.0) for frame in scored_frames]
    freq_peak = float(max(freq_scores)) if freq_scores else 0.0
    freq_boost = min(8.0, _normalize(freq_peak, 25.0, 80.0) * 8.0) if freq_peak > 25.0 else 0.0
    dataset_scores = [frame.get("dataset_score") for frame in scored_frames if frame.get("dataset_score") is not None] if dataset_reliable else []
    dataset_peak = float(max(dataset_scores)) if dataset_scores else 0.0
    dataset_mean = float(np.mean(sorted(dataset_scores, reverse=True)[:top_k])) if dataset_scores else 0.0
    dataset_boost = min(12.0, _normalize(dataset_peak, 38.0, 80.0) * 12.0) if dataset_peak > 38.0 else 0.0

    combined = (
        (0.22 * base_mean)            # Reduced from 0.25
        + (0.20 * peak_mean)          # Reduced from 0.24
        + (0.24 * critical_peak_mean)
        + (0.16 * faceswap_peak)
        + (0.14 * dataset_mean)
        + persistence_boost
        + (critical_hit_ratio * 14.0) # Increased from 10
        + reenactment_boost
        + freq_boost
        + dataset_boost
    )
    if temporal_model_score is not None:
        combined = (0.75 * combined) + (0.25 * float(temporal_model_score))

    diagnostics = {
        "top_window_mean": round(base_mean, 2),
        "forensic_peak_mean": round(peak_mean, 2),
        "critical_peak_mean": round(critical_peak_mean, 2),
        "critical_hit_ratio": round(critical_hit_ratio, 3),
        "persistence_ratio": round(persistence_ratio, 3),
        "faceswap_peak": round(faceswap_peak, 2),
        "dataset_peak": round(dataset_peak, 2),
        "dataset_mean": round(dataset_mean, 2),
        "reenactment_peak": round(reenactment_peak, 2),
        "freq_peak": round(freq_peak, 2),
        "temporal_model_score": round(float(temporal_model_score), 2) if temporal_model_score is not None else None,
    }
    return round(float(max(0.0, min(100.0, combined))), 2), diagnostics


def _video_support_profile(scored_frames, face_forensics, faceswap_analysis, temporal_model_score=None):
    total_frames = max(len(scored_frames), 1)
    direct_face_hits = sum(
        1 for frame in scored_frames if bool(frame.get("face_forensics", {}).get("face_detected"))
    )
    model_scores = [
        float(frame["model_score"])
        for frame in scored_frames
        if frame.get("model_score") is not None
    ]
    faceswap_scores = [
        float(frame.get("faceswap_analysis", {}).get("faceswap_score", 0.0))
        for frame in scored_frames
    ]
    dataset_scores = [
        float(frame["dataset_score"])
        for frame in scored_frames
        if frame.get("dataset_score") is not None
    ]
    profile = {
        "direct_face_ratio": round(direct_face_hits / total_frames, 3),
        "fallback_face_ratio": round(1.0 - (direct_face_hits / total_frames), 3),
        "faceswap_available": bool((faceswap_analysis or {}).get("available")),
        "max_faceswap_score": round(max(faceswap_scores, default=0.0), 2),
        "max_dataset_score": round(max(dataset_scores, default=0.0), 2),
        "max_model_score": round(max(model_scores), 2) if model_scores else None,
        "temporal_model_available": temporal_model_score is not None,
        "face_authenticity_score": round(float(face_forensics.get("face_authenticity_score", 0.0)), 2),
        "primary_face_detected": bool(face_forensics.get("face_detected")),
    }
    return profile


def _calibrate_video_confidence(
    conf_percent,
    scored_frames,
    video_consensus,
    face_forensics,
    faceswap_analysis,
    temporal_model_score=None,
):
    threshold = _decision_threshold("video")
    adjusted = float(conf_percent)
    profile = _video_support_profile(
        scored_frames,
        face_forensics,
        faceswap_analysis,
        temporal_model_score=temporal_model_score,
    )
    adjustments = []

    weak_signal_band = adjusted < (threshold + 10.0)
    direct_face_ratio = float(profile["direct_face_ratio"])
    persistence_ratio = float(video_consensus.get("persistence_ratio", 0.0))
    critical_hit_ratio = float(video_consensus.get("critical_hit_ratio", 0.0))
    face_authenticity = float(profile["face_authenticity_score"])
    max_faceswap_score = float(profile["max_faceswap_score"])
    max_dataset_score = float(profile["max_dataset_score"])

    if weak_signal_band and direct_face_ratio < float(getattr(ModelConfig, "VIDEO_WEAK_FACE_RATIO_FLOOR", 0.35)):
        adjusted *= 0.82
        adjustments.append("weak-face-lock")

    if weak_signal_band and not profile["faceswap_available"]:
        adjusted *= 0.86
        adjustments.append("no-faceswap-confirmation")

    if weak_signal_band and max_faceswap_score < 24.0:
        adjusted *= 0.92
        adjustments.append("low-faceswap-signal")

    if weak_signal_band and max_dataset_score >= threshold + 4.0:
        adjusted = max(adjusted, conf_percent + min(10.0, (max_dataset_score - threshold) * 0.30))
        adjustments.append("dataset-support")

    if (
        weak_signal_band
        and persistence_ratio < float(getattr(ModelConfig, "VIDEO_LOW_PERSISTENCE_FLOOR", 0.18))
        and critical_hit_ratio < float(getattr(ModelConfig, "VIDEO_LOW_CRITICAL_HIT_FLOOR", 0.18))
    ):
        adjusted *= 0.88
        adjustments.append("weak-temporal-support")

    if (
        weak_signal_band
        and face_authenticity >= float(getattr(ModelConfig, "VIDEO_AUTHENTICITY_SUPPORT_FLOOR", 65.0))
    ):
        adjusted *= 0.85
        adjustments.append("authenticity-support")

    if weak_signal_band and temporal_model_score is None and profile["max_model_score"] is None:
        adjusted *= 0.98
        adjustments.append("heuristic-only")

    adjusted = round(float(max(0.0, min(100.0, adjusted))), 2)
    profile["applied_adjustments"] = adjustments
    profile["calibrated"] = bool(adjustments)
    profile["raw_confidence"] = round(float(conf_percent), 2)
    profile["adjusted_confidence"] = adjusted
    return adjusted, profile


def run_image_inference(file_path, job_id):
    try:
        original_bgr = cv2.imread(file_path)
        if original_bgr is None:
            raise ValueError(f"CRIT_ERROR: Failed to load image matrix at {file_path}")

        set_progress(job_id, 30, "Calibrating neural face regions...")
        face_forensics = face_region_analyzer.analyze(
            original_bgr,
            job_id=job_id,
            media_type="image",
            frame_metadata={"selection_reason": "Primary uploaded image used for facial region analysis."},
            persist=True,
        )
        focus_region = _choose_image_focus_region(original_bgr)
        metrics = _artifact_metrics(focus_region)
        global_score = _score_from_metrics(metrics) * 100.0
        base_confidence = _blend_face_scores(global_score, face_forensics)
        freq_score = _frequency_heuristic_score(focus_region)

        faceswap_analysis = faceswap_detector.analyze(
            original_bgr,
            job_id=job_id,
            media_type="image",
            face_forensics=face_forensics,
            frame_metadata={"selection_reason": "Primary uploaded image used for face-swap detection."},
            persist=True,
        )
        reenactment_signal = _reenactment_signal_score(face_forensics, faceswap_analysis)
        dataset_profile = _get_dataset_reference_profile()
        reference_context = _reference_context(dataset_profile)
        dataset_score, dataset_evidence = score_observation(
            {
                "edge_energy": float(metrics["edge_energy"]),
                "noise_score": float(metrics["noise_score"]),
                "blocking_score": float(metrics["blocking_score"]),
                "color_divergence": float(metrics["color_divergence"]),
                "global_score": float(global_score),
                "freq_score": float(freq_score),
            },
            dataset_profile,
        )
        model_score = _predict_model_score(_choose_image_focus_region(original_bgr), face_forensics, faceswap_analysis)
        conf_percent = _blend_image_consensus(
            base_confidence,
            faceswap_analysis,
            dataset_score=dataset_score,
            freq_score=freq_score,
            reenactment_signal=reenactment_signal,
        )
        if model_score is not None:
            conf_percent = round(
                (conf_percent * float(getattr(ModelConfig, "HEURISTIC_BLEND_WEIGHT", 0.50)))
                + (float(model_score) * float(getattr(ModelConfig, "MODEL_BLEND_WEIGHT", 0.50))),
                2,
            )
        else:
            conf_percent = _apply_labeled_reference_adjustment(
                conf_percent,
                dataset_score=dataset_score,
                faceswap_score=float(faceswap_analysis.get("faceswap_score", 0.0)),
                freq_score=freq_score,
                reenactment_signal=reenactment_signal,
                media_type="image",
                model_score=None,
            )
        conf_percent = _stabilize_image_confidence(
            conf_percent,
            base_confidence,
            face_forensics,
            faceswap_analysis,
            dataset_score=dataset_score,
        )
        verdict = _verdict_from_confidence(conf_percent, media_type="image")
        deepfake_type = _classify_deepfake_type(verdict, faceswap_analysis, face_forensics=face_forensics)
        review_note = _review_note(conf_percent, media_type="image")
        evidence_strength = _evidence_strength(conf_percent, media_type="image")

        set_progress(job_id, 65, "Executing Forensic XAI Suite...")
        xai_data = xai_manager.generate_reports(
            original_bgr,
            job_id=job_id,
            media_type="image",
            frame_metadata={"selection_reason": "Primary uploaded image used for explainability generation."},
        )

        set_progress(job_id, 90, "Preparing forensic report card...")
        heatmap_rel = xai_data.get("primary_heatmap", "")
        detector = _get_mtcnn()
        if heatmap_rel:
            full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", heatmap_rel))
            if detector is not None:
                add_forensic_overlays_to_path(full_path, detector, conf_percent)

        reasons = generate_structured_reasons(False, conf_percent)
        reasons.append(face_forensics.get("landmark_integrity", {}).get("summary", ""))
        reasons.append(face_forensics.get("face_quality", {}).get("note", ""))
        reasons.extend(
            f"{region['label']}: {region['explanation']}"
            for region in face_forensics.get("top_regions", [])[:2]
        )
        reasons.extend(faceswap_analysis.get("explanations", []))
        if dataset_evidence.get("available") and dataset_score is not None:
            if dataset_score >= _decision_threshold("image"):
                reasons.append("Cross-dataset calibration aligns this upload with fake reference patterns from the integrated datasets.")
            elif dataset_score <= (_decision_threshold("image") - 8.0):
                reasons.append("Cross-dataset calibration aligns this upload more closely with authentic reference samples.")
            support_labels = [
                item.get("feature", "").replace("_", " ")
                for item in dataset_evidence.get("supporting_features", [])
                if item.get("feature")
            ]
            if support_labels:
                reasons.append(
                    "Dataset-backed signals were strongest in: " + ", ".join(support_labels[:3]) + "."
                )
        if freq_score >= 52.0:
            reasons.append("Frequency-domain inspection found elevated periodic artifacts often associated with synthetic generation.")
        if reenactment_signal >= float(getattr(ModelConfig, "REENACTMENT_SIGNAL_THRESHOLD", 38.0)):
            reasons.append("Mouth and contour geometry show reenactment-style distortion cues.")
        set_progress(job_id, 100, "Forensic Dashboard synchronized.")

        public_faceswap = _public_faceswap_payload(faceswap_analysis)
        if deepfake_type == "Face Swap":
            explanation = "Face-centered analysis indicates face-swap-like identity drift and facial boundary blending inconsistencies."
        elif deepfake_type == "Reenactment / Lip-Sync":
            explanation = "Facial geometry analysis indicates reenactment-style distortion across the mouth, contour, and central face regions."
        elif verdict == "DEEPFAKE":
            explanation = "Artifact analysis indicates elevated manipulation cues across the uploaded face region, reinforced by cross-dataset reference calibration."
        else:
            explanation = "Artifact analysis indicates strong biological consistency across the uploaded face region and matches the authentic side of the integrated dataset references."

        return sanitize_numpy(
            {
                "success": True,
                "verdict": verdict,
                "confidence": conf_percent,
                "explanation": explanation,
                "review_note": review_note,
                "evidence_strength": evidence_strength,
                "heatmap_url": heatmap_rel,
                "reasons": reasons,
                "xai_reports": xai_data.get("xai_reports", []),
                "xai_basic_reports": xai_data.get("xai_basic_reports", []),
                "xai_advanced_reports": xai_data.get("xai_advanced_reports", []),
                "xai_context": xai_data.get("xai_context", {}),
                "face_forensics": face_forensics,
                "faceswap_analysis": public_faceswap,
                "strongest_frame": public_faceswap.get("strongest_frame"),
                "deepfake_type": deepfake_type,
                "dataset_calibration": dataset_evidence,
                "model_status": _model_status_for_media("image"),
                "reference_datasets": reference_context["reference_datasets"],
                "calibration_mode": reference_context["calibration_mode"],
                "indicators": {
                    **_build_indicators(metrics, analyzed_frames=1),
                    "face_region_score": round(face_forensics["face_score"], 2),
                    "faceswap_score": round(float(public_faceswap.get("faceswap_score", 0.0)), 2),
                    "dataset_score": round(float(dataset_score), 2) if dataset_score is not None else None,
                    "frequency_score": round(float(freq_score), 2),
                    "reenactment_signal": round(float(reenactment_signal), 2),
                    "model_score": round(float(model_score), 2) if model_score is not None else None,
                },
                "file_type": "IMAGE",
            }
        )
    except Exception as exc:
        print(f"PIPELINE_CRASH: {exc}")
        return {"success": False, "error": f"Forensic Pipeline Failure: {exc}"}


def run_video_inference(file_path, job_id, mode="fast"):
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video stream payload.")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        video_duration_seconds = float(total_video_frames / video_fps) if video_fps else 0.0

        set_progress(job_id, 30, "Extracting strategic keyframe sequence...")
        dataset_profile = _get_dataset_reference_profile()
        reference_context = _reference_context(dataset_profile)
        frames = extract_timeline_frames(
            file_path,
            num_frames=int(getattr(ModelConfig, "VIDEO_DEEP_FRAME_COUNT", 24)),
            mode=mode,
        )
        if not frames:
            raise ValueError("Video payload contains zero extractable faces.")

        scored_frames = []
        set_progress(job_id, 45, "Calibrating neural face regions...")
        for idx, (frame_number, rgb_img) in enumerate(frames):
            sub_progress = 45 + int((idx / len(frames)) * 20)
            set_progress(job_id, sub_progress, f"Scanning facial landmarks [Frame {idx + 1}/{len(frames)}]...")

            bgr_frame = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            aligned_face = _choose_image_focus_region(bgr_frame)
            frame_context = {
                "frame_number": int(frame_number),
                "timestamp_label": f"{(frame_number / video_fps):.2f}s",
            }
            preview_face_forensics = face_region_analyzer.analyze(
                bgr_frame,
                job_id=f"{job_id}_frame_{frame_number}",
                media_type="video",
                frame_metadata=frame_context,
                persist=False,
            )
            metrics = _artifact_metrics(aligned_face)
            global_score = _score_from_metrics(metrics) * 100.0
            frame_confidence = _blend_face_scores(global_score, preview_face_forensics)
            preview_faceswap = faceswap_detector.analyze(
                bgr_frame,
                job_id=f"{job_id}_frame_{frame_number}",
                media_type="video",
                face_forensics=preview_face_forensics,
                frame_metadata=frame_context,
                persist=False,
            )
            dataset_score, _ = score_observation(
                {
                    "edge_energy": float(metrics["edge_energy"]),
                    "noise_score": float(metrics["noise_score"]),
                    "blocking_score": float(metrics["blocking_score"]),
                    "color_divergence": float(metrics["color_divergence"]),
                    "global_score": float(global_score),
                },
                dataset_profile,
            )
            freq_score = _frequency_heuristic_score(aligned_face)
            reenactment_signal = _reenactment_signal_score(preview_face_forensics, preview_faceswap)
            model_score = _predict_model_score(aligned_face, preview_face_forensics, preview_faceswap)
            combined_confidence = _blend_video_frame_consensus(
                frame_confidence,
                preview_faceswap,
                dataset_score=dataset_score,
                freq_score=freq_score,
                reenactment_signal=reenactment_signal,
                model_score=model_score,
            )
            if model_score is not None:
                combined_confidence = round(
                    (combined_confidence * float(getattr(ModelConfig, "HEURISTIC_BLEND_WEIGHT", 0.50)))
                    + (float(model_score) * float(getattr(ModelConfig, "MODEL_BLEND_WEIGHT", 0.50))),
                    2,
                )
            forensic_peak = _face_peak_score(preview_face_forensics, preview_faceswap)
            scored_frames.append(
                {
                    "frame_number": int(frame_number),
                    "timestamp_seconds": float(frame_number / video_fps),
                    "timestamp": f"{(frame_number / video_fps):.2f}s",
                    "confidence": frame_confidence,
                    "combined_confidence": combined_confidence,
                    "metrics": metrics,
                    "bgr_frame": bgr_frame,
                    "aligned_face": aligned_face,
                    "face_forensics": preview_face_forensics,
                    "faceswap_analysis": preview_faceswap,
                    "model_score": model_score,
                    "forensic_peak": forensic_peak,
                    "freq_score": freq_score,
                    "reenactment_signal": reenactment_signal,
                    "dataset_score": dataset_score,
                }
            )

        if not scored_frames:
            raise ValueError("Video payload contains zero analysable frames.")

        scored_frames.sort(key=lambda item: item["combined_confidence"], reverse=True)
        primary_frame = scored_frames[0]
        face_forensics = face_region_analyzer.analyze(
            primary_frame["bgr_frame"],
            job_id=job_id,
            media_type="video",
            frame_metadata={
                "frame_number": primary_frame["frame_number"],
                "timestamp_label": primary_frame["timestamp"],
                "selection_reason": "Top suspicious frame selected from the temporal sample set.",
            },
            persist=True,
        )
        top_faceswap = faceswap_detector.analyze(
            primary_frame["bgr_frame"],
            job_id=job_id,
            media_type="video",
            face_forensics=face_forensics,
            frame_metadata={
                "frame_number": primary_frame["frame_number"],
                "timestamp_label": primary_frame["timestamp"],
                "selection_reason": "Top suspicious frame selected from the temporal sample set.",
            },
            persist=True,
        )
        faceswap_analysis = faceswap_detector.aggregate_video_results(
            [frame["faceswap_analysis"] for frame in scored_frames],
            top_faceswap,
        )

        set_progress(job_id, 65, f"Running AI anomaly inference [{mode.upper()} MODE]...")
        xai_data = xai_manager.generate_reports(
            primary_frame["bgr_frame"],
            job_id=job_id,
            media_type="video",
            frame_metadata={
                "frame_number": primary_frame["frame_number"],
                "timestamp_seconds": round(primary_frame["timestamp_seconds"], 3),
                "timestamp_label": primary_frame["timestamp"],
                "selection_reason": "Top suspicious frame selected from the temporal sample set.",
            },
        )

        for idx, report in enumerate(xai_data["xai_reports"]):
            progress_value = 66 + int(((idx + 1) / len(xai_data["xai_reports"])) * 14)
            set_progress(job_id, progress_value, f"Forensic Scan: {report['method']} ready...")

        audio_results = None
        should_run_audio, audio_reason = _should_run_audio_scan(mode, video_duration_seconds)
        if should_run_audio:
            local_audio_suite = _get_audio_suite()
            if local_audio_suite is not None:
                set_progress(job_id, 82, "Running audio-visual support scan...")
                try:
                    audio_results = local_audio_suite.process_video_audio(
                        file_path,
                        job_id,
                        analysis_seconds=float(getattr(ModelConfig, "AUDIO_MAX_ANALYSIS_SECONDS", 12)),
                    )
                except Exception as audio_error:
                    audio_results = {"success": False, "error": str(audio_error)}
            else:
                audio_results = {"success": False, "error": "Audio suite unavailable."}
        else:
            skip_messages = {
                "demo_mode": "Skipping audio support scan in demo mode.",
                "disabled": "Audio support scan is disabled.",
                "fast_mode": "Skipping audio support scan in fast mode for quicker turnaround.",
                "video_too_long": "Skipping audio support scan for long video to keep turnaround fast.",
            }
            set_progress(job_id, 82, skip_messages.get(audio_reason, "Skipping audio support scan."))
            audio_results = {
                "success": False,
                "skipped": True,
                "reason": audio_reason,
            }

        suspicious_frames = []
        for idx, frame in enumerate(scored_frames[:3]):
            suspicious_frames.append(
                {
                    "frame_number": frame["frame_number"],
                    "timestamp": frame["timestamp"],
                    "confidence": frame["combined_confidence"],
                    "explanation": (
                        "Highest anomaly concentration in the sampled timeline."
                        if idx == 0
                        else "Repeated face-region anomalies compared with neighboring frames."
                    ),
                    "is_highest": idx == 0,
                }
            )

        temporal_model_score = _predict_temporal_model_score(scored_frames)
        raw_conf_percent, video_consensus = _video_consensus_confidence(
            scored_frames,
            temporal_model_score=temporal_model_score,
        )
        conf_percent, decision_support = _calibrate_video_confidence(
            raw_conf_percent,
            scored_frames,
            video_consensus,
            face_forensics,
            faceswap_analysis,
            temporal_model_score=temporal_model_score,
        )
        verdict = _verdict_from_confidence(conf_percent, media_type="video")
        deepfake_type = _classify_deepfake_type(verdict, faceswap_analysis, face_forensics=face_forensics)
        review_note = _review_note(conf_percent, media_type="video")
        evidence_strength = _evidence_strength(conf_percent, media_type="video")
        global_heatmap_url = xai_data["primary_heatmap"]

        set_progress(job_id, 95, "Synchronizing forensic logic dashboard...")
        reasons = generate_structured_reasons(True, conf_percent)
        reasons.append(face_forensics.get("landmark_integrity", {}).get("summary", ""))
        reasons.append(face_forensics.get("face_quality", {}).get("note", ""))
        reasons.extend(
            f"{region['label']}: {region['explanation']}"
            for region in face_forensics.get("top_regions", [])[:2]
        )
        reasons.extend(faceswap_analysis.get("explanations", []))
        if video_consensus.get("persistence_ratio", 0.0) >= float(getattr(ModelConfig, "VIDEO_REPEAT_FAKE_RATIO", 0.45)):
            reasons.append("Repeated facial anomaly evidence persists across the sampled timeline.")
        if video_consensus.get("dataset_peak", 0.0) >= _decision_threshold("video"):
            reasons.append("Cross-dataset calibration agrees with a fake-like pattern in the strongest sampled frames.")
        set_progress(job_id, 100, "Full Analysis Matrix Complete.")

        mean_metrics = {
            "edge_energy": float(np.mean([frame["metrics"]["edge_energy"] for frame in scored_frames])),
            "noise_score": float(np.mean([frame["metrics"]["noise_score"] for frame in scored_frames])),
            "blocking_score": float(np.mean([frame["metrics"]["blocking_score"] for frame in scored_frames])),
            "color_divergence": float(np.mean([frame["metrics"]["color_divergence"] for frame in scored_frames])),
        }

        full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", global_heatmap_url))
        detector = _get_mtcnn()
        if detector is not None:
            add_forensic_overlays_to_path(full_path, detector, conf_percent)

        public_faceswap = _public_faceswap_payload(faceswap_analysis)
        if deepfake_type == "Face Swap":
            explanation = (
                f"Spatio-temporal face analysis selected frame {primary_frame['frame_number']} at {primary_frame['timestamp']} "
                "as the strongest face-swap evidence anchor."
            )
        elif verdict == "DEEPFAKE":
            explanation = (
                f"Spatio-temporal analysis of {len(frames)} sampled frames found repeated manipulation cues, "
                f"with frame {primary_frame['frame_number']} at {primary_frame['timestamp']} acting as the strongest explainability anchor."
            )
        else:
            explanation = (
                f"Spatio-temporal analysis of {len(frames)} sampled frames did not find strong enough evidence for a deepfake verdict; "
                f"frame {primary_frame['frame_number']} at {primary_frame['timestamp']} was retained as the main review anchor."
            )

        return sanitize_numpy(
            {
                "success": True,
                "verdict": verdict,
                "confidence": conf_percent,
                "explanation": explanation,
                "review_note": review_note,
                "evidence_strength": evidence_strength,
                "heatmap_url": global_heatmap_url,
                "reasons": reasons,
                "xai_reports": xai_data.get("xai_reports", []),
                "xai_basic_reports": xai_data.get("xai_basic_reports", []),
                "xai_advanced_reports": xai_data.get("xai_advanced_reports", []),
                "xai_context": xai_data.get("xai_context", {}),
                "face_forensics": face_forensics,
                "faceswap_analysis": public_faceswap,
                "timeline_heatmaps": [global_heatmap_url],
                "suspicious_frames": suspicious_frames,
                "strongest_frame": public_faceswap.get("strongest_frame"),
                "forensic_details": {
                    "fps": video_fps,
                    "duration_seconds": round(video_duration_seconds, 2),
                    "frames_analyzed": len(frames),
                    "selected_frame_number": primary_frame["frame_number"],
                    "selected_frame_timestamp": primary_frame["timestamp"],
                    "temporal_stability": "STABLE" if conf_percent < 65 else "JITTER_DETECTED",
                    "raw_confidence": raw_conf_percent,
                    "top_window_mean": video_consensus.get("top_window_mean"),
                    "forensic_peak_mean": video_consensus.get("forensic_peak_mean"),
                    "critical_peak_mean": video_consensus.get("critical_peak_mean"),
                    "critical_hit_ratio": video_consensus.get("critical_hit_ratio"),
                    "persistence_ratio": video_consensus.get("persistence_ratio"),
                    "decision_support": decision_support,
                    "audio_scan": {
                        "requested": bool(should_run_audio),
                        "status": "completed" if audio_results and audio_results.get("success") else ("skipped" if audio_results and audio_results.get("skipped") else "unavailable"),
                        "reason": None if should_run_audio else audio_reason,
                    },
                },
                "audio_forensics": audio_results if audio_results and audio_results.get("success") else None,
                "deepfake_type": deepfake_type,
                "model_status": _model_status_for_media("video"),
                "reference_datasets": reference_context["reference_datasets"],
                "calibration_mode": reference_context["calibration_mode"],
                "indicators": {
                    **_build_indicators(mean_metrics, analyzed_frames=len(frames), total_frames=total_video_frames),
                    "face_region_score": round(face_forensics["face_score"], 2),
                    "faceswap_score": round(float(public_faceswap.get("faceswap_score", 0.0)), 2),
                    "dataset_score": round(float(video_consensus.get("dataset_peak", 0.0)), 2),
                    "model_score": max([frame["model_score"] for frame in scored_frames if frame["model_score"] is not None], default=None),
                    "temporal_model_score": temporal_model_score,
                    "forensic_peak_mean": video_consensus.get("forensic_peak_mean"),
                    "critical_peak_mean": video_consensus.get("critical_peak_mean"),
                    "critical_hit_ratio": video_consensus.get("critical_hit_ratio"),
                    "persistence_ratio": video_consensus.get("persistence_ratio"),
                },
                "file_type": "VIDEO",
            }
        )
    except Exception as exc:
        print(f"VIDEO_PIPELINE_CRASH: {exc}")
        return {"success": False, "error": f"Video Detection Logic Failed: {exc}"}
