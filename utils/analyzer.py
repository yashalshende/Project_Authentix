"""Lightweight deepfake risk estimation pipeline for AUTHENTIX.

This module provides practical, CPU-friendly analysis using visual artifact
heuristics. It is intentionally positioned as an academic prototype.
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Tuple

import cv2
import numpy as np


# Thresholds tuned for demo behavior and stability across common media.
SOBEL_LOW_THRESHOLD = 0.05
NOISE_HIGH_THRESHOLD = 0.12
BLOCKING_HIGH_THRESHOLD = 0.10
COLOR_DIVERGENCE_THRESHOLD = 0.08


def _normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v == min_v:
        return 0.0
    clipped = max(min(value, max_v), min_v)
    return float((clipped - min_v) / (max_v - min_v))


def _artifact_metrics(frame: np.ndarray) -> Dict[str, float]:
    """Compute interpretable artifact scores from a frame/image."""
    bgr = frame.astype(np.float32) / 255.0
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    # 1) Edge consistency via Sobel gradients.
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    # 2) High-frequency noise estimation.
    blurred = cv2.GaussianBlur(gray_f, (5, 5), 0)
    high_freq = gray_f - blurred
    noise_score = float(np.std(high_freq))

    # 3) Compression/blocking artifact approximation using 8x8 boundary jumps.
    diff_x = np.abs(np.diff(gray_f, axis=1))
    diff_y = np.abs(np.diff(gray_f, axis=0))
    block_x = float(np.mean(diff_x[:, 7::8])) if diff_x.shape[1] > 8 else 0.0
    block_y = float(np.mean(diff_y[7::8, :])) if diff_y.shape[0] > 8 else 0.0
    blocking = (block_x + block_y) / 2.0

    # 4) Color channel divergence (captures odd synthetic blending tendencies).
    b, g, r = cv2.split(bgr)
    rg_div = float(np.mean(np.abs(r - g)))
    gb_div = float(np.mean(np.abs(g - b)))
    rb_div = float(np.mean(np.abs(r - b)))
    color_divergence = (rg_div + gb_div + rb_div) / 3.0

    return {
        "edge_energy": edge_energy,
        "noise_score": noise_score,
        "blocking_score": blocking,
        "color_divergence": color_divergence,
    }


def _risk_from_metrics(metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    edge_component = 1.0 - _normalize(metrics["edge_energy"], 0.03, 0.22)
    noise_component = _normalize(metrics["noise_score"], 0.02, 0.20)
    block_component = _normalize(metrics["blocking_score"], 0.01, 0.22)
    color_component = _normalize(metrics["color_divergence"], 0.02, 0.25)

    risk = (
        (0.36 * edge_component)
        + (0.22 * noise_component)
        + (0.27 * block_component)
        + (0.15 * color_component)
    )
    risk = float(max(0.0, min(1.0, risk)))

    reasons: List[str] = []
    if metrics["edge_energy"] < SOBEL_LOW_THRESHOLD:
        reasons.append("low edge detail consistency")
    if metrics["noise_score"] > NOISE_HIGH_THRESHOLD:
        reasons.append("elevated high-frequency residual noise")
    if metrics["blocking_score"] > BLOCKING_HIGH_THRESHOLD:
        reasons.append("strong block-boundary compression artifacts")
    if metrics["color_divergence"] > COLOR_DIVERGENCE_THRESHOLD:
        reasons.append("unusual RGB channel divergence")

    if not reasons:
        reasons.append("no dominant synthetic artifact detected")

    return risk, reasons


def _verdict_from_risk(risk: float) -> str:
    if risk < 0.35:
        return "Real"
    if risk < 0.65:
        return "Suspicious"
    return "Likely Deepfake"


def analyze_image(image_path: str) -> Dict[str, object]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file.")

    metrics = _artifact_metrics(image)
    risk, reasons = _risk_from_metrics(metrics)
    verdict = _verdict_from_risk(risk)

    confidence = round(risk * 100.0, 2)
    explanation = (
        "Image evaluated using artifact heuristics: "
        + ", ".join(reasons)
        + "."
    )

    indicators = {
        "edge_energy": round(metrics["edge_energy"], 5),
        "noise_score": round(metrics["noise_score"], 5),
        "blocking_score": round(metrics["blocking_score"], 5),
        "color_divergence": round(metrics["color_divergence"], 5),
        "analyzed_frames": 1,
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "indicators": indicators,
    }


def analyze_video(video_path: str, max_frames: int = 20) -> Dict[str, object]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has no readable frames.")

    sample_count = min(max_frames, total_frames)
    positions = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)

    frame_risks: List[float] = []
    all_metrics: List[Dict[str, float]] = []
    aggregate_reasons: List[str] = []

    for frame_index in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        metrics = _artifact_metrics(frame)
        risk, reasons = _risk_from_metrics(metrics)
        frame_risks.append(risk)
        all_metrics.append(metrics)
        aggregate_reasons.extend(reasons)

    cap.release()

    if not frame_risks:
        raise ValueError("Unable to sample readable frames from video.")

    avg_risk = float(mean(frame_risks))
    verdict = _verdict_from_risk(avg_risk)
    confidence = round(avg_risk * 100.0, 2)

    mean_metrics = {
        "edge_energy": float(mean(m["edge_energy"] for m in all_metrics)),
        "noise_score": float(mean(m["noise_score"] for m in all_metrics)),
        "blocking_score": float(mean(m["blocking_score"] for m in all_metrics)),
        "color_divergence": float(mean(m["color_divergence"] for m in all_metrics)),
    }

    unique_reasons = sorted(set(aggregate_reasons))
    explanation = (
        f"Video sampled across {len(frame_risks)} frame(s); "
        "aggregated artifact profile indicates "
        + ", ".join(unique_reasons[:4])
        + "."
    )

    indicators = {
        "edge_energy": round(mean_metrics["edge_energy"], 5),
        "noise_score": round(mean_metrics["noise_score"], 5),
        "blocking_score": round(mean_metrics["blocking_score"], 5),
        "color_divergence": round(mean_metrics["color_divergence"], 5),
        "analyzed_frames": len(frame_risks),
        "total_frames": total_frames,
    }

    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "indicators": indicators,
    }
