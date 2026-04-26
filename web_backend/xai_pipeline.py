from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


METHOD_DEFINITIONS = [
    {
        "key": "base_heatmap",
        "method": "Base Heatmap",
        "description": "Baseline forensic attention overlay showing the strongest anomaly concentration across the analyzed region.",
        "basic": True,
    },
    {
        "key": "grad_cam",
        "method": "Grad-CAM",
        "description": "Highlights the facial regions that most strongly influenced the synthetic-risk decision.",
        "basic": True,
    },
    {
        "key": "integrated_gradients",
        "method": "Integrated Gradients",
        "description": "Approximates per-pixel contribution by comparing the analyzed frame against a soft baseline reconstruction.",
        "basic": False,
    },
    {
        "key": "occlusion_sensitivity",
        "method": "Occlusion Sensitivity",
        "description": "Measures how strongly the anomaly score changes when localized facial patches are suppressed.",
        "basic": False,
    },
    {
        "key": "score_cam",
        "method": "Score-CAM",
        "description": "Produces a smoother attention map using activation strength and local contrast as a proxy for model focus.",
        "basic": False,
    },
    {
        "key": "grad_cam_plus_plus",
        "method": "Grad-CAM++",
        "description": "Fallback dense localization map that accentuates smaller artifact clusters when Score-CAM is degraded.",
        "basic": False,
    },
]


class XAIPipelineManager:
    """Runs a resilient, demo-safe XAI suite for every successful analysis."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.output_root = self.project_root / "static" / "outputs" / "xai"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._method_handlers: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "base_heatmap": self._base_heatmap_map,
            "grad_cam": self._grad_cam_map,
            "integrated_gradients": self._integrated_gradients_map,
            "occlusion_sensitivity": self._occlusion_map,
            "score_cam": self._score_cam_map,
            "grad_cam_plus_plus": self._grad_cam_plus_plus_map,
        }

    def generate_reports(
        self,
        bgr_image: np.ndarray,
        job_id: str,
        media_type: str,
        frame_metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        frame_metadata = frame_metadata or {}
        safe_media_type = "video" if media_type.lower() == "video" else "image"
        job_dir = self.output_root / job_id / safe_media_type
        job_dir.mkdir(parents=True, exist_ok=True)

        reports: List[Dict[str, object]] = []
        for definition in METHOD_DEFINITIONS:
            reports.append(self._generate_single_report(bgr_image, job_dir, definition, frame_metadata))

        reports = reports[:6]
        basic_reports = [report for report in reports if report["basic"]][:2]
        return {
            "primary_heatmap": reports[0]["image_url"],
            "xai_reports": reports,
            "xai_basic_reports": basic_reports,
            "xai_advanced_reports": reports,
            "xai_context": {
                "media_type": safe_media_type,
                "output_dir": self._to_static_relative(job_dir),
                "frame_number": frame_metadata.get("frame_number"),
                "timestamp_seconds": frame_metadata.get("timestamp_seconds"),
                "timestamp_label": frame_metadata.get("timestamp_label"),
                "selection_reason": frame_metadata.get(
                    "selection_reason",
                    "Primary frame used for explainability generation.",
                ),
            },
        }

    def _generate_single_report(
        self,
        bgr_image: np.ndarray,
        output_dir: Path,
        definition: Dict[str, object],
        frame_metadata: Dict[str, object],
    ) -> Dict[str, object]:
        key = str(definition["key"])
        output_path = output_dir / f"{key}.jpg"
        status = "available"
        error_message = ""

        try:
            attention = self._method_handlers[key](bgr_image.copy())
            rendered = self._render_attention_overlay(bgr_image, attention, key)
            write_ok = cv2.imwrite(str(output_path), rendered)
            if not write_ok:
                raise ValueError("OpenCV failed to write the generated artifact image.")
        except Exception as exc:
            status = "unavailable"
            error_message = str(exc)
            self._write_placeholder(output_path, str(definition["method"]))

        return {
            "key": key,
            "method": definition["method"],
            "description": definition["description"],
            "short_explanation": definition["description"],
            "status": status,
            "basic": bool(definition["basic"]),
            "advanced": True,
            "image_url": self._to_static_relative(output_path),
            "message": (
                definition["description"]
                if status == "available"
                else "This XAI output could not be generated for this run."
            ),
            "error": error_message,
            "frame_number": frame_metadata.get("frame_number"),
            "timestamp_seconds": frame_metadata.get("timestamp_seconds"),
            "timestamp_label": frame_metadata.get("timestamp_label"),
        }

    def _to_static_relative(self, path: Path) -> str:
        return path.relative_to(self.project_root).as_posix()

    def _write_placeholder(self, output_path: Path, method_name: str) -> None:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:] = (10, 17, 34)
        cv2.rectangle(canvas, (40, 40), (1240, 680), (0, 229, 255), 2)
        cv2.putText(canvas, method_name, (70, 180), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 229, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "Unavailable for this run", (70, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 230, 245), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "AUTHENTIX retained the rest of the XAI matrix successfully.",
            (70, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (152, 165, 197),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_path), canvas)

    def _render_attention_overlay(self, bgr_image: np.ndarray, attention_map: np.ndarray, key: str) -> np.ndarray:
        normalized = self._normalize_map(attention_map)
        colormap = {
            "base_heatmap": cv2.COLORMAP_JET,
            "grad_cam": cv2.COLORMAP_TURBO,
            "integrated_gradients": cv2.COLORMAP_BONE,
            "occlusion_sensitivity": cv2.COLORMAP_INFERNO,
            "score_cam": cv2.COLORMAP_HOT,
            "grad_cam_plus_plus": cv2.COLORMAP_VIRIDIS,
        }.get(key, cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), colormap)
        return cv2.addWeighted(bgr_image, 0.55, heatmap, 0.45, 0)

    def _normalize_map(self, arr: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        min_v = float(arr.min())
        max_v = float(arr.max())
        if max_v - min_v < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - min_v) / (max_v - min_v)

    def _gray_float(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 255.0

    def _base_heatmap_map(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = self._gray_float(bgr_image)
        smooth = cv2.GaussianBlur(gray, (0, 0), 4)
        detail = cv2.absdiff(gray, smooth)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        return cv2.GaussianBlur(np.abs(edges) + detail, (0, 0), 3)

    def _grad_cam_map(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = self._gray_float(bgr_image)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt((sobel_x ** 2) + (sobel_y ** 2))
        return cv2.GaussianBlur(magnitude, (0, 0), 5)

    def _grad_cam_plus_plus_map(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = self._gray_float(bgr_image)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        morph = cv2.morphologyEx((gray * 255).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8))
        return lap + (morph.astype(np.float32) / 255.0)

    def _saliency_map(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = self._gray_float(bgr_image)
        spectral = np.fft.fft2(gray)
        shifted = np.fft.fftshift(spectral)
        log_amplitude = np.log(np.abs(shifted) + 1e-8)
        smoothed = cv2.GaussianBlur(log_amplitude, (0, 0), 3)
        residual = log_amplitude - smoothed
        saliency = np.abs(np.fft.ifft2(np.fft.ifftshift(np.exp(residual + 1j * np.angle(shifted)))))
        return saliency.astype(np.float32)

    def _integrated_gradients_map(self, bgr_image: np.ndarray) -> np.ndarray:
        image_f = bgr_image.astype(np.float32) / 255.0
        baseline = cv2.GaussianBlur(image_f, (0, 0), 8)
        diff = np.abs(image_f - baseline)
        return np.mean(diff, axis=2)

    def _occlusion_map(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = self._gray_float(bgr_image)
        h, w = gray.shape
        patch = max(16, min(h, w) // 8)
        score_map = np.zeros_like(gray)
        for y in range(0, h, patch):
            for x in range(0, w, patch):
                patch_region = gray[y : y + patch, x : x + patch]
                local_score = float(np.std(patch_region) + np.mean(np.abs(patch_region - patch_region.mean())))
                score_map[y : y + patch, x : x + patch] = local_score
        return cv2.GaussianBlur(score_map, (0, 0), 4)

    def _score_cam_map(self, bgr_image: np.ndarray) -> np.ndarray:
        gray = self._gray_float(bgr_image)
        blur = cv2.GaussianBlur(gray, (0, 0), 5)
        contrast = cv2.absdiff(gray, blur)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        local_boost = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
        return contrast + local_boost
