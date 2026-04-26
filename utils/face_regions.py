from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from web_backend.face_alignment import get_face_alignment_service


REGION_LAYOUT = [
    ("top_left", "Top Left"),
    ("top_centre", "Top Centre"),
    ("top_right", "Top Right"),
    ("middle_left", "Middle Left"),
    ("middle_centre", "Middle Centre"),
    ("middle_right", "Middle Right"),
    ("bottom_left", "Bottom Left"),
    ("bottom_centre", "Bottom Centre"),
    ("bottom_right", "Bottom Right"),
]

REGION_IMPORTANCE = {
    "top_left": 1.04,
    "top_centre": 1.16,
    "top_right": 1.04,
    "middle_left": 1.25,
    "middle_centre": 1.34,
    "middle_right": 1.25,
    "bottom_left": 1.18,
    "bottom_centre": 1.38,
    "bottom_right": 1.18,
}

REGION_FOCUS = {
    "top_left": "eyebrow edge and forehead boundary",
    "top_centre": "forehead blend and brow ridge",
    "top_right": "eyebrow edge and forehead boundary",
    "middle_left": "left eye and cheek texture",
    "middle_centre": "eyes, nose bridge, and central skin continuity",
    "middle_right": "right eye and cheek texture",
    "bottom_left": "jawline and cheek fusion",
    "bottom_centre": "lips, teeth, and mouth boundary",
    "bottom_right": "jawline and cheek fusion",
}


@dataclass
class FaceDetectionResult:
    face_crop: np.ndarray
    aligned_face: np.ndarray
    bbox: Tuple[int, int, int, int]
    eyes: List[Tuple[int, int]]
    anchor_points: Dict[str, Tuple[int, int]]
    detected: bool
    detector: str
    fallback_reason: str
    landmarks: List[Tuple[int, int]]


_FACE_MESH = None
_FACE_MESH_ATTEMPTED = False


def _get_face_mesh():
    global _FACE_MESH
    global _FACE_MESH_ATTEMPTED

    if _FACE_MESH_ATTEMPTED:
        return _FACE_MESH

    _FACE_MESH_ATTEMPTED = True
    try:
        import mediapipe as mp

        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
        )
    except Exception:
        _FACE_MESH = None

    return _FACE_MESH


class FaceRegionAnalyzer:
    def __init__(self, project_root: Optional[Path] = None, output_subdir: str = "static/outputs/faces") -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.output_root = self.project_root / output_subdir
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.alignment_service = get_face_alignment_service()

        cascades_dir = Path(cv2.data.haarcascades)
        self.face_detector = cv2.CascadeClassifier(str(cascades_dir / "haarcascade_frontalface_default.xml"))
        self.eye_detector = cv2.CascadeClassifier(str(cascades_dir / "haarcascade_eye.xml"))

    def analyze(
        self,
        bgr_image: np.ndarray,
        job_id: str,
        media_type: str,
        frame_metadata: Optional[Dict[str, object]] = None,
        persist: bool = True,
    ) -> Dict[str, object]:
        frame_metadata = frame_metadata or {}
        detection = self._detect_and_align_face(bgr_image)
        regions = self._split_face_into_grid(detection.aligned_face)
        landmark_integrity = self._compute_landmark_integrity(
            detection.aligned_face,
            detection.landmarks,
            detection.anchor_points,
            detection.eyes,
        )
        face_quality = self._compute_face_quality(detection.aligned_face, detection.detected)
        scored_regions = self._score_regions(regions, landmark_integrity, face_quality)

        aligned_face_url = ""
        grid_overlay_url = ""
        landmark_overlay_url = ""

        if persist:
            media_dir = self.output_root / job_id / media_type.lower()
            regions_dir = media_dir / "regions"
            regions_dir.mkdir(parents=True, exist_ok=True)

            aligned_face_path = media_dir / "aligned_face.jpg"
            grid_overlay_path = media_dir / "grid_overlay.jpg"
            landmark_overlay_path = media_dir / "landmark_overlay.jpg"
            cv2.imwrite(str(aligned_face_path), detection.aligned_face)
            cv2.imwrite(str(grid_overlay_path), self._draw_grid_overlay(detection.aligned_face.copy(), scored_regions))
            cv2.imwrite(
                str(landmark_overlay_path),
                self._draw_landmark_overlay(
                    detection.aligned_face.copy(),
                    detection.landmarks,
                    detection.anchor_points,
                    landmark_integrity,
                    detection.detected,
                ),
            )
            aligned_face_url = self._to_relative(aligned_face_path)
            grid_overlay_url = self._to_relative(grid_overlay_path)
            landmark_overlay_url = self._to_relative(landmark_overlay_path)

            for region in scored_regions:
                region_path = regions_dir / f"{region['key']}.jpg"
                cv2.imwrite(str(region_path), region["image"])
                region["image_url"] = self._to_relative(region_path)
                region.pop("image", None)
        else:
            for region in scored_regions:
                region["image_url"] = ""
                region.pop("image", None)

        ranked_regions = sorted(scored_regions, key=lambda item: item["score"], reverse=True)
        total_importance = max(sum(item["importance"] for item in ranked_regions), 1.0)
        face_anomaly_score = round(
            float(sum(item["score"] * item["importance"] for item in ranked_regions) / total_importance), 2
        )
        face_authenticity_score = round(100.0 - face_anomaly_score, 2)

        return {
            "face_detected": detection.detected,
            "detector": detection.detector,
            "fallback_reason": detection.fallback_reason,
            "face_bbox": {
                "x": int(detection.bbox[0]),
                "y": int(detection.bbox[1]),
                "w": int(detection.bbox[2]),
                "h": int(detection.bbox[3]),
            },
            "aligned_face_url": aligned_face_url,
            "grid_overlay_url": grid_overlay_url,
            "landmark_overlay_url": landmark_overlay_url,
            "region_grid": ranked_regions,
            "region_grid_ordered": self._ordered_regions(ranked_regions),
            "top_regions": ranked_regions[:3],
            "face_score": face_anomaly_score,
            "face_authenticity_score": face_authenticity_score,
            "face_quality": face_quality,
            "landmark_integrity": landmark_integrity,
            "selection_context": {
                "frame_number": frame_metadata.get("frame_number"),
                "timestamp_label": frame_metadata.get("timestamp_label"),
                "media_type": media_type.lower(),
            },
            "summary": self._build_summary(ranked_regions, detection.detected, detection.fallback_reason, landmark_integrity),
        }

    def extract_region_tensor_stack(self, rgb_face: np.ndarray, output_size: int = 128) -> np.ndarray:
        aligned = cv2.resize(rgb_face, (output_size * 3, output_size * 3))
        regions = self._split_face_into_grid(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        region_tensors = []
        for region in regions:
            crop = cv2.resize(region["image"], (output_size, output_size))
            region_tensors.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return np.stack(region_tensors, axis=0)

    def _detect_and_align_face(self, bgr_image: np.ndarray) -> FaceDetectionResult:
        alignment = self.alignment_service.align_primary_face(bgr_image)
        x0, y0, w, h = alignment.bbox
        x1 = min(bgr_image.shape[1], x0 + w)
        y1 = min(bgr_image.shape[0], y0 + h)
        face_crop = bgr_image[y0:y1, x0:x1].copy()
        if face_crop.size == 0:
            face_crop = alignment.aligned_face.copy()

        aligned_face = cv2.resize(alignment.aligned_face, (384, 384))
        landmarks = self._extract_landmarks(aligned_face)
        scaled_landmarks_5 = self._project_landmarks_to_aligned(alignment.landmarks_5, alignment.bbox, aligned_face.shape[1], aligned_face.shape[0])
        eyes = scaled_landmarks_5[:2] if len(scaled_landmarks_5) >= 2 else []
        anchor_points = self._build_anchor_points(aligned_face, landmarks, scaled_landmarks_5)

        detector_label = alignment.detector
        if landmarks:
            detector_label = f"{detector_label} + mediapipe-landmarks"
        elif anchor_points:
            detector_label = f"{detector_label} + guided-anchors"

        fallback_reason = alignment.fallback_reason
        if not fallback_reason and not alignment.detected:
            fallback_reason = "Guided facial crop used for analysis."

        return FaceDetectionResult(
            face_crop=face_crop,
            aligned_face=aligned_face,
            bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
            eyes=eyes,
            anchor_points=anchor_points,
            detected=alignment.detected,
            detector=detector_label,
            fallback_reason=fallback_reason,
            landmarks=landmarks,
        )

    def _align_by_eyes(self, face_crop: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18))
        if len(eyes) < 2:
            return face_crop, []

        selected = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
        selected = sorted(selected, key=lambda eye: eye[0])
        eye_centres = [(int(ex + ew / 2), int(ey + eh / 2)) for ex, ey, ew, eh in selected]

        left_eye, right_eye = eye_centres
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        centre = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rotation = cv2.getRotationMatrix2D(centre, angle, 1.0)
        aligned = cv2.warpAffine(
            face_crop,
            rotation,
            (face_crop.shape[1], face_crop.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        rotated_eyes = [self._apply_affine(rotation, point) for point in eye_centres]
        return aligned, rotated_eyes

    def _extract_landmarks(self, aligned_face: np.ndarray) -> List[Tuple[int, int]]:
        face_mesh = _get_face_mesh()
        if face_mesh is None:
            return []

        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []

        h, w = aligned_face.shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]

    def _compute_landmark_integrity(
        self,
        aligned_face: np.ndarray,
        landmarks: List[Tuple[int, int]],
        anchor_points: Dict[str, Tuple[int, int]],
        eyes: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, object]:
        if not landmarks:
            gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            mirrored = cv2.flip(gray[:, gray.shape[1] // 2 :], 1)
            left_half = gray[:, : mirrored.shape[1]]
            face_symmetry = float(max(0.0, min(100.0, ssim(left_half, mirrored, data_range=255) * 100.0)))
            boundary = np.hstack([gray[:, :24], gray[:, -24:]])
            contour_consistency = float(max(0.0, min(100.0, 100.0 - (cv2.Laplacian(boundary, cv2.CV_32F).var() / 40.0))))

            eye_alignment = 62.0
            mouth_geometry = 60.0
            if eyes and len(eyes) >= 2:
                left_eye, right_eye = eyes[:2]
                eye_angle = abs(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
                eye_alignment = max(0.0, 100.0 - (eye_angle * 8.0))
            elif anchor_points.get("left_eye") and anchor_points.get("right_eye"):
                left_eye = anchor_points["left_eye"]
                right_eye = anchor_points["right_eye"]
                eye_angle = abs(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
                eye_alignment = max(0.0, 100.0 - (eye_angle * 8.0))

            if anchor_points.get("mouth_left") and anchor_points.get("mouth_right") and anchor_points.get("mouth_top") and anchor_points.get("mouth_bottom"):
                mouth_left = anchor_points["mouth_left"]
                mouth_right = anchor_points["mouth_right"]
                mouth_top = anchor_points["mouth_top"]
                mouth_bottom = anchor_points["mouth_bottom"]
                mouth_width = max(np.hypot(mouth_right[0] - mouth_left[0], mouth_right[1] - mouth_left[1]), 1.0)
                mouth_height = max(np.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1]), 1.0)
                mouth_ratio = mouth_width / mouth_height
                mouth_geometry = max(0.0, 100.0 - abs(mouth_ratio - 3.2) * 18.0)

            return {
                "available": True,
                "fallback": True,
                "eye_alignment": round(eye_alignment, 2),
                "mouth_geometry": round(mouth_geometry, 2),
                "face_symmetry": round(face_symmetry, 2),
                "contour_consistency": round(contour_consistency, 2),
                "summary": "Guided landmark anchors remained active for eye, mouth, symmetry, and contour checks.",
            }

        def avg_point(indices: List[int]) -> Tuple[float, float]:
            pts = np.array([landmarks[idx] for idx in indices if idx < len(landmarks)], dtype=np.float32)
            if len(pts) == 0:
                return (0.0, 0.0)
            return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

        left_eye = avg_point([33, 133, 159, 145])
        right_eye = avg_point([362, 263, 386, 374])
        mouth_left = avg_point([61, 78, 191])
        mouth_right = avg_point([291, 308, 415])
        mouth_top = avg_point([13, 0, 37])
        mouth_bottom = avg_point([14, 17, 84])

        eye_angle = abs(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
        eye_alignment = max(0.0, 100.0 - (eye_angle * 8.0))

        mouth_width = max(np.hypot(mouth_right[0] - mouth_left[0], mouth_right[1] - mouth_left[1]), 1.0)
        mouth_height = max(np.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1]), 1.0)
        mouth_ratio = mouth_width / mouth_height
        mouth_geometry = max(0.0, 100.0 - abs(mouth_ratio - 3.2) * 18.0)

        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        mirrored = cv2.flip(gray[:, gray.shape[1] // 2 :], 1)
        left_half = gray[:, : mirrored.shape[1]]
        face_symmetry = float(max(0.0, min(100.0, ssim(left_half, mirrored, data_range=255) * 100.0)))

        boundary = np.hstack([gray[:, :24], gray[:, -24:]])
        contour_consistency = float(max(0.0, min(100.0, 100.0 - (cv2.Laplacian(boundary, cv2.CV_32F).var() / 40.0))))

        summary = (
            f"Eye alignment {eye_alignment:.0f}, mouth geometry {mouth_geometry:.0f}, "
            f"symmetry {face_symmetry:.0f}, contour consistency {contour_consistency:.0f}."
        )
        return {
            "available": True,
            "fallback": False,
            "eye_alignment": round(eye_alignment, 2),
            "mouth_geometry": round(mouth_geometry, 2),
            "face_symmetry": round(face_symmetry, 2),
            "contour_consistency": round(contour_consistency, 2),
            "summary": summary,
        }

    def _compute_face_quality(self, aligned_face: np.ndarray, detected: bool) -> Dict[str, object]:
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        sharpness_score = max(0.0, min(100.0, blur / 3.5))
        exposure_score = max(0.0, min(100.0, 100.0 - abs(brightness - 132.0) * 0.9))
        contrast_score = max(0.0, min(100.0, contrast * 2.4))
        quality_score = round((0.45 * sharpness_score) + (0.25 * exposure_score) + (0.30 * contrast_score), 2)

        note = "Good facial evidence quality."
        if quality_score < 45:
            note = "Lower image quality reduces how strongly region evidence should be trusted."
        elif not detected:
            note = "Guided facial crop used; treat region evidence as review support."

        return {
            "quality_score": quality_score,
            "sharpness": round(sharpness_score, 2),
            "exposure": round(exposure_score, 2),
            "contrast": round(contrast_score, 2),
            "note": note,
        }

    def _split_face_into_grid(self, aligned_face: np.ndarray) -> List[Dict[str, object]]:
        h, w = aligned_face.shape[:2]
        row_bounds = np.linspace(0, h, 4, dtype=int)
        col_bounds = np.linspace(0, w, 4, dtype=int)
        regions = []
        idx = 0
        for row in range(3):
            for col in range(3):
                key, label = REGION_LAYOUT[idx]
                y0, y1 = row_bounds[row], row_bounds[row + 1]
                x0, x1 = col_bounds[col], col_bounds[col + 1]
                regions.append(
                    {
                        "key": key,
                        "label": label,
                        "row": row,
                        "col": col,
                        "importance": REGION_IMPORTANCE[key],
                        "focus": REGION_FOCUS[key],
                        "image": aligned_face[y0:y1, x0:x1].copy(),
                        "box": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                    }
                )
                idx += 1
        return regions

    def _score_regions(
        self,
        regions: List[Dict[str, object]],
        landmark_integrity: Dict[str, object],
        face_quality: Dict[str, object],
    ) -> List[Dict[str, object]]:
        scored = []
        for region in regions:
            metrics = self._region_metrics(region["image"])
            raw_score = self._metrics_to_score(metrics)
            guided_score = raw_score

            if region["key"].startswith("middle"):
                guided_score += max(0.0, (70.0 - landmark_integrity["eye_alignment"]) * 0.08)
                guided_score += max(0.0, (70.0 - landmark_integrity["face_symmetry"]) * 0.06)
            if region["key"] == "bottom_centre":
                guided_score += max(0.0, (72.0 - landmark_integrity["mouth_geometry"]) * 0.12)
            if region["key"] in {"bottom_left", "bottom_right"}:
                guided_score += max(0.0, (70.0 - landmark_integrity["contour_consistency"]) * 0.08)
            if region["key"] == "top_centre":
                guided_score += max(0.0, (68.0 - landmark_integrity["contour_consistency"]) * 0.05)

            quality_factor = 0.82 if face_quality["quality_score"] < 45 else 1.0
            weighted = min(100.0, round(guided_score * region["importance"] * quality_factor, 2))

            scored.append(
                {
                    **region,
                    "score": weighted,
                    "base_score": round(raw_score, 2),
                    "status": self._status_from_score(weighted),
                    "explanation": self._region_explanation(region["key"], weighted, metrics),
                    "metrics": {k: round(v, 5) for k, v in metrics.items()},
                }
            )

        self._apply_symmetry_consensus(scored)
        return scored

    def _region_metrics(self, region_img: np.ndarray) -> Dict[str, float]:
        image = region_img.astype(np.float32) / 255.0
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = float(np.mean(np.sqrt((sobel_x ** 2) + (sobel_y ** 2))))
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual_noise = float(np.std(gray - blurred))
        diff_x = np.abs(np.diff(gray, axis=1))
        diff_y = np.abs(np.diff(gray, axis=0))
        blocking = float(
            ((np.mean(diff_x[:, 7::8]) if diff_x.shape[1] > 8 else 0.0) + (np.mean(diff_y[7::8, :]) if diff_y.shape[0] > 8 else 0.0))
            / 2.0
        )
        b, g, r = cv2.split(image)
        color_mismatch = float((np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0)
        lap_var = float(cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F).var() / 10000.0)
        return {
            "edge_energy": edge_energy,
            "residual_noise": residual_noise,
            "blocking": blocking,
            "color_mismatch": color_mismatch,
            "laplacian_variance": lap_var,
        }

    def _metrics_to_score(self, metrics: Dict[str, float]) -> float:
        edge_component = 1.0 - self._normalize(metrics["edge_energy"], 0.03, 0.22)
        noise_component = self._normalize(metrics["residual_noise"], 0.02, 0.18)
        block_component = self._normalize(metrics["blocking"], 0.01, 0.18)
        color_component = self._normalize(metrics["color_mismatch"], 0.02, 0.25)
        focus_component = self._normalize(metrics["laplacian_variance"], 0.01, 0.22)
        score = (
            (0.28 * edge_component)
            + (0.22 * noise_component)
            + (0.18 * block_component)
            + (0.14 * color_component)
            + (0.18 * focus_component)
        )
        return float(max(0.0, min(1.0, score)) * 100.0)

    def _apply_symmetry_consensus(self, scored_regions: List[Dict[str, object]]) -> None:
        pairs = [("top_left", "top_right"), ("middle_left", "middle_right"), ("bottom_left", "bottom_right")]
        by_key = {region["key"]: region for region in scored_regions}
        for left_key, right_key in pairs:
            left = by_key[left_key]
            right = by_key[right_key]
            gap = abs(left["base_score"] - right["base_score"])
            if gap > 9:
                left["score"] = min(100.0, round(left["score"] + gap * 0.3, 2))
                right["score"] = min(100.0, round(right["score"] + gap * 0.3, 2))
                left["status"] = self._status_from_score(left["score"])
                right["status"] = self._status_from_score(right["score"])
                if "asymmetry" not in left["explanation"].lower():
                    left["explanation"] += " Eye-face symmetry needs review."
                if "asymmetry" not in right["explanation"].lower():
                    right["explanation"] += " Eye-face symmetry needs review."

    def _region_explanation(self, key: str, score: float, metrics: Dict[str, float]) -> str:
        if key == "top_left":
            return "Mild eyebrow-edge inconsistency detected." if score >= 45 else "Forehead edge appears stable."
        if key == "top_centre":
            return "Forehead boundary blending warrants review." if score >= 45 else "Forehead transition looks natural."
        if key == "top_right":
            return "Subtle brow-texture mismatch noted." if score >= 45 else "Right brow region looks stable."
        if key == "middle_left":
            return "Eye-region asymmetry slightly elevated." if score >= 45 else "Left eye and cheek texture look natural."
        if key == "middle_centre":
            return "Nose bridge and central texture show review-level drift." if score >= 45 else "Central facial structure remains coherent."
        if key == "middle_right":
            return "Review-level cheek texture variation detected." if score >= 45 else "Natural cheek texture, low anomaly."
        if key == "bottom_left":
            return "Left jaw contour needs review." if score >= 45 else "Jawline transition appears stable."
        if key == "bottom_centre":
            return "Possible mouth-boundary blending irregularity." if score >= 45 else "Lip boundary looks mostly stable."
        return "Right jaw fusion signal slightly elevated." if score >= 45 else "Lower-right contour looks stable."

    def _status_from_score(self, score: float) -> str:
        if score >= 70:
            return "Suspicious"
        if score >= 45:
            return "Review"
        return "Stable"

    def _draw_grid_overlay(self, aligned_face: np.ndarray, scored_regions: List[Dict[str, object]]) -> np.ndarray:
        overlay = aligned_face.copy()
        top_keys = {region["key"] for region in sorted(scored_regions, key=lambda item: item["score"], reverse=True)[:3]}

        for region in scored_regions:
            x, y, rw, rh = region["box"]
            color = (54, 227, 154)
            fill = (24, 96, 58)
            if region["status"] == "Review":
                color = (0, 180, 255)
                fill = (24, 84, 116)
            if region["status"] == "Suspicious":
                color = (0, 92, 255)
                fill = (30, 56, 148)
            thickness = 3 if region["key"] in top_keys else 2
            cell = overlay[y : y + rh, x : x + rw]
            tint = np.full_like(cell, fill)
            overlay[y : y + rh, x : x + rw] = cv2.addWeighted(cell, 0.82, tint, 0.18, 0.0)
            cv2.rectangle(overlay, (x, y), (x + rw, y + rh), color, thickness)
            cv2.rectangle(overlay, (x + 8, y + 8), (x + min(rw - 8, 168), y + 36), (8, 12, 24), -1)
            cv2.putText(
                overlay,
                f"{region['label']}  {region['score']:.0f}",
                (x + 8, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(overlay, (12, overlay.shape[0] - 42), (overlay.shape[1] - 12, overlay.shape[0] - 12), (8, 12, 24), -1)
        cv2.putText(
            overlay,
            "3x3 facial evidence grid",
            (24, overlay.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 229, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _draw_landmark_overlay(
        self,
        aligned_face: np.ndarray,
        landmarks: List[Tuple[int, int]],
        anchor_points: Dict[str, Tuple[int, int]],
        landmark_integrity: Dict[str, object],
        detected: bool,
    ) -> np.ndarray:
        overlay = aligned_face.copy()
        if landmarks:
            for point in landmarks[::10]:
                cv2.circle(overlay, point, 1, (0, 229, 255), -1)

        self._draw_anchor_guides(overlay, anchor_points)
        cv2.putText(
            overlay,
            f"Sym {landmark_integrity['face_symmetry']:.0f} | Eye {landmark_integrity['eye_alignment']:.0f}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 229, 255),
            2,
            cv2.LINE_AA,
        )
        status_line = "Detected face landmarks" if detected and landmarks else "Guided landmark overlay"
        cv2.putText(
            overlay,
            status_line,
            (16, overlay.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (220, 230, 245),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _ordered_regions(self, ranked_regions: List[Dict[str, object]]) -> List[Dict[str, object]]:
        region_map = {region["key"]: region for region in ranked_regions}
        return [region_map[key] for key, _ in REGION_LAYOUT]

    def _build_summary(
        self,
        ranked_regions: List[Dict[str, object]],
        detected: bool,
        fallback_reason: str,
        landmark_integrity: Dict[str, object],
    ) -> str:
        top_regions = ", ".join(f"{region['label']} ({region['score']:.0f})" for region in ranked_regions[:3])
        prefix = "Aligned face evidence captured." if detected else (fallback_reason or "Guided facial crop used.")
        return f"{prefix} Top review zones: {top_regions}. {landmark_integrity['summary']}"

    def _build_anchor_points(
        self,
        aligned_face: np.ndarray,
        landmarks: List[Tuple[int, int]],
        landmarks_5: List[Tuple[int, int]],
    ) -> Dict[str, Tuple[int, int]]:
        h, w = aligned_face.shape[:2]

        def avg(indices: List[int]) -> Tuple[int, int]:
            pts = np.array([landmarks[idx] for idx in indices if idx < len(landmarks)], dtype=np.float32)
            if len(pts) == 0:
                return (0, 0)
            return int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

        if landmarks:
            return {
                "left_eye": avg([33, 133, 159, 145]),
                "right_eye": avg([362, 263, 386, 374]),
                "nose_tip": avg([1, 4, 5]),
                "mouth_left": avg([61, 78, 191]),
                "mouth_right": avg([291, 308, 415]),
                "mouth_top": avg([13, 0, 37]),
                "mouth_bottom": avg([14, 17, 84]),
                "left_cheek": avg([234, 93]),
                "right_cheek": avg([454, 323]),
                "chin": avg([152]),
                "forehead": avg([10]),
            }

        if len(landmarks_5) >= 5:
            left_eye = tuple(map(int, landmarks_5[0]))
            right_eye = tuple(map(int, landmarks_5[1]))
            nose_tip = tuple(map(int, landmarks_5[2]))
            mouth_left = tuple(map(int, landmarks_5[3]))
            mouth_right = tuple(map(int, landmarks_5[4]))
            mouth_mid_x = int((mouth_left[0] + mouth_right[0]) / 2)
            mouth_mid_y = int((mouth_left[1] + mouth_right[1]) / 2)
            return {
                "left_eye": left_eye,
                "right_eye": right_eye,
                "nose_tip": nose_tip,
                "mouth_left": mouth_left,
                "mouth_right": mouth_right,
                "mouth_top": (mouth_mid_x, max(0, mouth_mid_y - 10)),
                "mouth_bottom": (mouth_mid_x, min(h - 1, mouth_mid_y + 10)),
                "left_cheek": (max(0, left_eye[0] - 34), min(h - 1, nose_tip[1] + 18)),
                "right_cheek": (min(w - 1, right_eye[0] + 34), min(h - 1, nose_tip[1] + 18)),
                "chin": (mouth_mid_x, min(h - 1, mouth_mid_y + 54)),
                "forehead": (int((left_eye[0] + right_eye[0]) / 2), max(0, int(min(left_eye[1], right_eye[1]) - 58))),
            }

        return {
            "left_eye": (int(w * 0.33), int(h * 0.38)),
            "right_eye": (int(w * 0.67), int(h * 0.38)),
            "nose_tip": (int(w * 0.50), int(h * 0.54)),
            "mouth_left": (int(w * 0.40), int(h * 0.70)),
            "mouth_right": (int(w * 0.60), int(h * 0.70)),
            "mouth_top": (int(w * 0.50), int(h * 0.67)),
            "mouth_bottom": (int(w * 0.50), int(h * 0.74)),
            "left_cheek": (int(w * 0.25), int(h * 0.58)),
            "right_cheek": (int(w * 0.75), int(h * 0.58)),
            "chin": (int(w * 0.50), int(h * 0.86)),
            "forehead": (int(w * 0.50), int(h * 0.20)),
        }

    def _draw_anchor_guides(self, canvas: np.ndarray, anchor_points: Dict[str, Tuple[int, int]]) -> None:
        cyan = (0, 229, 255)
        white = (220, 230, 245)
        left_eye = anchor_points.get("left_eye")
        right_eye = anchor_points.get("right_eye")
        nose_tip = anchor_points.get("nose_tip")
        mouth_left = anchor_points.get("mouth_left")
        mouth_right = anchor_points.get("mouth_right")
        mouth_top = anchor_points.get("mouth_top")
        mouth_bottom = anchor_points.get("mouth_bottom")
        forehead = anchor_points.get("forehead")
        chin = anchor_points.get("chin")

        for point in anchor_points.values():
            cv2.circle(canvas, point, 3, cyan, -1)

        if left_eye and right_eye:
            cv2.line(canvas, left_eye, right_eye, cyan, 2)
        if forehead and nose_tip:
            cv2.line(canvas, forehead, nose_tip, white, 1)
        if left_eye and nose_tip:
            cv2.line(canvas, left_eye, nose_tip, white, 1)
        if right_eye and nose_tip:
            cv2.line(canvas, right_eye, nose_tip, white, 1)
        if mouth_left and mouth_right:
            cv2.line(canvas, mouth_left, mouth_right, cyan, 2)
        if mouth_top and mouth_bottom:
            cv2.line(canvas, mouth_top, mouth_bottom, white, 1)
        if nose_tip and mouth_top:
            cv2.line(canvas, nose_tip, mouth_top, white, 1)
        if mouth_bottom and chin:
            cv2.line(canvas, mouth_bottom, chin, white, 1)

    def _project_landmarks_to_aligned(
        self,
        landmarks_5: List[Tuple[int, int]],
        bbox: Tuple[int, int, int, int],
        out_w: int,
        out_h: int,
    ) -> List[Tuple[int, int]]:
        if not landmarks_5:
            return []
        x0, y0, bw, bh = bbox
        bw = max(1, bw)
        bh = max(1, bh)
        projected = []
        for px, py in landmarks_5:
            proj_x = int(np.clip(((float(px) - x0) / bw) * out_w, 0, out_w - 1))
            proj_y = int(np.clip(((float(py) - y0) / bh) * out_h, 0, out_h - 1))
            projected.append((proj_x, proj_y))
        return projected

    def _apply_affine(self, matrix: np.ndarray, point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        new_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        new_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        return int(new_x), int(new_y)

    def _to_relative(self, path: Path) -> str:
        return path.relative_to(self.project_root).as_posix()

    def _normalize(self, value: float, low: float, high: float) -> float:
        clipped = min(max(value, low), high)
        if high - low == 0:
            return 0.0
        return float((clipped - low) / (high - low))
