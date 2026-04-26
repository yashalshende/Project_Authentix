from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from core_engine.config import ModelConfig
from web_backend.face_alignment import FaceAlignmentResult, FaceAlignmentService, get_face_alignment_service
from web_backend.identity_embedding import IdentityEmbeddingAnalyzer


class FaceSwapDetector:
    def __init__(self, project_root: Optional[Path] = None, output_subdir: str = "static/outputs/faceswap") -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.output_root = self.project_root / output_subdir
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.alignment_service: FaceAlignmentService = get_face_alignment_service()
        self.identity_analyzer = IdentityEmbeddingAnalyzer()

    def analyze(
        self,
        bgr_image: np.ndarray,
        job_id: str,
        media_type: str,
        face_forensics: Optional[Dict[str, object]] = None,
        frame_metadata: Optional[Dict[str, object]] = None,
        persist: bool = True,
    ) -> Dict[str, object]:
        face_forensics = face_forensics or {}
        frame_metadata = frame_metadata or {}
        alignment = self.alignment_service.align_primary_face(bgr_image)

        media_key = media_type.lower()
        media_dir = self.output_root / job_id / media_key
        if persist:
            media_dir.mkdir(parents=True, exist_ok=True)

        if not alignment.usable:
            artifacts = self._persist_unavailable_artifacts(media_dir, alignment, persist)
            return self._unavailable_result(
                reason=alignment.fallback_reason,
                frame_metadata=frame_metadata,
                artifacts=artifacts,
            )

        aligned_face = alignment.aligned_face
        identity = self.identity_analyzer.analyze(aligned_face, alignment)
        boundary = self._boundary_signal(aligned_face)
        landmark = self._landmark_signal(face_forensics, alignment)
        texture = self._texture_signal(aligned_face)
        region = self._region_signal(face_forensics)
        quality_score = float(face_forensics.get("face_quality", {}).get("quality_score", 60.0))

        score = (
            identity["identity_inconsistency_score"] * 0.34
            + boundary["boundary_anomaly_score"] * 0.24
            + landmark["landmark_mismatch_score"] * 0.16
            + texture["texture_mismatch_score"] * 0.14
            + region["region_consensus_score"] * 0.12
        )

        quality_note = ""
        if quality_score < float(getattr(ModelConfig, "LOW_FACE_QUALITY_GATE", 40.0)):
            score *= 0.82
            quality_note = "Low face quality reduced the face-swap confidence for this run."

        score = round(float(max(0.0, min(100.0, score))), 2)
        suspicious_regions = self._suspicious_regions(face_forensics)
        explanations = self._explanations(identity, boundary, landmark, texture, suspicious_regions)
        artifacts = self._persist_artifacts(media_dir, aligned_face, identity, boundary, landmark, persist)

        available = True
        summary = (
            "Face-swap evidence is materially elevated across identity and boundary cues."
            if score >= getattr(ModelConfig, "FACE_SWAP_THRESHOLD", 58.0)
            else "Face-swap-specific evidence is limited in this run."
        )
        if quality_note:
            summary = f"{summary} {quality_note}"

        return {
            "available": available,
            "faceswap_score": score,
            "identity_inconsistency_score": round(float(identity["identity_inconsistency_score"]), 2),
            "boundary_anomaly_score": round(float(boundary["boundary_anomaly_score"]), 2),
            "landmark_mismatch_score": round(float(landmark["landmark_mismatch_score"]), 2),
            "texture_mismatch_score": round(float(texture["texture_mismatch_score"]), 2),
            "region_consensus_score": round(float(region["region_consensus_score"]), 2),
            "temporal_identity_drift_score": 0.0,
            "summary": summary,
            "explanations": explanations,
            "suspicious_regions": suspicious_regions,
            "strongest_frame": None,
            "embedding_source": identity["embedding_source"],
            "identity_consistency_score": round(float(100.0 - identity["identity_inconsistency_score"]), 2),
            "face_detected": alignment.detected,
            "detector": alignment.detector,
            "frame_context": {
                "frame_number": frame_metadata.get("frame_number"),
                "timestamp_label": frame_metadata.get("timestamp_label"),
            },
            "artifacts": artifacts,
            "_identity_vector": identity["descriptor_embedding"],
        }

    def aggregate_video_results(self, frame_results: List[Dict[str, object]], top_frame: Dict[str, object]) -> Dict[str, object]:
        if not frame_results:
            return self._unavailable_result(
                reason="No analysable faces were available for face-swap aggregation.",
                frame_metadata={},
                artifacts=self._empty_artifacts(),
            )

        identity_vectors = [item.get("_identity_vector") for item in frame_results]
        temporal_score = self.identity_analyzer.compute_temporal_identity_drift(identity_vectors)
        top_frame_score = float(top_frame.get("faceswap_score", 0.0))

        blended = (
            np.mean([item.get("identity_inconsistency_score", 0.0) for item in frame_results]) * 0.28
            + np.mean([item.get("boundary_anomaly_score", 0.0) for item in frame_results]) * 0.20
            + np.mean([item.get("landmark_mismatch_score", 0.0) for item in frame_results]) * 0.14
            + np.mean([item.get("texture_mismatch_score", 0.0) for item in frame_results]) * 0.12
            + np.mean([item.get("region_consensus_score", 0.0) for item in frame_results]) * 0.12
            + temporal_score * 0.14
        )
        persistence_ratio = sum(1 for item in frame_results if float(item.get("region_consensus_score", 0.0)) >= 36.0 or float(item.get("boundary_anomaly_score", 0.0)) >= 32.0) / max(len(frame_results), 1)
        blended += persistence_ratio * 14.0
        blended = round(float(max(0.0, min(100.0, blended))), 2)

        strongest_frame = {
            "frame_number": top_frame.get("frame_context", {}).get("frame_number"),
            "timestamp": top_frame.get("frame_context", {}).get("timestamp_label"),
            "faceswap_score": round(top_frame_score, 2),
            "summary": top_frame.get("summary", ""),
            "image_url": (top_frame.get("artifacts") or {}).get("aligned_face_url", ""),
        }

        return {
            **{k: v for k, v in top_frame.items() if not k.startswith("_")},
            "faceswap_score": max(blended, top_frame_score),
            "temporal_identity_drift_score": temporal_score,
            "strongest_frame": strongest_frame,
            "summary": (
                "Video-level aggregation points to face identity drift and boundary inconsistency."
                if max(blended, top_frame_score) >= getattr(ModelConfig, "FACE_SWAP_THRESHOLD", 58.0)
                else "Video-level face-swap evidence remains limited across sampled frames."
            ),
        }

    def _boundary_signal(self, aligned_face: np.ndarray) -> Dict[str, object]:
        masks = self._boundary_masks(aligned_face.shape[:2])
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2HSV).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)

        ring_vals = gray[masks["ring"] > 0]
        inner_vals = gray[masks["inner"] > 0]
        forehead_vals = gray[masks["forehead"] > 0]
        jaw_vals = gray[masks["jaw"] > 0]

        seam_contrast = abs(float(np.mean(ring_vals)) - float(np.mean(inner_vals))) / 128.0
        seam_variance = abs(float(np.std(ring_vals)) - float(np.std(inner_vals))) / 64.0
        frequency_gap = abs(float(np.mean(np.abs(laplacian[masks["ring"] > 0]))) - float(np.mean(np.abs(laplacian[masks["inner"] > 0])))) / 48.0
        hue_gap = abs(float(np.mean(hsv[:, :, 0][masks["ring"] > 0])) - float(np.mean(hsv[:, :, 0][masks["inner"] > 0]))) / 90.0
        forehead_gap = abs(float(np.mean(forehead_vals)) - float(np.mean(inner_vals))) / 128.0
        jaw_gap = abs(float(np.mean(jaw_vals)) - float(np.mean(inner_vals))) / 128.0

        raw = (
            min(1.0, seam_contrast) * 24.0
            + min(1.0, seam_variance) * 18.0
            + min(1.0, frequency_gap) * 22.0
            + min(1.0, hue_gap) * 14.0
            + min(1.0, forehead_gap) * 10.0
            + min(1.0, jaw_gap) * 12.0
        )
        return {
            "boundary_anomaly_score": round(float(max(0.0, min(100.0, raw))), 2),
            "masks": masks,
        }

    def _landmark_signal(self, face_forensics: Dict[str, object], alignment: FaceAlignmentResult) -> Dict[str, object]:
        integrity = face_forensics.get("landmark_integrity", {}) or {}
        if integrity.get("available"):
            anomaly = (
                (100.0 - float(integrity.get("eye_alignment", 70.0))) * 0.30
                + (100.0 - float(integrity.get("mouth_geometry", 70.0))) * 0.28
                + (100.0 - float(integrity.get("face_symmetry", 70.0))) * 0.24
                + (100.0 - float(integrity.get("contour_consistency", 70.0))) * 0.18
            )
            anomaly = round(float(max(0.0, min(100.0, anomaly))), 2)
            summary = integrity.get("summary", "")
        else:
            anomaly = 32.0 if alignment.landmarks_5 else 0.0
            summary = "Detailed landmark mesh unavailable; using geometric fallback."
        return {
            "landmark_mismatch_score": anomaly,
            "summary": summary,
        }

    def _texture_signal(self, aligned_face: np.ndarray) -> Dict[str, object]:
        masks = self._boundary_masks(aligned_face.shape[:2])
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2HSV)
        lbp = local_binary_pattern(gray, 16, 2, method="uniform")

        center_hist = self._masked_histogram(lbp, masks["inner"], bins=18, max_value=18)
        ring_hist = self._masked_histogram(lbp, masks["ring"], bins=18, max_value=18)
        lbp_gap = float(np.sum(np.abs(center_hist - ring_hist)))

        center_lap = float(cv2.Laplacian(gray * (masks["inner"] > 0).astype(np.uint8), cv2.CV_32F).var())
        ring_lap = float(cv2.Laplacian(gray * (masks["ring"] > 0).astype(np.uint8), cv2.CV_32F).var())
        lap_gap = abs(center_lap - ring_lap) / 2500.0

        center_sat = float(np.mean(hsv[:, :, 1][masks["inner"] > 0]))
        ring_sat = float(np.mean(hsv[:, :, 1][masks["ring"] > 0]))
        sat_gap = abs(center_sat - ring_sat) / 80.0

        raw = min(1.0, lbp_gap) * 46.0 + min(1.0, lap_gap) * 32.0 + min(1.0, sat_gap) * 22.0
        return {"texture_mismatch_score": round(float(max(0.0, min(100.0, raw))), 2)}

    def _region_signal(self, face_forensics: Dict[str, object]) -> Dict[str, object]:
        regions = face_forensics.get("region_grid_ordered", []) or []
        if not regions:
            return {"region_consensus_score": 0.0}

        emphasis = {
            "middle_left": 1.15,
            "middle_centre": 1.28,
            "middle_right": 1.15,
            "bottom_centre": 1.24,
            "bottom_left": 1.08,
            "bottom_right": 1.08,
            "top_centre": 1.05,
        }
        weighted = 0.0
        total = 0.0
        for region in regions:
            weight = emphasis.get(region.get("key"), 0.82)
            weighted += float(region.get("score", 0.0)) * weight
            total += weight
        score = weighted / max(total, 1.0)
        top_regions = sorted((float(region.get("score", 0.0)) for region in regions), reverse=True)[:3]
        top_mean = float(np.mean(top_regions)) if top_regions else 0.0
        score = (0.58 * score) + (0.42 * top_mean)
        return {"region_consensus_score": round(float(score), 2)}

    def _suspicious_regions(self, face_forensics: Dict[str, object]) -> List[Dict[str, object]]:
        top_regions = face_forensics.get("top_regions", []) or face_forensics.get("region_grid_ordered", []) or []
        suspicious = []
        for region in top_regions[:3]:
            suspicious.append(
                {
                    "label": region.get("label", "Unknown Region"),
                    "score": round(float(region.get("score", 0.0)), 2),
                    "explanation": region.get("explanation", ""),
                }
            )
        return suspicious

    def _explanations(
        self,
        identity: Dict[str, object],
        boundary: Dict[str, object],
        landmark: Dict[str, object],
        texture: Dict[str, object],
        suspicious_regions: List[Dict[str, object]],
    ) -> List[str]:
        explanations = []
        if float(identity["identity_inconsistency_score"]) >= 55:
            explanations.append("Possible face identity mismatch detected.")
        if float(boundary["boundary_anomaly_score"]) >= 55:
            explanations.append("Jawline or cheek boundary blending looks suspicious.")
        if float(landmark["landmark_mismatch_score"]) >= 55:
            explanations.append("Landmark alignment appears inconsistent.")
        if float(texture["texture_mismatch_score"]) >= 55:
            explanations.append("Skin texture continuity looks uneven across central and boundary regions.")
        if suspicious_regions:
            explanations.append(
                "Top flagged regions: "
                + ", ".join(f"{item['label']} ({item['score']:.0f})" for item in suspicious_regions[:3])
                + "."
            )
        if suspicious_regions and float(np.mean([item["score"] for item in suspicious_regions[:2]])) >= 42.0:
            explanations.append("Repeated face-region irregularity persists across the analysed face.")
        if not explanations:
            explanations.append("Face-swap-specific signals remain low in this run.")
        return explanations

    def _persist_artifacts(
        self,
        media_dir: Path,
        aligned_face: np.ndarray,
        identity: Dict[str, object],
        boundary: Dict[str, object],
        landmark: Dict[str, object],
        persist: bool,
    ) -> Dict[str, str]:
        if not persist:
            return self._empty_artifacts()

        aligned_path = media_dir / "aligned_face.jpg"
        boundary_path = media_dir / "boundary_overlay.jpg"
        landmark_path = media_dir / "landmark_overlay.jpg"
        identity_path = media_dir / "identity_overlay.jpg"

        cv2.imwrite(str(aligned_path), aligned_face)
        cv2.imwrite(str(boundary_path), self._draw_boundary_overlay(aligned_face.copy(), boundary))
        cv2.imwrite(str(landmark_path), self._draw_landmark_overlay(aligned_face.copy(), landmark))
        cv2.imwrite(str(identity_path), self._draw_identity_overlay(aligned_face.copy(), identity))

        return {
            "aligned_face_url": self._to_relative(aligned_path),
            "boundary_overlay_url": self._to_relative(boundary_path),
            "landmark_overlay_url": self._to_relative(landmark_path),
            "identity_overlay_url": self._to_relative(identity_path),
        }

    def _persist_unavailable_artifacts(self, media_dir: Path, alignment: FaceAlignmentResult, persist: bool) -> Dict[str, str]:
        if not persist:
            return self._empty_artifacts()
        aligned_path = media_dir / "aligned_face.jpg"
        cv2.imwrite(str(aligned_path), alignment.aligned_face)
        return {
            "aligned_face_url": self._to_relative(aligned_path),
            "boundary_overlay_url": "",
            "landmark_overlay_url": "",
            "identity_overlay_url": "",
        }

    def _draw_boundary_overlay(self, aligned_face: np.ndarray, boundary: Dict[str, object]) -> np.ndarray:
        masks = boundary["masks"]
        overlay = aligned_face.copy()
        colored = np.zeros_like(overlay)
        colored[masks["ring"] > 0] = (0, 96, 255)
        colored[masks["inner"] > 0] = (0, 214, 255)
        overlay = cv2.addWeighted(overlay, 0.72, colored, 0.28, 0.0)
        cv2.putText(
            overlay,
            f"Boundary anomaly {boundary['boundary_anomaly_score']:.1f}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 214, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _draw_landmark_overlay(self, aligned_face: np.ndarray, landmark: Dict[str, object]) -> np.ndarray:
        overlay = aligned_face.copy()
        cv2.putText(
            overlay,
            f"Landmark mismatch {landmark['landmark_mismatch_score']:.1f}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 214, 255),
            2,
            cv2.LINE_AA,
        )
        if landmark.get("summary"):
            cv2.putText(
                overlay,
                str(landmark["summary"])[:54],
                (16, overlay.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _draw_identity_overlay(self, aligned_face: np.ndarray, identity: Dict[str, object]) -> np.ndarray:
        overlay = aligned_face.copy()
        for patch in identity.get("patch_scores", []):
            x, y, w, h = patch["box"]
            drift = float(patch.get("drift", 0.0))
            color = (0, 214, 255)
            if drift >= 0.28:
                color = (0, 92, 255)
            elif drift >= 0.18:
                color = (0, 170, 255)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                overlay,
                f"{patch['name']} {drift * 100.0:.0f}",
                (x + 4, max(18, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"Identity inconsistency {identity['identity_inconsistency_score']:.1f}",
            (16, overlay.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _boundary_masks(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        h, w = shape
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, int(h * 0.52))
        cv2.ellipse(outer, center, (int(w * 0.42), int(h * 0.48)), 0, 0, 360, 255, -1)
        cv2.ellipse(inner, center, (int(w * 0.30), int(h * 0.36)), 0, 0, 360, 255, -1)
        ring = cv2.subtract(outer, inner)

        forehead = np.zeros((h, w), dtype=np.uint8)
        forehead[: int(h * 0.24), :] = 255
        forehead = cv2.bitwise_and(forehead, ring)

        jaw = np.zeros((h, w), dtype=np.uint8)
        jaw[int(h * 0.66) :, :] = 255
        jaw = cv2.bitwise_and(jaw, ring)

        return {
            "outer": outer,
            "inner": inner,
            "ring": ring,
            "forehead": forehead,
            "jaw": jaw,
        }

    def _masked_histogram(self, values: np.ndarray, mask: np.ndarray, bins: int, max_value: int) -> np.ndarray:
        selected = values[mask > 0]
        if selected.size == 0:
            return np.zeros((bins,), dtype=np.float32)
        hist, _ = np.histogram(selected, bins=bins, range=(0, max_value), density=True)
        return hist.astype(np.float32)

    def _empty_artifacts(self) -> Dict[str, str]:
        return {
            "aligned_face_url": "",
            "boundary_overlay_url": "",
            "landmark_overlay_url": "",
            "identity_overlay_url": "",
        }

    def _unavailable_result(self, reason: str, frame_metadata: Dict[str, object], artifacts: Dict[str, str]) -> Dict[str, object]:
        return {
            "available": False,
            "faceswap_score": 0.0,
            "identity_inconsistency_score": 0.0,
            "boundary_anomaly_score": 0.0,
            "landmark_mismatch_score": 0.0,
            "texture_mismatch_score": 0.0,
            "region_consensus_score": 0.0,
            "temporal_identity_drift_score": 0.0,
            "summary": reason or "Insufficient face evidence for dedicated face-swap analysis.",
            "explanations": [reason or "This face-swap output could not be generated for this run."],
            "suspicious_regions": [],
            "strongest_frame": None,
            "embedding_source": "unavailable",
            "identity_consistency_score": 0.0,
            "face_detected": False,
            "detector": "unavailable",
            "frame_context": {
                "frame_number": frame_metadata.get("frame_number"),
                "timestamp_label": frame_metadata.get("timestamp_label"),
            },
            "artifacts": artifacts,
            "_identity_vector": None,
        }

    def _to_relative(self, path: Path) -> str:
        return path.relative_to(self.project_root).as_posix()
