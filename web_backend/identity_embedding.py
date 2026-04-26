from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class IdentityEmbeddingAnalyzer:
    def __init__(self) -> None:
        self.lbp_radius = 2
        self.lbp_points = self.lbp_radius * 8

    def analyze(self, aligned_face: np.ndarray, alignment_result=None) -> Dict[str, object]:
        aligned_face = cv2.resize(aligned_face, (384, 384))
        full_descriptor = self._descriptor(aligned_face)
        patch_descriptors = self._extract_patch_descriptors(aligned_face)

        patch_scores: List[Dict[str, object]] = []
        centre_vector = patch_descriptors["center"]["descriptor"]
        left_vector = patch_descriptors["left"]["descriptor"]
        right_vector = patch_descriptors["right"]["descriptor"]

        centre_drift_values = []
        pairwise_drift_values = []
        for name, payload in patch_descriptors.items():
            cosine = self._cosine_similarity(full_descriptor, payload["descriptor"])
            centre_cosine = self._cosine_similarity(centre_vector, payload["descriptor"])
            drift = 1.0 - centre_cosine
            centre_drift_values.append(drift)
            pairwise_drift_values.append(1.0 - cosine)
            patch_scores.append(
                {
                    "name": name,
                    "box": payload["box"],
                    "cosine_similarity": round(float(cosine), 4),
                    "drift": round(float(drift), 4),
                }
            )

        left_right_gap = 1.0 - self._cosine_similarity(left_vector, right_vector)
        center_periphery_gap = float(np.mean(centre_drift_values))
        pairwise_dispersion = float(np.std(pairwise_drift_values))
        global_consistency = float(np.mean([entry["cosine_similarity"] for entry in patch_scores]))

        score = (
            min(1.0, max(0.0, center_periphery_gap * 2.2)) * 40.0
            + min(1.0, max(0.0, left_right_gap * 2.0)) * 35.0
            + min(1.0, max(0.0, pairwise_dispersion * 5.5)) * 25.0
        )
        score = round(float(max(0.0, min(100.0, score))), 2)

        arcface_embedding = None
        if alignment_result is not None:
            arcface_embedding = getattr(alignment_result, "embedding", None)
        embedding_source = "descriptor-fallback"
        if arcface_embedding is not None:
            embedding_source = "insightface-arcface + descriptor"
            arcface_embedding = np.asarray(arcface_embedding, dtype=np.float32).reshape(-1)

        summary = (
            "Possible identity drift across facial sub-regions."
            if score >= 58
            else "Identity cues remain mostly consistent across facial sub-regions."
        )

        return {
            "identity_inconsistency_score": score,
            "global_consistency": round(global_consistency * 100.0, 2),
            "left_right_gap": round(left_right_gap * 100.0, 2),
            "center_periphery_gap": round(center_periphery_gap * 100.0, 2),
            "dispersion": round(pairwise_dispersion * 100.0, 2),
            "patch_scores": patch_scores,
            "embedding_source": embedding_source,
            "descriptor_embedding": full_descriptor,
            "arcface_embedding": arcface_embedding,
            "summary": summary,
        }

    def compute_temporal_identity_drift(self, identity_vectors: List[np.ndarray]) -> float:
        valid = [self._normalize_vector(np.asarray(vector, dtype=np.float32).reshape(-1)) for vector in identity_vectors if vector is not None]
        if len(valid) < 2:
            return 0.0

        drifts = []
        for prev, curr in zip(valid[:-1], valid[1:]):
            if prev.shape != curr.shape:
                continue
            drifts.append(1.0 - self._cosine_similarity(prev, curr))
        if not drifts:
            return 0.0

        score = min(100.0, max(0.0, float(np.mean(drifts) * 230.0)))
        return round(score, 2)

    def _extract_patch_descriptors(self, aligned_face: np.ndarray) -> Dict[str, Dict[str, object]]:
        h, w = aligned_face.shape[:2]
        boxes = {
            "center": (int(w * 0.24), int(h * 0.20), int(w * 0.76), int(h * 0.82)),
            "upper": (int(w * 0.20), int(h * 0.08), int(w * 0.80), int(h * 0.46)),
            "lower": (int(w * 0.20), int(h * 0.48), int(w * 0.80), int(h * 0.94)),
            "left": (int(w * 0.06), int(h * 0.18), int(w * 0.50), int(h * 0.88)),
            "right": (int(w * 0.50), int(h * 0.18), int(w * 0.94), int(h * 0.88)),
        }
        payload = {}
        for name, (x0, y0, x1, y1) in boxes.items():
            crop = aligned_face[y0:y1, x0:x1]
            payload[name] = {
                "box": [x0, y0, x1 - x0, y1 - y0],
                "descriptor": self._descriptor(crop),
            }
        return payload

    def _descriptor(self, bgr_image: np.ndarray) -> np.ndarray:
        image = cv2.resize(bgr_image, (96, 96))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        color_hist = []
        for channel in cv2.split(hsv):
            hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).flatten()
            color_hist.append(hist)
        color_hist = np.concatenate(color_hist).astype(np.float32)

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        grad_hist = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [24], [0, 256]).flatten().astype(np.float32)

        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_points + 3, range=(0, self.lbp_points + 2), density=True)
        lbp_hist = lbp_hist.astype(np.float32)

        pooled = cv2.resize(gray, (8, 8)).astype(np.float32).flatten() / 255.0
        stats = np.array(
            [
                float(np.mean(gray)) / 255.0,
                float(np.std(gray)) / 255.0,
                float(np.mean(magnitude)) / 255.0,
                float(np.std(magnitude)) / 255.0,
                float(cv2.Laplacian(gray, cv2.CV_32F).var() / 4000.0),
            ],
            dtype=np.float32,
        )

        descriptor = np.concatenate([color_hist, grad_hist, lbp_hist, pooled, stats], axis=0)
        return self._normalize_vector(descriptor)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _cosine_similarity(self, first: np.ndarray, second: np.ndarray) -> float:
        first = self._normalize_vector(np.asarray(first, dtype=np.float32).reshape(-1))
        second = self._normalize_vector(np.asarray(second, dtype=np.float32).reshape(-1))
        length = min(first.shape[0], second.shape[0])
        if length == 0:
            return 0.0
        return float(np.clip(np.dot(first[:length], second[:length]), -1.0, 1.0))
