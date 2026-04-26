from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


_INSIGHTFACE_APP = None
_INSIGHTFACE_ATTEMPTED = False


@dataclass
class FaceAlignmentResult:
    aligned_face: np.ndarray
    bbox: Tuple[int, int, int, int]
    landmarks_5: List[Tuple[int, int]]
    detector: str
    detected: bool
    usable: bool
    detection_confidence: float
    fallback_reason: str
    embedding: Optional[np.ndarray] = None


class FaceAlignmentService:
    def __init__(self, output_size: int = 384) -> None:
        self.output_size = int(output_size)
        cascades_dir = Path(cv2.data.haarcascades)
        self.face_detector = cv2.CascadeClassifier(str(cascades_dir / "haarcascade_frontalface_default.xml"))
        self.eye_detector = cv2.CascadeClassifier(str(cascades_dir / "haarcascade_eye.xml"))

    def align_primary_face(self, bgr_image: np.ndarray) -> FaceAlignmentResult:
        insight = self._align_with_insightface(bgr_image)
        if insight is not None:
            return insight
        return self._align_with_fallback(bgr_image)

    def _align_with_insightface(self, bgr_image: np.ndarray) -> Optional[FaceAlignmentResult]:
        app = _get_insightface_app()
        if app is None:
            return None

        try:
            faces = app.get(bgr_image)
        except Exception:
            return None

        if not faces:
            return None

        best_face = max(
            faces,
            key=lambda face: float(getattr(face, "det_score", 0.0)) * float(
                max(1.0, (getattr(face, "bbox", [0, 0, 1, 1])[2] - getattr(face, "bbox", [0, 0, 1, 1])[0]))
                * max(1.0, (getattr(face, "bbox", [0, 0, 1, 1])[3] - getattr(face, "bbox", [0, 0, 1, 1])[1]))
            ),
        )

        bbox = np.asarray(getattr(best_face, "bbox", [0, 0, bgr_image.shape[1], bgr_image.shape[0]]), dtype=np.float32)
        x0 = max(0, int(bbox[0]))
        y0 = max(0, int(bbox[1]))
        x1 = min(bgr_image.shape[1], int(bbox[2]))
        y1 = min(bgr_image.shape[0], int(bbox[3]))

        kps = np.asarray(getattr(best_face, "kps", []), dtype=np.float32)
        aligned_face = self._norm_crop(bgr_image, kps) if kps.size >= 10 else self._crop_and_resize(bgr_image, x0, y0, x1, y1)

        embedding = getattr(best_face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(best_face, "embedding", None)
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        landmarks_5 = [(int(point[0]), int(point[1])) for point in kps.tolist()] if kps.size >= 10 else []
        return FaceAlignmentResult(
            aligned_face=aligned_face,
            bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
            landmarks_5=landmarks_5,
            detector="insightface-buffalo_l",
            detected=True,
            usable=True,
            detection_confidence=float(getattr(best_face, "det_score", 0.98) or 0.98),
            fallback_reason="",
            embedding=embedding,
        )

    def _align_with_fallback(self, bgr_image: np.ndarray) -> FaceAlignmentResult:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(96, 96))
        if len(faces) == 0:
            h, w = bgr_image.shape[:2]
            side = min(h, w)
            x = max(0, (w - side) // 2)
            y = max(0, (h - side) // 2)
            crop = self._crop_and_resize(bgr_image, x, y, x + side, y + side)
            return FaceAlignmentResult(
                aligned_face=crop,
                bbox=(x, y, side, side),
                landmarks_5=[],
                detector="fallback-centre-crop",
                detected=False,
                usable=False,
                detection_confidence=0.0,
                fallback_reason="No primary face could be aligned; face-swap analysis fell back to generic media evidence.",
                embedding=None,
            )

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        margin_x = int(w * 0.18)
        margin_y = int(h * 0.20)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(bgr_image.shape[1], x + w + margin_x)
        y1 = min(bgr_image.shape[0], y + h + margin_y)
        face_crop = bgr_image[y0:y1, x0:x1]
        aligned_face, landmarks_5 = self._align_by_eyes(face_crop)
        return FaceAlignmentResult(
            aligned_face=cv2.resize(aligned_face, (self.output_size, self.output_size)),
            bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
            landmarks_5=landmarks_5,
            detector="opencv-face + eye-align",
            detected=True,
            usable=True,
            detection_confidence=0.81,
            fallback_reason="",
            embedding=None,
        )

    def _align_by_eyes(self, face_crop: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18))
        if len(eyes) < 2:
            resized = cv2.resize(face_crop, (self.output_size, self.output_size))
            return resized, []

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
        rotated = [self._apply_affine(rotation, point) for point in eye_centres]
        pseudo_landmarks = [
            rotated[0],
            rotated[1],
            ((rotated[0][0] + rotated[1][0]) // 2, int((rotated[0][1] + rotated[1][1]) * 0.62)),
            (int(rotated[0][0] * 0.92), int(face_crop.shape[0] * 0.72)),
            (int(rotated[1][0] * 1.02), int(face_crop.shape[0] * 0.72)),
        ]
        return aligned, pseudo_landmarks

    def _norm_crop(self, bgr_image: np.ndarray, kps: np.ndarray) -> np.ndarray:
        template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        scale = self.output_size / 112.0
        template *= scale
        transform, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), template, method=cv2.LMEDS)
        if transform is None:
            x0, y0 = np.min(kps[:, 0]), np.min(kps[:, 1])
            x1, y1 = np.max(kps[:, 0]), np.max(kps[:, 1])
            return self._crop_and_resize(bgr_image, int(x0), int(y0), int(x1), int(y1))
        return cv2.warpAffine(
            bgr_image,
            transform,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _crop_and_resize(self, bgr_image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        crop = bgr_image[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        if crop.size == 0:
            crop = cv2.resize(bgr_image, (self.output_size, self.output_size))
            return crop
        return cv2.resize(crop, (self.output_size, self.output_size))

    def _apply_affine(self, matrix: np.ndarray, point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        new_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        new_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        return int(new_x), int(new_y)


def _get_insightface_app():
    global _INSIGHTFACE_APP
    global _INSIGHTFACE_ATTEMPTED

    if _INSIGHTFACE_ATTEMPTED:
        return _INSIGHTFACE_APP

    _INSIGHTFACE_ATTEMPTED = True
    try:
        from insightface.app import FaceAnalysis

        _INSIGHTFACE_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _INSIGHTFACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception:
        _INSIGHTFACE_APP = None
    return _INSIGHTFACE_APP


def get_face_alignment_service() -> FaceAlignmentService:
    return FaceAlignmentService()
