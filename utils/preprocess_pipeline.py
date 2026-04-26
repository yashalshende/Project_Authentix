import json
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import albumentations as A
import cv2
import numpy as np

from utils.face_regions import FaceRegionAnalyzer, REGION_LAYOUT
from web_backend.face_alignment import FaceAlignmentService
from web_backend.faceswap_detector import FaceSwapDetector
from web_backend.identity_embedding import IdentityEmbeddingAnalyzer


LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "preprocessing_skipped.log")
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, format="[%(asctime)s] %(levelname)s: %(message)s")


def get_training_augmentations():
    return A.Compose(
        [
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.45),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.05, rotate_limit=12, p=0.45),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    )


def get_validation_augmentations():
    return A.Compose([A.Resize(256, 256)])


def extract_timeline_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open or decode video block.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

    extracted = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            extracted.append((int(idx), frame))

    cap.release()
    return extracted


def _relative(path, root):
    return os.path.relpath(path, root).replace("\\", "/")


def process_single_media(
    file_path,
    output_dir,
    label,
    dataset_source,
    manipulation_type="Unknown",
    subject_id="",
    is_training=True,
    frames_per_video=8,
):
    alignment_service = FaceAlignmentService()
    face_region_analyzer = FaceRegionAnalyzer()
    identity_analyzer = IdentityEmbeddingAnalyzer()
    faceswap_detector = FaceSwapDetector()
    augmentor = get_training_augmentations() if is_training else get_validation_augmentations()

    exported_records = []
    filename = os.path.basename(file_path)
    file_id = os.path.splitext(filename)[0]
    is_video = filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    try:
        frames_data = extract_timeline_frames(file_path, num_frames=frames_per_video) if is_video else [(0, cv2.imread(file_path))]
        if not frames_data:
            return []

        faces_root = os.path.join(output_dir, "faces")
        regions_root = os.path.join(output_dir, "regions")
        landmarks_root = os.path.join(output_dir, "landmarks")
        embeddings_root = os.path.join(output_dir, "embeddings")
        masks_root = os.path.join(output_dir, "masks")
        metadata_root = os.path.join(output_dir, "metadata")
        os.makedirs(faces_root, exist_ok=True)
        os.makedirs(regions_root, exist_ok=True)
        os.makedirs(landmarks_root, exist_ok=True)
        os.makedirs(embeddings_root, exist_ok=True)
        os.makedirs(masks_root, exist_ok=True)
        os.makedirs(metadata_root, exist_ok=True)

        for frame_idx, bgr_frame in frames_data:
            if bgr_frame is None:
                continue

            alignment = alignment_service.align_primary_face(bgr_frame)
            if not alignment.usable:
                continue

            augmented = augmentor(image=alignment.aligned_face)
            aligned_face = cv2.resize(augmented["image"], (256, 256))
            identity_features = identity_analyzer.analyze(aligned_face, alignment)
            faceswap_preview = faceswap_detector.analyze(
                bgr_frame,
                job_id=f"prep_{dataset_source}_{file_id}_{frame_idx}",
                media_type="image",
                face_forensics=face_region_analyzer.analyze(
                    bgr_frame,
                    job_id=f"prep_regions_{dataset_source}_{file_id}_{frame_idx}",
                    media_type="image",
                    frame_metadata={},
                    persist=False,
                ),
                frame_metadata={},
                persist=False,
            )

            record_id = f"{dataset_source}_{manipulation_type}_{label}_{file_id}_f{frame_idx}"
            face_dir = os.path.join(faces_root, record_id)
            region_dir = os.path.join(regions_root, record_id)
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(region_dir, exist_ok=True)

            aligned_path = os.path.join(face_dir, "aligned_face.jpg")
            cv2.imwrite(aligned_path, aligned_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            region_stack = face_region_analyzer.extract_region_tensor_stack(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB), output_size=128)
            for region_idx, (region_key, _) in enumerate(REGION_LAYOUT):
                region_bgr = cv2.cvtColor(region_stack[region_idx], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(region_dir, f"{region_key}.jpg"), region_bgr)

            landmarks_path = os.path.join(landmarks_root, f"{record_id}.json")
            with open(landmarks_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "bbox": list(alignment.bbox),
                        "landmarks_5": alignment.landmarks_5,
                        "detector": alignment.detector,
                        "confidence": alignment.detection_confidence,
                    },
                    handle,
                    indent=2,
                )

            embedding_path = os.path.join(embeddings_root, f"{record_id}.npy")
            np.save(embedding_path, identity_features["descriptor_embedding"].astype(np.float32))

            boundary_mask = (faceswap_detector._boundary_masks(aligned_face.shape[:2])["ring"] > 0).astype(np.uint8) * 255
            boundary_mask_path = os.path.join(masks_root, f"{record_id}.png")
            cv2.imwrite(boundary_mask_path, boundary_mask)

            exported_records.append(
                {
                    "file_path": _relative(aligned_path, output_dir),
                    "region_dir": _relative(region_dir, output_dir),
                    "landmarks_path": _relative(landmarks_path, output_dir),
                    "embedding_path": _relative(embedding_path, output_dir),
                    "boundary_mask_path": _relative(boundary_mask_path, output_dir),
                    "parent_video_id": file_id,
                    "dataset_source": dataset_source,
                    "label": int(label),
                    "faceswap_label": 1 if str(manipulation_type).lower() == "faceswap" else 0,
                    "manipulation_type": manipulation_type,
                    "subject_id": subject_id or file_id.split("_")[0],
                    "frame_idx": int(frame_idx),
                    "identity_dispersion": identity_features["dispersion"],
                    "left_right_identity_gap": identity_features["left_right_gap"],
                    "center_periphery_gap": identity_features["center_periphery_gap"],
                    "landmark_eye_alignment": 100.0 - float(faceswap_preview["landmark_mismatch_score"]),
                    "landmark_mouth_geometry": 100.0 - float(faceswap_preview["landmark_mismatch_score"]),
                    "landmark_face_symmetry": max(0.0, 100.0 - float(faceswap_preview["identity_inconsistency_score"])),
                    "landmark_contour_consistency": max(0.0, 100.0 - float(faceswap_preview["boundary_anomaly_score"])),
                    "boundary_anomaly_score": faceswap_preview["boundary_anomaly_score"],
                    "texture_mismatch_score": faceswap_preview["texture_mismatch_score"],
                    "face_quality_score": 100.0 - min(float(faceswap_preview["faceswap_score"]), 100.0),
                }
            )

        return exported_records
    except Exception as exc:
        logging.error(f"{file_path} - FAILED: {exc}\n{traceback.format_exc()}")
        return []


def run_pipeline_batch(media_list, output_directory, workers=4):
    master_metadata = []
    os.makedirs(output_directory, exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_media, item[0], output_directory, *item[1:]): item[0]
            for item in media_list
        }
        for future in as_completed(futures):
            records = future.result()
            if records:
                master_metadata.extend(records)

    metadata_path = os.path.join(output_directory, "metadata", "unified_schema_export.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(master_metadata, handle, indent=2)

    print(f"Extraction pipeline complete. Generated {len(master_metadata)} aligned face records.")


if __name__ == "__main__":
    print("Pre-processing pipeline ready. Use run_pipeline_batch(...) to prepare aligned faces and face-swap metadata.")
