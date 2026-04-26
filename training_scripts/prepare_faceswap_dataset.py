import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from utils.preprocess_pipeline import run_pipeline_batch


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def split_from_subject(subject_id: str) -> str:
    digest = hashlib.md5(subject_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:2], 16) % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "val"
    return "test"


def collect_video_paths(root: Path):
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTS)

#Dataset Collection for Training
def collect_faceforensics(ffpp_root: Path, compression: str, priority_manipulations):
    media = []
    original_root = ffpp_root / "original_sequences" / "youtube" / compression / "videos"
    for path in collect_video_paths(original_root):
        subject_id = path.stem.split("_")[0]
        media.append((str(path), 0, "FFPP", "original", subject_id, True, 8))

    for manipulation in priority_manipulations:
        manip_root = ffpp_root / "manipulated_sequences" / manipulation / compression / "videos"
        for path in collect_video_paths(manip_root):
            subject_id = path.stem.split("_")[0]
            media.append((str(path), 1, "FFPP", manipulation, subject_id, True, 8))
    return media


def collect_celebdf(celebdf_root: Path):
    media = []
    for folder_name, label, manip_type in [("Celeb-real", 0, "original"), ("Celeb-synthesis", 1, "deepfake")]:
        folder = celebdf_root / folder_name
        for path in collect_video_paths(folder):
            subject_id = path.stem.split("_")[0]
            media.append((str(path), label, "CelebDF", manip_type, subject_id, True, 8))
    return media


def collect_dfdc(dfdc_root: Path):
    media = []
    candidates = []
    if (dfdc_root / "train_sample_videos").exists():
        candidates.extend(collect_video_paths(dfdc_root / "train_sample_videos"))
    else:
        candidates.extend(collect_video_paths(dfdc_root))

    for path in candidates:
        lowered = str(path).lower()
        label = 1 if "fake" in lowered or "manip" in lowered else 0
        manip_type = "unknown" if label == 0 else "deepfake"
        subject_id = path.stem.split("_")[0]
        media.append((str(path), label, "DFDC", manip_type, subject_id, True, 8))
    return media


def build_csv(output_dir: Path):
    export_path = output_dir / "metadata" / "unified_schema_export.json"
    if not export_path.exists():
        raise FileNotFoundError(f"Missing export metadata at {export_path}")

    rows = json.loads(export_path.read_text(encoding="utf-8"))
    frame_df = pd.DataFrame(rows)
    if frame_df.empty:
        raise ValueError("No extracted samples were generated.")

    frame_df["split"] = frame_df["subject_id"].fillna("unknown").map(split_from_subject)
    csv_path = output_dir / "metadata" / "faceswap_samples.csv"
    frame_df.to_csv(csv_path, index=False)
    print(f"Face-swap training CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare aligned face-swap training data for AUTHENTIX.")
    parser.add_argument("--ffpp", type=str, default="", help="FaceForensics++ root directory")
    parser.add_argument("--celebdf", type=str, default="", help="Celeb-DF root directory")
    parser.add_argument("--dfdc", type=str, default="", help="DFDC root directory")
    parser.add_argument("--out", type=str, required=True, help="Output cache directory")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--compression", type=str, default="c23")
    parser.add_argument("--priority-manipulations", nargs="+", default=["FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    media = []
    if args.ffpp:
        media.extend(collect_faceforensics(Path(args.ffpp), args.compression, args.priority_manipulations))
    if args.celebdf:
        media.extend(collect_celebdf(Path(args.celebdf)))
    if args.dfdc:
        media.extend(collect_dfdc(Path(args.dfdc)))

    media = [
        (path, label, source, manipulation, subject_id, True, args.frames_per_video)
        for path, label, source, manipulation, subject_id, _, _ in media
    ]

    if not media:
        raise ValueError("No dataset media found. Place datasets manually in the configured roots and retry.")

    run_pipeline_batch(media, str(output_dir), workers=args.workers)
    build_csv(output_dir)


if __name__ == "__main__":
    main()
