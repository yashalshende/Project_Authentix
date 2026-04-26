import os
import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

#dataset
def setup_directory_structure():
    """Builds the deepfake dataset directory architecture."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    dirs = [
        "1_raw_videos/DFDC",
        "1_raw_videos/Celeb-DF",
        "1_raw_videos/FaceForensics++",
        "2_extracted_faces",
        "3_metadata"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    logging.info("Dataset hierarchy successfully initialized.")
    return base_dir

#DFDC metadata
def parse_dfdc_metadata(raw_dir, extract_dir):
    """Integrates the DFDC JSON metadata into the standard Authentix CSV layout."""
    dfdc_path = os.path.join(raw_dir, "DFDC")
    json_files = list(Path(dfdc_path).rglob("metadata.json"))
    
    records = []
    for jpath in json_files:
        with open(jpath, 'r') as f:
            data = json.load(f)
            for video_name, meta in data.items():
                label = 1 if meta['label'] == 'FAKE' else 0
                subset = "train" if np.random.rand() > 0.15 else "val"
                records.append({
                    "file_path": f"DFDC/{video_name.replace('.mp4', '.jpg')}",
                    "label": label,
                    "faceswap_label": label,  # DFDC relies heavily on face-swapping
                    "manipulation_type": "faceswap" if label == 1 else "authentic",
                    "split": subset,
                    "dataset_source": "DFDC",
                    "parent_video_id": meta.get('original', video_name)
                })
    return records

def parse_celeb_df_metadata(raw_dir, extract_dir):
    """Integrates Celeb-DF's List_of_testing_videos.txt into the standard format."""
    celeb_txt = os.path.join(raw_dir, "Celeb-DF", "List_of_testing_videos.txt")
    records = []
    if os.path.exists(celeb_txt):
        with open(celeb_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    label = 1 if parts[0] == '0' else 0 # 0 means fake in CelebDF lists sometimes, verify logic
                    video_name = parts[1]
                    records.append({
                        "file_path": f"Celeb-DF/{video_name.replace('.mp4', '.jpg')}",
                        "label": label,
                        "faceswap_label": label,
                        "manipulation_type": "deepfake",
                        "split": "test",
                        "dataset_source": "Celeb-DF",
                        "parent_video_id": video_name
                    })
    return records

def build_unified_model(base_dir):
    """Generates the unified classification metadata for the DataLoader."""
    raw_dir = os.path.join(base_dir, "1_raw_videos")
    extract_dir = os.path.join(base_dir, "2_extracted_faces")
    
    logging.info("Scanning for DFDC Datasets...")
    dfdc_records = parse_dfdc_metadata(raw_dir, extract_dir)
    
    logging.info("Scanning for Celeb-DF Datasets...")
    celeb_records = parse_celeb_df_metadata(raw_dir, extract_dir)
    
    # Fake synthetic generation for FaceForensics++ as placeholder
    ff_records = []
    
    all_records = dfdc_records + celeb_records + ff_records
    if not all_records:
        logging.warning("No raw dataset metadata found. Providing skeleton CSV.")
        all_records = [{
            "file_path": "placeholder.jpg", "label": 0, "faceswap_label": 0, 
            "manipulation_type": "real", "split": "train", "dataset_source": "system", "parent_video_id": "0000"
        }]

    df = pd.DataFrame(all_records)
    csv_out = os.path.join(base_dir, "3_metadata", "unified_train.csv")
    df.to_csv(csv_out, index=False)
    logging.info(f"Unified Data Model generated at {csv_out} with {len(df)} records.")

if __name__ == "__main__":
    import numpy as np
    logging.info("Initializing Authentix Multi-Dataset Integrator...")
    base_data = setup_directory_structure()
    build_unified_model(base_data)
    
    print("\n--- DATASET DOWNLOAD INSTRUCTIONS ---")
    print("Due to the size of these databases (>650GB total), run the following off-band:")
    print("1. DFDC: 'kaggle competitions download -c deepfake-detection-challenge -p data/1_raw_videos/DFDC'")
    print("2. Celeb-DF: Fill out the institutional form on their repo to get Google Drive links.")
    print("3. FaceForensics++: Use the official download.py script provided by Technical University of Munich.")
