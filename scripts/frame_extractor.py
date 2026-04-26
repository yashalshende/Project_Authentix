import cv2
import os
import glob
import numpy as np

# Find project root so script can be run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'raw_videos')
OUT_FRAMES_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'extracted_frames')

def extract_frames(video_path, output_dir, label, num_frames=15):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Could not read frame count for {video_path}")
        cap.release()
        return

    # Calculate exactly 15 evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    count = 0
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Format: videoname_01.jpg
            out_filename = f"{video_name}_{idx:02d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, frame)
            count += 1
            
    cap.release()
    print(f"Extracted {count} frames from {video_name}.mp4")

def process_all_videos(raw_dir, output_dir, num_frames=15):
    for label in ['real', 'fake']:
        input_folder = os.path.join(raw_dir, label)
        output_folder = os.path.join(output_dir, label)
        
        # Ensure output subdirectory exists
        os.makedirs(output_folder, exist_ok=True)
        
        video_files = glob.glob(os.path.join(input_folder, '*.mp4'))
        print(f"\n--- Found {len(video_files)} {label} videos ---")
        
        for video_path in video_files:
            extract_frames(video_path, output_folder, label, num_frames)

if __name__ == '__main__':
    print("Starting Frame Extraction...")
    print(f"Looking for videos in: {RAW_VIDEOS_DIR}")
    process_all_videos(RAW_VIDEOS_DIR, OUT_FRAMES_DIR, num_frames=15)
    print("\nFrame extraction completed! Check dataset/extracted_frames folder.")
