import os
import cv2
import glob
from mtcnn import MTCNN

# Find project root so script can be run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

FRAMES_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'extracted_frames')
CROPPED_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'cropped_faces')

def crop_faces(frames_dir, output_dir, target_size=(224, 224), expand_ratio=0.15):
    detector = MTCNN()
    
    for label in ['real', 'fake']:
        input_folder = os.path.join(frames_dir, label)
        output_folder = os.path.join(output_dir, label)
        os.makedirs(output_folder, exist_ok=True)
        
        image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
        print(f"\n--- Found {len(image_files)} frames to crop in {label} folder ---")
        
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None: 
                print(f"Skipping unreadable image {img_path}")
                continue
            
            # MTCNN algorithm requires RGB input
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)
            
            if len(results) > 0:
                # Find the largest bounding box (the main face) instead of background faces
                best_face = max(results, key=lambda b: b['box'][2] * b['box'][3])
                x, y, w, h = best_face['box']
                
                # Expand box slightly to include full chin and forehead
                x1 = max(0, x - int(expand_ratio * w))
                y1 = max(0, y - int(expand_ratio * h))
                x2 = min(img.shape[1], x + w + int(expand_ratio * w))
                y2 = min(img.shape[0], y + h + int(expand_ratio * h))
                
                cropped = img[y1:y2, x1:x2]
                
                try:
                    # Resize to model standard 224x224
                    resized = cv2.resize(cropped, target_size)
                    out_name = os.path.basename(img_path)
                    out_path = os.path.join(output_folder, out_name)
                    cv2.imwrite(out_path, resized)
                except Exception as e:
                    print(f"Error resizing {img_path}: {e}")
            else:
                print(f"No face detected in {img_path}")

if __name__ == '__main__':
    print("Starting Face Cropping with MTCNN...")
    crop_faces(FRAMES_DIR, CROPPED_DIR)
    print("\nFace cropping completed! Check dataset/cropped_faces folder.")
