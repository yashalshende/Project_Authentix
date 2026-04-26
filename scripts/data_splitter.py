import os
import splitfolders

# Find project root so script can be run from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CROPPED_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'cropped_faces')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'model_data')

def split_dataset(input_dir, output_dir, ratio=(0.7, 0.15, 0.15)):
    # splitfolders automatically reads the 'real' and 'fake' subfolders inside CROPPED_DIR
    # and splits the images into train, val, and test according to the given ratio.
    print(f"Splitting data from {input_dir} into {output_dir}...")
    
    # Check if we have data to split
    real_count = len(os.listdir(os.path.join(input_dir, 'real'))) if os.path.exists(os.path.join(input_dir, 'real')) else 0
    fake_count = len(os.listdir(os.path.join(input_dir, 'fake'))) if os.path.exists(os.path.join(input_dir, 'fake')) else 0
    
    if real_count == 0 and fake_count == 0:
        print("Error: No images found in dataset/cropped_faces. Please run face_cropper.py first.")
        return

    # Seed is used for reproducibility so you always get the same split on the same data
    splitfolders.ratio(
        input_dir, 
        output=output_dir, 
        seed=1337, 
        ratio=ratio, 
        group_prefix=None, # If set to 2, it would group by prefix. We simply randomize here.
        move=False # We copy the images instead of moving them in case we make a mistake
    )
    print("\nData splitting completed successfully!")

if __name__ == '__main__':
    print("Starting Train/Val/Test Split...")
    split_dataset(CROPPED_DIR, OUTPUT_DIR)
