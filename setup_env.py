import os
import subprocess
import sys

# Directories to create
dirs = [
    r'dataset\raw_videos\real',
    r'dataset\raw_videos\fake',
    r'dataset\extracted_frames',
    r'dataset\cropped_faces\real',
    r'dataset\cropped_faces\fake',
    r'model_data\train',
    r'model_data\val',
    r'model_data\test',
    r'model_data\saved_models',
    r'scripts',
    r'web_app\static',
    r'web_app\templates',
    r'web_app\uploads'
]

print("Creating directories...")
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created {d}")

print("Installing packages...")
pip_executable = os.path.join('.venv', 'Scripts', 'pip.exe')
if not os.path.exists(pip_executable):
    pip_executable = sys.executable + " -m pip"

subprocess.check_call([
    pip_executable, 'install', 
    'tensorflow', 'visualkeras', 'mtcnn', 'split-folders', 'Flask', 'opencv-python', 'pandas'
])

print("Setup complete!")
