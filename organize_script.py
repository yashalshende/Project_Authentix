import os
import shutil
import re

project_dir = r"d:\Project Authentix"

# Step 1: Create directories
dirs_to_create = [
    'core_engine/config',
    'core_engine/data',
    'core_engine/layers',
    'core_engine/models',
    'core_engine/xai',
    'training_scripts/training',
    'training_scripts/evaluation',
    'training_scripts/dataset_prep',
]

for d in dirs_to_create:
    os.makedirs(os.path.join(project_dir, d), exist_ok=True)
    # create __init__.py so Python recognizes them as packages
    init_path = os.path.join(project_dir, d, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            pass

# Step 2: Move files into their respective categorical folders
moves = [
    ('core_engine/config.py', 'core_engine/config/config.py'),
    ('core_engine/dataset_loader.py', 'core_engine/data/dataset_loader.py'),
    ('core_engine/attention.py', 'core_engine/layers/attention.py'),
    ('core_engine/region_fusion.py', 'core_engine/layers/region_fusion.py'),
    ('core_engine/frequency_net.py', 'core_engine/models/frequency_net.py'),
    ('core_engine/fusion_model.py', 'core_engine/models/fusion_model.py'),
    ('core_engine/spatial_net.py', 'core_engine/models/spatial_net.py'),
    ('core_engine/temporal_net.py', 'core_engine/models/temporal_net.py'),
    ('core_engine/xai_cam.py', 'core_engine/xai/xai_cam.py'),

    ('training_scripts/cross_dataset_eval.py', 'training_scripts/evaluation/cross_dataset_eval.py'),
    ('training_scripts/eval_utils.py', 'training_scripts/evaluation/eval_utils.py'),
    ('training_scripts/eval.py', 'training_scripts/evaluation/eval.py'),
    ('training_scripts/prepare_faceswap_dataset.py', 'training_scripts/dataset_prep/prepare_faceswap_dataset.py'),
    ('training_scripts/train_utils.py', 'training_scripts/training/train_utils.py'),
    ('training_scripts/train.py', 'training_scripts/training/train.py'),
]

for src, dst in moves:
    src_path = os.path.join(project_dir, src)
    dst_path = os.path.join(project_dir, dst)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved {src} to {dst}")

# Step 3: Update imports across all Python scripts
patterns = [
    (re.compile(r'from core_engine\.attention import'), r'from core_engine.layers.attention import'),
    (re.compile(r'import core_engine\.attention\b'), r'import core_engine.layers.attention'),
    (re.compile(r'from core_engine\.config import'), r'from core_engine.config.config import'),
    (re.compile(r'import core_engine\.config\b'), r'import core_engine.config.config'),
    (re.compile(r'from core_engine\.dataset_loader import'), r'from core_engine.data.dataset_loader import'),
    (re.compile(r'import core_engine\.dataset_loader\b'), r'import core_engine.data.dataset_loader'),
    (re.compile(r'from core_engine\.frequency_net import'), r'from core_engine.models.frequency_net import'),
    (re.compile(r'import core_engine\.frequency_net\b'), r'import core_engine.models.frequency_net'),
    (re.compile(r'from core_engine\.fusion_model import'), r'from core_engine.models.fusion_model import'),
    (re.compile(r'import core_engine\.fusion_model\b'), r'import core_engine.models.fusion_model'),
    (re.compile(r'from core_engine\.region_fusion import'), r'from core_engine.layers.region_fusion import'),
    (re.compile(r'import core_engine\.region_fusion\b'), r'import core_engine.layers.region_fusion'),
    (re.compile(r'from core_engine\.spatial_net import'), r'from core_engine.models.spatial_net import'),
    (re.compile(r'import core_engine\.spatial_net\b'), r'import core_engine.models.spatial_net'),
    (re.compile(r'from core_engine\.temporal_net import'), r'from core_engine.models.temporal_net import'),
    (re.compile(r'import core_engine\.temporal_net\b'), r'import core_engine.models.temporal_net'),
    (re.compile(r'from core_engine\.xai_cam import'), r'from core_engine.xai.xai_cam import'),
    (re.compile(r'import core_engine\.xai_cam\b'), r'import core_engine.xai.xai_cam'),
    (re.compile(r'from training_scripts\.cross_dataset_eval import'), r'from training_scripts.evaluation.cross_dataset_eval import'),
    (re.compile(r'from training_scripts\.eval_utils import'), r'from training_scripts.evaluation.eval_utils import'),
    (re.compile(r'from training_scripts\.eval import'), r'from training_scripts.evaluation.eval import'),
    (re.compile(r'from training_scripts\.prepare_faceswap_dataset import'), r'from training_scripts.dataset_prep.prepare_faceswap_dataset import'),
    (re.compile(r'from training_scripts\.train_utils import'), r'from training_scripts.training.train_utils import'),
    (re.compile(r'from training_scripts\.train import'), r'from training_scripts.training.train import')
]

for root, _, files in os.walk(project_dir):
    if 'venv' in root or '.venv' in root or '__pycache__' in root or '.vscode' in root:
        continue
    for file in files:
        if file.endswith('.py') and file != 'organize_script.py':
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                original_content = content
                for pat, repl in patterns:
                    content = pat.sub(repl, content)
                if content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f'Updated imports in {filepath}')
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

print("Organization and automatic import updates complete!")
