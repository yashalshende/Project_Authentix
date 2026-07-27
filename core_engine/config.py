"""
AUTHENTIX: Model Configuration Hub
"""

import os


class ModelConfig:
    DEMO_MODE_ACTIVE = True  # Set to False to utilize the PyTorch model for higher accuracy instead of heuristics
    
    # 1. Inputs
    IMG_SIZE = 256
    IN_CHANNELS = 3
    NUM_FACE_REGIONS = 9
    REGION_SIZE = 128
    
    # 2. Main Architecture Parameters
    SPATIAL_BACKBONE = 'efficientnet_v2_s'    # Lightweight v2 backbone optimized for accuracy and feature depth
    NUM_CLASSES = 1                       # Binary outcome (0=Real, 1=Fake) 
                                          # Optimizes to BSEWithLogitsLoss efficiently
    
    # 3. Frequency Core Settings
    WAVELET = 'haar'
    FREQ_FEATURES_DIM = 256               # Dimensional output length assigned to Frequency Branch
    
    # 4. Attention mechanism Settings
    USE_CBAM = True                       # Toggle Convolutional Block Attention Module
    
    # 5. Temporal Video Tracking Settings
    SEQ_LENGTH = 10                       # Standard frame ingestion sequence
    LSTM_HIDDEN = 256                     # Compressed chronological cell logic
    LSTM_LAYERS = 1                       # Strictly 1 layer avoiding overfitting CPUs
    
    # 6. Fusion Classification
    FUSION_DIM = 512
    REGION_EMBED_DIM = 256
    USE_FACE_REGION_BRANCH = True
    FACE_REGION_WEIGHT = 0.65
    GLOBAL_FACE_WEIGHT = 0.35
    FACE_SWAP_ENABLED = True
    FACE_SWAP_THRESHOLD = 46.0          # Lowered slightly to catch heuristic-only face swap evidence sooner
    FACE_SWAP_TYPE_THRESHOLD = 58.0
    FACE_SWAP_SIGNAL_MIN = 45.0         # Lowered: triggers on more reenactment signals (was 55)
    REENACTMENT_SIGNAL_THRESHOLD = 38.0 # NEW: landmark+boundary combined trigger for reenactment
    LOW_FACE_QUALITY_GATE = 40.0
    IDENTITY_DISPERSION_WINDOW = 5
    VIDEO_FACE_SWAP_FRAMES = 8
    FACE_SWAP_AUX_DIM = 10
    USE_FACE_SWAP_AUX_FEATURES = True
    DROPOUT_RATE = 0.5
    FINAL_DEEPFAKE_THRESHOLD = 50.0
    VIDEO_FINAL_DEEPFAKE_THRESHOLD = 44.0   # Runtime calibration can refine this further from labeled references
    LOW_CONFIDENCE_FLOOR = 35.0             # Lowered from 40 for more sensitive floor
    HIGH_CONFIDENCE_FLOOR = 72.0            # Lowered from 78
    VIDEO_WEAK_FACE_RATIO_FLOOR = 0.35
    VIDEO_LOW_PERSISTENCE_FLOOR = 0.18
    VIDEO_LOW_CRITICAL_HIT_FLOOR = 0.18
    VIDEO_AUTHENTICITY_SUPPORT_FLOOR = 65.0
    VIDEO_FAST_FRAME_COUNT = 8              # Reduce fast-mode latency while keeping enough temporal coverage
    VIDEO_DEEP_FRAME_COUNT = 18             # Keep deep mode thorough without the previous 24-frame overhead
    VIDEO_TOPK_FRAMES = 6                   # Increased from 5
    VIDEO_PERSISTENCE_THRESHOLD = 38.0      # Lowered from 44: catch more persistent anomalies
    VIDEO_REPEAT_FAKE_RATIO = 0.35          # Lowered from 0.45: trigger on less repetition
    TEMPORAL_CONSENSUS_BOOST = 20.0         # Increased from 14: persistence matters more
    HEURISTIC_BLEND_WEIGHT = 0.18          # Keep more weight on tuned heuristics when checkpoints are unavailable
    MODEL_BLEND_WEIGHT = 0.82
    IMAGE_BASE_BLEND_WEIGHT = 0.34
    IMAGE_FACESWAP_BLEND_WEIGHT = 0.28
    IMAGE_DATASET_BLEND_WEIGHT = 0.24
    IMAGE_FREQUENCY_BLEND_WEIGHT = 0.08
    IMAGE_REENACTMENT_BLEND_WEIGHT = 0.06
    DATASET_REFERENCE_SAMPLE_SIZE = 24
    AUDIO_ANALYSIS_ENABLED = True
    AUDIO_ANALYSIS_IN_FAST_MODE = False
    AUDIO_MAX_ANALYSIS_SECONDS = 12
    AUDIO_SKIP_FOR_LONG_VIDEOS_SECONDS = 45
    FACE_FORENSIC_PEAK_WEIGHT = 0.40
    FACE_FORENSIC_BASE_WEIGHT = 0.35
    FACE_SWAP_BRANCH_WEIGHT = 0.25
    FAKE_CLASS_WEIGHT = 2.6
    FACE_SWAP_CLASS_WEIGHT = 1.8
    FOCAL_GAMMA = 2.0
    OHEM_KEEP_RATIO = 0.72
    THRESHOLD_SEARCH_MIN = 0.30
    THRESHOLD_SEARCH_MAX = 0.72
    THRESHOLD_SEARCH_STEPS = 29
    PRECISION_FLOOR = 0.45
    CHECKPOINT_RECALL_WEIGHT = 0.45
    CHECKPOINT_F1_WEIGHT = 0.25
    CHECKPOINT_AUC_WEIGHT = 0.20
    CHECKPOINT_PRECISION_WEIGHT = 0.10
    HARD_FAKE_OVERSAMPLE = 2.8
    FACE_SWAP_OVERSAMPLE = 1.6
    HARD_SAMPLE_SCORE = 55.0
    TEMPORAL_TRAINING_ENABLED = True
    BEST_MODEL_PATH = os.path.join("models", "checkpoints", "authentix_best_model.pth")
    BEST_TEMPORAL_MODEL_PATH = os.path.join("models", "checkpoints", "authentix_temporal_best_model.pth")
    
    # 7. Complete Training Architecture Settings
    EPOCHS = 50
    EARLY_STOP_PATIENCE = 7               # Unchanged validation metrics threshold before convergence
    CHECKPOINT_DIR = "../models/checkpoints" # Storing intermediate state dictionaries securely
    LOG_DIR = "../models/logs"
    
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
