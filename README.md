# AUTHENTIX
**Advanced Multi-Modal Deepfake Diagnostic Framework**

## 📌 Project Overview
AUTHENTIX is a cutting-edge academic prototype designed to detect, analyze, and explain AI-synthesized media (deepfakes) across both images and video sequences. Built with a focus on **Explainable AI (XAI)**, it not only provides a binary authenticity classification but also generates forensic thermal traces to visually explain its neural decision-making process.

---

## 🛑 Problem Statement
The exponential rise of generative AI models (GANs, Diffusion Networks, and highly accessible deepfake software) poses a severe threat to digital identity, political integrity, and personal privacy. Traditional spatial detectors are falling behind as synthetic media improves in visual quality. A modern detection system requires a robust, multi-modal approach combining spatial, frequency, and temporal analysis, alongside explainable intelligence, to effectively counter these threats.

---

## ✨ Key Features
*   **Dual-Analysis Modalities:** Choose between *Fast Mode* (single focal frame analysis) and *Deep Sequence Mode* (10-frame chronological timeline tracking).
*   **XAI Attention Maps:** Generates dynamic Grad-CAM heatmaps highlighting exact regions of structural manipulation, blending anomalies, and unnatural noise matrices.
*   **Hybrid Inference Engine:** Fuses Spatial Convolutional features with Discrete Wavelet Transforms (DWT) to catch invisible frequency anomalies.
*   **Persistent Threat Telemetry:** Local SQLite database automatically logs previous analyses for retrospective historical audits.
*   **Forensic Reporting:** Dynamically generates a downloadable HTML/PDF diagnostic report for every analyzed payload.
*   **Lightweight Lab Demo Mode:** Intelligent software bypass built-in to allow college presentations to run instantly on standard low-RAM laptops without loading heavy C++ GPU tensors.

---

## 🛠️ Tech Stack
*   **Deep Learning Backend:** PyTorch, Torchvision, Facenet-PyTorch (MTCNN)
*   **Computer Vision:** OpenCV (cv2), MediaPipe, Albumentations
*   **Web Server / API:** Python 3.x, Flask, Werkzeug, SQLite3
*   **Interface Layer:** Vanilla HTML5, CSS3 (Neon Cyberpunk Theme), JavaScript (ES6)

---

## 📊 Dataset Usage
AUTHENTIX is engineered and structurally optimized to be trained on the industry standard deepfake datasets:
*   **FaceForensics++ (FF++)**: Primary foundation for spatial compression and blending anomalies.
*   **Celeb-DF (V2)**: Utilized to reduce false positives against high-quality, seamless face-swaps.
*   **Deepfake Detection Challenge (DFDC)**: Used for evaluating extreme edge cases involving heavy compression and diverse lighting.

> Note: the training datasets are intentionally not committed to this GitHub repository. Keep your local dataset folders outside version control and point the training pipeline to your own copies.

---

## 🧠 Neural Architecture
The core engine (`core_engine/`) relies on a triple-threat architecture:
1.  **Spatial Branch**: `EfficientNet-B0` equipped with Convolutional Block Attention Modules (CBAM) isolates specific pixel-level modifications and edge inconsistencies.
2.  **Frequency Branch**: Applies Two-Dimensional Discrete Wavelet Transforms (2D-DWT via Haar filters) across the RGB channels to isolate hidden GAN noise and frequency-domain synthetic traces.
3.  **Temporal Branch**: For videos, a dedicated 1-Layer `LSTM` captures chronological inconsistencies and unnatural optical flow across a 10-frame extracted sliding window.

---

## ⚙️ Preprocessing Pipeline
1.  **Extraction**: Frame extraction via OpenCV `VideoCapture`.
2.  **Facial Tracking**: `MTCNN` detects the primary face subject within the frame.
3.  **Alignment & Cropping**: The bounding box is expanded by a `1.3x` margin ratio to capture crucial jawline blending artifacts.
4.  **Normalization**: Matrix resized to `256x256`, normalized between `[0, 1]`, and converted to standard PyTorch tensors `(N, C, H, W)`.

---

## 🚀 How to Run Locally

### 1. Requirements
Ensure you have `Python 3.8+` installed.

### 2. Installation
Clone the repository and install the dependencies natively:
```bash
pip install -r requirements.txt
```

This repository excludes local runtime artifacts such as SQLite databases, uploaded media, generated outputs, logs, and virtual environment files. Those will be created locally as you run the project.

### 3. Execution
Start the Flask Web Server:
```bash
python app.py
```
*Navigate to `http://127.0.0.1:5000/` in your browser.*

### 4. 💻 Important: Lab Demo Mode
For academic presentations where laptops lack GPU access (or to evaluate the UI without training weights), AUTHENTIX is pre-configured in **Demo Mode**. This bypasses heavy GPU initializations and simulates XAI heatmaps instantly.
*   To switch to your trained `.pth` PyTorch weights, edit `core_engine/config.py`:
    ```python
    class ModelConfig:
        DEMO_MODE_ACTIVE = False # Set to False to unbind simulation
    ```

---

## 📡 API Endpoints
| Endpoint | Method | Functionality |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the main UI Application dashboard. |
| `/api/analyze` | `POST` | Ingests `multipart/form-data` payload, routes to Core Network, returns JSON metrics and XAI paths. Accepts `mode` (fast/deep). |
| `/api/history` | `GET` | Returns an array of previous SQLite structural telemetry logs. |
| `/api/history/clear` | `POST` | Clears all telemetry records safely. |
| `/api/report/<job_id>` | `GET` | Renders a printable static HTML forensic report. |

---

## 📁 Repository Structure
```text
AUTHENTIX/
│
├── app.py                      # Flask Application Entrypoint
├── README.md                   # This file
├── requirements.txt            # Python Dependencies
│
├── core_engine/                # Deepfake Machine Learning Logic
│   ├── config.py               # Hyperparameters & Demo Toggle
│   ├── spatial_net.py          # EfficientNet + CBAM
│   ├── frequency_net.py        # Discrete Wavelet Transforms
│   ├── temporal_net.py         # LSTM Video Tracker
│   ├── fusion_model.py         # Multi-Branch Classifier 
│   └── xai_cam.py              # Explainable AI Heatmap Logic
│
├── web_backend/                # Core API Microservices
│   ├── inference_service.py    # Bridges PyTorch to Web endpoints
│   ├── database.py             # SQLite Management
│   ├── utils.py                # File handling & Security
│   └── report_generator.py     # PDF/HTML Export generation
│
├── templates/                  # Frontend Structures
│   ├── index.html              # Interactive Cyberpunk Dashboard
│   └── report_template.html    # Forensic Document Layout
│
└── static/                     # Frontend Assets
    ├── style.css               # Animations and Theming
    ├── script.js               # Async Logic
    └── outputs/                # Grad-CAM storage directory
```

---

## ⚠️ Known Limitations
*   Requires at least one visible frontal face to evaluate properly.
*   Highly compressed social-media videos (e.g., WhatsApp forwards) might degrade the high-frequency spectrum, leading to a marginalized reduction in baseline accuracy. 
*   System requires approximately `4GB VRAM` to handle sequential video inference natively when not utilizing `DEMO_MODE_ACTIVE`.

---

## 🔮 Future Improvements
1.  **Audio-Visual Modality**: Incorporating a Mel-Spectrogram tracker to detect asynchronous audio lip-syncing (Wav2Lip fakes).
2.  **Vision Transformers (ViT)**: Replacing the EfficientNet backbone with ViTs for stronger global context tracking.
3.  **Real-Time Browser Edge Computing**: Exporting the Fusion engine to ONNX to run directly inside the browser payload via WebAssembly.

---
*Developed for University Academic Demonstrations. Copyright © 2026.*
