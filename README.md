
---

# CAR-SafetyLogic-Using-CLS

This project implements a deep learning-based safety logic to detect vehicle movement anomalies in industrial production lines. It utilizes a **CNN-LSTM** architecture to classify whether a vehicle is moving normally or has stopped (Emergency) by analyzing a sequence of image frames.

## 1. Project Overview

In a production line, vehicles must move a certain distance within a specific timeframe. If a vehicle remains stationary across  frames, the system triggers an emergency signal.

* **Backbone:** EfficientNet-B0 (Spatial Feature Extraction)
* **Temporal Model:** LSTM (Temporal Sequence Analysis)
* **Target Hardware:** NVIDIA RTX 3080 (Dual GPU optimized)

## 2. Directory Structure

```text
CAR-SafetyLogic-Using-CLS/
├── Dataset/
│   ├── normal/        # Folders containing moving vehicle frame sequences
│   │   ├── seq_01/    # frame_001.jpg, frame_002.jpg ...
│   │   └── ...
│   └── emergency/     # Folders containing stationary vehicle frame sequences
│       ├── seq_01/    # frame_001.jpg, frame_002.jpg ...
│       └── ...
├── model.py           # EfficientNet-B0 + LSTM Architecture
├── train.py           # Dataset loader and Training pipeline
└── inference.py       # Sequence-based inference on frame directories

```

## 3. Environment Setup

### Prerequisites

* Windows 10/11
* NVIDIA RTX 3080 (CUDA 12.x confirmed)
* Anaconda or Miniconda

### Installation

```bash
# Create and activate conda environment
conda create -n car_safety python=3.9 -y
conda activate car_safety

# Install PyTorch for CUDA 12.x (Optimized for RTX 30-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install opencv-python efficientnet_pytorch pandas scikit-learn matplotlib Pillow

```

## 4. Component Details

### `model.py`

Defines the `CarSafetyModel` class. It extracts spatial features from each frame using a pre-trained **EfficientNet-B0** and passes the feature sequence into a **2-layer LSTM** to capture motion patterns.

### `train.py`

Handles data loading and model training.

* **FrameSequenceDataset:** Groups individual image frames into a sequence of length .
* **Sliding Window:** Creates training samples by sliding across the frame list to maximize data efficiency.
* **Optimization:** Uses Adam optimizer and CrossEntropyLoss for binary classification.

### `inference.py`

Performs status prediction on a target directory.

* Maintains a temporal buffer of 10 frames.
* Performs "sliding window" inference to provide near real-time monitoring.
* Outputs **NORMAL** or **EMERGENCY** based on the predicted state.

## 5. Usage

### Training

Place your labeled sequence data in the `Dataset/` folder and run:

```bash
python train.py

```

This will save the best model weights as `car_safety_model.pth`.

### Inference

Point the `test_folder` path in `inference.py` to your target frame directory and run:

```bash
python inference.py

```

---
