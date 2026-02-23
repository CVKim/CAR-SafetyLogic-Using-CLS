This is a professional **README.md** tailored for your project. It explains the architecture, environment setup, and how to use the scripts you've developed for your RTX 3080 environment.

---

# Vehicle Safety Monitoring System (Vision AI)

This project implements a deep learning-based safety logic to detect vehicle movement anomalies. It utilizes a **CNN-LSTM** architecture to classify whether a vehicle is moving normally or has stopped (Emergency) by analyzing a sequence of image frames.

## 1. Project Overview

In a production line, vehicles must move a certain distance within a specific timeframe. If a vehicle remains stationary across frames, the system triggers an emergency signal.

- **Backbone:** EfficientNet-B0 (Feature Extraction)
- **Temporal Model:** LSTM (Sequence Analysis)
- **Input:** Sequential image frames ()

## 2. Directory Structure

```text
CarSafetyProject/
├── Dataset/
│   ├── normal/        # Subfolders with normal movement sequences
│   │   ├── seq1/      # img1.jpg, img2.jpg...
│   │   └── seq2/
│   └── emergency/     # Subfolders with stopped/abnormal sequences
│       ├── seq1/
│       └── seq2/
├── model.py           # Model Architecture (EfficientNet + LSTM)
├── train.py           # Training Script
└── inference.py       # Real-time/Offline Inference Script

```

## 3. Environment Setup

### Prerequisites

- Windows 10/11
- NVIDIA RTX 3080 (Dual GPU supported)
- Anaconda or Miniconda

### Installation

```bash
# Create and activate conda environment
conda create -n car_safety python=3.9 -y
conda activate car_safety

# Install PyTorch for CUDA 12.x (Optimized for RTX 30x0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install opencv-python efficientnet_pytorch pandas scikit-learn matplotlib Pillow

```

## 4. Component Details

### `model.py`

Defines the `CarSafetyModel` class. It extracts spatial features from each frame using a pre-trained **EfficientNet-B0** and passes the feature sequence into a **2-layer LSTM** to capture motion patterns.

### `train.py`

Handles data loading and model training.

- **FrameSequenceDataset:** A custom PyTorch Dataset that groups images from subfolders into sequences.
- **Sliding Window:** It creates training samples by sliding across the frame list.
- **Optimization:** Uses Adam optimizer and CrossEntropyLoss.

### `inference.py`

Performs status prediction on a target folder of images.

- It maintains a buffer of frames and performs a "sliding window" inference.
- Outputs either **NORMAL** or **EMERGENCY** based on the temporal change between frames.

## 5. Usage

### Training

Place your labeled data in the `Dataset/` folder and run:

```bash
python train.py

```

### Inference

Point the script to your test frame directory in `inference.py` and run:

```bash
python inference.py

```
