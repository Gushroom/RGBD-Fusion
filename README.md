# RGB-D Multimodal Fusion for Semantic Segmentation

A PyTorch implementation of multimodal fusion networks combining RGB and Depth data for semantic segmentation, with progressive complexity from CNN baselines to Transformer-based fusion.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Analysis](#dataset-analysis)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)

---

## Project Overview

### Goal
Experiment with multimodal neural network fusion techniques using RGB and Depth modalities for semantic segmentation. The project follows a progressive approach: starting with simple baselines and advancing to transformer-based fusion.

### Why Segmentation Over Classification?
**Initial Approach**: Started with image classification (30 classes, 324 images)
- **Problem**: Model achieved 92% accuracy by epoch 17, then plateaued with severe overfitting (100% train, 92% eval)
- **Root Cause**: Dataset too small for classification task, no room for meaningful fusion experiments

**Current Approach**: Switched to semantic segmentation (32 classes including background)
- **Advantages**:
  - ~307K predictions per image (480×640 pixels) vs 1 label
  - Much harder task, won't overfit as quickly
  - Better showcase of RGB-Depth fusion benefits
  - Depth information crucial for boundaries, occlusions, and material properties

### Hardware Requirements
- **Recommended**: Google Colab Free (T4 GPU) - perfectly adequate
- **Training Time**: 2-3 hours per baseline model
- **Why Colab is sufficient**: Small dataset (324 images), lightweight models (11-22M parameters)

---

## Dataset Analysis

### MM5_ALIGNED Dataset

**Location**: `~/Downloads/MM5_ALIGNED/` or `MM5_ALIGNED/`

**Statistics**:
- **Total Images**: 324 (248 train, 76 eval)
- **Image Size**: 480×640 pixels (resized to 224×224 for training)
- **Segmentation Classes**: 32 (including background)
- **Modalities Available**: RGB (8 lighting variants), Depth (5 variants), Thermal, Infrared, UV

**Class Distribution** (from CSV analysis):
```
Most common: Mandarin (57), Apple Green (54), Lemon (49)
Least common: Cup (3), Apple (9), Onion (12)
Imbalanced: Some classes have 10x more samples than others
```

### Key Findings

#### 1. **Data Format Quirks**
- Split files (`list_train_c1.txt`) contain **only image IDs**, not paths
- `train_dataset.csv` and `eval_dataset.csv` contain the actual metadata
- Multi-object images exist (e.g., "Mandarin,Lemon") - we take first subcategory only
- `label_mapping.csv` had malformed data (JSON + CSV mixed) - handled in dataset loader

#### 2. **Modality Choices**
**RGB**: Using `RGB1/` (could experiment with RGB2-RGB8 for different lighting)

**Depth**: Using `D_FocusN/` (normalized, focused depth)
- Verified range: [18, 234] uint8
- Alternative: `D16/` with range [0, 1402] uint16 (higher precision, not tested yet)
- Normalization: `(depth - 18) / (234 - 18)` → [0, 1]

**Annotations**:
- `ANNO_CLASS/`: Grayscale images with pixel values [0-31] representing class indices
- `ANNO_VIS_CLASS/`: Color-coded visualizations (for humans, not used in training)
- Background = class 0, objects = classes 1-31

#### 3. **Class Mapping**
31 object classes + background:
```
1: Lemon, 2: Lemon Bad, 3: Lemon Fake
4: Mirror, 5: Bowl
6: Mandarin, 7: Mandarin Bad, 8: Mandarin Fake
... (see label_mapping.csv for full list)
31: Carrot Fake
```

Classes include quality variants (Good/Bad/Fake) and states (Half, Peel), making RGB-Depth fusion especially valuable for distinguishing:
- Real vs Fake fruits (depth reveals 3D structure)
- Good vs Bad quality (depth shows deformations)
- Mirrors vs real objects (depth sees through reflections)

---

## Project Structure

```
rgbd_fusion/
├── README.md                      # This file
├── SEGMENTATION_QUICKSTART.md     # Quick training guide
├── pyproject.toml                 # UV project config
├── uv.lock
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # Classification dataset (deprecated)
│   └── segmentation_dataset.py    # Segmentation dataset (ACTIVE)
│
├── models/
│   ├── __init__.py
│   ├── resnet_fusion.py           # Classification models (deprecated)
│   └── resnet_unet.py             # Segmentation models (ACTIVE)
│       ├── ResNetUNet             # Phase 1: RGB-only baseline
│       ├── ResNetUNetEarlyFusion  # Phase 2: RGB+D early fusion
│       └── (Transformer fusion)   # Phase 3: To be implemented
│
├── configs/
│   ├── seg_rgb_baseline.yaml      # Phase 1 config
│   ├── seg_early_fusion.yaml      # Phase 2 config
│   └── (transformer config)       # Phase 3: To be implemented
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                 # Metrics & visualization
│
├── train.py                       # Classification training (deprecated)
├── train_segmentation.py          # Segmentation training (ACTIVE)
├── main.py                        # Testing/debugging script
│
├── experiments/                   # Auto-generated, training outputs
│   ├── seg_rgb_baseline/
│   │   ├── seg_rgb_baseline_best.pth
│   │   └── seg_rgb_baseline_epoch*.pth
│   └── seg_early_fusion/
│
└── MM5_ALIGNED/                   # Dataset (not in repo)
    ├── RGB1/, RGB2/, ..., RGB8/
    ├── D_FocusN/, D16/, D_Focus/
    ├── ANNO_CLASS/                # Segmentation masks (used)
    ├── ANNO_VIS_CLASS/            # Visualizations (not used)
    ├── train_dataset.csv
    ├── eval_dataset.csv
    ├── label_mapping.csv
    └── classes.txt
```

---

## Installation

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip
- CUDA-capable GPU (optional, works on CPU)

### Setup
```bash
# Clone/navigate to project
cd rgbd_fusion

# Install dependencies (using UV)
uv sync

# Or with pip
pip install torch torchvision tqdm tensorboard scikit-learn matplotlib seaborn pandas opencv-python pyyaml
```

## Quick Start

```bash
bash train.sh
```

