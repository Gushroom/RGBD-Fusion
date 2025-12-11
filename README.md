# RGB-D Multimodal Fusion for Semantic Segmentation

A PyTorch implementation of multimodal fusion networks combining RGB and Depth data for semantic segmentation, with progressive complexity and multiple merging techniques.

## Participants
Tianhao Li

Tania Shokor

Rua Mohamad

## Description of the Conducted Research 

### Motivation

### Objective

### Dataset

### Methodology

## Project Structure

```
rgbd_fusion/
.
├── batch_train.sh
├── configs
│   ├── attn_fusion
│   ├── early_fusion
│   ├── late_fusion
│   ├── mid_fusion
│   ├── rgb_baseline
│   ├── se_fusion
├── data
│   ├── dataset_merge.py
│   ├── dataset.py
│   ├── __init__.py
├── eval.py
├── experiments
├── losses
│   ├── dice.py
│   └── __pycache__
├── MM5_SEG
│   ├── ANNO_CLASS
│   ├── classes.txt
│   ├── D_FocusN
│   ├── eval_dataset.csv
│   ├── label_mapping.csv
│   ├── RGB1
│   ├── RGB2
│   ├── RGB3
│   ├── RGB4
│   ├── RGB5
│   ├── RGB6
│   ├── RGB7
│   ├── RGB8
│   └── train_dataset.csv
├── models
│   ├── attn_fusion.py
│   ├── earlyfusion.py
│   ├── __init__.py
│   ├── latefusion.py
│   ├── midfusion.py
│   ├── resnet_unet.py
│   ├── sefusion.py
│   └── upblock.py
├── pyproject.toml
├── README.md
├── remap_dataset.py
├── results.txt
├── train.py
├── train.sh
├── utils
│   ├── __init__.py
│   └── metrics.py
└── uv.lock
```

---

## Installation

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip
- CUDA-capable GPU (optional, works on CPU)

### Setup
```bash
# Install dependencies (using UV)
uv sync

# Or with pip
pip install torch torchvision tqdm tensorboard scikit-learn matplotlib seaborn pandas opencv-python pyyaml
```

## Quick Start

### Generate collapsed segmentation dataset
```bash
uv run remap_dataset.py
```

```bash
bash batch_train.sh
```

## Results

