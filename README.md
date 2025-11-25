# RGB-D Multimodal Fusion for Semantic Segmentation

A PyTorch implementation of multimodal fusion networks combining RGB and Depth data for semantic segmentation, with progressive complexity and multiple merging techniques.

## Project Structure

```
rgbd_fusion/
├── configs
│   ├── cls_rgb_baseline.yaml
│   ├── depth_baseline.yaml
│   ├── rgbd_fusion.yaml
│   ├── seg_early_fusion.yaml
│   └── seg_rgb_baseline.yaml
├── data
│   ├── __init__.py
│   ├── dataset_cls.py
│   └── dataset_seg.py
├── eval_seg.py
├── experiments
│   ├── seg_early_fusion_dice
│   ├── seg_early_fusion_light(3)
│   ├── seg_early_fusion_light(5)
│   ├── seg_rgb_baseline
│   ├── seg_rgb_light(3)
│   └── seg_rgb_light(5)
├── losses
│   └── dice.py
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
│   ├── __init__.py
│   ├── resnet_fusion.py
│   └── resnet_unet.py
├── pyproject.toml
├── README.md
├── remap_dataset.py
├── train_cls.py
├── train_seg.py
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
bash train.sh
```

