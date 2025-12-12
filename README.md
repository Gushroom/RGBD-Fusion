# RGB-D Multimodal Fusion for Semantic Segmentation

A PyTorch implementation of multimodal fusion networks combining RGB and Depth data for semantic segmentation, with progressive complexity and multiple merging techniques.

## Participants
Tianhao Li

Tania Shokor

Rua Mohamad

## Description of the Conducted Research 

### Motivation
Multimodal perception is essential for reliable scene understanding in real-world applications such as robotics, autonomous systems, inspection and agriculture. It is often insufficient to rely solely on RGB alone, since it is highly sensitive to lighting variations, color ambiguity and occlusions.

### Objective
#### Goal
Evaluate how different RGB-D fusion techniques affect semantic segmentation performance on the MM5 dataset.

#### Hypothesis
Mid-level, late-level and transformer-based RGB-D fusion methods will outperform early fusion and unimodal baselines.



### Why We Switched From Classification to Segmentation
The initial experiment attempted image classificatin using 324 images across 30 categories. The model overfitted severely:
* Train accuracy: 100%
* Validation accuracy: 92%

Reasons for abondoning classification:
* The dataset is too small for a 30-class classification task
* Class imabalance further harmed generalization
* Fusion could not be meaningfully evaluated because RGB alone already solved the task

Thus, we switched to segmentation, since it provides pixel-level supervision, making it more suitable to study modality fusion.

### Dataset
The [MM5_ALIGNED](https://github.com/martinbrennernz/MM5-Dataset) dataset is a recent dataset contaaining synchronized RGB, Depth, Thermal, UV and NIR images. It is well suited for fusion research because:
* Provides lighting variations (8 lighting setups)
* Depth data provides high-quality geometric cues for testing fusion effectiveness
* Includes imperfect, damaged and fake variants of real-world fruits and objects
* Clear pixel-wise segmentation masks are provided for all images

#### Dataset Specifications
* Total images: 324
  * 248 train / 76 val
* Modalities used:
  * RGB1-RGB8
  * Depth (D_FocusN)
* Resolution: 480x640 &rarr; 224x224
* Segmentation classes: 32 (31 object types + background)
* Object types: Good / Bad / Fake variants, multi-class and imbalanced

### Pipeline Overview
#### Preprocessing
* Resize to 224x224
* Normalize RGB using ImageNet mean/std
* Normalize Depth to [0, 1] range
* Load segmentation masks (0-31 pixel values)

#### Model Backbone
All experiments use a UNet encoder-decoder architecture built in PyTorch

#### Evaluation Protocol
* All models run across 8 RGB illumination variants
* Metric: mIoU

### Methodology
1. RGB Baseline
The baseline model is a ResNet18-based UNet, using only RGB images as input. The ResNet-18 encoder extracts hierarchical features from the RGB input, while a symmetric decoder progressively upsamples these features to produce per-pixel semantic predictions. Skip connections are employed between encoder and decoder layers to preserve spatial details.

Key features:
* Standard ResNet-18 encoder with pretrained ImageNet weights
* UNet style decoder with upsampling blocks (UpBlock)
* Final classification layer predicts the desired number of semantic classes.

This baseline serves as a reference to evaluate the gains from the addition of depth data. 

3. Early Fusion
The early fusion model concatenates the RGB and depth channels at the input, forming a 4-channel input (RGB+D). The fused input is fed into a modified ResNet18 encoder.

Key features:
* The first convolutional layer is adapted to accept 4 channels
* If using pretrained weights, RGB weights are copied, and the depth channel is initialized as the average of RGB weights.
* Encoder and decoder are otherwise similar to the RGB-only baseline.

Early fusion uses the depth information at the very start, potentially helping the network learn complementary features jointly with RGB.

5. Mid-level Fusion
The mid-level fusion model merges RGB and depth features after intermediate layers in the encoder (in layer3). Seperate ResNet18 encoders are used for RGB and depth, and the resulting features are fused at multiple scales before being fed into a shared decoder.

Key features:
* Independent RGB and depth encoders
* Fusion occurs after intermediate layers (at layer3) using concatenation
* Decoder uses skip connections from the RGB encoder and fused features.

Mid-level fusion allows the network to first learn modality-specific features, and then to combine complementary representations at a higher semantic level.

7. Late Fusion
The late fusion model involves two fully independent RGB and depth U-Nets. Each of which produce their own predictions, which is fused at the end, using averaging.

Key features:
* Two separate ResNetUNet models for RGB and Depth
* Depth encoder initialized with grayscale weights derived from the RGB model
* Prediction logits are merged by averaging them

Late fusion is resilient to modality-specific noise since the full capacity of each modality is preserved and only the final predictions are combined.

9. Squeeze-and-Excitation Fusion
The SE fusion model involves adaptive gating and multi-scale fusion. Such model is able to learn to dynamically weight RGB vs depth based on featue quality, fuse information at multiple scales, and weight skip connections from both modalities by attention and confidence.

Key features:
* RGB and depth features are fuused at multiple encoder scales using the AdaptiveGatedFusion module
* Depth confidence is estimated per pixel to weight reliable regions higher
* Both channel attention and spatial attention refine fused features

SE fusion provides feature adaptivity, allowing the network to trust depth where it is reliable, while relying more on RGB elewhere.


11. Attention-based Fusion
The attention-based fusion model includes a lightweight cross-attention mechanism between RGB and depth features at a higher semantic layer (layer3).

Key features:
* Features from RGB and depth are projected and passed through a small multi-head attention block (CrossAttention)
* The fused representation is used in the decoder alongside standard skip connections

Cross-attention allows the network to selectively highlight complementary information from depth and RGB features. This approach explicitly models interactions between the two modalities, improving segmentation in challenging regions.

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
![fusion heatmap](./fusion_heatmap.png)
![fusion performance](./fusion_performance.png)
![fusion radar](./fusion_radar.png)
![fusion summary](./fusion_summary.png)

## Reference List
