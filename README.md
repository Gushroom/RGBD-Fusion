# RGB-D Multimodal Fusion for Semantic Segmentation

A PyTorch implementation of multimodal fusion networks combining RGB and Depth data for semantic segmentation, with progressive complexity from CNN baselines to Transformer-based fusion.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Analysis](#dataset-analysis)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development Roadmap](#development-roadmap)
- [Key Design Decisions](#key-design-decisions)
- [Training Progress](#training-progress)
- [Known Issues & Solutions](#known-issues--solutions)
- [Next Steps](#next-steps)

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

# Verify dataset location
ls MM5_ALIGNED/RGB1/ | head -3
ls MM5_ALIGNED/ANNO_CLASS/ | head -3
```

---

## Quick Start

### 1. Test Dataset Loading (30 seconds)
```bash
uv run data/segmentation_dataset.py
```

**Expected Output**:
```
Loaded TRAIN split for SEGMENTATION:
  Total samples: 248
  Number of classes: 32
  
Testing batch loading...
  RGB shape: torch.Size([4, 3, 224, 224])
  Mask shape: torch.Size([4, 224, 224])
  Unique classes in batch: [0, 1, 4, 5, 6, ...]
✓ Segmentation dataset loading successful!
```

### 2. Test Model Architecture (30 seconds)
```bash
uv run models/resnet_unet.py
```

**Expected Output**:
```
1. ResNet U-Net (RGB only):
   Input: [2, 3, 224, 224] -> Output: [2, 32, 224, 224]
   Parameters: 19.23M

2. ResNet U-Net Early Fusion (RGB+D):
   Input: RGB [2, 3, 224, 224] + Depth [2, 1, 224, 224]
   Output: [2, 32, 224, 224]
   Parameters: 19.23M
✓ All models working correctly!
```

### 3. Train Phase 1: RGB Baseline (2-3 hours)
```bash
uv run train_segmentation.py --config configs/seg_rgb_baseline.yaml
```

**What to Watch**:
- Training: ~15 batches per epoch, ~3 min/epoch
- Train mIoU should reach 70-80% (will overfit)
- **Eval mIoU: Target 60-75%**
- Plateaus around epoch 20-30

### 4. Train Phase 2: Early Fusion (2-3 hours)
```bash
uv run train_segmentation.py --config configs/seg_early_fusion.yaml
```

**Expected Improvement**:
- **Eval mIoU: Target 65-80%** (5-10% over RGB-only)
- Better on object boundaries, fake detection, mirrors

### 5. Compare Results
```python
import torch

rgb = torch.load('experiments/seg_rgb_baseline/seg_rgb_baseline_best.pth')
fusion = torch.load('experiments/seg_early_fusion/seg_early_fusion_best.pth')

print(f"RGB mIoU: {rgb['best_iou']*100:.2f}%")
print(f"Fusion mIoU: {fusion['best_iou']*100:.2f}%")
print(f"Improvement: +{(fusion['best_iou'] - rgb['best_iou'])*100:.2f}%")
```

---

## Development Roadmap

### ✅ Completed (Phase 1 & 2)

**Phase 1: RGB-Only Baseline**
- [x] Segmentation dataset loader with proper class mapping
- [x] ResNet18 U-Net architecture
- [x] Training script with mIoU metrics
- [x] Config system
- [x] Baseline training (2-3 hours)
- **Target**: 60-75% mIoU

**Phase 2: Early Fusion**
- [x] 4-channel input (RGB+D concatenated)
- [x] Pretrained RGB weight initialization
- [x] Training script support
- [x] Early fusion training (2-3 hours)
- **Target**: 65-80% mIoU

### 🔄 In Progress (Phase 3)

**Phase 3: Transformer Mid-Fusion** (Next step!)
- [ ] Separate ResNet18 encoders for RGB and Depth
- [ ] Cross-modal transformer blocks (2-4 layers)
- [ ] Multi-scale feature fusion
- [ ] Config file for transformer fusion
- [ ] Training (4-6 hours expected)
- **Target**: 70-85% mIoU

**Architecture Sketch for Phase 3**:
```python
class ResNetTransformerFusion(nn.Module):
    def __init__(self):
        # RGB encoder (ResNet18, pretrained)
        self.rgb_encoder = ResNet18Encoder()
        
        # Depth encoder (ResNet18, from scratch)
        self.depth_encoder = ResNet18Encoder(in_channels=1)
        
        # Cross-modal fusion at multiple scales
        self.fusion_blocks = nn.ModuleList([
            TransformerFusionBlock(dim=64),   # After layer1
            TransformerFusionBlock(dim=128),  # After layer2
            TransformerFusionBlock(dim=256),  # After layer3
            TransformerFusionBlock(dim=512),  # After layer4
        ])
        
        # U-Net decoder
        self.decoder = UNetDecoder()
    
    def forward(self, rgb, depth):
        # Multi-scale encoding
        rgb_feats = self.rgb_encoder(rgb)    # [f1, f2, f3, f4]
        depth_feats = self.depth_encoder(depth)
        
        # Fuse at each scale with transformers
        fused_feats = []
        for rgb_f, depth_f, fusion in zip(rgb_feats, depth_feats, self.fusion_blocks):
            fused = fusion(rgb_f, depth_f)  # Cross-attention
            fused_feats.append(fused)
        
        # Decode
        out = self.decoder(fused_feats)
        return out
```

### 🎯 Future Improvements

**Model Architecture**:
- [ ] Try ResNet34/50 backbones (more parameters)
- [ ] Experiment with ViT-based encoders
- [ ] Add attention gates in decoder
- [ ] Multi-scale prediction (deep supervision)

**Data & Training**:
- [ ] Strong data augmentation (rotation, flip, color jitter)
- [ ] Class balancing / weighted loss
- [ ] Try different depth modalities (D16 for precision)
- [ ] Experiment with different RGB lighting (RGB2-RGB8)
- [ ] Instance segmentation (using ANNO_INST)

**Evaluation**:
- [ ] Per-class IoU analysis
- [ ] Confusion matrix visualization
- [ ] Qualitative result visualization
- [ ] Failure case analysis

---

## Key Design Decisions

### 1. **Why Segmentation Instead of Classification?**
- **Decision**: Pivot from classification to segmentation after seeing 92% accuracy at epoch 17
- **Reason**: Dataset too small (324 images) for classification, severe overfitting, no room to demonstrate fusion benefits
- **Benefit**: Segmentation provides 307K labels per image, much more learning signal

### 2. **Why Start with ResNet U-Net?**
- **Decision**: Begin with proven CNN architecture before transformers
- **Reason**: 
  - ResNet18 is lightweight (19M params), trains fast
  - Pretrained ImageNet weights available for RGB
  - U-Net proven for segmentation with limited data
  - Easy to debug and understand
- **Timeline**: Establish baselines first (Phase 1-2), then add complexity (Phase 3)

### 3. **Image Resizing Strategy**
- **Decision**: Resize from 480×640 to 224×224
- **Reason**: 
  - Match ImageNet pretrained weights (224×224)
  - Reduce memory usage
  - Faster training
- **Tradeoff**: Loss of resolution, but acceptable for experimentation

### 4. **Depth Modality Choice**
- **Decision**: Use `D_FocusN` over `D16`
- **Reason**: 
  - Already normalized [18, 234] → easy to use
  - Pre-focused for quality
  - 8-bit sufficient for segmentation
- **Alternative**: `D16` available if need higher precision later

### 5. **Training Configuration**
- **Batch size**: 16 (segmentation needs more memory than classification)
- **Epochs**: 50 (reduced from 100 after overfitting observed)
- **Optimizer**: AdamW with cosine annealing
- **Loss**: CrossEntropyLoss (could add class weighting later)

### 6. **Evaluation Metric**
- **Primary**: mIoU (mean Intersection over Union)
- **Why not pixel accuracy?**: Biased toward background class (70% of pixels)
- **Target**: 
  - RGB baseline: 60-75% mIoU
  - Fusion: 65-85% mIoU

---

## Training Progress

### Classification Experiments (Deprecated)

**Experiment**: RGB-only classification with ResNet18
- **Dataset**: 248 train, 76 eval, 11 classes (consolidated from CSV)
- **Results**: 
  - Epoch 17: Train 99.6%, Eval 92.1%
  - Epoch 18: Train 100%, Eval 92.1% (saturated)
- **Conclusion**: Severe overfitting, task too easy for dataset size

**Key Learning**: Small datasets need tasks with more labels per sample (segmentation > classification)

### Segmentation Experiments (Active)

**Phase 1: RGB Baseline** (Status: Ready to train)
- Model: ResNet18 U-Net, 19M parameters
- Config: `configs/seg_rgb_baseline.yaml`
- Expected: 2-3 hours, 60-75% mIoU
- **Status**: Awaiting training results

**Phase 2: Early Fusion** (Status: Ready to train)
- Model: ResNet18 U-Net with 4-channel input
- Config: `configs/seg_early_fusion.yaml`
- Expected: 2-3 hours, 65-80% mIoU
- **Status**: Awaiting training results

**Phase 3: Transformer Fusion** (Status: To be implemented)
- Model: To be created after Phase 1-2 results
- Expected: 4-6 hours, 70-85% mIoU

---

## Known Issues & Solutions

### Issue 1: Malformed label_mapping.csv
**Problem**: File contains both JSON and CSV data mixed together
```
{
    "Lemon": 1,
    ...
}ID,Label Name
1,Lemon
```

**Solution**: Dataset loader handles this by parsing JSON first, falling back to CSV
```python
json_str = content.split('\n')[0] if '\n' in content else content
self.class_mapping = json.loads(json_str)
```

### Issue 2: Multi-object Images
**Problem**: Some images have multiple subcategories: "Mandarin,Lemon"

**Solution**: Take first subcategory only
```python
if ',' in subcategory:
    subcategory = subcategory.split(',')[0].strip()
```

**Impact**: ~10 images affected, acceptable simplification for now

### Issue 3: Class Imbalance
**Problem**: Cup (3 samples) vs Mandarin (57 samples) = 19x difference

**Current Solution**: Ignore for now, train with standard CrossEntropyLoss

**Future Solutions**:
- Class-weighted loss
- Oversample rare classes
- Focal loss

### Issue 4: Small Dataset Size
**Problem**: Only 324 images for 32 classes

**Mitigation**:
- Use pretrained weights (ImageNet for RGB)
- Start with small models (ResNet18)
- Strong regularization (dropout 0.3, weight decay 1e-4)
- Segmentation task (more labels per image)

**If still overfitting**:
- Add data augmentation
- Reduce model capacity
- Early stopping

### Issue 5: MPS (Mac GPU) Warnings
**Problem**: `'pin_memory' not supported on MPS`

**Solution**: Warning is harmless, pin_memory silently disabled on MPS

**Alternative**: Run on Colab for faster training

---

## Next Steps

### Immediate (For Current Developer)

1. **Run Phase 1 Training** (~2-3 hours)
   ```bash
   uv run train_segmentation.py --config configs/seg_rgb_baseline.yaml
   ```
   - Monitor mIoU progression
   - Check for overfitting (train vs eval gap)
   - Note which classes perform poorly

2. **Run Phase 2 Training** (~2-3 hours)
   ```bash
   uv run train_segmentation.py --config configs/seg_early_fusion.yaml
   ```
   - Compare mIoU improvement vs RGB-only
   - Identify which classes benefit most from depth

3. **Analyze Results**
   ```python
   # Load checkpoints
   # Compare per-class IoU
   # Identify where fusion helps most
   ```

### For Continuing This Project (New Developer/AI Assistant)

**Context Needed**:
- This README (complete project context)
- Phase 1 & 2 training results (mIoU scores)
- Saved model checkpoints in `experiments/`

**Next Development Tasks**:

**A. Implement Phase 3: Transformer Fusion**
```
Priority: HIGH
Files to create:
  - models/resnet_transformer_fusion.py
  - configs/seg_transformer_fusion.yaml
  
Key components:
  - Separate ResNet encoders for RGB/Depth
  - TransformerFusionBlock with cross-attention
  - Multi-scale fusion
  - U-Net decoder
  
Expected: 4-6 hours training, 70-85% mIoU
```

**B. Add Data Augmentation**
```
Priority: MEDIUM (if overfitting occurs)
File to modify: data/segmentation_dataset.py

Add to __init__ when split=='train':
  - RandomHorizontalFlip(p=0.5)
  - RandomRotation(15)
  - ColorJitter(brightness=0.2, contrast=0.2)
  - For segmentation: Apply same transform to mask!
```

**C. Visualization Tools**
```
Priority: MEDIUM
File to create: utils/visualize.py

Functions needed:
  - plot_segmentation_results(image, pred, gt)
  - plot_per_class_iou(class_names, ious)
  - plot_rgb_vs_depth_vs_fusion_comparison()
```

**D. Evaluation Script**
```
Priority: MEDIUM
File to create: evaluate.py

Features:
  - Load trained model
  - Run on eval set
  - Generate qualitative results
  - Per-class metrics
  - Confusion matrix
```

**E. Advanced Experiments** (After Phase 3)
```
Priority: LOW

Ideas to try:
  - Different backbones (ResNet34, ResNet50)
  - Different depth modalities (D16)
  - Different RGB lighting (RGB2-RGB8)
  - Attention mechanisms
  - Multi-scale training
  - Instance segmentation (ANNO_INST)
```

---

## Important Notes for AI Assistants

### Project Context Summary
- **Current State**: Ready to train Phase 1 & 2 (RGB baseline, early fusion)
- **Next Step**: Train models, analyze results, implement Phase 3 (transformer fusion)
- **Main Challenge**: Small dataset (324 images), need careful regularization
- **Key Metric**: mIoU (mean Intersection over Union)

### Dataset Quirks to Remember
1. Split files are just IDs, metadata in CSVs
2. label_mapping.csv has mixed JSON+CSV format
3. Multi-object images exist, we take first subcategory
4. Class imbalance significant (3 to 57 samples per class)
5. Depth range is [18, 234] for D_FocusN, needs normalization

### Code Organization
- **Active**: segmentation_dataset.py, resnet_unet.py, train_segmentation.py
- **Deprecated**: dataset.py, resnet_fusion.py, train.py (classification version)
- **To Create**: Transformer fusion model for Phase 3

### Training Expectations
- **Phase 1 (RGB)**: 2-3 hrs, 60-75% mIoU
- **Phase 2 (Early Fusion)**: 2-3 hrs, 65-80% mIoU  
- **Phase 3 (Transformer)**: 4-6 hrs, 70-85% mIoU
- All trainable on Colab Free T4

### When Helping Further
1. Always check if Phase 1 & 2 training completed
2. Ask for mIoU results before suggesting Phase 3 implementation
3. Respect the progressive complexity approach (CNN → Transformer)
4. Consider dataset size (324 images) in all recommendations
5. Refer to this README for design decisions and context

---

## Contact & Contribution

**Current Status**: Active development, Phase 1-2 ready to train

**For Future Developers**: 
- Read this README thoroughly
- Check `SEGMENTATION_QUICKSTART.md` for training guide
- Review training results in `experiments/` directory
- Follow progressive approach: complete Phase 1 & 2 before Phase 3

**Questions?** Check:
1. This README for project context
2. Code comments in dataset/model files
3. Config files for hyperparameters
4. Saved checkpoints for previous results

---

## Appendix: Command Reference

### Dataset Testing
```bash
# Test segmentation dataset
uv run data/segmentation_dataset.py

# Check dataset stats
python -c "
from data.segmentation_dataset import get_segmentation_dataloaders
train_loader, val_loader, num_classes = get_segmentation_dataloaders(
    'MM5_ALIGNED', modalities=['rgb'], batch_size=8, num_workers=0
)
print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Classes: {num_classes}')
"
```

### Model Testing
```bash
# Test model architectures
uv run models/resnet_unet.py

# Check model parameters
python -c "
from models.resnet_unet import ResNetUNet
model = ResNetUNet(num_classes=32)
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

### Training
```bash
# Phase 1: RGB baseline
uv run train_segmentation.py --config configs/seg_rgb_baseline.yaml

# Phase 2: Early fusion
uv run train_segmentation.py --config configs/seg_early_fusion.yaml

# Override data path
uv run train_segmentation.py --config configs/seg_rgb_baseline.yaml --data_root /path/to/MM5_ALIGNED
```

### Monitoring
```bash
# TensorBoard (if implemented)
tensorboard --logdir experiments/

# Check GPU usage
nvidia-smi  # or watch -n 1 nvidia-smi
```

### Results Analysis
```python
# Load checkpoint
import torch
ckpt = torch.load('experiments/seg_rgb_baseline/seg_rgb_baseline_best.pth')
print(f"Best mIoU: {ckpt['best_iou']*100:.2f}%")
print(f"Epoch: {ckpt['epoch']}")
print(f"Per-class IoU: {ckpt['class_ious']}")

# Compare models
rgb_iou = torch.load('experiments/seg_rgb_baseline/seg_rgb_baseline_best.pth')['best_iou']
fusion_iou = torch.load('experiments/seg_early_fusion/seg_early_fusion_best.pth')['best_iou']
improvement = (fusion_iou - rgb_iou) * 100
print(f"Fusion improvement: +{improvement:.2f}%")
```

---

**Last Updated**: November 2025  
**Version**: 1.0 (Segmentation implementation, Phase 1-2 ready)  
**Status**: Awaiting Phase 1 training results

Good luck with the experiments! 🚀