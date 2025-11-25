import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json


class RGBDSegmentationDataset(Dataset):
    """Dataset for RGB + Depth semantic segmentation"""
    
    def __init__(
        self,
        root_dir,
        split='train',  # 'train' or 'eval'
        modality='rgbd',  # 'rgb', 'depth', or 'rgbd'
        rgb_dir='RGB5',
        depth_dir='D_FocusN',
        anno_dir='ANNO_CLASS',
        transform=None,
        img_size=(224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.rgb_dir = self.root_dir / rgb_dir
        self.depth_dir = self.root_dir / depth_dir
        self.anno_dir = self.root_dir / anno_dir
        self.transform = transform
        self.split = split
        self.img_size = img_size
        
        # Load class mapping
        self.load_class_mapping()
        
        # Load CSV data to get valid IDs
        csv_file = 'train_dataset.csv' if split == 'train' else 'eval_dataset.csv'
        self.df = pd.read_csv(self.root_dir / csv_file)
        
        # Build samples list (just IDs, labels come from segmentation masks)
        self.samples = self._build_samples()
        
        print(f"\n{'='*60}")
        print(f"Loaded {split.upper()} split for SEGMENTATION:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of classes: {self.num_classes} (including background)")
        print(f"  Modality: {modality}")
        print(f"  Image size: {img_size}")
        print(f"{'='*60}\n")
    
    def load_class_mapping(self):
        """Load class ID-to-name mapping from CSV"""

        csv_path = self.root_dir / "label_mapping.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"label_mapping.csv not found at: {csv_path}")

        df = pd.read_csv(csv_path)

        # Create mapping: name → ID
        self.class_mapping = dict(zip(df["Label Name"], df["ID"]))

        # Number of classes (include background = 0)
        self.num_classes = max(self.class_mapping.values()) + 1

        # Class names sorted by ID, with background at index 0
        self.class_names = ["Background"] + [
            name for name, _ in sorted(self.class_mapping.items(), key=lambda x: x[1])
        ]

        print(f"Loaded {self.num_classes} classes from CSV:")
        for i, name in enumerate(self.class_names[:10]):
            print(f"  {i}: {name}")
        if len(self.class_names) > 10:
            print(f"  ... and {len(self.class_names)-10} more")
    
    def _build_samples(self):
        """Build list of valid image IDs"""
        samples = []
        
        for idx, row in self.df.iterrows():
            image_id = row['ID']
            
            # Check if annotation exists
            anno_path = self.anno_dir / f"{image_id}.png"
            if anno_path.exists():
                samples.append(image_id)
        
        return samples
    
    def _load_image(self, image_id, is_depth=False):
        """Load and preprocess image by ID"""
        img_filename = f"{image_id}.png"
        
        if is_depth:
            full_path = self.depth_dir / img_filename
            
            if not full_path.exists():
                raise FileNotFoundError(f"Depth image not found: {full_path}")
            
            img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load depth: {full_path}")
            
            # Resize
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), 
                           interpolation=cv2.INTER_LINEAR)
            
            # Normalize D_FocusN: uint8 [18, 234] -> float32 [0, 1]
            img = img.astype(np.float32)
            img = (img - 18.0) / (234.0 - 18.0)
            img = np.clip(img, 0, 1)
            
            # Add channel dimension: [H, W] -> [1, H, W]
            img = img[np.newaxis, :, :]
        else:
            # Load RGB
            full_path = self.rgb_dir / img_filename
            
            if not full_path.exists():
                raise FileNotFoundError(f"RGB image not found: {full_path}")
            
            img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load RGB: {full_path}")
            
            # Resize
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                           interpolation=cv2.INTER_LINEAR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # [H, W, C] -> [C, H, W]
            img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def _load_annotation(self, image_id):
        """Load segmentation mask"""
        anno_path = self.anno_dir / f"{image_id}.png"
        
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation not found: {anno_path}")
        
        # Load as grayscale (class indices)
        mask = cv2.imread(str(anno_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load annotation: {anno_path}")
        
        # Resize with nearest neighbor (preserve class indices)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_NEAREST)
        
        return mask.astype(np.int64)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_id = self.samples[idx]
        
        # Load annotation (always needed)
        mask = self._load_annotation(image_id)
        
        # Load based on modality
        if self.modality == 'rgb':
            rgb = self._load_image(image_id, is_depth=False)
            sample = torch.from_numpy(rgb).float()
            
        elif self.modality == 'depth':
            depth = self._load_image(image_id, is_depth=True)
            sample = torch.from_numpy(depth).float()
            
        elif self.modality == 'rgbd':
            rgb = self._load_image(image_id, is_depth=False)
            depth = self._load_image(image_id, is_depth=True)
            
            # Apply transforms if provided
            if self.transform:
                # Stack for joint transform, then split
                stacked = np.concatenate([rgb, depth], axis=0)  # [4, H, W]
                # TODO: Add proper augmentation that handles mask
                rgb = stacked[:3]
                depth = stacked[3:4]
            
            sample = {
                'rgb': torch.from_numpy(rgb).float(),
                'depth': torch.from_numpy(depth).float()
            }
        
        mask = torch.from_numpy(mask).long()
        
        return sample, mask


def get_segmentation_dataloaders(
    data_root,
    train_split='train',
    eval_split='eval',
    modalities=['rgb'],
    batch_size=16,  # Smaller batch size for segmentation
    num_workers=4,
    img_size=(224, 224),
    rgb_dir='RGB1',
    depth_dir='D_FocusN'
):
    """Create train and validation dataloaders for segmentation
    
    Args:
        data_root: Path to dataset root
        train_split: Name of train split
        eval_split: Name of eval split
        modalities: List of modalities ['rgb'], ['depth'], or ['rgb', 'depth']
        batch_size: Batch size (default 16 for segmentation)
        num_workers: Number of data loading workers
        img_size: Target image size (H, W)
        rgb_dir: RGB directory name
        depth_dir: Depth directory name
    """
    
    # Determine modality mode
    if len(modalities) == 2 or 'rgbd' in modalities:
        modality = 'rgbd'
    elif 'rgb' in modalities:
        modality = 'rgb'
    elif 'depth' in modalities:
        modality = 'depth'
    else:
        raise ValueError(f"Invalid modalities: {modalities}")
    
    train_dataset = RGBDSegmentationDataset(
        root_dir=data_root,
        split=train_split,
        modality=modality,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        transform=None,
        img_size=img_size
    )
    
    val_dataset = RGBDSegmentationDataset(
        root_dir=data_root,
        split=eval_split,
        modality=modality,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        transform=None,
        img_size=img_size
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.num_classes


# Test the dataset
if __name__ == '__main__':
    root = 'MM5_ALIGNED'
    
    print("\n" + "="*60)
    print("Testing Segmentation Dataset")
    print("="*60)
    
    train_loader, val_loader, num_classes = get_segmentation_dataloaders(
        data_root=root,
        modalities=['rgb'],
        batch_size=4,
        num_workers=0,
        img_size=(224, 224)
    )
    
    print(f"\nDataset Summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test one batch
    print(f"\nTesting batch loading...")
    for batch, masks in train_loader:
        if isinstance(batch, dict):
            print(f"  RGB shape: {batch['rgb'].shape}")
            print(f"  Depth shape: {batch['depth'].shape}")
        else:
            print(f"  Batch shape: {batch.shape}")
        print(f"  Mask shape: {masks.shape}")
        print(f"  Unique classes in batch: {torch.unique(masks).tolist()}")
        print(f"  Mask value range: [{masks.min()}, {masks.max()}]")
        break
    
    print("\n✓ Segmentation dataset loading successful!")