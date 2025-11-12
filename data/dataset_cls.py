import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class RGBDDataset(Dataset):
    """Dataset for RGB + Depth multimodal classification"""
    
    def __init__(
        self,
        root_dir,
        split='train',  # 'train' or 'eval'
        modality='rgbd',  # 'rgb', 'depth', or 'rgbd'
        rgb_dir='RGB1',  # Which RGB lighting to use
        depth_dir='D_FocusN',  # Which depth modality to use
        transform=None,
        img_size=(224, 224)  # Target image size (H, W)
    ):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.rgb_dir = self.root_dir / rgb_dir
        self.depth_dir = self.root_dir / depth_dir
        self.transform = transform
        self.split = split
        self.img_size = img_size
        
        # Load CSV data
        csv_file = 'train_dataset.csv' if split == 'train' else 'eval_dataset.csv'
        self.df = pd.read_csv(self.root_dir / csv_file)
        
        # Extract unique subcategories to build class mapping
        all_subcategories = self._extract_all_subcategories()
        self.classes = sorted(list(set(all_subcategories)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Build samples list: (image_id, class_index)
        self.samples = self._build_samples()
        
        print(f"\n{'='*60}")
        print(f"Loaded {split.upper()} split:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Modality: {modality}")
        print(f"  RGB dir: {rgb_dir}")
        print(f"  Depth dir: {depth_dir}")
        self._print_class_distribution()
        print(f"{'='*60}\n")
    
    def _extract_all_subcategories(self):
        """Extract all unique subcategories from both train and eval CSVs"""
        all_categories = []
        
        for csv_file in ['train_dataset.csv', 'eval_dataset.csv']:
            csv_path = self.root_dir / csv_file
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                for subcategory in df['Subcategory'].values:
                    if pd.isna(subcategory):
                        continue
                    # Clean and take first if comma-separated
                    subcategory = str(subcategory).strip().strip('"').strip()
                    if ',' in subcategory:
                        subcategory = subcategory.split(',')[0].strip()
                    if subcategory and subcategory != 'Subcategory':  # Skip header
                        all_categories.append(subcategory)
        
        return all_categories
    
    def _build_samples(self):
        """Build list of (image_id, label) tuples"""
        samples = []
        
        for idx, row in self.df.iterrows():
            image_id = row['ID']
            subcategory = row['Subcategory']
            
            # Skip invalid entries
            if pd.isna(subcategory) or subcategory == 'Subcategory':
                continue
            
            # Clean subcategory
            subcategory = str(subcategory).strip().strip('"').strip()
            
            # Take first subcategory if comma-separated
            if ',' in subcategory:
                subcategory = subcategory.split(',')[0].strip()
            
            # Get class index
            if subcategory in self.class_to_idx:
                label = self.class_to_idx[subcategory]
                samples.append((image_id, label))
            else:
                print(f"Warning: Unknown subcategory '{subcategory}' for ID {image_id}")
        
        return samples
    
    def _print_class_distribution(self):
        """Print distribution of samples per class"""
        from collections import Counter
        label_counts = Counter([label for _, label in self.samples])
        
        print(f"\n  Class distribution:")
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            count = label_counts[cls_idx]
            print(f"    {cls_idx:2d}. {cls_name:<20s}: {count:3d} samples")
    
    def _load_image(self, image_id, is_depth=False):
        """Load and preprocess image by ID"""
        img_filename = f"{image_id}.png"
        
        if is_depth:
            full_path = self.depth_dir / img_filename
            
            if not full_path.exists():
                raise FileNotFoundError(f"Depth image not found: {full_path}")
            
            # Load depth as grayscale
            img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load depth: {full_path}")
            
            # Resize to target size
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            
            img = img.astype(np.float32)
            img = img / 255.0 # Normalize to [0, 1]
            img = np.clip(img, 0, 1)  # Clip any outliers
            
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
            
            # Resize to target size
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # [H, W, C] -> [C, H, W]
            img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        
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
            
            # Apply transforms if provided (same for both modalities)
            if self.transform:
                # Stack for joint transform, then split
                stacked = np.concatenate([rgb, depth], axis=0)  # [4, H, W]
                stacked = self.transform(image=np.transpose(stacked, (1, 2, 0)))['image']
                stacked = np.transpose(stacked, (2, 0, 1))
                rgb = stacked[:3]
                depth = stacked[3:4]
            
            sample = {
                'rgb': torch.from_numpy(rgb).float(),
                'depth': torch.from_numpy(depth).float()
            }
        
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label


def get_dataloaders(
    data_root,
    train_split='train',
    eval_split='eval',
    modalities=['rgb', 'depth'],
    batch_size=32,
    num_workers=4,
    img_size=(224, 224),
    rgb_dir='RGB1',
    depth_dir='D_FocusN'
):
    """Create train and validation dataloaders
    
    Args:
        data_root: Path to dataset root
        train_split: Name of train split ('train')
        eval_split: Name of eval split ('eval')
        modalities: List of modalities to use ['rgb'], ['depth'], or ['rgb', 'depth']
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Target image size (H, W)
        rgb_dir: RGB directory name
        depth_dir: Depth directory name
    """
    
    # Determine modality mode from list
    if len(modalities) == 2 or 'rgbd' in modalities:
        modality = 'rgbd'
    elif 'rgb' in modalities:
        modality = 'rgb'
    elif 'depth' in modalities:
        modality = 'depth'
    else:
        raise ValueError(f"Invalid modalities: {modalities}")
    
    # Simple augmentation for now (can enhance later)
    train_transform = None  # Can add albumentations here
    val_transform = None
    
    train_dataset = RGBDDataset(
        root_dir=data_root,
        split=train_split,
        modality=modality,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        transform=train_transform,
        img_size=img_size
    )
    
    val_dataset = RGBDDataset(
        root_dir=data_root,
        split=eval_split,
        modality=modality,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        transform=val_transform,
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
    
    # Test RGB only
    print("\n" + "="*60)
    print("Testing RGBD Dataset")
    print("="*60)
    
    train_loader, val_loader, num_classes = get_dataloaders(
        data_root=root,
        train_split='train',
        eval_split='eval',
        modalities=['rgb', 'depth'],
        batch_size=8,
        num_workers=0,
        img_size=(224, 224)
    )
    
    print(f"\nDataset Summary:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test one batch
    print(f"\nTesting batch loading...")
    for batch, labels in train_loader:
        if isinstance(batch, dict):
            print(f"  RGB shape: {batch['rgb'].shape}")
            print(f"  Depth shape: {batch['depth'].shape}")
            print(f"  RGB range: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
            print(f"  Depth range: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")
        else:
            print(f"  Batch shape: {batch.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample labels: {labels[:5].tolist()}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        break
    
    print("\n✓ Dataset loading successful!")