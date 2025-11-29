import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RGBDSegmentationDataset(Dataset):
    """Dataset for RGB + Depth semantic segmentation"""
    
    def __init__(
        self,
        root_dir,
        split='train',
        modality='rgbd',
        rgb_dir='RGB1',
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
        
        self.load_class_mapping()
        
        csv_file = 'train_dataset.csv' if split == 'train' else 'eval_dataset.csv'
        self.df = pd.read_csv(self.root_dir / csv_file)
        
        self.samples = self._build_samples()
        
        print(f"\n{'='*60}")
        print(f"Loaded {split.upper()} split for SEGMENTATION:")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Number of classes: {self.num_classes} (including background)")
        print(f"  Modality: {modality}")
        print(f"  Image size: {img_size}")
        print(f"{'='*60}\n")
    
    def load_class_mapping(self):
        csv_path = self.root_dir / "label_mapping.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"label_mapping.csv not found at: {csv_path}")

        df = pd.read_csv(csv_path)

        self.class_mapping = dict(zip(df["Label Name"], df["ID"]))
        self.num_classes = max(self.class_mapping.values()) + 1

        self.class_names = ["Background"] + [
            name for name, _ in sorted(self.class_mapping.items(), key=lambda x: x[1])
        ]

        print(f"Loaded {self.num_classes} classes from CSV:")
        for i, name in enumerate(self.class_names[:10]):
            print(f"  {i}: {name}")
        if len(self.class_names) > 10:
            print(f"  ... and {len(self.class_names)-10} more")
    
    def _build_samples(self):
        samples = []
        
        for idx, row in self.df.iterrows():
            image_id = row['ID']
            
            anno_path = self.anno_dir / f"{image_id}.png"
            if anno_path.exists():
                samples.append(image_id)
        
        return samples
    
    def _load_image(self, image_id, is_depth=False):
        img_filename = f"{image_id}.png"
        
        if is_depth:
            full_path = self.depth_dir / img_filename
            
            if not full_path.exists():
                raise FileNotFoundError(f"Depth image not found: {full_path}")
            
            img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load depth: {full_path}")
            
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), 
                           interpolation=cv2.INTER_NEAREST)
            
            img = img.astype(np.float32)
            img = (img - 138.94) / 52.13
            
            img = img[..., np.newaxis]
            
            return img
        else:
            full_path = self.rgb_dir / img_filename
            
            if not full_path.exists():
                raise FileNotFoundError(f"RGB image not found: {full_path}")
            
            img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load RGB: {full_path}")
            
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                           interpolation=cv2.INTER_LINEAR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
    
    def _load_annotation(self, image_id):
        anno_path = self.anno_dir / f"{image_id}.png"
        
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation not found: {anno_path}")
        
        mask = cv2.imread(str(anno_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load annotation: {anno_path}")
        
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_NEAREST)
        
        return mask.astype(np.int64)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_id = self.samples[idx]

        rgb = None
        depth = None

        mask = self._load_annotation(image_id)

        if self.modality in ['rgb', 'rgbd']:
            rgb = self._load_image(image_id, is_depth=False)

        if self.modality in ['depth', 'rgbd']:
            depth = self._load_image(image_id, is_depth=True)

        if self.transform:
            if self.modality == 'rgb':
                out = self.transform(image=rgb, mask=mask)
                rgb = out["image"]
                mask = out["mask"]

            elif self.modality == 'depth':
                out = self.transform(image=depth, mask=mask)
                depth = out["image"]
                mask = out["mask"]

            else:
                out = self.transform(image=rgb, depth=depth, mask=mask)
                rgb = out["image"]
                depth = out["depth"]
                mask = out["mask"]
        else:
            if self.modality == 'rgb':
                rgb = torch.from_numpy(rgb.transpose(2, 0, 1))
            elif self.modality == 'depth':
                depth = torch.from_numpy(depth.transpose(2, 0, 1))
            else:
                rgb = torch.from_numpy(rgb.transpose(2, 0, 1))
                depth = torch.from_numpy(depth.transpose(2, 0, 1))
            mask = torch.from_numpy(mask)

        if self.modality == 'rgb':
            rgb = rgb.float() / 255.0
            rgb = (rgb - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return rgb, mask.long()

        elif self.modality == 'depth':
            return depth.float(), mask.long()

        else:
            rgb = rgb.float() / 255.0
            rgb = (rgb - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return {"rgb": rgb, "depth": depth.float()}, mask.long()


def build_transforms(img_size=(224, 224), modality='rgbd', is_train=True):
    if is_train:
        if modality == 'rgbd':
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                ToTensorV2(),
            ]
            return A.Compose(transforms, additional_targets={'depth': 'image'})
        else:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3),
            ]
            if modality == 'rgb':
                transforms.extend([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                ])
            transforms.append(ToTensorV2())
            return A.Compose(transforms)
    else:
        transforms = [ToTensorV2()]
        if modality == 'rgbd':
            return A.Compose(transforms, additional_targets={'depth': 'image'})
        else:
            return A.Compose(transforms)


def get_dataloaders(
    data_root,
    train_split='train',
    eval_split='eval',
    modalities=['rgb'],
    batch_size=16,
    num_workers=4,
    img_size=(224, 224),
    rgb_dir='RGB1',
    depth_dir='D_FocusN'
):
    if len(modalities) == 2 or 'rgbd' in modalities:
        modality = 'rgbd'
    elif 'rgb' in modalities:
        modality = 'rgb'
    elif 'depth' in modalities:
        modality = 'depth'
    else:
        raise ValueError(f"Invalid modalities: {modalities}")

    train_transform = build_transforms(img_size=img_size, modality=modality, is_train=True)
    val_transform = build_transforms(img_size=img_size, modality=modality, is_train=False)
    
    train_dataset = RGBDSegmentationDataset(
        root_dir=data_root,
        split=train_split,
        modality=modality,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        transform=train_transform,
        img_size=img_size
    )
    
    val_dataset = RGBDSegmentationDataset(
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