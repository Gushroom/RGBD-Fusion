import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RGBDSegmentationDataset(Dataset):
    """Dataset for RGB + Depth semantic segmentation with multiple RGB directories.
       Each image is duplicated across all RGB directories to increase dataset size.
    """

    def __init__(
        self,
        root_dir,
        split='train',  # 'train' or 'eval'
        modality='rgbd',  # 'rgb', 'depth', or 'rgbd'
        rgb_dirs=('RGB1',),  # list/tuple of RGB dirs
        depth_dir='D_FocusN',
        anno_dir='ANNO_CLASS',
        transform=True,
        img_size=(224, 224)
    ):
        self.root_dir = Path(root_dir)
        self.modality = modality

        # Support multiple RGB dirs
        self.rgb_dirs = [self.root_dir / d for d in rgb_dirs]
        for d in self.rgb_dirs:
            if not d.exists():
                raise FileNotFoundError(f"RGB directory missing: {d}")

        self.depth_dir = self.root_dir / depth_dir
        self.anno_dir = self.root_dir / anno_dir
        self.transform = transform
        self.split = split
        self.img_size = img_size

        # Load class mapping
        self.load_class_mapping()

        # Load CSV
        csv_file = 'train_dataset.csv' if split == 'train' else 'eval_dataset.csv'
        self.df = pd.read_csv(self.root_dir / csv_file)

        # Build samples (expand across all RGB dirs)
        self.samples = self._build_samples()

        print(f"\nLoaded {split.upper()} split: {len(self.samples)} samples | Classes: {self.num_classes}")

    def load_class_mapping(self):
        csv_path = self.root_dir / "label_mapping.csv"
        df = pd.read_csv(csv_path)
        self.class_mapping = dict(zip(df["Label Name"], df["ID"]))
        self.num_classes = max(self.class_mapping.values()) + 1
        self.class_names = ["Background"] + [n for n, _ in sorted(self.class_mapping.items(), key=lambda x: x[1])]

    def _build_samples(self):
        """
        Each sample is a tuple (image_id, rgb_dir_name)
        This duplicates images across all RGB directories.
        """
        samples = []
        for _, row in self.df.iterrows():
            image_id = row['ID']
            anno_path = self.anno_dir / f"{image_id}.png"
            if not anno_path.exists():
                continue
            for rgb_dir in self.rgb_dirs:
                samples.append((image_id, rgb_dir.name))
        return samples

    def _load_rgb(self, sample):
        image_id, rgb_dir_name = sample
        rgb_dir = self.root_dir / rgb_dir_name
        full_path = rgb_dir / f"{image_id}.png"
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_depth(self, image_id):
        full_path = self.depth_dir / f"{image_id}.png"
        img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = (img - 18.0) / (234.0 - 18.0)
        img = np.clip(img, 0, 1)
        return img[..., None]  # H,W,1

    def _load_mask(self, image_id):
        full_path = self.anno_dir / f"{image_id}.png"
        mask = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id, _ = sample

        rgb = self._load_rgb(sample) if self.modality in ['rgb', 'rgbd'] else None
        depth = self._load_depth(image_id) if self.modality in ['depth', 'rgbd'] else None
        mask = self._load_mask(image_id)

        # ---------- APPLY TRANSFORMS ----------
        if self.transform:
            if self.modality == 'rgb':
                augmented = self.transform(image=rgb, mask=mask)
                rgb = augmented['image']
                mask = augmented['mask']
                return rgb, mask.long()

            elif self.modality == 'depth':
                augmented = self.transform(image=depth, mask=mask)
                depth = augmented['image']
                mask = augmented['mask']
                return depth, mask.long()

            else:  # RGBD
                combined = np.concatenate([rgb, depth], axis=-1)
                augmented = self.transform(image=combined, mask=mask)
                combined = augmented['image']
                rgb = combined[:3]
                depth = combined[3:4]
                return {'rgb': rgb, 'depth': depth}, augmented['mask'].long()

        # No transform
        rgb = torch.from_numpy(np.transpose(rgb / 255.0, (2,0,1))).float() if rgb is not None else None
        depth = torch.from_numpy(np.transpose(depth, (2,0,1))).float() if depth is not None else None
        mask = torch.from_numpy(mask).long()

        if self.modality == 'rgb': return rgb, mask
        if self.modality == 'depth': return depth, mask
        return {'rgb': rgb, 'depth': depth}, mask



def build_segmentation_transforms(img_size=(224,224)):
    return A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_REFLECT,p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),
        ToTensorV2(),
    ], additional_targets={})


from torch.utils.data import DataLoader

def get_segmentation_dataloaders(
    data_root,
    modalities=['rgb'],
    batch_size=8,
    num_workers=4,
    img_size=(224, 224),
    rgb_dirs=('RGB1','RGB2','RGB3','RGB4','RGB5','RGB6','RGB7','RGB8'),
    depth_dir='D_FocusN'
):

    modality = 'rgbd' if ('rgb' in modalities and 'depth' in modalities) else modalities[0]

    transform = build_segmentation_transforms(img_size)

    train_dataset = RGBDSegmentationDataset(
        root_dir=data_root,
        split='train',
        modality=modality,
        rgb_dirs=rgb_dirs,
        depth_dir=depth_dir,
        transform=transform,
        img_size=img_size
    )

    val_dataset = RGBDSegmentationDataset(
        root_dir=data_root,
        split='eval',
        modality=modality,
        rgb_dirs=rgb_dirs,
        depth_dir=depth_dir,
        transform=build_segmentation_transforms(img_size),
        img_size=img_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.num_classes
