import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import cv2

# -----------------------------
# CONFIGURATION
# -----------------------------
OLD_ROOT = Path("MM5_ALIGNED")
NEW_ROOT = Path("MM5_SEG")
RGB_DIRS = [f"RGB{i}" for i in range(1, 9)]
DEPTH_DIR = "D_FocusN"
ANNO_DIR = "ANNO_CLASS"

# Mapping old IDs -> new category IDs
# Example merge: Lemon, Mandarin, Tableware, Others...
# You can modify the new mapping to match your needs
NEW_CLASSES = {
    1: "Lemon",         # 1,2,3,10
    2: "Mandarin",      # 6,7,8,11,12
    3: "Tableware",     # 4,5,9,13,27
    4: "Onion",         # 14,15
    5: "Grapes",        # 16-21
    6: "Apple",         # 22-26
    7: "Pear",          # 28,29
    8: "Carrot"         # 30,31
}

# Define old IDs mapping to new IDs
OLD_TO_NEW = {
    # Lemon
    1: 1, 2: 1, 3: 1, 10: 1,
    # Mandarin
    6: 2, 7: 2, 8: 2, 11: 2, 12: 2,
    # Tableware / Others
    4: 3, 5: 3, 9: 3, 13: 3, 27: 3,
    # Onion
    14: 4, 15: 4,
    # Grapes
    16: 5, 17: 5, 18: 5, 19: 5, 20: 5, 21: 5,
    # Apple
    22: 6, 23: 6, 24: 6, 25: 6, 26: 6,
    # Pear
    28: 7, 29: 7,
    # Carrot
    30: 8, 31: 8
}

# -----------------------------
# UTILS
# -----------------------------
def remap_mask(mask):
    """Remap old class IDs to new class IDs"""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for old_id, new_id in OLD_TO_NEW.items():
        new_mask[mask == old_id] = new_id
    return new_mask

def copy_and_remap_images(split_csv):
    df = pd.read_csv(OLD_ROOT / split_csv)
    new_entries = []

    for idx, row in df.iterrows():
        img_id = str(row['ID'])
        # Remap mask
        old_mask_path = ANNO_DIR + f"/{img_id}.png"
        old_mask_full = OLD_ROOT / old_mask_path
        if not old_mask_full.exists():
            print(f"Mask not found: {old_mask_full}")
            continue

        mask = cv2.imread(str(old_mask_full), cv2.IMREAD_UNCHANGED)
        new_mask = remap_mask(mask)

        # Save new mask
        new_mask_dir = NEW_ROOT / ANNO_DIR
        new_mask_dir.mkdir(parents=True, exist_ok=True)
        new_mask_path = new_mask_dir / f"{img_id}.png"
        cv2.imwrite(str(new_mask_path), new_mask)

        # Copy RGBs
        new_rgb_paths = {}
        for rgb_dir in RGB_DIRS:
            src = OLD_ROOT / rgb_dir / f"{img_id}.png"
            dst_dir = NEW_ROOT / rgb_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{img_id}.png"
            if src.exists():
                shutil.copy2(src, dst)
                new_rgb_paths[rgb_dir] = str(dst)
            else:
                print(f"RGB missing: {src}")

        # Copy depth
        src_depth = OLD_ROOT / DEPTH_DIR / f"{img_id}.png"
        dst_depth_dir = NEW_ROOT / DEPTH_DIR
        dst_depth_dir.mkdir(parents=True, exist_ok=True)
        dst_depth = dst_depth_dir / f"{img_id}.png"
        if src_depth.exists():
            shutil.copy2(src_depth, dst_depth)
        else:
            print(f"Depth missing: {src_depth}")

        new_entries.append({'ID': img_id})

    # Save new CSV
    new_csv_path = NEW_ROOT / split_csv
    new_df = pd.DataFrame(new_entries)
    new_df.to_csv(new_csv_path, index=False)
    print(f"Saved new CSV: {new_csv_path}")

def save_label_mapping():
    """Save new merged label mapping to CSV"""
    new_mapping = pd.DataFrame({
        'ID': list(NEW_CLASSES.keys()),
        'Label Name': list(NEW_CLASSES.values())
    })
    new_mapping_path = NEW_ROOT / "label_mapping.csv"
    new_mapping.to_csv(new_mapping_path, index=False)
    print(f"Saved new label mapping CSV: {new_mapping_path}")

    # Optional: save classes.txt
    classes_txt_path = NEW_ROOT / "classes.txt"
    with open(classes_txt_path, 'w') as f:
        for cls_name in NEW_CLASSES.values():
            f.write(cls_name + "\n")
    print(f"Saved classes.txt: {classes_txt_path}")

# -----------------------------
# MAIN
# -----------------------------
def main():
    # Process train and eval splits
    for split_csv in ["train_dataset.csv", "eval_dataset.csv"]:
        print(f"Processing {split_csv}...")
        copy_and_remap_images(split_csv)

    # Save new mapping
    save_label_mapping()
    print("\nDataset conversion complete! New dataset at:", NEW_ROOT)

if __name__ == "__main__":
    main()
