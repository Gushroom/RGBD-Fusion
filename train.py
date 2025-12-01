import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import get_dataloaders
from models import *
from losses.dice import DiceLoss


def confusion_matrix(preds, targets, num_classes, ignore_index=None):
    """
    Build confusion matrix for a batch.
    preds: Tensor long, shape [N, H, W]
    targets: Tensor long, shape [N, H, W]
    returns: conf (num_classes x num_classes) numpy array where
        conf[t, p] counts number of pixels with true class t and predicted class p
    """
    with torch.no_grad():
        preds = preds.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()

    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    # filter out-of-range (safety)
    valid = (targets >= 0) & (targets < num_classes)
    preds = preds[valid]
    targets = targets[valid]

    # combine into single index
    indices = targets * num_classes + preds
    conf = np.bincount(indices, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    return conf

def compute_iou_from_conf(conf, ignore_index=None):
    """
    Compute per-class IoU and mean IoU from confusion matrix.
    conf shape: [num_classes, num_classes] where conf[t,p]
    """
    num_classes = conf.shape[0]
    true_positive = np.diag(conf).astype(np.float64)
    false_positive = conf.sum(axis=0) - true_positive
    false_negative = conf.sum(axis=1) - true_positive
    union = true_positive + false_positive + false_negative

    ious = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            ious[c] = np.nan
        else:
            if union[c] == 0:
                ious[c] = np.nan
            else:
                ious[c] = true_positive[c] / union[c]

    # mean over non-nan classes
    valid = ~np.isnan(ious)
    mean_iou = np.nanmean(ious[valid]) if valid.any() else 0.0
    return mean_iou, ious


def train_epoch(model, dataloader, criterion, optimizer, device, modality_mode='rgb', num_classes=32, ignore_index=None):
    """Train for one epoch and compute dataset-level mIoU"""
    model.train()
    running_loss = 0.0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    pbar = tqdm(dataloader, desc='Training')
    for data, masks in pbar:
        masks = masks.to(device)

        # Forward
        if modality_mode == 'fusion':
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            outputs = model(rgb, depth)
        elif modality_mode == 'rgb':
            if isinstance(data, dict):
                rgb = data['rgb'].to(device)
            else:
                rgb = data.to(device)
            outputs = model(rgb)
        elif modality_mode == 'depth':
            if isinstance(data, dict):
                depth = data['depth'].to(device)
            else:
                depth = data.to(device)
            outputs = model(depth)

        # Loss
        loss = criterion(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update confusion matrix
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            conf = confusion_matrix(preds, masks, num_classes=num_classes, ignore_index=ignore_index)
            conf_matrix += conf

        # Optional: show batch loss in progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1)
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_miou, per_class_iou = compute_iou_from_conf(conf_matrix, ignore_index=ignore_index)

    return epoch_loss, epoch_miou, per_class_iou


def eval_epoch(model, dataloader, criterion, device, modality_mode='rgb', num_classes=32, ignore_index=None):
    model.eval()
    running_loss = 0.0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for data, masks in pbar:
            masks = masks.to(device)

            # forward (same as before)
            if modality_mode == 'fusion':
                rgb = data['rgb'].to(device)
                depth = data['depth'].to(device)
                outputs = model(rgb, depth)
            elif modality_mode == 'rgb':
                if isinstance(data, dict):
                    rgb = data['rgb'].to(device)
                else:
                    rgb = data.to(device)
                outputs = model(rgb)
            elif modality_mode == 'depth':
                if isinstance(data, dict):
                    depth = data['depth'].to(device)
                else:
                    depth = data.to(device)
                outputs = model(depth)

            # loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # preds
            preds = outputs.argmax(dim=1)

            # update confusion matrix
            conf = confusion_matrix(preds, masks, num_classes=num_classes, ignore_index=ignore_index)
            conf_matrix += conf

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1)
            })

    epoch_loss = running_loss / len(dataloader)
    mean_iou, per_class_iou = compute_iou_from_conf(conf_matrix, ignore_index=ignore_index)

    return epoch_loss, mean_iou, per_class_iou



def train(config):
    """Main training function with Dice + CrossEntropy loss"""
    history = {
        "train_loss": [],
        "train_iou": [],
        "eval_loss": [],
        "eval_iou": [],
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading segmentation data...")
    train_loader, eval_loader, num_classes = get_dataloaders(
        data_root=config['data_root'],
        modalities=config['modalities'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        rgb_dir=config['rgb_folder']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Eval samples: {len(eval_loader.dataset)}")
    print(f"Num classes: {num_classes}")
    
    # Create model
    print(f"\nCreating model: {config['model_type']}")
    if config['model_type'] == 'rgb_baseline':
        model = ResNetUNet(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'rgb'
    elif config['model_type'] == 'early_fusion':
        model = ResNetUNetEarlyFusion(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'fusion'
    elif config['model_type'] == 'mid_fusion':
        model = ResNetUNetMidFusion(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'fusion'
    elif config['model_type'] == 'late_fusion':
        model = ResNetUNetLateFusion(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'fusion'
    elif config['model_type'] == 'attn_fusion':
        model = ResNetUNetAttnFusion(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'fusion'
    elif config['model_type'] == 'se_fusion':
        model = ResNetUNetSEFusion(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'fusion'
    elif config['model_type'] == 'transformer_fusion':
        model = ResNetTransformerFusion(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'fusion'
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    alpha = config.get('dice_alpha', 0.5)  # weight for Dice vs CE

    def combined_loss(logits, targets):
        return alpha * dice_loss(logits, targets) + (1 - alpha) * ce_loss(logits, targets)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Training loop
    best_iou = 0.0
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_iou, train_class_ious = train_epoch(
            model, train_loader, combined_loss, optimizer, device, modality_mode, num_classes, ignore_index=0
        )
        
        # Evaluate
        eval_loss, eval_iou, class_ious = eval_epoch(
            model, eval_loader, combined_loss, device, modality_mode, num_classes, ignore_index=0
        )
        
        # Scheduler step
        scheduler.step()
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_iou*100:.2f}%")
        print(f"Eval Loss: {eval_loss:.4f} | Eval mIoU: {eval_iou*100:.2f}%")

        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["eval_loss"].append(eval_loss)
        history["eval_iou"].append(eval_iou)
        
        # Save best model
        if eval_iou > best_iou:
            best_iou = eval_iou
            save_path = save_dir / f"{config['exp_name']}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'class_ious': class_ious,
                'config': config
            }, save_path)
            print(f"Saved best model: {save_path} (mIoU: {best_iou*100:.2f}%)")
        
        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0:
            save_path = save_dir / f"{config['exp_name']}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_iou': train_iou,
                'eval_iou': eval_iou,
                'config': config
            }, save_path)
    
    print(f"\nTraining complete! Best eval mIoU: {best_iou*100:.2f}%")
    with open(f"results.txt", "a") as f:
        f.write(f"{config['exp_name']}: {best_iou*100:.2f} \n")

    plt.figure(figsize=(12, 8))

    epochs = range(1, config['epochs'] + 1)

    # Loss subplot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    # IoU subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["eval_iou"], label="Eval mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("IoU Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plot_path = save_dir / f"{config['exp_name']}_training_curves.png"
    plt.savefig(plot_path)
    plt.close()
    return best_iou


def main():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (YAML)')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Override data root path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data_root if provided
    if args.data_root:
        config['data_root'] = args.data_root
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
