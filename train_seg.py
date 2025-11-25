import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data.dataset_seg import get_segmentation_dataloaders
from models.resnet_unet import ResNetUNet, ResNetUNetEarlyFusion
from losses.dice import DiceLoss


def calculate_iou(pred, target, num_classes, ignore_index=None):
    """Calculate mean IoU (Intersection over Union)"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # No ground truth or prediction
        else:
            ious.append((intersection / union).item())
    
    # Calculate mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0, ious


def train_epoch(model, dataloader, criterion, optimizer, device, modality_mode='rgb'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for data, masks in pbar:
        masks = masks.to(device)
        
        # Handle different modality modes
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
        
        # Metrics
        running_loss += loss.item()
        
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            miou, _ = calculate_iou(pred, masks, num_classes=outputs.shape[1])
            running_iou += miou
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'mIoU': running_iou / (pbar.n + 1)
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_iou


def eval_epoch(model, dataloader, criterion, device, modality_mode='rgb', num_classes=32):
    """Evaluate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    all_class_ious = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for data, masks in pbar:
            masks = masks.to(device)
            
            # Handle different modality modes
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
            running_loss += loss.item()
            
            # Metrics
            pred = outputs.argmax(dim=1)
            miou, class_ious = calculate_iou(pred, masks, num_classes=num_classes)
            running_iou += miou
            all_class_ious.append(class_ious)
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'mIoU': running_iou / (pbar.n + 1)
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    # Average per-class IoU
    avg_class_ious = np.nanmean(all_class_ious, axis=0)
    
    return epoch_loss, epoch_iou, avg_class_ious


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
    train_loader, eval_loader, num_classes = get_segmentation_dataloaders(
        data_root=config['data_root'],
        train_split=config['train_split'],
        eval_split=config['eval_split'],
        modalities=config['modalities'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Eval samples: {len(eval_loader.dataset)}")
    print(f"Num classes: {num_classes}")
    
    # Create model
    print(f"\nCreating model: {config['model_type']}")
    if config['model_type'] == 'unet_rgb':
        model = ResNetUNet(num_classes=num_classes, pretrained=config['pretrained'])
        modality_mode = 'rgb'
    elif config['model_type'] == 'unet_early_fusion':
        model = ResNetUNetEarlyFusion(num_classes=num_classes, pretrained=config['pretrained'])
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
        train_loss, train_iou = train_epoch(
            model, train_loader, combined_loss, optimizer, device, modality_mode
        )
        
        # Evaluate
        eval_loss, eval_iou, class_ious = eval_epoch(
            model, eval_loader, combined_loss, device, modality_mode, num_classes
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

    plt.figure(figsize=(12, 8))

    epochs = range(1, config['epochs'] + 1)

    # Loss subplot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["eval_loss"], label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()

    # IoU subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_iou"], label="Train mIoU")
    plt.plot(epochs, history["eval_iou"], label="Eval mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("IoU Curves")
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