import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from pathlib import Path

from data.dataset_cls import get_dataloaders
from models.resnet_fusion import ResNetSingleModality, RGBDEarlyFusion, RGBDFusionModel


def train_epoch(model, dataloader, criterion, optimizer, device, modality_mode='fusion'):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for data, labels in pbar:
        labels = labels.to(device)
        
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
        
        # Backward pass
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def eval_epoch(model, dataloader, criterion, device, modality_mode='fusion'):
    """Evaluate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for data, labels in pbar:
            labels = labels.to(device)
            
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
            
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(config):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, eval_loader, num_classes = get_dataloaders(
        data_root=config['data_root'],
        train_split=config['train_split'],
        eval_split=config['eval_split'],
        modalities=config['modalities'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        rgb_dir=config['rgb_dir']
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Eval samples: {len(eval_loader.dataset)}")
    print(f"Num classes: {num_classes}")
    
    # Create model
    print(f"Creating model: {config['model_type']}")
    if config['model_type'] == 'fusion':
        model = RGBDFusionModel(num_classes=num_classes)
        modality_mode = 'fusion'
    elif config['model_type'] == 'rgb_only':
        model = ResNetSingleModality(num_classes=num_classes)
        modality_mode = 'rgb'
    elif config['model_type'] == 'depth_only':
        model = ResNetSingleModality(num_classes=num_classes, in_channels=1)
        modality_mode = 'depth'
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
    best_acc = 0.0
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, modality_mode
        )
        
        # Evaluate
        eval_loss, eval_acc = eval_epoch(
            model, eval_loader, criterion, device, modality_mode
        )
        
        # Scheduler step
        scheduler.step()
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2f}%")
        
        # Save best model
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = save_dir / f"{config['exp_name']}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, save_path)
            print(f"Saved best model: {save_path} (Acc: {best_acc:.2f}%)")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['save_freq'] == 0:
            save_path = save_dir / f"{config['exp_name']}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'eval_acc': eval_acc,
                'config': config
            }, save_path)
    
    print(f"\nTraining complete! Best eval accuracy: {best_acc:.2f}%")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train RGBD Fusion Model')
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