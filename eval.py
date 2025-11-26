import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision.transforms import functional as TF

from data.dataset_seg import get_segmentation_dataloaders
from models.resnet_unet import ResNetUNet, ResNetUNetEarlyFusion


def decode_mask(mask, class_colors):
    """Convert class indices to RGB mask for visualization"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(class_colors):
        rgb_mask[mask == cls_idx] = color
    return rgb_mask


def visualize_segmentation(model, dataloader, class_colors, device, modality_mode='fusion', num_samples=5):
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for data, masks in dataloader:
            masks = masks.to(device)

            if modality_mode == 'fusion':
                rgb = data['rgb'].to(device)
                depth = data['depth'].to(device)
                outputs = model(rgb, depth)
                rgb_np = rgb.cpu().numpy()
            elif modality_mode == 'rgb':
                rgb = data['rgb'].to(device)
                outputs = model(rgb)
                rgb_np = rgb.cpu().numpy()
            else:
                depth = data['depth'].to(device)
                outputs = model(depth)
                rgb_np = depth.cpu().numpy()

            preds = outputs.argmax(dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            batch_size = preds.shape[0]
            for i in range(batch_size):
                pred_mask_rgb = decode_mask(preds[i], class_colors)
                gt_mask_rgb = decode_mask(masks[i], class_colors)

                # RGB input (transpose from [C,H,W] to [H,W,C])
                rgb_img = np.transpose(rgb_np[i][:3], (1, 2, 0))
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # normalize to [0,1]

                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb_img)
                axes[0].set_title("RGB Input")
                axes[0].axis('off')

                axes[1].imshow(gt_mask_rgb)
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')

                axes[2].imshow(pred_mask_rgb)
                axes[2].set_title("Prediction")
                axes[2].axis('off')

                plt.show()

                samples_shown += 1
                if samples_shown >= num_samples:
                    return


if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataloaders
    train_loader, eval_loader, num_classes = get_segmentation_dataloaders(
        data_root=config['data_root'],
        train_split=config['train_split'],
        eval_split=config['eval_split'],
        modalities=config['modalities'],
        batch_size=1,
        num_workers=4,
        img_size=config['img_size']
    )

    # Define a simple color map for classes
    np.random.seed(42)
    class_colors = [tuple(np.random.randint(0, 256, 3)) for _ in range(num_classes)]

    # Load model
    if config['model_type'] == 'unet_rgb':
        model = ResNetUNet(num_classes=num_classes, pretrained=False)
        modality_mode = 'rgb'
    elif config['model_type'] == 'unet_early_fusion':
        model = ResNetUNetEarlyFusion(num_classes=num_classes, pretrained=False)
        modality_mode = 'fusion'
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Visualize
    visualize_segmentation(model, eval_loader, class_colors, device, modality_mode, args.num_samples)
