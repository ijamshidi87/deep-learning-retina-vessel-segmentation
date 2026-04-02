
"""
Test Model
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model_unet import UNet2Block, UNet3Block, UNet4Block
from data_loader import RetinaDataset
from metrics import dice_score, iou_score


def test_model(model_path, model_type='unet2', threshold=0.7):
    """
    Test a trained U-Net model on test dataset.
    
    Args:
        model_path (str): Path to model checkpoint (.pth file)
        model_type (str): Model architecture ('unet2', 'unet3', or 'unet4')
        threshold (float): Threshold for binary prediction (default: 0.5)
        
    Returns:
        dict: Test results containing mean_dice and mean_iou
    """
    print("="*70)
    print("Testing Model")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Architecture: {model_type}")
    print(f"Threshold: {threshold}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    if model_type == 'unet2':
        model = UNet2Block(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            dropout=config.DROPOUT
        )
    elif model_type == 'unet3':
        model = UNet3Block(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            dropout=config.DROPOUT
        )
    elif model_type == 'unet4':
        model = UNet4Block(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    
    # Create test dataloader
    print("Loading test dataset...")
    test_dataset = RetinaDataset(
        image_dir=config.TEST_IMG_DIR,
        mask_dir=config.TEST_MASK_DIR,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        transform=None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Evaluate
    print("Evaluating...")
    all_dice = []
    all_iou = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = (outputs > threshold).float()
            
            # Calculate metrics
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            
            all_dice.append(dice)
            all_iou.append(iou)
    
    # Calculate mean scores
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    
    # Print results
    print("\n" + "="*70)
    print("Test Results")
    print("="*70)
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean IoU Score:  {mean_iou:.4f}")
    print("="*70)
    
    return {
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'dice_scores': all_dice,
        'iou_scores': all_iou
    }


if __name__ == "__main__":
    # Example usage
    results = test_model(
        model_path='Models/unet4_without_norm.pth',
        model_type='unet4',
        threshold=0.7
    )
    
    print(f"\nFinal Results:")
    print(f"Dice: {results['mean_dice']:.4f}")
    print(f"IoU: {results['mean_iou']:.4f}")
