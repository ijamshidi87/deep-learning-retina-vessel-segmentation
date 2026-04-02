# Retinal Blood Vessel Segmentation with U-Net

**Course**: Deep Learning — Homework 3  
**Instructor**: Professor Jun Bai  
**Author**: Iman Jamshidi  

## Overview

This project implements U-Net for medical image segmentation of retinal blood vessels. Three U-Net variants (2, 3, and 4 encoder-decoder blocks) were trained and compared with and without input normalization, using a combined BCE + Dice loss function.

## Dataset

- **Task**: Binary segmentation — vessel (1) vs background (0)
- **Samples**: 100 retinal images with ground truth masks
- **Split**: 64 train / 16 validation / 20 test
- **Input size**: 256×256×3 (RGB), output: 256×256×1 (binary mask)
- **Augmentation**: random horizontal/vertical flips and ±15° rotations

## Models & Results

| Model | Dice (with norm) | IoU (with norm) | Dice (no norm) | IoU (no norm) |
|---|---|---|---|---|
| U-Net 2 Blocks | 0.7443 | 0.5931 | 0.7427 | 0.5914 |
| U-Net 3 Blocks | 0.7376 | 0.5849 | 0.7445 | 0.5934 |
| **U-Net 4 Blocks** | **0.7454** ✅ | **0.5946** | 0.7446 | 0.5935 |

Best model: **U-Net 4 Blocks with normalization** — Dice: 0.7454, IoU: 0.5946

## Model Architectures

| Model | Parameters |
|---|---|
| U-Net 2 Blocks | ~1.87M |
| U-Net 3 Blocks | ~7.76M |
| U-Net 4 Blocks | ~31M |

## Key Findings

- Input normalization made training more stable and improved convergence
- Without normalization, U-Net 2 nearly failed (Dice ≈ 0.18–0.19 on some samples)
- All normalized models produced clean vessel masks with consistent Dice scores (0.73–0.80)
- Combined BCE + Dice loss effectively handles class imbalance in vessel segmentation
- Deeper models did not always outperform shallower ones — U-Net 4 with normalization was best overall

## Files

| File | Description |
|---|---|
| `model_unet.py` | U-Net architecture (2, 3, and 4 blocks) |
| `trainer.py` | Training loop with early stopping and LR scheduler |
| `config.py` | Hyperparameters and configuration |
| `data_loader.py` | Dataset loader with augmentation |
| `metrics.py` | Dice loss, IoU, and evaluation metrics |
| `test_model.py` | Testing script (run locally) |
| `test_model.ipynb` | Google Colab notebook for easy testing |
| `unet4_with_norm.pth` | Saved weights of the best model |
| `Iman_jamshidi_HW3.pdf` | Full written report |

## How to Test (Google Colab — Recommended)

1. Upload `test_model.ipynb` to [Google Colab](https://colab.research.google.com)
2. Run all cells in order
3. Expected output:
```
Mean Dice Score: 0.7454
Mean IoU Score:  0.5946
```

## How to Train
```python
# In config.py, set desired architecture:
NUM_BLOCKS = 4       # options: 2, 3, 4
NORMALIZE = True     # options: True, False

# Then run:
python trainer.py
```

## Requirements
```bash
pip install torch torchvision opencv-python numpy matplotlib albumentations tqdm
```
