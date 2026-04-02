                    HOMEWORK 3 - Medical Image Segmentation
                                 Iman Jamshidi
================================================================================

PROJECT DESCRIPTION
-------------------
This project implements U-Net architecture for Retina Blood Vessel Segmentation.
Three different U-Net variants (2, 3, and 4 blocks) are trained and compared
with and without input normalization.


FILES INCLUDED
--------------
1. Code Files:
   - model_unet.py      : U-Net architecture (2, 3, 4 blocks)
   - config.py          : Configuration and hyperparameters
   - data_loader.py     : Dataset loader with augmentation
   - metrics.py         : Dice loss, IoU, evaluation metrics
   - trainer.py         : Training loop and utilities
   - test_model.py      : Testing function (MAIN FILE FOR TESTING)

2. Notebooks:
   - Test_model.ipynb   : Google Colab notebook for easy testing

3. Model Weights:
   - unet4_with_norm.pth : Best performing model (370 MB)


REQUIREMENTS
------------
- Python 3.7+
- PyTorch 1.10+
- torchvision
- opencv-python
- numpy
- matplotlib
- albumentations
- tqdm

Install dependencies:
    pip install torch torchvision opencv-python numpy matplotlib albumentations tqdm


DATASET STRUCTURE
-----------------
Data/
├── train/
│   ├── image/
│   └── mask/
├── val/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/


HOW TO TEST THE MODEL
---------------------

METHOD 1: Using Google Colab (RECOMMENDED)
-------------------------------------------
This is the easiest way to test the model without any setup.

1. Upload Test_model.ipynb to Google Colab
   - Go to https://colab.research.google.com
   - Click "Upload" and select Test_model.ipynb

2. Run all cells in order (Cell 1 through Cell 6)
   - The notebook will automatically:
     * Install required dependencies
     * Load Python files from your Google Drive (or prompt manual upload)
     * Download the trained model (unet4_with_norm.pth)
     * Download test dataset
     * Run testing and display results

3. Expected output:
   Mean Dice Score: 0.7454
   Mean IoU Score:  0.5946


METHOD 2: Using Command Line/Terminal
--------------------------------------
If you prefer to run locally on your computer:

1. Create a project directory with this structure:
   project/
   ├── model_unet.py
   ├── config.py
   ├── data_loader.py
   ├── metrics.py
   ├── test_model.py
   ├── Models/
   │   └── unet4_with_norm.pth
   └── Data/
       └── test/
           ├── image/
           └── mask/

2. Install dependencies:
   pip install torch torchvision opencv-python numpy matplotlib albumentations tqdm

3. Run test from command line:
   python test_model.py

4. Or use the function directly in Python:
   from test_model import test_model
   
   results = test_model(
       model_path='Models/unet4_with_norm.pth',
       model_type='unet4',
       threshold=0.7
   )
   
   print(f"Dice: {results['mean_dice']:.4f}")
   print(f"IoU: {results['mean_iou']:.4f}")


HOW TO TRAIN A NEW MODEL
-------------------------
1. Adjust hyperparameters in config.py if needed
2. Run training:
   python trainer.py

Or use Google Colab notebooks provided for training.


MODEL ARCHITECTURES
-------------------
- U-Net 2 Blocks: ~1.87M parameters
- U-Net 3 Blocks: ~7.76M parameters
- U-Net 4 Blocks: ~31M parameters


BEST MODEL RESULTS (Test Set)
------------------------------
Model: U-Net 4 Blocks (with normalization)
- Test Dice Score: 0.7454
- Test IoU Score:  0.5946
- Threshold: 0.7



