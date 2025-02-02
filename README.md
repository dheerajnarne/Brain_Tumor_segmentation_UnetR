# Brain Tumor Segmentation using UNETR

## Overview
Brain tumor segmentation is a critical task in medical image analysis, aiming to accurately identify tumor regions in MRI scans. This project leverages **UNETR (U-Net Transformer)**, a novel deep learning architecture that combines the strengths of Vision Transformers (ViTs) and U-Net for precise segmentation. Unlike traditional CNN-based methods, UNETR captures **long-range dependencies** using self-attention mechanisms, making it highly effective for complex medical imaging tasks.

## Why UNETR?
### 1. **Vision Transformers for Long-Range Dependencies**
Unlike conventional CNNs, which have a **limited receptive field**, Vision Transformers (ViTs) divide images into **patches** and process them as tokens, allowing the model to learn **global context** through self-attention. This capability is particularly beneficial for brain tumor segmentation, where tumors may have irregular shapes and varying intensities.

### 2. **Hybrid UNETR Architecture**
- **Encoder:** Uses a **Transformer** backbone (ViT) to extract hierarchical representations.
- **Decoder:** Uses a **U-Net style** upsampling path for precise localization.
- **Skip Connections:** Bridges the encoder and decoder to retain spatial details lost in deep feature extraction.

## Dataset
This project utilizes the **BraTS (Brain Tumor Segmentation) dataset**, which includes multimodal MRI scans (T1, T1ce, T2, FLAIR) with expert-annotated tumor regions. The dataset is preprocessed using:
- **N4ITK bias field correction**
- **Normalization (z-score scaling)**
- **Resampling to a common resolution**

## Model Architecture
The UNETR model consists of:
- **Patch Embedding Layer:** Splits MRI scans into non-overlapping patches.
- **Transformer Encoder:** Uses **multi-head self-attention (MHSA)** to capture **long-range dependencies**.
- **Bottleneck:** Integrates learned features for hierarchical understanding.
- **Decoder:** Upsamples feature maps for pixel-wise segmentation.
- **Skip Connections:** Enhances fine-grained details by merging encoder and decoder features.

## Training Details
- **Optimizer:** AdamW
- **Loss Function:** Dice Loss + Cross-Entropy Loss
- **Learning Rate:** 1e-4 with cosine decay
- **Batch Size:** 4
- **Augmentations:** Random flipping, rotation, intensity shifts
- **Framework:** PyTorch & MONAI

## Results
The model achieves **high Dice Similarity Coefficient (DSC)** across different tumor subregions:
- **Whole Tumor (WT):** X%
- **Tumor Core (TC):** X%
- **Enhancing Tumor (ET):** X%

## Installation
To set up the environment, run:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, run:
```bash
python train.py --epochs 100 --batch_size 4
```
To test the model, run:
```bash
python test.py --checkpoint path/to/model.pth
```

## Conclusion
UNETR leverages **Vision Transformers** to capture long-range dependencies, making it highly effective for **brain tumor segmentation**. By integrating the **global understanding of ViTs** with **U-Net’s localization strength**, it achieves **state-of-the-art performance** on medical imaging tasks.

---
### References
- Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image Segmentation" (MICCAI 2022)
- BraTS Challenge: https://www.med.upenn.edu/cbica/brats2021/

