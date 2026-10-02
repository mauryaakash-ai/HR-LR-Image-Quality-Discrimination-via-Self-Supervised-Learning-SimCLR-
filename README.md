# HR-LR Image Quality Discrimination via Self-Supervised Learning (SimCLR)

This repository implements a self-supervised representation learning framework based on **SimCLR** (*A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020) specifically adapted for **High-Resolution (HR) vs. Low-Resolution (LR) / Image Degradation Discrimination**.

---

## Table of Contents

1. [Overview & Problem Formulation](#overview--problem-formulation)
2. [Datasets & Data Pipeline](#datasets--data-pipeline)
3. [Benchmark & Experimental Results](#benchmark--experimental-results)
4. [Comparison: Official SimCLR Paper/Repo vs. Our HR-LR Implementation](#comparison-official-simclr-paperrepo-vs-our-hr-lr-implementation)
5. [Repository Structure](#repository-structure)
6. [Quickstart & Usage](#quickstart--usage)
   - [1. Self-Supervised Pretraining](#1-self-supervised-pretraining-runpy)
   - [2. Paired HR vs LR Fine-Tuning](#2-paired-hr-vs-lr-fine-tuning-finetune_hr_lrpy)
   - [3. Linear Evaluation (Frozen Encoder)](#3-linear-evaluation-linear_eval_hr_lrpy)
   - [4. Inference & Paired Demo](#4-inference--demo-infer_hr_lrpy)
7. [Checkpoints & Reproducibility](#checkpoints--reproducibility)
8. [References](#references)

---

## Overview & Problem Formulation

Standard computer vision backbones are trained for semantic classification (e.g., ImageNet), which explicitly encourages invariance to scale, blur, and high-frequency variations. However, tasks such as **Super-Resolution (SR)**, **Image Quality Assessment (IQA)**, and **Degradation Estimation** require the exact opposite: **representations that are highly sensitive to spatial resolution, frequency details, and blur while remaining invariant across diverse semantic content**.

### Key Idea
1. **Self-Supervised Pretraining (SimCLR)**: Pretrain a ResNet-18 backbone on unlabelled high-resolution image patches using contrastive learning (NT-Xent loss + LARS optimizer).
2. **Frequency-Preserving Augmentations**: Unlike standard SimCLR which uses heavy Gaussian blur to enforce semantic abstraction, we disable Gaussian blur to preserve high-frequency sharpness cues.
3. **Paired Same-Location Degradation Fine-Tuning**: Extract matching patches from the *exact same spatial coordinates* in paired HR and LR images:
   - **Label 0 (HR / Clean)**: Native high-resolution sharp crop.
   - **Label 1 (LR / Degraded)**: Downscaled ($\times 4$) and bicubic-upsampled crop from the identical scene location.
   - By keeping the visual content identical, the model is forced to discriminate purely on genuine resolution degradation and blur rather than scene semantics.

```
       ┌──────────────────────┐
       │   DIV2K HR Image     │───► Crop (x, y, 256, 256) ─────────────► HR Patch (Label 0) ──┐
       └──────────────────────┘                                                                │
                   │                                                                           ▼
                   │ (x4 Bicubic Downsampling)                                          ┌──────────────┐
                   ▼                                                                    │  ResNet-18   │──► HR / LR
       ┌──────────────────────┐                                                         │ Pretrained / │    Decision
       │ DIV2K LR Image (x4)  │───► Crop (x/4, y/4, 64, 64) ──► Resize (256, 256) ────► LR Patch (Label 1) ──┘
       └──────────────────────┘                                (Bicubic Upsample)
```

---

## Datasets & Data Pipeline

### 1. DIV2K Dataset (Primary Dataset)
The repository uses the **DIV2K (DIVerse 2K resolution)** dataset, a standard benchmark in super-resolution and image restoration:
- **DIV2K Training Set (`Dataset/DIV2K_train_HR` & `DIV2K_train_LR_bicubic_X4/X4`)**:
  - 800 high-resolution images ($2040 \times 1356$ average resolution) in 2K RGB PNG format.
  - Paired $\times 4$ downsampled images generated with bicubic degradation (`DIV2K_train_LR_bicubic_X4`).
  - Scale $\times 2$ downsampled subsets (`DIV2K_train_LR_bicubic/X2`).
- **DIV2K Validation Set (`Dataset/DIV2K_valid_HR` & `DIV2K_valid_LR_bicubic_X4/X4`)**:
  - 100 high-resolution images (`0801.png` - `0900.png`) and corresponding $\times 4$ LR images.

### 2. Patch Extraction & Data Loading Strategy
- **On-the-Fly Pretraining Loader (`DIV2KPatchDataset` in `data.py`)**:
  - Loads 800 full-resolution images into RAM.
  - Dynamically samples random $96 \times 96$ or $256 \times 256$ crops on the fly (50 patches/image = 40,000 virtual patches per epoch).
  - Generates two stochastic augmented views per crop (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale).
- **Paired Coordinate-Preserving Loader (`HRLRPairedDataset` in `finetune_hr_lr.py`)**:
  - Splits 800 images into **700 training pairs** (DIV2K 0001–0700) and **100 validation pairs** (DIV2K 0701–0800).
  - Computes exact spatial coordinates $(x, y)$ in HR and $(x/4, y/4)$ in LR.
  - Upsamples the LR patch back to $256 \times 256$ using bicubic interpolation to preserve natural degradation artifacts.
  - Produces $700 \times 30 \times 2 = 42,000$ training patch samples per epoch and $100 \times 30 \times 2 = 6,000$ validation patch samples.

---

## Benchmark & Experimental Results

### 1. Self-Supervised Pretraining Benchmark (SimCLR)

| Experiment / Configuration | Backbone | Patch Size | Batch Size | Epochs | Loss | Top-1 Contrast Acc | Key Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SimCLR HR (with Gaussian Blur)** | ResNet-18 | $96 \times 96$ | 128 | 50 | 8.1534 | 62.56% | Baseline with official SimCLR blur augmentation |
| **SimCLR HR (No-Blur Ablation)** | ResNet-18 | $96 \times 96$ | 128 | 50 | 8.1094 | **64.59%** (+2.03%) | Disabling blur preserves high frequencies |
| **SimCLR HR Pretraining ($256 \times 256$)** | ResNet-18 | $256 \times 256$ | 128 | 100 | **6.4191** | **88.78%** (Peak: 89.20%) | Large patch size captures rich texture & edge structures |

---

### 2. HR vs. LR Discrimination & Evaluation Benchmark

| Evaluation Protocol | Encoder Status | Patch Size | Pairing Strategy | Train Accuracy | Validation Accuracy | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Probe (Unaligned Crops)** | Frozen | $96 \times 96$ | Random / Independent | 68.30% | 67.61% | Model conflates scene semantics with resolution |
| **End-to-End Fine-Tuning (Unaligned)** | Unfrozen | $96 \times 96$ | Random / Independent | 84.92% | 83.71% | Fine-tuning encoder improves over frozen unaligned probe |
| **Paired Fine-Tuning ($96 \times 96$)** | Unfrozen | $96 \times 96$ | Paired Same-Location | 99.77% | **99.93%** | Paired coordinates isolate genuine degradation |
| **Paired Fine-Tuning ($256 \times 256$, Bicubic)** | Unfrozen | $256 \times 256$ | Paired Same-Location | 99.94% | **99.98%** | **Best Performance**; near-perfect discrimination |
| **Random Encoder Baseline (Linear Probe)** | Random (Untrained) | $256 \times 256$ | Paired Same-Location | 79.10% | 78.80% | Untrained random weights baseline |
| **Pretrained Encoder (Linear Probe)** | Frozen Pretrained | $256 \times 256$ | Paired Same-Location | 99.96% | **99.60%** (+20.80%) | Linear probe on frozen features verifies SSL representation quality |

#### Detailed Linear Evaluation Metrics (`linear_eval_hr_lr.py` on 1,000 validation patches):
```text
============================================================
           LINEAR EVALUATION RESULTS (DIV2K HR vs LR)
============================================================
Random Encoder Baseline Val Accuracy:     78.80%
Pretrained Encoder Train Accuracy:        99.96%
Pretrained Encoder Validation Accuracy:   99.60%
Representation Quality Gain:              +20.80%
============================================================

Classification Report (Validation Split):
                  precision    recall  f1-score   support
   HR (Clean: 0)       1.00      0.99      1.00       500
LR (Degraded: 1)       0.99      1.00      1.00       500
        accuracy                           1.00      1000
```

---

## Comparison: Official SimCLR Paper/Repo vs. Our HR-LR Implementation

| Component / Hyperparameter | Official SimCLR Paper & Repo (`google-research/simclr`) | Our Implementation (`hr_lr_repo`) | Technical Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Core Paper Reference** | Chen et al., ICML 2020 / arXiv:2002.05709 | Modified SimCLR for Resolution & Blur Sensitivity | Adapted contrastive objective for quality/degradation tasks |
| **Framework / Language** | TensorFlow 1.x / 2.x & TPU Mesh | Native PyTorch with CUDA acceleration | Modern, modular PyTorch implementation |
| **Optimizer** | LARS (`lars_optimizer.py`) | Faithful PyTorch LARS (`model_util.py`) | Exact match: 1D parameter exclusion, weight decay pre-trust ratio, `classic_momentum=True` |
| **Learning Rate Schedule** | Linear Warmup (10 epochs) + Cosine Decay | Linear Warmup (10 epochs) + Cosine Decay | Scaled by $\text{LR} \times \frac{\text{BatchSize}}{256}$ |
| **Loss Function** | NT-Xent ($\tau=0.5$), Sum of $A \to B$ & $B \to A$ | NT-Xent ($\tau=0.5$), Masked Diagonal with $-\infty$ | Faithfully reproduces official asymmetric sum without redundant normalization |
| **Projection Head** | 3-Layer MLP with BatchNorm + ReLU $\to$ 128-d | 3-Layer MLP with BatchNorm + ReLU $\to$ 128-d | Preserves non-linear transformation before contrastive projection space |
| **Gaussian Blur Augmentation** | **Enabled ($p=0.5$)** | **Disabled ($p=0.0$)** | **Critical Difference**: Official SimCLR removes high-frequency spatial shortcuts; we preserve sharpness cues so the network remains sensitive to blur |
| **Input Data Format** | Resized whole ImageNet images ($224 \times 224$) | Dynamic on-the-fly random patch crops from 2K images ($256 \times 256$) | High-resolution patches capture genuine sensor & optical textures |
| **Downstream Evaluation** | ImageNet 1000-class classification | Paired Same-Location HR vs. LR ($\times 4$ Bicubic) | Evaluates representation sensitivity to resolution degradation rather than semantic object class |

---

## Repository Structure

```
hr_lr_repo/
├── README.md                               # Comprehensive project documentation and benchmarks
├── run.py                                  # Self-supervised SimCLR pretraining script
├── finetune_hr_lr.py                       # Paired same-location HR vs LR fine-tuning script
├── linear_eval_hr_lr.py                    # Linear evaluation / probing on frozen representations
├── infer_hr_lr.py                          # Inference script for paired/standalone image evaluation
├── data.py                                 # DIV2K patch dataset and frequency-preserving transforms
├── resnet.py                               # ResNet backbone with customizable projection head
├── objective.py                            # NT-Xent contrastive loss implementation
├── model_util.py                           # LARS optimizer, projection head, LR schedules
├── Dataset/                                # Dataset root directory
│   ├── DIV2K_train_HR/                     # 800 high-resolution training images
│   ├── DIV2K_train_LR_bicubic_X4/X4/       # 800 x4 bicubic downsampled training images
│   ├── DIV2K_train_LR_bicubic/X2/          # 800 x2 bicubic downsampled training images
│   ├── DIV2K_valid_HR/                     # 100 high-resolution validation images
│   ├── DIV2K_valid_LR_bicubic_X4/X4/       # 100 x4 bicubic downsampled validation images
│   └── DIV2K_valid_LR_bicubic/X2/          # 100 x2 bicubic downsampled validation images
└── checkpoints_pretrain_256/               # Pretrained SimCLR model checkpoints (epoch 100)
    └── checkpoint_epoch0100.pth
```

---

## Quickstart & Usage

### Environment Setup
```bash
pip install torch torchvision tqdm Pillow scikit-learn numpy
```

---

### 1. Self-Supervised Pretraining (`run.py`)

To pretrain the ResNet-18 backbone using SimCLR on $256 \times 256$ DIV2K HR patches:

```bash
python run.py \
  --hr_dir Dataset/DIV2K_train_HR \
  --model_dir ./checkpoints_pretrain_256 \
  --patch_size 256 \
  --patches_per_image 50 \
  --resnet_depth 18 \
  --train_batch_size 128 \
  --train_epochs 100 \
  --learning_rate 0.3 \
  --warmup_epochs 10 \
  --checkpoint_epochs 10
```

---

### 2. Paired HR vs LR Fine-Tuning (`finetune_hr_lr.py`)

To fine-tune the pretrained encoder with paired same-location patches ($256 \times 256$ HR vs. $64 \times 64 \to 256 \times 256$ Bicubic LR):

```bash
python finetune_hr_lr.py \
  --hr_dir Dataset/DIV2K_train_HR \
  --lr_dir Dataset/DIV2K_train_LR_bicubic_X4/X4 \
  --checkpoint ./checkpoints_pretrain_256/checkpoint_epoch0100.pth \
  --save_dir ./checkpoints_finetune_256_bicubic \
  --hr_patch_size 256 \
  --patches_per_image 30 \
  --batch_size 32 \
  --epochs 200 \
  --lr_encoder 0.001 \
  --lr_classifier 0.01
```

---

### 3. Linear Evaluation (`linear_eval_hr_lr.py`)

To assess the quality of the frozen pretrained features via logistic regression probing vs. an untrained random baseline:

```bash
python linear_eval_hr_lr.py \
  --checkpoint ./checkpoints_pretrain_256/checkpoint_epoch0100.pth \
  --hr_patch_size 256 \
  --patches_per_image 10
```

---

### 4. Inference & Demo (`infer_hr_lr.py`)

#### Paired Image Quality Verification:
```bash
python infer_hr_lr.py \
  --hr_image Dataset/DIV2K_valid_HR/0801.png \
  --lr_image Dataset/DIV2K_valid_LR_bicubic_X4/X4/0801x4.png \
  --checkpoint ./checkpoints_finetune_256_bicubic/best_paired_classifier.pth
```

**Output:**
```
======================================================================
                     PAIRED HR vs LR PREDICTION
======================================================================
HR Image:   0801.png (2040x1356)
  Prediction: HR (Clean) | Confidence: HR=100.00%, LR=0.00%

LR Image:   0801x4.png (510x339)
  Prediction: LR (Degraded) | Confidence: HR=0.00%, LR=100.00%
======================================================================
```

#### Batch Directory Evaluation:
```bash
python infer_hr_lr.py \
  --input Dataset/DIV2K_valid_HR \
  --checkpoint ./checkpoints_finetune_256_bicubic/best_paired_classifier.pth
```

---

## Checkpoints & Reproducibility

- `checkpoints_pretrain_256/checkpoint_epoch0100.pth`: Pretrained SimCLR encoder after 100 epochs on DIV2K HR patches (Contrast Acc: 88.78%, Loss: 6.4191).
- `checkpoints_finetune_256_bicubic/best_paired_classifier.pth`: Best paired HR-LR classifier weights (**99.98%** Validation Accuracy).

---

## References

1. **SimCLR v1**: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton. *"A Simple Framework for Contrastive Learning of Visual Representations"*, ICML 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709).
2. **SimCLRv2**: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, Geoffrey Hinton. *"Big Self-Supervised Models are Strong Semi-Supervised Learners"*, NeurIPS 2020. [arXiv:2006.10029](https://arxiv.org/abs/2006.10029).
3. **Official SimCLR Repository**: [google-research/simclr](https://github.com/google-research/simclr) (TensorFlow).

