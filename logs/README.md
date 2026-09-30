# Structured Experiment Logs

This directory contains organized, structured logs for all self-supervised pretraining, paired fine-tuning, and linear evaluation experiments.

---

## Directory Hierarchy

```text
logs/
|-- pretraining/
|   |-- simclr_pretrain_256_100epochs.log        # Main SimCLR pretraining (256x256 patches, 100 epochs)
|   |-- simclr_pretrain_96_blur_50epochs.log     # Baseline pretraining with Gaussian blur (96x96 patches)
|   `-- simclr_pretrain_96_noblur_50epochs.log   # Ablation pretraining without Gaussian blur (96x96 patches)
|-- finetuning/
|   |-- finetune_256_paired_bicubic.log          # Best paired HR-LR fine-tuning (256x256, 99.98% Val Acc)
|   |-- finetune_256_paired_bicubic_samelocation.log # Paired fine-tuning validation run (99.97% Val Acc)
|   |-- finetune_96_paired_samelocation.log      # Paired fine-tuning on 96x96 patches (99.93% Val Acc)
|   |-- finetune_96_unpaired_end2end.log         # Unaligned crop fine-tuning (83.71% Val Acc)
|   `-- finetune_96_unpaired_linear_probe.log    # Unaligned crop linear probe (67.61% Val Acc)
`-- linear_eval/
    `-- linear_eval_div2k.log                    # Frozen feature linear probing vs random baseline (99.60%)
```

---

## 1. Pretraining Logs (`logs/pretraining/`)

| Log File | Architecture | Patch Size | Augmentation | Epochs | Final Loss | Contrast Accuracy | Key Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `simclr_pretrain_256_100epochs.log` | ResNet-18 | 256 x 256 | No-Blur (Sharpness Preserving) | 100 | 6.4191 | **88.78%** (Peak: 89.20%) | Main model checkpoint for downstream tasks |
| `simclr_pretrain_96_noblur_50epochs.log` | ResNet-18 | 96 x 96 | No-Blur (Sharpness Preserving) | 50 | 8.1094 | **64.59%** | Disabling blur preserves high-frequency details |
| `simclr_pretrain_96_blur_50epochs.log` | ResNet-18 | 96 x 96 | With Gaussian Blur (Official) | 50 | 8.1534 | **62.56%** | Baseline with official SimCLR blur |

---

## 2. Fine-Tuning Logs (`logs/finetuning/`)

| Log File | Patch Size | Sampling / Pairing Strategy | Train Split | Val Split | Train Acc | Best Val Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `finetune_256_paired_bicubic.log` | 256 x 256 | Paired Same-Location (x4 Bicubic) | 700 pairs | 100 pairs | 99.94% | **99.98%** |
| `finetune_256_paired_bicubic_samelocation.log` | 256 x 256 | Paired Same-Location (x4 Bicubic) | 700 pairs | 100 pairs | 99.94% | **99.97%** |
| `finetune_96_paired_samelocation.log` | 96 x 96 | Paired Same-Location (x4 Bicubic) | 700 pairs | 100 pairs | 99.77% | **99.93%** |
| `finetune_96_unpaired_end2end.log` | 96 x 96 | Unaligned / Independent Crops | 700 pairs | 100 pairs | 84.92% | **83.71%** |
| `finetune_96_unpaired_linear_probe.log` | 96 x 96 | Unaligned / Independent Crops | 700 pairs | 100 pairs | 68.30% | **67.61%** |

---

## 3. Linear Evaluation Logs (`logs/linear_eval/`)

| Log File | Protocol | Evaluated Features | Train Acc | Val Acc | Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `linear_eval_div2k.log` | Logistic Regression Probe | Frozen SimCLR Pretrained (256x256) | 99.96% | **99.60%** | **+20.80%** |
| `linear_eval_div2k.log` | Logistic Regression Probe | Random Untrained Encoder Baseline | 79.10% | **78.80%** | Baseline |
