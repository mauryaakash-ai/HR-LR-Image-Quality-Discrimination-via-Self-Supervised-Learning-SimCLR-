import os
import argparse
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from resnet import ResNetSimCLR


def parse_args():
    parser = argparse.ArgumentParser(description='Linear Evaluation on DIV2K HR vs LR Paired Dataset')
    parser.add_argument('--hr_dir', type=str, default='Dataset/DIV2K_train_HR',
                        help='Directory containing HR images')
    parser.add_argument('--lr_dir', type=str, default='Dataset/DIV2K_train_LR_bicubic_X4/X4',
                        help='Directory containing LR images (X4 bicubic)')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints_pretrain_256/checkpoint_epoch0100.pth',
                        help='Path to pretrained SimCLR model checkpoint')
    parser.add_argument('--resnet_depth', type=int, default=18,
                        help='ResNet backbone depth (default: 18)')
    parser.add_argument('--proj_out_dim', type=int, default=128,
                        help='Projection output dimension (default: 128)')
    parser.add_argument('--hr_patch_size', type=int, default=256,
                        help='HR patch size (default: 256)')
    parser.add_argument('--patches_per_image', type=int, default=10,
                        help='Number of patches extracted per image (default: 10)')
    parser.add_argument('--val_images', type=int, default=100,
                        help='Number of images reserved for validation split')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction (default: 64)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


class HRLRPairedExtractDataset(Dataset):
    """
    Dataset extracting paired same-location HR (label 0) and LR (label 1) patches
    for linear probing feature extraction.
    """
    def __init__(self, hr_dir, lr_dir, hr_patch_size=256, patches_per_image=10,
                 split='train', val_images=100, seed=42):
        random.seed(seed)
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // 4
        self.patches_per_image = patches_per_image

        hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if split == 'train':
            hr_files = hr_files[val_images:]
            lr_files = lr_files[val_images:]
        else:
            hr_files = hr_files[:val_images]
            lr_files = lr_files[:val_images]

        print(f"[{split}] Loading {len(hr_files)} paired images...")
        self.hr_images, self.lr_images = [], []
        for f in hr_files:
            self.hr_images.append(Image.open(os.path.join(hr_dir, f)).convert('RGB'))
        for f in lr_files:
            self.lr_images.append(Image.open(os.path.join(lr_dir, f)).convert('RGB'))

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.lr_transform = T.Compose([
            T.Resize((hr_patch_size, hr_patch_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.locations = []
        for img_idx, hr_img in enumerate(self.hr_images):
            hr_w, hr_h = hr_img.size
            max_x = max(0, hr_w - self.hr_patch_size)
            max_y = max(0, hr_h - self.hr_patch_size)
            for _ in range(self.patches_per_image):
                hr_x = random.randint(0, max_x) if max_x > 0 else 0
                hr_y = random.randint(0, max_y) if max_y > 0 else 0
                self.locations.append((img_idx, hr_x, hr_y))

    def __len__(self):
        return len(self.locations) * 2

    def __getitem__(self, idx):
        n = len(self.locations)
        if idx < n:
            img_idx, hr_x, hr_y = self.locations[idx]
            hr_img = self.hr_images[img_idx]
            hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.hr_patch_size, hr_y + self.hr_patch_size))
            return self.to_tensor(hr_patch), 0
        else:
            img_idx, hr_x, hr_y = self.locations[idx - n]
            lr_img = self.lr_images[img_idx]
            lr_x, lr_y = hr_x // 4, hr_y // 4
            lr_patch = lr_img.crop((lr_x, lr_y, lr_x + self.lr_patch_size, lr_y + self.lr_patch_size))
            return self.lr_transform(lr_patch), 1


def extract_features(encoder, data_loader, device):
    encoder.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            feats = encoder(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build datasets
    train_ds = HRLRPairedExtractDataset(args.hr_dir, args.lr_dir, args.hr_patch_size,
                                        args.patches_per_image, split='train',
                                        val_images=args.val_images, seed=args.seed)
    val_ds = HRLRPairedExtractDataset(args.hr_dir, args.lr_dir, args.hr_patch_size,
                                      args.patches_per_image, split='val',
                                      val_images=args.val_images, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 1. Evaluate Pretrained Encoder (Frozen)
    print(f"\nLoading pretrained checkpoint: {args.checkpoint}")
    model = ResNetSimCLR(depth=args.resnet_depth, proj_out_dim=args.proj_out_dim)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    encoder = model.backbone.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("\nExtracting pretrained features...")
    X_train, y_train = extract_features(encoder, train_loader, device)
    X_val, y_val = extract_features(encoder, val_loader, device)
    print(f"Pretrained Train features: {X_train.shape} | Val features: {X_val.shape}")

    # Normalize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Train linear classifier on frozen representations
    print("\nFitting Logistic Regression linear probe on pretrained features...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
    clf.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_s)) * 100
    val_acc = accuracy_score(y_val, clf.predict(X_val_s)) * 100

    # 2. Evaluate Random Untrained Encoder (Baseline)
    print("\nEvaluating Random Untrained Encoder Baseline...")
    rand_model = ResNetSimCLR(depth=args.resnet_depth, proj_out_dim=args.proj_out_dim)
    rand_encoder = rand_model.backbone.to(device)
    rand_encoder.eval()

    X_train_rand, y_train_rand = extract_features(rand_encoder, train_loader, device)
    X_val_rand, y_val_rand = extract_features(rand_encoder, val_loader, device)

    scaler_rand = StandardScaler()
    X_train_rand_s = scaler_rand.fit_transform(X_train_rand)
    X_val_rand_s = scaler_rand.transform(X_val_rand)

    clf_rand = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
    clf_rand.fit(X_train_rand_s, y_train_rand)
    rand_val_acc = accuracy_score(y_val_rand, clf_rand.predict(X_val_rand_s)) * 100

    print("\n" + "="*60)
    print("           LINEAR EVALUATION RESULTS (DIV2K HR vs LR)")
    print("="*60)
    print(f"Random Encoder Baseline Val Accuracy:     {rand_val_acc:.2f}%")
    print(f"Pretrained Encoder Train Accuracy:        {train_acc:.2f}%")
    print(f"Pretrained Encoder Validation Accuracy:   {val_acc:.2f}%")
    print(f"Representation Quality Gain:              +{val_acc - rand_val_acc:.2f}%")
    print("="*60)
    print("\nClassification Report (Validation Split):")
    print(classification_report(y_val, clf.predict(X_val_s), target_names=['HR (Clean: 0)', 'LR (Degraded: 1)']))


if __name__ == '__main__':
    main()
