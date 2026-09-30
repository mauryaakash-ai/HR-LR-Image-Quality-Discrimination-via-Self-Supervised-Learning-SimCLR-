import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from resnet import ResNetSimCLR


class HRLRPairedDataset(Dataset):
    """
    Same location se HR aur LR patches — content same, quality different.
    HR patch (sharp) → label 0
    LR patch (degraded, same content) → label 1
    Scale factor = 4 (DIV2K X4)
    """
    def __init__(self, hr_dir, lr_dir, hr_patch_size=96,
                 patches_per_image=30, split='train', val_images=100):
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // 4  # 96//4 = 24px from LR
        self.patches_per_image = patches_per_image

        hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])
        lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])

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
        print(f"[{split}] Cached! {len(self.hr_images)} pairs")
        print(f"  HR patch: {hr_patch_size}x{hr_patch_size} | LR patch: {self.lr_patch_size}x{self.lr_patch_size} → resized to {hr_patch_size}x{hr_patch_size}")

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        # LR patch resize to match HR patch size for encoder input
        self.lr_transform = T.Compose([
            T.Resize((hr_patch_size, hr_patch_size),
                     interpolation=T.InterpolationMode.BICUBIC),  # BICUBIC — matches natural SR degradation
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # --- FIX (Error 4): precompute SAME (img_idx, hr_x, hr_y) location
        # for both HR and LR branches, instead of calling random.randint()
        # independently in each branch (which broke same-location pairing).
        self.locations = []
        for img_idx, hr_img in enumerate(self.hr_images):
            hr_w, hr_h = hr_img.size
            max_x = hr_w - self.hr_patch_size
            max_y = hr_h - self.hr_patch_size
            for _ in range(self.patches_per_image):
                hr_x = random.randint(0, max_x)
                hr_y = random.randint(0, max_y)
                self.locations.append((img_idx, hr_x, hr_y))

    def __len__(self):
        # Each precomputed location gives 1 HR + 1 LR sample
        return len(self.locations) * 2

    def __getitem__(self, idx):
        n = len(self.locations)
        if idx < n:
            # HR patch
            img_idx, hr_x, hr_y = self.locations[idx]
            hr_img = self.hr_images[img_idx]
            hr_patch = hr_img.crop((hr_x, hr_y,
                                    hr_x + self.hr_patch_size,
                                    hr_y + self.hr_patch_size))
            return self.to_tensor(hr_patch), 0  # label 0 = HR (clean)

        else:
            # LR patch — SAME (img_idx, hr_x, hr_y) as its paired HR sample
            img_idx, hr_x, hr_y = self.locations[idx - n]
            lr_img = self.lr_images[img_idx]

            # Map to LR coordinates (scale factor = 4)
            lr_x = hr_x // 4
            lr_y = hr_y // 4
            lr_patch = lr_img.crop((lr_x, lr_y,
                                    lr_x + self.lr_patch_size,
                                    lr_y + self.lr_patch_size))
            # Resize LR patch to same size as HR patch (BICUBIC to preserve genuine degradation)
            return self.lr_transform(lr_patch), 1  # label 1 = LR (degraded)


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item() * 100


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune ResNet-18 for Paired HR vs LR Classification')
    parser.add_argument('--hr_dir', type=str, default='Dataset/DIV2K_train_HR',
                        help='Directory containing HR images')
    parser.add_argument('--lr_dir', type=str, default='Dataset/DIV2K_train_LR_bicubic_X4/X4',
                        help='Directory containing LR images (X4 bicubic)')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints_pretrain_256/checkpoint_epoch0100.pth',
                        help='Pretrained SimCLR model checkpoint')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints_finetune_256_bicubic',
                        help='Directory to save fine-tuned checkpoints')
    parser.add_argument('--hr_patch_size', type=int, default=256,
                        help='HR patch size (default: 256)')
    parser.add_argument('--patches_per_image', type=int, default=30,
                        help='Patches extracted per image (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total fine-tuning epochs (default: 200)')
    parser.add_argument('--val_images', type=int, default=100,
                        help='Number of images reserved for validation (default: 100)')
    parser.add_argument('--lr_encoder', type=float, default=0.001,
                        help='Encoder fine-tuning learning rate (default: 0.001)')
    parser.add_argument('--lr_classifier', type=float, default=0.01,
                        help='Classifier learning rate (default: 0.01)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader workers (default: 1)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    hr_dir            = args.hr_dir
    lr_dir            = args.lr_dir
    checkpoint        = args.checkpoint
    hr_patch_size     = args.hr_patch_size
    patches_per_image = args.patches_per_image
    batch_size        = args.batch_size
    epochs            = args.epochs
    num_workers       = args.num_workers

    train_ds = HRLRPairedDataset(hr_dir, lr_dir, hr_patch_size,
                                 patches_per_image, 'train', args.val_images)
    val_ds   = HRLRPairedDataset(hr_dir, lr_dir, hr_patch_size,
                                 patches_per_image, 'val',   args.val_images)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = ResNetSimCLR(depth=18, proj_out_dim=128)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    encoder = model.backbone.to(device)
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True

    classifier = nn.Linear(512, 2).to(device)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    optimizer = torch.optim.SGD([ 
        {'params': encoder.parameters(),    'lr': args.lr_encoder},
        {'params': classifier.parameters(), 'lr': args.lr_classifier}
    ], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting paired HR vs LR fine-tuning ({epochs} epochs)...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        total_loss, total_acc = 0.0, 0.0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            feats  = encoder(imgs)
            logits = classifier(feats)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)
        scheduler.step()

        encoder.eval()
        classifier.eval()
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats  = encoder(imgs)
                logits = classifier(feats)
                val_acc += accuracy(logits, labels)

        train_acc = total_acc / len(train_loader)
        val_acc   = val_acc   / len(val_loader)
        print(f"Epoch [{epoch:2d}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_paired_classifier.pth')
            torch.save({
                'encoder':    encoder.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch, 'val_acc': val_acc},
                save_path)

    print(f"\n=== Best Val Accuracy: {best_acc:.2f}% ===")
    print("HR=0 (clean), LR=1 (degraded) — paired same-location patches")


if __name__ == '__main__':
    main()

