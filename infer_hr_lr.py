import os
import argparse
import random
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from resnet import get_resnet


def parse_args():
    parser = argparse.ArgumentParser(description='Inference and Evaluation for HR vs LR Image Classifier')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to single image file or directory of images')
    parser.add_argument('--hr_image', type=str, default=None,
                        help='Path to HR reference image (for paired test)')
    parser.add_argument('--lr_image', type=str, default=None,
                        help='Path to LR degraded image (for paired test)')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints_finetune_256_bicubic/best_paired_classifier.pth',
                        help='Path to fine-tuned classifier checkpoint')
    parser.add_argument('--resnet_depth', type=int, default=18,
                        help='ResNet depth (default: 18)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size for evaluation (default: 256)')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Scale factor for downsampled LR images (default: 4)')
    return parser.parse_args()


class HRLRClassifier(nn.Module):
    def __init__(self, depth=18, num_classes=2):
        super(HRLRClassifier, self).__init__()
        self.encoder, in_dim = get_resnet(depth=depth)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits


def predict_patch(patch_img, model, transform, device='cuda'):
    tensor = transform(patch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(logits.argmax(dim=1).item())
    return pred_label, probs


def evaluate_paired_images(hr_path, lr_path, model, transform, patch_size=256, scale_factor=4, device='cuda'):
    img_hr = Image.open(hr_path).convert('RGB')
    img_lr = Image.open(lr_path).convert('RGB')
    w_hr, h_hr = img_hr.size

    # Choose center coordinate
    max_x = max(0, w_hr - patch_size)
    max_y = max(0, h_hr - patch_size)
    x = max_x // 2
    y = max_y // 2

    # Crop HR
    patch_hr = img_hr.crop((x, y, x + patch_size, y + patch_size))
    pred_hr, prob_hr = predict_patch(patch_hr, model, transform, device)

    # Crop paired LR at matching spatial coordinates and bicubic resize
    lr_patch_size = patch_size // scale_factor
    lr_x, lr_y = x // scale_factor, y // scale_factor
    patch_lr = img_lr.crop((lr_x, lr_y, lr_x + lr_patch_size, lr_y + lr_patch_size)).resize((patch_size, patch_size), Image.BICUBIC)
    pred_lr, prob_lr = predict_patch(patch_lr, model, transform, device)

    print("\n" + "="*70)
    print("                     PAIRED HR vs LR PREDICTION")
    print("="*70)
    print(f"HR Image:   {os.path.basename(hr_path)} ({w_hr}x{h_hr})")
    print(f"  Prediction: {'HR (Clean)' if pred_hr == 0 else 'LR (Degraded)'} | Confidence: HR={prob_hr[0]*100:.2f}%, LR={prob_hr[1]*100:.2f}%")
    print(f"\nLR Image:   {os.path.basename(lr_path)} ({img_lr.size[0]}x{img_lr.size[1]})")
    print(f"  Prediction: {'LR (Degraded)' if pred_lr == 1 else 'HR (Clean)'} | Confidence: HR={prob_lr[0]*100:.2f}%, LR={prob_lr[1]*100:.2f}%")
    print("="*70)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = HRLRClassifier(depth=args.resnet_depth, num_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder'])
    model.classifier.load_state_dict(ckpt['classifier'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (Val Acc: {ckpt.get('val_acc', 0.0):.2f}%)")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if args.hr_image and args.lr_image:
        evaluate_paired_images(args.hr_image, args.lr_image, model, transform,
                               args.patch_size, args.scale_factor, device)
        return

    if args.input is None:
        # Default run paired test on valid set sample
        hr_sample = 'Dataset/DIV2K_valid_HR/0801.png'
        lr_sample = 'Dataset/DIV2K_valid_LR_bicubic_X4/X4/0801x4.png'
        if os.path.exists(hr_sample) and os.path.exists(lr_sample):
            print("No input specified. Running demo paired evaluation on DIV2K validation sample...")
            evaluate_paired_images(hr_sample, lr_sample, model, transform,
                                   args.patch_size, args.scale_factor, device)
            return
        else:
            raise ValueError("Please provide --input or both --hr_image and --lr_image.")

    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        raise ValueError(f"Input path not found: {args.input}")

    print(f"\nEvaluating {len(files)} image patch(es)...")
    print("-" * 75)
    print(f"{'Image Name':<35} | {'Prediction':<25} | {'HR Prob':<8} | {'LR Prob':<8}")
    print("-" * 75)
    for fpath in files:
        fname = os.path.basename(fpath)
        img = Image.open(fpath).convert('RGB')
        w, h = img.size
        if w >= args.patch_size and h >= args.patch_size:
            patch = img.crop(((w-args.patch_size)//2, (h-args.patch_size)//2,
                              (w-args.patch_size)//2 + args.patch_size,
                              (h-args.patch_size)//2 + args.patch_size))
        else:
            patch = img.resize((args.patch_size, args.patch_size), Image.BICUBIC)
        pred, probs = predict_patch(patch, model, transform, device)
        lbl_name = 'HR (Clean: 0)' if pred == 0 else 'LR (Degraded: 1)'
        print(f"{fname:<35} | {lbl_name:<25} | {probs[0]*100:6.2f}% | {probs[1]*100:6.2f}%")
    print("-" * 75)


if __name__ == '__main__':
    main()
