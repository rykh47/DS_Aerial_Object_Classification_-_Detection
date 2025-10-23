import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.data.datasets import build_dataloaders
from src.models.custom_cnn import build_model as build_custom
from src.models.transfer import build_model as build_transfer
from src.utils.common import get_device, load_checkpoint, ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--arch", type=str, default="resnet50", choices=["custom_cnn", "resnet50", "mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", type=str, default="runs/eval")
    p.add_argument("--prefer_gpu", action="store_true")
    return p.parse_args()


def plot_confusion(cm: np.ndarray, class_names: List[str], out_path: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate():
    args = parse_args()
    device = get_device(prefer_gpu=args.prefer_gpu)
    ensure_dir(args.out_dir)

    _, _, test_loader, class_names = build_dataloaders(
        data_root=args.data_root, img_size=args.img_size, batch_size=args.batch_size, num_workers=0
    )

    num_classes = len(class_names)
    if args.arch == "custom_cnn":
        model = build_custom(num_classes)
    else:
        model = build_transfer(args.arch, num_classes=num_classes, pretrained=False)

    ckpt = load_checkpoint(args.weights, map_location=str(device))
    model.load_state_dict(ckpt.model_state)
    model.to(device)
    model.eval()

    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_tgts.extend(targets.cpu().tolist())

    acc = accuracy_score(all_tgts, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_tgts, all_preds, average="binary")
    cm = confusion_matrix(all_tgts, all_preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_tgts, all_preds, target_names=class_names))

    plot_confusion(cm, class_names, os.path.join(args.out_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    evaluate()


