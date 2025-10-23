import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src.data.datasets import build_dataloaders
from src.models.custom_cnn import build_model as build_custom
from src.models.transfer import build_model as build_transfer
from src.utils.common import set_seed, get_device, select_num_workers, ensure_dir, save_json, save_checkpoint, Checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--arch", type=str, default="resnet50", choices=["custom_cnn", "resnet50", "mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--prefer_gpu", action="store_true")
    return p.parse_args()


def evaluate_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            losses.append(loss.item())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_tgts.extend(targets.cpu().tolist())
    acc = accuracy_score(all_tgts, all_preds)
    return float(sum(losses) / max(1, len(losses))), float(acc)


def train():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_gpu=args.prefer_gpu)
    ensure_dir(args.out_dir)

    num_workers = select_num_workers(4)
    train_loader, valid_loader, _, class_names = build_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )
    save_json({"class_names": class_names}, os.path.join(args.out_dir, "class_names.json"))

    num_classes = len(class_names)
    if args.arch == "custom_cnn":
        model = build_custom(num_classes)
    else:
        model = build_transfer(args.arch, num_classes=num_classes, pretrained=True)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc = evaluate_epoch(model, valid_loader, device)

        print(f"Epoch {epoch:03d}: train_loss={running_loss/len(train_loader):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            ckpt_path = os.path.join(args.out_dir, "best.pt")
            save_checkpoint(
                ckpt_path,
                Checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    best_metric=best_acc,
                ),
            )
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation accuracy: {best_acc:.4f}. Weights saved to {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    train()


