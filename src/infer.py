import argparse
import json
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch

from src.models.custom_cnn import build_model as build_custom
from src.models.transfer import build_model as build_transfer
from src.utils.common import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--class_names", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--arch", type=str, default="resnet50", choices=["custom_cnn", "resnet50", "mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--prefer_gpu", action="store_true")
    return p.parse_args()


def build_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def main():
    args = parse_args()
    device = get_device(prefer_gpu=args.prefer_gpu)
    with open(args.class_names, "r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)["class_names"]

    num_classes = len(class_names)
    if args.arch == "custom_cnn":
        model = build_custom(num_classes)
    else:
        model = build_transfer(args.arch, num_classes=num_classes, pretrained=False)
    ckpt = load_checkpoint(args.weights, map_location=str(device))
    model.load_state_dict(ckpt.model_state)
    model.to(device)
    model.eval()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = build_transform(args.img_size)
    image = t(image=image)["image"]
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        confidence = float(probs[pred_idx])

    print(json.dumps({"prediction": pred_name, "confidence": confidence}, indent=2))


if __name__ == "__main__":
    main()


