import argparse
import os

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="yolo/data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out_dir", type=str, default="runs/yolo")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, project=args.out_dir, name="")


if __name__ == "__main__":
    main()


