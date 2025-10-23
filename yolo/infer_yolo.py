import argparse
import os

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True, help="image, directory, or video path")
    p.add_argument("--out_dir", type=str, default="runs/yolo_infer")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model = YOLO(args.weights)
    model.predict(source=args.source, project=args.out_dir, name="", save=True)


if __name__ == "__main__":
    main()


