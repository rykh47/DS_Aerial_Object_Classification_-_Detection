## Aerial Object Classification & Detection

Deep learning solution to classify aerial images as Bird or Drone, with an optional object detection component (YOLOv8). Includes data preprocessing, augmentation, model training (custom CNN and transfer learning), evaluation, and a Streamlit app.

### Project Structure
```
.
├── classification_dataset/           # Expected dataset root (train/val/test subfolders)
├── object_detection_Dataset/         # YOLOv8-format dataset root
├── src/
│   ├── data/
│   │   └── datasets.py               # Dataset and augmentations
│   ├── models/
│   │   ├── custom_cnn.py             # Custom CNN classifier
│   │   └── transfer.py               # Transfer learning backbones
│   ├── utils/
│   │   └── common.py                 # Utilities (seed, device, checkpoint)
│   ├── train.py                      # Training CLI
│   ├── evaluate.py                   # Evaluation metrics & plots
│   └── infer.py                      # Inference utilities
├── streamlit_app.py                  # Streamlit UI for classification/detection
├── yolo/
│   ├── data.yaml                     # YOLOv8 dataset config
│   ├── train_yolo.py                 # YOLOv8 training helper
│   └── infer_yolo.py                 # YOLOv8 inference helper
├── requirements.txt
└── README.md
```

### Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Classification Dataset Layout
```
classification_dataset/
  train/
    bird/
    drone/
  valid/
    bird/
    drone/
  test/
    bird/
    drone/
```

### Train (Classification)
```bash
python -m src.train \
  --data_root classification_dataset \
  --arch resnet50 \# options: custom_cnn, resnet50, mobilenet_v3_small, efficientnet_b0
  --img_size 224 \
  --batch_size 32 \
  --epochs 25 \
  --out_dir runs/cls_resnet50
```

### Evaluate
```bash
python -m src.evaluate --data_root classification_dataset --weights runs/cls_resnet50/best.pt --img_size 224
```

### Inference
```bash
python -m src.infer --weights runs/cls_resnet50/best.pt --class_names runs/cls_resnet50/class_names.json --image path/to/image.jpg --img_size 224
```

### YOLOv8 (Optional)
Edit `yolo/data.yaml` to point to your `object_detection_Dataset` splits, then:
```bash
python yolo/train_yolo.py --epochs 50 --imgsz 640 --out_dir runs/yolo
python yolo/infer_yolo.py --weights runs/yolo/weights/best.pt --source path/to/images_or_dir
```

### Streamlit App
```bash
streamlit run streamlit_app.py
```

### Notes
- Set `--arch custom_cnn` to use the provided custom CNN.
- Transfer learning backbones supported: ResNet50, MobileNetV3 Small, EfficientNet-B0.
- Uses Albumentations for robust data augmentation.

