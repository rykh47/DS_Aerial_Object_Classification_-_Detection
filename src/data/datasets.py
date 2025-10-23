import os
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


def get_transforms(img_size: int = 224, is_train: bool = True) -> A.Compose:
    if is_train:
        return A.Compose(
            [
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=20, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=1, max_height=int(0.1 * img_size), max_width=int(0.1 * img_size), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )


class AlbumentationsImageFolder(Dataset):
    def __init__(self, root: str, transform: Optional[A.Compose] = None) -> None:
        self.base = datasets.ImageFolder(root=root)
        self.transform = transform
        self.samples = self.base.samples
        self.class_to_idx = self.base.class_to_idx
        self.classes = self.base.classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(image), target


def build_dataloaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    test_dir = os.path.join(data_root, "test")

    train_ds = AlbumentationsImageFolder(train_dir, get_transforms(img_size, True))
    valid_ds = AlbumentationsImageFolder(valid_dir, get_transforms(img_size, False))
    test_ds = AlbumentationsImageFolder(test_dir, get_transforms(img_size, False))

    class_names = train_ds.classes

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype=torch.long)
        return images, targets

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader, class_names


