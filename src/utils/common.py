import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # macOS Metal
        return torch.device("mps")
    return torch.device("cpu")


def select_num_workers(default: int = 2) -> int:
    try:
        import multiprocessing as mp

        cpu_count = mp.cpu_count()
        return max(0, min(default, cpu_count - 1))
    except Exception:
        return 0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    epoch: int
    best_metric: float


def save_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "epoch": checkpoint.epoch,
            "best_metric": checkpoint.best_metric,
        },
        path,
    )


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Checkpoint:
    obj = torch.load(path, map_location=map_location)
    return Checkpoint(
        model_state=obj["model_state"],
        optimizer_state=obj.get("optimizer_state", {}),
        epoch=int(obj.get("epoch", -1)),
        best_metric=float(obj.get("best_metric", 0.0)),
    )


