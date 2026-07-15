from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_summary(model: torch.nn.Module) -> str:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    frozen = total - trainable
    return (
        f"Total params     : {total:>12,}\n"
        f"Trainable params : {trainable:>12,}\n"
        f"Frozen params    : {frozen:>12,}"
    )


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path):
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
