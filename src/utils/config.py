"""Configuration loading and project path resolution."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """Dict with attribute access and project-relative path resolution."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve a key under ``paths:`` to an absolute path and create it."""
        p = PROJECT_ROOT / self["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we rely on, including PyTorch when it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch absent or its CUDA libraries are broken; NumPy seeding still applies.
        pass
