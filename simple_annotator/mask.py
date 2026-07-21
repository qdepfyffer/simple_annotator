"""Mask file locations and handling"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MASK_SUFFIX = "_L"


def count_annotated(image_dir: Path | str, extensions: set[str]) -> tuple[int, int]:
    """Count how many images in a folder already have masks"""
    image_dir = Path(image_dir)
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]
    done = sum(1 for p in images if mask_path(p).exists())
    return done, len(images)


def mask_path(image_path: Path) -> Path:
    """Full path to mask image for a given source image"""
    image_path = Path(image_path)
    return labels_dir(image_path) / (image_path.stem + MASK_SUFFIX + ".png")


def labels_dir(image_path: Path) -> Path:
    """Sibling folder for image masks: <parent>_labels"""
    parent = image_path.parent
    return parent.parent / (parent.name + "_labels")


def load_mask(path: Path) -> np.ndarray | None:
    """Load RGB mask if it exists"""
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGBA").convert("RGB"))


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Write RGB mask array to disk and create labels folder if necessary"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)
