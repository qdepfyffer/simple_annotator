"""
Mask file locations and handling
"Mask" = saved annotations for a single image, saved as a png equal in resolution to the source image
Every pixel in the mask is colored according to the class color assigned to the superpixel it belongs to
This file ONLY handles where those masks go and how they get saved/read

For reference:
photos/img1.jpg             <- If this is your source image
photos_labels/img1_L.png    <- This will be your mask image
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# This determines the suffix that is appended to label images
# _L was used here just because that was the old convention and there's no real reason to change it
MASK_SUFFIX = "_L"


def count_annotated(image_dir: Path | str, extensions: set[str]) -> tuple[int, int]:
    """
    Count how many images in a folder already have masks
    This is calculated based on whether mask_path() exists for each image in the current directory
    """
    image_dir = Path(image_dir)
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]
    # An incomplete mask will still count as "annotated" purely because it exists
    # There's no way for this program to know whether you've properly annotated an image, so this is the simplest approach
    # Count the number of images in the current folder that have an existing mask_path()
    done = sum(1 for p in images if mask_path(p).exists())
    return done, len(images)


def mask_path(image_path: Path) -> Path:
    """
    Determines the full path to the mask image for a given source image
    photos/img1.jpg -> photos_labels/img1.png
    Masks are ALWAYS saved as a .png (throwback to when our masks were being saved as .jpeg) because png is lossless
    """
    image_path = Path(image_path)
    # `stem` gets us the file path without its extension, and pathlib lets us join path components with `/`
    return labels_dir(image_path) / (image_path.stem + MASK_SUFFIX + ".png")


def labels_dir(image_path: Path) -> Path:
    """
    Sibling folder for image masks: <parent>_labels, sits next to the folder holding the images being annotated
    """
    parent = image_path.parent
    return parent.parent / (parent.name + "_labels")


def load_mask(path: Path) -> np.ndarray | None:
    """
    Load RGB mask if it exists as an (H, W, 3) uint8 RGB array or None if there's no mask
    """
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGBA").convert("RGB"))


def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    Write RGB mask array to disk and create labels folder if necessary
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The path here determines the file extension and PIL infers the format from that
    Image.fromarray(mask).save(path)
