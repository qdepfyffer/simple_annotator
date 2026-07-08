"""
Per-image annotation state
AnnotationSession owns everything about a single open image
Class index 0 is the default and will be colored in saved masks, but not the on-screen overlay
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from .mask import load_mask, mask_path, save_mask


@dataclass(frozen=True)
class Class:
    name: str
    color: tuple[int, int, int]
    

DEFAULT_CLASSES: tuple[Class, ...] = (
    Class("Noise", (165, 42, 42)),
    Class("Sunlit", (0, 255, 0)),
)


class AnnotationSession:
    def __init__(self, image_path: Path, image: np.ndarray, labels: np.ndarray,
                 classes: tuple[Class, ...] = DEFAULT_CLASSES) -> None:
        self.image_path = Path(image_path)
        self.image = image
        self.labels = labels
        self.classes = classes
        self.mask_path = mask_path(image_path)
        
        n_segments = int(labels.max()) + 1
        self.segment_class = np.zeros(n_segments, dtype=int)
        
        self._undo: list[tuple[int, int, int]] = []
        self._redo: list[tuple[int, int, int]] = []
        self.dirty = False
        self.load_warning: str | None = None  # Error message to show user when a load issue occurs

        self._load_existing()

    # === EDITING ======================================================================================================

    def assign(self, x: int, y: int, class_index: int) -> bool:
        """Assign superpixel under (x, y) to the specified class, and return true if the superpixel was changed"""
        sid = int(self.labels[y, x])
        old = int(self.segment_class[sid])
        if old == class_index:
            return False
        self.segment_class[sid] = class_index
        self._undo.append((sid, old, class_index))
        self._redo.clear()
        self.dirty = True
        return True

    def undo(self) -> bool:
        if not self._undo:
            return False
        sid, old, new = self._undo.pop()
        self.segment_class[sid] = old
        self._redo.append((sid, old, new))
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        sid, old, new = self._redo.pop()
        self.segment_class[sid] = new
        self._undo.append((sid, old, new))
        self.dirty = True
        return True

    # === RENDERING ====================================================================================================

    def _colors(self) -> np.ndarray:
        return np.array([c.color for c in self.classes], dtype=np.uint8)

    def render_mask(self) -> np.ndarray:
        """Opaque RGB mask that gets saved where pixels are colored by their classes"""
        return self._colors()[self.segment_class][self.labels]

    def render_display(self, boundary_color: tuple[int, int, int] = (1, 1, 0), alpha: float = 0.5) -> np.ndarray:
        """Image + superpixel borders with non-default segments tinted"""
        display = self.image.astype(np.float64)
        painted_px = (self.segment_class !=0)[self.labels]
        mask_rgb = self.render_mask().astype(np.float64)
        display[painted_px] = (1 - alpha) * display[painted_px] + alpha * mask_rgb[painted_px]
        display = display.astype(np.uint8)
        return img_as_ubyte(mark_boundaries(img_as_float(display), self.labels, color=boundary_color, mode="inner"))

    # === PERSISTENCE ==================================================================================================

    def _load_existing(self) -> None:
        """Reconstruct assignments from existing mask"""
        existing = load_mask(self.mask_path)
        if existing is None:
            return
        if existing.shape[:2] != self.image.shape[:2]:
            mask_h, mask_w = existing.shape[:2]
            image_h, image_w = self.image.shape[:2]
            self.load_warning = (f"Existing mask {self.mask_path.name} is {mask_w}x{mask_h} "
                                 f"but the image is {image_w}x{image_h}, so it was not loaded.")  # Shape mismatch err
            return
        flat_labels = self.labels.ravel()
        flat_mask = existing.reshape(-1, 3)
        uniq, first_idx = np.unique(flat_labels, return_index=True)
        rep_colors = flat_mask[first_idx]
        color_to_class: dict[tuple[int, ...], int] = {tuple(c.color): i for i, c in enumerate(self.classes)}
        for sid, color in zip(uniq, rep_colors):
            self.segment_class[sid] = color_to_class.get(tuple(int(v) for v in color), 0)


    def save(self) -> None:
        save_mask(self.render_mask(), self.mask_path)
        self.dirty = False
