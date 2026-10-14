"""
Per-image annotation state
AnnotationSession owns everything about a single open image
Class index 0 is the default and will be colored in saved masks, but not the on-screen overlay
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

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
        self._boundary_mask = find_boundaries(labels, mode="inner")
        self._slices = {p.label - 1: p.slice for p in regionprops(labels + 1)}
        self.classes = classes
        self.mask_path = mask_path(image_path)
        
        n_segments = int(labels.max()) + 1
        self.segment_class = np.zeros(n_segments, dtype=int)
        
        self._undo: list[tuple[int, int, int]] = []
        self._redo: list[tuple[int, int, int]] = []
        self.dirty = False
        self.load_warning: str | None = None  # Error message to show user when a load issue occurs

        self._display: np.ndarray | None = None
        self._display_key: tuple[tuple[int, int, int], float, bool] = ((255, 255, 0), 0.5, True)
        self._load_existing()

    @property
    def annotated(self) -> bool:
        """True if session has any non-default superpixel annotations"""
        return bool(self.segment_class.any())

    # === EDITING ======================================================================================================

    def assign(self, x: int, y: int, class_index: int) -> bool:
        """Assign superpixel under (x, y) to the specified class, and return true if the superpixel was changed"""
        sid = int(self.labels[y, x])
        old = int(self.segment_class[sid])
        if old == class_index:
            return False
        self.segment_class[sid] = class_index
        self._patch(sid)
        self._undo.append((sid, old, class_index))
        self._redo.clear()
        self.dirty = True
        return True


    def undo(self) -> bool:
        if not self._undo:
            return False
        sid, old, new = self._undo.pop()
        self.segment_class[sid] = old
        self._patch(sid)
        self._redo.append((sid, old, new))
        self.dirty = True
        return True


    def redo(self) -> bool:
        if not self._redo:
            return False
        sid, old, new = self._redo.pop()
        self.segment_class[sid] = new
        self._patch(sid)
        self._undo.append((sid, old, new))
        self.dirty = True
        return True

    # === RENDERING ====================================================================================================

    def render_display(self, boundary_color: tuple[int, int, int] = (255, 255, 0), alpha: float = 0.5,
                       boundaries: bool = True) -> np.ndarray:
        """Returns the cached picture"""
        key = (boundary_color, alpha, boundaries)
        display = self._display
        if display is None or key != self._display_key:
            display = self._build_display(boundary_color, alpha, boundaries)
            self._display = display
            self._display_key = key
        return display


    def render_mask(self) -> np.ndarray:
        """Opaque RGB mask that gets saved where pixels are colored by their classes"""
        return self._colors()[self.segment_class][self.labels]


    def _build_display(self, boundary_color: tuple[int, int, int], alpha: float,
                       boundaries: bool) -> np.ndarray:
        """Image + superpixel borders with non-default segments tinted"""
        display = self.image.copy()
        painted_px = (self.segment_class != 0)[self.labels]
        tint = self.render_mask()
        display[painted_px] = ((1 - alpha) * display[painted_px] + alpha * tint[painted_px]).astype(np.uint8)
        if boundaries:
            display[self._boundary_mask] = boundary_color
        return display


    def _colors(self) -> np.ndarray:
        return np.array([c.color for c in self.classes], dtype=np.uint8)


    def _patch(self, sid: int) -> None:
        """Redraw one superpixel within the cached display"""
        display = self._display
        if display is None:
            return
        boundary_color, alpha, boundaries = self._display_key
        box = self._slices[sid]
        inside = self.labels[box] == sid
        region = display[box]
        pixels = self.image[box][inside]
        class_index = self.segment_class[sid]
        if class_index != 0:
            pixels = ((1 - alpha) * pixels + alpha * self._colors()[class_index]).astype(np.uint8)
        region[inside] = pixels
        if boundaries:
            region[inside & self._boundary_mask[box]] = boundary_color

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
