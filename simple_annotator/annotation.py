"""
Per-image annotation state

The annotation session owns everything about the singular open image. All of the state pertaining to the open image should live here.

Class index 0 (non-sunlit for the CIWA+ project) is the default and will be colored in saved masks, but not the on-screen overlay.
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
    """
    Label categories - their names and the colors assigned to them
    """
    name: str
    color: tuple[int, int, int]
    
# Position in this tuple IS THE CLASS INDEX - Index 0 will be treated as the default
DEFAULT_CLASSES: tuple[Class, ...] = (
    Class("Noise", (165, 42, 42)),
    Class("Sunlit", (0, 255, 0)),
)


class AnnotationSession:
    """Contains all mutable state for a single open image. New AnnotationSessions are created whenever a new image is opened`"""
    def __init__(self, image_path: Path, image: np.ndarray, labels: np.ndarray,
                 classes: tuple[Class, ...] = DEFAULT_CLASSES) -> None:
        """
        Initialize AnnotationSession from a segmented image
        """
        # Set image path and path to mask being created
        self.image_path = Path(image_path)
        self.mask_path = mask_path(image_path)

        self.image = image  # (H, W, 3) uint8 RGB
        self.labels = labels  # (H, W) int superpixel map produced by segmenter
        self.classes = classes


        self._boundary_mask = find_boundaries(labels, mode="inner")
        self._slices = {p.label - 1: p.slice for p in regionprops(labels + 1)}
        
        n_segments = int(labels.max()) + 1  # Superpixel ids are 0..max, max + 1 segments total
        self.segment_class = np.zeros(n_segments, dtype=int)  # One class id per superpixel

        # Initialize undo/redo stacks 
        self._undo: list[tuple[int, int, int]] = []
        self._redo: list[tuple[int, int, int]] = []
        self.dirty = False  # True if changes haven't been written to disk, so False when the image is first opened

        # Cached display, None until the display is first built + rendered
        self._display: np.ndarray | None = None

        # Stores values the cached display was built with so we can rebuild if the user requests a display built with different values
        self._display_key: tuple[tuple[int, int, int], float, bool] = ((255, 255, 0), 0.5, True)  

        self.load_warning: str | None = None  # Error message to show user when a load issue occurs

        # Call _load_existing() at the end so if a mask exists we can load it after everything about the AnnotationSession is initialized
        self._load_existing()

    @property
    def annotated(self) -> bool:
        """
        True if the AnnotationSession has any non-default superpixel annotations (at least one superpixel has been annotated as a non-default class)
        """
        return bool(self.segment_class.any())

    # === EDITING ======================================================================================================

    def assign(self, x: int, y: int, class_index: int) -> bool:
        """
        Assign superpixel under (x, y) to the specified class
        Return true if the superpixel was changed
        Return false if not (so we can avoid a redraw and not push a no-op onto the undo stack)
        """
        sid = int(self.labels[y, x])  # Superpixel ID where clicked
        old = int(self.segment_class[sid])  # Class of that current superpixel
        if old == class_index:
            return False  # No change, return false
        self.segment_class[sid] = class_index
        self._patch(sid)  # Update cached display just for this superpixel
        self._undo.append((sid, old, class_index))
        self._redo.clear()  # New edits need to clear redo stack
        self.dirty = True
        return True


    def undo(self) -> bool:
        """
        Undo the most recent edit
        Return false if there's nothing to undo
        """
        if not self._undo:
            return False
        sid, old, new = self._undo.pop()  # int superpixel id, int class before, int class after
        self.segment_class[sid] = old  # Restore the superpixel to its previously held class
        self._patch(sid)
        self._redo.append((sid, old, new))  # Append this undo to the redo stack (lol)
        self.dirty = True
        return True


    def redo(self) -> bool:
        """
        Redo the most recently undone edit
        Return false if there's nothing to redo
        """
        if not self._redo:
            return False
        sid, old, new = self._redo.pop()  # int superpixel id, int class before, int class after
        self.segment_class[sid] = new  # Restore the superpixel to its previously held class
        self._patch(sid)
        self._undo.append((sid, old, new)) # Push the reversed edit back onto the undo stack
        self.dirty = True
        return True

    def reset(self) -> bool:
        """
        Assign the default class back to every superpixel
        Return true if something changes
        """
        if not self.annotated:
            return False

        # Reset every segment's class to 0 (0 is treated as the default class)
        self.segment_class[:] = 0

        # Clear undo and redo caches since the states they existed for no longer exist
        # We could (probably) make the reset undo-able but for now that's out of scope (I don't want to right now)
        self._undo.clear()
        self._redo.clear()

        # Clear display so it gets fully refreshed next time it's drawn
        self._display = None

        # Mask has changed = dirty
        self.dirty = True
        return True

    # === RENDERING ====================================================================================================

    def render_display(self, boundary_color: tuple[int, int, int] = (255, 255, 0), alpha: float = 0.5,
                       boundaries: bool = True) -> np.ndarray:
        """
        Returns the cached picture to be displayed on-screen
        Full rebuild of the image only happens the very first time or when the display key is changed
        Edits are incrementally applied so a full rebuild isn't necessary with every annotation
        """
        key = (boundary_color, alpha, boundaries)  # Cache key so we can determine when we need to fully rebuild the display
        display = self._display
        if display is None or key != self._display_key:
            display = self._build_display(boundary_color, alpha, boundaries)
            self._display = display
            self._display_key = key
        return display


    def render_mask(self) -> np.ndarray:
        """
        Opaque (H, W, 3) RGB mask that gets saved, where pixels are colored by their classes by ID
        """
        return self._colors()[self.segment_class][self.labels]


    def _build_display(self, boundary_color: tuple[int, int, int], alpha: float,
                       boundaries: bool) -> np.ndarray:
        """
        Image + superpixel borders with non-default segments tinted
        """
        display = self.image.copy()
        painted_px = (self.segment_class != 0)[self.labels]  # Pixels that belong to a non-default annotated superpixel 
        tint = self.render_mask()  # Get class color for every pixel in the mask
        display[painted_px] = ((1 - alpha) * display[painted_px] + alpha * tint[painted_px]).astype(np.uint8)  # Tint the pixels that should be tinted based on their annotations
        if boundaries:
            # If boundaries are being displayed, tint the pixels in the display where the boundaries lie as the boundary color
            display[self._boundary_mask] = boundary_color
        return display


    def _colors(self) -> np.ndarray:
        """
        (n_classes, 3) uint8 array of class colors, where row i is the color of class i.
        """
        return np.array([c.color for c in self.classes], dtype=np.uint8)


    def _patch(self, sid: int) -> None:
        """
        Redraw one superpixel within the cached display, in place
        Only rebuilds the bounding box of the changed superpixel
        """
        display = self._display
        if display is None: 
            return  # Nothing cached yet, nothing to update
        boundary_color, alpha, boundaries = self._display_key  # Fetch necessary info for updating pixels
        box = self._slices[sid]  # Bounding box of the superpixel
        inside = self.labels[box] == sid  # Which pixels in the bounding box belong to this superpixel
        region = display[box]  # Region of the cached display containing this superpixel
        pixels = self.image[box][inside]  # Pixels in the original image that are part of the update
        class_index = self.segment_class[sid]  # Fetch class of this superpixel
        if class_index != 0:
            # Same blend here as in _build_display()
            pixels = ((1 - alpha) * pixels + alpha * self._colors()[class_index]).astype(np.uint8)
        region[inside] = pixels
        if boundaries:
            region[inside & self._boundary_mask[box]] = boundary_color  # Re-draw boundaries since we just overwrote them

    # === PERSISTENCE ==================================================================================================

    def _load_existing(self) -> None:
        """
        Reconstruct superpixel assignments from existing mask on disk
        Masks are stored on disk as colors, not arrays of class ids or class ids per superpixel, so this basically reverses what `render_mask` does
        Look at each superpixel and color it based on the pixel at that position in the mask, mapping the color of the mask back to a class ID
        Note that this relies on deterministic segmentation
        """
        existing = load_mask(self.mask_path)
        if existing is None:
            return  # No mask yet, don't do anything
        if existing.shape[:2] != self.image.shape[:2]:
            # We cant use a mask that doesn't match the size of the image, return with a still-empty session and inform the user
            mask_h, mask_w = existing.shape[:2]
            image_h, image_w = self.image.shape[:2]
            self.load_warning = (f"Existing mask {self.mask_path.name} is {mask_w}x{mask_h} "
                                 f"but the image is {image_w}x{image_h}, so it was not loaded.")  # Shape mismatch err
            return
        # Flatten the labels and mask
        flat_labels = self.labels.ravel()
        flat_mask = existing.reshape(-1, 3)  # One RGB row per pixel

        # Get one sample pixel's color per superpixel
        uniq, first_idx = np.unique(flat_labels, return_index=True)
        rep_colors = flat_mask[first_idx]

        # Reverse lookup for class
        color_to_class: dict[tuple[int, ...], int] = {tuple(c.color): i for i, c in enumerate(self.classes)}

        # Assign classes to segments using above info
        for sid, color in zip(uniq, rep_colors):
            self.segment_class[sid] = color_to_class.get(tuple(int(v) for v in color), 0)  # Fall back to 0 if we get some non-class color


    def save(self) -> None:
        """
        Write the current mask to disk
        """
        save_mask(self.render_mask(), self.mask_path)
        self.dirty = False  # Mask is no longer dirty after being saved
