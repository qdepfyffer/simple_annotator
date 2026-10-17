"""
Segmentation algorithm registry
To add an algorithm, write a function that maps a float RGB image + params to a 2-D integer label array, list the
tunable params, and register() the segmenter
Segmenters must behave deterministically for mask loading to work
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from skimage.segmentation import slic as _slic
from skimage.util import img_as_float

type ParamValues = dict[str, float]


# Frozen makes instances immutable
@dataclass(frozen=True)
class Param:
    """
    A tunable parameter of a segmentation algorithm
    (Schema only)
    """
    key: str
    label: str
    type: type
    default: float
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Segmenter:
    """
    A named segmentation algorithm and schema for parameters
    """
    key: str
    label: str
    params: tuple[Param, ...]
    func: Callable[..., np.ndarray]

    def defaults(self) -> ParamValues:
        return {p.key: p.default for p in self.params}


REGISTRY: dict[str, Segmenter] = {}

def register(segmenter: Segmenter) -> None:
    """
    Add a segmenter to REGISTRY (replaces any entry with the same key)
    """
    REGISTRY[segmenter.key] = segmenter


def run_segmenter(key: str, image: np.ndarray, params: ParamValues) -> np.ndarray:
    """
    Run the named segmenter on an RGB image, returning a 2-D label array
    """
    segmenter = REGISTRY[key]  # If you get a key error here double check your key
    return segmenter.func(img_as_float(image), **params)


# === SLIC =============================================================================================================

def _run_slic(image: np.ndarray, n_segments: float, sigma: float, compactness: float) -> np.ndarray:
    """
    Wrapper for scikit-image slic with this app's conventions
    """
    return _slic(image, n_segments = int(n_segments), sigma = sigma, compactness = compactness, start_label=0)

register(Segmenter(key="slic",
                   label="SLIC",
                   params=(
                       # Param (key, label, type, default, minimum, maximum)
                       Param("n_segments", "Segments", int, 800, minimum=1),
                       Param("sigma", "Sigma", float, 1.5, minimum=0.0),
                       Param("compactness", "Compactness", float, 50.0, minimum=0.0)
                   ),
                   func=_run_slic
                   ))

