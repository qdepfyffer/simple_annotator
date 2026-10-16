"""
Persistent user settings
Stores selected segmenter and per-segmenter parameters as JSON in your platform's user-config directory
Note that defaults are not hard coded here, and loading a settings file that's missing some settings will load without issue and fall back to defaults
"""


from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_config_dir

from . import segmentation

APP_NAME = "simple_annotator"
CONFIG_PATH = Path(user_config_dir(APP_NAME, appauthor=False)) / "config.json"


def _default_params() -> dict[str, segmentation.ParamValues]:
    """
    Fresh default parameters for all registered segmentation algorithms
    """
    return {key: seg.defaults() for key, seg in segmentation.REGISTRY.items()}

# Note to self, @dataclass is pretty cool - auto-generates __init__, __repr__, and __eq__ for you
@dataclass
class Settings:
    """
    Everything configurable by the user
    """

    # Key to `segmentation.REGISTRY` determining active segmenter
    segmenter: str = "slic"

    # Params for all the registered segmenters
    params: dict[str, segmentation.ParamValues] = field(default_factory=_default_params)

    # Color of superpixel border overlay
    boundary_color: str = "#ffff00"

    def segmenter_params(self) -> segmentation.ParamValues:
        """
        Parameters for the current segmenter
        """
        return self.params[self.segmenter]


def load() -> Settings:
    """
    Load settings and merge saved values over defaults
    """
    settings = Settings()  # Seeded with defaults from registry
    if not CONFIG_PATH.exists():
        return settings  # No point trying to load a config that doesn't exist

    try:
        data = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return settings  # Fall back to defaults on failure

    # Only accept the segmenter in settings if it's actually part of the registry
    if data.get("segmenter") in segmentation.REGISTRY:
        settings.segmenter = data["segmenter"]

    # Pull in any saved values that exist
    saved = data.get("params", {})
    for seg_key, seg_params in settings.params.items():
        for param_key in seg_params:
            if param_key in saved.get(seg_key, {}):
                seg_params[param_key] = saved[seg_key][param_key]

    # This is to make sure we get an actual hex color
    color = data.get("boundary_color")
    if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):  # I love regex
        settings.boundary_color = color

    return settings


def save(settings: Settings) -> None:
    """
    Save settings to disk and create config directory if necessary
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)  # Let it make required parent folders and don't worry if the directory already exists

    # This is how the settings files are structured and they're saved with indent=2 so they're actually readable
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "segmenter": settings.segmenter,
                "params": settings.params,
                "boundary_color": settings.boundary_color,
            },
            indent=2
        )
    )
