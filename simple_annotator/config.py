"""
Persistent user settings
Stores selected segmenter and per-segmenter parameters as JSON in your platform's user-config directory
"""


from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from platformdirs import user_config_dir
from . import segmentation

APP_NAME = "simple_annotator"
CONFIG_PATH = Path(user_config_dir(APP_NAME, appauthor=False)) / "config.json"


def _default_params() -> dict[str, dict[str, float]]:
    """Fresh default parameters for all registered segmentation algorithms"""
    return {key: seg.defaults() for key, seg in segmentation.REGISTRY.items()}


@dataclass
class Settings:
    segmenter: str = "slic"
    params: dict[str, dict[str, float]] = field(default_factory=_default_params)

    def segmenter_params(self) -> dict[str, float]:
        """Parameters for the current segmenter"""
        return self.params[self.segmenter]


def load() -> Settings:
    """Load settings and merge saved values over defaults"""
    settings = Settings()  # Seeded with defaults from registry
    if not CONFIG_PATH.exists():
        return settings

    try:
        data = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return settings  # Fall back to defaults on failure

    if data.get("segmenter") in segmentation.REGISTRY:
        settings.segmenter = data["segmenter"]

    saved = data.get("params", {})
    for seg_key, seg_params in settings.params.items():
        for param_key in seg_params:
            if param_key in saved.get(seg_key, {}):
                seg_params[param_key] = saved[seg_key][param_key]

    return settings


def save(settings: Settings) -> None:
    """Save settings to disk and create config directory if necessary"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "segmenter": settings.segmenter,
                "params": settings.params
            },
            indent=2
        )
    )