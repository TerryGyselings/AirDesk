"""Chargement de config/gestures.yaml dans des dataclasses typées."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "gestures.yaml"


@dataclass(frozen=True)
class EngagementConfig:
    hold_ms: float = 300
    lost_timeout_ms: float = 1000


@dataclass(frozen=True)
class PinchConfig:
    enter: float = 0.25
    exit: float = 0.40
    click_max_ms: float = 300


@dataclass(frozen=True)
class PointerConfig:
    pixels_per_frame_width: float = 2400
    landmark: int = 9


@dataclass(frozen=True)
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 8.0
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class Config:
    engagement: EngagementConfig
    pinch: PinchConfig
    pointer: PointerConfig
    filter: FilterConfig


def load_config(path: Path | None = None) -> Config:
    """Lit le fichier YAML. Les clés absentes gardent leur valeur par défaut."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    raw: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    return Config(
        engagement=EngagementConfig(**raw.get("engagement", {})),
        pinch=PinchConfig(**raw.get("pinch", {})),
        pointer=PointerConfig(**raw.get("pointer", {})),
        filter=FilterConfig(**raw.get("filter", {})),
    )
