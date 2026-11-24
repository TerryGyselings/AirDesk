"""Chargement des réglages dans des dataclasses typées.

Trois niveaux, du plus faible au plus fort :

1. les valeurs par défaut des dataclasses ci-dessous ;
2. config/gestures.yaml (versionné, commun à toute l'équipe), puis config/local.yaml
   (non versionné, propre à une machine : quelle caméra, quel backend...) ;
3. les options de la ligne de commande (`apply_cli_overrides`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "gestures.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"


@dataclass(frozen=True)
class EngagementConfig:
    hold_ms: float = 300
    lost_timeout_ms: float = 1000


@dataclass(frozen=True)
class PinchConfig:
    enter: float = 0.25
    exit: float = 0.40
    click_max_ms: float = 300
    # Règle anti-poing (voir interpretation/pinch_rules.py). PAS ENCORE utilisée par la
    # machine à états : décision après mesures. "A" = règle actuelle.
    rule: str = "A"
    min_index_extension: float = 1.0
    min_index_extension_3d: float = 1.0
    # Relâchement reconnu seulement si pinch_ratio >= exit pendant ce temps (0 = immédiat,
    # comportement actuel). PAS ENCORE utilisé par la machine à états.
    release_grace_ms: float = 0


@dataclass(frozen=True)
class QualityConfig:
    """Garde-fou de taille de main (interpretation/quality.py). PAS ENCORE branché."""

    min_hand_size_ratio: float = 0.0  # 0 = désactivé
    hand_size_window_ms: float = 2000


@dataclass(frozen=True)
class FingersConfig:
    """Seuils d'extension (bout -> centre de paume / taille de main) d'un doigt replié."""

    folded_below: float = 0.8
    folded_below_3d: float = 0.8


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
class CameraConfig:
    """Caméra à utiliser. Dépend de la machine : se règle dans config/local.yaml."""

    index: int = 0
    backend: str = "auto"  # "auto" = selon le système, sinon "dshow", "msmf"...
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class Config:
    engagement: EngagementConfig
    pinch: PinchConfig
    pointer: PointerConfig
    filter: FilterConfig
    camera: CameraConfig
    fingers: FingersConfig
    quality: QualityConfig


def read_yaml(path: Path) -> dict:
    """Lit un fichier YAML ; renvoie {} s'il n'existe pas ou s'il est vide."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_settings(base: dict, override: dict) -> dict:
    """Fusionne deux réglages lus en YAML, section par section puis clé par clé.

    Une clé présente dans `override` remplace celle de `base` ; les autres sont conservées.
    Exemple : base {"pinch": {"enter": 0.25, "exit": 0.4}} et override {"pinch": {"enter": 0.2}}
    donnent {"pinch": {"enter": 0.2, "exit": 0.4}}. Les dictionnaires reçus ne sont pas modifiés.
    """
    merged: dict = {}
    for section, values in base.items():
        merged[section] = dict(values or {})
    for section, values in override.items():
        if section not in merged:
            merged[section] = {}
        for key, value in (values or {}).items():
            merged[section][key] = value
    return merged


def load_config(path: Path | None = None, local_path: Path | None = LOCAL_CONFIG_PATH) -> Config:
    """Lit gestures.yaml (ou `path`), puis le surcharge avec local.yaml s'il existe.

    `local_path=None` ignore local.yaml (utile dans les tests).
    Les clés absentes gardent leur valeur par défaut.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    raw = read_yaml(path)
    if local_path is not None:
        raw = merge_settings(raw, read_yaml(local_path))

    return Config(
        engagement=EngagementConfig(**raw.get("engagement", {})),
        pinch=PinchConfig(**raw.get("pinch", {})),
        pointer=PointerConfig(**raw.get("pointer", {})),
        filter=FilterConfig(**raw.get("filter", {})),
        camera=CameraConfig(**raw.get("camera", {})),
        fingers=FingersConfig(**raw.get("fingers", {})),
        quality=QualityConfig(**raw.get("quality", {})),
    )


def apply_cli_overrides(
    camera: CameraConfig,
    index: int | None = None,
    backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> CameraConfig:
    """Applique les options de la ligne de commande, qui priment sur les fichiers.

    Une option non donnée vaut None et laisse la valeur des fichiers intacte.
    `replace` crée une copie modifiée (les dataclasses sont figées, `frozen=True`).
    """
    changes: dict = {}
    if index is not None:
        changes["index"] = index
    if backend is not None:
        changes["backend"] = backend
    if width is not None:
        changes["width"] = width
    if height is not None:
        changes["height"] = height
    return replace(camera, **changes)


def save_camera_settings(
    path: Path, index: int, backend: str, width: int, height: int
) -> None:
    """Écrit (ou remplace) la section `camera:` de local.yaml, sans toucher aux autres sections.

    Attention : le fichier est réécrit par le module yaml, donc les commentaires d'un local.yaml
    existant sont perdus.
    """
    raw = read_yaml(path)
    raw["camera"] = {"index": index, "backend": backend, "width": width, "height": height}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Réglages propres à cette machine (non versionné). Voir local.example.yaml.\n")
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
