"""Chargement des réglages dans des dataclasses typées.

Trois niveaux, du plus faible au plus fort :

1. les valeurs par défaut des dataclasses ci-dessous ;
2. config/gestures.yaml (versionné, commun à toute l'équipe), puis config/local.yaml
   (non versionné, propre à une machine : quelle caméra, quel backend...) ;
3. les options de la ligne de commande (`apply_cli_overrides`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "gestures.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"


@dataclass(frozen=True)
class EngagementConfig:
    hold_ms: float = 300
    lost_timeout_ms: float = 1000
    # Main baissée sous la zone d'interaction pendant rest_ms -> veille (0 = désactivé).
    rest_ms: float = 0
    # La main ouverte n'active que si le point de pointage est dans la zone d'interaction.
    activate_in_zone: bool = False


@dataclass(frozen=True)
class PinchConfig:
    enter: float = 0.25
    exit: float = 0.40
    click_max_ms: float = 300
    # Règle anti-poing (voir interpretation/pinch_rules.py). Valeurs par défaut du code =
    # comportement d'origine ; les valeurs en service sont dans config/gestures.yaml.
    rule: str = "A"
    min_index_extension: float = 1.0
    min_index_extension_3d: float = 1.0
    # Relâchement reconnu seulement si la condition de sortie tient pendant ce temps
    # (0 = immédiat).
    release_grace_ms: float = 0
    # Mesure comparée à enter / exit : "ratio" (pinch_ratio), "seg78", "seg68" (pinch_seg).
    measure: str = "ratio"
    # Sortir aussi du pincement quand les autres doigts se replient (règles C, D...).
    exit_on_fold: bool = False
    # Entrée en pincement seulement si la paume fait face à la caméra (palm_width).
    require_palm_facing: bool = False
    min_palm_width: float = 0.0
    # Un pincement ne peut commencer que si le point de pointage est dans la zone.
    only_in_zone: bool = False
    # Un relâchement en cours n'est annulé que si la mesure redescend sous exit - cette marge
    # (0 = comportement d'origine). Voir PinchHysteresis.
    release_cancel_margin: float = 0.0
    # Relâchement immédiat au-dessus de exit + cette marge (doigts franchement ouverts).
    # 0 = désactivé (toujours le délai release_grace_ms).
    release_fast_margin: float = 0.0
    # Vitesse du poignet maximale au contact (tailles de main / s). 0 = pas de condition.
    max_contact_speed: float = 0.0
    # Mode de clic : "release" (clic émis au relâchement, d'origine) ou "press" (comme une
    # souris : Press dès le contact confirmé, Release au relâchement ; le système distingue
    # clic et glisser). En mode press :
    click_mode: str = "release"
    min_press_frames: int = 2  # images de pincement d'affilée avant Press (anti-glitch)
    drag_threshold: float = 0.15  # tailles de main : au-delà, le curseur suit (glisser)
    drag_catchup_ms: float = 100  # rattrapage progressif du curseur au début du glisser
    # Double-clic aimanté (mode press, pointage absolu) : un 2e Press à moins de
    # double_click_ms du précédent et à moins de double_click_snap tailles de main de son point
    # de contact est émis au pixel du 1er clic. Dans gestures.yaml, double_click_ms "auto" =
    # délai de double-clic du système (lu par app.py). double_click_snap 0 = pas d'aimant.
    double_click_ms: float | str = 500
    double_click_snap: float = 0.0
    # Relâchement relatif : remontée de la mesure au-dessus de son minimum pendant l'appui,
    # pendant un appui sur place seulement. 0 = désactivé.
    release_rise: float = 0.0
    # Règles F / F3D : pas de condition de fermeture pour un contact moins de ce temps (ms)
    # après la fin du pincement précédent (2e tape d'un double-clic). 0 = pas d'exemption.
    closing_grace_ms: float = 0.0
    # L'exemption ne vaut que pour un vrai candidat 2e tape : main à moins de
    # double_click_snap du clic précédent ET vitesse du poignet sous ce seuil (tailles de
    # main / s). 0 = jamais d'exemption.
    closing_grace_max_speed: float = 0.0
    # Règles F / F3D (non branchées) : vitesse de fermeture minimale des doigts.
    min_closing_speed: float = 0.0
    min_closing_speed_3d: float = 0.0


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
class ZoneConfig:
    """Zone d'interaction : partie de l'image (fractions 0 à 1) projetée sur tout l'écran."""

    left: float = 0.15
    right: float = 0.85
    top: float = 0.20
    bottom: float = 0.80


@dataclass(frozen=True)
class PointerConfig:
    # "absolute" : la zone d'interaction de l'image correspond à l'écran entier.
    # "relative" : comme un pavé tactile, la main déplace le curseur (gain ci-dessous).
    mode: str = "absolute"
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    # Écran couvert en pointage absolu : "primary" (principal) ou "all" (rectangle
    # englobant de tous les écrans).
    screen: str = "primary"
    pixels_per_frame_width: float = 2400  # gain du mode relatif
    landmark: int = 9


@dataclass(frozen=True)
class FilterConfig:
    min_cutoff: float = 1.0
    beta: float = 8.0
    d_cutoff: float = 1.0


def default_absolute_filter() -> FilterConfig:
    """Coupure plus basse en absolu, où le bruit de la main est plus amplifié."""
    return FilterConfig(min_cutoff=0.5)


@dataclass(frozen=True)
class FilterModes:
    """Réglages du filtre 1€ pour chaque mode de pointage (le bruit n'y est pas amplifié
    pareil : voir config/gestures.yaml)."""

    absolute: FilterConfig = field(default_factory=default_absolute_filter)
    relative: FilterConfig = field(default_factory=FilterConfig)

    def for_mode(self, mode: str) -> FilterConfig:
        if mode == "absolute":
            return self.absolute
        return self.relative


@dataclass(frozen=True)
class HandConfig:
    """Choix de la main suivie quand deux mains sont visibles."""

    preferred: str = "right"  # main pilote : "right", "left" ou "any" (côté de l'utilisateur)
    max_jump: float = 1.5  # saut maximal du poignet entre deux images, en tailles de main
    # Suivre la main d'appoint une fois actif (futurs gestes à deux mains). Coûte ~10 ms par
    # image : MediaPipe doit alors chercher 2 mains à chaque image.
    track_support: bool = False
    # En veille, une main non pilote visible fait chercher 2 mains pendant ce temps (ms).
    two_hands_hold_ms: float = 1000
    # Une fois actif, on ne cherche 2 mains qu'après ce nombre d'images sans main pilote.
    search_after_lost_frames: int = 3


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
    filter: FilterModes
    camera: CameraConfig
    fingers: FingersConfig
    quality: QualityConfig
    hand: HandConfig


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

    # Sous-sections imbriquées : pointer.zone et filter.absolute / filter.relative.
    pointer_raw = dict(raw.get("pointer", {}))
    if "zone" in pointer_raw:
        pointer_raw["zone"] = ZoneConfig(**pointer_raw["zone"])

    return Config(
        engagement=EngagementConfig(**raw.get("engagement", {})),
        pinch=PinchConfig(**raw.get("pinch", {})),
        pointer=PointerConfig(**pointer_raw),
        filter=filter_modes(raw.get("filter", {})),
        camera=CameraConfig(**raw.get("camera", {})),
        fingers=FingersConfig(**raw.get("fingers", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        hand=HandConfig(**raw.get("hand", {})),
    )


def filter_modes(raw: dict) -> FilterModes:
    """Section `filter` : réglages par mode, ou ancien format à plat (appliqué aux deux)."""
    if "absolute" in raw or "relative" in raw:
        absolute = FilterModes().absolute
        relative = FilterModes().relative
        if "absolute" in raw:
            absolute = FilterConfig(**raw["absolute"])
        if "relative" in raw:
            relative = FilterConfig(**raw["relative"])
        return FilterModes(absolute=absolute, relative=relative)
    if raw:
        same = FilterConfig(**raw)
        return FilterModes(absolute=same, relative=same)
    return FilterModes()


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


def save_local_section(path: Path, section: str, values: dict) -> None:
    """Écrit des clés dans une section de local.yaml, sans toucher au reste.

    Les autres clés de la section et les autres sections sont conservées ; les clés données
    remplacent les anciennes (ex. section "pointer", values {"zone": {...}}).
    Attention : le fichier est réécrit par le module yaml, donc les commentaires d'un local.yaml
    existant sont perdus.
    """
    raw = read_yaml(path)
    current = dict(raw.get(section) or {})
    for key, value in values.items():
        current[key] = value
    raw[section] = current
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Réglages propres à cette machine (non versionné). Voir local.example.yaml.\n")
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)


def save_camera_settings(
    path: Path, index: int, backend: str, width: int, height: int
) -> None:
    """Écrit (ou remplace) la caméra dans local.yaml (`scripts/check_camera.py --save`)."""
    save_local_section(
        path, "camera", {"index": index, "backend": backend, "width": width, "height": height}
    )


def save_zone(path: Path, zone: ZoneConfig) -> None:
    """Écrit la zone d'interaction étalonnée dans local.yaml (étalonnage)."""
    values = {"left": round(zone.left, 3), "right": round(zone.right, 3),
              "top": round(zone.top, 3), "bottom": round(zone.bottom, 3)}
    save_local_section(path, "pointer", {"zone": values})
