"""Descripteurs calculés à partir des 21 points de la main.

Toutes les distances sont divisées par la « taille de la main » (poignet -> base du majeur),
pour que les seuils restent valables que l'on soit à 50 cm ou à 2 m de la caméra.

On compare des distances entre points plutôt que des coordonnées y : un test du type
« bout du doigt plus haut que l'articulation » échoue dès que la main est penchée.

Unités. MediaPipe donne x divisé par la LARGEUR de l'image, y divisé par sa HAUTEUR, et z
(profondeur relative au poignet) à peu près à la même échelle que x. En 640x480, une même
longueur vaut donc moins en x qu'en y : une main tournée de 90° semblerait changer de taille.
Les extensions de doigts sont calculées en unités homogènes : X = x·aspect, Y = y,
Z = z·aspect, avec aspect = largeur / hauteur. (`pinch_ratio` et `extended_fingers` gardent
leur calcul d'origine pour ne pas changer le comportement actuel de la machine à états.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

# (articulation PIP, bout) pour index, majeur, annulaire, auriculaire
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]

# Points qui délimitent la paume : poignet et base des quatre doigts.
PALM_POINTS = [0, 5, 9, 13, 17]
# Bouts des doigts, dans l'ordre index, majeur, annulaire, auriculaire.
FINGER_TIPS = [8, 12, 16, 20]

# Seuil d'extension sous lequel un doigt est compté comme replié (valeur provisoire, réglée
# dans config/gestures.yaml après mesures : voir scripts/analyse_gestures.py).
DEFAULT_FOLDED_BELOW = 0.8

Point = tuple[float, float, float]


@dataclass
class HandFeatures:
    hand_size: float  # distance poignet -> base du majeur (unités normalisées de l'image)
    pinch_ratio: float  # distance pouce-index / taille de main (petit = pincement)
    extended_fingers: int  # nombre de doigts tendus parmi index, majeur, annulaire, auriculaire
    open_palm: bool  # les quatre doigts tendus
    # Extension = distance bout du doigt -> centre de la paume / taille de main.
    # Ordre : index, majeur, annulaire, auriculaire. Tendu ≈ 1,5 ; replié ≈ 0,4.
    finger_extensions: tuple[float, float, float, float]  # dans le plan de l'image (2D)
    index_extension: float  # = finger_extensions[0]
    folded_fingers: int  # nombre d'extensions 2D sous le seuil `folded_below`
    # Même chose avec la profondeur z : un index pointé vers la caméra paraît court en 2D
    # (raccourcissement de perspective) mais garde sa longueur en 3D.
    finger_extensions_3d: tuple[float, float, float, float]
    index_extension_3d: float
    folded_fingers_3d: int


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_3d(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def is_finger_extended(landmarks: list[Point], pip: int, tip: int) -> bool:
    """Un doigt est tendu si son bout est nettement plus loin du poignet que son articulation."""
    wrist = landmarks[WRIST]
    return distance_2d(landmarks[tip], wrist) > 1.15 * distance_2d(landmarks[pip], wrist)


def to_homogeneous(landmarks: list[Point], aspect: float) -> list[Point]:
    """Convertit en unités homogènes (même échelle sur les trois axes) : voir l'en-tête."""
    points = []
    for x, y, z in landmarks:
        points.append((x * aspect, y, z * aspect))
    return points


def palm_center(landmarks: list[Point]) -> Point:
    """Centre de la paume : moyenne du poignet et de la base des quatre doigts."""
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for i in PALM_POINTS:
        x, y, z = landmarks[i]
        sx += x
        sy += y
        sz += z
    n = len(PALM_POINTS)
    return (sx / n, sy / n, sz / n)


def finger_extensions(
    landmarks: list[Point], use_depth: bool = False
) -> tuple[float, float, float, float]:
    """Extension de chaque doigt : distance bout -> centre de la paume / taille de la main.

    Seules des distances entre points interviennent : le résultat ne dépend ni de la position
    de la main dans l'image, ni de sa rotation, ni de sa distance à la caméra.
    `landmarks` doit être en unités homogènes (voir `to_homogeneous`).
    Avec `use_depth`, z est pris en compte (distances 3D, taille de main 3D comprise).
    """
    if use_depth:
        distance = distance_3d
    else:
        distance = distance_2d
    hand_size = distance(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if hand_size < 1e-6:
        hand_size = 1e-6
    center = palm_center(landmarks)
    values = []
    for tip in FINGER_TIPS:
        values.append(distance(landmarks[tip], center) / hand_size)
    return (values[0], values[1], values[2], values[3])


def count_folded(extensions: tuple[float, ...], folded_below: float) -> int:
    """Nombre de doigts dont l'extension est sous le seuil (doigts repliés)."""
    count = 0
    for value in extensions:
        if value < folded_below:
            count += 1
    return count


def normalize_landmarks(landmarks: list[Point], aspect: float = 1.0) -> list[Point]:
    """Points relatifs au poignet et divisés par la taille de la main, en unités homogènes.

    Après normalisation : poignet en (0, 0, 0), base du majeur à une distance 2D de 1.
    C'est ce qu'on enregistre dans les jeux de données : la forme de la main, sans sa
    position ni sa taille dans l'image (et aucune image, RGPD).
    """
    points = to_homogeneous(landmarks, aspect)
    wx, wy, wz = points[WRIST]
    hand_size = distance_2d(points[WRIST], points[MIDDLE_MCP])
    if hand_size < 1e-6:
        hand_size = 1e-6
    normalized = []
    for x, y, z in points:
        normalized.append(((x - wx) / hand_size, (y - wy) / hand_size, (z - wz) / hand_size))
    return normalized


def compute_features(
    landmarks: list[Point],
    folded_below: float = DEFAULT_FOLDED_BELOW,
    folded_below_3d: float = DEFAULT_FOLDED_BELOW,
    aspect: float = 1.0,
) -> HandFeatures:
    """Calcule tous les descripteurs d'une main.

    `aspect` = largeur / hauteur de l'image (ex. 640/480), pour les extensions.
    `folded_below`, `folded_below_3d` : seuils d'extension d'un doigt replié (gestures.yaml).
    """
    hand_size = distance_2d(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if hand_size < 1e-6:
        hand_size = 1e-6

    pinch_ratio = distance_2d(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_size

    extended = 0
    for i in range(len(FINGERS)):
        pip, tip = FINGERS[i]
        if is_finger_extended(landmarks, pip, tip):
            extended += 1

    homogeneous = to_homogeneous(landmarks, aspect)
    extensions = finger_extensions(homogeneous, use_depth=False)
    extensions_3d = finger_extensions(homogeneous, use_depth=True)

    return HandFeatures(
        hand_size=hand_size,
        pinch_ratio=pinch_ratio,
        extended_fingers=extended,
        open_palm=extended == 4,
        finger_extensions=extensions,
        index_extension=extensions[0],
        folded_fingers=count_folded(extensions, folded_below),
        finger_extensions_3d=extensions_3d,
        index_extension_3d=extensions_3d[0],
        folded_fingers_3d=count_folded(extensions_3d, folded_below_3d),
    )
