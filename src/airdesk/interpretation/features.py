"""Descripteurs calculés à partir des 21 points de la main.

Toutes les distances sont divisées par la « taille de la main » (poignet -> base du majeur),
pour que les seuils restent valables que l'on soit à 50 cm ou à 2 m de la caméra.

On compare des distances au poignet plutôt que des coordonnées y : un test du type
« bout du doigt plus haut que l'articulation » échoue dès que la main est penchée.
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

Point = tuple[float, float, float]


@dataclass
class HandFeatures:
    hand_size: float  # distance poignet -> base du majeur (unités normalisées de l'image)
    pinch_ratio: float  # distance pouce-index / taille de main (petit = pincement)
    extended_fingers: int  # nombre de doigts tendus parmi index, majeur, annulaire, auriculaire
    open_palm: bool  # les quatre doigts tendus


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_finger_extended(landmarks: list[Point], pip: int, tip: int) -> bool:
    """Un doigt est tendu si son bout est nettement plus loin du poignet que son articulation."""
    wrist = landmarks[WRIST]
    return distance_2d(landmarks[tip], wrist) > 1.15 * distance_2d(landmarks[pip], wrist)


def compute_features(landmarks: list[Point]) -> HandFeatures:
    hand_size = distance_2d(landmarks[WRIST], landmarks[MIDDLE_MCP])
    if hand_size < 1e-6:
        hand_size = 1e-6

    pinch_ratio = distance_2d(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_size

    extended = 0
    for i in range(len(FINGERS)):
        pip, tip = FINGERS[i]
        if is_finger_extended(landmarks, pip, tip):
            extended += 1

    return HandFeatures(
        hand_size=hand_size,
        pinch_ratio=pinch_ratio,
        extended_fingers=extended,
        open_palm=extended == 4,
    )
