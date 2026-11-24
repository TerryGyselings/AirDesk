"""Dessine le squelette des mains, la zone d'interaction et l'état du système."""

from __future__ import annotations

import cv2
import numpy as np

from airdesk.config import ZoneConfig
from airdesk.interpretation.features import HandFeatures
from airdesk.interpretation.state_machine import State
from airdesk.perception.hand_tracker import Hand

# Os de la main : paires de points à relier.
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # majeur
    (9, 13), (13, 14), (14, 15), (15, 16),   # annulaire
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # auriculaire + paume
]

# Couleurs BGR par état.
STATE_COLORS = {
    State.IDLE: (160, 160, 160),
    State.ACTIVE: (80, 200, 80),
    State.PINCHING: (0, 200, 255),
    State.DRAGGING: (255, 120, 0),
}
IGNORED_COLOR = (110, 110, 110)  # main visible sans rôle
SUPPORT_COLOR = (170, 130, 90)  # main d'appoint (bleu-gris), sans effet pour l'instant
ZONE_COLOR = (255, 255, 0)


def draw_hand(image: np.ndarray, hand: Hand, color: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    pts = []
    for lm in hand.landmarks:
        pts.append((int(lm[0] * w), int(lm[1] * h)))
    for a, b in BONES:
        cv2.line(image, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(image, p, 4, (255, 255, 255), -1)


def draw_debug(
    image: np.ndarray,
    hands: list[Hand],
    selected: int | None,
    features: HandFeatures | None,
    state: State,
    fps: float,
    processing_ms: float,
    zone: ZoneConfig | None = None,
    support: int | None = None,
) -> np.ndarray:
    """`hands` : toutes les mains détectées ; `selected` : indice de la main pilote ;
    `support` : indice de la main d'appoint (sans effet pour l'instant).

    `zone` : zone d'interaction, dessinée pour savoir où placer la main.
    """
    h, w = image.shape[:2]
    color = STATE_COLORS[state]

    if zone is not None:
        top_left = (int(zone.left * w), int(zone.top * h))
        bottom_right = (int(zone.right * w), int(zone.bottom * h))
        cv2.rectangle(image, top_left, bottom_right, ZONE_COLOR, 1)

    for i in range(len(hands)):
        if i == selected:
            draw_hand(image, hands[i], color)
        elif i == support:
            draw_hand(image, hands[i], SUPPORT_COLOR)
        else:
            draw_hand(image, hands[i], IGNORED_COLOR)

    lines = [
        f"Etat : {state.value}",
        f"{fps:.0f} img/s   traitement {processing_ms:.0f} ms",
    ]
    if features is not None:
        lines.append(
            f"pincement {features.pinch_ratio:.2f}   paume {features.palm_width:.2f}"
            f"   doigts replies {features.folded_fingers}"
        )
    else:
        lines.append("aucune main suivie")
    if len(hands) > 1:
        lines.append(f"{len(hands)} mains : seule la main pilote agit")
    if state == State.IDLE:
        lines.append("Main pilote ouverte dans le cadre pour activer")

    y = 30
    for text in lines:
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30
    return image
