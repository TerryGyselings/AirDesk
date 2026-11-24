"""Dessine le squelette de la main et l'état du système sur l'image de la webcam."""

from __future__ import annotations

import cv2
import numpy as np

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


def draw_debug(
    image: np.ndarray,
    hand: Hand | None,
    features: HandFeatures | None,
    state: State,
    fps: float,
    processing_ms: float,
) -> np.ndarray:
    h, w = image.shape[:2]
    color = STATE_COLORS[state]

    if hand is not None:
        pts = []
        for lm in hand.landmarks:
            pts.append((int(lm[0] * w), int(lm[1] * h)))
        for a, b in BONES:
            cv2.line(image, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(image, p, 4, (255, 255, 255), -1)

    lines = [
        f"Etat : {state.value}",
        f"{fps:.0f} img/s   traitement {processing_ms:.0f} ms",
    ]
    if features is not None:
        lines.append(
            f"pincement {features.pinch_ratio:.2f}   doigts tendus {features.extended_fingers}"
        )
    else:
        lines.append("aucune main")
    if state == State.IDLE:
        lines.append("Montre ta main ouverte pour activer")

    y = 30
    for text in lines:
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 30
    return image
