"""Suivi de la main avec MediaPipe Hand Landmarker (API « Tasks »).

Points renvoyés par MediaPipe (21 par main), coordonnées normalisées entre 0 et 1 :

    0 poignet
    1-4   pouce   (4 = bout)
    5-8   index   (5 = base, 6 = articulation PIP, 8 = bout)
    9-12  majeur  (9 = base)
    13-16 annulaire
    17-20 auriculaire

Référence : Zhang et al., « MediaPipe Hands: On-device Real-time Hand Tracking », 2020.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "hand_landmarker.task"


def user_side(handedness: str) -> str:
    """Côté de la main pour l'UTILISATEUR ("right" / "left") à partir de l'étiquette MediaPipe.

    MediaPipe suppose une image en miroir (« selfie »), ce qu'on lui donne (voir Camera).
    Pourtant, mesuré le 2026-11-24 sur 2 sessions faites main DROITE, il répond « Left » sur
    97 % des images (2 330/2 332 et 2 260/2 365) : avec notre chaîne, l'étiquette est
    inversée. On la retourne donc ici, à un seul endroit (voir docs/journal.md).
    """
    if handedness == "Left":
        return "right"
    if handedness == "Right":
        return "left"
    return "unknown"


@dataclass
class Hand:
    """Une main détectée."""

    landmarks: list[tuple[float, float, float]]  # 21 points (x, y, z)
    handedness: str  # "Left" ou "Right", étiquette BRUTE de MediaPipe (inversée, voir user_side)
    score: float  # confiance de la classification gauche/droite

    @property
    def side(self) -> str:
        """Main de l'utilisateur : "right" ou "left" (étiquette corrigée)."""
        return user_side(self.handedness)


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Télécharge le modèle au premier lancement (environ 8 Mo), puis le réutilise."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Téléchargement du modèle MediaPipe vers {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Détecte au plus `max_hands` mains par image."""

    def __init__(self, model_path: Path | None = None, max_hands: int = 1):
        model = ensure_model(model_path or DEFAULT_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, rgb, timestamp_ms: int) -> list[Hand]:
        """Renvoie les mains présentes dans l'image RGB (liste vide si aucune)."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        hands: list[Hand] = []
        for i in range(len(result.hand_landmarks)):
            points = []
            for lm in result.hand_landmarks[i]:
                points.append((lm.x, lm.y, lm.z))
            category = result.handedness[i][0]
            hands.append(Hand(points, category.category_name, category.score))
        return hands
