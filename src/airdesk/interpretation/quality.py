"""Garde-fou de qualité : écarter les images où la main est mal vue.

Quand la main se met de profil ou sort en partie du champ, la distance poignet -> base du
majeur (`hand_size`) s'effondre dans l'image. Comme tous les descripteurs sont divisés par
hand_size, ils explosent alors (pinch_ratio > 1 en plein pincement) et peuvent faire lâcher
un glisser. On compare donc chaque hand_size à la médiane des dernières secondes : une chute
brutale signale une image peu fiable, que l'interprétation traite comme « main absente ».

PAS ENCORE utilisé par la machine à états (voir scripts/analyse_gestures.py).
"""

from __future__ import annotations

import statistics
from collections import deque

MIN_HISTORY = 5  # en dessous, pas assez de recul : on accepte l'image


class HandSizeGuard:
    """Accepte ou rejette chaque image selon le rapport hand_size / médiane récente.

    `min_ratio` : rapport minimal accepté (0 = garde-fou désactivé).
    `window_ms` : durée de l'historique servant à la médiane.

    Toutes les images entrent dans l'historique, même rejetées : si la main s'éloigne
    réellement de la caméra, la médiane suit en une demi-fenêtre environ, et l'on ne rejette
    pas indéfiniment une main simplement plus petite.
    """

    def __init__(self, min_ratio: float, window_ms: float = 2000):
        self.min_ratio = min_ratio
        self.window_ms = window_ms
        self._history: deque[tuple[float, float]] = deque()  # (t_ms, hand_size)

    def reset(self) -> None:
        self._history.clear()

    def ratio(self, t_ms: float, hand_size: float) -> float | None:
        """hand_size / médiane de l'historique récent (None si historique trop court).

        Met aussi l'historique à jour : à appeler une seule fois par image.
        """
        while self._history and t_ms - self._history[0][0] > self.window_ms:
            self._history.popleft()
        value = None
        if len(self._history) >= MIN_HISTORY:
            sizes = []
            for _, size in self._history:
                sizes.append(size)
            median = statistics.median(sizes)
            if median > 0:
                value = hand_size / median
        self._history.append((t_ms, hand_size))
        return value

    def accept(self, t_ms: float, hand_size: float) -> bool:
        """Vrai si l'image est fiable. À appeler une seule fois par image."""
        value = self.ratio(t_ms, hand_size)
        if self.min_ratio <= 0 or value is None:
            return True
        return value >= self.min_ratio
