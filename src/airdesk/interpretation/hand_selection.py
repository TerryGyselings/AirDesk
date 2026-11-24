"""Choix de la main suivie quand plusieurs mains sont visibles.

Deux temps, comme dans un dialogue : on choisit d'abord QUI parle (la main qui fait le geste
d'activation), puis on écoute cette main-là, et pas l'autre.

- En veille : on retient une main qui fait le geste d'activation (main ouverte), en
  préférant le côté configuré (hand.preferred). Sans main ouverte : la main préférée, sinon
  la plus grande (la plus proche de la caméra).
- Une fois actif (verrou) : à chaque image, la main dont le poignet est le plus proche de sa
  position précédente, en tailles de main. Si même la plus proche a sauté de plus de
  `max_jump` tailles de main, la main suivie est considérée comme absente : on ne bascule pas
  sur l'autre main quand la main suivie sort du champ. Le verrou saute au retour en veille.

Logique pure : reçoit des descriptions de mains, renvoie un indice. Testable sans webcam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HandCandidate:
    wrist: tuple[float, float]  # poignet en unités homogènes (x multiplié par l'aspect)
    hand_size: float  # taille de main dans les mêmes unités
    side: str  # "right" / "left" (côté de l'utilisateur)
    open_palm: bool  # fait le geste d'activation


class HandSelector:
    def __init__(self, preferred: str = "right", max_jump: float = 1.5):
        if preferred not in ("right", "left", "any"):
            raise ValueError(f"hand.preferred inconnu : '{preferred}'. Choix : right, left, any.")
        self.preferred = preferred
        self.max_jump = max_jump
        self._last_wrist: tuple[float, float] | None = None

    def select(self, candidates: list[HandCandidate], idle: bool) -> int | None:
        """Indice de la main suivie dans `candidates`, ou None (aucune main retenue).

        `idle` : la machine à états est en veille. Hors veille, la main est verrouillée : on
        la suit par continuité depuis la dernière position retenue. Si elle reste perdue
        plus de lost_timeout_ms, la machine repasse en veille, ce qui libère le verrou.
        """
        if idle:
            choice = self._choose_for_activation(candidates)
        else:
            choice = self._follow(candidates)
        if choice is not None:
            self._last_wrist = candidates[choice].wrist
        return choice

    def _prefers(self, candidate: HandCandidate) -> bool:
        return self.preferred == "any" or candidate.side == self.preferred

    def _largest(self, indexes: list[int], candidates: list[HandCandidate]) -> int:
        best = indexes[0]
        for i in indexes[1:]:
            if candidates[i].hand_size > candidates[best].hand_size:
                best = i
        return best

    def _choose_for_activation(self, candidates: list[HandCandidate]) -> int | None:
        if not candidates:
            return None
        # Par ordre de priorité : main ouverte du bon côté, main ouverte, bon côté, n'importe.
        groups = [[], [], [], []]
        for i in range(len(candidates)):
            c = candidates[i]
            if c.open_palm and self._prefers(c):
                groups[0].append(i)
            elif c.open_palm:
                groups[1].append(i)
            elif self._prefers(c):
                groups[2].append(i)
            else:
                groups[3].append(i)
        for group in groups:
            if group:
                return self._largest(group, candidates)
        return None

    def _follow(self, candidates: list[HandCandidate]) -> int | None:
        if not candidates:
            return None
        if self._last_wrist is None:
            return self._choose_for_activation(candidates)
        best = None
        best_jump = math.inf
        for i in range(len(candidates)):
            c = candidates[i]
            size = c.hand_size if c.hand_size > 1e-6 else 1e-6
            jump = math.dist(c.wrist, self._last_wrist) / size
            if jump < best_jump:
                best = i
                best_jump = jump
        if best_jump > self.max_jump:
            return None  # la main suivie n'est plus là : on ignore les autres
        return best
