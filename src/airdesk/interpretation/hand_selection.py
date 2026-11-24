"""Rôle de chaque main visible : main pilote, main d'appoint.

Les deux mains n'ont pas le même rôle. Dans le modèle de la « chaîne cinématique » de
Guiard (1987), la main dominante agit avec précision (ici : pointer, cliquer) et la main non
dominante sert de référence ou d'appoint (ex. futur zoom à deux mains). D'où deux rôles :

- **main pilote** : seule une main du côté `hand.preferred` peut l'être. Elle seule active
  AirDesk et pilote le curseur ;
- **main d'appoint** : l'autre main, seulement pendant que la main pilote est active. Elle n'a
  aucun effet seule, et perd tout rôle dès que la main pilote est perdue. (Aucun geste à deux
  mains n'est encore implémenté : la structure est prête, l'appoint est ignoré.)

Pourquoi ? Constaté en usage réel : la main pilote sort du champ, AirDesk repasse en veille,
et l'autre main, ouverte au repos 300 ms, suffisait à reprendre le contrôle. Désormais l'autre
main ne devient JAMAIS pilote (sauf `preferred: any`, l'ancien comportement).

- En veille : main pilote = une main du bon côté, ouverte de préférence, sinon la plus grande.
- Une fois actif (verrou) : la main du bon côté dont le poignet est le plus proche de sa
  position précédente, en tailles de main ; au-delà de `max_jump`, la main pilote est absente.

Référence : Guiard, Y. (1987). Asymmetric division of labor in human skilled bimanual
action: The kinematic chain as a model. Journal of Motor Behavior, 19(4), 486-517.

Logique pure : reçoit des descriptions de mains, renvoie des indices. Testable sans webcam.
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


@dataclass(frozen=True)
class Selection:
    pilot: int | None  # indice de la main pilote, ou None (aucune main ne pilote)
    support: int | None  # indice de la main d'appoint (seulement main pilote active)


class HandSelector:
    def __init__(self, preferred: str = "right", max_jump: float = 1.5):
        if preferred not in ("right", "left", "any"):
            raise ValueError(f"hand.preferred inconnu : '{preferred}'. Choix : right, left, any.")
        self.preferred = preferred
        self.max_jump = max_jump
        self._last_wrist: tuple[float, float] | None = None

    def select(self, candidates: list[HandCandidate], idle: bool) -> Selection:
        """Rôles des mains visibles.

        `idle` : la machine à états est en veille. Hors veille, la main pilote est
        verrouillée : on la suit par continuité. Si elle reste perdue plus de
        lost_timeout_ms, la machine repasse en veille, ce qui libère le verrou.
        """
        if idle:
            pilot = self._choose_pilot(candidates)
            support = None  # en veille, la main d'appoint n'a pas de rôle
        else:
            pilot = self._follow(candidates)
            support = None
            if pilot is not None:
                support = self._choose_support(candidates, pilot)
        if pilot is not None:
            self._last_wrist = candidates[pilot].wrist
        return Selection(pilot, support)

    def can_pilot(self, candidate: HandCandidate) -> bool:
        return self.preferred == "any" or candidate.side == self.preferred

    def _largest(self, indexes: list[int], candidates: list[HandCandidate]) -> int:
        best = indexes[0]
        for i in indexes[1:]:
            if candidates[i].hand_size > candidates[best].hand_size:
                best = i
        return best

    def _choose_pilot(self, candidates: list[HandCandidate]) -> int | None:
        open_hands = []
        other_hands = []
        for i in range(len(candidates)):
            if not self.can_pilot(candidates[i]):
                continue  # une main de l'autre côté ne pilote jamais
            if candidates[i].open_palm:
                open_hands.append(i)
            else:
                other_hands.append(i)
        if open_hands:
            return self._largest(open_hands, candidates)
        if other_hands:
            return self._largest(other_hands, candidates)
        return None

    def _follow(self, candidates: list[HandCandidate]) -> int | None:
        if self._last_wrist is None:
            return self._choose_pilot(candidates)
        best = None
        best_jump = math.inf
        for i in range(len(candidates)):
            c = candidates[i]
            if not self.can_pilot(c):
                continue
            size = c.hand_size if c.hand_size > 1e-6 else 1e-6
            jump = math.dist(c.wrist, self._last_wrist) / size
            if jump < best_jump:
                best = i
                best_jump = jump
        if best is None or best_jump > self.max_jump:
            return None  # la main pilote n'est plus là : aucune autre main ne la remplace
        return best

    def _choose_support(self, candidates: list[HandCandidate], pilot: int) -> int | None:
        others = []
        for i in range(len(candidates)):
            if i != pilot:
                others.append(i)
        if not others:
            return None
        return self._largest(others, candidates)
