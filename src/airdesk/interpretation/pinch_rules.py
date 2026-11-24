"""Règles candidates pour décider qu'une main « pince », et logique d'entrée / sortie.

Problème : poing fermé = pouce posé sur l'index replié, donc pinch_ratio petit, donc un faux
pincement (« Midas touch », Jacob 1991). Les règles ajoutent des conditions sur l'extension
des doigts (voir features.py) :

    A    : pinch_ratio < enter                                         (règle actuelle)
    B    : A  et  index tendu (extension > min_index_extension)
    C    : B  et  au moins un doigt parmi majeur/annulaire/auriculaire non replié
    D    : A  et  au moins un doigt parmi majeur/annulaire/auriculaire non replié
           (pas de condition sur l'index, qui se plie pour pincer : voir le journal)
    B3D, C3D, D3D : mêmes règles avec les extensions 3D

PAS ENCORE utilisé par la machine à états : ces règles sont comparées sur des mesures
(scripts/analyse_gestures.py) avant de décider laquelle adopter.
"""

from __future__ import annotations

from dataclasses import dataclass

RULES = ["A", "B", "C", "D", "B3D", "C3D", "D3D"]

# Pour chaque règle : (index tendu exigé ?, autre doigt non replié exigé ?, en 3D ?)
RULE_CONDITIONS = {
    "A": (False, False, False),
    "B": (True, False, False),
    "C": (True, True, False),
    "D": (False, True, False),
    "B3D": (True, False, True),
    "C3D": (True, True, True),
    "D3D": (False, True, True),
}


@dataclass(frozen=True)
class RuleThresholds:
    enter: float = 0.25
    min_index_extension: float = 1.0
    folded_below: float = 0.8
    min_index_extension_3d: float = 1.0
    folded_below_3d: float = 0.8


def pinch_candidate(
    rule: str,
    pinch_ratio: float,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
) -> bool:
    """Vrai si la règle `rule` considère que la main entre en pincement.

    `extensions` et `extensions_3d` : index, majeur, annulaire, auriculaire.
    """
    if rule not in RULE_CONDITIONS:
        raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
    needs_index, needs_other_finger, use_3d = RULE_CONDITIONS[rule]

    if not pinch_ratio < thresholds.enter:
        return False

    if use_3d:
        values = extensions_3d
        min_index = thresholds.min_index_extension_3d
        folded_below = thresholds.folded_below_3d
    else:
        values = extensions
        min_index = thresholds.min_index_extension
        folded_below = thresholds.folded_below

    if needs_index and not values[0] > min_index:
        return False

    if needs_other_finger:
        # Un poing a tous ses doigts repliés ; un vrai pincement en laisse au moins un.
        other_finger_open = False
        for i in range(1, 4):
            if values[i] >= folded_below:
                other_finger_open = True
        if not other_finger_open:
            return False

    return True


class PinchHysteresis:
    """Entrée et sortie du pincement, image par image.

    - Entrée : la règle dit « pincement ».
    - Sortie : pinch_ratio >= exit pendant au moins `release_grace_ms` d'affilée.
      Une seule image « relâchée » au milieu d'un pincement (le suivi qui décroche un instant)
      ne fait donc plus lâcher un glisser. Avec release_grace_ms = 0, une image suffit :
      c'est le comportement actuel de la machine à états. Contrepartie : le relâchement
      volontaire est reconnu `release_grace_ms` plus tard (latence).

    `update` renvoie "start", "end" ou None. Après "end", `release_ms` donne l'instant où le
    relâchement a commencé (la vraie fin du pincement).
    Les images sans main ne passent pas ici : c'est à l'appelant de décider (voir
    gesture_analysis.simulate_episodes, qui imite la machine à états).
    """

    def __init__(
        self, rule: str, thresholds: RuleThresholds, exit: float, release_grace_ms: float = 0
    ):
        self.rule = rule
        self.thresholds = thresholds
        self.exit = exit
        self.release_grace_ms = release_grace_ms
        self.reset()

    def reset(self) -> None:
        self.pinched = False
        self.release_ms: float | None = None  # début du relâchement en cours, sinon None

    def update(
        self,
        t_ms: float,
        pinch_ratio: float,
        extensions: tuple[float, ...],
        extensions_3d: tuple[float, ...],
    ) -> str | None:
        if not self.pinched:
            if pinch_candidate(self.rule, pinch_ratio, extensions, extensions_3d,
                               self.thresholds):
                self.pinched = True
                self.release_ms = None
                return "start"
            return None

        if pinch_ratio < self.exit:
            self.release_ms = None  # fausse alerte : on pince toujours
            return None
        if self.release_ms is None:
            self.release_ms = t_ms
        if t_ms - self.release_ms >= self.release_grace_ms:
            self.pinched = False
            return "end"
        return None
