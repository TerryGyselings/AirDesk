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

# Mesures de pincement possibles (valeur comparée à enter / exit) :
#   ratio : distance bout du pouce -> bout de l'index / taille de main (pinch_ratio)
#   seg78 : distance bout du pouce -> segment 7-8 de l'index (pinch_seg)
#   seg68 : distance bout du pouce -> segment 6-8 de l'index (pinch_seg_68)
MEASURES = ["ratio", "seg78", "seg68"]

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
    pinch_value: float,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
) -> bool:
    """Vrai si la règle `rule` considère que la main entre en pincement.

    `pinch_value` : la mesure de pincement choisie (pinch_ratio par défaut, voir MEASURES),
    comparée à `thresholds.enter`.
    `extensions` et `extensions_3d` : index, majeur, annulaire, auriculaire.
    """
    if rule not in RULE_CONDITIONS:
        raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
    needs_index, needs_other_finger, use_3d = RULE_CONDITIONS[rule]

    if not pinch_value < thresholds.enter:
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

    if needs_other_finger and not other_finger_open(values, folded_below):
        return False

    return True


def other_finger_open(extensions: tuple[float, ...], folded_below: float) -> bool:
    """Vrai si au moins un doigt parmi majeur, annulaire, auriculaire n'est pas replié.

    Un poing a tous ses doigts repliés ; un vrai pincement en laisse au moins un ouvert.
    """
    for i in range(1, 4):
        if extensions[i] >= folded_below:
            return True
    return False


def fingers_folded_for_rule(
    rule: str,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
) -> bool:
    """Vrai si la règle exige un autre doigt ouvert (C, D...) et qu'aucun ne l'est plus."""
    _, needs_other_finger, use_3d = RULE_CONDITIONS[rule]
    if not needs_other_finger:
        return False
    if use_3d:
        return not other_finger_open(extensions_3d, thresholds.folded_below_3d)
    return not other_finger_open(extensions, thresholds.folded_below)


def pinch_measure(features, measure: str) -> float:
    """Valeur de la mesure de pincement choisie, pour des descripteurs HandFeatures."""
    if measure == "ratio":
        return features.pinch_ratio
    if measure == "seg78":
        return features.pinch_seg
    if measure == "seg68":
        return features.pinch_seg_68
    raise ValueError(f"Mesure de pincement inconnue : '{measure}'. "
                     f"Choix possibles : {', '.join(MEASURES)}.")


class PinchHysteresis:
    """Entrée et sortie du pincement, image par image.

    - Entrée : la règle dit « pincement » (mesure < enter, plus les conditions de la règle).
    - Sortie : mesure >= exit (hystérésis : exit > enter), ou, si `exit_on_fold`, plus aucun
      autre doigt ouvert pour une règle qui l'exige (C, D...) : un poing entré par erreur ne
      reste alors pas « pincé » indéfiniment.
    - La condition de sortie doit tenir pendant `release_grace_ms` d'affilée : une seule
      image « relâchée » (le suivi qui décroche un instant) ne fait pas lâcher un glisser.
      Avec 0, une image suffit (comportement d'origine). Contrepartie : le relâchement
      volontaire est reconnu `release_grace_ms` plus tard (latence).

    La « mesure » est pinch_ratio, pinch_seg ou pinch_seg_68 (voir MEASURES) : c'est à
    l'appelant de passer la bonne valeur à `update`, et de régler enter / exit pour elle.

    `update` renvoie "start", "end" ou None. Après "end", `release_ms` donne l'instant où le
    relâchement a commencé (la vraie fin du pincement).
    Les images sans main ne passent pas ici : c'est à l'appelant de décider (la machine à
    états ne relâche pas sur une image sans main, seulement après lost_timeout_ms).
    """

    def __init__(
        self,
        rule: str,
        thresholds: RuleThresholds,
        exit: float,
        release_grace_ms: float = 0,
        exit_on_fold: bool = False,
    ):
        if rule not in RULE_CONDITIONS:
            raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
        self.rule = rule
        self.thresholds = thresholds
        self.exit = exit
        self.release_grace_ms = release_grace_ms
        self.exit_on_fold = exit_on_fold
        self.reset()

    def reset(self) -> None:
        self.pinched = False
        self.release_ms: float | None = None  # début du relâchement en cours, sinon None

    def update(
        self,
        t_ms: float,
        pinch_value: float,
        extensions: tuple[float, ...],
        extensions_3d: tuple[float, ...],
    ) -> str | None:
        if not self.pinched:
            if pinch_candidate(self.rule, pinch_value, extensions, extensions_3d,
                               self.thresholds):
                self.pinched = True
                self.release_ms = None
                return "start"
            return None

        releasing = pinch_value >= self.exit
        if self.exit_on_fold and fingers_folded_for_rule(
            self.rule, extensions, extensions_3d, self.thresholds
        ):
            releasing = True
        if not releasing:
            self.release_ms = None  # fausse alerte : on pince toujours
            return None
        if self.release_ms is None:
            self.release_ms = t_ms
        if t_ms - self.release_ms >= self.release_grace_ms:
            self.pinched = False
            return "end"
        return None
