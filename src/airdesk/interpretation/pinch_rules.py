"""Règles candidates pour décider qu'une main « pince » (entrée dans le pincement).

Problème : poing fermé = pouce posé sur l'index replié, donc pinch_ratio petit, donc un faux
pincement (« Midas touch », Jacob 1991). Les règles ajoutent des conditions sur l'extension
des doigts (voir features.py) :

    A    : pinch_ratio < enter                                      (règle actuelle)
    B    : A  et  index tendu (extension 2D > min_index_extension)
    C    : B  et  au moins un doigt parmi majeur/annulaire/auriculaire non replié
    B3D, C3D : mêmes règles avec les extensions 3D (robustes au raccourcissement de
               perspective quand l'index pointe vers la caméra)

PAS ENCORE utilisé par la machine à états : ces règles sont comparées sur des mesures
(scripts/analyse_gestures.py) avant de décider laquelle adopter. La sortie du pincement,
elle, ne change pas (pinch_ratio >= exit, hystérésis).
"""

from __future__ import annotations

from dataclasses import dataclass

RULES = ["A", "B", "C", "B3D", "C3D"]


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
    if rule not in RULES:
        raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")

    if not pinch_ratio < thresholds.enter:
        return False
    if rule == "A":
        return True

    if rule in ("B", "C"):
        values = extensions
        min_index = thresholds.min_index_extension
        folded_below = thresholds.folded_below
    else:
        values = extensions_3d
        min_index = thresholds.min_index_extension_3d
        folded_below = thresholds.folded_below_3d

    if not values[0] > min_index:
        return False
    if rule in ("B", "B3D"):
        return True

    # Règle C : un poing a tous ses doigts repliés ; un vrai pincement en laisse au moins un.
    for i in range(1, 4):
        if values[i] >= folded_below:
            return True
    return False
