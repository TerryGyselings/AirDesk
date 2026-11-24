"""Petites fonctions statistiques partagées (interprétation et analyses)."""

from __future__ import annotations

import math


def percentile(values: list[float], p: float) -> float:
    """Percentile par la méthode du « rang le plus proche ».

    On trie les valeurs et on prend celle de rang ceil(p/100 × n) (rangs comptés à partir de 1).
    Le 95e percentile est donc une valeur réellement mesurée, telle que 95 % des mesures
    lui sont inférieures ou égales. Plus parlant que la moyenne pour la latence : quelques
    images lentes suffisent à rendre le curseur saccadé, même si la moyenne est bonne.
    """
    if not values:
        raise ValueError("Aucune valeur : percentile impossible.")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    if rank < 1:
        rank = 1
    return ordered[rank - 1]
