"""Pointage absolu : projeter la zone d'interaction de l'image sur l'écran.

En pointage relatif (comme un pavé tactile), le curseur et la main finissent par se
désynchroniser : chaque petite erreur s'accumule. En pointage absolu, une position de la
main correspond toujours à la même position du curseur.

On ne projette pas toute l'image : ses bords sont difficiles à atteindre (la main sort du
champ et n'est plus détectée). Une zone centrale, plus petite, est étirée sur l'écran ; hors
zone, le curseur reste collé au bord.

Logique pure : on travaille en fractions de l'écran (0 à 1). La conversion en pixels, qui
dépend de l'écran, est faite par l'actionneur (actuator/screen.py).
"""

from __future__ import annotations

from airdesk.config import ZoneConfig


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def zone_to_screen(x: float, y: float, zone: ZoneConfig) -> tuple[float, float]:
    """Position dans l'image (0 à 1) -> position sur l'écran (0 à 1), collée aux bords.

    Exemple avec la zone par défaut (15 %..85 % en x) : x = 0,15 -> 0 (bord gauche),
    x = 0,50 -> 0,5 (milieu), x = 0,85 -> 1 (bord droit), x = 0,95 -> 1 (collé au bord).
    """
    width = zone.right - zone.left
    height = zone.bottom - zone.top
    if width <= 0 or height <= 0:
        raise ValueError("Zone d'interaction vide : il faut left < right et top < bottom.")
    u = (x - zone.left) / width
    v = (y - zone.top) / height
    return clamp(u), clamp(v)


def in_zone(x: float, y: float, zone: ZoneConfig) -> bool:
    """Vrai si le point est dans la zone d'interaction (bords compris, sans bornage)."""
    return zone.left <= x <= zone.right and zone.top <= y <= zone.bottom


def below_zone(y: float, zone: ZoneConfig) -> bool:
    """Vrai si le point est sous le bord bas de la zone (main baissée)."""
    return y > zone.bottom

