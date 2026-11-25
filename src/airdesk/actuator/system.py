"""Réglages du système d'exploitation utiles à AirDesk (dépendances système : ici seulement)."""

from __future__ import annotations

import sys

DEFAULT_DOUBLE_CLICK_MS = 500  # valeur par défaut de Windows


def system_double_click_ms() -> int | None:
    """Délai de double-clic du système, en ms (None si on ne sait pas le lire).

    Windows : GetDoubleClickTime (user32), réglable dans Paramètres > Souris. Deux clics plus
    espacés que ce délai ne forment pas un double-clic. Ailleurs : None (l'appelant utilise
    DEFAULT_DOUBLE_CLICK_MS).
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        value = ctypes.windll.user32.GetDoubleClickTime()
    except (AttributeError, OSError):
        return None
    if value <= 0:
        return None
    return int(value)


def resolve_double_click_ms(setting) -> float:
    """Valeur de `pinch.double_click_ms` : "auto" = délai du système (500 ms à défaut),
    sinon le nombre donné."""
    if setting == "auto":
        value = system_double_click_ms()
        if value is None:
            return DEFAULT_DOUBLE_CLICK_MS
        return value
    return float(setting)
