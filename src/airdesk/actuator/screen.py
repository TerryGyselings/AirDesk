"""Géométrie des écrans, pour le pointage absolu.

PyMonCtl (multiplateforme, installé avec PyWinCtl) donne la position et la taille de chaque
écran en pixels du bureau. PyWinCtl.getScreenSize() est déprécié au profit de PyMonCtl.

Limite : les coordonnées de pynput et de PyMonCtl coïncident quand l'échelle d'affichage de
Windows est à 100 %. Avec une autre échelle (125 %, 150 %...), un décalage est possible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenRect:
    """Rectangle d'écran en pixels du bureau (x, y = coin haut gauche)."""

    x: int
    y: int
    width: int
    height: int

    def to_pixels(self, u: float, v: float) -> tuple[int, int]:
        """Fractions de l'écran (0 à 1) -> pixels. 0 = premier pixel, 1 = dernier pixel."""
        px = self.x + round(u * (self.width - 1))
        py = self.y + round(v * (self.height - 1))
        return px, py


def bounding_rect(screens: list[ScreenRect]) -> ScreenRect:
    """Plus petit rectangle qui contient tous les écrans.

    Attention : si les écrans n'ont pas la même taille ou sont décalés, ce rectangle contient
    des zones sans écran, où le système bloque le curseur.
    """
    if not screens:
        raise ValueError("Aucun écran.")
    left = screens[0].x
    top = screens[0].y
    right = screens[0].x + screens[0].width
    bottom = screens[0].y + screens[0].height
    for s in screens[1:]:
        left = min(left, s.x)
        top = min(top, s.y)
        right = max(right, s.x + s.width)
        bottom = max(bottom, s.y + s.height)
    return ScreenRect(left, top, right - left, bottom - top)


def detect_screen(which: str = "primary") -> ScreenRect:
    """Écran couvert en pointage absolu : "primary" (principal) ou "all" (tous les écrans).

    PyMonCtl est importé ici seulement, pour que les tests n'en dépendent pas.
    """
    import pymonctl

    if which == "primary":
        monitor = pymonctl.getPrimary()
        return ScreenRect(monitor.position.x, monitor.position.y, monitor.size.width,
                          monitor.size.height)
    if which == "all":
        screens = []
        for monitor in pymonctl.getAllMonitors():
            screens.append(ScreenRect(monitor.position.x, monitor.position.y,
                                      monitor.size.width, monitor.size.height))
        return bounding_rect(screens)
    raise ValueError(f"pointer.screen inconnu : '{which}'. Choix possibles : primary, all.")
