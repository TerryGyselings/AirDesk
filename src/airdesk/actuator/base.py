"""Interface commune des actionneurs."""

from __future__ import annotations

from airdesk.interpretation.state_machine import Click, Event, Move, Press, Release


class Actuator:
    """Classe de base : convertit les déplacements normalisés en pixels, puis délègue.

    Pointage relatif (comme un pavé tactile) : un déplacement de la main sur toute la largeur
    de l'image déplace le curseur de `pixels_per_frame_width` pixels.
    """

    def __init__(self, pixels_per_frame_width: float):
        self.gain = pixels_per_frame_width
        self._rest_x = 0.0  # fractions de pixel accumulées, pour ne pas les perdre
        self._rest_y = 0.0

    def apply(self, event: Event) -> None:
        if isinstance(event, Move):
            fx = event.dx * self.gain + self._rest_x
            fy = event.dy * self.gain + self._rest_y
            px = int(fx)
            py = int(fy)
            self._rest_x = fx - px
            self._rest_y = fy - py
            if px != 0 or py != 0:
                self.move_by(px, py)
        elif isinstance(event, Click):
            self.click()
        elif isinstance(event, Press):
            self.press()
        elif isinstance(event, Release):
            self.release()

    # À implémenter par chaque actionneur concret.
    def move_by(self, dx_px: int, dy_px: int) -> None:
        raise NotImplementedError

    def click(self) -> None:
        raise NotImplementedError

    def press(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class DryRunActuator(Actuator):
    """N'agit pas sur le système : affiche seulement les actions. Pour tester sans risque."""

    def move_by(self, dx_px: int, dy_px: int) -> None:
        pass  # trop verbeux à afficher à 30 images/s

    def click(self) -> None:
        print("[simulation] clic")

    def press(self) -> None:
        print("[simulation] bouton enfoncé (début du glisser)")

    def release(self) -> None:
        print("[simulation] bouton relâché (fin du glisser)")
