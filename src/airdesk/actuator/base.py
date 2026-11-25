"""Interface commune des actionneurs."""

from __future__ import annotations

from airdesk.actuator.screen import ScreenRect
from airdesk.interpretation.state_machine import (
    Click,
    Event,
    Move,
    MoveTo,
    Press,
    PressAt,
    Release,
    ReleaseAt,
)


class Actuator:
    """Classe de base : convertit les événements en pixels, puis délègue.

    - Pointage relatif (Move, comme un pavé tactile) : un déplacement de la main sur toute la
      largeur de l'image déplace le curseur de `pixels_per_frame_width` pixels.
    - Pointage absolu (MoveTo) : une fraction de l'écran (0 à 1) devient une position en
      pixels de `screen`.
    - Clics épinglés (PressAt, ReleaseAt) : le curseur est placé au point épinglé, puis le
      bouton est enfoncé / relâché. Pas de retour à la main juste après le Press : bouton
      enfoncé, ce retour serait vu comme un début de glisser (au-delà de 4 px sous Windows).
      C'est la machine à états qui ramène ensuite le curseur vers la main (MoveTo, avec
      rattrapage) après le Release.
    """

    def __init__(self, pixels_per_frame_width: float, screen: ScreenRect | None = None):
        self.gain = pixels_per_frame_width
        self.screen = screen
        self._rest_x = 0.0  # fractions de pixel accumulées, pour ne pas les perdre
        self._rest_y = 0.0

    def to_pixels(self, u: float, v: float) -> tuple[int, int]:
        if self.screen is None:
            raise RuntimeError("Pointage absolu sans écran : passer `screen` à l'actionneur.")
        return self.screen.to_pixels(u, v)

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
        elif isinstance(event, MoveTo):
            x, y = self.to_pixels(event.u, event.v)
            self.move_to(x, y)
        elif isinstance(event, Click):
            self.click()
        elif isinstance(event, Press):
            self.press()
        elif isinstance(event, Release):
            self.release()
        elif isinstance(event, PressAt):
            x, y = self.to_pixels(event.u, event.v)
            self.press_at(x, y)
        elif isinstance(event, ReleaseAt):
            x, y = self.to_pixels(event.u, event.v)
            self.release_at(x, y)

    def press_at(self, x_px: int, y_px: int) -> None:
        """Bouton enfoncé au point épinglé : on y place le curseur, puis on appuie."""
        self.move_to(x_px, y_px)
        self.press()

    def release_at(self, x_px: int, y_px: int) -> None:
        """Bouton relâché au point épinglé (le curseur y est déjà s'il est resté figé)."""
        self.move_to(x_px, y_px)
        self.release()

    # À implémenter par chaque actionneur concret.
    def move_by(self, dx_px: int, dy_px: int) -> None:
        raise NotImplementedError

    def move_to(self, x_px: int, y_px: int) -> None:
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

    def move_to(self, x_px: int, y_px: int) -> None:
        pass

    def click(self) -> None:
        print("[simulation] clic")

    def press(self) -> None:
        print("[simulation] bouton enfoncé")

    def release(self) -> None:
        print("[simulation] bouton relâché")

    def press_at(self, x_px: int, y_px: int) -> None:
        print(f"[simulation] bouton enfoncé en ({x_px}, {y_px})")

    def release_at(self, x_px: int, y_px: int) -> None:
        print(f"[simulation] bouton relâché en ({x_px}, {y_px})")
