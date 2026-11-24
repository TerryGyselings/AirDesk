"""Pilotage de la vraie souris avec pynput (Windows, macOS, Linux X11).

- macOS : autoriser le terminal (ou l'éditeur) dans
  Réglages Système > Confidentialité et sécurité > Accessibilité.
- Linux Wayland : l'injection d'événements est bloquée par le système ; lancer une session X11.
"""

from __future__ import annotations

from pynput.mouse import Button, Controller

from airdesk.actuator.base import Actuator


class MouseActuator(Actuator):
    def __init__(self, pixels_per_frame_width: float):
        super().__init__(pixels_per_frame_width)
        self._mouse = Controller()

    def move_by(self, dx_px: int, dy_px: int) -> None:
        self._mouse.move(dx_px, dy_px)

    def click(self) -> None:
        self._mouse.click(Button.left, 1)

    def press(self) -> None:
        self._mouse.press(Button.left)

    def release(self) -> None:
        self._mouse.release(Button.left)
