"""Tests de l'actionneur : clics épinglés (PressAt, ReleaseAt), sans toucher à la vraie souris."""

from airdesk.actuator.base import Actuator
from airdesk.actuator.screen import ScreenRect
from airdesk.interpretation.state_machine import MoveTo, PressAt, Release, ReleaseAt


class RecordingActuator(Actuator):
    """Faux actionneur : enregistre les appels au lieu d'agir sur le système."""

    def __init__(self):
        super().__init__(1000, ScreenRect(0, 0, 1001, 501))  # pixels ronds : 0,5 -> 500
        self.calls = []

    def move_to(self, x_px, y_px):
        self.calls.append(("move_to", x_px, y_px))

    def press(self):
        self.calls.append(("press",))

    def release(self):
        self.calls.append(("release",))


def test_press_at_moves_to_the_pinned_point_then_presses():
    a = RecordingActuator()
    a.apply(MoveTo(0.52, 0.5))  # position réelle de la main
    a.apply(PressAt(0.5, 0.5))  # point du contact
    assert a.calls == [("move_to", 520, 250), ("move_to", 500, 250), ("press",)]


def test_no_move_back_while_the_button_is_down():
    """Pas de retour vers la main juste après le Press : ce serait un début de glisser."""
    a = RecordingActuator()
    a.apply(PressAt(0.5, 0.5))
    a.apply(ReleaseAt(0.5, 0.5))
    assert a.calls == [("move_to", 500, 250), ("press",), ("move_to", 500, 250), ("release",)]


def test_plain_release_does_not_move():
    a = RecordingActuator()
    a.apply(Release())
    assert a.calls == [("release",)]
