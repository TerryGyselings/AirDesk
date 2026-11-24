"""Tests du pointage absolu : zone de l'image -> écran, puis fractions d'écran -> pixels."""

import pytest

from airdesk.actuator.base import Actuator
from airdesk.actuator.screen import ScreenRect, bounding_rect
from airdesk.config import ZoneConfig
from airdesk.interpretation.pointing import zone_to_screen
from airdesk.interpretation.state_machine import MoveTo

ZONE = ZoneConfig(left=0.15, right=0.85, top=0.15, bottom=0.75)


def test_zone_center_is_screen_center():
    u, v = zone_to_screen(0.50, 0.45, ZONE)
    assert u == pytest.approx(0.5)
    assert v == pytest.approx(0.5)


def test_zone_corners_are_screen_corners():
    assert zone_to_screen(0.15, 0.15, ZONE) == (0.0, 0.0)
    u, v = zone_to_screen(0.85, 0.75, ZONE)
    assert u == pytest.approx(1.0)
    assert v == pytest.approx(1.0)


def test_outside_the_zone_sticks_to_the_edges():
    assert zone_to_screen(0.02, 0.50, ZONE)[0] == 0.0  # trop à gauche
    assert zone_to_screen(0.99, 0.50, ZONE)[0] == 1.0  # trop à droite
    assert zone_to_screen(0.50, 0.01, ZONE)[1] == 0.0  # trop haut
    assert zone_to_screen(0.50, 0.95, ZONE)[1] == 1.0  # trop bas


def test_off_center_zone():
    zone = ZoneConfig(left=0.5, right=1.0, top=0.0, bottom=0.5)
    u, v = zone_to_screen(0.75, 0.25, zone)
    assert u == pytest.approx(0.5)
    assert v == pytest.approx(0.5)


def test_empty_zone_is_rejected_in_french():
    with pytest.raises(ValueError, match="Zone d'interaction vide"):
        zone_to_screen(0.5, 0.5, ZoneConfig(left=0.6, right=0.4))


# --- Écran et pixels ---------------------------------------------------------------------------

SCREEN = ScreenRect(0, 0, 2560, 1440)


def test_screen_fractions_to_pixels_include_both_edges():
    assert SCREEN.to_pixels(0.0, 0.0) == (0, 0)
    assert SCREEN.to_pixels(1.0, 1.0) == (2559, 1439)  # dernier pixel, pas en dehors
    assert SCREEN.to_pixels(0.5, 0.5) == (1280, 720)


def test_screen_offset_is_applied():
    second = ScreenRect(2560, -806, 1440, 2560)
    assert second.to_pixels(0.0, 0.0) == (2560, -806)
    assert second.to_pixels(1.0, 1.0) == (2560 + 1439, -806 + 2559)


def test_bounding_rect_of_two_screens_with_a_negative_offset():
    """La configuration réelle de P01 : principal 2560x1440, second vertical à droite, décalé
    vers le haut. Le rectangle englobant contient des zones sans écran (limite documentée)."""
    rect = bounding_rect([SCREEN, ScreenRect(2560, -806, 1440, 2560)])
    assert rect == ScreenRect(0, -806, 4000, 2560)


class RecordingActuator(Actuator):
    def __init__(self, screen):
        super().__init__(2400, screen)
        self.positions = []

    def move_to(self, x_px, y_px):
        self.positions.append((x_px, y_px))


def test_actuator_converts_move_to_into_pixels():
    actuator = RecordingActuator(SCREEN)
    actuator.apply(MoveTo(1.0, 0.0))
    actuator.apply(MoveTo(0.5, 0.5))
    assert actuator.positions == [(2559, 0), (1280, 720)]


def test_move_to_without_screen_is_a_clear_error():
    actuator = RecordingActuator(None)
    with pytest.raises(RuntimeError, match="Pointage absolu sans écran"):
        actuator.apply(MoveTo(0.5, 0.5))
