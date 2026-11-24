"""Tests du pointage absolu : zone de l'image -> écran, puis fractions d'écran -> pixels."""

import pytest

from airdesk.actuator.base import Actuator
from airdesk.actuator.screen import ScreenRect, bounding_rect
from airdesk.config import ZoneConfig
from airdesk.interpretation.pointing import (
    TARGETS,
    StillnessDetector,
    below_zone,
    center_offset,
    in_zone,
    target_position,
    zone_from_corners,
    zone_to_screen,
)
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


def test_in_zone_and_below_zone():
    assert in_zone(0.5, 0.5, ZONE)
    assert in_zone(0.15, 0.75, ZONE)  # bords compris
    assert not in_zone(0.10, 0.5, ZONE)
    assert below_zone(0.80, ZONE)
    assert not below_zone(0.75, ZONE)
    assert not below_zone(0.05, ZONE)  # au-dessus n'est pas « baissée »


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


# --- Étalonnage guidé ---------------------------------------------------------------------------

STILL = 0.1  # vitesse du poignet d'une main immobile (tailles de main / s)
MOVING = 1.0


def feed_still(detector, start_ms, duration_ms, point=(0.5, 0.5), speed=STILL):
    """Images toutes les 33 ms ; renvoie (dernier t, dernière progression, point capturé)."""
    t = start_ms
    progress = 0.0
    captured = None
    while t < start_ms + duration_ms:
        progress, captured = detector.update(t, speed, point)
        if captured is not None:
            break
        t += 33
    return t, progress, captured


def test_targets_order_and_positions():
    assert TARGETS == ["centre", "haut_gauche", "haut_droite", "bas_droite", "bas_gauche"]
    assert target_position("centre") == (0.5, 0.5)
    assert target_position("haut_gauche") == (0.03, 0.03)
    assert target_position("bas_droite") == (0.97, 0.97)
    with pytest.raises(ValueError, match="Cible inconnue"):
        target_position("milieu")


def test_still_hand_is_captured_after_one_second():
    detector = StillnessDetector(max_speed=0.5, hold_ms=1000)
    t, progress, captured = feed_still(detector, 0, 2000, point=(0.4, 0.6))
    assert captured == (0.4, 0.6)
    assert 1000 <= t < 1040  # capture dès la première image après 1 s


def test_progress_grows_then_restarts_when_the_hand_moves():
    detector = StillnessDetector(max_speed=0.5, hold_ms=1000)
    progress, _ = detector.update(0, STILL, (0.5, 0.5))
    assert progress == 0.0
    progress, _ = detector.update(500, STILL, (0.5, 0.5))
    assert progress == pytest.approx(0.5)
    progress, _ = detector.update(533, MOVING, (0.55, 0.5))  # la main bouge : on repart de zéro
    assert progress == 0.0
    detector.update(566, STILL, (0.55, 0.5))
    progress, captured = detector.update(1500, STILL, (0.55, 0.5))
    assert captured is None  # seulement 934 ms depuis le redémarrage
    assert progress == pytest.approx(0.934)


def test_missing_hand_or_unknown_speed_restarts():
    detector = StillnessDetector(max_speed=0.5, hold_ms=1000)
    detector.update(0, STILL, (0.5, 0.5))
    assert detector.update(500, STILL, None) == (0.0, None)  # main pilote absente
    detector.update(533, STILL, (0.5, 0.5))
    assert detector.update(600, None, (0.5, 0.5)) == (0.0, None)  # vitesse inconnue


def test_captured_point_is_the_median():
    """Un point aberrant isolé (main mal détectée une image) ne déplace pas la mesure."""
    detector = StillnessDetector(max_speed=0.5, hold_ms=1000)
    captured = None
    t = 0
    i = 0
    while captured is None:
        point = (0.9, 0.1) if i == 10 else (0.5 + (i % 3) * 0.001, 0.6)
        _, captured = detector.update(t, STILL, point)
        t += 33
        i += 1
    assert captured[0] == pytest.approx(0.501)
    assert captured[1] == pytest.approx(0.6)


def test_after_a_capture_the_hand_must_move_before_the_next_one():
    """Sinon la cible suivante serait capturée au même endroit, la main étant encore immobile."""
    detector = StillnessDetector(max_speed=0.5, hold_ms=1000)
    t, _, first = feed_still(detector, 0, 2000)
    assert first is not None
    assert detector.waiting_for_movement
    t, progress, second = feed_still(detector, t + 33, 3000)  # toujours immobile : rien
    assert second is None
    assert progress == 0.0
    detector.update(t + 33, MOVING, (0.3, 0.3))  # la main part vers la cible suivante
    assert not detector.waiting_for_movement
    _, _, second = feed_still(detector, t + 66, 2000, point=(0.2, 0.2))
    assert second == (0.2, 0.2)


def test_reset_cancels_the_movement_requirement():
    """Retour arrière : on reprend la cible précédente sans avoir à bouger d'abord."""
    detector = StillnessDetector(max_speed=0.5, hold_ms=1000)
    feed_still(detector, 0, 2000)
    detector.reset()
    assert not detector.waiting_for_movement


CORNERS = {
    "haut_gauche": (0.30, 0.40),
    "haut_droite": (0.70, 0.42),
    "bas_droite": (0.72, 0.80),
    "bas_gauche": (0.28, 0.78),
}


def test_zone_from_corners_averages_the_edges_and_stretches_by_three_percent():
    """Bords mesurés : gauche (0.30 + 0.28)/2 = 0.29, droite 0.71, haut 0.41, bas 0.79.
    Étirés de 3 % : chaque coin mesuré doit tomber sur sa cible, à 3 % du bord de l'écran."""
    zone = zone_from_corners(CORNERS, target_margin=0.03)
    width = 0.42 / 0.94
    assert zone.left == pytest.approx(0.29 - 0.03 * width)
    assert zone.right == pytest.approx(0.71 + 0.03 * width)
    u, v = zone_to_screen(0.29, 0.41, zone)
    assert (u, v) == (pytest.approx(0.03), pytest.approx(0.03))
    u, v = zone_to_screen(0.71, 0.79, zone)
    assert (u, v) == (pytest.approx(0.97), pytest.approx(0.97))


def test_corners_out_of_order_are_refused():
    upside_down = dict(CORNERS)
    upside_down["haut_gauche"] = (0.30, 0.85)  # plus bas que bas_gauche
    with pytest.raises(ValueError, match="plus bas"):
        zone_from_corners(upside_down)
    mirrored = dict(CORNERS)
    mirrored["haut_gauche"] = (0.75, 0.40)  # plus à droite que haut_droite
    with pytest.raises(ValueError, match="plus à droite"):
        zone_from_corners(mirrored)


def test_too_small_zone_is_refused():
    small = {
        "haut_gauche": (0.45, 0.40),
        "haut_droite": (0.55, 0.40),
        "bas_droite": (0.55, 0.80),
        "bas_gauche": (0.45, 0.80),
    }
    with pytest.raises(ValueError, match="Zone trop petite"):
        zone_from_corners(small, min_size=0.15)


def test_stretched_zone_is_clamped_to_the_image():
    wide = {
        "haut_gauche": (0.01, 0.01),
        "haut_droite": (0.99, 0.01),
        "bas_droite": (0.99, 0.99),
        "bas_gauche": (0.01, 0.99),
    }
    zone = zone_from_corners(wide)
    assert (zone.left, zone.top, zone.right, zone.bottom) == (0.0, 0.0, 1.0, 1.0)


def test_missing_corner_is_refused():
    incomplete = dict(CORNERS)
    del incomplete["bas_gauche"]
    with pytest.raises(ValueError, match="Coin manquant"):
        zone_from_corners(incomplete)


def test_center_offset():
    zone = zone_from_corners(CORNERS)
    assert center_offset(zone, (0.50, 0.60)) < 0.01  # centre au milieu des coins
    assert center_offset(zone, (0.60, 0.60)) > 0.15  # centre décalé vers la droite
