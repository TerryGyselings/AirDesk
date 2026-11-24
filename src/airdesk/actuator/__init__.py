"""Couche 4 — Actionneur : transforme les événements en actions sur le système."""

from airdesk.actuator.base import Actuator, DryRunActuator
from airdesk.actuator.screen import ScreenRect, detect_screen

__all__ = ["Actuator", "DryRunActuator", "ScreenRect", "create_actuator", "detect_screen"]


def create_actuator(
    dry_run: bool, pixels_per_frame_width: float, screen: ScreenRect | None = None
) -> Actuator:
    """Choisit l'actionneur.

    pynput est importé ici seulement, pour que les tests n'en dépendent pas.
    `screen` : écran couvert en pointage absolu (voir detect_screen).
    """
    if dry_run:
        return DryRunActuator(pixels_per_frame_width, screen)
    from airdesk.actuator.mouse import MouseActuator

    return MouseActuator(pixels_per_frame_width, screen)
