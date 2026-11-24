"""Couche 4 — Actionneur : transforme les événements en actions sur le système."""

from airdesk.actuator.base import Actuator, DryRunActuator

__all__ = ["Actuator", "DryRunActuator", "create_actuator"]


def create_actuator(dry_run: bool, pixels_per_frame_width: float) -> Actuator:
    """Choisit l'actionneur.

    pynput est importé ici seulement, pour que les tests n'en dépendent pas.
    """
    if dry_run:
        return DryRunActuator(pixels_per_frame_width)
    from airdesk.actuator.mouse import MouseActuator

    return MouseActuator(pixels_per_frame_width)
