"""Machine à états d'engagement.

Elle répond au problème du « Midas touch » (Jacob, 1991) : la main bouge tout le temps,
donc rien ne doit se déclencher tant que l'utilisateur ne l'a pas clairement demandé.

    VEILLE ──(main ouverte maintenue hold_ms)──▶ ACTIF
    ACTIF ──(pincement)──▶ PINCEMENT
    PINCEMENT ──(relâché avant click_max_ms)──▶ ACTIF      + Click
    PINCEMENT ──(maintenu après click_max_ms)──▶ GLISSER    + Press
    GLISSER ──(relâché)──▶ ACTIF                            + Release
    n'importe quel état ──(main perdue > lost_timeout_ms)──▶ VEILLE

Cette classe ne connaît ni la caméra, ni la souris : elle reçoit des mesures et renvoie
des événements. C'est ce qui permet de la tester sans webcam (voir tests/).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from airdesk.config import EngagementConfig, PinchConfig
from airdesk.interpretation.features import HandFeatures


class State(Enum):
    IDLE = "veille"
    ACTIVE = "actif"
    PINCHING = "pincement"
    DRAGGING = "glisser"


@dataclass(frozen=True)
class Move:
    """Déplacement relatif du curseur, en unités normalisées de l'image."""

    dx: float
    dy: float


@dataclass(frozen=True)
class Click:
    pass


@dataclass(frozen=True)
class Press:
    pass


@dataclass(frozen=True)
class Release:
    pass


Event = Move | Click | Press | Release


class GestureStateMachine:
    def __init__(self, engagement: EngagementConfig, pinch: PinchConfig):
        self.engagement = engagement
        self.pinch = pinch
        self.state = State.IDLE
        self._arm_start_ms: float | None = None
        self._pinch_start_ms: float = 0.0
        self._last_seen_ms: float | None = None
        self._prev_pointer: tuple[float, float] | None = None

    def update(
        self,
        t_ms: float,
        features: HandFeatures | None,
        pointer: tuple[float, float] | None,
    ) -> list[Event]:
        """Fait avancer la machine d'une image. `features` et `pointer` valent None sans main."""
        events: list[Event] = []

        # --- Pas de main visible ---
        if features is None or pointer is None:
            self._arm_start_ms = None
            self._prev_pointer = None
            lost_for = 0.0
            if self._last_seen_ms is not None:
                lost_for = t_ms - self._last_seen_ms
            if self.state != State.IDLE and lost_for > self.engagement.lost_timeout_ms:
                if self.state == State.DRAGGING:
                    events.append(Release())
                self.state = State.IDLE
            return events

        self._last_seen_ms = t_ms

        # --- Veille : on attend une main ouverte maintenue ---
        if self.state == State.IDLE:
            if features.open_palm:
                if self._arm_start_ms is None:
                    self._arm_start_ms = t_ms
                elif t_ms - self._arm_start_ms >= self.engagement.hold_ms:
                    self.state = State.ACTIVE
                    self._arm_start_ms = None
                    self._prev_pointer = pointer  # pas de saut du curseur à l'activation
            else:
                self._arm_start_ms = None
            return events

        # --- Déplacement du curseur (sauf pendant un pincement court, pour un clic précis) ---
        if self._prev_pointer is not None and self.state != State.PINCHING:
            dx = pointer[0] - self._prev_pointer[0]
            dy = pointer[1] - self._prev_pointer[1]
            if dx != 0 or dy != 0:
                events.append(Move(dx, dy))
        self._prev_pointer = pointer

        # --- Pincement avec hystérésis ---
        currently_pinched = self.state in (State.PINCHING, State.DRAGGING)
        if currently_pinched:
            pinched = features.pinch_ratio < self.pinch.exit
        else:
            pinched = features.pinch_ratio < self.pinch.enter

        if self.state == State.ACTIVE:
            if pinched:
                self.state = State.PINCHING
                self._pinch_start_ms = t_ms

        elif self.state == State.PINCHING:
            if not pinched:
                events.append(Click())
                self.state = State.ACTIVE
            elif t_ms - self._pinch_start_ms > self.pinch.click_max_ms:
                events.append(Press())
                self.state = State.DRAGGING

        elif self.state == State.DRAGGING:
            if not pinched:
                events.append(Release())
                self.state = State.ACTIVE

        return events
