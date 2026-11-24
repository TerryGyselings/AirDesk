"""Machine à états d'engagement.

Elle répond au problème du « Midas touch » (Jacob, 1991) : la main bouge tout le temps,
donc rien ne doit se déclencher tant que l'utilisateur ne l'a pas clairement demandé.

    VEILLE ──(main ouverte maintenue hold_ms, dans la zone)──▶ ACTIF
    ACTIF ──(pincement, dans la zone)──▶ PINCEMENT
    PINCEMENT ──(relâché avant click_max_ms)──▶ ACTIF      + Click
    PINCEMENT ──(maintenu après click_max_ms)──▶ GLISSER    + Press
    GLISSER ──(relâché)──▶ ACTIF                            + Release
    n'importe quel état ──(main perdue > lost_timeout_ms)──▶ VEILLE
    n'importe quel état ──(main sous la zone > rest_ms)──▶ VEILLE  (+ Release si glisser)

« Main baissée = veille » : poser la main, c'est arrêter de piloter. Ces règles de zone sont
activées par gestures.yaml (activate_in_zone, only_in_zone, rest_ms).

Cette classe ne connaît ni la caméra, ni la souris : elle reçoit des mesures et renvoie
des événements. C'est ce qui permet de la tester sans webcam (voir tests/).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from airdesk.config import EngagementConfig, FingersConfig, PinchConfig, PointerConfig
from airdesk.interpretation.features import HandFeatures
from airdesk.interpretation.pinch_rules import PinchHysteresis, RuleThresholds, pinch_measure
from airdesk.interpretation.pointing import below_zone, in_zone, zone_to_screen


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
class MoveTo:
    """Position absolue du curseur, en fractions de l'écran (0 à 1) : pointage absolu."""

    u: float
    v: float


@dataclass(frozen=True)
class Click:
    pass


@dataclass(frozen=True)
class Press:
    pass


@dataclass(frozen=True)
class Release:
    pass


Event = Move | MoveTo | Click | Press | Release


class GestureStateMachine:
    """Machine à états d'engagement et de pincement.

    L'entrée / sortie du pincement est déléguée à PinchHysteresis (pinch_rules.py), réglée
    par config/gestures.yaml : règle (A = pinch_ratio seul, D = + un autre doigt ouvert...),
    mesure (pinch_ratio ou pinch_seg), enter / exit, délai de relâchement, sortie sur repli.
    Avec rule A, measure ratio et release_grace_ms 0, on retrouve le comportement d'origine.
    Choix de la configuration : voir scripts/analyse_gestures.py et docs/journal.md.

    Pointage : `pointer.mode` "absolute" émet MoveTo (zone de l'image -> écran entier),
    "relative" émet Move (déplacements). Sans `pointer`, pointage relatif (comportement
    d'origine). Dans les deux cas, le curseur reste figé pendant un pincement court, pour
    que le clic tombe à l'endroit visé.
    """

    def __init__(
        self,
        engagement: EngagementConfig,
        pinch: PinchConfig,
        fingers: FingersConfig | None = None,
        pointer: PointerConfig | None = None,
    ):
        if fingers is None:
            fingers = FingersConfig()
        if pointer is None:
            pointer = PointerConfig(mode="relative")
        self.engagement = engagement
        self.pinch = pinch
        self.pointer = pointer
        self.state = State.IDLE
        thresholds = RuleThresholds(
            enter=pinch.enter,
            min_index_extension=pinch.min_index_extension,
            folded_below=fingers.folded_below,
            min_index_extension_3d=pinch.min_index_extension_3d,
            folded_below_3d=fingers.folded_below_3d,
            require_palm_facing=pinch.require_palm_facing,
            min_palm_width=pinch.min_palm_width,
        )
        self._pinch_detector = PinchHysteresis(
            pinch.rule, thresholds, pinch.exit, pinch.release_grace_ms, pinch.exit_on_fold
        )
        self._arm_start_ms: float | None = None
        self._pinch_start_ms: float = 0.0
        self._last_seen_ms: float | None = None
        self._prev_pointer: tuple[float, float] | None = None
        self._last_target: tuple[float, float] | None = None  # dernier MoveTo émis
        self._below_since_ms: float | None = None  # depuis quand la main est sous la zone

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
                self._pinch_detector.reset()
                self._below_since_ms = None
            return events

        self._last_seen_ms = t_ms

        zone = self.pointer.zone
        inside = in_zone(pointer[0], pointer[1], zone)

        # --- Veille : on attend une main ouverte maintenue (dans la zone, si demandé) ---
        if self.state == State.IDLE:
            arming = features.open_palm
            if self.engagement.activate_in_zone and not inside:
                arming = False  # main ouverte posée en bas, hors zone : ce n'est pas un appel
            if arming:
                if self._arm_start_ms is None:
                    self._arm_start_ms = t_ms
                elif t_ms - self._arm_start_ms >= self.engagement.hold_ms:
                    self.state = State.ACTIVE
                    self._arm_start_ms = None
                    self._below_since_ms = None
                    self._prev_pointer = pointer  # pas de saut du curseur à l'activation
                    self._pinch_detector.reset()
            else:
                self._arm_start_ms = None
            return events

        # --- Main baissée sous la zone pendant rest_ms : retour en veille ---
        if self.engagement.rest_ms > 0 and below_zone(pointer[1], zone):
            if self._below_since_ms is None:
                self._below_since_ms = t_ms
            elif t_ms - self._below_since_ms >= self.engagement.rest_ms:
                if self.state == State.DRAGGING:
                    events.append(Release())  # on ne laisse pas le bouton enfoncé
                # Un pincement court en cours est simplement annulé : pas de clic.
                self.state = State.IDLE
                self._below_since_ms = None
                self._arm_start_ms = None
                self._pinch_detector.reset()
                return events
        else:
            self._below_since_ms = None

        # --- Déplacement du curseur (sauf pendant un pincement court, pour un clic précis) ---
        if self.state != State.PINCHING:
            if self.pointer.mode == "absolute":
                target = zone_to_screen(pointer[0], pointer[1], self.pointer.zone)
                if target != self._last_target:
                    events.append(MoveTo(target[0], target[1]))
                    self._last_target = target
            elif self._prev_pointer is not None:
                dx = pointer[0] - self._prev_pointer[0]
                dy = pointer[1] - self._prev_pointer[1]
                if dx != 0 or dy != 0:
                    events.append(Move(dx, dy))
        self._prev_pointer = pointer

        # --- Pincement : entrée / sortie confiées à PinchHysteresis ---
        # Hors zone, un pincement ne peut pas COMMENCER (main baissée, au repos) ; un pincement
        # déjà commencé dans la zone continue, pour ne pas lâcher un glisser vers un bord.
        pinch_event = None
        may_start = inside or not self.pinch.only_in_zone
        if self._pinch_detector.pinched or may_start:
            pinch_event = self._pinch_detector.update(
                t_ms,
                pinch_measure(features, self.pinch.measure),
                features.finger_extensions,
                features.finger_extensions_3d,
                features.palm_width,
            )

        if self.state == State.ACTIVE:
            if pinch_event == "start":
                self.state = State.PINCHING
                self._pinch_start_ms = t_ms

        elif self.state == State.PINCHING:
            if pinch_event == "end":
                events.append(Click())
                self.state = State.ACTIVE
            elif t_ms - self._pinch_start_ms > self.pinch.click_max_ms:
                events.append(Press())
                self.state = State.DRAGGING

        elif self.state == State.DRAGGING:
            if pinch_event == "end":
                events.append(Release())
                self.state = State.ACTIVE

        return events
