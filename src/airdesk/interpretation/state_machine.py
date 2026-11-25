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
from airdesk.interpretation.pinch_rules import (
    ClosingSpeed,
    PinchHysteresis,
    RuleThresholds,
    pinch_measure,
)
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
            min_closing_speed=pinch.min_closing_speed,
            min_closing_speed_3d=pinch.min_closing_speed_3d,
        )
        self._pinch_detector = PinchHysteresis(
            pinch.rule, thresholds, pinch.exit, pinch.release_grace_ms, pinch.exit_on_fold,
            pinch.release_cancel_margin,
        )
        # Vitesse de fermeture des doigts (règles F / F3D, et colonnes du log).
        self._closing = ClosingSpeed()
        self._closing_3d = ClosingSpeed()
        self.closing_speed: float | None = None
        self.closing_speed_3d: float | None = None
        # Pourquoi un pincement ne peut pas commencer à cette image (colonne entry_block du
        # log) : "veille", "hors_zone", "pincement_en_cours", ou la raison de la règle
        # ("ratio", "paume", "autres_doigts"...) ; "" si l'entrée est possible.
        self.entry_block = ""
        self._arm_start_ms: float | None = None
        self._pinch_start_ms: float = 0.0
        self._last_seen_ms: float | None = None
        self._prev_pointer: tuple[float, float] | None = None
        self._last_target: tuple[float, float] | None = None  # dernier MoveTo émis
        self._below_since_ms: float | None = None  # depuis quand la main est sous la zone
        # Mode de clic « press » (comme une souris) : bouton enfoncé dès le contact confirmé.
        self._pressed = False  # bouton de la souris enfoncé (Press émis, pas encore Release)
        self._pinch_frames = 0  # images consécutives de pincement depuis le contact
        self._anchor: tuple[float, float] | None = None  # point de pointage au contact
        # Rattrapage : le curseur figé rejoint la main progressivement (voir _catch_up).
        self._catchup_from: tuple[float, float] | None = None  # position du curseur figé
        self._catchup_start_ms: float | None = None
        # Pour le log : instant du contact du pincement dont le premier Press / Click vient
        # d'être émis (None sinon), et distance au point d'ancrage pendant un appui.
        self.event_contact_ms: float | None = None
        self.anchor_distance: float | None = None

    def update(
        self,
        t_ms: float,
        features: HandFeatures | None,
        pointer: tuple[float, float] | None,
    ) -> list[Event]:
        """Fait avancer la machine d'une image. `features` et `pointer` valent None sans main."""
        events: list[Event] = []
        self.event_contact_ms = None
        self.anchor_distance = None

        # --- Pas de main visible ---
        if features is None or pointer is None:
            self._arm_start_ms = None
            self._prev_pointer = None
            self._closing.reset()
            self._closing_3d.reset()
            self.closing_speed = None
            self.closing_speed_3d = None
            self.entry_block = ""
            lost_for = 0.0
            if self._last_seen_ms is not None:
                lost_for = t_ms - self._last_seen_ms
            if self.state != State.IDLE and lost_for > self.engagement.lost_timeout_ms:
                if self.state == State.DRAGGING or self._pressed:
                    events.append(Release())
                self.state = State.IDLE
                self._pinch_detector.reset()
                self._below_since_ms = None
                self._pressed = False
            return events

        self._last_seen_ms = t_ms
        self.closing_speed = self._closing.update(t_ms, features.pinch_ratio)
        self.closing_speed_3d = self._closing_3d.update(t_ms, features.pinch_ratio_3d)

        zone = self.pointer.zone
        inside = in_zone(pointer[0], pointer[1], zone)

        # --- Veille : on attend une main ouverte maintenue (dans la zone, si demandé) ---
        if self.state == State.IDLE:
            self.entry_block = "veille"
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
                if self.state == State.DRAGGING or self._pressed:
                    events.append(Release())  # on ne laisse pas le bouton enfoncé
                # Un pincement court en cours (mode release) est simplement annulé : pas de clic.
                self._pressed = False
                self.state = State.IDLE
                self._below_since_ms = None
                self._arm_start_ms = None
                self._pinch_detector.reset()
                self.entry_block = "veille"
                return events
        else:
            self._below_since_ms = None

        # --- Déplacement du curseur (sauf pendant un pincement court, pour un clic précis) ---
        if self.state != State.PINCHING:
            if self.pointer.mode == "absolute":
                target = zone_to_screen(pointer[0], pointer[1], self.pointer.zone)
                target = self._catch_up(t_ms, target)
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
        already_pinched = self._pinch_detector.pinched
        if already_pinched or may_start:
            pinch_event = self._pinch_detector.update(
                t_ms,
                pinch_measure(features, self.pinch.measure),
                features.finger_extensions,
                features.finger_extensions_3d,
                features.palm_width,
                self.closing_speed,
                self.closing_speed_3d,
            )
        if already_pinched:
            self.entry_block = "pincement_en_cours"
        elif not may_start:
            self.entry_block = "hors_zone"
        else:
            self.entry_block = self._pinch_detector.last_block or ""

        if self.pinch.click_mode == "press":
            self._update_press_mode(t_ms, pinch_event, features, pointer, events)
            return events

        if self.state == State.ACTIVE:
            if pinch_event == "start":
                self.state = State.PINCHING
                self._pinch_start_ms = t_ms

        elif self.state == State.PINCHING:
            if pinch_event == "end":
                events.append(Click())
                self.event_contact_ms = self._pinch_start_ms
                self.state = State.ACTIVE
            elif self._pinch_detector.releasing:
                # Relâchement en cours de confirmation (release_grace_ms) : pas de Press. La
                # durée du pincement s'arrête au DÉBUT du relâchement ; s'il est confirmé, c'est
                # un clic, même si la confirmation fait dépasser click_max_ms.
                pass
            elif t_ms - self._pinch_start_ms > self.pinch.click_max_ms:
                events.append(Press())
                self.event_contact_ms = self._pinch_start_ms
                self.state = State.DRAGGING

        elif self.state == State.DRAGGING:
            if pinch_event == "end":
                events.append(Release())
                self.state = State.ACTIVE

        return events

    def _update_press_mode(self, t_ms, pinch_event, features, pointer, events) -> None:
        """Mode « press », comme une souris : c'est le système qui distingue clic et glisser.

        - Contact (la détection dit « start ») : PINCHING ; le curseur se fige.
        - Après `min_press_frames` images de pincement d'affilée : Press. Un pincement plus
          bref (glitch du suivi) ne produit rien.
        - Bouton enfoncé, le curseur reste figé tant que la main ne s'est pas éloignée du point
          de contact de plus de `drag_threshold` tailles de main ; au-delà : DRAGGING, le
          curseur suit (après un rattrapage progressif, voir _catch_up).
        - Relâchement confirmé : Release. Clic sur place = Press puis Release au même point ;
          deux clics rapides font un double-clic, comme avec une vraie souris.
        """
        if self.state == State.ACTIVE:
            if pinch_event == "start":
                self.state = State.PINCHING
                self._pinch_start_ms = t_ms
                self._pinch_frames = 1
                self._anchor = pointer
                self._pressed = False
                self._press_if_confirmed(events)
            return

        if self.state == State.PINCHING:
            if pinch_event == "end":
                if self._pressed:
                    events.append(Release())  # clic sur place
                # Sinon : pincement trop bref pour être confirmé, rien n'a été envoyé.
                self._pressed = False
                self.state = State.ACTIVE
                return
            if not self._pressed:
                # Image de pincement de plus, sauf si un relâchement est déjà en cours de
                # confirmation : le pincement n'est alors plus « présent ».
                if not self._pinch_detector.releasing:
                    self._pinch_frames += 1
                    self._press_if_confirmed(events)
                return
            self.anchor_distance = self._distance_from_anchor(pointer, features)
            if self.anchor_distance > self.pinch.drag_threshold:
                self.state = State.DRAGGING  # la main part : c'est un glisser
                self._start_catch_up(t_ms)
            return

        if self.state == State.DRAGGING:
            self.anchor_distance = self._distance_from_anchor(pointer, features)
            if pinch_event == "end":
                events.append(Release())
                self._pressed = False
                self.state = State.ACTIVE

    def _press_if_confirmed(self, events: list[Event]) -> None:
        if self._pinch_frames >= self.pinch.min_press_frames:
            events.append(Press())
            self._pressed = True
            self.event_contact_ms = self._pinch_start_ms

    def _distance_from_anchor(self, pointer, features) -> float:
        """Distance du point de pointage au point de contact, en tailles de main."""
        size = features.hand_size if features.hand_size > 1e-6 else 1e-6
        dx = pointer[0] - self._anchor[0]
        dy = pointer[1] - self._anchor[1]
        return (dx * dx + dy * dy) ** 0.5 / size

    def _start_catch_up(self, t_ms: float) -> None:
        """Le curseur, figé jusqu'ici, va rejoindre la main progressivement (_catch_up)."""
        self._catchup_start_ms = t_ms
        self._catchup_from = self._last_target

    def _catch_up(self, t_ms: float, target: tuple[float, float]) -> tuple[float, float]:
        """Quand le curseur cesse d'être figé (début d'un glisser ; pointage absolu), il
        rejoint la main progressivement pendant `drag_catchup_ms` au lieu de sauter du point
        figé à la main (le saut vaut jusqu'à drag_threshold tailles de main, une centaine de
        pixels).
        Interpolation linéaire : figé + k x (main - figé), k passant de 0 à 1."""
        if (self._catchup_start_ms is None or self._catchup_from is None
                or self.pinch.drag_catchup_ms <= 0):
            return target
        k = (t_ms - self._catchup_start_ms) / self.pinch.drag_catchup_ms
        if k >= 1:
            self._catchup_start_ms = None  # rattrapage terminé
            return target
        u0, v0 = self._catchup_from
        return (u0 + k * (target[0] - u0), v0 + k * (target[1] - v0))
