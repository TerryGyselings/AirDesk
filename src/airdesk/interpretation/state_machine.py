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


@dataclass(frozen=True)
class PressAt:
    """Bouton enfoncé en un point « épinglé » (fractions de l'écran) : le point du contact,
    ou celui du 1er clic d'un double-clic. L'actionneur y place le curseur avant d'appuyer."""

    u: float
    v: float


@dataclass(frozen=True)
class ReleaseAt:
    """Bouton relâché au point épinglé : le système voit un clic sur place."""

    u: float
    v: float


Event = Move | MoveTo | Click | Press | Release | PressAt | ReleaseAt


class GestureStateMachine:
    """Machine à états d'engagement et de pincement.

    L'entrée / sortie du pincement est déléguée à PinchHysteresis (pinch_rules.py), réglée
    par config/gestures.yaml : règle (A = pinch_ratio seul, D = + un autre doigt ouvert...),
    mesure (pinch_ratio ou pinch_seg), enter / exit, délai de relâchement, sortie sur repli.
    Avec rule A, measure ratio et release_grace_ms 0, on retrouve le comportement d'origine.
    Choix de la configuration : voir scripts/analyse_gestures.py et docs/journal.md.

    Pointage : `pointer.mode` "absolute" émet MoveTo (zone de l'image -> écran entier),
    "relative" émet Move (déplacements). Sans `pointer`, pointage relatif (comportement
    d'origine). Mode release : le curseur reste figé pendant tout le pincement. Mode press :
    il suit la main en permanence, sauf bouton enfoncé sur place (voir _update_press_mode).

    Clics épinglés (mode press, pointage absolu) : PressAt / ReleaseAt portent le point du
    clic (contact, ou 1er clic d'un double-clic). En pointage relatif, la machine ne connaît
    pas la position du curseur : Press / Release simples, sans double-clic aimanté.
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
            max_contact_speed=pinch.max_contact_speed,
            closing_grace_ms=pinch.closing_grace_ms,
        )
        self._pinch_detector = PinchHysteresis(
            pinch.rule, thresholds, pinch.exit, pinch.release_grace_ms, pinch.exit_on_fold,
            pinch.release_cancel_margin, pinch.release_fast_margin, pinch.release_rise,
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
        self._press_ms = 0.0  # instant du Press de l'appui en cours
        self._anchor: tuple[float, float] | None = None  # point de pointage au contact
        # Rattrapage : le curseur figé rejoint la main progressivement (voir _catch_up).
        self._catchup_from: tuple[float, float] | None = None  # position du curseur figé
        self._catchup_start_ms: float | None = None
        # Clics épinglés (pointage absolu) : point de l'écran où l'appui en cours a été émis.
        self._pin: tuple[float, float] | None = None
        # Dernier clic sur place, pour aimanter le 2e clic d'un double-clic (voir _pin_point) :
        # instant du Press, point épinglé, point de pointage au contact.
        self._last_click_ms: float | None = None
        self._last_click_pin: tuple[float, float] | None = None
        self._last_click_anchor: tuple[float, float] | None = None
        # Point de pointage du dernier clic sur place (les deux pointages), pour l'exemption
        # de fermeture de la 2e tape (voir _is_second_tap_candidate). None après un glisser.
        self._grace_anchor: tuple[float, float] | None = None
        # Pour le log (colonne second_tap) : ce contact peut-il être la 2e tape d'un
        # double-clic ? (près du clic précédent, poignet lent)
        self.second_tap_candidate = False
        self._snapped = False  # l'appui en cours est aimanté sur le clic précédent
        # Pour le log : instant du contact du pincement dont le premier Press / Click vient
        # d'être émis (None sinon), distance au point d'ancrage pendant un appui, et type de
        # l'événement épinglé de l'image ("press", "press_snap", "release", "release_snap").
        self.event_contact_ms: float | None = None
        self.anchor_distance: float | None = None
        self.pin_kind = ""

    def update(
        self,
        t_ms: float,
        features: HandFeatures | None,
        pointer: tuple[float, float] | None,
        wrist_speed: float | None = None,
    ) -> list[Event]:
        """Fait avancer la machine d'une image. `features` et `pointer` valent None sans main.

        `wrist_speed` : vitesse du poignet (tailles de main / s), pour la condition facultative
        `max_contact_speed` (None = inconnue, pas de refus)."""
        events: list[Event] = []
        self.event_contact_ms = None
        self.anchor_distance = None
        self.pin_kind = ""

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
                self._last_click_ms = None  # une série de clics ne survit pas à la veille
                self._grace_anchor = None
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
                self._last_click_ms = None
                self._grace_anchor = None
                self.state = State.IDLE
                self._below_since_ms = None
                self._arm_start_ms = None
                self._pinch_detector.reset()
                self.entry_block = "veille"
                return events
        else:
            self._below_since_ms = None

        # --- Déplacement du curseur ---
        # Figé pendant un pincement court, pour que le clic tombe à l'endroit visé : tout le
        # pincement en mode release, seulement bouton enfoncé en mode press (le curseur reste
        # alors au point épinglé : aucun déplacement bouton enfoncé, donc pas de mini-glisser).
        frozen = self.state == State.PINCHING and (
            self.pinch.click_mode != "press" or self._pressed)
        if not frozen:
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
        self.second_tap_candidate = False
        if not already_pinched:
            self.second_tap_candidate = self._is_second_tap_candidate(pointer, features,
                                                                      wrist_speed)
        if already_pinched or may_start:
            pinch_event = self._pinch_detector.update(
                t_ms,
                pinch_measure(features, self.pinch.measure),
                features.finger_extensions,
                features.finger_extensions_3d,
                features.palm_width,
                self.closing_speed,
                self.closing_speed_3d,
                wrist_speed,
                # Relâchement relatif (release_rise) : seulement pendant un appui sur place,
                # jamais pendant un glisser (il y couperait le glisser, voir docs/journal.md).
                self.pinch.click_mode == "press" and self.state == State.PINCHING,
                self.second_tap_candidate,
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

        - Contact (la détection dit « start ») : PINCHING ; le curseur suit encore la main.
        - Après `min_press_frames` images de pincement d'affilée : Press, épinglé au point du
          contact (PressAt en pointage absolu, voir _pin_point). Un pincement plus bref
          (glitch du suivi) ne produit rien.
        - Bouton enfoncé, le curseur reste au point épinglé tant que la main ne s'est pas
          éloignée du point de contact de plus de `drag_threshold` tailles de main ; au-delà :
          DRAGGING, le curseur suit (après un rattrapage progressif, voir _catch_up).
        - Relâchement confirmé : ReleaseAt au point épinglé (clic sur place), ou Release
          simple à la fin d'un glisser. Après un clic, le curseur rejoint la main tout de suite
          (rattrapage) : plus de verrou.
        """
        if self.state == State.ACTIVE:
            if pinch_event == "start":
                self.state = State.PINCHING
                self._pinch_start_ms = t_ms
                self._pinch_frames = 1
                self._anchor = pointer
                self._pressed = False
                self._press_if_confirmed(t_ms, features, events)
            return

        if self.state == State.PINCHING:
            if pinch_event == "end":
                if self._pressed:
                    self._release_in_place(t_ms, events)
                # Sinon : pincement trop bref pour être confirmé, rien n'a été envoyé.
                self._pressed = False
                self.state = State.ACTIVE
                return
            if not self._pressed:
                # Image de pincement de plus, sauf si un relâchement est déjà en cours de
                # confirmation : le pincement n'est alors plus « présent ».
                if not self._pinch_detector.releasing:
                    self._pinch_frames += 1
                    self._press_if_confirmed(t_ms, features, events)
                return
            self.anchor_distance = self._distance_from_anchor(pointer, features)
            if self.anchor_distance > self.pinch.drag_threshold:
                self.state = State.DRAGGING  # la main part : c'est un glisser
                self._last_click_ms = None  # un glisser interrompt la série de clics
                self._grace_anchor = None  # ni aimant, ni exemption après un glisser
                self._start_catch_up(t_ms)
            return

        if self.state == State.DRAGGING:
            self.anchor_distance = self._distance_from_anchor(pointer, features)
            if pinch_event == "end":
                events.append(Release())  # fin du glisser, à la position réelle
                self._pressed = False
                self.state = State.ACTIVE

    def _press_if_confirmed(self, t_ms: float, features, events: list[Event]) -> None:
        if self._pinch_frames < self.pinch.min_press_frames:
            return
        self._pressed = True
        self._press_ms = t_ms
        self.event_contact_ms = self._pinch_start_ms
        if self.pointer.mode != "absolute":
            events.append(Press())
            return
        self._pin = self._pin_point(t_ms, features)
        events.append(PressAt(self._pin[0], self._pin[1]))
        self.pin_kind = "press_snap" if self._snapped else "press"
        # Le curseur est désormais au point épinglé : point de départ du rattrapage à venir.
        self._last_target = self._pin

    def _pin_point(self, t_ms: float, features) -> tuple[float, float]:
        """Point de l'écran où émettre le Press : le point du contact, ou, pour le 2e clic
        d'un double-clic, les coordonnées exactes du 1er clic.

        Windows ne reconnaît un double-clic que si les deux clics tombent à ±2 pixels
        (SM_CXDOUBLECLK = 4) et à moins du délai de double-clic du système. Entre deux tapes,
        la main se déplace de 1 à 21 px (logs du 2026-11-24 et du 2026-11-25). Le 2e appui est
        donc « aimanté » sur le 1er s'il arrive à moins de `double_click_ms` du Press précédent
        ET à moins de `double_click_snap` tailles de main de son point de contact. Une série
        (triple-clic) reste aimantée sur le point du 1er clic."""
        own = zone_to_screen(self._anchor[0], self._anchor[1], self.pointer.zone)
        self._snapped = False
        if self._last_click_ms is None or self.pinch.double_click_snap <= 0:
            return own
        if t_ms - self._last_click_ms > self.pinch.double_click_ms:
            return own
        size = features.hand_size if features.hand_size > 1e-6 else 1e-6
        dx = self._anchor[0] - self._last_click_anchor[0]
        dy = self._anchor[1] - self._last_click_anchor[1]
        if (dx * dx + dy * dy) ** 0.5 / size > self.pinch.double_click_snap:
            return own
        self._snapped = True
        self._anchor = self._last_click_anchor  # la série garde le point du 1er clic
        return self._last_click_pin

    def _release_in_place(self, t_ms: float, events: list[Event]) -> None:
        """Fin d'un clic sur place : Release au point épinglé, puis le curseur rejoint la main."""
        self._grace_anchor = self._anchor  # point du clic, pour l'exemption de la 2e tape
        if self.pointer.mode != "absolute":
            events.append(Release())
            return
        events.append(ReleaseAt(self._pin[0], self._pin[1]))
        self.pin_kind = "release_snap" if self._snapped else "release"
        # Mémorisé pour aimanter le clic suivant (double-clic) : instant du Press.
        self._last_click_ms = self._press_ms
        self._last_click_pin = self._pin
        self._last_click_anchor = self._anchor
        self._start_catch_up(t_ms)

    def _is_second_tap_candidate(self, pointer, features, wrist_speed) -> bool:
        """Ce contact peut-il être la 2e tape d'un double-clic ? (exemption de fermeture de la
        règle F, voir PinchHysteresis). Il faut, après un clic sur place :
        - la main à moins de `double_click_snap` tailles de main du point de ce clic ;
        - un poignet lent : vitesse connue et < `closing_grace_max_speed`.
        Logs réels : 26 vraies 2es tapes à 1-22 px du clic, poignet 0,08 à 0,68 ; contacts
        exemptés à tort en traversant l'écran (log 17-54-53) à 203-333 px, poignet 1,84 à
        4,43. Le délai (closing_grace_ms) est vérifié par PinchHysteresis."""
        if self._grace_anchor is None or self.pinch.double_click_snap <= 0:
            return False
        if wrist_speed is None or wrist_speed >= self.pinch.closing_grace_max_speed:
            return False
        size = features.hand_size if features.hand_size > 1e-6 else 1e-6
        dx = pointer[0] - self._grace_anchor[0]
        dy = pointer[1] - self._grace_anchor[1]
        return (dx * dx + dy * dy) ** 0.5 / size <= self.pinch.double_click_snap

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
        """Quand le curseur cesse d'être figé (début d'un glisser, fin d'un clic ; pointage
        absolu), il rejoint la main progressivement pendant `drag_catchup_ms` au lieu de sauter
        du point figé à la main (le saut vaut jusqu'à drag_threshold tailles de main, une
        centaine de pixels, ou double_click_snap après un double-clic aimanté).
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
