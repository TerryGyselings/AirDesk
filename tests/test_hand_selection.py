"""Tests des rôles des mains : main pilote (côté préféré) et main d'appoint."""

import pytest

from airdesk.config import EngagementConfig, PinchConfig, PointerConfig
from airdesk.interpretation.features import HandFeatures
from airdesk.interpretation.hand_selection import HandCandidate, HandSelector, Selection
from airdesk.interpretation.state_machine import GestureStateMachine, State
from airdesk.perception.hand_tracker import Hand, user_side

SIZE = 0.15  # taille de main (unités homogènes)


def hand(x, y, side="right", open_palm=False, size=SIZE):
    return HandCandidate((x, y), size, side, open_palm)


# --- En veille : seule la main du côté préféré peut devenir pilote ----------------------------


def test_idle_pilot_is_the_preferred_side():
    selector = HandSelector(preferred="right")
    candidates = [hand(0.3, 0.5, "left", open_palm=True), hand(0.7, 0.5, "right", open_palm=True)]
    assert selector.select(candidates, idle=True) == Selection(pilot=1, support=None)


def test_other_side_never_becomes_pilot_even_with_the_activation_gesture():
    """La main gauche s'ouvre, la droite non : la gauche ne pilote pas pour autant."""
    selector = HandSelector(preferred="right")
    candidates = [hand(0.3, 0.5, "left", open_palm=True), hand(0.7, 0.5, "right")]
    assert selector.select(candidates, idle=True).pilot == 1
    assert selector.select([hand(0.3, 0.5, "left", open_palm=True)], idle=True).pilot is None


def test_idle_prefers_an_open_hand_then_the_largest():
    selector = HandSelector(preferred="right")
    candidates = [hand(0.3, 0.5, "right", size=0.20), hand(0.7, 0.5, "right", open_palm=True)]
    assert selector.select(candidates, idle=True).pilot == 1
    candidates = [hand(0.3, 0.5, "right", size=0.10), hand(0.7, 0.5, "right", size=0.20)]
    assert selector.select(candidates, idle=True).pilot == 1


def test_any_keeps_the_former_behaviour():
    """preferred: any : n'importe quelle main peut piloter (ancien comportement)."""
    selector = HandSelector(preferred="any")
    candidates = [hand(0.3, 0.5, "left", open_palm=True), hand(0.7, 0.5, "right")]
    assert selector.select(candidates, idle=True).pilot == 0


def test_no_hand_gives_no_pilot():
    assert HandSelector().select([], idle=True) == Selection(None, None)
    assert HandSelector().select([], idle=False) == Selection(None, None)


def test_unknown_preference_is_rejected_in_french():
    with pytest.raises(ValueError, match="hand.preferred inconnu"):
        HandSelector(preferred="both")


# --- Une fois actif : verrou par continuité, main d'appoint -----------------------------------


def test_pilot_is_followed_when_two_hands_cross():
    """Main droite (pilote) de gauche à droite, main gauche en sens inverse, 20 % plus bas.
    L'ordre de la liste s'inverse à chaque image ; la main gauche est l'appoint."""
    selector = HandSelector(preferred="right", max_jump=1.5)
    pilot_x = 0.20
    other_x = 0.80
    first = [hand(pilot_x, 0.40, "right", open_palm=True), hand(other_x, 0.60, "left")]
    assert selector.select(first, idle=True).pilot == 0
    for frame in range(1, 31):
        pilot_x += 0.02  # 0,13 taille de main par image
        other_x -= 0.02
        pilot = hand(pilot_x, 0.40, "right")
        other = hand(other_x, 0.60, "left", open_palm=True)
        candidates = [pilot, other] if frame % 2 == 0 else [other, pilot]
        roles = selector.select(candidates, idle=False)
        assert candidates[roles.pilot] is pilot, f"image {frame} : mauvaise main pilote"
        assert candidates[roles.support] is other, f"image {frame} : mauvaise main d'appoint"


def test_support_exists_only_while_the_pilot_is_active():
    selector = HandSelector(preferred="right")
    candidates = [hand(0.7, 0.5, "right", open_palm=True), hand(0.3, 0.5, "left")]
    assert selector.select(candidates, idle=True).support is None  # en veille : pas d'appoint
    assert selector.select(candidates, idle=False).support == 1  # actif : appoint
    assert selector.select([hand(0.3, 0.5, "left")], idle=False) == Selection(None, None)


def test_pilot_too_far_is_lost_and_never_replaced():
    selector = HandSelector(preferred="right", max_jump=1.5)
    selector.select([hand(0.30, 0.50, "right", open_palm=True)], idle=True)
    selector.select([hand(0.31, 0.50, "right")], idle=False)
    assert selector.select([hand(0.80, 0.50, "right")], idle=False).pilot is None  # saut
    # La main gauche, même tout près de la dernière position, ne remplace pas la pilote.
    assert selector.select([hand(0.32, 0.50, "left", open_palm=True)], idle=False).pilot is None
    # La main pilote revient près de sa dernière position : elle est retrouvée.
    assert selector.select([hand(0.33, 0.50, "right")], idle=False).pilot == 0


# --- Le scénario constaté en usage réel (log 2026-11-24_17-33-21) -----------------------------

ENGAGEMENT = EngagementConfig(hold_ms=300, lost_timeout_ms=1000)
POINTER = PointerConfig(mode="absolute")  # zone par défaut 15-85 % x 20-80 %


def features_for(open_palm: bool) -> HandFeatures:
    """Descripteurs simplifiés : main ouverte (4 doigts tendus) ou demi-fermée, sans pincement."""
    extension = 1.5 if open_palm else 0.6
    extensions = (extension, extension, extension, extension)
    return HandFeatures(
        hand_size=0.2, pinch_ratio=0.8, extended_fingers=4 if open_palm else 2,
        open_palm=open_palm, finger_extensions=extensions, index_extension=extension,
        folded_fingers=0, finger_extensions_3d=extensions, index_extension_3d=extension,
        folded_fingers_3d=0, pinch_seg=0.8, pinch_seg_68=0.8, palm_width=0.8, palm_facing=1.0,
    )


def run_frame(selector, machine, t, candidates):
    """Une image, comme dans app.py : rôles des mains, puis la main pilote seule à la machine."""
    roles = selector.select(candidates, idle=machine.state == State.IDLE)
    if roles.pilot is None:
        return machine.update(t, None, None)
    c = candidates[roles.pilot]
    return machine.update(t, features_for(c.open_palm), (c.wrist[0], c.wrist[1]))


@pytest.mark.parametrize("other_open", [False, True])
def test_pilot_leaves_and_the_other_hand_never_takes_control(other_open):
    """La main pilote sort du champ ; l'autre main reste visible, ouverte ou non, bien au-delà
    du retour en veille (1 s) et de hold_ms : AirDesk ne s'active pas avec elle. Le contrôle
    ne revient que quand la main pilote réapparaît et fait elle-même le geste d'activation."""
    selector = HandSelector(preferred="right", max_jump=1.5)
    machine = GestureStateMachine(ENGAGEMENT, PinchConfig(), pointer=POINTER)
    left = hand(0.30, 0.50, "left", open_palm=other_open)

    t = 0
    while t <= 400:  # activation par la main droite
        run_frame(selector, machine, t, [hand(0.60, 0.50, "right", open_palm=True), left])
        t += 33
    assert machine.state == State.ACTIVE

    events = []
    while t <= 4000:  # la main droite sort du champ ; la gauche reste 3,6 s
        events += run_frame(selector, machine, t, [left])
        t += 33
    assert machine.state == State.IDLE  # retour en veille après la perte de la pilote
    assert events == []  # et aucun mouvement, aucun clic venus de la main gauche

    while t <= 4500:  # la main droite revient, ouverte : activation complète
        run_frame(selector, machine, t, [left, hand(0.60, 0.50, "right", open_palm=True)])
        t += 33
    assert machine.state == State.ACTIVE


# --- Latéralité : étiquette MediaPipe corrigée -------------------------------------------------


def test_mediapipe_label_is_inverted_for_the_user():
    """Mesuré sur 2 sessions main droite : MediaPipe répond « Left » (voir journal)."""
    assert user_side("Left") == "right"
    assert user_side("Right") == "left"
    assert user_side("?") == "unknown"
    assert Hand([(0.0, 0.0, 0.0)] * 21, "Left", 0.9).side == "right"
