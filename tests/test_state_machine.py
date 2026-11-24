from airdesk.config import EngagementConfig, PinchConfig
from airdesk.interpretation.features import HandFeatures
from airdesk.interpretation.state_machine import (
    Click,
    GestureStateMachine,
    Move,
    Press,
    Release,
    State,
)

OPEN = HandFeatures(hand_size=0.2, pinch_ratio=0.8, extended_fingers=4, open_palm=True)
FIST = HandFeatures(hand_size=0.2, pinch_ratio=0.8, extended_fingers=0, open_palm=False)
PINCH = HandFeatures(hand_size=0.2, pinch_ratio=0.10, extended_fingers=3, open_palm=False)
# Entre les deux seuils (0.25 < 0.33 < 0.40) : sert à tester l'hystérésis.
HALF = HandFeatures(hand_size=0.2, pinch_ratio=0.33, extended_fingers=3, open_palm=False)

P = (0.5, 0.5)


def new_machine():
    return GestureStateMachine(EngagementConfig(hold_ms=300, lost_timeout_ms=1000), PinchConfig())


def activate(m):
    """Main ouverte de t=0 à t=330 ms -> état ACTIF."""
    t = 0
    while t <= 330:
        m.update(t, OPEN, P)
        t += 33
    assert m.state == State.ACTIVE
    return t


def test_starts_idle_and_ignores_moves():
    m = new_machine()
    events = m.update(0, FIST, (0.1, 0.1))
    events += m.update(33, FIST, (0.9, 0.9))
    assert m.state == State.IDLE
    assert events == []


def test_open_palm_must_be_held_to_activate():
    m = new_machine()
    m.update(0, OPEN, P)
    m.update(100, OPEN, P)
    assert m.state == State.IDLE  # pas encore 300 ms
    m.update(200, FIST, P)  # interruption : le compteur repart de zéro
    m.update(400, OPEN, P)
    m.update(600, OPEN, P)
    assert m.state == State.IDLE
    m.update(750, OPEN, P)
    assert m.state == State.ACTIVE


def test_moves_are_relative_once_active():
    m = new_machine()
    t = activate(m)
    events = m.update(t, OPEN, (0.6, 0.45))
    assert len(events) == 1
    assert isinstance(events[0], Move)
    assert abs(events[0].dx - 0.1) < 1e-9
    assert abs(events[0].dy + 0.05) < 1e-9


def test_short_pinch_is_a_click():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    assert m.state == State.PINCHING
    events = m.update(t + 150, OPEN, P)
    assert Click() in events
    assert m.state == State.ACTIVE


def test_long_pinch_is_a_drag():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    events = m.update(t + 400, PINCH, P)
    assert Press() in events
    assert m.state == State.DRAGGING

    events = m.update(t + 433, PINCH, (0.6, 0.5))  # pendant le glisser, le curseur suit
    assert len(events) == 1
    assert isinstance(events[0], Move)
    assert abs(events[0].dx - 0.1) < 1e-9

    events = m.update(t + 466, OPEN, (0.6, 0.5))
    assert Release() in events
    assert m.state == State.ACTIVE


def test_hysteresis_keeps_drag_when_fingers_open_slightly():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 400, PINCH, P)
    assert m.state == State.DRAGGING
    m.update(t + 433, HALF, P)  # doigts un peu entrouverts : on reste en glisser
    assert m.state == State.DRAGGING


def test_half_pinch_does_not_start_pinch():
    m = new_machine()
    t = activate(m)
    m.update(t, HALF, P)
    assert m.state == State.ACTIVE


def test_lost_hand_returns_to_idle_and_releases_drag():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 400, PINCH, P)
    assert m.state == State.DRAGGING

    events = m.update(t + 900, None, None)  # perdue depuis 500 ms : on attend encore
    assert m.state == State.DRAGGING
    assert events == []

    events = m.update(t + 1500, None, None)  # perdue depuis 1100 ms
    assert m.state == State.IDLE
    assert Release() in events
