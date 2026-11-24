"""Tests du choix de la main suivie quand deux mains sont visibles."""

import pytest

from airdesk.interpretation.hand_selection import HandCandidate, HandSelector
from airdesk.perception.hand_tracker import Hand, user_side

SIZE = 0.15  # taille de main (unités homogènes)


def hand(x, y, side="right", open_palm=False, size=SIZE):
    return HandCandidate((x, y), size, side, open_palm)


# --- En veille : quelle main fait le geste d'activation ? -------------------------------------


def test_idle_prefers_the_open_hand_on_the_preferred_side():
    selector = HandSelector(preferred="right")
    candidates = [hand(0.3, 0.5, "left", open_palm=True), hand(0.7, 0.5, "right", open_palm=True)]
    assert selector.select(candidates, idle=True) == 1


def test_idle_activation_gesture_beats_side_preference():
    """La main gauche fait le geste, la droite non : c'est la gauche qui veut piloter."""
    selector = HandSelector(preferred="right")
    candidates = [hand(0.3, 0.5, "left", open_palm=True), hand(0.7, 0.5, "right")]
    assert selector.select(candidates, idle=True) == 0


def test_idle_without_gesture_takes_the_preferred_side_then_the_largest():
    selector = HandSelector(preferred="right")
    assert selector.select([hand(0.3, 0.5, "left"), hand(0.7, 0.5, "right")], idle=True) == 1
    any_side = HandSelector(preferred="any")
    candidates = [hand(0.3, 0.5, "left", size=0.10), hand(0.7, 0.5, "right", size=0.20)]
    assert any_side.select(candidates, idle=True) == 1  # la plus grande = la plus proche


def test_no_hand_gives_none():
    assert HandSelector().select([], idle=True) is None
    assert HandSelector().select([], idle=False) is None


def test_unknown_preference_is_rejected_in_french():
    with pytest.raises(ValueError, match="hand.preferred inconnu"):
        HandSelector(preferred="both")


# --- Une fois actif : verrou par continuité ----------------------------------------------------


def test_locked_hand_is_followed_when_two_hands_cross():
    """Main droite (suivie) de gauche à droite, main gauche en sens inverse, 20 % plus bas.
    L'ordre de la liste s'inverse à chaque image : seul le suivi par continuité tient."""
    selector = HandSelector(preferred="right", max_jump=1.5)
    followed_x = 0.20
    other_x = 0.80
    first = [hand(followed_x, 0.40, "right", open_palm=True), hand(other_x, 0.60, "left")]
    assert selector.select(first, idle=True) == 0
    for frame in range(1, 31):
        followed_x += 0.02  # 0,13 taille de main par image
        other_x -= 0.02
        followed = hand(followed_x, 0.40, "right")
        other = hand(other_x, 0.60, "left", open_palm=True)  # l'autre « fait le geste » : ignoré
        if frame % 2 == 0:
            candidates = [followed, other]
        else:
            candidates = [other, followed]
        choice = selector.select(candidates, idle=False)
        assert candidates[choice] is followed, f"image {frame} : mauvaise main suivie"


def test_jump_beyond_max_jump_means_the_followed_hand_is_lost():
    """La main suivie sort du champ ; il ne reste que l'autre, loin : on ne bascule pas."""
    selector = HandSelector(preferred="right", max_jump=1.5)
    selector.select([hand(0.30, 0.50, "right", open_palm=True)], idle=True)
    selector.select([hand(0.31, 0.50, "right")], idle=False)
    assert selector.select([hand(0.80, 0.50, "left")], idle=False) is None
    # La main suivie revient près de sa dernière position : elle est retrouvée.
    candidates = [hand(0.80, 0.50, "left"), hand(0.33, 0.50, "right")]
    assert selector.select(candidates, idle=False) == 1


def test_lock_is_released_when_back_to_idle():
    selector = HandSelector(preferred="right", max_jump=1.5)
    selector.select([hand(0.30, 0.50, "right", open_palm=True)], idle=True)
    selector.select([hand(0.30, 0.50, "right")], idle=False)
    # Retour en veille : on choisit de nouveau d'après le geste, même loin de l'ancienne main.
    candidates = [hand(0.30, 0.50, "right"), hand(0.80, 0.50, "left", open_palm=True)]
    assert selector.select(candidates, idle=True) == 1


# --- Latéralité : étiquette MediaPipe corrigée -------------------------------------------------


def test_mediapipe_label_is_inverted_for_the_user():
    """Mesuré sur 2 sessions main droite : MediaPipe répond « Left » (voir journal)."""
    assert user_side("Left") == "right"
    assert user_side("Right") == "left"
    assert user_side("?") == "unknown"
    assert Hand([(0.0, 0.0, 0.0)] * 21, "Left", 0.9).side == "right"
