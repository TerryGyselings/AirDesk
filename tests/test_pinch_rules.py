"""Tests des règles candidates de détection du pincement et de la logique d'entrée / sortie."""

import pytest

from airdesk.interpretation.pinch_rules import (
    BLOCK_CLOSING,
    BLOCK_OTHER_FINGERS,
    BLOCK_PALM,
    BLOCK_RATIO,
    RULES,
    ClosingSpeed,
    PinchHysteresis,
    RuleThresholds,
    entry_block_reason,
    pinch_candidate,
    pinch_measure,
)

T = RuleThresholds(enter=0.25, min_index_extension=1.0, folded_below=0.8,
                   min_index_extension_3d=1.0, folded_below_3d=0.8)

TENDUS = (1.5, 1.5, 1.5, 1.5)
POING = (0.4, 0.4, 0.4, 0.4)
INDEX_SEUL = (1.4, 0.4, 0.4, 0.4)  # index tendu, autres doigts repliés (« montrer du doigt »)
INDEX_PLIE = (0.5, 1.5, 1.5, 1.5)  # pincement réel où l'index se plie, autres doigts ouverts


def detected_by(pinch_ratio, extensions, extensions_3d=None):
    """Liste des règles qui détectent un pincement."""
    if extensions_3d is None:
        extensions_3d = extensions
    rules = []
    for rule in RULES:
        if pinch_candidate(rule, pinch_ratio, extensions, extensions_3d, T):
            rules.append(rule)
    return rules


def test_real_pinch_is_detected_by_every_rule():
    assert detected_by(0.1, TENDUS) == ["A", "B", "C", "D", "B3D", "C3D", "D3D"]


def test_no_rule_fires_when_fingers_are_apart():
    assert detected_by(0.6, TENDUS) == []


def test_fist_only_fools_rule_a():
    """Poing fermé, pouce sur l'index replié : pinch_ratio petit, tous les doigts repliés."""
    assert detected_by(0.1, POING) == ["A"]


def test_rule_c_and_d_need_another_finger_not_folded():
    """Index tendu mais les trois autres doigts repliés : B l'accepte, C et D le refusent."""
    assert detected_by(0.1, INDEX_SEUL) == ["A", "B", "B3D"]


def test_rule_d_accepts_a_pinch_with_a_bent_index():
    """Le constat des mesures : l'index se plie pour pincer. B et C le rejettent, pas D."""
    assert detected_by(0.1, INDEX_PLIE) == ["A", "D", "D3D"]


def test_3d_rules_accept_an_index_pointed_at_the_camera():
    """Index court en 2D (perspective), normal en 3D."""
    short_in_2d = (0.6, 1.5, 1.5, 1.5)
    assert detected_by(0.1, short_in_2d, TENDUS) == ["A", "D", "B3D", "C3D", "D3D"]


def test_unknown_rule_is_rejected_in_french():
    with pytest.raises(ValueError, match="Règle inconnue"):
        pinch_candidate("Z", 0.1, TENDUS, TENDUS, T)


# --- Entrée / sortie avec délai de relâchement -------------------------------------------------


def run(hysteresis, ratios, step_ms=33):
    """Envoie une suite de pinch_ratio (une image toutes les 33 ms) ; renvoie les événements."""
    events = []
    for i in range(len(ratios)):
        event = hysteresis.update(i * step_ms, ratios[i], TENDUS, TENDUS)
        if event is not None:
            events.append((event, i * step_ms))
    return events


def test_clean_release_without_grace():
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=0)
    assert run(h, [0.8, 0.1, 0.1, 0.5, 0.8]) == [("start", 33), ("end", 99)]


def test_without_grace_one_glitch_frame_breaks_the_pinch():
    """Comportement actuel : une seule image au-dessus de exit lâche le pincement."""
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=0)
    events = run(h, [0.1, 0.1, 1.2, 0.1, 0.1])
    assert events == [("start", 0), ("end", 66), ("start", 99)]


def test_grace_ignores_a_single_glitch_frame():
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=66)
    assert run(h, [0.1, 0.1, 1.2, 0.1, 0.1]) == [("start", 0)]
    assert h.pinched


def test_grace_end_time_is_when_the_release_started():
    """Relâché à 66 ms, confirmé 66 ms plus tard : la fin retenue est 66 ms."""
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=66)
    events = run(h, [0.1, 0.1, 0.6, 0.6, 0.6, 0.6])
    assert events == [("start", 0), ("end", 132)]
    assert h.release_ms == 66


def test_hysteresis_between_enter_and_exit_keeps_the_pinch():
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=0)
    assert run(h, [0.1, 0.3, 0.35, 0.3]) == [("start", 0)]


def run_with_fingers(hysteresis, frames):
    """frames : liste de (pinch_value, extensions) ; une image toutes les 33 ms."""
    events = []
    for i in range(len(frames)):
        value, extensions = frames[i]
        event = hysteresis.update(i * 33, value, extensions, extensions)
        if event is not None:
            events.append((event, i * 33))
    return events


def test_exit_on_fold_releases_when_the_other_fingers_fold():
    """Entré avec les doigts ouverts, puis poing fermé en gardant le pouce sur l'index."""
    frames = [(0.1, TENDUS), (0.1, POING), (0.1, POING)]
    without = PinchHysteresis("D", T, exit=0.40)
    assert run_with_fingers(without, frames) == [("start", 0)]
    with_fold = PinchHysteresis("D", T, exit=0.40, exit_on_fold=True)
    assert run_with_fingers(with_fold, frames) == [("start", 0), ("end", 33)]


def test_exit_on_fold_respects_the_release_grace():
    """Un repli d'une seule image (suivi qui décroche) ne relâche pas avec un délai de 66 ms."""
    frames = [(0.1, TENDUS), (0.1, POING), (0.1, TENDUS), (0.1, TENDUS)]
    h = PinchHysteresis("D", T, exit=0.40, release_grace_ms=66, exit_on_fold=True)
    assert run_with_fingers(h, frames) == [("start", 0)]


def test_exit_on_fold_has_no_effect_on_rule_a():
    """La règle A n'exige pas d'autre doigt ouvert : rien à surveiller."""
    h = PinchHysteresis("A", T, exit=0.40, exit_on_fold=True)
    assert run_with_fingers(h, [(0.1, POING), (0.1, POING)]) == [("start", 0)]


PALM = RuleThresholds(enter=0.25, folded_below=0.8, require_palm_facing=True,
                      min_palm_width=0.40)


def test_palm_condition_rejects_a_hand_in_profile():
    assert pinch_candidate("D", 0.1, TENDUS, TENDUS, PALM, palm_width=0.80)
    assert not pinch_candidate("D", 0.1, TENDUS, TENDUS, PALM, palm_width=0.15)
    assert not pinch_candidate("A", 0.1, TENDUS, TENDUS, PALM, palm_width=0.15)  # toute règle


def test_unknown_palm_width_does_not_block():
    """Ancien jeu de données sans les 21 points : pas de largeur de paume, pas de blocage."""
    assert pinch_candidate("D", 0.1, TENDUS, TENDUS, PALM, palm_width=None)


def test_palm_condition_disabled_has_no_effect():
    assert pinch_candidate("D", 0.1, TENDUS, TENDUS, T, palm_width=0.15)


def test_palm_condition_only_affects_the_entry():
    """Une fois pincé, tourner la main de profil ne relâche pas (seule l'entrée est filtrée)."""
    h = PinchHysteresis("D", PALM, exit=0.40)
    assert h.update(0, 0.1, TENDUS, TENDUS, palm_width=0.80) == "start"
    assert h.update(33, 0.1, TENDUS, TENDUS, palm_width=0.15) is None
    assert h.pinched


def test_pinch_measure_selects_the_value():
    class Features:
        pinch_ratio = 0.5
        pinch_seg = 0.2
        pinch_seg_68 = 0.1

    assert pinch_measure(Features, "ratio") == 0.5
    assert pinch_measure(Features, "seg78") == 0.2
    assert pinch_measure(Features, "seg68") == 0.1
    with pytest.raises(ValueError, match="Mesure de pincement inconnue"):
        pinch_measure(Features, "angle")


# --- Raison du refus d'entrée (colonne entry_block) ---------------------------------------------

REFUS = RuleThresholds(enter=0.25, folded_below=0.8, require_palm_facing=True,
                       min_palm_width=0.40, min_closing_speed=1.0, min_closing_speed_3d=1.0)


def test_entry_block_reason_gives_the_first_missing_condition():
    assert entry_block_reason("D", 0.6, POING, POING, REFUS, 0.1) == BLOCK_RATIO  # avant tout
    assert entry_block_reason("D", 0.1, POING, POING, REFUS, 0.1) == BLOCK_PALM
    assert entry_block_reason("D", 0.1, POING, POING, REFUS, 0.8) == BLOCK_OTHER_FINGERS
    assert entry_block_reason("D", 0.1, TENDUS, TENDUS, REFUS, 0.8) is None


def test_rule_f_needs_the_fingers_to_be_closing():
    assert entry_block_reason("F", 0.1, TENDUS, TENDUS, REFUS, 0.8, closing_speed=2.0) is None
    assert entry_block_reason("F", 0.1, TENDUS, TENDUS, REFUS, 0.8,
                              closing_speed=0.2) == BLOCK_CLOSING
    # Vitesse inconnue (pas assez d'historique) : on ne sait pas si les doigts se ferment.
    assert entry_block_reason("F", 0.1, TENDUS, TENDUS, REFUS, 0.8) == BLOCK_CLOSING
    # F3D regarde la vitesse calculée en 3D.
    assert entry_block_reason("F3D", 0.1, TENDUS, TENDUS, REFUS, 0.8, closing_speed=2.0,
                              closing_speed_3d=0.1) == BLOCK_CLOSING


def test_pinch_candidate_agrees_with_entry_block_reason():
    for rule in RULES:
        for extensions in (TENDUS, POING):
            accepted = pinch_candidate(rule, 0.1, extensions, extensions, T, 0.8, 2.0, 2.0)
            reason = entry_block_reason(rule, 0.1, extensions, extensions, T, 0.8, 2.0, 2.0)
            assert accepted == (reason is None)


def test_hysteresis_reports_why_entry_is_refused():
    h = PinchHysteresis("D", REFUS, exit=0.40)
    h.update(0, 0.1, POING, POING, palm_width=0.8)
    assert h.last_block == BLOCK_OTHER_FINGERS
    h.update(33, 0.1, TENDUS, TENDUS, palm_width=0.8)
    assert h.pinched
    assert h.last_block is None


def test_ratio3d_measure():
    class Features:
        pinch_ratio = 0.1
        pinch_ratio_3d = 0.5
        pinch_seg = 0.1
        pinch_seg_68 = 0.1

    assert pinch_measure(Features, "ratio3d") == 0.5


# --- Vitesse de fermeture -----------------------------------------------------------------------


def test_closing_speed_over_200_ms():
    speed = ClosingSpeed(window_ms=200)
    assert speed.update(0, 0.8) is None  # pas encore 200 ms d'historique
    assert speed.update(100, 0.6) is None
    # De 0,8 à 0,4 en 200 ms : la mesure baisse de 2 par seconde (doigts qui se ferment).
    assert speed.update(200, 0.4) == pytest.approx(2.0)
    # Doigts qui s'ouvrent : vitesse négative.
    assert speed.update(400, 0.6) == pytest.approx(-1.0)


def test_closing_speed_reset():
    speed = ClosingSpeed(window_ms=200)
    speed.update(0, 0.8)
    speed.reset()
    assert speed.update(250, 0.4) is None


# --- Relâchement qui oscille autour d'exit ------------------------------------------------------


def test_release_is_not_cancelled_by_a_dip_just_below_exit():
    """Avec une marge de 0,05, un relâchement commencé n'est annulé que sous 0,35."""
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=66, release_cancel_margin=0.05)
    assert run(h, [0.1, 0.1, 0.41, 0.39, 0.41]) == [("start", 0), ("end", 132)]
    assert h.release_ms == 66


def test_without_margin_a_dip_cancels_the_release():
    """Comportement d'origine : un passage juste sous exit annule la confirmation."""
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=66)
    assert run(h, [0.1, 0.1, 0.41, 0.39, 0.41]) == [("start", 0)]


def test_a_real_re_pinch_still_cancels_the_release():
    h = PinchHysteresis("A", T, exit=0.40, release_grace_ms=66, release_cancel_margin=0.05)
    assert run(h, [0.1, 0.1, 0.41, 0.20, 0.20]) == [("start", 0)]
    assert h.pinched
    assert not h.releasing
