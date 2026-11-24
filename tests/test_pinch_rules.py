"""Tests des règles candidates de détection du pincement et de la logique d'entrée / sortie."""

import pytest

from airdesk.interpretation.pinch_rules import (
    RULES,
    PinchHysteresis,
    RuleThresholds,
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
