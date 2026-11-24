"""Tests des règles candidates de détection du pincement (A, B, C, en 2D et 3D)."""

import pytest

from airdesk.interpretation.pinch_rules import RULES, RuleThresholds, pinch_candidate

T = RuleThresholds(enter=0.25, min_index_extension=1.0, folded_below=0.8,
                   min_index_extension_3d=1.0, folded_below_3d=0.8)

TENDUS = (1.5, 1.5, 1.5, 1.5)
POING = (0.4, 0.4, 0.4, 0.4)
INDEX_SEUL = (1.4, 0.4, 0.4, 0.4)  # index tendu, autres doigts repliés (« montrer du doigt »)


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
    assert detected_by(0.1, TENDUS) == ["A", "B", "C", "B3D", "C3D"]


def test_no_rule_fires_when_fingers_are_apart():
    assert detected_by(0.6, TENDUS) == []


def test_fist_only_fools_rule_a():
    """Poing fermé, pouce sur l'index replié : pinch_ratio petit, index replié."""
    assert detected_by(0.1, POING) == ["A"]


def test_rule_c_needs_another_finger_not_folded():
    """Index tendu mais les trois autres doigts repliés : B l'accepte, C le refuse."""
    assert detected_by(0.1, INDEX_SEUL) == ["A", "B", "B3D"]


def test_3d_rules_accept_an_index_pointed_at_the_camera():
    """Index court en 2D (perspective), normal en 3D."""
    short_in_2d = (0.6, 1.5, 1.5, 1.5)
    assert detected_by(0.1, short_in_2d, TENDUS) == ["A", "B3D", "C3D"]


def test_unknown_rule_is_rejected_in_french():
    with pytest.raises(ValueError, match="Règle inconnue"):
        pinch_candidate("D", 0.1, TENDUS, TENDUS, T)
