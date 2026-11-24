"""Tests de l'analyse des jeux de données étiquetés, sur de petits CSV écrits à la main.

tests/data/sample_dataset.csv (entraînement), seuils enter 0.25 / exit 0.40 :
- pincement_maintenu : 2 images de mise en place (t < 1 s, ignorées), puis
  P1 (pince 0.10, index 1.4 / 3D 1.45), P2 (0.12, 1.3 / 1.35),
  P3 (0.15, index 0.9 en 2D mais 1.40 en 3D : perspective), P4 (0.30 : au-dessus de enter) ;
- poing : 4 images détectées (index 0.4 à 0.6) dont 3 sous enter, plus 1 main perdue ;
- parasites : X1 (0.20, index 0.7), X2 (0.60), X3 (0.22, index 1.2, autres doigts repliés) ;
- main_ouverte : 2 images ;
- pincements_repetes : 3 pincements, dont un passage à 0.32 (entre enter et exit) qui ne
  doit pas en compter un de plus ;
- pincement_vers_camera : 2 pincements, index court en 2D (0.7 / 0.6) mais long en 3D.
"""

from pathlib import Path

import pytest

from airdesk.gesture_analysis import (
    POSITIVE_LABEL,
    confusion,
    count_episodes,
    count_sequences,
    describe,
    detection_rate,
    frame_samples,
    propose_threshold,
    propose_thresholds,
    read_dataset,
    stats_by_label,
    thresholds_from_proposals,
    youden,
)
from airdesk.interpretation.pinch_rules import RuleThresholds

DATA = Path(__file__).parent / "data"
T = RuleThresholds(enter=0.25, min_index_extension=1.0, folded_below=0.8,
                   min_index_extension_3d=1.0, folded_below_3d=0.8)
EXIT = 0.40


@pytest.fixture
def train():
    return read_dataset(DATA / "sample_dataset.csv")


@pytest.fixture
def test_set():
    return read_dataset(DATA / "sample_dataset_test.csv")


# --- Lecture et distributions -----------------------------------------------------------------


def test_read_dataset(train):
    assert len(train) == 30
    lost = train[9]
    assert lost.label == "poing"
    assert not lost.hand_present
    assert train[4].extensions == (0.9, 1.5, 1.5, 1.5)
    assert train[4].extensions_3d == (1.40, 1.5, 1.5, 1.5)


def test_settling_second_is_excluded_from_the_positive_label(train):
    positives = frame_samples(train, POSITIVE_LABEL)
    assert len(positives) == 4
    assert positives[0].t_in_sequence_ms == 1000


def test_detection_rate(train):
    assert detection_rate(train, "poing") == pytest.approx(4 / 5)
    assert detection_rate(train, "main_ouverte") == 1.0


def test_stats_of_the_fist(train):
    s = stats_by_label(train)["poing"]["pinch_ratio"]
    assert s.n == 4
    assert s.minimum == 0.15
    assert s.median == pytest.approx(0.19)  # moyenne des deux valeurs centrales 0.18 et 0.20
    assert s.p5 == 0.15  # rang ceil(0.05 x 4) = 1
    assert s.p95 == 0.40  # rang ceil(0.95 x 4) = 4
    assert s.maximum == 0.40


def test_describe_without_values():
    assert describe([]) is None


# --- Matrices de confusion --------------------------------------------------------------------


def false_positives(samples, rule):
    return confusion(samples, rule, T).false_positives


def test_rule_a_confuses_the_fist_with_a_pinch(train):
    c = confusion(train, "A", T)
    assert (c.true_positives, c.false_negatives) == (3, 1)  # P4 est au-dessus de enter
    assert c.false_positives["poing"] == 3
    assert c.negatives["poing"] == 4  # la main perdue n'est pas comptée
    assert c.false_positives["parasites"] == 2
    assert c.false_positives["main_ouverte"] == 0


def test_rule_b_removes_the_fist_but_misses_the_foreshortened_index(train):
    c = confusion(train, "B", T)
    assert c.true_positives == 2  # P3 : index court en 2D
    assert c.false_positives["poing"] == 0
    assert c.false_positives["parasites"] == 1  # X3 : index tendu, autres repliés


def test_rule_c_also_removes_the_pointing_parasite(train):
    c = confusion(train, "C", T)
    assert c.true_positives == 2
    assert c.false_positives["parasites"] == 0


def test_3d_rules_keep_the_foreshortened_pinch(train):
    b3d = confusion(train, "B3D", T)
    c3d = confusion(train, "C3D", T)
    assert b3d.true_positives == 3
    assert b3d.false_positives["poing"] == 0
    assert c3d.true_positives == 3
    assert c3d.false_positives["parasites"] == 0
    assert c3d.recall == pytest.approx(3 / 4)


# --- Épisodes ---------------------------------------------------------------------------------


def test_episodes_use_the_hysteresis(train):
    """0.32 après un relâchement (0.45) est entre enter et exit : pas de nouvel épisode."""
    assert count_episodes(train, "pincements_repetes", "A", T, EXIT) == 3


def test_episodes_toward_the_camera_need_3d(train):
    assert count_episodes(train, "pincement_vers_camera", "A", T, EXIT) == 2
    assert count_episodes(train, "pincement_vers_camera", "B", T, EXIT) == 0
    assert count_episodes(train, "pincement_vers_camera", "B3D", T, EXIT) == 2
    assert count_episodes(train, "pincement_vers_camera", "C3D", T, EXIT) == 2


def test_count_sequences(train):
    assert count_sequences(train, "pincements_repetes") == 1
    assert count_sequences(train + train, "pincements_repetes") == 2  # deux fichiers


# --- Seuils proposés --------------------------------------------------------------------------


def test_youden_picks_the_best_cut():
    threshold, j = youden([1.35, 1.40, 1.45], [0.45, 0.55, 0.65, 0.7, 1.2])
    assert threshold == pytest.approx(1.275)
    assert j == 1.0


def test_propose_threshold_reports_overlap():
    """Index 2D : P5 des pincements 0.9 < P95 des négatifs 1.2 -> marge négative."""
    p = propose_threshold([1.4, 1.3, 0.9], [0.4, 0.5, 0.6, 0.7, 1.2])
    assert p.p5_positives == 0.9
    assert p.p95_negatives == 1.2
    assert p.margin == pytest.approx(-0.3)
    assert p.threshold == pytest.approx(1.05)
    assert p.youden_threshold == pytest.approx(0.8)
    assert p.youden_j == pytest.approx(0.8)


def test_propose_threshold_needs_both_groups():
    assert propose_threshold([1.4], []) is None


def test_proposals_on_the_training_set(train):
    proposals = propose_thresholds(train, enter=0.25)
    index_3d = proposals["min_index_extension_3d"]
    assert index_3d.threshold == pytest.approx(1.275)
    assert index_3d.margin == pytest.approx(0.15)
    assert (index_3d.n_positives, index_3d.n_negatives) == (3, 5)
    assert proposals["min_index_extension"].margin == pytest.approx(-0.3)
    assert proposals["folded_below"].threshold == pytest.approx(1.05)


def test_thresholds_learned_on_train_are_applied_to_test(train, test_set):
    """Sur le test, le poing a un index 3D de 1.29 : le seuil appris (1.275) le laisse passer.
    Recalculé sur le test seul, le seuil aurait été 1.295 : la validation serait faussée."""
    learned = thresholds_from_proposals(propose_thresholds(train, 0.25), T)
    assert learned.min_index_extension_3d == pytest.approx(1.275)
    c = confusion(test_set, "B3D", learned)
    assert c.true_positives == 1
    assert c.false_positives["poing"] == 1
    # La règle C3D, elle, rejette ce poing (autres doigts repliés).
    assert confusion(test_set, "C3D", learned).false_positives["poing"] == 0


def test_missing_proposal_keeps_the_current_threshold():
    learned = thresholds_from_proposals({"min_index_extension": None}, T)
    assert learned.min_index_extension == T.min_index_extension
    assert learned.enter == T.enter
