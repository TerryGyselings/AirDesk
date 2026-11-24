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

from dataclasses import replace
from pathlib import Path

import pytest

from airdesk.gesture_analysis import (
    END_HAND_LOST,
    END_RELEASED,
    END_SEQUENCE,
    EPISODE_LABELS,
    EXT3D_COLUMNS,
    EXT_COLUMNS,
    FALSE_EPISODE_LABELS,
    NEGATIVE_LABELS,
    OUT_OF_VOCABULARY_LABELS,
    POSITIVE_LABEL,
    PROFILE_LABEL,
    EpisodeMetrics,
    EpisodeSettings,
    Sample,
    SweepRow,
    best_separation,
    confusion,
    count_episodes,
    count_sequences,
    describe,
    detection_rate,
    episode_metrics,
    frame_samples,
    profile_separations,
    propose_hand_size_ratio,
    propose_threshold,
    propose_thresholds,
    rank,
    read_dataset,
    recommend,
    rejected_by_guard,
    ruptures_in,
    simulate_sequence,
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


# --- Règle D et geste hors vocabulaire ---------------------------------------------------------


def test_rule_d_keeps_every_pinch_below_enter_and_rejects_fist_and_parasites(train):
    """P1, P2, P3 ont les autres doigts ouverts (P3 a l'index court) : tous acceptés."""
    for rule in ("D", "D3D"):
        c = confusion(train, rule, T)
        assert (c.true_positives, c.false_negatives) == (3, 1)
        assert c.false_positives["poing"] == 0
        assert c.false_positives["parasites"] == 0


def test_rest_hand_in_profile_is_a_negative_label():
    assert "main_repos_profil" in NEGATIVE_LABELS
    assert "main_repos_profil" in FALSE_EPISODE_LABELS


def test_palm_measures_are_computed_from_landmarks(tmp_path):
    """Paume de face, points normalisés : base index (-0.3, -1) et auriculaire (0.5, -1)."""
    points = [(0.0, 0.0, 0.0)] * 21
    points[9] = (0.0, -1.0, 0.0)
    points[5] = (-0.3, -1.0, 0.0)
    points[17] = (0.5, -1.0, 0.0)
    header = ["label", "t_in_sequence_ms", "hand_present", "pinch_ratio", "hand_size"]
    header += EXT_COLUMNS + EXT3D_COLUMNS
    values = ["main_repos_profil", "0", "1", "0.1", "0.2"] + ["1.5"] * 8
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
        values += [str(points[i][0]), str(points[i][1]), str(points[i][2])]
    path = tmp_path / "paume.csv"
    path.write_text(",".join(header) + "\n" + ",".join(values) + "\n", encoding="utf-8")
    sample = read_dataset(path)[0]
    assert sample.palm_width == pytest.approx(0.8)
    assert sample.palm_facing == pytest.approx(1.0)


def palm_frame(label, i, palm):
    s = frame(label, i, 0.1)
    s.palm_width = palm
    return s


def test_min_palm_width_is_proposed_from_profile_rest_hands():
    """Pincements paume de face (0.7-0.8) contre mains au repos de profil (0.10-0.20)."""
    samples = []
    for i in range(30, 60):
        samples.append(palm_frame(POSITIVE_LABEL, i, 0.7 + (i % 3) * 0.05))
    for i in range(30):
        samples.append(palm_frame("main_repos_profil", i, 0.10 + (i % 3) * 0.05))
    p = propose_thresholds(samples, enter=0.25)["min_palm_width"]
    assert p.p5_positives == pytest.approx(0.7)
    assert p.p95_negatives == pytest.approx(0.2)
    assert p.threshold == pytest.approx(0.45)
    assert p.margin == pytest.approx(0.5)
    learned = thresholds_from_proposals({"min_palm_width": p}, replace(T, min_palm_width=0.4))
    assert learned.min_palm_width == pytest.approx(0.45)


def test_palm_proposal_without_separation_keeps_the_provisional_value():
    overlapping = propose_threshold([0.3, 0.5, 0.7], [0.4, 0.6, 0.8])
    assert overlapping.margin < 0
    current = replace(T, min_palm_width=0.40)
    learned = thresholds_from_proposals({"min_palm_width": overlapping}, current)
    assert learned.min_palm_width == 0.40


def test_best_separation_finds_the_right_direction():
    """palm_width : pincement plus LARGE ; pinch_ratio : pincement plus PETIT."""
    wide = best_separation("palm_width", [0.6, 0.7, 0.8], [0.1, 0.2, 0.3])
    assert wide.higher_is_pinch
    assert wide.proposal.threshold == pytest.approx(0.45)
    assert wide.proposal.margin == pytest.approx(0.3)
    assert wide.proposal.youden_j == 1.0
    small = best_separation("pinch_ratio", [0.05, 0.10, 0.15], [0.40, 0.50, 0.60])
    assert not small.higher_is_pinch
    assert small.proposal.threshold == pytest.approx(0.275)  # entre 0.15 et 0.40
    assert small.proposal.margin == pytest.approx(0.25)
    assert small.proposal.youden_threshold == pytest.approx(0.275)
    assert small.proposal.youden_j == 1.0


def test_profile_comparison_needs_the_label(train):
    """Anciens fichiers (protocole v1) : pas de main au repos de profil, pas de comparaison."""
    assert profile_separations(train, POSITIVE_LABEL) == []


def test_profile_comparison_ranks_descriptors():
    samples = []
    for i in range(30, 60):  # pincement maintenu : paume large, poignet haut
        s = palm_frame(POSITIVE_LABEL, i, 0.7)
        s.wrist_y = 0.5
        samples.append(s)
    for i in range(30):  # main au repos de profil : paume étroite, poignet au même niveau
        s = palm_frame(PROFILE_LABEL, i, 0.15)
        s.wrist_y = 0.5 + (i % 2) * 0.01
        samples.append(s)
    ranking = profile_separations(samples, POSITIVE_LABEL)
    assert ranking[0].descriptor == "palm_width"  # sépare parfaitement : en tête
    by_name = {}
    for sep in ranking:
        by_name[sep.descriptor] = sep
    assert by_name["wrist_y"].proposal.youden_j < 1.0


def test_toward_camera_is_out_of_vocabulary():
    assert "pincement_vers_camera" in OUT_OF_VOCABULARY_LABELS
    assert "pincement_vers_camera" not in EPISODE_LABELS
    assert POSITIVE_LABEL != "pincement_vers_camera"


# --- Simulation par épisode (séquences construites ici, une image toutes les 33 ms) -----------

STEP = 33
OPEN_OTHERS = (1.4, 1.5, 1.5, 1.5)  # autres doigts ouverts : D accepte
FIST = (0.4, 0.4, 0.4, 0.4)  # tous repliés : D refuse


def frame(label, i, pinch_ratio, extensions=OPEN_OTHERS, hand_size=0.2):
    """Image n° i (t = 33 x i). pinch_ratio None = pas de main."""
    if pinch_ratio is None:
        return Sample(label, i * STEP, None, None, None)
    return Sample(label, i * STEP, pinch_ratio, extensions, extensions, hand_size)


def held_pinch_with_glitch():
    """Pincement maintenu 3 s ; à l'image 46 (1 518 ms), le suivi décroche une image :
    pinch_ratio 1.2 et taille de main divisée par 4 (main vue de profil)."""
    frames = []
    for i in range(92):
        if i == 46:
            frames.append(frame(POSITIVE_LABEL, i, 1.2, hand_size=0.05))
        else:
            frames.append(frame(POSITIVE_LABEL, i, 0.1))
    return frames


def settings(rule="A", exit=0.40, grace=0, guard=0.0):
    return EpisodeSettings(rule, T, exit, release_grace_ms=grace, guard_ratio=guard)


def test_one_glitch_frame_breaks_a_held_pinch_without_grace():
    episodes = simulate_sequence(held_pinch_with_glitch(), settings())
    assert episodes[0].end_ms == 1518
    assert episodes[0].ended_by == END_RELEASED
    assert episodes[1].ended_by == END_SEQUENCE
    assert ruptures_in(held_pinch_with_glitch(), settings()) == 1


def test_release_grace_absorbs_the_glitch():
    assert ruptures_in(held_pinch_with_glitch(), settings(grace=66)) == 0


def test_hand_size_guard_absorbs_the_glitch():
    """L'image décrochée est écartée (comme une main absente) : le pincement tient."""
    assert ruptures_in(held_pinch_with_glitch(), settings(guard=0.5)) == 0


def test_pinch_never_detected_counts_as_a_failure():
    frames = []
    for i in range(92):
        frames.append(frame(POSITIVE_LABEL, i, 0.6))
    assert ruptures_in(frames, settings()) == 1


def hand_lost_during_pinch(missing_frames):
    """Pincement jusqu'à l'image 45 (1 485 ms), puis main absente, puis pincement à nouveau."""
    frames = []
    for i in range(46):
        frames.append(frame(POSITIVE_LABEL, i, 0.1))
    for i in range(46, 46 + missing_frames):
        frames.append(frame(POSITIVE_LABEL, i, None))
    for i in range(46 + missing_frames, 46 + missing_frames + 30):
        frames.append(frame(POSITIVE_LABEL, i, 0.1))
    return frames


def test_short_hand_loss_keeps_the_pinch():
    """Comme la machine à états : une absence de 500 ms ne relâche pas le pincement."""
    frames = hand_lost_during_pinch(15)
    episodes = simulate_sequence(frames, settings())
    assert len(episodes) == 1
    assert ruptures_in(frames, settings()) == 0


def test_long_hand_loss_ends_the_pinch():
    """Absence de plus de lost_timeout_ms (1 000 ms) : fin, datée de la dernière image vue."""
    frames = hand_lost_during_pinch(34)
    episodes = simulate_sequence(frames, settings())
    assert episodes[0].ended_by == END_HAND_LOST
    assert episodes[0].end_ms == 45 * STEP
    assert ruptures_in(frames, settings()) == 1


def fist_with_brief_thumb_contact():
    """Poing 2 s ; pouce sur l'index replié de l'image 15 à 21 (~230 ms)."""
    frames = []
    for i in range(61):
        ratio = 0.1 if 15 <= i <= 21 else 0.8
        frames.append(frame("poing", i, ratio, extensions=FIST))
    return frames


def repeated_pinches():
    """Trois pincements de 200 ms, séparés par des relâchements francs."""
    frames = []
    for i in range(90):
        in_pinch = i % 30 < 6
        frames.append(frame("pincements_repetes", i, 0.1 if in_pinch else 0.8))
    return frames


def test_false_click_on_a_fist_with_rule_a_but_not_d():
    samples = fist_with_brief_thumb_contact() + repeated_pinches()
    with_a = episode_metrics(samples, settings("A"), click_max_ms=400, seconds=10)
    assert with_a.false_clicks == 1
    assert with_a.false_drags == 0
    assert with_a.false_durations == [231]  # de 495 ms (image 15) à 726 ms (image 22)
    with_d = episode_metrics(samples, settings("D"), click_max_ms=400, seconds=10)
    assert with_d.false_clicks == 0


def test_repeated_pinches_are_counted_against_the_expected_number():
    m = episode_metrics(repeated_pinches(), settings("D"), click_max_ms=400, seconds=10)
    assert (m.repeated_detected, m.repeated_expected) == (3, 10)


def test_count_sequences_and_episodes_still_work(train):
    assert count_sequences(train, "pincements_repetes") == 1
    assert count_episodes(train, "pincements_repetes", "D", T, EXIT) == 3


# --- Garde-fou de taille de main ---------------------------------------------------------------


def held_pinch_with_four_glitches():
    frames = []
    for i in range(92):
        if i in (40, 50, 60, 70):
            frames.append(frame(POSITIVE_LABEL, i, 1.2, hand_size=0.05))
        else:
            frames.append(frame(POSITIVE_LABEL, i, 0.1))
    return frames


def test_hand_size_ratio_is_proposed_between_healthy_and_glitch_frames():
    """Saines : rapport 1.0 ; décrochées : 0.05 / 0.2 = 0.25 -> seuil au milieu, 0.625."""
    p = propose_hand_size_ratio(held_pinch_with_four_glitches(), exit=0.40, window_ms=2000)
    assert p.p5_positives == pytest.approx(1.0)
    assert p.p95_negatives == pytest.approx(0.25)
    assert p.threshold == pytest.approx(0.625)
    assert p.margin == pytest.approx(0.75)
    assert p.n_negatives == 4


def test_rejected_frames_are_counted_per_label():
    counts = rejected_by_guard(held_pinch_with_four_glitches(), 0.625, 2000)
    assert counts[POSITIVE_LABEL] == (4, 92)


# --- Recommandation automatique ----------------------------------------------------------------


def metrics(ruptures, clicks, drags, repeated):
    return EpisodeMetrics(ruptures, clicks, drags, [], repeated, 10, 0)


def row(rule, exit, grace, train, test=None):
    """Une ligne de balayage ; `test` par défaut : des chiffres catastrophiques, pour vérifier
    qu'ils n'influencent PAS le choix."""
    if test is None:
        test = metrics(9, 9, 9, 0)
    return SweepRow(settings(rule, exit, grace), train, test)


def describe_row(r):
    return (r.settings.rule, r.settings.exit, r.settings.release_grace_ms)


def test_recommend_filters_and_ranks_on_train_only():
    rows = [
        row("A", 0.40, 0, metrics(0, 3, 0, 10)),  # trop de faux clics
        row("D", 0.40, 0, metrics(2, 1, 0, 9)),
        row("D", 0.40, 66, metrics(0, 0, 0, 8)),
        row("D", 0.50, 100, metrics(0, 0, 0, 9)),
        row("D3D", 0.50, 66, metrics(0, 1, 0, 9)),  # même score que la précédente, délai plus court
        row("D3D", 0.40, 0, metrics(0, 0, 1, 10)),  # un faux glisser : exclue
    ]
    assert describe_row(recommend(rows)) == ("D3D", 0.50, 66)


def test_test_figures_never_change_the_choice():
    """Même entraînement, test excellent pour l'une et catastrophique pour l'autre : le choix
    ne bouge pas (sinon on réglerait les paramètres sur le test)."""
    good_train = metrics(0, 0, 0, 10)
    rows = [
        row("D", 0.40, 0, good_train, test=metrics(9, 9, 9, 0)),
        row("D", 0.50, 66, good_train, test=metrics(0, 0, 0, 10)),
    ]
    assert describe_row(recommend(rows)) == ("D", 0.40, 0)


def test_recommend_returns_none_when_nothing_qualifies():
    rows = [row("A", 0.40, 0, metrics(0, 2, 0, 10)), row("D", 0.40, 0, metrics(0, 0, 1, 10))]
    assert recommend(rows) is None


def test_unreliable_ruptures_are_left_out_of_the_ranking():
    """Ruptures non fiables : on classe par faux clics, puis épisodes répétés détectés."""
    rows = [
        row("A", 0.40, 0, metrics(0, 1, 0, 8)),  # 0 rupture mais 1 faux clic
        row("D", 0.40, 0, metrics(3, 0, 0, 7)),  # 3 ruptures, 0 faux clic, 7 répétés
        row("D", 0.50, 66, metrics(5, 0, 0, 9)),  # 5 ruptures, 0 faux clic, 9 répétés
    ]
    assert describe_row(recommend(rows, ruptures_reliable=True)) == ("A", 0.40, 0)
    assert describe_row(recommend(rows, ruptures_reliable=False)) == ("D", 0.50, 66)


def test_rank_puts_qualifying_rows_first():
    rows = [row("A", 0.40, 0, metrics(0, 3, 0, 10)), row("D", 0.40, 0, metrics(4, 0, 0, 5))]
    ranked = rank(rows)
    assert ranked[0].settings.rule == "D"


# --- Sortie sur repli des doigts et mesure de pincement ----------------------------------------


def fist_entered_by_mistake():
    """Main pincée doigts ouverts une image (entrée de D), puis poing fermé 2 s, pouce contre
    l'index : pinch_ratio reste petit. Sans sortie sur repli, D reste « pincé » jusqu'au bout."""
    frames = [frame("poing", 0, 0.1, extensions=OPEN_OTHERS)]
    for i in range(1, 61):
        frames.append(frame("poing", i, 0.2, extensions=FIST))
    return frames


def test_exit_on_fold_ends_a_fist_entered_by_mistake():
    without = simulate_sequence(fist_entered_by_mistake(), settings("D"))
    assert without[0].ended_by == END_SEQUENCE
    assert without[0].duration_ms == 60 * STEP  # un faux glisser de 2 s
    with_fold = simulate_sequence(
        fist_entered_by_mistake(), replace(settings("D"), exit_on_fold=True)
    )
    assert with_fold[0].ended_by == END_RELEASED
    assert with_fold[0].duration_ms == STEP  # relâché dès la première image de poing


def test_measure_is_used_by_the_simulation():
    """pinch_ratio reste haut (pouce sous le bout de l'index) mais pinch_seg est petit."""
    frames = []
    for i in range(30):
        s = frame("pincements_repetes", i, 0.6)
        s.pinch_seg = 0.1 if 10 <= i < 20 else 0.6
        s.pinch_seg_68 = s.pinch_seg
        frames.append(s)
    assert len(simulate_sequence(frames, settings())) == 0
    seg = replace(settings(), measure="seg78")
    assert len(simulate_sequence(frames, seg)) == 1


def test_missing_measure_gives_a_clear_error(train):
    with pytest.raises(ValueError, match="indisponible"):
        simulate_sequence(train[:3], replace(settings(), measure="seg78"))


def test_read_dataset_computes_pinch_seg_from_landmarks(tmp_path):
    """Pouce (4) posé au milieu de la dernière phalange (7-8) : pinch_seg nul, pinch_ratio non."""
    points = [(0.0, 0.0, 0.0)] * 21
    points[9] = (0.0, -1.0, 0.0)  # base du majeur : taille de main = 1
    points[6] = (0.3, -1.4, 0.0)
    points[7] = (0.3, -1.6, 0.0)
    points[8] = (0.3, -1.8, 0.0)
    points[4] = (0.3, -1.7, 0.0)
    header = ["label", "t_in_sequence_ms", "hand_present", "pinch_ratio", "hand_size"]
    header += EXT_COLUMNS + EXT3D_COLUMNS
    values = ["pincement_maintenu", "1000", "1", "0.5", "0.2"] + ["1.5"] * 8
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
        values += [str(points[i][0]), str(points[i][1]), str(points[i][2])]
    path = tmp_path / "points.csv"
    path.write_text(",".join(header) + "\n" + ",".join(values) + "\n", encoding="utf-8")
    sample = read_dataset(path)[0]
    assert sample.pinch_seg == pytest.approx(0.0)
    assert sample.pinch_seg_68 == pytest.approx(0.0)
    assert sample.pinch_ratio == 0.5
