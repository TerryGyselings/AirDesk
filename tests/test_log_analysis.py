"""Tests de l'analyse des logs, sur un petit CSV écrit à la main (tests/data/sample_log.csv).

Contenu du log d'exemple (une image toutes les 50 ms, de 0 à 1250 ms, soit 26 images) :
- 50-100 ms : pinch_ratio à 0,30 puis 0,35, entre enter (0,25) et exit (0,40) : PAS un pincement ;
- 150-300 ms : pincement court, relâché à 300 ms (0,45 ≥ exit) -> 150 ms, clic ;
- 400-1000 ms : pincement long, Press à 750 ms, relâché à 1000 ms -> 600 ms, glisser ;
- 1100 ms : pincement, puis la main disparaît à 1200 ms -> 100 ms, aucune issue.
Temps de traitement : 10 ms partout sauf 12, 11, 30 et 40 ms.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from airdesk.config import load_config
from airdesk.interpretation.state_machine import Click, Move
from airdesk.log_analysis import (
    END_HAND_LOST,
    END_LOG,
    END_RELEASED,
    OUTCOME_CLICK,
    OUTCOME_DRAG,
    OUTCOME_NONE,
    Pinch,
    Row,
    block_reason,
    estimated_latencies,
    latencies,
    mean_and_p95,
    percentile,
    pinch_durations,
    pinch_thresholds,
    press_episodes,
    read_log,
    read_metadata,
    still_periods,
    summarize,
    text_histogram,
    timing_by_mode,
    wrist_speed_quartiles,
)
from airdesk.session_log import COLUMNS, SessionLog, build_metadata, event_names

DATA = Path(__file__).parent / "data"
SAMPLE_CSV = DATA / "sample_log.csv"
SAMPLE_JSON = DATA / "sample_log.json"


@pytest.fixture
def rows():
    return read_log(SAMPLE_CSV)


# --- Lecture ----------------------------------------------------------------------------------


def test_read_log_converts_types(rows):
    assert len(rows) == 26
    first = rows[0]
    assert first.timestamp_ms == 0
    assert first.hand_present is True
    assert first.pinch_ratio == 0.8
    assert first.extended_fingers == 4
    assert first.events == []
    assert rows[20].events == ["move", "release"]
    no_hand = rows[-1]
    assert no_hand.hand_present is False
    assert no_hand.pinch_ratio is None
    assert no_hand.extended_fingers is None


def test_thresholds_come_from_the_session_json():
    enter, exit_, click_max_ms = pinch_thresholds(read_metadata(SAMPLE_JSON))
    assert (enter, exit_, click_max_ms) == (0.25, 0.4, 300)


# --- Résumé -----------------------------------------------------------------------------------


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 95) == 10  # rang ceil(9,5) = 10
    assert percentile(values, 50) == 5  # rang ceil(5) = 5
    assert percentile([7], 95) == 7


def test_summary_of_the_sample(rows):
    s = summarize(rows)
    assert s.frames == 26
    assert s.duration_s == 1.25
    assert s.fps_mean == pytest.approx(20.0)  # 25 intervalles en 1,25 s
    # (22 × 10 + 12 + 11 + 30 + 40) / 26
    assert s.processing_mean_ms == pytest.approx(313 / 26)
    # 26 valeurs : rang ceil(0,95 × 26) = 25, l'avant-dernière une fois triées
    assert s.processing_p95_ms == 30.0
    assert s.clicks == 1
    assert s.drags == 1


# --- Pincements -------------------------------------------------------------------------------


def test_pinch_durations_use_the_hysteresis(rows):
    pinches = pinch_durations(rows, enter=0.25, exit=0.40)
    found = []
    for p in pinches:
        found.append((p.start_ms, p.duration_ms, p.outcome, p.ended_by))
    assert found == [
        (150, 150, OUTCOME_CLICK, END_RELEASED),
        (400, 600, OUTCOME_DRAG, END_RELEASED),
        (1100, 100, OUTCOME_NONE, END_HAND_LOST),
    ]


def test_pinch_durations_do_not_depend_on_recorded_states(rows):
    """On efface les états : les durées doivent rester les mêmes."""
    without_states = []
    for row in rows:
        without_states.append(
            Row(row.timestamp_ms, row.hand_present, row.pinch_ratio, row.extended_fingers,
                "", row.events, row.processing_ms)
        )
    durations = []
    for p in pinch_durations(without_states, 0.25, 0.40):
        durations.append(p.duration_ms)
    assert durations == [150, 600, 100]


def test_stricter_enter_threshold_misses_the_light_pinch(rows):
    """Avec enter = 0,18, le pincement court (minimum 0,20) n'est plus vu."""
    starts = []
    for p in pinch_durations(rows, enter=0.18, exit=0.40):
        starts.append(p.start_ms)
    assert starts == [400, 1100]


def test_pinch_still_open_at_the_end_of_the_log():
    rows = [
        Row(0, True, 0.8, 4, "actif", [], 10.0),
        Row(50, True, 0.1, 1, "pincement", [], 10.0),
        Row(100, True, 0.1, 1, "pincement", [], 10.0),
    ]
    pinches = pinch_durations(rows, 0.25, 0.40)
    assert len(pinches) == 1
    assert pinches[0].duration_ms == 50
    assert pinches[0].ended_by == END_LOG


# --- Histogramme ------------------------------------------------------------------------------


def test_text_histogram_bins_and_marker():
    lines = text_histogram([150, 600, 100], bin_ms=50, marker=300, marker_label="click_max_ms")
    assert len(lines) == 13  # tranches 0-50 ... 600-650
    assert lines[2] == "  100-150   ms | # 1"
    assert lines[3] == "  150-200   ms | # 1"
    assert lines[6].endswith("<- click_max_ms")  # tranche 300-350
    assert lines[12] == "  600-650   ms | # 1"
    assert lines[0] == "    0-50    ms |"


def test_text_histogram_scales_long_bars():
    lines = text_histogram([10] * 100, bin_ms=50, bar_width=20)
    assert lines[0].count("#") == 20
    assert lines[0].endswith("100")


def test_text_histogram_without_values():
    assert text_histogram([]) == ["(aucune valeur)"]


# --- Écriture (SessionLog) puis relecture -----------------------------------------------------


@dataclass
class FakeFeatures:
    pinch_ratio: float
    extended_fingers: int
    finger_extensions: tuple = (1.5, 1.5, 1.5, 1.5)
    folded_fingers: int = 0
    finger_extensions_3d: tuple = (1.5, 1.5, 1.5, 1.5)
    folded_fingers_3d: int = 0
    pinch_seg: float = 0.7
    pinch_seg_68: float = 0.6
    palm_width: float = 0.8
    palm_facing: float = 1.0
    pinch_ratio_3d: float = 0.8


def test_event_names():
    assert event_names([Move(0.1, 0.0), Click()]) == "move+click"
    assert event_names([]) == ""


def row_at(t, speed, cursor, hand=True):
    return Row(t, hand, 0.8 if hand else None, 4 if hand else None, "actif", [], 10.0,
               wrist_speed=speed, cursor=cursor)


def still_rows(cursor_x_at, duration_s=3.0, fps=30):
    """Main immobile pendant `duration_s` ; position x du curseur donnée par cursor_x_at(t_s)."""
    rows = []
    n = int(duration_s * fps)
    for i in range(n):
        t_ms = round(i * 1000 / fps)
        rows.append(row_at(t_ms, 0.1, (round(cursor_x_at(t_ms / 1000)), 500)))
    return rows


def test_still_periods_keep_only_long_still_runs():
    """1,5 s immobile, un geste rapide, puis 0,5 s immobile (trop court pour compter)."""
    rows = []
    t = 0
    for i in range(46):  # 0 à 1 485 ms
        rows.append(row_at(t, 0.1, (1000 + (2 if i % 2 else -2), 500)))
        t += 33
    for _ in range(5):
        rows.append(row_at(t, 3.0, (1200, 600)))  # main qui bouge vite
        t += 33
    for _ in range(15):  # 0,5 s seulement
        rows.append(row_at(t, 0.1, (1200, 600)))
        t += 33
    periods = still_periods(rows)
    assert len(periods) == 1
    p = periods[0]
    assert p.frames == 46
    assert p.drift_x == pytest.approx(2.0)  # écart-type brut de ±2 px
    assert p.drift_y == 0.0
    assert p.jitter(300) == pytest.approx(2.0, abs=0.3)  # bruit rapide : vu par les deux
    assert p.jitter(1000) == pytest.approx(2.0, abs=0.3)


def test_slow_sine_tremor_is_measured_with_the_one_second_window():
    """Tremblement de ±3 px à 1 Hz (lent, comme après le filtre 1€) : écart-type 3/√2 ≈ 2,1 px.
    La fenêtre de 1 s le mesure ; celle de 300 ms le prend en partie pour de la dérive."""
    rows = still_rows(lambda t: 1000 + 3 * math.sin(2 * math.pi * 1.0 * t))
    p = still_periods(rows)[0]
    expected = 3 / math.sqrt(2)
    assert p.jitter(1000) == pytest.approx(expected, rel=0.15)
    assert p.jitter(300) < 0.5 * expected  # mesure optimiste avec la fenêtre courte


def test_slow_drift_is_not_counted_as_tremor():
    """Dérive régulière de 20 px/s, sans bruit : grande dérive, tremblement quasi nul."""
    rows = still_rows(lambda t: 1000 + 20 * t)
    p = still_periods(rows)[0]
    assert p.drift > 10
    assert p.jitter(300) < 0.5
    assert p.jitter(1000) < 0.5


def test_short_period_gives_no_value_for_the_long_window():
    rows = still_rows(lambda t: 1000, duration_s=1.2)  # 36 images : 6 résidus sur 1 s
    p = still_periods(rows)[0]
    assert p.jitter(1000) is None
    assert p.jitter(300) is not None


def test_wrist_speed_quartiles():
    rows = []
    for i, speed in enumerate([0.1, 0.2, 0.3, 0.4]):
        rows.append(row_at(i * 33, speed, None))
    assert wrist_speed_quartiles(rows) == (0.1, pytest.approx(0.25), 0.3)
    assert wrist_speed_quartiles([row_at(0, None, None)]) is None


def test_click_after_the_release_grace_is_attributed_to_the_pinch():
    """Avec release_grace_ms = 66, le clic est émis 2 images APRÈS la fin du pincement
    mesurée sur pinch_ratio : sans fenêtre, l'issue serait « aucune »."""
    rows = [
        Row(0, True, 0.8, 4, "actif", [], 10.0),
        Row(33, True, 0.1, 1, "pincement", [], 10.0),
        Row(66, True, 0.1, 1, "pincement", [], 10.0),
        Row(99, True, 0.6, 4, "pincement", [], 10.0),  # fin mesurée (pinch_ratio >= exit)
        Row(132, True, 0.6, 4, "pincement", [], 10.0),
        Row(165, True, 0.6, 4, "actif", ["click"], 10.0),  # clic confirmé 66 ms plus tard
        Row(198, True, 0.8, 4, "actif", [], 10.0),
    ]
    without = pinch_durations(rows, 0.25, 0.40)
    assert without[0].outcome == OUTCOME_NONE  # le bug constaté
    with_window = pinch_durations(rows, 0.25, 0.40, outcome_extra_ms=66 + 33)
    assert with_window[0].outcome == OUTCOME_CLICK
    assert with_window[0].duration_ms == 66  # la durée, elle, ne change pas


def test_outcome_window_stops_at_the_next_pinch():
    rows = [
        Row(0, True, 0.1, 1, "pincement", [], 10.0),
        Row(33, True, 0.6, 4, "pincement", [], 10.0),  # fin du 1er
        Row(66, True, 0.1, 1, "pincement", ["click"], 10.0),  # début du 2e (avec un clic)
        Row(99, True, 0.6, 4, "actif", [], 10.0),
    ]
    pinches = pinch_durations(rows, 0.25, 0.40, outcome_extra_ms=200)
    assert pinches[0].outcome == OUTCOME_NONE  # le clic appartient au 2e pincement
    assert pinches[1].outcome == OUTCOME_CLICK


def test_still_periods_need_a_hand_and_a_cursor():
    rows = []
    for i in range(60):
        rows.append(row_at(i * 33, 0.1, None))  # pointage relatif : pas de curseur absolu
    assert still_periods(rows) == []


def test_old_logs_without_new_columns_are_still_readable(tmp_path):
    path = tmp_path / "ancien.csv"
    path.write_text(
        "timestamp_ms,hand_present,pinch_ratio,extended_fingers,state,events,processing_ms\n"
        "0,1,0.8,4,actif,,10.0\n",
        encoding="utf-8",
    )
    row = read_log(path)[0]
    assert row.wrist_speed is None
    assert row.cursor is None


def test_session_log_round_trip(tmp_path):
    cfg = load_config(tmp_path / "absent.yaml", local_path=None)
    camera = {"index": 1, "backend": "dshow", "width": 640, "height": 480}
    metadata = build_metadata(cfg, camera, git={"commit": None, "dirty": None})
    log = SessionLog(metadata, folder=tmp_path, now=datetime(2026, 11, 24, 15, 30, 5))

    with log:
        log.write_frame(0, FakeFeatures(0.8123456, 4), "actif", [], 9.876)
        log.write_frame(33, None, "veille", [Move(0.1, 0.2), Click()], 5.0)

    assert log.csv_path.name == "2026-11-24_15-30-05.csv"
    assert log.json_path.name == "2026-11-24_15-30-05.json"

    rows = read_log(log.csv_path)
    assert rows[0].pinch_ratio == 0.8123  # 4 décimales
    assert rows[0].processing_ms == 9.88
    assert rows[1].hand_present is False
    assert rows[1].events == ["move", "click"]

    saved = read_metadata(log.json_path)
    assert saved["config"]["pinch"]["click_max_ms"] == 300
    assert saved["config"]["camera"]["index"] == 0
    assert saved["camera"] == camera
    assert saved["git"] == {"commit": None, "dirty": None}
    assert "platform" in saved["system"]
    assert "airdesk_version" in saved
    assert saved["columns"] == COLUMNS
    assert pinch_thresholds(saved) == (0.25, 0.40, 300)


def test_block_reason_of_a_refused_pinch():
    """Pincement sans issue : la raison la plus fréquente pendant le pincement."""
    rows = [
        Row(0, True, 0.6, 4, "actif", [], 10.0, entry_block="ratio"),
        Row(33, True, 0.2, 4, "actif", [], 10.0, entry_block="paume"),
        Row(66, True, 0.1, 4, "actif", [], 10.0, entry_block="paume"),
        Row(99, True, 0.1, 4, "actif", [], 10.0, entry_block="autres_doigts"),
        Row(132, True, 0.6, 4, "actif", [], 10.0, entry_block="ratio"),
    ]
    assert block_reason(rows, 33, 99) == ("paume", 2, 3)


def test_block_reason_is_none_for_old_logs():
    rows = [Row(0, True, 0.1, 4, "actif", [], 10.0), Row(33, True, 0.1, 4, "actif", [], 10.0)]
    assert block_reason(rows, 0, 33) is None


def test_timing_by_mode():
    rows = []
    for i in range(20):
        rows.append(Row(i * 33, True, 0.8, 4, "veille", [], 20.0 if i < 10 else 30.0,
                        tracker_hands=2))
    for i in range(20, 40):
        rows.append(Row(i * 33, True, 0.8, 4, "actif", [], 9.0, tracker_hands=1))
    modes = timing_by_mode(rows)
    assert [m.hands for m in modes] == [1, 2]
    one, two = modes
    assert (one.frames, one.mean_ms, one.slow_share) == (20, 9.0, 0.0)
    assert (two.frames, two.mean_ms, two.p95_ms, two.max_ms) == (20, 25.0, 30.0, 30.0)
    assert two.slow_share == 0.5  # 10 images sur 20 au-dessus de 25 ms
    assert timing_by_mode([Row(0, True, 0.8, 4, "actif", [], 9.0)]) == []


def test_new_log_columns_round_trip(tmp_path):
    cfg = load_config(tmp_path / "absent.yaml", local_path=None)
    log = SessionLog(build_metadata(cfg, {}, git={"commit": None, "dirty": None}),
                     folder=tmp_path, now=datetime(2026, 11, 24, 18, 0, 0))
    with log:
        log.write_frame(0, FakeFeatures(0.1, 4), "actif", [], 9.0, closing_speed=1.5,
                        closing_speed_3d=1.2, entry_block="paume", tracker_hands=1)
    row = read_log(log.csv_path)[0]
    assert row.entry_block == "paume"
    assert row.tracker_hands == 1


# --- Mode press : latence, déplacement pendant les clics, saut au début des glisser -----------


def press_row(t, state, events=(), contact=None, distance=None, cursor=(1000, 500)):
    return Row(t, True, 0.1, 3, state, list(events), 9.0, cursor=cursor, contact_ms=contact,
               anchor_distance=distance)


def press_log():
    """Un clic sur place (déplacement max 0,08) puis un glisser (saut de 150 px)."""
    return [
        press_row(0, "pincement"),
        press_row(33, "pincement", ["press"], contact=0, cursor=(1000, 500)),
        press_row(66, "pincement", distance=0.05),
        press_row(99, "pincement", distance=0.08),
        press_row(132, "actif", ["release"]),
        press_row(500, "pincement"),
        press_row(533, "pincement", ["press"], contact=500, cursor=(1200, 500)),
        press_row(566, "pincement", distance=0.10, cursor=(1250, 500)),
        press_row(599, "glisser", distance=0.30, cursor=(1350, 500)),
        press_row(632, "glisser", ["moveto"], distance=0.40, cursor=(1400, 500)),
        press_row(665, "actif", ["release"]),
    ]


def test_latency_from_the_contact_column():
    found = latencies(press_log())
    assert [x.latency_ms for x in found] == [33, 33]
    assert mean_and_p95([33, 33, 400]) == (pytest.approx(155.33, abs=0.01), 400)


def test_press_episodes_measure_movement_and_jump():
    clicks, drag = press_episodes(press_log())
    assert not clicks.dragged
    assert clicks.max_anchor_distance == 0.08
    assert drag.dragged
    assert drag.jump_px == 150  # curseur figé (1200, 500) -> main au passage en glisser
    assert drag.max_anchor_distance == 0.40


def test_press_mode_clicks_and_drags_are_counted():
    s = summarize(press_log())
    assert (s.clicks, s.drags) == (1, 1)


def test_press_mode_outcome_is_a_click_without_drag():
    rows = []
    ratios = [0.1, 0.1, 0.1, 0.8]
    states = ["pincement", "pincement", "pincement", "actif"]
    events = [[], ["press"], [], ["release"]]
    for i in range(4):
        rows.append(Row(i * 33, True, ratios[i], 3, states[i], events[i], 9.0))
    pinch = pinch_durations(rows, 0.25, 0.40)[0]
    assert pinch.outcome == OUTCOME_CLICK


def test_estimated_latency_for_old_logs():
    rows = [
        Row(0, True, 0.8, 4, "actif", [], 9.0),
        Row(33, True, 0.1, 3, "pincement", [], 9.0),
        Row(333, True, 0.8, 3, "actif", [], 9.0),
        Row(400, True, 0.8, 3, "actif", ["click"], 9.0),
    ]
    found = estimated_latencies(rows, [Pinch(33, 333, OUTCOME_CLICK, END_RELEASED)])
    assert [x.latency_ms for x in found] == [367]

