"""Tests de l'analyse des logs, sur un petit CSV écrit à la main (tests/data/sample_log.csv).

Contenu du log d'exemple (une image toutes les 50 ms, de 0 à 1250 ms, soit 26 images) :
- 50-100 ms : pinch_ratio à 0,30 puis 0,35, entre enter (0,25) et exit (0,40) : PAS un pincement ;
- 150-300 ms : pincement court, relâché à 300 ms (0,45 ≥ exit) -> 150 ms, clic ;
- 400-1000 ms : pincement long, Press à 750 ms, relâché à 1000 ms -> 600 ms, glisser ;
- 1100 ms : pincement, puis la main disparaît à 1200 ms -> 100 ms, aucune issue.
Temps de traitement : 10 ms partout sauf 12, 11, 30 et 40 ms.
"""

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
    Row,
    percentile,
    pinch_durations,
    pinch_thresholds,
    read_log,
    read_metadata,
    summarize,
    text_histogram,
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


def test_event_names():
    assert event_names([Move(0.1, 0.0), Click()]) == "move+click"
    assert event_names([]) == ""


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
