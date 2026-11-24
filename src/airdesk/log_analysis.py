"""Analyse d'un journal de mesures (`airdesk --log`) : logique pure, testable sans webcam.

Le script `scripts/analyse_log.py` se contente d'appeler ces fonctions et d'afficher.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    """Une ligne du CSV, avec des types Python (et None quand aucune main n'est visible)."""

    timestamp_ms: int
    hand_present: bool
    pinch_ratio: float | None
    extended_fingers: int | None
    state: str
    events: list[str]
    processing_ms: float


def read_log(path: Path) -> list[Row]:
    """Lit le CSV écrit par SessionLog."""
    rows: list[Row] = []
    with open(path, encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            pinch_ratio = None
            if record["pinch_ratio"] != "":
                pinch_ratio = float(record["pinch_ratio"])
            extended_fingers = None
            if record["extended_fingers"] != "":
                extended_fingers = int(record["extended_fingers"])
            events = []
            if record["events"] != "":
                events = record["events"].split("+")
            rows.append(
                Row(
                    timestamp_ms=int(record["timestamp_ms"]),
                    hand_present=record["hand_present"] == "1",
                    pinch_ratio=pinch_ratio,
                    extended_fingers=extended_fingers,
                    state=record["state"],
                    events=events,
                    processing_ms=float(record["processing_ms"]),
                )
            )
    return rows


def read_metadata(path: Path) -> dict:
    """Lit le JSON des réglages écrit à côté du CSV."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pinch_thresholds(metadata: dict) -> tuple[float, float, float]:
    """(enter, exit, click_max_ms) en vigueur pendant la session enregistrée."""
    pinch = metadata["config"]["pinch"]
    return pinch["enter"], pinch["exit"], pinch["click_max_ms"]


def percentile(values: list[float], p: float) -> float:
    """Percentile par la méthode du « rang le plus proche ».

    On trie les valeurs et on prend celle de rang ceil(p/100 × n) (rangs comptés à partir de 1).
    Le 95e percentile est donc une valeur réellement mesurée, telle que 95 % des mesures
    lui sont inférieures ou égales. Plus parlant que la moyenne pour la latence : quelques
    images lentes suffisent à rendre le curseur saccadé, même si la moyenne est bonne.
    """
    if not values:
        raise ValueError("Aucune valeur : percentile impossible.")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    if rank < 1:
        rank = 1
    return ordered[rank - 1]


@dataclass
class Summary:
    frames: int
    duration_s: float
    fps_mean: float
    processing_mean_ms: float
    processing_p95_ms: float
    clicks: int
    drags: int


def summarize(rows: list[Row]) -> Summary:
    """Chiffres globaux de la session.

    FPS moyen = (nombre d'images - 1) / durée entre la première et la dernière image :
    n images délimitent n - 1 intervalles.
    """
    if len(rows) < 2:
        raise ValueError("Il faut au moins deux images pour calculer une cadence.")
    duration_s = (rows[-1].timestamp_ms - rows[0].timestamp_ms) / 1000.0
    fps_mean = 0.0
    if duration_s > 0:
        fps_mean = (len(rows) - 1) / duration_s

    processing = []
    clicks = 0
    drags = 0
    for row in rows:
        processing.append(row.processing_ms)
        if "click" in row.events:
            clicks += 1
        if "press" in row.events:  # un glisser commence par un Press
            drags += 1

    return Summary(
        frames=len(rows),
        duration_s=duration_s,
        fps_mean=fps_mean,
        processing_mean_ms=sum(processing) / len(processing),
        processing_p95_ms=percentile(processing, 95),
        clicks=clicks,
        drags=drags,
    )


# Issues possibles d'un pincement, telles qu'observées dans le log.
OUTCOME_CLICK = "clic"
OUTCOME_DRAG = "glisser"
OUTCOME_NONE = "aucune"  # ex. pincement pendant la veille, ou main perdue

END_RELEASED = "relâché"
END_HAND_LOST = "main perdue"
END_LOG = "fin du log"


@dataclass
class Pinch:
    start_ms: int
    end_ms: int
    outcome: str  # ce que la machine à états a réellement émis
    ended_by: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def pinch_durations(rows: list[Row], enter: float, exit: float) -> list[Pinch]:
    """Retrouve chaque pincement à partir de `pinch_ratio` et de l'hystérésis enter/exit.

    Début : `pinch_ratio` passe sous `enter`. Fin : il repasse à `exit` ou au-dessus
    (mêmes comparaisons que la machine à états), ou la main disparaît.
    On ne se sert PAS des états enregistrés pour mesurer la durée : ils dépendent de
    `click_max_ms`, le seuil qu'on cherche justement à régler.
    L'issue observée (clic / glisser) est gardée à côté, pour comparer.
    """
    pinches: list[Pinch] = []
    pinched = False
    start_ms = 0
    events_seen: list[str] = []

    for row in rows:
        if row.pinch_ratio is None:  # pas de main
            if pinched:
                pinches.append(
                    Pinch(start_ms, row.timestamp_ms, outcome_of(events_seen), END_HAND_LOST)
                )
                pinched = False
            continue

        if not pinched:
            if row.pinch_ratio < enter:
                pinched = True
                start_ms = row.timestamp_ms
                events_seen = list(row.events)
            continue

        # Pincement en cours. Les événements de l'image de fin comptent aussi :
        # c'est là que la machine émet Click ou Release.
        for name in row.events:
            events_seen.append(name)
        if row.pinch_ratio >= exit:
            pinches.append(
                Pinch(start_ms, row.timestamp_ms, outcome_of(events_seen), END_RELEASED)
            )
            pinched = False

    if pinched:
        pinches.append(Pinch(start_ms, rows[-1].timestamp_ms, outcome_of(events_seen), END_LOG))
    return pinches


def outcome_of(events: list[str]) -> str:
    """Ce qu'a donné un pincement d'après les événements émis pendant sa durée."""
    if "press" in events:
        return OUTCOME_DRAG
    if "click" in events:
        return OUTCOME_CLICK
    return OUTCOME_NONE


def text_histogram(
    values: list[float],
    bin_ms: int = 50,
    bar_width: int = 40,
    marker: float | None = None,
    marker_label: str = "",
) -> list[str]:
    """Histogramme en texte, une ligne par tranche de `bin_ms` millisecondes.

        100-150 ms | ##### 5
        150-200 ms | ## 2   <- click_max_ms

    Les barres sont mises à l'échelle si une tranche dépasse `bar_width` valeurs.
    `marker` signale la tranche qui contient une valeur remarquable (ex. un seuil).
    """
    if not values:
        return ["(aucune valeur)"]

    counts: dict[int, int] = {}
    highest_bin = 0
    for value in values:
        b = int(value // bin_ms)
        counts[b] = counts.get(b, 0) + 1
        if b > highest_bin:
            highest_bin = b
    marker_bin = None
    if marker is not None:
        marker_bin = int(marker // bin_ms)
        if marker_bin > highest_bin:
            highest_bin = marker_bin

    largest = max(counts.values())
    scale = 1.0
    if largest > bar_width:
        scale = bar_width / largest

    lines = []
    for b in range(highest_bin + 1):
        count = counts.get(b, 0)
        bar = "#" * round(count * scale)
        if count > 0 and bar == "":
            bar = "#"  # une valeur isolée reste visible même après mise à l'échelle
        low = b * bin_ms
        line = f"{low:>5}-{low + bin_ms:<5} ms | {bar} {count if count else ''}".rstrip()
        if b == marker_bin:
            line += f"   <- {marker_label}"
        lines.append(line)
    return lines
