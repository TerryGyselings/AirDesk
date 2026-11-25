"""Analyse d'un journal de mesures (`airdesk --log`) : logique pure, testable sans webcam.

Le script `scripts/analyse_log.py` se contente d'appeler ces fonctions et d'afficher.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

# percentile est utilisé ici et reste importable depuis ce module (scripts, gesture_analysis).
from airdesk.stats import percentile


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
    # Colonnes ajoutées plus tard (absentes des anciens logs) :
    wrist_speed: float | None = None  # tailles de main par seconde
    cursor: tuple[int, int] | None = None  # position calculée du curseur (pointage absolu)
    entry_block: str = ""  # pourquoi un pincement ne pouvait pas commencer ("" : il le pouvait)
    tracker_hands: int | None = None  # mains demandées à MediaPipe (1 ou 2)
    contact_ms: float | None = None  # contact des doigts, sur l'image du premier Press/Click
    anchor_distance: float | None = None  # mode press : distance au point de contact


def optional_float(record: dict, name: str) -> float | None:
    """Valeur d'une colonne, ou None si elle est vide ou absente (ancien format de log)."""
    value = record.get(name)
    if value is None or value == "":
        return None
    return float(value)


def read_log(path: Path) -> list[Row]:
    """Lit le CSV écrit par SessionLog."""
    rows: list[Row] = []
    with open(path, encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            cursor = None
            cursor_x = optional_float(record, "cursor_x")
            cursor_y = optional_float(record, "cursor_y")
            if cursor_x is not None and cursor_y is not None:
                cursor = (int(cursor_x), int(cursor_y))
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
                    wrist_speed=optional_float(record, "wrist_speed"),
                    cursor=cursor,
                    entry_block=record.get("entry_block") or "",
                    tracker_hands=int(record["tracker_hands"]) if record.get("tracker_hands")
                    else None,
                    contact_ms=optional_float(record, "contact_ms"),
                    anchor_distance=optional_float(record, "anchor_distance"),
                )
            )
    return rows


# --- Tremblement du curseur quand la main est immobile -----------------------------------------

# Main « immobile » : poignet à moins de 0,5 taille de main par seconde (≈ 4-5 cm/s) pendant
# au moins 1 s. Le bruit du suivi seul donne de l'ordre de 0,1 taille de main par seconde.
IMMOBILE_SPEED = 0.5
IMMOBILE_MIN_MS = 1000
JITTER_GOAL_PX = 2.0  # objectif : tremblement du curseur sous 2 pixels, main immobile
# Fenêtres de la moyenne glissante qui sépare le tremblement (rapide) de la dérive (lente).
# Le résultat DÉPEND de ce choix : le curseur est déjà lissé par le filtre 1€ (coupure 0,5 Hz
# à basse vitesse), donc le tremblement restant est lent, et une fenêtre courte (300 ms) en
# compte une partie comme de la dérive : mesure optimiste. L'objectif est jugé sur 1 s.
JITTER_WINDOWS_MS = (300, 1000)
JITTER_GOAL_WINDOW_MS = 1000
MIN_RESIDUALS = 10  # en dessous, période trop courte pour la fenêtre : « n/a »


def frame_interval_ms(rows: list[Row]) -> float:
    """Durée typique d'une image : médiane des écarts entre horodatages successifs."""
    gaps = []
    for i in range(1, len(rows)):
        gap = rows[i].timestamp_ms - rows[i - 1].timestamp_ms
        if gap > 0:
            gaps.append(gap)
    if not gaps:
        return 33.3
    return statistics.median(gaps)


def window_frames(window_ms: float, interval_ms: float) -> int:
    """Nombre d'images d'une fenêtre centrée : arrondi, puis impair (autant d'images de chaque
    côté de l'image centrale)."""
    frames = round(window_ms / interval_ms)
    if frames % 2 == 0:
        frames += 1
    return max(frames, 3)


def residual_std(values: list[float], frames: int) -> float | None:
    """Écart-type de `valeur − moyenne glissante centrée sur `frames` images`.

    La moyenne glissante suit les variations lentes (la dérive) ; ce qui reste est le
    tremblement. Seules les images où la fenêtre entière tient dans la série sont utilisées.
    None s'il en reste moins de MIN_RESIDUALS.
    """
    half = frames // 2
    residuals = []
    for i in range(half, len(values) - half):
        window = values[i - half:i + half + 1]
        residuals.append(values[i] - sum(window) / len(window))
    if len(residuals) < MIN_RESIDUALS:
        return None
    return statistics.pstdev(residuals)


@dataclass
class StillPeriod:
    start_ms: int
    end_ms: int
    frames: int
    drift_x: float  # écart-type brut de la position du curseur (dérive + tremblement), px
    drift_y: float
    # Tremblement (plus grand des deux axes) pour chaque fenêtre de JITTER_WINDOWS_MS, en px ;
    # None si la période est trop courte pour la fenêtre.
    jitter_by_window: dict[int, float | None]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def drift(self) -> float:
        return max(self.drift_x, self.drift_y)

    def jitter(self, window_ms: int = JITTER_GOAL_WINDOW_MS) -> float | None:
        return self.jitter_by_window.get(window_ms)


def still_periods(
    rows: list[Row],
    max_speed: float = IMMOBILE_SPEED,
    min_duration_ms: int = IMMOBILE_MIN_MS,
    windows_ms: tuple[int, ...] = JITTER_WINDOWS_MS,
) -> list[StillPeriod]:
    """Périodes où la main est immobile, avec dérive et tremblement du curseur.

    Une période = images consécutives avec main, vitesse du poignet connue et < max_speed,
    et position du curseur connue (pointage absolu), pendant au moins min_duration_ms.
    """
    interval = frame_interval_ms(rows)
    periods: list[StillPeriod] = []
    run: list[Row] = []
    for row in rows + [None]:  # None final : ferme la dernière période
        still = (
            row is not None
            and row.hand_present
            and row.wrist_speed is not None
            and row.wrist_speed < max_speed
            and row.cursor is not None
        )
        if still:
            run.append(row)
            continue
        if run and run[-1].timestamp_ms - run[0].timestamp_ms >= min_duration_ms:
            xs = []
            ys = []
            for r in run:
                xs.append(r.cursor[0])
                ys.append(r.cursor[1])
            jitters: dict[int, float | None] = {}
            for window_ms in windows_ms:
                frames = window_frames(window_ms, interval)
                jx = residual_std(xs, frames)
                jy = residual_std(ys, frames)
                jitters[window_ms] = None if jx is None or jy is None else max(jx, jy)
            periods.append(StillPeriod(run[0].timestamp_ms, run[-1].timestamp_ms, len(run),
                                       statistics.pstdev(xs), statistics.pstdev(ys), jitters))
        run = []
    return periods


def wrist_speed_quartiles(rows: list[Row]) -> tuple[float, float, float] | None:
    """(P25, médiane, P75) de la vitesse du poignet, pour régler le seuil d'immobilité."""
    speeds = []
    for row in rows:
        if row.wrist_speed is not None:
            speeds.append(row.wrist_speed)
    if not speeds:
        return None
    return percentile(speeds, 25), statistics.median(speeds), percentile(speeds, 75)


def read_metadata(path: Path) -> dict:
    """Lit le JSON des réglages écrit à côté du CSV."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pinch_thresholds(metadata: dict) -> tuple[float, float, float]:
    """(enter, exit, click_max_ms) en vigueur pendant la session enregistrée."""
    pinch = metadata["config"]["pinch"]
    return pinch["enter"], pinch["exit"], pinch["click_max_ms"]


def click_mode(metadata: dict) -> str:
    """Mode de clic de la session ("release" pour les logs d'avant le mode press)."""
    return metadata["config"]["pinch"].get("click_mode", "release")


def release_grace(metadata: dict) -> float:
    """Délai de relâchement en vigueur pendant la session (0 pour les anciens logs)."""
    return metadata["config"]["pinch"].get("release_grace_ms", 0)


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

    # Compté de la même façon dans les deux modes de clic :
    # - glisser = la machine passe à l'état « glisser » ;
    # - clic = événement Click (mode release), ou Release depuis l'état « pincement » sans
    #   être passé en glisser (mode press : Press puis Release sur place).
    processing = []
    clicks = 0
    drags = 0
    previous_state = ""
    for row in rows:
        processing.append(row.processing_ms)
        if "click" in row.events:
            clicks += 1
        if "release" in row.events and previous_state == "pincement":
            clicks += 1
        if row.state == "glisser" and previous_state != "glisser":
            drags += 1
        previous_state = row.state

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


def pinch_durations(
    rows: list[Row], enter: float, exit: float, outcome_extra_ms: float = 0
) -> list[Pinch]:
    """Retrouve chaque pincement à partir de `pinch_ratio` et de l'hystérésis enter/exit.

    Début : `pinch_ratio` passe sous `enter`. Fin : il repasse à `exit` ou au-dessus
    (mêmes comparaisons que la machine à états), ou la main disparaît.
    On ne se sert PAS des états enregistrés pour mesurer la durée : ils dépendent de
    `click_max_ms`, le seuil qu'on cherche justement à régler.

    L'issue observée (clic / glisser) est gardée à côté, pour comparer. Avec un délai de
    relâchement (release_grace_ms), la machine émet le clic APRÈS la fin mesurée ici : on
    cherche donc les événements jusqu'à `fin + outcome_extra_ms` (en pratique
    release_grace_ms + une image), sans dépasser le début du pincement suivant.
    """
    # 1) Début et fin de chaque pincement.
    spans: list[tuple[int, int, str]] = []
    pinched = False
    start_ms = 0
    for row in rows:
        if row.pinch_ratio is None:  # pas de main
            if pinched:
                spans.append((start_ms, row.timestamp_ms, END_HAND_LOST))
                pinched = False
            continue
        if not pinched:
            if row.pinch_ratio < enter:
                pinched = True
                start_ms = row.timestamp_ms
            continue
        if row.pinch_ratio >= exit:
            spans.append((start_ms, row.timestamp_ms, END_RELEASED))
            pinched = False
    if pinched:
        spans.append((start_ms, rows[-1].timestamp_ms, END_LOG))

    # 2) Issue : événements émis pendant le pincement et juste après (délai de relâchement).
    pinches: list[Pinch] = []
    for i in range(len(spans)):
        start, end, ended_by = spans[i]
        limit = end + outcome_extra_ms
        if i + 1 < len(spans) and spans[i + 1][0] <= limit:
            limit = spans[i + 1][0] - 1  # ne pas déborder sur le pincement suivant
        events_seen: list[str] = []
        dragged = False
        for row in rows:
            if start <= row.timestamp_ms <= limit:
                for name in row.events:
                    events_seen.append(name)
                if row.state == "glisser":
                    dragged = True
        pinches.append(Pinch(start, end, outcome_of(events_seen, dragged), ended_by))
    return pinches


def block_reason(rows: list[Row], start_ms: int, end_ms: int) -> tuple[str, int, int] | None:
    """Raison la plus fréquente du refus d'entrée pendant un pincement : (raison, images
    concernées, images du pincement). None si le log n'a pas la colonne entry_block (ancien
    format) ou si rien n'a bloqué."""
    counts: dict[str, int] = {}
    total = 0
    has_column = False
    for row in rows:
        if start_ms <= row.timestamp_ms <= end_ms and row.hand_present:
            total += 1
            if row.entry_block:
                has_column = True
                counts[row.entry_block] = counts.get(row.entry_block, 0) + 1
    if not has_column or not counts:
        return None
    best = None
    for reason, count in counts.items():
        if best is None or count > counts[best]:
            best = reason
    return best, counts[best], total


@dataclass
class ModeTiming:
    """Temps de traitement des images traitées avec 1 ou 2 mains demandées à MediaPipe."""

    hands: int
    frames: int
    mean_ms: float
    p95_ms: float
    max_ms: float
    slow_share: float  # part des images au-dessus de SLOW_FRAME_MS


SLOW_FRAME_MS = 25.0  # une image qui prend plus de 25 ms fait perdre de la fluidité


def timing_by_mode(rows: list[Row]) -> list[ModeTiming]:
    """Temps de traitement séparé par mode (1 main / 2 mains), pour trouver d'où viennent les
    pics. Vide pour un log sans la colonne tracker_hands."""
    by_mode: dict[int, list[float]] = {}
    for row in rows:
        if row.tracker_hands is not None:
            by_mode.setdefault(row.tracker_hands, []).append(row.processing_ms)
    result = []
    for hands in sorted(by_mode):
        values = by_mode[hands]
        slow = 0
        for v in values:
            if v > SLOW_FRAME_MS:
                slow += 1
        result.append(ModeTiming(hands, len(values), sum(values) / len(values),
                                 percentile(values, 95), max(values), slow / len(values)))
    return result


def outcome_of(events: list[str], dragged: bool) -> str:
    """Ce qu'a donné un pincement, d'après les événements émis pendant sa durée et le passage
    de la machine à l'état « glisser ». En mode press, un clic est un Press suivi d'un
    Release sans passer en glisser ; en mode release, c'est un Click."""
    if dragged:
        return OUTCOME_DRAG
    if "click" in events or "press" in events:
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


# --- Latence du clic, déplacement pendant les clics, saut au début des glisser --------------


@dataclass
class Latency:
    """Contact des doigts -> premier événement du pincement (Press ou Click)."""

    contact_ms: float
    event_ms: int
    event: str  # "press" ou "click"

    @property
    def latency_ms(self) -> float:
        return self.event_ms - self.contact_ms


def latencies(rows: list[Row]) -> list[Latency]:
    """Une latence par pincement qui a produit un Press ou un Click (colonne contact_ms)."""
    found = []
    for row in rows:
        if row.contact_ms is None:
            continue
        event = "press"
        if "click" in row.events:
            event = "click"
        found.append(Latency(row.contact_ms, row.timestamp_ms, event))
    return found


def estimated_latencies(rows: list[Row], pinches: list[Pinch]) -> list[Latency]:
    """Pour les logs sans colonne contact_ms : contact = début du pincement retrouvé dans
    pinch_ratio, événement = premier Press / Click émis ensuite, avant le pincement suivant
    (et au plus 3 s après). Estimation : même seuil enter que la machine, mais sans les
    conditions de la règle (un pincement refusé ne produit pas d'événement, donc pas de
    latence)."""
    found = []
    for i in range(len(pinches)):
        start = pinches[i].start_ms
        limit = start + 3000
        if i + 1 < len(pinches) and pinches[i + 1].start_ms - 1 < limit:
            limit = pinches[i + 1].start_ms - 1
        for row in rows:
            if row.timestamp_ms < start or row.timestamp_ms > limit:
                continue
            if "click" in row.events or "press" in row.events:
                event = "click" if "click" in row.events else "press"
                found.append(Latency(start, row.timestamp_ms, event))
                break
    return found


def mean_and_p95(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), percentile(values, 95)


@dataclass
class PressEpisode:
    """Mode press : un appui (Press ... Release)."""

    start_ms: int
    max_anchor_distance: float | None  # plus grande distance au point de contact
    dragged: bool  # passé en glisser
    jump_px: float | None  # glisser : écart curseur figé -> main au passage en glisser


def press_episodes(rows: list[Row]) -> list[PressEpisode]:
    """Découpe les appuis du mode press : du Press au Release (ou à la fin du log).

    Saut : distance en pixels entre la position du curseur à l'image du Press (le curseur
    reste figé là) et la position calculée de la main à l'image où l'état passe en
    « glisser » ; c'est le saut qu'aurait le curseur sans rattrapage (drag_catchup_ms)."""
    episodes = []
    current = None
    anchor_cursor = None
    for row in rows:
        if "press" in row.events and current is None:
            current = PressEpisode(row.timestamp_ms, None, False, None)
            anchor_cursor = row.cursor
            continue
        if current is None:
            continue
        if row.anchor_distance is not None:
            if (current.max_anchor_distance is None
                    or row.anchor_distance > current.max_anchor_distance):
                current.max_anchor_distance = row.anchor_distance
        if row.state == "glisser" and not current.dragged:
            current.dragged = True
            if anchor_cursor is not None and row.cursor is not None:
                dx = row.cursor[0] - anchor_cursor[0]
                dy = row.cursor[1] - anchor_cursor[1]
                current.jump_px = math.hypot(dx, dy)
        if "release" in row.events:
            episodes.append(current)
            current = None
    if current is not None:
        episodes.append(current)
    return episodes

