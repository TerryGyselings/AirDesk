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

from airdesk.interpretation.pinch_rules import BLOCK_CLOSING

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
    pin_kind: str = ""  # clic épinglé : press, press_snap, release, release_snap
    pin_offset_px: float | None = None  # écart main -> point épinglé (px)
    closing_speed: float | None = None  # vitesse de fermeture des doigts (règles F / F3D)


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
                    pin_kind=record.get("pin_kind") or "",
                    pin_offset_px=optional_float(record, "pin_offset_px"),
                    closing_speed=optional_float(record, "closing_speed"),
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


@dataclass(frozen=True)
class ClosingSettings:
    """Réglages de la condition de fermeture (règles F / F3D) pendant la session."""

    rule: str
    min_closing_speed: float
    closing_grace_ms: float

    @property
    def active(self) -> bool:
        return self.rule in ("F", "F3D")


def closing_settings(pinch: dict) -> ClosingSettings:
    """D'après la section `pinch` de la config (0 pour les logs d'avant ces réglages)."""
    return ClosingSettings(pinch.get("rule", "A"), pinch.get("min_closing_speed", 0.0),
                           pinch.get("closing_grace_ms", 0.0))


def has_press(events: list[str]) -> bool:
    """Bouton enfoncé à cette image : Press, ou PressAt (clic épinglé, logs du 2026-11-25)."""
    return "press" in events or "pressat" in events


def has_release(events: list[str]) -> bool:
    """Bouton relâché à cette image : Release, ou ReleaseAt (clic épinglé)."""
    return "release" in events or "releaseat" in events


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
        if has_release(row.events) and previous_state == "pincement":
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
    if "click" in events or has_press(events):
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
            if "click" in row.events or has_press(row.events):
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
        if has_press(row.events) and current is None:
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
        if has_release(row.events):
            episodes.append(current)
            current = None
    if current is not None:
        episodes.append(current)
    return episodes


# --- Vitesse du poignet au contact (condition « mouvement », max_contact_speed) -------------


@dataclass
class ContactSpeed:
    """Un pincement qui a donné un clic ou un glisser, et la vitesse du poignet au contact."""

    contact_ms: int
    speed: float  # tailles de main / s, à l'image du contact
    outcome: str  # OUTCOME_CLICK ou OUTCOME_DRAG


def contact_speeds(rows: list[Row]) -> list[ContactSpeed]:
    """Contact = première image à l'état « pincement » après « actif » (les deux modes de
    clic). Issue : glisser si la machine passe à l'état « glisser », clic si un Click ou un
    Release arrive avant ; les pincements sans issue sont ignorés."""
    found = []
    for i in range(1, len(rows)):
        row = rows[i]
        if row.state != "pincement" or rows[i - 1].state != "actif":
            continue
        if row.wrist_speed is None:
            continue
        outcome = None
        for later in rows[i:]:
            if later.state == "glisser":
                outcome = OUTCOME_DRAG
                break
            if "click" in later.events or has_release(later.events):
                outcome = OUTCOME_CLICK
                break
            if later.state != "pincement":
                break
        if outcome is not None:
            found.append(ContactSpeed(row.timestamp_ms, row.wrist_speed, outcome))
    return found


# --- Clics épinglés : écart entre la main et le point épinglé --------------------------------


def pin_offsets(rows: list[Row]) -> dict[str, list[float]]:
    """Écarts (px) entre la main et le point épinglé, par type d'événement épinglé (press,
    press_snap, release, release_snap). press : dérive pendant les min_press_frames images ;
    release : dérive pendant l'appui ; *_snap : distance au 1er clic d'un double-clic."""
    found: dict[str, list[float]] = {}
    for row in rows:
        if row.pin_kind and row.pin_offset_px is not None:
            if row.pin_kind not in found:
                found[row.pin_kind] = []
            found[row.pin_kind].append(row.pin_offset_px)
    return found


# --- Relâchement relatif (release_rise) : rejeu sur pinch_ratio -------------------------------


@dataclass
class RiseSplit:
    """Un relâchement relatif simulé suivi d'un nouveau contact dans le même appui."""

    release_ms: int  # instant du relâchement relatif
    contact_ms: int  # instant du nouveau contact
    in_drag: bool  # l'appui (tel qu'enregistré) est passé en glisser : une rupture


def rise_splits(rows: list[Row], rise: float, enter: float, guard_px: float) -> list[RiseSplit]:
    """Rejoue `release_rise` sur les appuis enregistrés (images à l'état « pincement » ou
    « glisser » d'affilée) et liste les endroits où il couperait l'appui en deux.

    Simulation simplifiée, sur pinch_ratio seul :
    - relâchement si la mesure dépasse son minimum depuis le contact de `rise` et dépasse
      enter, seulement tant que la main est à moins de `guard_px` pixels du point de contact
      (la garde de la machine : drag_threshold, ici en pixels, car le log ne garde pas la
      taille de la main ; 0,15 taille de main ≈ 95 px, voir docs/journal.md) ;
    - nouveau contact quand la mesure repasse sous enter (sans les autres conditions de la
      règle D).
    Dans un appui passé en glisser, chaque coupure est une rupture du glisser (in_drag)."""
    splits: list[RiseSplit] = []
    episode: list[Row] = []
    for row in rows:
        if row.state in ("pincement", "glisser"):
            episode.append(row)
            continue
        if episode:
            splits += _rise_splits_in_episode(episode, rise, enter, guard_px)
            episode = []
    if episode:  # log terminé pendant un appui
        splits += _rise_splits_in_episode(episode, rise, enter, guard_px)
    return splits


def _rise_splits_in_episode(episode, rise, enter, guard_px) -> list[RiseSplit]:
    in_drag = False
    for row in episode:
        if row.state == "glisser":
            in_drag = True
    found = []
    minimum = None
    anchor = None
    moved = False
    released_at = None
    for row in episode:
        if row.pinch_ratio is None:
            continue
        if anchor is None:
            anchor = row.cursor
        if not moved and anchor is not None and row.cursor is not None:
            dx = row.cursor[0] - anchor[0]
            dy = row.cursor[1] - anchor[1]
            moved = math.hypot(dx, dy) > guard_px
        if released_at is not None:
            if row.pinch_ratio < enter:  # nouveau contact
                found.append(RiseSplit(released_at, row.timestamp_ms, in_drag))
                released_at = None
                minimum = row.pinch_ratio
                anchor = row.cursor
                moved = False
            continue
        if minimum is None or row.pinch_ratio < minimum:
            minimum = row.pinch_ratio
        if not moved and row.pinch_ratio >= minimum + rise and row.pinch_ratio >= enter:
            released_at = row.timestamp_ms
    return found


# --- Vitesse de fermeture des doigts (règles F / F3D) ----------------------------------------


@dataclass
class ContactClosing:
    """Un contact accepté (la machine passe de « actif » à « pincement »)."""

    contact_ms: int
    closing_speed: float | None  # à l'image du contact (None : historique trop court)
    since_release_ms: int | None  # temps depuis le dernier relâchement (None : aucun avant)
    outcome: str  # OUTCOME_CLICK, OUTCOME_DRAG ou OUTCOME_NONE (Press jamais confirmé)


def contact_closing_speeds(rows: list[Row]) -> list[ContactClosing]:
    """Vitesse de fermeture au contact de chaque pincement accepté, et temps écoulé depuis le
    relâchement précédent : un contact juste après un relâchement (2e tape d'un double-clic)
    est exempté de la condition de fermeture (closing_grace_ms)."""
    found = []
    last_release_ms = None
    for i in range(len(rows)):
        row = rows[i]
        if i > 0 and row.state == "pincement" and rows[i - 1].state == "actif":
            since = None
            if last_release_ms is not None:
                since = row.timestamp_ms - last_release_ms
            found.append(ContactClosing(row.timestamp_ms, row.closing_speed, since,
                                        outcome_after(rows, i)))
        if has_release(row.events) or "click" in row.events:
            last_release_ms = row.timestamp_ms
    return found


def outcome_after(rows: list[Row], start: int) -> str:
    """Issue d'un appui qui commence à la ligne `start` : glisser si la machine passe à l'état
    « glisser », clic si un Click ou un Release arrive avant, sinon aucune."""
    for later in rows[start:]:
        if later.state == "glisser":
            return OUTCOME_DRAG
        if "click" in later.events or has_release(later.events):
            return OUTCOME_CLICK
        if later.state != "pincement":
            break
    return OUTCOME_NONE


@dataclass
class ClosingRefusal:
    """Un pincement (vu dans pinch_ratio) refusé pour « fermeture » : doigts trop lents."""

    start_ms: int
    duration_ms: int
    max_closing_speed: float | None  # la plus haute pendant le pincement (meilleure chance)


def refused_for_closing(rows: list[Row], pinches: list[Pinch]) -> list[ClosingRefusal]:
    """Pincements sans issue dont la raison de refus la plus fréquente est « fermeture »."""
    found = []
    for p in pinches:
        if p.outcome != OUTCOME_NONE:
            continue
        blocked = block_reason(rows, p.start_ms, p.end_ms)
        if blocked is None or blocked[0] != BLOCK_CLOSING:
            continue
        best = None
        for row in rows:
            if p.start_ms <= row.timestamp_ms <= p.end_ms and row.closing_speed is not None:
                if best is None or row.closing_speed > best:
                    best = row.closing_speed
        found.append(ClosingRefusal(p.start_ms, p.duration_ms, best))
    return found
