"""Analyse d'un jeu de données étiqueté (scripts/record_gestures.py) : logique pure, testée.

Question posée : quelle règle de détection du pincement (voir interpretation/pinch_rules.py)
reconnaît les vrais pincements sans se déclencher sur un poing ou un geste parasite ?

Vérité terrain :
- positif : « pincement_maintenu » (sans la première seconde, le temps de prendre la pose) ;
- négatifs : main ouverte, poing, poing qui s'ouvre et se ferme, main relâchée, parasites ;
- « pincements_repetes » : un pincement par seconde au métronome, on compte les ÉPISODES ;
- « pincement_vers_camera » : HORS VOCABULAIRE (décision de conception : ce geste, autres
  doigts repliés, n'est pas un pincement d'AirDesk). Affiché pour information seulement.

Deux niveaux de mesure :
- image par image (matrice de confusion) : utile pour choisir les seuils ;
- par épisode (simulation de l'entrée / sortie du pincement) : ce que ressent l'utilisateur,
  un faux épisode court = un faux clic, une rupture = un glisser lâché.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from airdesk.interpretation.features import (
    palm_facing,
    palm_width,
    pinch_ratio_3d,
    pinch_segment,
)
from airdesk.interpretation.pinch_rules import (
    ClosingSpeed,
    PinchHysteresis,
    RuleThresholds,
    pinch_candidate,
)
from airdesk.interpretation.quality import HandSizeGuard
from airdesk.stats import percentile

POSITIVE_LABEL = "pincement_maintenu"
NEGATIVE_LABELS = ["main_ouverte", "poing", "poing_ouvre_ferme", "main_relachee", "parasites",
                   "main_repos_profil", "main_repos_profil_gauche", "va_et_vient_gauche"]
# Négatifs dont on compte les faux épisodes (la main ouverte sert à l'engagement, elle est
# vue à part dans la matrice image par image). main_repos_profil : protocole v3 et suivants ;
# main_repos_profil_gauche et va_et_vient_gauche : protocole v4.
FALSE_EPISODE_LABELS = ["poing", "poing_ouvre_ferme", "main_relachee", "parasites",
                        "main_repos_profil", "main_repos_profil_gauche", "va_et_vient_gauche"]
# Pincements au métronome (un par seconde) : au centre, et main tournée vers la gauche (v4).
EPISODE_LABELS = ["pincements_repetes", "pincements_repetes_gauche"]
# Main de profil (repos ou pivot vers la gauche) : la cible des règles E et F.
PROFILE_NEGATIVE_LABELS = ["main_repos_profil", "main_repos_profil_gauche", "va_et_vient_gauche"]
OUT_OF_VOCABULARY_LABELS = ["pincement_vers_camera"]
SETTLE_MS = 1000  # début de « pincement_maintenu » ignoré (mise en place du geste)
LOST_TIMEOUT_MS = 1000  # comme engagement.lost_timeout_ms de la machine à états

EXT_COLUMNS = ["ext_index", "ext_middle", "ext_ring", "ext_pinky"]
EXT3D_COLUMNS = ["ext3d_index", "ext3d_middle", "ext3d_ring", "ext3d_pinky"]


@dataclass
class Sample:
    """Une image du jeu de données (descripteurs seulement)."""

    label: str
    t_in_sequence_ms: int
    pinch_ratio: float | None  # None : pas de main détectée
    extensions: tuple[float, float, float, float] | None
    extensions_3d: tuple[float, float, float, float] | None
    hand_size: float | None = None  # None : inconnue (le garde-fou laisse alors passer)
    # Distance pouce -> segment de l'index, recalculée à partir des 21 points enregistrés
    # (None si le fichier n'a pas les colonnes des points, comme les CSV d'exemple).
    pinch_seg: float | None = None
    pinch_seg_68: float | None = None
    # Orientation de la paume, recalculée elle aussi à partir des 21 points.
    palm_width: float | None = None
    palm_facing: float | None = None
    # Hauteur du poignet dans l'image (0 = haut, 1 = bas), protocole v2 et suivants.
    wrist_y: float | None = None
    # Distance pouce-index en 3D, recalculée à partir des 21 points (v1 compris).
    pinch_ratio_3d: float | None = None
    # Vitesse de fermeture des doigts (baisse par seconde sur ~200 ms) de pinch_ratio et de
    # pinch_ratio_3d, recalculée sur la série de chaque prise (voir add_closing_speeds).
    closing_speed: float | None = None
    closing_speed_3d: float | None = None
    # Vitesse du poignet (tailles de main / s), protocole v2 et suivants.
    wrist_speed: float | None = None

    @property
    def hand_present(self) -> bool:
        return self.pinch_ratio is not None


def measure_value(s: Sample, measure: str) -> float:
    """Valeur de la mesure de pincement choisie (voir pinch_rules.MEASURES)."""
    if measure == "ratio":
        value = s.pinch_ratio
    elif measure == "ratio3d":
        value = s.pinch_ratio_3d
    elif measure == "seg78":
        value = s.pinch_seg
    elif measure == "seg68":
        value = s.pinch_seg_68
    else:
        raise ValueError(f"Mesure inconnue : {measure}")
    if value is None:
        raise ValueError(f"Mesure « {measure} » indisponible : le jeu de données n'a pas les "
                         "colonnes des 21 points (x0, y0, ...).")
    return value


def landmarks_from_record(record: dict) -> list[tuple[float, float, float]] | None:
    """Les 21 points normalisés d'une ligne de CSV, ou None si les colonnes manquent."""
    if not record.get("x0"):
        return None
    points = []
    for i in range(21):
        points.append((float(record[f"x{i}"]), float(record[f"y{i}"]), float(record[f"z{i}"])))
    return points


def read_dataset(path: Path) -> list[Sample]:
    """Lit un CSV de record_gestures.py. Les colonnes des 21 points sont ignorées ici."""
    samples: list[Sample] = []
    with open(path, encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            if record["hand_present"] != "1":
                samples.append(Sample(record["label"], int(record["t_in_sequence_ms"]),
                                      None, None, None))
                continue
            ext = []
            for name in EXT_COLUMNS:
                ext.append(float(record[name]))
            ext3d = []
            for name in EXT3D_COLUMNS:
                ext3d.append(float(record[name]))
            hand_size = None
            if record.get("hand_size"):
                hand_size = float(record["hand_size"])
            pinch_seg = None
            pinch_seg_68 = None
            width = None
            facing = None
            ratio_3d = None
            points = landmarks_from_record(record)
            if points is not None:
                # Points déjà relatifs au poignet, en unités homogènes et divisés par la
                # taille de main : même calcul qu'en direct (features.py).
                pinch_seg = pinch_segment(points, 7)
                pinch_seg_68 = pinch_segment(points, 6)
                width = palm_width(points)
                facing = palm_facing(points)
                ratio_3d = pinch_ratio_3d(points)
            elif record.get("pinch_ratio_3d"):
                ratio_3d = float(record["pinch_ratio_3d"])  # colonne directe (v4)
            samples.append(
                Sample(
                    label=record["label"],
                    t_in_sequence_ms=int(record["t_in_sequence_ms"]),
                    pinch_ratio=float(record["pinch_ratio"]),
                    extensions=(ext[0], ext[1], ext[2], ext[3]),
                    extensions_3d=(ext3d[0], ext3d[1], ext3d[2], ext3d[3]),
                    hand_size=hand_size,
                    pinch_seg=pinch_seg,
                    pinch_seg_68=pinch_seg_68,
                    palm_width=width,
                    palm_facing=facing,
                    wrist_y=float(record["wrist_y"]) if record.get("wrist_y") else None,
                    pinch_ratio_3d=ratio_3d,
                    wrist_speed=float(record["wrist_speed"]) if record.get("wrist_speed")
                    else None,
                )
            )
    add_closing_speeds(samples)
    return samples


def add_closing_speeds(samples: list[Sample]) -> None:
    """Calcule closing_speed et closing_speed_3d image par image, comme en direct.

    Même calcul que la machine à états (pinch_rules.ClosingSpeed, fenêtre de 200 ms) : on
    repart de zéro à chaque nouvelle prise (t_in_sequence_ms qui recule) et après une image
    sans main.
    """
    closing = ClosingSpeed()
    closing_3d = ClosingSpeed()
    previous_label = None
    previous_t = -1
    for s in samples:
        if s.label != previous_label or s.t_in_sequence_ms < previous_t:
            closing.reset()
            closing_3d.reset()
        previous_label = s.label
        previous_t = s.t_in_sequence_ms
        if not s.hand_present:
            closing.reset()
            closing_3d.reset()
            continue
        s.closing_speed = closing.update(s.t_in_sequence_ms, s.pinch_ratio)
        if s.pinch_ratio_3d is not None:
            s.closing_speed_3d = closing_3d.update(s.t_in_sequence_ms, s.pinch_ratio_3d)


def labels_in_order(samples: list[Sample]) -> list[str]:
    """Étiquettes présentes, dans l'ordre de première apparition."""
    labels: list[str] = []
    for s in samples:
        if s.label not in labels:
            labels.append(s.label)
    return labels


def frame_samples(samples: list[Sample], label: str) -> list[Sample]:
    """Images utilisables d'une étiquette : main détectée, mise en place exclue (positif)."""
    selected = []
    for s in samples:
        if s.label != label or not s.hand_present:
            continue
        if label == POSITIVE_LABEL and s.t_in_sequence_ms < SETTLE_MS:
            continue
        selected.append(s)
    return selected


def split_sequences(samples: list[Sample], label: str) -> list[list[Sample]]:
    """Les prises successives d'une étiquette (une par fichier, en pratique).

    Une nouvelle prise commence quand t_in_sequence_ms repart en arrière.
    """
    sequences: list[list[Sample]] = []
    previous_t = None
    for s in samples:
        if s.label != label:
            continue
        if previous_t is None or s.t_in_sequence_ms < previous_t:
            sequences.append([])
        sequences[-1].append(s)
        previous_t = s.t_in_sequence_ms
    return sequences


def count_sequences(samples: list[Sample], label: str) -> int:
    return len(split_sequences(samples, label))


# --- Distributions ----------------------------------------------------------------------------


@dataclass
class Stats:
    n: int
    minimum: float
    p5: float
    median: float
    p95: float
    maximum: float


def describe(values: list[float]) -> Stats | None:
    """Minimum, percentiles 5 et 95 (rang le plus proche), médiane, maximum."""
    if not values:
        return None
    return Stats(
        n=len(values),
        minimum=min(values),
        p5=percentile(values, 5),
        median=statistics.median(values),
        p95=percentile(values, 95),
        maximum=max(values),
    )


# Descripteurs résumés pour chaque étiquette.
# « autres_max » = la plus grande extension parmi majeur, annulaire, auriculaire :
# petite si tous ces doigts sont repliés (poing), c'est ce que testent les règles C et D.
DESCRIPTORS = ["pinch_ratio", "pinch_seg", "pinch_seg_68", "ext_index", "ext_index_3d",
               "ext_autres_max", "ext_autres_max_3d", "palm_width", "palm_facing", "wrist_y"]


def descriptor_value(name: str, s: Sample) -> float | None:
    """Valeur du descripteur `name` pour une image (main présente) ; None si indisponible."""
    if name == "pinch_ratio":
        return s.pinch_ratio
    if name == "pinch_seg":
        return s.pinch_seg
    if name == "pinch_seg_68":
        return s.pinch_seg_68
    if name == "palm_width":
        return s.palm_width
    if name == "palm_facing":
        return s.palm_facing
    if name == "wrist_y":
        return s.wrist_y
    if name == "ext_index":
        return s.extensions[0]
    if name == "ext_index_3d":
        return s.extensions_3d[0]
    if name == "ext_autres_max":
        return max(s.extensions[1:])
    if name == "ext_autres_max_3d":
        return max(s.extensions_3d[1:])
    raise ValueError(f"Descripteur inconnu : {name}")


def detection_rate(samples: list[Sample], label: str) -> float:
    """Part des images de l'étiquette où une main a été détectée."""
    total = 0
    present = 0
    for s in samples:
        if s.label == label:
            total += 1
            if s.hand_present:
                present += 1
    if total == 0:
        return 0.0
    return present / total


def stats_by_label(samples: list[Sample]) -> dict[str, dict[str, Stats | None]]:
    """Pour chaque étiquette, la distribution de chaque descripteur de DESCRIPTORS."""
    result: dict[str, dict[str, Stats | None]] = {}
    for label in labels_in_order(samples):
        selected = frame_samples(samples, label)
        result[label] = {}
        for name in DESCRIPTORS:
            values = []
            for s in selected:
                value = descriptor_value(name, s)
                if value is not None:
                    values.append(value)
            result[label][name] = describe(values)
    return result


# --- Règles : matrice de confusion image par image ---------------------------------------------


def detects(rule: str, s: Sample, thresholds: RuleThresholds) -> bool:
    """La règle accepte-t-elle cette image ? Avec la vitesse de fermeture (règles F / F3D) :
    sans elle, F refusait toutes les images."""
    return pinch_candidate(rule, s.pinch_ratio, s.extensions, s.extensions_3d, thresholds,
                           s.palm_width, s.closing_speed, s.closing_speed_3d)


@dataclass
class Confusion:
    """Matrice « pincement vs pas pincement », image par image."""

    true_positives: int
    false_negatives: int
    false_positives: dict[str, int]  # par étiquette négative
    negatives: dict[str, int]  # nombre d'images par étiquette négative

    @property
    def recall(self) -> float:
        """Taux de vrais positifs : part des images pincées reconnues."""
        total = self.true_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total


def confusion(samples: list[Sample], rule: str, thresholds: RuleThresholds) -> Confusion:
    tp = 0
    fn = 0
    for s in frame_samples(samples, POSITIVE_LABEL):
        if detects(rule, s, thresholds):
            tp += 1
        else:
            fn += 1
    false_positives = {}
    negatives = {}
    for label in NEGATIVE_LABELS:
        selected = frame_samples(samples, label)
        negatives[label] = len(selected)
        count = 0
        for s in selected:
            if detects(rule, s, thresholds):
                count += 1
        false_positives[label] = count
    return Confusion(tp, fn, false_positives, negatives)


# --- Simulation par épisode ---------------------------------------------------------------------

END_RELEASED = "relâché"
END_HAND_LOST = "main perdue"
END_SEQUENCE = "fin de séquence"


@dataclass
class Episode:
    start_ms: int
    end_ms: int
    ended_by: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class EpisodeSettings:
    """Réglages de la simulation : ce qu'on fait varier dans le balayage.

    `thresholds.enter` et `exit` s'appliquent à la mesure choisie (`measure`).
    """

    rule: str
    thresholds: RuleThresholds
    exit: float
    release_grace_ms: float = 0
    guard_ratio: float = 0.0  # 0 = garde-fou désactivé
    window_ms: float = 2000
    lost_timeout_ms: float = LOST_TIMEOUT_MS
    measure: str = "ratio"
    exit_on_fold: bool = False
    release_cancel_margin: float = 0.0
    release_fast_margin: float = 0.0

    def describe(self) -> str:
        fold = ", sortie si repli" if self.exit_on_fold else ""
        palm = ""
        if self.thresholds.require_palm_facing:
            palm = f", paume >= {self.thresholds.min_palm_width:.2f}"
        return (f"règle {self.rule}, mesure {self.measure}, enter {self.thresholds.enter:.2f}, "
                f"exit {self.exit:.2f}, délai {self.release_grace_ms:g} ms{fold}{palm}")


def simulate_sequence(sequence: list[Sample], settings: EpisodeSettings) -> list[Episode]:
    """Épisodes de pincement d'une prise, en imitant la machine à états.

    - Entrée / sortie : PinchHysteresis (règle, mesure, exit, délai, sortie si repli).
    - Une image sans main, ou écartée par le garde-fou de taille, ne relâche PAS le pincement
      (la machine à états ne réagit pas à une image sans main) ; seule une absence de plus de
      `lost_timeout_ms` y met fin.
    """
    hysteresis = PinchHysteresis(
        settings.rule, settings.thresholds, settings.exit, settings.release_grace_ms,
        settings.exit_on_fold, settings.release_cancel_margin, settings.release_fast_margin,
    )
    guard = HandSizeGuard(settings.guard_ratio, settings.window_ms)
    episodes: list[Episode] = []
    start_ms = 0
    last_seen_ms = None

    for s in sequence:
        t = s.t_in_sequence_ms
        usable = s.hand_present
        if usable and s.hand_size is not None:
            usable = guard.accept(t, s.hand_size)
        if not usable:
            lost_for = 0
            if last_seen_ms is not None:
                lost_for = t - last_seen_ms
            if hysteresis.pinched and lost_for > settings.lost_timeout_ms:
                episodes.append(Episode(start_ms, last_seen_ms, END_HAND_LOST))
                hysteresis.reset()
            continue

        last_seen_ms = t
        value = measure_value(s, settings.measure)
        event = hysteresis.update(t, value, s.extensions, s.extensions_3d, s.palm_width,
                                  s.closing_speed, s.closing_speed_3d, s.wrist_speed)
        if event == "start":
            start_ms = t
        elif event == "end":
            episodes.append(Episode(start_ms, hysteresis.release_ms, END_RELEASED))

    if hysteresis.pinched and sequence:
        episodes.append(Episode(start_ms, sequence[-1].t_in_sequence_ms, END_SEQUENCE))
    return episodes


def count_episodes(
    samples: list[Sample], label: str, rule: str, thresholds: RuleThresholds, exit: float
) -> int:
    """Nombre de pincements détectés dans les prises `label` (sans délai ni garde-fou)."""
    settings = EpisodeSettings(rule, thresholds, exit)
    total = 0
    for sequence in split_sequences(samples, label):
        total += len(simulate_sequence(sequence, settings))
    return total


@dataclass
class EpisodeMetrics:
    """Ce que ressent l'utilisateur, pour un jeu de données et des réglages donnés."""

    ruptures: int  # pincement maintenu : sorties après la 1re seconde (idéal 0)
    false_clicks: int  # négatifs : faux épisodes courts (<= click_max_ms)
    false_drags: int  # négatifs : faux épisodes longs
    false_durations: list[int]  # durées des faux épisodes (ms)
    repeated_detected: int  # pincements répétés détectés...
    repeated_expected: int  # ... sur le nombre attendu (un par seconde)
    out_of_vocabulary: int  # épisodes dans le geste hors vocabulaire (information)


def ruptures_in(sequence: list[Sample], settings: EpisodeSettings) -> int:
    """Ruptures d'un pincement maintenu : fins d'épisode après la mise en place.

    Si le pincement n'est jamais actif après la première seconde, c'est aussi un échec : 1.
    """
    episodes = simulate_sequence(sequence, settings)
    ruptures = 0
    active_after_settle = False
    for e in episodes:
        if e.end_ms >= SETTLE_MS:
            active_after_settle = True
            if e.ended_by != END_SEQUENCE:
                ruptures += 1
    if not active_after_settle:
        ruptures += 1
    return ruptures


def episode_metrics(
    samples: list[Sample], settings: EpisodeSettings, click_max_ms: float, seconds: int
) -> EpisodeMetrics:
    ruptures = 0
    for sequence in split_sequences(samples, POSITIVE_LABEL):
        ruptures += ruptures_in(sequence, settings)

    durations: list[int] = []
    for label in FALSE_EPISODE_LABELS:
        for sequence in split_sequences(samples, label):
            for e in simulate_sequence(sequence, settings):
                durations.append(e.duration_ms)
    false_clicks = 0
    for d in durations:
        if d <= click_max_ms:
            false_clicks += 1

    repeated = 0
    repeated_sequences = 0
    for label in EPISODE_LABELS:
        for sequence in split_sequences(samples, label):
            repeated_sequences += 1
            repeated += len(simulate_sequence(sequence, settings))

    out_of_vocabulary = 0
    for label in OUT_OF_VOCABULARY_LABELS:
        for sequence in split_sequences(samples, label):
            out_of_vocabulary += len(simulate_sequence(sequence, settings))

    return EpisodeMetrics(
        ruptures=ruptures,
        false_clicks=false_clicks,
        false_drags=len(durations) - false_clicks,
        false_durations=durations,
        repeated_detected=repeated,
        repeated_expected=seconds * repeated_sequences,
        out_of_vocabulary=out_of_vocabulary,
    )


@dataclass
class SweepRow:
    settings: EpisodeSettings
    train: EpisodeMetrics
    test: EpisodeMetrics | None


def sweep(
    train: list[Sample],
    test: list[Sample] | None,
    configurations: list[EpisodeSettings],
    click_max_ms: float,
    seconds_train: int,
    seconds_test: int,
) -> list[SweepRow]:
    """Métriques par épisode de chaque configuration, sur l'entraînement (et le test)."""
    rows = []
    for settings in configurations:
        train_metrics = episode_metrics(train, settings, click_max_ms, seconds_train)
        test_metrics = None
        if test is not None:
            test_metrics = episode_metrics(test, settings, click_max_ms, seconds_test)
        rows.append(SweepRow(settings, train_metrics, test_metrics))
    return rows


MAX_FALSE_CLICKS = 1  # toléré (sur l'entraînement) pour une recommandation
MAX_FALSE_DRAGS = 0


def qualifies(m: EpisodeMetrics) -> bool:
    """Conditions minimales : aucun faux glisser, au plus un faux clic."""
    return m.false_drags <= MAX_FALSE_DRAGS and m.false_clicks <= MAX_FALSE_CLICKS


def ranking_key(row: SweepRow, ruptures_reliable: bool = True) -> tuple:
    """Clé de classement (plus petite = meilleure), calculée sur l'ENTRAÎNEMENT.

    - Ruptures fiables : le moins de ruptures, puis le plus de pincements répétés détectés.
    - Ruptures non fiables (l'utilisateur a peut-être relâché pendant le pincement maintenu) :
      le moins de faux glisser, puis de faux clics, puis le plus de répétés détectés.
    - Enfin, le plus petit délai de relâchement (moins de latence).
    """
    m = row.train
    grace = row.settings.release_grace_ms
    if ruptures_reliable:
        return (m.ruptures, -m.repeated_detected, grace)
    return (m.false_drags, m.false_clicks, -m.repeated_detected, grace)


def rank(rows: list[SweepRow], ruptures_reliable: bool = True) -> list[SweepRow]:
    """Configurations triées : d'abord celles qui remplissent les conditions, par classement."""
    def passing_key(row: SweepRow) -> tuple:
        return ranking_key(row, ruptures_reliable)

    def failing_key(row: SweepRow) -> tuple:
        # Les moins mauvaises d'abord : le moins de faux glisser, puis de faux clics.
        return (row.train.false_drags, row.train.false_clicks,
                ranking_key(row, ruptures_reliable))

    passing = []
    failing = []
    for row in rows:
        if qualifies(row.train):
            passing.append(row)
        else:
            failing.append(row)
    passing.sort(key=passing_key)
    failing.sort(key=failing_key)
    return passing + failing


def recommend(rows: list[SweepRow], ruptures_reliable: bool = True) -> SweepRow | None:
    """Configuration recommandée, ou None si aucune ne remplit les conditions.

    Filtres ET classement sur l'ENTRAÎNEMENT uniquement : avec des centaines de
    configurations, choisir d'après le test reviendrait à régler les paramètres sur le test,
    qui ne mesurerait plus rien. Les chiffres du test ne servent qu'à vérifier le choix.
    """
    ranked = rank(rows, ruptures_reliable)
    if ranked and qualifies(ranked[0].train):
        return ranked[0]
    return None


# --- Garde-fou de taille de main ---------------------------------------------------------------


def hand_size_ratios(sequence: list[Sample], window_ms: float) -> list[tuple[Sample, float]]:
    """(image, hand_size / médiane récente) pour chaque image avec main d'une prise."""
    guard = HandSizeGuard(0.0, window_ms)
    result = []
    for s in sequence:
        if not s.hand_present or s.hand_size is None:
            continue
        value = guard.ratio(s.t_in_sequence_ms, s.hand_size)
        if value is not None:
            result.append((s, value))
    return result


def propose_hand_size_ratio(
    samples: list[Sample], exit: float, window_ms: float
) -> Proposal | None:
    """Seuil de rapport de taille qui sépare les images saines des images « décrochées ».

    Dans le pincement maintenu (après la première seconde), la main pince par protocole :
    une image avec pinch_ratio >= exit est un décrochage du suivi. Si ces images ont une
    taille de main anormalement petite, un seuil sur le rapport les écartera.
    """
    healthy = []
    glitches = []
    for sequence in split_sequences(samples, POSITIVE_LABEL):
        for s, value in hand_size_ratios(sequence, window_ms):
            if s.t_in_sequence_ms < SETTLE_MS:
                continue
            if s.pinch_ratio >= exit:
                glitches.append(value)
            else:
                healthy.append(value)
    return propose_threshold(healthy, glitches)


def rejected_by_guard(
    samples: list[Sample], min_ratio: float, window_ms: float
) -> dict[str, tuple[int, int]]:
    """Par étiquette : (images écartées par le garde-fou, images avec main)."""
    result = {}
    for label in labels_in_order(samples):
        rejected = 0
        present = 0
        for sequence in split_sequences(samples, label):
            guard = HandSizeGuard(min_ratio, window_ms)
            for s in sequence:
                if not s.hand_present:
                    continue
                present += 1
                if s.hand_size is not None and not guard.accept(s.t_in_sequence_ms,
                                                                s.hand_size):
                    rejected += 1
        result[label] = (rejected, present)
    return result


# --- Proposition de seuils --------------------------------------------------------------------


@dataclass
class Proposal:
    """Seuil proposé pour une règle « valeur > seuil => positif »."""

    threshold: float  # milieu de l'écart entre P95(négatifs) et P5(positifs)
    margin: float  # P5(positifs) - P95(négatifs) ; négatif = les distributions se chevauchent
    p5_positives: float
    p95_negatives: float
    youden_threshold: float  # seuil qui maximise J = TPR - FPR
    youden_j: float
    n_positives: int
    n_negatives: int


def youden(positives: list[float], negatives: list[float]) -> tuple[float, float]:
    """Meilleur seuil au sens de l'indice de Youden J = TPR - FPR (Youden, 1950).

    On essaie comme seuil le milieu de chaque paire de valeurs consécutives (triées) et on
    garde celui qui maximise J. TPR = part des positifs au-dessus du seuil, FPR = part des
    négatifs au-dessus. J = 1 : séparation parfaite ; J = 0 : pas mieux que le hasard.
    """
    values = sorted(set(positives) | set(negatives))
    best_threshold = values[0]
    best_j = -1.0
    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2
        tp = 0
        for v in positives:
            if v > threshold:
                tp += 1
        fp = 0
        for v in negatives:
            if v > threshold:
                fp += 1
        j = tp / len(positives) - fp / len(negatives)
        if j > best_j:
            best_j = j
            best_threshold = threshold
    return best_threshold, best_j


def propose_threshold(positives: list[float], negatives: list[float]) -> Proposal | None:
    """Seuil qui sépare des valeurs positives (hautes) de valeurs négatives (basses).

    Marge = P5(positifs) - P95(négatifs) : en écartant 5 % de valeurs extrêmes de chaque
    côté, quel espace libre reste-t-il entre les deux groupes ? Le seuil proposé est au milieu,
    pour laisser autant de marge de chaque côté. Renvoie None s'il manque un des deux groupes.
    """
    if not positives or not negatives:
        return None
    p5 = percentile(positives, 5)
    p95 = percentile(negatives, 95)
    youden_threshold, youden_j = youden(positives, negatives)
    return Proposal(
        threshold=(p5 + p95) / 2,
        margin=p5 - p95,
        p5_positives=p5,
        p95_negatives=p95,
        youden_threshold=youden_threshold,
        youden_j=youden_j,
        n_positives=len(positives),
        n_negatives=len(negatives),
    )


def propose_thresholds(samples: list[Sample], enter: float) -> dict[str, Proposal | None]:
    """Seuils proposés pour les règles B, C et D (2D et 3D), sur les données d'entraînement.

    Seules comptent les images que la règle A accepte (pinch_ratio < enter) : ce sont les
    cas où il faut un critère de plus. Positifs = pincement maintenu ; négatifs = images
    négatives (poing, parasites...) qui passent la règle A, donc les faux positifs actuels.
    """
    positives = []
    negatives = []
    for s in frame_samples(samples, POSITIVE_LABEL):
        if s.pinch_ratio < enter:
            positives.append(s)
    for label in NEGATIVE_LABELS:
        for s in frame_samples(samples, label):
            if s.pinch_ratio < enter:
                negatives.append(s)

    # Seuil à proposer -> descripteur qu'il coupe.
    # « palm_facing » n'est pas un réglage : proposé seulement pour comparer avec palm_width.
    threshold_descriptors = {
        "min_index_extension": "ext_index",
        "folded_below": "ext_autres_max",
        "min_index_extension_3d": "ext_index_3d",
        "folded_below_3d": "ext_autres_max_3d",
        "min_palm_width": "palm_width",
        "palm_facing (comparaison)": "palm_facing",
    }
    proposals: dict[str, Proposal | None] = {}
    for threshold_name, descriptor in threshold_descriptors.items():
        pos_values = []
        for s in positives:
            value = descriptor_value(descriptor, s)
            if value is not None:
                pos_values.append(value)
        neg_values = []
        for s in negatives:
            value = descriptor_value(descriptor, s)
            if value is not None:
                neg_values.append(value)
        proposals[threshold_name] = propose_threshold(pos_values, neg_values)
    return proposals


@dataclass
class Separation:
    """Meilleure séparation d'un indice entre deux groupes, dans le bon sens."""

    descriptor: str
    higher_is_pinch: bool  # True : pincement si valeur > seuil ; False : si valeur < seuil
    proposal: Proposal


def best_separation(
    descriptor: str, pinch_values: list[float], other_values: list[float]
) -> Separation | None:
    """Essaie les deux sens (« pincement = valeur haute » et « = valeur basse ») et garde celui
    qui a le meilleur indice de Youden. Pour le sens « bas », on travaille sur les valeurs
    opposées (−x), puis on remet le seuil dans le bon signe.
    """
    if not pinch_values or not other_values:
        return None
    high = propose_threshold(pinch_values, other_values)
    negated_pinch = []
    for v in pinch_values:
        negated_pinch.append(-v)
    negated_other = []
    for v in other_values:
        negated_other.append(-v)
    low = propose_threshold(negated_pinch, negated_other)
    if high.youden_j >= low.youden_j:
        return Separation(descriptor, True, high)
    # Remise dans le bon signe : seuil et percentiles changent de signe.
    flipped = Proposal(
        threshold=-low.threshold,
        margin=low.margin,
        p5_positives=-low.p5_positives,  # = P95 des valeurs de pincement
        p95_negatives=-low.p95_negatives,  # = P5 des autres valeurs
        youden_threshold=-low.youden_threshold,
        youden_j=low.youden_j,
        n_positives=low.n_positives,
        n_negatives=low.n_negatives,
    )
    return Separation(descriptor, False, flipped)


PROFILE_LABEL = "main_repos_profil"
PROFILE_DESCRIPTORS = ["pinch_ratio", "pinch_seg", "palm_width", "palm_facing", "ext_autres_max",
                       "wrist_y"]


def profile_separations(samples: list[Sample], pinch_label: str) -> list[Separation]:
    """Pour chaque indice, séparation entre `pinch_label` et la main au repos de profil,
    classée du meilleur (Youden le plus haut) au moins bon. Liste vide sans l'étiquette."""
    pinch = frame_samples(samples, pinch_label)
    profile = frame_samples(samples, PROFILE_LABEL)
    if not pinch or not profile:
        return []
    results = []
    for name in PROFILE_DESCRIPTORS:
        pinch_values = []
        for s in pinch:
            value = descriptor_value(name, s)
            if value is not None:
                pinch_values.append(value)
        other_values = []
        for s in profile:
            value = descriptor_value(name, s)
            if value is not None:
                other_values.append(value)
        separation = best_separation(name, pinch_values, other_values)
        if separation is not None:
            results.append(separation)

    def youden_of(separation: Separation) -> float:
        return separation.proposal.youden_j

    results.sort(key=youden_of, reverse=True)
    return results


def thresholds_from_proposals(
    proposals: dict[str, Proposal | None], current: RuleThresholds
) -> RuleThresholds:
    """Seuils proposés quand ils existent, sinon les seuils actuels.

    Pour min_palm_width, la proposition n'est retenue que si elle sépare vraiment (marge > 0) :
    sinon on garde la valeur actuelle (provisoire). La condition de paume reste activée ou non
    comme dans `current`.
    """
    values = {}
    for name in ["min_index_extension", "folded_below", "min_index_extension_3d",
                 "folded_below_3d", "min_palm_width"]:
        proposal = proposals.get(name)
        if proposal is None or (name == "min_palm_width" and proposal.margin <= 0):
            values[name] = getattr(current, name)
        else:
            values[name] = proposal.threshold
    return RuleThresholds(enter=current.enter, require_palm_facing=current.require_palm_facing,
                          **values)


# --- Main de profil vers la gauche (protocole v4) -----------------------------------------------
#
# Usage réel : en pointage absolu, la main droite qui vise la GAUCHE de l'écran pivote avec
# l'avant-bras et passe de profil ; elle fait alors de faux pincements, comme la main au repos
# de profil. Hypothèses : en 3D, pouce et index ne se touchent pas (pinch_ratio_3d) ; et un
# vrai pincement FERME activement les doigts (closing_speed), un pivot non.

ENTRY_DESCRIPTORS = ["pinch_ratio", "pinch_ratio_3d", "closing_speed", "closing_speed_3d",
                     "palm_width", "wrist_speed"]


def entry_value(name: str, s: Sample) -> float | None:
    if name == "pinch_ratio":
        return s.pinch_ratio
    if name == "pinch_ratio_3d":
        return s.pinch_ratio_3d
    if name == "closing_speed":
        return s.closing_speed
    if name == "closing_speed_3d":
        return s.closing_speed_3d
    if name == "palm_width":
        return s.palm_width
    if name == "wrist_speed":
        return s.wrist_speed
    raise ValueError(f"Descripteur inconnu : {name}")


def entry_moments(samples: list[Sample], label: str, enter: float) -> list[Sample]:
    """Images où pinch_ratio PASSE sous `enter` (le moment où la règle A entrerait).

    On compare les descripteurs à ces moments plutôt qu'image par image : la vitesse de
    fermeture n'a de sens qu'au moment où les doigts se rapprochent.
    """
    moments = []
    for sequence in split_sequences(samples, label):
        above = True  # au début de la prise, on considère les doigts écartés
        for s in sequence:
            if not s.hand_present:
                above = True
                continue
            if s.pinch_ratio < enter:
                if above:
                    moments.append(s)
                above = False
            else:
                above = True
    return moments


def entry_values(moments: list[Sample], name: str) -> list[float]:
    values = []
    for s in moments:
        value = entry_value(name, s)
        if value is not None:
            values.append(value)
    return values


def entry_separations(samples: list[Sample], enter: float) -> list[Separation]:
    """Vrais pincements (répétés, au centre et à gauche) contre main de profil (repos et
    va-et-vient), au moment de l'entrée ; du meilleur indice (Youden) au moins bon."""
    pinch_moments = []
    for label in EPISODE_LABELS:
        pinch_moments += entry_moments(samples, label, enter)
    profile_moments = []
    for label in PROFILE_NEGATIVE_LABELS:
        profile_moments += entry_moments(samples, label, enter)
    results = []
    for name in ENTRY_DESCRIPTORS:
        separation = best_separation(name, entry_values(pinch_moments, name),
                                     entry_values(profile_moments, name))
        if separation is not None:
            results.append(separation)

    def youden_of(separation: Separation) -> float:
        return separation.proposal.youden_j

    results.sort(key=youden_of, reverse=True)
    return results


@dataclass
class EntryDetail:
    """Passages sous enter d'une étiquette, et médianes des indices à ces moments."""

    label: str
    count: int
    medians: dict[str, float | None]


def entry_details(samples: list[Sample], labels: list[str], enter: float) -> list[EntryDetail]:
    details = []
    for label in labels:
        moments = entry_moments(samples, label, enter)
        medians: dict[str, float | None] = {}
        for name in ENTRY_DESCRIPTORS:
            values = entry_values(moments, name)
            medians[name] = statistics.median(values) if values else None
        details.append(EntryDetail(label, len(moments), medians))
    return details


@dataclass
class ProfileMetrics:
    """Ce que donne une règle face à la main de profil."""

    false_clicks: int  # sur main_repos_profil, main_repos_profil_gauche, va_et_vient_gauche
    false_drags: int
    detected: dict[str, tuple[int, int]]  # pincements répétés : (détectés, attendus)


def profile_metrics(
    samples: list[Sample], settings: EpisodeSettings, click_max_ms: float, seconds: int
) -> ProfileMetrics:
    clicks = 0
    drags = 0
    for label in PROFILE_NEGATIVE_LABELS:
        for sequence in split_sequences(samples, label):
            for e in simulate_sequence(sequence, settings):
                if e.duration_ms <= click_max_ms:
                    clicks += 1
                else:
                    drags += 1
    detected = {}
    for label in EPISODE_LABELS:
        sequences = split_sequences(samples, label)
        found = 0
        for sequence in sequences:
            found += len(simulate_sequence(sequence, settings))
        detected[label] = (found, seconds * len(sequences))
    return ProfileMetrics(clicks, drags, detected)


def has_profile_data(samples: list[Sample]) -> bool:
    """Vrai si le jeu de données contient au moins une séquence du protocole v4."""
    labels = labels_in_order(samples)
    return "pincements_repetes_gauche" in labels or "va_et_vient_gauche" in labels
