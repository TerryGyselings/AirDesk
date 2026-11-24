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

from airdesk.interpretation.pinch_rules import PinchHysteresis, RuleThresholds, pinch_candidate
from airdesk.interpretation.quality import HandSizeGuard
from airdesk.log_analysis import percentile

POSITIVE_LABEL = "pincement_maintenu"
NEGATIVE_LABELS = ["main_ouverte", "poing", "poing_ouvre_ferme", "main_relachee", "parasites"]
# Négatifs dont on compte les faux épisodes (la main ouverte sert à l'engagement, elle est
# vue à part dans la matrice image par image).
FALSE_EPISODE_LABELS = ["poing", "poing_ouvre_ferme", "main_relachee", "parasites"]
EPISODE_LABELS = ["pincements_repetes"]
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

    @property
    def hand_present(self) -> bool:
        return self.pinch_ratio is not None


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
            samples.append(
                Sample(
                    label=record["label"],
                    t_in_sequence_ms=int(record["t_in_sequence_ms"]),
                    pinch_ratio=float(record["pinch_ratio"]),
                    extensions=(ext[0], ext[1], ext[2], ext[3]),
                    extensions_3d=(ext3d[0], ext3d[1], ext3d[2], ext3d[3]),
                    hand_size=hand_size,
                )
            )
    return samples


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
DESCRIPTORS = ["pinch_ratio", "ext_index", "ext_index_3d", "ext_autres_max", "ext_autres_max_3d"]


def descriptor_value(name: str, s: Sample) -> float:
    """Valeur du descripteur `name` pour une image (main présente)."""
    if name == "pinch_ratio":
        return s.pinch_ratio
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
                values.append(descriptor_value(name, s))
            result[label][name] = describe(values)
    return result


# --- Règles : matrice de confusion image par image ---------------------------------------------


def detects(rule: str, s: Sample, thresholds: RuleThresholds) -> bool:
    return pinch_candidate(rule, s.pinch_ratio, s.extensions, s.extensions_3d, thresholds)


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
    """Réglages de la simulation : ce qu'on fait varier dans le balayage."""

    rule: str
    thresholds: RuleThresholds
    exit: float
    release_grace_ms: float = 0
    guard_ratio: float = 0.0  # 0 = garde-fou désactivé
    window_ms: float = 2000
    lost_timeout_ms: float = LOST_TIMEOUT_MS


def simulate_sequence(sequence: list[Sample], settings: EpisodeSettings) -> list[Episode]:
    """Épisodes de pincement d'une prise, en imitant la machine à états.

    - Entrée / sortie : PinchHysteresis (règle, exit, délai de relâchement).
    - Une image sans main, ou écartée par le garde-fou de taille, ne relâche PAS le pincement
      (la machine à états ne réagit pas à une image sans main) ; seule une absence de plus de
      `lost_timeout_ms` y met fin.
    """
    hysteresis = PinchHysteresis(
        settings.rule, settings.thresholds, settings.exit, settings.release_grace_ms
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
        event = hysteresis.update(t, s.pinch_ratio, s.extensions, s.extensions_3d)
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
    rule: str
    exit: float
    release_grace_ms: float
    train: EpisodeMetrics
    test: EpisodeMetrics | None


def sweep(
    train: list[Sample],
    test: list[Sample] | None,
    rules: list[str],
    exits: list[float],
    graces: list[float],
    thresholds: RuleThresholds,
    guard_ratio: float,
    window_ms: float,
    click_max_ms: float,
    seconds_train: int,
    seconds_test: int,
) -> list[SweepRow]:
    """Toutes les combinaisons règle × exit × délai de relâchement, sur train (et test)."""
    rows = []
    for rule in rules:
        for exit_value in exits:
            for grace in graces:
                settings = EpisodeSettings(rule, thresholds, exit_value, grace, guard_ratio,
                                           window_ms)
                train_metrics = episode_metrics(train, settings, click_max_ms, seconds_train)
                test_metrics = None
                if test is not None:
                    test_metrics = episode_metrics(test, settings, click_max_ms, seconds_test)
                rows.append(SweepRow(rule, exit_value, grace, train_metrics, test_metrics))
    return rows


MAX_FALSE_CLICKS = 1  # toléré sur le test pour une recommandation
MAX_FALSE_DRAGS = 0


def recommend(rows: list[SweepRow]) -> SweepRow | None:
    """Configuration recommandée, ou None si aucune ne remplit les conditions.

    Conditions, sur le TEST (sur l'entraînement s'il n'y a pas de test) : aucun faux glisser
    et au plus un faux clic. Parmi elles : le moins de ruptures, puis le plus de pincements
    répétés détectés, puis le plus petit délai de relâchement (moins de latence).
    """
    best = None
    best_key = None
    for row in rows:
        m = row.test if row.test is not None else row.train
        if m.false_drags > MAX_FALSE_DRAGS or m.false_clicks > MAX_FALSE_CLICKS:
            continue
        key = (m.ruptures, -m.repeated_detected, row.release_grace_ms)
        if best_key is None or key < best_key:
            best = row
            best_key = key
    return best


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
    threshold_descriptors = {
        "min_index_extension": "ext_index",
        "folded_below": "ext_autres_max",
        "min_index_extension_3d": "ext_index_3d",
        "folded_below_3d": "ext_autres_max_3d",
    }
    proposals: dict[str, Proposal | None] = {}
    for threshold_name, descriptor in threshold_descriptors.items():
        pos_values = []
        for s in positives:
            pos_values.append(descriptor_value(descriptor, s))
        neg_values = []
        for s in negatives:
            neg_values.append(descriptor_value(descriptor, s))
        proposals[threshold_name] = propose_threshold(pos_values, neg_values)
    return proposals


def thresholds_from_proposals(
    proposals: dict[str, Proposal | None], current: RuleThresholds
) -> RuleThresholds:
    """Seuils proposés quand ils existent, sinon les seuils actuels."""
    values = {}
    for name in ["min_index_extension", "folded_below", "min_index_extension_3d",
                 "folded_below_3d"]:
        proposal = proposals.get(name)
        if proposal is None:
            values[name] = getattr(current, name)
        else:
            values[name] = proposal.threshold
    return RuleThresholds(enter=current.enter, **values)
