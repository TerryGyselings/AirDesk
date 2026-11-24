"""Analyse d'un jeu de données étiqueté (scripts/record_gestures.py) : logique pure, testée.

Question posée : quelle règle de détection du pincement (voir interpretation/pinch_rules.py)
reconnaît les vrais pincements sans se déclencher sur un poing ou un geste parasite ?

Vérité terrain :
- positif image par image : « pincement_maintenu » (sans la première seconde, le temps de
  prendre la pose) ;
- négatifs image par image : main ouverte, poing, poing qui s'ouvre et se ferme, main
  relâchée, gestes parasites ;
- « pincements_repetes » et « pincement_vers_camera » alternent pincé / relâché au rythme d'un
  métronome (un pincement par seconde) : pas de vérité image par image, on y compte les
  ÉPISODES détectés et on les compare au nombre attendu.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

from airdesk.interpretation.pinch_rules import RuleThresholds, pinch_candidate
from airdesk.log_analysis import percentile

POSITIVE_LABEL = "pincement_maintenu"
NEGATIVE_LABELS = ["main_ouverte", "poing", "poing_ouvre_ferme", "main_relachee", "parasites"]
EPISODE_LABELS = ["pincements_repetes", "pincement_vers_camera"]
SETTLE_MS = 1000  # début de « pincement_maintenu » ignoré (mise en place du geste)

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
            samples.append(
                Sample(
                    label=record["label"],
                    t_in_sequence_ms=int(record["t_in_sequence_ms"]),
                    pinch_ratio=float(record["pinch_ratio"]),
                    extensions=(ext[0], ext[1], ext[2], ext[3]),
                    extensions_3d=(ext3d[0], ext3d[1], ext3d[2], ext3d[3]),
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
# petite si tous ces doigts sont repliés (poing), c'est ce que teste la règle C.
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


# --- Règles : matrice de confusion et épisodes -------------------------------------------------


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


def count_episodes(
    samples: list[Sample], label: str, rule: str, thresholds: RuleThresholds, exit: float
) -> int:
    """Nombre de pincements détectés dans les séquences `label`, avec hystérésis.

    Entrée : la règle dit « pincement ». Sortie : pinch_ratio >= exit, ou main perdue.
    Une nouvelle séquence (t_in_sequence_ms qui repart en arrière) remet l'état à zéro.
    """
    episodes = 0
    pinched = False
    previous_t = -1
    for s in samples:
        if s.label != label:
            continue
        if s.t_in_sequence_ms < previous_t:  # nouvelle séquence (autre fichier, autre prise)
            pinched = False
        previous_t = s.t_in_sequence_ms
        if not s.hand_present:
            pinched = False
            continue
        if pinched:
            if s.pinch_ratio >= exit:
                pinched = False
        elif detects(rule, s, thresholds):
            pinched = True
            episodes += 1
    return episodes


def count_sequences(samples: list[Sample], label: str) -> int:
    """Nombre de prises de l'étiquette (une par fichier, en pratique)."""
    count = 0
    previous_t = None
    for s in samples:
        if s.label != label:
            continue
        if previous_t is None or s.t_in_sequence_ms < previous_t:
            count += 1
        previous_t = s.t_in_sequence_ms
    return count


# --- Proposition de seuils --------------------------------------------------------------------


@dataclass
class Proposal:
    """Seuil proposé pour une règle « valeur > seuil => pincement »."""

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
    """Seuils proposés pour les règles B et C (2D et 3D), à partir des données d'entraînement.

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
