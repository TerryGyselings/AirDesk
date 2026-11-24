"""Analyse des jeux de données étiquetés : quelle règle détecte le pincement sans le poing ?

Usage (venv activé, depuis la racine du projet) :

    python scripts/analyse_gestures.py data/datasets/A.csv data/datasets/B.csv \
        --test data/datasets/C.csv

- Fichiers donnés sans option : ENTRAÎNEMENT. Les seuils y sont proposés.
- --test : un fichier enregistré à part (autre session, idéalement autre participant). Les
  règles y sont évaluées avec les seuils appris sur l'entraînement, SANS les recalculer :
  c'est l'estimation honnête de ce que donnera la règle sur de nouvelles données.

Les seuils en vigueur (enter, exit...) sont lus dans le .json de chaque jeu de données.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from airdesk.config import load_config
from airdesk.gesture_analysis import (
    DESCRIPTORS,
    EPISODE_LABELS,
    NEGATIVE_LABELS,
    POSITIVE_LABEL,
    confusion,
    count_episodes,
    count_sequences,
    detection_rate,
    labels_in_order,
    propose_thresholds,
    read_dataset,
    stats_by_label,
    thresholds_from_proposals,
)
from airdesk.interpretation.pinch_rules import RULES, RuleThresholds
from airdesk.log_analysis import read_metadata

SHORT_NAMES = {
    "main_ouverte": "ouverte",
    "poing": "poing",
    "poing_ouvre_ferme": "poing o/f",
    "main_relachee": "relachee",
    "parasites": "parasites",
    "pincements_repetes": "repetes",
    "pincement_vers_camera": "vers cam.",
}


def load_settings(csv_paths: list[Path]) -> tuple[RuleThresholds, float, int]:
    """(seuils actuels, exit, durée d'une séquence en s) d'après les .json des jeux de données."""
    found = []
    for path in csv_paths:
        json_path = path.with_suffix(".json")
        if json_path.exists():
            found.append(read_metadata(json_path))
        else:
            print(f"ATTENTION : {json_path.name} introuvable.")
    if found:
        config = found[0]["config"]
        seconds = found[0].get("dataset", {}).get("seconds", 10)
        for other in found[1:]:
            if other["config"]["pinch"] != config["pinch"] or \
                    other["config"]["fingers"] != config["fingers"]:
                print("ATTENTION : les fichiers n'ont pas été enregistrés avec les mêmes seuils ;"
                      " ceux du premier sont utilisés.")
        pinch = config["pinch"]
        fingers = config["fingers"]
    else:
        print("ATTENTION : aucun .json, seuils lus dans la config ACTUELLE.")
        cfg = load_config()
        pinch = {"enter": cfg.pinch.enter, "exit": cfg.pinch.exit,
                 "min_index_extension": cfg.pinch.min_index_extension,
                 "min_index_extension_3d": cfg.pinch.min_index_extension_3d}
        fingers = {"folded_below": cfg.fingers.folded_below,
                   "folded_below_3d": cfg.fingers.folded_below_3d}
        seconds = 10
    thresholds = RuleThresholds(
        enter=pinch["enter"],
        min_index_extension=pinch["min_index_extension"],
        folded_below=fingers["folded_below"],
        min_index_extension_3d=pinch["min_index_extension_3d"],
        folded_below_3d=fingers["folded_below_3d"],
    )
    return thresholds, pinch["exit"], seconds


def print_distributions(samples) -> None:
    labels = labels_in_order(samples)
    print("Détection de la main par étiquette :")
    for label in labels:
        print(f"  {label:<22} {detection_rate(samples, label) * 100:5.1f} %")
    stats = stats_by_label(samples)
    for name in DESCRIPTORS:
        print(f"\n{name} :")
        print(f"  {'étiquette':<22} {'n':>5} {'min':>6} {'P5':>6} {'méd.':>6} "
              f"{'P95':>6} {'max':>6}")
        for label in labels:
            s = stats[label][name]
            if s is None:
                print(f"  {label:<22} {0:>5}")
                continue
            print(f"  {label:<22} {s.n:>5} {s.minimum:6.2f} {s.p5:6.2f} {s.median:6.2f} "
                  f"{s.p95:6.2f} {s.maximum:6.2f}")


def print_rules(samples, thresholds: RuleThresholds, exit_: float, seconds: int) -> None:
    header = f"  {'règle':<5} | {'rappel pincement':<17}"
    for label in NEGATIVE_LABELS:
        header += f" | FP {SHORT_NAMES[label]:<9}"
    for label in EPISODE_LABELS:
        header += f" | {SHORT_NAMES[label]:<9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rule in RULES:
        c = confusion(samples, rule, thresholds)
        total = c.true_positives + c.false_negatives
        line = f"  {rule:<5} | {c.true_positives:>4}/{total:<4} {c.recall * 100:5.1f} % "
        for label in NEGATIVE_LABELS:
            cell = f"{c.false_positives[label]}/{c.negatives[label]}"
            line += f" | {cell:<12}"
        for label in EPISODE_LABELS:
            expected = seconds * count_sequences(samples, label)
            found = count_episodes(samples, label, rule, thresholds, exit_)
            line += f" | {f'{found}/{expected}':<9}"
        print(line)
    print("  rappel = images de pincement maintenu reconnues ; FP x/n = faux pincements sur n")
    print("  images ; colonnes répétés / vers caméra = épisodes détectés / pincements attendus.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparaison des règles de pincement.")
    parser.add_argument("train", type=Path, nargs="+", help="jeux de données d'entraînement")
    parser.add_argument("--test", type=Path, default=None, help="jeu de données de test")
    parser.add_argument("--enter", type=float, default=None, help="tester une autre valeur")
    parser.add_argument("--exit", type=float, default=None, help="tester une autre valeur")
    args = parser.parse_args()

    current, exit_, seconds = load_settings(args.train)
    if args.enter is not None:
        current = RuleThresholds(args.enter, current.min_index_extension, current.folded_below,
                                 current.min_index_extension_3d, current.folded_below_3d)
    if args.exit is not None:
        exit_ = args.exit

    train = []
    for path in args.train:
        train += read_dataset(path)
    print(f"ENTRAÎNEMENT : {', '.join(str(p) for p in args.train)} ({len(train)} images)")
    if args.test is not None:
        print(f"TEST         : {args.test}")
    else:
        print("\nATTENTION : pas de fichier --test. Les résultats ci-dessous sont OPTIMISTES :")
        print("les seuils sont évalués sur les données mêmes qui ont servi à les choisir.")
    print(f"\nSeuils en vigueur à l'enregistrement : enter={current.enter}, exit={exit_}, "
          f"min_index_extension={current.min_index_extension} "
          f"(3D {current.min_index_extension_3d}), folded_below={current.folded_below} "
          f"(3D {current.folded_below_3d})")

    print("\n================ ENTRAÎNEMENT : distributions ================\n")
    print_distributions(train)

    print("\n================ ENTRAÎNEMENT : seuils proposés ================")
    print("Parmi les images que la règle A accepte (pinch_ratio < enter) : positifs = pincement")
    print("maintenu, négatifs = poing, parasites... (les faux pincements actuels).\n")
    proposals = propose_thresholds(train, current.enter)
    for name, p in proposals.items():
        if p is None:
            print(f"  {name:<24} : pas de proposition (aucun faux positif de la règle A à "
                  "corriger, ou aucun pincement maintenu)")
            continue
        verdict = "OK" if p.margin > 0 else "CHEVAUCHEMENT : pas de seuil sans erreur"
        print(f"  {name:<24} : {p.threshold:.2f}  (marge {p.margin:+.2f} ; P5 pincement "
              f"{p.p5_positives:.2f}, P95 négatifs {p.p95_negatives:.2f} ; Youden "
              f"{p.youden_threshold:.2f}, J={p.youden_j:.2f} ; n={p.n_positives}/{p.n_negatives})"
              f"  {verdict}")
    stats = stats_by_label(train)
    positive_ratio = stats.get(POSITIVE_LABEL, {}).get("pinch_ratio")
    if positive_ratio is not None and positive_ratio.p95 >= current.enter:
        print(f"\n  Contrôle de enter : P95 du pinch_ratio en pincement maintenu = "
              f"{positive_ratio.p95:.2f} >= enter ({current.enter}) : des pincements réels "
              "passent au-dessus du seuil d'entrée.")

    learned = thresholds_from_proposals(proposals, current)
    print(f"\nSeuils utilisés pour les règles B/C : min_index_extension="
          f"{learned.min_index_extension:.2f} (3D {learned.min_index_extension_3d:.2f}), "
          f"folded_below={learned.folded_below:.2f} (3D {learned.folded_below_3d:.2f})")

    print("\n================ ENTRAÎNEMENT : règles ================\n")
    print_rules(train, learned, exit_, seconds)

    if args.test is not None:
        test = read_dataset(args.test)
        _, _, test_seconds = load_settings([args.test])  # durée des séquences du fichier test
        print(f"\n================ TEST ({len(test)} images, mêmes seuils) ================\n")
        print_rules(test, learned, exit_, test_seconds)


if __name__ == "__main__":
    main()
