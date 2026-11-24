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
from dataclasses import dataclass
from pathlib import Path

from airdesk.config import load_config
from airdesk.gesture_analysis import (
    DESCRIPTORS,
    EPISODE_LABELS,
    NEGATIVE_LABELS,
    OUT_OF_VOCABULARY_LABELS,
    POSITIVE_LABEL,
    SweepRow,
    confusion,
    count_episodes,
    count_sequences,
    detection_rate,
    labels_in_order,
    propose_hand_size_ratio,
    propose_thresholds,
    read_dataset,
    recommend,
    rejected_by_guard,
    stats_by_label,
    sweep,
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
    "pincement_vers_camera": "hors voc.",
}

# Balayage des métriques par épisode.
SWEEP_RULES = ["A", "D", "D3D"]
SWEEP_EXITS = [0.40, 0.50]
SWEEP_GRACES = [0, 66, 100, 150]


@dataclass
class Settings:
    """Réglages lus dans les .json des jeux de données."""

    thresholds: RuleThresholds
    exit: float
    click_max_ms: float
    window_ms: float
    seconds: int
    participants: list[str]


def load_settings(csv_paths: list[Path]) -> Settings:
    found = []
    for path in csv_paths:
        json_path = path.with_suffix(".json")
        if json_path.exists():
            found.append(read_metadata(json_path))
        else:
            print(f"ATTENTION : {json_path.name} introuvable.")
    if found:
        config = found[0]["config"]
        for other in found[1:]:
            if other["config"]["pinch"] != config["pinch"] or \
                    other["config"]["fingers"] != config["fingers"]:
                print("ATTENTION : les fichiers n'ont pas été enregistrés avec les mêmes seuils ;"
                      " ceux du premier sont utilisés.")
        pinch = config["pinch"]
        fingers = config["fingers"]
        quality = config.get("quality", {})
        seconds = found[0].get("dataset", {}).get("seconds", 10)
        participants = []
        for metadata in found:
            participants.append(metadata.get("dataset", {}).get("participant", "?"))
    else:
        print("ATTENTION : aucun .json, seuils lus dans la config ACTUELLE.")
        cfg = load_config()
        pinch = {"enter": cfg.pinch.enter, "exit": cfg.pinch.exit,
                 "click_max_ms": cfg.pinch.click_max_ms,
                 "min_index_extension": cfg.pinch.min_index_extension,
                 "min_index_extension_3d": cfg.pinch.min_index_extension_3d}
        fingers = {"folded_below": cfg.fingers.folded_below,
                   "folded_below_3d": cfg.fingers.folded_below_3d}
        quality = {"hand_size_window_ms": cfg.quality.hand_size_window_ms}
        seconds = 10
        participants = ["?"]
    thresholds = RuleThresholds(
        enter=pinch["enter"],
        min_index_extension=pinch["min_index_extension"],
        folded_below=fingers["folded_below"],
        min_index_extension_3d=pinch["min_index_extension_3d"],
        folded_below_3d=fingers["folded_below_3d"],
    )
    return Settings(
        thresholds=thresholds,
        exit=pinch["exit"],
        click_max_ms=pinch["click_max_ms"],
        window_ms=quality.get("hand_size_window_ms", 2000),
        seconds=seconds,
        participants=participants,
    )


def title(text: str) -> None:
    print(f"\n================ {text} ================\n")


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
    """Matrice image par image + épisodes (sans délai ni garde-fou) pour chaque règle."""
    header = f"  {'règle':<5} | {'rappel pincement':<17}"
    for label in NEGATIVE_LABELS:
        header += f" | FP {SHORT_NAMES[label]:<9}"
    for label in EPISODE_LABELS + OUT_OF_VOCABULARY_LABELS:
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
        for label in OUT_OF_VOCABULARY_LABELS:
            found = count_episodes(samples, label, rule, thresholds, exit_)
            line += f" | {found:<9}"
        print(line)
    print("  rappel = images de pincement maintenu reconnues ; FP x/n = faux pincements sur n")
    print("  images ; repetes = épisodes détectés / attendus ; hors voc. = épisodes dans le")
    print("  pincement index vers la caméra (hors vocabulaire, pour information).")


def pair(train_value, test_value) -> str:
    if test_value is None:
        return f"{train_value}"
    return f"{train_value} / {test_value}"


def print_sweep(rows: list[SweepRow], has_test: bool) -> None:
    print(f"  {'règle':<5} {'exit':>5} {'délai':>6} | {'ruptures':<9} | {'faux clics':<10} | "
          f"{'faux gliss.':<11} | {'répétés':<13} | {'hors voc.':<9}")
    print("  " + "-" * 90)
    for row in rows:
        tr = row.train
        te = row.test
        ruptures = pair(tr.ruptures, te.ruptures if te else None)
        clicks = pair(tr.false_clicks, te.false_clicks if te else None)
        drags = pair(tr.false_drags, te.false_drags if te else None)
        repeated_test = None
        if te is not None:
            repeated_test = f"{te.repeated_detected}/{te.repeated_expected}"
        repeated = pair(f"{tr.repeated_detected}/{tr.repeated_expected}", repeated_test)
        out_of_vocabulary = pair(tr.out_of_vocabulary, te.out_of_vocabulary if te else None)
        print(
            f"  {row.rule:<5} {row.exit:5.2f} {row.release_grace_ms:>4} ms"
            f" | {ruptures:<9} | {clicks:<10} | {drags:<11} | {repeated:<13}"
            f" | {out_of_vocabulary:<9}"
        )
    where = "entraînement / test" if has_test else "entraînement"
    print(f"  Cellules : {where}. ruptures = sorties de pincement pendant le pincement maintenu")
    print("  (idéal 0) ; faux clics / faux glisser = faux épisodes courts / longs sur poing,")
    print("  poing o/f, main relâchée, parasites ; répétés = détectés / attendus.")

    best = recommend(rows)
    on = "le TEST" if has_test else "l'ENTRAÎNEMENT (optimiste : pas de test)"
    print(f"\n  Recommandation (sur {on} : 0 faux glisser, au plus 1 faux clic ; puis le moins")
    print("  de ruptures, le plus de répétés détectés, le plus petit délai) :")
    if best is None:
        print("  -> AUCUNE configuration ne remplit les conditions.")
        return
    tr = best.train
    te = best.test
    print(f"  -> règle {best.rule}, exit {best.exit:.2f}, release_grace_ms {best.release_grace_ms}")
    print(f"     entraînement : {tr.ruptures} rupture(s), {tr.false_clicks} faux clic(s), "
          f"{tr.false_drags} faux glisser, répétés {tr.repeated_detected}/"
          f"{tr.repeated_expected}")
    if te is not None:
        print(f"     test         : {te.ruptures} rupture(s), {te.false_clicks} faux clic(s), "
              f"{te.false_drags} faux glisser, répétés {te.repeated_detected}/"
              f"{te.repeated_expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparaison des règles de pincement.")
    parser.add_argument("train", type=Path, nargs="+", help="jeux de données d'entraînement")
    parser.add_argument("--test", type=Path, default=None, help="jeu de données de test")
    parser.add_argument("--enter", type=float, default=None, help="tester une autre valeur")
    parser.add_argument("--exit", type=float, default=None, help="tester une autre valeur")
    args = parser.parse_args()

    settings = load_settings(args.train)
    current = settings.thresholds
    if args.enter is not None:
        current = RuleThresholds(args.enter, current.min_index_extension, current.folded_below,
                                 current.min_index_extension_3d, current.folded_below_3d)
    exit_ = settings.exit if args.exit is None else args.exit

    train = []
    for path in args.train:
        train += read_dataset(path)
    test = None
    test_settings = None
    print(f"ENTRAÎNEMENT : {', '.join(str(p) for p in args.train)} ({len(train)} images)")
    if args.test is not None:
        test = read_dataset(args.test)
        test_settings = load_settings([args.test])
        print(f"TEST         : {args.test} ({len(test)} images)")
        shared = set(settings.participants) & set(test_settings.participants)
        if shared:
            print(f"\nATTENTION : participant(s) {', '.join(sorted(shared))} présent(s) dans "
                  "l'entraînement ET le test.")
            print("Le test mesure la stabilité d'une séance à l'autre, pas la généralisation")
            print("à une personne nouvelle.")
    else:
        print("\nATTENTION : pas de fichier --test. Les résultats ci-dessous sont OPTIMISTES :")
        print("les seuils sont évalués sur les données mêmes qui ont servi à les choisir.")
    print(f"\nSeuils en vigueur à l'enregistrement : enter={current.enter}, exit={exit_}, "
          f"click_max_ms={settings.click_max_ms}, folded_below={current.folded_below} "
          f"(3D {current.folded_below_3d})")

    title("ENTRAÎNEMENT : distributions")
    print_distributions(train)

    title("ENTRAÎNEMENT : seuils proposés")
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
    positive_ratio = stats_by_label(train).get(POSITIVE_LABEL, {}).get("pinch_ratio")
    if positive_ratio is not None and positive_ratio.p95 >= current.enter:
        print(f"\n  Contrôle de enter : P95 du pinch_ratio en pincement maintenu = "
              f"{positive_ratio.p95:.2f} >= enter ({current.enter}) : des images de vrai "
              "pincement passent au-dessus du seuil d'entrée.")
    learned = thresholds_from_proposals(proposals, current)
    print(f"\nSeuils utilisés pour B/C/D : min_index_extension="
          f"{learned.min_index_extension:.2f} (3D {learned.min_index_extension_3d:.2f}), "
          f"folded_below={learned.folded_below:.2f} (3D {learned.folded_below_3d:.2f})")

    title("ENTRAÎNEMENT : règles, image par image")
    print_rules(train, learned, exit_, settings.seconds)
    if test is not None:
        title(f"TEST : règles, image par image ({len(test)} images, mêmes seuils)")
        print_rules(test, learned, exit_, test_settings.seconds)

    title("Garde-fou taille de main (appris sur l'entraînement)")
    print("Pincement maintenu après 1 s : images saines (pinch_ratio < exit) contre images")
    print(f"« décrochées » (pinch_ratio >= exit) ; rapport = hand_size / médiane des "
          f"{settings.window_ms:g} ms précédentes.\n")
    guard = propose_hand_size_ratio(train, exit_, settings.window_ms)
    guard_ratio = 0.0
    if guard is None:
        print("  Pas de proposition : aucune image décrochée (ou aucune saine).")
    else:
        verdict = "OK" if guard.margin > 0 else "CHEVAUCHEMENT : le rapport sépare mal"
        print(f"  min_hand_size_ratio proposé : {guard.threshold:.2f}  (marge {guard.margin:+.2f}"
              f" ; P5 saines {guard.p5_positives:.2f}, P95 décrochées "
              f"{guard.p95_negatives:.2f} ; Youden {guard.youden_threshold:.2f}, "
              f"J={guard.youden_j:.2f} ; n={guard.n_positives}/{guard.n_negatives})  {verdict}")
    if guard is not None and guard.margin <= 0:
        print("\n  -> Garde-fou NON RETENU : les images décrochées n'ont pas une taille de main")
        print("     anormale ; un seuil sur ce rapport écarterait surtout des images saines.")
    elif guard is not None:
        guard_ratio = guard.threshold
        datasets = [("entraînement", train)]
        if test is not None:
            datasets.append(("test", test))
        for name, samples in datasets:
            print(f"\n  Images écartées à {guard_ratio:.2f} ({name}) :")
            for label, (rejected, present) in rejected_by_guard(
                samples, guard_ratio, settings.window_ms
            ).items():
                share = 100 * rejected / present if present else 0.0
                print(f"    {label:<22} {rejected:>4}/{present:<4} ({share:4.1f} %)")

    test_seconds = test_settings.seconds if test_settings else settings.seconds
    variants = [("SANS garde-fou", 0.0)]
    if guard_ratio > 0:
        variants.append((f"AVEC garde-fou ({guard_ratio:.2f})", guard_ratio))
    for label, ratio in variants:
        title(f"Épisodes, {label}")
        rows = sweep(train, test, SWEEP_RULES, SWEEP_EXITS, SWEEP_GRACES, learned, ratio,
                     settings.window_ms, settings.click_max_ms, settings.seconds, test_seconds)
        print_sweep(rows, test is not None)


if __name__ == "__main__":
    main()
