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
import csv
from dataclasses import dataclass, replace
from pathlib import Path

from airdesk.config import load_config
from airdesk.gesture_analysis import (
    DESCRIPTORS,
    ENTRY_DESCRIPTORS,
    EPISODE_LABELS,
    NEGATIVE_LABELS,
    OUT_OF_VOCABULARY_LABELS,
    POSITIVE_LABEL,
    PROFILE_DESCRIPTORS,
    PROFILE_LABEL,
    PROFILE_NEGATIVE_LABELS,
    EpisodeSettings,
    SweepRow,
    confusion,
    count_episodes,
    count_sequences,
    detection_rate,
    entry_details,
    entry_separations,
    has_profile_data,
    labels_in_order,
    profile_metrics,
    profile_separations,
    propose_hand_size_ratio,
    propose_thresholds,
    rank,
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
    "main_repos_profil": "profil",
    "main_repos_profil_gauche": "profil g.",
    "va_et_vient_gauche": "va-vient",
    "pincements_repetes": "repetes",
    "pincements_repetes_gauche": "repet. g.",
    "pincement_vers_camera": "hors voc.",
}

# Balayage des métriques par épisode. D3D n'apportait rien sur les premières mesures : il
# reste dans le tableau image par image mais pas dans le balayage.
SWEEP_RULES = ["A", "D"]
SWEEP_GRACES = [0, 66, 150]
# Couples (enter, exit) par mesure. Une distance au segment de l'index est toujours plus
# petite qu'une distance au bout : les seuils de pinch_ratio ne conviennent pas à pinch_seg.
RATIO_THRESHOLDS = [(0.25, 0.40), (0.25, 0.50)]
SEG_THRESHOLDS = []
for _enter in (0.15, 0.20, 0.25):
    for _exit in (0.30, 0.40, 0.50):
        SEG_THRESHOLDS.append((_enter, _exit))
SHOWN_ROWS = 15  # configurations affichées (toutes dans --sweep-csv)


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
    # Les jeux de données enregistrés avant la condition de paume n'ont pas min_palm_width :
    # on prend alors la valeur provisoire actuelle de gestures.yaml.
    min_palm_width = pinch.get("min_palm_width")
    if not min_palm_width:
        min_palm_width = load_config(local_path=None).pinch.min_palm_width
    thresholds = RuleThresholds(
        enter=pinch["enter"],
        min_index_extension=pinch["min_index_extension"],
        folded_below=fingers["folded_below"],
        min_index_extension_3d=pinch["min_index_extension_3d"],
        folded_below_3d=fingers["folded_below_3d"],
        min_palm_width=min_palm_width,
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


def configurations(learned: RuleThresholds, measures: list[str]) -> list[EpisodeSettings]:
    """Toutes les configurations du balayage."""
    configs = []
    for rule in SWEEP_RULES:
        # Les variantes « repli » et « paume » n'ont de sens que pour la règle D.
        folds = [False, True] if rule == "D" else [False]
        palms = [False, True] if rule == "D" else [False]
        for measure in measures:
            pairs = RATIO_THRESHOLDS if measure == "ratio" else SEG_THRESHOLDS
            for enter, exit_value in pairs:
                for palm in palms:
                    thresholds = replace(learned, enter=enter, require_palm_facing=palm)
                    for grace in SWEEP_GRACES:
                        for exit_on_fold in folds:
                            configs.append(EpisodeSettings(
                                rule, thresholds, exit_value, grace, measure=measure,
                                exit_on_fold=exit_on_fold,
                            ))
    return configs


def print_sweep_row(row: SweepRow) -> None:
    s = row.settings
    tr = row.train
    te = row.test
    fold = "oui" if s.exit_on_fold else "non"
    palm = "oui" if s.thresholds.require_palm_facing else "non"
    ruptures = pair(tr.ruptures, te.ruptures if te else None)
    clicks = pair(tr.false_clicks, te.false_clicks if te else None)
    drags = pair(tr.false_drags, te.false_drags if te else None)
    repeated_test = None
    if te is not None:
        repeated_test = f"{te.repeated_detected}/{te.repeated_expected}"
    repeated = pair(f"{tr.repeated_detected}/{tr.repeated_expected}", repeated_test)
    out_of_vocabulary = pair(tr.out_of_vocabulary, te.out_of_vocabulary if te else None)
    print(
        f"  {s.rule:<3} {s.measure:<6} {s.thresholds.enter:4.2f} {s.exit:4.2f} "
        f"{s.release_grace_ms:>4g} {fold:<4} {palm:<4}"
        f" | {ruptures:<7} | {clicks:<8} | {drags:<9} | {repeated:<13} | {out_of_vocabulary:<7}"
    )


def print_sweep(rows: list[SweepRow], has_test: bool, ruptures_reliable: bool) -> None:
    header = (f"  {'rg.':<3} {'mesure':<6} {'ent.':>4} {'exit':>4} {'dél.':>4} {'repli':<4}"
              f" {'paume':<4}"
              f" | {'rupt.':<7} | {'f.clics':<8} | {'f.gliss.':<9} | {'répétés':<13} | hors v.")
    ranked = rank(rows, ruptures_reliable)
    print(f"  {len(rows)} configurations ; les {SHOWN_ROWS} premières du classement "
          "(calculé sur l'entraînement) :\n")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in ranked[:SHOWN_ROWS]:
        print_sweep_row(row)
    for row in rows:
        s = row.settings
        if (s.rule, s.measure, s.thresholds.enter, s.exit, s.release_grace_ms) == \
                ("A", "ratio", 0.25, 0.40, 0) and not s.exit_on_fold and \
                not s.thresholds.require_palm_facing:
            print("  " + "." * (len(header) - 2))
            print("  référence (configuration d'origine) :")
            print_sweep_row(row)
    where = "entraînement / test" if has_test else "entraînement"
    print(f"\n  Cellules : {where}. rupt. = sorties pendant le pincement maintenu (idéal 0) ;")
    print("  f.clics / f.gliss. = faux épisodes courts / longs (poing, poing o/f, main relâchée,")
    print("  parasites, main au repos de profil) ; répétés = détectés / attendus ; repli =")
    print("  sortie si les doigts se replient ; paume = entrée seulement paume face caméra.")

    best = recommend(rows, ruptures_reliable)
    print("\n  Recommandation : filtres (0 faux glisser, au plus 1 faux clic) et classement sur")
    print("  l'ENTRAÎNEMENT seulement ; avec autant de configurations, choisir sur le test")
    print("  reviendrait à régler les paramètres sur le test, qui ne mesurerait plus rien.")
    if ruptures_reliable:
        print("  Classement : le moins de ruptures, le plus de répétés, le plus petit délai.")
    else:
        print("  Classement (ruptures NON fiables) : le moins de faux glisser, de faux clics,")
        print("  le plus de répétés détectés, le plus petit délai.")
    if best is None:
        print("  -> AUCUNE configuration ne remplit les conditions sur l'entraînement.")
        return
    print(f"  -> {best.settings.describe()}")
    for name, m in [("entraînement", best.train), ("test (vérification)", best.test)]:
        if m is None:
            continue
        print(f"     {name:<20}: {m.ruptures} rupture(s), {m.false_clicks} faux clic(s), "
              f"{m.false_drags} faux glisser, répétés {m.repeated_detected}/"
              f"{m.repeated_expected}, hors vocabulaire {m.out_of_vocabulary}")


def write_sweep_csv(path: Path, rows: list[SweepRow]) -> None:
    """Toutes les configurations du balayage dans un CSV (pour un tableur)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["rule", "measure", "enter", "exit", "release_grace_ms", "exit_on_fold",
                  "require_palm_facing", "min_palm_width"]
        for dataset in ("train", "test"):
            for name in ("ruptures", "false_clicks", "false_drags", "repeated_detected",
                         "repeated_expected", "out_of_vocabulary"):
                header.append(f"{dataset}_{name}")
        writer.writerow(header)
        for row in rows:
            s = row.settings
            line = [s.rule, s.measure, s.thresholds.enter, s.exit, s.release_grace_ms,
                    s.exit_on_fold, s.thresholds.require_palm_facing, s.thresholds.min_palm_width]
            for m in (row.train, row.test):
                if m is None:
                    line += [""] * 6
                else:
                    line += [m.ruptures, m.false_clicks, m.false_drags, m.repeated_detected,
                             m.repeated_expected, m.out_of_vocabulary]
            writer.writerow(line)


LEFT_DETAIL_LABELS = EPISODE_LABELS + PROFILE_NEGATIVE_LABELS


def cell(value) -> str:
    return "     -" if value is None else f"{value:6.2f}"


def closing_threshold(separations, descriptor: str) -> float:
    """Seuil de vitesse de fermeture proposé par la comparaison au moment de l'entrée.

    Retenu seulement si les vrais pincements se ferment PLUS vite (sens « valeur haute =
    pincement ») ; sinon 0, c'est-à-dire sans effet.
    """
    for sep in separations:
        if sep.descriptor == descriptor and sep.higher_is_pinch:
            return sep.proposal.threshold
    return 0.0


def print_left_profile(train, test, learned: RuleThresholds, exit_: float, click_max_ms: float,
                       seconds: int, test_seconds: int) -> None:
    """Main tournée vers la gauche (protocole v4) : quel indice sépare les vrais pincements
    du pivot de profil, et que donnent les règles A, D, E, F, F3D, D + paume ?"""
    if not has_profile_data(train):
        print("  Pas de séquence du protocole v4 (pincements_repetes_gauche, "
              "main_repos_profil_gauche, va_et_vient_gauche) dans l'entraînement : enregistrer "
              "une session v4 (python scripts/record_gestures.py).")
        return
    enter = learned.enter
    print(f"  Passages de pinch_ratio sous enter ({enter}) et médianes des indices à ces "
          "moments (entraînement) :")
    header = f"    {'étiquette':<26} {'n':>4}"
    for name in ENTRY_DESCRIPTORS:
        header += f" {name:>16}"
    print(header)
    for detail in entry_details(train, LEFT_DETAIL_LABELS, enter):
        line = f"    {detail.label:<26} {detail.count:>4}"
        for name in ENTRY_DESCRIPTORS:
            line += f" {cell(detail.medians[name]):>16}"
        print(line)

    separations = entry_separations(train, enter)
    print("\n  Vrais pincements (répétés, centre et gauche) contre main de profil (repos et "
          "va-et-vient), au moment de l'entrée, du meilleur indice au moins bon :")
    for sep in separations:
        p = sep.proposal
        sense = ">" if sep.higher_is_pinch else "<"
        verdict = "sépare" if p.margin > 0 else "chevauchement"
        print(f"    {sep.descriptor:<17} pincement si {sense} {p.threshold:7.2f}   "
              f"J={p.youden_j:4.2f}   marge {p.margin:+.2f}  {verdict}  "
              f"(n={p.n_positives}/{p.n_negatives})")

    min_closing = closing_threshold(separations, "closing_speed")
    min_closing_3d = closing_threshold(separations, "closing_speed_3d")
    print(f"\n  Seuils proposés : min_closing_speed = {min_closing:.2f}, "
          f"min_closing_speed_3d = {min_closing_3d:.2f} (0 = pas de séparation dans le bon "
          "sens)")

    current = load_config(local_path=None).pinch
    thresholds = replace(learned, min_closing_speed=min_closing,
                         min_closing_speed_3d=min_closing_3d)
    palm = replace(thresholds, require_palm_facing=True)
    candidates = [
        ("A", EpisodeSettings("A", thresholds, exit_, current.release_grace_ms,
                              release_cancel_margin=current.release_cancel_margin)),
        ("D", EpisodeSettings("D", thresholds, exit_, current.release_grace_ms,
                              release_cancel_margin=current.release_cancel_margin)),
        ("E (D, 3D)", EpisodeSettings("D", thresholds, exit_, current.release_grace_ms,
                                      measure="ratio3d",
                                      release_cancel_margin=current.release_cancel_margin)),
        ("F", EpisodeSettings("F", thresholds, exit_, current.release_grace_ms,
                              release_cancel_margin=current.release_cancel_margin)),
        ("F3D", EpisodeSettings("F3D", thresholds, exit_, current.release_grace_ms,
                                release_cancel_margin=current.release_cancel_margin)),
        ("D + paume", EpisodeSettings("D", palm, exit_, current.release_grace_ms,
                                      release_cancel_margin=current.release_cancel_margin)),
    ]
    print(f"\n  Épisodes (délai {current.release_grace_ms:g} ms, marge d'annulation "
          f"{current.release_cancel_margin:g}, enter {enter} / exit {exit_}) ; cellules "
          "entraînement / test :")
    print(f"    {'règle':<11} | {'faux clics profil':<17} | {'faux gliss. profil':<18} | "
          f"{'répétés':<13} | {'répétés gauche':<14}")
    for name, settings in candidates:
        tr = profile_metrics(train, settings, click_max_ms, seconds)
        te = profile_metrics(test, settings, click_max_ms, test_seconds) if test else None

        def repeated(m, label):
            found, expected = m.detected[label]
            return f"{found}/{expected}"

        clicks = pair(tr.false_clicks, te.false_clicks if te else None)
        drags = pair(tr.false_drags, te.false_drags if te else None)
        center = pair(repeated(tr, "pincements_repetes"),
                      repeated(te, "pincements_repetes") if te else None)
        left = pair(repeated(tr, "pincements_repetes_gauche"),
                    repeated(te, "pincements_repetes_gauche") if te else None)
        print(f"    {name:<11} | {clicks:<17} | {drags:<18} | {center:<13} | {left:<14}")
    print("    Faux clics / glisser profil : sur main_repos_profil, main_repos_profil_gauche et "
          "va_et_vient_gauche. Une bonne règle : peu de faux épisodes, sans perdre les "
          "pincements répétés à gauche.")


def print_profile_comparison(samples, name: str) -> None:
    """Quel indice sépare le mieux la main au repos de profil des vrais pincements ?"""
    if PROFILE_LABEL not in labels_in_order(samples):
        print(f"  ({name}) Pas de séquence {PROFILE_LABEL} : il faut une session au protocole v3 "
              "(python scripts/record_gestures.py). min_palm_width n'est pas modifié.")
        return
    labels = [PROFILE_LABEL, POSITIVE_LABEL, "pincements_repetes"]
    stats = stats_by_label(samples)
    print(f"  ({name}) Distributions (min / P5 / médiane / P95 / max) :")
    for descriptor in PROFILE_DESCRIPTORS:
        print(f"    {descriptor} :")
        for label in labels:
            s = stats.get(label, {}).get(descriptor)
            if s is None:
                print(f"      {label:<22} -")
                continue
            print(f"      {label:<22} {s.minimum:6.2f} {s.p5:6.2f} {s.median:6.2f} {s.p95:6.2f} "
                  f"{s.maximum:6.2f}  (n={s.n})")
    for pinch_label in (POSITIVE_LABEL, "pincements_repetes"):
        print(f"\n  ({name}) Séparation {pinch_label} / {PROFILE_LABEL}, du meilleur indice au "
              "moins bon (Youden J ; marge = écart entre P5/P95 des deux groupes) :")
        for sep in profile_separations(samples, pinch_label):
            p = sep.proposal
            sense = ">" if sep.higher_is_pinch else "<"
            verdict = "sépare" if p.margin > 0 else "chevauchement"
            print(f"    {sep.descriptor:<15} pincement si {sense} {p.threshold:6.2f}   "
                  f"J={p.youden_j:4.2f} (seuil Youden {p.youden_threshold:6.2f})   "
                  f"marge {p.margin:+.2f}  {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparaison des règles de pincement.")
    parser.add_argument("train", type=Path, nargs="+", help="jeux de données d'entraînement")
    parser.add_argument("--test", type=Path, default=None, help="jeu de données de test")
    parser.add_argument("--enter", type=float, default=None, help="tester une autre valeur")
    parser.add_argument("--exit", type=float, default=None, help="tester une autre valeur")
    parser.add_argument("--ruptures-non-fiables", action="store_true",
                        help="le pincement maintenu a pu être relâché : ruptures hors classement")
    parser.add_argument("--sweep-csv", type=Path, default=None,
                        help="écrire toutes les configurations du balayage dans ce CSV")
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
          f"folded_below={learned.folded_below:.2f} (3D {learned.folded_below_3d:.2f}), "
          f"min_palm_width={learned.min_palm_width:.2f}")
    palm_proposal = proposals.get("min_palm_width")
    if palm_proposal is None or palm_proposal.margin <= 0:
        print(f"  min_palm_width : valeur PROVISOIRE gardée ({learned.min_palm_width:.2f}), la "
              "proposition ne sépare pas (il faut la séquence main_repos_profil, protocole v3).")

    title("ENTRAÎNEMENT : règles, image par image")
    print_rules(train, learned, exit_, settings.seconds)
    if test is not None:
        title(f"TEST : règles, image par image ({len(test)} images, mêmes seuils)")
        print_rules(test, learned, exit_, test_settings.seconds)

    title("Main au repos de profil contre pincements")
    print_profile_comparison(train, "entraînement")
    if test is not None:
        print_profile_comparison(test, "test")

    title("Main tournée vers la gauche (protocole v4)")
    test_seconds_v4 = test_settings.seconds if test_settings else settings.seconds
    print_left_profile(train, test, learned, exit_, settings.click_max_ms, settings.seconds,
                       test_seconds_v4)

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

    # pinch_seg n'est disponible que si les 21 points ont été enregistrés.
    measures = ["ratio"]
    for s in train:
        if s.hand_present:
            if s.pinch_seg is not None:
                measures += ["seg78", "seg68"]
            break
    ruptures_reliable = not args.ruptures_non_fiables

    test_seconds = test_settings.seconds if test_settings else settings.seconds
    variants = [("SANS garde-fou", 0.0)]
    if guard_ratio > 0:
        variants.append((f"AVEC garde-fou ({guard_ratio:.2f})", guard_ratio))
    for label, ratio in variants:
        title(f"Épisodes, {label}")
        if not ruptures_reliable:
            print("  ATTENTION : ruptures déclarées NON FIABLES pour ces fichiers (le pincement")
            print("  maintenu a peut-être été relâché volontairement) : elles sont affichées mais")
            print("  ne servent pas au classement.\n")
        configs = []
        for config in configurations(learned, measures):
            configs.append(replace(config, guard_ratio=ratio, window_ms=settings.window_ms))
        rows = sweep(train, test, configs, settings.click_max_ms, settings.seconds,
                     test_seconds)
        print_sweep(rows, test is not None, ruptures_reliable)
        if args.sweep_csv is not None:
            path = args.sweep_csv
            if ratio > 0:
                path = path.with_name(f"{path.stem}_garde-fou{path.suffix}")
            write_sweep_csv(path, rows)
            print(f"\n  Tableau complet : {path}")


if __name__ == "__main__":
    main()
