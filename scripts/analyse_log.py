"""Analyse d'un journal de mesures enregistré avec `airdesk --log`.

Usage (venv activé, depuis la racine du projet) :

    python scripts/analyse_log.py                      # le log le plus récent de data/logs/
    python scripts/analyse_log.py data/logs/xxx.csv    # un log précis

Les seuils (pinch.enter, pinch.exit, click_max_ms) sont lus dans le .json enregistré avec
le log : ce sont ceux qui étaient en vigueur pendant la mesure, pas ceux d'aujourd'hui.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from airdesk.config import load_config
from airdesk.log_analysis import (
    IMMOBILE_MIN_MS,
    IMMOBILE_SPEED,
    JITTER_GOAL_PX,
    pinch_durations,
    pinch_thresholds,
    read_log,
    read_metadata,
    still_periods,
    summarize,
    text_histogram,
)
from airdesk.session_log import LOG_DIR


def latest_log(folder: Path) -> Path | None:
    """Le CSV le plus récent (les noms AAAA-MM-JJ_HH-MM-SS se trient dans l'ordre du temps)."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        return None
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse d'un log AirDesk (airdesk --log).")
    parser.add_argument("csv", type=Path, nargs="?", help="fichier CSV (défaut : le plus récent)")
    parser.add_argument("--immobile-speed", type=float, default=IMMOBILE_SPEED,
                        help="vitesse du poignet sous laquelle la main est jugée immobile "
                             f"(tailles de main / s, défaut {IMMOBILE_SPEED})")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = latest_log(LOG_DIR)
        if csv_path is None:
            sys.exit(f"Aucun log dans {LOG_DIR}. Lancer d'abord `airdesk --dry-run --log`.")

    rows = read_log(csv_path)
    print(f"Log : {csv_path}")

    json_path = csv_path.with_suffix(".json")
    if json_path.exists():
        metadata = read_metadata(json_path)
        enter, exit_, click_max_ms = pinch_thresholds(metadata)
        camera = metadata.get("camera", {})
        git = metadata.get("git", {})
        commit = git.get("commit") or "inconnu"
        if git.get("dirty"):
            commit += " (avec modifications non commitées)"
        print(f"Session du {metadata.get('started_at')}", end="")
        print(f" | AirDesk {metadata.get('airdesk_version')}")
        print(f"Commit : {commit}")
        print(
            f"Caméra : index {camera.get('index')}, {camera.get('backend')}, "
            f"{camera.get('width')}x{camera.get('height')} {camera.get('pixel_format')}, "
            f"{camera.get('fps_measured')} i/s à l'ouverture"
        )
    else:
        print(f"ATTENTION : {json_path.name} introuvable, seuils lus dans la config ACTUELLE.")
        pinch = load_config().pinch
        enter, exit_, click_max_ms = pinch.enter, pinch.exit, pinch.click_max_ms
    print(f"Seuils : enter={enter}, exit={exit_}, click_max_ms={click_max_ms}\n")

    s = summarize(rows)
    print(f"Images : {s.frames} en {s.duration_s:.2f} s, soit {s.fps_mean:.1f} images/s en moyenne")
    print(
        f"Traitement : {s.processing_mean_ms:.1f} ms en moyenne, "
        f"{s.processing_p95_ms:.1f} ms au 95e percentile"
    )
    print(f"Événements : {s.clicks} clic(s), {s.drags} glisser\n")

    pinches = pinch_durations(rows, enter, exit_)
    print(f"Pincements (d'après pinch_ratio, hystérésis {enter} / {exit_}) : {len(pinches)}")
    if pinches:
        print(f"{'début (ms)':>10} | {'durée':>7} | {'issue observée':<14} | {'fin':<11} | seuil")
    durations = []
    for p in pinches:
        durations.append(p.duration_ms)
        # Ce que donnerait le seuil : en dessous de click_max_ms, la machine émet un clic.
        expected = "clic" if p.duration_ms <= click_max_ms else "glisser"
        print(
            f"{p.start_ms:>10} | {p.duration_ms:>4} ms | {p.outcome:<14} | {p.ended_by:<11} | "
            f"{expected}"
        )

    print("\nDurée des pincements :")
    for line in text_histogram(durations, marker=click_max_ms, marker_label="click_max_ms"):
        print("  " + line)
    print(
        "\nPour régler click_max_ms : les clics volontaires doivent tomber AVANT le seuil, "
        "les glisser bien après.\nLa durée de mesure dépend de la cadence : à 30 i/s, "
        "précision d'environ 33 ms."
    )

    print_jitter(rows, args.immobile_speed)


def print_jitter(rows, max_speed: float) -> None:
    """Tremblement du curseur pendant les périodes où la main est immobile."""
    print(f"\nTremblement du curseur, main immobile (poignet < {max_speed:g} taille de main/s "
          f"pendant >= {IMMOBILE_MIN_MS / 1000:g} s) :")
    has_cursor = False
    for row in rows:
        if row.cursor is not None:
            has_cursor = True
            break
    if not has_cursor:
        print("  Pas de position de curseur dans ce log (pointage relatif, ou log enregistré "
              "avant cette mesure).")
        return
    periods = still_periods(rows, max_speed)
    if not periods:
        print("  Aucune période immobile assez longue : garder la main immobile 2-3 s, "
              "plusieurs fois, pendant l'enregistrement.")
        return
    print(f"  {'début (ms)':>10} | {'durée':>7} | {'images':>6} | {'é.-t. x':>8} | {'é.-t. y':>8}")
    jitters = []
    for p in periods:
        jitters.append(p.jitter)
        print(f"  {p.start_ms:>10} | {p.duration_ms:>4} ms | {p.frames:>6} | {p.std_x:6.2f} px |"
              f" {p.std_y:6.2f} px")
    median = statistics.median(jitters)
    worst = max(jitters)
    verdict = "OK" if worst < JITTER_GOAL_PX else "AU-DESSUS de l'objectif"
    print(f"  {len(periods)} période(s) ; tremblement (plus grand écart-type x/y) : médiane "
          f"{median:.2f} px, maximum {worst:.2f} px -> objectif < {JITTER_GOAL_PX:g} px : "
          f"{verdict}")
    if worst >= JITTER_GOAL_PX:
        print("  Piste : baisser filter.absolute.min_cutoff dans gestures.yaml (plus de lissage "
              "à basse vitesse), puis vérifier que le curseur ne traîne pas.")


if __name__ == "__main__":
    main()
