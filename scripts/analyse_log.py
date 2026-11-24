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
    JITTER_GOAL_WINDOW_MS,
    JITTER_WINDOWS_MS,
    OUTCOME_NONE,
    SLOW_FRAME_MS,
    block_reason,
    frame_interval_ms,
    pinch_durations,
    pinch_thresholds,
    read_log,
    read_metadata,
    release_grace,
    still_periods,
    summarize,
    text_histogram,
    timing_by_mode,
    wrist_speed_quartiles,
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
    grace_ms = 0
    if json_path.exists():
        metadata = read_metadata(json_path)
        enter, exit_, click_max_ms = pinch_thresholds(metadata)
        grace_ms = release_grace(metadata)
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
        grace_ms = pinch.release_grace_ms
    print(f"Seuils : enter={enter}, exit={exit_}, click_max_ms={click_max_ms}, "
          f"release_grace_ms={grace_ms:g}\n")

    s = summarize(rows)
    print(f"Images : {s.frames} en {s.duration_s:.2f} s, soit {s.fps_mean:.1f} images/s en moyenne")
    print(
        f"Traitement : {s.processing_mean_ms:.1f} ms en moyenne, "
        f"{s.processing_p95_ms:.1f} ms au 95e percentile"
    )
    modes = timing_by_mode(rows)
    if modes:
        print("Traitement par mode MediaPipe (mains demandées) :")
        for m in modes:
            print(f"  {m.hands} main(s) : {m.frames:>5} images, moyenne {m.mean_ms:5.1f} ms, "
                  f"P95 {m.p95_ms:5.1f} ms, max {m.max_ms:5.1f} ms, "
                  f"{m.slow_share * 100:4.1f} % au-dessus de {SLOW_FRAME_MS:g} ms")
    else:
        print("Traitement par mode : colonne tracker_hands absente (log enregistré avant).")
    print(f"Événements : {s.clicks} clic(s), {s.drags} glisser\n")

    # Avec un délai de relâchement, le clic est émis après la fin mesurée sur pinch_ratio :
    # on cherche l'issue jusqu'à fin + délai + une image.
    outcome_extra_ms = grace_ms + frame_interval_ms(rows)
    pinches = pinch_durations(rows, enter, exit_, outcome_extra_ms)
    print(f"Pincements (d'après pinch_ratio, hystérésis {enter} / {exit_}) : {len(pinches)}"
          f" ; issue cherchée jusqu'à {outcome_extra_ms:.0f} ms après la fin")
    has_entry_block = False
    for row in rows:
        if row.entry_block:
            has_entry_block = True
            break
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
        if p.outcome == OUTCOME_NONE:
            # Pourquoi aucun clic ni glisser : raison de refus la plus fréquente pendant ce
            # pincement (colonne entry_block, logs récents seulement).
            blocked = block_reason(rows, p.start_ms, p.end_ms)
            if blocked is not None:
                reason, count, total = blocked
                print(f"{'':>10}   -> refusé : {reason} ({count}/{total} images)")
            elif not has_entry_block:
                print(f"{'':>10}   -> raison inconnue (colonne entry_block absente de ce log)")

    print("\nDurée des pincements :")
    for line in text_histogram(durations, marker=click_max_ms, marker_label="click_max_ms"):
        print("  " + line)
    print(
        "\nPour régler click_max_ms : les clics volontaires doivent tomber AVANT le seuil, "
        "les glisser bien après.\nLa durée de mesure dépend de la cadence : à 30 i/s, "
        "précision d'environ 33 ms."
    )

    print_jitter(rows, args.immobile_speed)


def cell(value: float | None) -> str:
    return "    n/a" if value is None else f"{value:5.2f} px"


def print_jitter(rows, max_speed: float) -> None:
    """Tremblement du curseur pendant les périodes où la main est immobile (toujours affiché)."""
    print(f"\nTremblement du curseur, main immobile (poignet < {max_speed:g} taille de main/s "
          f"pendant >= {IMMOBILE_MIN_MS / 1000:g} s) :")
    quartiles = wrist_speed_quartiles(rows)
    if quartiles is None:
        print("  Vitesse du poignet absente de ce log (enregistré avant cette mesure).")
    else:
        p25, median, p75 = quartiles
        print(f"  Vitesse du poignet sur tout le log : P25 {p25:.2f}, médiane {median:.2f}, "
              f"P75 {p75:.2f} taille de main/s (réglage : --immobile-speed)")
    has_cursor = False
    for row in rows:
        if row.cursor is not None:
            has_cursor = True
            break
    if not has_cursor:
        print("  0 période immobile trouvée : pas de position de curseur dans ce log (pointage "
              "relatif, ou log enregistré avant cette mesure).")
        return
    periods = still_periods(rows, max_speed)
    if not periods:
        print("  0 période immobile trouvée : garder la main immobile 2-3 s, plusieurs fois, ou "
              "relever --immobile-speed.")
        return

    short, long = JITTER_WINDOWS_MS
    print(f"  Dérive = écart-type brut ; tremblement = écart-type autour d'une moyenne glissante "
          f"de {short} ms ou {long} ms.")
    print("  Le tremblement DÉPEND de cette fenêtre : le curseur est déjà lissé par le filtre 1€,")
    print(f"  son tremblement restant est lent, et la fenêtre de {short} ms en compte une partie "
          "comme de la dérive")
    print(f"  (mesure optimiste). L'objectif < {JITTER_GOAL_PX:g} px est jugé sur "
          f"{JITTER_GOAL_WINDOW_MS} ms.\n")
    print(f"  {'début (ms)':>10} | {'durée':>7} | {'images':>6} | {'dérive':>8} | "
          f"{f'tremb. {short}':>10} | {f'tremb. {long}':>11}")
    goal_values = []
    for p in periods:
        print(f"  {p.start_ms:>10} | {p.duration_ms:>4} ms | {p.frames:>6} | {p.drift:5.2f} px | "
              f"{cell(p.jitter(short)):>10} | {cell(p.jitter(long)):>11}")
        if p.jitter(JITTER_GOAL_WINDOW_MS) is not None:
            goal_values.append(p.jitter(JITTER_GOAL_WINDOW_MS))
    if not goal_values:
        print(f"  {len(periods)} période(s), toutes trop courtes pour la fenêtre de "
              f"{JITTER_GOAL_WINDOW_MS} ms : garder la main immobile plus longtemps.")
        return
    median = statistics.median(goal_values)
    worst = max(goal_values)
    verdict = "OK" if worst < JITTER_GOAL_PX else "AU-DESSUS de l'objectif"
    print(f"  {len(periods)} période(s) ; tremblement sur {JITTER_GOAL_WINDOW_MS} ms : médiane "
          f"{median:.2f} px, maximum {worst:.2f} px -> objectif < {JITTER_GOAL_PX:g} px : "
          f"{verdict}")
    if worst >= JITTER_GOAL_PX:
        print("  Piste : baisser filter.absolute.min_cutoff dans gestures.yaml (plus de lissage "
              "à basse vitesse), puis vérifier que le curseur ne traîne pas.")


if __name__ == "__main__":
    main()
