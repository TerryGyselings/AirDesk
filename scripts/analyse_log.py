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
from dataclasses import asdict
from pathlib import Path

from airdesk.config import load_config
from airdesk.interpretation.pinch_rules import BLOCK_CLOSING
from airdesk.log_analysis import (
    IMMOBILE_MIN_MS,
    IMMOBILE_SPEED,
    JITTER_GOAL_PX,
    JITTER_GOAL_WINDOW_MS,
    JITTER_WINDOWS_MS,
    OUTCOME_NONE,
    SLOW_FRAME_MS,
    block_reason,
    click_mode,
    closing_settings,
    contact_closing_speeds,
    contact_speeds,
    estimated_latencies,
    frame_interval_ms,
    latencies,
    mean_and_p95,
    percentile,
    pin_offsets,
    pinch_durations,
    pinch_thresholds,
    press_episodes,
    read_log,
    read_metadata,
    refused_for_closing,
    release_grace,
    rise_splits,
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
    parser.add_argument("--rise-guard-px", type=float, default=RISE_GUARD_PX,
                        help="rejeu de release_rise : distance au point de contact (px) "
                             "au-delà de laquelle il n'agit plus (≈ drag_threshold, défaut "
                             f"{RISE_GUARD_PX:g})")
    parser.add_argument("--contact-speed", type=float, default=CONTACT_SPEED,
                        help="vitesse du poignet au contact au-delà de laquelle un pincement "
                             "est listé (candidat max_contact_speed, défaut "
                             f"{CONTACT_SPEED})")
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
        mode = click_mode(metadata)
        closing = closing_settings(metadata["config"]["pinch"])
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
        mode = pinch.click_mode
        closing = closing_settings(asdict(pinch))
    print(f"Seuils : enter={enter}, exit={exit_}, click_max_ms={click_max_ms}, "
          f"release_grace_ms={grace_ms:g} | mode de clic : {mode} | règle {closing.rule}\n")

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
    contacts = contact_closing_speeds(rows)
    if pinches:
        print(f"{'début (ms)':>10} | {'durée':>7} | {'issue observée':<14} | {'fin':<11} | seuil"
              " | fermeture au contact")
    durations = []
    for p in pinches:
        durations.append(p.duration_ms)
        # Ce que donnerait le seuil : en dessous de click_max_ms, la machine émet un clic.
        # (En mode press, click_max_ms ne sert pas : c'est le déplacement qui décide.)
        expected = "clic" if p.duration_ms <= click_max_ms else "glisser"
        if mode == "press":
            expected = "-"
        print(
            f"{p.start_ms:>10} | {p.duration_ms:>4} ms | {p.outcome:<14} | {p.ended_by:<11} | "
            f"{expected:<5} | {contact_cell(contacts, p)}"
        )
        if p.outcome == OUTCOME_NONE:
            # Pourquoi aucun clic ni glisser : raison de refus la plus fréquente pendant ce
            # pincement (colonne entry_block, logs récents seulement).
            blocked = block_reason(rows, p.start_ms, p.end_ms)
            if blocked is not None:
                reason, count, total = blocked
                print(f"{'':>10}   -> refusé : {reason} ({count}/{total} images)")
                if reason == BLOCK_CLOSING:
                    best = max_closing_speed(rows, p.start_ms, p.end_ms)
                    print(f"{'':>10}      vitesse de fermeture max {fmt(best)} pendant "
                          f"{p.duration_ms} ms (seuil {closing.min_closing_speed:g})")
            elif not has_entry_block:
                print(f"{'':>10}   -> raison inconnue (colonne entry_block absente de ce log)")

    print()
    print_latency(rows, mode, pinches)
    if mode == "press":
        print_press_episodes(rows)
    print_contact_speeds(rows, args.contact_speed)
    print_pin_offsets(rows)
    print_closing_summary(rows, pinches, contacts, closing)
    print_rise_replay(rows, enter, args.rise_guard_px)

    print("\nDurée des pincements :")
    for line in text_histogram(durations, marker=click_max_ms, marker_label="click_max_ms"):
        print("  " + line)
    print(
        "\nPour régler click_max_ms : les clics volontaires doivent tomber AVANT le seuil, "
        "les glisser bien après.\nLa durée de mesure dépend de la cadence : à 30 i/s, "
        "précision d'environ 33 ms."
    )

    print_jitter(rows, args.immobile_speed)


# Rejeu de release_rise : valeurs comparées, et garde en pixels (0,15 taille de main ≈ 95 px
# dans les logs du 2026-11-24 et 25 : 570 à 700 px par taille de main).
RISE_VALUES = (0.12, 0.15, 0.18)
RISE_GUARD_PX = 95.0


def fmt(value) -> str:
    return "inconnue" if value is None else f"{value:.2f}"


def contact_cell(contacts, pinch) -> str:
    """Vitesse de fermeture au contact accepté pendant ce pincement (vide si refusé)."""
    for c in contacts:
        if pinch.start_ms <= c.contact_ms <= pinch.end_ms:
            return fmt(c.closing_speed)
    return ""


def max_closing_speed(rows, start_ms, end_ms):
    """Plus haute vitesse de fermeture pendant un pincement refusé (sa meilleure chance)."""
    best = None
    for row in rows:
        if start_ms <= row.timestamp_ms <= end_ms and row.closing_speed is not None:
            if best is None or row.closing_speed > best:
                best = row.closing_speed
    return best


def print_closing_summary(rows, pinches, contacts, closing) -> None:
    """Règles F / F3D : marge réelle entre les vitesses de fermeture des contacts acceptés et
    min_closing_speed ; refus « fermeture » ; contacts exemptés (juste après un relâchement).
    Pour un log enregistré en règle D, montre ce que F aurait refusé."""
    speeds = []
    for c in contacts:
        if c.closing_speed is not None:
            speeds.append(c.closing_speed)
    if not speeds:
        return
    refusals = refused_for_closing(rows, pinches)
    note = ""
    if not closing.active:
        # Log enregistré sans condition de fermeture : on compare aux réglages ACTUELS, pour
        # voir ce que F aurait fait de ces contacts.
        closing = closing_settings(asdict(load_config().pinch))
        note = " ; log sans règle F : comparaison avec les réglages actuels"
    threshold = closing.min_closing_speed
    print(f"Vitesse de fermeture (règle {closing.rule}, min_closing_speed {threshold:g}, "
          f"closing_grace_ms {closing.closing_grace_ms:g}{note}) :")
    print(f"  {len(contacts)} contact(s) accepté(s), {len(refusals)} refus « fermeture »")
    print(f"  au contact des acceptés : min {min(speeds):.2f}, P5 {percentile(speeds, 5):.2f}, "
          f"médiane {statistics.median(speeds):.2f}, max {max(speeds):.2f}")
    exempted = []
    below = []
    for c in contacts:
        if c.closing_speed is not None and c.closing_speed >= threshold:
            continue
        label = f"{c.contact_ms} ({fmt(c.closing_speed)}"
        if c.since_release_ms is not None:
            label += f", {c.since_release_ms} ms après un relâchement"
        label += ")"
        if c.since_release_ms is not None and c.since_release_ms <= closing.closing_grace_ms:
            exempted.append(label)
        else:
            below.append(label)
    if exempted:
        print(f"  sous le seuil, exemptés (2e tape) : {len(exempted)} : " + " ; ".join(exempted))
    if below:
        what = "acceptés hors exemption" if not note else "que F aurait refusés"
        print(f"  sous le seuil, {what} : {len(below)} : " + " ; ".join(below))


def print_pin_offsets(rows) -> None:
    """Clics épinglés : écart en pixels entre la main et le point où l'événement est émis.
    press : dérive pendant la confirmation du Press ; release : dérive pendant l'appui
    (ce que l'épinglage corrige) ; *_snap : distance au 1er clic d'un double-clic aimanté."""
    found = pin_offsets(rows)
    if not found:
        return
    print("Clics épinglés, écart main -> point épinglé (px) :")
    for kind in ("press", "release", "press_snap", "release_snap"):
        values = found.get(kind)
        if not values:
            continue
        mean, p95 = mean_and_p95(values)
        print(f"  {kind:<12} {len(values):>3} : médiane {statistics.median(values):5.1f}, "
              f"P95 {p95:5.1f}, max {max(values):5.1f}")
    snaps = len(found.get("press_snap", []))
    print(f"  -> {snaps} clic(s) aimanté(s) sur le clic précédent (double-clic).")


def print_rise_replay(rows, enter: float, guard_px: float) -> None:
    """Ce que ferait release_rise sur ce log : appuis coupés en deux. Dans un glisser, c'est
    une rupture (à éviter) ; dans un clic, c'est une tape fusionnée qui serait séparée."""
    print(f"Rejeu de release_rise (garde {guard_px:g} px, simulation sur pinch_ratio) :")
    for rise in RISE_VALUES:
        splits = rise_splits(rows, rise, enter, guard_px)
        ruptures = []
        separated = []
        for split in splits:
            if split.in_drag:
                ruptures.append(str(split.release_ms))
            else:
                separated.append(str(split.release_ms))
        line = (f"  {rise:.2f} : {len(ruptures)} rupture(s) de glisser, "
                f"{len(separated)} appui(s) de clic coupé(s)")
        if ruptures:
            line += " | ruptures à " + ", ".join(ruptures) + " ms"
        if separated:
            line += " | coupés à " + ", ".join(separated) + " ms"
        print(line)


# Seuil candidat de max_contact_speed (logs du 2026-11-24, voir docs/journal.md).
CONTACT_SPEED = 1.8


def print_contact_speeds(rows, threshold: float) -> None:
    """Vitesse du poignet au contact des pincements suivis d'un clic ou d'un glisser : un
    vrai clic se fait main presque immobile (condition facultative max_contact_speed)."""
    found = contact_speeds(rows)
    if not found:
        return
    speeds = []
    for c in found:
        speeds.append(c.speed)
    mean, p95 = mean_and_p95(speeds)
    print(f"Vitesse du poignet au contact ({len(speeds)} pincements suivis d'un clic ou d'un "
          f"glisser) : médiane {statistics.median(speeds):.2f}, P95 {p95:.2f}, "
          f"max {max(speeds):.2f} taille de main/s")
    fast = []
    for c in found:
        if c.speed >= threshold:
            fast.append(f"{c.contact_ms} ms ({c.outcome}, {c.speed:.2f})")
    if fast:
        print(f"  Au-dessus de {threshold:g} (refusés par max_contact_speed = {threshold:g}) : "
              + " ; ".join(fast))
    else:
        print(f"  Aucun au-dessus de {threshold:g}.")


def print_latency(rows, mode: str, pinches) -> None:
    """Latence de l'interprétation : contact des doigts -> premier Press / Click.

    Sans la caméra ni MediaPipe (communs aux deux modes) : c'est la partie que le mode de
    clic change. Mode release : durée du pincement + release_grace_ms ; mode press : environ
    min_press_frames images."""
    found = latencies(rows)
    how = "mesurée"
    if not found:
        found = estimated_latencies(rows, pinches)
        how = "ESTIMÉE depuis pinch_ratio (log sans colonne contact_ms)"
    if not found:
        print(f"Latence contact -> clic (mode {mode}) : aucun clic.")
        return
    values = []
    for latency in found:
        values.append(latency.latency_ms)
    mean, p95 = mean_and_p95(values)
    print(f"Latence contact -> Press/Click (mode {mode}, {how}) : {len(values)} pincement(s), "
          f"moyenne {mean:.0f} ms, P95 {p95:.0f} ms")


def print_press_episodes(rows) -> None:
    """Mode press : déplacement de la main pendant les clics (pour régler drag_threshold) et
    saut du curseur au début des glisser (compensé par drag_catchup_ms)."""
    episodes = press_episodes(rows)
    if not episodes:
        return
    click_moves = []
    jumps = []
    for e in episodes:
        if e.dragged:
            if e.jump_px is not None:
                jumps.append(e.jump_px)
        elif e.max_anchor_distance is not None:
            click_moves.append(e.max_anchor_distance)
    if click_moves:
        values = []
        for value in sorted(click_moves):
            values.append(f"{value:.2f}")
        print("Déplacement max. de la main pendant les clics (tailles de main) : "
              + " ".join(values))
        print("  -> drag_threshold doit rester au-dessus des clics volontaires.")
    if jumps:
        values = []
        for value in jumps:
            values.append(f"{value:.0f}")
        print(f"Saut du curseur au début des glisser (px, sans rattrapage) : {' '.join(values)}"
              f" | médiane {statistics.median(jumps):.0f}, max {max(jumps):.0f}")


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
