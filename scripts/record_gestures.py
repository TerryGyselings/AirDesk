"""Enregistrement guidé et étiqueté de gestes, pour régler la détection du pincement.

Usage (venv activé, depuis la racine du projet) :

    python scripts/record_gestures.py --participant P01
    python scripts/record_gestures.py --seconds 5        # séquences plus courtes (essai)

Une fenêtre guide l'utilisateur à travers 8 séquences de 10 s, chacune précédée d'un compte
à rebours de 3 s. Chaque image est enregistrée dans data/datasets/AAAA-MM-JJ_HH-MM-SS.csv
avec son étiquette (la séquence en cours), tous les descripteurs et les 21 points de la main
normalisés (relatifs au poignet, divisés par la taille de la main). Un .json à côté garde les
réglages et le déroulé. AUCUNE IMAGE n'est enregistrée (RGPD).

`q` ou Échap : arrêter (ce qui est déjà enregistré est conservé).

Remarque d'architecture : c'est un outil de recherche, hors de l'application ; comme
check_camera.py, il relie lui-même capture, perception et interprétation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2

from airdesk.capture import Camera
from airdesk.config import PROJECT_ROOT, apply_cli_overrides, load_config
from airdesk.interpretation.features import (
    WRIST,
    compute_features,
    normalize_landmarks,
    wrist_speed,
)
from airdesk.interpretation.pinch_rules import ClosingSpeed
from airdesk.perception import HandTracker
from airdesk.session_log import HAND_COLUMNS, build_metadata, hand_columns
from airdesk.ui.debug_view import BONES

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
WINDOW_NAME = "AirDesk - enregistrement (q ou Echap pour arreter)"
METRONOME_FLASH_MS = 300  # durée d'affichage de « PINCE ! » au début de chaque seconde
# Version du protocole, notée dans le JSON : 2 = consigne du pincement maintenu précisée
# (doigts collés, petits mouvements) et position / vitesse du poignet enregistrées ;
# 3 = séquence négative « main_repos_profil » et colonnes palm_width / palm_facing ;
# 4 = main tournée vers la GAUCHE de l'écran (pivot de l'avant-bras en pointage absolu) :
#     pincements_repetes_gauche (positif), main_repos_profil_gauche et va_et_vient_gauche
#     (négatifs) ; colonnes pinch_ratio_3d, closing_speed et closing_speed_3d.
PROTOCOL_VERSION = 4
VA_ET_VIENT_PERIOD_MS = 2000  # un aller-retour centre -> gauche -> centre toutes les 2 s


@dataclass(frozen=True)
class Sequence:
    label: str  # étiquette écrite dans le CSV
    instructions: tuple[str, ...]  # consigne affichée (sans accents : police OpenCV limitée)
    # Indication rythmée : None, "pince" (« PINCE ! » une fois par seconde) ou
    # "va_et_vient" (« ALLER » puis « RETOUR », un aller-retour toutes les 2 s).
    cue: str | None = None


SEQUENCES = [
    Sequence("main_ouverte", ("Main ouverte, doigts ecartes,", "face a la camera. Bouge-la.")),
    Sequence(
        "pincements_repetes",
        ("Pince pouce-index a chaque PINCE !", "puis relache bien entre deux."),
        cue="pince",
    ),
    Sequence(
        "pincement_maintenu",
        ("Pince et GARDE pouce et index COLLES 10 s.",
         "Petits mouvements lents, sans relacher,", "sans revenir en arriere."),
    ),
    Sequence("poing", ("Poing ferme, pouce pose sur l'index replie.", "Bouge le poing.")),
    Sequence("poing_ouvre_ferme", ("Ouvre et ferme le poing,", "sans pincer.")),
    Sequence("main_relachee", ("Main relachee, naturelle,", "comme au repos devant l'ecran.")),
    Sequence("parasites", ("Gestes libres SANS pincer :", "se gratter, montrer, saluer...")),
    Sequence(
        "main_repos_profil",
        ("Main detendue DE COTE (profil),", "pouce contre l'index, comme au repos.",
         "Bouge un peu, sans montrer la paume."),
    ),
    Sequence(
        "pincements_repetes_gauche",
        ("Main placee comme pour viser la GAUCHE", "de l'ecran (avant-bras tourne) :",
         "pince a chaque PINCE !"),
        cue="pince",
    ),
    Sequence(
        "main_repos_profil_gauche",
        ("Main au repos, placee a GAUCHE", "(comme pour viser la gauche de l'ecran),",
         "SANS pincer. Bouge un peu."),
    ),
    Sequence(
        "va_et_vient_gauche",
        ("SANS pincer : main du centre vers la", "GAUCHE de l'ecran puis retour,",
         "un aller-retour toutes les 2 s."),
        cue="va_et_vient",
    ),
    Sequence(
        "pincement_vers_camera",
        ("Index pointe VERS la camera :", "pince a chaque PINCE !"),
        cue="pince",
    ),
]


def dataset_columns() -> list[str]:
    columns = ["label", "t_in_sequence_ms", "timestamp_ms", "handedness"]
    columns += HAND_COLUMNS
    columns.append("hand_size")
    # Position du poignet dans l'image (0 à 1, image en miroir) et sa vitesse en tailles de
    # main par seconde : permet de vérifier si les décrochages arrivent pendant les gestes
    # rapides. Vitesse vide à la première image et après une perte de la main.
    columns += ["wrist_x", "wrist_y", "wrist_speed"]
    # Vitesse de fermeture des doigts (baisse de pinch_ratio / pinch_ratio_3d par seconde,
    # sur ~200 ms) : un vrai pincement rapproche activement les doigts.
    columns += ["closing_speed", "closing_speed_3d"]
    for i in range(21):
        for axis in ("x", "y", "z"):
            columns.append(f"{axis}{i}")
    return columns


def speed_since(hand, previous, timestamp_ms: int, aspect: float) -> float | None:
    """Vitesse du poignet depuis l'image précédente (tailles de main par seconde).

    `previous` : (timestamp_ms, landmarks) de l'image précédente avec main, ou None.
    """
    if previous is None:
        return None
    previous_ms, previous_landmarks = previous
    dt_s = (timestamp_ms - previous_ms) / 1000.0
    return wrist_speed(hand.landmarks, previous_landmarks, dt_s, aspect)


def dataset_row(
    label: str, t_ms: int, timestamp_ms: int, hand, features, aspect: float, speed=None,
    closing=None, closing_3d=None,
) -> list:
    """Une ligne du CSV : étiquette, descripteurs, poignet, 21 points normalisés.

    Sans main, toutes les colonnes après hand_present sont vides.
    """
    row = [label, t_ms, timestamp_ms, hand.handedness if hand is not None else ""]
    row += hand_columns(features)
    if hand is None:
        row.append("")
        row += ["", "", ""]
        row += ["", ""]
        row += [""] * 63
        return row
    row.append(f"{features.hand_size:.4f}")
    wrist_x, wrist_y, _ = hand.landmarks[WRIST]
    row += [f"{wrist_x:.4f}", f"{wrist_y:.4f}", "" if speed is None else f"{speed:.3f}"]
    row += ["" if closing is None else f"{closing:.3f}",
            "" if closing_3d is None else f"{closing_3d:.3f}"]
    for x, y, z in normalize_landmarks(hand.landmarks, aspect):
        row += [f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"]
    return row


def write_json(path: Path, metadata: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def put_text(image, text, org, scale=0.6, color=(255, 255, 255), thickness=2) -> None:
    """Texte lisible sur tout fond : contour noir puis texte."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_overlay(image, hand, features, sequence, number, total, t_ms, duration_ms, recording):
    h, w = image.shape[:2]
    if hand is not None:
        pts = []
        for lm in hand.landmarks:
            pts.append((int(lm[0] * w), int(lm[1] * h)))
        for a, b in BONES:
            cv2.line(image, pts[a], pts[b], (80, 200, 80), 2)

    put_text(image, f"{number}/{total}  {sequence.label}", (12, 28), 0.65, (0, 220, 255))
    y = 58
    for line in sequence.instructions:
        put_text(image, line, (12, y), 0.6)
        y += 28

    if recording:
        cv2.circle(image, (w - 30, 25), 9, (0, 0, 255), -1)
        put_text(image, "REC", (w - 85, 32), 0.6, (0, 0, 255))
        # Barre de progression en bas de l'image.
        cv2.rectangle(image, (0, h - 8), (int(w * t_ms / duration_ms), h), (0, 0, 255), -1)
        if sequence.cue == "pince" and t_ms % 1000 < METRONOME_FLASH_MS:
            put_text(image, "PINCE !", (w // 2 - 90, h // 2), 1.6, (0, 255, 255), 4)
        elif sequence.cue == "va_et_vient":
            if t_ms % VA_ET_VIENT_PERIOD_MS < VA_ET_VIENT_PERIOD_MS // 2:
                put_text(image, "<- ALLER (gauche)", (w // 2 - 170, h // 2), 1.1,
                         (0, 255, 255), 3)
            else:
                put_text(image, "RETOUR (centre) ->", (w // 2 - 170, h // 2), 1.1,
                         (255, 200, 0), 3)
    else:
        remaining = (duration_ms - t_ms) // 1000 + 1
        put_text(image, "Prepare-toi...", (w // 2 - 90, h // 2 - 50), 0.8)
        put_text(image, str(remaining), (w // 2 - 20, h // 2 + 30), 2.5, (0, 220, 255), 5)

    if features is not None:
        put_text(
            image,
            f"pince {features.pinch_ratio:.2f} (3D {features.pinch_ratio_3d:.2f})  "
            f"paume {features.palm_width:.2f}  replies {features.folded_fingers}",
            (12, h - 20),
            0.5,
        )
    else:
        put_text(image, "aucune main", (12, h - 20), 0.5, (0, 0, 255))
    return image


def run_phase(frames, tracker, cfg, aspect, sequence, number, total, duration_ms, writer):
    """Affiche (et enregistre si `writer`) pendant `duration_ms`. False si l'utilisateur quitte."""
    start_ms = None
    previous = None  # (timestamp_ms, landmarks) de la dernière image avec main
    closing = ClosingSpeed()
    closing_3d = ClosingSpeed()
    for frame in frames:
        if start_ms is None:
            start_ms = frame.timestamp_ms
        t_ms = frame.timestamp_ms - start_ms
        if t_ms >= duration_ms:
            return True

        hands = tracker.detect(frame.rgb, frame.timestamp_ms)
        hand = hands[0] if hands else None
        features = None
        speed = None
        closing_speed = None
        closing_speed_3d = None
        if hand is not None:
            features = compute_features(
                hand.landmarks, cfg.fingers.folded_below, cfg.fingers.folded_below_3d, aspect
            )
            speed = speed_since(hand, previous, frame.timestamp_ms, aspect)
            previous = (frame.timestamp_ms, hand.landmarks)
            closing_speed = closing.update(frame.timestamp_ms, features.pinch_ratio)
            closing_speed_3d = closing_3d.update(frame.timestamp_ms, features.pinch_ratio_3d)
        else:
            previous = None  # après une perte de la main, les vitesses repartent de zéro
            closing.reset()
            closing_3d.reset()
        if writer is not None:
            writer.writerow(
                dataset_row(sequence.label, t_ms, frame.timestamp_ms, hand, features, aspect,
                            speed, closing_speed, closing_speed_3d)
            )

        image = draw_overlay(
            frame.bgr, hand, features, sequence, number, total, t_ms, duration_ms,
            recording=writer is not None,
        )
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
    return False  # le flux de la caméra s'est arrêté


def main() -> None:
    parser = argparse.ArgumentParser(description="Enregistrement guidé de gestes étiquetés.")
    parser.add_argument("--participant", default="P00",
                        help="code pseudonyme (ex. P01), jamais un nom (RGPD)")
    parser.add_argument("--seconds", type=int, default=10, help="durée de chaque séquence")
    parser.add_argument("--countdown", type=int, default=3, help="compte à rebours (s)")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    camera_cfg = apply_cli_overrides(cfg.camera, args.camera, args.backend, args.width, args.height)
    cfg = replace(cfg, camera=camera_cfg)
    backend = None if camera_cfg.backend == "auto" else camera_cfg.backend

    print(f"Ouverture de la caméra {camera_cfg.index}...")
    with Camera(camera_cfg.index, camera_cfg.width, camera_cfg.height, backend=backend) as cam, \
            HandTracker(max_hands=1) as tracker:
        aspect = cam.info.width / cam.info.height
        metadata = build_metadata(cfg, cam.info.summary())
        metadata["dataset"] = {
            "participant": args.participant,
            "protocol_version": PROTOCOL_VERSION,
            "seconds": args.seconds,
            "countdown_s": args.countdown,
            "metronome": "un PINCE ! par seconde dans les séquences à métronome ; un "
                         "aller-retour toutes les 2 s pour va_et_vient_gauche",
            "aspect": aspect,
            "landmarks": "x,y,z relatifs au poignet, unités homogènes (x et z multipliés par "
                         "aspect), divisés par la taille de main 2D",
            "sequences": [],
            "completed": [],
        }
        for seq in SEQUENCES:
            metadata["dataset"]["sequences"].append(
                {"label": seq.label, "instructions": " ".join(seq.instructions),
                 "cue": seq.cue}
            )

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = DATASET_DIR / f"{stem}.csv"
        json_path = DATASET_DIR / f"{stem}.json"
        print(f"Enregistrement dans {csv_path}")
        write_json(json_path, metadata)  # tout de suite, au cas où le script s'arrête en route

        frames = cam.frames()
        total = len(SEQUENCES)
        with open(csv_path, "w", encoding="utf-8", newline="", buffering=1) as f:
            writer = csv.writer(f)
            writer.writerow(dataset_columns())
            for number, seq in enumerate(SEQUENCES, start=1):
                if not run_phase(frames, tracker, cfg, aspect, seq, number, total,
                                 args.countdown * 1000, writer=None):
                    break
                if not run_phase(frames, tracker, cfg, aspect, seq, number, total,
                                 args.seconds * 1000, writer=writer):
                    break
                metadata["dataset"]["completed"].append(seq.label)
                print(f"  {number}/{total} {seq.label} : terminé")

        # Réécrit à la fin pour y noter les séquences réellement terminées.
        write_json(json_path, metadata)

    cv2.destroyAllWindows()
    done = len(metadata["dataset"]["completed"])
    print(f"{done}/{total} séquences enregistrées. Analyse : "
          f"python scripts/analyse_gestures.py {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
