"""Enregistrement des mesures d'une session (`airdesk --log`), pour régler les seuils.

Deux fichiers par session, dans data/logs/ (ignoré par git) :

- `AAAA-MM-JJ_HH-MM-SS.csv` : une ligne par image traitée ;
- `AAAA-MM-JJ_HH-MM-SS.json` : les réglages utilisés (config complète, version, commit git,
  système, caméra). Sans lui, on ne saurait plus quel seuil était en place pendant la mesure.

On n'enregistre que des nombres et des mots, jamais d'image (RGPD).
Ce module ne fait que de l'écriture : il est appelé par app.py, comme les autres couches.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from airdesk.config import PROJECT_ROOT, Config

LOG_DIR = PROJECT_ROOT / "data" / "logs"

# Colonnes qui décrivent la main, remplies par `hand_columns`. Partagées avec les jeux de
# données de scripts/record_gestures.py.
HAND_COLUMNS = [
    "hand_present",  # 1 si une main est détectée, sinon 0
    "pinch_ratio",  # distance pouce-index / taille de la main (vide sans main)
    "extended_fingers",  # doigts tendus (index, majeur, annulaire, auriculaire), vide sans main
    # Extension de chaque doigt (bout -> centre de paume / taille de main), en 2D puis en 3D
    "ext_index",
    "ext_middle",
    "ext_ring",
    "ext_pinky",
    "folded_fingers",  # doigts repliés (extension 2D sous fingers.folded_below)
    "ext3d_index",
    "ext3d_middle",
    "ext3d_ring",
    "ext3d_pinky",
    "folded_fingers_3d",
    "pinch_seg",  # distance pouce -> segment 7-8 de l'index / taille de main
    "pinch_seg_68",  # distance pouce -> segment 6-8 de l'index / taille de main
]

COLUMNS = (
    ["timestamp_ms"]  # horodatage de l'image (ms depuis l'ouverture de la caméra)
    + HAND_COLUMNS
    + [
        "state",  # état de la machine : veille, actif, pincement, glisser
        "events",  # événements émis, joints par « + » (ex. « move+click »), vide sinon
        "processing_ms",  # temps de traitement de l'image (détection + interprétation)
    ]
)


def git_commit(folder: Path = PROJECT_ROOT) -> dict:
    """Commit git courant et présence de modifications non commitées.

    Renvoie {"commit": None, "dirty": None} si git est absent ou si ce n'est pas un dépôt :
    l'enregistrement ne doit jamais échouer pour ça.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=folder, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=folder, capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    # « dirty » : le code mesuré n'est pas exactement celui du commit.
    return {"commit": commit, "dirty": status != ""}


def airdesk_version() -> str | None:
    try:
        return version("airdesk")
    except PackageNotFoundError:
        return None


def build_metadata(config: Config, camera: dict, git: dict | None = None) -> dict:
    """Rassemble tout ce qu'il faut pour interpréter le log plus tard.

    `camera` décrit l'ouverture réelle (backend, résolution, format, FPS mesuré).
    `git` est calculé si non fourni (paramètre utile pour les tests).
    """
    if git is None:
        git = git_commit()
    return {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "airdesk_version": airdesk_version(),
        "git": git,
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "camera": camera,
        "config": asdict(config),  # gestes + caméra, après fusion local.yaml et ligne de commande
        "columns": COLUMNS,
    }


def hand_columns(features) -> list:
    """Valeurs des colonnes HAND_COLUMNS, dans le même ordre.

    Sans main (`features` None) : hand_present = 0 et les autres colonnes vides.
    Partagé avec scripts/record_gestures.py, pour que logs et jeux de données aient les
    mêmes colonnes et le même format.
    """
    if features is None:
        return [0] + [""] * (len(HAND_COLUMNS) - 1)
    row = [1, f"{features.pinch_ratio:.4f}", features.extended_fingers]
    for value in features.finger_extensions:
        row.append(f"{value:.4f}")
    row.append(features.folded_fingers)
    for value in features.finger_extensions_3d:
        row.append(f"{value:.4f}")
    row.append(features.folded_fingers_3d)
    row.append(f"{features.pinch_seg:.4f}")
    row.append(f"{features.pinch_seg_68:.4f}")
    return row


def event_names(events: list) -> str:
    """[Move(...), Click()] -> "move+click"."""
    names = []
    for event in events:
        names.append(type(event).__name__.lower())
    return "+".join(names)


class SessionLog:
    """Écrit le JSON des réglages à l'ouverture, puis une ligne CSV par image.

        with SessionLog(metadata) as log:
            log.write_frame(t_ms, features, "actif", events, 12.3)
    """

    def __init__(self, metadata: dict, folder: Path = LOG_DIR, now: datetime | None = None):
        if now is None:
            now = datetime.now()
        # Pas de « : » dans le nom : caractère interdit dans les noms de fichiers Windows.
        stem = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = folder / f"{stem}.csv"
        self.json_path = folder / f"{stem}.json"
        self.metadata = metadata
        self._file = None
        self._writer = None

    def __enter__(self) -> SessionLog:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        # newline="" : recommandé par le module csv, sinon lignes vides en plus sous Windows.
        # buffering=1 : chaque ligne est écrite sur le disque aussitôt ; si AirDesk est fermé
        # brutalement, le log reste lisible jusqu'à la dernière image.
        self._file = open(self.csv_path, "w", encoding="utf-8", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        return self

    def write_frame(
        self, timestamp_ms: int, features, state: str, events: list, processing_ms: float
    ) -> None:
        """Une ligne par image. `features` vaut None quand aucune main n'est visible."""
        row = [timestamp_ms]
        row.extend(hand_columns(features))
        row.append(state)
        row.append(event_names(events))
        row.append(f"{processing_ms:.2f}")
        self._writer.writerow(row)

    def __exit__(self, *exc) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
