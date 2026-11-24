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

COLUMNS = [
    "timestamp_ms",  # horodatage de l'image (ms depuis l'ouverture de la caméra)
    "hand_present",  # 1 si une main est détectée, sinon 0
    "pinch_ratio",  # distance pouce-index / taille de la main (vide sans main)
    "extended_fingers",  # doigts tendus (index, majeur, annulaire, auriculaire), vide sans main
    "state",  # état de la machine : veille, actif, pincement, glisser
    "events",  # événements émis, joints par « + » (ex. « move+click »), vide sinon
    "processing_ms",  # temps de traitement de l'image (détection + interprétation)
]


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
        if features is None:
            hand_present = 0
            pinch_ratio = ""
            extended_fingers = ""
        else:
            hand_present = 1
            pinch_ratio = f"{features.pinch_ratio:.4f}"
            extended_fingers = features.extended_fingers
        self._writer.writerow(
            [
                timestamp_ms,
                hand_present,
                pinch_ratio,
                extended_fingers,
                state,
                event_names(events),
                f"{processing_ms:.2f}",
            ]
        )

    def __exit__(self, *exc) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
