"""Diagnostic de la webcam : quels index, backends et résolutions donnent une image utilisable ?

Usage (depuis la racine du projet, venv activé) :

    python scripts/check_camera.py           # diagnostic seul
    python scripts/check_camera.py --save    # + écrit la 1re combinaison OK dans local.yaml

Pour chaque index de caméra (0 à 3), chaque backend du système et deux résolutions
(1280x720 et 640x480), le script ouvre la caméra, lit 30 images puis affiche ce qu'il a
réellement obtenu. Avec DirectShow, il essaie les deux ordres de réglage du format MJPG
(avant / après la résolution). Chaque essai est limité dans le temps : un pilote qui bloque
est signalé « bloqué (timeout) » sans arrêter le diagnostic.

Avec --save, la première combinaison OK (dans l'ordre index, backend, puis 1280x720 avant
640x480 : la meilleure résolution d'abord) est écrite dans config/local.yaml. AirDesk
l'essaie alors directement au démarrage, au lieu de tout parcourir.

Aucune image n'est enregistrée (RGPD) : on ne garde que des chiffres.
"""

from __future__ import annotations

import argparse
import platform

import cv2

from airdesk.capture.camera import (
    PLATFORM_BACKENDS,
    STATUS_CLOSED,
    STATUS_TIMEOUT,
    ProbeResult,
    probe_with_timeout,
)
from airdesk.config import LOCAL_CONFIG_PATH, save_camera_settings

INDEXES = [0, 1, 2, 3]
RESOLUTIONS = [(1280, 720), (640, 480)]
FPS = 30
FRAMES_TO_READ = 30
# Plus long que les 5 s de l'application : ici on lit 30 images, soit 3 s à elles seules pour
# une webcam lente (10 i/s), plus environ 3 s d'ouverture avec DirectShow.
TIMEOUT_S = 10.0


def mjpg_orders(backend: str) -> list[bool]:
    """Ordres de réglage à essayer : les deux pour DirectShow, un seul (sans objet) sinon."""
    if backend == "dshow":
        return [True, False]
    return [True]


def format_line(r: ProbeResult) -> str:
    """Fin de ligne du tableau (tout ce qui suit l'essai)."""
    if r.status == STATUS_CLOSED:
        opened = "KO"
    elif r.status == STATUS_TIMEOUT:
        opened = "?"
    else:
        opened = "OK"
    if r.width:
        resolution = f"{r.width}x{r.height}"
        fps = f"{r.fps:.1f}"
        brightness = f"{r.brightness:.1f}"
        pixel_format = r.pixel_format
    else:
        resolution = fps = brightness = pixel_format = "-"
    verdict = "OK" if r.ok else r.status.upper()
    if r.detail:
        verdict += f" ({r.detail})"
    return (
        f"{opened:<4} | {resolution:<10} | {pixel_format:<6} | {fps:>5} | {brightness:>6} | "
        f"{verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic des webcams pour AirDesk.")
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"écrire la première combinaison OK dans {LOCAL_CONFIG_PATH.name}",
    )
    args = parser.parse_args()

    # Sans ça, OpenCV écrit ses avertissements (index inexistant...) au milieu du tableau.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    system = platform.system()
    backends = PLATFORM_BACKENDS.get(system, ["any"])
    print(f"Système : {system} | OpenCV {cv2.__version__}", end="")
    print(f" | backends testés : {', '.join(backends)}")
    print(f"{FRAMES_TO_READ} images lues par essai, {TIMEOUT_S:g} s maximum par essai.\n")

    header = (
        f"{'idx':>3} | {'backend':<7} | {'demandé':<8} | {'MJPG':<5} | {'ouv.':<4} | "
        f"{'obtenu':<10} | {'format':<6} | {'FPS':>5} | {'lumin.':>6} | verdict"
    )
    print(header)
    print("-" * len(header))

    working: list[int] = []
    first_ok: ProbeResult | None = None
    for index in INDEXES:
        for backend in backends:
            for width, height in RESOLUTIONS:
                for mjpg_first in mjpg_orders(backend):
                    if backend == "dshow":
                        order = "avant" if mjpg_first else "après"
                    else:
                        order = "-"
                    # Début de ligne affiché AVANT l'essai (flush=True) : si un pilote bloque,
                    # on voit tout de suite quelle combinaison est en cause.
                    print(
                        f"{index:>3} | {backend:<7} | {f'{width}x{height}':<8} | {order:<5} | ",
                        end="",
                        flush=True,
                    )
                    r = probe_with_timeout(
                        TIMEOUT_S,
                        index=index,
                        backend=backend,
                        width=width,
                        height=height,
                        fps=FPS,
                        frames=FRAMES_TO_READ,
                        mjpg_first=mjpg_first,
                    )
                    print(format_line(r), flush=True)
                    if r.ok and index not in working:
                        working.append(index)
                    if r.ok and first_ok is None:
                        first_ok = r

    print()
    if working:
        print(f"Index utilisables : {', '.join(str(i) for i in working)}  (airdesk --camera N)")
    else:
        print("Aucune caméra n'a donné d'image utilisable.")

    if args.save:
        if first_ok is None:
            print(f"--save : rien à enregistrer, {LOCAL_CONFIG_PATH} n'est pas modifié.")
        else:
            width, height = first_ok.requested
            save_camera_settings(
                LOCAL_CONFIG_PATH, first_ok.index, first_ok.backend, width, height
            )
            print(
                f"Enregistré dans {LOCAL_CONFIG_PATH} : caméra {first_ok.index}, "
                f"backend {first_ok.backend}, {width}x{height}"
            )
    print()
    print("Lecture :")
    print("- MJPG avant/après : format MJPG demandé avant ou après la résolution (DSHOW).")
    print("- IMAGE NOIRE : la caméra s'ouvre mais n'envoie rien (carte d'acquisition sans source,")
    print("  cache sur l'objectif...).")
    print("- IMAGE FIGÉE : toujours la même image (caméra virtuelle éteinte, ex. OBS).")
    print("- TROP LENT : moins de 20 images/s en 1280x720, préférer une résolution plus basse.")
    print("- BLOQUÉ : le pilote n'a pas répondu à temps (fréquent avec MSMF).")


if __name__ == "__main__":
    main()
