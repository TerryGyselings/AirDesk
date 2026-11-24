"""Point d'entrée : `airdesk` ou `python -m airdesk`."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="airdesk",
        description="Piloter le bureau avec la main devant une webcam.",
        epilog="Les options caméra non données sont lues dans config/local.yaml.",
    )
    # default=None : on distingue « option non donnée » (on garde local.yaml) d'une valeur choisie.
    parser.add_argument(
        "--camera", type=int, default=None, help="index de la webcam (défaut : local.yaml, sinon 0)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "avfoundation", "v4l2", "any"],
        default=None,
        help="API vidéo utilisée par OpenCV (défaut : local.yaml, sinon auto)",
    )
    parser.add_argument("--width", type=int, default=None, help="largeur demandée (défaut : 1280)")
    parser.add_argument("--height", type=int, default=None, help="hauteur demandée (défaut : 720)")
    parser.add_argument("--config", type=Path, default=None, help="fichier YAML de réglages")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ne touche pas à la vraie souris, affiche seulement les actions",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="sans la fenêtre de debug"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="enregistre les mesures de chaque image dans data/logs/ (nombres seulement)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="étalonnage guidé de la zone d'interaction (5 cibles en plein écran, à son "
             "rythme), enregistré dans config/local.yaml",
    )
    args = parser.parse_args()

    # Import tardif : `airdesk --help` reste instantané même si MediaPipe est long à charger.
    from airdesk.app import calibrate, run

    try:
        if args.calibrate:
            calibrate(args.camera, args.config, args.backend, args.width, args.height)
            return
        run(
            camera_index=args.camera,
            backend=args.backend,
            width=args.width,
            height=args.height,
            config_path=args.config,
            dry_run=args.dry_run,
            show_preview=not args.no_preview,
            log=args.log,
        )
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # Erreurs attendues (caméra introuvable...) : message clair, sans trace Python.
        raise SystemExit(f"\nErreur : {exc}") from None


if __name__ == "__main__":
    main()
