"""Point d'entrée : `airdesk` ou `python -m airdesk`."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="airdesk", description="Piloter le bureau avec la main devant une webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="index de la webcam (défaut : 0)")
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "avfoundation", "v4l2", "any"],
        default="auto",
        help="API vidéo utilisée par OpenCV (défaut : auto = selon le système)",
    )
    parser.add_argument("--width", type=int, default=1280, help="largeur demandée (défaut : 1280)")
    parser.add_argument("--height", type=int, default=720, help="hauteur demandée (défaut : 720)")
    parser.add_argument("--config", type=Path, default=None, help="fichier YAML de réglages")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ne touche pas à la vraie souris, affiche seulement les actions",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="sans la fenêtre de debug"
    )
    args = parser.parse_args()

    # Import tardif : `airdesk --help` reste instantané même si MediaPipe est long à charger.
    from airdesk.app import run

    backend = None if args.backend == "auto" else args.backend

    try:
        run(
            camera_index=args.camera,
            backend=backend,
            width=args.width,
            height=args.height,
            config_path=args.config,
            dry_run=args.dry_run,
            show_preview=not args.no_preview,
        )
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # Erreurs attendues (caméra introuvable...) : message clair, sans trace Python.
        raise SystemExit(f"\nErreur : {exc}") from None


if __name__ == "__main__":
    main()
