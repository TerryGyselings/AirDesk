"""Boucle principale : relie les cinq couches.

    webcam -> squelette de la main -> filtre 1€ -> machine à états -> souris
                                                       \\-> fenêtre de debug
                                                       \\-> journal de mesures (--log)
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import cv2

from airdesk.actuator import create_actuator
from airdesk.capture import Camera
from airdesk.config import apply_cli_overrides, load_config
from airdesk.interpretation import GestureStateMachine, compute_features
from airdesk.perception import HandTracker, PointFilter
from airdesk.session_log import SessionLog, build_metadata
from airdesk.ui import draw_debug

WINDOW_NAME = "AirDesk - debug (q ou Echap pour quitter)"


def run(
    camera_index: int | None = None,
    config_path: Path | None = None,
    dry_run: bool = False,
    show_preview: bool = True,
    backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
    log: bool = False,
) -> None:
    """Lance AirDesk. Les options caméra à None reprennent la valeur des fichiers de config."""
    cfg = load_config(config_path)
    # Priorité : ligne de commande > config/local.yaml > défauts.
    camera_cfg = apply_cli_overrides(cfg.camera, camera_index, backend, width, height)
    cfg = replace(cfg, camera=camera_cfg)  # config effective, enregistrée avec le log

    actuator = create_actuator(dry_run, cfg.pointer.pixels_per_frame_width)
    machine = GestureStateMachine(cfg.engagement, cfg.pinch)
    pointer_filter = PointFilter(cfg.filter.min_cutoff, cfg.filter.beta, cfg.filter.d_cutoff)

    fps = 0.0
    last_frame_time = time.perf_counter()

    print(f"Ouverture de la caméra {camera_cfg.index} (quelques secondes)...")
    forced_backend = None if camera_cfg.backend == "auto" else camera_cfg.backend
    camera = Camera(camera_cfg.index, camera_cfg.width, camera_cfg.height, backend=forced_backend)
    with camera as cam, HandTracker(max_hands=1) as tracker:
        # Les essais écartés expliquent pourquoi on n'a pas eu le backend ou la résolution
        # demandés (ex. « dshow, 1280x720 : trop lent » puis « 640x480 : ok »).
        for attempt in cam.attempts:
            if not attempt.ok:
                print(f"  essai écarté : {attempt.describe()}")
        info = cam.info
        print(
            f"Caméra {camera_cfg.index} : backend {info.backend.upper()}, "
            f"{info.width}x{info.height} {info.pixel_format}, {info.fps:.1f} images/s mesurées"
        )

        # Sans --log, nullcontext() est un « with » qui ne fait rien et donne None.
        if log:
            session = SessionLog(build_metadata(cfg, info.summary()))
        else:
            session = nullcontext()

        with session as logger:
            if logger is not None:
                print(f"Mesures enregistrées dans {logger.csv_path} (+ réglages dans le .json)")
            print("AirDesk démarré. Montre ta main ouverte face à la caméra pour activer.")
            for frame in cam.frames():
                t_start = time.perf_counter()

                hands = tracker.detect(frame.rgb, frame.timestamp_ms)
                hand = hands[0] if hands else None

                features = None
                pointer = None
                if hand is not None:
                    height, width = frame.bgr.shape[:2]
                    features = compute_features(
                        hand.landmarks,
                        cfg.fingers.folded_below,
                        cfg.fingers.folded_below_3d,
                        aspect=width / height,
                    )
                    raw = hand.landmarks[cfg.pointer.landmark]
                    pointer = pointer_filter(raw[0], raw[1], frame.timestamp_ms / 1000.0)
                else:
                    pointer_filter.reset()

                events = machine.update(frame.timestamp_ms, features, pointer)
                for event in events:
                    actuator.apply(event)

                processing_ms = (time.perf_counter() - t_start) * 1000.0
                now = time.perf_counter()
                dt = now - last_frame_time
                last_frame_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)  # moyenne glissante, plus lisible

                if logger is not None:
                    logger.write_frame(
                        frame.timestamp_ms, features, machine.state.value, events, processing_ms
                    )

                if show_preview:
                    image = draw_debug(
                        frame.bgr, hand, features, machine.state, fps, processing_ms
                    )
                    cv2.imshow(WINDOW_NAME, image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

    cv2.destroyAllWindows()
