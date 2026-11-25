"""Boucle principale : relie les cinq couches.

    webcam -> mains (MediaPipe) -> choix de la main -> filtre 1€ -> machine à états -> souris
                                                                        \\-> fenêtre de debug
                                                                        \\-> journal (--log)
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import cv2

from airdesk.actuator import create_actuator, detect_screen
from airdesk.actuator.system import resolve_double_click_ms
from airdesk.capture import Camera
from airdesk.config import apply_cli_overrides, load_config
from airdesk.interpretation import GestureStateMachine, State, compute_features
from airdesk.interpretation.features import MIDDLE_MCP, WRIST, distance_2d, wrist_speed
from airdesk.interpretation.hand_selection import (
    HandCandidate,
    HandSelector,
    TrackerModePolicy,
)
from airdesk.interpretation.pointing import zone_to_screen
from airdesk.interpretation.state_machine import PressAt, ReleaseAt
from airdesk.perception import HandTracker, PointFilter
from airdesk.session_log import SessionLog, build_metadata
from airdesk.ui import draw_debug

WINDOW_NAME = "AirDesk - debug (q ou Echap pour quitter)"
SLOW_CAMERA_FPS = 25


def warn_if_slow_camera(fps: float) -> None:
    """Prévient si la caméra est lente : souvent le manque de lumière (pose plus longue)."""
    if fps < SLOW_CAMERA_FPS:
        print(f"  ATTENTION : la caméra ne donne que {fps:.1f} images/s (< {SLOW_CAMERA_FPS}). "
              "Peu de lumière ? Éclairer la pièce : la webcam allonge son temps de pose quand "
              "il fait sombre, et le curseur devient moins fluide.")


def pin_offset_px(events, cursor, actuator) -> float | None:
    """Écart en pixels entre la main (position réelle, `cursor`) et le point épinglé d'un
    PressAt / ReleaseAt de l'image : ce que l'épinglage « corrige ». None sans événement
    épinglé."""
    if cursor is None:
        return None
    for event in events:
        if isinstance(event, (PressAt, ReleaseAt)):
            x, y = actuator.to_pixels(event.u, event.v)
            return math.hypot(cursor[0] - x, cursor[1] - y)
    return None


def hand_candidate(hand, features, aspect: float) -> HandCandidate:
    """Ce dont le sélecteur de main a besoin, en unités homogènes (x multiplié par l'aspect)."""
    wx, wy, _ = hand.landmarks[WRIST]
    mx, my, _ = hand.landmarks[MIDDLE_MCP]
    size = distance_2d((wx * aspect, wy, 0.0), (mx * aspect, my, 0.0))
    return HandCandidate((wx * aspect, wy), size, hand.side, features.open_palm)


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

    absolute = cfg.pointer.mode == "absolute"
    screen = None
    if absolute:
        screen = detect_screen(cfg.pointer.screen)
        print(f"Pointage absolu sur l'écran {cfg.pointer.screen} : {screen.width}x{screen.height}"
              f" en ({screen.x}, {screen.y})")
    else:
        print("Pointage relatif (comme un pavé tactile).")
    actuator = create_actuator(dry_run, cfg.pointer.pixels_per_frame_width, screen)
    # Délai du double-clic : "auto" = délai du système (lu ici, la machine à états reste
    # sans dépendance système). Enregistré tel quel dans le log.
    double_click_ms = resolve_double_click_ms(cfg.pinch.double_click_ms)
    cfg = replace(cfg, pinch=replace(cfg.pinch, double_click_ms=double_click_ms))
    if cfg.pinch.click_mode == "press" and absolute:
        print(f"Double-clic aimanté : {double_click_ms:g} ms, "
              f"{cfg.pinch.double_click_snap:g} taille de main")
    machine = GestureStateMachine(cfg.engagement, cfg.pinch, cfg.fingers, cfg.pointer)
    selector = HandSelector(cfg.hand.preferred, cfg.hand.max_jump)
    filter_cfg = cfg.filter.for_mode(cfg.pointer.mode)
    pointer_filter = PointFilter(filter_cfg.min_cutoff, filter_cfg.beta, filter_cfg.d_cutoff)

    fps = 0.0
    last_frame_time = time.perf_counter()
    previous_hand = None  # (timestamp_ms, landmarks) de la main suivie, pour sa vitesse
    tracker_policy = TrackerModePolicy(cfg.hand.preferred, cfg.hand.track_support,
                                       cfg.hand.two_hands_hold_ms,
                                       cfg.hand.search_after_lost_frames)
    pilot_found = False  # main pilote trouvée à l'image précédente
    other_hand_seen = False  # main non pilote vue à l'image précédente

    print(f"Ouverture de la caméra {camera_cfg.index} (quelques secondes)...")
    forced_backend = None if camera_cfg.backend == "auto" else camera_cfg.backend
    camera = Camera(camera_cfg.index, camera_cfg.width, camera_cfg.height, backend=forced_backend)
    with camera as cam, HandTracker(max_hands=2) as tracker:
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
        warn_if_slow_camera(info.fps)

        # Sans --log, nullcontext() est un « with » qui ne fait rien et donne None.
        if log:
            metadata = build_metadata(cfg, info.summary())
            if screen is not None:
                metadata["screen"] = {"x": screen.x, "y": screen.y, "width": screen.width,
                                      "height": screen.height}
            session = SessionLog(metadata)
        else:
            session = nullcontext()

        with session as logger:
            if logger is not None:
                print(f"Mesures enregistrées dans {logger.csv_path} (+ réglages dans le .json)")
            print("AirDesk démarré. Montre ta main ouverte face à la caméra pour activer.")
            for frame in cam.frames():
                t_start = time.perf_counter()
                frame_height, frame_width = frame.bgr.shape[:2]
                aspect = frame_width / frame_height

                # 1) Les mains, leurs descripteurs, puis le rôle de chacune.
                # 2 mains à chercher seulement quand c'est utile pour trouver la main pilote
                # (voir TrackerModePolicy) : sinon MediaPipe relance à chaque image la détection
                # de paume, coûteuse (19 ms au lieu de 9).
                idle = machine.state == State.IDLE
                tracker_hands = tracker_policy.hands_to_search(
                    frame.timestamp_ms, idle, pilot_found, other_hand_seen)
                hands = tracker.detect(frame.rgb, frame.timestamp_ms, tracker_hands)
                all_features = []
                candidates = []
                for hand in hands:
                    f = compute_features(hand.landmarks, cfg.fingers.folded_below,
                                         cfg.fingers.folded_below_3d, aspect=aspect)
                    all_features.append(f)
                    candidates.append(hand_candidate(hand, f, aspect))
                roles = selector.select(candidates, idle=idle)
                selected = roles.pilot
                # Avec le modèle 1 main, MediaPipe peut s'accrocher à l'autre main : le
                # sélecteur la refuse, et TrackerModePolicy fait chercher 2 mains.
                pilot_found = selected is not None
                other_hand_seen = len(hands) > (1 if pilot_found else 0)

                features = None
                pointer = None
                hand = None
                speed = None
                if selected is not None:
                    hand = hands[selected]
                    features = all_features[selected]
                    raw = hand.landmarks[cfg.pointer.landmark]
                    pointer = pointer_filter(raw[0], raw[1], frame.timestamp_ms / 1000.0)
                    if previous_hand is not None:
                        dt_s = (frame.timestamp_ms - previous_hand[0]) / 1000.0
                        speed = wrist_speed(hand.landmarks, previous_hand[1], dt_s, aspect)
                    previous_hand = (frame.timestamp_ms, hand.landmarks)
                else:
                    pointer_filter.reset()
                    previous_hand = None

                # 2) Interprétation puis action.
                events = machine.update(frame.timestamp_ms, features, pointer, speed)
                for event in events:
                    actuator.apply(event)

                processing_ms = (time.perf_counter() - t_start) * 1000.0
                now = time.perf_counter()
                dt = now - last_frame_time
                last_frame_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)  # moyenne glissante, plus lisible

                if logger is not None:
                    wrist = None
                    cursor = None
                    if hand is not None:
                        wrist = (hand.landmarks[WRIST][0], hand.landmarks[WRIST][1])
                        if absolute:
                            u, v = zone_to_screen(pointer[0], pointer[1], cfg.pointer.zone)
                            cursor = actuator.to_pixels(u, v)
                    pin_offset = pin_offset_px(events, cursor, actuator)
                    logger.write_frame(
                        frame.timestamp_ms, features, machine.state.value, events,
                        processing_ms, wrist=wrist, wrist_speed=speed, cursor=cursor,
                        closing_speed=machine.closing_speed,
                        closing_speed_3d=machine.closing_speed_3d,
                        entry_block=machine.entry_block,
                        tracker_hands=tracker_hands,
                        contact_ms=machine.event_contact_ms,
                        anchor_distance=machine.anchor_distance,
                        pin_kind=machine.pin_kind,
                        pin_offset_px=pin_offset,
                        second_tap=machine.second_tap_candidate,
                    )

                if show_preview:
                    # La zone sert aussi en mode relatif (activation, pincement, main baissée).
                    image = draw_debug(
                        frame.bgr, hands, selected, features, machine.state, fps,
                        processing_ms, cfg.pointer.zone, support=roles.support,
                    )
                    cv2.imshow(WINDOW_NAME, image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

    cv2.destroyAllWindows()

