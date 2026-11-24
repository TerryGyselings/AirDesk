"""Boucle principale : relie les cinq couches.

    webcam -> mains (MediaPipe) -> choix de la main -> filtre 1€ -> machine à états -> souris
                                                                        \\-> fenêtre de debug
                                                                        \\-> journal (--log)
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import cv2

from airdesk.actuator import create_actuator, detect_screen
from airdesk.capture import Camera
from airdesk.config import LOCAL_CONFIG_PATH, apply_cli_overrides, load_config, save_zone
from airdesk.interpretation import GestureStateMachine, State, compute_features
from airdesk.interpretation.features import MIDDLE_MCP, WRIST, distance_2d, wrist_speed
from airdesk.interpretation.hand_selection import HandCandidate, HandSelector
from airdesk.interpretation.pointing import (
    TARGETS,
    StillnessDetector,
    center_offset,
    target_position,
    zone_from_corners,
    zone_to_screen,
)
from airdesk.perception import HandTracker, PointFilter
from airdesk.session_log import SessionLog, build_metadata
from airdesk.ui import draw_debug
from airdesk.ui.calibration_view import (
    draw_capture,
    draw_refusal,
    draw_validation,
    make_canvas,
)
from airdesk.ui.debug_view import IGNORED_COLOR, draw_hand

WINDOW_NAME = "AirDesk - debug (q ou Echap pour quitter)"
CALIBRATION_WINDOW = "AirDesk - etalonnage"
SLOW_CAMERA_FPS = 25


def warn_if_slow_camera(fps: float) -> None:
    """Prévient si la caméra est lente : souvent le manque de lumière (pose plus longue)."""
    if fps < SLOW_CAMERA_FPS:
        print(f"  ATTENTION : la caméra ne donne que {fps:.1f} images/s (< {SLOW_CAMERA_FPS}). "
              "Peu de lumière ? Éclairer la pièce : la webcam allonge son temps de pose quand "
              "il fait sombre, et le curseur devient moins fluide.")


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
    machine = GestureStateMachine(cfg.engagement, cfg.pinch, cfg.fingers, cfg.pointer)
    selector = HandSelector(cfg.hand.preferred, cfg.hand.max_jump)
    filter_cfg = cfg.filter.for_mode(cfg.pointer.mode)
    pointer_filter = PointFilter(filter_cfg.min_cutoff, filter_cfg.beta, filter_cfg.d_cutoff)

    fps = 0.0
    last_frame_time = time.perf_counter()
    previous_hand = None  # (timestamp_ms, landmarks) de la main suivie, pour sa vitesse
    pilot_lost = True  # au départ, il faut chercher la main pilote

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
                # 2 mains à chercher seulement quand il faut trouver la main pilote (veille, ou
                # main pilote perdue à l'image précédente) : sinon MediaPipe relance à chaque
                # image la détection de paume, coûteuse (19 ms au lieu de 9).
                idle = machine.state == State.IDLE
                search_two = idle or pilot_lost or cfg.hand.track_support
                hands = tracker.detect(frame.rgb, frame.timestamp_ms, 2 if search_two else 1)
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
                # sélecteur la refuse, et l'image suivante cherche de nouveau 2 mains.
                pilot_lost = selected is None

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
                    wrist = None
                    cursor = None
                    if hand is not None:
                        wrist = (hand.landmarks[WRIST][0], hand.landmarks[WRIST][1])
                        if absolute:
                            u, v = zone_to_screen(pointer[0], pointer[1], cfg.pointer.zone)
                            cursor = actuator.to_pixels(u, v)
                    logger.write_frame(
                        frame.timestamp_ms, features, machine.state.value, events,
                        processing_ms, wrist=wrist, wrist_speed=speed, cursor=cursor,
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


# --- Étalonnage guidé de la zone d'interaction (`airdesk --calibrate`) -------------------------

KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_BACKSPACE = 8
REACHED_DISTANCE = 0.06  # validation : coin « atteint » si le point passe à moins de 6 % de l'écran


def calibrate(
    camera_index: int | None = None,
    config_path: Path | None = None,
    backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Étalonnage guidé, à son rythme : 5 cibles en plein écran, puis validation.

    Pour chaque cible (centre, haut-gauche, haut-droite, bas-droite, bas-gauche), la main
    pilote est capturée quand elle reste immobile 1 s (interpretation/pointing.py). Les 4 coins
    donnent la zone, le centre sert de contrôle. Validation : un point suit la main avec la
    nouvelle zone, sans toucher à la vraie souris. Entrée enregistre dans config/local.yaml.
    """
    cfg = load_config(config_path)
    cal = cfg.calibration
    camera_cfg = apply_cli_overrides(cfg.camera, camera_index, backend, width, height)
    forced_backend = None if camera_cfg.backend == "auto" else camera_cfg.backend

    screen = detect_screen("primary")
    canvas_w = screen.width // 2  # dessiné à mi-résolution, étiré en plein écran
    canvas_h = screen.height // 2
    targets = []
    for name in TARGETS:
        u, v = target_position(name, cal.target_margin)
        targets.append((name, u, v))
    corner_targets = targets[1:]

    selector = HandSelector(cfg.hand.preferred, cfg.hand.max_jump)
    detector = StillnessDetector(cal.still_speed, cal.hold_ms)
    abs_filter = cfg.filter.absolute
    pointer_filter = PointFilter(abs_filter.min_cutoff, abs_filter.beta, abs_filter.d_cutoff)

    step = "capture"  # capture -> validation, ou refus
    current = 0
    measures: dict[str, tuple[float, float]] = {}
    zone = None
    message = ""
    warning = None
    reached: set[str] = set()
    previous_hand = None
    saved = False
    old_zone = cfg.pointer.zone

    cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(CALIBRATION_WINDOW, screen.x, screen.y)  # sur l'écran principal
    cv2.setWindowProperty(CALIBRATION_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"Ouverture de la caméra {camera_cfg.index}...")
    camera = Camera(camera_cfg.index, camera_cfg.width, camera_cfg.height, backend=forced_backend)
    with camera as cam, HandTracker(max_hands=2) as tracker:
        warn_if_slow_camera(cam.info.fps)
        for frame in cam.frames():
            frame_height, frame_width = frame.bgr.shape[:2]
            aspect = frame_width / frame_height

            # Main pilote seulement (l'autre main est ignorée, comme dans AirDesk).
            hands = tracker.detect(frame.rgb, frame.timestamp_ms, 2)
            candidates = []
            for hand in hands:
                f = compute_features(hand.landmarks, cfg.fingers.folded_below,
                                     cfg.fingers.folded_below_3d, aspect=aspect)
                candidates.append(hand_candidate(hand, f, aspect))
            pilot = selector.select(candidates, idle=True).pilot

            camera_image = frame.bgr
            point = None
            speed = None
            for i in range(len(hands)):
                color = (80, 200, 80) if i == pilot else IGNORED_COLOR
                draw_hand(camera_image, hands[i], color)
            if pilot is not None:
                hand = hands[pilot]
                raw = hand.landmarks[cfg.pointer.landmark]
                point = (raw[0], raw[1])
                if previous_hand is not None:
                    dt_s = (frame.timestamp_ms - previous_hand[0]) / 1000.0
                    speed = wrist_speed(hand.landmarks, previous_hand[1], dt_s, aspect)
                previous_hand = (frame.timestamp_ms, hand.landmarks)
            else:
                previous_hand = None

            canvas = make_canvas(canvas_w, canvas_h)
            if step == "capture":
                progress, captured = detector.update(frame.timestamp_ms, speed, point)
                if captured is not None:
                    measures[targets[current][0]] = captured
                    current += 1
                    if current == len(targets):
                        try:
                            zone = zone_from_corners(measures, cal.target_margin, cal.min_size)
                            offset = center_offset(zone, measures["centre"])
                            warning = None
                            if offset > cal.max_center_offset:
                                warning = (f"Attention : le centre vise est decale de "
                                           f"{offset * 100:.0f} % de la zone. Verifie, ou "
                                           "recommence (R).")
                            reached = set()
                            pointer_filter.reset()
                            step = "validation"
                        except ValueError as exc:
                            message = str(exc)
                            step = "refus"
                if step == "capture":
                    draw_capture(canvas, targets, current, progress, point is not None,
                                 camera_image, detector.waiting_for_movement)

            if step == "validation":
                cursor = None
                if point is not None:
                    fx, fy = pointer_filter(point[0], point[1], frame.timestamp_ms / 1000.0)
                    cursor = zone_to_screen(fx, fy, zone)
                    for name, u, v in corner_targets:
                        near_x = abs(cursor[0] - u) < REACHED_DISTANCE
                        near_y = abs(cursor[1] - v) < REACHED_DISTANCE
                        if near_x and near_y:
                            reached.add(name)
                else:
                    pointer_filter.reset()
                draw_validation(canvas, corner_targets, reached, cursor, warning, camera_image)
            elif step == "refus":
                draw_refusal(canvas, message, camera_image)

            cv2.imshow(CALIBRATION_WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESCAPE:
                break  # annuler : rien n'est modifié
            if step == "capture" and key == KEY_BACKSPACE and current > 0:
                current -= 1  # reprendre la cible précédente
                measures.pop(targets[current][0], None)
                detector.reset()
            if step in ("validation", "refus") and key in (ord("r"), ord("R")):
                step = "capture"  # recommencer depuis le centre
                current = 0
                measures = {}
                detector.reset()
            if step == "validation" and key == KEY_ENTER:
                save_zone(LOCAL_CONFIG_PATH, zone)
                saved = True
                break

    cv2.destroyAllWindows()
    if not saved:
        print("Étalonnage annulé : config/local.yaml n'est pas modifié.")
        return
    print(f"Zone avant : x {old_zone.left:.2f}..{old_zone.right:.2f}, "
          f"y {old_zone.top:.2f}..{old_zone.bottom:.2f}")
    print(f"Zone après : x {zone.left:.2f}..{zone.right:.2f}, y {zone.top:.2f}..{zone.bottom:.2f}"
          f" ; centre décalé de {center_offset(zone, measures['centre']) * 100:.0f} %")
    print(f"Enregistrée dans {LOCAL_CONFIG_PATH} (pointer.zone). Lancer `airdesk` pour l'utiliser.")
