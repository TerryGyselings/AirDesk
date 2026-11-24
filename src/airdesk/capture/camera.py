"""Lecture de la webcam avec OpenCV."""

from __future__ import annotations

import platform
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import cv2
import numpy as np

# --- Backends vidéo -------------------------------------------------------------------------
# OpenCV ne parle pas directement à la webcam : il passe par une API du système, le « backend ».
# Sous Windows il en existe deux. Sans précision, OpenCV prend MSMF (Media Foundation), qui
# renvoie chez certaines webcams des images entièrement noires, voire bloque indéfiniment.
# DirectShow (DSHOW), plus ancien, est en pratique plus fiable : on l'essaie donc en premier.

BACKENDS: dict[str, int] = {
    "dshow": cv2.CAP_DSHOW,  # Windows, DirectShow
    "msmf": cv2.CAP_MSMF,  # Windows, Media Foundation (défaut d'OpenCV)
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
    "v4l2": cv2.CAP_V4L2,  # Linux, Video4Linux2
    "any": cv2.CAP_ANY,  # laisser OpenCV choisir
}

# Ordre d'essai par système, tel que renvoyé par platform.system().
# "any" n'y figure pas : sur ces systèmes, OpenCV choisirait l'un de ces mêmes backends,
# ce qui referait un essai identique (et, sous Windows, un second blocage MSMF possible).
PLATFORM_BACKENDS: dict[str, list[str]] = {
    "Windows": ["dshow", "msmf"],
    "Darwin": ["avfoundation"],
    "Linux": ["v4l2"],
}


def backends_for_platform(system: str, forced: str | None = None) -> list[str]:
    """Liste des backends à essayer, dans l'ordre, pour un système donné.

    `system` est la valeur de `platform.system()` ("Windows", "Darwin", "Linux").
    Si `forced` est donné (ex. "msmf"), on n'essaie que celui-là.
    Sur un système inconnu, on laisse OpenCV choisir ("any").
    """
    if forced is not None:
        if forced not in BACKENDS:
            noms = ", ".join(BACKENDS)
            raise ValueError(f"Backend caméra inconnu : '{forced}'. Choix possibles : {noms}.")
        return [forced]
    return list(PLATFORM_BACKENDS.get(system, ["any"]))


def resolutions_to_try(width: int, height: int) -> list[tuple[int, int]]:
    """Résolution demandée, puis 640x480 (que presque toutes les webcams savent fournir)."""
    resolutions = [(width, height)]
    if (width, height) != (640, 480):
        resolutions.append((640, 480))
    return resolutions


def mean_brightness(image: np.ndarray) -> float:
    """Luminosité moyenne d'une image, de 0 (noir) à 255 (blanc).

    Sert à détecter une image noire : une vraie scène, même sombre, dépasse rarement 5.
    """
    return float(np.mean(image))


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Écart moyen (0 à 255) entre deux images de même taille, pixel par pixel.

    Une vraie webcam a toujours un peu de bruit de capteur (écart de l'ordre de 1 à 3, même
    devant une scène immobile). Un écart nul signale une image figée : caméra virtuelle
    éteinte (OBS Virtual Camera affiche alors une image fixe), carte d'acquisition, etc.
    """
    # Conversion en int16 : en uint8, 10 - 20 donnerait 246 au lieu de -10.
    diff = a.astype(np.int16) - b.astype(np.int16)
    return float(np.mean(np.abs(diff)))


def fourcc_to_str(code: float) -> str:
    """Traduit le code FOURCC renvoyé par OpenCV en texte lisible ("MJPG", "YUY2"...).

    Un FOURCC est un nombre de 32 bits qui contient 4 caractères ASCII, un par octet,
    l'octet de poids faible en premier. Certains backends (MSMF) renvoient un autre code,
    non imprimable : on l'affiche alors en nombre, précédé de « # ».
    """
    value = int(code)
    if value <= 0:
        return "?"
    letters = ""
    for i in range(4):
        byte = (value >> (8 * i)) & 0xFF
        if byte == 0:
            continue
        if byte < 32 or byte > 126:  # pas un caractère ASCII imprimable
            return f"#{value}"
        letters += chr(byte)
    return letters.strip() or "?"


def configure_capture(
    cap: cv2.VideoCapture,
    backend: str,
    width: int,
    height: int,
    fps: int,
    mjpg_first: bool = True,
) -> None:
    """Demande le format, la résolution et la cadence voulus à une capture déjà ouverte.

    Avec DirectShow, on demande aussi le format MJPG (images compressées en JPEG par la
    webcam). Sinon DirectShow reste souvent en YUY2 (non compressé) : le débit USB ne laisse
    alors passer que 5 à 10 images/s en 1280x720. Selon les pilotes, MJPG n'est accepté que
    s'il est demandé avant la résolution, ou au contraire après : `mjpg_first` choisit l'ordre.

    La caméra est libre de refuser : il faut relire ce qu'on a obtenu ensuite.
    """
    mjpg = cv2.VideoWriter.fourcc(*"MJPG")
    if backend == "dshow" and mjpg_first:
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if backend == "dshow" and not mjpg_first:
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FPS, fps)


# --- Essai d'ouverture ----------------------------------------------------------------------

# Résultats possibles d'un essai.
STATUS_OK = "ok"
STATUS_CLOSED = "ne s'ouvre pas"
STATUS_NO_FRAME = "pas d'image"
STATUS_BLACK = "image noire"
STATUS_FROZEN = "image figée"
STATUS_TOO_SLOW = "trop lent"
STATUS_TIMEOUT = "bloqué (timeout)"
STATUS_ERROR = "erreur"


@dataclass
class ProbeResult:
    """Ce qu'a donné un essai d'ouverture (index, backend, résolution demandée)."""

    index: int
    backend: str
    requested: tuple[int, int]
    mjpg_first: bool
    status: str = STATUS_CLOSED
    width: int = 0  # résolution réellement obtenue
    height: int = 0
    pixel_format: str = "?"  # "MJPG", "YUY2"...
    fps: float = 0.0  # cadence mesurée (et non celle annoncée par le pilote)
    brightness: float = 0.0  # luminosité moyenne des images lues
    motion: float = 0.0  # écart moyen entre images successives (0 = image figée)
    detail: str = ""  # message d'erreur éventuel
    capture: cv2.VideoCapture | None = None  # capture encore ouverte si keep_open et status ok

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def describe(self) -> str:
        """Résumé d'une ligne, en français, pour les messages d'erreur."""
        w, h = self.requested
        text = f"index {self.index}, {self.backend}, {w}x{h} : {self.status}"
        if self.status in (STATUS_BLACK, STATUS_FROZEN, STATUS_TOO_SLOW, STATUS_OK):
            text += (
                f" (obtenu {self.width}x{self.height} {self.pixel_format}, "
                f"{self.fps:.1f} i/s, luminosité {self.brightness:.1f})"
            )
        if self.detail:
            text += f" ({self.detail})"
        return text


def probe(
    index: int,
    backend: str,
    width: int,
    height: int,
    fps: int = 30,
    frames: int = 10,
    min_brightness: float = 5.0,
    min_fps: float = 20.0,
    min_motion: float = 0.5,
    mjpg_first: bool = True,
    keep_open: bool = False,
    capture_factory: Callable = cv2.VideoCapture,
    clock: Callable[[], float] = time.perf_counter,
) -> ProbeResult:
    """Ouvre une caméra, lit quelques images de « chauffe » et juge si elle est utilisable.

    Échecs possibles : ne s'ouvre pas, ne rend pas d'image, image noire (luminosité moyenne
    sous `min_brightness`), image figée (écart moyen entre images successives sous
    `min_motion`), ou trop lente : en 1280x720 et au-delà, une cadence mesurée sous
    `min_fps` rendrait le curseur saccadé, on préfère alors une résolution plus basse.

    `capture_factory` et `clock` sont remplaçables pour tester sans webcam.
    Si `keep_open` est vrai et que l'essai réussit, la capture reste ouverte dans
    `result.capture` ; dans tous les autres cas elle est libérée.
    """
    result = ProbeResult(index, backend, (width, height), mjpg_first)
    cap = capture_factory(index, BACKENDS[backend])
    try:
        if not cap.isOpened():
            return result

        configure_capture(cap, backend, width, height, fps, mjpg_first)

        # Première image hors chronomètre : le démarrage du flux peut prendre une seconde.
        ok, image = cap.read()
        if not ok or image is None:
            result.status = STATUS_NO_FRAME
            return result

        frames_ok = 0
        brightness_sum = 0.0
        motion_sum = 0.0
        previous = image
        t_start = clock()
        for _ in range(frames):
            ok, image = cap.read()
            if not ok or image is None:
                continue
            frames_ok += 1
            brightness_sum += mean_brightness(image)
            if image.shape == previous.shape:
                motion_sum += frame_difference(image, previous)
            previous = image
        elapsed = clock() - t_start

        if frames_ok == 0:
            result.status = STATUS_NO_FRAME
            return result

        result.height, result.width = image.shape[:2]
        result.pixel_format = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        if elapsed > 0:
            result.fps = frames_ok / elapsed
        result.brightness = brightness_sum / frames_ok
        result.motion = motion_sum / frames_ok

        if result.brightness < min_brightness:
            result.status = STATUS_BLACK
        elif result.motion < min_motion:
            result.status = STATUS_FROZEN
        elif width * height >= 1280 * 720 and result.fps < min_fps:
            result.status = STATUS_TOO_SLOW
        else:
            result.status = STATUS_OK
        return result
    finally:
        if result.ok and keep_open:
            result.capture = cap
        else:
            cap.release()


def probe_with_timeout(timeout_s: float = 5.0, **probe_args) -> ProbeResult:
    """Lance `probe(...)` dans un thread et abandonne au bout de `timeout_s` secondes.

    Pourquoi un thread ? Certains pilotes (MSMF sous Windows) bloquent `VideoCapture(...)`
    ou `read()` indéfiniment, et Python ne peut pas interrompre un appel bloqué dans du code C.
    On fait donc l'essai à côté : si le délai est dépassé, on n'attend plus et on passe à la
    suite. Le thread est « daemon » : il ne retiendra pas la fermeture du programme.
    Limite : le thread bloqué garde la caméra ; si un jour il se débloque, il la libère.
    """
    box: dict = {"result": None, "abandoned": False}
    lock = threading.Lock()

    def worker() -> None:
        try:
            result = probe(**probe_args)
        except Exception as exc:  # cv2.error, pilote défaillant... on le note, sans planter
            requested = (probe_args["width"], probe_args["height"])
            mjpg_first = probe_args.get("mjpg_first", True)
            result = ProbeResult(probe_args["index"], probe_args["backend"], requested, mjpg_first)
            result.status = STATUS_ERROR
            result.detail = str(exc).strip()
        with lock:
            if box["abandoned"]:
                # Trop tard : personne n'attend plus ce résultat, on rend la caméra.
                if result.capture is not None:
                    result.capture.release()
                return
            box["result"] = result

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_s)

    with lock:
        if box["result"] is not None:
            return box["result"]
        box["abandoned"] = True

    requested = (probe_args["width"], probe_args["height"])
    mjpg_first = probe_args.get("mjpg_first", True)
    result = ProbeResult(probe_args["index"], probe_args["backend"], requested, mjpg_first)
    result.status = STATUS_TIMEOUT
    result.detail = f"aucune réponse en {timeout_s:g} s"
    return result


def open_capture(
    index: int,
    backends: list[str],
    resolutions: list[tuple[int, int]],
    timeout_s: float = 5.0,
    **probe_args,
) -> tuple[ProbeResult | None, list[ProbeResult]]:
    """Essaie chaque résolution, et pour chacune chaque backend, jusqu'au premier qui marche.

    Ordre : on garde d'abord la résolution demandée en changeant de backend, puis seulement
    on baisse la résolution (une image plus petite rend le suivi de la main moins précis).
    Renvoie (essai réussi avec sa capture ouverte, ou None ; liste de tous les essais).
    Les autres réglages (fps, seuils, capture_factory, clock) sont transmis à `probe`.
    """
    attempts: list[ProbeResult] = []
    for width, height in resolutions:
        for backend in backends:
            result = probe_with_timeout(
                timeout_s,
                index=index,
                backend=backend,
                width=width,
                height=height,
                keep_open=True,
                **probe_args,
            )
            attempts.append(result)
            if result.ok:
                return result, attempts
    return None, attempts


def find_working_indexes(
    candidates: list[int],
    backends: list[str],
    timeout_s: float = 5.0,
    **probe_args,
) -> list[int]:
    """Parmi `candidates`, les index de caméra qui donnent une image utilisable en 640x480.

    Sert à suggérer un autre index quand celui demandé ne marche pas : c'est le même test
    que `scripts/check_camera.py`, en plus court.
    """
    working: list[int] = []
    for index in candidates:
        for backend in backends:
            result = probe_with_timeout(
                timeout_s, index=index, backend=backend, width=640, height=480, **probe_args
            )
            if result.ok:
                working.append(index)
                break
    return working


def build_error_message(index: int, attempts: list[ProbeResult], working: list[int]) -> str:
    """Message d'erreur en français : ce qui a été essayé, puis quoi vérifier."""
    lines = [f"Impossible d'obtenir une image utilisable de la caméra {index}.", "Essais :"]
    for attempt in attempts:
        lines.append(f"  - {attempt.describe()}")
    lines.append("À vérifier :")
    if working:
        noms = ", ".join(str(i) for i in working)
        lines.append(
            f"  - une image a été trouvée sur l'index {noms} : `airdesk --camera {working[0]}`"
        )
    else:
        lines.append("  - un autre index : `airdesk --camera 1` (0 est parfois virtuel)")
    lines.append(
        "  - qu'aucune autre application n'utilise la caméra (Teams, Zoom, Discord, navigateur...)"
    )
    lines.append(
        "  - Windows : Paramètres > Confidentialité et sécurité > Caméra > autoriser les "
        "applications de bureau ; macOS : Réglages Système > Confidentialité > Caméra"
    )
    lines.append("  - le cache de l'objectif ou la touche de désactivation de la caméra")
    lines.append("  - un autre backend (`--backend dshow` ou `msmf`) ou `--width 640 --height 480`")
    lines.append("  - le diagnostic complet : `python scripts/check_camera.py`")
    return "\n".join(lines)


# --- Caméra utilisée par l'application ------------------------------------------------------


@dataclass
class Frame:
    """Une image de la webcam, déjà retournée en miroir (comme dans un miroir de salle de bain)."""

    bgr: np.ndarray  # format natif OpenCV, utile pour l'affichage
    rgb: np.ndarray  # format attendu par MediaPipe
    timestamp_ms: int  # horodatage monotone, obligatoire pour MediaPipe en mode VIDEO


class Camera:
    """Ouvre une webcam et fournit les images une par une.

    S'utilise avec `with` pour être sûr que la caméra est libérée :

        with Camera(0) as cam:
            print(cam.info.describe())  # backend, résolution et cadence réellement obtenus
            for frame in cam.frames():
                ...

    À l'ouverture, plusieurs backends puis une résolution plus basse sont essayés (voir
    `open_capture`). Les seuils ci-dessous concernent la caméra, pas les gestes : ils ne sont
    donc pas dans config/gestures.yaml.

    - `backend` : None = choix automatique selon le système, sinon "dshow", "msmf"...
    - `timeout_s` : durée maximale d'un essai (un pilote peut bloquer indéfiniment)
    - `warmup_frames` : images lues pour juger un essai (luminosité, mouvement, cadence)
    - `min_brightness` : luminosité moyenne minimale (0-255) ; en dessous, image noire
    - `min_motion` : écart moyen minimal entre deux images ; en dessous, image figée
    - `min_fps` : cadence minimale en 1280x720 et au-delà ; en dessous, résolution suivante
    - `system`, `capture_factory`, `clock` : remplaçables dans les tests
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: str | None = None,
        timeout_s: float = 5.0,
        warmup_frames: int = 10,
        min_brightness: float = 5.0,
        min_motion: float = 0.5,
        min_fps: float = 20.0,
        system: str | None = None,
        capture_factory: Callable = cv2.VideoCapture,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_s = timeout_s
        if system is None:
            system = platform.system()
        # Vérifié dès maintenant : un nom de backend inconnu lève ValueError tout de suite.
        self.backends = backends_for_platform(system, backend)
        self._probe_args = {
            "fps": fps,
            "frames": warmup_frames,
            "min_brightness": min_brightness,
            "min_motion": min_motion,
            "min_fps": min_fps,
            "capture_factory": capture_factory,
            "clock": clock,
        }
        self.info: ProbeResult | None = None  # l'essai retenu (backend, résolution, fps...)
        self.attempts: list[ProbeResult] = []  # tous les essais, y compris les échecs
        self._cap: cv2.VideoCapture | None = None
        self._t0 = time.monotonic()

    def __enter__(self) -> Camera:
        resolutions = resolutions_to_try(self.width, self.height)
        result, self.attempts = open_capture(
            self.index, self.backends, resolutions, self.timeout_s, **self._probe_args
        )
        if result is None:
            # Avant d'abandonner, on cherche si un autre index donne une image, pour le suggérer.
            others = []
            for i in range(4):
                if i != self.index:
                    others.append(i)
            working = find_working_indexes(
                others, self.backends, self.timeout_s, **self._probe_args
            )
            raise RuntimeError(build_error_message(self.index, self.attempts, working))
        self.info = result
        self._cap = result.capture
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("La caméra n'est pas ouverte : utiliser `with Camera(...) as cam`.")
        last_ts = -1
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                break
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = int((time.monotonic() - self._t0) * 1000)
            # MediaPipe exige des horodatages strictement croissants.
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            yield Frame(bgr=bgr, rgb=rgb, timestamp_ms=ts)
