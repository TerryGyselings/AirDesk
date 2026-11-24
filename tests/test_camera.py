"""Tests de l'ouverture de la caméra, avec une fausse VideoCapture (aucune webcam nécessaire).

La fausse caméra se comporte selon un « scénario » : pour chaque (index, backend, largeur
demandée), on décrit ce qu'elle fait (s'ouvre ou non, image noire, figée, lente, bloquée...).
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from airdesk.capture.camera import (
    BACKENDS,
    STATUS_BLACK,
    STATUS_ERROR,
    STATUS_FROZEN,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_TOO_SLOW,
    Camera,
    backends_for_platform,
    fourcc_to_str,
    frame_difference,
    mean_brightness,
    open_capture,
    probe_with_timeout,
    resolutions_to_try,
)

# Pour retrouver le nom d'un backend à partir de la constante OpenCV reçue par la fabrique.
BACKEND_NAMES = {}
for _name, _code in BACKENDS.items():
    BACKEND_NAMES[_code] = _name


@dataclass
class Behavior:
    """Comportement de la fausse caméra pour un essai donné."""

    opens: bool = True
    brightness: int = 100  # 0 = image noire
    fps: float = 30.0  # cadence simulée (via la fausse horloge)
    frozen: bool = False  # toujours exactement la même image
    blocks: bool = False  # read() ne rend jamais la main (comme MSMF)
    raises: bool = False  # read() lève une exception (pilote défaillant)


class FakeClock:
    """Horloge contrôlée par la fausse caméra : chaque image « coûte » 1/fps seconde."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeCapture:
    """Imite cv2.VideoCapture : isOpened, set, get, read, release."""

    def __init__(self, behaviors, index, backend, clock, unblock):
        self.behaviors = behaviors
        self.index = index
        self.backend = backend
        self.clock = clock
        self.unblock = unblock
        self.props = {}
        self.released = False
        self.reads = 0

    def behavior(self) -> Behavior:
        width = int(self.props.get(3, 0))  # 3 = cv2.CAP_PROP_FRAME_WIDTH
        key = (self.index, self.backend, width)
        if key in self.behaviors:
            return self.behaviors[key]
        # Pas de règle pour cette largeur : règle valable pour toutes les largeurs (None),
        # ou, si aucune largeur n'est encore demandée (isOpened avant set), n'importe laquelle.
        for (index, backend, rule_width), behavior in self.behaviors.items():
            if index != self.index or backend != self.backend:
                continue
            if rule_width is None or width == 0:
                return behavior
        return Behavior(opens=False)

    def isOpened(self):
        return self.behavior().opens

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        b = self.behavior()
        if b.blocks:
            self.unblock.wait()  # bloque jusqu'à la fin du test
            return False, None
        if b.raises:
            raise RuntimeError("pilote en panne")
        self.reads += 1
        self.clock.now += 1.0 / b.fps
        width = int(self.props.get(3, 640))
        height = int(self.props.get(4, 480))  # 4 = cv2.CAP_PROP_FRAME_HEIGHT
        value = b.brightness
        if not b.frozen and b.brightness > 0:
            # Bruit de capteur simulé : toute l'image alterne d'un niveau (écart moyen = 1).
            value += self.reads % 2
        # Images 40 fois plus petites que demandé : les tests restent rapides.
        image = np.full((height // 40, width // 40, 3), value, dtype=np.uint8)
        return True, image

    def release(self):
        self.released = True


class FakeFactory:
    """Remplace cv2.VideoCapture et garde la trace de toutes les captures créées."""

    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.clock = FakeClock()
        self.unblock = threading.Event()
        self.created = []

    def __call__(self, index, api):
        cap = FakeCapture(self.behaviors, index, BACKEND_NAMES[api], self.clock, self.unblock)
        self.created.append(cap)
        return cap


@pytest.fixture
def make_factory():
    """Crée des fausses fabriques et débloque leurs threads à la fin du test."""
    factories = []

    def make(behaviors):
        factory = FakeFactory(behaviors)
        factories.append(factory)
        return factory

    yield make
    for factory in factories:
        factory.unblock.set()


def make_camera(factory, index=0, **kwargs):
    """Caméra « Windows » branchée sur la fausse fabrique, avec un délai court pour les tests."""
    return Camera(
        index,
        system="Windows",
        timeout_s=0.3,
        capture_factory=factory,
        clock=factory.clock,
        **kwargs,
    )


# --- Fonctions pures --------------------------------------------------------------------------


def test_backend_order_depends_on_platform():
    assert backends_for_platform("Windows") == ["dshow", "msmf"]
    assert backends_for_platform("Darwin") == ["avfoundation"]
    assert backends_for_platform("Linux") == ["v4l2"]
    assert backends_for_platform("SunOS") == ["any"]


def test_forced_backend_is_the_only_one_tried():
    assert backends_for_platform("Windows", "msmf") == ["msmf"]


def test_unknown_backend_is_rejected_in_french():
    with pytest.raises(ValueError, match="Backend caméra inconnu"):
        backends_for_platform("Windows", "directx")


def test_resolutions_fall_back_to_640x480_without_duplicate():
    assert resolutions_to_try(1280, 720) == [(1280, 720), (640, 480)]
    assert resolutions_to_try(640, 480) == [(640, 480)]


def test_brightness_of_black_and_white_images():
    assert mean_brightness(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0
    assert mean_brightness(np.full((4, 4, 3), 255, dtype=np.uint8)) == 255.0


def test_frame_difference_does_not_wrap_around():
    """En uint8, 10 - 20 donnerait 246 : on vérifie que l'écart vaut bien 10."""
    a = np.full((2, 2), 10, dtype=np.uint8)
    b = np.full((2, 2), 20, dtype=np.uint8)
    assert frame_difference(a, b) == 10.0
    assert frame_difference(a, a) == 0.0


def test_fourcc_to_str():
    mjpg = ord("M") | ord("J") << 8 | ord("P") << 16 | ord("G") << 24
    assert fourcc_to_str(mjpg) == "MJPG"
    assert fourcc_to_str(0) == "?"
    assert fourcc_to_str(22) == "#22"  # code non imprimable renvoyé par MSMF


# --- Essai isolé ------------------------------------------------------------------------------


def test_probe_gives_up_on_a_driver_that_never_answers(make_factory):
    factory = make_factory({(0, "msmf", None): Behavior(blocks=True)})
    t_start = time.perf_counter()
    result = probe_with_timeout(
        0.2, index=0, backend="msmf", width=640, height=480, capture_factory=factory
    )
    assert result.status == STATUS_TIMEOUT
    assert time.perf_counter() - t_start < 2.0  # on n'a pas attendu indéfiniment


def test_probe_reports_driver_exceptions_instead_of_crashing(make_factory):
    factory = make_factory({(0, "dshow", None): Behavior(raises=True)})
    result = probe_with_timeout(
        1.0, index=0, backend="dshow", width=640, height=480, capture_factory=factory
    )
    assert result.status == STATUS_ERROR
    assert "pilote en panne" in result.detail
    assert factory.created[0].released


# --- Repli automatique ------------------------------------------------------------------------


def test_first_backend_that_works_is_kept_open(make_factory):
    factory = make_factory({(0, "dshow", None): Behavior()})
    with make_camera(factory) as cam:
        assert cam.info.backend == "dshow"
        assert (cam.info.width, cam.info.height) == (1280 // 40, 720 // 40)
        assert len(cam.attempts) == 1
        assert not factory.created[0].released  # la capture retenue reste ouverte
    assert factory.created[0].released  # et elle est libérée en sortie de `with`


def test_black_image_falls_back_to_next_backend(make_factory):
    factory = make_factory(
        {(0, "dshow", None): Behavior(brightness=0), (0, "msmf", None): Behavior()}
    )
    with make_camera(factory) as cam:
        assert cam.info.backend == "msmf"
        assert cam.attempts[0].status == STATUS_BLACK
        assert factory.created[0].released  # la capture noire a bien été rendue


def test_frozen_image_is_rejected(make_factory):
    """Cas de l'OBS Virtual Camera éteinte : image correcte mais toujours identique."""
    factory = make_factory(
        {(0, "dshow", None): Behavior(frozen=True), (0, "msmf", None): Behavior()}
    )
    with make_camera(factory) as cam:
        assert cam.attempts[0].status == STATUS_FROZEN
        assert cam.info.backend == "msmf"


def test_slow_720p_then_blocked_msmf_falls_back_to_640x480(make_factory):
    """Cas réel de la C920 : DSHOW en YUY2 à 10 i/s en 720p, MSMF bloqué, 640x480 OK."""
    factory = make_factory(
        {
            (0, "dshow", 1280): Behavior(fps=10),
            (0, "dshow", 640): Behavior(fps=30),
            (0, "msmf", None): Behavior(blocks=True),
        }
    )
    with make_camera(factory) as cam:
        statuses = []
        for attempt in cam.attempts:
            statuses.append(attempt.status)
        assert statuses == [STATUS_TOO_SLOW, STATUS_TIMEOUT, STATUS_OK]
        assert cam.info.backend == "dshow"
        assert cam.info.requested == (640, 480)


def test_slow_camera_is_accepted_at_low_resolution(make_factory):
    """Le seuil de cadence ne s'applique qu'à partir de 1280x720."""
    factory = make_factory({(0, "dshow", None): Behavior(fps=12)})
    with make_camera(factory, width=640, height=480) as cam:
        assert cam.info.ok


def test_forced_backend_is_respected(make_factory):
    factory = make_factory({(0, "dshow", None): Behavior(), (0, "msmf", None): Behavior()})
    with make_camera(factory, backend="msmf") as cam:
        assert cam.info.backend == "msmf"


def test_open_capture_tries_backends_before_lowering_resolution(make_factory):
    factory = make_factory({})  # rien ne s'ouvre
    result, attempts = open_capture(
        0, ["dshow", "msmf"], [(1280, 720), (640, 480)], capture_factory=factory
    )
    assert result is None
    order = []
    for attempt in attempts:
        order.append((attempt.backend, attempt.requested))
    assert order == [
        ("dshow", (1280, 720)),
        ("msmf", (1280, 720)),
        ("dshow", (640, 480)),
        ("msmf", (640, 480)),
    ]


def test_total_failure_gives_clear_french_error_with_working_index(make_factory):
    """Index 0 noir partout ; l'index 1 marche : le message doit le suggérer."""
    factory = make_factory(
        {
            (0, "dshow", None): Behavior(brightness=0),
            (0, "msmf", None): Behavior(brightness=0),
            (1, "dshow", None): Behavior(),
        }
    )
    with pytest.raises(RuntimeError) as error:
        with make_camera(factory, index=0):
            pass
    message = str(error.value)
    assert "caméra 0" in message
    assert "image noire" in message
    assert "autre application" in message
    assert "Confidentialité" in message
    assert "index 1" in message
    for cap in factory.created:
        assert cap.released  # aucune caméra laissée ouverte après l'échec
