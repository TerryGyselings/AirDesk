"""Pointage absolu : projeter la zone d'interaction de l'image sur l'écran.

En pointage relatif (comme un pavé tactile), le curseur et la main finissent par se
désynchroniser : chaque petite erreur s'accumule. En pointage absolu, une position de la
main correspond toujours à la même position du curseur.

On ne projette pas toute l'image : ses bords sont difficiles à atteindre (la main sort du
champ et n'est plus détectée). Une zone centrale, plus petite, est étirée sur l'écran ; hors
zone, le curseur reste collé au bord.

Logique pure : on travaille en fractions de l'écran (0 à 1). La conversion en pixels, qui
dépend de l'écran, est faite par l'actionneur (actuator/screen.py).
"""

from __future__ import annotations

import statistics

from airdesk.config import ZoneConfig


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def zone_to_screen(x: float, y: float, zone: ZoneConfig) -> tuple[float, float]:
    """Position dans l'image (0 à 1) -> position sur l'écran (0 à 1), collée aux bords.

    Exemple avec la zone par défaut (15 %..85 % en x) : x = 0,15 -> 0 (bord gauche),
    x = 0,50 -> 0,5 (milieu), x = 0,85 -> 1 (bord droit), x = 0,95 -> 1 (collé au bord).
    """
    width = zone.right - zone.left
    height = zone.bottom - zone.top
    if width <= 0 or height <= 0:
        raise ValueError("Zone d'interaction vide : il faut left < right et top < bottom.")
    u = (x - zone.left) / width
    v = (y - zone.top) / height
    return clamp(u), clamp(v)


def in_zone(x: float, y: float, zone: ZoneConfig) -> bool:
    """Vrai si le point est dans la zone d'interaction (bords compris, sans bornage)."""
    return zone.left <= x <= zone.right and zone.top <= y <= zone.bottom


def below_zone(y: float, zone: ZoneConfig) -> bool:
    """Vrai si le point est sous le bord bas de la zone (main baissée)."""
    return y > zone.bottom


# --- Étalonnage guidé de la zone d'interaction -------------------------------------------------
#
# L'utilisateur vise 5 cibles affichées en plein écran (centre puis 4 coins), à son rythme. Pour
# chacune, il place sa main là où il voudrait être pour viser ce point, confortablement, et la
# garde immobile : la position est capturée automatiquement. Les 4 coins donnent la zone, le
# centre sert de contrôle. (Un premier essai, 5 s de mouvements libres, donnait des zones
# incohérentes : un nuage de points ne dit pas où l'on veut viser.)

TARGETS = ["centre", "haut_gauche", "haut_droite", "bas_droite", "bas_gauche"]


def target_position(name: str, margin: float = 0.03) -> tuple[float, float]:
    """Position d'une cible à l'écran, en fractions (0 à 1). Coins à `margin` des bords."""
    low = margin
    high = 1.0 - margin
    positions = {
        "centre": (0.5, 0.5),
        "haut_gauche": (low, low),
        "haut_droite": (high, low),
        "bas_droite": (high, high),
        "bas_gauche": (low, high),
    }
    if name not in positions:
        raise ValueError(f"Cible inconnue : '{name}'. Choix : {', '.join(TARGETS)}.")
    return positions[name]


class StillnessDetector:
    """Détecte une main immobile pendant `hold_ms` et renvoie sa position médiane.

    À chaque image, `update` reçoit la vitesse du poignet (tailles de main par seconde) et le
    point de pointage. Tant que la vitesse reste sous `max_speed`, la progression monte de 0
    à 1 ; si la main bouge, disparaît (point None) ou si la vitesse est inconnue (None), tout
    repart de zéro. Une fois `hold_ms` atteint, le point capturé est la MÉDIANE des points de
    la période (x et y séparément) : un point aberrant isolé ne la déplace pas.

    Après une capture, la main est encore immobile : sans précaution, la cible suivante serait
    capturée au même endroit. Le détecteur attend donc d'abord un MOUVEMENT (une vitesse au-
    dessus du seuil) avant de pouvoir capturer de nouveau.
    """

    def __init__(self, max_speed: float = 0.5, hold_ms: float = 1000):
        self.max_speed = max_speed
        self.hold_ms = hold_ms
        self._needs_movement = False
        self.reset()

    def reset(self) -> None:
        """Repart de zéro (sans exiger de mouvement : pour revenir à une cible, par exemple)."""
        self._start_ms: float | None = None
        self._points: list[tuple[float, float]] = []
        self._needs_movement = False

    @property
    def waiting_for_movement(self) -> bool:
        return self._needs_movement

    def update(
        self, t_ms: float, wrist_speed: float | None, point: tuple[float, float] | None
    ) -> tuple[float, tuple[float, float] | None]:
        """(progression de 0 à 1, point capturé ou None)."""
        if self._needs_movement:
            if wrist_speed is not None and wrist_speed >= self.max_speed:
                self._needs_movement = False  # la main est partie vers la cible suivante
            return 0.0, None
        still = point is not None and wrist_speed is not None and wrist_speed < self.max_speed
        if not still:
            self._start_ms = None
            self._points = []
            return 0.0, None
        if self._start_ms is None:
            self._start_ms = t_ms
        self._points.append(point)
        held = t_ms - self._start_ms
        if held < self.hold_ms:
            return held / self.hold_ms, None
        xs = []
        ys = []
        for x, y in self._points:
            xs.append(x)
            ys.append(y)
        captured = (statistics.median(xs), statistics.median(ys))
        self.reset()
        self._needs_movement = True
        return 1.0, captured


def zone_from_corners(
    corners: dict[str, tuple[float, float]],
    target_margin: float = 0.03,
    min_size: float = 0.15,
) -> ZoneConfig:
    """Zone d'interaction à partir des 4 coins mesurés (positions de la main dans l'image).

    1. Chaque bord est la moyenne des deux mesures qui le concernent : gauche = moyenne des x de
       haut_gauche et bas_gauche, haut = moyenne des y de haut_gauche et haut_droite, etc.
    2. Les cibles étaient à `target_margin` des bords de l'écran (et non sur le bord) : on
       étire la zone d'autant, pour que la main mesurée tombe exactement sur la cible.
       Ex. avec 3 % : largeur = (droite - gauche) / 0,94.
    3. On borne à l'image [0, 1].

    Refus (ValueError, en français) si les coins sont dans le désordre ou si la zone mesurée
    fait moins de `min_size` de large ou de haut.
    """
    for name in ("haut_gauche", "haut_droite", "bas_droite", "bas_gauche"):
        if name not in corners:
            raise ValueError(f"Coin manquant : {name}.")
    tl = corners["haut_gauche"]
    tr = corners["haut_droite"]
    br = corners["bas_droite"]
    bl = corners["bas_gauche"]

    problems = []
    if tl[1] >= bl[1] or tr[1] >= br[1]:
        problems.append("un coin du haut est plus bas que le coin du bas correspondant")
    if tl[0] >= tr[0] or bl[0] >= br[0]:
        problems.append("un coin de gauche est plus à droite que le coin de droite correspondant")
    if problems:
        raise ValueError("Coins dans le désordre : " + " ; ".join(problems) + ". Recommencer "
                         "(R) en visant les cibles dans l'ordre affiché.")

    left = (tl[0] + bl[0]) / 2
    right = (tr[0] + br[0]) / 2
    top = (tl[1] + tr[1]) / 2
    bottom = (bl[1] + br[1]) / 2
    width = right - left
    height = bottom - top
    if width < min_size or height < min_size:
        raise ValueError(f"Zone trop petite ({width:.2f} x {height:.2f}, minimum "
                         f"{min_size:.2f}) : écarter davantage la main entre les coins, ou se "
                         "rapprocher de la caméra. Recommencer (R).")

    # Étirement : les mesures correspondent aux cibles, à target_margin des bords de l'écran.
    full_width = width / (1 - 2 * target_margin)
    full_height = height / (1 - 2 * target_margin)
    return ZoneConfig(
        left=clamp(left - target_margin * full_width),
        right=clamp(right + target_margin * full_width),
        top=clamp(top - target_margin * full_height),
        bottom=clamp(bottom + target_margin * full_height),
    )


def center_offset(zone: ZoneConfig, center_point: tuple[float, float]) -> float:
    """Écart du centre mesuré au milieu de la zone, en fractions de la zone (0 = parfait).

    Un grand écart signale une mesure ratée ou une posture très asymétrique : la zone
    calculée depuis les coins ne correspond pas à l'endroit où l'on vise le centre.
    """
    u, v = zone_to_screen(center_point[0], center_point[1], zone)
    return max(abs(u - 0.5), abs(v - 0.5))
