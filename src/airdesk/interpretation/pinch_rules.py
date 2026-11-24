"""Règles candidates pour décider qu'une main « pince », et logique d'entrée / sortie.

Problème : poing fermé = pouce posé sur l'index replié, donc pinch_ratio petit, donc un faux
pincement (« Midas touch », Jacob 1991). Les règles ajoutent des conditions sur l'extension
des doigts (voir features.py) :

    A    : mesure < enter                                              (règle d'origine)
    B    : A  et  index tendu (extension > min_index_extension)
    C    : B  et  au moins un doigt parmi majeur/annulaire/auriculaire non replié
    D    : A  et  au moins un doigt parmi majeur/annulaire/auriculaire non replié
           (pas de condition sur l'index, qui se plie pour pincer : voir le journal)
    B3D, C3D, D3D : mêmes règles avec les extensions 3D
    F    : D  et  doigts qui se FERMENT (closing_speed >= min_closing_speed)
    F3D  : D  et  closing_speed_3d >= min_closing_speed_3d
    (E = D avec la mesure « ratio3d » : distance pouce-index en 3D, voir MEASURES.)

F et F3D testent une hypothèse issue de l'usage réel : une main qui pivote de profil (en
allant viser la gauche de l'écran) voit son pouce passer DEVANT l'index pour la caméra, sans
geste de fermeture ; un vrai pincement, lui, rapproche activement les doigts.
"""

from __future__ import annotations

from dataclasses import dataclass

RULES = ["A", "B", "C", "D", "B3D", "C3D", "D3D", "F", "F3D"]

# Mesures de pincement possibles (valeur comparée à enter / exit) :
#   ratio   : distance bout du pouce -> bout de l'index / taille de main (pinch_ratio)
#   ratio3d : la même en 3D, avec la profondeur z (pinch_ratio_3d)
#   seg78   : distance bout du pouce -> segment 7-8 de l'index (pinch_seg)
#   seg68   : distance bout du pouce -> segment 6-8 de l'index (pinch_seg_68)
MEASURES = ["ratio", "ratio3d", "seg78", "seg68"]

# Pour chaque règle : (index tendu exigé ?, autre doigt non replié exigé ?, extensions 3D ?,
# vitesse de fermeture exigée : None, "2d" ou "3d")
RULE_CONDITIONS = {
    "A": (False, False, False, None),
    "B": (True, False, False, None),
    "C": (True, True, False, None),
    "D": (False, True, False, None),
    "B3D": (True, False, True, None),
    "C3D": (True, True, True, None),
    "D3D": (False, True, True, None),
    "F": (False, True, False, "2d"),
    "F3D": (False, True, False, "3d"),
}

# Raisons d'un refus d'entrée en pincement (colonne entry_block du log), dans l'ordre testé.
BLOCK_RATIO = "ratio"  # doigts pas assez proches (mesure >= enter)
BLOCK_PALM = "paume"  # paume de profil (require_palm_facing)
BLOCK_INDEX = "index"  # index pas assez tendu (règles B, C)
BLOCK_OTHER_FINGERS = "autres_doigts"  # majeur, annulaire et auriculaire repliés (poing)
BLOCK_CLOSING = "fermeture"  # doigts qui ne se ferment pas assez vite (règles F, F3D)


@dataclass(frozen=True)
class RuleThresholds:
    enter: float = 0.25
    min_index_extension: float = 1.0
    folded_below: float = 0.8
    min_index_extension_3d: float = 1.0
    folded_below_3d: float = 0.8
    # Condition supplémentaire, quelle que soit la règle : paume tournée vers la caméra.
    # Une main au repos de profil (pouce contre l'index) ressemble sinon à un pincement.
    require_palm_facing: bool = False
    min_palm_width: float = 0.0
    # Règles F / F3D : vitesse de fermeture minimale (baisse de la mesure par seconde).
    min_closing_speed: float = 0.0
    min_closing_speed_3d: float = 0.0


def entry_block_reason(
    rule: str,
    pinch_value: float,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
    palm_width: float | None = None,
    closing_speed: float | None = None,
    closing_speed_3d: float | None = None,
) -> str | None:
    """Pourquoi la règle refuse d'entrer en pincement : None si elle accepte, sinon la PREMIÈRE
    condition qui manque (BLOCK_RATIO, BLOCK_PALM, BLOCK_INDEX, BLOCK_OTHER_FINGERS,
    BLOCK_CLOSING). Sert à la décision ET au log (comprendre un pincement refusé).

    `pinch_value` : la mesure choisie (pinch_ratio par défaut, voir MEASURES).
    `palm_width` inconnue (None) ne bloque pas ; une vitesse de fermeture inconnue bloque les
    règles F / F3D (pas assez d'historique pour savoir si les doigts se ferment).
    """
    if rule not in RULE_CONDITIONS:
        raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
    needs_index, needs_other_finger, use_3d, closing = RULE_CONDITIONS[rule]

    if not pinch_value < thresholds.enter:
        return BLOCK_RATIO

    if thresholds.require_palm_facing and palm_width is not None:
        if palm_width < thresholds.min_palm_width:
            return BLOCK_PALM

    if use_3d:
        values = extensions_3d
        min_index = thresholds.min_index_extension_3d
        folded_below = thresholds.folded_below_3d
    else:
        values = extensions
        min_index = thresholds.min_index_extension
        folded_below = thresholds.folded_below

    if needs_index and not values[0] > min_index:
        return BLOCK_INDEX

    if needs_other_finger and not other_finger_open(values, folded_below):
        return BLOCK_OTHER_FINGERS

    if closing == "2d":
        if closing_speed is None or closing_speed < thresholds.min_closing_speed:
            return BLOCK_CLOSING
    elif closing == "3d":
        if closing_speed_3d is None or closing_speed_3d < thresholds.min_closing_speed_3d:
            return BLOCK_CLOSING

    return None


def pinch_candidate(
    rule: str,
    pinch_value: float,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
    palm_width: float | None = None,
    closing_speed: float | None = None,
    closing_speed_3d: float | None = None,
) -> bool:
    """Vrai si la règle `rule` considère que la main entre en pincement."""
    reason = entry_block_reason(rule, pinch_value, extensions, extensions_3d, thresholds,
                                palm_width, closing_speed, closing_speed_3d)
    return reason is None


def other_finger_open(extensions: tuple[float, ...], folded_below: float) -> bool:
    """Vrai si au moins un doigt parmi majeur, annulaire, auriculaire n'est pas replié.

    Un poing a tous ses doigts repliés ; un vrai pincement en laisse au moins un ouvert.
    """
    for i in range(1, 4):
        if extensions[i] >= folded_below:
            return True
    return False


def fingers_folded_for_rule(
    rule: str,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
) -> bool:
    """Vrai si la règle exige un autre doigt ouvert (C, D...) et qu'aucun ne l'est plus."""
    _, needs_other_finger, use_3d, _ = RULE_CONDITIONS[rule]
    if not needs_other_finger:
        return False
    if use_3d:
        return not other_finger_open(extensions_3d, thresholds.folded_below_3d)
    return not other_finger_open(extensions, thresholds.folded_below)


def pinch_measure(features, measure: str) -> float:
    """Valeur de la mesure de pincement choisie, pour des descripteurs HandFeatures."""
    if measure == "ratio":
        return features.pinch_ratio
    if measure == "ratio3d":
        return features.pinch_ratio_3d
    if measure == "seg78":
        return features.pinch_seg
    if measure == "seg68":
        return features.pinch_seg_68
    raise ValueError(f"Mesure de pincement inconnue : '{measure}'. "
                     f"Choix possibles : {', '.join(MEASURES)}.")


class ClosingSpeed:
    """Vitesse de fermeture des doigts : baisse de la mesure de pincement par seconde.

    closing_speed(t) = (mesure(t - window_ms) - mesure(t)) / (window_ms en s), positive quand
    les doigts se rapprochent. None tant qu'il manque `window_ms` d'historique (début, ou
    après une perte de la main : appeler reset()). La valeur « à t − window_ms » est la plus
    récente mesure au moins aussi ancienne.
    """

    def __init__(self, window_ms: float = 200):
        self.window_ms = window_ms
        self.reset()

    def reset(self) -> None:
        self._history: list[tuple[float, float]] = []  # (t_ms, mesure)

    def update(self, t_ms: float, value: float) -> float | None:
        self._history.append((t_ms, value))
        # On garde juste ce qu'il faut : la plus récente mesure plus vieille que la fenêtre.
        while len(self._history) >= 2 and t_ms - self._history[1][0] >= self.window_ms:
            self._history.pop(0)
        old_t, old_value = self._history[0]
        if t_ms - old_t < self.window_ms:
            return None
        return (old_value - value) / ((t_ms - old_t) / 1000.0)


class PinchHysteresis:
    """Entrée et sortie du pincement, image par image.

    - Entrée : la règle dit « pincement » (voir entry_block_reason). Après chaque appel,
      `last_block` donne la raison du refus d'entrée (None si accepté ou déjà pincé).
    - Sortie : mesure >= exit (hystérésis : exit > enter), ou, si `exit_on_fold`, plus aucun
      autre doigt ouvert pour une règle qui l'exige (C, D...).
    - La condition de sortie doit tenir pendant `release_grace_ms` : une seule image
      « relâchée » (le suivi qui décroche un instant) ne fait pas lâcher un glisser. Avec 0,
      une image suffit (comportement d'origine).
    - Une fois un relâchement commencé, il n'est ANNULÉ que si la mesure redescend sous
      `exit - release_cancel_margin`. Constaté en usage réel : en relâchant lentement, la
      mesure oscille autour d'exit (0,3983, 0,4035, 0,3969, 0,4014...) ; sans marge, chaque
      passage juste sous exit annulait la confirmation, et un clic de 344 ms devenait un
      glisser. Avec 0 : comportement d'origine.

    `update` renvoie "start", "end" ou None. `release_ms` : début du relâchement en cours
    (None sinon) ; après "end", c'est l'instant où le relâchement a commencé, la vraie fin du
    pincement. `releasing` : un relâchement est en cours de confirmation.
    Les images sans main ne passent pas ici : la machine à états ne relâche pas sur une image
    sans main, seulement après lost_timeout_ms.
    """

    def __init__(
        self,
        rule: str,
        thresholds: RuleThresholds,
        exit: float,
        release_grace_ms: float = 0,
        exit_on_fold: bool = False,
        release_cancel_margin: float = 0.0,
    ):
        if rule not in RULE_CONDITIONS:
            raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
        self.rule = rule
        self.thresholds = thresholds
        self.exit = exit
        self.release_grace_ms = release_grace_ms
        self.exit_on_fold = exit_on_fold
        self.release_cancel_margin = release_cancel_margin
        self.reset()

    def reset(self) -> None:
        self.pinched = False
        self.release_ms: float | None = None  # début du relâchement en cours, sinon None
        self.last_block: str | None = None

    @property
    def releasing(self) -> bool:
        return self.pinched and self.release_ms is not None

    def update(
        self,
        t_ms: float,
        pinch_value: float,
        extensions: tuple[float, ...],
        extensions_3d: tuple[float, ...],
        palm_width: float | None = None,
        closing_speed: float | None = None,
        closing_speed_3d: float | None = None,
    ) -> str | None:
        if not self.pinched:
            self.last_block = entry_block_reason(
                self.rule, pinch_value, extensions, extensions_3d, self.thresholds,
                palm_width, closing_speed, closing_speed_3d,
            )
            if self.last_block is None:
                self.pinched = True
                self.release_ms = None
                return "start"
            return None

        self.last_block = None
        releasing = pinch_value >= self.exit
        if self.exit_on_fold and fingers_folded_for_rule(
            self.rule, extensions, extensions_3d, self.thresholds
        ):
            releasing = True
        if not releasing:
            # Un relâchement en cours n'est annulé que par un vrai repincement.
            if self.release_ms is not None and \
                    pinch_value >= self.exit - self.release_cancel_margin:
                releasing = True
            else:
                self.release_ms = None  # fausse alerte : on pince toujours
                return None
        if self.release_ms is None:
            self.release_ms = t_ms
        if t_ms - self.release_ms >= self.release_grace_ms:
            self.pinched = False
            return "end"
        return None
