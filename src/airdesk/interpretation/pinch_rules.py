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
BLOCK_MOTION = "mouvement"  # poignet trop rapide au contact (max_contact_speed)


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
    # Vitesse du poignet maximale au contact (tailles de main / s) ; 0 = pas de condition.
    # Un vrai clic se fait main presque immobile ; un pincement pendant un geste rapide est
    # souvent involontaire (log 19-12-46 : 72 140 et 72 421 ms).
    max_contact_speed: float = 0.0
    # Règles F / F3D : la condition de fermeture n'est pas exigée pour un contact qui arrive
    # moins de closing_grace_ms après la fin du pincement précédent (2e tape d'un double-clic,
    # voir PinchHysteresis). 0 = pas d'exemption.
    closing_grace_ms: float = 0.0


def rule_has_entry_condition(rule: str) -> bool:
    """Vrai si la règle exige une condition qui ne se voit qu'AU CONTACT (vitesse de
    fermeture des doigts, règles F / F3D). Pendant un pincement maintenu, les doigts ne se
    ferment plus : la règle refuse ces images, et un rappel « image par image » n'a pas de
    sens pour elle (voir scripts/analyse_gestures.py)."""
    if rule not in RULE_CONDITIONS:
        raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
    return RULE_CONDITIONS[rule][3] is not None


def entry_block_reason(
    rule: str,
    pinch_value: float,
    extensions: tuple[float, ...],
    extensions_3d: tuple[float, ...],
    thresholds: RuleThresholds,
    palm_width: float | None = None,
    closing_speed: float | None = None,
    closing_speed_3d: float | None = None,
    wrist_speed: float | None = None,
    skip_closing: bool = False,
) -> str | None:
    """Pourquoi la règle refuse d'entrer en pincement : None si elle accepte, sinon la PREMIÈRE
    condition qui manque (BLOCK_RATIO, BLOCK_PALM, BLOCK_MOTION, BLOCK_INDEX,
    BLOCK_OTHER_FINGERS, BLOCK_CLOSING). Sert à la décision ET au log (comprendre un
    pincement refusé).

    `pinch_value` : la mesure choisie (pinch_ratio par défaut, voir MEASURES).
    `palm_width` inconnue (None) ne bloque pas ; une vitesse de fermeture inconnue bloque les
    règles F / F3D (pas assez d'historique pour savoir si les doigts se ferment).
    `wrist_speed` inconnue (main qui vient d'apparaître) ne bloque pas.
    `skip_closing` : la condition de fermeture (F / F3D) n'est pas exigée (exemption juste
    après un relâchement, voir PinchHysteresis) ; les autres conditions restent.
    """
    if rule not in RULE_CONDITIONS:
        raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
    needs_index, needs_other_finger, use_3d, closing = RULE_CONDITIONS[rule]

    if not pinch_value < thresholds.enter:
        return BLOCK_RATIO

    if thresholds.require_palm_facing and palm_width is not None:
        if palm_width < thresholds.min_palm_width:
            return BLOCK_PALM

    if thresholds.max_contact_speed > 0 and wrist_speed is not None:
        if wrist_speed >= thresholds.max_contact_speed:
            return BLOCK_MOTION

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

    if skip_closing:
        return None
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
    wrist_speed: float | None = None,
) -> bool:
    """Vrai si la règle `rule` considère que la main entre en pincement."""
    reason = entry_block_reason(rule, pinch_value, extensions, extensions_3d, thresholds,
                                palm_width, closing_speed, closing_speed_3d, wrist_speed)
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
    - Relâchement RAPIDE : au-dessus de `exit + release_fast_margin`, les doigts sont
      franchement ouverts, le relâchement est immédiat (sans délai). Le délai et la marge
      d'annulation ne servent qu'aux valeurs proches d'exit (oscillation). Constaté en usage
      réel : entre deux tapes rapides (double-clic), les doigts ne restent ouverts que 1 ou 2
      images ; avec le délai seul, les tapes fusionnaient en un seul appui (log 19-12-46,
      65-67 s : 4 appuis pour 14 tapes ; 11 avec une marge de 0,05). 0 = désactivé.
    - Exemption de fermeture (règles F / F3D, `thresholds.closing_grace_ms`) : un contact
      qui arrive moins de closing_grace_ms après la fin du pincement précédent n'a pas à
      montrer une vitesse de fermeture. Pour la 2e tape d'un double-clic, la vitesse est
      mesurée sur 200 ms et remonte jusqu'au creux de la 1re tape : elle est faible ou
      négative. Rejeu sur les logs réels : F à 0,4 aurait refusé 14 contacts, tous des 2es
      tapes 31 à 125 ms après un relâchement (dont les 4 doubles-clics du log
      2026-11-25_17-10-10). Le faux pincement que F vise (pouce posé, main de profil au
      repos) ne suit pas un relâchement récent.
      L'exemption exige aussi que l'appelant signale un vrai candidat 2e tape
      (`second_tap_candidate`, calculé par la machine : main près du point du clic précédent
      et poignet lent). Sans cela, en traversant l'écran vite après un clic, des contacts
      exemptés à tort devenaient des glisser (log 2026-11-25_17-54-53 : 62 125 et 64 827 ms,
      poignet à 3,46 et 4,43, à 252 et 203 px du clic). Par défaut : pas candidat.
    - Relâchement RELATIF (`release_rise`) : relâchement immédiat quand la mesure remonte de
      plus de `release_rise` au-dessus de son minimum depuis le contact (et au-dessus de
      enter), même sans atteindre exit. Deux tapes rapides font deux creux séparés par une
      petite remontée qui ne dépasse pas exit (log 2026-11-25_16-35-12 : 0,11 → 0,392 →
      0,128 à 13 250 ms ; 0,162 → 0,338 → 0,105 à 58 375 ms). C'est l'idée de la
      « proéminence » d'un détecteur de pics : on juge la remontée par rapport au creux, pas
      à un seuil fixe. Seulement si l'appelant l'autorise (`allow_rise_release`) : la machine
      ne le fait que pendant un appui sur place, car pendant un glisser les doigts bougent
      autant (ruptures, voir docs/journal.md). 0 = désactivé.

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
        release_fast_margin: float = 0.0,
        release_rise: float = 0.0,
    ):
        if rule not in RULE_CONDITIONS:
            raise ValueError(f"Règle inconnue : '{rule}'. Choix possibles : {', '.join(RULES)}.")
        self.rule = rule
        self.thresholds = thresholds
        self.exit = exit
        self.release_grace_ms = release_grace_ms
        self.exit_on_fold = exit_on_fold
        self.release_cancel_margin = release_cancel_margin
        self.release_fast_margin = release_fast_margin
        self.release_rise = release_rise
        self.reset()

    def reset(self) -> None:
        self.pinched = False
        self.release_ms: float | None = None  # début du relâchement en cours, sinon None
        self.last_block: str | None = None
        self.min_value: float | None = None  # minimum de la mesure depuis le contact
        self.last_end_ms: float | None = None  # fin du dernier pincement (exemption F / F3D)

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
        wrist_speed: float | None = None,
        allow_rise_release: bool = False,
        second_tap_candidate: bool = False,
    ) -> str | None:
        if not self.pinched:
            # Exemption F / F3D : 2e tape d'un double-clic, juste après un relâchement ET
            # signalée comme candidate par l'appelant (main lente, près du clic précédent).
            skip_closing = False
            grace = self.thresholds.closing_grace_ms
            recent = self.last_end_ms is not None and t_ms - self.last_end_ms <= grace
            if grace > 0 and recent and second_tap_candidate:
                skip_closing = True
            self.last_block = entry_block_reason(
                self.rule, pinch_value, extensions, extensions_3d, self.thresholds,
                palm_width, closing_speed, closing_speed_3d, wrist_speed, skip_closing,
            )
            if self.last_block is None:
                self.pinched = True
                self.release_ms = None
                self.min_value = pinch_value
                return "start"
            return None

        self.last_block = None
        # Remontée nette au-dessus du creux (tape suivante d'un double-clic) : relâchement.
        if self.min_value is None or pinch_value < self.min_value:
            self.min_value = pinch_value
        if (allow_rise_release and self.release_rise > 0
                and pinch_value >= self.min_value + self.release_rise
                and pinch_value >= self.thresholds.enter):
            if self.release_ms is None:
                self.release_ms = t_ms
            self.pinched = False
            self.last_end_ms = t_ms
            return "end"
        # Doigts franchement ouverts : relâchement immédiat, sans délai de confirmation.
        if self.release_fast_margin > 0 and pinch_value > self.exit + self.release_fast_margin:
            if self.release_ms is None:
                self.release_ms = t_ms
            self.pinched = False
            self.last_end_ms = t_ms
            return "end"
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
            self.last_end_ms = t_ms
            return "end"
        return None
