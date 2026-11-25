from dataclasses import replace

import pytest

from airdesk.config import (
    EngagementConfig,
    FingersConfig,
    PinchConfig,
    PointerConfig,
    load_config,
)
from airdesk.interpretation.features import HandFeatures
from airdesk.interpretation.pinch_rules import MEASURES, RULES
from airdesk.interpretation.pointing import zone_to_screen
from airdesk.interpretation.state_machine import (
    Click,
    GestureStateMachine,
    Move,
    MoveTo,
    Press,
    PressAt,
    Release,
    ReleaseAt,
    State,
)


def features(
    pinch_ratio: float,
    extended_fingers: int,
    extension: float,
    pinch_seg: float | None = None,
    palm_width: float = 0.8,
) -> HandFeatures:
    """Descripteurs simplifiés : même extension (bout -> paume) pour les quatre doigts.

    `pinch_seg` vaut pinch_ratio par défaut ; `palm_width` 0.8 = paume de face.
    """
    if pinch_seg is None:
        pinch_seg = pinch_ratio
    extensions = (extension, extension, extension, extension)
    folded = 4 if extension < 0.8 else 0
    return HandFeatures(
        hand_size=0.2,
        pinch_ratio=pinch_ratio,
        extended_fingers=extended_fingers,
        open_palm=extended_fingers == 4,
        finger_extensions=extensions,
        index_extension=extension,
        folded_fingers=folded,
        finger_extensions_3d=extensions,
        index_extension_3d=extension,
        folded_fingers_3d=folded,
        pinch_seg=pinch_seg,
        pinch_seg_68=pinch_seg,
        palm_width=palm_width,
        palm_facing=palm_width / 0.8,
        pinch_ratio_3d=pinch_ratio,
    )


OPEN = features(pinch_ratio=0.8, extended_fingers=4, extension=1.5)
FIST = features(pinch_ratio=0.8, extended_fingers=0, extension=0.4)
PINCH = features(pinch_ratio=0.10, extended_fingers=3, extension=1.4)
# Entre les deux seuils (0.25 < 0.33 < 0.40) : sert à tester l'hystérésis.
HALF = features(pinch_ratio=0.33, extended_fingers=3, extension=1.4)

P = (0.5, 0.5)


def new_machine():
    return GestureStateMachine(EngagementConfig(hold_ms=300, lost_timeout_ms=1000), PinchConfig())


def activate(m):
    """Main ouverte de t=0 à t=330 ms -> état ACTIF."""
    t = 0
    while t <= 330:
        m.update(t, OPEN, P)
        t += 33
    assert m.state == State.ACTIVE
    return t


def test_starts_idle_and_ignores_moves():
    m = new_machine()
    events = m.update(0, FIST, (0.1, 0.1))
    events += m.update(33, FIST, (0.9, 0.9))
    assert m.state == State.IDLE
    assert events == []


def test_open_palm_must_be_held_to_activate():
    m = new_machine()
    m.update(0, OPEN, P)
    m.update(100, OPEN, P)
    assert m.state == State.IDLE  # pas encore 300 ms
    m.update(200, FIST, P)  # interruption : le compteur repart de zéro
    m.update(400, OPEN, P)
    m.update(600, OPEN, P)
    assert m.state == State.IDLE
    m.update(750, OPEN, P)
    assert m.state == State.ACTIVE


def test_moves_are_relative_once_active():
    m = new_machine()
    t = activate(m)
    events = m.update(t, OPEN, (0.6, 0.45))
    assert len(events) == 1
    assert isinstance(events[0], Move)
    assert abs(events[0].dx - 0.1) < 1e-9
    assert abs(events[0].dy + 0.05) < 1e-9


def test_short_pinch_is_a_click():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    assert m.state == State.PINCHING
    events = m.update(t + 150, OPEN, P)
    assert Click() in events
    assert m.state == State.ACTIVE


def test_long_pinch_is_a_drag():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    events = m.update(t + 400, PINCH, P)
    assert Press() in events
    assert m.state == State.DRAGGING

    events = m.update(t + 433, PINCH, (0.6, 0.5))  # pendant le glisser, le curseur suit
    assert len(events) == 1
    assert isinstance(events[0], Move)
    assert abs(events[0].dx - 0.1) < 1e-9

    events = m.update(t + 466, OPEN, (0.6, 0.5))
    assert Release() in events
    assert m.state == State.ACTIVE


def test_hysteresis_keeps_drag_when_fingers_open_slightly():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 400, PINCH, P)
    assert m.state == State.DRAGGING
    m.update(t + 433, HALF, P)  # doigts un peu entrouverts : on reste en glisser
    assert m.state == State.DRAGGING


def test_half_pinch_does_not_start_pinch():
    m = new_machine()
    t = activate(m)
    m.update(t, HALF, P)
    assert m.state == State.ACTIVE


def test_lost_hand_returns_to_idle_and_releases_drag():
    m = new_machine()
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 400, PINCH, P)
    assert m.state == State.DRAGGING

    events = m.update(t + 900, None, None)  # perdue depuis 500 ms : on attend encore
    assert m.state == State.DRAGGING
    assert events == []

    events = m.update(t + 1500, None, None)  # perdue depuis 1100 ms
    assert m.state == State.IDLE
    assert Release() in events


# --- Configuration choisie d'après les mesures : règle D, délai de relâchement 66 ms -----------
# Les tests ci-dessus utilisent PinchConfig() par défaut, c'est-à-dire la règle A d'origine :
# ils vérifient que « rule: A » redonne exactement l'ancien comportement.

# Poing fermé, pouce posé sur l'index replié : pinch_ratio petit, tous les doigts repliés.
FIST_THUMB_ON_INDEX = features(pinch_ratio=0.10, extended_fingers=0, extension=0.4)
ENGAGEMENT = EngagementConfig(hold_ms=300, lost_timeout_ms=1000)
FINGERS = FingersConfig(folded_below=0.87, folded_below_3d=0.91)


def machine_d(release_grace_ms=66, exit_on_fold=False, measure="ratio"):
    pinch = PinchConfig(rule="D", release_grace_ms=release_grace_ms, exit_on_fold=exit_on_fold,
                        measure=measure)
    return GestureStateMachine(ENGAGEMENT, pinch, FINGERS)


def feed(m, t, hand, duration_ms):
    """Envoie la même main pendant `duration_ms` (une image toutes les 33 ms)."""
    events = []
    end = t + duration_ms
    while t < end:
        events += m.update(t, hand, P)
        t += 33
    return t, events


def test_rule_a_takes_a_fist_for_a_pinch():
    """Le problème d'origine, reproduit : avec la règle A, le poing finit en glisser."""
    m = new_machine()
    t = activate(m)
    t, events = feed(m, t, FIST_THUMB_ON_INDEX, 600)
    assert Press() in events
    assert m.state == State.DRAGGING


def test_rule_d_ignores_a_fist():
    m = machine_d()
    t = activate(m)
    t, events = feed(m, t, FIST_THUMB_ON_INDEX, 600)
    assert events == []
    assert m.state == State.ACTIVE


def test_rule_d_still_clicks_with_the_release_grace():
    """Le relâchement n'est reconnu qu'après 66 ms : le clic arrive 2 images plus tard."""
    m = machine_d()
    t = activate(m)
    m.update(t, PINCH, P)
    assert m.state == State.PINCHING
    assert m.update(t + 150, OPEN, P) == []  # relâchement en cours, pas encore confirmé
    assert m.update(t + 183, OPEN, P) == []
    events = m.update(t + 216, OPEN, P)  # 66 ms après le début du relâchement
    assert Click() in events
    assert m.state == State.ACTIVE


def test_release_grace_keeps_a_drag_through_a_one_frame_glitch():
    """Une image où le suivi « ouvre » les doigts : sans délai, le glisser est lâché."""
    for grace, still_dragging in [(0, False), (66, True)]:
        m = machine_d(release_grace_ms=grace)
        t = activate(m)
        m.update(t, PINCH, P)
        m.update(t + 400, PINCH, P)
        assert m.state == State.DRAGGING
        events = m.update(t + 433, OPEN, P)  # décrochage d'une image
        events += m.update(t + 466, PINCH, P)
        assert (m.state == State.DRAGGING) == still_dragging
        assert (Release() in events) != still_dragging


def test_exit_on_fold_releases_a_drag_when_the_hand_closes():
    """Avec exit_on_fold : refermer la main en poing pendant un glisser le lâche, même si le
    pouce reste sur l'index."""
    m = machine_d(release_grace_ms=0, exit_on_fold=True)
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 400, PINCH, P)
    assert m.state == State.DRAGGING
    events = m.update(t + 433, FIST_THUMB_ON_INDEX, P)
    assert Release() in events
    assert m.state == State.ACTIVE


def test_measure_seg78_is_used():
    """Pouce posé sous le bout de l'index : pinch_ratio 0.6 (écarté), pinch_seg 0.1 (contact)."""
    thumb_below_tip = features(pinch_ratio=0.6, extended_fingers=3, extension=1.4,
                               pinch_seg=0.1)
    with_ratio = machine_d(measure="ratio")
    t = activate(with_ratio)
    with_ratio.update(t, thumb_below_tip, P)
    assert with_ratio.state == State.ACTIVE
    with_seg = machine_d(measure="seg78")
    t = activate(with_seg)
    with_seg.update(t, thumb_below_tip, P)
    assert with_seg.state == State.PINCHING


# --- Main au repos de profil ---------------------------------------------------------------------

# Main détendue, de côté : pouce contre l'index (pinch_ratio petit), autres doigts à moitié
# ouverts (extension 1.0, au-dessus de folded_below), mais paume étroite (vue de profil).
REST_PROFILE = features(pinch_ratio=0.10, extended_fingers=1, extension=1.0, palm_width=0.15)


def machine_d_palm(require_palm_facing):
    pinch = PinchConfig(rule="D", release_grace_ms=66, require_palm_facing=require_palm_facing,
                        min_palm_width=0.40)
    return GestureStateMachine(ENGAGEMENT, pinch, FINGERS)


def test_rest_hand_in_profile_fools_rule_d_alone():
    """Le défaut constaté en usage réel : la règle D seule accepte la main de profil."""
    m = machine_d_palm(require_palm_facing=False)
    t = activate(m)
    t, events = feed(m, t, REST_PROFILE, 600)
    assert Press() in events


def test_palm_condition_rejects_the_rest_hand_in_profile():
    m = machine_d_palm(require_palm_facing=True)
    t = activate(m)
    t, events = feed(m, t, REST_PROFILE, 600)
    assert events == []
    assert m.state == State.ACTIVE


def test_palm_condition_keeps_the_ok_gesture():
    """Pincement « OK », paume de face (0.8) : toujours reconnu."""
    m = machine_d_palm(require_palm_facing=True)
    t = activate(m)
    m.update(t, PINCH, P)
    assert m.state == State.PINCHING


# --- Pointage absolu ---------------------------------------------------------------------------


def machine_absolute():
    return GestureStateMachine(ENGAGEMENT, PinchConfig(), pointer=PointerConfig(mode="absolute"))


def test_absolute_mode_emits_screen_positions():
    """Zone par défaut 15 %..85 % en x, 20 %..80 % en y : (0.50, 0.50) = milieu de l'écran."""
    m = machine_absolute()
    t = activate(m)
    events = m.update(t, OPEN, (0.50, 0.50))
    assert events == [MoveTo(pytest.approx(0.5), pytest.approx(0.5))]
    assert m.update(t + 33, OPEN, (0.50, 0.50)) == []  # même position : rien à envoyer
    events = m.update(t + 66, OPEN, (0.95, 0.05))  # hors zone : collé au coin haut droit
    assert events == [MoveTo(1.0, 0.0)]


def test_absolute_cursor_is_frozen_during_a_short_pinch():
    m = machine_absolute()
    t = activate(m)
    m.update(t, OPEN, (0.50, 0.45))
    m.update(t + 33, PINCH, (0.50, 0.45))
    assert m.state == State.PINCHING
    events = m.update(t + 66, PINCH, (0.60, 0.45))  # la main bouge pendant le pincement
    assert events == []  # le curseur ne suit pas : le clic tombe là où on visait
    events = m.update(t + 99, OPEN, (0.60, 0.45))
    assert Click() in events
    events = m.update(t + 132, OPEN, (0.60, 0.45))  # de nouveau actif : le curseur rejoint la main
    assert len(events) == 1
    assert isinstance(events[0], MoveTo)


def test_absolute_cursor_follows_during_a_drag():
    m = machine_absolute()
    t = activate(m)
    m.update(t, PINCH, (0.50, 0.45))
    m.update(t + 400, PINCH, (0.50, 0.45))
    assert m.state == State.DRAGGING
    events = m.update(t + 433, PINCH, (0.60, 0.45))
    assert len(events) == 1
    assert isinstance(events[0], MoveTo)


def test_without_pointer_config_the_machine_stays_relative():
    """Comportement d'origine : sans PointerConfig, Move (relatif)."""
    m = new_machine()
    t = activate(m)
    events = m.update(t, OPEN, (0.6, 0.5))
    assert isinstance(events[0], Move)


# --- Main baissée = veille ----------------------------------------------------------------------
# Zone par défaut : x 0.15..0.85, y 0.20..0.80. IN = dans la zone, BELOW = sous la zone.

IN = (0.50, 0.50)
BELOW = (0.50, 0.90)


def machine_zone(click_max_ms=300):
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000, rest_ms=500,
                                  activate_in_zone=True)
    pinch = PinchConfig(only_in_zone=True, click_max_ms=click_max_ms)
    return GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="absolute"))


def activate_at(m, position):
    t = 0
    while t <= 330:
        m.update(t, OPEN, position)
        t += 33
    return t


def test_open_hand_outside_the_zone_does_not_activate():
    m = machine_zone()
    activate_at(m, BELOW)
    assert m.state == State.IDLE
    activate_at(m, IN)
    assert m.state == State.ACTIVE


def test_pinch_cannot_start_outside_the_zone():
    m = machine_zone()
    t = activate_at(m, IN)
    m.update(t, PINCH, (0.50, 0.85))  # main baissée (sous la zone), pouce contre index
    assert m.state == State.ACTIVE
    m.update(t + 33, PINCH, IN)  # dans la zone : le pincement commence
    assert m.state == State.PINCHING


def test_pinch_started_in_the_zone_continues_outside():
    """Glisser une fenêtre vers le bord : on peut sortir de la zone sans lâcher."""
    m = machine_zone()
    t = activate_at(m, IN)
    m.update(t, PINCH, IN)
    m.update(t + 400, PINCH, IN)
    assert m.state == State.DRAGGING
    events = m.update(t + 433, PINCH, (0.95, 0.50))  # hors zone, à droite
    assert m.state == State.DRAGGING
    assert Release() not in events


def test_hand_lowered_for_rest_ms_returns_to_idle_and_releases_the_drag():
    m = machine_zone()
    t = activate_at(m, IN)
    m.update(t, PINCH, IN)
    m.update(t + 400, PINCH, IN)
    assert m.state == State.DRAGGING
    t, events = feed_at(m, t + 433, PINCH, BELOW, 300)  # 300 ms sous la zone : on glisse encore
    assert m.state == State.DRAGGING
    assert Release() not in events
    t, events = feed_at(m, t, PINCH, BELOW, 300)  # au-delà de 500 ms : veille
    assert m.state == State.IDLE
    assert events.count(Release()) == 1


def test_short_pinch_is_cancelled_without_click_when_the_hand_is_lowered():
    # click_max_ms plus long que rest_ms : sinon le pincement deviendrait d'abord un glisser.
    m = machine_zone(click_max_ms=1000)
    t = activate_at(m, IN)
    m.update(t, PINCH, IN)
    assert m.state == State.PINCHING
    t, events = feed_at(m, t + 33, HALF, BELOW, 600)  # descend en gardant un demi-pincement
    assert m.state == State.IDLE
    assert Click() not in events
    assert Press() not in events


def test_lowered_open_hand_does_not_reactivate():
    """Sans activate_in_zone, une main ouverte posée en bas se réactiverait aussitôt."""
    m = machine_zone()
    t = activate_at(m, IN)
    t, _ = feed_at(m, t, OPEN, BELOW, 2000)
    assert m.state == State.IDLE


def feed_at(m, t, hand, position, duration_ms):
    events = []
    end = t + duration_ms
    while t < end:
        events += m.update(t, hand, position)
        t += 33
    return t, events


def test_project_configuration_builds_a_machine():
    """config/gestures.yaml est valide : la machine se construit avec la règle choisie."""
    cfg = load_config(local_path=None)
    m = GestureStateMachine(cfg.engagement, cfg.pinch, cfg.fingers)
    assert m.pinch.rule in RULES
    assert m.pinch.measure in MEASURES


def test_project_configuration_keeps_the_second_tap_grace():
    """Règle F en service : sans exemption, les doubles-clics rapides seraient refusés ou
    retardés (garde-fou : une ligne du yaml perdue lors d'une modification)."""
    pinch = load_config(local_path=None).pinch
    if pinch.rule in ("F", "F3D"):
        assert pinch.closing_grace_ms > 0
        assert pinch.closing_grace_max_speed > 0
        assert pinch.double_click_snap > 0


# --- Clic devenu glisser (log 2026-11-24_18-19-43, 79 296 ms) ----------------------------------
# pinch_ratio réel, image par image : pincement de 79 296 à 79 640 ms, puis relâchement lent
# qui OSCILLE autour d'exit (0,40). Autres doigts ouverts (extension ≈ 1,05).
REAL_RELEASE = [
    (79218, 0.4835), (79265, 0.4474), (79296, 0.2498), (79343, 0.0984), (79390, 0.0807),
    (79438, 0.0886), (79484, 0.1287), (79515, 0.2518), (79563, 0.3639), (79609, 0.3983),
    (79640, 0.4035), (79688, 0.3969), (79718, 0.4014), (79765, 0.3970), (79813, 0.4069),
    (79843, 0.4117), (79890, 0.3904), (79921, 0.4033), (79968, 0.4241), (80015, 0.4399),
]


def replay_real_release(release_cancel_margin):
    pinch = PinchConfig(rule="D", release_grace_ms=66, click_max_ms=400,
                        release_cancel_margin=release_cancel_margin)
    m = GestureStateMachine(ENGAGEMENT, pinch, FINGERS)
    activate(m)
    events = []
    for t, ratio in REAL_RELEASE:
        events += m.update(t, features(pinch_ratio=ratio, extended_fingers=3, extension=1.05), P)
    return events


def test_real_click_of_344_ms_is_a_click_not_a_drag():
    """Avec la marge : le relâchement commencé à 79 640 ms (344 ms) est confirmé malgré les
    petits passages sous 0,40, et c'est un clic."""
    events = replay_real_release(release_cancel_margin=0.05)
    assert Click() in events
    assert Press() not in events


def test_real_case_without_the_margin_reproduces_the_bug():
    """Sans marge (comportement d'origine), la même séquence donnait un glisser."""
    events = replay_real_release(release_cancel_margin=0.0)
    assert Press() in events
    assert Click() not in events


def test_no_press_while_a_release_is_being_confirmed():
    """Relâchement commencé à 380 ms, confirmé 66 ms plus tard (446 ms > click_max 400) :
    c'est un clic, la durée s'arrête au début du relâchement."""
    pinch = PinchConfig(release_grace_ms=66, click_max_ms=400)
    m = GestureStateMachine(ENGAGEMENT, pinch)
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 350, PINCH, P)
    assert m.update(t + 380, OPEN, P) == []  # relâchement en cours de confirmation
    assert m.update(t + 413, OPEN, P) == []  # 413 ms > 400 : mais pas de Press
    events = m.update(t + 446, OPEN, P)
    assert events == [Click()]


# --- Raison du refus d'entrée ------------------------------------------------------------------


def test_entry_block_explains_each_situation():
    m = machine_zone()
    m.update(0, OPEN, IN)
    assert m.entry_block == "veille"
    t = activate_at(m, IN)
    m.update(t, OPEN, IN)
    assert m.entry_block == "ratio"  # doigts écartés
    m.update(t + 33, PINCH, (0.50, 0.85))
    assert m.entry_block == "hors_zone"
    m.update(t + 66, PINCH, IN)
    assert m.state == State.PINCHING
    m.update(t + 99, PINCH, IN)
    assert m.entry_block == "pincement_en_cours"


def test_entry_block_names_the_rule_condition():
    m = machine_d()
    t = activate(m)
    m.update(t, FIST_THUMB_ON_INDEX, P)
    assert m.entry_block == "autres_doigts"


def test_closing_speed_is_computed_by_the_machine():
    m = new_machine()
    t = activate(m)
    m.update(t, features(pinch_ratio=0.8, extended_fingers=4, extension=1.5), P)
    m.update(t + 100, features(pinch_ratio=0.6, extended_fingers=4, extension=1.5), P)
    m.update(t + 200, features(pinch_ratio=0.4, extended_fingers=4, extension=1.5), P)
    assert m.closing_speed == pytest.approx(2.0)
    m.update(t + 233, None, None)  # main perdue : historique effacé
    assert m.closing_speed is None


# --- Mode de clic « press » (comme une souris) -------------------------------------------------
# hand_size = 0,2 dans features() : drag_threshold 0,15 = 0,03 en coordonnées image.


def machine_press(drag_catchup_ms=100, release_grace_ms=0, min_press_frames=2):
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000, rest_ms=500)
    pinch = PinchConfig(click_mode="press", min_press_frames=min_press_frames,
                        drag_threshold=0.15, drag_catchup_ms=drag_catchup_ms,
                        release_grace_ms=release_grace_ms)
    return GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="absolute"))


def activate_press(m):
    """Activation, puis une image main ouverte : le curseur est déjà sur la main (P)."""
    t = activate(m)
    m.update(t, OPEN, P)
    return t + 33


def moves(events):
    found = []
    for event in events:
        if isinstance(event, MoveTo):
            found.append(event)
    return found


def pin(position):
    """Point de l'écran (fractions) d'un point de l'image, zone par défaut."""
    return zone_to_screen(position[0], position[1], PointerConfig(mode="absolute").zone)


def press_at(position):
    u, v = pin(position)
    return PressAt(u, v)


def release_at(position):
    u, v = pin(position)
    return ReleaseAt(u, v)


def test_press_comes_one_frame_after_the_contact():
    """Latence : Press dès la 2e image de pincement (min_press_frames = 2), pas au relâchement."""
    m = machine_press()
    t = activate_press(m)
    assert m.update(t, PINCH, P) == []  # contact : 1re image, pas encore confirmé
    assert m.state == State.PINCHING
    assert m.update(t + 33, PINCH, P) == [press_at(P)]
    assert m.event_contact_ms == t  # pour la colonne contact_ms du log
    assert m.pin_kind == "press"
    assert m.update(t + 66, PINCH, P) == []
    assert m.event_contact_ms is None  # seulement sur l'image du Press
    assert m.pin_kind == ""


def test_click_in_place_is_pinned_to_the_contact_point():
    """La main dérive pendant le clic : Press et Release au point du contact. Le curseur suit
    la main jusqu'au Press, puis reste au point épinglé tant que le bouton est enfoncé."""
    m = machine_press()
    t = activate_press(m)
    events = m.update(t, PINCH, (0.50, 0.50))  # contact
    press_frame = m.update(t + 33, PINCH, (0.51, 0.50))  # la main a dérivé de 0,05
    assert press_frame == [MoveTo(*pin((0.51, 0.50))), press_at((0.50, 0.50))]
    events += m.update(t + 66, PINCH, (0.50, 0.51))
    events += m.update(t + 600, PINCH, (0.51, 0.51))  # appui long sans bouger : pas de glisser
    assert events == []  # bouton enfoncé : aucun MoveTo, pas de mini-glisser
    release = m.update(t + 633, OPEN, (0.51, 0.51))
    assert release == [release_at((0.50, 0.50))]
    assert m.pin_kind == "release"
    assert m.state == State.ACTIVE


def test_cursor_follows_the_hand_right_after_a_click():
    """Plus de verrou après le clic : le curseur repart vers la main à l'image suivante."""
    m = machine_press(drag_catchup_ms=0)
    t = activate_press(m)
    click_at(m, t, (0.50, 0.50))
    assert moves(m.update(t + 99, OPEN, (0.60, 0.50))) == [MoveTo(*pin((0.60, 0.50)))]


def test_click_max_ms_is_not_used_in_press_mode():
    m = machine_press()
    t = activate_press(m)
    m.update(t, PINCH, P)
    m.update(t + 2000, PINCH, P)
    assert m.state == State.PINCHING  # 2 s sans bouger : toujours un appui sur place


def test_drag_starts_beyond_the_threshold_and_catches_up():
    m = machine_press(drag_catchup_ms=100)
    t = activate_press(m)
    m.update(t, PINCH, (0.50, 0.50))
    assert m.update(t + 33, PINCH, (0.50, 0.50)) == [press_at((0.50, 0.50))]
    assert m.update(t + 66, PINCH, (0.52, 0.50)) == []  # 0,10 taille de main : figé
    assert m.state == State.PINCHING
    assert m.update(t + 99, PINCH, (0.60, 0.50)) == []  # 0,5 taille de main : glisser
    assert m.state == State.DRAGGING
    assert m.anchor_distance == pytest.approx(0.5)
    hand_u = (0.60 - 0.15) / 0.70
    # Rattrapage : à 33 ms sur 100, le curseur a fait 33 % du chemin.
    first = moves(m.update(t + 132, PINCH, (0.60, 0.50)))[0]
    assert first.u == pytest.approx(0.5 + 0.33 * (hand_u - 0.5))
    middle = moves(m.update(t + 165, PINCH, (0.60, 0.50)))[0]
    assert first.u < middle.u < hand_u
    last = moves(m.update(t + 199, PINCH, (0.60, 0.50)))[0]  # 100 ms : il a rejoint la main
    assert last.u == pytest.approx(hand_u)
    events = m.update(t + 232, OPEN, (0.60, 0.50))
    assert events == [Release()]  # fin du glisser : position réelle, pas épinglée


def test_drag_without_catch_up_jumps_to_the_hand():
    m = machine_press(drag_catchup_ms=0)
    t = activate_press(m)
    m.update(t, PINCH, (0.50, 0.50))
    m.update(t + 33, PINCH, (0.50, 0.50))
    m.update(t + 66, PINCH, (0.60, 0.50))
    first = moves(m.update(t + 99, PINCH, (0.60, 0.50)))[0]
    assert first.u == pytest.approx((0.60 - 0.15) / 0.70)


def test_double_click_without_snap_is_two_clicks_at_their_own_points():
    m = machine_press()
    t = activate_press(m)
    events = []
    for start in (t, t + 200):
        events += m.update(start, PINCH, P)
        events += m.update(start + 33, PINCH, P)
        events += m.update(start + 66, OPEN, P)
    events += m.update(t + 300, OPEN, P)
    assert events == [press_at(P), release_at(P), press_at(P), release_at(P)]


def test_one_frame_glitch_produces_nothing():
    m = machine_press()
    t = activate_press(m)
    events = m.update(t, PINCH, P)
    events += m.update(t + 33, OPEN, P)
    events += m.update(t + 66, OPEN, P)
    assert events == []
    assert m.state == State.ACTIVE


def test_no_press_while_the_release_is_being_confirmed():
    """Contact d'une image, puis relâchement en cours de confirmation : pas un appui."""
    m = machine_press(release_grace_ms=66)
    t = activate_press(m)
    events = m.update(t, PINCH, P)
    events += m.update(t + 33, OPEN, P)  # relâchement commencé
    events += m.update(t + 66, OPEN, P)
    events += m.update(t + 99, OPEN, P)  # relâchement confirmé
    assert events == []


def test_lost_hand_releases_a_pressed_button():
    m = machine_press()
    t = activate_press(m)
    m.update(t, PINCH, P)
    m.update(t + 33, PINCH, P)
    m.update(t + 66, None, None)
    assert m.update(t + 1200, None, None) == [Release()]
    assert m.state == State.IDLE


def test_lowered_hand_releases_a_pressed_button():
    m = machine_press()
    t = activate_press(m)
    m.update(t, PINCH, P)
    m.update(t + 33, PINCH, P)
    events = []
    for dt in range(66, 700, 33):
        events += m.update(t + dt, PINCH, (0.50, 0.90))  # main baissée sous la zone
    assert Release() in events
    assert m.state == State.IDLE


def test_relative_pointing_keeps_plain_press_and_release():
    """En pointage relatif, la machine ne connaît pas la position du curseur : pas d'épinglage."""
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000)
    pinch = PinchConfig(click_mode="press", double_click_snap=0.08)
    m = GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="relative"))
    t = activate(m)
    events = m.update(t, PINCH, P)
    events += m.update(t + 33, PINCH, P)
    events += m.update(t + 66, PINCH, (0.51, 0.50))  # bouton enfoncé : figé
    events += m.update(t + 99, OPEN, (0.51, 0.50))
    assert events == [Press(), Release()]


def test_release_mode_logs_the_contact_on_the_click():
    m = machine_zone()
    t = activate_at(m, IN)
    m.update(t, PINCH, IN)
    events = m.update(t + 33, OPEN, IN)
    assert events == [Click()]
    assert m.event_contact_ms == t


# --- Double-clic aimanté (synthétique) ----------------------------------------------------------
# hand_size = 0,2 : double_click_snap 0,08 = 0,016 en coordonnées image.


def machine_snap(snap=0.08, double_click_ms=500):
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000, rest_ms=500)
    pinch = PinchConfig(click_mode="press", drag_threshold=0.15, drag_catchup_ms=100,
                        double_click_ms=double_click_ms, double_click_snap=snap,
                        release_fast_margin=0.05)
    return GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="absolute"))


def click_at(m, t, position):
    events = m.update(t, PINCH, position)
    events += m.update(t + 33, PINCH, position)
    events += m.update(t + 66, OPEN, position)
    return events


def pinned(events):
    found = []
    for event in events:
        if isinstance(event, (PressAt, ReleaseAt)):
            found.append(event)
    return found


def test_fast_double_tap_lands_on_the_first_click_pixel():
    m = machine_snap()
    t = activate_press(m)
    first = click_at(m, t, (0.50, 0.50))
    m.update(t + 99, OPEN, (0.51, 0.50))  # la main dérive
    second = m.update(t + 132, PINCH, (0.51, 0.51))  # 0,07 taille de main du 1er contact
    second += m.update(t + 165, PINCH, (0.51, 0.51))
    assert m.pin_kind == "press_snap"
    second += m.update(t + 198, PINCH, (0.52, 0.51))  # bouton enfoncé : pas de MoveTo
    second += m.update(t + 231, OPEN, (0.52, 0.51))
    assert m.pin_kind == "release_snap"
    assert pinned(first) == [press_at((0.50, 0.50)), release_at((0.50, 0.50))]
    assert pinned(second) == [press_at((0.50, 0.50)), release_at((0.50, 0.50))]
    # Entre le PressAt et le ReleaseAt du 2e clic, le curseur ne bouge pas.
    between = second[second.index(press_at((0.50, 0.50))):]
    assert moves(between) == []


def test_second_tap_too_far_is_not_snapped():
    m = machine_snap()
    t = activate_press(m)
    click_at(m, t, (0.50, 0.50))
    second = click_at(m, t + 132, (0.53, 0.50))  # 0,15 taille de main > 0,08
    assert pinned(second) == [press_at((0.53, 0.50)), release_at((0.53, 0.50))]


def test_second_tap_too_late_is_not_snapped():
    m = machine_snap(double_click_ms=500)
    t = activate_press(m)
    click_at(m, t, (0.50, 0.50))
    second = click_at(m, t + 600, (0.51, 0.50))  # 600 ms après le 1er Press
    assert pinned(second) == [press_at((0.51, 0.50)), release_at((0.51, 0.50))]


def test_no_snap_after_a_drag():
    m = machine_snap()
    t = activate_press(m)
    m.update(t, PINCH, (0.50, 0.50))
    m.update(t + 33, PINCH, (0.50, 0.50))
    m.update(t + 66, PINCH, (0.60, 0.50))  # glisser
    m.update(t + 99, OPEN, (0.60, 0.50))
    second = click_at(m, t + 132, (0.50, 0.50))
    assert m.pin_kind == "release"
    assert pinned(second) == [press_at((0.50, 0.50)), release_at((0.50, 0.50))]


def test_snap_disabled_with_zero():
    m = machine_snap(snap=0)
    t = activate_press(m)
    click_at(m, t, (0.50, 0.50))
    second = click_at(m, t + 132, (0.51, 0.50))
    assert pinned(second) == [press_at((0.51, 0.50)), release_at((0.51, 0.50))]


# --- Séquences réelles ---------------------------------------------------------------------------
# (t ms, pinch_ratio, extensions index/majeur/annulaire/auriculaire, poignet (x, y), vitesse du
# poignet). Le poignet sert de point de pointage (approximation : le log ne garde pas le point 9
# filtré).
# Double-clics rapides du log 2026-11-24_19-12-46 :
REAL_TAPS = [
    (64921, 0.7662, (1.05, 0.98, 1.01, 0.86), (0.7026, 0.7865), 0.273),
    (64969, 0.7343, (1.03, 0.98, 1.00, 0.85), (0.7014, 0.7864), 0.185),
    (65015, 0.5577, (0.93, 0.98, 1.03, 0.86), (0.7023, 0.7896), 0.416),
    (65046, 0.1512, (0.77, 0.96, 1.02, 0.84), (0.7002, 0.7882), 0.566),
    (65094, 0.1464, (0.76, 0.95, 1.02, 0.85), (0.7001, 0.7889), 0.081),
    (65125, 0.2585, (0.80, 0.95, 1.01, 0.83), (0.7001, 0.7880), 0.153),
    (65171, 0.4328, (0.89, 0.97, 1.00, 0.83), (0.7010, 0.7869), 0.196),
    (65219, 0.1339, (0.76, 0.96, 1.03, 0.84), (0.7000, 0.7887), 0.257),
    (65250, 0.1081, (0.73, 0.89, 1.00, 0.83), (0.7010, 0.7923), 0.692),
    (65312, 0.1109, (0.74, 0.94, 0.98, 0.82), (0.7005, 0.7932), 0.093),
    (65344, 0.5524, (0.89, 0.99, 1.00, 0.81), (0.7001, 0.7925), 0.151),
    (65390, 0.5678, (0.92, 1.00, 1.01, 0.82), (0.7002, 0.7911), 0.171),
    (65437, 0.1513, (0.76, 0.95, 1.04, 0.83), (0.7012, 0.7954), 0.531),
    (65469, 0.0870, (0.72, 0.89, 1.02, 0.82), (0.7013, 0.8008), 0.912),
    (65515, 0.1267, (0.73, 0.94, 1.00, 0.81), (0.7008, 0.7991), 0.211),
    (65546, 0.3963, (0.86, 0.99, 1.01, 0.82), (0.6997, 0.7954), 0.710),
    (65594, 0.1562, (0.75, 0.85, 1.01, 0.82), (0.6991, 0.7959), 0.111),
    (65640, 0.1070, (0.72, 0.85, 0.98, 0.81), (0.7031, 0.8003), 0.812),
    (65671, 0.1962, (0.76, 0.90, 0.98, 0.82), (0.6996, 0.7963), 1.082),
    (65719, 0.3892, (0.86, 0.96, 0.99, 0.81), (0.7001, 0.7920), 0.498),
    (65750, 0.1340, (0.74, 0.80, 0.99, 0.80), (0.7011, 0.7961), 0.765),
    (65796, 0.1176, (0.74, 0.85, 0.97, 0.80), (0.7012, 0.7981), 0.230),
    (65844, 0.4602, (0.83, 0.94, 1.00, 0.82), (0.7004, 0.7955), 0.330),
    (65875, 0.5199, (0.86, 0.92, 1.00, 0.81), (0.6985, 0.7927), 0.669),
    (65921, 0.1699, (0.75, 0.84, 0.99, 0.79), (0.6993, 0.7967), 0.504),
    (65969, 0.1223, (0.74, 0.87, 0.96, 0.79), (0.7012, 0.7997), 0.448),
    (66015, 0.3312, (0.81, 0.91, 0.97, 0.79), (0.6991, 0.7950), 0.667),
    (66062, 0.8171, (0.99, 1.00, 0.98, 0.76), (0.6971, 0.7929), 0.396),
    (66094, 0.8273, (1.01, 1.03, 1.00, 0.76), (0.6986, 0.7910), 0.470),
    (66140, 0.2751, (0.73, 0.98, 1.04, 0.78), (0.7005, 0.7954), 0.613),
    (66171, 0.1223, (0.73, 0.91, 0.99, 0.77), (0.7022, 0.8012), 1.085),
    (66219, 0.1780, (0.73, 0.93, 0.96, 0.78), (0.6992, 0.7993), 0.500),
    (66265, 0.4601, (0.85, 0.99, 0.99, 0.78), (0.6983, 0.7933), 0.745),
    (66296, 0.4488, (0.84, 0.97, 1.00, 0.78), (0.6973, 0.7923), 0.284),
    (66344, 0.1339, (0.73, 0.86, 1.00, 0.79), (0.7000, 0.7968), 0.649),
    (66375, 0.1395, (0.73, 0.88, 0.97, 0.79), (0.7005, 0.7975), 0.170),
    (66421, 0.4625, (0.81, 0.96, 0.98, 0.79), (0.6996, 0.7916), 0.720),
    (66469, 0.6779, (0.96, 1.02, 1.00, 0.78), (0.6979, 0.7885), 0.453),
    (66500, 0.4434, (0.83, 1.00, 1.02, 0.78), (0.6989, 0.7904), 0.425),
    (66546, 0.1187, (0.74, 0.91, 1.00, 0.80), (0.7001, 0.7942), 0.494),
    (66577, 0.1557, (0.76, 0.90, 0.97, 0.78), (0.7000, 0.7963), 0.361),
    (66640, 0.4993, (0.87, 0.97, 0.99, 0.78), (0.6987, 0.7935), 0.286),
    (66687, 0.7920, (1.01, 1.01, 0.98, 0.78), (0.6976, 0.7890), 0.561),
    (66719, 0.7853, (1.02, 1.01, 0.99, 0.77), (0.6980, 0.7882), 0.176),
    (66765, 0.2891, (0.79, 0.93, 1.00, 0.79), (0.6994, 0.7906), 0.371),
    (66812, 0.1611, (0.76, 0.92, 0.98, 0.79), (0.7012, 0.7948), 0.558),
    (66844, 0.2288, (0.78, 0.94, 0.97, 0.77), (0.6995, 0.7933), 0.457),
    (66890, 0.6323, (0.94, 1.00, 1.00, 0.78), (0.6998, 0.7904), 0.353),
    (66921, 0.7089, (1.01, 1.01, 0.99, 0.78), (0.7004, 0.7870), 0.631),
    (66969, 0.5462, (0.84, 1.01, 1.01, 0.79), (0.7011, 0.7868), 0.106),
    (67015, 0.1446, (0.75, 0.94, 0.98, 0.79), (0.7012, 0.7892), 0.297),
    (67046, 0.1538, (0.76, 0.93, 0.98, 0.79), (0.7013, 0.7924), 0.557),
    (67094, 0.2967, (0.81, 0.94, 0.97, 0.77), (0.7005, 0.7901), 0.281),
    (67125, 0.5738, (0.93, 0.97, 0.98, 0.77), (0.7015, 0.7905), 0.242),
    (67171, 0.6942, (0.99, 0.99, 0.98, 0.78), (0.7011, 0.7880), 0.313),
    (67219, 0.4105, (0.84, 0.94, 0.99, 0.78), (0.7017, 0.7868), 0.164),
    (67265, 0.1174, (0.74, 0.89, 1.00, 0.79), (0.7023, 0.7924), 0.678),
    (67312, 0.1456, (0.75, 0.91, 0.97, 0.78), (0.7017, 0.7948), 0.297),
    (67344, 0.2712, (0.80, 0.93, 0.96, 0.77), (0.7009, 0.7930), 0.357),
    (67390, 0.5718, (0.94, 0.97, 0.96, 0.77), (0.7014, 0.7920), 0.147),
    (67437, 0.5188, (0.89, 0.95, 0.97, 0.77), (0.7021, 0.7893), 0.333),
    (67469, 0.1359, (0.75, 0.88, 0.97, 0.78), (0.7027, 0.7912), 0.354),
    (67515, 0.1021, (0.74, 0.87, 0.97, 0.78), (0.7028, 0.7950), 0.452),
    (67546, 0.2051, (0.77, 0.90, 0.97, 0.78), (0.7014, 0.7929), 0.500),
    (67594, 0.4966, (0.91, 0.96, 0.97, 0.76), (0.7015, 0.7914), 0.173),
    (67640, 0.4635, (0.87, 0.94, 0.97, 0.77), (0.7019, 0.7901), 0.173),
    (67671, 0.1261, (0.74, 0.85, 0.97, 0.78), (0.7031, 0.7927), 0.541),
    (67719, 0.0951, (0.73, 0.87, 0.98, 0.79), (0.7042, 0.7958), 0.395),
    (67750, 0.1671, (0.74, 0.87, 0.95, 0.77), (0.7023, 0.7942), 0.528),
    (67796, 0.4870, (0.87, 0.93, 0.96, 0.77), (0.7019, 0.7910), 0.388),
]
REAL_FAST_MOVES = [
    (71937, 0.5553, (0.97, 1.02, 0.96, 0.78), (0.5310, 0.5170), 0.578),
    (71969, 0.5597, (0.96, 1.01, 0.96, 0.79), (0.5261, 0.5136), 1.025),
    (72015, 0.5599, (0.94, 0.99, 0.95, 0.77), (0.5192, 0.5157), 0.907),
    (72062, 0.5769, (0.95, 0.99, 0.93, 0.77), (0.5129, 0.5278), 1.375),
    (72094, 0.3084, (0.81, 0.98, 0.94, 0.78), (0.5087, 0.5514), 3.300),
    (72140, 0.2202, (0.81, 0.88, 0.90, 0.75), (0.5056, 0.5839), 3.062),
    (72171, 0.3876, (0.84, 0.97, 0.97, 0.83), (0.4970, 0.6129), 4.498),
    (72219, 0.5962, (0.93, 1.01, 0.96, 0.84), (0.4950, 0.6308), 1.703),
    (72265, 0.6036, (0.96, 1.02, 0.98, 0.79), (0.4957, 0.6392), 0.811),
    (72296, 0.5864, (0.94, 1.01, 0.97, 0.81), (0.4955, 0.6348), 0.634),
    (72344, 0.5071, (0.92, 1.03, 1.00, 0.81), (0.4949, 0.6234), 1.047),
    (72375, 0.4342, (0.88, 0.99, 0.96, 0.81), (0.4972, 0.6068), 2.380),
    (72421, 0.2366, (0.78, 1.01, 0.99, 0.83), (0.4977, 0.5876), 1.868),
    (72469, 0.2366, (0.81, 1.02, 0.98, 0.79), (0.4971, 0.5706), 1.598),
    (72500, 0.5320, (0.95, 1.04, 0.96, 0.77), (0.4938, 0.5523), 2.769),
    (72546, 0.5393, (0.96, 1.03, 0.98, 0.81), (0.4932, 0.5399), 1.236),
    (72594, 0.5199, (0.94, 1.02, 0.96, 0.81), (0.4944, 0.5297), 0.979),
    (72640, 0.5036, (0.93, 1.00, 0.96, 0.81), (0.4940, 0.5247), 0.493),
    (72687, 0.4978, (0.93, 1.00, 0.97, 0.82), (0.4971, 0.5233), 0.411),
    (72719, 0.5005, (0.94, 1.01, 0.98, 0.82), (0.4971, 0.5221), 0.162),
    (72765, 0.5062, (0.94, 1.01, 0.98, 0.82), (0.4967, 0.5213), 0.105),
]


# Tapes rapides fusionnées en un seul appui, log 2026-11-25_16-35-12 (deux creux séparés par
# une remontée qui ne dépasse pas exit = 0,40) :
MERGED_TAPS_13250 = [
    (13171, 0.6047, (0.96, 0.98, 0.87, 0.69), (0.6792, 0.8434), 0.181),
    (13218, 0.4274, (0.85, 0.98, 0.90, 0.70), (0.6811, 0.8424), 0.254),
    (13250, 0.1765, (0.81, 0.99, 0.92, 0.72), (0.6810, 0.8419), 0.059),
    (13281, 0.1187, (0.83, 0.98, 0.91, 0.74), (0.6800, 0.8400), 0.338),
    (13311, 0.1140, (0.83, 0.97, 0.90, 0.73), (0.6816, 0.8390), 0.342),
    (13343, 0.1096, (0.83, 0.97, 0.91, 0.74), (0.6795, 0.8382), 0.389),
    (13375, 0.1477, (0.82, 0.97, 0.88, 0.71), (0.6799, 0.8385), 0.088),
    (13406, 0.2446, (0.84, 0.95, 0.86, 0.70), (0.6784, 0.8385), 0.283),
    (13436, 0.3922, (0.90, 0.92, 0.85, 0.70), (0.6794, 0.8394), 0.231),
    (13484, 0.3621, (0.87, 0.92, 0.85, 0.70), (0.6823, 0.8408), 0.363),
    (13514, 0.1613, (0.82, 0.95, 0.89, 0.72), (0.6830, 0.8408), 0.128),
    (13546, 0.1275, (0.81, 0.95, 0.88, 0.71), (0.6813, 0.8399), 0.321),
    (13577, 0.1291, (0.81, 0.94, 0.87, 0.71), (0.6817, 0.8391), 0.132),
    (13625, 0.1286, (0.83, 0.96, 0.89, 0.73), (0.6793, 0.8373), 0.328),
    (13656, 0.1719, (0.82, 0.95, 0.88, 0.72), (0.6776, 0.8378), 0.321),
    (13686, 0.2328, (0.83, 0.94, 0.87, 0.71), (0.6788, 0.8399), 0.380),
    (13718, 0.2928, (0.85, 0.92, 0.86, 0.70), (0.6796, 0.8394), 0.152),
    (13750, 0.3273, (0.87, 0.91, 0.85, 0.70), (0.6783, 0.8388), 0.234),
    (13781, 0.3391, (0.86, 0.90, 0.84, 0.70), (0.6805, 0.8389), 0.401),
    (13811, 0.3461, (0.88, 0.91, 0.86, 0.71), (0.6786, 0.8384), 0.370),
    (13843, 0.3623, (0.87, 0.90, 0.85, 0.71), (0.6809, 0.8391), 0.427),
    (13875, 0.3822, (0.88, 0.90, 0.84, 0.71), (0.6797, 0.8369), 0.367),
    (13906, 0.3962, (0.89, 0.89, 0.84, 0.71), (0.6803, 0.8390), 0.306),
    (13936, 0.3991, (0.89, 0.89, 0.83, 0.70), (0.6805, 0.8375), 0.217),
    (13968, 0.3987, (0.88, 0.88, 0.83, 0.69), (0.6811, 0.8375), 0.113),
    (14014, 0.3996, (0.88, 0.89, 0.84, 0.70), (0.6810, 0.8376), 0.019),
    (14046, 0.4009, (0.89, 0.89, 0.85, 0.72), (0.6811, 0.8373), 0.045),
    (14077, 0.3989, (0.89, 0.89, 0.84, 0.70), (0.6815, 0.8393), 0.281),
    (14125, 0.4088, (0.90, 0.89, 0.83, 0.71), (0.6818, 0.8411), 0.158),
    (14156, 0.4234, (0.90, 0.89, 0.83, 0.71), (0.6810, 0.8411), 0.155),
    (14186, 0.4288, (0.90, 0.89, 0.83, 0.70), (0.6816, 0.8415), 0.124),
]
MERGED_TAPS_58375 = [
    (58281, 0.5125, (0.97, 0.92, 0.85, 0.71), (0.7009, 0.8120), 0.342),
    (58311, 0.4984, (0.96, 0.91, 0.84, 0.70), (0.7014, 0.8138), 0.293),
    (58343, 0.4337, (0.94, 0.91, 0.86, 0.71), (0.7048, 0.8149), 0.677),
    (58375, 0.1650, (0.86, 0.91, 0.89, 0.74), (0.7051, 0.8147), 0.056),
    (58421, 0.1624, (0.85, 0.89, 0.85, 0.72), (0.7074, 0.8139), 0.314),
    (58452, 0.1703, (0.84, 0.90, 0.86, 0.72), (0.7066, 0.8139), 0.148),
    (58484, 0.1873, (0.87, 0.91, 0.87, 0.73), (0.7055, 0.8139), 0.204),
    (58514, 0.3127, (0.88, 0.89, 0.86, 0.73), (0.7067, 0.8127), 0.289),
    (58546, 0.3380, (0.90, 0.89, 0.88, 0.72), (0.7059, 0.8145), 0.289),
    (58577, 0.1334, (0.83, 0.89, 0.89, 0.76), (0.7054, 0.8136), 0.147),
    (58609, 0.1117, (0.84, 0.88, 0.88, 0.73), (0.7076, 0.8127), 0.426),
    (58639, 0.1051, (0.80, 0.88, 0.88, 0.74), (0.7060, 0.8108), 0.431),
    (58671, 0.1189, (0.85, 0.91, 0.89, 0.74), (0.7059, 0.8119), 0.158),
    (58702, 0.3003, (0.87, 0.88, 0.85, 0.73), (0.7066, 0.8088), 0.470),
    (58734, 0.3834, (0.89, 0.89, 0.85, 0.73), (0.7068, 0.8083), 0.081),
    (58781, 0.4281, (0.91, 0.91, 0.86, 0.74), (0.7057, 0.8075), 0.160),
    (58827, 0.4156, (0.90, 0.90, 0.85, 0.74), (0.7072, 0.8068), 0.213),
    (58859, 0.4192, (0.91, 0.89, 0.85, 0.74), (0.7069, 0.8059), 0.134),
    (58889, 0.4370, (0.92, 0.91, 0.86, 0.75), (0.7050, 0.8046), 0.436),
]


def replay(sequence, rule="D", fingers=None, hand_size=0.2, **pinch_args):
    """Rejoue une séquence réelle en mode press, règle D par défaut (réglages du 2026-11-24).
    `hand_size` : taille de la main (le log ne la garde pas ; 0,2 par défaut)."""
    pinch = PinchConfig(rule=rule, click_mode="press", release_grace_ms=66,
                        release_cancel_margin=0.05, **pinch_args)
    m = GestureStateMachine(ENGAGEMENT, pinch, fingers, pointer=PointerConfig(mode="absolute"))
    t0, _, _, position, _ = sequence[0]
    t = t0 - 400
    while t < t0:
        m.update(t, OPEN, position)  # activation au point de départ
        t += 33
    events = []
    for t, ratio, extensions, position, speed in sequence:
        f = replace(features(pinch_ratio=ratio, extended_fingers=3, extension=1.4),
                    finger_extensions=extensions, finger_extensions_3d=extensions,
                    index_extension=extensions[0], hand_size=hand_size)
        for event in m.update(t, f, position, speed):
            events.append((t, event))
    return events


def count(events, kind):
    """Nombre d'événements de la classe `kind` (PressAt, ReleaseAt...)."""
    n = 0
    for _, event in events:
        if isinstance(event, kind):
            n += 1
    return n


def test_real_fast_taps_merged_without_the_fast_release():
    """Bug constaté : 14 tapes rapides, mais le bouton reste enfoncé entre elles."""
    events = replay(REAL_TAPS)
    assert count(events, PressAt) == 5
    assert count(events, ReleaseAt) == 4


def test_real_fast_taps_are_separated_with_the_fast_release():
    events = replay(REAL_TAPS, release_fast_margin=0.05)
    assert count(events, PressAt) == 11
    assert count(events, ReleaseAt) == 11
    # Jamais deux Press sans Release entre les deux.
    pressed = False
    for _, event in events:
        if isinstance(event, PressAt):
            assert not pressed
            pressed = True
        elif isinstance(event, ReleaseAt):
            pressed = False


def test_real_taps_land_on_the_first_click_pixel_with_the_snap():
    """Avec l'aimant, les 11 clics de la série tombent au pixel du 1er : Windows voit des
    doubles-clics, alors que la main a bougé entre les tapes."""
    events = replay(REAL_TAPS, release_fast_margin=0.05, double_click_snap=0.08)
    points = []
    for _, event in events:
        if isinstance(event, (PressAt, ReleaseAt)):
            points.append((event.u, event.v))
    assert len(points) == 22
    for point in points:
        assert point == points[0]
    # Sans aimant, les points diffèrent (la main bouge de quelques pixels entre les tapes).
    plain = replay(REAL_TAPS, release_fast_margin=0.05)
    plain_points = set()
    for _, event in plain:
        if isinstance(event, PressAt):
            plain_points.add((event.u, event.v))
    assert len(plain_points) > 1


def test_real_fast_moves_are_refused_by_the_contact_speed():
    """72 140 et 72 421 ms : pincements pendant un geste rapide (3,06 et 1,87 au contact)."""
    assert count(replay(REAL_FAST_MOVES, release_fast_margin=0.05), PressAt) == 2
    assert count(replay(REAL_FAST_MOVES, release_fast_margin=0.05, max_contact_speed=1.8),
                 PressAt) == 0


def test_real_slow_release_is_still_a_single_click_with_the_fast_release():
    """Le cas du 79 296 ms (oscillation autour d'exit) reste un clic."""
    pinch = PinchConfig(rule="D", release_grace_ms=66, click_max_ms=400,
                        release_cancel_margin=0.05, release_fast_margin=0.05)
    m = GestureStateMachine(ENGAGEMENT, pinch, FINGERS)
    activate(m)
    events = []
    for t, ratio in REAL_RELEASE:
        events += m.update(t, features(pinch_ratio=ratio, extended_fingers=3, extension=1.05), P)
    assert events.count(Click()) == 1
    assert Press() not in events


# --- Relâchement relatif (release_rise) ---------------------------------------------------------


@pytest.mark.parametrize("sequence", [MERGED_TAPS_13250, MERGED_TAPS_58375])
def test_merged_real_taps_are_one_press_without_release_rise(sequence):
    events = replay(sequence, release_fast_margin=0.05)
    assert count(events, PressAt) == 1


@pytest.mark.parametrize("sequence", [MERGED_TAPS_13250, MERGED_TAPS_58375])
def test_release_rise_separates_the_merged_real_taps(sequence):
    events = replay(sequence, release_fast_margin=0.05, release_rise=0.15)
    assert count(events, PressAt) == 2
    assert count(events, ReleaseAt) == 2


def test_release_rise_is_not_used_during_a_drag():
    """Pendant un glisser, une remontée de la mesure sous exit ne relâche pas le bouton."""
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000)
    pinch = PinchConfig(click_mode="press", drag_threshold=0.15, release_rise=0.15)
    m = GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="absolute"))
    t = activate(m)
    m.update(t, PINCH, (0.50, 0.50))
    m.update(t + 33, PINCH, (0.50, 0.50))  # Press
    m.update(t + 66, PINCH, (0.60, 0.50))  # glisser
    assert m.state == State.DRAGGING
    events = m.update(t + 99, HALF, (0.62, 0.50))  # 0,10 -> 0,33 : remontée de 0,23
    assert not any(isinstance(event, (Release, ReleaseAt)) for event in events)
    assert m.state == State.DRAGGING


def test_release_rise_releases_a_click_in_place():
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000)
    pinch = PinchConfig(click_mode="press", release_rise=0.15)
    m = GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="absolute"))
    t = activate(m)
    m.update(t, PINCH, P)
    m.update(t + 33, PINCH, P)  # Press
    assert m.update(t + 66, HALF, P) == [release_at(P)]  # 0,10 -> 0,33, sous exit 0,40


# --- Règle F : exemption de fermeture juste après un relâchement ------------------------------
# Les 4 doubles-clics aimantés du log 2026-11-25_17-10-10 (enregistré en règle D). À la 2e
# tape, la vitesse de fermeture sur 200 ms remonte jusqu'au creux de la 1re : -0,26 / -0,33 /
# -0,15 / 0,21, sous le seuil de 0,4.
REAL_DOUBLE_CLICKS = [
    [  # 2e tape à 19578 ms
        (18906, 0.4103, (0.90, 0.97, 0.91, 0.77), (0.6554, 0.7596), 0.383),
        (18938, 0.3977, (0.89, 0.97, 0.91, 0.77), (0.6550, 0.7597), 0.100),
        (18969, 0.4081, (0.89, 0.96, 0.91, 0.76), (0.6554, 0.7594), 0.122),
        (19000, 0.4037, (0.90, 0.97, 0.91, 0.76), (0.6553, 0.7583), 0.255),
        (19031, 0.4046, (0.90, 0.97, 0.92, 0.77), (0.6552, 0.7582), 0.029),
        (19063, 0.4085, (0.90, 0.97, 0.92, 0.77), (0.6549, 0.7580), 0.103),
        (19094, 0.4125, (0.89, 0.96, 0.91, 0.77), (0.6558, 0.7572), 0.324),
        (19141, 0.4096, (0.90, 0.97, 0.92, 0.77), (0.6549, 0.7573), 0.181),
        (19172, 0.4148, (0.90, 0.96, 0.91, 0.77), (0.6550, 0.7568), 0.116),
        (19203, 0.4022, (0.89, 0.96, 0.91, 0.77), (0.6552, 0.7569), 0.066),
        (19235, 0.4060, (0.91, 0.97, 0.92, 0.78), (0.6548, 0.7570), 0.116),
        (19266, 0.3926, (0.89, 0.96, 0.92, 0.77), (0.6552, 0.7568), 0.121),
        (19297, 0.2624, (0.83, 0.95, 0.91, 0.77), (0.6551, 0.7564), 0.094),
        (19328, 0.1401, (0.81, 0.95, 0.89, 0.80), (0.6545, 0.7569), 0.207),
        (19375, 0.1289, (0.80, 0.93, 0.90, 0.80), (0.6539, 0.7546), 0.350),
        (19406, 0.1147, (0.80, 0.92, 0.90, 0.80), (0.6533, 0.7552), 0.205),
        (19438, 0.1290, (0.81, 0.92, 0.90, 0.80), (0.6532, 0.7551), 0.044),
        (19469, 0.2127, (0.81, 0.92, 0.89, 0.79), (0.6535, 0.7554), 0.097),
        (19500, 0.2764, (0.85, 0.94, 0.91, 0.78), (0.6544, 0.7550), 0.289),
        (19547, 0.3229, (0.87, 0.94, 0.92, 0.79), (0.6557, 0.7563), 0.298),
        (19578, 0.1810, (0.79, 0.93, 0.90, 0.81), (0.6551, 0.7547), 0.387),
        (19610, 0.1358, (0.79, 0.89, 0.88, 0.80), (0.6553, 0.7554), 0.145),
        (19641, 0.1202, (0.80, 0.90, 0.88, 0.79), (0.6555, 0.7546), 0.167),
        (19672, 0.1370, (0.80, 0.91, 0.87, 0.80), (0.6541, 0.7551), 0.401),
        (19703, 0.2539, (0.81, 0.91, 0.87, 0.77), (0.6554, 0.7542), 0.404),
        (19735, 0.4186, (0.88, 0.94, 0.91, 0.78), (0.6565, 0.7555), 0.406),
        (19766, 0.5244, (0.90, 0.96, 0.90, 0.77), (0.6568, 0.7548), 0.174),
        (19797, 0.5698, (0.91, 0.96, 0.91, 0.78), (0.6578, 0.7549), 0.293),
        (19828, 0.5806, (0.91, 0.94, 0.88, 0.75), (0.6577, 0.7543), 0.128),
        (19860, 0.5793, (0.92, 0.96, 0.90, 0.77), (0.6579, 0.7550), 0.152),
        (19891, 0.5723, (0.90, 0.94, 0.89, 0.76), (0.6585, 0.7546), 0.199),
    ],
    [  # 2e tape à 25000 ms
        (24344, 0.4838, (0.89, 0.91, 0.88, 0.78), (0.6596, 0.7490), 0.172),
        (24375, 0.4941, (0.89, 0.92, 0.89, 0.79), (0.6601, 0.7492), 0.144),
        (24406, 0.4862, (0.89, 0.92, 0.89, 0.80), (0.6598, 0.7494), 0.100),
        (24438, 0.4887, (0.89, 0.92, 0.89, 0.78), (0.6599, 0.7488), 0.115),
        (24469, 0.4869, (0.89, 0.92, 0.89, 0.79), (0.6599, 0.7494), 0.119),
        (24500, 0.4868, (0.90, 0.92, 0.90, 0.79), (0.6593, 0.7495), 0.162),
        (24531, 0.4964, (0.90, 0.93, 0.90, 0.80), (0.6589, 0.7500), 0.176),
        (24563, 0.5018, (0.91, 0.93, 0.90, 0.79), (0.6593, 0.7501), 0.113),
        (24594, 0.4980, (0.90, 0.93, 0.90, 0.80), (0.6585, 0.7509), 0.287),
        (24625, 0.4946, (0.91, 0.94, 0.91, 0.80), (0.6586, 0.7517), 0.164),
        (24656, 0.4844, (0.92, 0.94, 0.91, 0.80), (0.6588, 0.7528), 0.236),
        (24688, 0.4302, (0.90, 0.94, 0.91, 0.79), (0.6576, 0.7526), 0.339),
        (24750, 0.3415, (0.88, 0.95, 0.91, 0.79), (0.6576, 0.7536), 0.104),
        (24781, 0.1706, (0.81, 0.92, 0.89, 0.79), (0.6565, 0.7535), 0.309),
        (24813, 0.1027, (0.80, 0.92, 0.90, 0.82), (0.6571, 0.7539), 0.174),
        (24844, 0.0944, (0.79, 0.91, 0.88, 0.81), (0.6564, 0.7522), 0.410),
        (24875, 0.1026, (0.80, 0.91, 0.89, 0.81), (0.6548, 0.7534), 0.551),
        (24906, 0.1286, (0.80, 0.91, 0.88, 0.80), (0.6554, 0.7532), 0.176),
        (24938, 0.2409, (0.83, 0.92, 0.89, 0.79), (0.6553, 0.7522), 0.214),
        (24969, 0.3243, (0.88, 0.95, 0.93, 0.81), (0.6554, 0.7530), 0.194),
        (25000, 0.2436, (0.82, 0.93, 0.90, 0.78), (0.6554, 0.7534), 0.083),
        (25031, 0.1561, (0.81, 0.94, 0.91, 0.82), (0.6568, 0.7548), 0.512),
        (25063, 0.1167, (0.81, 0.92, 0.92, 0.82), (0.6557, 0.7546), 0.332),
        (25094, 0.1078, (0.81, 0.92, 0.91, 0.81), (0.6550, 0.7542), 0.208),
        (25141, 0.1485, (0.81, 0.93, 0.90, 0.80), (0.6550, 0.7536), 0.085),
        (25172, 0.2665, (0.84, 0.93, 0.90, 0.78), (0.6558, 0.7535), 0.247),
        (25203, 0.3912, (0.90, 0.96, 0.93, 0.80), (0.6563, 0.7540), 0.171),
        (25235, 0.4538, (0.90, 0.96, 0.92, 0.80), (0.6570, 0.7537), 0.218),
        (25266, 0.4786, (0.91, 0.96, 0.91, 0.79), (0.6578, 0.7539), 0.229),
        (25297, 0.4842, (0.92, 0.95, 0.92, 0.79), (0.6581, 0.7537), 0.076),
        (25328, 0.4910, (0.92, 0.97, 0.92, 0.79), (0.6583, 0.7532), 0.126),
    ],
    [  # 2e tape à 29735 ms
        (29063, 0.4429, (0.91, 0.93, 0.90, 0.79), (0.6549, 0.7600), 0.062),
        (29094, 0.4352, (0.90, 0.93, 0.89, 0.78), (0.6546, 0.7590), 0.236),
        (29141, 0.4270, (0.91, 0.94, 0.90, 0.79), (0.6546, 0.7604), 0.192),
        (29172, 0.4072, (0.91, 0.93, 0.89, 0.78), (0.6547, 0.7599), 0.111),
        (29203, 0.3895, (0.91, 0.94, 0.90, 0.78), (0.6554, 0.7602), 0.204),
        (29235, 0.3636, (0.91, 0.94, 0.90, 0.78), (0.6546, 0.7605), 0.226),
        (29266, 0.3306, (0.90, 0.94, 0.90, 0.78), (0.6562, 0.7605), 0.438),
        (29297, 0.3426, (0.91, 0.95, 0.92, 0.80), (0.6556, 0.7620), 0.360),
        (29328, 0.3493, (0.90, 0.94, 0.90, 0.78), (0.6561, 0.7609), 0.280),
        (29375, 0.3367, (0.91, 0.95, 0.91, 0.79), (0.6556, 0.7614), 0.114),
        (29406, 0.2494, (0.87, 0.93, 0.90, 0.78), (0.6554, 0.7615), 0.065),
        (29438, 0.1207, (0.80, 0.92, 0.90, 0.80), (0.6551, 0.7622), 0.169),
        (29469, 0.0922, (0.80, 0.91, 0.90, 0.80), (0.6544, 0.7603), 0.454),
        (29500, 0.0921, (0.81, 0.92, 0.90, 0.80), (0.6543, 0.7602), 0.039),
        (29547, 0.0887, (0.81, 0.92, 0.91, 0.80), (0.6535, 0.7601), 0.147),
        (29578, 0.1171, (0.80, 0.91, 0.90, 0.79), (0.6545, 0.7601), 0.297),
        (29610, 0.2160, (0.84, 0.92, 0.89, 0.78), (0.6547, 0.7608), 0.143),
        (29641, 0.3049, (0.85, 0.90, 0.88, 0.77), (0.6566, 0.7600), 0.541),
        (29672, 0.3672, (0.89, 0.92, 0.91, 0.80), (0.6560, 0.7605), 0.181),
        (29703, 0.3031, (0.87, 0.93, 0.90, 0.79), (0.6550, 0.7602), 0.308),
        (29735, 0.1286, (0.79, 0.92, 0.90, 0.80), (0.6551, 0.7607), 0.117),
        (29766, 0.0940, (0.80, 0.92, 0.90, 0.80), (0.6539, 0.7596), 0.429),
        (29797, 0.0943, (0.81, 0.92, 0.92, 0.81), (0.6545, 0.7592), 0.180),
        (29828, 0.0908, (0.81, 0.92, 0.90, 0.79), (0.6534, 0.7590), 0.316),
        (29860, 0.1302, (0.80, 0.92, 0.89, 0.79), (0.6546, 0.7587), 0.347),
        (29891, 0.2201, (0.84, 0.92, 0.90, 0.79), (0.6542, 0.7596), 0.233),
        (29938, 0.2806, (0.87, 0.92, 0.91, 0.79), (0.6558, 0.7588), 0.324),
        (29969, 0.3197, (0.87, 0.92, 0.91, 0.80), (0.6558, 0.7582), 0.121),
        (30000, 0.3334, (0.89, 0.93, 0.92, 0.80), (0.6562, 0.7587), 0.138),
        (30047, 0.3421, (0.88, 0.93, 0.91, 0.79), (0.6561, 0.7573), 0.194),
        (30078, 0.3414, (0.88, 0.92, 0.91, 0.80), (0.6562, 0.7570), 0.079),
    ],
    [  # 2e tape à 38266 ms
        (37578, 0.5538, (0.93, 0.96, 0.93, 0.81), (0.6947, 0.7164), 0.049),
        (37610, 0.5528, (0.92, 0.96, 0.92, 0.81), (0.6945, 0.7162), 0.065),
        (37641, 0.5440, (0.91, 0.95, 0.92, 0.80), (0.6948, 0.7162), 0.102),
        (37672, 0.5384, (0.91, 0.95, 0.92, 0.81), (0.6946, 0.7155), 0.166),
        (37703, 0.5296, (0.91, 0.95, 0.92, 0.80), (0.6951, 0.7151), 0.156),
        (37735, 0.5259, (0.91, 0.95, 0.93, 0.81), (0.6959, 0.7145), 0.262),
        (37766, 0.5186, (0.91, 0.96, 0.92, 0.81), (0.6958, 0.7147), 0.040),
        (37797, 0.5057, (0.91, 0.96, 0.93, 0.82), (0.6953, 0.7148), 0.156),
        (37828, 0.5102, (0.91, 0.95, 0.93, 0.81), (0.6956, 0.7144), 0.138),
        (37860, 0.5068, (0.90, 0.95, 0.92, 0.81), (0.6959, 0.7151), 0.178),
        (37891, 0.5169, (0.90, 0.95, 0.93, 0.81), (0.6961, 0.7157), 0.125),
        (37938, 0.5202, (0.90, 0.96, 0.92, 0.81), (0.6968, 0.7156), 0.128),
        (37969, 0.5104, (0.90, 0.96, 0.93, 0.81), (0.6971, 0.7161), 0.158),
        (38000, 0.4454, (0.87, 0.96, 0.93, 0.81), (0.6970, 0.7156), 0.117),
        (38031, 0.2990, (0.84, 0.96, 0.95, 0.83), (0.6979, 0.7148), 0.294),
        (38078, 0.0993, (0.79, 0.94, 0.95, 0.86), (0.6973, 0.7154), 0.136),
        (38110, 0.0935, (0.77, 0.92, 0.94, 0.85), (0.6964, 0.7142), 0.347),
        (38141, 0.1068, (0.78, 0.91, 0.93, 0.84), (0.6972, 0.7143), 0.238),
        (38172, 0.1777, (0.80, 0.93, 0.94, 0.84), (0.6961, 0.7121), 0.588),
        (38203, 0.2570, (0.82, 0.91, 0.92, 0.81), (0.6964, 0.7143), 0.477),
        (38235, 0.2957, (0.84, 0.93, 0.91, 0.82), (0.6972, 0.7151), 0.271),
        (38266, 0.2497, (0.85, 0.94, 0.93, 0.83), (0.6975, 0.7148), 0.106),
        (38297, 0.1023, (0.79, 0.93, 0.95, 0.85), (0.6975, 0.7158), 0.228),
        (38344, 0.1099, (0.78, 0.91, 0.93, 0.85), (0.6965, 0.7137), 0.362),
        (38375, 0.1116, (0.80, 0.92, 0.93, 0.85), (0.6957, 0.7152), 0.416),
        (38406, 0.0969, (0.79, 0.92, 0.94, 0.84), (0.6957, 0.7148), 0.093),
        (38438, 0.1141, (0.78, 0.92, 0.93, 0.83), (0.6957, 0.7139), 0.201),
        (38469, 0.2482, (0.83, 0.92, 0.92, 0.82), (0.6960, 0.7149), 0.233),
        (38500, 0.3018, (0.83, 0.92, 0.91, 0.81), (0.6958, 0.7159), 0.221),
        (38531, 0.3691, (0.85, 0.92, 0.91, 0.81), (0.6964, 0.7161), 0.162),
        (38563, 0.4041, (0.85, 0.91, 0.90, 0.80), (0.6963, 0.7166), 0.116),
        (38594, 0.4109, (0.86, 0.91, 0.90, 0.80), (0.6964, 0.7165), 0.025),
    ],
]


def replay_rule_f(sequence, closing_grace_ms):
    """Réglages de l'essai du 2026-11-25 : règle F, min_closing_speed 0,4."""
    return replay(sequence, rule="F", min_closing_speed=0.4, closing_grace_ms=closing_grace_ms,
                  closing_grace_max_speed=1.0, release_fast_margin=0.05, release_rise=0.15,
                  double_click_snap=0.08)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_rule_f_keeps_real_double_clicks_with_the_grace(index):
    events = replay_rule_f(REAL_DOUBLE_CLICKS[index], closing_grace_ms=300)
    assert count(events, PressAt) == 2
    assert count(events, ReleaseAt) == 2


def press_times(events):
    times = []
    for t, event in events:
        if isinstance(event, PressAt):
            times.append(t)
    return times


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_rule_f_without_grace_refuses_or_delays_the_second_tap(index):
    """Sans exemption, la 2e tape est refusée (19 578 ms), ou acceptée seulement quand la
    fenêtre de 200 ms a dépassé le pic entre les deux tapes : 94 à 141 ms plus tard, presque au
    relâchement (25 000, 29 735 et 38 266 ms)."""
    with_grace = press_times(replay_rule_f(REAL_DOUBLE_CLICKS[index], closing_grace_ms=300))
    without = press_times(replay_rule_f(REAL_DOUBLE_CLICKS[index], closing_grace_ms=0))
    assert without[0] == with_grace[0]  # 1re tape : fermeture rapide, acceptée dans les deux cas
    if len(without) == 2:
        assert without[1] - with_grace[1] >= 90
    else:
        assert len(without) == 1


# --- folded_below 0,65 en règle F (log 2026-11-25_17-41-02) -------------------------------------
# Clic à gauche refusé pour « autres_doigts » à 102 531 ms : main tournée vers la gauche, les
# autres doigts paraissent courts à l'image (ext_autres_max 0,83 < 0,87).
LEFT_CLICK_102531 = [
    (101906, 0.5198, (0.88, 0.83, 0.84, 0.75), (0.7143, 0.7361), 0.901),
    (101952, 0.5303, (0.90, 0.85, 0.85, 0.75), (0.7152, 0.7330), 0.421),
    (101984, 0.5183, (0.89, 0.84, 0.85, 0.76), (0.7148, 0.7306), 0.453),
    (102046, 0.5075, (0.87, 0.84, 0.84, 0.76), (0.7172, 0.7284), 0.366),
    (102093, 0.5126, (0.88, 0.84, 0.85, 0.76), (0.7195, 0.7256), 0.519),
    (102125, 0.5020, (0.87, 0.83, 0.85, 0.76), (0.7207, 0.7250), 0.314),
    (102171, 0.4756, (0.86, 0.82, 0.83, 0.75), (0.7232, 0.7244), 0.424),
    (102218, 0.4919, (0.86, 0.82, 0.84, 0.76), (0.7245, 0.7251), 0.233),
    (102250, 0.4956, (0.87, 0.83, 0.84, 0.77), (0.7257, 0.7250), 0.299),
    (102296, 0.5061, (0.86, 0.82, 0.83, 0.76), (0.7267, 0.7250), 0.160),
    (102327, 0.4980, (0.85, 0.82, 0.83, 0.76), (0.7279, 0.7251), 0.302),
    (102375, 0.5051, (0.85, 0.83, 0.84, 0.76), (0.7288, 0.7235), 0.250),
    (102421, 0.4971, (0.84, 0.82, 0.84, 0.77), (0.7288, 0.7229), 0.073),
    (102452, 0.5035, (0.85, 0.82, 0.83, 0.77), (0.7291, 0.7231), 0.100),
    (102500, 0.4195, (0.80, 0.82, 0.85, 0.77), (0.7301, 0.7245), 0.226),
    (102531, 0.1477, (0.70, 0.80, 0.83, 0.75), (0.7302, 0.7286), 0.754),
    (102577, 0.1454, (0.68, 0.77, 0.83, 0.74), (0.7302, 0.7280), 0.069),
    (102625, 0.1361, (0.68, 0.77, 0.82, 0.74), (0.7290, 0.7280), 0.198),
    (102656, 0.1822, (0.74, 0.77, 0.83, 0.75), (0.7294, 0.7272), 0.181),
    (102702, 0.2737, (0.76, 0.76, 0.82, 0.77), (0.7297, 0.7263), 0.119),
    (102750, 0.3878, (0.80, 0.79, 0.84, 0.77), (0.7291, 0.7271), 0.137),
    (102796, 0.4793, (0.84, 0.82, 0.84, 0.77), (0.7288, 0.7266), 0.089),
]


def replay_with_folded_below(sequence, folded_below):
    fingers = FingersConfig(folded_below=folded_below, folded_below_3d=0.91)
    return replay(sequence, rule="F", fingers=fingers, min_closing_speed=0.4,
                  closing_grace_ms=300, closing_grace_max_speed=1.0, double_click_snap=0.08,
                  release_fast_margin=0.05, release_rise=0.15)


def test_left_click_refused_with_087_is_accepted_with_065():
    assert count(replay_with_folded_below(LEFT_CLICK_102531, 0.87), PressAt) == 0
    events = replay_with_folded_below(LEFT_CLICK_102531, 0.65)
    assert count(events, PressAt) == 1
    assert count(events, ReleaseAt) == 1


def test_fast_closing_fist_is_still_refused_with_065():
    """Poing qui se ferme vite (la vitesse de fermeture ne le distingue pas d'un clic) : autres
    doigts à 0,45, sous 0,65 (poing des sessions v4 : P95 0,40 / 0,49, max 0,56)."""
    fist = (1.0, 0.45, 0.45, 0.45)
    sequence = []
    ratios = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.1, 0.08, 0.08, 0.08]
    for i in range(len(ratios)):
        sequence.append((1000 + 33 * i, ratios[i], fist, (0.5, 0.5), 0.2))
    assert count(replay_with_folded_below(sequence, 0.65), PressAt) == 0


# --- Exemption de fermeture : seulement pour un vrai candidat 2e tape ----------------------------
# Log 2026-11-25_17-54-53, traversée rapide de l'écran : un clic sur place (64 296 ms), puis
# 31 ms après son relâchement un contact à 4,43 tailles de main / s, à 203 px du clic
# (64 827 ms), qui était exempté et devenait un glisser de 891 ms. Taille de main ≈ 0,33
# (anchor_distance 0,125 pour 0,041 de déplacement à 64 750 ms).
FAST_CROSSING_64827 = [
    (64264, 0.3144, (1.15, 1.36, 1.32, 1.19), (0.2807, 0.8456), 0.250),
    (64296, 0.1785, (1.12, 1.35, 1.35, 1.21), (0.2779, 0.8456), 0.746),
    (64343, 0.1978, (1.13, 1.33, 1.33, 1.17), (0.2775, 0.8453), 0.087),
    (64375, 0.2654, (1.10, 1.30, 1.32, 1.17), (0.2779, 0.8450), 0.102),
    (64421, 0.1671, (1.12, 1.32, 1.31, 1.18), (0.2780, 0.8449), 0.028),
    (64468, 0.1609, (1.14, 1.34, 1.32, 1.17), (0.2812, 0.8469), 0.634),
    (64500, 0.1740, (1.13, 1.34, 1.33, 1.18), (0.2828, 0.8454), 0.546),
    (64546, 0.1998, (1.13, 1.34, 1.31, 1.18), (0.2850, 0.8465), 0.421),
    (64577, 0.1992, (1.11, 1.32, 1.29, 1.14), (0.2862, 0.8463), 0.340),
    (64625, 0.2197, (1.11, 1.32, 1.28, 1.14), (0.2870, 0.8453), 0.192),
    (64671, 0.1638, (1.07, 1.29, 1.26, 1.13), (0.2942, 0.8466), 1.303),
    (64702, 0.1819, (1.06, 1.24, 1.26, 1.13), (0.3023, 0.8477), 2.146),
    (64750, 0.1282, (1.04, 1.23, 1.20, 1.04), (0.3190, 0.8525), 2.744),
    (64796, 0.6664, (1.15, 1.16, 1.18, 1.00), (0.3402, 0.8614), 3.651),
    (64827, 0.1549, (1.07, 1.22, 1.19, 1.02), (0.3575, 0.8652), 4.431),
    (64875, 0.1107, (0.96, 1.13, 1.07, 0.90), (0.3886, 0.8769), 5.007),
    (64905, 0.1597, (0.99, 1.05, 1.03, 0.90), (0.4275, 0.8836), 9.325),
    (64968, 0.1078, (0.90, 1.01, 0.97, 0.80), (0.4668, 0.8894), 4.366),
    (65014, 0.0981, (0.89, 1.01, 0.96, 0.73), (0.5033, 0.8909), 5.537),
    (65046, 0.1146, (0.87, 0.98, 0.94, 0.76), (0.5410, 0.8900), 8.120),
    (65093, 0.1216, (0.82, 0.94, 0.92, 0.78), (0.5745, 0.8856), 4.867),
]


def replay_crossing(**overrides):
    """Réglages en service (règle F, folded_below 0,65...), avec des surcharges."""
    settings = {"min_closing_speed": 0.4, "closing_grace_ms": 300,
                "closing_grace_max_speed": 1.0, "double_click_snap": 0.08,
                "release_fast_margin": 0.05, "release_rise": 0.15}
    for name in overrides:
        settings[name] = overrides[name]
    fingers = FingersConfig(folded_below=0.65, folded_below_3d=0.91)
    return replay(FAST_CROSSING_64827, rule="F", fingers=fingers, hand_size=0.33, **settings)


def test_fast_contact_after_a_click_is_not_exempted():
    """64 827 ms n'est plus exempté (fermeture 0,32 < 0,4) : refusé. LIMITE : à 65 014 ms, un
    pic de fermeture (2,61, suivi bruité en plein mouvement) fait quand même entrer un contact,
    Press à 65 046 ms. La vitesse du poignet (5,54) le distingue : max_contact_speed."""
    events = replay_crossing()
    assert press_times(events) == [64343, 65046]
    assert count(events, ReleaseAt) == 1  # le clic sur place de 64 296 ms


def test_fast_contact_was_exempted_without_the_candidate_conditions():
    """Conditions relâchées (aimant 1 taille de main, pas de limite de vitesse) : on retrouve
    le défaut, le contact de 64 827 ms est exempté et appuie dès l'image suivante."""
    events = replay_crossing(double_click_snap=1.0, closing_grace_max_speed=100)
    assert press_times(events) == [64343, 64875]


def machine_rule_f():
    engagement = EngagementConfig(hold_ms=300, lost_timeout_ms=1000)
    pinch = PinchConfig(rule="F", click_mode="press", min_closing_speed=0.4,
                        closing_grace_ms=300, closing_grace_max_speed=1.0,
                        double_click_snap=0.08, drag_threshold=0.15)
    return GestureStateMachine(engagement, pinch, pointer=PointerConfig(mode="absolute"))


def re_pinch_after(m, t, position):
    """Relâche, puis repince aussitôt au même endroit, poignet lent : fermeture négative sur
    200 ms (la mesure remonte depuis le creux précédent)."""
    m.update(t, OPEN, position, 0.2)  # relâchement
    m.update(t + 33, PINCH, position, 0.2)  # nouveau contact 33 ms après
    return m.second_tap_candidate


def test_second_tap_candidate_after_a_click_in_place():
    m = machine_rule_f()
    t = activate(m)
    m.update(t, OPEN, P, 0.2)
    for dt in range(300, 600, 33):  # appui de 300 ms : la fenêtre de 200 ms reste pincée
        m.update(t + dt, PINCH, P, 0.2)
    assert m.state == State.PINCHING
    assert re_pinch_after(m, t + 600, P)
    assert m.closing_speed <= 0  # pas de fermeture mesurée...
    assert m.state == State.PINCHING  # ... mais exempté : la 2e tape est acceptée


def test_no_second_tap_candidate_after_a_drag():
    m = machine_rule_f()
    t = activate(m)
    m.update(t, OPEN, P, 0.2)
    m.update(t + 300, PINCH, P, 0.2)
    m.update(t + 333, PINCH, P, 0.2)  # Press
    for dt in range(366, 600, 33):
        m.update(t + dt, PINCH, (0.60, 0.50), 0.2)  # glisser
    assert m.state == State.DRAGGING
    assert not re_pinch_after(m, t + 600, (0.60, 0.50))
    assert m.closing_speed <= 0
    assert m.state == State.ACTIVE  # pas d'exemption : refusé
    assert m.entry_block == "fermeture"
