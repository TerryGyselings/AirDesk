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


def replay(sequence, **pinch_args):
    """Rejoue une séquence réelle en mode press, règle D (réglages du 2026-11-24)."""
    pinch = PinchConfig(rule="D", click_mode="press", release_grace_ms=66,
                        release_cancel_margin=0.05, **pinch_args)
    m = GestureStateMachine(ENGAGEMENT, pinch, pointer=PointerConfig(mode="absolute"))
    t0, _, _, position, _ = sequence[0]
    t = t0 - 400
    while t < t0:
        m.update(t, OPEN, position)  # activation au point de départ
        t += 33
    events = []
    for t, ratio, extensions, position, speed in sequence:
        f = replace(features(pinch_ratio=ratio, extended_fingers=3, extension=1.4),
                    finger_extensions=extensions, finger_extensions_3d=extensions,
                    index_extension=extensions[0])
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
