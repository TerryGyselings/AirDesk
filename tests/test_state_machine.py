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
from airdesk.interpretation.state_machine import (
    Click,
    GestureStateMachine,
    Move,
    MoveTo,
    Press,
    Release,
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


def test_press_comes_one_frame_after_the_contact():
    """Latence : Press dès la 2e image de pincement (min_press_frames = 2), pas au relâchement."""
    m = machine_press()
    t = activate_press(m)
    assert m.update(t, PINCH, P) == []  # contact : 1re image, pas encore confirmé
    assert m.state == State.PINCHING
    assert m.update(t + 33, PINCH, P) == [Press()]
    assert m.event_contact_ms == t  # pour la colonne contact_ms du log
    assert m.update(t + 66, PINCH, P) == []
    assert m.event_contact_ms is None  # seulement sur l'image du Press


def test_click_in_place_is_press_then_release_with_a_frozen_cursor():
    m = machine_press()
    t = activate_press(m)
    events = m.update(t, PINCH, (0.50, 0.50))
    events += m.update(t + 33, PINCH, (0.51, 0.50))  # petits tremblements : 0,05 taille de main
    events += m.update(t + 66, PINCH, (0.50, 0.51))
    events += m.update(t + 600, PINCH, (0.51, 0.51))  # appui long sans bouger : pas de glisser
    events += m.update(t + 633, OPEN, (0.51, 0.51))
    assert events == [Press(), Release()]  # aucun MoveTo : le curseur est resté figé
    assert m.state == State.ACTIVE


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
    m.update(t + 33, PINCH, (0.50, 0.50))  # Press
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
    assert Release() in events


def test_drag_without_catch_up_jumps_to_the_hand():
    m = machine_press(drag_catchup_ms=0)
    t = activate_press(m)
    m.update(t, PINCH, (0.50, 0.50))
    m.update(t + 33, PINCH, (0.50, 0.50))
    m.update(t + 66, PINCH, (0.60, 0.50))
    first = moves(m.update(t + 99, PINCH, (0.60, 0.50)))[0]
    assert first.u == pytest.approx((0.60 - 0.15) / 0.70)


def test_double_click_is_two_press_release_at_the_same_place():
    m = machine_press()
    t = activate_press(m)
    events = []
    for start in (t, t + 200):
        events += m.update(start, PINCH, P)
        events += m.update(start + 33, PINCH, P)
        events += m.update(start + 66, OPEN, P)
    events += m.update(t + 300, OPEN, P)
    assert events == [Press(), Release(), Press(), Release()]


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


def test_release_mode_logs_the_contact_on_the_click():
    m = machine_zone()
    t = activate_at(m, IN)
    m.update(t, PINCH, IN)
    events = m.update(t + 33, OPEN, IN)
    assert events == [Click()]
    assert m.event_contact_ms == t

