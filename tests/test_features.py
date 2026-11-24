import math

import pytest

from airdesk.interpretation.features import (
    compute_features,
    count_folded,
    normalize_landmarks,
    palm_center,
    point_segment_distance,
    wrist_speed,
)


def make_hand(fingers_extended: bool, pinch: bool):
    """Main synthétique verticale : poignet en bas, doigts vers le haut (y diminue vers le haut)."""
    pts = [(0.0, 0.0, 0.0)] * 21
    pts = list(pts)
    pts[0] = (0.50, 0.80, 0.0)  # poignet
    pts[9] = (0.50, 0.60, 0.0)  # base du majeur -> taille de main = 0.20

    tip_y = 0.35 if fingers_extended else 0.62  # doigt replié : bout revenu vers la paume
    columns = [0.44, 0.50, 0.55, 0.60]
    fingers = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for i in range(4):
        mcp, pip, dip, tip = fingers[i]
        x = columns[i]
        pts[mcp] = (x, 0.60, 0.0)
        pts[pip] = (x, 0.50, 0.0)
        pts[dip] = (x, 0.45, 0.0)
        pts[tip] = (x, tip_y, 0.0)

    # pouce
    pts[1] = (0.40, 0.72, 0.0)
    pts[2] = (0.36, 0.65, 0.0)
    pts[3] = (0.34, 0.58, 0.0)
    pts[4] = pts[8] if pinch else (0.30, 0.52, 0.0)
    return pts


def test_open_palm_detected():
    f = compute_features(make_hand(fingers_extended=True, pinch=False))
    assert f.extended_fingers == 4
    assert f.open_palm


def test_fist_is_not_open_palm():
    f = compute_features(make_hand(fingers_extended=False, pinch=False))
    assert f.extended_fingers == 0
    assert not f.open_palm


def test_pinch_ratio_small_when_thumb_touches_index():
    pinched = compute_features(make_hand(fingers_extended=True, pinch=True))
    apart = compute_features(make_hand(fingers_extended=True, pinch=False))
    assert pinched.pinch_ratio < 0.05
    assert apart.pinch_ratio > 0.5


def test_features_do_not_depend_on_distance_to_camera():
    """Une main deux fois plus petite (plus loin) donne les mêmes rapports."""
    hand = make_hand(fingers_extended=True, pinch=False)
    small = []
    for i in range(len(hand)):
        x, y, z = hand[i]
        small.append((0.5 + (x - 0.5) / 2, 0.5 + (y - 0.5) / 2, z))
    a = compute_features(hand)
    b = compute_features(small)
    assert abs(a.pinch_ratio - b.pinch_ratio) < 1e-9
    assert a.extended_fingers == b.extended_fingers
    for i in range(4):
        assert abs(a.finger_extensions[i] - b.finger_extensions[i]) < 1e-9


# --- Extension des doigts (bout -> centre de la paume / taille de main) ------------------------
#
# Main synthétique : centre de paume = moyenne de (0.50, 0.80), (0.44, 0.60), (0.50, 0.60),
# (0.55, 0.60), (0.60, 0.60) = (0.518, 0.64). Taille de main 0.20.
# Index tendu (0.44, 0.35) : distance 0.300 -> extension 1.50.
# Index replié (0.44, 0.62) : distance 0.080 -> extension 0.40.


def rotate_90(points):
    """Rotation de 90° autour du centre de l'image (0.5, 0.5)."""
    rotated = []
    for x, y, z in points:
        rotated.append((0.5 - (y - 0.5), 0.5 + (x - 0.5), z))
    return rotated


def to_image_coordinates(points, width, height):
    """Points définis en unités « carrées » (x480 px) -> coordonnées MediaPipe d'une image
    width x height (x divisé par la largeur, y par la hauteur, z à l'échelle de x)."""
    converted = []
    for x, y, z in points:
        px = x * height
        py = y * height
        converted.append((px / width, py / height, z * height / width))
    return converted


def test_palm_center_is_the_mean_of_wrist_and_finger_bases():
    cx, cy, cz = palm_center(make_hand(fingers_extended=True, pinch=False))
    assert math.isclose(cx, 0.518)
    assert math.isclose(cy, 0.64)
    assert cz == 0.0


def test_open_hand_has_long_extensions_and_no_folded_finger():
    f = compute_features(make_hand(fingers_extended=True, pinch=False))
    assert abs(f.index_extension - 1.50) < 0.01
    for value in f.finger_extensions:
        assert value > 1.4
    assert f.folded_fingers == 0


def test_fist_has_short_extensions_and_four_folded_fingers():
    f = compute_features(make_hand(fingers_extended=False, pinch=False))
    assert abs(f.index_extension - 0.40) < 0.01
    for value in f.finger_extensions:
        assert value < 0.5
    assert f.folded_fingers == 4


def test_fist_with_thumb_on_folded_index_looks_like_a_pinch_but_index_is_folded():
    """Le bug constaté : poing fermé + pouce sur l'index replié = pinch_ratio minuscule.
    Seule l'extension de l'index permet de le distinguer d'un vrai pincement."""
    fist = compute_features(make_hand(fingers_extended=False, pinch=True))
    real_pinch = compute_features(make_hand(fingers_extended=True, pinch=True))
    assert fist.pinch_ratio < 0.05  # ressemble à un pincement...
    assert real_pinch.pinch_ratio < 0.05
    assert fist.index_extension < 0.8  # ... mais l'index est replié
    assert real_pinch.index_extension > 1.2


def test_extensions_do_not_depend_on_rotation():
    hand = make_hand(fingers_extended=True, pinch=False)
    a = compute_features(hand)
    b = compute_features(rotate_90(hand))
    for i in range(4):
        assert abs(a.finger_extensions[i] - b.finger_extensions[i]) < 1e-9


def test_aspect_correction_makes_extensions_rotation_invariant_in_640x480():
    """En 640x480, x est divisé par 640 et y par 480 : sans correction, tourner la main change
    ses proportions. Avec aspect = 640/480, on retrouve les mêmes extensions."""
    hand = make_hand(fingers_extended=True, pinch=False)
    upright = to_image_coordinates(hand, 640, 480)
    turned = to_image_coordinates(rotate_90(hand), 640, 480)
    a = compute_features(upright, aspect=640 / 480)
    b = compute_features(turned, aspect=640 / 480)
    assert abs(a.index_extension - b.index_extension) < 1e-9
    assert abs(a.index_extension - 1.50) < 0.01
    # Sans la correction, les deux postures ne donnent pas la même valeur.
    c = compute_features(upright)
    d = compute_features(turned)
    assert abs(c.index_extension - d.index_extension) > 0.03  # ≈ 1,48 contre 1,54


def test_index_pointing_at_camera_is_short_in_2d_but_not_in_3d():
    """Raccourcissement de perspective : bout de l'index au-dessus de sa base dans l'image,
    mais 0.28 plus près de la caméra (z négatif = plus proche, à l'échelle de x)."""
    hand = make_hand(fingers_extended=True, pinch=False)
    hand[8] = (0.44, 0.56, -0.28)
    f = compute_features(hand)
    assert f.index_extension < 0.8  # en 2D, l'index paraît replié
    assert abs(f.index_extension_3d - 1.51) < 0.01  # en 3D, il est bien tendu
    assert f.folded_fingers == 1
    assert f.folded_fingers_3d == 0


def test_point_segment_distance():
    a = (0.0, 0.0, 0.0)
    b = (0.0, 1.0, 0.0)
    assert point_segment_distance((0.5, 0.5, 0.0), a, b) == 0.5  # projection dans le segment
    assert point_segment_distance((0.0, 2.0, 0.0), a, b) == 1.0  # au-delà de b : distance à b
    assert point_segment_distance((0.0, -3.0, 0.0), a, b) == 3.0  # avant a : distance à a
    assert point_segment_distance((1.0, 1.0, 0.0), a, a) == math.sqrt(2)  # segment réduit à a


def test_pinch_seg_stays_small_when_the_thumb_touches_below_the_tip():
    """Pouce posé sur la dernière phalange, sous le bout : pinch_ratio le croit écarté,
    pinch_seg voit le contact. Hypothèse à vérifier sur les mesures (moins de ruptures)."""
    hand = make_hand(fingers_extended=True, pinch=False)
    hand[4] = (0.44, 0.42, 0.0)  # entre 7 (0.45) et 8 (0.35) sur l'index (x = 0.44)
    f = compute_features(hand)
    assert f.pinch_ratio > 0.3  # 0.07 / 0.20 = 0.35
    assert f.pinch_seg < 0.01
    assert f.pinch_seg_68 < 0.01


def test_pinch_seg_does_not_depend_on_scale_or_rotation():
    hand = make_hand(fingers_extended=True, pinch=False)
    hand[4] = (0.40, 0.40, 0.0)
    small = []
    for x, y, z in hand:
        small.append((0.5 + (x - 0.5) / 2, 0.5 + (y - 0.5) / 2, z))
    a = compute_features(hand)
    b = compute_features(small)
    c = compute_features(rotate_90(hand))
    assert abs(a.pinch_seg - b.pinch_seg) < 1e-9
    assert abs(a.pinch_seg - c.pinch_seg) < 1e-9


def turn_around_vertical_axis(points, degrees):
    """Tourne la main autour d'un axe vertical passant par le poignet (vers le profil).
    x se contracte (cosinus) et z apparaît (sinus) ; z à l'échelle de x, comme MediaPipe."""
    angle = math.radians(degrees)
    x0 = points[0][0]
    turned = []
    for x, y, z in points:
        dx = x - x0
        turned.append((x0 + dx * math.cos(angle), y, z + dx * math.sin(angle)))
    return turned


def test_palm_width_and_facing_from_front_to_profile():
    """Main synthétique : base index (0.44) -> base auriculaire (0.60) = 0.16, taille 0.20.
    De face 0.80 ; tournée de 60° 0.40 ; de profil (80°) 0.14 ; palm_facing = cos(angle)."""
    hand = make_hand(fingers_extended=True, pinch=False)
    front = compute_features(hand)
    assert front.palm_width == pytest.approx(0.80)
    assert front.palm_facing == pytest.approx(1.0)
    turned = compute_features(turn_around_vertical_axis(hand, 60))
    assert turned.palm_width == pytest.approx(0.40)
    assert turned.palm_facing == pytest.approx(0.5)
    profile = compute_features(turn_around_vertical_axis(hand, 80))
    assert profile.palm_width == pytest.approx(0.139, abs=0.001)
    assert profile.palm_facing == pytest.approx(math.cos(math.radians(80)))


def test_palm_measures_do_not_depend_on_scale():
    hand = turn_around_vertical_axis(make_hand(fingers_extended=True, pinch=False), 45)
    small = []
    for x, y, z in hand:
        small.append((0.5 + (x - 0.5) / 2, 0.5 + (y - 0.5) / 2, z / 2))
    a = compute_features(hand)
    b = compute_features(small)
    assert a.palm_width == pytest.approx(b.palm_width)
    assert a.palm_facing == pytest.approx(b.palm_facing)


def test_wrist_speed_in_hand_sizes_per_second():
    """Le poignet avance de 0.02 (largeur d'image) en 33 ms, image 640x480, taille 0.20 :
    0.02 x 4/3 / 0.033 / 0.20 = 4.04 tailles de main par seconde."""
    hand = make_hand(fingers_extended=True, pinch=False)
    moved = []
    for x, y, z in hand:
        moved.append((x + 0.02, y, z))
    assert wrist_speed(moved, hand, 0.033, aspect=640 / 480) == pytest.approx(4.04, abs=0.01)
    assert wrist_speed(moved, hand, 0.0) is None


def test_count_folded_uses_the_threshold():
    assert count_folded((1.5, 0.79, 0.8, 0.2), folded_below=0.8) == 2
    assert count_folded((1.5, 0.79, 0.8, 0.2), folded_below=0.5) == 1


def test_normalize_landmarks_puts_wrist_at_origin_and_hand_size_at_one():
    hand = make_hand(fingers_extended=True, pinch=False)
    normalized = normalize_landmarks(hand)
    assert normalized[0] == (0.0, 0.0, 0.0)
    x9, y9, _ = normalized[9]
    assert math.isclose(math.hypot(x9, y9), 1.0)
    # Même forme, plus loin de la caméra (deux fois plus petite) : mêmes points normalisés.
    small = []
    for x, y, z in hand:
        small.append((0.5 + (x - 0.5) / 2, 0.5 + (y - 0.5) / 2, z / 2))
    normalized_small = normalize_landmarks(small)
    for i in range(21):
        for axis in range(3):
            assert abs(normalized[i][axis] - normalized_small[i][axis]) < 1e-9
