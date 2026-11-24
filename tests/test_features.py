from airdesk.interpretation.features import compute_features


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
