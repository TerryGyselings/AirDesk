"""Tests du garde-fou de taille de main."""

from airdesk.interpretation.quality import MIN_HISTORY, HandSizeGuard


def feed(guard, sizes, step_ms=33, start_ms=0):
    """Envoie une suite de hand_size ; renvoie la liste des décisions (True = acceptée)."""
    decisions = []
    for i in range(len(sizes)):
        decisions.append(guard.accept(start_ms + i * step_ms, sizes[i]))
    return decisions


def test_short_history_accepts_everything():
    guard = HandSizeGuard(min_ratio=0.5)
    assert feed(guard, [0.2, 0.01, 0.2, 0.01]) == [True, True, True, True]
    assert MIN_HISTORY == 5


def test_sudden_collapse_is_rejected():
    """Main de profil : la taille apparente tombe au quart de la médiane récente."""
    guard = HandSizeGuard(min_ratio=0.5)
    decisions = feed(guard, [0.2] * 10 + [0.05, 0.2])
    assert decisions[10] is False
    assert decisions[11] is True


def test_gradual_move_away_is_accepted():
    """La main s'éloigne doucement (-1 % par image) : jamais de chute brutale."""
    guard = HandSizeGuard(min_ratio=0.5)
    sizes = []
    size = 0.2
    for _ in range(150):
        sizes.append(size)
        size *= 0.99
    assert all(feed(guard, sizes))


def test_median_follows_a_real_change_of_distance():
    """Toutes les images entrent dans l'historique : après un vrai recul, la médiane suit et
    les images sont de nouveau acceptées (pas de rejet sans fin)."""
    guard = HandSizeGuard(min_ratio=0.5, window_ms=2000)
    decisions = feed(guard, [0.2] * 60 + [0.08] * 90)
    assert decisions[60] is False  # chute brutale : rejetée d'abord
    assert decisions[-1] is True  # puis acceptée quand la médiane a suivi


def test_zero_ratio_disables_the_guard():
    guard = HandSizeGuard(min_ratio=0.0)
    assert all(feed(guard, [0.2] * 10 + [0.001]))
