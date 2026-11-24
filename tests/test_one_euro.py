import random

from airdesk.perception.one_euro import OneEuroFilter


def test_first_sample_is_returned_unchanged():
    f = OneEuroFilter()
    assert f(0.5, 0.0) == 0.5


def test_constant_signal_stays_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=8.0)
    out = 0.0
    for i in range(100):
        out = f(0.3, i / 30)
    assert abs(out - 0.3) < 1e-9


def test_noise_is_reduced_when_hand_is_still():
    """Main immobile + tremblement : la sortie doit trembler beaucoup moins que l'entrée."""
    random.seed(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=8.0)
    raw = []
    filtered = []
    for i in range(300):
        x = 0.5 + random.uniform(-0.005, 0.005)
        raw.append(x)
        filtered.append(f(x, i / 30))

    def spread(values):
        tail = values[100:]
        return max(tail) - min(tail)

    assert spread(filtered) < 0.5 * spread(raw)


def test_fast_movement_has_little_lag():
    """Main qui se déplace vite : la sortie doit rester proche de l'entrée."""
    f = OneEuroFilter(min_cutoff=1.0, beta=8.0)
    out = 0.0
    x = 0.0
    for i in range(30):
        x = i * 0.02  # traverse 60 % de l'image en 1 s
        out = f(x, i / 30)
    assert abs(out - x) < 0.05
