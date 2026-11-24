"""Filtre 1€ : lisse le tremblement du curseur sans ajouter trop de retard.

Principe : un filtre passe-bas dont la fréquence de coupure s'adapte à la vitesse.
- main lente  -> coupure basse  -> fort lissage, le curseur ne tremble pas ;
- main rapide -> coupure haute  -> peu de lissage, le curseur suit sans traîner.

Référence : Casiez, Roussel & Vogel, « 1€ Filter », CHI 2012.
https://gery.casiez.net/1euro/
"""

from __future__ import annotations

import math


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    """Coefficient de lissage d'un filtre passe-bas du premier ordre."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    """Filtre 1€ pour UNE valeur (par exemple x)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    def __call__(self, x: float, t_s: float) -> float:
        # Premier échantillon : rien à lisser.
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t_s
            return x

        dt = t_s - self._t_prev
        if dt <= 0:
            return self._x_prev

        # 1) Vitesse estimée, elle-même lissée.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # 2) Coupure adaptative : plus on va vite, plus on laisse passer.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t_s
        return x_hat


class PointFilter:
    """Deux filtres 1€, un pour x et un pour y."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t_s: float) -> tuple[float, float]:
        return self.fx(x, t_s), self.fy(y, t_s)
