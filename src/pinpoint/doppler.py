"""Doppler physics: ideal straight-line flyover model and spectrogram fitting.

Conventions: `t` is reception time at the observer. A sound emitted at time `te` from
distance R(te) arrives at te + R(te)/c, and is heard at f0 * c / (c + v_r(te)) where
v_r is the source's radial velocity (positive = receding) at *emission* time.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import differential_evolution

from .audio import Spec

C = 343.0  # m/s; shortcut: fixed ~20 C air, no temperature/wind correction


def ideal_curve(t, f0, v, d, tc, c=C):
    """Frequency heard at reception times t for a source on a straight line at speed v
    with closest (slant) distance d, whose CPA sound reaches the observer at time tc."""
    s = np.asarray(t, float) - tc + d / c  # reception time relative to CPA emission
    a = c * c - v * v
    # emission time tau solves (s - tau) c = sqrt(d^2 + v^2 tau^2); root with tau < s
    tau = (c * c * s - np.sqrt(c * c * v * v * s * s + a * d * d)) / a
    vr = v * v * tau / np.sqrt(d * d + v * v * tau * tau)
    return f0 * c / (c + vr)


def curve_score(spec: Spec, t, f, harmonics=(1,)) -> float:
    """Mean clipped brightness along a curve (and its harmonics). NaNs (off-spectrogram) count as 0."""
    vals = [spec.sample(t, f * h) for h in harmonics]
    v = np.clip(np.nan_to_num(np.concatenate(vals), nan=0.0), -2, 6)
    return float(v.mean())


@dataclass
class Event:
    t0: float  # analysed window, seconds into recording
    t1: float
    f0: float  # rest (source) frequency of the fitted line, Hz
    v: float  # m/s
    d: float  # slant CPA distance, m
    tc: float  # reception time of CPA, seconds into recording
    score: float
    harmonics: tuple = (1,)  # multiples of f0 visible (and scored)

    def to_dict(self):
        return asdict(self)


def fit_event(spec: Spec, t0, t1, fmin, fmax, harmonics=(1,), seed=0) -> Event:
    """Fit an ideal flyover curve to the brightest Doppler ridge inside the box.
    `harmonics` are multiples of the fitted line also scored (e.g. (1, 2) if the 2x line is visible)."""
    box = spec.crop(t0 - 5, t1 + 5, fmin * 0.9, fmax * max(harmonics) * 1.1)
    t = box.t[(box.t >= t0) & (box.t <= t1)]

    def neg(p):
        f0, v, d, tc = p
        return -curve_score(box, t, ideal_curve(t, f0, v, d, tc), harmonics)

    bounds = [(fmin, fmax), (15, 180), (80, 8000), (t0, t1)]
    r = differential_evolution(neg, bounds, seed=seed, popsize=40, maxiter=300, tol=1e-7,
                               polish=True)
    f0, v, d, tc = r.x
    return Event(t0, t1, float(f0), float(v), float(d), float(tc), float(-r.fun), tuple(harmonics))
