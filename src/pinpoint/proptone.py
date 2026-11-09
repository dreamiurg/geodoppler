"""Propeller base tone: which aircraft types could have made an event.

An engine/propeller's tones are integer (or half-integer: 4-stroke exhaust) multiples of a shaft rate. The
event's de-Dopplered spectrum turns them into a comb whose spacing `fb` is one of a few low orders; the fitted
f0 is just whichever tooth was brightest. Per type, FAMILIES lists (rpm_lo, rpm_hi, order): fb = order * rpm / 60.
Validation (experiments/f0-type): +-5 % bands pass all 20 matches of correct runs and reject 10 of 29 in wrong runs.
Most rpm limits are typical values, not manuals. Unknown types pass, as adsb.is_jet keeps them.
"""

import numpy as np
import pandas as pd

from .audio import Spec
from .doppler import Event, ideal_curve

PISTON_RPM = (1700, 2800)  # approach/descent ~1700-2000 to climb 2700, plus overspeed slack
PISTON_MAX_FB = 150.0  # Hz; any direct-drive piston, mapped or not: 2800 rpm x 3 blades / 60 = 140


def _piston(cyl, blades):
    # comb spacing: half-order (exhaust), shaft, 1.5 (6-cyl), 2nd, firing order, blade pass
    orders = {0.5, 1.0, 2.0, cyl / 2, float(blades)} | ({1.5} if cyl == 6 else set())
    return [(*PISTON_RPM, o) for o in sorted(orders)]


def _turboprop(rpm_lo, rpm_hi, *blades):  # governed prop rpm: shaft and blade pass
    return [(rpm_lo, rpm_hi, 1.0)] + [(rpm_lo, rpm_hi, float(b)) for b in blades]


_ROTAX = [(4500 / 2.43, 5800 / 2.43, b) for b in (2.0, 3.0)] + [(4500, 5800, o) for o in (1.0, 2.0)]  # 2.43:1 gearbox
FAMILIES = {
    **{t: _piston(4, 2) for t in ("C172", "C152", "C150", "P28A", "P28R", "P28B", "C72R", "C77R", "C177", "M20P",
                                  "M20T", "AA5", "AA1", "PA44", "RV9", "RV7", "RV6", "RV8", "RV10", "BE23", "BE19")},
    **{t: _piston(4, 3) for t in ("SR20", "DA40")},
    **{t: _piston(6, 2) for t in ("PA32", "P32R", "C182", "C82R", "BE35", "BE33", "NAVI", "C210", "PA24", "C185")},
    **{t: _piston(6, 3) for t in ("SR22", "S22T", "BE36", "BE58", "BE55", "C206", "T206", "C82T", "PA46", "P210", "LNC4")},
    **{t: _ROTAX for t in ("FDCT", "PIVI", "CTSW", "TL20")},
    "PC12": _turboprop(1500, 1750, 4, 5), "C208": _turboprop(1700, 1900, 3, 4), "BE20": _turboprop(1600, 2000, 3, 4),
    "B350": _turboprop(1600, 2000, 4, 5), "B190": _turboprop(1550, 1700, 4), "P46T": _turboprop(1800, 2030, 4, 5),
    "TBM7": _turboprop(1900, 2000, 4, 5), "TBM8": _turboprop(1900, 2000, 4, 5), "TBM9": _turboprop(1900, 2050, 5),
    "SB20": _turboprop(950, 1100, 6), "C30J": _turboprop(980, 1050, 6),
    "AT76": _turboprop(800, 1200, 6), "AT75": _turboprop(800, 1200, 6), "DH8D": _turboprop(850, 1050, 6),
}

F = np.arange(25.0, 590.0, 0.5)  # de-Dopplered spectrum bins
FB = np.arange(14.0, 200.0, 0.1)  # candidate comb spacings


def base_tone(spec: Spec, e: Event) -> tuple[float, float]:
    """(fb, score): comb spacing of the spectrum along the event's fitted curve, scored tooth minus gap.
    fb/2 halves the score (empty teeth) and 2*fb cancels (odd teeth land on the gaps), so the true spacing wins."""
    t = spec.t[(spec.t >= e.t0) & (spec.t <= e.t1)]
    r = ideal_curve(t, 1.0, e.v, e.d, e.tc)
    S = spec.sample(np.broadcast_to(t, (len(F), len(t))), F[:, None] * r[None, :])
    p = np.nan_to_num(S, nan=0.0).clip(-2, 6).mean(1)
    best = np.full(len(FB), np.nan)
    for i, fb in enumerate(FB):
        n = np.arange(1, int(580.0 / fb) + 1)
        n = n[n * fb >= 30.0]
        if len(n) >= 2:
            best[i] = np.interp(n * fb, F, p).mean() - np.interp((n + 0.5) * fb, F, p).mean()
    k = int(np.nanargmax(best))
    return float(FB[k]), float(best[k])


def plausible(type_code, fb, fb_score, types, tol=0.05, min_score=0.4) -> bool:
    """False only when the comb estimate is reliable and fb fits none of the type's families.
    Unmapped pistons (ICAO desc ..P) get the class rule fb <= PISTON_MAX_FB; other unknown types pass."""
    if fb is None or fb_score is None or fb_score < min_score:
        return True
    fam = FAMILIES.get(type_code or "")
    if fam:
        return any(lo * o / 60 * (1 - tol) <= fb <= hi * o / 60 * (1 + tol) for lo, hi, o in fam)
    desc = types.get(type_code or "", {}).get("desc", "")
    return not (len(desc) == 3 and desc[2] == "P" and fb > PISTON_MAX_FB * (1 + tol))


def keep(type_col: pd.Series, fb, fb_score, types) -> np.ndarray:
    """Row mask of a categorical `type` column: aircraft that could have made an event with this base tone."""
    ok = [c for c in type_col.cat.categories if plausible(c, fb, fb_score, types)]
    return type_col.isin(ok).to_numpy()
