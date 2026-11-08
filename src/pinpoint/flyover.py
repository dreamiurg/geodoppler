"""Model the Doppler curve a real ADS-B track would produce at an observer.

Uses emission-time radial velocity: the frequency heard at reception time tr comes from
where the aircraft was when it emitted that sound (te, with tr = te + R(te)/c).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .doppler import C

R_EARTH = 6371000.0


def enu(lat, lon, alt_m, lat0, lon0, elev0=0.0):
    """Local tangent-plane metres around the observer.
    shortcut: equirectangular, fine within ~30 km; use pyproj for longer baselines."""
    k = np.pi / 180 * R_EARTH
    return (np.asarray(lon) - lon0) * k * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * k, np.asarray(alt_m) - elev0


class Track:
    """One aircraft pass, resampled to 1 s and lightly smoothed (ADS-B positions jitter)."""

    def __init__(self, t, lat, lon, alt_m, max_gap=30.0):
        t = np.asarray(t, float)
        self.t = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
        self.lat = np.interp(self.t, t, lat)
        self.lon = np.interp(self.t, t, lon)
        self.alt = gaussian_filter1d(np.interp(self.t, t, alt_m), 5)
        # mask samples inside long reporting gaps: interpolated straight lines there are guesses
        gap = np.diff(t)
        idx = np.searchsorted(t, self.t, side="right") - 1
        idx = np.clip(idx, 0, len(gap) - 1)
        self.ok = gap[idx] <= max_gap

    def curve(self, lat0, lon0, elev0=0.0, c=C):
        """Return (reception_time, f/f0, slant_m) per emission sample, and the reception time of CPA."""
        x, y, z = enu(self.lat, self.lon, self.alt, lat0, lon0, elev0)
        x, y = gaussian_filter1d(x, 3), gaussian_filter1d(y, 3)
        R = np.sqrt(x * x + y * y + z * z)
        vr = np.gradient(R, self.t)
        tr = self.t + R / c
        ratio = c / (c + vr)
        i = int(np.argmin(np.where(self.ok, R, np.inf)))
        return tr, ratio, R, tr[i], R[i]
