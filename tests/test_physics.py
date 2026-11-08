import numpy as np

from pinpoint.doppler import C, ideal_curve
from pinpoint.flyover import Track
from pinpoint.search import Points


def synthetic_track(lat0, lon0, v=120.0, alt=3000.0, offset=2000.0, dur=600):
    """Straight east-bound flight at constant speed, passing `offset` m north of the observer at t=0."""
    t = np.arange(-dur / 2, dur / 2 + 1, 2.0)
    k = np.pi / 180 * 6371000.0
    lat = np.full_like(t, lat0 + offset / k)
    lon = lon0 + v * t / (k * np.cos(np.radians(lat0)))
    return t, lat, lon, np.full_like(t, alt)


def test_cpa_heard_at_rest_frequency_and_asymptotes():
    f0, v, d, tc = 100.0, 120.0, 3000.0, 50.0
    assert abs(ideal_curve(tc, f0, v, d, tc) - f0) < 1e-6
    far = ideal_curve(np.array([tc - 1e5, tc + 1e5]), f0, v, d, tc)
    assert np.allclose(far, [f0 * C / (C - v), f0 * C / (C + v)], rtol=1e-3)


def test_track_model_matches_closed_form():
    """Track.curve (numeric, emission-time) and ideal_curve (closed form) must agree on a straight flight."""
    lat0, lon0 = 40.0, -100.0
    t, lat, lon, alt = synthetic_track(lat0, lon0)
    tr, ratio, _, t_heard, rmin = Track(t, lat, lon, alt).curve(lat0, lon0)
    assert abs(rmin - np.hypot(2000.0, 3000.0)) < 5
    assert abs(t_heard - rmin / C) < 1.0  # CPA emitted at t=0, heard after slant/c
    sel = np.abs(tr - t_heard) < 120
    expect = ideal_curve(tr[sel], 1.0, 120.0, rmin, t_heard)
    assert np.max(np.abs(ratio[sel] - expect)) < 0.004


def test_grid_window_equals_brute_force():
    pts = Points.grid((40.0, -74.0, 40.2, -73.7), 200)
    box = (40.05, 40.11, -73.9, -73.81)
    brute = np.flatnonzero((pts.lat >= box[0]) & (pts.lat <= box[1]) & (pts.lon >= box[2]) & (pts.lon <= box[3]))
    assert set(pts.window(*box)) == set(brute)
    sub = pts.subset(brute[::3])
    assert set(sub.window(*box)) == set(brute[::3])
