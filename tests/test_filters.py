import numpy as np
import pandas as pd

from pinpoint.doppler import Event
from pinpoint.search import Points, Tolerances, silence_filter
from test_physics import synthetic_track


def test_silence_guard_ignores_passes_inside_an_events_sound_window():
    """Another aircraft close by 45 s after event 1's CPA is part of that flyover's sound, not a break in silence;
    one 200 s after it is."""
    lat0, lon0 = 40.0, -100.0
    pts = Points.grid((lat0 - 0.001, lon0 - 0.001, lat0 + 0.001, lon0 + 0.001), 200)
    evs = [Event(0, 0, 100, 60, 1000, 0, 1), Event(0, 0, 100, 60, 1000, 1000, 1)]
    hits = pd.DataFrame({"g": [0], "t0": [10_000.0], "t1": [11_000.0], "icao0": ["a"], "icao1": ["b"]})
    for after, kept in ((45, True), (200, False)):
        t, lat, lon, alt = synthetic_track(lat0, lon0, v=60.0, alt=300.0, offset=0.0, dur=300)
        df = pd.DataFrame({"icao": "x", "t": t + 10_000 + after, "lat": lat, "lon": lon, "alt_m": alt, "gs": 60.0})
        assert (len(silence_filter(hits, df, evs, pts, Tolerances())) == 1) == kept


def test_doppler_scale_recovers_radial_velocity_scale():
    """A track's Doppler line painted with its radial velocity scaled by k: doppler_scale finds k back."""
    from pinpoint.audio import Spec
    from pinpoint.doppler import C
    from pinpoint.flyover import Track
    from pinpoint.search import doppler_scale
    lat0, lon0 = 40.0, -100.0
    t, lat, lon, alt = synthetic_track(lat0, lon0, v=60.0, alt=800.0, offset=600.0)
    tr = Track(t + 1000, lat, lon, alt)  # unix time = spectrogram seconds (start 0)
    t_rec, ratio, _, tcpa, _ = tr.curve(lat0, lon0)
    vr = C / ratio - C
    st, sf = np.arange(0, 2000, 0.5), np.arange(20, 300, 0.5)
    for k in (1.0, 1.1):
        S = np.zeros((len(sf), len(st)), np.float32)
        sel = np.abs(st - tcpa) < 90
        f = 100 * C / (C + k * np.interp(st[sel], t_rec, vr))
        S[np.rint((f - sf[0]) / 0.5).astype(int), np.flatnonzero(sel)] = 5.0
        e = Event(tcpa - 90, tcpa + 90, 100.0, 60.0, 1000.0, tcpa, 1.0)
        assert abs(doppler_scale(Spec(st, sf, S), tr, e, lat0, lon0, 0.0) - k) <= 0.011


def test_base_tone_plausibility():
    from pinpoint.adsb import _types
    from pinpoint.proptone import plausible
    T = _types()
    assert plausible("C172", 79.2, 0.79, T)  # 2-blade piston, 2400 rpm x 2 / 60 = 80 Hz (U1bQZeogcsU)
    assert not plausible("C72R", 176.5, 0.77, T)  # 0a7boP2vrtY wrong match: no C172 family near 176 Hz
    assert plausible("C72R", 176.5, 0.30, T)  # comb estimate unreliable: no verdict
    assert not plausible("BL8", 190.0, 0.9, T)  # unmapped piston (L1P): class rule fb <= 150 Hz
    assert plausible("BL8", 120.0, 0.9, T) and plausible("", 190.0, 0.9, T)  # in class / unknown type
