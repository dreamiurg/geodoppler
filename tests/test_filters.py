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


def test_verify_credits_a_fast_sparsely_reported_pass():
    """A 170 m/s aircraft reported every 20 s crosses the verify box in 7 rows; verification must still use its
    whole track (it dropped every segment under 10 rows)."""
    from pinpoint.audio import Spec
    from pinpoint.doppler import C
    from pinpoint.flyover import Track
    from pinpoint.search import verify_extra
    lat0, lon0 = 40.0, -100.0
    t, lat, lon, alt = (a[::10] for a in synthetic_track(lat0, lon0, v=170.0, alt=1000.0, offset=7500.0, dur=800))
    t = t + 1000
    t_rec, ratio, _, tcpa, _ = Track(t, lat, lon, alt).curve(lat0, lon0)
    st, sf = np.arange(0, 2000, 0.5), np.arange(20, 300, 0.5)
    S = np.zeros((len(sf), len(st)), np.float32)
    sel = np.abs(st - tcpa) < 90
    S[np.rint((100 * np.interp(st[sel], t_rec, ratio) - sf[0]) / 0.5).astype(int), np.flatnonzero(sel)] = 5.0
    df = pd.DataFrame({"icao": "x", "t": t, "lat": lat, "lon": lon, "alt_m": alt, "gs": 170.0})
    assert ((np.abs(df.lat - lat0) < 0.11) & (np.abs(df.lon - lon0) < 0.14)).sum() < 10
    cands = pd.DataFrame({"lat": [lat0], "lon": [lon0], "score": [1.0], "start": [0.0], "icao1": ["p"], "icao2": ["q"]})
    primary = [Event(0, 0, 100, 60, 1000, 0, 1), Event(0, 0, 100, 60, 1000, 100, 1)]
    e = Event(tcpa - 90, tcpa + 90, 100.0, 170.0, 7600.0, tcpa, 1.0)
    v = verify_extra(Spec(st, sf, S), cands, [e], primary, df, Tolerances(), log=lambda m: None)
    assert v.xicao1[0] == "x" and v.x1[0] > 2


def test_offset_filter_in_blocks_equals_one_join():
    """Blocking the grid-point join (memory) must not change the hits or their order."""
    from pinpoint.search import offset_filter
    rng = np.random.default_rng(0)
    evs = [Event(0, 0, 100, 60, 1000, 100.0, 1), Event(0, 0, 100, 60, 1000, 400.0, 1)]

    def cands(n, t0):
        return pd.DataFrame({"g": rng.integers(0, 40, n), "t": t0 + rng.uniform(-30, 30, n), "d": 1.0, "gs": 60.0,
                             "icao": rng.choice(["a", "b", "c", "d"], n)})
    c = [cands(300, 1000.0), cands(300, 1300.0)]
    whole = offset_filter(c, evs, Tolerances(), block=10**9)
    assert len(whole) > 100
    pd.testing.assert_frame_equal(whole, offset_filter(c, evs, Tolerances(), block=3))
