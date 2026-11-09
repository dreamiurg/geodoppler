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
