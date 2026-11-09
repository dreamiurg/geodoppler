"""e3hz7EunI18: four near-identical events 244 s apart (f0 164, v 74-76, d 2.0-2.2 km). One aircraft flying
circuits? List aircraft inside the search bbox, in the window of those events (truth start), that loop:
cumulative heading change >= 3 turns, with their median ground speed. offset_filter forbids one aircraft
explaining both search events, so such an aircraft could never be matched."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import adsb, cli  # noqa: E402

from rank import LAB, truth_unix  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "e3hz7EunI18"
t0, t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 6341, float(sys.argv[3]) if len(sys.argv) > 3 else 7074
c = LAB[run]
T = truth_unix(c)
la0, lo0, la1, lo1 = cli.parse_bbox(c["search_bbox"])
df = pd.read_parquet(adsb.ensure_day(c["adsb_day"], cli.DATA))
w = df[(df.t > T + t0 - 300) & (df.t < T + t1 + 300) & (df.lat > la0) & (df.lat < la1) & (df.lon > lo0) & (df.lon < lo1)]
for icao, g in w.groupby("icao", observed=True):
    if len(g) < 20:
        continue
    x = np.radians(g.lon.to_numpy()) * np.cos(np.radians(g.lat.mean()))
    y = np.radians(g.lat.to_numpy())
    hd = np.unwrap(np.arctan2(np.diff(y), np.diff(x)))
    turns = abs(hd[-1] - hd[0]) / (2 * np.pi)
    if turns >= 3:
        print(f"{icao} {g.type.iloc[0]!s:5} turns {turns:4.1f} median gs {g.gs.median():5.1f} m/s  "
              f"centre {g.lat.mean():.3f},{g.lon.mean():.3f}  rows {len(g)}")
