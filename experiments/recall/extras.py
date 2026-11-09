"""Why are extra events unexplained at the time-given answer? For each extra event, every aircraft (rotorcraft
included) passing within 20 km of the time-given spot with its CPA within +-90 s of the locked time:
rmin vs the verify gate (d * 1.45), tcpa - when vs the gate (10 s), and the brightness score verify would give.
usage: python extras.py VARIANT RUN [RUN ...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import adsb, cli, search  # noqa: E402
from pinpoint.doppler import C  # noqa: E402
from pinpoint.dem import Elevation  # noqa: E402
from pinpoint.flyover import Track  # noqa: E402

from rank import LAB  # noqa: E402

RUNS = (Path(__file__).resolve().parents[2] / "runs")
OUT = Path("/Volumes/DevScratch/pinpoint/exp/recall")
types = adsb._types()
var = sys.argv[1]
elev = Elevation(cli.DATA)
for run in sys.argv[2:]:
    c = LAB[run]
    v = pd.read_parquet(OUT / var / run / ("verified.parquet" if (OUT / var / run / "verified.parquet").exists() else "candidates.parquet"))
    b = v.iloc[0]
    sp, all_evs = cli.load_spec(RUNS / run), cli.load_events(RUNS / run)
    evs, extra = cli.split_events(all_evs)
    df = pd.read_parquet(adsb.ensure_day(c["adsb_day"], cli.DATA))
    e0 = float(elev([b.lat], [b.lon])[0])
    tol = search.Tolerances()
    rel = np.arange(-70.0, 71.0, sp.t[1] - sp.t[0])
    ks = np.linspace(*tol.airspeed, tol.airspeed_steps)
    print(f"== {run} {b.lat:.4f},{b.lon:.4f} pair {b.icao1}+{b.icao2} score {b.score:.3f}")
    for j, e in enumerate(extra):
        when = b.start + e.tc
        near = df[(df.t > when - 500) & (df.t < when + 500) & (np.abs(df.lat - b.lat) < 0.2) & (np.abs(df.lon - b.lon) < 0.27)]
        rows = []
        for icao, seg in search.segments(near, min_len=10):
            tr = Track(seg.t.to_numpy(), seg.lat.to_numpy(), seg.lon.to_numpy(), seg.alt_m.to_numpy())
            t_rec, ratio, _, tcpa, rmin = tr.curve(b.lat, b.lon, e0)
            if rmin > 20000 or abs(tcpa - when) > 90:
                continue
            vr = np.interp(tcpa + rel, t_rec, C / ratio - C)
            f0s = e.f0 * (1 + np.linspace(-tol.f0_rel, tol.f0_rel, 25))
            sc, _ = search._sweep(sp, tcpa - b.start + rel, vr, np.arange(-2, 2.01, 0.5), ks, f0s, e.harmonics)
            gs = float(np.interp(tcpa, seg.t, seg.gs))
            ok = rmin <= e.d * (1 + tol.slant_rel) and abs(tcpa - when) <= tol.offset_s
            rows.append(f"    {icao} {seg.type.iloc[0]!s:5} rmin {rmin / 1000:5.1f}k (gate {e.d * 1.45 / 1000:.1f}k) "
                        f"dt {tcpa - when:+5.0f}s gs {gs:4.0f} score {sc:.2f}{' PASS' if ok else ''}")
        print(f"  extra {e.tc:.0f}s f0 {e.f0:.0f} v {e.v:.0f} d {e.d / 1000:.1f}k fit {e.score:.2f} explained by "
              f"{b.get(f'xicao{j + 1}') or '-'} ({b.get(f'x{j + 1}', 0):.2f})")
        print("\n".join(rows) if rows else "    no aircraft within 20 km / 90 s")
