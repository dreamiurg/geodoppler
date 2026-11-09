"""Where does the truth-time match die? For each case, run search.locate's front half (candidate passes, offset
filter, silence filter) with the start fixed to go-live +- 60 s, under progressively relaxed gates:

  base       default Tolerances, rotorcraft dropped (what `locate --start` does)
  +rotor     rotorcraft kept
  +speed     speed gate off (any ground speed)
  +slant     slant gate 0.1-3x the fitted d (default 0.55-1.45x)
  +offset    CPA-offset gate +-30 s (default 10)
  +same      the same aircraft may explain both events (offset_filter requires different icao)

Jets are not in props.parquet at all (dropped when the day was reduced), so they cannot be tested here.
Counts are aircraft pairs (combinations) surviving offset+silence, summed over tiles.
usage: python gates.py RUN [RUN ...]   (heavy: run under lockf -k /tmp/pinpoint-cpu.lock)
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import adsb, cli, search  # noqa: E402
from pinpoint.dem import Elevation  # noqa: E402

from rank import LAB, truth_unix  # noqa: E402

RUNS = (Path(__file__).resolve().parents[2] / "runs")


def offset_any(cands, events, tol, same):
    a = cands[0].rename(columns={c: f"{c}0" for c in ["t", "d", "gs", "icao"]})
    b = cands[1].rename(columns={c: f"{c}1" for c in ["t", "d", "gs", "icao"]})
    m = a.merge(b, on="g")
    want = events[1].tc - events[0].tc
    ok = np.abs((m.t1 - m.t0) - want) <= tol.offset_s
    if not same:
        ok &= m.icao1 != m.icao0
    else:  # same aircraft: its two CPAs must be distinct passes
        ok &= (m.icao1 != m.icao0) | (np.abs(m.t1 - m.t0) > 120)
    return m[ok].reset_index(drop=True)


def stage(sp, evs, df, bbox, tol, start, elev, same=False):
    pairs, n1, n2, off = set(), set(), set(), 0
    for tb in cli.tiles(bbox):
        pad = 0.15
        sub = df[(df.lat > tb[0] - pad) & (df.lat < tb[2] + pad) & (df.lon > tb[1] - pad * 1.4) & (df.lon < tb[3] + pad * 1.4)]
        pts = search.Points.grid(tb, 200.0, elev)
        c = [search.candidate_passes(sub, e, pts, tol, elev, start) for e in evs]
        n1 |= set(c[0].icao)
        n2 |= set(c[1].icao)
        hits = offset_any(c, evs, tol, same)
        off += hits.g.nunique()
        if len(hits):
            hits = search.silence_filter(hits, sub, evs, pts, tol, elev)
        pairs |= set(zip(hits.icao0, hits.icao1))
    return len(n1), len(n2), off, sorted(pairs)


def main(run):
    c = LAB[run]
    d = RUNS / run
    sp, all_evs = cli.load_spec(d), cli.load_events(d)
    evs, _ = cli.split_events(all_evs)
    t0 = truth_unix(c)
    start = (t0 - 60, t0 + 60)
    raw = pd.concat([pd.read_parquet(adsb.ensure_day(x, cli.DATA)) for x in [c["adsb_day"]]], ignore_index=True)
    raw = raw.astype({"icao": "category", "type": "category"})
    norotor = cli.load_days([c["adsb_day"]])
    elev = Elevation(cli.DATA)
    bbox = cli.parse_bbox(c["search_bbox"])
    t = search.Tolerances()
    steps = [("base", norotor, t, False), ("+rotor", raw, t, False),
             ("+speed", raw, replace(t, speed=1e4), False),
             ("+slant", raw, replace(t, speed=1e4, slant_rel=0.9), False),
             ("+offset", raw, replace(t, speed=1e4, slant_rel=0.9, offset_s=30.0), False),
             ("+same", raw, replace(t, speed=1e4, slant_rel=0.9, offset_s=30.0), True)]
    out = {}
    for name, df, tol, same in steps:
        n1, n2, off, pairs = stage(sp, evs, df, bbox, tol, start, elev, same)
        out[name] = {"ev1_aircraft": n1, "ev2_aircraft": n2, "offset_points": off, "pairs": len(pairs),
                     "examples": [f"{a}+{b}" for a, b in pairs[:5]]}
        print(f"{run} {name:8} ev1 {n1:3} ev2 {n2:3} offset-points {off:6} pairs after silence {len(pairs):3} {pairs[:3]}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    f = Path("/Volumes/DevScratch/pinpoint/exp/recall/gates.json")
    if f.exists():
        res = json.loads(f.read_text())
    for r in sys.argv[1:]:
        if r in res:
            continue
        res[r] = main(r)
        f.write_text(json.dumps(res, indent=1))
