"""exit 70 check: rows the unblocked offset_filter join would build on LYoUoV2jRL8's crashing tile.
usage (under the CPU lock): python joinsize.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import cli, search  # noqa: E402
from pinpoint.dem import Elevation  # noqa: E402

if __name__ == "__main__":
    run = (Path(__file__).resolve().parents[2] / "runs/LYoUoV2jRL8")
    sp, evs = cli.load_spec(run), cli.load_events(run)
    evs, _ = cli.split_events(evs)
    df = cli.load_days(["2026-11-07"])
    elev = Elevation(cli.DATA)
    tb = next(cli.tiles(cli.parse_bbox("28.5,-81.6,150")))
    pad = 0.15
    sub = df[(df.lat > tb[0] - pad) & (df.lat < tb[2] + pad) & (df.lon > tb[1] - pad * 1.4) & (df.lon < tb[3] + pad * 1.4)]
    pts = search.Points.grid(tb, 200.0, elev)
    tol = search.Tolerances()
    with search.pool(sp, evs, elev) as ex:
        c = [search.candidate_passes(sub, e, pts, tol, elev, None, ex, (tuple(tb), 200.0, True)) for e in evs]
    n1 = np.bincount(c[0].g, minlength=len(pts)).astype(np.int64)
    n2 = np.bincount(c[1].g, minlength=len(pts)).astype(np.int64)
    rows = int((n1 * n2).sum())
    print(f"tile {tb}: event rows {len(c[0]):,} / {len(c[1]):,}; unblocked join rows {rows:,} "
          f"(~{rows * 11 * 8 / 1e9:.0f} GB at 11 numeric columns, before pandas overhead)")
    hits = search.offset_filter(c, evs, tol)
    print(f"blocked offset_filter: {len(hits):,} hits at {hits.g.nunique():,} points")
