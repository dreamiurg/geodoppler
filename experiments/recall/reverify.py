"""Verification on more than the best point per combination, from heldout main's candidates.parquet (no new search).
points=1 must reproduce main's verified.parquet; points=P verifies up to P refined points of each of the top 300
combinations. Writes /Volumes/DevScratch/pinpoint/exp/recall/reverify_P.json.
usage (under the CPU lock): python reverify.py P[w] [RUN ...]   (w: verify with whole tracks, see verify_extra)"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import cli, search  # noqa: E402
from pinpoint.dem import Elevation  # noqa: E402

from rank import LAB, truth_unix  # noqa: E402

MAIN = Path("/Volumes/DevScratch/pinpoint/exp/heldout/main")
RUNS = (Path(__file__).resolve().parents[2] / "runs")

if __name__ == "__main__":
    P, whole = int(sys.argv[1].rstrip("w")), sys.argv[1].endswith("w")
    only = sys.argv[2:]
    out_f = Path(f"/Volumes/DevScratch/pinpoint/exp/recall/reverify_{sys.argv[1]}.json")
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    elev = Elevation(cli.DATA)
    days = {}
    for run, c in LAB.items():
        if (only and run not in only) or run in res or not (MAIN / run / "candidates.parquet").exists():
            continue
        sp, all_evs = cli.load_spec(RUNS / run), cli.load_events(RUNS / run)
        evs, extra = cli.split_events(all_evs)
        cands = pd.read_parquet(MAIN / run / "candidates.parquet")
        if c["adsb_day"] not in days:
            days = {c["adsb_day"]: cli.load_days([c["adsb_day"]])}
        df = days[c["adsb_day"]]
        v = search.verify_extra(sp, cands, extra, evs, df, search.Tolerances(), log=lambda m: None, elev=elev, points=P, whole=whole) \
            if extra else cands
        if P == 1 and not whole and extra:
            ref = pd.read_parquet(MAIN / run / "verified.parquet")
            assert (ref.score.to_numpy() == v.score.to_numpy()).all(), run
        t = truth_unix(c)
        b = v.iloc[0]
        ok = (v.start - t).abs() <= 60
        res[run] = {"score": float(b.score), "err": float(b.start - t), "correct": bool(abs(b.start - t) <= 60),
                    "lat": float(b.lat), "lon": float(b.lon), "icao": f"{b.icao1}+{b.icao2}",
                    "truth_row": int(ok.to_numpy().argmax()) + 1 if ok.any() else None,
                    "truth_score": float(v.score[ok].iloc[0]) if ok.any() else None}
        print(run, res[run], flush=True)
        out_f.write_text(json.dumps(res, indent=1))
