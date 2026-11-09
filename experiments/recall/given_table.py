"""Table for given.py results next to the unconstrained (heldout main) answer.
usage: python given_table.py VARIANT [BASE]   BASE default /Volumes/DevScratch/pinpoint/exp/heldout/main"""
import json
import sys
from pathlib import Path

import pandas as pd

from rank import LAB, truth_unix

OUT = Path("/Volumes/DevScratch/pinpoint/exp/recall")
var = OUT / sys.argv[1]
base = Path(sys.argv[2] if len(sys.argv) > 2 else "/Volumes/DevScratch/pinpoint/exp/heldout/main")
rows = []
for run, c in LAB.items():
    d = var / run
    if not d.exists():
        continue
    row = {"run": run}
    b = base / run / "result.json"
    if b.exists():
        r = json.loads(b.read_text())
        err = pd.Timestamp(r["recording_start_utc"]).timestamp() - pd.Timestamp(c["release_timestamp_utc"]).timestamp()
        row.update(main_ok=abs(err) <= 60, main_score=r["score"])
    if (d / "result.json").exists():
        r = json.loads((d / "result.json").read_text())
        err = pd.Timestamp(r["recording_start_utc"]).timestamp() - pd.Timestamp(c["release_timestamp_utc"]).timestamp()
        ds = r["decoy_scores"] or {}
        row.update(given=round(r["score"], 3), err=round(err), k="/".join(f"{k:.2f}" for k in r["doppler_scale"]),
                   decoy_max=max(ds.values(), default=0.0), n_decoy_match=sum(v > 0 for v in ds.values()),
                   ratio=r["decoy_ratio"], conf=r["confidence"], lat=r["lat"], lon=r["lon"],
                   explained=f"{len(r['aircraft'])}/{len(r['aircraft']) + len(r['events_unexplained'])}",
                   icao="+".join(a["icao"] for a in r["aircraft"]))
    else:
        row["given"] = "no match" if (d / "failed").exists() else "missing"
    rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df.to_string(index=False))
df.to_json(var / "table.json", orient="records", indent=1)
