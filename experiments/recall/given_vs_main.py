"""Where is the time-given winner's aircraft pair in the unconstrained (heldout main) search?
  in-main: same pair present; rank among combinations (verified), its best point's score/start error vs winner."""
import sys
from pathlib import Path

import pandas as pd

from rank import LAB, truth_unix

MAIN = Path("/Volumes/DevScratch/pinpoint/exp/heldout/main")
GIVEN = Path("/Volumes/DevScratch/pinpoint/exp/recall") / (sys.argv[1] if len(sys.argv) > 1 else "start60")
for run, c in LAB.items():
    g = GIVEN / run / "verified.parquet"
    g = g if g.exists() else GIVEN / run / "candidates.parquet"
    if not g.exists():
        continue
    t = truth_unix(c)
    gw = pd.read_parquet(g).iloc[0]
    pair = (gw.icao1, gw.icao2)
    line = f"{run:12} given {gw.icao1}+{gw.icao2} {gw.score:.3f} err {gw.start - t:+.0f} | "
    for f in ("candidates", "verified"):
        p = MAIN / run / f"{f}.parquet"
        if not p.exists():
            line += f"{f[:4]}: - "
            continue
        m = pd.read_parquet(p).reset_index(drop=True)
        combos = m.drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
        hit = combos[(combos.icao1 == pair[0]) & (combos.icao2 == pair[1])]
        if hit.empty:
            line += f"{f[:4]}: pair absent ({len(combos)} combos)  "
        else:
            r = hit.iloc[0]
            line += f"{f[:4]}: combo#{hit.index[0] + 1}/{len(combos)} {r.score:.3f} vs {m.score[0]:.3f} err {r.start - t:+.0f}  "
    print(line)
