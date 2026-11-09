"""How meaningful is "a candidate within 60 s of the truth at combo rank R"? Same lookup at 40 fake truths
(truth +- 10..200 min, step 10 min): the fraction of fake truths that get a candidate with combo rank <= R.
usage: python null_rank.py EXPDIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rank import LAB, truth_unix

SHIFTS = [s * 600 for s in range(-20, 21) if s]


def best_combo_rank(x, t):
    ok = np.flatnonzero((x.start - t).abs().to_numpy() <= 60)
    if not len(ok):
        return None
    i = ok[0]
    return x.iloc[:i + 1].drop_duplicates(["icao1", "icao2"]).shape[0]


if __name__ == "__main__":
    exp = Path(sys.argv[1])
    for d in sorted(p for p in exp.iterdir() if p.is_dir() and p.name in LAB):
        f = d / "verified.parquet" if (d / "verified.parquet").exists() else d / "candidates.parquet"
        if not f.exists():
            continue
        x = pd.read_parquet(f).reset_index(drop=True)
        t = truth_unix(LAB[d.name])
        r = best_combo_rank(x, t)
        null = [best_combo_rank(x, t + s) for s in SHIFTS]
        hit = [n for n in null if n is not None]
        p = None if r is None else sum(n <= r for n in hit) / len(SHIFTS)
        print(f"{d.name:12} {f.stem[:4]} truth combo#{r}  fake truths with a hit: {len(hit)}/{len(SHIFTS)}"
              f"  P(fake rank <= truth rank) = {p if p is None else round(p, 2)}  fake ranks {sorted(hit)[:8]}")
