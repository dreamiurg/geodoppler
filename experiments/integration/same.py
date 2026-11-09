"""Bit-identity of a variant case against the saved runs/<case> outputs: same.py VARIANT CASE"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

v, case = sys.argv[1:3]
a, b = Path("/Users/dreamiurg/src/dreamiurg/pinpoint/runs") / case, Path("/Volumes/DevScratch/pinpoint/exp/integration") / v / case
for f in ("candidates.parquet", "verified.parquet"):
    if not (a / f).exists():
        continue
    x, y = pd.read_parquet(a / f), pd.read_parquet(b / f)
    num = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    same = x.shape == y.shape and list(x.columns) == list(y.columns)
    d = float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if same else float("nan")
    eq = same and all((x[c].astype(str) == y[c].astype(str)).all() for c in x.columns if c not in num)
    print(f"{f}: shapes {x.shape} {y.shape}, max num diff {d}, other cols equal {eq}")
ra, rb = json.loads((a / "result.json").read_text()), json.loads((b / "result.json").read_text())
print("result.json identical:", ra == rb, [k for k in ra if ra[k] != rb.get(k)])
