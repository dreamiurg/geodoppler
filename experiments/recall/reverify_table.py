"""Compare re-verification variants (reverify_<V>.json) against heldout main: correct count, flips, truth rows."""
import json
import sys
from pathlib import Path

D = Path("/Volumes/DevScratch/pinpoint/exp/recall")
names = sys.argv[1:] or ["1", "8", "1w", "8w"]
vs = {n: json.loads((D / f"reverify_{n}.json").read_text()) for n in names if (D / f"reverify_{n}.json").exists()}
runs = sorted(set().union(*vs.values()))
print(f"{'run':12} " + " ".join(f"{n:>22}" for n in vs))
for r in runs:
    cells = []
    for n, v in vs.items():
        x = v.get(r)
        cells.append("-" if x is None else f"{'OK ' if x['correct'] else '   '}{x['err']:+8.0f} {x['score']:5.2f} t#{x['truth_row'] or '-'}")
    print(f"{r:12} " + " ".join(f"{c:>22}" for c in cells))
for n, v in vs.items():
    print(f"{n}: {sum(x['correct'] for x in v.values())} correct of {len(v)}")
