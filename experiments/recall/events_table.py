"""Per case: the fitted events (search pair marked *), from runs/<id>/events.json."""
import json
from pathlib import Path

from rank import LAB

RUNS = Path("/Users/dreamiurg/src/dreamiurg/pinpoint/runs")
for run, c in LAB.items():
    f = RUNS / run / "events.json"
    if not c["adsb_cached"] or not f.exists():
        continue
    ev = json.loads(f.read_text())
    top = sorted(range(len(ev)), key=lambda i: -ev[i]["score"])[:2]
    s = "  ".join(f"{'*' if i in top else ' '}{e['tc']:.0f}s f0={e['f0']:.0f} v={e['v']:.0f} d={e['d'] / 1000:.1f}k "
                  f"s={e['score']:.2f} h={'+'.join(map(str, e['harmonics']))}" for i, e in enumerate(ev))
    print(f"{run:12} {s}")
