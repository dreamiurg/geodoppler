"""Event times as wall-clock UTC (if the video starts at go-live), to spot streams that share audio or repeat."""
import datetime as dt
import json
import sys
from pathlib import Path

from rank import LAB

RUNS = (Path(__file__).resolve().parents[2] / "runs")

for r in sys.argv[1:]:
    c = LAB[r]
    rel = dt.datetime.fromisoformat(c["release_timestamp_utc"])
    ev = json.loads((RUNS / r / "events.json").read_text())
    print(f"{r:12} go-live {c['release_timestamp_utc']} slice@{c['section_start_s']:.0f}s dur {c['duration_s']}  "
          + " ".join((rel + dt.timedelta(seconds=c["section_start_s"] + e["tc"])).strftime("%H:%M:%S") for e in ev))
