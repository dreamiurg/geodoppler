"""Score a code variant on every labeled recording.

  python validation/eval.py run VARIANT [--only CASE ...]   snapshot src/, locate each case (resumable)
  python validation/eval.py score VARIANT [VARIANT ...]     per-case table + bucket counts, side by side
                                                            (VARIANT:margin = confidence from margin alone)

`run` copies src/pinpoint to EXP/<variant>/src (so editing src/ mid-run changes nothing), copies each case's
run dir (meta, spec, events, level_db; audio/overview symlinked) to EXP/<variant>/<case>/ and runs
`pinpoint locate` there under the shared CPU lock. A case with a result.json (or a recorded failure) is skipped.
Verdict: CORRECT = start within 60 s of the true go-live and, where the spot is known, within 3 km of it.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("/Users/dreamiurg/src/dreamiurg/pinpoint/runs")
EXP = Path("/Volumes/DevScratch/pinpoint/exp/integration")
PY = ROOT / ".venv" / "bin" / "python"

OHIO, KY, HURON = (41.11, -81.57), (38.83, -84.77), (41.38924, -82.56009)
# case, ADS-B day, bbox, true video start (UTC) or None, true spot or None
CASES = [
    ("U1bQZeogcsU", "2026-09-30", "41.3,-81.4,150", "2026-09-30 13:14:10", OHIO),
    ("J8yvJJS7qTU", "2026-10-21", "38.9,-84.6,120", "2026-10-21 14:53:58", KY),
    ("UsyJgZpSuus", "2026-10-27", "41.3,-81.4,150", "2026-10-27 16:55:38", OHIO),
    ("GPLUdm_Q9NQ", "2026-10-28", "41.3,-81.4,150", None, OHIO),  # no go-live time: place-only verdict
    ("u8qudttB8PI", "2026-10-18", "41.38924,-82.56009,100", "2026-10-18 11:46:17", HURON),
    # README/demo: found spot "matches the video (truth given to 0.01 deg)" -> reference, not independent truth
    ("4CxVxuMVkMc", "2026-09-08", "40.9,-73.5,120", "2026-09-08 20:17:28", (40.87, -73.52)),
    ("y-xpMDcnQvw", "2026-10-19", "41.3,-81.4,150", "2026-10-19 13:51:18", OHIO),
    ("y-xpMDcnQvw_akron", "2026-10-19", "41.11,-81.57,15", "2026-10-19 13:51:18", OHIO),
    ("0a7boP2vrtY", "2026-10-27", "44.0,-89.5,200", "2026-10-27 18:08:46", None),
    ("1Rh6h15nY70", "2026-10-15", "38.9,-84.6,120", "2026-10-15 14:02:15", KY),
    ("1Rh6h15nY70_nky", "2026-10-15", "38.83,-84.77,15", "2026-10-15 14:02:15", KY),
    ("AKgS6TS1X0M", "2026-03-27", "43.5,-84.5,250", "2026-03-27 13:09:28", None),
    ("BKWni6FXr2o", "2026-11-07", "36.3,-95.5,150", "2026-11-07 17:32:06", None),
    ("LdusZTziHeI", "2026-10-08", "38.9,-84.6,120", "2026-10-08 14:32:19", KY),
    ("TX52YFJBThg", "2026-09-10", "38.9,-84.6,120", "2026-09-10 14:18:11", KY),
    ("YQW34ztHbkE", "2026-02-01", "41.3,-81.4,150", "2026-02-01 14:21:03", OHIO),
    ("ZJ7e-vtV-Qs", "2026-03-06", "35.5,-79.5,300", "2026-03-06 11:05:26", None),
    ("ZQBMsh72FwM", "2026-09-06", "38.9,-84.6,120", "2026-09-06 14:02:57", KY),
    ("_1Xk8p7F5ko", "2026-10-02", "44.5,-89.5,250", "2026-10-02 11:00:12", None),
    ("kOhLIeejc2A", "2026-10-08", "41.3,-81.4,150", "2026-10-08 12:02:02", OHIO),
    ("kOhLIeejc2A_akron", "2026-10-08", "41.11,-81.57,15", "2026-10-08 12:02:02", OHIO),
    ("ru5b7rt969M", "2026-03-06", "34.6,-84.2,150", "2026-03-06 00:07:31", None),
    ("shhydRMX6e0", "2026-02-28", "35.5,-79.5,300", "2026-02-28 13:24:31", None),
]


def km(a, b):
    p = math.pi / 180
    h = math.sin((b[0] - a[0]) * p / 2) ** 2 + math.cos(a[0] * p) * math.cos(b[0] * p) * math.sin((b[1] - a[1]) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def run(variant, only):
    vdir = EXP / variant
    snap = vdir / "src"
    if not snap.exists():
        shutil.copytree(ROOT / "src", snap, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        print(f"reusing snapshot {snap}", file=sys.stderr)
    for case, day, bbox, _, _ in CASES:
        if only and case not in only:
            continue
        dst = vdir / case
        if (dst / "result.json").exists() or (dst / "failed").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("meta.json", "spec.npz", "events.json", "level_db"):
            shutil.copy2(RUNS / case / f, dst / f)
        for f in ("audio.webm", "overview"):
            if not (dst / f).exists():
                (dst / f).symlink_to((RUNS / case / f).resolve())
        code = "import sys, pinpoint; print('code:', pinpoint.__file__, file=sys.stderr); from pinpoint.cli import main; main()"
        cmd = ["lockf", "-k", "/tmp/pinpoint-cpu.lock", "env", f"PYTHONPATH={snap}", str(PY), "-c", code,
               "locate", str(dst), "--date", day, "--bbox", bbox, "--no-open"]
        t = time.time()
        with open(dst / "locate.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        (dst / "seconds").write_text(f"{time.time() - t:.1f}\n")
        if rc:
            (dst / "failed").write_text(f"exit {rc}\n")
        print(f"{variant} {case}: {'exit ' + str(rc) if rc else 'ok'} in {time.time() - t:.0f} s", file=sys.stderr, flush=True)


def locate_seconds(log):
    """First to last log timestamp: locate time without the wait for the CPU lock."""
    ts = [pd.Timestamp(line[1:9]) for line in log.read_text().splitlines() if line[:1] == "[" and line[9:10] == "]"]
    s = (ts[-1] - ts[0]).total_seconds() if ts else None
    return None if s is None else s % 86400


def score_case(vdir, case, true_start, spot, margin_only=False):
    d = vdir / case
    row = {"case": case}
    if not (d / "result.json").exists():
        return {**row, "verdict": "none" if (d / "failed").exists() else "missing", "bucket": "none"}
    r = json.loads((d / "result.json").read_text())
    err_s = None if true_start is None else abs((pd.Timestamp(r["recording_start_utc"]) - pd.Timestamp(true_start)).total_seconds())
    err_km = None if spot is None else km((r["lat"], r["lon"]), spot)
    ok = (err_s is None or err_s <= 60) and (err_km is None or err_km <= 3)
    # margin_only: confidence as before the Doppler-scale flag (same search, so same margins)
    high = (r["margin"] or 0) >= 0.15 if margin_only else r["confidence"] == "high"
    return {**row, "err_s": err_s, "err_km": err_km, "margin": r["margin"], "conf": r["confidence"],
            "k": r.get("doppler_scale"), "types": "+".join(a["type"] or "?" for a in r["aircraft"]),
            "verdict": "CORRECT" if ok else "WRONG",
            "bucket": ("correct" if ok else "wrong") + ("-high" if high else "-low"),
            "seconds": locate_seconds(d / "locate.log")}


BUCKETS = ["correct-high", "correct-low", "wrong-high", "wrong-low", "none"]


def fmt(s):
    if s["bucket"] == "none":
        return f"{s['verdict']:>28}"
    e_km = "-" if s["err_km"] is None else f"{s['err_km']:.1f}km"
    e_s = "-" if s["err_s"] is None else f"{s['err_s']:.0f}s"
    m = "-" if s["margin"] is None else f"{s['margin']:.3f}"
    return f"{s['bucket']:>12} {e_km:>8} {e_s:>7} {m:>6}"


def score(variants):
    rows = {v: [score_case(EXP / v.split(":")[0], c, ts, sp, v.endswith(":margin")) for c, _, _, ts, sp in CASES]
            for v in variants}
    print(f"{'case':18} " + " | ".join(f"{v[:36]:^36}" for v in variants))
    for i, (c, *_) in enumerate(CASES):
        print(f"{c:18} " + " | ".join(fmt(rows[v][i]) for v in variants))
    print()
    for v in variants:
        n = {b: sum(s["bucket"] == b for s in rows[v]) for b in BUCKETS}
        secs = sum(s.get("seconds") or 0 for s in rows[v])
        print(f"{v:28} " + "  ".join(f"{b} {n[b]}" for b in BUCKETS) + f"   ({secs:.0f} s locate)")
        (EXP / v.split(":")[0] / f"score{'_margin' if v.endswith(':margin') else ''}.json").write_text(json.dumps(rows[v], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("variant"); r.add_argument("--only", nargs="*")
    s = sub.add_parser("score"); s.add_argument("variants", nargs="+")
    a = ap.parse_args()
    run(a.variant, a.only) if a.cmd == "run" else score(a.variants)
