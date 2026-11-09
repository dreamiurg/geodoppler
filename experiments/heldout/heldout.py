"""Held-out eval on experiments/dataset/labeled.json (livestreams; truth = YouTube go-live time).

  python experiments/heldout/heldout.py run VARIANT [--only RUN ...]     snapshot src/, locate each cached case
  python experiments/heldout/heldout.py decoys VARIANT BASE [--n 5]      decoy searches only, reusing BASE's true day
  python experiments/heldout/heldout.py score VARIANT [BASE]             per-case table (+ decoy ratio if BASE)

Same mechanics as validation/eval.py: src/ is snapshotted to EXP/<variant>/src, each run dir's meta/spec/events/
level_db are copied so runs/ stays clean, and every locate runs under the shared CPU lock. Resumable.
CORRECT = found recording start within 60 s of release_timestamp_utc (no spots are known for these cases).
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "validation"))
from eval import km  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(__file__).resolve().parents[2]
RUNS = MAIN / "runs"
LABELED = MAIN / "experiments" / "dataset" / "labeled.json"
EXP = Path("/Volumes/DevScratch/pinpoint/exp/heldout")
PY = MAIN / ".venv" / "bin" / "python"
LOCK = ["lockf", "-k", "/tmp/pinpoint-cpu.lock"]


def cases(only=None):
    rows = [c for c in json.loads(LABELED.read_text()) if c["adsb_cached"] and (RUNS / c["run"] / "spec.npz").exists()]
    return [c for c in rows if not only or c["run"] in only]


def snapshot(variant):
    snap = EXP / variant / "src"
    if not snap.exists():
        shutil.copytree(ROOT / "src", snap, ignore=shutil.ignore_patterns("__pycache__"))
    return snap


def copy_case(case, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "spec.npz", "events.json", "level_db"):
        if not (dst / f).exists():
            shutil.copy2(RUNS / case / f, dst / f)
    for f in ("audio.webm", "overview"):
        if (RUNS / case / f).exists() and not (dst / f).exists():
            (dst / f).symlink_to((RUNS / case / f).resolve())


def run(variant, only):
    snap = snapshot(variant)
    for c in cases(only):
        dst = EXP / variant / c["run"]
        if (dst / "result.json").exists() or (dst / "failed").exists():
            continue
        copy_case(c["run"], dst)
        if not (dst / "events.json").exists():
            sys.exit(f"{c['run']}: no events.json (fit with `pinpoint events` first)")
        code = "import sys, pinpoint; print('code:', pinpoint.__file__, file=sys.stderr); from pinpoint.cli import main; main()"
        cmd = [*LOCK, "env", f"PYTHONPATH={snap}", str(PY), "-c", code,
               "locate", str(dst), "--date", c["adsb_day"], "--bbox", c["search_bbox"], "--no-open"]
        t = time.time()
        with open(dst / "locate.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            (dst / "failed").write_text(f"exit {rc}\n")
        print(f"{variant} {c['run']}: {'exit ' + str(rc) if rc else 'ok'} in {time.time() - t:.0f} s", file=sys.stderr, flush=True)


DECOY_CODE = """
import json, sys, time
from pathlib import Path
import pandas as pd
from pinpoint import cli, search
from pinpoint.dem import Elevation
run, date, bbox, n = Path(sys.argv[1]), sys.argv[2], cli.parse_bbox(sys.argv[3]), int(sys.argv[4])
sp, all_evs = cli.load_spec(run), cli.load_events(run)
evs, extra = cli.split_events(all_evs)
elev = Elevation(cli.DATA)
out = {}
with search.pool(sp, evs, elev) as ex:
    for d in cli.decoy_days(date, n):
        t = time.time()
        c = cli.search_day(sp, evs, extra, cli.load_days([d]), bbox, 200.0, None, elev, ex)
        b = None if c is None else c.iloc[0]
        out[d] = {"score": 0.0} if b is None else {"score": float(b.score), "lat": float(b.lat), "lon": float(b.lon),
                                                   "start": float(b.start), "seconds": round(time.time() - t, 1)}
        cli.log(f"decoy {d}: {out[d]['score']:.3f} in {time.time() - t:.0f} s")
(run / "decoys.json").write_text(json.dumps(out, indent=1))
"""


def decoys(variant, base, n):
    """Decoy searches only, through the variant's own search_day/decoy_days; the true day comes from BASE."""
    snap = snapshot(variant)
    for c in cases():
        b = EXP / base / c["run"]
        if not (b / "result.json").exists() or len(json.loads((b / "events.json").read_text())) < 3:
            continue
        dst = EXP / variant / c["run"]
        if (dst / "decoys.json").exists() or (dst / "failed").exists():
            continue
        copy_case(c["run"], dst)
        cmd = [*LOCK, "env", f"PYTHONPATH={snap}", str(PY), "-c", DECOY_CODE, str(dst), c["adsb_day"], c["search_bbox"], str(n)]
        t = time.time()
        with open(dst / "decoys.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            (dst / "failed").write_text(f"exit {rc}\n")
        print(f"{variant} {c['run']}: {'exit ' + str(rc) if rc else 'ok'} in {time.time() - t:.0f} s", file=sys.stderr, flush=True)


def log_seconds(log):
    ts = [pd.Timestamp(x[1:9]) for x in log.read_text().splitlines() if x[:1] == "[" and x[9:10] == "]"] if log.exists() else []
    return (ts[-1] - ts[0]).total_seconds() % 86400 if ts else None


def score(variant, base=None):
    rows = []
    for c in cases():
        d = EXP / (base or variant) / c["run"]
        row = {"run": c["run"], "day": c["adsb_day"], "n_events": len(json.loads((RUNS / c["run"] / "events.json").read_text()))}
        if not (d / "result.json").exists():
            rows.append({**row, "verdict": "none" if (d / "failed").exists() else "missing"})
            continue
        r = json.loads((d / "result.json").read_text())
        err = (pd.Timestamp(r["recording_start_utc"]) - pd.Timestamp(c["release_timestamp_utc"])).total_seconds()
        row.update(verdict="CORRECT" if abs(err) <= 60 else "WRONG", err_s=err, conf=r["confidence"], margin=r["margin"],
                   k=r.get("doppler_scale"), score=r["score"], lat=r["lat"], lon=r["lon"],
                   seconds=log_seconds(d / "locate.log"))
        dj = EXP / variant / c["run"] / "decoys.json"
        if base and dj.exists():
            full = json.loads(dj.read_text())
            ds = {d: v["score"] for d, v in full.items()}
            mx = max(ds.values(), default=0.0)
            # schedule-leak check: decoy best spot vs this case's answer (no true spots for these cases)
            row["decoy_km"] = {d: round(km((r["lat"], r["lon"]), (v["lat"], v["lon"])), 1) for d, v in full.items() if "lat" in v}
            row.update(decoys=ds, ratio=None if mx == 0 else r["score"] / mx,
                       decoy_seconds=log_seconds(EXP / variant / c["run"] / "decoys.log"))
            kflag = any(abs(k - 1) > 0.08 for k in r.get("doppler_scale") or [])
            row["conf_decoy"] = "high" if (row["ratio"] is None or row["ratio"] >= 1.2) and row["n_events"] >= 3 and not kflag else "low"
        rows.append(row)
    print(f"{'run':12} {'day':10} {'ev':>3} {'verdict':7} {'err_s':>8} {'conf':4} {'margin':>6} {'k':11} {'score':>6}"
          + ("  ratio  decoys(max)        conf_d" if base else ""))
    for r in rows:
        if "conf" not in r:
            print(f"{r['run']:12} {r['day']:10} {r['n_events']:>3} {r['verdict']}")
            continue
        k = "/".join(f"{x:.2f}" for x in r["k"] or [])
        line = (f"{r['run']:12} {r['day']:10} {r['n_events']:>3} {r['verdict']:7} {r['err_s']:>8.0f} {r['conf']:4} "
                f"{r['margin'] if r['margin'] is not None else float('nan'):>6.3f} {k:11} {r['score']:>6.3f}")
        if base and "decoys" in r:
            line += f"  {r['ratio'] if r['ratio'] is not None else float('inf'):5.2f}  {max(r['decoys'].values()):6.3f} ({len(r['decoys'])})        {r['conf_decoy']}"
        print(line)
    n = lambda f: sum(1 for r in rows if f(r))  # noqa: E731
    print(f"\n{len(rows)} cases: correct {n(lambda r: r['verdict'] == 'CORRECT')}, wrong {n(lambda r: r['verdict'] == 'WRONG')}, "
          f"none {n(lambda r: r['verdict'] in ('none', 'missing'))}")
    for v in ("CORRECT", "WRONG"):
        print(f"  {v}: high {n(lambda r: r['verdict'] == v and r['conf'] == 'high')}, low {n(lambda r: r['verdict'] == v and r['conf'] == 'low')}"
              + (f" | decoy rule: high {n(lambda r: r['verdict'] == v and r.get('conf_decoy') == 'high')}, "
                 f"low {n(lambda r: r['verdict'] == v and r.get('conf_decoy') == 'low')}" if base else ""))
    (EXP / variant / f"score{'_' + base if base else ''}.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("variant"); r.add_argument("--only", nargs="*")
    d = sub.add_parser("decoys"); d.add_argument("variant"); d.add_argument("base"); d.add_argument("--n", type=int, default=5)
    s = sub.add_parser("score"); s.add_argument("variant"); s.add_argument("base", nargs="?")
    a = ap.parse_args()
    {"run": lambda: run(a.variant, a.only), "decoys": lambda: decoys(a.variant, a.base, a.n),
     "score": lambda: score(a.variant, a.base)}[a.cmd]()
