"""Time-given locate: `locate --start <release_timestamp> --start-window W --decoys 5` on held-out cases.
Tells whether the search CAN find a match when the start is known, and how it scores against decoy days
(decoys use the same time of day). Run dirs are copied (heldout.copy_case); runs/ is never written.

  python experiments/recall/given.py VARIANT [--window 60] [--bbox conus] [--only RUN ...]
Output: /Volumes/DevScratch/pinpoint/exp/recall/<VARIANT>/<run>/{result.json,locate.log,...}
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "heldout"))
import heldout  # noqa: E402

OUT = Path("/Volumes/DevScratch/pinpoint/exp/recall")
ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--window", type=float, default=60)
    ap.add_argument("--bbox")
    ap.add_argument("--decoys", type=int, default=5)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    snap = OUT / a.variant / "src"
    if not snap.exists():
        shutil.copytree(ROOT / "src", snap, ignore=shutil.ignore_patterns("__pycache__"))
    for c in heldout.cases(a.only):
        dst = OUT / a.variant / c["run"]
        if (dst / "result.json").exists() or (dst / "failed").exists():
            continue
        heldout.copy_case(c["run"], dst)
        start = c["release_timestamp_utc"].replace(" ", "T")
        code = "from pinpoint.cli import main; main()"
        cmd = [*heldout.LOCK, "env", f"PYTHONPATH={snap}", str(heldout.PY), "-c", code, "locate", str(dst),
               "--date", c["adsb_day"], "--bbox", a.bbox or c["search_bbox"], "--start", start,
               "--start-window", str(a.window), "--decoys", str(a.decoys), "--no-open"]
        t = time.time()
        with open(dst / "locate.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            (dst / "failed").write_text(f"exit {rc}\n")
        print(f"{a.variant} {c['run']}: {'exit ' + str(rc) if rc else 'ok'} in {time.time() - t:.0f} s", file=sys.stderr, flush=True)
