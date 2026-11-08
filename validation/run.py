"""Run pinpoint over validation/candidates.json and score against known locations.

  uv run python validation/run.py [--radius-km 100] [--only ID ...]

Each video is searched within --radius-km of its true location (the "you know the city/region" case),
on its known UTC date, without using its known start time. Results: validation/results.json + results.md.
"""

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def km(a, b, c, d):
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def sh(*args, timeout=None):
    r = subprocess.run(["uv", "run", "pinpoint", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def daylight_section(c, minutes=120, local_hour=10.5):
    """Seconds into a long stream where local mid-morning starts (props fly by day). Returns (offset_s, start_dt)."""
    if not c.get("start_utc"):
        return 0.0, None
    start = datetime.fromisoformat(c["start_utc"].replace("Z", "+00:00"))
    want_utc = (local_hour - c["lon"] / 15) % 24  # solar local time is close enough
    for d in range(0, 2):
        t = start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=d, hours=want_utc)
        off = (t - start).total_seconds()
        if 0 <= off <= (c.get("duration_s") or 0) - minutes * 60:
            return off, t
    return 0.0, start


def run_one(c, radius):
    rec = {k: c.get(k) for k in ("id", "url", "title", "date_utc", "start_utc", "lat", "lon")}
    t = time.time()
    off, sec_start = daylight_section(c)
    rec["section_start_s"] = off
    date, days = c["date_utc"], 1
    if sec_start is not None:
        date = sec_start.strftime("%Y-%m-%d")
        rec["section_start_utc"] = sec_start.strftime("%Y-%m-%d %H:%M:%S")
        days = 1 if (sec_start + timedelta(hours=2)).date() == sec_start.date() else 2
    code, _, err = sh("fetch", c["url"], "--section-start", str(off))
    if code:
        return {**rec, "status": "fetch failed", "detail": err.strip().splitlines()[-1:]}
    run = ROOT / "runs" / c["id"]
    code, _, err = sh("events", str(run))
    evs = json.loads((run / "events.json").read_text()) if (run / "events.json").exists() else []
    rec["events"] = [round(e["tc"]) for e in evs]
    if code or not evs:
        return {**rec, "status": "no Doppler events", "detail": err.strip().splitlines()[-1:]}
    bbox = f"{c['lat']},{c['lon']},{radius}"
    code, _, err = sh("locate", str(run), "--date", date, "--days", str(days), "--bbox", bbox, "--no-open",
                      timeout=6 * 3600)
    rec["minutes"] = round((time.time() - t) / 60, 1)
    if code:
        return {**rec, "status": "no match", "detail": err.strip().splitlines()[-1:]}
    res = json.loads((run / "result.json").read_text())
    rec.update(found_lat=res["lat"], found_lon=res["lon"], score=res["score"], runner_up=res["runner_up_score"],
               found_start=res["recording_start_utc"], aircraft=[a["icao"] for a in res["aircraft"]],
               error_km=round(km(c["lat"], c["lon"], res["lat"], res["lon"]), 2), status="located")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=100)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    cands = json.loads((HERE / "candidates.json").read_text())
    out_p = HERE / "results.json"
    done = {r["id"]: r for r in json.loads(out_p.read_text())} if out_p.exists() else {}
    for c in cands:
        if a.only and c["id"] not in a.only:
            continue
        print(f"== {c['id']} {c['title'][:60]!r}", flush=True)
        try:
            done[c["id"]] = run_one(c, a.radius_km)
        except subprocess.TimeoutExpired:
            done[c["id"]] = {"id": c["id"], "url": c["url"], "status": "timeout"}
        print(json.dumps(done[c["id"]]), flush=True)
        out_p.write_text(json.dumps(list(done.values()), indent=2))
    write_md(list(done.values()))


def write_md(rows):
    lines = ["| Video | Date | Status | Error | Score (runner-up) | Start found / known |", "|---|---|---|---|---|---|"]
    for r in rows:
        err = f"{r['error_km']} km" if "error_km" in r else ""
        sc = f"{r['score']} ({r['runner_up']})" if "score" in r else ""
        st = f"{r.get('found_start', '')} / {r.get('section_start_utc') or '?'}" if "found_start" in r else ""
        lines.append(f"| [{(r.get('title') or r['id'])[:50]}]({r['url']}) | {r.get('date_utc', '')} | {r['status']} | {err} | {sc} | {st} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
