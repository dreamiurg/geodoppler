"""ADS-B history: fetch adsb.lol daily globe_history, reduce to non-jet tracks in a bbox.

adsb.lol publishes one readsb `globe_history` tarball per UTC day on GitHub
(adsblol/globe_history_<year>, ODbL). Each aircraft is traces/<xx>/trace_full_<icao>.json,
gzip-compressed, with rows [dt, lat, lon, alt_baro_ft|"ground", gs_kt, track, flags,
vrate, details, source, alt_geom_ft, ...] relative to the file's `timestamp`.
"""

import gzip
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

FT = 0.3048
KT = 0.514444
CONUS = (24.0, -125.0, 50.0, -66.0)  # lat0, lon0, lat1, lon1


def _types():
    return json.loads(resources.files("pinpoint.resources").joinpath("icao_aircraft_types.json").read_text())


def is_jet(type_code: str | None, types=None) -> bool:
    """True only for known jets; unknown types are kept (may be props)."""
    d = (types or _types()).get(type_code or "", {}).get("desc", "")
    return d.endswith("J")


def is_rotorcraft(type_code: str | None, types=None) -> bool:
    """True only for known helicopters/gyrodynes (ICAO desc H*); unknown types are kept."""
    return (types or _types()).get(type_code or "", {}).get("desc", "").startswith("H")


def download_day(day: str, data_dir: Path) -> Path:
    """Download + extract adsb.lol globe_history for YYYY-MM-DD into data_dir/adsb/<day>/traces."""
    out = data_dir / "adsb" / day
    if (out / ".complete").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    y, m, d = day.split("-")
    assets = []
    for suffix in ("prod-0", "prod-0tmp", "staging-0", "staging-0tmp"):  # naming varies by day
        tag = f"v{y}.{m}.{d}-planes-readsb-{suffix}"
        try:
            with urllib.request.urlopen(f"https://api.github.com/repos/adsblol/globe_history_{y}/releases/tags/{tag}") as r:
                assets = [(a["browser_download_url"], a["size"]) for a in json.load(r)["assets"]]
            break
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    if not assets:
        raise SystemExit(f"no adsb.lol history for {day}")
    # parts to disk with retry/resume (GitHub hands out transient 504s), then extract in order
    parts = []
    for u, size in sorted(assets):
        part = out / u.rsplit("/", 1)[1]
        for attempt in range(12):
            if part.exists() and part.stat().st_size == size:
                break  # resets mid-transfer (curl 18/56) need a fresh resumed request
            r = subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-all-errors", "--retry-delay", "10",
                                "-C", "-", "-o", str(part), u])
            if r.returncode == 0:
                break
            time.sleep(min(120, 10 * 2 ** attempt))
        else:
            raise SystemExit(f"download kept failing: {u}")
        parts.append(part)
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    if subprocess.run(["tar", "-x", "-C", str(out), "./traces"], stdin=cat.stdout).returncode or cat.wait():
        raise SystemExit(f"extracting {tag} failed")
    for part in parts:
        part.unlink()
    (out / ".complete").touch()
    return out


def _reduce_dir(args):
    d, bbox, types = args
    lat0, lon0, lat1, lon1 = bbox
    rows = []
    for p in Path(d).iterdir():
        try:
            j = orjson.loads(gzip.decompress(p.read_bytes()))
        except Exception:  # corrupt/partial file: skip it, one aircraft is not worth failing the day
            continue
        if is_jet(j.get("t"), types):
            continue
        ts = j["timestamp"]
        icao, typ = j["icao"], j.get("t") or ""
        for r in j["trace"]:
            lat, lon = r[1], r[2]
            if not (lat0 <= lat <= lat1 and lon0 <= lon <= lon1):
                continue
            alt = r[10] if len(r) > 10 and isinstance(r[10], (int, float)) else r[3]
            if not isinstance(alt, (int, float)) or r[4] is None:
                continue  # on ground or no speed
            rows.append((icao, typ, ts + r[0], lat, lon, alt * FT, r[4] * KT))
    return rows


def ensure_day(day: str, data_dir: Path) -> Path:
    """props.parquet for the day; raw traces (~3 GB) are deleted once reduced."""
    out = data_dir / "adsb" / day / "props.parquet"
    if not out.exists():
        if not (out.parent / "traces").exists():
            (out.parent / ".complete").unlink(missing_ok=True)  # traces were already reduced and deleted
        build_parquet(download_day(day, data_dir))
        shutil.rmtree(out.parent / "traces", ignore_errors=True)
    return out


def build_parquet(day_dir: Path, bbox=CONUS, workers: int = 10) -> Path:
    out = day_dir / "props.parquet"
    if out.exists():
        return out
    types = _types()
    dirs = sorted(p for p in (day_dir / "traces").iterdir() if p.is_dir())
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for chunk in ex.map(_reduce_dir, [(d, bbox, types) for d in dirs]) for r in chunk]
    df = pd.DataFrame(rows, columns=["icao", "type", "t", "lat", "lon", "alt_m", "gs"])
    df = df.astype({"icao": "category", "type": "category", "alt_m": "float32", "gs": "float32"})
    tmp = out.with_suffix(".tmp")
    df.sort_values(["icao", "t"]).to_parquet(tmp, index=False)
    tmp.replace(out)  # a crash mid-write must not leave a parquet that looks complete
    return out
