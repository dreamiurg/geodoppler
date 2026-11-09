"""pinpoint: locate where a recording was made from aircraft Doppler sounds + ADS-B history.

  pinpoint fetch URL                      download audio + metadata, render spectrogram overviews
  pinpoint events RUN [--box ...]         fit Doppler events (auto-detect if no --box)
  pinpoint locate RUN --date D --bbox B   search ADS-B tracks, write candidates + map.html
  pinpoint run URL --date D --bbox B      all of the above
  pinpoint frames RUN / pin RUN LAT LON   stills for visual clues; map a location judged from them
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from . import adsb, audio, doppler, mapview, search
from .dem import Elevation

DATA = Path(os.environ.get("PINPOINT_DATA", Path.home() / ".cache" / "pinpoint"))
RUNS = Path(os.environ.get("PINPOINT_RUNS", "runs"))
MAX_MINUTES = float(os.environ.get("PINPOINT_MAX_MINUTES", 120))
REGIONS = {"conus": adsb.CONUS}
META_KEYS = ["id", "title", "uploader", "upload_date", "duration", "description", "location", "webpage_url", "tags"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------- fetch ----------

def ytdlp(*args, timeout=300) -> str:
    """The system yt-dlp: YouTube breaks old versions often, and a package manager keeps it current."""
    exe = shutil.which("yt-dlp")
    if not exe:
        raise SystemExit("yt-dlp not found on PATH (brew install yt-dlp)")
    return subprocess.run([exe, "--no-warnings", *args], check=True, capture_output=True, text=True,
                          timeout=timeout).stdout


def fetch(url: str, section_start: float = 0.0) -> Path:
    """`section_start`: seconds into the video to start analysing (long livestreams: pick daylight)."""
    info = json.loads(ytdlp("--dump-single-json", "--skip-download", "--no-playlist", url))
    run = RUNS / info["id"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "meta.json").write_text(json.dumps({**{k: info.get(k) for k in META_KEYS},
                                              "release_timestamp": info.get("release_timestamp"),
                                              "section_start_s": section_start}, indent=2))
    if not media(run, "audio"):
        log(f"downloading audio for {info['id']}: {info.get('title')!r}")
        # whole track over plain https (section cuts via ffmpeg hang on long live archives); cut locally
        ytdlp("-q", "--no-playlist", "-f", "bestaudio[protocol^=http]/bestaudio/best", "-o", str(run / "audio.%(ext)s"),
              info["webpage_url"], timeout=1800)
    spec = load_spec(run)
    level = float((run / "level_db").read_text()) if (run / "level_db").exists() else None
    (run / "meta.json").write_text(json.dumps({**json.loads((run / "meta.json").read_text()), "level_db": level}, indent=2))
    if level is not None and level < -70:
        log(f"WARNING: audio is silent ({level:.0f} dBFS): no microphone, nothing acoustic to work with")
    overview(spec, run)
    log(f"run dir: {run}")
    return run


def media(run: Path, stem: str) -> list[Path]:
    """Finished downloads only (yt-dlp leaves .part/.ytdl files when interrupted)."""
    return [p for p in sorted(run.glob(f"{stem}.*")) if p.suffix not in (".part", ".ytdl")]


def load_spec(run: Path) -> audio.Spec:
    cache = run / "spec.npz"
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    start = float(meta.get("section_start_s") or 0)
    if cache.exists():
        z = np.load(cache)
        if "start" in z and float(z["start"]) == start:
            return audio.Spec(z["t"], z["f"], z["S"])
        if "start" in z:  # analysed window moved: event times etc. refer to the old one
            for stale in ("events.json", "events.png", "candidates.parquet", "verified.parquet", "result.json"):
                (run / stale).unlink(missing_ok=True)
    return _build_spec(run, cache, meta)


def _build_spec(run: Path, cache: Path, meta: dict) -> audio.Spec:
    # shortcut: first MAX_MINUTES only; long livestreams would need chunked spectrograms
    start = float(meta.get("section_start_s") or 0)
    x = audio.load(str(media(run, "audio")[0]), max_s=MAX_MINUTES * 60, start_s=start)
    level = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))
    (run / "level_db").write_text(f"{level:.1f}\n")
    sp = audio.suppress_steady(audio.spectrogram(x))
    np.savez_compressed(cache, t=sp.t, f=sp.f, S=sp.S, start=start)
    return sp


def _plot(ax, sp: audio.Spec):
    ax.pcolormesh(sp.t, sp.f, sp.S, vmin=0, vmax=6, cmap="magma", shading="auto", rasterized=True)


def overview(sp: audio.Spec, run: Path, chunk=600.0, fmax=500.0):
    """One PNG per 10 minutes, 20-500 Hz: what a person (or Claude) scans for Doppler curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = run / "overview"
    out.mkdir(exist_ok=True)
    for i, t0 in enumerate(np.arange(0, sp.t[-1], chunk)):
        fig, ax = plt.subplots(figsize=(20, 6))
        _plot(ax, sp.crop(t0, t0 + chunk, 20, fmax))
        ax.set_xlabel("seconds into recording")
        ax.set_ylabel("Hz")
        ax.set_xticks(np.arange(t0, t0 + chunk + 1, 30))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{int(t0):05d}s.png", dpi=60)
        plt.close(fig)


def frames(run: Path, n=24) -> Path:
    """Evenly spaced stills for visual clues (signs, terrain, vegetation, architecture)."""
    meta = json.loads((run / "meta.json").read_text())
    vid = next(iter(media(run, "video")), None)
    if vid is None:
        ytdlp("-q", "--no-playlist", "-f", "bv*[height<=720]/b[height<=720]/bv*/b", "-o", str(run / "video.%(ext)s"),
              meta["webpage_url"])
        vid = media(run, "video")[0]
    out = run / "frames"
    out.mkdir(exist_ok=True)
    dur = float(meta.get("duration") or 60)
    for i, t in enumerate(np.linspace(dur * 0.02, dur * 0.98, n)):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.1f}", "-i", str(vid), "-frames:v", "1",
                        "-vf", "scale='min(1280,iw)':-2", str(out / f"{i:02d}_{int(t):05d}s.jpg")], check=True)
    log(f"frames: {out}")
    return out


def pin(run: Path, lat: float, lon: float, note: str, radius_m: float, open_map=True):
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    path = mapview.render_pin(run / "map.html", meta.get("title") or run.name, lat, lon, note, radius_m)
    (run / "result.json").write_text(json.dumps({"lat": lat, "lon": lon, "basis": note, "radius_m": radius_m,
                                                 "google_maps": f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"}, indent=2))
    log(f"map: {path.resolve()}")
    if open_map:
        webbrowser.open(path.resolve().as_uri())
    return path


# ---------- events ----------

def parse_box(s: str):
    """t0,t1,fmin,fmax[,h1+h2]  e.g. 2080,2240,80,190,1+2"""
    p = s.split(",")
    h = tuple(int(x) for x in p[4].split("+")) if len(p) > 4 else (1,)
    return (*map(float, p[:4]), h)


def events(run: Path, boxes=None) -> list[doppler.Event]:
    """Fit events; all go to events.json. `locate` searches with the two strongest and uses the rest to verify."""
    sp = load_spec(run)
    if boxes:
        evs = [doppler.fit_event(sp, t0, t1, f0, f1, h) for t0, t1, f0, f1, h in boxes]
    else:
        log("auto-detecting Doppler events (a few minutes per hour of audio)")
        evs = doppler.detect_events(sp, log=log)
    evs.sort(key=lambda e: e.tc)
    for e in evs:
        log(f"event tc={e.tc:.1f}s v={e.v:.0f} m/s slant={e.d:.0f} m f0={e.f0:.1f} Hz score={e.score:.2f}")
    (run / "events.json").write_text(json.dumps([e.to_dict() for e in evs], indent=2))
    if evs:
        plot_events(sp, evs, run / "events.png")
    else:
        log("no Doppler events found")
    return evs


def plot_events(sp, evs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(evs), figsize=(9 * len(evs), 6), squeeze=False)
    for ax, e in zip(axs[0], evs):
        _plot(ax, sp.crop(e.t0 - 20, e.t1 + 20, e.f0 * 0.5, e.f0 * max(e.harmonics) * 1.7))
        t = np.linspace(e.t0, e.t1, 400)
        for h in e.harmonics:
            ax.plot(t, h * doppler.ideal_curve(t, e.f0, e.v, e.d, e.tc), "c-", lw=0.8, alpha=0.6)
        ax.set_title(f"tc={e.tc:.1f}s  v={e.v:.0f} m/s  slant={e.d:.0f} m  score={e.score:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def load_events(run: Path):
    return [doppler.Event(**{**d, "harmonics": tuple(d["harmonics"])}) for d in json.loads((run / "events.json").read_text())]


# ---------- locate ----------

def parse_bbox(s: str):
    if s.lower() in REGIONS:
        return REGIONS[s.lower()]
    v = [float(x) for x in s.split(",")]
    if len(v) == 3:  # lat,lon,radius_km
        la, lo, r = v
        dla, dlo = r / 111.0, r / (111.0 * np.cos(np.radians(la)))
        return (la - dla, lo - dlo, la + dla, lo + dlo)
    return tuple(v)


def tiles(bbox, size=2.0):
    lat0, lon0, lat1, lon1 = bbox
    for a in np.arange(lat0, lat1, size):
        for b in np.arange(lon0, lon1, size):
            yield (a, b, min(a + size, lat1), min(b + size, lon1))


def locate(run: Path, date: str, bbox, step=200.0, open_map=True, start=None, days=1) -> dict:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise SystemExit(f"--date must be YYYY-MM-DD, got {date!r}")
    sp, all_evs = load_spec(run), load_events(run)
    if not all_evs:
        raise SystemExit("no events: run `pinpoint events` first")
    if len(all_evs) < 2 and start is None:
        # validation: 1 event, unknown time -> 2.7 h search, wrong spot, 0.8 % margin
        raise SystemExit("only one flyover: need a second one or a known start time (--start)")
    by_score = sorted(all_evs, key=lambda e: -e.score)
    evs = sorted(by_score[:2], key=lambda e: e.tc)
    extra = sorted(by_score[2:], key=lambda e: e.tc)
    log(f"searching with events at {', '.join(f'{e.tc:.0f}s' for e in evs)}; verifying with "
        f"{', '.join(f'{e.tc:.0f}s' for e in extra) or 'none'}")
    log(f"ADS-B for {date} (first use downloads ~4 GB from adsb.lol)")
    dates = [(pd.Timestamp(date) + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    df = pd.concat([pd.read_parquet(adsb.ensure_day(d, DATA)) for d in dates], ignore_index=True)
    df = df.astype({"icao": "category", "type": "category"})
    if days > 1:  # each day's parquet is already (icao, t)-sorted
        df = df.sort_values(["icao", "t"], kind="stable")
    elev = Elevation(DATA)
    res = []
    with search.pool(sp, evs, elev) as ex:
        for tb in tiles(bbox):
            pad = 0.15
            sub = df[(df.lat > tb[0] - pad) & (df.lat < tb[2] + pad) & (df.lon > tb[1] - pad * 1.4) & (df.lon < tb[3] + pad * 1.4)]
            log(f"tile {tb[0]:.1f},{tb[1]:.1f}..{tb[2]:.1f},{tb[3]:.1f}: {sub.icao.nunique()} aircraft")
            r = search.locate(sp, evs, sub, tb, step=step, log=lambda m: log("  " + m), start=start, elev=elev, ex=ex)
            if len(r):
                res.append(r)
    if not res:
        raise SystemExit("no location matched all events; widen --bbox, check --date, or refit events")
    cands = pd.concat(res, ignore_index=True).sort_values("score", ascending=False).reset_index(drop=True)
    cands.to_parquet(run / "candidates.parquet")
    if extra:
        cands = search.verify_extra(sp, cands, extra, evs, df, search.Tolerances(), log=log, elev=elev)
        cands.to_parquet(run / "verified.parquet")
    return report(run, cands, df, evs, date, open_map, extra)


def report(run, cands, df, evs, date, open_map=True, extra=()) -> dict:
    n = len(evs)
    best = cands.iloc[0]
    combo = [f"icao{i + 1}" for i in range(n)]
    per_combo = cands.drop_duplicates(combo)
    runner = per_combo.iloc[1] if len(per_combo) > 1 else None
    types = df.drop_duplicates("icao").set_index("icao")["type"].astype(str).to_dict()
    # (event time in recording, icao) for every event the best candidate explains
    heard = [(e.tc, best[f"icao{i + 1}"]) for i, e in enumerate(evs)]
    heard += [(e.tc, best[f"xicao{j + 1}"]) for j, e in enumerate(extra) if best.get(f"xicao{j + 1}")]
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    off = float(meta.get("section_start_s") or 0)  # spectrogram t=0 is this many seconds into the video
    tracks = []
    for tc, icao in sorted(heard):
        t_cpa = best.start + tc
        g = df[(df.icao == icao) & ((df.t - t_cpa).abs() < 900)]
        tracks.append({"path": g[["lat", "lon"]].round(5).values.tolist(),
                       "label": html.escape(f"{icao} {types.get(icao, '')}: heard at {tc + off:.0f}s = {pd.to_datetime(t_cpa, unit='s'):%H:%M:%S} UTC")})
    margin = None if runner is None else (best.score - runner.score) / runner.score
    # validation: correct answers had 18-50 % margins, wrong ones 0.04-8 %
    confidence = "unknown" if margin is None else "high" if margin >= 0.15 else "low"
    result = {
        "lat": round(float(best.lat), 5), "lon": round(float(best.lon), 5), "score": round(float(best.score), 3),
        "runner_up_score": None if runner is None else round(float(runner.score), 3),
        "margin": None if margin is None else round(float(margin), 3), "confidence": confidence,
        "recording_start_utc": f"{pd.to_datetime(best.start - off, unit='s'):%Y-%m-%d %H:%M:%S}",
        "analysed_from_s": off,
        "aircraft": [{"heard_at_s": round(tc + off, 1), "icao": icao, "type": types.get(icao, "")} for tc, icao in sorted(heard)],
        "events_unexplained": [round(e.tc + off, 1) for j, e in enumerate(extra) if not best.get(f"xicao{j + 1}")],
        "date": date, "candidates": len(cands),
        "google_maps": f"https://www.google.com/maps?q={best.lat:.5f},{best.lon:.5f}",
    }
    (run / "result.json").write_text(json.dumps(result, indent=2))
    info = {
        "Best estimate": f"<a href='{result['google_maps']}' target=_blank>{result['lat']}, {result['lon']}</a>",
        "Confidence": f"{confidence} (margin {result['margin']:.0%} over the next aircraft combination)" if margin is not None else confidence,
        "Score": f"{result['score']} (next aircraft combination: {result['runner_up_score']})",
        "Recording started": result["recording_start_utc"] + " UTC",
        "Aircraft heard": "<br>".join(html.escape(f"{a['heard_at_s']:.0f}s: {a['icao']} {a['type']}") for a in result["aircraft"]),
        "Events unexplained": ", ".join(f"{t:.0f}s" for t in result["events_unexplained"]) or "none",
    }
    title = meta.get("title") or run.name
    path = mapview.render(cands, run / "map.html", title, info, tracks)
    log(f"best: {result['lat']}, {result['lon']}  score {result['score']} (runner-up {result['runner_up_score']}), confidence {confidence}")
    log(f"map: {path.resolve()}")
    if open_map:
        webbrowser.open(path.resolve().as_uri())
    print(json.dumps(result, indent=2))
    return result


# ---------- entry ----------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="pinpoint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("url")
    f.add_argument("--section-start", type=float, default=0, help="seconds into the video to start analysing")
    fr = sub.add_parser("frames", help="extract stills for visual clues"); fr.add_argument("run", type=Path)
    fr.add_argument("-n", type=int, default=24)
    pn = sub.add_parser("pin", help="map a location judged from non-acoustic clues")
    pn.add_argument("run", type=Path); pn.add_argument("lat", type=float); pn.add_argument("lon", type=float)
    pn.add_argument("--note", default="judged from visual/metadata clues"); pn.add_argument("--radius", type=float, default=500)
    pn.add_argument("--no-open", action="store_true")
    e = sub.add_parser("events"); e.add_argument("run", type=Path)
    for p in (e,):
        p.add_argument("--box", action="append", type=parse_box, help="t0,t1,fmin,fmax[,1+2] seconds/Hz; repeat per event")
    for p in (sub.add_parser("locate"), sub.add_parser("run")):
        p.add_argument("target", help="run dir (locate) or URL (run)")
        p.add_argument("--date", help="UTC day YYYY-MM-DD (default: upload date)")
        p.add_argument("--bbox", type=parse_bbox, default="conus",
                       help="lat0,lon0,lat1,lon1 | lat,lon,radius_km | conus")
        p.add_argument("--step", type=float, default=200.0, help="coarse grid metres")
        p.add_argument("--days", type=int, default=1, help="consecutive UTC days of ADS-B from --date (recording across midnight: 2)")
        p.add_argument("--start", help="known recording start, UTC ISO time, e.g. 2026-09-08T20:17:40")
        p.add_argument("--start-window", type=float, default=120, help="+/- seconds around --start")
        p.add_argument("--no-open", action="store_true")
        if p.prog.endswith("run"):
            p.add_argument("--box", action="append", type=parse_box)
            p.add_argument("--section-start", type=float, default=0, help="seconds into the video to start analysing")
    a = ap.parse_args(argv)

    if a.cmd == "fetch":
        fetch(a.url, a.section_start)
    elif a.cmd == "frames":
        frames(a.run, a.n)
    elif a.cmd == "pin":
        pin(a.run, a.lat, a.lon, a.note, a.radius, not a.no_open)
    elif a.cmd == "events":
        events(a.run, a.box)
    else:
        run = fetch(a.target, a.section_start) if a.cmd == "run" else Path(a.target)
        if a.cmd == "run":
            events(run, a.box)
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        d = a.date or (meta.get("upload_date") and f"{meta['upload_date'][:4]}-{meta['upload_date'][4:6]}-{meta['upload_date'][6:]}")
        start = None
        if a.start:
            ts = pd.Timestamp(a.start)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            s0 = ts.timestamp() + float(meta.get("section_start_s") or 0)  # --start is the video's start
            start = (s0 - a.start_window, s0 + a.start_window)
            d = a.date or f"{ts:%Y-%m-%d}"
        if not d:
            raise SystemExit("--date required")
        locate(run, d, a.bbox, a.step, not a.no_open, start, a.days)
