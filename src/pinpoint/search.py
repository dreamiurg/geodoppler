"""Find where a recording was made from its Doppler events and a day of ADS-B tracks.

Pipeline (after the source video's method):
  1. candidate passes: per event, track segments whose ground speed fits the measured speed;
     for every grid point within reach, the delayed CPA time and slant distance.
  2. offset filter: grid points that heard event-1 and event-2 passes with the measured
     CPA-time offset (and each at the measured slant distance).
  3. silence filter: no other aircraft passing close between consecutive events.
  4. scoring: model the exact emission-time Doppler curve of each real track at each surviving
     point (refined to a finer grid) and sum spectrogram brightness under it. Event 1's time and
     every event's f0 absorb measurement error; later events' times stay locked to geometry.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .audio import Spec
from .doppler import Event, C
from .flyover import Track, enu

DEG = np.pi / 180 * 6371000.0


@dataclass
class Tolerances:
    speed: float = 30.0  # m/s; ADS-B ground speed vs Doppler airspeed (wind) + fit error
    slant_rel: float = 0.45  # relative slant distance error of the curve fit
    offset_s: float = 10.0  # CPA-time offset error between events
    silence_rel: float = 0.6  # another pass closer than this x min(event slant) breaks silence
    shift_s: float = 6.0  # event-1 placement sweep for scoring
    f0_rel: float = 0.04  # f0 sweep around the fitted value


def grid(bbox, step_m):
    lat0, lon0, lat1, lon1 = bbox
    dlat = step_m / DEG
    dlon = step_m / (DEG * np.cos(np.radians((lat0 + lat1) / 2)))
    la, lo = np.meshgrid(np.arange(lat0, lat1, dlat), np.arange(lon0, lon1, dlon), indexing="ij")
    return la.ravel(), lo.ravel()


def segments(df: pd.DataFrame, max_gap=60.0, min_len=20):
    """Split each aircraft's day into contiguous passes."""
    for icao, g in df.groupby("icao", observed=True, sort=False):
        t = g.t.to_numpy()
        cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
        for s in np.split(np.arange(len(t)), cuts):
            if len(s) >= min_len:
                yield str(icao), g.iloc[s]


def passes(seg: pd.DataFrame, glat, glon, dmax, elev=None):
    """For grid points within dmax of a pass: delayed CPA time, slant distance, ground speed at CPA."""
    tr = Track(seg.t.to_numpy(), seg.lat.to_numpy(), seg.lon.to_numpy(), seg.alt_m.to_numpy())
    tr_t, tr_lat, tr_lon, tr_alt = tr.t[tr.ok][::2], tr.lat[tr.ok][::2], tr.lon[tr.ok][::2], tr.alt[tr.ok][::2]
    if len(tr_t) < 3:
        return None
    pad_lat = dmax / DEG
    pad_lon = dmax / (DEG * np.cos(np.radians(tr_lat.mean())))
    m = ((glat > tr_lat.min() - pad_lat) & (glat < tr_lat.max() + pad_lat)
         & (glon > tr_lon.min() - pad_lon) & (glon < tr_lon.max() + pad_lon))
    gi = np.flatnonzero(m)
    if not len(gi):
        return None
    lat_c, lon_c = tr_lat.mean(), tr_lon.mean()
    gx, gy, _ = enu(glat[gi], glon[gi], 0, lat_c, lon_c)
    ax, ay, _ = enu(tr_lat, tr_lon, 0, lat_c, lon_c)
    ge = np.zeros(len(gi)) if elev is None else elev(glat[gi], glon[gi])
    best_d = np.full(len(gi), np.inf)
    best_k = np.zeros(len(gi), int)
    for k0 in range(0, len(tr_t), 256):  # chunk to bound memory
        sl = slice(k0, k0 + 256)
        d2 = ((gx[:, None] - ax[None, sl]) ** 2 + (gy[:, None] - ay[None, sl]) ** 2
              + (tr_alt[None, sl] - ge[:, None]) ** 2)
        k = d2.argmin(1)
        dk = np.sqrt(d2[np.arange(len(gi)), k])
        better = dk < best_d
        best_d[better], best_k[better] = dk[better], k[better] + k0
    keep = best_d <= dmax
    # CPA at the ends of the pass is a truncation artefact, not a real closest approach
    keep &= (best_k > 0) & (best_k < len(tr_t) - 1)
    if not keep.any():
        return None
    k = best_k[keep]
    gs = np.interp(tr_t[k], seg.t.to_numpy(), seg.gs.to_numpy())
    return pd.DataFrame({"g": gi[keep], "t": tr_t[k] + best_d[keep] / C, "d": best_d[keep], "gs": gs})


def candidate_passes(df, ev: Event, glat, glon, tol: Tolerances, elev=None):
    v0, v1 = ev.v - tol.speed, ev.v + tol.speed
    d0, d1 = ev.d * (1 - tol.slant_rel), ev.d * (1 + tol.slant_rel)
    fast = df[(df.gs >= v0) & (df.gs <= v1)]
    out = []
    for icao, seg in segments(fast):
        p = passes(seg, glat, glon, d1, elev)
        if p is None:
            continue
        p = p[(p.d >= d0) & (p.gs >= v0) & (p.gs <= v1)]
        if len(p):
            out.append(p.assign(icao=icao))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["g", "t", "d", "gs", "icao"])


def offset_filter(cands: list[pd.DataFrame], events: list[Event], tol: Tolerances):
    """Join per-event passes on grid point where CPA offsets match the recording."""
    cur = cands[0].rename(columns={c: f"{c}0" for c in ["t", "d", "gs", "icao"]})
    for i in range(1, len(events)):
        nxt = cands[i].rename(columns={c: f"{c}{i}" for c in ["t", "d", "gs", "icao"]})
        cur = cur.merge(nxt, on="g")
        want = events[i].tc - events[0].tc
        cur = cur[(np.abs((cur[f"t{i}"] - cur["t0"]) - want) <= tol.offset_s) & (cur[f"icao{i}"] != cur["icao0"])]
    return cur.reset_index(drop=True)


def silence_filter(hits, df, events, glat, glon, tol: Tolerances, elev=None):
    """Drop points where another non-jet aircraft would have passed close between events."""
    if hits.empty:
        return hits
    n = len(events)
    g = np.unique(hits.g.to_numpy())
    sub_lat, sub_lon = glat[g], glon[g]
    thr = tol.silence_rel * min(e.d for e in events)
    t_lo, t_hi = hits.t0.min(), hits[f"t{n - 1}"].max()
    window = df[(df.t >= t_lo - 300) & (df.t <= t_hi + 300)]
    others = []
    for icao, seg in segments(window, min_len=5):
        p = passes(seg, sub_lat, sub_lon, thr, elev)
        if p is not None:
            others.append(p.assign(icao=icao, g=g[p.g.to_numpy()]))
    if not others:
        return hits
    o = pd.concat(others, ignore_index=True)
    m = hits.reset_index().merge(o, on="g", suffixes=("", "_o"))
    own = np.zeros(len(m), bool)
    for i in range(n):
        own |= m.icao_o.to_numpy() == m[f"icao{i}"].to_numpy()
    guard = 30.0  # s; the events themselves span ~±60 s around CPA
    between = (m.t_o > m.t0 + guard) & (m.t_o < m[f"t{n - 1}"] - guard)
    bad = m.loc[between & ~own, "index"].unique()
    return hits.drop(index=bad).reset_index(drop=True)


def _track(df, icao, t, span=600):
    g = df[(df.icao == icao) & (np.abs(df.t - t) < span)]
    return Track(g.t.to_numpy(), g.lat.to_numpy(), g.lon.to_numpy(), g.alt_m.to_numpy())


def score_point(spec: Spec, tracks, events, lat, lon, tol: Tolerances, elev0=0.0, half=70.0):
    """Locked multi-event score at one observer location. Returns (total, per-event, start_unix)."""
    cur = [tr.curve(lat, lon, elev0) for tr in tracks]
    # event 1: sweep placement shift and f0
    tr0, ratio0, _, tcpa0, _ = cur[0]
    e0 = events[0]
    rel = np.arange(-half, half + 1, spec.t[1] - spec.t[0])
    shifts = np.arange(-tol.shift_s, tol.shift_s + 0.01, 0.5)
    f0s = e0.f0 * (1 + np.linspace(-tol.f0_rel, tol.f0_rel, 33))
    r0 = np.interp(tcpa0 + rel, tr0, ratio0)
    best = (-np.inf, 0.0)
    for sh in shifts:
        t = e0.tc + sh + rel
        s = _sweep_f0(spec, t, r0, f0s, e0.harmonics)
        if s > best[0]:
            best = (s, sh)
    s0, sh = best
    start = tcpa0 - (e0.tc + sh)
    scores = [s0]
    for (tr, ratio, _, tcpa, _), e in zip(cur[1:], events[1:]):
        r = np.interp(tcpa + rel, tr, ratio)
        t = tcpa - start + rel  # locked: geometry alone places it
        scores.append(_sweep_f0(spec, t, r, e.f0 * (1 + np.linspace(-tol.f0_rel, tol.f0_rel, 33)), e.harmonics))
    return float(sum(scores)), scores, float(start)


def _sweep_f0(spec: Spec, t, ratio, f0s, harmonics):
    f = f0s[:, None] * ratio[None, :]
    vals = sum(np.nan_to_num(spec.sample(np.broadcast_to(t, f.shape), f * h), nan=0.0).clip(-2, 6) for h in harmonics)
    return float((vals / len(harmonics)).mean(1).max())


def locate(spec: Spec, events: list[Event], df: pd.DataFrame, bbox, step=200.0, fine=100.0,
           tol: Tolerances = Tolerances(), elev=None, log=print) -> pd.DataFrame:
    glat, glon = grid(bbox, step)
    log(f"grid: {len(glat):,} points at {step:.0f} m")
    cands = []
    for i, e in enumerate(events):
        c = candidate_passes(df, e, glat, glon, tol, elev)
        log(f"event {i + 1}: {len(c):,} (point, pass) candidates from {c.icao.nunique() if len(c) else 0} aircraft")
        cands.append(c)
    hits = offset_filter(cands, events, tol)
    log(f"offset filter: {len(hits):,} hits at {hits.g.nunique() if len(hits) else 0:,} points")
    if len(events) > 1:
        hits = silence_filter(hits, df, events, glat, glon, tol, elev)
        log(f"silence filter: {len(hits):,} hits at {hits.g.nunique() if len(hits) else 0:,} points")
    if hits.empty:
        return pd.DataFrame()
    n = len(events)
    tracks = {}
    rows = []
    off = (fine / 2) / DEG
    for r in hits.itertuples():
        key = tuple((getattr(r, f"icao{i}"), round(getattr(r, f"t{i}") / 600)) for i in range(n))
        if key not in tracks:
            tracks[key] = [_track(df, getattr(r, f"icao{i}"), getattr(r, f"t{i}")) for i in range(n)]
        la, lo = glat[r.g], glon[r.g]
        k = np.cos(np.radians(la))
        for dla, dlo in ((-off, -off / k), (-off, off / k), (off, -off / k), (off, off / k)):
            e0 = 0.0 if elev is None else float(elev(np.array([la + dla]), np.array([lo + dlo]))[0])
            tot, per, start = score_point(spec, tracks[key], events, la + dla, lo + dlo, tol, e0)
            rows.append({"lat": la + dla, "lon": lo + dlo, "score": tot, **{f"s{i + 1}": per[i] for i in range(n)},
                         "start": start, **{f"icao{i + 1}": key[i][0] for i in range(n)}})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    log(f"scored {len(out):,} points at {fine:.0f} m")
    return out
