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

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

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
    lock_slack_s: float = 2.0  # later events: play around the geometry-locked placement
    f0_rel: float = 0.04  # f0 sweep around the fitted value
    airspeed: tuple = (0.85, 1.15)  # radial-velocity scale sweep (wind along track)
    airspeed_steps: int = 7


class Points:
    """Observer locations. A regular grid answers window queries by arithmetic, not masking."""

    def __init__(self, lat, lon, regular=None, mask=None):
        self.lat, self.lon, self._reg, self.mask = np.asarray(lat), np.asarray(lon), regular, mask

    def subset(self, idx):
        """Same grid, but window queries only return these indices."""
        mask = np.zeros(len(self.lat), bool)
        mask[idx] = True
        return Points(self.lat, self.lon, self._reg, mask)

    @classmethod
    def grid(cls, bbox, step_m):
        lat0, lon0, lat1, lon1 = bbox
        dlat = step_m / DEG
        dlon = step_m / (DEG * np.cos(np.radians((lat0 + lat1) / 2)))
        la, lo = np.arange(lat0, lat1, dlat), np.arange(lon0, lon1, dlon)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        return cls(LA.ravel(), LO.ravel(), (lat0, lon0, dlat, dlon, len(la), len(lo)))

    def __len__(self):
        return len(self.lat)

    def window(self, la0, la1, lo0, lo1):
        if self._reg is None:
            return np.flatnonzero((self.lat >= la0) & (self.lat <= la1) & (self.lon >= lo0) & (self.lon <= lo1))
        lat0, lon0, dlat, dlon, n, m = self._reg
        i0, i1 = max(0, int(np.ceil((la0 - lat0) / dlat))), min(n - 1, int(np.floor((la1 - lat0) / dlat)))
        j0, j1 = max(0, int(np.ceil((lo0 - lon0) / dlon))), min(m - 1, int(np.floor((lo1 - lon0) / dlon)))
        if i0 > i1 or j0 > j1:
            return np.empty(0, int)
        idx = (np.arange(i0, i1 + 1)[:, None] * m + np.arange(j0, j1 + 1)[None, :]).ravel()
        return idx if self.mask is None else idx[self.mask[idx]]


def segments(df: pd.DataFrame, max_gap=60.0, min_len=20):
    """Split each aircraft's day into contiguous passes."""
    for icao, g in df.groupby("icao", observed=True, sort=False):
        t = g.t.to_numpy()
        cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
        for s in np.split(np.arange(len(t)), cuts):
            if len(s) >= min_len:
                yield str(icao), g.iloc[s]


def passes(seg: pd.DataFrame, pts: Points, dmax, elev=None, chunk=32):
    """For points within dmax of a pass: delayed CPA time, slant distance, ground speed at CPA.
    shortcut: one CPA per point per pass (global min); a pass that circles a point twice keeps only the closest."""
    tr = Track(seg.t.to_numpy(), seg.lat.to_numpy(), seg.lon.to_numpy(), seg.alt_m.to_numpy())
    tr_t, tr_lat, tr_lon, tr_alt = tr.t[tr.ok][::2], tr.lat[tr.ok][::2], tr.lon[tr.ok][::2], tr.alt[tr.ok][::2]
    if len(tr_t) < 3:
        return None
    lat_c, lon_c = tr_lat.mean(), tr_lon.mean()
    pad_lat, pad_lon = dmax / DEG, dmax / (DEG * np.cos(np.radians(lat_c)))
    ax, ay, _ = enu(tr_lat, tr_lon, 0, lat_c, lon_c)
    G, D, K = [], [], []
    for k0 in range(0, len(tr_t), chunk):
        sl = slice(k0, k0 + chunk)
        gi = pts.window(tr_lat[sl].min() - pad_lat, tr_lat[sl].max() + pad_lat,
                        tr_lon[sl].min() - pad_lon, tr_lon[sl].max() + pad_lon)
        if not len(gi):
            continue
        gx, gy, _ = enu(pts.lat[gi], pts.lon[gi], 0, lat_c, lon_c)
        ge = 0.0 if elev is None else elev(pts.lat[gi], pts.lon[gi])[:, None]
        d2 = (gx[:, None] - ax[None, sl]) ** 2 + (gy[:, None] - ay[None, sl]) ** 2 + (tr_alt[None, sl] - ge) ** 2
        k = d2.argmin(1)
        dk = np.sqrt(d2[np.arange(len(gi)), k])
        m = dk <= dmax
        G.append(gi[m]); D.append(dk[m]); K.append(k[m] + k0)
    G = np.concatenate(G) if G else np.empty(0, int)
    if not len(G):
        return None
    D, K = np.concatenate(D), np.concatenate(K)
    o = np.lexsort((D, G))
    first = np.r_[True, G[o][1:] != G[o][:-1]]
    G, D, K = G[o][first], D[o][first], K[o][first]
    # CPA at the ends of the pass is a truncation artefact, not a real closest approach
    keep = (K > 0) & (K < len(tr_t) - 1)
    if not keep.any():
        return None
    G, D, K = G[keep], D[keep], K[keep]
    gs = np.interp(tr_t[K], seg.t.to_numpy(), seg.gs.to_numpy())
    return pd.DataFrame({"g": G, "t": tr_t[K] + D / C, "d": D, "gs": gs})


def candidate_passes(df, ev: Event, pts: Points, tol: Tolerances, elev=None, start=None):
    """`start`: optional (lo, hi) unix bounds on when the recording began."""
    v0, v1 = ev.v - tol.speed, ev.v + tol.speed
    d0, d1 = ev.d * (1 - tol.slant_rel), ev.d * (1 + tol.slant_rel)
    fast = df[(df.gs >= v0) & (df.gs <= v1)]
    if start is not None:
        fast = fast[(fast.t >= start[0] + ev.tc - 900) & (fast.t <= start[1] + ev.tc + 900)]
    out = []
    for icao, seg in segments(fast):
        p = passes(seg, pts, d1, elev)
        if p is None:
            continue
        p = p[(p.d >= d0) & (p.gs >= v0) & (p.gs <= v1)]
        if start is not None:
            p = p[(p.t - ev.tc >= start[0]) & (p.t - ev.tc <= start[1])]
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


def silence_filter(hits, df, events, pts: Points, tol: Tolerances, elev=None):
    """Drop points where another non-jet aircraft would have passed close between events."""
    if hits.empty:
        return hits
    n = len(events)
    g = np.unique(hits.g.to_numpy())
    sub = pts.subset(g)
    thr = tol.silence_rel * min(e.d for e in events)
    t_lo, t_hi = hits.t0.min(), hits[f"t{n - 1}"].max()
    window = df[(df.t >= t_lo - 300) & (df.t <= t_hi + 300)]
    others = []
    for icao, seg in segments(window, min_len=5):
        p = passes(seg, sub, thr, elev)
        if p is not None:
            others.append(p.assign(icao=icao))
    if not others:
        return hits
    o = pd.concat(others, ignore_index=True)[["g", "t", "icao"]].rename(columns={"t": "t_o", "icao": "icao_o"})
    m = hits.reset_index().merge(o, on="g")
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
    """Locked multi-event score at one observer location. Returns (total, per-event, start_unix).

    Event 1 absorbs measurement error (placement shift, f0). Later events are placed by geometry
    alone, with only `lock_slack_s` of play. Every event may scale its radial velocity by
    `airspeed` (wind: Doppler follows speed through the air, ADS-B reports ground speed)."""
    rel = np.arange(-half, half + 1, spec.t[1] - spec.t[0])
    ks = np.linspace(*tol.airspeed, tol.airspeed_steps)
    start, scores = None, []
    for tr, e in zip(tracks, events):
        t_rec, ratio, _, tcpa, _ = tr.curve(lat, lon, elev0)
        vr = np.interp(tcpa + rel, t_rec, C / ratio - C)
        f0s = e.f0 * (1 + np.linspace(-tol.f0_rel, tol.f0_rel, 25))
        if start is None:
            base, shifts = e.tc, np.arange(-tol.shift_s, tol.shift_s + 0.01, 0.5)
        else:
            base, shifts = tcpa - start, np.arange(-tol.lock_slack_s, tol.lock_slack_s + 0.01, 0.5)
        sc, sh = _sweep(spec, base + rel, vr, shifts, ks, f0s, e.harmonics)
        if start is None:
            start = tcpa - (e.tc + sh)
        scores.append(sc)
    return float(sum(scores)), scores, float(start)


def _sweep(spec: Spec, t, vr, shifts, ks, f0s, harmonics):
    """Best mean clipped brightness over placement shift x airspeed scale x f0. Returns (score, shift)."""
    ratio = C / (C + ks[:, None] * vr[None, :])  # (k, t)
    f = f0s[None, :, None] * ratio[:, None, :]  # (k, f0, t)
    best, best_sh = -np.inf, 0.0
    for sh in shifts:
        tt = np.broadcast_to(t + sh, f.shape)
        v = sum(np.nan_to_num(spec.sample(tt, f * h), nan=0.0).clip(-2, 6) for h in harmonics) / len(harmonics)
        m = v.mean(-1).max()
        if m > best:
            best, best_sh = m, sh
    return float(best), float(best_sh)


def verify_extra(spec: Spec, cands: pd.DataFrame, extra: list[Event], primary: list[Event], df: pd.DataFrame,
                 tol: Tolerances, top=300, log=print) -> pd.DataFrame:
    """Re-rank the best point of the top combinations by how well *other* detected events are explained:
    for each extra event, the best real aircraft whose geometry puts its flyover at that moment
    (locked to the recording start found from event 1). No aircraft -> 0, so a spurious event costs nothing."""
    n = len(primary)
    combo = [f"icao{i + 1}" for i in range(n)]
    best = cands.drop_duplicates(combo).head(top).copy()
    log(f"verifying {len(best)} best combinations against {len(extra)} extra event(s)")
    rel = np.arange(-70.0, 71.0, spec.t[1] - spec.t[0])
    ks = np.linspace(*tol.airspeed, tol.airspeed_steps)
    extra_scores = np.zeros((len(best), len(extra)))
    extra_icao = np.full((len(best), len(extra)), "", dtype=object)
    for j, e in enumerate(extra):
        f0s = e.f0 * (1 + np.linspace(-tol.f0_rel, tol.f0_rel, 25))
        dmax = e.d * (1 + tol.slant_rel)
        for i, r in enumerate(best.itertuples()):
            when = r.start + e.tc
            near = df[(df.t > when - 400) & (df.t < when + 400)
                      & (np.abs(df.lat - r.lat) < dmax / DEG + 0.01)
                      & (np.abs(df.lon - r.lon) < dmax / (DEG * np.cos(np.radians(r.lat))) + 0.01)]
            for icao, seg in segments(near, min_len=10):
                if icao in {getattr(r, c) for c in combo}:
                    continue
                tr = Track(seg.t.to_numpy(), seg.lat.to_numpy(), seg.lon.to_numpy(), seg.alt_m.to_numpy())
                t_rec, ratio, _, tcpa, rmin = tr.curve(r.lat, r.lon)
                if rmin > dmax or abs(tcpa - when) > tol.offset_s:
                    continue
                vr = np.interp(tcpa + rel, t_rec, C / ratio - C)
                shifts = np.arange(-tol.lock_slack_s, tol.lock_slack_s + 0.01, 0.5)
                sc, _ = _sweep(spec, tcpa - r.start + rel, vr, shifts, ks, f0s, e.harmonics)
                if sc > extra_scores[i, j]:
                    extra_scores[i, j], extra_icao[i, j] = sc, icao
    for j in range(len(extra)):
        best[f"x{j + 1}"] = extra_scores[:, j]
        best[f"xicao{j + 1}"] = extra_icao[:, j]
    best["base"] = best["score"]
    best["score"] = best["base"] + extra_scores.sum(1)
    return best.sort_values("score", ascending=False).reset_index(drop=True)


def df_for(df, hits, n):
    """Only the tracks scoring needs, to keep worker start-up cheap."""
    want = set().union(*(set(hits[f"icao{i}"]) for i in range(n)))
    return df[df.icao.isin(want)]


_W = {}


def _init_worker(spec, events, df, elev):
    _W.update(spec=spec, events=events, df=df, elev=elev)


def _score_job(job):
    """Score points for one aircraft combination. `split` > 0 expands each coarse point into 4 at split/2 offsets."""
    jid, key, lats, lons, tol, split = job
    spec, events, df, elev = (_W[k] for k in ("spec", "events", "df", "elev"))
    tracks = [_track(df, icao, t) for icao, t in key]
    off = (split / 2) / DEG
    subs = ((-1, -1), (-1, 1), (1, -1), (1, 1)) if split else ((0, 0),)
    rows = []
    for la, lo in zip(lats, lons):
        k = np.cos(np.radians(la))
        for dla, dlo in ((a * off, b * off / k) for a, b in subs):
            e0 = 0.0 if elev is None else float(elev(np.array([la + dla]), np.array([lo + dlo]))[0])
            tot, per, start = score_point(spec, tracks, events, la + dla, lo + dlo, tol, e0)
            rows.append({"lat": la + dla, "lon": lo + dlo, "score": tot, **{f"s{i + 1}": p for i, p in enumerate(per)},
                         "start": start, **{f"icao{i + 1}": a for i, (a, _) in enumerate(key)}, "job": jid})
    return rows


def locate(spec: Spec, events: list[Event], df: pd.DataFrame, bbox, step=200.0, fine=100.0,
           tol: Tolerances = Tolerances(), elev=None, log=print, workers=10, start=None,
           refine=300, per_combo=8) -> pd.DataFrame:
    pts = Points.grid(bbox, step)
    glat, glon = pts.lat, pts.lon
    log(f"grid: {len(pts):,} points at {step:.0f} m")
    cands = []
    for i, e in enumerate(events):
        c = candidate_passes(df, e, pts, tol, elev, start)
        log(f"event {i + 1}: {len(c):,} (point, pass) candidates from {c.icao.nunique() if len(c) else 0} aircraft")
        cands.append(c)
    hits = offset_filter(cands, events, tol)
    log(f"offset filter: {len(hits):,} hits at {hits.g.nunique() if len(hits) else 0:,} points")
    if len(events) > 1:
        hits = silence_filter(hits, df, events, pts, tol, elev)
        log(f"silence filter: {len(hits):,} hits at {hits.g.nunique() if len(hits) else 0:,} points")
    if hits.empty:
        return pd.DataFrame()
    n = len(events)
    # crop the spectrogram to what scoring can touch, so workers get a small copy
    fmax = max(e.f0 * max(e.harmonics) for e in events) * (1.6 + tol.f0_rel)
    fmin = min(e.f0 for e in events) * (0.6 - tol.f0_rel)
    small = spec.crop(events[0].tc - 300, events[-1].tc + 300, fmin, fmax)
    keys = [tuple((getattr(r, f"icao{i}"), getattr(r, f"t{i}")) for i in range(n)) for r in hits.itertuples()]
    hits = hits.assign(key=[tuple((a, round(t / 600)) for a, t in k) for k in keys], key_t=keys)
    fast = replace(tol, airspeed=(1.0, 1.0), airspeed_steps=1)
    jobs = [(jid, grp.key_t.iloc[0], glat[grp.g.to_numpy()], glon[grp.g.to_numpy()], fast, fine)
            for jid, (_, grp) in enumerate(hits.groupby("key", sort=False))]
    log(f"scoring {len(hits):,} hits in {len(jobs):,} aircraft combinations (no wind)")
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(small, events, df_for(df, hits, n), elev)) as ex:
        first = pd.DataFrame([r for part in ex.map(_score_job, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
                              for r in part]).sort_values("score", ascending=False)
        # stage 2: the best points of the top combinations, with the wind (airspeed) sweep
        keep = first.drop_duplicates("job").head(refine)["job"]
        top = first[first.job.isin(keep)].groupby("job", sort=False).head(per_combo)
        jobs2 = [(jid, jobs[jid][1], g.lat.to_numpy(), g.lon.to_numpy(), tol, 0) for jid, g in top.groupby("job", sort=False)]
        log(f"refining {len(top):,} points of the top {len(jobs2)} combinations with wind")
        rows = [r for part in ex.map(_score_job, jobs2, chunksize=max(1, len(jobs2) // (workers * 4))) for r in part]
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    log(f"scored {len(first):,} points at {fine:.0f} m; refined {len(out):,}")
    return out
