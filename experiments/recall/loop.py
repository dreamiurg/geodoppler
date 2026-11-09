"""Is the audio a loop? Correlate the whitened spectrogram with itself at lags around a suspected period,
over the whole slice (events included and excluded). A looped recording gives a sharp peak at one lag; a
periodic real source (machine, aircraft circuit) gives a peak only near the events.
usage: python loop.py RUN LAG_LO LAG_HI"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import cli  # noqa: E402

run, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = Path("/Users/dreamiurg/src/dreamiurg/pinpoint/runs") / run
sp = cli.load_spec(d)
evs = cli.load_events(d)
S = sp.crop(0, sp.t[-1], 40, 1000).S.astype(np.float32)
S = (S - S.mean(0)) / (S.std(0) + 1e-6)  # per-frame normalisation
dt = sp.t[1] - sp.t[0]
near = np.zeros(S.shape[1], bool)
for e in evs:
    near |= np.abs(sp.t[:S.shape[1]] - e.tc) < 120
res = []
for lag in np.arange(lo, hi, dt):
    k = int(round(lag / dt))
    a, b = S[:, :-k], S[:, k:]
    c = (a * b).mean(0)  # per-frame correlation
    quiet = ~near[:-k] & ~near[k:]
    res.append((lag, c.mean(), c[quiet].mean() if quiet.any() else np.nan))
res = np.array(res)
i = np.nanargmax(res[:, 2])
print(f"{run}: lags {lo}-{hi}s  best quiet-frame corr {res[i, 2]:.3f} at {res[i, 0]:.1f}s "
      f"(median {np.nanmedian(res[:, 2]):.3f}); best all-frame {res[:, 1].max():.3f} at {res[np.argmax(res[:, 1]), 0]:.1f}s")
