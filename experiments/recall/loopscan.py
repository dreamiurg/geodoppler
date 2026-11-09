"""Scan every case for looped audio: coarse (2 s x 10 Hz-band) spectrogram autocorrelation, lags 60 s-1 h.
A looped recording shows one lag far above the rest (qoUTMghkS1k: 0.79 at 2481.5 s at full resolution)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pinpoint import cli  # noqa: E402

from rank import LAB  # noqa: E402

RUNS = (Path(__file__).resolve().parents[2] / "runs")
for run, c in LAB.items():
    if not c["adsb_cached"] or not (RUNS / run / "spec.npz").exists():
        continue
    sp = cli.load_spec(RUNS / run).crop(0, 1e9, 40, 1000)
    S = sp.S.astype(np.float32)
    nf, nt = S.shape[0] // 4 * 4, S.shape[1] // 4 * 4
    S = S[:nf, :nt].reshape(nf // 4, 4, nt // 4, 4).mean((1, 3))  # (bands, 2 s frames)
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    lags = np.arange(30, min(1800, S.shape[1] // 2))
    cor = np.array([(S[:, :-k] * S[:, k:]).mean() for k in lags])
    i = int(np.argmax(cor))
    z = (cor[i] - np.median(cor)) / (np.median(np.abs(cor - np.median(cor))) * 1.4826 + 1e-9)
    print(f"{run:12} peak corr {cor[i]:.3f} at {lags[i] * 2 * (sp.t[1] - sp.t[0]) * 2:.0f} s  robust z {z:.0f}"
          f"{'  <-- LOOP?' if cor[i] > 0.1 else ''}", flush=True)
