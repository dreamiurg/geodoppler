"""Audio loading and whitened low-frequency spectrograms."""

import subprocess
from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

SR = 4000  # aircraft prop fundamentals sit well below 1 kHz


def load(path: str, sr: int = SR, max_s: float | None = None) -> np.ndarray:
    """Decode any media file to mono float32 at `sr` via ffmpeg (first `max_s` seconds if given)."""
    limit = ["-t", str(max_s)] if max_s else []
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, *limit, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


@dataclass
class Spec:
    t: np.ndarray  # frame centres, seconds from start of recording
    f: np.ndarray  # Hz
    S: np.ndarray  # whitened dB, shape (len(f), len(t)); ~0 is background

    def crop(self, t0, t1, f0, f1) -> "Spec":
        mt = (self.t >= t0) & (self.t <= t1)
        mf = (self.f >= f0) & (self.f <= f1)
        return Spec(self.t[mt], self.f[mf], self.S[np.ix_(mf, mt)])

    def sample(self, t: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Brightness at (t, f) points, nearest bin; NaN outside."""
        dt, df = self.t[1] - self.t[0], self.f[1] - self.f[0]
        it = np.rint((t - self.t[0]) / dt).astype(int)
        jf = np.rint((f - self.f[0]) / df).astype(int)
        ok = (it >= 0) & (it < len(self.t)) & (jf >= 0) & (jf < len(self.f))
        out = np.full(t.shape, np.nan)
        out[ok] = self.S[jf[ok], it[ok]]
        return out


def spectrogram(x: np.ndarray, sr: int = SR, win_s: float = 2.0, hop_s: float = 0.5,
                fmin: float = 20, fmax: float = 600) -> Spec:
    n = int(win_s * sr)
    f, t, P = ss.spectrogram(x, sr, nperseg=n, noverlap=n - int(hop_s * sr), window="hann",
                             nfft=2 * n)
    m = (f >= fmin) & (f <= fmax)
    S = 10 * np.log10(P[m] + 1e-20)
    # Whiten: remove per-frequency median (stationary hum) and per-frame median (broadband bursts).
    S -= np.median(S, axis=1, keepdims=True)
    S -= np.median(S, axis=0, keepdims=True)
    S /= np.median(np.abs(S)) * 1.4826 + 1e-9  # robust z-score
    return Spec(t, f[m], S.astype(np.float32))


def suppress_steady(spec: Spec, span_s: float = 25.0) -> Spec:
    """Remove lines that hold a constant frequency for ~span_s (engines idling, hum),
    keeping sweeping Doppler ridges."""
    from scipy.ndimage import median_filter
    k = max(3, int(span_s / (spec.t[1] - spec.t[0])) | 1)
    return Spec(spec.t, spec.f, spec.S - median_filter(spec.S, size=(1, k), mode="nearest"))
