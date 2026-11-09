"""For each held-out case: is a candidate with start within 60 s of the truth in candidates/verified.parquet?
usage: python rank.py EXPDIR   (EXPDIR = a heldout.py variant dir, e.g. /Volumes/DevScratch/pinpoint/exp/heldout/main)"""
import json
import sys
from pathlib import Path

import pandas as pd

LAB = {c["run"]: c for c in json.loads((Path(__file__).resolve().parents[2] / "experiments/dataset/labeled.json").read_text())}


def truth_unix(c):
    """unix time of spectrogram t=0 if the video starts at go-live"""
    return pd.Timestamp(c["release_timestamp_utc"]).timestamp() + c["section_start_s"]


def rank(d: Path):
    c = LAB[d.name]
    truth = truth_unix(c)
    out = {}
    for f in ("candidates", "verified"):
        if not (d / f"{f}.parquet").exists():
            continue
        x = pd.read_parquet(d / f"{f}.parquet").reset_index(drop=True)
        err = x.start - truth
        ok = x[err.abs() <= 60]
        r = {"n": len(x), "best": float(x.score[0]), "nearest_s": float(err.abs().min())}
        if len(ok):
            i = ok.index[0]
            r.update(row=int(i + 1), combo=int(x.iloc[:i + 1].drop_duplicates(["icao1", "icao2"]).shape[0]),
                     score=float(x.score[i]), err=float(err[i]), icao=f"{x.icao1[i]}+{x.icao2[i]}")
        out[f] = r
    return out


if __name__ == "__main__":
    exp = Path(sys.argv[1])
    for d in sorted(p for p in exp.iterdir() if p.is_dir() and p.name in LAB):
        r = rank(d)
        parts = [d.name]
        for f, v in r.items():
            if "row" in v:
                parts.append(f"{f[:4]}: row {v['row']}/{v['n']} combo#{v['combo']} {v['score']:.3f} vs {v['best']:.3f} "
                             f"err {v['err']:+.0f} {v['icao']}")
            else:
                parts.append(f"{f[:4]}: none of {v['n']} (nearest {v['nearest_s']:.0f} s)")
        print(" | ".join(parts))
