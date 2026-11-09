"""Write experiments/recall/cases.txt: one row per held-out case with the evidence behind its failure class."""
import json
import re
from pathlib import Path

import pandas as pd

from rank import LAB, rank, truth_unix  # noqa: F401

HERE = Path(__file__).parent
D = Path("/Volumes/DevScratch/pinpoint/exp/recall")
MAIN = Path("/Volumes/DevScratch/pinpoint/exp/heldout/main")
RUNS = Path("/Users/dreamiurg/src/dreamiurg/pinpoint/runs")
given = {r["run"]: r for r in json.loads((D / "start60" / "table.json").read_text())}
nullp = {}
for line in (D / "null_rank.txt").read_text().splitlines():
    m = re.match(r"(\S+)\s+\S+ truth combo#(\S+)\s+fake truths with a hit: (\d+)/40\s+P\(fake rank <= truth rank\) = (\S+)", line)
    if m:
        nullp[m[1]] = (m[2], m[4])
gates = json.loads((D / "gates.json").read_text())

LOOP = {"qoUTMghkS1k": 2481, "vbPMpKPCs7M": 2481, "LajMZuJBPLA": 2836}
CAUSE = {  # class: evidence (see REPORT.txt)
    "qoUTMghkS1k": ("A loop", "audio repeats every 2481.5 s (corr 0.68-0.79); the 3 'flyovers' are one sound replayed"),
    "vbPMpKPCs7M": ("A loop", "same 2481.5 s loop as qoUT; time-given answer even reads high (ratio 1.22)"),
    "LajMZuJBPLA": ("A loop", "audio repeats every 2836 s (corr 0.75)"),
    "3xPxFS5evIM": ("B speed", "ev2 fit v=172: no non-jet track within +-30 m/s at the truth time (52 with speed open)"),
    "Xl3klV87YNs": ("B speed", "ev2 fit v=124: 3 non-jet tracks in range, no pair; 53 with speed open; level -74 dBFS"),
    "HuYzXxa5gNE": ("B speed", "ev1 fit v=139 (close harmonic comb, fit off): 0 tracks in range, 104 with speed open"),
    "vDHQcL2Hvo0": ("B speed", "ev1 fit v=168; 13 events look like broadband noise/clicks; 0 tracks in range"),
    "Bf2uUbj8qvE": ("B speed", "(no answer in main) 35 events, search pair v=135/124: 0 tracks in range for both"),
    "dNs5EMP91vc": ("C no pair", "tracks exist for both events (13/26) but none time-locked; not in ADS-B, or misfit"),
    "cv3itW_Q6H4": ("C no pair", "tracks exist (12/29), no time-locked pair; with rotorcraft 98 offset points, silence kills them"),
    "Kz69coKyH9Q": ("D outranked", "truth pair combo#14 (null P 0.05); spot 1.5 km from wTRF12TtDsQ's correct answer, same aircraft a883cc; "
                    "lost a 0.75 extra (PC12) to the verify box clip -> FIXED by whole-track verify"),
    "TbVt7uWmnlQ": ("D outranked", "truth pair combo#7 (null P 0.03); lost a C208 extra to the verify box clip -> FIXED by whole-track verify"),
    "AxxRAFZS_5Y": ("D outranked", "truth pair combo#15 (P 0.05); same pair+spot as yAsLE5tgxZE (shared audio); extras: no aircraft"),
    "yAsLE5tgxZE": ("D outranked", "truth pair combo#10 (P 0.03); same pair+spot as AxxRAFZS_5Y"),
    "iSYDIGV3Ppg": ("D outranked", "truth pair combo#32 (P 0.05); time-given spot ~360 km from Axx/yAs although the 4 streams hear "
                    "the same events: at most one camp is right"),
    "3AELK17k8lg": ("D outranked", "truth pair combo#30 (P 0.07); k 0.91/1.15; spot disagrees with Axx/yAs (same events)"),
    "aJcOrb0Sq-A": ("D outranked", "truth pair combo#10 (P 0.03); 1 extra, no aircraft near it"),
    "b4xV0V7oHe0": ("D outranked", "truth pair combo#8 (P 0.03); k 1.15; several fits hit the d=8 km bound; P180 extra at 13.8 km "
                    "fails the 11.5 km slant gate"),
    "WBZAVCwDe3o": ("D outranked", "same pair as main's #2 combo but another point: verify only scores each combination's best "
                    "point; time-given point verifies to 2.62 > winner 2.49 (verifying 8 points/combo did not fix it)"),
    "-CKmIa-V-Iw": ("E chance", "truth pair combo#30 (P 0.15); time-given ratio 1.52 but k 1.03/0.86"),
    "JWKdGun5GRo": ("E chance", "truth pair combo#23 (P 0.12), k 0.82/1.17"),
    "7NzTIUwkPE8": ("E chance", "truth pair combo#101 (P 0.38)"),
    "9O6hpXOevBM": ("E chance", "truth pair combo#106 (P 0.33); time-given k 0.84/1.22"),
    "DhAIJYgqmMc": ("E chance", "combo#206 (P 0.57); level -81 dBFS; time-given k 0.80/0.90"),
    "e3hz7EunI18": ("E chance", "combo#235 (P 0.62); 4 near-identical events every 244 s (circuits? same-icao pairs are "
                    "forbidden; no matching circuit aircraft found at 75 m/s)"),
    "d2EqZDmIw_k": ("E chance", "combo#253 (P 0.62)"),
    "DbcjVt0lm2E": ("E chance", "combo#202 (P 0.45); time-given k 0.84/0.82"),
    "YOTJzpyb6cc": ("E chance", "no truth candidate in verified (cand combo#504); time-given k 1.25/1.09"),
    "V5AuHCx1qCw": ("E chance", "no truth candidate in verified; time-given pair at combo#229 in main with err +119"),
    "D5KjAKFEx-Q": ("E chance", "no truth candidate in verified; time-given ratio 0.82"),
    "Ane3idAqQac": ("E chance", "time-given pair absent from main: a6f737's full-day segment keeps only its global CPA "
                    "(1459 s earlier) - passes() one-CPA-per-pass shortcut; time-given ratio 0.85 anyway"),
    "Cs4nYfpcC0E": ("ok", "correct in main at a 0.7 % margin; whole-track verify gives a wrong pair +0.16 of tiny extras (0.14, "
                    "0.02) and it wins 4.37 vs 4.34"),
    "LYoUoV2jRL8": ("F crash", "exit 70 = 949M-row offset join (OOM) -> FIXED; time-given: 2.18, ratio 1.17, err -4"),
}

rows = []
for run, c in LAB.items():
    if not c["adsb_cached"] or not (RUNS / run / "spec.npz").exists():
        continue
    ev = json.loads((RUNS / run / "events.json").read_text())
    top = sorted(ev, key=lambda e: -e["score"])[:2]
    r = MAIN / run / "result.json"
    if r.exists():
        res = json.loads(r.read_text())
        err = pd.Timestamp(res["recording_start_utc"]).timestamp() - pd.Timestamp(c["release_timestamp_utc"]).timestamp()
        main = f"{'CORRECT' if abs(err) <= 60 else 'wrong':7} {err:+7.0f} {res['confidence']:4}"
    else:
        main = f"{'none':7} {'':7} {'':4}"
    g = given.get(run, {})
    gs = (f"{g['given']:.2f} err {g['err']:+4.0f} ratio {g['ratio'] if g['ratio'] is not None else float('nan'):.2f} "
          f"k {g['k']} {g['conf']}") if isinstance(g.get("given"), float) else str(g.get("given", "-"))
    cause, note = CAUSE.get(run, ("ok", "correct in main"))
    w = Path("/Volumes/DevScratch/pinpoint/exp/heldout/whole") / run / "result.json"
    if w.exists():
        we = pd.Timestamp(json.loads(w.read_text())["recording_start_utc"]).timestamp() - pd.Timestamp(c["release_timestamp_utc"]).timestamp()
        after = "CORRECT" if abs(we) <= 60 else "wrong"
    else:
        after = "none"
    nr = tuple("-" if x == "None" else x for x in nullp.get(run, ("-", "-")))
    rows.append(f"{run:12} {len(ev):3} {'/'.join(str(round(e['v'])) for e in top):7} {main} | {gs:44} | "
                f"{nr[0]:>4} {nr[1]:>4} | {after:7} | {cause:11} | {note}")
hdr = (f"{'run':12} {'ev':>3} {'v1/v2':7} {'main':7} {'err_s':>7} {'conf':4} | {'time-given (--start go-live --start-window 60 --decoys 5)':44} | "
       f"{'#tr':>4} {'nullP':>4} | {'fixed':7} | {'class':11} | evidence")
txt = ("Per-case evidence for experiments/recall/REPORT.txt. v1/v2 = fitted speeds (m/s) of the two search events.\n"
       "#tr = truth pair's combination rank in main verified.parquet (candidates if not verified); nullP = share of 40 fake\n"
       "truths (go-live +-10..200 min) that get a candidate at that rank or better. Classes: A loop = looped audio;\n"
       "B speed = no non-jet track at a search event's fitted speed at the truth time; C no pair = tracks exist, none\n"
       "time-locked; D outranked = a truth-time pair with signal (nullP <= 0.07) loses; E chance = truth-time pair no\n"
       "better than chance; F crash. fixed = verdict with both fixes (heldout variant whole).\n\n" + hdr + "\n" + "\n".join(rows) + "\n")
(HERE / "cases.txt").write_text(txt)
print(txt)
