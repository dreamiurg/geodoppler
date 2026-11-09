---
name: pinpoint
description: Find where a YouTube video (or any recording) was filmed and show it on a map. Use when the user gives a video URL and asks where it was taken, to geolocate a clip, or says "pinpoint". Primary method matches faint propeller-aircraft Doppler curves in the audio against public ADS-B flight history (works with zero visual clues); falls back to metadata and visual clues.
---

# pinpoint

Run every command from the repo root with `uv run pinpoint ...`. This file lives in `.claude/skills/pinpoint/` of the
repo; if the skill is symlinked from elsewhere, resolve the link to find the root.
Work happens in `runs/<video_id>/`. ADS-B days cache in `~/.cache/pinpoint/adsb/<day>` (~4 GB download + 40 s build per new day).

Do not ask the user questions mid-run. Decide, state the assumption in the final report.

## 1. Fetch and read the clues

```bash
uv run pinpoint fetch '<URL>'
```

Read `runs/<id>/meta.json`. Note, in this order of strength:
- `location` (YouTube recording location) -> usually the answer already; still confirm.
- Places, dates, times in title / description / tags. "Filmed September 8th" style hints set `--date`.
- `upload_date` is the fallback date (recordings are often uploaded weeks later: prefer explicit hints).
- Links to the raw/original/full recording ("Link to full recording: ...") -> analyse THAT video instead:
  edited videos with music and narration hide the aircraft and break the time offsets.
- Livestream archives: `release_timestamp` in meta.json is the stream start (UTC). Long streams: analyse
  a daylight 2-hour slice with `fetch URL --section-start SECONDS` (props fly by day).

## 2. Acoustic path (the strong one)

Look at every `runs/<id>/overview/*.png` (10 min each, 20-500 Hz). You are looking for **smooth falling
S-shaped tones lasting 1-3 minutes**, often with a parallel line at 2x the frequency. That is a propeller
aircraft flying past (Doppler). Steady horizontal lines are hum/engines idling: ignore them.

```bash
uv run pinpoint events runs/<id>                       # auto-detect all flyovers
uv run pinpoint events runs/<id> --box 2080,2240,80,190,1+2 --box 2990,3140,95,185
```

`--box t0,t1,fmin,fmax[,1+2]`: seconds into the recording, Hz range of the line's sweep, and which
harmonics are visible. Read `runs/<id>/events.png`: the cyan model must sit on the ridge. If it sits on a
flat line or noise, tighten the box and refit. Keep every real flyover: `locate` searches with the two
strongest and uses the others to verify candidates (each extra explained event raises the true spot).

Then search. Region from the clues; tighter is faster and safer:

```bash
uv run pinpoint locate runs/<id> --date 2026-09-08 --bbox 40.9,-73.5,120     # lat,lon,radius_km
uv run pinpoint locate runs/<id> --date 2026-09-08 --bbox 39.9,-74.5,41.9,-72.5
uv run pinpoint locate runs/<id> --date 2026-09-08 --bbox conus               # hours; run in background
```

Recording that may cross midnight UTC: add `--days 2`. Known start time (on-screen clock, livestream
start, phone call): add `--start 2026-09-08T20:17:30` (the video's own start, also with `--section-start`;
`recording_start_utc` and `heard_at_s` in result.json are video time too). One event plus a known time is NOT enough for
`--bbox conus` (tested: wrong state, 0.6 % margin, 5.5 h); it can work inside a small region. One event
with no known time is refused (tested: 69 km off, 0.8 % margin): go to section 3.

Add `--decoys 5` whenever there are >= 3 events: it reruns the same search on the 5 nearest other cached
ADS-B days (never downloads; about 6x the time). Output: `runs/<id>/map.html`, `result.json`,
`candidates.parquet`. `result.json` has `confidence`. With decoys: `high` when `score` is >= 1.2x the best
decoy-day score (`decoy_ratio`, `decoy_scores`), there are >= 3 events, and every primary event fits its
aircraft at a Doppler scale within 1 ± 0.08 (`doppler_scale`), else `low`. A 2-event recording is always
`low` with decoys: the true day only stands out through the extra events. Without decoys: `high` when
`score` beats the best *other aircraft combination* by >= 15 % and the Doppler-scale check passes; that
rule is weak (a wrong answer had a 32 % margin, a right one 11 %). Report a `low` result as "no reliable acoustic
fix" and fall back to section 3; mention the acoustic guess only as a weak hint. Boats, cars and trains
also make Doppler curves; a recording full of them (marinas, roads) gives `low` results.

## 3. Fallback: metadata and visual clues

When there are no Doppler curves, no ADS-B match, or the recording is outside ADS-B coverage:

```bash
uv run pinpoint frames runs/<id>             # 24 stills into runs/<id>/frames/
```

Look at the frames like a geolocator: signs and text, language, road markings, plates, vegetation, terrain,
architecture, sun position, landmarks. Combine with the metadata. Then:

```bash
uv run pinpoint pin runs/<id> 40.87 -73.52 --note "shoreline + water tower in frame 07" --radius 300
```

## 4. Show it

Open `runs/<id>/map.html` (the commands do `open` unless `--no-open`), send it with SendUserFile, and
report: coordinates + Google Maps link, method (acoustic / visual / metadata), confidence and margin,
recording start time (acoustic path gives it to the second), aircraft heard, assumptions (date, region).

## Limits

- Needs the audio to carry aircraft sound: an unedited stretch (cuts break the time offsets) and a
  non-jet flyover within ~8 km. Jets are filtered out. Jets low on approach DO make clean tones (seen under CVG, 107-144 m/s),
  but too many look-alike tracks share the approach path; backyards under airline approaches fail.
- Wide searches (150-300 km): `high` can be wrong (19-32 % margins seen before rotorcraft were dropped and the
  Doppler-scale check added; a 0.9 km position error also trips that check, so a right answer can read `low`). For livestream
  archives, check found `recording_start_utc` against `release_timestamp` after the run; right answers
  matched within 33 s (some up to 16 s before the go-live time). Watch for looped "1 hour" uploads: identical events at a fixed offset.
- ADS-B = adsb.lol community receivers: dense in the US/Europe, thin elsewhere; history from 2023 on.
- Analyses the first 2 hours of audio (PINPOINT_MAX_MINUTES). Searching a 2x2 degree box takes ~4 min in a
  busy area; `conus` takes hours.
