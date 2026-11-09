# pinpoint

Find where a video was recorded from the sound of small aircraft flying past it.

A propeller aircraft passing within a few km leaves a falling "Doppler curve" in the audio, often
inaudible but clear in a spectrogram. Its shape gives the aircraft's speed and closest distance; the gap
between two curves gives a time offset. Public ADS-B history (adsb.lol) records every transponder-equipped
aircraft's track. pinpoint searches a grid of possible observer locations for the place where real tracks
reproduce the observed curves, then models the exact curve each real track would make there and scores
it against the spectrogram. Method after colsto's video
[I discovered a sound that leaks your exact location](https://youtu.be/UytYrhYLOKI).

## Use

```bash
uv sync
uv run pinpoint run 'https://youtu.be/4CxVxuMVkMc' --date 2026-09-08 --bbox 40.9,-73.5,120
```

Step by step (what the Claude skill in `.claude/skills/pinpoint` does):

```bash
uv run pinpoint fetch URL                       # runs/<id>/: meta.json, audio, overview/*.png spectrograms
uv run pinpoint events runs/<id>                # auto-detect flyovers -> events.json, events.png
uv run pinpoint events runs/<id> --box 2080,2240,80,190,1+2 --box 2990,3140,95,185   # or pick them
uv run pinpoint locate runs/<id> --date YYYY-MM-DD --bbox lat0,lon0,lat1,lon1   # or lat,lon,km | conus
uv run pinpoint locate runs/<id> --bbox conus --start 2026-09-08T20:17:30       # one event + known time
uv run pinpoint frames runs/<id>                # stills, for visual clues when there is no aircraft audio
uv run pinpoint pin runs/<id> LAT LON --note "..."   # map a location judged from other clues
```

Output in `runs/<id>/`: `map.html` (Leaflet, satellite, candidate heat, matched flight tracks),
`result.json` (best lat/lon, recording start time to the second, aircraft heard, score margin).

Needs `yt-dlp` and `ffmpeg` on PATH. ADS-B days cache under `$PINPOINT_DATA` (default
`~/.cache/pinpoint`): first use of a day downloads ~4 GB and keeps a ~700 MB parquet of non-jet tracks.

## Pipeline

1. **Events** (`doppler.py`): whitened spectrogram with steady lines suppressed; sliding-window fit of an
   ideal straight-line flyover (f0, speed, slant distance, CPA time) by differential evolution;
   visible harmonics detected and scored too.
2. **Candidates** (`search.py`): 200 m grid; for each event, track segments whose ground speed fits; for every
   grid point in reach, delayed CPA time (CPA + slant/c) and slant distance. Jets and rotorcraft (known ICAO
   types) are dropped up front.
3. **Filters**: CPA-time offset between the two strongest events (±10 s); silence (no other non-jet passes
   close in between, ignoring ±60 s around each event's CPA).
4. **Scoring**: per surviving point (100 m), the emission-time Doppler curve of each real track
   (`flyover.py`), brightness summed along it. Event 1 absorbs measurement error (±6 s, ±4 % f0); event 2 is
   locked to geometry (±2 s). Stage 1 at the plain radial velocity; stage 2 re-scores the best points of every
   aircraft combination with a ±15 % radial-velocity scale sweep. That scale absorbs sound-speed and geometry
   error, not wind: Doppler follows ground speed, and wind only shifts the effective sound speed (~W/c).
5. **Verification**: every other detected event must be explained, time-locked, by some real aircraft at
   the candidate; that adds to the score (0 if none, so a spurious event costs nothing).
6. **Confidence**: `high` needs a >= 15 % margin over the next aircraft combination and every primary event
   explained at a Doppler scale within 1 ± 0.08 (`doppler_scale` in result.json); right matches sit at 0.94-1.05.

## Validation

22 public YouTube recordings with a known place and day (`validation/candidates.json`), searched within
100 km of the truth on the known UTC day, start time unknown (`validation/run.py`). Most recordings carry
no usable aircraft sound, so the useful number is "how often it is right when it answers, and does it
know when it is wrong".

| Recording | Flyovers | Result | Error | Margin -> confidence |
| --- | --- | --- | --- | --- |
| colsto raw audio (4CxVxuMVkMc), 2°×2° box | 3 | located, start 20:17:28 UTC | matches the video (truth given to 0.01°) | 18 % -> high |
| Huron OH railcam, Oct 18 2026 (u8qudttB8PI) | 8 | located, 5 aircraft matched | 0.21 km | 50 % -> high |
| Huron OH railcam, Aug 15 2026 (cq68vWhv8DQ) | 2 | wrong spot | 25 km | 6 % -> low |
| Panama City Beach boat ramp (7IrHU1tydh4) | 2 | wrong spot (boat Doppler?) | 105 km | 8 % -> low |
| Huron OH railcam, Sep 1 2026 (SemiF2yUFzk) | 1 | wrong spot; now refused | 69 km | 0.8 % -> low |
| colsto audio, 1 event + known start, all CONUS | 1 | wrong state (Kansas) | ~2200 km | 0.6 % -> low |
| Viroqua WI field recording (MjWQyWLvoVw) | 4 | no match (recording date unsure) | | |
| Oshkosh 2025 arrivals (x2sVBsOfsmI) | 12 | wrong spot (landing traffic) | 22 km | 0.04 % -> low |
| 6 airport / beach webcams, 2 field recordings, 2 railcam days | 0 | no flyovers detected | | |
| 6 webcam archives | | silent audio track | | |

In this 100 km set every `high` answer was right and every wrong answer was flagged `low` (only 2 high
answers). The wider 150-300 km searches in `demo.md` broke that: two `high` answers (19 % and 32 %) were
wrong, one of them matched medical helicopters. There, a found recording start that matches a livestream's
real go-live time is what separates right from wrong. After dropping rotorcraft, widening the silence guard,
sweeping every combination and adding the Doppler-scale check, none of 23 labeled recordings
(`validation/eval.py`) gets a wrong `high`; the helicopter case is found at 150 km (0.9 km off, `low`). Airport plane-spotting streams fail: ground operations and landings are
not straight constant-speed flyovers. Full table: `validation/results.md`.

## Limits

- Needs unedited audio with at least one non-jet flyover within ~8 km (two for an unknown time).
  Jets low on approach do make clean Doppler tones, but they are excluded: too many look-alike tracks.
- Coverage = adsb.lol receivers (dense in US/Europe), history from 2023.
- Ground elevation comes from Copernicus GLO-90 (downloaded per 1° tile on first use).
- One UTC day of tracks per search; a recording across midnight UTC needs `--days 2`.
- One flyover is not enough without a known start time, and not enough for a CONUS-wide search even with one.
