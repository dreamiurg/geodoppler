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
   grid point in reach, delayed CPA time (CPA + slant/c) and slant distance.
3. **Filters**: CPA-time offset between the two strongest events (±10 s); silence (no other non-jet passes
   close in between).
4. **Scoring**: per surviving point (100 m), the emission-time Doppler curve of each real track
   (`flyover.py`), brightness summed along it. Event 1 absorbs measurement error (±6 s, ±4 % f0); event 2 is
   locked to geometry (±2 s). Stage 1 without wind; stage 2 re-scores the top 300 aircraft combinations
   with a ±15 % airspeed sweep (Doppler follows speed through the air, ADS-B gives ground speed).
5. **Verification**: every other detected event must be explained, time-locked, by some real aircraft at
   the candidate; that adds to the score (0 if none, so a spurious event costs nothing).

## Validation

| Recording | Search area | Result | Truth |
| --- | --- | --- | --- |
| colsto raw audio (4CxVxuMVkMc), 3 flyovers | 2°×2° around NYC | 40.87, −73.52, start 20:17:28 UTC, margin 18 % | 40.87, −73.5 (video) |

See `validation/` for the wider set.

## Limits

- Needs unedited audio with at least one non-jet flyover within ~8 km (two for an unknown time).
- Coverage = adsb.lol receivers (dense in US/Europe), history from 2023.
- Ground elevation is taken as 0 m; inland high terrain biases slant distances.
- One UTC day of tracks per search; a recording across midnight UTC needs both days.
