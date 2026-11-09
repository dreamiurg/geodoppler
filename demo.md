# pinpoint demo: where was this video recorded?

The tool hears small aircraft passing the microphone. Each flyover is a falling Doppler tone. It matches
those tones against public ADS-B flight history (adsb.lol) and finds the one spot on the ground where the
real aircraft tracks would sound exactly like the recording. It reads no title, description, tags, GPS or
frames: only the audio, a date, and a search area.

## What the tool was given, and how we check it

- **Audio:** a 2-hour daytime slice of the recording.
- **Date:** the day of the recording (the livestream's day).
- **Search area:** a wide circle around the region the uploader names ("Northeast Ohio" -> a 150 km circle,
  about 70,000 km²). It does **not** get the address, the time of day, or any hand tuning: flyovers are
  auto-detected.
- **Check 1, place:** the answer must fall where the uploader says the camera is.
- **Check 2, time:** the tool also works out when the recording started, to the second. For livestream
  archives YouTube knows the real go-live time, which the tool never sees. A right answer gets both.

## Results that pass both checks

The feeder cams are private homes, so their coordinates here are rounded to 0.01° (about 1 km). The distances
between answers were measured on the unrounded values.

| Video | Views | Search area | Answer | Place check | Time check (found / real) | Aircraft heard | Margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Chickadee's LIVE Bird Feeder - Northeast Ohio, Sep 30 2026](https://www.youtube.com/watch?v=U1bQZeogcsU) | 264 | 150 km circle over NE Ohio | 41.11, -81.57 (Akron / Cuyahoga Falls) | in NE Ohio, as stated | 13:14:01 / 13:14:10 UTC (9 s) | Mooney M20, C-130J, 2x Cessna 172 | 63 % |
| [Chickadee's LIVE Bird Feeder - Northeast Ohio, Oct 28 2026](https://www.youtube.com/watch?v=GPLUdm_Q9NQ) | 183 | same 150 km circle | 41.11, -81.57 (0.45 km from the Sep 30 answer) | in NE Ohio, as stated | 14:07:30 / 14:07:18 UTC (12 s) | Cessna 172, Piper Malibu Meridian, Piper Arrow | 32 % |
| [Bird & Squirrel Cam, Northern Kentucky, Oct 21 2026](https://www.youtube.com/watch?v=J8yvJJS7qTU) | 599 | 120 km circle over Northern Kentucky | 38.83, -84.77 (Boone County) | in Northern Kentucky, as stated | 14:53:56 / 14:53:58 UTC (2 s) | Saab 2000, Beech Baron, Twin Bonanza | 23 % |
| [Huron, Ohio LIVE Train Camera, Oct 18 2026](https://www.youtube.com/watch?v=u8qudttB8PI) | n/a | 300 km circle | 41.3876, -82.5579 | 0.26 km from the camera | 11:46:50 / 11:46:17 UTC (33 s) | 5 light aircraft | 15 % (50 % at 100 km) |
| [colsto raw audio](https://www.youtube.com/watch?v=4CxVxuMVkMc) (the video that inspired this) | n/a | 2°x2° box | 40.87, -73.52 | matches the video | 20:17:28 UTC, matches the video | Cessna 208, Pilatus PC-12, ... | 18 % |

### Same backyard, different day

Blind, same 150 km search: the Ohio cam's [Oct 27 2026 archive](https://www.youtube.com/watch?v=UsyJgZpSuus)
lands at 41.11, -81.57, **0.2 km from the Sep 30 answer**, found start 16:55:31 vs real 16:55:38 UTC
(7 s), from two other aircraft (Cessna 177RG, King Air). The tool rated it `low` (11 %, only 2 flyovers),
so it would not have claimed it; both checks pass anyway.

The Ohio cam's Oct 19 archive, searched only within 15 km of the Sep 30 answer (a confirmation search, not
a blind one), lands 0.96 km from the Sep 30 spot with a found start of 13:51:12 vs the real 13:51:18 UTC
(6 s), from three different aircraft (Piper Saratoga, Cessna 172RG, Grumman AA-5), margin 34 %. The same
check on Oct 8 is inconclusive (`low`, 8 %).

## Honest funnel (Nov 8 2026 search for demo videos)

- ~200 YouTube results for home-cam and backyard phrases; most were AI music/ambience loops.
- 52 real US outdoor recordings screened; 36 had 2+ flyovers; 20 were searched blind (one crashed).
- 4 new recordings passed both checks: 3 days of the Ohio bird feeder (all within 0.5 km of each other,
  three different sets of aircraft) and 1 day of the Kentucky bird feeder.
- Wrong answers, and how they were caught:
  - Same Ohio cam, Oct 19: said "high" (19 %) but landed near Pittsburgh, matched 3 medical helicopters,
    and the start time was 6 h off. Caught by the time check.
  - Green Lake WI weather cam, Oct 27: said "high" (32 %) but landed in northern Illinois, start 1.5 h off.
    Caught by the time check.
  - North Carolina (2 days), North Georgia, Oklahoma, Wisconsin, Michigan, Boston, Kentucky (Sep 6, 8,
    10, Oct 15), Ohio (Oct 8, Feb 1): all flagged `low`.
- Of the 5 blind `high` answers, 3 passed the time check and 2 failed it.
- So: "high" alone is not proof at wide search areas. "High" plus a start time that matches the real
  go-live time is.

## Why most videos fail

- No aircraft: most recordings never hear a small plane within ~8 km in a 2-hour window.
- Jets: near airline airports (Northern Kentucky under CVG) the clean tones are jets on approach. The tool
  ignores jets today (too many look-alike tracks on the same approach path).
- Edited or looped audio: a "1 hour" Boston backyard recording was a 30-minute loop played twice.
- Unknown date: home recordings rarely say what day they were made; the upload date is often wrong.
