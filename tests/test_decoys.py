from pinpoint import cli


def test_decoy_days_nearest_cached_only_never_the_recordings_days(tmp_path, monkeypatch):
    """Only days with a props.parquet count (an uncached decoy would download ~4 GB); with --days 2 a decoy needs
    its next day cached too and must not overlap the recording's two days."""
    for d in ("2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-11-03", "2026-11-04"):
        (tmp_path / "adsb" / d).mkdir(parents=True)
        (tmp_path / "adsb" / d / "props.parquet").touch()
    (tmp_path / "adsb" / "2026-10-30").mkdir()  # traces only, no parquet: not cached
    monkeypatch.setattr(cli, "DATA", tmp_path)
    assert cli.decoy_days("2026-10-27", 3) == ["2026-10-26", "2026-10-28", "2026-10-25"]
    assert cli.decoy_days("2026-10-27", 9) == ["2026-10-26", "2026-10-28", "2026-10-25", "2026-10-29", "2026-11-03", "2026-11-04"]
    assert cli.decoy_days("2026-10-27", 9, days=2) == ["2026-10-25", "2026-11-03"]
