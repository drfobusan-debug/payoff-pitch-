"""The reader that answers the only question phase one exists to ask."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cfb_engine.audit.availability import read_log, summarize
from cfb_engine.data.injuries import InjuryRow, NewsItem, log_availability

SEEN = datetime(2026, 9, 3, 15, 0, tzinfo=timezone.utc)


def _row(team: str = "iowa") -> InjuryRow:
    return InjuryRow("Cade McNamara", "9", team, "QB", "Out", "Knee", "Oct 4th", "")


def _write(path, *, spread, when: datetime, news: datetime | None, out_side="home"):
    """Log Nebraska @ Iowa, with the absence on ``out_side``."""
    log_availability(
        path,
        home="Iowa",
        away="Nebraska",
        rows=[_row("iowa" if out_side == "home" else "nebraska")],
        spread=spread,
        news={"9": NewsItem("9", "Out Saturday", news)} if news else {},
        observed=when,
    )


def test_repeat_sightings_collapse_into_one_row(tmp_path):
    path = tmp_path / "availability.jsonl"
    _write(path, spread=-6.5, when=SEEN, news=SEEN - timedelta(hours=2))
    _write(path, spread=-4.0, when=SEEN + timedelta(hours=6), news=SEEN - timedelta(hours=2))
    (sighting,) = read_log(path)
    assert sighting.observations == 2
    assert sighting.lead_s == 7200.0
    # Home team lost him and the home spread rose 2.5: the market moved against
    # them after we already knew, so those points were still available.
    assert sighting.move_after == 2.5
    assert sighting.team_is_home


def test_the_sign_follows_the_short_handed_team(tmp_path):
    """An away absence moves the home spread the other way; the reading must not flip."""
    path = tmp_path / "availability.jsonl"
    _write(path, spread=-6.5, when=SEEN, news=None, out_side="away")
    _write(path, spread=-9.0, when=SEEN + timedelta(hours=4), news=None, out_side="away")
    (sighting,) = read_log(path)
    assert not sighting.team_is_home
    assert sighting.move_after == 2.5


def test_a_market_that_never_moves_reads_as_zero(tmp_path):
    path = tmp_path / "availability.jsonl"
    _write(path, spread=-6.5, when=SEEN, news=SEEN - timedelta(minutes=30))
    _write(path, spread=-6.5, when=SEEN + timedelta(hours=8), news=None)
    (sighting,) = read_log(path)
    assert sighting.move_after == 0.0
    assert "median +0.00 pts" in summarize([sighting])


def test_summary_reports_lead_time_and_leftover_movement(tmp_path):
    path = tmp_path / "availability.jsonl"
    _write(path, spread=-3.0, when=SEEN, news=SEEN - timedelta(hours=1))
    _write(path, spread=-1.0, when=SEEN + timedelta(hours=5), news=None)
    text = summarize(read_log(path))
    assert "1 absences logged, 1 with a posting time." in text
    assert "median +1.0h" in text
    assert "1/1 still moved 0.25+ against the team" in text


def test_missing_and_corrupt_rows_do_not_stop_the_read(tmp_path):
    path = tmp_path / "availability.jsonl"
    assert read_log(path) == []
    _write(path, spread=-3.0, when=SEEN, news=None)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n\n[1,2]\n")
    assert len(read_log(path)) == 1
    assert summarize([]) == "No absences logged yet."


def test_one_sighting_has_no_movement_to_report(tmp_path):
    path = tmp_path / "availability.jsonl"
    _write(path, spread=None, when=SEEN, news=None)
    (sighting,) = read_log(path)
    assert sighting.move_after is None
    assert "no absence has been seen twice yet" in summarize([sighting])
