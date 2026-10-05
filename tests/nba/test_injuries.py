from datetime import date, datetime, timezone
from pathlib import Path

from nba_engine.data import injuries
from nba_engine.data.espn_injuries import FeedRow, latest_feed_path, parse_feed, write_feed
from nba_engine.data.oddsapi import SLATE_TZ

FIX = Path(__file__).parent / "fixtures"
PDF = (FIX / "Injury-Report_2025-12-10_05PM.pdf").read_bytes()


def test_hour_labels_and_url():
    assert injuries.hour_label(11) == "11AM"
    assert injuries.hour_label(12) == "12PM"
    assert injuries.hour_label(17) == "05PM"
    assert injuries.url_for(date(2025, 12, 10), 17).endswith("Injury-Report_2025-12-10_05PM.pdf")


def test_official_report_parses_every_player_under_its_game_and_team():
    rows = injuries.parse_pdf(PDF, report="2025-12-10T17:00")
    assert len(rows) == 32
    assert all(r.team and r.player and r.matchup and r.status in injuries.STATUSES for r in rows)
    wemby = next(r for r in rows if r.player == "Wembanyama, Victor")
    assert (wemby.matchup, wemby.team, wemby.status) == ("SAS@LAL", "San Antonio Spurs", "Out")
    assert wemby.reason == "Injury/Illness - Left Calf; Strain"
    booker = next(r for r in rows if r.player == "Booker, Devin")
    assert (booker.game_date, booker.tip_et, booker.status) == (
        "12/10/2025",
        "07:30",
        "Questionable",
    )
    zion = next(r for r in rows if r.player.startswith("Williamson"))
    assert zion.player == "Williamson, Zion" and "SUBMITTED" not in zion.reason


def test_archive_is_content_addressed(tmp_path):
    day = date(2025, 12, 10)
    assert injuries.archive(tmp_path, day, 17, PDF) is not None
    assert injuries.archive(tmp_path, day, 18, PDF) is None
    assert len(injuries.read_latest(tmp_path, day)) == 32


def test_a_report_republished_in_the_same_hour_is_the_newest(tmp_path):
    day = date(2025, 12, 10)
    first = injuries.archive(
        tmp_path, day, 18, PDF, captured=datetime(2025, 12, 10, 23, 31, tzinfo=timezone.utc)
    )
    second = injuries.archive(
        tmp_path,
        day,
        18,
        PDF + b"\n%republished\n",
        captured=datetime(2025, 12, 10, 23, 36, tzinfo=timezone.utc),
    )
    assert first is not None and second is not None
    held = injuries.report_csvs(tmp_path, day)
    assert [p.stem.split("_")[1] for p in held] == ["20251210T233100Z", "20251210T233600Z"]
    assert held[-1].stem == second.stem


class _Client(injuries.InjuryClient):
    def __init__(self) -> None:
        self.asked: list[int] = []

    def fetch(self, day: date, hour: int) -> bytes | None:
        self.asked.append(hour)
        return PDF if hour == 17 else None


def test_capture_only_asks_for_published_hours(tmp_path):
    client = _Client()
    client.capture(
        tmp_path, date(2025, 12, 10), now=datetime(2025, 12, 10, 17, 20, tzinfo=SLATE_TZ)
    )
    assert client.asked == list(range(11, 18))


def test_espn_feed_state_is_archived_only_when_a_status_changes(tmp_path):
    payload = {
        "injuries": [
            {
                "displayName": "Phoenix Suns",
                "injuries": [
                    {
                        "status": "Questionable",
                        "date": "2025-12-10T20:00Z",
                        "shortComment": "groin",
                        "athlete": {
                            "displayName": "Devin Booker",
                            "links": [
                                {
                                    "rel": ["playercard"],
                                    "href": "https://www.espn.com/nba/player/_/id/3136193/devin-booker",
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }
    rows = parse_feed(payload)
    assert rows == [
        FeedRow("PHX", "Devin Booker", "3136193", "Questionable", "2025-12-10T20:00Z", "groin")
    ]
    day = date(2025, 12, 10)
    assert write_feed(rows, tmp_path, day, "2025-12-10T20:00:00Z") is not None
    reworded = [FeedRow(**{**rows[0].__dict__, "comment": "groin strain"})]
    assert write_feed(reworded, tmp_path, day, "2025-12-10T20:05:00Z") is None
    out = [FeedRow(**{**rows[0].__dict__, "status": "Out"})]
    assert write_feed(out, tmp_path, date(2025, 12, 11), "2025-12-11T15:00:00Z") is not None
    assert latest_feed_path(tmp_path, date(2025, 12, 12)) is not None
