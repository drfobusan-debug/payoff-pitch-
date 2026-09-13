"""The read beside each game: gathered from what exists, invented from nothing.

Contracts:

* the italic market line under a title quotes the market's consensus rung and
  the model's number on the same row, escaped;
* a field no source supplied is left out of the card rather than filled in;
* a card built without any brief renders exactly the plays and vetoes it always
  did -- the colour is additive, and pricing never sees it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_engine.audit.ledger import LedgerEntry
from nfl_engine.data.color import parse_summary
from nfl_engine.features.ratings import RatingBook, TeamRating
from nfl_engine.output.brief import GameBrief, TeamBrief, build_briefs
from nfl_engine.output.card import build_card, market_reads, render_html, render_markdown

SEASON, WEEK = 2026, 3


def row(
    matchup: str,
    market: str,
    side: str,
    *,
    line: float | None,
    model: float,
    fair: float | None,
    screens: str = "thin_edge",
) -> LedgerEntry:
    return LedgerEntry(
        season=SEASON,
        week=WEEK,
        date="2026-09-27",
        matchup=matchup,
        market=market,
        side=side,
        line=line,
        book="dk",
        odds=-110.0,
        opposite_odds=-110.0,
        tier="Moderate buy",
        model_prob=model,
        fair_prob=fair,
        ev_model=0.02,
        ev_fair=(model - (fair or model)) * 1.9,
        paired_books=3,
        screens=screens,
        result="",
        clv=None,
        source="engine",
        captured_at="2026-09-24T18:00:00Z",
        kickoff_utc="2026-09-27T17:00:00Z",
    )


@pytest.fixture
def rows() -> list[LedgerEntry]:
    m = "GB @ CHI"
    return [
        row(m, "moneyline", "CHI", line=None, model=0.47, fair=0.44),
        row(m, "moneyline", "GB", line=None, model=0.53, fair=0.56),
        row(m, "spread", "CHI", line=2.5, model=0.51, fair=0.50, screens=""),
        row(m, "spread", "CHI", line=-0.5, model=0.44, fair=0.43),
        row(m, "spread", "CHI", line=6.5, model=0.66, fair=0.64),
        row(m, "total", "over", line=44.5, model=0.48, fair=0.50),
        row(m, "total", "over", line=41.5, model=0.58, fair=0.60),
        row(m, "total", "under", line=44.5, model=0.52, fair=0.50),
    ]


def test_the_market_line_reads_the_consensus_rung_off_the_ledger(
    rows: list[LedgerEntry],
) -> None:
    reads = {r.market: r for r in market_reads(rows)}
    assert reads["spread"].selection() == "CHI +2.5"
    assert reads["total"].selection() == "over 44.5"
    assert (reads["moneyline"].fair_prob, reads["moneyline"].model_prob) == (0.44, 0.47)
    card = build_card(rows, season=SEASON, week=WEEK)
    page = render_html(card)
    assert "ATS CHI +2.5 market 50.0% · model 51.0%" in page
    assert "Total over 44.5 market 50.0% · model 48.0%" in page
    assert "ML CHI market 44.0% · model 47.0%" in page
    assert "_moneyline CHI market 44.0% · model 47.0%" in render_markdown(card)


def test_a_card_without_briefs_renders_as_before(rows: list[LedgerEntry]) -> None:
    card = build_card(rows, season=SEASON, week=WEEK)
    page = render_html(card)
    assert "GB @ CHI" in page and "CHI +2.5" in page
    assert "Vetoed: thin_edge x7" in page
    assert "<table class='teams'>" not in page
    assert "Venue" not in page and "Who matters" not in page


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(season=SEASON, week=1, gameday="2026-09-13", gametime="13:00", home_team="CHI", away_team="MIN", home_score=24.0, away_score=17.0),
            dict(season=SEASON, week=2, gameday="2026-09-20", gametime="13:00", home_team="DET", away_team="CHI", home_score=31.0, away_score=27.0),
            dict(season=SEASON, week=1, gameday="2026-09-13", gametime="16:25", home_team="GB", away_team="DAL", home_score=30.0, away_score=20.0),
            dict(season=SEASON, week=2, gameday="2026-09-20", gametime="13:00", home_team="GB", away_team="SEA", home_score=27.0, away_score=13.0),
            dict(
                season=SEASON, week=WEEK, gameday="2026-09-27", gametime="13:00",
                home_team="CHI", away_team="GB", home_score=float("nan"), away_score=float("nan"),
                roof="outdoors", surface="grass", stadium="Soldier Field", div_game=1,
                home_rest=7, away_rest=7, home_qb_name="Caleb Williams", away_qb_name="Jordan Love",
            ),
        ]
    )  # fmt: skip


def _ratings(games_used: int = 400) -> RatingBook:
    return RatingBook(
        teams={
            "CHI": TeamRating(
                "CHI", off_epa=-0.02, def_epa=0.01, off_success=0.44, def_success=0.45
            ),
            "GB": TeamRating("GB", off_epa=0.08, def_epa=-0.03, off_success=0.48, def_success=0.42),
            "DET": TeamRating(
                "DET", off_epa=0.05, def_epa=0.02, off_success=0.47, def_success=0.46
            ),
        },
        games_used=games_used,
    )


def test_the_brief_reads_form_from_the_schedule_and_ranks_from_the_book(
    rows: list[LedgerEntry],
) -> None:
    briefs = build_briefs(rows, season=SEASON, week=WEEK, schedule=_schedule(), ratings=_ratings())
    b = briefs["GB @ CHI"]
    assert (b.home.record, b.home.streak, b.home.last) == ("1-1", "L1", "L 27-31 @ DET")
    assert (b.away.record, b.away.streak) == ("2-0", "W2")
    assert (b.away.off_rank, b.away.def_rank, b.away.net_rank) == (1, 1, 1)
    assert b.home.rated and (b.home.def_rank, b.home.net_rank) == (2, 3)
    assert (b.venue, b.roof, b.surface, b.div_game) == ("Soldier Field", "outdoors", "grass", True)
    assert (b.home.qb, b.away.qb) == ("Caleb Williams", "Jordan Love")
    assert b.kickoff_local == "Sun Sep 27 13:00 ET"
    # The number each side quotes, read back off the rung nearest even money.
    assert (b.market_spread, b.model_spread) == (2.5, 2.5)
    assert (b.market_total, b.model_total) == (44.5, 44.5)


def test_an_unusable_book_leaves_the_team_unrated(rows: list[LedgerEntry]) -> None:
    briefs = build_briefs(rows, season=SEASON, week=WEEK, ratings=_ratings(games_used=40))
    b = briefs["GB @ CHI"]
    assert not b.home.rated and b.home.net_rank is None
    assert b.home.record == "" and b.venue is None
    page = render_html(build_card(rows, season=SEASON, week=WEEK, briefs=briefs))
    assert "unrated" in page and "Venue" not in page


def test_the_take_and_context_render_only_what_the_brief_has(
    rows: list[LedgerEntry],
) -> None:
    briefs = build_briefs(rows, season=SEASON, week=WEEK, schedule=_schedule(), ratings=_ratings())
    b = briefs["GB @ CHI"]
    b.home.out = ["WR <Rome> Odunze (Out, knee)"]
    b.headline = "Packers & Bears renew rivalry"
    b.tags = ["rivalry", "division"]
    b.city = "Chicago, IL"
    b.temperature_f, b.gust_mph, b.condition = 54.0, 18.0, "Rain"
    page = render_html(build_card(rows, season=SEASON, week=WEEK, briefs=briefs))
    assert "Green Bay Packers at Chicago Bears" in page
    assert (
        "Green Bay Packers (2-0, W2, rated #1) visits Chicago Bears (1-1, L1, rated #3) in Chicago."
        in page
    )
    assert "The market has Green Bay Packers by 2.5 with the total at 44.5." in page
    assert "Green Bay Packers&#x27; 1st-ranked offense" in page
    assert "WR &lt;Rome&gt; Odunze (Out, knee)" in page and "<Rome>" not in page
    assert "Packers &amp; Bears renew rivalry" in page
    assert "<span class='chip'>rivalry</span>" in page
    assert "Soldier Field, Chicago, IL (grass)" in page
    assert "kickoff forecast rain, 54°F, gusts to 18 mph" in page
    assert "division game" in page
    assert (
        "Under center: Jordan Love for Green Bay Packers, Caleb Williams for Chicago Bears." in page
    )


def test_the_take_in_a_dome_says_indoors_and_no_forecast(rows: list[LedgerEntry]) -> None:
    brief = GameBrief(
        matchup="GB @ CHI",
        home=TeamBrief("CHI", "Chicago Bears"),
        away=TeamBrief("GB", "Green Bay Packers"),
        venue="Some Dome",
        roof="dome",
        temperature_f=80.0,
        condition="Sunny",
    )
    page = render_html(build_card(rows, season=SEASON, week=WEEK, briefs={"GB @ CHI": brief}))
    assert "Some Dome; indoors." in page and "Sunny" not in page


ESPN = {
    "header": {
        "competitions": [
            {
                "neutralSite": False,
                "notes": [{"headline": "NFC North division game"}],
                "competitors": [
                    {
                        "team": {"abbreviation": "CHI"},
                        "record": [{"type": "total", "displayValue": "1-1"}],
                    },
                    {
                        "team": {"abbreviation": "GB"},
                        "record": [{"type": "total", "displayValue": "2-0"}],
                    },
                ],
                "broadcasts": [{"media": {"shortName": "FOX"}}],
            }
        ]
    },
    "againstTheSpread": [
        {"team": {"abbreviation": "GB"}, "records": [{"displayValue": "2-0-0"}]},
    ],
    "leaders": [
        {
            "team": {"abbreviation": "GB"},
            "leaders": [
                {
                    "name": "passingYards",
                    "leaders": [
                        {
                            "displayValue": "612 YDS, 5 TD",
                            "athlete": {
                                "displayName": "Jordan Love",
                                "position": {"abbreviation": "QB"},
                            },
                        }
                    ],
                }
            ],
        }
    ],
    "injuries": [
        {
            "team": {"abbreviation": "CHI"},
            "injuries": [
                {
                    "status": "Out",
                    "details": {"type": "Knee"},
                    "athlete": {"displayName": "Rome Odunze", "position": {"abbreviation": "WR"}},
                },
                {
                    "status": "Questionable",
                    "details": {"type": "Ankle"},
                    "athlete": {"displayName": "D.J. Moore", "position": {"abbreviation": "WR"}},
                },
            ],
        }
    ],
    "lastFiveGames": [
        {
            "team": {"abbreviation": "CHI"},
            "events": [
                {
                    "gameResult": "L",
                    "score": "27-31",
                    "atVs": "@",
                    "opponent": {"abbreviation": "DET"},
                },
                {
                    "gameResult": "W",
                    "score": "24-17",
                    "atVs": "vs",
                    "opponent": {"abbreviation": "MIN"},
                },
            ],
        }
    ],
    "gameInfo": {
        "venue": {
            "fullName": "Soldier Field",
            "grass": True,
            "indoor": False,
            "address": {"city": "Chicago", "state": "IL"},
        },
        "weather": {"temperature": 54, "precipitation": 60, "gust": 18, "displayValue": "Rain"},
    },
    "article": {
        "type": "Preview",
        "headline": "Packers, Bears renew the NFL's oldest rivalry",
        "story": "<p>The Packers visit the Bears with the division lead on the line.</p>",
    },
    "predictor": {"homeTeam": {"gameProjection": "41.3"}},
    "pickcenter": [{"details": "CHI +2.5", "overUnder": 44.5}],
}


def test_parse_summary_reads_the_espn_payload_and_only_the_payload() -> None:
    color = parse_summary(ESPN, home="CHI", away="GB")
    assert (color.home.record, color.away.record, color.away.ats) == ("1-1", "2-0", "2-0-0")
    assert color.away.leaders == ["QB Jordan Love — 612 YDS, 5 TD"]
    assert color.home.out == ["WR Rome Odunze (Out, knee)"]
    assert color.home.questionable == ["WR D.J. Moore (ankle)"]
    assert color.home.last_five[0] == "L 27-31 @ DET" and color.home.streak() == "L1"
    assert (color.venue, color.city, color.grass, color.indoor) == (
        "Soldier Field",
        "Chicago, IL",
        True,
        False,
    )
    assert (color.temperature_f, color.precip_pct, color.gust_mph, color.condition) == (
        54.0,
        60.0,
        18.0,
        "Rain",
    )
    assert color.headline == "Packers, Bears renew the NFL's oldest rivalry"
    assert color.story and "division lead" in color.story and "<p>" not in color.story
    assert color.fpi_home == 41.3
    assert color.broadcast == "FOX" and color.spread_note == "CHI +2.5 / 44.5"
    assert "rivalry" in color.tags and "division" in color.tags
    # Nothing there, nothing invented.
    empty = parse_summary({}, home="CHI", away="GB")
    assert empty.venue is None and empty.home.record is None and empty.tags == []
    assert empty.home.leaders == [] and empty.home.out == [] and empty.fpi_home is None
