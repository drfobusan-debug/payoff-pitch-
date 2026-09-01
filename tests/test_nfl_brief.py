"""What the morning brief is allowed to say, and what it must refuse to.

The brief is prose over the same ledger rows the tables are built from, so the
tests that matter are the ones about honesty: the two terms with a fit behind
them may be described as moving the total, and rest, travel, cold and injury may
not, however much a reader would like them to.
"""

from __future__ import annotations

from mlb_engine.data.openmeteo import VenueWeather
from nfl_engine.audit.ledger import LedgerEntry
from nfl_engine.data.schedule import ScheduleContext
from nfl_engine.market.ev import MONEYLINE, SPREAD, TOTAL
from nfl_engine.output.brief import GameContext, write_brief
from nfl_engine.output.card import build_card, render_html, render_markdown

MATCHUP = "NYJ @ BUF"


def _entry(**overrides: object) -> LedgerEntry:
    row = LedgerEntry(
        season=2026,
        week=1,
        date="2026-09-13",
        matchup=MATCHUP,
        market=SPREAD,
        side="NYJ",
        line=6.5,
        book="pinnacle",
        odds=-105.0,
        opposite_odds=-115.0,
        tier="Moderate buy",
        model_prob=0.56,
        fair_prob=0.51,
        ev_model=0.05,
        ev_fair=0.043,
        paired_books=4,
        kickoff_utc="2026-09-13T17:00:00Z",
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _brief_text(entries: list[LedgerEntry], context: GameContext | None = None, **kwargs) -> str:
    written = write_brief(MATCHUP, entries, week=kwargs.pop("week", 1), context=context, **kwargs)
    return " ".join([written.headline, *written.paragraphs])


def test_the_lede_names_the_price_the_total_and_whether_it_is_divisional() -> None:
    entries = [
        _entry(),
        _entry(side="BUF", line=-6.5),
        _entry(market=TOTAL, side="over", line=44.5),
    ]
    context = GameContext(schedule=ScheduleContext(div_game=True, roof="outdoors"))

    written = write_brief(MATCHUP, entries, week=1, context=context)

    assert written.headline == "Divisional: BUF by 6.5, total 44.5."


def test_the_market_implied_win_is_the_devigged_one() -> None:
    entries = [
        _entry(market=MONEYLINE, side="BUF", line=None, odds=-280.0, fair_prob=0.72),
        _entry(market=MONEYLINE, side="NYJ", line=None, odds=+230.0, fair_prob=0.28),
    ]

    text = _brief_text(entries)

    assert "BUF 72% / NYJ 28% once the hold is taken out" in text


def test_the_edge_is_stated_as_model_against_fair() -> None:
    text = _brief_text([_entry()])

    assert "Moderate buy on NYJ +6.5 at -105 (pinnacle)" in text
    assert "wins it 56% against a fair 51%" in text
    assert "+5.0 points of edge over the fair price" in text


def test_a_game_with_only_vetoes_says_which_screen_refused_it() -> None:
    text = _brief_text([_entry(tier="Pass", screens="model_disagrees;price_band")])

    assert "Nothing survives on this game" in text
    assert "model_disagrees, price_band" in text


def test_wind_is_the_one_thing_described_as_moving_the_total() -> None:
    context = GameContext(
        schedule=ScheduleContext(roof="outdoors", div_game=False),
        weather=VenueWeather(wind_mph=18.5, gust_mph=27.0, precipitation=0.0, temperature_f=44.0),
    )

    text = _brief_text([_entry()], context)

    assert "18 mph at 44F takes 2.0 off the total" in text
    assert "Gusts to 27." in text


def test_an_indoor_game_says_the_weather_never_enters_it() -> None:
    context = GameContext(schedule=ScheduleContext(roof="dome", div_game=True))

    assert "Indoors, so the weather never enters it." in _brief_text([_entry()], context)


def test_an_unknown_roof_is_not_written_up_as_a_calm_day() -> None:
    text = _brief_text([_entry()], GameContext(schedule=ScheduleContext(div_game=True)))

    assert "No roof on file" in text
    assert "mph" not in text


def test_rest_is_reported_and_explicitly_not_priced() -> None:
    context = GameContext(
        schedule=ScheduleContext(roof="dome", home_rest=13, away_rest=4, div_game=True)
    )

    text = _brief_text([_entry()], context)

    assert "BUF on 13 days' rest to NYJ's 4 -- a 9-day edge to BUF" in text
    assert "reported, not priced" in text
    assert "home off a bye (measured t=+0.05, not priced)" in text
    assert "away on a short week (measured t=-0.61, not priced)" in text


def test_cold_is_context_and_not_an_adjustment() -> None:
    context = GameContext(
        schedule=ScheduleContext(roof="outdoors", home_rest=7, away_rest=7),
        weather=VenueWeather(wind_mph=8.5, gust_mph=None, precipitation=0.0, temperature_f=21.0),
    )

    text = _brief_text([_entry()], context)

    assert "cold (measured t=+1.85 on margin, not priced)" in text


def test_clv_is_reported_on_the_side_taken() -> None:
    text = _brief_text([_entry(clv=0.023, close_odds=-118.0)])

    assert "NYJ +2.3 pts (closed -118), the market came to us" in text
    assert "Mean +2.3 points" in text


def test_an_absence_is_named_and_the_price_is_said_not_to_carry_it() -> None:
    text = _brief_text([_entry()], absences="Out -- NYJ: QB A. Rodgers [reported, not priced]")

    assert "QB A. Rodgers" in text
    assert "The absence itself is not charged to the price" in text


def test_a_game_with_no_context_at_all_still_writes_a_brief() -> None:
    written = write_brief(MATCHUP, [_entry()], week=1)

    assert written.headline == "Matchup: BUF by 6.5."
    assert any("Moderate buy on NYJ" in para for para in written.paragraphs)


def test_the_brief_reaches_the_rendered_card() -> None:
    entries = [_entry()]
    context = {
        MATCHUP: GameContext(
            schedule=ScheduleContext(roof="outdoors", div_game=True),
            weather=VenueWeather(
                wind_mph=20.0, gust_mph=None, precipitation=0.0, temperature_f=40.0
            ),
        )
    }

    card = build_card(entries, season=2026, week=1, context=context)
    text, page = render_markdown(card), render_html(card)

    assert card.games[0].brief is not None
    assert "**Divisional: BUF by 6.5.**" in text
    assert "takes 2.3 off the total" in text
    assert "class='lede'" in page
    assert "takes 2.3 off the total" in page


def test_a_card_built_without_context_is_unchanged_in_substance() -> None:
    card = build_card([_entry()], season=2026, week=1)
    text = render_markdown(card)

    assert "| NYJ +6.5 | -105 | pinnacle |" in text
    assert "No roof on file" in text
