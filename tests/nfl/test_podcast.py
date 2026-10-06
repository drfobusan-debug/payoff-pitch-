"""NFL podcast picks: team names, placement on the week's games, grading against
nflverse finals and closes, the card blocks, and that none of it reaches a play."""

from __future__ import annotations

import dataclasses
from datetime import date as Date
from pathlib import Path

import pandas as pd
import pytest

from engine_common.podcasts import picks as core
from engine_common.podcasts.extract import Extraction, Pick
from engine_common.podcasts.render import PodcastView
from engine_common.podcasts.shows import BY_KEY, NFL
from nfl_engine import podcast
from nfl_engine.audit.ledger import LedgerEntry, load_ledger, merge_ledger
from nfl_engine.output.card import build_card, render_html

SEASON, WEEK = 2026, 2
PUB = "2026-09-10T15:00:00+00:00"
SHOW = BY_KEY["schwartz"]


def row(
    matchup: str,
    market: str,
    side: str,
    *,
    line: float | None = None,
    odds: float = -110.0,
    screens: str = "",
    kickoff: str = "2026-09-13T17:00:00Z",
) -> LedgerEntry:
    return LedgerEntry(
        season=SEASON,
        week=WEEK,
        date="2026-09-13",
        matchup=matchup,
        market=market,
        side=side,
        line=line,
        book="dk",
        odds=odds,
        opposite_odds=-110.0,
        tier="Strong buy",
        model_prob=0.55,
        fair_prob=0.5,
        ev_model=0.05,
        ev_fair=0.04,
        paired_books=3,
        screens=screens,
        captured_at="2026-09-09T18:00:00Z",
        kickoff_utc=kickoff,
    )


@pytest.fixture
def entries() -> list[LedgerEntry]:
    return [
        row("BUF @ KC", "moneyline", "KC", odds=-130.0),
        row("BUF @ KC", "moneyline", "BUF", odds=110.0, screens="disagreement"),
        row("BUF @ KC", "spread", "KC", line=-2.5),
        row("BUF @ KC", "spread", "BUF", line=2.5, screens="thin_market"),
        row("BUF @ KC", "total", "over", line=47.5, screens="thin_market"),
        row("BUF @ KC", "total", "under", line=47.5, odds=-105.0, screens="thin_market"),
        row("NYJ @ NYG", "moneyline", "NYG", odds=-120.0, kickoff="2026-09-13T20:25:00Z"),
        row("NYJ @ NYG", "spread", "NYG", line=-1.5, kickoff="2026-09-13T20:25:00Z"),
    ]


def _pick(**kw: object) -> Pick:
    base: dict[str, object] = dict(
        pick_id="p1",
        show=SHOW.key,
        show_name=SHOW.name,
        host="Geoff Schwartz",
        episode_guid="g1",
        episode_title="Week 2 picks",
        published=PUB,
        kind="official",
        market="game_ats",
        team="Bills",
        opponent="Chiefs",
        side=None,
        line=2.5,
        price=None,
        units=None,
        seconds=300,
        quote="give me the Bills plus two and a half",
        reason="Allen off a loss",
        description="Bills +2.5",
        league=NFL,
        edge="my number is Bills +1",
        fair_line=1.0,
    )
    base.update(kw)
    return Pick(**base)  # type: ignore[arg-type]


def _ex(*picks: Pick) -> Extraction:
    return Extraction("g1", "m", "picks-v2", "sha", list(picks), [], list(SHOW.leagues))


def _frame(
    home: float = 24, away: float = 27, spread: float = 3.0, total: float = 48.5
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                season=SEASON,
                week=WEEK,
                away_team="BUF",
                home_team="KC",
                home_score=home,
                away_score=away,
                spread_line=spread,
                total_line=total,
            )
        ]
    )


@pytest.mark.parametrize(
    ("spoken", "code"),
    [
        ("Chiefs", "KC"),
        ("Kansas City", "KC"),
        ("KC", "KC"),
        ("the Niners", "SF"),
        ("Jaguars", "JAX"),
        ("JAC", "JAX"),
        ("Green Bay", "GB"),
        ("New York", None),
        ("Los Angeles", None),
        ("LA", None),
        ("", None),
    ],
)
def test_spoken_teams_map_to_one_code_or_none(spoken: str, code: str | None) -> None:
    assert podcast.team_code(spoken) == code


def test_games_come_from_the_engines_own_rows_with_the_home_spread(
    entries: list[LedgerEntry],
) -> None:
    bench = dataclasses.replace(entries[0], source="benchmark", matchup="DAL @ PHI")
    games = {g.matchup: g for g in podcast.to_games([*entries, bench], season=SEASON, week=WEEK)}
    assert set(games) == {"BUF @ KC", "NYJ @ NYG"}
    kc = games["BUF @ KC"]
    assert (kc.home, kc.away, kc.day) == ("KC", "BUF", Date(2026, 9, 13))
    assert kc.home_spread == -2.5
    assert kc.kickoff is not None and kc.kickoff.hour == 17


def test_a_pick_lands_on_its_game_from_its_teams_side(entries: list[LedgerEntry]) -> None:
    slate = podcast.slate_picks([_ex(_pick())], entries, season=SEASON, week=WEEK)
    [pl] = slate.placed
    assert (pl.matchup, pl.team_side, pl.line, pl.label) == ("BUF @ KC", "away", 2.5, "BUF +2.5")
    assert not slate.unmatched


def test_other_leagues_and_unknown_teams_never_land(entries: list[LedgerEntry]) -> None:
    cfb = _pick(pick_id="c", league="cfb", team="Iowa")
    vague = _pick(pick_id="v", team="New York", opponent=None)
    slate = podcast.slate_picks([_ex(cfb, vague)], entries, season=SEASON, week=WEEK)
    assert not slate.placed
    assert [p.pick_id for p in slate.unmatched] == ["v"]


def test_a_pick_published_after_kickoff_is_not_placed(entries: list[LedgerEntry]) -> None:
    late = _pick(published="2026-09-13T18:00:00+00:00")
    assert not podcast.slate_picks([_ex(late)], entries, season=SEASON, week=WEEK).placed


def test_official_supersedes_the_same_hosts_lean(entries: list[LedgerEntry]) -> None:
    lean = _pick(pick_id="l", kind="lean", published="2026-09-09T15:00:00+00:00")
    slate = podcast.slate_picks([_ex(lean, _pick())], entries, season=SEASON, week=WEEK)
    assert [pl.pick.kind for pl in slate.placed] == ["official"]


def test_engine_view_reads_only_unscreened_plays(entries: list[LedgerEntry]) -> None:
    slate = podcast.slate_picks(
        [_ex(_pick(), _pick(pick_id="k", team="Chiefs", line=-2.5))],
        entries,
        season=SEASON,
        week=WEEK,
    )
    views = {pl.pick.pick_id: podcast.engine_view(pl) for pl in slate.placed}
    assert views == {"p1": "engine disagrees", "k": "engine agrees"}
    total = _pick(pick_id="t", market="game_total", side="under", team="Bills", line=47.5)
    [pl] = podcast.slate_picks([_ex(total)], entries, season=SEASON, week=WEEK).placed
    assert podcast.engine_view(pl) == ""  # the engine passed every total


def test_grading_uses_the_hosts_number_and_the_nflverse_close(entries: list[LedgerEntry]) -> None:
    [pl] = podcast.slate_picks([_ex(_pick())], entries, season=SEASON, week=WEEK).placed
    key = (SEASON, WEEK, "BUF", "KC")
    graded = podcast.grade_row(pl, podcast.finals(_frame())[key])
    # BUF won by 3 at +2.5; KC closed -3 so BUF +3 at the close: half a point worse.
    assert graded["result"] == core.WIN
    assert (graded["close_line"], graded["clv_pts"]) == ("3", "-0.5")
    # No price said: the engine's own price on BUF +2.5 is used and labelled.
    assert (graded["price"], graded["price_source"]) == ("-110", "board")
    assert graded["edge"] == "my number is Bills +1" and graded["fair_line"] == "1"


def test_a_stated_price_beats_the_board_and_totals_grade_at_the_number(
    entries: list[LedgerEntry],
) -> None:
    total = _pick(
        pick_id="t", market="game_total", side="under", team="Bills", line=47.5, price=-108
    )
    [pl] = podcast.slate_picks([_ex(total)], entries, season=SEASON, week=WEEK).placed
    graded = podcast.grade_row(pl, podcast.finals(_frame())[(SEASON, WEEK, "BUF", "KC")])
    assert (graded["result"], graded["price"], graded["price_source"]) == (
        core.LOSS,
        "-108",
        "stated",
    )
    assert graded["clv_pts"] == "-1"  # under 47.5 against a 48.5 close


def test_unplayed_or_unpriced_picks_count_no_units(entries: list[LedgerEntry]) -> None:
    ml = _pick(pick_id="m", market="game_ml", team="Giants", opponent="Jets", line=None)
    [pl] = podcast.slate_picks([_ex(ml)], entries, season=SEASON, week=WEEK).placed
    graded = podcast.grade_row(pl, None)
    assert graded["result"] == "" and graded["units"] == ""
    alt = _pick(pick_id="a", line=6.5, quote="Bills plus six and a half")
    [pl] = podcast.slate_picks([_ex(alt)], entries, season=SEASON, week=WEEK).placed
    assert podcast.board_price(pl) is None  # the engine never priced BUF +6.5


def test_grade_week_writes_only_the_podcast_ledger(
    tmp_path: Path, entries: list[LedgerEntry]
) -> None:
    store = tmp_path / "store"
    (store / "podcast_picks").mkdir(parents=True)
    (store / "podcast_picks" / "schwartz__x.json").write_text(_ex(_pick()).to_json())
    production = tmp_path / "ledger.csv"
    merge_ledger(production, entries)
    before = production.read_bytes()
    rows = podcast.grade_week(
        store, load_ledger(production), season=SEASON, week=WEEK, frame=_frame()
    )
    assert [r["result"] for r in rows] == [core.WIN]
    assert core.load_ledger(core.ledger_path(store, NFL))[0]["league"] == NFL
    assert production.read_bytes() == before


def test_card_prints_the_pick_under_its_game_with_the_edge_and_record(
    tmp_path: Path, entries: list[LedgerEntry]
) -> None:
    store = tmp_path / "store"
    (store / "podcast_picks").mkdir(parents=True)
    (store / "podcast_picks" / "schwartz__x.json").write_text(_ex(_pick()).to_json())
    podcast.grade_week(store, entries, season=SEASON, week=WEEK, frame=_frame())
    card = build_card(entries, season=SEASON, week=WEEK)
    plain = render_html(card)
    view = podcast.view(store, entries, season=SEASON, week=WEEK)
    assert isinstance(view, PodcastView)
    page = render_html(card, view)
    kc = page.index("Buffalo") if "Buffalo" in page else page.index("BUF @ KC")
    block = page.index("Schwartz Football Show", kc)
    assert block < page.index("NYJ @ NYG")
    assert "my number is Bills +1" in page and "Geoff Schwartz" in page
    assert "1-0" in page
    assert "underpowered" in page.lower()
    # The plays themselves are the same with or without the shows.
    assert [p.side for p in card.plays()] == [
        p.side for p in build_card(entries, season=SEASON, week=WEEK).plays()
    ]
    assert "Schwartz Football Show" not in plain


def test_no_store_means_no_view(tmp_path: Path, entries: list[LedgerEntry]) -> None:
    assert podcast.view(tmp_path, entries, season=SEASON, week=WEEK) is None
    assert podcast.grade_week(tmp_path, entries, season=SEASON, week=WEEK, frame=_frame()) == []
