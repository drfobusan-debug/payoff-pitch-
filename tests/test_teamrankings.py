"""TeamRankings on its own rows, graded the same way and counted separately.

A second model is the only check on the engine that is not the engine or the
price, so what matters in these tests is the two things that make it worth
having: that its picks are read as *it* published them (star rating, price,
"Lay Off"), and that its record can never leak into ours.

The HTML is a row captured verbatim from their grid.
"""

from __future__ import annotations

import argparse
from datetime import date as Date

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.audit.ledger import (
    ENGINE,
    LEDGER_FIELDS,
    LedgerEntry,
    engine_metrics,
    engine_rows,
    load_ledger,
    update_ledger,
)
from mlb_engine.audit.outside import (
    TEAMRANKINGS,
    entries_from_picks,
    grade_pick,
    head_to_head,
    star_tier,
)
from mlb_engine.config import Credentials
from mlb_engine.data.results import GameResult
from mlb_engine.data.teamrankings import (
    TeamRankingsClient,
    TeamRating,
    TRPick,
    annotate,
    load_picks,
    load_ratings,
    merge_picks,
    merge_ratings,
    parse_picks,
    parse_ratings,
    save_picks,
    save_ratings,
)
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

_MATCHUP_URL = "https://www.teamrankings.com/mlb/matchup/diamondbacks-braves-2026-08-14"

ROW = f"""
<tr class="div_714 team_1426">
<td data-sort="957">957<br />958</td>
<td>Arizona<br />Atlanta</td>
<td class="picks green" data-sort="2-60.93"><a href="{_MATCHUP_URL}">
  <div class="picks-block-in">Arizona
  <div class="tr_stars"><span class="tr_stars_2"></span></div></div></a></td>
<td class="picks red" data-sort="2-03.23"><a href="{_MATCHUP_URL}">
  <div class="picks-block-in">ATL -1.5 +105
  <div class="tr_stars"><span class="tr_stars_2"></span></div></div></a></td>
<td class="picks green" data-sort="1-50.07"><a href="{_MATCHUP_URL}">
  <div class="picks-block-in">Under 8.0
  <div class="tr_stars"><span class="tr_stars_1"></span></div></div></a></td>
<td class="picks red" data-sort="2-03.23"><a href="{_MATCHUP_URL}">
  <div class="picks-block-in">ATL +179
  <div class="tr_stars"><span class="tr_stars_2"></span></div></div></a></td>
</tr>
"""

LAY_OFF_ROW = ROW.replace("ATL -1.5 +105", "Lay Off").replace("ATL +179", "Lay Off")


def _picks() -> dict[str, TRPick]:
    return {p.market: p for p in parse_picks(ROW)}


def _result(home: int, away: int) -> GameResult:
    return GameResult(
        game_pk=1, final=True, home_runs=home, away_runs=away, f5_home=0, f5_away=0
    )


def test_the_grid_reads_as_four_calls_on_one_game() -> None:
    picks = _picks()
    assert set(picks) == {"game_winner", "game_rl", "game_total", "game_ml"}
    assert all(p.date == "2026-08-14" for p in picks.values())
    # Their long club names become our codes, visitor first, so the row joins to
    # the same matchup string our own recommendations carry.
    assert all(p.matchup == "AZ @ ATL" for p in picks.values())


def test_selections_are_our_own_market_keys() -> None:
    """Otherwise the two ledgers cannot be read side by side."""
    picks = _picks()
    assert picks["game_ml"].selection == "ATL ML"
    assert picks["game_rl"].selection == "ATL -1.5"
    assert picks["game_total"].selection == "Under 8.0"


def test_their_price_is_kept_and_ours_is_not_invented() -> None:
    picks = _picks()
    assert picks["game_ml"].american == 179.0
    assert picks["game_rl"].american == 105.0
    # The grid quotes no price on a total or on the projected winner. A number
    # they did not publish is not filled in.
    assert picks["game_total"].american is None
    assert picks["game_winner"].american is None


def test_stars_are_read_as_published() -> None:
    picks = _picks()
    assert picks["game_ml"].stars == 2
    assert picks["game_total"].stars == 1
    assert star_tier(2) == "2 stars"
    assert star_tier(1) == "1 star"
    # A row with no star markup says so rather than claiming one star.
    assert star_tier(0) == "unrated"


def test_their_published_numbers_are_kept_apart_by_column() -> None:
    """A win probability and an edge are different quantities."""
    picks = _picks()
    assert picks["game_winner"].win_prob == 0.6093
    assert picks["game_total"].win_prob == 0.5007
    assert picks["game_ml"].value == 0.0323
    assert picks["game_rl"].value == 0.0323
    # The value columns publish no probability, and the forecast columns no edge.
    assert picks["game_ml"].win_prob is None
    assert picks["game_winner"].value is None


def test_the_projected_winner_is_not_the_money_line_pick() -> None:
    """They are separate columns, and on this game they disagree.

    Their winner column takes Arizona; their money-line column takes Atlanta at
    +179. Folding the two together would credit the model with a bet it did not
    make -- and, on a "Lay Off" game, with one it explicitly declined.
    """
    picks = _picks()
    assert picks["game_winner"].team == "AZ"
    assert picks["game_ml"].team == "ATL"


def test_lay_off_is_no_pick_rather_than_a_pick() -> None:
    markets = {p.market for p in parse_picks(LAY_OFF_ROW)}
    assert markets == {"game_winner", "game_total"}


def test_a_row_we_cannot_read_is_dropped_whole() -> None:
    """Half-reading a row would put a pick on the wrong team."""
    assert parse_picks(ROW.replace("Atlanta", "Some New Club")) == []
    assert parse_picks(ROW.replace(_MATCHUP_URL, "/mlb/matchup/no-date-here")) == []


def test_grading_matches_the_box_score() -> None:
    picks = _picks()
    # Atlanta 5, Arizona 2: 7 runs under 8, Atlanta covers -1.5, Arizona loses.
    res = _result(home=5, away=2)
    assert grade_pick(picks["game_ml"], res) == WIN
    assert grade_pick(picks["game_rl"], res) == WIN
    assert grade_pick(picks["game_total"], res) == WIN
    assert grade_pick(picks["game_winner"], res) == LOSS

    # Arizona 6, Atlanta 3: nine runs, so the under loses and so does Atlanta.
    flipped = _result(home=3, away=6)
    assert grade_pick(picks["game_ml"], flipped) == LOSS
    assert grade_pick(picks["game_total"], flipped) == LOSS
    assert grade_pick(picks["game_winner"], flipped) == WIN


def test_a_total_landing_on_the_number_is_a_push() -> None:
    assert grade_pick(_picks()["game_total"], _result(home=4, away=4)) == PUSH


def test_a_one_run_win_loses_the_run_line() -> None:
    assert grade_pick(_picks()["game_rl"], _result(home=4, away=3)) == LOSS


def _entries(home: int = 5, away: int = 2) -> list[LedgerEntry]:
    return entries_from_picks(
        list(_picks().values()),
        {1: _result(home, away)},
        {"AZ @ ATL": 1},
        Date(2026, 8, 14),
    )


def test_rows_are_theirs_and_say_so() -> None:
    rows = _entries()
    assert len(rows) == 4
    assert {r.source for r in rows} == {TEAMRANKINGS}
    assert {r.book for r in rows} == {"teamrankings"}
    # Their star rating stands where our tier would be, unconverted.
    by_market = {r.market: r for r in rows}
    assert by_market["game_ml"].tier == "2 stars"
    assert by_market["game_total"].tier == "1 star"


def test_the_forecast_column_is_graded_but_never_staked() -> None:
    """A projected winner has no price, so it cannot be a wager."""
    winner = next(r for r in _entries() if r.market == "game_winner")
    assert winner.result == LOSS
    assert winner.pnl == 0.0


def test_an_unpriced_total_is_paid_at_the_standard_price() -> None:
    total = next(r for r in _entries() if r.market == "game_total")
    assert total.result == WIN
    assert total.pnl == 0.91


def test_a_game_with_no_final_score_is_not_graded() -> None:
    assert entries_from_picks(list(_picks().values()), {}, {"AZ @ ATL": 1}, Date(2026, 8, 14)) == []
    assert entries_from_picks(list(_picks().values()), {1: _result(5, 2)}, {}, Date(2026, 8, 14)) == []


def test_picks_from_another_slate_are_not_graded_into_this_one() -> None:
    stale = [p for p in parse_picks(ROW.replace("2026-08-14", "2026-08-13"))]
    assert entries_from_picks(stale, {1: _result(5, 2)}, {"AZ @ ATL": 1}, Date(2026, 8, 14)) == []


def _ours(**kw) -> LedgerEntry:
    base = dict(
        date="2026-08-14",
        matchup="AZ @ ATL",
        category="game",
        market="game_total",
        selection="Over 8.0",
        line=8.0,
        book="draftkings",
        odds=-110.0,
        tier="Strong buy",
        model_prob=0.55,
        ev=0.05,
        result=LOSS,
        pnl=-1.0,
    )
    base.update(kw)
    return LedgerEntry(**base)


def test_our_measurements_do_not_count_their_bets() -> None:
    """The whole point of a benchmark is that it is not inside the thing it checks.

    Their record is graded and stored, and then every measurement of the engine
    is taken through ``engine_rows``: their 4-0 on a night would otherwise be
    our PPV, our ROI and our calibration basis.
    """
    ours = [_ours(), _ours(selection="Under 8.0", result=WIN, pnl=0.91)]
    mixed = ours + _entries()
    assert engine_rows(mixed) == ours
    assert all(r.source == ENGINE for r in engine_rows(mixed))
    assert engine_metrics(engine_rows(mixed)) == engine_metrics(ours)
    # And the unfiltered ledger really would have been contaminated.
    assert engine_metrics(mixed).n > engine_metrics(ours).n


def test_an_old_ledger_row_is_ours(tmp_path) -> None:
    """Every row written before the benchmark existed is the engine's own."""
    path = tmp_path / "ledger.csv"
    header = ",".join(f for f in LEDGER_FIELDS if f != "source")
    path.write_text(f"{header}\n2026-08-01,AZ @ ATL,game,game_total,Over 8.0{',' * 26}\n")
    rows = load_ledger(path)
    assert len(rows) == 1
    assert rows[0].source == ENGINE
    assert engine_rows(rows) == rows


def test_a_re_audit_replaces_both_ledgers_for_the_date(tmp_path) -> None:
    """Their rows are written in the same call as ours, so neither is orphaned."""
    path = tmp_path / "ledger.csv"
    update_ledger(path, [_ours(), *_entries()], Date(2026, 8, 14))
    again = update_ledger(path, [_ours(), *_entries()], Date(2026, 8, 14))
    assert len(again) == 5
    assert sum(1 for r in again if r.source == TEAMRANKINGS) == 4


def test_an_earlier_slate_of_theirs_survives_a_later_audit(tmp_path) -> None:
    path = tmp_path / "ledger.csv"
    yesterday = [r for r in _entries()]
    update_ledger(path, yesterday, Date(2026, 8, 14))
    kept = update_ledger(path, [_ours(date="2026-08-15")], Date(2026, 8, 15))
    assert sum(1 for r in kept if r.source == TEAMRANKINGS) == 4


def test_the_head_to_head_shows_what_each_of_us_did() -> None:
    rows = {r.market: r for r in head_to_head([_ours()], _entries())}
    total = rows["game_total"]
    assert (total.ours, total.our_result) == ("Over 8.0", LOSS)
    assert (total.theirs, total.their_tier, total.their_result) == ("Under 8.0", "1 star", WIN)
    assert total.contested and not total.agree
    # A market we passed on is still shown: their call there is the interesting one.
    assert rows["game_ml"].ours == ""
    assert rows["game_ml"].theirs == "ATL ML"
    # The forecast column is not a bet, so it is not in the bet comparison.
    assert "game_winner" not in rows


def test_agreement_is_agreement_on_the_same_selection() -> None:
    rows = {r.market: r for r in head_to_head([_ours(selection="Under 8.0")], _entries())}
    assert rows["game_total"].agree
    assert not rows["game_total"].contested


def test_a_pick_we_passed_on_is_not_read_as_our_bet() -> None:
    passed = _ours(tier="Pass", selection="Under 8.0")
    rows = {r.market: r for r in head_to_head([passed], _entries())}
    assert rows["game_total"].ours == ""


def test_a_capture_round_trips_and_the_fresher_one_wins(tmp_path) -> None:
    path = tmp_path / "teamrankings_2026-08-14.json"
    picks = list(_picks().values())
    save_picks(path, picks)
    assert load_picks(path) == picks

    moved = [p for p in parse_picks(ROW.replace("Under 8.0", "Under 8.5"))]
    merged = merge_picks(picks, moved)
    # A moved line is the same call revised, not a second bet -- the capture runs
    # several times before first pitch, and two rows would settle it twice.
    totals = [p.selection for p in merged if p.market == "game_total"]
    assert totals == ["Under 8.5"]
    assert len(merge_picks(picks, picks)) == len(picks)


def test_a_re_captured_slate_settles_each_market_once() -> None:
    """The whole benchmark is unreadable if their P&L counts a market twice."""
    picks = list(_picks().values())
    moved = list(parse_picks(ROW.replace("Under 8.0", "Under 8.5")))
    rows = entries_from_picks(
        picks + moved, {1: _result(4, 3)}, {"AZ @ ATL": 1}, Date(2026, 8, 14)
    )
    assert [r.market for r in rows].count("game_total") == 1
    assert sum(1 for r in rows if r.market == "game_ml") == 1


def test_a_missing_capture_is_not_an_error(tmp_path) -> None:
    assert load_picks(tmp_path / "nothing.json") == []
    (tmp_path / "junk.json").write_text("not json")
    assert load_picks(tmp_path / "junk.json") == []


# --- the join onto our own bets --------------------------------------------


def _rec(**kw) -> Recommendation:
    base = dict(
        game_date=Date(2026, 8, 14),
        game_pk=1,
        matchup="AZ @ ATL",
        category="game",
        market="game_total",
        selection="Under 8.5",
        model_prob=0.55,
        line=8.5,
        side="under",
        tier=Tier.MODERATE,
    )
    base.update(kw)
    return Recommendation(**base)


def test_each_of_their_four_columns_lands_on_the_bet_it_is_about() -> None:
    """Four columns are four statements; folding them into one loses three."""
    picks = list(_picks().values())
    total = _rec()
    runline = _rec(market="game_rl", selection="ATL -1.5", line=-1.5, side="")
    money = _rec(market="game_ml", selection="ATL ML", line=None, side="")
    assert annotate([total, runline, money], picks) == 3
    assert total.tr_pick.startswith("Under 8.0")
    assert runline.tr_pick.startswith("ATL -1.5")
    assert money.tr_pick.startswith("ATL ML")
    # The projected winner is a projection, not a bet, so it rides along on
    # every game market rather than being mistaken for their money-line pick --
    # on this game the two disagree: the model likes Arizona, the price makes
    # Atlanta the value.
    assert all(r.tr_winner == "AZ ML 61% 2\u2605" for r in (total, runline, money))
    assert money.tr_pick.startswith("ATL ML")


def test_laying_off_the_price_is_not_a_pick_on_the_winner() -> None:
    """Their winner column always names a side; their value columns may decline."""
    money = _rec(market="game_ml", selection="ATL ML", line=None, side="")
    assert annotate([money], parse_picks(LAY_OFF_ROW)) == 0
    assert money.tr_pick is None and money.tr_agrees is None
    # ... and the projection is still shown, in its own column, so declining to
    # bet the price is never printed as a pick they did not make.
    assert money.tr_winner.startswith("AZ ML")


def test_the_mark_says_whose_side_they_are_on() -> None:
    picks = list(_picks().values())
    with_them = _rec(selection="Under 8.5")
    against = _rec(selection="Over 8.5", side="over")
    annotate([with_them, against], picks)
    assert with_them.tr_agrees is True and with_them.tr_mark == "\u2605"
    assert against.tr_agrees is False and against.tr_mark == "\u2717"
    # Their two stars belong to their side, so a cross never shows our row two
    # stars for a bet they took against us.
    assert against.tr_stars == 1  # the total's own rating, not the money line's


def test_a_prop_is_left_alone_because_they_do_not_price_props() -> None:
    prop = _rec(market="batter_hits", selection="Corbin Carroll H o0.5", side="over")
    assert annotate([prop], list(_picks().values())) == 0
    assert prop.tr_pick is None and prop.tr_winner is None


def test_their_numbers_are_printed_in_their_own_units() -> None:
    picks = _picks()
    # A win probability where they publish one, the edge they see where they do
    # not -- the money-line column is about the price, not about who wins.
    assert picks["game_total"].summary == "Under 8.0 50% 1\u2605"
    assert picks["game_ml"].summary == "ATL ML +3.2% val 2\u2605"


def test_a_slate_without_the_login_captures_nothing_rather_than_yesterday() -> None:
    """Signed out their grid serves the last *played* slate, results included."""
    client = TeamRankingsClient(Credentials(teamrankings_user=None, teamrankings_pass=None))
    assert client._signed_in() is None


# A luck-rating table row, captured verbatim: the team cell carries the record,
# and the columns after the rating are their splits, which we do not store.
RATING_ROW = """
<tr><th>Rank</th><th>Team</th><th>Rating</th><th>v 1-5</th></tr>
<tr><td>1</td><td><a href="/mlb/team/tampa-bay-rays/">Tampa Bay (74-48)</a></td>
<td>11.76</td><td>6-6</td></tr>
<tr><td>2</td><td><a href="/mlb/team/chicago-white-sox/">Chi Sox (64-58)</a></td>
<td>-6.68</td><td>4-7</td></tr>
<tr><td>3</td><td><a href="/mlb/team/sacramento-athletics/">Sacramento (55-70)</a></td>
<td>0.40</td><td>2-9</td></tr>
"""


def test_a_rating_table_is_read_as_rank_and_number() -> None:
    table = parse_ratings(RATING_ROW)
    assert table["TB"] == (1, 11.76)
    assert table["CWS"] == (2, -6.68)
    # Their Athletics are "Sacramento"; a club named some other way must not be
    # silently ranked as a different club.
    assert table["ATH"] == (3, 0.40)
    assert len(table) == 3


def test_a_header_or_an_unreadable_row_is_dropped_not_guessed() -> None:
    assert parse_ratings("<tr><td>x</td><td>Tampa Bay</td><td>1.0</td></tr>") == {}
    assert parse_ratings("<tr><td>1</td><td>Tampa Bay</td><td>--</td></tr>") == {}
    assert parse_ratings("<tr><td>1</td><td>Sheffield Wednesday</td><td>1.0</td></tr>") == {}


def test_ratings_round_trip_and_the_fresher_capture_wins(tmp_path) -> None:
    path = tmp_path / "tr_ratings_2026-08-16.json"
    first = [TeamRating(date="2026-08-16", team="TB", luck=11.76, luck_rank=1)]
    save_ratings(path, first)
    assert load_ratings(path) == first

    later = [TeamRating(date="2026-08-16", team="TB", luck=9.5, luck_rank=3)]
    merged = merge_ratings(load_ratings(path), later)
    assert [(r.luck, r.luck_rank) for r in merged] == [(9.5, 3)]
    # A different day is a different row: the whole point is that these accrue.
    across = merge_ratings(merged, [TeamRating(date="2026-08-17", team="TB", luck=9.0)])
    assert len(across) == 2


def test_a_missing_ratings_capture_is_not_an_error(tmp_path) -> None:
    assert load_ratings(tmp_path / "nothing.json") == []
    (tmp_path / "junk.json").write_text("not json")
    assert load_ratings(tmp_path / "junk.json") == []


def test_a_rating_that_is_not_a_number_is_dropped() -> None:
    """"nan" parses as a float and serialises to JSON nothing can read back."""
    assert parse_ratings("<tr><td>1</td><td>Tampa Bay</td><td>nan</td></tr>") == {}
    assert parse_ratings("<tr><td>1</td><td>Tampa Bay</td><td>inf</td></tr>") == {}


def test_the_ratings_are_captured_even_when_tonight_is_not_on_the_grid(monkeypatch, tmp_path) -> None:
    """Their grid rolls over late; the rating tables have no date on them at all.

    Capturing them after the slate check meant a run before the rollover took
    the "not on the grid" exit and stored no ratings for that day -- and a day
    of ratings missed cannot be recovered.
    """
    from mlb_engine import cli

    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    cfg = cli.load_config()
    monkeypatch.setattr(
        cli.TeamRankingsClient,
        "fetch_ratings",
        lambda self, date: [TeamRating(date=date, team="TB", luck=1.0)],
    )
    assert cli._capture_tr_ratings(cfg) == 1
    assert list((tmp_path / "audit").glob("tr_ratings_*.json"))


def test_a_ratings_failure_never_reaches_the_caller(monkeypatch, tmp_path) -> None:
    """The picks are the benchmark; a research sample must not take them down."""
    from mlb_engine import cli

    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    cfg = cli.load_config()

    def boom(self, date):
        raise RuntimeError("their site moved")

    monkeypatch.setattr(cli.TeamRankingsClient, "fetch_ratings", boom)
    assert cli._capture_tr_ratings(cfg) == 0
    assert not list(tmp_path.rglob("tr_ratings_*.json"))


def test_an_anonymous_session_cookie_is_not_a_login(monkeypatch) -> None:
    """``tr_session`` is handed out before any login, so it proved nothing.

    Checking for it accepted every rejected password: the client returned a
    signed-out session, the capture filed the free grid -- the last slate
    already played -- and the run reported success. Only ``tru`` means
    subscriber.
    """
    import requests

    from mlb_engine.data import teamrankings as tr

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self, names: tuple[str, ...]) -> None:
            self.cookies = requests.cookies.RequestsCookieJar()
            for name in names:
                self.cookies.set(name, "x")

        def get(self, *a, **k) -> FakeResponse:
            return FakeResponse()

        def post(self, *a, **k) -> FakeResponse:
            return FakeResponse()

    creds = Credentials(teamrankings_user="a@b.c", teamrankings_pass="pw")

    monkeypatch.setattr(requests, "Session", lambda: FakeSession(("tr_session", "trv3")))
    assert TeamRankingsClient(creds)._signed_in() is None

    monkeypatch.setattr(requests, "Session", lambda: FakeSession(("tr_session", "tru")))
    assert TeamRankingsClient(creds)._signed_in() is not None
    assert tr.SUBSCRIBER_COOKIE == "tru"


def test_capturing_a_slate_already_played_says_so(monkeypatch, tmp_path, capsys) -> None:
    """Signed out the newest grid date is in the past, which read as a success."""
    from mlb_engine import cli

    monkeypatch.setenv("MLBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MLBE_STATE_SYNC", "0")
    stale = "2020-05-05"
    picks = [
        TRPick(
            date=stale, matchup="AAA @ BBB", market="game_ml", selection="AAA ML",
            line=None, side="", team="AAA", team_side="away", american=-120, stars=2,
        )
    ]
    monkeypatch.setattr(cli.TeamRankingsClient, "fetch", lambda self, date=None: picks)
    monkeypatch.setattr(cli, "_capture_tr_ratings", lambda cfg: 0)

    args = argparse.Namespace(date=None)
    assert cli.cmd_teamrankings(args) == 0
    out = capsys.readouterr().out
    assert stale in out
    assert "TEAMRANKINGS_EMAIL" in out

    # Asked for that date explicitly, it is a backfill and not a warning.
    args = argparse.Namespace(date=stale)
    assert cli.cmd_teamrankings(args) == 0
    assert "TEAMRANKINGS_EMAIL" not in capsys.readouterr().out


def test_a_grid_on_another_slate_is_not_a_broken_scrape(caplog) -> None:
    """Signed out their grid serves the slate already played, and a filter that
    empties logs the same "0 picks" a dead parse does -- so name the date it is
    showing."""
    import logging

    from mlb_engine.data.teamrankings import TeamRankingsClient

    client = TeamRankingsClient()
    client._get = lambda url: ROW  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO):
        assert client.fetch(date="2026-08-15") == []

    text = caplog.text
    assert "showing 2026-08-14" in text and "2026-08-15" in text


def test_the_slate_on_the_grid_is_still_returned(caplog) -> None:
    import logging

    from mlb_engine.data.teamrankings import TeamRankingsClient

    client = TeamRankingsClient()
    client._get = lambda url: ROW  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO):
        picks = client.fetch(date="2026-08-14")

    assert len(picks) == 4
    assert "showing" not in caplog.text
