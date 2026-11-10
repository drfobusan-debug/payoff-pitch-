from __future__ import annotations

from datetime import date as Date

import pytest

from cbb_engine import audit, bands, history
from cbb_engine.data import espn
from engine_runtime import ledger, quotes

DAY = "2025-01-10"


def _row(eid: str, market: str, side: str, line: float | None, *, american: float = -110, p: float = 0.5,
         outcome: str = "win", day: str = DAY, book: str = "dk") -> ledger.Row:
    r = ledger.Row(slate_date=day, event_id=eid, tip_utc=f"{day}T23:00:00Z", market=market, side=side, line=line,
                   book=book, pass_id="p", pass_kind="board", bet_of_record=True, status="live", american=american, p_fair=p)
    r.outcome = outcome
    r.pnl = audit.pnl(r, outcome)
    return r


def _game(home: str, away: str, *, home_rank: int | None = None, away_rank: int | None = None,
          neutral: bool = False, conference: bool = False) -> espn.Game:
    return espn.Game("1", f"{DAY}T23:00Z", away, home, "STATUS_FINAL", neutral, conference, (30, 30), (35, 35),
                     away_rank=away_rank, home_rank=home_rank)


def test_scoreboard_reads_ap_rank_and_unranked_99_is_none() -> None:
    comp = {"competitors": [
        {"homeAway": "home", "team": {"displayName": "Kentucky Wildcats"}, "curatedRank": {"current": 17}},
        {"homeAway": "away", "team": {"displayName": "Toledo Rockets"}, "curatedRank": {"current": 99}},
    ]}
    (g,) = espn.parse_scoreboard({"events": [{"id": "9", "competitions": [comp]}]})
    assert (g.home_rank, g.away_rank) == (17, None)


def test_match_game_returns_the_game_and_match_final_its_score() -> None:
    games = [_game("Michigan St Spartans", "Iowa Hawkeyes"), _game("Duke Blue Devils", "UNC Tar Heels")]
    assert history.match_game("Michigan State Spartans", "Iowa Hawkeyes", games) is games[0]
    assert history.match_final("Michigan State Spartans", "Iowa Hawkeyes", games) == (60, 70)


def test_main_rows_keep_one_row_per_side_at_the_most_quoted_line_and_median_price() -> None:
    rows = [_row("e", "game_ats", "home", -3.0, american=a, book=b) for a, b in ((-105, "a"), (-110, "b"), (-115, "c"))]
    rows.append(_row("e", "game_ats", "home", -2.5, american=-130, book="d"))
    rows.append(_row("e", "game_ats", "home", -3.0, american=-110, book="x", outcome=""))  # ungraded
    (r,) = bands.main_rows(rows)
    assert (r.line, r.american) == (-3.0, -110)


def test_market_expected_is_the_vig_and_a_push_expects_nothing() -> None:
    assert bands.market_expected(_row("e", "game_ats", "home", -3.0)) == pytest.approx(0.5 * 100 / 110 - 0.5)
    assert bands.market_expected(_row("e", "game_ats", "home", -3.0, outcome="push")) == 0.0


def test_rest_days_counts_back_to_the_previous_board_with_the_team() -> None:
    boards = {Date(2025, 1, 9): [_game("A Team", "B Team")], Date(2025, 1, 7): [_game("C Team", "D Team")]}
    assert bands.rest_days("A Team", Date(2025, 1, 10), boards) == 1
    assert bands.rest_days("D Team", Date(2025, 1, 10), boards) == 3
    assert bands.rest_days("E Team", Date(2025, 1, 10), boards) is None


def test_last_results_read_the_latest_game_before_the_day() -> None:
    logs = [
        bands.LogGame(Date(2025, 1, 1), "A", 80, 90, 70),
        bands.LogGame(Date(2025, 1, 5), "A", 70, 60, 70),  # latest before the day: a loss
        bands.LogGame(Date(2025, 1, 1), "B", 60, 60, 50),
        bands.LogGame(Date(2025, 1, 10), "B", 60, 40, 120),  # same day: not yet known
    ]
    assert bands.last_results(logs, Date(2025, 1, 10)) == {"A": True, "B": False}


def test_openers_take_the_earliest_pass_and_movement_reads_toward_the_side() -> None:
    def q(pid: str, at: str, market: str, side: str, line: float) -> quotes.Quote:
        return quotes.Quote(at, DAY, "e", f"{DAY}T23:00:00Z", market, side, "", line, "dk", -110, -110, pid, "board")
    shards = [q("late", "2025-01-10T20:00:00Z", "game_ats", "home", -3.0), q("early", "2025-01-10T12:00:00Z", "game_ats", "home", -1.5),
              q("early", "2025-01-10T12:00:00Z", "game_total", "over", 148.0)]
    op = bands.openers(shards)["e"]
    assert op == {("game_ats", "home"): -1.5, ("game_total", "over"): 148.0}
    ctx = {"e": bands.GameCtx(opener=op)}
    reads = {b.name: b for b in bands.read_bands([_row("e", "game_ats", "home", -3.0), _row("e", "game_total", "over", 149.0)], ctx)}
    assert reads["line moved >= 1 toward the side ATS"].n == 1
    assert reads["line moved >= 1 toward the side total"].n == 1


def test_read_bands_rest_spots_and_per_game_sums() -> None:
    rested = bands.Team(rest=3)
    tired = bands.Team(rest=1)
    ctx = {"e": bands.GameCtx(home=rested, away=tired), "f": bands.GameCtx(home=tired, away=rested)}
    rows = [_row("e", "game_ats", "home", -2.0), _row("e", "game_ats", "away", 2.0, outcome="loss"),
            _row("f", "game_ats", "away", -4.0, outcome="loss"), _row("g", "game_ats", "home", -1.0)]
    reads = {b.name: b for b in bands.read_bands(rows, ctx)}
    rv = reads["rested (2+ days) vs a back-to-back ATS"]
    assert (rv.n, rv.record, len(rv.games)) == (2, "1-1", 2)
    assert reads["team on a back-to-back ATS"].record == "0-1"
    assert reads["home favourite ATS"].n == 1  # event g has no context and is left out


def test_context_reads_rank_venue_power_torvik_rank_tempo_and_last_result() -> None:
    board = [_game("Kentucky Wildcats", "Toledo Rockets", home_rank=17, neutral=True, conference=False)]
    logs = [bands.LogGame(Date(2025, 1, 1), "Kentucky", 70, 90, 60), bands.LogGame(Date(2025, 1, 1), "Toledo", 70, 60, 90)]
    ctx = bands.context({"e": ("Kentucky Wildcats", "Toledo Rockets")}, Date(2025, 1, 10), {Date(2025, 1, 10): board},
                        logs, {"Kentucky": "SEC", "Toledo": "MAC"}, tempo={"Toledo": "slow"},
                        ranks={"Kentucky": 5, "Toledo": 250})["e"]
    assert ctx.neutral and not ctx.conference
    assert (ctx.home.ap, ctx.home.power, ctx.home.net_rank, ctx.home.off_loss) == (17, True, 5, False)
    assert (ctx.away.ap, ctx.away.power, ctx.away.net_rank, ctx.away.off_loss) == (None, False, 250, True)
    assert (ctx.home.slow, ctx.away.slow) == (None, True)


def test_a_logged_margin_never_stands_in_for_torvik_rank() -> None:
    board = [_game("Kentucky Wildcats", "Toledo Rockets")]
    logs = [bands.LogGame(Date(2025, 1, 1), "Kentucky", 70, 90, 60)]
    ctx = bands.context({"e": ("Kentucky Wildcats", "Toledo Rockets")}, Date(2025, 1, 10), {Date(2025, 1, 10): board},
                        logs, {"Kentucky": "SEC", "Toledo": "MAC"})["e"]
    assert ctx.home.net_rank is None and ctx.away.net_rank is None
    assert bands.torvik_ranks([{"team": "A", "rank": "3"}, {"team": "B", "rank": ""}]) == {"A": 3}


def test_bounce_back_and_weak_game_bands() -> None:
    weak_loser = bands.Team(net_rank=250, off_loss=True)
    weak_winner = bands.Team(net_rank=300, off_loss=False)
    good_loser = bands.Team(net_rank=50, off_loss=True)
    ctx = {"e": bands.GameCtx(home=weak_loser, away=weak_winner), "f": bands.GameCtx(home=good_loser, away=weak_loser)}
    rows = [_row("e", "game_ats", "home", -3.0), _row("e", "game_ats", "away", 3.0, outcome="loss"),
            _row("f", "game_ats", "home", -8.0), _row("f", "game_ats", "away", 8.0, outcome="loss")]
    reads = {b.name: b for b in bands.read_bands(rows, ctx)}
    assert reads["weak game (both Torvik rank > 200) favourite ATS"].record == "1-0"
    assert reads["weak game (both Torvik rank > 200) underdog ATS"].record == "0-1"
    assert reads["bounce back: side off a loss ATS"].record == "1-0"  # e home only; f has both off a loss
    assert reads["bounce back: both off a loss, favourite ATS"].record == "1-0"
    assert reads["bounce back: both off a loss, underdog ATS"].record == "0-1"


def _on(day: str, tip: str, **kw: object) -> ledger.Row:
    r = _row("e", "game_total", "over", 140.5, day=day, **kw)  # type: ignore[arg-type]
    r.tip_utc = tip
    return r


def test_a_game_on_three_slates_counts_once_from_its_own_tip_day() -> None:
    tip = "2026-11-03T00:30:00Z"  # Monday 7:30 pm ET, so Monday Nov 2 is its slate
    rows = [_on(d, tip) for d in ("2026-11-01", "2026-11-02", "2026-11-03")]
    (r,) = bands.main_rows(rows)
    assert r.slate_date == "2026-11-02"


def test_without_its_tip_day_slate_the_latest_earlier_slate_wins() -> None:
    tip = "2026-11-03T00:30:00Z"
    assert [r.slate_date for r in bands.one_slate_per_game([_on(d, tip) for d in ("2026-11-01", "2026-11-03")])] == [
        "2026-11-01"
    ]
    assert [r.slate_date for r in bands.one_slate_per_game([_on("2026-11-04", tip)])] == ["2026-11-04"]


def _shot_game(day: Date, team: str, two: tuple[float, float], three: tuple[float, float]) -> bands.LogGame:
    return bands.LogGame(day, team, 70, 70, 70, two, three, (10.0, 20.0), (5.0, 15.0))


def test_shooting_gaps_need_ten_prior_games_and_pool_over_shots() -> None:
    logs = [_shot_game(Date(2025, 1, d), "A", (12.0, 20.0), (4.0, 20.0)) for d in range(1, 10)]
    assert bands.shooting(logs, Date(2025, 1, 20)) == {}  # nine games
    logs.append(_shot_game(Date(2025, 1, 10), "A", (12.0, 20.0), (4.0, 20.0)))
    assert bands.shooting(logs, Date(2025, 1, 10)) == {}  # the tenth is on the day itself
    two, three = bands.shooting(logs, Date(2025, 1, 11))["A"]
    assert two == pytest.approx(0.6 - 0.5) and three == pytest.approx(0.2 - 1 / 3)


def test_exploratory_inside_shooting_band_takes_only_the_underdog() -> None:
    inside = bands.Team(two=0.05, three=-0.03)
    outside = bands.Team(two=-0.02, three=0.04)
    ctx = {"e": bands.GameCtx(home=outside, away=inside), "f": bands.GameCtx(home=inside, away=outside),
           "g": bands.GameCtx(home=outside, away=bands.Team(two=0.05))}
    rows = [_row("e", "game_ats", "home", -4.0, outcome="loss"), _row("e", "game_ats", "away", 4.0),
            _row("f", "game_ats", "home", -4.0), _row("f", "game_ats", "away", 4.0, outcome="loss"),
            _row("g", "game_ats", "away", 4.0)]
    band = {b.name: b for b in bands.read_bands(rows, ctx)}["exploratory: underdog better 2P%, worse 3P% (10+ games) ATS"]
    assert (band.n, band.record) == (1, "1-0")  # e's away dog; f's inside team is the favourite; g has no 3P gap


def test_context_carries_shooting_gaps() -> None:
    board = [_game("Kentucky Wildcats", "Toledo Rockets")]
    logs = [_shot_game(Date(2025, 1, d), "Kentucky", (12.0, 20.0), (4.0, 20.0)) for d in range(1, 11)]
    ctx = bands.context({"e": ("Kentucky Wildcats", "Toledo Rockets")}, Date(2025, 1, 12), {Date(2025, 1, 12): board},
                        logs, {"Kentucky": "SEC", "Toledo": "MAC"})["e"]
    assert ctx.home.two == pytest.approx(0.1) and ctx.away.two is None
