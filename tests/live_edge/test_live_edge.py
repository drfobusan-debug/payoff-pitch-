from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import erf, sqrt
from pathlib import Path

from live_edge.engine import (
    Flag,
    LiveEngine,
    Thresholds,
    consensus_prior,
    devig_pairs,
    grade_flag,
    match_event,
)
from live_edge.espn import LiveGame, parse_pregame_line, parse_scoreboard
from live_edge.oddsapi import age_seconds, parse_events
from live_edge.roller import LiveState, Prior, Roller

NOW = datetime(2026, 9, 20, 18, 0, tzinfo=timezone.utc)
STAMP = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- roller ----
def test_pregame_roll_reproduces_prior():
    r = Roller("nfl")
    prior = Prior(margin=3.0, total=45.0, margin_sd=13.2, total_sd=13.4)
    d = r.roll(prior, LiveState(0, 0, period=1, clock_secs=900))
    # The fitted line-carry coefficient (a=1.032) is applied even at kickoff.
    assert abs(d.margin_mu - 3.0) < 0.2
    assert abs(d.total_mu - 45.0) < 3.0
    assert abs(d.margin_sd - 13.2) < 0.6  # fitted curve at f=1 (15.41) scaled to the NFL
    assert 0.55 < d.p_home_win() < 0.62
    assert abs(d.p_home_cover(-3.0) - 0.5) < 0.01
    assert abs(d.p_over(d.total_mu) - 0.5) < 1e-6


def test_underdog_early_lead_is_shrunk_toward_prior():
    """A 10-0 Q1 lead by a 8.5-pt dog should NOT be priced like a coin flip + 10."""
    r = Roller("nfl")
    prior = Prior(margin=8.5, total=41.5, margin_sd=13.2, total_sd=13.4)
    d = r.roll(prior, LiveState(home_score=0, away_score=10, period=1, clock_secs=300))
    score_only = 0.5 * (1 + erf(-10 / (d.margin_sd * sqrt(2))))  # ignore the prior entirely
    assert d.p_home_win() > score_only + 0.15  # prior still pulls the favourite back
    assert 0.3 < d.p_home_win() < 0.5  # but the ten points make the dog the live favourite


def test_variance_collapses_with_clock():
    r = Roller("cfb")
    prior = Prior(0.0, 55.0, 16.0, 13.0)
    early = r.roll(prior, LiveState(7, 7, 1, 600))
    late = r.roll(prior, LiveState(21, 21, 4, 120))
    assert late.margin_sd < early.margin_sd / 3
    assert abs(late.p_home_win() - 0.5) < 0.05


def test_end_of_game_is_deterministic():
    r = Roller("nfl")
    d = r.roll(Prior(0.0, 44.0, 13.2, 13.4), LiveState(24, 17, 4, 0))
    assert d.p_home_win() > 0.999
    assert d.p_home_cover(-7.0) > 0.4  # push-ish territory at exactly the margin
    assert d.p_over(40.5) > 0.999


def test_possession_and_field_position_shift_margin():
    r = Roller("nfl")
    base = LiveState(14, 14, 3, 600)
    red_zone = LiveState(14, 14, 3, 600, possession_home=True, yards_to_goal=5)
    prior = Prior(0.0, 44.0, 13.2, 13.4)
    assert r.roll(prior, red_zone).margin_mu > r.roll(prior, base).margin_mu + 1.0


# ------------------------------------------------------------------ espn ----
def _espn_payload(state="in", clock=808.0, period=2, poss="34"):
    return {
        "events": [
            {
                "id": "401872934",
                "date": "2026-09-20T17:00Z",
                "competitions": [
                    {
                        "status": {
                            "clock": clock,
                            "displayClock": "13:28",
                            "period": period,
                            "type": {"state": state},
                        },
                        "situation": {
                            "possession": poss,
                            "yardLine": 75,
                            "lastPlay": {"probability": {"homeWinPercentage": 0.687}},
                        },
                        "competitors": [
                            {
                                "homeAway": "home",
                                "id": "34",
                                "score": "7",
                                "team": {
                                    "id": "34",
                                    "displayName": "Houston Texans",
                                    "location": "Houston",
                                    "name": "Texans",
                                    "abbreviation": "HOU",
                                },
                            },
                            {
                                "homeAway": "away",
                                "id": "4",
                                "score": "3",
                                "team": {
                                    "id": "4",
                                    "displayName": "Cincinnati Bengals",
                                    "location": "Cincinnati",
                                    "name": "Bengals",
                                    "abbreviation": "CIN",
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    }


def test_parse_scoreboard_live_game():
    (g,) = parse_scoreboard(_espn_payload())
    assert g.event_id == "401872934"
    assert g.status == "in"
    assert g.matchup == "Cincinnati Bengals @ Houston Texans"
    assert g.state == LiveState(7, 3, 2, 808, possession_home=True, yards_to_goal=25)
    assert g.espn_home_wp == 0.687
    assert "houstontexans" in g.home_aliases and "hou" in g.home_aliases


def test_parse_scoreboard_pregame_has_no_situation():
    (g,) = parse_scoreboard(_espn_payload(state="pre"))
    assert g.state.period == 1 and g.state.clock_secs == 900
    assert g.state.possession_home is None and g.state.yards_to_goal is None


def test_parse_pregame_line_median_of_providers():
    summary = {
        "pickcenter": [
            {"spread": -2.5, "overUnder": 45.5},
            {"spread": -3.0, "overUnder": 46.0},
            {"spread": -3.0, "overUnder": 45.5},
            {"spread": None, "overUnder": 44.0},
        ]
    }
    assert parse_pregame_line(summary) == (-3.0, 45.5)
    assert parse_pregame_line({"pickcenter": []}) is None
    assert parse_pregame_line({}) is None


# --------------------------------------------------------------- odds api ----
def _bookmaker(key, home, away, stamp, home_ml, away_ml):
    return {
        "key": key,
        "last_update": stamp,
        "markets": [
            {
                "key": "h2h",
                "last_update": stamp,
                "outcomes": [
                    {"name": home, "price": home_ml},
                    {"name": away, "price": away_ml},
                ],
            },
            {
                "key": "spreads",
                "last_update": stamp,
                "outcomes": [
                    {"name": home, "price": -110, "point": -3.0},
                    {"name": away, "price": -110, "point": 3.0},
                ],
            },
            {
                "key": "totals",
                "last_update": stamp,
                "outcomes": [
                    {"name": "Over", "price": -105, "point": 45.5},
                    {"name": "Under", "price": -115, "point": 45.5},
                ],
            },
        ],
    }


def _board(
    home="Houston Texans",
    away="Cincinnati Bengals",
    stamp=STAMP,
    home_ml=-150,
    away_ml=130,
    books=("draftkings", "fanduel"),
):
    return [
        {
            "id": "abc",
            "commence_time": "2026-09-20T17:00:00Z",
            "home_team": home,
            "away_team": away,
            "bookmakers": [_bookmaker(b, home, away, stamp, home_ml, away_ml) for b in books],
        }
    ]


def test_parse_events_and_pairs():
    (ev,) = parse_events(_board())
    assert ev.home == "Houston Texans" and len(ev.quotes) == 12
    pairs = ev.pairs()
    assert len(pairs) == 6
    markets = {a.market for a, _ in pairs}
    assert markets == {"ml", "spread", "total"}
    sp = next((a, b) for a, b in pairs if a.market == "spread")
    assert sp[0].side == "home" and sp[0].line == -3.0
    assert sp[1].side == "away" and sp[1].line == 3.0


def test_devig_pairs_are_proportional_and_stale_quotes_drop():
    (ev,) = parse_events(_board())
    rows = devig_pairs(ev, NOW + timedelta(seconds=30), max_age=90)
    assert len(rows) == 6
    ml = next(r for r in rows if r[0].market == "ml")
    _, p_home, _, p_away = ml
    assert abs(p_home + p_away - 1.0) < 1e-9
    assert 0.56 < p_home < 0.58  # -150/+130 -> 0.60/0.435 vig -> ~0.58
    assert devig_pairs(ev, NOW + timedelta(seconds=600), max_age=90) == []


def test_age_seconds():
    assert age_seconds(STAMP, NOW) == 0.0
    assert age_seconds(STAMP, NOW + timedelta(seconds=45)) == 45.0
    assert age_seconds("", NOW) is None


def test_consensus_prior_from_board():
    (ev,) = parse_events(_board())
    pr = consensus_prior(ev, "nfl")
    assert pr is not None
    assert pr.margin == 3.0 and pr.total == 45.5


# --------------------------------------------------------------- matching ----
def test_match_event_by_alias():
    (g,) = parse_scoreboard(_espn_payload())
    (ev,) = parse_events(_board())
    assert match_event(g, [ev]) is ev
    other = parse_events(_board(home="Dallas Cowboys", away="Washington Commanders"))
    assert match_event(g, other) is None


# ----------------------------------------------------------------- engine ----
def _live_game(home=0, away=10, period=1, clock=300, status="in"):
    return LiveGame(
        event_id="401872935",
        home="Tampa Bay Buccaneers",
        away="Cleveland Browns",
        status=status,
        state=LiveState(home, away, period, clock),
        home_aliases=("tampabaybuccaneers", "tb"),
        away_aliases=("clevelandbrowns", "cle"),
    )


def _live_board(home_ml, away_ml, stamp=STAMP):
    return parse_events(_board("Tampa Bay Buccaneers", "Cleveland Browns", stamp, home_ml, away_ml))


def test_engine_flags_after_two_ticks_and_grades(tmp_path: Path):
    eng = LiveEngine(
        "nfl", root=tmp_path, thresholds=Thresholds(), prior_fallback=lambda s, e: (-8.5, 41.5)
    )
    # Book has overreacted: TB (8.5 favourite) down 10 in Q1, priced as a +150 dog.
    board = _live_board(home_ml=250, away_ml=-300)
    r1 = eng.tick([_live_game()], board, now=NOW)
    assert r1.priced == 1 and r1.flags == []
    r2 = eng.tick([_live_game(clock=240)], board, now=NOW + timedelta(seconds=60))
    ml = [f for f in r2.flags if f.market == "ml"]
    assert len(ml) == 1 and ml[0].side == "home" and ml[0].price == 250
    assert ml[0].p_model > ml[0].p_market + 0.06
    # No duplicate flag on the third tick.
    r3 = eng.tick([_live_game(clock=180)], board, now=NOW + timedelta(seconds=120))
    assert not any(f.market == "ml" and f.side == "home" for f in r3.flags)
    # Final: TB wins 24-20 -> ML flag graded win at +150.
    r4 = eng.tick([_live_game(24, 20, 4, 0, status="post")], [], now=NOW + timedelta(hours=3))
    flags = eng.load_flags()
    graded = [f for f in flags if f.market == "ml"]
    assert r4.graded >= 1 and graded[0].result == "win" and graded[0].pnl == 2.5
    assert (tmp_path / "priors.json").exists()
    tape = list((tmp_path / "tape").glob("nfl_*.jsonl"))
    assert tape and len(tape[0].read_text().splitlines()) == 3


def test_single_book_outlier_does_not_flag(tmp_path: Path):
    eng = LiveEngine("nfl", root=tmp_path, prior_fallback=lambda s, e: (-8.5, 41.5))
    board = parse_events(
        _board("Tampa Bay Buccaneers", "Cleveland Browns", STAMP, 250, -300, books=("betmgm",))
    )
    for i in range(3):
        rep = eng.tick([_live_game()], board, now=NOW + timedelta(seconds=60 * i))
        assert rep.flags == []


def test_engine_no_prior_is_reported_not_priced(tmp_path: Path):
    eng = LiveEngine("nfl", root=tmp_path, prior_fallback=lambda s, e: None)
    rep = eng.tick([_live_game()], _live_board(250, -300), now=NOW)
    assert rep.live == 1 and rep.priced == 0
    assert rep.no_prior == ["Cleveland Browns @ Tampa Bay Buccaneers"]


def test_engine_fair_price_produces_no_flag(tmp_path: Path):
    eng = LiveEngine("nfl", root=tmp_path, prior_fallback=lambda s, e: (-8.5, 41.5))
    dist = Roller("nfl").roll(Prior(8.5, 41.5, 13.2, 13.4), LiveState(0, 10, 1, 300))
    p = dist.p_home_win()
    fair_home = -round(100 * p / (1 - p)) if p > 0.5 else round(100 * (1 - p) / p)
    fair_away = -fair_home
    board = _live_board(fair_home, fair_away)
    for i in range(3):
        rep = eng.tick([_live_game()], board, now=NOW + timedelta(seconds=60 * i))
        assert not [f for f in rep.flags if f.market == "ml"]


def test_engine_garbage_time_gate(tmp_path: Path):
    eng = LiveEngine("nfl", root=tmp_path, prior_fallback=lambda s, e: (-8.5, 41.5))
    board = _live_board(250, -300)
    for i in range(3):
        rep = eng.tick([_live_game(20, 20, 4, 120)], board, now=NOW + timedelta(seconds=60 * i))
        assert rep.flags == []


# ---------------------------------------------------------------- grading ----
def _flag(market, side, line, price=-110):
    return Flag(
        STAMP,
        "nfl",
        "e",
        "A @ B",
        market,
        side,
        line,
        "dk",
        price,
        0.6,
        0.5,
        0.1,
        0.1,
        2,
        600,
        7,
        3,
        3.0,
        45.0,
        None,
    )


def test_grade_flag_all_markets():
    assert grade_flag(_flag("ml", "home", None, 150), 24, 20) == ("win", 1.5)
    assert grade_flag(_flag("ml", "away", None), 24, 20) == ("loss", -1.0)
    assert grade_flag(_flag("ml", "home", None), 20, 20) == ("push", 0.0)
    assert grade_flag(_flag("spread", "home", -3.0), 24, 20)[0] == "win"
    assert grade_flag(_flag("spread", "home", -4.0), 24, 20) == ("push", 0.0)
    assert grade_flag(_flag("spread", "away", 3.0), 24, 20)[0] == "loss"
    assert grade_flag(_flag("spread", "away", 7.5), 24, 20)[0] == "win"
    assert grade_flag(_flag("total", "over", 43.5), 24, 20)[0] == "win"
    assert grade_flag(_flag("total", "under", 43.5), 24, 20)[0] == "loss"
    assert grade_flag(_flag("total", "over", 44.0), 24, 20) == ("push", 0.0)
    assert grade_flag(_flag("spread", "home", -3.0), 24, 20)[1] == round(100 / 110, 4)
