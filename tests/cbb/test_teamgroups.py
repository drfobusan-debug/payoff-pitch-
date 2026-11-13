from __future__ import annotations

from datetime import date as Date

import pytest

from cbb_engine import audit_report, bands, teamgroups
from cbb_engine.data import espn
from engine_runtime import faststart, ledger, quotes

DAY = "2026-11-10"
TIP = f"{DAY}T23:00:00Z"


def _q(eid: str, market: str, side: str, line: float, book: str, at: str = f"{DAY}T22:50:00Z") -> quotes.Quote:
    return quotes.Quote(at, DAY, eid, TIP, market, side, "", line, book, -110, -110, "c1", "close")


def test_result_is_favourite_cover_margin_and_points_over_the_total() -> None:
    home_fav = teamgroups.Called("e", -5.0, 140.0, 8, 150)
    assert home_fav.result("spread") == 3.0 and home_fav.result("total") == 10.0
    away_fav = teamgroups.Called("e", 4.0, None, -10, 130)
    assert away_fav.result("spread") == 6.0 and away_fav.result("total") is None
    assert teamgroups.Called("e", 0.0, 140.0, 1, 141).result("spread") is None


def test_called_reads_the_timely_close_median_and_the_espn_final_either_way_round() -> None:
    shards = [_q("e1", "game_ats", "home", -6.0, b) for b in ("a", "b")] + [_q("e1", "game_ats", "home", -5.0, "c")]
    shards += [_q("e1", "game_total", "over", 150.5, "a")]
    shards += [_q("e1", "game_total", "over", 160.0, "a", at=f"{DAY}T20:00:00Z")]  # early close: ignored
    shards += [_q("e2", "game_total", "over", 140.0, "a")]  # no final matched
    # ESPN lists the neutral-site game the other way round: Duke (odds home) is ESPN's away team
    g = espn.Game(espn_id="1", start_utc="", away="Duke", home="Kansas", state="STATUS_FINAL", neutral=True,
                  conference_game=False, away_halves=(40, 40), home_halves=(35, 35))
    out = teamgroups.called(shards, {"e1": ("Duke", "Kansas"), "e2": ("Army", "Navy")}, [g])
    assert set(out) == {"e1"}
    assert out["e1"] == teamgroups.Called("e1", -6.0, 150.5, 10, 150, 75)


def _row(eid: str, market: str, side: str, line: float, *, clv: float, close: float) -> ledger.Row:
    return ledger.Row(slate_date=DAY, event_id=eid, tip_utc=TIP, market=market, side=side, line=line, book="draftkings",
                      pass_id="p", pass_kind="board", bet_of_record=True, status="research", american=-110.0, p_fair=0.5,
                      outcome="win", pnl=0.909, close_status="pre_tip", clv_consensus=clv, close_consensus=close)


def test_read_groups_games_by_pairing_and_reads_the_favourite_rows() -> None:
    fast, slow = bands.Team(pace="fast", power=True), bands.Team(pace="slow", power=False)
    ctx = {"a": bands.GameCtx(home=fast, away=fast), "b": bands.GameCtx(home=fast, away=slow),
           "c": bands.GameCtx(home=slow, away=slow), "d": bands.GameCtx(home=slow, away=fast)}
    games = {
        "a": teamgroups.Called("a", -4.0, 150.0, 10, 160),  # fav covers by 6, Over by 10
        "b": teamgroups.Called("b", -4.0, 150.0, 2, 140),
        "c": teamgroups.Called("c", 3.0, 130.0, -1, 128),  # away fav by 3, won by 1: -2
        "d": teamgroups.Called("d", -2.0, 140.0, 0, 142),
    }
    rows = [_row("a", "game_ats", "home", -4.0, clv=1.0, close=0.53), _row("a", "game_ats", "away", 4.0, clv=-1.0, close=0.47),
            _row("b", "game_ats", "home", -4.0, clv=3.0, close=0.55)]
    reads = {(g.group, g.market): g for g in teamgroups.read(rows, ctx, games)}
    assert reads[("all games", "spread")].games == 4
    both_fast = reads[("both fast", "spread")]
    assert both_fast.games == 1 and both_fast.miss == 6.0 and both_fast.bias == 6.0
    assert both_fast.clv_n == 1 and both_fast.clv == 1.0
    assert both_fast.close_ev == pytest.approx(0.53 * (1 + 100 / 110) - 1)
    mixed = reads[("fast vs slow", "spread")]
    assert mixed.games == 2 and mixed.miss == pytest.approx(2.0) and mixed.vs_rest == pytest.approx(2.0 - 4.0)
    assert reads[("Power vs mid-major", "spread")].games == 2
    assert ("Power vs mid-major", "total") not in reads
    assert reads[("Power vs mid-major, Power fav 10-29.5", "total")].games == 0
    assert reads[("mid-major vs mid-major", "total")].bias == -2.0
    assert mixed.verdict == "underpowered (2/150 games)"


def test_power_vs_mid_total_keeps_only_power_favoured_by_10_to_29_5() -> None:
    power, mid = bands.Team(power=True), bands.Team(power=False)
    ctx = {e: bands.GameCtx(home=power, away=mid) for e in ("a", "b", "c", "d", "e")}
    ctx["f"] = bands.GameCtx(home=mid, away=power)
    games = {
        "a": teamgroups.Called("a", -9.5, 150.0, 20, 140),
        "b": teamgroups.Called("b", -10.0, 150.0, 20, 140),
        "c": teamgroups.Called("c", -29.75, 150.0, 20, 140),
        "d": teamgroups.Called("d", -30.0, 150.0, 20, 140),
        "e": teamgroups.Called("e", None, 150.0, 20, 140),
        "f": teamgroups.Called("f", 15.0, 150.0, -20, 160),  # Power team on the road, favoured by 15
    }
    reads = {(g.group, g.market): g for g in teamgroups.read([], ctx, games)}
    trimmed = reads[("Power vs mid-major, Power fav 10-29.5", "total")]
    assert trimmed.games == 3 and trimmed.bias == pytest.approx((-10 - 10 + 10) / 3)
    assert reads[("Power vs mid-major", "spread")].games == 5


def test_verdict_calls_only_at_150_games_and_two_se() -> None:
    base = dict(dimension="tempo", group="g", market="total", miss=10.0, miss_se=0.5, bias=0.0, bias_se=0.5,
                clv_n=0, clv=0.0, clv_se=0.0, close_ev=0.0, close_ev_se=0.0)
    assert teamgroups.GroupRead(games=149, vs_rest=5.0, vs_rest_se=0.5, **base).verdict.startswith("underpowered")  # type: ignore[arg-type]
    assert teamgroups.GroupRead(games=150, vs_rest=1.1, vs_rest_se=0.5, **base).verdict == "close less accurate than the rest"  # type: ignore[arg-type]
    assert teamgroups.GroupRead(games=150, vs_rest=0.9, vs_rest_se=0.5, **base).verdict == "no difference"  # type: ignore[arg-type]


def test_tempo_thirds_cut_torvik_adjusted_tempo() -> None:
    rows = [{"team": t, "adjt": str(p)} for t, p in (("A", 60), ("B", 65), ("C", 70), ("D", 75), ("E", 80), ("F", 85))]
    rows.append({"team": "G", "adjt": ""})  # unreadable: left out
    assert bands.tempo_thirds(rows) == {"A": "slow", "B": "slow", "C": "mid", "D": "mid", "E": "fast", "F": "fast"}


def test_audit_report_prints_the_team_groups_table() -> None:
    g = teamgroups.GroupRead("tempo", "both fast", "total", 3, 9.0, 1.0, 0.5, 1.2, 2.0, 1.0, 2, 0.4, 0.2, -0.03, 0.01)
    rep = audit_report.build({DAY: []}, Date.fromisoformat(DAY), groups=[g])
    text = audit_report.render_text(rep)
    assert "Team groups" in text and "both fast" in text and "underpowered (3/150 games)" in text
    assert "returning-minutes" in text


def test_audit_report_prints_the_fast_start_table() -> None:
    obs = [faststart.Obs(f"g{i}", f"2026-11-0{1 + i // 40}", 140.0, 66.0 + (i % 9 - 4) * 3, 140.0 + (i % 9 - 4) * 3) for i in range(80)]
    fast = faststart.read("half", obs)
    assert fast is not None
    text = audit_report.render_text(audit_report.build({DAY: []}, Date.fromisoformat(DAY), fast=[fast]))
    assert "Fast starts" in text and "half" in text and "in-play totals" in text
