"""Graded-only market bands: main-line rows, favourite/underdog, back-to-backs, verdicts."""

from __future__ import annotations

from nhl_engine.audit import bands, ledger_report
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.audit.screens import Context


def _row(
    market="game_ml",
    side="H",
    consensus=0.6,
    pnl=0.5,
    line=None,
    slate="2026-10-10",
    matchup="A @ H",
    entity="",
    tier="Pass",
    model=None,
) -> LedgerRow:
    return LedgerRow(
        slate_date=slate,
        matchup=matchup,
        home="H",
        away="A",
        market=market,
        side=side,
        entity=entity,
        line=line,
        ot_rule="incl_ot_so",
        book="b",
        american=-150.0,
        books=3,
        consensus=consensus,
        model_prob=consensus if model is None else model,
        push_prob=0.0,
        edge=0.0,
        ev=0.0,
        tier=tier,
        outcome="win" if pnl > 0 else "loss",
        pnl=pnl,
    )


def _reads(rows, ctx=None):
    return {b.name: b for b in bands.read_bands(rows, ctx or Context())}


def test_sides_split_by_home_away_and_favourite():
    r = _reads([_row(side="H", consensus=0.6, pnl=0.67), _row(side="A", consensus=0.4, pnl=-1)])
    assert r["ML home favourite"].n == 1 and r["ML home favourite"].units == 0.67
    assert r["ML away underdog"].record == "0-1"
    assert r["ML home underdog"].n == 0 and r["ML away favourite"].n == 0


def test_main_total_line_and_puck_line_only():
    rows = [
        _row("game_total", "Over", 0.62, -1, line=5.5),
        _row("game_total", "Over", 0.49, 0.9, line=6.5),
        _row("game_total", "Under", 0.51, -1, line=6.5),
        _row("game_pl", "H", 0.4, 1.2, line=-1.5),
        _row("game_pl", "H", 0.2, 3.0, line=-2.5),
    ]
    r = _reads(rows)
    assert r["Total Over"].units == 0.9 and r["Total Under"].n == 1
    assert r["Puck line home underdog"].units == 1.2


def test_back_to_back_bands_read_the_context():
    ctx = Context(b2b=frozenset({"2026-10-10|A"}))
    rows = [
        _row(side="A", consensus=0.45, pnl=-1),
        _row(side="H", consensus=0.55, pnl=0.8),
        _row("game_total", "Over", 0.5, -1, line=6.0),
        _row("game_total", "Under", 0.5, 0.9, line=6.0),
    ]
    r = _reads(rows, ctx)
    assert r["ML road team on a back-to-back"].n == 1
    assert r["ML rested team vs a back-to-back"].units == 0.8
    assert r["Total Under, a team on a back-to-back"].n == 1


def test_sixty_minute_favourite_underdog_draw():
    rows = [
        _row("game_ml3", "H", 0.45, 1.2),
        _row("game_ml3", "A", 0.33, -1),
        _row("game_ml3", "Draw", 0.22, -1),
    ]
    r = _reads(rows)
    assert r["60-min favourite"].units == 1.2 and r["60-min home favourite"].n == 1
    assert r["60-min underdog"].n == 1 and r["60-min draw"].n == 1


def test_verdict_needs_the_sample_and_both_halves():
    win = bands.BandRead("x", [1.0, -1.0, 1.0] * 40)
    assert win.verdict == "wins: clears the bar"
    lose = bands.BandRead("x", [-1.0, -1.0, 1.0] * 40)
    assert lose.verdict == "loses: refusal candidate"
    assert bands.BandRead("x", [1.0] * 50).verdict == "probation (50/100 games)"


def test_audit_report_renders_the_bands():
    audit = ledger_report.build([_row()])
    md = ledger_report.render_md(audit, as_of="t")
    assert "Graded-only bands" in md and "| ML home favourite | 1 |" in md
    assert "ML home favourite" in ledger_report.render_html(audit, as_of="t")


def test_prop_bands_read_unders_per_player_and_se_by_game():
    rows = [
        _row("g_saves", "under", 0.5, 0.9, line=24.5, entity="G1", model=0.65),
        _row("g_saves", "under", 0.5, -1, line=25.5, entity="G2", model=0.55),
        _row("g_saves", "over", 0.5, 0.9, line=24.5, entity="G1"),
        _row("sk_g", "under", 0.7, 0.4, line=0.5, entity="S1", tier="Strong buy"),
        _row("sk_g", "under", 0.7, -1, line=0.5, entity="S2"),
        _row("sk_g", "under", 0.9, 0.1, line=1.5, entity="S1"),
        _row("sk_g", "under", 0.7, 0.4, line=0.5, entity="S3", matchup="B @ C"),
    ]
    r = _reads(rows)
    assert r["Goalie saves Under"].n == 2 and r["Goalie saves Under"].record == "1-1"
    assert r["Goalie saves Under, sim 10+ pts above the market"].n == 1
    no_goal = r["Player no goal (goals Under 0.5)"]
    assert no_goal.n == 3 and len(no_goal.games) == 2
    assert r["Player no goal, sim rates it a buy"].units == 0.4


def test_the_study_leans_are_graded_like_any_band():
    ctx = Context(b2b=frozenset({"2026-10-10|A"}))
    rows = [
        _row(side="H", consensus=0.70, pnl=0.4),
        _row(side="A", consensus=0.30, pnl=-1),
        _row("game_pl", "H", 0.45, 1.1, line=-1.5),
    ]
    r = _reads(rows, ctx)
    assert r["ML lean: home team vs a road back-to-back"].units == 0.4
    assert r["Puck line lean: home team vs a road back-to-back"].units == 1.1
    assert r["ML lean: home favourite >= 65%"].n == 1
    assert set(bands.LEANS) >= {
        "ML lean: home team vs a road back-to-back",
        "ML lean: home favourite >= 65%",
        f"ML lean (fade): starting goalie {bands.GSAX_GAP:.2f}+ GSAx/60 better",
    }


def test_the_goalie_lean_reads_the_priced_gsax_and_skips_rows_without_it():
    hot = _row(side="H", pnl=-1)
    hot.home_goalie_gsax60, hot.away_goalie_gsax60 = 0.30, 0.05
    cold = _row(side="A", matchup="B @ H", pnl=0.9)
    cold.home_goalie_gsax60, cold.away_goalie_gsax60 = 0.30, 0.05
    old = _row(side="H", matchup="C @ H", pnl=0.5)  # recorded before the field existed
    name = f"ML lean (fade): starting goalie {bands.GSAX_GAP:.2f}+ GSAx/60 better"
    read = _reads([hot, cold, old])[name]
    assert (read.n, read.units) == (1, -1)


def test_tonight_lists_each_lean_on_an_ungraded_card():
    ctx = Context(b2b=frozenset({"2026-10-10|A"}))
    rows = [
        _row(side="H", consensus=0.70),
        _row(side="H", consensus=0.70),  # a second book on the same side
        _row("game_pl", "H", 0.45, line=-1.5),
        _row("game_pl", "H", 0.30, line=-2.5),  # alternate line: not read
    ]
    for r in rows:
        r.outcome = r.pnl = None
    got = bands.tonight(rows, ctx)
    assert got["ML lean: home team vs a road back-to-back"] == ["H ML (A @ H)"]
    assert got["Puck line lean: home team vs a road back-to-back"] == ["H -1.5 (A @ H)"]
    assert got["ML lean: home favourite >= 65%"] == ["H ML (A @ H)"]
