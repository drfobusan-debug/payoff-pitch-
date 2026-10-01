"""Starter length from the book's pitcher-outs line and his median pitch count."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from mlb_engine.features.market_length import (
    LEAGUE_PITCHES_PER_BF,
    OUTS_SD,
    market_outs_lines,
    market_outs_mean,
    starter_length,
)
from mlb_engine.market.ev import MarketQuote
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig

AS_OF = date(2026, 8, 20)
M = "NYY @ BOS"


def _starts(pitches: list[int], bf: int = 24, outs: int = 18) -> pd.DataFrame:
    """One row per pitch; the last ``bf`` pitches of each start end a PA."""
    rows = []
    for i, n in enumerate(pitches):
        d = AS_OF - timedelta(days=5 * (len(pitches) - i))
        events = [None] * (n - bf) + ["single"] * (bf - outs) + ["strikeout"] * outs
        rows += [{"game_date": d, "events": e} for e in events]
    return pd.DataFrame(rows)


def test_lines_need_two_sided_price_and_match_canonical_name() -> None:
    quotes = {
        (M, "pitcher_outs", "José Berríos Outs o17.5"): [
            MarketQuote("a", -120.0, opposite_american=100.0),
            MarketQuote("b", -110.0, opposite_american=-110.0),
        ],
        (M, "pitcher_outs", "Jose Berrios Outs o15.5"): [MarketQuote("a", -300.0)],
        (M, "pitcher_outs", "Other Guy Outs o16.5"): [
            MarketQuote("a", -110.0, opposite_american=-110.0)
        ],
        ("X @ Y", "pitcher_outs", "Jose Berrios Outs o16.5"): [
            MarketQuote("a", -110.0, opposite_american=-110.0)
        ],
    }
    lines = market_outs_lines(quotes, M, "Jose Berrios")
    assert [ln for ln, _ in lines] == [17.5]
    fair = sorted(q.no_vig_prob for q in quotes[(M, "pitcher_outs", "José Berríos Outs o17.5")])
    assert lines[0][1] == pytest.approx(sum(fair) / 2)


def test_market_mean_from_even_money_is_the_line() -> None:
    assert market_outs_mean([(16.5, 0.5)]) == pytest.approx(16.5)


def test_market_mean_moves_with_price_and_skips_extremes() -> None:
    mu = market_outs_mean([(16.5, 0.6), (20.5, 0.99)])
    assert mu is not None and mu > 16.5
    assert mu == pytest.approx(market_outs_mean([(16.5, 0.6)]))
    assert market_outs_mean([(14.5, 0.99)]) is None
    assert market_outs_mean([]) is None


def test_market_mean_reads_two_lines_consistently() -> None:
    from statistics import NormalDist

    n = NormalDist(17.0, OUTS_SD)
    lines = [(15.5, 1 - n.cdf(15.5)), (17.5, 1 - n.cdf(17.5))]
    assert market_outs_mean(lines) == pytest.approx(17.0)


def _length(rows: pd.DataFrame, lines: list[tuple[float, float]], **kw: float):
    args = dict(weight=1.0, bf_buffer=0.0, bf_sd=3.0, pitch_buffer=8, max_bf=30)
    args.update(kw)
    return starter_length(rows, AS_OF, lines, **args)  # type: ignore[arg-type]


def test_too_few_starts_leaves_workload_cap_in_charge() -> None:
    assert _length(_starts([90]), [(17.5, 0.5)]) is None


def test_no_line_hooks_on_median_pitch_count_of_last_six() -> None:
    rows = _starts([40, 100, 100, 90, 92, 94, 96])  # the 40 is a seventh start back
    length = _length(rows, [])
    assert length is not None and length.source == "pitch_count"
    assert length.pitch_cap == 95 + 8
    assert length.bf_cap == 24 and length.bf_sd == 0.0


def test_market_line_sets_batters_faced_hook() -> None:
    rows = _starts([90] * 6)  # 24 BF per 18 outs, shrunk toward the league
    deep = _length(rows, [(20.5, 0.5)])
    short = _length(rows, [(14.5, 0.5)])
    assert deep is not None and short is not None
    assert deep.source == "market" and deep.market_outs == pytest.approx(20.5)
    assert deep.bf_cap > short.bf_cap
    # The pitch hook never undercuts the market's length on an average count.
    assert deep.pitch_cap >= round(deep.bf_cap * LEAGUE_PITCHES_PER_BF)


def test_weight_blends_toward_recent_batters_faced() -> None:
    rows = _starts([90] * 6)
    book = _length(rows, [(26.5, 0.5)], weight=1.0)
    own = _length(rows, [(26.5, 0.5)], weight=0.0)
    assert book is not None and own is not None
    assert own.bf_cap == 24
    assert book.bf_cap > own.bf_cap


def test_max_bf_caps_the_hook() -> None:
    length = _length(_starts([90] * 6), [(26.5, 0.5)], max_bf=25)
    assert length is not None and length.bf_cap == 25


def _cfg(**kw: float) -> TeamSimConfig:
    rates = {"K": 0.22, "BB": 0.08, "HBP": 0.01, "1B": 0.14, "2B": 0.045, "3B": 0.004,
             "HR": 0.03, "OUT": 0.471}
    return TeamSimConfig(bat_vs_starter=[rates] * 9, bat_vs_pen=[rates] * 9, **kw)  # type: ignore[arg-type]


def test_zero_bf_sd_keeps_the_simulation_identical() -> None:
    a = MonteCarlo(300, seed=3).simulate(_cfg(), _cfg())
    b = MonteCarlo(300, seed=3).simulate(_cfg(starter_bf_sd=0.0), _cfg(starter_bf_sd=0.0))
    assert np.array_equal(a.pit["home"]["outs"], b.pit["home"]["outs"])
    assert np.array_equal(a.home_runs_full, b.home_runs_full)


def test_bf_sd_spreads_the_starters_outs() -> None:
    fixed = MonteCarlo(2000, seed=3).simulate(_cfg(starter_pitch_cap=200), _cfg(starter_pitch_cap=200))
    spread = MonteCarlo(2000, seed=3).simulate(
        _cfg(starter_pitch_cap=200, starter_bf_sd=4.0), _cfg(starter_pitch_cap=200, starter_bf_sd=4.0)
    )
    assert spread.pit["home"]["outs"].std() > fixed.pit["home"]["outs"].std()
