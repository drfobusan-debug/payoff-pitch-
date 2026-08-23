"""A free pass is not a walk, and the walks prop settles on walks.

The BB outcome bucket is a free pass -- walk, intentional walk and hit-by-pitch --
because all three put the batter on first, which is what the run models need. The
pitcher-walks prop does not settle that way: ``data.results`` reads the box score's
``baseOnBalls`` and nothing else. So the prop was priced on 0.1007 of plate
appearances and graded against 0.0893 of them, roughly a quarter of a walk per
start.

The correction is a change of units rather than a fitted factor, and two
independent measurements off 2,890 starts agree it is the right one: starters free
pass 0.0929 per batter faced, 0.0929 * 0.8843 = 0.08215, and counting walks alone
off the same starts gives 0.0821.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import mlb_engine.models.montecarlo as mc
from mlb_engine.config import Config, EVThresholds
from mlb_engine.features.rolling import (
    LEAGUE_WALK_SHARE_OF_FREE_PASS,
    WALK_EVENTS,
)
from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.pipeline import Pipeline


def _mean_walks(n_sims: int = 6000, p_bb: float = 0.30) -> float:
    """Mean walks charged to the starter, at an exaggerated rate for precision."""
    rates = {
        "1B": 0.0,
        "2B": 0.0,
        "3B": 0.0,
        "HR": 0.0,
        "BB": p_bb,
        "K": 0.0,
        "OUT": 1.0 - p_bb,
    }
    cfg = TeamSimConfig(
        bat_vs_starter=[dict(rates) for _ in range(9)],
        bat_vs_pen=[dict(rates) for _ in range(9)],
        starter_pitch_cap=10_000,  # let the batters-faced cap be the only hook
    )
    res = MonteCarlo(n_sims, seed=11).simulate(cfg, cfg)
    return float(res.pit["home"]["BB"].mean())


def test_the_starter_is_charged_only_the_walks_among_his_free_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The charged count is the free-pass count times the league walk share.

    Measured against the same simulation with the thinning switched off, so the
    assertion is about the ratio rather than a hand-derived batters-faced count.
    """
    thinned = _mean_walks()
    monkeypatch.setattr(mc, "WALK_SHARE_OF_FREE_PASS", 1.0)
    free_passes = _mean_walks()
    assert thinned < free_passes
    assert abs(thinned / free_passes - LEAGUE_WALK_SHARE_OF_FREE_PASS) < 0.02


def test_hit_by_pitch_still_puts_the_man_on_first() -> None:
    """The thinning is a scoring decision and must not touch the run environment.

    This is the trap the change is written around: removing the hit-by-pitch from
    the walk *rate* would have taken the runner off the bases with it, and the run
    models are correct precisely because a free pass advances runners.
    """
    assert "hit_by_pitch" in WALK_EVENTS
    lg = {
        "1B": 0.1417,
        "2B": 0.0416,
        "3B": 0.0036,
        "HR": 0.0311,
        "BB": 0.1020,
        "K": 0.2228,
        "OUT": 0.4572,
    }
    cfg = TeamSimConfig(
        bat_vs_starter=[dict(lg) for _ in range(9)],
        bat_vs_pen=[dict(lg) for _ in range(9)],
    )
    res = MonteCarlo(4000, seed=5).simulate(cfg, cfg)
    runs = np.concatenate([res.home_runs_full, res.away_runs_full])
    # The league's own vector must still score like the league, ~4.6 R/team/game.
    assert 4.2 < runs.mean() < 5.1


def test_the_walk_share_matches_the_measured_event_split() -> None:
    """Pinned to the starters it prices, and cross-checked against the league.

    The constant is the share measured over starts, 0.8843, because a starter is
    what the prop is about. The league-wide event split over all 116,384 plate
    appearances is 0.8867 -- relievers hit marginally more batters -- and the two
    agreeing to 0.0024 is the check that neither is an artefact of one population.
    """
    walk, ibb, hbp = 0.086558, 0.002698, 0.011402
    league = (walk + ibb) / (walk + ibb + hbp)
    assert abs(league - 0.8867) < 5e-4
    assert abs(LEAGUE_WALK_SHARE_OF_FREE_PASS - 0.8843) < 5e-5
    assert abs(LEAGUE_WALK_SHARE_OF_FREE_PASS - league) < 3e-3


# A 0.55 side at +100 / -110 does not clear the shipped conviction floor once the
# probability is anchored to the market, and the walks veto is what is under test.
LEVELS_OFF = EVThresholds(min_prob=0.0, max_ev=1.0)


def _bb_recs(cfg: Config, p_over: float) -> dict[tuple[str, str], object]:
    """Price every pitcher prop at ``p_over``, both sides quoted +100 / -110.

    That price devigs to a fair 0.4884, so whichever side is given 0.55 carries a
    +0.062 edge -- inside the 0.02-0.08 buy band, and therefore a buy unless a
    gate stops it. Without that the row passes on the ordinary EV screens and the
    test proves nothing about the veto.
    """
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    p._calibrator = SimpleNamespace(apply=lambda market, prob: prob)
    p._shrink = None
    p._splits = {}
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    pitcher = SimpleNamespace(name="Some Pitcher", mlbam_id=42)
    n = 1000
    arr = np.zeros(n)
    arr[: int(n * p_over)] = 8.0  # 8 clears every line in the table, 0 clears none
    res = SimpleNamespace(
        pit={"home": {k: arr.copy() for k in ("K", "outs", "H", "BB", "ER")}}
    )
    quotes = {
        (
            "MATCH",
            f"pitcher_{stat.lower()}",
            keys.pitcher_prop(pitcher.name, lab, ln, side),
        ): [MarketQuote(book="dk", american=100.0, opposite_american=-110.0)]
        for stat, lab, lines in (
            ("K", "Ks", (4.5, 5.5, 6.5)),
            ("outs", "Outs", (15.5, 17.5)),
            ("H", "Hits", (4.5, 5.5)),
            ("BB", "Walks", (1.5, 2.5)),
            ("ER", "ER", (2.5, 3.5)),
        )
        for ln in lines
        for side in ("over", "under")
    }
    recs = p._pitcher_props(game, "MATCH", res, "home", pitcher, quotes)
    # One line per market: the lowest of each, which every stat's array clears.
    return {(r.market, r.side): r for r in recs if r.line in (1.5, 4.5, 15.5)}


def test_the_walks_under_is_vetoed_even_when_it_is_the_value_side() -> None:
    """pitcher_bb is the one market whose over was the profitable side.

    At 0.45 the under is the side the model likes, and it is exactly the side the
    graded rows say not to take -- so the veto has to bite on a row that would
    otherwise be bought, not merely on one the EV screens already declined.
    """
    recs = _bb_recs(Config(ev=LEVELS_OFF), p_over=0.45)
    under = recs[("pitcher_bb", "under")]
    assert under.tier is Tier.PASS
    assert under.pass_gate == "bb_under"
    # Its own gate name, so the audit can grade what the veto declined.
    assert under.pass_gate != "contact_floor"
    # The veto is this market's alone. No other pitcher under carries it, and
    # those without a probability adjustment of their own are still bought off
    # the identical price and probability -- pitcher_outs is excluded because its
    # own outs-bias term moves the edge out of the band on its own.
    for market in ("pitcher_h", "pitcher_outs", "pitcher_k"):
        assert recs[(market, "under")].pass_gate != "bb_under", market
    for market in ("pitcher_h", "pitcher_k"):
        assert recs[(market, "under")].tier is not Tier.PASS, market


def test_the_walks_over_is_left_alone() -> None:
    """The veto is directional; the over is the side that made money."""
    over = _bb_recs(Config(ev=LEVELS_OFF), p_over=0.55)[("pitcher_bb", "over")]
    assert over.tier is not Tier.PASS
    assert over.pass_gate is None


def test_the_walks_under_can_be_re_enabled() -> None:
    """The veto is a stance on an unvalidated level, so it has to be reversible."""
    cfg = replace(Config(ev=LEVELS_OFF), pitcher_bb_under_gate=False)
    under = _bb_recs(cfg, p_over=0.45)[("pitcher_bb", "under")]
    assert under.tier is not Tier.PASS
    assert under.pass_gate is None
