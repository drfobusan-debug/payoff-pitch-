"""SIERA has to land on the published scale, not just rank pitchers correctly.

The existing SIERA test compares an extreme ace against an extreme scrub, which
passes whatever constant the formula is offset by. That is exactly how a
four-tenths-of-a-run scale error survived: the *ordering* was always right. The
tests here pin the absolute level instead, because the ace/scrub cut points are
quoted in published SIERA units and the value is written into the ledger.
"""

from __future__ import annotations

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.features.siera import (
    SIERA_LEAGUE_ANCHOR,
    faces_ace,
    faces_scrub,
    pitcher_siera,
)

# League-average rates measured on the Statcast cache: K/PA .220, BB/PA .086,
# and a batted-ball mix of 42.4% GB / 26.6% FB / 7.3% PU / 23.7% LD over the
# 67.8% of plate appearances that put a ball in play.
LEAGUE_PA = 1000
LEAGUE_K = 220
LEAGUE_BB = 86
LEAGUE_GB = 287
LEAGUE_FB = 180
LEAGUE_PU = 49


def _mk(n_pa: int, k: int, bb: int, gb: int, fb: int, pu: int) -> pd.DataFrame:
    """A pitcher's Statcast slice with the given PA outcome counts."""
    events = ["strikeout"] * k + ["walk"] * bb + ["field_out"] * (n_pa - k - bb)
    bip = n_pa - k - bb
    bb_type = (
        [None] * (k + bb)
        + ["ground_ball"] * gb
        + ["fly_ball"] * fb
        + ["popup"] * pu
        + [None] * (bip - gb - fb - pu)
    )
    return pd.DataFrame({"events": events, "bb_type": bb_type})


def _league_average() -> pd.DataFrame:
    return _mk(LEAGUE_PA, LEAGUE_K, LEAGUE_BB, LEAGUE_GB, LEAGUE_FB, LEAGUE_PU)


def test_the_league_average_arm_reads_league_average_siera() -> None:
    """The regression test for the scale bug.

    Fed league-average rates the bare Swartz polynomial returns 3.64, because
    its coefficients were fitted to a different run environment and FanGraphs
    re-centres the output every season. Without that step every starter read
    ~0.4 runs better than his published number.
    """
    league = pitcher_siera(_league_average())
    # The tolerance is 0.15 rather than something tighter because SIERA is a
    # polynomial: evaluating it at the league's *mean* rates (3.74 raw) is not
    # the same as the mean of the league's individual SIERAs (3.64 raw, which is
    # what the offset is pinned to). The gap is Jensen's inequality, not slop.
    assert abs(league.siera - SIERA_LEAGUE_ANCHOR) < 0.15
    assert league.siera > 3.9  # 3.74 before re-centring, so this is the guard


def test_the_ace_and_scrub_cuts_select_sane_shares_of_the_league() -> None:
    """The cut points are only meaningful if they sit either side of average.

    On the raw scale 3.4 sat *below* the league mean of 3.64, so a merely
    average starter tripped the ace gate and suppressed the batter's over.
    """
    cfg = Config()
    league = pitcher_siera(_league_average())
    assert cfg.singles_siera_ace < league.siera < cfg.singles_siera_bad
    assert not faces_ace(league, cfg.singles_siera_ace)
    assert not faces_scrub(league, cfg.singles_siera_bad)


def test_re_centring_is_a_shift_and_preserves_ordering() -> None:
    ace = pitcher_siera(_mk(200, 76, 10, 60, 30, 5))
    scrub = pitcher_siera(_mk(200, 24, 22, 30, 55, 12))
    league = pitcher_siera(_league_average())
    assert ace.siera < league.siera < scrub.siera
    # The rates behind the number are untouched by the shift.
    assert abs(league.so_rate - 0.220) < 0.001
    assert abs(league.bb_rate - 0.086) < 0.001


def test_a_thin_sample_still_refuses_to_gate() -> None:
    thin = pitcher_siera(_mk(40, 9, 3, 12, 8, 2))
    assert not thin.has_data
    assert not faces_ace(thin, Config().singles_siera_ace)
    assert not faces_scrub(thin, Config().singles_siera_bad)
