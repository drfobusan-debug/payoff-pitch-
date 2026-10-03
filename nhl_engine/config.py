"""Central configuration for the NHL engine.

Credentials come from environment variables (nothing sensitive is committed).
Everything else has a default overridable via an ``NHLE_``-prefixed variable.
The Odds API key is shared with the other engines so one ``engine.env`` serves
all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def data_dir() -> Path:
    """Where the archive, cache and (later) the ledger live."""
    raw = os.getenv("NHLE_DATA_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".nhl_engine"


def cache_dir() -> Path:
    return data_dir() / "cache"


def output_dir() -> Path:
    raw = os.getenv("NHLE_OUTPUT_DIR")
    return Path(raw).expanduser() if raw else data_dir() / "output"


@dataclass(frozen=True)
class Credentials:
    """Credentials for data sources (never logged)."""

    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )

    # Same Gmail App Password the MLB/NFL/CFB senders use: one engine.env serves all.
    gmail_user: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    )
    gmail_app_password: str | None = field(default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD"))

    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)

    def has_email(self) -> bool:
        return bool(self.gmail_app_password)


@dataclass(frozen=True)
class Delivery:
    """Where the card goes; shares the MLB SMTP variables with ``NHLE_`` overrides."""

    email_to: str | None = field(
        default_factory=lambda: os.getenv("NHLE_EMAIL_TO") or os.getenv("MLBE_EMAIL_TO")
    )
    smtp_host: str = field(
        default_factory=lambda: (
            os.getenv("NHLE_SMTP_HOST") or os.getenv("SMTP_HOST") or "smtp.gmail.com"
        )
    )
    smtp_port: int = field(
        default_factory=lambda: _env_int("NHLE_SMTP_PORT", _env_int("SMTP_PORT", 465))
    )


@dataclass(frozen=True)
class CaptureParams:
    """How much of the board one capture pass is allowed to spend.

    A featured-board pull is one bulk call (3 credits). Every period, team-total
    and prop market is a *per-event* call costing about one credit per market
    that returns data -- measured 25 credits for one event with all 31 keys on
    2026-09-28 -- so a full pass over a 15-game slate is ~400 credits. The cap
    keeps a misconfigured schedule from draining the key in a day.
    """

    max_events: int = field(default_factory=lambda: _env_int("NHLE_CAPTURE_MAX_EVENTS", 20))
    # Only archive events starting within this many hours; keeps the per-event
    # spend on today's games rather than the whole posted week.
    horizon_hours: int = field(default_factory=lambda: _env_int("NHLE_CAPTURE_HORIZON_H", 30))


def studies_dir() -> Path:
    return data_dir() / "studies"


def priors_dir() -> Path:
    return data_dir() / "priors"


# Fitted by scripts/nhl/reliability_study.py on 2026-09-29, MoneyPuck regular
# seasons 2015-2024 (312 team-seasons, 685 goalie-seasons >= 600 min).
# k is in the metric's exposure units (seconds of that strength state; shots
# for Sv%), i.e. the exposure at which the observation and the prior weigh
# equally. Shown alongside: split-half r at half a season, and k in games.
#
#   metric         r_split  k_games   k_sec      r_yy
#   xgf60_5v5        0.79     10.2     30155     0.62
#   xga60_5v5        0.73     14.7     43207     0.51
#   cf60_5v5         0.88      5.6     16398     0.67
#   ca60_5v5         0.85      6.8     20017     0.59
#   hdxgf60_5v5      0.63     22.7     66886     0.62
#   hdxga60_5v5      0.60     26.4     77834     0.60
#   gf60_5v5         0.38     63.9    188519     0.44
#   ga60_5v5         0.49     41.3    121853     0.46
#   fin60_5v5        0.18    183.2    540234     0.30   (finishing: ~2 seasons to half-trust)
#   tsv60_5v5        0.20    160.4    473218     0.21
#   pp_xgf60         0.50     39.3     10971     0.57
#   pp_xga60         0.24    126.9     35433     0.33   (shorthanded-against: r<0.25)
#   pk_xga60         0.35     73.1     20423     0.36
#   sh_xgf60         0.33     79.8     22273     0.25   (SH offence: r<0.35)
#   pp_share         0.34     76.4    278177     0.31
#   pk_share         0.53     34.2    124385     0.45
#   pen_drawn60      0.57     29.3    106749     0.54
#   pen_taken60      0.65     21.1     76988     0.63
#   goalie gsax60    0.15    104.2    349639     0.14   (n_half ~1028 min)
#   goalie svpct     0.23     62.0      1759     0.23   (k in shots; display only)
#
# Ramp check (first-N-games rate vs rest, measured/predicted): xgf60 0.59/0.65
# at N=10, 0.69/0.75 at 20, 0.72/0.80 at 41 -- the n/(n+k) model over-predicts
# repeatability by ~0.06 for the xG rates, i.e. real ramps are slightly slower
# than k says; the shrink is, if anything, too light, never too heavy.
# Goalie GSAx/60 barely repeats within a season (r=0.15 at ~1000 minutes) and
# year-to-year (0.14): a goalie's number is mostly prior until he has ~100 games.
# Sv% repeats a little better (0.23) but that includes the team defence in front
# of him, which is why it stays display-only.
RELIABILITY_STUDY = "scripts/nhl/reliability_study.py 2026-09-29, seasons 2015-2024"

_K_SEC: dict[str, float] = {
    "xgf60_5v5": 30155.0,
    "xga60_5v5": 43207.0,
    "cf60_5v5": 16398.0,
    "ca60_5v5": 20017.0,
    "hdxgf60_5v5": 66886.0,
    "hdxga60_5v5": 77834.0,
    "gf60_5v5": 188519.0,
    "ga60_5v5": 121853.0,
    "fin60_5v5": 540234.0,
    "tsv60_5v5": 473218.0,
    "pp_xgf60": 10971.0,
    "pp_xga60": 35433.0,
    "pk_xga60": 20423.0,
    "sh_xgf60": 22273.0,
    "pp_share": 278177.0,
    "pk_share": 124385.0,
    "pen_drawn60": 106749.0,
    "pen_taken60": 76988.0,
}
_R_YY: dict[str, float] = {
    "xgf60_5v5": 0.62,
    "xga60_5v5": 0.51,
    "cf60_5v5": 0.67,
    "ca60_5v5": 0.59,
    "hdxgf60_5v5": 0.62,
    "hdxga60_5v5": 0.60,
    "gf60_5v5": 0.44,
    "ga60_5v5": 0.46,
    "fin60_5v5": 0.30,
    "tsv60_5v5": 0.21,
    "pp_xgf60": 0.57,
    "pp_xga60": 0.33,
    "pk_xga60": 0.36,
    "sh_xgf60": 0.25,
    "pp_share": 0.31,
    "pk_share": 0.45,
    "pen_drawn60": 0.54,
    "pen_taken60": 0.63,
}


# scripts/nhl/league_level_study.py 2026-10-03, seasons 2016-2025: the league
# rate every team is divided by, as ``w x season-to-date + (1 - w) x last
# season``, ``w = n / (n + k)`` with ``n`` team-games played, against the rest of
# the season (SSE over checkpoints 16..512 team-games, 9 seasons). xG rates
# settle in ~100 team-games, goals/finishing/PP-PK in 400-800; penalty rates and
# PP/PK time shares never beat last season's league rate within a season (k=inf).
# Read raw, two days of October finishing (+0.35 goals/60 on 10-02) took ~0.6
# goals off every game the sim priced.
LEAGUE_STUDY = "scripts/nhl/league_level_study.py 2026-10-03, seasons 2016-2025"

_LEAGUE_K_GAMES: dict[str, float] = {
    "xgf60_5v5": 100.0,
    "xga60_5v5": 100.0,
    "cf60_5v5": 25.0,
    "ca60_5v5": 50.0,
    "hdxgf60_5v5": 100.0,
    "hdxga60_5v5": 100.0,
    "gf60_5v5": 400.0,
    "ga60_5v5": 800.0,
    "fin60_5v5": 400.0,
    "tsv60_5v5": 800.0,
    "pp_xgf60": 400.0,
    "pp_xga60": 1600.0,
    "pk_xga60": 400.0,
    "sh_xgf60": 800.0,
    "pp_share": float("inf"),
    "pk_share": float("inf"),
    "pen_drawn60": float("inf"),
    "pen_taken60": float("inf"),
}


@dataclass(frozen=True)
class ShrinkParams:
    """EB ``k`` per team metric (exposure seconds) and the goalie GSAx ``k``."""

    team_k: dict[str, float] = field(default_factory=lambda: dict(_K_SEC))
    # League rate: k in team-games toward last season's league (LEAGUE_STUDY).
    league_k_games: dict[str, float] = field(default_factory=lambda: dict(_LEAGUE_K_GAMES))
    goalie_k_season: float = 349639.0
    # Same k for the discounted career sum: exposures add, the prior does not change.
    goalie_k_career: float = 349639.0
    goalie_r_yy: float = 0.14
    study: str = RELIABILITY_STUDY


@dataclass(frozen=True)
class PriorParams:
    """Preseason prior: regression weights and the refresh/freeze schedule."""

    r_yy: dict[str, float] = field(default_factory=lambda: dict(_R_YY))
    refresh_days: int = 7
    # Weekly rebuilds through October, then frozen for the season (§5.1).
    freeze_month_day: tuple[int, int] = (10, 31)
    study: str = RELIABILITY_STUDY


# scripts/nhl/goalie_study.py 2026-09-29, seasons 2015-2024, 26,086 appearances,
# 232 goalies. GSAx/60 pooled by career NHL minutes *before* the game:
#     0-180 min   -0.229 ± 0.082  n=505     <- call-up anchor
#   180-600 min   -0.086 ± 0.059  n=746
#   600-2000      -0.117 ± 0.037  n=2192
#   2000-5000     -0.037 ± 0.026  n=4230
#   5000-15000    +0.015 ± 0.018  n=8834
#   15000+        +0.023 ± 0.017  n=9579
# Days since previous appearance, within goalie-season (rust/fatigue):
#   1d +0.076±0.053  2d -0.026±0.018  3-4d -0.042±0.018  5-7d -0.006±0.026
#   8-14d +0.107±0.036  15-30d +0.075±0.066  31+d +0.218±0.052
# Long-rest games grade *better*, not worse -- consistent with "benched after a
# bad stretch, then regressed" selection, not with rust. No TSLS penalty ships.
# Age (delta method, later-season age): 24-26 -0.002±0.043 n=107; 27-29
# -0.024±0.039 n=141; 30-32 -0.041±0.037 n=119; 33-35 -0.037±0.049 n=76;
# 36+ -0.078±0.075 n=41. Direction consistent with decline from ~27 at roughly
# -0.03 GSAx/60 per season, but no bucket clears 2 SE; ships at 0, registered
# as a candidate for the ledger.
# Team change (n=126 movers >= 600 min both seasons): residual vs the gap in
# prior-season 5v5 xGA/60 (new - old team) has slope -0.23, corr -0.15 -- the
# expected sign (xG does not remove all of the defence) at ~1.7 SE. Ships at 0.
GOALIE_STUDY = "scripts/nhl/goalie_study.py 2026-09-29, seasons 2015-2024"


@dataclass(frozen=True)
class GoalieParams:
    """Goalie status floors and the knobs the goalie study measured (see above)."""

    # Below this many NHL minutes a goalie is a call-up: his anchor is the
    # call-up distribution, not the league rate. The bucket edge the study used.
    min_minutes: float = field(
        default_factory=lambda: float(_env_int("NHLE_GOALIE_MIN_MINUTES", 180))
    )
    # Pooled GSAx/60 of appearances under the floor (study, bucket 0-180 min).
    callup_prior_gsax60: float | None = -0.229
    # Measured, not significant, ship at 0 (see GOALIE_STUDY): per-season GSAx/60
    # change after age 27, and GSAx/60 per unit of defensive gap on a team change.
    age_decline_per_season: float = 0.0
    age_decline_from: int = 27
    team_change_slope: float = 0.0
    # Within-goalie GSAx/60 by rest bucket: measured positive at long rest
    # (selection), so no rust or fatigue term ships.
    tsls_penalty_gsax60: float = 0.0
    study: str = GOALIE_STUDY


# scripts/nhl/sim_params_study.py 2026-09-29, seasons 2022-2024, 3,936 games.
# All-situation xG/sec by the shooting team's score state, relative to tied
# (SE in parentheses; exact exposure from the goal timeline, nets in):
#   trail 2+ 1.027 (.018)  trail 1 1.036 (.016)  lead 1 0.965 (.015)  lead 2+ 0.924 (.016)
# Tied-state xG/sec by period vs the mean: P1 0.983  P2 1.081  P3 0.939.
# Home share of tied xG 0.530; league goals per xG 0.983.
# Penalties taken 3.46 per team-60; 5v4 seconds per penalty 78.8 (majors,
# coincidentals and 5v3 fold into an *effective* clean-minor rate below).
# Goalie pulls (shift charts, 3,879 games, deficit at the pull): time left at the
# pull, quantiles 10/25/50/75/90 (s), pooled over the three seasons:
#   down 1: 71 94 117 142 175   down 2: 100 141 176 220 275   down 3: 160 206 293 386 666
# Pulls per team-game trailing by d with 5:00 left: down 1 or 2 ~1.0 (every
# team pulls), down 3 0.29 (287 of 1,000). While pulled: 6.8 goals/60 for the attacking side
# (xG says 11.5 -- 6v5 xG overstates, so goals are used) and 19.3/60 into the
# empty net.
# OT (840 tied-after-60 games): decided in OT 69.3%; P(home wins OT) =
# sigmoid(0.181 (.084) + 3.37 (1.60) x [home 5v5 xG share - away 5v5 xG share])
# -- both terms ~2.1 SE, shipped at the measured value. Shootout home share
# 0.512 ± 0.031 (n=258): not distinguishable from a coin, ships at 0.500.
SIM_STUDY = "scripts/nhl/sim_params_study.py 2026-09-29, seasons 2022-2024"


@dataclass(frozen=True)
class SimParams:
    """Game-simulation parameters (master plan §5.3); every number from SIM_STUDY."""

    draws: int = field(default_factory=lambda: _env_int("NHLE_SIM_DRAWS", 20000))
    step_seconds: int = 20
    score_mult: dict[str, float] = field(
        default_factory=lambda: {
            "trail2": 1.027,
            "trail1": 1.036,
            "tied": 1.0,
            "lead1": 0.965,
            "lead2": 0.924,
        }
    )
    period_mult: tuple[float, float, float] = (0.983, 1.081, 0.939)
    home_xg_share: float = 0.530
    goals_per_xg: float = 0.983
    # effective clean minors per team-60: 3.46 penalties x 78.8 / 120 s
    minors_per_60: float = 2.27
    minor_seconds: int = 120
    # P(pull | trailing by d with 5:00 left) and time-left-at-pull quantiles (s)
    pull_prob: dict[int, float] = field(default_factory=lambda: {1: 1.0, 2: 1.0, 3: 0.29})
    pull_quantiles: dict[int, tuple[float, ...]] = field(
        default_factory=lambda: {
            1: (71.0, 94.0, 117.0, 142.0, 175.0),
            2: (100.0, 141.0, 176.0, 220.0, 275.0),
            3: (160.0, 206.0, 293.0, 386.0, 666.0),
        }
    )
    en_goals60_for: float = 6.8  # 6v5 attacking side, per pulled 60
    en_goals60_against: float = 19.3  # into the empty net, per pulled 60
    p_ot_goal: float = 0.693
    # Multiplier on the xG-derived goal rates (5v5, PP, SH). At league-average
    # teams and goalies the sim scored 1.031/1.038/1.022/1.047 x too few reg+OT
    # goals in 2022-2025 (LEAGUE_STUDY); pooled 1.035.
    goal_level: float = 1.035
    ot_home_intercept: float = 0.181
    ot_strength_slope: float = 3.37
    so_home: float = 0.5
    study: str = SIM_STUDY


# Selection gates for the first card (master plan §5.9). Edge is model minus the
# devigged consensus; the ceiling treats a large disagreement as a model error.
# Every gate stamps ``pass_gate`` -- nothing is filtered -- and is graded in
# probation before it may open.
@dataclass(frozen=True)
class GateParams:
    min_edge: float = 0.02
    strong_edge_gap: float = 0.02
    max_edge: float = 0.06
    min_ev: float = 0.0
    # Longest price we will buy, per market family (American odds).
    max_buy_odds: dict[str, float] = field(
        default_factory=lambda: {"ml": 160.0, "pl": 200.0, "total": 125.0, "period": 150.0}
    )
    # A quote older than this is stale; the card prices only a live board.
    max_quote_age_minutes: int = 90
    # Period markets and team totals are archived and priced but not yet buys
    # (probation, master plan §6 Phase 2); ML/PL/totals are the live markets.
    live_markets: tuple[str, ...] = ("game_ml", "game_pl", "game_total")
    # Game markets price off ``p_bet = (1 - w) x p_sim + w x consensus``; edge,
    # EV, tier and every gate read p_bet, the ledger keeps p_sim as model_prob.
    # Not fitted on NHL (21 graded games); the MLB engine's fitted cap (0.90)
    # until the NHL ledger can refit it. At 0.90 a buy needs the sim 20 points
    # off the market to clear min_edge.
    market_anchor: float = field(default_factory=lambda: _env_float("NHLE_MARKET_ANCHOR", 0.90))
    anchored_markets: tuple[str, ...] = (
        "game_ml",
        "game_ml3",
        "game_pl",
        "game_pl_alt",
        "game_total",
        "game_total_alt",
    )


@dataclass(frozen=True)
class Config:
    creds: Credentials = field(default_factory=Credentials)
    delivery: Delivery = field(default_factory=Delivery)
    capture: CaptureParams = field(default_factory=CaptureParams)
    shrink: ShrinkParams = field(default_factory=ShrinkParams)
    prior: PriorParams = field(default_factory=PriorParams)
    goalie: GoalieParams = field(default_factory=GoalieParams)
    sim: SimParams = field(default_factory=SimParams)
    gates: GateParams = field(default_factory=GateParams)
    # Carry the archive on the shared engine-state branch so a second machine
    # (or a rebuilt one) has every price this one ever saw.
    state_sync: bool = field(default_factory=lambda: _env_bool("NHLE_STATE_SYNC", True))


def load_config() -> Config:
    return Config()
