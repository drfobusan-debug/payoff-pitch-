"""Central configuration for the MLB prediction engine.

Values are sourced from environment variables where relevant (credentials in
particular) so that nothing sensitive is committed. Everything else has sensible
defaults that can be overridden via environment variables prefixed ``MLBE_``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from mlb_engine.features.regression import (
    SINGLES_BARREL_SLOPE,
    SINGLES_GB_SLOPE,
    SINGLES_LD_SLOPE,
)
from mlb_engine.features.rolling import HR_PRIOR_WEIGHT


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _data_dir() -> Path:
    return Path(os.getenv("MLBE_DATA_DIR", str(Path.home() / ".mlb_engine")))


def default_ros_prior_path() -> str:
    """Where ``mlb-engine ros-prior`` writes the hitter projection file."""
    return str(_data_dir() / "ros_hitters.csv")


def _env_csv(name: str) -> tuple[str, ...] | None:
    """Comma-separated override, or ``None`` to keep the caller's default."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


@dataclass(frozen=True)
class RollingWindows:
    """Rolling look-back windows (in days) for stat aggregation."""

    # Six weeks, not four. Over 2,894 starts, the six-week read is the stronger
    # predictor of the next start on 66 of 100 metric/target pairs and wins every
    # held-out target (next-start xwOBA R^2 0.087 vs 0.075, IP 0.067 vs 0.055).
    # Replaying 54 slates moves favoured PPV .5831 -> .5867, date-clustered 95%
    # CI [+0.04, +0.64] pp, and +1.54 pp on pitcher strikeouts.
    pitcher_form_days: int = field(default_factory=lambda: _env_int("MLBE_PITCHER_FORM_DAYS", 42))
    batter_home_away_days: int = field(
        default_factory=lambda: _env_int("MLBE_BATTER_HOME_AWAY_DAYS", 21)
    )
    batter_vs_rhp_days: int = field(default_factory=lambda: _env_int("MLBE_BATTER_VS_RHP_DAYS", 21))
    batter_vs_lhp_days: int = field(default_factory=lambda: _env_int("MLBE_BATTER_VS_LHP_DAYS", 42))
    biomech_days: int = field(default_factory=lambda: _env_int("MLBE_BIOMECH_DAYS", 28))
    # Team-level platoon and venue splits for the preview, which need a far
    # longer look-back than an individual hitter does -- not because a club
    # changes more slowly, but because ``team_splits.MIN_SPLIT_PA`` is 500 and a
    # club sees roughly 390 PA against left-handers in six weeks. Over 42 days
    # exactly 1 of 30 clubs clears the floor vs LHP; at 60 days, 21; at 90, all
    # 30. Shorter than this and the platoon line simply stops printing.
    team_split_days: int = field(default_factory=lambda: _env_int("MLBE_TEAM_SPLIT_DAYS", 90))
    # Bullpen: relievers' last ~3 weeks and batters' late-inning last ~3 weeks.
    bullpen_days: int = field(default_factory=lambda: _env_int("MLBE_BULLPEN_DAYS", 21))
    bullpen_min_inning: int = field(default_factory=lambda: _env_int("MLBE_BULLPEN_MIN_INNING", 6))
    # A separate, longer window for the bullpen's stuff and command signals.
    # Split-half reliability of a 3-week relief read (30 pens, ~270 batters faced
    # each): K% 0.66, whiff 0.58, velocity 0.67, but xwOBA 0.37, BB% 0.19,
    # hard-hit 0.13, HR/BF 0.06. Out of sample against the next three weeks, K%
    # scores 0.73 on 42 days vs 0.66 on 21, and in a joint regression the 42-day
    # read takes +0.68 against +0.14 for the last three weeks. 0 keeps the single
    # 21-day window for everything.
    bullpen_skill_days: int = field(
        default_factory=lambda: _env_int("MLBE_BULLPEN_SKILL_DAYS", 0)
    )
    # Share of the empirical-Bayes correction to apply to a starter's
    # contact-quality rates (xwOBA/wOBA allowed, BABIP, hard-hit, barrel) before
    # they drive the hit and HR multipliers. Split-half across adjacent six-week
    # blocks: xwOBA r=0.31, hard-hit r=0.24, BABIP r=0.10, barrel r=0.09, against
    # K% r=0.52 and CSW r=0.50 for the command signals, which are left raw.
    # 0.0 is the legacy raw behaviour; 1.0 applies the measured weight in full.
    starter_contact_shrink: float = field(
        default_factory=lambda: _env_float("MLBE_STARTER_CONTACT_SHRINK", 0.0)
    )
    # Share of a bullpen's distance from the league mean xwOBA to keep, set to
    # the measured split-half reliability of a three-week read. The run-line
    # underdog gate keeps reading the unshrunk mean (``BullpenProfile.xwoba_raw``)
    # so its .330 threshold still means what it was calibrated against; this
    # value is what the reader and any level term see.
    bullpen_xwoba_shrink: float = field(
        default_factory=lambda: _env_float("MLBE_BULLPEN_XWOBA_SHRINK", 0.37)
    )
    # Share of the fitted four-seam velocity term to charge on a starter's
    # strikeout rate: his level against the league, plus how his most recent
    # start sat against his own window. Velocity is the only read that survives
    # a single start (r=.93 between consecutive starts, against .20 for K/PA and
    # .15 for CSW%), and adding both terms improved held-out strikeout deviance
    # from 1.05839 to 1.05661 over 2,082 starts.
    #
    # Still 0.0, on evidence rather than on caution. As a *rate* forecast it
    # clears the bar that retired the stuff multiplier: weekly walk-forward
    # wRMSE 0.09634 against 0.09749 for the blended rate alone, and the dose
    # search keeps ~0.8 of it where it kept none of stuff
    # (scripts/vfa_k_price_study.py). Priced through the simulator on nine
    # graded slates it does not: strikeout Brier .20197 -> .20166 but log loss
    # .60567 -> .62127, and 16 of 18 other markets get worse, because scaling a
    # starter's K rate rescales every other outcome he allows
    # (scripts/vfa_k_backtest.py). 1.0 charges the fitted slopes in full.
    vfa_k_weight: float = field(
        default_factory=lambda: _env_float("MLBE_VFA_K_WEIGHT", 0.0)
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def _env_set(name: str, default: tuple[str, ...]) -> frozenset[str]:
    """A comma-separated override, where an empty value means the empty set."""
    raw = os.getenv(name)
    if raw is None:
        return frozenset(default)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


# Cumulative-audit false-positive pockets. The model is over-confident on these
# counting-prop overs (measured PPV well below the ~52.4% breakeven), so a global
# thin-edge guard still lets marginal, unprofitable buys through. Each must show
# a bigger edge over the devigged price before it can fire; the env override
# ``MLBE_MIN_EDGE_<MARKET>`` still wins.
_OVERBET_EDGE_FLOORS: dict[str, float] = {
    "batter_tb": 0.05,
    "batter_r": 0.05,
    "batter_hrr": 0.04,
    "batter_1b": 0.04,
}

# Markets whose buys the accumulated ledger cannot justify at any edge. Over 27
# graded slates (1,894 buys) every batter market except doubles lost money on
# the side the engine backed: hits 48.3% for -14.0% ROI (n=447), runs 29.7%
# (n=128), total bases 11.1% (n=9), H+R+RBI 43.8% (n=16), against doubles'
# +36.3% (n=73). Raising an edge floor is the wrong instrument when the market
# itself is under water at every edge, so these overs are hard-passed and keep
# grading as shadow bets -- the ledger still records the price, the tier reason
# and the result, so a rebuilt batter model can be graded before it is trusted
# with money. ``MLBE_NO_BUY_<MARKET>=0`` re-enables one, and
# ``MLBE_NO_BUY_<MARKET>=1`` disqualifies any other.
#
# Doubles were the one batter market carrying that comparison and no longer
# are: on the current basis their buys are 14.3% for -15.5% ROI (n=70), which
# is what the passed rows do anyway. They are screened by price rather than
# listed here -- see ``doubles_max_buy_odds``.
#
# Home runs, singles and RBI lost money too and are deliberately *not* here:
# each already has a price band or probability floor fitted to its own graded
# rows (``hr_min_buy_odds``, ``singles_min_buy_odds``, ``rbi_min_buy_prob``),
# which is the sharper instrument. Disqualification is for the markets with no
# surviving profitable pocket to screen for.
_NO_BUY_MARKETS: frozenset[str] = frozenset(
    {"batter_h", "batter_hrr", "batter_r", "batter_tb"}
)

# Longest price a buy may be taken at, per market, overriding the global
# ``EVThresholds.max_buy_odds``. Plus-money buys went 28.5% (n=933, -15.5% ROI)
# against 50.7% at minus money, but the cure has to be aimed: props are
# one-sided by construction -- a home run is honestly +500 -- and moneylines
# already have ``away_ml_refuse_odds``, which locates the damage on the road dog
# specifically. Run lines are what is left uncovered, and they split cleanly:
# +11.8% at -110 or shorter against -21.2% at plus money, where taking +1.5
# means paying a premium to need the fewest runs.
_MAX_BUY_ODDS_BY_MARKET: dict[str, float] = {
    "game_rl": 109.0,
    "f5_rl": 109.0,
}

# Weight given to the devigged market price per market, overriding the global
# ``Config.market_anchor``. Scoring both probability sources on the 10,497
# real-priced graded rows, the market is the better forecaster everywhere the
# engine bets (Brier: batter props .2180 vs .2210, F5 .2425 vs .2674, moneyline
# .2470 vs .2597, pitcher props .2461 vs .2769, run lines .2414 vs .2567) --
# except totals, where the model wins (.2446 vs .2480) and is also the only
# profitable buy bucket (+16 units on n=93).
#
# Totals are therefore pinned at zero rather than left to inherit the global
# weight: anchoring scales the measured edge by ``1 - w``, so raising the global
# toll to make the engine defer where it is beaten would silently double the
# edge required in the one market it is not.
_MARKET_ANCHOR_BY_MARKET: dict[str, float] = {
    "game_total": 0.0,
    "f5_total": 0.0,
}


@dataclass(frozen=True)
class EVThresholds:
    """Cutoffs for buy tiers: an EV floor to clear, then edge to rank on."""

    # The price has to pay at all: EV per $1 at the best number we can bet must
    # exceed this. Deliberately 0 rather than a margin, because EV = decimal_odds
    # x edge, so an EV *margin* is a cheaper bar the longer the price -- the old
    # 0.03 floor left the thin-edge band 73% plus-money, and it lost 13.5% per
    # unit. The bar is sized in edge below, not here.
    min_ev: float = field(default_factory=lambda: _env_float("MLBE_MIN_EV", 0.0))
    # Minimum model edge over the no-vig market price required to buy (thin-edge
    # guard). Raise it to trade volume for realized PPV/NPV.
    min_edge: float = field(default_factory=lambda: _env_float("MLBE_MIN_EDGE", 0.02))
    # Extra edge, in probability points over ``min_edge``, that promotes a buy to
    # Strong. EV cannot do this job: EV = decimal_odds x edge, so an EV cutoff is
    # a *cheaper* bar at longer prices and the Strong tier filled up with
    # plus-money dogs (median price +101 vs -123 for Moderate) and inverted --
    # 39.9% (n=153) against Moderate's 46.9%.
    strong_edge_gap: float = field(
        default_factory=lambda: _env_float("MLBE_EDGE_STRONG_GAP", 0.02)
    )
    # Disagreement with the devigged market beyond which the edge is treated as a
    # model error rather than a bet. Realized win rate falls as the model departs
    # from the price: over the real-priced rows, buys inside 8 points went 51.0%
    # and buys past it 39.0% (-18.6% ROI). 1.0 disables the cap.
    max_edge: float = field(default_factory=lambda: _env_float("MLBE_MAX_EDGE", 0.08))
    # Strict selection: when set, downgrade every Moderate buy to Pass so only
    # Strong buys fire.
    strong_only: bool = field(default_factory=lambda: _env_bool("MLBE_STRONG_ONLY", False))
    # Longest American price a buy may be taken at. Off globally and applied per
    # market (see ``_MAX_BUY_ODDS_BY_MARKET``), because a long price means
    # opposite things on a two-sided team market and a one-sided prop.
    max_buy_odds: float = field(
        default_factory=lambda: _env_float("MLBE_MAX_BUY_ODDS", math.inf)
    )
    # Never buy this market's over, whatever the price (see
    # ``_NO_BUY_MARKETS``); the fade keeps its own screens.
    no_buy: bool = False

    def for_market(self, market: str) -> EVThresholds:
        """Per-market thresholds, overridable via ``MLBE_MIN_EDGE_<MARKET>`` etc.

        Lets a market the audit flags as a false-positive pocket (e.g.
        ``pitcher_outs``) be tightened independently -- raising its buy bar lifts
        realized PPV -- without touching the rest of the sheet. Falls back to the
        global cutoffs when no override is set.
        """
        suffix = market.upper()
        # The flagged floor raises the global guard, it never lowers it: a
        # tightened MLBE_MIN_EDGE must not loosen the leakiest markets.
        d_edge = max(_OVERBET_EDGE_FLOORS.get(market, 0.0), self.min_edge)
        return EVThresholds(
            min_ev=_env_float(f"MLBE_MIN_EV_{suffix}", self.min_ev),
            min_edge=_env_float(f"MLBE_MIN_EDGE_{suffix}", d_edge),
            # A gap, not an absolute edge, so a market with a raised floor keeps
            # a Moderate band above it instead of grading every buy Strong.
            strong_edge_gap=_env_float(
                f"MLBE_EDGE_STRONG_GAP_{suffix}", self.strong_edge_gap
            ),
            max_edge=_env_float(f"MLBE_MAX_EDGE_{suffix}", self.max_edge),
            strong_only=_env_bool(f"MLBE_STRONG_ONLY_{suffix}", self.strong_only),
            max_buy_odds=_env_float(
                f"MLBE_MAX_BUY_ODDS_{suffix}",
                _MAX_BUY_ODDS_BY_MARKET.get(market, self.max_buy_odds),
            ),
            no_buy=_env_bool(f"MLBE_NO_BUY_{suffix}", market in _NO_BUY_MARKETS),
        )


@dataclass(frozen=True)
class RunLineGates:
    """NPV gates that veto a run-line selection outright.

    Each gate removes selections whose *failure* to cover is highly predictable,
    lifting realized NPV at the cost of bet volume. They ship disabled so each
    can be A/B'd against the ledger one at a time (see ``runline_metrics``).
    """

    # Favorite -1.5: a low-power lineup facing a ground-ball starter has almost
    # no blowout path (needs multi-run homers it cannot hit).
    # On by default: over eight graded slates this gate removed six favorite
    # -1.5s that went 1-5, against a 59.6% baseline for the run lines it kept.
    # Thresholds are the engine's own tracked-batted-ball scale, which reads a
    # few points below the public leaderboards the .140/.50 figures come from.
    iso_gb: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_ISO_GB", True))
    iso_max: float = field(default_factory=lambda: _env_float("MLBE_RL_ISO_MAX", 0.170))
    gb_min: float = field(default_factory=lambda: _env_float("MLBE_RL_GB_MIN", 0.40))

    # Underdog +1.5: a starter putting runners on and giving up hard contact is
    # a blowout waiting to happen.
    # Both underdog gates stay off: over the same eight slates they removed 35
    # +1.5s that won at 66-75%, i.e. they deleted winners. The sim already
    # prices weak-starter and weak-bullpen matchups, so the gates double-count.
    dog_sp: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_DOG_SP", False))
    dog_sp_whip_max: float = field(default_factory=lambda: _env_float("MLBE_RL_DOG_WHIP_MAX", 1.45))
    dog_sp_hard_hit_max: float = field(
        default_factory=lambda: _env_float("MLBE_RL_DOG_HARD_HIT_MAX", 0.45)
    )

    # Underdog +1.5: a bullpen that cannot strand inherited runners hands the
    # favorite the late-innings cushion that breaks the spread.
    dog_pen: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_DOG_PEN", False))
    dog_pen_xwoba_max: float = field(
        default_factory=lambda: _env_float("MLBE_RL_DOG_PEN_XWOBA_MAX", 0.330)
    )
    dog_pen_k_min: float = field(default_factory=lambda: _env_float("MLBE_RL_DOG_PEN_K_MIN", 0.18))

    # Favorite -1.5: low-total games trend to 1-run margins. Redundant with the
    # simulated margin distribution, so it is off by default.
    low_total: bool = field(default_factory=lambda: _env_bool("MLBE_RL_GATE_TOTAL", False))
    low_total_max: float = field(default_factory=lambda: _env_float("MLBE_RL_TOTAL_MAX", 7.0))


@dataclass(frozen=True)
class Credentials:
    """Credentials for subscription data sources and SMTP (never logged)."""

    fangraphs_user: str | None = field(default_factory=lambda: os.getenv("FANGRAPHS_USER"))
    fangraphs_pass: str | None = field(default_factory=lambda: os.getenv("FANGRAPHS_PASS"))
    rotowire_user: str | None = field(default_factory=lambda: os.getenv("ROTOWIRE_USER"))
    rotowire_pass: str | None = field(default_factory=lambda: os.getenv("ROTOWIRE_PASS"))
    vsin_user: str | None = field(default_factory=lambda: os.getenv("VSIN_USER"))
    vsin_pass: str | None = field(default_factory=lambda: os.getenv("VSIN_PASS"))
    # The Odds API (https://the-odds-api.com) key: multi-book ML/run-line/total prices.
    # Prefer the vendor-canonical THE_ODDS_API_KEY; fall back to the legacy ODDS_API_KEY.
    odds_api_key: str | None = field(
        default_factory=lambda: os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    # Generic SMTP credentials used by nightly audit emails.
    smtp_server: str | None = field(default_factory=lambda: os.getenv("SMTP_SERVER"))
    smtp_port: int = field(default_factory=lambda: _env_int("SMTP_PORT", 587))
    smtp_user: str | None = field(default_factory=lambda: os.getenv("SMTP_USER"))
    smtp_pass: str | None = field(default_factory=lambda: os.getenv("SMTP_PASS"))
    # Gmail credentials used by daily-card emails.
    gmail_user: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    )
    gmail_app_password: str | None = field(
        default_factory=lambda: os.getenv("GMAIL_APP_PASSWORD")
    )

    def has_fangraphs(self) -> bool:
        return bool(self.fangraphs_user and self.fangraphs_pass)

    def has_rotowire(self) -> bool:
        return bool(self.rotowire_user and self.rotowire_pass)

    def has_vsin(self) -> bool:
        return bool(self.vsin_user and self.vsin_pass)

    def has_odds_api(self) -> bool:
        return bool(self.odds_api_key)

    def has_email(self) -> bool:
        return bool(self.gmail_app_password)


@dataclass(frozen=True)
class Config:
    windows: RollingWindows = field(default_factory=RollingWindows)
    ev: EVThresholds = field(default_factory=EVThresholds)
    runline_gates: RunLineGates = field(default_factory=RunLineGates)
    creds: Credentials = field(default_factory=Credentials)

    # RBI hard-rule threshold: on-base pct of preceding 3 batters over 3wk window.
    rbi_obp_threshold: float = field(default_factory=lambda: _env_float("MLBE_RBI_OBP", 0.345))

    # Contact-quality floor on batter props: exclude low-power bats from the
    # power markets (HR/XBH/TB) below MLBE_POWER_XSLG_FLOOR xSLG, and whiff-prone
    # bats from the contact markets (H/1B/H+R+RBI) above MLBE_CONTACT_K_CEILING
    # K%. Live by default; set MLBE_POWER_FLOOR=0 to disable.
    power_floor: bool = field(default_factory=lambda: _env_bool("MLBE_POWER_FLOOR", True))
    # .360 rather than .400: the floor was set against an expected slugging read
    # per batted ball, whose league mean is .486, and cut the bottom sixth of a
    # slate's starters. Against a calibrated xSLG (league mean .400) the same
    # .400 would cut half of them; .360 holds the cut where it was (15.9% of the
    # Aug 8 lineups against 16.7%).
    power_xslg_floor: float = field(
        default_factory=lambda: _env_float("MLBE_POWER_XSLG_FLOOR", 0.360)
    )
    contact_k_ceiling: float = field(
        default_factory=lambda: _env_float("MLBE_CONTACT_K_CEILING", 0.25)
    )

    # Hierarchical batter splits: regress each home/away and platoon split
    # toward the batter's own overall rate rather than toward the league. Set
    # MLBE_BATTER_SPLIT_PRIOR=0 to restore the flat league prior everywhere.
    batter_split_prior: bool = field(
        default_factory=lambda: _env_bool("MLBE_BATTER_SPLIT_PRIOR", True)
    )

    # Rest-of-season projections as the batter prior: a hitter's window
    # regresses toward his own projection at the per-outcome strengths in
    # OUTCOME_PRIOR_STRENGTH instead of toward the league mean at a flat 60 PA.
    #
    # This was built off and switched off, because the export it was written for
    # (THE BAT X via FanGraphs) needs a subscription and does not survive the
    # Cloudflare challenge from this machine, so there was never a file to read.
    # It falls back to a Marcel off the free official season lines
    # (``scripts.ros_prior_study marcel``, or ``mlb-engine ros-prior``), which
    # writes ``ros_hitters.csv`` into the data directory -- so the default is the
    # standard path rather than off, and an operator with no file still gets
    # exactly today's behaviour.
    #
    # A subscriber can hand the better projection over the wall instead: any
    # projection CSV dropped in ``projections_dir`` is read first and the Marcel
    # covers whoever it does not list. THE BAT X spreads hitters 25% wider than
    # the Marcel does (sd of projected wOBA .0278 against .0222, r .80 between
    # them), which is the direction that matters -- compressing the lineup is
    # this engine's known failure, not over-separating it.
    #
    # Measured forward, 113 hitters over the 8,494 PA in the three weeks after a
    # 07-22 cutoff, priors built from seasons the holdout cannot reach:
    #
    #     vector                            log loss / PA   vs today
    #     league rate for everyone               1.46643     +0.0024
    #     window, flat 60 PA to league (today)   1.46403       0
    #     window, fitted strengths to league     1.45845     -0.0056
    #     window, fitted strengths to projection 1.45411     -0.0099  (5.8 SE)
    #
    # The line worth reading twice is the first: today's hitter model beats
    # giving every hitter the league line by 0.8 SE. Set MLBE_ROS_PRIOR to a
    # path to point elsewhere, or to an empty string to restore the league mean.
    ros_prior_path: str | None = field(
        default_factory=lambda: os.environ.get("MLBE_ROS_PRIOR", default_ros_prior_path()) or None
    )

    # Which dropped-in projection to prefer when the folder holds several,
    # matched against the file name. ATC is the default because it is an
    # accuracy-weighted ensemble of the systems below it, and a model whose job
    # is to *rank* hitters wants the projection that is rarely badly wrong about
    # anyone over the one that is sharpest about some. Set MLBE_PROJECTION_SOURCE
    # to batx to anchor on the Statcast batted-ball system instead; an empty
    # value takes whichever export was written most recently.
    projection_source: str = field(
        default_factory=lambda: os.environ.get("MLBE_PROJECTION_SOURCE", "atc")
    )

    # Per-outcome shrinkage on the bullpen aggregate (PEN_PRIOR_STRENGTH), whose
    # three-week sample is thin enough that at the flat 60 PA the pen vector is
    # mostly binomial noise -- and whose doubles/triples-allowed spread across the
    # 30 pens is *entirely* noise.
    #
    # On by default since the out-of-time check that was owed: 330 team-windows
    # (Apr-Jul, 21-day read scored against the *next* 21 days of relief wOBA
    # allowed), regressing what happened next on what the window said --
    #
    #     read              sd      slope on the next window   RMSE
    #     raw             .0371            0.15                .0504
    #     flat 60 PA      .0300            0.19                .0462
    #     fitted priors   .0106            0.62                .0398
    #     league mean      ---              ---                .0397
    #
    # A slope of 0.15 is the definition of a read used at six times its worth:
    # production was handing the simulator a pen line whose spread is five sixths
    # sampling error, and paying for it -- the raw and flat-60 reads forecast the
    # next three weeks *worse than assuming every pen is league average*. The
    # fitted priors are the first version that does not.
    pen_shrink: bool = field(default_factory=lambda: _env_bool("MLBE_PEN_SHRINK", True))

    # Singles "Under" screen: exclude the singles/H/H+R+RBI OVER for batters with
    # a strong structural anti-singles profile (high K%, fly-ball tilt -- the two
    # flags that survived the out-of-time fit). Live by default; set
    # MLBE_SINGLES_UNDER=0 to disable, MLBE_SINGLES_UNDER_MIN to retune the score.
    singles_under: bool = field(
        default_factory=lambda: _env_bool("MLBE_SINGLES_UNDER", True)
    )
    singles_under_min: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_UNDER_MIN", 3.0)
    )
    # Score a singles UNDER must show before it is bought. The profile is the
    # only screen the fade side has, and it is the one thing measured to separate
    # a paying singles under from a losing one: on the shadow book, buys on a
    # batter over the strikeout flag went 75.9% for +24.2%, and buys below it
    # 45.1% for -16.9% (n=29 / 71). 2.0 is the strikeout flag on its own.
    singles_under_buy_min: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_UNDER_BUY_MIN", 2.0)
    )

    # Opposing-starter SIERA gate on the batter singles/hit market. Since singles
    # PPV is weak, lean on matchup quality: skip the singles/H/H+R+RBI OVER when
    # the batter faces an ace (SIERA < ace floor, e.g. Arraez vs Skubal), and do
    # NOT apply the singles-Under exclusion when he faces a scrub (SIERA > bad
    # ceiling, e.g. a power bat vs Gallen -- a weak arm inflates cheap singles).
    # SIERA is computed from Statcast; the gate stays neutral when the opposing
    # starter has too few PA. Set MLBE_SINGLES_SIERA=0 to disable.
    singles_siera: bool = field(
        default_factory=lambda: _env_bool("MLBE_SINGLES_SIERA", True)
    )
    singles_siera_ace: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_SIERA_ACE", 3.4)
    )
    singles_siera_bad: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_SIERA_BAD", 4.4)
    )

    # Thin-Statcast starter gate: when a probable starter has fewer than
    # MLBE_THIN_SP_MIN_PITCHES tracked pitches in the window (e.g. a debut/call-up
    # with no MLB data), the engine has no real read on him and falls back to an
    # optimistic prior -- which manufactures phantom edges on that game's
    # starter-driven markets (game/F5 ML, run line, totals, and his pitcher props).
    # Veto those to Pass rather than bet a matchup the model can't price. Live by
    # default; set MLBE_THIN_SP_GATE=0 to disable.
    thin_starter_gate: bool = field(
        default_factory=lambda: _env_bool("MLBE_THIN_SP_GATE", True)
    )
    thin_starter_min_pitches: int = field(
        default_factory=lambda: _env_int("MLBE_THIN_SP_MIN_PITCHES", 150)
    )

    # Default recipient for the nightly audit email.
    audit_email: str = field(
        default_factory=lambda: os.getenv("MLBE_AUDIT_EMAIL", "drfobusan@gmail.com")
    )
    # Daily-card email delivery. Env names mirror scripts/email_results.py
    # (MLB_EMAIL_TO / SMTP_HOST / SMTP_PORT), with MLBE_-prefixed overrides.
    email_to: str | None = field(
        default_factory=lambda: os.getenv("MLBE_EMAIL_TO") or os.getenv("MLB_EMAIL_TO")
    )
    smtp_host: str = field(
        default_factory=lambda: os.getenv("MLBE_SMTP_HOST")
        or os.getenv("SMTP_HOST")
        or "smtp.gmail.com"
    )
    smtp_port: int = field(
        default_factory=lambda: _env_int("MLBE_SMTP_PORT", _env_int("SMTP_PORT", 465))
    )

    # Monte Carlo simulation count per game.
    mc_sims: int = field(default_factory=lambda: _env_int("MLBE_MC_SIMS", 20000))

    # Apply the historical isotonic probability calibration before EV/tiers.
    calibrate: bool = field(
        default_factory=lambda: os.getenv("MLBE_CALIBRATE", "1") not in ("0", "false", "")
    )
    # Compress the over-confident tails after calibration (see ConfidenceShrink).
    shrink_tails: bool = field(default_factory=lambda: _env_bool("MLBE_SHRINK_TAILS", True))
    shrink_pivot: float = field(default_factory=lambda: _env_float("MLBE_SHRINK_PIVOT", 0.60))
    shrink_slope: float = field(default_factory=lambda: _env_float("MLBE_SHRINK_SLOPE", 0.55))

    # Post-simulation TB/RBI selector scaling. The selector's park/weather and
    # batted-ball terms are already applied inside the simulation, so scaling
    # the simulated count arrays again double-counts them; kept only as an
    # escape hatch for reproducing pre-fix runs.
    legacy_prop_post_mult: bool = field(
        default_factory=lambda: _env_bool("MLBE_LEGACY_PROP_POST_MULT", False)
    )

    # High pitcher-K lines (o6.5+) are a cumulative false-positive pocket
    # (~38% hit vs the ~52.4% breakeven): the model over-projects strikeouts at
    # the top of the ladder. Gate any K prop whose line exceeds this cap to Pass
    # so only the reliable low lines (o4.5/o5.5) can be bought.
    pitcher_k_max_buy_line: float = field(
        default_factory=lambda: _env_float("MLBE_PITCHER_K_MAX_LINE", 5.5)
    )

    # Walks-allowed unders are vetoed while the model's walk level is unvalidated.
    #
    # Pricing both sides of every prop opened this side up, and pitcher_bb is the
    # one market whose *over* was the profitable side: on the 149 graded rows
    # carrying both prices the over returned +2.84% and the under -17.88%, with a
    # bootstrap interval of [-32.7%, -3.0%] that excludes zero. The cause is small
    # and the loss is not, which is the point -- pitchers walked 2+ in 55.0% of
    # those starts against a devigged market price of 49.6%, a 5.4-point miss that
    # is only 1.32 SE, but the under is the short side and short prices turn a
    # coin-flip miss into a fifth of the stake.
    #
    # The model then leans that way by construction: it averaged .4649 on P(BB>=2)
    # against the market's .4961, so at a .03 minimum edge it takes 81 unders and
    # 50 overs -- 62% of its walk buys on the side that lost. That lean is not
    # evidence, because every one of those rows was priced before the league walk
    # prior was corrected, so the veto stands until slates graded on the current
    # basis can measure the level. Set MLBE_PITCHER_BB_UNDER=1 to allow them.
    pitcher_bb_under_gate: bool = field(
        default_factory=lambda: not _env_bool("MLBE_PITCHER_BB_UNDER", False)
    )

    # Home-run overs only pay inside a price band. Graded buys by price:
    # +300-400 lost 50% of stake on 10 bets, +400-500 returned +20.5% on 19, and
    # everything from +500 up collapsed -- 62 bets at +500 or longer won three
    # times between them (-37 units, the single largest pocket left on the card).
    # The long prices are where a small absolute probability error is a huge
    # relative one, so the model's edge there is noise wearing a big number.
    # This is a hard price screen: a home-run buy must sit inside the band no
    # matter what EV or probability the model reports for it.
    hr_min_buy_odds: float = field(
        default_factory=lambda: _env_float("MLBE_HR_MIN_BUY_ODDS", 400.0)
    )
    hr_max_buy_odds: float = field(
        default_factory=lambda: _env_float("MLBE_HR_MAX_BUY_ODDS", 700.0)
    )

    # Singles are won or lost on the price, not on the read. The hit rate on a
    # singles buy is ~35-39% whatever the book charges -- our probability barely
    # moves with the market's -- so the same pick profits at +200 and loses two
    # thirds of stake at -130. Graded: 34 minus-money buys cost 15.4 units,
    # while the 78 at +150 or better returned +4.4%. A price floor rather than a
    # model claim: it stops us paying a premium for a read we do not have.
    singles_min_buy_odds: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_MIN_BUY_ODDS", 100.0)
    )

    # Doubles are refused at +300 and longer, which on the card so far is 69 of
    # 70 graded buys: this is a shut market with a door left open, and it is
    # written as a price rather than as a disqualification because a double is
    # only ever a long price when the model is the one claiming the edge.
    #
    # The market is calibrated everywhere except where it bets. Over 6,656
    # graded o0.5 rows the model reads .140 -> 14.0% actual, .165 -> 14.8%,
    # .188 -> 17.7%, and then .258 -> 15.0% (n=346). The confident tail is the
    # only broken band and the buy list is drawn entirely from it, which is why
    # the selection is worth nothing: bought rows hit 14.3% (n=70) against
    # passed rows' 14.2% (n=6,586).
    #
    # It is a price ceiling and not a band because no band survives contact
    # with the rows: +300-350 went 0 for 5, +350-400 -14.4%, +400-450 -25.4%,
    # +450-500 -16.1%, +500 and longer +18.1% on three winners in nineteen.
    # Fitting the cutoff to that shape would be fitting it to one hitter's good
    # night. What the data does support is the general fact the home-run band
    # already encodes -- at 20% and longer a fraction of a point of probability
    # error is a fifth of the stake -- plus the measured absence of any edge at
    # all on this market.
    #
    # Three cautions, kept here because the next person to read the cell will
    # find them: 70 buys is under the 100 ``probation`` needs to condemn a
    # market, the halves of that window disagree (+16% then -61%), and the
    # engine's own doubles multiplier was already stripped to sprint speed in
    # #129 for the same reason. The evidence for shutting the buys is the
    # 6,656-row calibration table, not the 70. Refusals keep grading as shadow
    # bets, so ``screen_probation`` will say to lift this if it starts deleting
    # winners -- which is also why the screen runs last in ``_mk`` rather than
    # beside the other price bands: run early it inherits the rows the contact
    # floor would have refused anyway, and is then judged on bets it never
    # removed. ``MLBE_DOUBLES_MAX_BUY_ODDS=100000`` disables it.
    doubles_max_buy_odds: float = field(
        default_factory=lambda: _env_float("MLBE_DOUBLES_MAX_BUY_ODDS", 300.0)
    )

    # Player-prop markets that get an under recommendation as well as an over.
    # Every prop was over-only, so a fade could only ever be a Pass; the
    # shadow book (1,560 gradeable unders at EV>2%, -0.1%) says the under is
    # not free money, so it is bet on its own EV like any other side.
    #
    # Home runs, doubles and triples are deliberately absent. They are rare
    # events, so the under is a heavy favourite whose vig swallows any edge,
    # and after #132/#138 the engine's extra-base numbers are near-flat by
    # design -- fading them would be betting the prior, not a read.
    # Shortest price an under may be bought at. A deep favourite has no room:
    # the graded shadow book returns -2.1% on unders priced worse than -300 and
    # -0.7% between -300 and -200 (they win 77% and 70%, and still lose), while
    # -200 to -150 returns +2.0% and plus money +1.9%. It bites hardest on RBI
    # unders, whose median shadow price was -350 for -10.7%.
    prop_under_min_price: float = field(
        default_factory=lambda: _env_float("MLBE_PROP_UNDER_MIN_PRICE", -250.0)
    )

    prop_under_markets: frozenset[str] = field(
        default_factory=lambda: _env_set(
            "MLBE_PROP_UNDER_MARKETS",
            (
                "batter_h",
                "batter_1b",
                "batter_tb",
                "batter_hrr",
                "batter_rbi",
                "pitcher_k",
                "pitcher_outs",
                "pitcher_h",
                "pitcher_bb",
                "pitcher_er",
            ),
        )
    )

    # Let the simulator lift a hitter mid-game instead of batting the same nine
    # to the last out. The hazard is measured (``features.removal``): 10.1% per
    # appearance once the opposing starter is gone, 22.7% for a wrong-handed bat
    # batting 9th, and the substitute who takes over is a worse hitter, so the
    # branch moves hits and total bases down through lost opportunity rather than
    # by cutting anyone's rates. On: the reprice it was waiting for scores it
    # against three independent measurements -- the credited-appearance ratio
    # lands on .9538 against a play-by-play .9541, the substitute share runs
    # 2.5% at the top of the order to 7.8% at the bottom against a measured
    # 3.1% to 8.0%, and both agree with the box score's own ``battingOrder``
    # codes. It takes 1.0-1.4pp off every batter market, which is about 40% of
    # the +1.75-3.40pp the graded ledger says those markets are long by; the
    # rest is level rather than opportunity and is not corrected here.
    # ``MLBE_REMOVAL_HAZARD=0`` reverts to batting the same nine.
    removal_hazard: bool = field(
        default_factory=lambda: _env_bool("MLBE_REMOVAL_HAZARD", True)
    )

    # RBI overs are the one market where a conviction floor works: 20.5 of the
    # 21.5 units that market lost came from buys under 40% model probability
    # (11 bets under 30% alone lost 61.8% of stake), while everything above the
    # floor was roughly flat. All of them were plus-money o0.5 tickets -- cheap
    # lottery lines the EV screen liked precisely because the payout was long.
    rbi_min_buy_prob: float = field(
        default_factory=lambda: _env_float("MLBE_RBI_MIN_BUY_PROB", 0.40)
    )

    # The conviction ceiling. Batter-prop buys since the screens shipped on
    # 08-04, by model probability, against every one of those rows all-time:
    #
    #                since 08-04                 all 1,320 graded
    #   p < .50     n=171  won 33.3%   +4.8%     -8.3%
    #   .50-.58     n= 65  won 46.2%   -8.0%    -15.7%
    #   .58-.62     n= 32  won 50.0%  -13.0%    -17.2%
    #   p >= .62    n=188  won 56.9%  -11.4%    -16.5%  (model said 66.9%)
    #
    # Confidence is inverted. Every band improved once the screens landed
    # except the most confident one, which is 14pp long, is the only band still
    # losing double digits, and loses at short prices (-16.5%, n=239) as much
    # as at long ones. This is a screen, not a recalibration -- it moves no
    # probability, so the 08-17 refit grades against the same basis and can
    # retire the ceiling if it fixes the level.
    # ``MLBE_BATTER_MAX_BUY_PROB=1`` disables it.
    batter_max_buy_prob: float = field(
        default_factory=lambda: _env_float("MLBE_BATTER_MAX_BUY_PROB", 0.62)
    )

    # The road moneyline underdog is the only sides cell the graded card
    # condemns twice. Split four ways by venue and role:
    #
    #   away ML dog   n=77  won 28.6%  -26.1u  -33.9%   (train -40.1, test -23.8)
    #   home ML dog   n=35  won 45.7%   +1.1u   +3.0%
    #   away ML fav   n=41  won 51.2%   -3.2u   -7.7%
    #   home ML fav   n=63  won 50.8%   -7.3u  -11.6%
    #
    # Neither "underdogs" nor "road teams" is the problem; their intersection
    # is. It is the highest-variance side on the board and the one where the
    # book's edge on lineups, travel and bullpen availability is largest, so our
    # probability error is both biggest and most amplified by the payout -- the
    # same failure as the +700 home run. It applies to the full game and the
    # first five alike, because each condemns itself without the other's help:
    #
    #   game_ml away dog  n=42  -39.2%   (train -43.5, test -27.0)
    #   f5_ml   away dog  n=35  -27.6%   (train -33.7, test -21.8)
    #
    # Run lines are deliberately untouched: every rule fitted to them reverses
    # sign across the window, including the home +1.5 that made 34.5% in July
    # and lost 11.9% in August.
    away_ml_refuse_odds: float = field(
        default_factory=lambda: _env_float("MLBE_AWAY_ML_REFUSE_ODDS", 100.0)
    )

    # Pitcher-outs is a cumulative false-NEGATIVE pocket: over the graded window
    # the model under-projected outs in its meaty 0.45-0.60 band, which actually
    # cashed 53-70% (vs the 45-55% it priced), so profitable outs-overs were
    # passed. This adds a small, capped upward bias to the calibrated
    # pitcher_outs over-probability -- only below ``pitcher_outs_bias_max_prob``,
    # where the miss was measured -- to lift those bets back over the buy bar.
    # Conservative default (well under the measured 8-15pp gap); provisional
    # pending CLV validation. Set MLBE_PITCHER_OUTS_PROB_BIAS=0 to disable.
    pitcher_outs_prob_bias: float = field(
        default_factory=lambda: _env_float("MLBE_PITCHER_OUTS_PROB_BIAS", 0.04)
    )
    pitcher_outs_bias_max_prob: float = field(
        default_factory=lambda: _env_float("MLBE_PITCHER_OUTS_BIAS_MAX_PROB", 0.62)
    )

    # Barrel rate is a negative for singles: power hitters take the same number
    # of hits but convert them to extra bases. The slope prices the half of that
    # effect the simulated K rate does not already carry; ``power_split`` stops
    # the distribution-tail bonus lifting 1B by the same factor it lifts HR.
    singles_barrel: bool = field(default_factory=lambda: _env_bool("MLBE_SINGLES_BARREL", True))
    singles_barrel_slope: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_BARREL_SLOPE", SINGLES_BARREL_SLOPE)
    )
    tail_power_split: bool = field(
        default_factory=lambda: _env_bool("MLBE_TAIL_POWER_SPLIT", True)
    )

    # Batted-ball mix is the other half of the same story: a single is a ground
    # ball or a line drive, almost never a ball hit in the air. Previously off,
    # because an eight-slate fit could not separate the ground-ball slope from
    # zero. Both slopes are now fitted out of time -- a 42-day window predicting
    # the *following* 21 days, over 862 batter-windows -- and clear p<1e-4 taken
    # alone, so the terms are live.
    singles_gb: bool = field(default_factory=lambda: _env_bool("MLBE_SINGLES_GB", True))
    singles_gb_slope: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_GB_SLOPE", SINGLES_GB_SLOPE)
    )
    singles_ld_slope: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_LD_SLOPE", SINGLES_LD_SLOPE)
    )
    # Shape within those classes -- pulled grounders, soft line drives, line
    # drives an outfielder can cut off. Smaller effects than the mix itself
    # (p .013-.046), so they get their own switch.
    singles_shape: bool = field(
        default_factory=lambda: _env_bool("MLBE_SINGLES_SHAPE", True)
    )

    # Blend each hitter's observed HR/PA toward what his batted balls were worth
    # against the walls they were hit toward (xHR/PA). On by default: raw HR/PA
    # carries the park and weather noise expected HR exists to strip out.
    xhr_blend: bool = field(default_factory=lambda: _env_bool("MLBE_XHR_BLEND", True))
    xhr_prior_weight: float = field(
        default_factory=lambda: _env_float("MLBE_XHR_PRIOR_WEIGHT", HR_PRIOR_WEIGHT)
    )
    # Having neutralised the parks he came from, apply the one he is walking
    # into: his own batted balls re-scored against tonight's fences. A scalar
    # park factor cannot tell a pull-heavy lefty from an opposite-field bat.
    xhr_park: bool = field(default_factory=lambda: _env_bool("MLBE_XHR_PARK", True))

    # The ballpark on the singles line. The runs park factor is mostly home runs
    # and carries no singles signal, so singles were priced identically at
    # Coors and Dodger Stadium; ``Park.singles_factor`` is the component version,
    # measured per park and shrunk to its split-half reliability.
    park_singles: bool = field(
        default_factory=lambda: _env_bool("MLBE_PARK_SINGLES", True)
    )

    # The ballpark on the doubles line, and the widest of the component factors:
    # ``Park.xbh_factor`` spans 0.86..1.25 after shrinking, against singles'
    # 0.945..1.035, because outfield geometry varies more than fence distance
    # does. Total bases were priced identically at Coors and Wrigley without it.
    park_xbh: bool = field(default_factory=lambda: _env_bool("MLBE_PARK_XBH", True))

    # Read a bullpen's contact quality as talent rather than as luck. The pen
    # covers ~44% of a hitter's plate appearances and was running through the
    # small-sample luck corrections built for starters -- on ~1,240 pooled
    # batted balls those invert, suppressing the pens that allow the most hits.
    pen_contact_level: bool = field(
        default_factory=lambda: _env_bool("MLBE_PEN_CONTACT_LEVEL", True)
    )

    # Bridge innings: once the starter is hooked in a close game, the simulator
    # used to hand every remaining inning to the pen's 8th+ leverage profile,
    # so a 6th-inning hand-off was charged the closer's rates. With this on, the
    # innings before the 8th are priced off the arms that actually cover them
    # (relief before the 8th) and the leverage profile starts in the 8th. It
    # moves full-game and total prices, so it can be turned off to reprice a
    # slate the old way for comparison.
    pen_bridge: bool = field(default_factory=lambda: _env_bool("MLBE_PEN_BRIDGE", True))
    # Extend the starter's arsenal matching (pitch-mix usage x per-class SwStr%
    # vs the hitter's per-class whiff/xwOBA) to the bullpen matchups, read
    # separately for the bridge and leverage subsets of the corps.
    pen_arsenal: bool = field(default_factory=lambda: _env_bool("MLBE_PEN_ARSENAL", True))

    # Odds API credit budget. The vendor bills markets x regions per request, so
    # a 16-game slate at every market it can name costs ~230 credits. Props are
    # restricted to the markets with a positive graded edge (see
    # data.oddsapi.DEFAULT_PROP_MARKETS); MLBE_ODDS_PROPS takes a comma list.
    odds_props: tuple[str, ...] | None = field(
        default_factory=lambda: _env_csv("MLBE_ODDS_PROPS")
    )
    odds_f5: bool = field(default_factory=lambda: _env_bool("MLBE_ODDS_F5", True))
    # Re-running the same slate inside the TTL costs nothing.
    odds_cache_ttl: int = field(default_factory=lambda: _env_int("MLBE_ODDS_CACHE_TTL", 1800))
    # Credits held in reserve so one runaway slate cannot drain the plan.
    odds_min_credits: int = field(default_factory=lambda: _env_int("MLBE_ODDS_MIN_CREDITS", 200))

    # How long a forecast is reused before it is pulled again. The point is not
    # the API quota (Open-Meteo is free) but reproducibility: the forecast for a
    # park moves between calls, so an uncached run of the same slate off the same
    # odds board priced 6,050 of 6,705 rows differently from the run before it,
    # mean 1.23pp and up to 6.9pp -- larger than most of the changes the engine is
    # asked to measure. Matching the odds TTL means one slate is priced on one
    # forecast and a re-run reproduces it; a past date never re-fetches at all.
    weather_cache_ttl: int = field(
        default_factory=lambda: _env_int("MLBE_WEATHER_CACHE_TTL", 1800)
    )

    # Weight given to the devigged market price when forming the probability the
    # EV screen bets on (see market.ev.anchor_to_market). The model's own
    # probability is untouched, so PPV/NPV and the calibration refit still
    # measure the model. Because the screen is affine in the probability, a weight
    # w is equivalent to demanding edge >= threshold / (1 - w): it raises the toll
    # on disagreeing with the market rather than making the engine defer to it.
    # Nine retro-priced slates: ROI -5.4% at 0, -4.1% at 0.4, -3.5% at 0.6 on a
    # third as many bets, -12.9% at 0.8 -- every interval spans zero, so this
    # shrinks a loss rather than earning a profit; judge a weight on closing
    # line value, which resolves in far fewer bets than ROI.
    #
    # Still off, despite 27 graded slates putting the market ahead of the model
    # on Brier and log loss in every market the engine bets except totals, and
    # for a mechanical reason rather than a lack of evidence: every edge floor,
    # price band and probability floor on the card was fitted against unanchored
    # probabilities, and a global weight rescales all of them at once
    # (edge -> edge x (1 - w)). Raise it per market with
    # ``MLBE_MARKET_ANCHOR_<MARKET>`` and re-grade that market's floors with it.
    market_anchor: float = field(default_factory=lambda: _env_float("MLBE_MARKET_ANCHOR", 0.0))

    # Run-line luck-gap tier nudge (season actual RD vs xwOBA-based xRD). Reads the
    # daily-built team-form cache; OFF by default until the graded-data backtest
    # sets the threshold.
    runline_luck_gap: bool = field(
        default_factory=lambda: _env_bool("MLBE_RL_LUCK_GAP", False)
    )

    # Shared state (see mlb_engine/state.py). On by default because the runs
    # that need it most are scheduled ones on disposable machines, and it is
    # best-effort: no remote, no branch or no credentials just means the run
    # keeps its own local state.
    state_sync: bool = field(default_factory=lambda: _env_bool("MLBE_STATE_SYNC", True))
    state_branch: str = field(
        default_factory=lambda: os.getenv("MLBE_STATE_BRANCH", "engine-state")
    )

    # Directories.
    data_dir: Path = field(default_factory=_data_dir)

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def odds_cache_dir(self) -> Path:
        return self.cache_dir / "oddsapi"

    @property
    def weather_cache_dir(self) -> Path:
        return self.cache_dir / "weather"

    @property
    def calibration_file(self) -> Path:
        """Isotonic map to price with: a locally refit one wins if it exists.

        ``mlb-engine calibrate`` writes ``calibration_live.json`` into the data
        directory from the audit ledger, so an operator who has graded history
        prices off their own results instead of the packaged 2024 fit.
        """
        override = os.getenv("MLBE_CALIBRATION_FILE")
        if override:
            return Path(override)
        return self.data_dir / "calibration_live.json"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audit"

    @property
    def fangraphs_dir(self) -> Path:
        """Default drop-in folder for FanGraphs custom-report exports."""
        return self.data_dir / "fangraphs"

    @property
    def projections_dir(self) -> Path:
        """Drop-in folder for rest-of-season projection exports.

        The matching CSV here becomes the batter prior, with the Marcel filling
        in the hitters it omits. Standard-view FanGraphs exports carry the
        columns needed (``PA``, ``H``, ``2B``, ``3B``, ``HR``, ``BB``, ``SO``,
        ``MLBAMID``); an export without ``MLBAMID`` cannot be joined to Statcast
        and is skipped with a warning rather than name-matched.

        ``MLBE_PROJECTIONS_DIR`` points this at a folder the exports already land
        in -- a browser's download folder, typically, which is the difference
        between a file that is refreshed daily and one that is refreshed when
        somebody remembers to move it. Sharing a folder with every other download
        is why ``projection_source`` is matched strictly there.
        """
        override = os.getenv("MLBE_PROJECTIONS_DIR")
        if override:
            return Path(override).expanduser()
        return self.data_dir / "projections"

    @property
    def batx_dir(self) -> Path:
        """Priced THE BAT X exports, one CSV per slate (scripts/batx_study.py)."""
        return self.data_dir / "batx"

    @property
    def team_form_path(self) -> Path:
        """Cached daily-built season team-form baseline (luck-gap inputs)."""
        return self.cache_dir / "team_form.json"

    def anchor_for(self, market: str) -> float:
        """Anchor weight for one market: per-market default, then env override.

        The default is not uniform because the model's accuracy against the
        price is not uniform -- see ``_MARKET_ANCHOR_BY_MARKET``. A market with
        its own default ignores the global weight, so raising the global toll
        cannot start taxing the one market that out-forecasts the price.
        """
        return _env_float(
            f"MLBE_MARKET_ANCHOR_{market.upper()}",
            _MARKET_ANCHOR_BY_MARKET.get(market, self.market_anchor),
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.cache_dir,
            self.output_dir,
            self.audit_dir,
            self.fangraphs_dir,
            self.batx_dir,
            self.projections_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
