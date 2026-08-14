"""Central configuration for the MLB prediction engine.

Values are sourced from environment variables where relevant (credentials in
particular) so that nothing sensitive is committed. Everything else has sensible
defaults that can be overridden via environment variables prefixed ``MLBE_``.
"""

from __future__ import annotations

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
    # Share of a bullpen's distance from the league mean xwOBA to keep. 1.0 is
    # the raw three-week mean; 0.37 is its measured reliability.
    bullpen_xwoba_shrink: float = field(
        default_factory=lambda: _env_float("MLBE_BULLPEN_XWOBA_SHRINK", 1.0)
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


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

    # Rest-of-season projections as the batter prior. The path holds the file
    # written by ``scripts.ros_prior_study prior``; when it is set and readable,
    # a hitter's window regresses toward his own projection at the per-outcome
    # strengths in OUTCOME_PRIOR_STRENGTH instead of toward the league mean at a
    # flat 60 PA. Off by default: it moves every batter probability, so it is
    # meant to arrive with a calibration refit rather than mid-window.
    ros_prior_path: str | None = field(
        default_factory=lambda: os.getenv("MLBE_ROS_PRIOR") or None
    )

    # Per-outcome shrinkage on the bullpen aggregate (PEN_PRIOR_STRENGTH), whose
    # three-week sample is thin enough that at the flat 60 PA the pen vector is
    # mostly binomial noise -- and whose doubles/triples-allowed spread across the
    # 30 pens is *entirely* noise. Off by default for the same reason as the ROS
    # prior: it moves every pen-driven probability, so it ships with a refit.
    pen_shrink: bool = field(default_factory=lambda: _env_bool("MLBE_PEN_SHRINK", False))

    # Singles "Under" screen: exclude the singles/H/H+R+RBI OVER for batters with
    # a strong structural anti-singles profile (TTO volume, fly-ball tilt, elite
    # power contact, pull-heavy grounders). Live by default; set
    # MLBE_SINGLES_UNDER=0 to disable, MLBE_SINGLES_UNDER_MIN to retune the score.
    singles_under: bool = field(
        default_factory=lambda: _env_bool("MLBE_SINGLES_UNDER", True)
    )
    singles_under_min: float = field(
        default_factory=lambda: _env_float("MLBE_SINGLES_UNDER_MIN", 3.0)
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

    # Let the simulator lift a hitter mid-game instead of batting the same nine
    # to the last out. The hazard is measured (``features.removal``): 8.6% per
    # appearance once the opposing starter is gone, 24.2% for a wrong-handed bat
    # batting 9th, and the substitute who takes over is a worse hitter, so the
    # branch moves hits and total bases down through lost opportunity rather than
    # by cutting anyone's rates. Off until the reprice scores it: it changes every
    # batter number the engine produces, and the live calibration map was fitted
    # without it.
    removal_hazard: bool = field(
        default_factory=lambda: _env_bool("MLBE_REMOVAL_HAZARD", False)
    )

    # RBI overs are the one market where a conviction floor works: 20.5 of the
    # 21.5 units that market lost came from buys under 40% model probability
    # (11 bets under 30% alone lost 61.8% of stake), while everything above the
    # floor was roughly flat. All of them were plus-money o0.5 tickets -- cheap
    # lottery lines the EV screen liked precisely because the payout was long.
    rbi_min_buy_prob: float = field(
        default_factory=lambda: _env_float("MLBE_RBI_MIN_BUY_PROB", 0.40)
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

    # Weight given to the devigged market price when forming the probability the
    # EV screen bets on (see market.ev.anchor_to_market). The model's own
    # probability is untouched, so PPV/NPV and the calibration refit still
    # measure the model. Because the screen is affine in the probability, a weight
    # w is equivalent to demanding edge >= threshold / (1 - w): it raises the toll
    # on disagreeing with the market rather than making the engine defer to it.
    # Default 0 (off). Nine retro-priced slates: ROI -5.4% at 0, -4.1% at 0.4,
    # -3.5% at 0.6 on a third as many bets, -12.9% at 0.8 -- every interval still
    # spans zero, so this shrinks a loss rather than earning a profit. Judge a
    # weight on closing line value, which resolves in far fewer bets than ROI.
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
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MLBE_DATA_DIR", str(Path.home() / ".mlb_engine")))
    )

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def odds_cache_dir(self) -> Path:
        return self.cache_dir / "oddsapi"

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
    def team_form_path(self) -> Path:
        """Cached daily-built season team-form baseline (luck-gap inputs)."""
        return self.cache_dir / "team_form.json"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.cache_dir,
            self.output_dir,
            self.audit_dir,
            self.fangraphs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    cfg = Config()
    cfg.ensure_dirs()
    return cfg
