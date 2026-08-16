"""Post-model selection gates for moneyline buys (sharp money, bullpen depletion).

Backtesting the graded ``game_ml`` props showed that the model's own EV/edge is
the *wrong* way round -- higher-EV moneyline buys won less often (EV AUC 0.33,
p=0.004 over 102 graded rows), because the engine picks the highest-EV side and
that is systematically the losing dog. The metric that actually separates
winning moneyline buys from losers is the VSIN betting split: the share of the
**handle** (money) on the bet side, and especially handle% minus bets% -- the
classic *sharp money* indicator (AUC 0.80, p=0.027 on the buys).

This gate demotes a ``game_ml`` BUY to Pass unless the money split confirms the
side (handle% at least keeps pace with ticket%), which in effect stops the
engine from buying purely on an inverted EV signal. It is a post-model
selection gate (never changes a probability), env-tunable with a kill-switch,
and stays neutral when no VSIN split is available so it cannot punish games we
have no public-money read on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# A moneyline buy is kept only when handle% - bets% >= this threshold, i.e. the
# money is at least keeping pace with the tickets on our side (sharp agreement).
# Winning buys averaged +19.7 here, losers -2.6; 0.0 is the neutral default and
# is meant to be tuned against a longer graded window (see MLBE_ML_* env knobs).
DEFAULT_MIN_DIVERGENCE = 0.0

# Positive sharp signal: a side the engine *passed* on (EV too low) is promoted
# to a buy when the money strongly backs it. Backtesting the passes showed sides
# with handle% - bets% >= +5 won 62% (n=32) vs the engine's own EV-driven buys at
# 30% -- the market's money was a better predictor than the model's EV. The price
# guard stops us from upgrading into heavy chalk at a bad number: skip when the
# no-vig implied probability already exceeds DEFAULT_MAX_FAIR_PROB.
DEFAULT_UPGRADE_DIVERGENCE = 5.0
DEFAULT_MAX_FAIR_PROB = 0.65


# --- bullpen-depletion gate ------------------------------------------------
# The simulator hands the ball to a *generic* pen built from three weeks of
# late-inning relief rates, so it prices bullpen **skill** and never bullpen
# **availability**: a team whose high-leverage arms worked three days running
# still gets its season-quality pen in every simulated 7th-9th. The 0-100
# StatsAPI fatigue proxy already exists on the slate (it feeds the run-line PPV
# gate and the comeback flags) but never reached the moneyline, which is the
# market most exposed to it -- the full-game ML is decided in exactly the innings
# the depleted arms cover, while F5 markets never see them.
#
# Only a *relative* depletion is actionable: a gassed pen matters when the
# opponent's is fresher, so both pens at 70 is a wash the sim's neutral pen
# already approximates.
#
# That reasoning was never scored, and it does not survive being scored. The
# proxy was rebuilt per team-game over 3,956 team-games across two windows and
# read against the game it was about to be spent in: the sides this gate would
# have demoted won .529 and .507 against those same teams' own rates of .493
# and .506 -- if anything *better* than usual -- and r(fatigue, win) is -0.005
# and +0.000. The mechanism is missing upstream too: over 970 team-games the
# proxy does not predict the relief wOBA the pen goes on to allow (r = -0.035;
# ``scripts/pen_read_study.py``). So the fatigue branch is off by default. It
# describes who threw yesterday, and it was quietly spending moneyline buys on
# a coin flip.
#
# The Rotowire branch is a different signal -- arms declared unavailable rather
# than pitch counts inferred to be tired -- and stays on, though there is no
# history of that feed on this box to grade it against either.
DEFAULT_PEN_FATIGUE = False
DEFAULT_PEN_DEPLETED = 60.0  # same 0-100 threshold the run-line gate calls depleted
DEFAULT_PEN_EDGE = 15.0  # fatigue points our pen must trail by before demoting
DEFAULT_MIN_AVAILABILITY = 0.25  # 0..1 rested share (Rotowire feed, when live)


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val not in ("0", "false", "False", "")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class MLSharpGate:
    """Config + logic for the moneyline sharp-money confirmation gate."""

    enabled: bool = True
    min_divergence: float = DEFAULT_MIN_DIVERGENCE
    upgrade_enabled: bool = True
    upgrade_divergence: float = DEFAULT_UPGRADE_DIVERGENCE
    max_fair_prob: float = DEFAULT_MAX_FAIR_PROB

    @classmethod
    def from_env(cls) -> MLSharpGate:
        return cls(
            enabled=_env_flag("MLBE_ML_SHARP_GATE", True),
            min_divergence=_env_float("MLBE_ML_MIN_DIVERGENCE", DEFAULT_MIN_DIVERGENCE),
            upgrade_enabled=_env_flag("MLBE_ML_SHARP_UPGRADE", True),
            upgrade_divergence=_env_float(
                "MLBE_ML_UPGRADE_DIVERGENCE", DEFAULT_UPGRADE_DIVERGENCE
            ),
            max_fair_prob=_env_float("MLBE_ML_UPGRADE_MAX_FAIR", DEFAULT_MAX_FAIR_PROB),
        )

    def allows(
        self,
        handle_pct: float | None,
        bets_pct: float | None,
    ) -> tuple[bool, str]:
        """Return (keep_buy, reason).

        ``keep_buy`` is False only when the gate is enabled, a VSIN split is
        available, and the handle-minus-bets divergence on our side falls below
        the confirmation threshold.
        """
        if not self.enabled:
            return True, ""
        if handle_pct is None or bets_pct is None:
            return True, "ml-gate: neutral (no handle/bets split)"
        div = handle_pct - bets_pct
        if div < self.min_divergence:
            return False, (
                f"ml-gate: PASS (handle-bets {div:+.0f} < {self.min_divergence:+.0f}; "
                f"no sharp confirmation)"
            )
        return True, f"ml-gate: OK (handle-bets {div:+.0f})"

    def upgrades(
        self,
        handle_pct: float | None,
        bets_pct: float | None,
        fair_prob: float | None,
    ) -> tuple[bool, str]:
        """Return (promote_to_buy, reason) for a side the engine passed on.

        Promotes only when the gate + upgrade are enabled, a VSIN split shows
        the money strongly on our side (handle - bets >= ``upgrade_divergence``),
        and the price is not heavy chalk (no-vig implied <= ``max_fair_prob``).
        """
        if not (self.enabled and self.upgrade_enabled):
            return False, ""
        if handle_pct is None or bets_pct is None:
            return False, ""
        div = handle_pct - bets_pct
        if div < self.upgrade_divergence:
            return False, ""
        if fair_prob is not None and fair_prob > self.max_fair_prob:
            return False, (
                f"ml-upgrade: skip chalk (fair {fair_prob:.2f} > {self.max_fair_prob:.2f})"
            )
        return True, f"ml-upgrade: BUY (sharp handle-bets {div:+.0f})"


@dataclass(frozen=True)
class MLPenGate:
    """Demote a full-game moneyline buy whose own bullpen cannot cover the late innings.

    A post-model selection gate (it never touches a probability), applied to
    ``game_ml`` only. Two branches, and only one of them is on: the Rotowire
    availability read, and the pitch-count fatigue proxy, which was measured
    against 3,956 team-games, predicted nothing, and now needs
    ``MLBE_ML_PEN_FATIGUE=1`` to demote anything.
    """

    enabled: bool = True
    fatigue_enabled: bool = DEFAULT_PEN_FATIGUE
    depleted: float = DEFAULT_PEN_DEPLETED
    min_edge: float = DEFAULT_PEN_EDGE
    min_availability: float = DEFAULT_MIN_AVAILABILITY

    @classmethod
    def from_env(cls) -> MLPenGate:
        return cls(
            enabled=_env_flag("MLBE_ML_PEN_GATE", True),
            fatigue_enabled=_env_flag("MLBE_ML_PEN_FATIGUE", DEFAULT_PEN_FATIGUE),
            depleted=_env_float("MLBE_ML_PEN_DEPLETED", DEFAULT_PEN_DEPLETED),
            min_edge=_env_float("MLBE_ML_PEN_EDGE", DEFAULT_PEN_EDGE),
            min_availability=_env_float(
                "MLBE_ML_PEN_MIN_AVAIL", DEFAULT_MIN_AVAILABILITY
            ),
        )

    def allows(
        self,
        own_fatigue: float | None,
        opp_fatigue: float | None,
        own_availability: float | None = None,
    ) -> tuple[bool, str]:
        """Return (keep_buy, reason) for the side this recommendation backs.

        ``own_fatigue`` / ``opp_fatigue`` are the 0-100 StatsAPI depletion proxy
        (higher = less high-leverage depth left). ``own_availability`` is the
        Rotowire 0..1 rested share, which overrides the proxy when the feed is
        live because it reads actual arm-by-arm unavailability rather than
        inferring it from pitch counts.
        """
        if not self.enabled:
            return True, ""
        if own_availability is not None and own_availability <= self.min_availability:
            return False, (
                f"ml-pen: PASS (pen availability {own_availability:.2f} "
                f"<= {self.min_availability:.2f})"
            )
        if not self.fatigue_enabled:
            return True, "ml-pen: OK (workload proxy not scored against results)"
        if own_fatigue is None:
            return True, "ml-pen: neutral (no bullpen workload read)"
        if own_fatigue < self.depleted:
            return True, f"ml-pen: OK (fatigue {own_fatigue:.0f})"
        if opp_fatigue is not None and (own_fatigue - opp_fatigue) < self.min_edge:
            return True, (
                f"ml-pen: OK (both pens worked: {own_fatigue:.0f} vs {opp_fatigue:.0f})"
            )
        opp_txt = "unknown" if opp_fatigue is None else f"{opp_fatigue:.0f}"
        return False, (
            f"ml-pen: PASS (own pen depleted {own_fatigue:.0f} vs opp {opp_txt}; "
            "sim prices pen skill, not availability)"
        )
