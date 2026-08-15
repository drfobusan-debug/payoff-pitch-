"""Daily slate *preview* report — the reader-facing article that ships with a run.

Where the nightly audit grades the past, this report previews the day ahead. For
every game on the slate it tells the matchup story the pipeline already computed
and persisted as a :class:`~mlb_engine.preview.GamePreview`:

* each starting pitcher's stuff/command line vs. the lineup he faces,
* each bullpen's contact/command line vs. that same lineup,
* who is regressing positively (buy-low) or negatively (due to cool off),
* the shape of game the simulator expects — blowout vs. coin-flip, low- vs.
  high-run,
* park and weather context,
* the moneyline's market-implied probability, the model's probability, and the
  edge between them,
* and the engine's best bets for that game, in bold.

It renders a Morningstar-style HTML/PDF (same house style as the Audit Desk) plus
an energetic, sportscaster-cadence MP3 narration. It never re-runs the simulation
— everything comes from the persisted previews.

This is a model preview, not betting advice.
"""

from __future__ import annotations

import logging
import re
from datetime import date as Date
from pathlib import Path

import numpy as np

from mlb_engine.features.regression import BL_BABIP
from mlb_engine.features.trend import FLAT_CSW, FLAT_SIERA, FLAT_VFA
from mlb_engine.market.tiers import Tier
from mlb_engine.output.audit_insight import (
    GOLD,
    INK,
    MUTE,
    NAVY,
    RED,
    _fig_b64,
    market_label,
    to_mp3,
    to_pdf,
)
from mlb_engine.preview import BullpenLine, GamePreview, LineupLine, StarterLine
from mlb_engine.recommendations import Recommendation

logger = logging.getLogger(__name__)

_TIER_RANK = {Tier.STRONG.value: 0, Tier.MODERATE.value: 1}


# --- interpretation helpers ------------------------------------------------
def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"


def game_shape(gp: GamePreview) -> tuple[str, str]:
    """Classify the expected game into (headline, one-line description)."""
    if gp.total_mean >= 9.5:
        run_env = "high-scoring"
    elif gp.total_mean <= 7.5:
        run_env = "low-scoring"
    else:
        run_env = "average-run"

    if gp.p_blowout >= 0.34 and gp.p_blowout >= gp.p_close:
        margin = "blowout-leaning"
    elif gp.p_close >= 0.30:
        margin = "coin-flip"
    else:
        margin = "modest-margin"

    fav = gp.fav_team or (gp.home if gp.p_home_win >= 0.5 else gp.away)
    label = f"{run_env.capitalize()}, {margin}"
    desc = (
        f"Sim projects ~{gp.total_mean:.1f} total runs and a {abs(gp.xrd):.1f}-run "
        f"lean toward {fav}. Blowout odds {_pct(gp.p_blowout)}, one-run-or-tie "
        f"{_pct(gp.p_close)}."
    )
    return label, desc


def _edge_cls(edge: float | None) -> str:
    if edge is None:
        return ""
    return "pos" if edge >= 0 else "neg"


# --- charts ----------------------------------------------------------------
def _projection_rows(gp: GamePreview) -> list[tuple[str, float, float]] | None:
    """Each offense's projected wOBA vs this starter and vs a league-average arm."""
    rows = []
    for bats, lu, sl in (
        (gp.home, gp.home_lineup, gp.away_starter),
        (gp.away, gp.away_lineup, gp.home_starter),
    ):
        if lu.proj_woba is None or lu.proj_woba_vs_league is None:
            return None
        rows.append((f"{bats} bats — vs {sl.name}", lu.proj_woba, lu.proj_woba_vs_league))
    return rows


def _projection_chart(rows: list[tuple[str, float, float]]) -> str:
    """Bars of the projected matchup wOBA against the same order's neutral mark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    vs_arm = [r[1] for r in rows]
    vs_lg = [r[2] for r in rows]
    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 0.72 * len(labels) + 1.2))
    ax.barh(y - h / 2, vs_arm, height=h, color=NAVY, zorder=3, label="Projected vs this starter")
    ax.barh(y + h / 2, vs_lg, height=h, color=MUTE, zorder=3, label="Same order vs average arm")
    for yi, (a, lg) in enumerate(zip(vs_arm, vs_lg, strict=False)):
        ax.text(a + 0.004, yi - h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK)
        ax.text(lg + 0.004, yi + h / 2, f"{lg:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max([*vs_arm, *vs_lg]) * 1.18)
    ax.set_xlabel("Projected wOBA per plate appearance (log5 matchup)", fontsize=9, color=MUTE)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("What each order projects to do tonight", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


def _matchup_chart(gp: GamePreview) -> str:
    """The projected-wOBA bars, or the old xwOBA levels for older previews.

    In the fallback both league baselines are drawn, because those two bars
    measure different populations and can't be read against each other.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = _projection_rows(gp)
    if proj is not None:
        return _projection_chart(proj)
    rows = [
        (f"{gp.home} bats — vs {gp.away} SP", gp.home_lineup.xwoba, gp.away_starter.xwoba_allowed),
        (f"{gp.home} bats — vs {gp.away} pen", gp.home_lineup.xwoba, gp.away_pen.xwoba_allowed or 0.0),
        (f"{gp.away} bats — vs {gp.home} SP", gp.away_lineup.xwoba, gp.home_starter.xwoba_allowed),
        (f"{gp.away} bats — vs {gp.home} pen", gp.away_lineup.xwoba, gp.home_pen.xwoba_allowed or 0.0),
    ]
    labels = [r[0] for r in rows]
    bats = [r[1] for r in rows]
    arms = [r[2] for r in rows]
    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(labels) + 1.0))
    ax.barh(y - h / 2, bats, height=h, color=NAVY, zorder=3, label="Lineup xwOBA")
    ax.barh(y + h / 2, arms, height=h, color=GOLD, zorder=3, label="xwOBA allowed")
    for yi, (b, a) in enumerate(zip(bats, arms, strict=False)):
        ax.text(b + 0.004, yi - h / 2, f"{b:.3f}", va="center", fontsize=8, color=INK)
        ax.text(a + 0.004, yi + h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max([*bats, *arms, 0.35]) * 1.18)
    lg_bats = gp.home_lineup.league_xwoba
    lg_arms = gp.home_starter.league_xwoba_allowed
    if lg_bats is not None:
        ax.axvline(lg_bats, color=NAVY, ls=":", lw=1.2, zorder=4, label="League lineup")
    if lg_arms is not None:
        ax.axvline(lg_arms, color=GOLD, ls=":", lw=1.2, zorder=4, label="League pitching")
    ax.set_xlabel("xwOBA — read each bar against its own dotted league line", fontsize=9, color=MUTE)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Bats and the arms they face, vs. league", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


def _shape_chart(gp: GamePreview) -> str:
    """Horizontal bars for the projected game shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{gp.home} win", f"{gp.away} win", "One run or tie", "Blowout (4+)"]
    vals = [gp.p_home_win * 100, (1 - gp.p_home_win) * 100, gp.p_close * 100, gp.p_blowout * 100]
    colors = [NAVY, RED, GOLD, MUTE]
    fig, ax = plt.subplots(figsize=(7.6, 2.2))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for yi, v in enumerate(vals):
        ax.text(v + 1.0, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Simulated probability (%)", fontsize=9, color=MUTE)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Projected game shape", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


# --- HTML pieces -----------------------------------------------------------
def _starter_row(tag: str, sl: StarterLine) -> str:
    spin = "" if sl.spin is None else f", {sl.spin:.0f} rpm"
    hard = "—" if sl.hard_hit_allowed is None else f"{sl.hard_hit_allowed * 100:.0f}%"
    return (
        f"<tr><td class='l'><b>{tag}</b> {sl.name}</td>"
        f"<td>{sl.k_pct * 100:.0f}% (x{sl.xk_pct * 100:.0f})</td>"
        f"<td>{sl.bb_pct * 100:.0f}% (x{sl.xbb_pct * 100:.0f})</td>"
        f"<td>{sl.csw * 100:.0f}%</td><td>{sl.swstr * 100:.0f}%</td><td>{hard}</td>"
        f"<td>{sl.xwoba_allowed:.3f}</td><td>{sl.barrel_allowed * 100:.0f}%{spin}</td></tr>"
    )


def _trend_phrase(delta: float | None, flat: float, unit: str, *, lower_is_better: bool) -> str:
    """Describe a half-window change: direction, size, and whose favour it is.

    ``lower_is_better`` is from the pitcher's side, so a falling SIERA improves
    and falling velocity slips.
    """
    if delta is None:
        return "too thin to read"
    if abs(delta) < flat:
        return f"flat ({delta:+{unit}})"
    improving = (delta < 0) if lower_is_better else (delta > 0)
    word = "improving" if improving else "slipping"
    cls = "pos" if improving else "neg"
    return f"<span class='{cls}'>{word}</span> ({delta:+{unit}})"


def starter_trend_sentence(team: str, sl: StarterLine) -> str:
    """Prose for one starter's form direction and his contact-luck gap."""
    siera = "SIERA unavailable (thin sample)" if sl.siera is None else f"SIERA {sl.siera:.2f}"
    bits = [
        f"{siera}, {_trend_phrase(sl.siera_trend, FLAT_SIERA, '.2f', lower_is_better=True)}",
        f"stuff {_trend_phrase(sl.stuff_trend, FLAT_CSW, '.1%', lower_is_better=False)} on CSW%",
        f"velocity {_trend_phrase(sl.vfa_trend, FLAT_VFA, '.1f', lower_is_better=False)} mph",
    ]
    if sl.babip_allowed is None:
        luck = ""
    else:
        gap = sl.dxwoba * 1000
        woba_allowed = sl.xwoba_allowed - sl.dxwoba
        babip = "a lucky" if sl.babip_allowed < BL_BABIP - 0.020 else (
            "an unlucky" if sl.babip_allowed > BL_BABIP + 0.020 else "a normal"
        )
        if gap >= 15:
            read = (
                f"<span class='neg'>the hits are owed</span> — his contact deserved "
                f"{gap:.0f} points more damage than it did"
            )
        elif gap <= -15:
            read = (
                f"<span class='pos'>he's been hit harder than the contact deserved</span> "
                f"by {-gap:.0f} points, so the line should improve"
            )
        else:
            read = f"results match the contact ({gap:+.0f} points), nothing owed either way"
        luck = (
            f" He's allowed {woba_allowed:.3f} wOBA on {sl.xwoba_allowed:.3f} xwOBA with "
            f"{babip} {sl.babip_allowed:.3f} BABIP ({BL_BABIP:.3f} league): {read}."
        )
    return f"<p><b>{sl.name} ({team}).</b> " + "; ".join(bits) + f".{luck}</p>"


def _starter_trends(gp: GamePreview) -> str:
    return starter_trend_sentence(gp.home, gp.home_starter) + starter_trend_sentence(
        gp.away, gp.away_starter
    )


def _split_clause(lu: LineupLine) -> str:
    hand = {"R": "right-handers", "L": "left-handers"}.get(lu.vs_hand or "", "this hand")
    if lu.split_woba is None or lu.split_rank is None or lu.split_of is None:
        return f"has too thin a sample against {hand} to rank"
    return (
        f"hits {hand} at a {lu.split_woba:.3f} wOBA, "
        f"<b>{lu.split_rank} of {lu.split_of}</b> — the <b>{lu.split_bucket} third</b>"
    )


def _venue_clause(lu: LineupLine) -> str:
    if lu.home_woba is None or lu.away_woba is None or lu.is_home is None:
        return ""
    here, there = ("at home", "on the road") if lu.is_home else ("on the road", "at home")
    mine = lu.home_woba if lu.is_home else lu.away_woba
    theirs = lu.away_woba if lu.is_home else lu.home_woba
    diff = mine - theirs
    splits = f"{mine:.3f} wOBA {here} vs {theirs:.3f} {there}"
    if abs(diff) < 0.010:
        shape = f"which is no help either way — {splits}"
    elif diff > 0:
        shape = f"their better half — {splits}"
    else:
        shape = f"their weaker half — {splits}"
    return f" They're {here} tonight, {shape}."


# Points of projected wOBA below which neither side owns the half.
WASH_WOBA = 0.008
# Fallback scale: points of league-relative xwOBA (previews without projections).
WASH_XWOBA = 0.010


def matchup_gap(lu: LineupLine, sl: StarterLine) -> float:
    """Points of projected wOBA this arm costs the order, positive = bats ahead.

    The simulator already prices each hitter against this starter by log5 on the
    seven PA outcomes, with the hitter's platoon and home/road context inside it.
    ``proj_woba`` is that lineup average, and ``proj_woba_vs_league`` is the same
    order against a league-average arm, so their difference isolates the starter.
    Previews written before those fields existed fall back to comparing each
    side's xwOBA with its own league baseline.
    """
    if lu.proj_woba is not None and lu.proj_woba_vs_league is not None:
        return lu.proj_woba - lu.proj_woba_vs_league
    bats, arm = _league_relative(lu, sl)
    return bats - arm


def _league_relative(lu: LineupLine, sl: StarterLine) -> tuple[float, float]:
    """Each side's xwOBA against the league baseline for *its own* statistic.

    A lineup's xwOBA (mean over hitters) and a starter's xwOBA allowed (mean over
    his batted balls) sit on different scales, so comparing them raw would hand
    the bats every matchup. Both are centred first: positive means better than
    league at what that side does.
    """
    bats = lu.xwoba - lu.league_xwoba if lu.league_xwoba is not None else 0.0
    arm = sl.league_xwoba_allowed - sl.xwoba_allowed if sl.league_xwoba_allowed else 0.0
    return bats, arm


def edge_side(lu: LineupLine, sl: StarterLine) -> str:
    """``bats`` / ``arm`` / ``wash`` for one offense against one starter."""
    projected = lu.proj_woba is not None and lu.proj_woba_vs_league is not None
    gap = matchup_gap(lu, sl)
    if abs(gap) < (WASH_WOBA if projected else WASH_XWOBA):
        return "wash"
    return "bats" if gap > 0 else "arm"


def _vs_league(points: float) -> str:
    if abs(points) < 5:
        return "league-average"
    return f"{abs(points):.0f} points {'better' if points > 0 else 'worse'} than league"


def _gap_clause(bats_team: str, lu: LineupLine, sl: StarterLine) -> str:
    """The evidence behind the verdict, in the strongest form available."""
    if lu.proj_woba is not None and lu.proj_woba_vs_league is not None:
        gap = (lu.proj_woba - lu.proj_woba_vs_league) * 1000
        direction = "above" if gap > 0 else "below"
        return (
            f"the sim projects {bats_team}'s order at a {lu.proj_woba:.3f} wOBA against "
            f"{sl.name}, {abs(gap):.0f} points {direction} the {lu.proj_woba_vs_league:.3f} "
            "that same order projects against a league-average arm"
        )
    bats, arm = _league_relative(lu, sl)
    return (
        f"the lineup's {lu.xwoba:.3f} xwOBA is {_vs_league(bats * 1000)} for a batting order "
        f"and {sl.name}'s {sl.xwoba_allowed:.3f} allowed is {_vs_league(arm * 1000)} for a pitcher"
    )


def matchup_verdict(bats_team: str, arm_team: str, lu: LineupLine, sl: StarterLine) -> str:
    """One sentence on who wins a bats-vs-arm half, and why."""
    side = edge_side(lu, sl)
    if side == "wash":
        verdict = f"<b>Wash</b> — {bats_team}'s bats and {sl.name} price out even"
    elif side == "bats":
        verdict = f"<b>Edge: {bats_team}'s bats</b>"
    else:
        verdict = f"<b>Edge: {sl.name} ({arm_team})</b>"
    return (
        f"<p>{verdict}. Against an order that {_split_clause(lu)}, "
        f"{_gap_clause(bats_team, lu, sl)}.{_venue_clause(lu)}</p>"
    )


# Bullpen reader thresholds: workload proxy at/above which a pen is worked, and
# the per-arm wOBA spread that makes the choice of reliever matter.
FATIGUE_DEPLETED = 60.0
WIDE_ARM_SPREAD = 0.040
TIGHT_ARM_SPREAD = 0.020


def _better_arm(a: StarterLine, b: StarterLine) -> tuple[StarterLine, StarterLine, float] | None:
    """The better of two starters by SIERA, with the gap in runs."""
    if a.siera is None or b.siera is None:
        return None
    better, worse = (a, b) if a.siera <= b.siera else (b, a)
    return better, worse, worse.siera - better.siera  # type: ignore[operator]


def _margin_word(gap: float) -> str:
    if gap >= 1.25:
        return "by a wide margin"
    if gap >= 0.50:
        return "clearly"
    return "narrowly"


def starter_duel(gp: GamePreview) -> str:
    """Who is the better pitcher tonight, and where the two disagree."""
    home, away = gp.home_starter, gp.away_starter
    teams = {home.name: gp.home, away.name: gp.away}
    ranked = _better_arm(home, away)
    if ranked is None:
        lead = (
            f"<b>No SIERA read on both arms</b> — {home.name} and {away.name} are compared on "
            "contact and stuff only"
        )
    else:
        better, worse, gap = ranked
        lead = (
            f"<b>Better pitcher: {better.name} ({teams[better.name]})</b> {_margin_word(gap)} — "
            f"{better.siera:.2f} SIERA to {worse.name}'s {worse.siera:.2f}"
        )
    bits = []
    for tag, sl in ((gp.home, home), (gp.away, away)):
        hard = "" if sl.hard_hit_allowed is None else f", {sl.hard_hit_allowed * 100:.0f}% hard-hit"
        bits.append(
            f"{sl.name} ({tag}) misses bats at {sl.swstr * 100:.0f}% SwStr and allows "
            f"{sl.xwoba_allowed:.3f} xwOBA{hard}"
        )
    return f"<p>{lead}. " + "; ".join(bits) + ".</p>"


def _fresh_clause(bp: BullpenLine) -> str:
    """Rested or worked, from the workload proxy and the 3-day load.

    The proxy counts arms on back-to-back days or a heavy two-day pitch count at
    20 points each, so it reads back as a number of gassed relievers.
    """
    load = "" if bp.recent_load is None else f", three-day workload {bp.recent_load:.2f}× normal"
    if bp.fatigue is None:
        return "Workload unknown" + load
    gassed = round(bp.fatigue / 20)
    if bp.fatigue >= FATIGUE_DEPLETED:
        state = f"<span class='neg'>Worked</span> — {gassed} arms gassed"
    elif bp.fatigue > 0:
        state = f"About normal — {gassed} arm{'s' if gassed != 1 else ''} gassed"
    else:
        state = "<span class='pos'>Fresh</span> — nobody on back-to-back days or a heavy two-day count"
    return state + load


def _volatility_clause(bp: BullpenLine) -> str:
    if bp.arm_spread is None:
        return "too few arms with real work to judge how much the choice of reliever matters"
    if bp.arm_spread >= WIDE_ARM_SPREAD:
        read = "<span class='neg'>volatile</span> — which reliever appears matters more than the average"
    elif bp.arm_spread <= TIGHT_ARM_SPREAD:
        read = "<span class='pos'>uniform</span> — any arm out of it does about the same job"
    else:
        read = "normal spread between its best and worst arm"
    arms = "" if bp.arms is None else f" across {bp.arms} arms"
    return f"{read} ({bp.arm_spread:.3f} wOBA spread{arms})"


def bullpen_verdict(bats_team: str, pen_team: str, bp: BullpenLine) -> str:
    """One pen: how rested, how it projects against this order, how volatile."""
    if bp.proj_woba is None:
        proj = "no projection against this order (thin relief sample)"
    else:
        close = (
            ""
            if bp.proj_woba_close is None
            else f", {bp.proj_woba_close:.3f} once the 8th-inning arms take it"
        )
        proj = f"projects {bp.proj_woba:.3f} wOBA against {bats_team}'s order{close}"
    walk = ""
    if bp.zone_pct is not None and bp.zone_pct < 0.40:
        walk = f" It's also a walk trap at {bp.zone_pct * 100:.0f}% zone."
    return (
        f"<p><b>{pen_team}'s pen.</b> {_fresh_clause(bp)}; {proj}; "
        f"{_volatility_clause(bp)}.{walk}</p>"
    )


def _pen_edge(gp: GamePreview) -> str:
    """Which pen is the better bet to hold, given the order it must face."""
    home, away = gp.home_pen, gp.away_pen
    if home.proj_woba is None or away.proj_woba is None:
        return ""
    gap = (away.proj_woba - home.proj_woba) * 1000
    if abs(gap) < 8:
        return f"<p>Late-inning edge: <b>even</b> — both pens project within {abs(gap):.0f} points.</p>"
    better = gp.home if gap > 0 else gp.away
    return (
        f"<p>Late-inning edge: <b>{better}'s pen</b>, by {abs(gap):.0f} points of projected "
        "wOBA against the order it has to face.</p>"
    )


def _bullpens(gp: GamePreview) -> str:
    return (
        bullpen_verdict(gp.away, gp.home, gp.home_pen)
        + bullpen_verdict(gp.home, gp.away, gp.away_pen)
        + _pen_edge(gp)
    )


def _rank_bucket(rank: int, of: int) -> str:
    third = of / 3.0
    if rank <= third:
        return "top"
    return "middle" if rank <= 2 * third else "bottom"


def lineup_profile(team: str, lu: LineupLine) -> str:
    """How this offense hits in general, then how it hits in tonight's situation."""
    if lu.team_woba is None or lu.team_rank is None or lu.team_of is None:
        general = f"a {lu.woba:.3f} wOBA / {lu.xwoba:.3f} xwOBA batting order"
    else:
        bucket = _rank_bucket(lu.team_rank, lu.team_of)
        general = (
            f"a {lu.team_woba:.3f} wOBA club overall, <b>{lu.team_rank} of {lu.team_of}</b> "
            f"({bucket} third), hitting {lu.xwoba:.3f} xwOBA on contact"
        )
    situ = [_split_clause(lu)]
    if lu.is_home is not None and lu.home_woba is not None and lu.away_woba is not None:
        where = "at home" if lu.is_home else "on the road"
        mine = lu.home_woba if lu.is_home else lu.away_woba
        rank = (
            ""
            if lu.venue_rank is None or lu.venue_of is None
            else f", {lu.venue_rank} of {lu.venue_of} in that split"
        )
        other = lu.away_woba if lu.is_home else lu.home_woba
        swing = (mine - other) * 1000
        gap = f"{abs(swing):.0f} points {'better' if swing > 0 else 'worse'} than the {other:.3f}"
        situ.append(
            f"{where} they hit {mine:.3f}{rank}, {gap} they hit "
            f"{'on the road' if lu.is_home else 'at home'}"
        )
    return f"<p><b>{team}.</b> They are {general}. Tonight: " + "; ".join(situ) + ".</p>"


def _lineup_profiles(gp: GamePreview) -> str:
    return lineup_profile(gp.home, gp.home_lineup) + lineup_profile(gp.away, gp.away_lineup)


def _matchup_verdicts(gp: GamePreview) -> str:
    return matchup_verdict(gp.home, gp.away, gp.home_lineup, gp.away_starter) + matchup_verdict(
        gp.away, gp.home, gp.away_lineup, gp.home_starter
    )


def _reg_bits(gp: GamePreview) -> str:
    def side(team: str, lu) -> str:
        hot = ", ".join(f"{f.name} (+{f.points:.0f})" for f in lu.hot) or "—"
        cold = ", ".join(f"{f.name} (−{f.points:.0f})" for f in lu.cold) or "—"
        return (
            f"<p><b>{team}.</b> <span class='neg'>Due to cool off:</span> {hot}. "
            f"<span class='pos'>Buy-low / due to heat up:</span> {cold}.</p>"
        )

    return side(gp.home, gp.home_lineup) + side(gp.away, gp.away_lineup)


# "Aaron Judge HR o0.5" -> "Aaron Judge"; the side belongs in the sentence,
# not in the name, and it is no longer always the over.
_HR_SUFFIX = re.compile(r"\s+(?:HR\s+)?[ou]\d+(?:\.\d+)?$")


def top_hr_prop(hr_recs: list[Recommendation]) -> Recommendation | None:
    """The likeliest man in the game to homer.

    Only the over answers that question. Both sides of a prop are priced, and
    the under's probability is the complement, so ranking every side by model
    probability returns the *weakest* bat in the lineup at better than 95%.
    """
    priced = [r for r in hr_recs if r.model_prob is not None and r.side == "over"]
    return max(priced, key=lambda r: r.model_prob) if priced else None


def _hr_line(hr_recs: list[Recommendation]) -> str:
    best = top_hr_prop(hr_recs)
    if best is None:
        return "<p class='hr'><b>Top HR prop:</b> no home-run market priced for this game.</p>"
    odds = "" if best.market_american is None else f" ({best.market_american:+.0f})"
    name = _HR_SUFFIX.sub("", best.selection)
    return (
        f"<p class='hr'><b>Top HR prop:</b> {name}{odds} — model gives him "
        f"<b>{best.model_prob * 100:.1f}%</b> to go yard, the best shot in this game.</p>"
    )


def _best_bets_block(gp: GamePreview) -> str:
    if not gp.best_bets:
        return "<p><b>Best bets:</b> none clear the buy threshold — the model passes this game.</p>"
    items = ""
    for b in gp.best_bets:
        odds = "" if b.odds is None else f" ({b.odds:+.0f})"
        edge = "" if b.edge is None else f", edge {b.edge * 100:+.1f}%"
        items += (
            f"<li><b>{b.selection}{odds}</b> — {market_label(b.market)}, "
            f"model {b.model_prob * 100:.0f}%{edge} · <i>{b.tier}</i></li>"
        )
    return f"<p class='bets'><b>Best bets</b></p><ul class='bets'>{items}</ul>"


def _slate_best_bets_block(
    previews: list[GamePreview], recs: list[Recommendation]
) -> str:
    """Every buy across the slate, strongest first, bold, at the bottom.

    Built from the full ``recs`` (not ``GamePreview.best_bets``, which the
    pipeline truncates to the top four per game) so the count is the true number
    of Strong/Moderate plays and no qualifying bet is silently dropped.
    """
    labels = {gp.game_pk: f"{gp.away}@{gp.home}" for gp in previews}
    rows = [r for r in recs if r.tier in (Tier.STRONG, Tier.MODERATE)]
    rows.sort(key=lambda r: (_TIER_RANK.get(r.tier.value, 9), -(r.edge or 0.0)))
    if not rows:
        return (
            "<div class='slatebets'><h2>Slate best bets</h2>"
            "<p>The model passes the entire board today — no selection clears the buy threshold.</p></div>"
        )
    items = ""
    for r in rows:
        odds = "" if r.market_american is None else f" ({r.market_american:+.0f})"
        edge = "" if r.edge is None else f", edge {r.edge * 100:+.1f}%"
        where = f" ({labels[r.game_pk]})" if r.game_pk in labels else ""
        items += (
            f"<li><b>{r.selection}{odds}</b> — {market_label(r.market)}"
            f"{where}, model {r.model_prob * 100:.0f}%{edge} · <i>{r.tier.value}</i></li>"
        )
    return (
        "<div class='slatebets'><h2>Slate best bets</h2>"
        f"<p class='sbnote'>{len(rows)} plays clear the buy threshold, strongest first:</p>"
        f"<ul class='bets big'>{items}</ul></div>"
    )


def _game_section(gp: GamePreview, hr_recs: list[Recommendation]) -> str:
    shape_label, shape_desc = game_shape(gp)
    env_bits = []
    if gp.park_name:
        pf = "" if gp.park_factor is None else f" (park factor {gp.park_factor:.2f})"
        env_bits.append(f"{gp.park_name}{pf}")
    if gp.roof:
        env_bits.append(f"roof {gp.roof}")
    if gp.wx_summary:
        hr = "" if gp.wx_hr_mult is None else f", HR carry ×{gp.wx_hr_mult:.2f}"
        env_bits.append(f"{gp.wx_summary}{hr}")
    env = " · ".join(env_bits) if env_bits else "no park/weather data"

    ml = (
        f"<div class='ml'>Moneyline: <b>{gp.fav_team}</b> favored — market implies "
        f"<b>{_pct(gp.fav_implied)}</b>, model says "
        f"<b>{_pct(gp.home_ml_prob if gp.fav_side == 'home' else gp.away_ml_prob)}</b>, "
        f"edge <span class='{_edge_cls(gp.fav_edge)}'>{_pct(gp.fav_edge)}</span>.</div>"
    )

    starter_tbl = (
        "<table><tr><th class='l'>Starter</th><th>K% (x)</th><th>BB% (x)</th>"
        "<th>CSW%</th><th>SwStr%</th><th>Hard-hit%</th><th>xwOBA</th>"
        "<th>Barrel% / spin</th></tr>"
        + _starter_row(gp.home, gp.home_starter)
        + _starter_row(gp.away, gp.away_starter)
        + "</table>"
    )

    return (
        f"<div class='game'><h2>{gp.away} @ {gp.home}</h2>"
        f"<p class='env'>{env}</p>"
        f"<div class='shape'><span class='tag'>{shape_label}</span> {shape_desc}</div>"
        f"{ml}"
        f"<h3>Who's the better pitcher</h3>{starter_duel(gp)}{starter_tbl}"
        f"<h3>Who wins the batter-pitcher matchup</h3>{_matchup_verdicts(gp)}"
        f"<img class='chart' src='data:image/png;base64,{_matchup_chart(gp)}'/>"
        f"<h3>How these lineups hit — overall and tonight</h3>{_lineup_profiles(gp)}"
        f"<h3>Who's pitching well, and who's due to turn</h3>{_starter_trends(gp)}"
        f"<h3>Hitters due to cool off or heat up</h3>{_reg_bits(gp)}"
        f"<h3>The bullpens: rested, effective, volatile?</h3>{_bullpens(gp)}"
        f"<img class='chart' src='data:image/png;base64,{_shape_chart(gp)}'/>"
        f"{_hr_line(hr_recs)}"
        f"{_best_bets_block(gp)}"
        "</div>"
    )


CSS = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:4px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#16324f;}
h1{font-size:22pt;color:#16324f;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-size:15pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:20px 0 6px;}
h3{font-size:11pt;color:#16324f;margin:14px 0 4px;}
p{margin:6px 0;}
.lead{font-size:11pt;}
.game{page-break-inside:avoid;border-bottom:2px solid #eceef1;padding-bottom:10px;margin-bottom:6px;}
.env{color:#6b7280;font-style:italic;font-size:9.4pt;margin:2px 0;}
.shape{background:#eef2f6;border-left:4px solid #16324f;padding:6px 10px;margin:8px 0;font-size:9.8pt;}
.shape .tag{display:inline-block;background:#16324f;color:#fff;padding:1px 8px;border-radius:10px;font-size:8.4pt;font-family:'DejaVu Sans',sans-serif;margin-right:6px;}
.ml{font-size:10pt;margin:6px 0;}
table{border-collapse:collapse;width:100%;font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:8.6pt;margin:8px 0;}
th{background:#16324f;color:#fff;padding:5px 6px;text-align:center;font-weight:600;}
td{border-bottom:1px solid #e6e8ec;padding:5px 6px;text-align:center;}
tr:nth-child(even) td{background:#f5f6f8;}
td.l,th.l{text-align:left;}
.pos{color:#2e7d32;font-weight:bold;}.neg{color:#b23b3b;font-weight:bold;}
img.chart{width:100%;margin:6px 0 2px;}
p.bets{margin:10px 0 2px;font-size:11pt;color:#16324f;}
ul.bets{margin:2px 0 4px 0;font-size:10pt;}
ul.bets b{color:#111;}
.hr{background:#fff8e6;border-left:4px solid #c8a02e;padding:5px 10px;margin:8px 0;font-size:9.8pt;}
.slatebets{page-break-inside:avoid;background:#0f2438;color:#f4f6f8;border-radius:6px;padding:12px 16px;margin:22px 0 8px;}
.slatebets h2{color:#ffd76a;border:none;margin:0 0 4px;}
.slatebets .sbnote{color:#c6ccd4;font-style:italic;font-size:9.4pt;margin:0 0 6px;}
ul.bets.big{font-size:10.5pt;}
ul.bets.big b{color:#fff;}
.slatebets i{color:#ffd76a;}
.callout{background:#eef2f6;border-left:4px solid #16324f;padding:8px 12px;margin:10px 0;font-size:9.6pt;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""


def _hr_by_game(recs: list[Recommendation]) -> dict[int, list[Recommendation]]:
    out: dict[int, list[Recommendation]] = {}
    for r in recs:
        if r.market == "batter_hr":
            out.setdefault(r.game_pk, []).append(r)
    return out


def build_preview_report(
    day: Date, previews: list[GamePreview], recs: list[Recommendation] | None = None
) -> tuple[str, str]:
    hr_map = _hr_by_game(recs or [])
    nice = day.strftime("%A, %B %-d, %Y")
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Slate Preview</div>"
        "<h1>Today's Slate</h1>"
        "<p class='sub'>Arms vs. bats, who's regressing, the shape of the game, and where the edge is.</p>"
        f"<div class='dateline'>Slate previewed · {nice}</div></div>"
    )
    n_bets = sum(len(p.best_bets) for p in previews)
    lead = (
        f"Good morning — here's the {len(previews)}-game board for {nice.split(',')[0]}. For every matchup "
        "we call which side owns it — each lineup's expected offense against the arms it draws (starter first, "
        "then the bullpen it meets late), with where that offense ranks against the hand it faces and how it "
        "hits home versus away — read each starter's form direction on SIERA, stuff and velocity, "
        "flag the hitters the Statcast model says are running hot or cold, and let the simulator "
        "call the shape of the game. Then we put the moneyline's market-implied number next to the model's and "
        f"read the edge. The engine's flagged <b>{n_bets}</b> best bets across the slate — they're in bold under "
        "each game and gathered at the very bottom. Each game also gets its single most likely home-run prop. "
        "This is a model preview, not betting advice."
    )
    body = "".join(_game_section(gp, hr_map.get(gp.game_pk, [])) for gp in previews)
    body += _slate_best_bets_block(previews, recs or [])
    fine = (
        "<p class='fine'>Methodology: probabilities and run distribution come from the engine's Monte Carlo game "
        "simulation and F5 Markov model; xwOBA lines are trailing-window Statcast. Starter trends split the same "
        "six-week window in half and report the recent half minus the earlier one, for the three signals that "
        "repeat on three weeks of pitches (velocity, CSW%, SIERA); contact quality is excluded because it does "
        "not. A starter's BABIP-vs-xwOBA gap is the wOBA he allowed against the wOBA his contact "
        "deserved. Matchup verdicts are the simulator's own log5 projection — each hitter's rates in "
        "his platoon and home/road context against this starter's — averaged over the order and "
        "compared with the same order against a league-average arm. "
        "Platoon and home/road ranks are club-level wOBA over the trailing window, ranked among teams with at "
        "least 150 plate appearances in the split. Bullpen lines are the same log5 projection against "
        "the pen as a whole and against its 8th-inning arms; volatility is the standard deviation of "
        "wOBA allowed across individual relievers with 25+ batters faced, and freshness is the "
        "StatsAPI workload proxy alongside the three-day load. Regression flags are the gap "
        "between a hitter's actual and expected wOBA (points). Implied probability is the devig-free American-odds "
        "conversion of the best posted price; edge is model minus implied. Model preview, not investment advice.</p>"
    )
    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{body}{fine}</body></html>"
    )
    narr = _narration(day, previews, hr_map)
    return html, narr


def _narrate_bullpens(gp: GamePreview) -> str:
    """Spoken bullpen read: who's rested, who holds, who's volatile."""
    bits = []
    for pen_team, bp in ((gp.home, gp.home_pen), (gp.away, gp.away_pen)):
        if bp.fatigue is not None and bp.fatigue >= FATIGUE_DEPLETED:
            bits.append(f"The {pen_team} bullpen is worked")
        elif bp.arm_spread is not None and bp.arm_spread >= WIDE_ARM_SPREAD:
            bits.append(f"The {pen_team} bullpen is volatile arm to arm")
    home, away = gp.home_pen, gp.away_pen
    if home.proj_woba is not None and away.proj_woba is not None:
        gap = (away.proj_woba - home.proj_woba) * 1000
        if abs(gap) >= 8:
            better = gp.home if gap > 0 else gp.away
            bits.append(f"and late innings favor the {better} pen")
    return "" if not bits else ", ".join(bits) + ". "


def _narrate_arms(gp: GamePreview) -> str:
    """Spoken version of the better-pitcher verdict."""
    ranked = _better_arm(gp.home_starter, gp.away_starter)
    if ranked is None:
        return ""
    better, worse, gap = ranked
    team = gp.home if better is gp.home_starter else gp.away
    return (
        f"{better.name} is the better arm {_margin_word(gap)}, {better.siera:.2f} SIERA "
        f"for {team} against {worse.siera:.2f}. "
    )


def _narrate_matchup(gp: GamePreview) -> str:
    """Spoken version of the two bats-vs-arm verdicts and the split ranks."""
    out = [_narrate_arms(gp)]
    for bats, lu, sl in (
        (gp.home, gp.home_lineup, gp.away_starter),
        (gp.away, gp.away_lineup, gp.home_starter),
    ):
        side = edge_side(lu, sl)
        if side == "wash":
            who = f"{bats}'s bats and {sl.name} are a wash"
        elif side == "bats":
            who = f"{bats}'s bats have the edge on {sl.name}"
        else:
            who = f"{sl.name} has the edge on the {bats} bats"
        rank = ""
        if lu.split_rank is not None and lu.split_of is not None:
            hand = "lefties" if lu.vs_hand == "L" else "righties"
            rank = (
                f", and {bats} is {lu.split_rank} of {lu.split_of} against {hand}, "
                f"{lu.split_bucket} third"
            )
        out.append(f"{who}{rank}. ")
    return "".join(out) + _narrate_bullpens(gp)


def _narration(
    day: Date, previews: list[GamePreview], hr_map: dict[int, list[Recommendation]] | None = None
) -> str:
    hr_map = hr_map or {}
    nice = day.strftime("%A, %B %-d")
    parts = [
        f"What's up everybody, welcome into the Payoff Pitch Slate Preview for {nice}. "
        f"We got {len(previews)} games on the board, so let's run the card. ",
    ]
    for gp in previews:
        shape_label, _ = game_shape(gp)
        edge = gp.fav_edge
        edge_txt = ""
        if edge is not None and gp.fav_implied is not None:
            side = "value" if edge >= 0 else "no value"
            edge_txt = (
                f" The market implies {gp.fav_implied * 100:.0f} percent, the model says "
                f"{(gp.home_ml_prob if gp.fav_side == 'home' else gp.away_ml_prob) * 100:.0f}, "
                f"so there's {side} on {gp.fav_team}."
            )
        parts.append(
            f"{gp.away} at {gp.home}. The sim likes a {shape_label.lower()} game, "
            f"about {gp.total_mean:.1f} runs, leaning {gp.fav_team}.{edge_txt} "
        )
        parts.append(_narrate_matchup(gp))
        if gp.best_bets:
            b = gp.best_bets[0]
            odds = "" if b.odds is None else f" at {b.odds:+.0f}"
            parts.append(
                f"Best bet here: {b.selection}{odds}, {market_label(b.market)}, "
                f"model's got it at {b.model_prob * 100:.0f} percent. "
            )
        else:
            parts.append("No bet here, the model passes. ")
        hr_best = top_hr_prop(hr_map.get(gp.game_pk, []))
        if hr_best is not None:
            name = hr_best.selection.replace(" HR o0.5", "").replace(" o0.5", "")
            parts.append(
                f"If you want a longball, {name} is the top home-run shot here at "
                f"{hr_best.model_prob * 100:.0f} percent. "
            )
    strong = [
        (gp, b)
        for gp in previews
        for b in gp.best_bets
        if _TIER_RANK.get(b.tier, 9) == 0
    ]
    strong.sort(key=lambda t: -(t[1].edge or 0.0))
    if strong:
        parts.append("Alright, the headline plays of the day. ")
        for _gp, b in strong[:5]:
            odds = "" if b.odds is None else f" at {b.odds:+.0f}"
            parts.append(f"{b.selection}{odds}, {market_label(b.market)}. ")
    parts.append(
        "That's the slate. Bet the edges, skip the coin-flips, and we'll grade it all tomorrow. "
        "Payoff Pitch, out."
    )
    return "".join(parts)


# --- top-level entry point -------------------------------------------------
def generate_daily_preview(
    previews: list[GamePreview],
    slate_date: Date,
    cfg,
    *,
    email: bool,
    to: str | None,
    recs: list[Recommendation] | None = None,
    extra_attachments: list[tuple[str, bytes]] | None = None,
) -> dict[str, Path | None]:
    """Build the preview PDF + MP3 and optionally email them with the ledger."""
    out: dict[str, Path | None] = {"pdf": None, "mp3": None, "html": None}
    if not previews:
        logger.warning("no previews for %s; skipping slate preview report", slate_date)
        return out

    html, narr = build_preview_report(slate_date, previews, recs)
    iso = slate_date.isoformat()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.output_dir / f"slate_preview_{iso}.html"
    html_path.write_text(html)
    out["html"] = html_path

    attachments: list[tuple[str, bytes]] = list(extra_attachments or [])
    try:
        pdf_bytes = to_pdf(html)
        pdf_path = cfg.output_dir / f"PayoffPitch_Slate_{iso}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        out["pdf"] = pdf_path
        attachments.insert(0, (pdf_path.name, pdf_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slate preview PDF not written: %s", exc)

    try:
        mp3_path = cfg.output_dir / f"PayoffPitch_Slate_{iso}.mp3"
        mp3_bytes = to_mp3(narr, mp3_path)
        out["mp3"] = mp3_path
        attachments.append((mp3_path.name, mp3_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slate preview MP3 not written: %s", exc)

    print("Slate preview -> " + ", ".join(str(p) for p in (out["pdf"], out["mp3"]) if p))

    if email and attachments:
        from mlb_engine.output.email import EmailNotConfigured, send_card_email

        body_html = (
            "<div style='font-family:Georgia,serif;color:#1a1a1a;max-width:640px'>"
            "<h2 style='color:#16324f'>Payoff Pitch — Slate Preview</h2>"
            f"<p>Your preview for <b>{slate_date.strftime('%A, %B %-d, %Y')}</b> is attached:</p>"
            "<ul><li><b>Slate article (PDF)</b> — per-game arms-vs-bats, regression watch, game shape, "
            "weather, moneyline edge, and the best bets in bold.</li>"
            "<li><b>Audio narration (MP3)</b> — the same read, sportscaster style.</li>"
            "<li><b>Excel bet sheet</b> — every priced market (when attached).</li></ul>"
            "<p style='color:#6b7280;font-size:13px'>Model preview, not investment advice.</p></div>"
        )
        try:
            recipient = send_card_email(
                cfg,
                subject=f"Payoff Pitch — Slate Preview ({iso})",
                html_body=body_html,
                text_body="Your Payoff Pitch slate preview (PDF + audio) is attached.",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed slate preview ({len(attachments)} attachments) to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Slate preview email not sent: {exc}")

    return out
