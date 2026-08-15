"""Turn a game's situational context into point adjustments.

Two outputs per game: a **margin** delta (home-team perspective, feeding
moneyline and ATS) and a **total** delta (feeding over/under), plus a
human-readable reason for each nudge so the article and ledger can explain
why the model departed from raw ratings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cfb_engine.config import FeatureParams
from cfb_engine.features.context import GameContext


@dataclass
class Adjustment:
    margin_delta: float = 0.0  # home perspective (+ helps home)
    total_delta: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def _add_margin(self, pts: float, reason: str) -> None:
        if abs(pts) >= 0.05:
            self.margin_delta += pts
            self.reasons.append(reason)

    def _add_total(self, pts: float, reason: str) -> None:
        if abs(pts) >= 0.05:
            self.total_delta += pts
            self.reasons.append(reason)

    def _add_total_or_note(self, pts: float, reason: str) -> None:
        """Price the nudge, or -- when it is switched off -- just name it.

        A condition the model observes but does not charge for still belongs on
        the card: the reader can see it, and the ledger records what the layer
        would have done if a graded season ever justifies turning it on.
        """
        if abs(pts) >= 0.05:
            self._add_total(pts, reason)
        else:
            self.reasons.append(f"{reason} [observed, not scored]")


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def compute_adjustment(
    ctx: GameContext, params: FeatureParams, base_hfa: float, home_ab: str, away_ab: str
) -> Adjustment:
    """Situational margin/total deltas for one game.

    ``base_hfa`` is the model's default home-field points; at a neutral site it
    is swapped out for ``params.neutral_site_hfa`` via a corrective delta so the
    caller does not need to special-case the mean.
    """
    adj = Adjustment()
    if not params.enabled:
        return adj

    # -- home field: neutralize HFA at neutral sites ----------------------
    if ctx.neutral_site:
        adj._add_margin(params.neutral_site_hfa - base_hfa, "Neutral site (HFA removed)")

    # -- rest / fatigue ---------------------------------------------------
    if ctx.rest_home is not None and ctx.rest_away is not None:
        net = ctx.rest_home - ctx.rest_away
        pts = _clamp(net * params.rest_pts_per_day, params.rest_max_pts)
        if pts:
            side = home_ab if net > 0 else away_ab
            adj._add_margin(pts, f"Rest edge {side} ({ctx.rest_home}d vs {ctx.rest_away}d)")
        if ctx.rest_home >= params.bye_days:
            adj._add_margin(params.bye_bonus_pts, f"{home_ab} off a bye")
        if ctx.rest_away >= params.bye_days:
            adj._add_margin(-params.bye_bonus_pts, f"{away_ab} off a bye")

    # -- travel: away team's road miles -----------------------------------
    if ctx.travel_away_miles is not None and ctx.travel_away_miles >= params.travel_min_miles:
        pts = _clamp(
            (ctx.travel_away_miles / 1000.0) * params.travel_pts_per_1000mi,
            params.travel_max_pts,
        )
        adj._add_margin(pts, f"{away_ab} travels {ctx.travel_away_miles:.0f} mi")

    # -- weather: totals only, skipped indoors ----------------------------
    # The point values default to zero (see FeatureParams), so conditions are
    # named on the card without moving the total. ``_add_total`` drops a zero
    # nudge and its reason with it, hence the explicit report.
    if not ctx.dome:
        if ctx.wind_mph is not None and ctx.wind_mph > params.wind_threshold_mph:
            over = ctx.wind_mph - params.wind_threshold_mph
            cut = min(over * params.wind_total_per_mph, params.wind_total_max)
            adj._add_total_or_note(-cut, f"Wind {ctx.wind_mph:.0f} mph")
        if ctx.precipitation is not None and ctx.precipitation > 0:
            adj._add_total_or_note(-params.precip_total_pts, "Precipitation")
        if ctx.temperature_f is not None and ctx.temperature_f < params.cold_threshold_f:
            adj._add_total_or_note(-params.cold_total_pts, f"Cold {ctx.temperature_f:.0f}F")

    return adj
