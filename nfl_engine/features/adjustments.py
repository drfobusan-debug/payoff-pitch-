"""Situational deltas, in points, and the ones that measured as nothing.

Every term here was fitted against the **closing line's residual** over 1999-2025
-- not against the outcome. A term that predicts the score but is already in the
line is worth nothing to a bet, and most of the famous ones are exactly that.

What survived (n=5,431 graded games, 2006-2025 for the schedule terms):

    wind on the total          -0.33 pts/mph   t -3.33   priced at -0.11
    divisional game, total     -1.06 pts       t -2.82

What did not, and is therefore carried at zero weight:

    rest differential, margin  +0.06 pts/day   t +0.87
    short week (home / away)   -0.38 / -0.49   t -0.47 / -0.61
    post-bye (home / away)     +0.03 / -0.69   t +0.05 / -0.99
    neutral site, margin       -0.50           t -0.35
    cold (< 35F), margin       +1.33           t +1.85
    cold, total                +0.81           t +1.12
    backup QB, margin          +0.63           t +1.84
    wind, margin               -0.02 pts/mph   t -0.62

The rest and travel results are the ones worth stating plainly, because they are
the most written-about angles in football: **the market prices them already.** A
Thursday game after a Sunday is not a secret, and neither is a bye. The engine
measures them at zero rather than asserting the folk number, because a small
unfounded delta on a market whose residual is 13.2 points is a phantom edge, and
that is the mistake that cost the MLB engine five terms this month.

The QB term is the uncomfortable one at t = +1.84: fading a backup is *nearly*
significant on the margin, but it covers only 50.7% of the time (n=899), which is
below the 52.4% a -110 price needs. It is reported, not priced.

Wind is the one real finding, and it needs one caveat
----------------------------------------------------
games.csv's own wind column is a report that is **missing for 46% of 2022**, and
the missingness is not random, so the historical fit is run on ERA5 reanalysis
instead -- every hour of every season at the venue, from Open-Meteo's free
archive. On that source the market moves the total only -0.11 pts/mph while the
score moves -0.33, and the gap does not close in any era.

But a bettor has a *forecast*, not a reanalysis. A day-ahead Open-Meteo forecast
correlates r=+0.72 with the reported game-time wind, so the term must be applied
to the forecast and will attenuate accordingly. That is why
``WIND_TOTAL_PTS_PER_MPH`` is deliberately set to the *market-relative* slope
rather than the raw one, and why the engine's own total still starts from the
market's number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Points off the total per mph, *relative to the closing total* -- the residual
# slope, so it is already net of the -0.11 pts/mph the market itself prices.
# The long history with temperature, divisional and the total's own level
# controlled gives -0.21 (t -5.20, n=3,751); ERA5 over 2021-2025 gives -0.33
# (t -3.33, n=920). The shallower of the two ships: a term fitted on one
# five-season window is the one to distrust.
WIND_TOTAL_PTS_PER_MPH = -0.20
# A threshold fits no better than a straight line (|t| is flat from a knee of 0
# to 6), so there is no knee: the effect is linear in the wind from zero.
# The tail is thin, though -- only 118 games above 20 mph in 27 seasons.
WIND_MAX_TOTAL_POINTS = -5.0
# The deltas are *centred*: an average outdoor game gets nothing, because the
# market's total already contains the average game. Applied uncentred, the slope
# would quietly take 1.6 points off every total in the league.
WIND_MEAN_MPH = 8.5
DIV_GAME_TOTAL_POINTS = -1.06
DIV_GAME_SHARE = 0.376

INDOOR_ROOFS = frozenset({"dome", "closed"})


@dataclass(frozen=True)
class Situation:
    """Everything about a game that is not the two teams."""

    roof: str | None = None
    wind_mph: float | None = None
    temp_f: float | None = None
    home_rest: int | None = None
    away_rest: int | None = None
    neutral_site: bool = False
    # ``None`` means unknown rather than false: an unknown divisional flag must
    # not silently push the total the way "not divisional" does.
    div_game: bool | None = None

    def indoors(self) -> bool:
        return (self.roof or "").strip().lower() in INDOOR_ROOFS


@dataclass(frozen=True)
class Adjustment:
    """Points to add to the ratings-implied total and margin, and why."""

    total_points: float = 0.0
    margin_points: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return "; ".join(self.notes) if self.notes else "no situational adjustment"


def wind_total_delta(situation: Situation) -> float:
    """Points off the total for wind, against an average outdoor game.

    Indoors is zero because the roof settles it, and a calm outdoor game gets a
    small *positive* delta -- the average total already assumes some wind.
    """
    if situation.indoors() or situation.wind_mph is None:
        return 0.0
    excess = max(float(situation.wind_mph), 0.0) - WIND_MEAN_MPH
    return max(excess * WIND_TOTAL_PTS_PER_MPH, WIND_MAX_TOTAL_POINTS)


def div_total_delta(situation: Situation) -> float:
    """Divisional games run under; an unknown flag is worth nothing either way."""
    if situation.div_game is None:
        return 0.0
    flag = 1.0 if situation.div_game else 0.0
    return DIV_GAME_TOTAL_POINTS * (flag - DIV_GAME_SHARE)


def adjust(situation: Situation) -> Adjustment:
    """The situational block: two terms with evidence, and nothing else."""
    total = 0.0
    notes: list[str] = []
    wind = wind_total_delta(situation)
    if wind != 0.0 and situation.wind_mph is not None:
        total += wind
        notes.append(f"wind {situation.wind_mph:.0f}mph {wind:+.1f} total")
    div = div_total_delta(situation)
    if div != 0.0:
        total += div
        label = "divisional" if situation.div_game else "non-divisional"
        notes.append(f"{label} {div:+.1f} total")
    return Adjustment(total_points=total, margin_points=0.0, notes=tuple(notes))


def unpriced_notes(situation: Situation) -> tuple[str, ...]:
    """The situations that get *reported* because they measured as nothing.

    Kept visible so a card can say "short week, no adjustment" rather than
    leaving the reader to assume the engine missed it.
    """
    notes: list[str] = []
    if situation.home_rest is not None and situation.home_rest <= 5:
        notes.append("home on a short week (measured t=-0.47, not priced)")
    if situation.away_rest is not None and situation.away_rest <= 5:
        notes.append("away on a short week (measured t=-0.61, not priced)")
    if situation.home_rest is not None and situation.home_rest >= 11:
        notes.append("home off a bye (measured t=+0.05, not priced)")
    if situation.away_rest is not None and situation.away_rest >= 11:
        notes.append("away off a bye (measured t=-0.99, not priced)")
    if situation.neutral_site:
        notes.append("neutral site (measured t=-0.35, not priced)")
    if (
        situation.temp_f is not None
        and situation.temp_f < 35.0
        and not situation.indoors()
    ):
        notes.append("cold (measured t=+1.85 on margin, not priced)")
    return tuple(notes)
