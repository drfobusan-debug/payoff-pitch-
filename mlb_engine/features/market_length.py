"""Starter length read off the book's pitcher-outs line.

The workload cap (``features.workload``) guesses a starter's hook from his own
recent outings. Graded against 973 starts from 07-19 to 09-30 2026, that guess
missed actual outs by 4.04 RMSE; the mean implied by the book's two-sided outs
line missed by 3.65, and a blend fitted on the first half put 0.95 of the
weight on the book. Six-start median pitch count, converted to outs at the
pitcher's own pitches-per-batter and batters-per-out, missed by 3.90: it beats
the workload cap but adds nothing once the book's line is known (coefficient
0.10 +/- 0.07 on its gap to the book), so it is the hook the starter is given
when the book has not posted a line.

The length enters the simulator as a batters-faced hook drawn per simulation
around the target, so the book's read on the hook flows into every line that
depends on it: his K/H/BB/ER, the plate appearances each hitter takes against
him rather than the bullpen, and the bullpen innings in the game total.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from statistics import NormalDist, median

import pandas as pd

from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote

# Spread of a starter's outs around the book's mean, implied by the 600
# pitcher-days quoted at two or more lines (median; IQR 2.0-5.8).
OUTS_SD = 3.26
# League rates from the 2026 starts with 9+ batters faced.
LEAGUE_BF_PER_OUT = 1.436
LEAGUE_PITCHES_PER_BF = 3.877
# Starts of the pitcher's own work behind the median pitch count: RMSE on the
# next start falls to 13.85 pitches at six and is flat beyond it.
PITCH_COUNT_STARTS = 6
# Prior weight, in starts, holding a rate toward the league before it is his.
RATE_PRIOR_STARTS = 3
_MIN_PROB = 0.03
_N01 = NormalDist()
_OUTS_SEL = re.compile(r"^(?P<name>.*) Outs o(?P<line>\d+(?:\.\d+)?)$")

Quotes = Mapping[tuple[str, str, str], Sequence[MarketQuote]]


@dataclass(frozen=True)
class StarterLength:
    """The hook a starter is simulated with, and where it came from."""

    bf_cap: int
    bf_sd: float
    pitch_cap: int
    source: str  # "market" | "pitch_count"
    market_outs: float | None = None


def market_outs_lines(quotes: Quotes, matchup: str, pitcher: str) -> list[tuple[float, float]]:
    """``(line, no-vig over probability)`` per outs line with a two-sided price.

    One-way quotes carry half the hold in their implied probability, so they
    are dropped rather than read as a fair price; books at the same line are
    pooled by their median.
    """
    want = keys.canonical(pitcher)
    by_line: dict[float, list[float]] = {}
    for (m, market, selection), qs in quotes.items():
        if m != matchup or market != "pitcher_outs":
            continue
        hit = _OUTS_SEL.match(selection)
        if hit is None or keys.canonical(hit["name"]) != want:
            continue
        probs = [q.no_vig_prob for q in qs if q.opposite_american is not None]
        if probs:
            by_line.setdefault(float(hit["line"]), []).extend(probs)
    return sorted((line, median(ps)) for line, ps in by_line.items())


def market_outs_mean(lines: Sequence[tuple[float, float]], sd: float = OUTS_SD) -> float | None:
    """The mean outs a set of over prices implies, under a normal of spread ``sd``.

    Each line gives ``line + sd * z(p)``; lines nearer even money pin the mean
    tightest, so they are weighted by ``p * (1 - p)``. Near-certain prices
    (alt lines far from the mean) say little about it and are skipped.
    """
    num = den = 0.0
    for line, p in lines:
        if not _MIN_PROB < p < 1 - _MIN_PROB:
            continue
        w = p * (1 - p)
        num += w * (line + sd * _N01.inv_cdf(p))
        den += w
    return num / den if den else None


def _prior_starts(pit_rows: pd.DataFrame, as_of: Date, n: int) -> pd.DataFrame:
    """Per-start pitches, batters faced and outs for his last ``n`` outings."""
    if "game_date" not in pit_rows or "events" not in pit_rows:
        return pd.DataFrame(columns=["pitches", "bf", "outs"])
    rows = pit_rows[pit_rows["game_date"] < as_of]
    if rows.empty:
        return pd.DataFrame(columns=["pitches", "bf", "outs"])
    ev = rows["events"]
    outs = ev.map(_outs_on).fillna(0)
    per = pd.DataFrame(
        {"game_date": rows["game_date"], "pitch": 1, "bf": ev.notna().astype(int), "outs": outs}
    ).groupby("game_date").agg(pitches=("pitch", "sum"), bf=("bf", "sum"), outs=("outs", "sum"))
    return per.sort_index().tail(n)


_OUT_EVENTS = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "fielders_choice_out": 1,
    "sac_fly": 1, "sac_bunt": 1, "other_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "strikeout_double_play": 2,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
}


def _outs_on(event: object) -> int:
    return _OUT_EVENTS.get(event, 0) if isinstance(event, str) else 0


def starter_length(
    pit_rows: pd.DataFrame,
    as_of: Date,
    lines: Sequence[tuple[float, float]],
    *,
    weight: float,
    bf_buffer: float,
    bf_sd: float,
    pitch_buffer: int,
    max_bf: int,
) -> StarterLength | None:
    """The starter's simulated hook from the book's outs line and his pitch count.

    With a two-sided outs line, the batters-faced hook is centred on the outs
    the book implies, converted at his own batters per out and blended
    (``weight`` on the book) with his recent batters faced; the pitch hook then
    only ends an outing that runs long on pitches. Without a line he is hooked
    at his recent batters faced or his median pitch count over his last
    ``PITCH_COUNT_STARTS`` starts, whichever comes first. ``None`` when he has
    fewer than two starts to read, which leaves the workload cap in charge.
    """
    starts = _prior_starts(pit_rows, as_of, PITCH_COUNT_STARTS)
    if len(starts) < 2:
        return None
    recent_bf = float(starts["bf"].mean())
    prior_outs = RATE_PRIOR_STARTS * 15.0
    bf_per_out = (float(starts["bf"].sum()) + LEAGUE_BF_PER_OUT * prior_outs) / (
        float(starts["outs"].sum()) + prior_outs
    )
    pitch_cap = int(round(float(starts["pitches"].median()))) + pitch_buffer
    mkt = market_outs_mean(lines)
    if mkt is None:
        return StarterLength(
            bf_cap=int(min(max_bf, round(recent_bf + bf_buffer))),
            bf_sd=0.0,
            pitch_cap=pitch_cap,
            source="pitch_count",
        )
    target = weight * mkt * bf_per_out + (1.0 - weight) * recent_bf
    bf_cap = int(min(max_bf, round(target + bf_buffer)))
    return StarterLength(
        bf_cap=bf_cap,
        bf_sd=bf_sd,
        pitch_cap=max(pitch_cap, int(round(bf_cap * LEAGUE_PITCHES_PER_BF)) + pitch_buffer),
        source="market",
        market_outs=mkt,
    )
