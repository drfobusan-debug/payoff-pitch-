"""Price forecasts from the recorded history.

Holt's linear (double exponential) smoothing on log price, fitted to the daily
market snapshots the ledger has collected. Log space keeps the trend a
percentage, so a $2 card and a $200 box get the same treatment, and the
forecast can never cross zero.

The interval is the residual spread scaled by sqrt(horizon): honest about the
fact that a few weeks of daily snapshots say little about next year. With
fewer than ``MIN_POINTS`` observations the forecast is flat at the last price
and says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

MIN_POINTS = 7


@dataclass
class Forecast:
    last: float
    horizon_days: int
    point: float
    low: float
    high: float
    daily_trend: float  # fractional change per day
    n: int

    @property
    def change(self) -> float:
        return self.point / self.last - 1

    @property
    def flat(self) -> bool:
        return self.n < MIN_POINTS

    @property
    def direction(self) -> str:
        if self.flat or abs(self.change) < 0.03:
            return "flat"
        return "up" if self.change > 0 else "down"


def _daily_series(rows: Sequence[tuple[str, float]]) -> list[float]:
    """Collapse snapshots to one price per calendar day (the last seen)."""
    by_day: dict[str, float] = {}
    for stamp, price in rows:
        if price and price > 0:
            by_day[stamp[:10]] = price
    return [by_day[d] for d in sorted(by_day)]


def holt(
    series: Sequence[float], alpha: float = 0.5, beta: float = 0.1
) -> tuple[float, float, float]:
    """Return (level, trend, residual_sd) of Holt's linear method on log(series)."""
    y = [math.log(v) for v in series]
    level, trend = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    resid: list[float] = []
    for v in y[1:]:
        pred = level + trend
        resid.append(v - pred)
        new_level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    sd = (sum(r * r for r in resid) / len(resid)) ** 0.5 if resid else 0.0
    return level, trend, sd


def forecast(rows: Sequence[tuple[str, float]], horizon_days: int = 30) -> Forecast | None:
    series = _daily_series(rows)
    if not series:
        return None
    last = series[-1]
    if len(series) < MIN_POINTS:
        return Forecast(last, horizon_days, last, last, last, 0.0, len(series))
    level, trend, sd = holt(series)
    # Damp the trend so a hot week does not extrapolate into a hot year.
    phi = 0.98
    damped = sum(phi**k for k in range(1, horizon_days + 1))
    point = math.exp(level + trend * damped)
    band = 1.64 * sd * math.sqrt(horizon_days)
    return Forecast(
        last=last,
        horizon_days=horizon_days,
        point=point,
        low=math.exp(level + trend * damped - band),
        high=math.exp(level + trend * damped + band),
        daily_trend=math.exp(trend) - 1,
        n=len(series),
    )


def days_between(a: str, b: str) -> int:
    fa = datetime.fromisoformat(a[:10])
    fb = datetime.fromisoformat(b[:10])
    return (fb - fa).days
