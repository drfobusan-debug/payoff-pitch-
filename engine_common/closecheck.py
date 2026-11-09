"""Does the devigged close mean what it says, price range by price range?

Close-implied ROI (``promotion.close_roi``) trusts the closing no-vig probability.
That trust has to be checked against results separately in each price range,
because a devig can be honest at -110 and still flatter +600 long shots. For
every graded buy with a timely close this groups rows by the closing no-vig
probability of our side and prints what the close said against what happened.
Graded-only: it never votes in the stake bar and never touches a price.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from engine_common.promotion import close_roi

BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("long shot  < 25%", 0.0, 0.25),
    ("dog     25-40%", 0.25, 0.40),
    ("pick    40-60%", 0.40, 0.60),
    ("fav     60-75%", 0.60, 0.75),
    ("chalk   >= 75%", 0.75, 1.01),
)


@dataclass(frozen=True)
class Graded:
    close_prob: float  # closing no-vig prob of the side we took
    american: float  # the price we took
    won: bool  # pushes are left out before this point
    pnl: float  # realized units at flat 1u


@dataclass(frozen=True)
class BucketRead:
    label: str
    n: int
    close_prob: float  # mean closing no-vig prob
    hit: float  # realized win rate
    gap: float  # hit - close_prob, probability points
    gap_se: float
    close_roi: float  # mean close-implied ROI of the prices taken
    roi: float  # realized ROI

    @property
    def verdict(self) -> str:
        if self.n < 2:
            return "-"
        z = self.gap / self.gap_se if self.gap_se > 0 else 0.0
        if abs(z) >= 2:
            return "close too high" if z < 0 else "close too low"
        return "fits"


def read(rows: Iterable[Graded]) -> list[BucketRead]:
    xs = list(rows)
    out = []
    for label, lo, hi in BUCKETS:
        rs = [r for r in xs if lo <= r.close_prob < hi]
        if not rs:
            continue
        n = len(rs)
        p = sum(r.close_prob for r in rs) / n
        hit = sum(r.won for r in rs) / n
        se = math.sqrt(sum((float(r.won) - hit) ** 2 for r in rs) / (n - 1) / n) if n > 1 else 0.0
        out.append(
            BucketRead(
                label,
                n,
                round(p, 4),
                round(hit, 4),
                round(hit - p, 4),
                round(se, 4),
                round(sum(close_roi(r.close_prob, r.american) for r in rs) / n, 4),
                round(sum(r.pnl for r in rs) / n, 4),
            )
        )
    return out


TITLE = "Close vs results by price range (graded buys with a timely close; does the devig hold?)"
HEAD = ("close range", "n", "close prob", "hit rate", "gap ± SE", "close ROI", "ROI", "verdict")


def row(b: BucketRead) -> tuple[str, ...]:
    return (
        b.label,
        str(b.n),
        f"{b.close_prob:.1%}",
        f"{b.hit:.1%}",
        f"{100 * b.gap:+.1f} ± {100 * b.gap_se:.1f}",
        f"{b.close_roi:+.1%}",
        f"{b.roi:+.1%}",
        b.verdict,
    )


__all__ = ["BUCKETS", "HEAD", "TITLE", "BucketRead", "Graded", "read", "row"]
