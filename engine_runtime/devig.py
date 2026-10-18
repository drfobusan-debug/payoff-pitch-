"""Power de-vig and the execution-book-excluded consensus median.

The runtime's own copy of the power method (``nfl_engine.market.fair`` is the
reference): every implied probability is raised to the common exponent that
makes the market sum to one. Proportional de-vig manufactures a
favourite-longshot slope; power does not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from statistics import median

from engine_common.odds import american_to_prob

MIN_HOLD = 1e-9


def hold(implied: Sequence[float]) -> float:
    return sum(implied) - 1.0


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-10) -> float | None:
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def devig_proportional(implied: Sequence[float]) -> list[float]:
    total = sum(implied)
    return list(implied) if total <= 0 else [p / total for p in implied]


def devig_power(implied: Sequence[float]) -> list[float]:
    if abs(hold(implied)) < MIN_HOLD:
        return list(implied)
    if min(implied) <= 0.0:
        return devig_proportional(implied)

    def excess(k: float) -> float:
        return sum(p**k for p in implied) - 1.0

    k = _bisect(excess, 0.2, 5.0)
    if k is None:
        return devig_proportional(implied)
    out = [p**k for p in implied]
    total = sum(out)
    return [p / total for p in out]


def novig(american: float, opposite: float) -> float:
    """Power no-vig probability of ``american`` against its paired ``opposite``."""
    return devig_power([american_to_prob(american), american_to_prob(opposite)])[0]


def consensus(
    quotes: Iterable[tuple[str, float, float | None]],
    *,
    exclude_books: Sequence[str] = (),
) -> tuple[float | None, int, int]:
    """Median no-vig probability across paired books, execution books excluded.

    ``quotes`` are ``(book, american, opposite_american)``. Returns
    ``(fair, books_seen, books_used)``; ``fair`` is ``None`` when no paired
    non-execution book is quoting (the ``thin_market`` input).
    """
    seen = 0
    fair: list[float] = []
    for book, american, opposite in quotes:
        seen += 1
        if opposite is None or book in exclude_books:
            continue
        fair.append(novig(american, opposite))
    return (float(median(fair)) if fair else None, seen, len(fair))


__all__ = ["consensus", "devig_power", "devig_proportional", "hold", "novig"]
