"""THE BAT X's projection beside ours, on the card and in the ledger.

Opta's read (``data.opta``) reaches the card and stops there. BAT X goes one
step further and is written to the ledger row, because the head-to-head that
decides whether it belongs anywhere near a price is a *graded* question:
fitting

    logit(win) ~ a + b*logit(model) + c*logit(market) + d*logit(batx)

needs the three probabilities on the same graded row, and reconstructing the
third one a month later means keeping every daily export and re-joining by
name. Persisting it at pricing time makes the study a query.

What is deliberately absent: any path from this column into ``model_prob``,
``bet_prob``, the tiers or the screens. On the first four graded slates BAT X
scored +0.53 next to the price -- but only on the markets it prices from its
own feed, with an interval touching zero once whole player-dates are
resampled, and flat (-0.08) on the markets where the probability is our
distribution imposed on their mean. That is a second opinion worth reading,
not a forecast worth paying.

The file read here is what ``scripts/batx_study.py price`` writes:
``date, player, team, market, line, batx_prob``, where ``batx_prob`` is always
P(over).
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Trailing tokens the engine appends to a player's name in ``selection``.
_SEL_SUFFIX = re.compile(
    r"\s+(H\+R\+RBI|1B|2B|3B|HR|TB|H|R|RBI|Ks|Walks|Hits|ER|Outs)\s+[ou][\d.]+$"
)


@dataclass(frozen=True)
class BatxRow:
    """One projected prop: ``prob`` is P(over), as the feed prices it."""

    player: str
    market: str
    line: float
    prob: float


def _norm(name: str) -> str:
    plain = unicodedata.normalize("NFKD", str(name).casefold())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    plain = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", plain.strip())
    return re.sub(r"[^a-z0-9]+", "", plain)


def _key(market: str, player: str, line: float) -> str:
    return f"{market}|{_norm(player)}|{line:g}"


def player_from_selection(selection: str) -> str:
    return _SEL_SUFFIX.sub("", str(selection))


def load_rows(path: Path) -> list[BatxRow]:
    """Read a priced BAT X export, skipping rows that cannot be joined."""
    if not path.exists():
        return []
    rows: list[BatxRow] = []
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            for raw in csv.DictReader(fh):
                try:
                    rows.append(
                        BatxRow(
                            player=raw["player"],
                            market=raw["market"],
                            line=float(raw["line"]),
                            prob=float(raw["batx_prob"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        log.warning("could not read BAT X projections at %s", path, exc_info=True)
        return []
    return rows


def annotate(recs: list, rows: list[BatxRow]) -> int:
    """Stamp each recommendation with BAT X's probability *for our side*.

    The feed quotes P(over) and the ledger grades whichever side we bet, so an
    under has to be flipped here. Leaving it unflipped stores the complement of
    the forecast, which reads as BAT X disagreeing precisely when it agrees --
    and every counting prop the engine fades is an under.
    """
    by_key: dict[str, BatxRow] = {}
    for row in rows:
        by_key.setdefault(_key(row.market, row.player, row.line), row)

    hits = 0
    for rec in recs:
        if rec.line is None or rec.side not in ("over", "under"):
            continue
        match = by_key.get(_key(rec.market, player_from_selection(rec.selection), rec.line))
        if match is None:
            continue
        rec.batx_prob = round(1.0 - match.prob if rec.side == "under" else match.prob, 6)
        hits += 1
    return hits
