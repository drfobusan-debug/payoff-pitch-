"""CFB fast-start study: does the score at Q1/half/Q3 stay in the final?

The pregame number is the timely close (:func:`cfb_engine.audit.clv.timely_close`) on the
Over side of ``game_total``; the quarters are CFBD's line scores, the final their sum
(overtime included, as the total settles). Graded only: nothing here prices or buys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

from cfb_engine.audit.clv import ClosingQuote, timely_close
from cfb_engine.data.teamnames import label_matches, norm
from engine_runtime import faststart

CFB_CUTS = {"end of Q1": 1, "half": 2, "end of Q3": 3}


@dataclass(frozen=True)
class LineScore:
    home: str
    away: str
    start: str
    home_q: tuple[int, ...]
    away_q: tuple[int, ...]


def _game(away: str, home: str, day: str, games: Sequence[LineScore]) -> LineScore | None:
    near = {(Date.fromisoformat(day) + timedelta(days=k)).isoformat() for k in (-1, 0, 1)}
    pool = [g for g in games if g.start[:10] in near]
    pair = frozenset({norm(away), norm(home)})
    exact = [g for g in pool if frozenset({norm(g.home), norm(g.away)}) == pair]
    if len(exact) == 1:
        return exact[0]
    found = [
        g
        for g in pool
        if (label_matches(home, g.home) and label_matches(away, g.away))
        or (label_matches(home, g.away) and label_matches(away, g.home))
    ]
    return found[0] if len(found) == 1 else None


def reads(
    closings: Mapping[str, Mapping[str, ClosingQuote]], games: Sequence[LineScore]
) -> list[faststart.Read]:
    """``closings`` is slate date -> that slate's closing snapshot."""
    by_cut: dict[str, list[faststart.Obs]] = {c: [] for c in CFB_CUTS}
    for day, closing in closings.items():
        for key, q in closing.items():
            matchup, market, side = key.split("|")
            if market != "game_total" or side != "Over" or q.line is None or not timely_close(q):
                continue
            away, _, home = matchup.partition(" @ ")
            g = _game(away, home, day, games)
            if g is None or len(g.home_q) != len(g.away_q) or len(g.home_q) < 4:
                continue
            periods = [h + a for h, a in zip(g.home_q, g.away_q, strict=True)]
            for cut, o in faststart.checkpoints(
                f"{day} {matchup}", day, q.line, periods, float(sum(periods)), CFB_CUTS
            ).items():
                by_cut[cut].append(o)
    return [r for c, obs in by_cut.items() if (r := faststart.read(c, obs)) is not None]
