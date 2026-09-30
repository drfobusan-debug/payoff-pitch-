"""A market snapshot keyed so that a moved line still matches.

Both market snapshots this engine keeps -- the first board it sees for a slate
and the last one before kickoff -- record the same three things per side: the
best price, the no-vig probability, and **the handicap that price was for**.

The handicap is the part that was missing, and its absence was not cosmetic.
:mod:`cfb_engine.audit.clv` filed closing quotes under the full selection string
(``"ALA -7.0"``), so a spread that closed at -7.5 was stored under a key no bet
could ever look up, and ``compute_clv`` returned ``None``. Closing line value was
therefore recorded only for the bets whose number had *not* moved -- which is the
subset with no line movement to measure, and the opposite of the sample worth
having. Snapshots here are keyed on ``"<matchup>|<market>|<side>"`` with the line
carried in the value, so the lookup succeeds and the movement becomes the answer
instead of being the reason there isn't one.

The matchup belongs in the key for the same reason the handicap does not. A totals
side is called ``"Over"`` in every game on the board, so dropping the number from
``"Over 54.5"`` without naming the game would file every over on a Saturday under
one key, one game's total silently becoming the baseline for all of them.

Two snapshots, two merge rules, and they are deliberately opposites:

* the **first-seen board** keeps the earliest capture, because it is the baseline
  a day's movement is measured from;
* the **close** keeps the latest, because a Saturday runs from noon to past
  midnight and any single capture sees one kickoff window's close alongside the
  long-stale opener of every later game.

"First seen" is honest and "opening" would not be: it is the first board *this
engine* pulled for the slate, which on a Saturday-morning run is hours or days
after the market actually opened. It is a baseline, not the true open.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cfb_engine.market import keys
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote, evaluate
from cfb_engine.schemas import Slate


@dataclass(frozen=True)
class SideQuote:
    """Best price for one side of one market, with the number it was priced at."""

    american: float
    no_vig_prob: float
    line: float | None = None


def _quote(quotes: list[MarketQuote], line: float | None) -> SideQuote | None:
    if not quotes:
        return None
    res = evaluate(0.5, quotes)
    return SideQuote(res.best_quote.american, res.fair_prob, line)


def key(matchup: str, market: str, selection: str) -> str:
    """The line-independent snapshot key for one side of one game's market."""
    return f"{matchup}|{market}|{keys.side_of(selection)}"


def board_quotes(slate: Slate, board: dict[str, GameOdds]) -> dict[str, SideQuote]:
    """Snapshot every priced side of the main lines, keyed line-independently."""
    out: dict[str, SideQuote] = {}
    for game in slate.games:
        odds = board.get(game.matchup())
        if odds is None:
            continue
        matchup = game.matchup()
        home, away = game.home.abbrev, game.away.abbrev
        for ab in (home, away):
            sq = _quote(odds.ml.get(ab, []), None)
            if sq is not None:
                out[key(matchup, "game_ml", keys.game_ml(ab))] = sq
        point = odds.main_spread()
        if point is not None and point in odds.spreads:
            for ab, pt in ((home, point), (away, -point)):
                sq = _quote(odds.spreads[point].get(ab, []), pt)
                if sq is not None:
                    out[key(matchup, "game_ats", keys.game_ats(ab, pt))] = sq
        line = odds.main_total()
        if line is not None and line in odds.totals:
            for is_over, side in ((True, "over"), (False, "under")):
                sq = _quote(odds.totals[line].get(side, []), line)
                if sq is not None:
                    out[key(matchup, "game_total", keys.game_total(is_over, line))] = sq
    return out


def merge_first_wins(
    existing: dict[str, SideQuote], fresh: dict[str, SideQuote]
) -> dict[str, SideQuote]:
    """Baseline merge: a side already captured keeps its earliest quote."""
    return {**fresh, **existing}


def merge_last_wins(
    existing: dict[str, SideQuote], fresh: dict[str, SideQuote]
) -> dict[str, SideQuote]:
    """Closing merge: the latest capture wins, but nothing captured is dropped."""
    return {**existing, **fresh}


def save(quotes: dict[str, SideQuote], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        k: {"american": q.american, "no_vig_prob": q.no_vig_prob, "line": q.line}
        for k, q in quotes.items()
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load(path: Path) -> dict[str, SideQuote]:
    """Read a snapshot, tolerating the two-part ``"<market>|<selection>"`` keys of
    files written before this schema, so an existing audit directory keeps working.

    A legacy key keeps its own shape, having no matchup to normalise onto;
    :func:`cfb_engine.audit.clv.compute_clv` looks it up as a fallback.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, SideQuote] = {}
    for k, v in raw.items():
        parts = k.split("|")
        line = v.get("line")
        if line is None:
            line = _line_in(parts[-1])
        out[k if len(parts) < 3 else key(parts[0], parts[1], parts[2])] = SideQuote(
            v["american"], v["no_vig_prob"], line
        )
    return out


def _line_in(selection: str) -> float | None:
    """The handicap a legacy key carried in its selection: ``"ALA -7.5"`` -> -7.5."""
    try:
        return float(selection.rpartition(" ")[2])
    except ValueError:
        return None
