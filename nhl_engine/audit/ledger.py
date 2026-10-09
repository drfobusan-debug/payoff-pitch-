"""Prediction rows, the write-once ledger, and grading with CLV (master plan §7).

``predictions_<date>.json`` is written once. Later passes (goalie rerun,
pre-drop) write ``cards/card_<date>_<tag>.json`` so the history of what the
engine said is kept, but the row that gets graded is the first one: a card
cannot be improved after the fact. ``--force`` writes a new *versioned* file
beside it rather than overwriting.

Grading joins each row to the official ``GameResult``, settles it under its
own ``ot_rule``, records the dual-rule flag for totals, and computes CLV as
the change in devigged consensus between pricing and the last archived
quote of the day (positive = the close moved our way). The price paid is not
in it, so the hold does not read as the market moving against us.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import date as Date
from pathlib import Path

from nhl_engine.audit.grade import dual_rule, pnl, settle
from nhl_engine.data.capture import QuoteRow, pregame
from nhl_engine.market.board import Selection, selections
from nhl_engine.market.pricing import Priced
from nhl_engine.schemas import GameResult

log = logging.getLogger("nhl_engine")


@dataclass
class LedgerRow:
    slate_date: str
    matchup: str
    home: str
    away: str
    market: str
    side: str
    entity: str
    line: float | None
    ot_rule: str
    book: str
    american: float
    books: int
    consensus: float
    model_prob: float
    push_prob: float
    edge: float
    ev: float
    tier: str
    gates: list[str] = field(default_factory=list)
    pass_gate: bool = False
    kelly: float = 0.0
    home_goalie: str = ""
    away_goalie: str = ""
    goalie_status: str = ""
    lineup_source: str = ""
    priced_at: str = ""
    card_tag: str = ""
    reasons: list[str] = field(default_factory=list)
    # grading
    outcome: str | None = None
    pnl: float | None = None
    close_american: float | None = None
    close_consensus: float | None = None
    clv: float | None = None
    dual_rule_differs: bool | None = None
    graded_at: str = ""
    # Probability edge/EV were read from (market-anchored); None = model_prob.
    bet_prob: float | None = None

    @property
    def is_buy(self) -> bool:
        return self.pass_gate and self.tier != "Pass"

    @property
    def key(self) -> tuple[str, str, str, str, float | None]:
        return (self.matchup, self.market, self.side, self.entity, self.line)


def row_from(
    p: Priced,
    *,
    slate_date: Date,
    home: str,
    away: str,
    home_goalie: str,
    away_goalie: str,
    goalie_status: str,
    lineup_source: str,
    priced_at: str,
    card_tag: str,
) -> LedgerRow:
    return LedgerRow(
        slate_date=slate_date.isoformat(),
        matchup=p.sel.matchup,
        home=home,
        away=away,
        market=p.sel.market,
        side=p.sel.side,
        entity=p.sel.entity,
        line=p.sel.line,
        ot_rule=p.ot_rule,
        book=p.sel.best_book,
        american=p.sel.best_american,
        books=p.sel.books,
        consensus=p.sel.consensus,
        model_prob=p.sim_prob if p.sim_prob is not None else p.model.win,
        push_prob=p.model.push,
        edge=p.edge,
        ev=p.ev,
        tier=p.tier.value,
        gates=list(p.gates),
        pass_gate=p.pass_gate,
        kelly=p.kelly(),
        home_goalie=home_goalie,
        away_goalie=away_goalie,
        goalie_status=goalie_status,
        lineup_source=lineup_source,
        priced_at=priced_at,
        card_tag=card_tag,
        reasons=list(p.reasons),
        bet_prob=p.model.win,
    )


_FIELDS = {f.name for f in fields(LedgerRow)}


def save_rows(rows: list[LedgerRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=1) + "\n", encoding="utf-8")
    return path


def load_rows(path: Path) -> list[LedgerRow]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[LedgerRow] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            out.append(LedgerRow(**{k: v for k, v in item.items() if k in _FIELDS}))
    return out


def predictions_path(data_dir: Path, slate: Date) -> Path:
    return data_dir / "ledger" / f"predictions_{slate.isoformat()}.json"


def graded_path(data_dir: Path, slate: Date) -> Path:
    return data_dir / "ledger" / f"graded_{slate.isoformat()}.json"


def unregradable(data_dir: Path) -> list[Date]:
    """Nights with a graded file but no predictions file, so no audit can regrade them."""
    nights = (
        Date.fromisoformat(p.stem.removeprefix("graded_"))
        for p in sorted((data_dir / "ledger").glob("graded_*.json"))
    )
    return [d for d in nights if not predictions_path(data_dir, d).exists()]


def card_path(data_dir: Path, slate: Date, tag: str) -> Path:
    return data_dir / "cards" / f"card_{slate.isoformat()}_{tag}.json"


def write_once(rows: list[LedgerRow], path: Path, *, force: bool = False) -> tuple[Path, bool]:
    """Write the ledger unless it exists. ``force`` writes ``<stem>.v<N>.json`` instead."""
    if not path.exists():
        return save_rows(rows, path), True
    if not force:
        return path, False
    n = 2
    while (alt := path.with_name(f"{path.stem}.v{n}{path.suffix}")).exists():
        n += 1
    return save_rows(rows, alt), True


def close_consensus(
    board: list[Selection],
) -> dict[tuple[str, str, str, str, float | None], Selection]:
    return {s.key: s for s in board}


def grade_rows(
    rows: list[LedgerRow],
    results: dict[str, GameResult],
    day_quotes: list[QuoteRow],
    *,
    graded_at: str,
    starts: Mapping[str, str] | None = None,
) -> list[LedgerRow]:
    """Settle every row with a final result; rows without one are left ungraded.

    With ``starts`` (matchup -> puck drop UTC) the close is the last quote before
    puck drop, so in-play boards never stand in for the closing line.
    """
    if starts is not None:
        day_quotes = pregame(day_quotes, starts)
    close = close_consensus(selections(day_quotes)) if day_quotes else {}
    out: list[LedgerRow] = []
    for r in rows:
        res = results.get(r.matchup)
        if res is None or not res.is_final:
            out.append(r)
            continue
        outcome = settle(
            market=r.market, side=r.side, entity=r.entity, line=r.line, ot_rule=r.ot_rule, res=res
        )
        r.outcome = outcome
        r.pnl = pnl(outcome, r.american)
        r.dual_rule_differs = dual_rule(
            market=r.market, side=r.side, entity=r.entity, line=r.line, res=res
        )
        c = close.get(r.key)
        if c is not None:
            r.close_american = c.best_american
            r.close_consensus = c.consensus
            r.clv = c.consensus - r.consensus
        r.graded_at = graded_at
        out.append(r)
    return out


__all__ = [
    "LedgerRow",
    "card_path",
    "grade_rows",
    "graded_path",
    "load_rows",
    "predictions_path",
    "row_from",
    "unregradable",
    "save_rows",
    "write_once",
]
