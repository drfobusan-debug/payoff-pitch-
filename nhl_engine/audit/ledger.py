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

Each graded row also carries its ``puck_drop`` and the ``close_at`` capture
time, so the audit can drop rows priced after puck drop and count CLV only
where the close was taken within ``STALE_CLOSE_MIN`` of puck drop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import date as Date
from pathlib import Path

from nhl_engine.audit.grade import PPPLookup, dual_rule, pnl, settle, settle_ppp
from nhl_engine.data.capture import QuoteRow, parse_utc, pregame
from nhl_engine.market.board import Selection, selections
from nhl_engine.market.pricing import Priced
from nhl_engine.schemas import GameResult

log = logging.getLogger("nhl_engine")

STALE_CLOSE_MIN = 45.0  # a close older than this before puck drop is not a close


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
    puck_drop: str = ""  # UTC, from the NHL schedule at grading
    close_at: str = ""  # capture time of the quote the close was read from
    # The unanchored sim's own stamp on anchored markets (graded-only).
    sim_tier: str = ""
    sim_edge: float | None = None
    sim_buy: bool = False
    # The starters' GSAx/60 as priced, for the graded-only goalie lean in the bands.
    home_goalie_gsax60: float | None = None
    away_goalie_gsax60: float | None = None

    @property
    def is_buy(self) -> bool:
        return self.pass_gate and self.tier != "Pass"

    @property
    def priced_late(self) -> bool:
        """Priced at or after puck drop: an in-play quote, not a prediction."""
        drop, priced = parse_utc(self.puck_drop), parse_utc(self.priced_at)
        return drop is not None and priced is not None and priced >= drop

    @property
    def close_lead_min(self) -> float | None:
        """Minutes from the close quote to puck drop; None when either is unknown."""
        drop, seen = parse_utc(self.puck_drop), parse_utc(self.close_at)
        if drop is None or seen is None:
            return None
        return (drop - seen).total_seconds() / 60.0

    @property
    def close_fresh(self) -> bool:
        lead = self.close_lead_min
        return lead is not None and 0 <= lead <= STALE_CLOSE_MIN

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
    home_goalie_gsax60: float | None = None,
    away_goalie_gsax60: float | None = None,
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
        sim_tier=p.sim_tier,
        sim_edge=p.sim_edge,
        sim_buy=p.sim_buy,
        home_goalie_gsax60=home_goalie_gsax60,
        away_goalie_gsax60=away_goalie_gsax60,
    )


_FIELDS = {f.name for f in fields(LedgerRow)}


def save_rows(rows: list[LedgerRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=1) + "\n", encoding="utf-8")
    return path


def save_outcomes(rows: list[LedgerRow], path: Path) -> Path:
    """Write back only ``outcome``/``pnl``; every other key in the file stays exactly as written."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != len(rows):
        raise ValueError(f"{path}: rows do not line up with the file")
    for item, r in zip(raw, rows, strict=True):
        item["outcome"], item["pnl"] = r.outcome, r.pnl
    path.write_text(json.dumps(raw, indent=1) + "\n", encoding="utf-8")
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
    """Write each game's rows once: an existing ledger keeps its games and gains new ones.

    ``force`` writes the whole card to ``<stem>.v<N>.json`` instead.
    """
    if not path.exists():
        return save_rows(rows, path), True
    if not force:
        kept = load_rows(path)
        have = {r.matchup for r in kept}
        new = [r for r in rows if r.matchup not in have]
        if not new:
            return path, False
        return save_rows(kept + new, path), True
    n = 2
    while (alt := path.with_name(f"{path.stem}.v{n}{path.suffix}")).exists():
        n += 1
    return save_rows(rows, alt), True


def close_consensus(
    board: list[Selection],
) -> dict[tuple[str, str, str, str, float | None], Selection]:
    return {s.key: s for s in board}


def stamp_timing(
    rows: list[LedgerRow], starts: Mapping[str, str], day_quotes: list[QuoteRow]
) -> None:
    """Set ``puck_drop`` and ``close_at`` (last pre-drop capture of the row's selection)."""
    seen: dict[tuple[str, str, str, str, float | None], str] = {}
    for q in pregame(day_quotes, starts):
        k = (q.matchup, q.market, q.side, q.entity, q.line)
        if q.captured_at > seen.get(k, ""):
            seen[k] = q.captured_at
    for r in rows:
        r.puck_drop = starts.get(r.matchup, r.puck_drop)
        if r.clv is not None:
            r.close_at = seen.get(r.key, r.close_at)


def backfill_ppp(
    rows: list[LedgerRow], results: Mapping[str, GameResult], ppp_of: PPPLookup
) -> int:
    """Settle ``sk_ppp`` rows a past audit left ungraded; every other field stays as written."""
    done = 0
    for r in rows:
        res = results.get(r.matchup)
        if r.market != "sk_ppp" or r.outcome is not None or res is None or not res.is_final:
            continue
        outcome = settle_ppp(side=r.side, entity=r.entity, line=r.line, res=res, ppp_of=ppp_of)
        if outcome is not None:
            r.outcome, r.pnl = outcome, pnl(outcome, r.american)
            done += 1
    return done


def grade_rows(
    rows: list[LedgerRow],
    results: dict[str, GameResult],
    day_quotes: list[QuoteRow],
    *,
    graded_at: str,
    starts: Mapping[str, str] | None = None,
    ppp_of: PPPLookup | None = None,
) -> list[LedgerRow]:
    """Settle every row with a final result; rows without one are left ungraded.

    ``ppp_of`` (player_id, nhl_game_id -> PPP) settles ``sk_ppp`` off the game log.

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
        if outcome is None and r.market == "sk_ppp" and ppp_of is not None:
            outcome = settle_ppp(side=r.side, entity=r.entity, line=r.line, res=res, ppp_of=ppp_of)
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
    if starts is not None:
        stamp_timing(out, starts, day_quotes)
    return out


__all__ = [
    "STALE_CLOSE_MIN",
    "LedgerRow",
    "card_path",
    "grade_rows",
    "graded_path",
    "load_rows",
    "predictions_path",
    "row_from",
    "unregradable",
    "save_rows",
    "stamp_timing",
    "write_once",
]
