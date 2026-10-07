"""Every priced NBA selection, recorded at pricing and graded the next morning.

Plan §7 and §10. Four rules carried over from the other engines:

**Refused rows are rows.** A selection the gates refuse is written with the
gates that refused it, so each gate's false negatives are a query (``gates``).

**The price of record is set before tip.** A pass writes its own immutable
file under ``ledger/<date>/``. The graded row for a position is its earliest
buy, or, if no pass bought it, the latest pass that priced it: both choices
use only what was known before tip, never the result.

**The close is the last pre-tip capture, at the book that priced the row.**
Quotes captured at or after tip are dropped. CLV is the closing no-vig
probability at that book minus the no-vig probability of the price taken;
``clv_ev`` is the EV of the price taken under the closing consensus fair.

**A pulled prop keeps its row.** It is graded on the result (ROI, PPV/NPV,
calibration) with ``close_status = pulled`` (or ``moved`` when the book
re-posted it at another line), and its CLV goes in ``pre_pull_clv`` against
the last quote captured before it went, never in ``clv``.

Everything is paper (``mode``) until probation clears a market (§11).
"""

from __future__ import annotations

import csv
import gzip
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import date as Date
from pathlib import Path

from engine_common.odds import american_to_prob
from nba_engine.audit.settle import pnl, settle_row
from nba_engine.data.capture import MARKET_MAP, QuoteRow
from nba_engine.data.oddsapi import parse_utc
from nba_engine.market import overbias
from nba_engine.market.board import Selection, ev_per_unit, latest, novig, selections
from nba_engine.schemas import GameResult

PAPER = "paper"
PASS = "Pass"
NO_MODEL = "no_model"
NO_EXEC_BOOK = "no_exec_book"
NO_FAIR = "no_fair"
CLOSE, MOVED, PULLED = "close", "moved", "pulled"

PosKey = tuple[str, str, str, str, float | None]  # (event_id, market, side, entity, line)
QuoteKey = tuple[str, str, str, str, float | None, str]  # PosKey + book


@dataclass
class LedgerRow:
    slate_date: str
    tip_utc: str
    event_id: str
    espn_id: str
    matchup: str
    market: str
    ot_rule: str
    side: str
    entity: str
    line: float | None
    book: str  # the DraftKings/BetMGM book that priced it; "" when neither posted
    american: float | None
    opposite_american: float | None  # the other side at the same book, for its no-vig
    exec_books: int  # 0, 1 or 2 execution books posting this line
    exec_prices: str  # "betmgm:-105;draftkings:-110"
    books: int
    paired_books: int
    fair: float | None  # consensus no-vig, Over bias removed
    model_prob: float | None
    edge: float | None  # model_prob - fair
    ev: float | None  # EV per unit of the price taken under model_prob, else under fair
    tier: str
    gates: str  # refusal reasons, ";"-joined; empty on a buy
    priced_at: str  # when the price taken was captured
    pass_tag: str
    versions: str  # "over_bias=...;team_ratings=..."
    mode: str = PAPER
    # -- grading
    outcome: str = ""  # win | loss | push | void; "" = ungraded
    void_reason: str = ""  # dnp | absent | not_played
    pnl: float | None = None
    away_score: int | None = None
    home_score: int | None = None
    close_status: str = ""  # close | moved | pulled
    close_captured_at: str = ""
    close_american: float | None = None
    close_opposite: float | None = None
    close_fair: float | None = None
    clv: float | None = None
    clv_ev: float | None = None
    pre_pull_american: float | None = None
    pre_pull_captured_at: str = ""
    pre_pull_clv: float | None = None
    graded_at: str = ""

    @property
    def position(self) -> PosKey:
        return (self.event_id, self.market, self.side, self.entity, self.line)

    @property
    def is_buy(self) -> bool:
        return not self.gates and self.tier != PASS and self.american is not None

    @property
    def graded(self) -> bool:
        return self.outcome in ("win", "loss", "push")


FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LedgerRow))


def nv(american: float, opposite: float | None) -> float:
    """No-vig probability of ``american``; its implied probability when unpaired."""
    return novig(american, opposite) if opposite is not None else american_to_prob(american)


def quote_index(rows: Iterable[QuoteRow]) -> dict[QuoteKey, QuoteRow]:
    return {(r.event_id, r.market, r.side, r.entity, r.line, r.book): r for r in latest(rows)}


def price(
    sels: Iterable[Selection],
    quotes: Mapping[QuoteKey, QuoteRow],
    *,
    tips: Mapping[str, str],
    shift: Mapping[str, float],
    versions: str,
    pass_tag: str,
    model: Mapping[PosKey, float] | None = None,
    espn_ids: Mapping[str, str] | None = None,
) -> list[LedgerRow]:
    """One ledger row per selection on games that have not tipped.

    ``model`` maps a position to the model's probability. Without one the row is
    refused (``no_model``) but still recorded: it is the board's base rate.
    """
    out: list[LedgerRow] = []
    for s in sels:
        tip = tips.get(s.event_id, "")
        pos = (s.event_id, s.market, s.side, s.entity, s.line)
        fair = overbias.adjusted_fair(s, shift)
        p_model = (model or {}).get(pos)
        q = quotes.get((*pos, s.exec_book)) if s.exec_book else None
        gates = [
            g
            for g, refused in (
                (NO_MODEL, p_model is None),
                (NO_FAIR, fair is None),
                (NO_EXEC_BOOK, s.exec_american is None),
            )
            if refused
        ]
        p_used = p_model if p_model is not None else fair
        out.append(
            LedgerRow(
                slate_date=s.game_date,
                tip_utc=tip,
                event_id=s.event_id,
                espn_id=(espn_ids or {}).get(s.event_id, ""),
                matchup=s.matchup,
                market=s.market,
                ot_rule=_ot_rule(s.market),
                side=s.side,
                entity=s.entity,
                line=s.line,
                book=s.exec_book or "",
                american=s.exec_american,
                opposite_american=q.opposite_american if q else None,
                exec_books=s.exec_books,
                exec_prices=";".join(f"{b}:{p:g}" for b, p in sorted(s.exec_prices.items())),
                books=s.books,
                paired_books=s.paired_books,
                fair=None if fair is None else round(fair, 6),
                model_prob=p_model,
                edge=None if p_model is None or fair is None else round(p_model - fair, 6),
                ev=(
                    None
                    if p_used is None or s.exec_american is None
                    else round(ev_per_unit(p_used, s.exec_american), 6)
                ),
                tier=PASS,
                gates=";".join(gates),
                priced_at=s.exec_captured_at,
                pass_tag=pass_tag,
                versions=versions,
            )
        )
    return out


def _ot_rule(market: str) -> str:
    return next((rule for key, rule in MARKET_MAP.values() if key == market), "")


def pregame(rows: Iterable[QuoteRow], tips: Mapping[str, str]) -> list[QuoteRow]:
    """Quotes captured strictly before their game's tip; a game with no known tip keeps none."""
    out: list[QuoteRow] = []
    for r in rows:
        tip, seen = parse_utc(tips.get(r.event_id, "")), parse_utc(r.captured_at)
        if tip is not None and seen is not None and seen < tip:
            out.append(r)
    return out


def of_record(passes: Iterable[LedgerRow]) -> list[LedgerRow]:
    """Per position: the earliest pass that bought it, else the latest pass that priced it."""
    by_pos: dict[PosKey, list[LedgerRow]] = defaultdict(list)
    for r in passes:
        by_pos[r.position].append(r)
    out: list[LedgerRow] = []
    for rows in by_pos.values():
        rows.sort(key=lambda r: (r.priced_at, r.pass_tag))
        buys = [r for r in rows if r.is_buy]
        out.append(buys[0] if buys else rows[-1])
    out.sort(key=lambda r: (r.tip_utc, r.matchup, r.market, r.entity, r.line or 0.0, r.side))
    return out


def grade(
    rows: Iterable[LedgerRow],
    finals: Mapping[str, GameResult],
    day_quotes: Iterable[QuoteRow],
    *,
    shift: Mapping[str, float],
    graded_at: str,
) -> list[LedgerRow]:
    """Settle each row on its ESPN final (keyed by matchup) and stamp its close.

    Rows whose game has no final come back ungraded. The close is read only from
    quotes captured before tip; nothing captured after it is used.
    """
    rows = list(rows)
    tips = {r.event_id: r.tip_utc for r in rows}
    pre = pregame(day_quotes, tips)
    close_quotes = quote_index(pre)
    close_fair = {
        (s.event_id, s.market, s.side, s.entity, s.line): overbias.adjusted_fair(s, shift)
        for s in selections(pre)
    }
    newest = _newest_capture(pre)
    last_seen = _last_seen(pre)
    posted = {(k[0], k[1], k[2], k[3], k[5]) for k in close_quotes}
    out: list[LedgerRow] = []
    for r in rows:
        game = finals.get(r.matchup)
        if game is None:
            out.append(r)
            continue
        outcome, reason = settle_row(r.market, r.side, r.entity, r.line, game)
        g = replace(r, graded_at=graded_at, espn_id=r.espn_id or game.espn_id)
        if outcome is not None:
            g.outcome, g.void_reason = outcome, reason
            g.pnl = None if r.american is None else pnl(outcome, r.american)
        if game.is_final:
            g.away_score, g.home_score = game.final_away, game.final_home
        _stamp_close(g, close_quotes, close_fair, newest, last_seen, posted)
        out.append(g)
    return out


def _newest_capture(rows: Iterable[QuoteRow]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        k = (r.event_id, r.market)
        out[k] = max(out.get(k, ""), r.captured_at)
    return out


def _last_seen(rows: Iterable[QuoteRow]) -> dict[QuoteKey, QuoteRow]:
    out: dict[QuoteKey, QuoteRow] = {}
    for r in sorted(rows, key=lambda r: r.captured_at):
        out[(r.event_id, r.market, r.side, r.entity, r.line, r.book)] = r
    return out


def _stamp_close(
    g: LedgerRow,
    close_quotes: Mapping[QuoteKey, QuoteRow],
    close_fair: Mapping[PosKey, float | None],
    newest: Mapping[tuple[str, str], str],
    last_seen: Mapping[QuoteKey, QuoteRow],
    posted: set[tuple[str, str, str, str, str]],
) -> None:
    if g.american is None or not g.book:
        return
    g.close_captured_at = newest.get((g.event_id, g.market), "")
    if not g.close_captured_at:
        return
    taken = nv(g.american, g.opposite_american)
    q = close_quotes.get((*g.position, g.book))
    if q is not None:
        g.close_status = CLOSE
        g.close_american, g.close_opposite = q.american, q.opposite_american
        g.clv = round(nv(q.american, q.opposite_american) - taken, 6)
        fair = close_fair.get(g.position)
        g.close_fair = None if fair is None else round(fair, 6)
        g.clv_ev = None if fair is None else round(ev_per_unit(fair, g.american), 6)
        return
    moved = (g.event_id, g.market, g.side, g.entity, g.book) in posted
    g.close_status = MOVED if moved else PULLED
    last = last_seen.get((*g.position, g.book))
    if last is not None:
        g.pre_pull_american = last.american
        g.pre_pull_captured_at = last.captured_at
        g.pre_pull_clv = round(nv(last.american, last.opposite_american) - taken, 6)


# -- persistence ------------------------------------------------------------
def ledger_dir(data_dir: Path, day: Date) -> Path:
    return data_dir / "ledger" / day.isoformat()


def pass_path(data_dir: Path, day: Date, priced_at: str, tag: str) -> Path:
    stamp = priced_at.replace(":", "").replace("-", "")
    return ledger_dir(data_dir, day) / f"priced_{stamp}_{tag}.csv.gz"


def graded_path(data_dir: Path, day: Date) -> Path:
    return ledger_dir(data_dir, day) / "graded.csv.gz"


def save(rows: Iterable[LedgerRow], path: Path) -> Path:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: "" if v is None else v for k, v in asdict(r).items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
    tmp.replace(path)
    return path


def write_once(rows: list[LedgerRow], path: Path) -> Path | None:
    """Files under ``ledger/`` are immutable (state sync is a set union of paths)."""
    if path.exists() or not rows:
        return None
    return save(rows, path)


def load(path: Path) -> list[LedgerRow]:
    if not path.exists():
        return []
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [_row(raw) for raw in csv.DictReader(io.StringIO(text, newline=""))]


def _f(raw: Mapping[str, str], name: str) -> float | None:
    v = raw.get(name, "")
    return float(v) if v else None


def _i(raw: Mapping[str, str], name: str) -> int | None:
    v = raw.get(name, "")
    return int(float(v)) if v else None


def _row(raw: Mapping[str, str]) -> LedgerRow:
    def s(name: str) -> str:
        return raw.get(name, "")

    return LedgerRow(
        slate_date=s("slate_date"),
        tip_utc=s("tip_utc"),
        event_id=s("event_id"),
        espn_id=s("espn_id"),
        matchup=s("matchup"),
        market=s("market"),
        ot_rule=s("ot_rule"),
        side=s("side"),
        entity=s("entity"),
        line=_f(raw, "line"),
        book=s("book"),
        american=_f(raw, "american"),
        opposite_american=_f(raw, "opposite_american"),
        exec_books=_i(raw, "exec_books") or 0,
        exec_prices=s("exec_prices"),
        books=_i(raw, "books") or 0,
        paired_books=_i(raw, "paired_books") or 0,
        fair=_f(raw, "fair"),
        model_prob=_f(raw, "model_prob"),
        edge=_f(raw, "edge"),
        ev=_f(raw, "ev"),
        tier=s("tier") or PASS,
        gates=s("gates"),
        priced_at=s("priced_at"),
        pass_tag=s("pass_tag"),
        versions=s("versions"),
        mode=s("mode") or PAPER,
        outcome=s("outcome"),
        void_reason=s("void_reason"),
        pnl=_f(raw, "pnl"),
        away_score=_i(raw, "away_score"),
        home_score=_i(raw, "home_score"),
        close_status=s("close_status"),
        close_captured_at=s("close_captured_at"),
        close_american=_f(raw, "close_american"),
        close_opposite=_f(raw, "close_opposite"),
        close_fair=_f(raw, "close_fair"),
        clv=_f(raw, "clv"),
        clv_ev=_f(raw, "clv_ev"),
        pre_pull_american=_f(raw, "pre_pull_american"),
        pre_pull_captured_at=s("pre_pull_captured_at"),
        pre_pull_clv=_f(raw, "pre_pull_clv"),
        graded_at=s("graded_at"),
    )


def passes(data_dir: Path, day: Date) -> list[LedgerRow]:
    directory = ledger_dir(data_dir, day)
    out: list[LedgerRow] = []
    for path in sorted(directory.glob("priced_*.csv.gz")) if directory.is_dir() else []:
        out.extend(load(path))
    return out


def graded_rows(data_dir: Path, days: Iterable[Date]) -> list[LedgerRow]:
    out: list[LedgerRow] = []
    for day in days:
        out.extend(load(graded_path(data_dir, day)))
    return out


__all__ = [
    "CLOSE",
    "FIELDS",
    "MOVED",
    "NO_EXEC_BOOK",
    "NO_FAIR",
    "NO_MODEL",
    "PAPER",
    "PASS",
    "PULLED",
    "LedgerRow",
    "grade",
    "graded_path",
    "graded_rows",
    "ledger_dir",
    "load",
    "nv",
    "of_record",
    "pass_path",
    "passes",
    "pregame",
    "price",
    "quote_index",
    "save",
    "write_once",
]
