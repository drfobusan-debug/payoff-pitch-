"""The daily slate card: every game in tip-off order, the market's number beside ours.

Built from the day's recorded ledger rows of record (``ledger.of_record``), so
it re-renders for any priced day without an Odds API call, and it never writes
to, re-prices or re-gates a row. Per game: moneyline, spread and total with the
consensus no-vig probability, the engine's probability where one exists (else
"no model price — market only"), the DraftKings/BetMGM price taken, EV, Kelly,
tier and the gates that refused the row. Injuries, rest, travel, line movement
and the news alarm sit beside them as context and price nothing.

``render_pdf`` imports WeasyPrint lazily: a box without its system libraries
loses the PDF only, never the workbook or the email.
"""

from __future__ import annotations

import html
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date

from engine_common.bands import CALL_SE, MIN_N, BandRead
from engine_common.odds import american_to_prob
from engine_common.podcasts import render as podcast_render
from engine_common.podcasts.render import CSS as PODCAST_CSS
from engine_common.podcasts.render import game_block as podcast_game_block
from engine_common.podcasts.render import records_block as podcast_records_block
from nba_engine.audit import scorecard
from nba_engine.audit.ledger import LedgerRow
from nba_engine.data import teamnames
from nba_engine.data.capture import MARKET_MAP
from nba_engine.market.board import EXEC_BOOKS
from nba_engine.output.context import ESPN, OFFICIAL, GameContext, Injury, et_clock
from nba_engine.output.teamstats import Factors, Rated, Split, TeamStats

PodcastView = podcast_render.PodcastView[list[LedgerRow]]
MAIN_MARKETS = ("game_ml", "game_ats", "game_total")
MARKET_ORDER = tuple(key for key, _ in MARKET_MAP.values())
MARKET_LABEL = {
    "game_ml": "Moneyline",
    "game_ats": "Spread",
    "game_total": "Total",
    "h1_ml": "1H moneyline",
    "h1_ats": "1H spread",
    "h1_total": "1H total",
    "pl_pts": "Points",
    "pl_reb": "Rebounds",
    "pl_ast": "Assists",
    "pl_3pm": "Threes",
    "pl_pra": "Pts+Reb+Ast",
}
BOOK_SHORT = {"draftkings": "DK", "betmgm": "MGM"}
NO_MODEL_TEXT = "no model price — market only"
PAPER_NOTE = "Paper only: no stake is placed and no bankroll exists in this engine."
CONTEXT_NOTE = (
    "Injuries, rest, travel, line movement and the news alarm are shown for the reader;"
    " none of it moves a price or a gate."
)
_TEAM_NAMES = {
    code: " ".join(w[:1].upper() + w[1:] for w in name.split())
    for name, code in reversed(list(teamnames.BY_NAME.items()))
}


def team_name(code: str) -> str:
    return _TEAM_NAMES.get(code, code)


def kelly(p: float, american: float) -> float:
    """Full-Kelly stake fraction at ``american`` for a true win chance ``p`` (floored at 0)."""
    b = 1.0 / american_to_prob(american) - 1.0
    return max(0.0, (p * b - (1.0 - p)) / b)


def row_kelly(r: LedgerRow) -> float | None:
    """Kelly on the engine's probability only; a market-only row has none."""
    if r.model_prob is None or r.american is None:
        return None
    return round(kelly(r.model_prob, r.american), 6)


def tip_key(r: LedgerRow) -> tuple[str, str]:
    return (r.tip_utc or "9999", r.matchup)


def strength(r: LedgerRow) -> tuple[bool, float]:
    """Buys first, then the best EV: the order rows are listed in within a game."""
    return (not r.is_buy, -(r.ev if r.ev is not None else float("-inf")))


def market_rank(market: str) -> int:
    return MARKET_ORDER.index(market) if market in MARKET_ORDER else len(MARKET_ORDER)


def board_order(rows: Iterable[LedgerRow]) -> list[LedgerRow]:
    """Tip-off order, then market, player, line and side: the workbook's row order."""
    return sorted(
        rows,
        key=lambda r: (*tip_key(r), market_rank(r.market), r.entity, r.line or 0.0, r.side),
    )


def gate_list(r: LedgerRow) -> list[str]:
    return [g for g in r.gates.split(";") if g]


def params_version(r: LedgerRow) -> str:
    return r.versions or "none fitted"


def side_label(r: LedgerRow) -> str:
    if r.entity:
        return f"{r.entity} {r.side}"
    return r.side.capitalize() if r.side in ("over", "under") else r.side


def line_text(r: LedgerRow) -> str:
    if r.line is None:
        return ""
    return f"{r.line:+g}" if r.market.endswith("_ats") else f"{r.line:g}"


def american_text(american: float | None) -> str:
    if american is None:
        return "—"
    return f"+{american:g}" if american > 0 else f"{american:g}"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def signed_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f}%"


def ev_text(r: LedgerRow) -> str:
    """EV under the model when there is one; else the price against the consensus fair."""
    if r.ev is None:
        return "—"
    return signed_pct(r.ev) if r.model_prob is not None else f"{signed_pct(r.ev)} vs fair"


def exec_prices(r: LedgerRow) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in r.exec_prices.split(";"):
        book, _, price = part.partition(":")
        if book and price:
            out[book] = float(price)
    return out


def main_line(rows: Sequence[LedgerRow], market: str, home: str) -> list[LedgerRow]:
    """The two sides of a game market's main number: the line most books pair.

    A spread pairs on the line seen from the home side (home -3.5 with away
    +3.5), a total on its number; alternates stay in the workbook.
    """
    mine = [r for r in rows if r.market == market and not r.entity]
    if market.endswith("_ml"):
        return sorted(mine, key=lambda r: r.side == home)
    groups: dict[float, list[LedgerRow]] = {}
    for r in mine:
        if r.line is None:
            continue
        key = r.line if market.endswith("_total") or r.side == home else -r.line
        groups.setdefault(key, []).append(r)
    if not groups:
        return []
    best = max(
        groups.values(),
        key=lambda g: (max(r.paired_books for r in g), max(r.books for r in g), len(g)),
    )
    if market.endswith("_total"):
        return sorted(best, key=lambda r: r.side != "over")
    return sorted(best, key=lambda r: r.side == home)


@dataclass(frozen=True)
class GameCard:
    event_id: str
    matchup: str
    away: str
    home: str
    tip_utc: str
    rows: tuple[LedgerRow, ...]
    context: GameContext = field(default_factory=GameContext)

    @property
    def tip_et(self) -> str:
        return et_clock(self.tip_utc) or "tip time unknown"

    @property
    def title(self) -> str:
        return f"{team_name(self.away)} at {team_name(self.home)}"

    def main(self, market: str) -> list[LedgerRow]:
        return main_line(self.rows, market, self.home)

    def buys(self) -> list[LedgerRow]:
        return sorted((r for r in self.rows if r.is_buy), key=strength)

    def others(self) -> list[LedgerRow]:
        return [r for r in self.rows if r.market not in MAIN_MARKETS]

    def gates(self) -> Counter[str]:
        return Counter(g for r in self.rows for g in gate_list(r))


@dataclass(frozen=True)
class SlateCard:
    day: Date
    games: tuple[GameCard, ...]
    preseason: bool = False
    passes: tuple[str, ...] = ()  # pass tags of record, oldest first
    record: Mapping[str, list[scorecard.Metrics]] = field(default_factory=dict)
    graded_rows: int = 0
    integrity: scorecard.Integrity | None = None
    bands: tuple[BandRead, ...] = ()

    @property
    def rows(self) -> list[LedgerRow]:
        return [r for g in self.games for r in g.rows]

    def buys(self) -> list[LedgerRow]:
        """Tip-off order, strongest first within a game."""
        return [r for g in self.games for r in g.buys()]

    def gates(self) -> Counter[str]:
        return Counter(g for r in self.rows for g in gate_list(r))

    def versions(self) -> list[str]:
        return sorted({params_version(r) for r in self.rows})

    def title(self) -> str:
        label = "NBA Preseason" if self.preseason else "NBA"
        return f"{label} Slate · {self.day:%A, %B} {self.day.day}, {self.day.year}"

    def stem(self) -> str:
        return f"PayoffPitch_NBA{'_Preseason' if self.preseason else ''}_{self.day.isoformat()}"


def build_card(
    rows: Iterable[LedgerRow],
    *,
    day: Date,
    context: Mapping[str, GameContext] | None = None,
    graded: Sequence[LedgerRow] = (),
    preseason: bool = False,
    draws: int = 400,
    bands: Sequence[BandRead] = (),
) -> SlateCard:
    """Group the rows of record by game, in tip-off order. Reads rows, never writes them."""
    ordered = board_order(rows)
    by_game: dict[str, list[LedgerRow]] = {}
    for r in ordered:
        by_game.setdefault(r.event_id, []).append(r)
    games = []
    for eid, mine in by_game.items():
        first = mine[0]
        away, _, home = first.matchup.partition(" @ ")
        games.append(
            GameCard(
                event_id=eid,
                matchup=first.matchup,
                away=away,
                home=home,
                tip_utc=first.tip_utc,
                rows=tuple(mine),
                context=(context or {}).get(eid, GameContext()),
            )
        )
    games.sort(key=lambda g: (g.tip_utc or "9999", g.matchup))
    graded = list(graded)
    return SlateCard(
        day=day,
        games=tuple(games),
        preseason=preseason,
        passes=tuple(sorted({r.pass_tag for r in ordered}, key=lambda t: t)),
        record=scorecard.tables(graded, draws=draws) if graded else {},
        graded_rows=len(graded),
        integrity=scorecard.integrity(graded) if graded else None,
        bands=tuple(bands),
    )


# -- HTML --------------------------------------------------------------------
_STYLE = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:4px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#16324f;}
h1{font-size:22pt;color:#16324f;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-size:15pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:20px 0 6px;}
h2 .kick{float:right;font-size:8.6pt;color:#6b7280;font-family:'DejaVu Sans',sans-serif;font-weight:normal;padding-top:6px;}
p{margin:6px 0;}
.lead{font-size:11pt;}
.game{border-bottom:2px solid #eceef1;padding-bottom:10px;margin-bottom:6px;}
.mkt{margin:-2px 0 4px;font-size:9.4pt;color:#4b5563;}
table.board{width:100%;border-collapse:collapse;font-size:8.4pt;font-family:'DejaVu Sans',sans-serif;margin:6px 0 4px;}
table.board th{text-align:left;font-weight:normal;color:#6b7280;border-bottom:1px solid #d7dbe0;padding:2px 5px;font-size:7.4pt;text-transform:uppercase;letter-spacing:.4px;}
table.board td{padding:3px 5px;border-bottom:1px solid #eceef1;vertical-align:top;}
table.board td.mk{font-weight:bold;color:#16324f;}
table.board td.nomodel{color:#6b7280;font-style:italic;font-size:7.8pt;}
table.board td.gate{color:#6b7280;font-size:7.8pt;}
table.board tr.buy td{background:#fbf7ec;}
.muted{color:#6b7280;font-style:italic;font-size:8.6pt;}
.rk{color:#9aa0a8;font-size:6.6pt;font-weight:normal;}
h2{page-break-after:avoid;}
.keep,table.panel{page-break-inside:avoid;}
table.panel{width:100%;border-collapse:collapse;margin:6px 0 0;}
table.panel>tbody>tr>td,table.panel>tr>td{width:50%;vertical-align:top;padding:0 4px 0 0;}
table.tape{width:100%;border-collapse:collapse;font-size:7.8pt;font-family:'DejaVu Sans',sans-serif;}
table.tape th{font-weight:normal;color:#6b7280;font-size:7pt;text-transform:uppercase;letter-spacing:.3px;padding:1px 4px;text-align:right;border-bottom:1px solid #d7dbe0;}
table.tape th.cap{text-align:left;color:#16324f;font-weight:bold;}
table.tape th[colspan]{text-align:center;color:#16324f;}
table.tape td{padding:2px 4px;text-align:right;border-bottom:1px solid #eceef1;white-space:nowrap;}
table.tape td.mk{text-align:left;color:#16324f;font-weight:bold;}
table.tape .tn{background:#fbf7ec;font-weight:bold;}
table.tape tr.g td{color:#6b7280;font-size:7pt;}
.tapenote{font-size:7.2pt;color:#6b7280;font-style:italic;margin:2px 0 4px;}
.ctx{font-size:9.2pt;margin:3px 0;color:#2b2f36;}
p.bets{margin:10px 0 2px;font-size:11pt;color:#16324f;}
ul.bets{margin:2px 0 4px 0;font-size:10pt;}
ul.bets b{color:#111;}
.veto{color:#6b7280;font-size:8.8pt;margin:2px 0 6px;font-family:'DejaVu Sans',sans-serif;}
.slatebets{page-break-inside:avoid;background:#0f2438;color:#f4f6f8;border-radius:6px;padding:12px 16px;margin:22px 0 8px;}
.slatebets h2{color:#ffd76a;border:none;margin:0 0 4px;}
.slatebets .sbnote{color:#c6ccd4;font-style:italic;font-size:9.4pt;margin:0 0 6px;}
ul.bets.big{font-size:10.5pt;}ul.bets.big b{color:#fff;}.slatebets i{color:#ffd76a;}
table.record{width:100%;border-collapse:collapse;font-size:8.6pt;font-family:'DejaVu Sans',sans-serif;margin:6px 0;}
table.record th{text-align:left;font-weight:normal;color:#6b7280;border-bottom:1px solid #d7dbe0;padding:2px 6px;font-size:7.8pt;text-transform:uppercase;}
table.record td{padding:3px 6px;border-bottom:1px solid #eceef1;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""

_e = html.escape


def exec_sibling(game: GameCard, r: LedgerRow) -> LedgerRow | None:
    """The same side's row at the line DraftKings/BetMGM did post, when the main line lacks one.

    Rows are keyed by line, so a consensus main number the two exec books are not
    on (CHA +4 when they sit at +4.5) looks unbettable on the card; the bettable
    row is the sibling one line over."""
    line = r.line
    if exec_prices(r) or r.entity or r.market.endswith("_ml") or line is None:
        return None
    sibs = [
        x
        for x in game.rows
        if x.market == r.market and x.side == r.side and not x.entity and exec_prices(x)
    ]
    if not sibs:
        return None
    return min(sibs, key=lambda x: abs((x.line or 0.0) - line))


def _exec_cell(r: LedgerRow, sibling: LedgerRow | None = None) -> str:
    prices = exec_prices(r)
    if not prices:
        if sibling is not None:
            return (
                f"<span class='muted'>not at this number; at {_e(line_text(sibling))}:</span> "
                f"{_exec_cell(sibling)}"
            )
        return "<span class='muted'>neither DK nor BetMGM posted it</span>"
    bits = []
    for book in EXEC_BOOKS:
        if book not in prices:
            continue
        text = f"{BOOK_SHORT.get(book, book)} {american_text(prices[book])}"
        bits.append(f"<b>{_e(text)}</b>" if book == r.book else _e(text))
    return " · ".join(bits)


def _market_rows(game: GameCard, market: str) -> str:
    rows = game.main(market)
    label = MARKET_LABEL.get(market, market)
    if not rows:
        return (
            f"<tr><td class='mk'>{_e(label)}</td><td colspan='9' class='muted'>"
            "not on the board at the pass of record</td></tr>"
        )
    no_model = all(r.model_prob is None for r in rows)
    out = []
    for i, r in enumerate(rows):
        model = ""
        if no_model:
            if i == 0:
                model = f"<td class='nomodel' rowspan='{len(rows)}'>{NO_MODEL_TEXT}</td>"
        else:
            model = f"<td>{pct(r.model_prob)}</td>"
        cls = " class='buy'" if r.is_buy else ""
        gates = ", ".join(gate_list(r)) or "—"
        kel = row_kelly(r)
        out.append(
            f"<tr{cls}><td class='mk'>{_e(label) if i == 0 else ''}</td>"
            f"<td>{_e(side_label(r))}</td><td>{_e(line_text(r))}</td>"
            f"<td>{pct(r.fair)}</td>{model}<td>{_exec_cell(r, exec_sibling(game, r))}</td>"
            f"<td>{_e(ev_text(r))}</td><td>{pct(kel) if kel is not None else '—'}</td>"
            f"<td>{_e(r.tier)}</td><td class='gate'>{_e(gates)}</td></tr>"
        )
    return "".join(out)


def _board_table(game: GameCard) -> str:
    head = (
        "<tr><th>Market</th><th>Side</th><th>Line</th><th>Market %</th><th>Model %</th>"
        "<th>DK / BetMGM</th><th>EV</th><th>Kelly</th><th>Tier</th><th>Gates</th></tr>"
    )
    body = "".join(_market_rows(game, m) for m in MAIN_MARKETS)
    return f"<table class='board'><thead>{head}</thead><tbody>{body}</tbody></table>"


def _market_line(game: GameCard) -> str:
    bits = []
    for market, short in (("game_ml", "ML"), ("game_ats", "ATS"), ("game_total", "Total")):
        for r in game.main(market):
            if (market == "game_total" and r.side != "over") or (
                market != "game_total" and r.side != game.home
            ):
                continue
            model = pct(r.model_prob) if r.model_prob is not None else "—"
            name = side_label(r) + (f" {line_text(r)}" if r.line is not None else "")
            bits.append(f"{short} {name} market {pct(r.fair)} · model {model}")
    return f"<p class='mkt'><i>{_e(' | '.join(bits))}</i></p>" if bits else ""


def _injury_text(items: Sequence[Injury], team: str) -> str:
    mine = [i for i in items if i.team == team]
    if not mine:
        return f"{team}: none listed"
    by_status: dict[str, list[str]] = {}
    for i in mine:
        by_status.setdefault(i.status, []).append(i.player)
    return f"{team}: " + "; ".join(f"{s} {', '.join(p)}" for s, p in by_status.items())


def _context(game: GameCard) -> str:
    ctx = game.context
    out = []
    if ctx.rest:
        out.append(
            "<p class='ctx'><b>Rest &amp; travel</b> — "
            + _e(" · ".join(r.text() for r in ctx.rest))
            + "</p>"
        )
    off = [i for i in ctx.injuries if i.source == OFFICIAL]
    feed = [i for i in ctx.injuries if i.source == ESPN]
    if ctx.official_report:
        label = f"Official report {ctx.official_report[-5:]} ET"
        text = " · ".join(_injury_text(off, t) for t in (game.away, game.home))
    else:
        label, text = "Official report", "none archived for this date"
    out.append(f"<p class='ctx'><b>{_e(label)}</b> — {_e(text)}</p>")
    if ctx.espn_feed:
        text = " · ".join(_injury_text(feed, t) for t in (game.away, game.home))
    else:
        text = "no feed archived"
    out.append(f"<p class='ctx'><b>ESPN injuries</b> — {_e(text)}</p>")
    moves = ctx.moves_text(game.home) or "no board archived before tip"
    out.append(f"<p class='ctx'><b>Line move, open → now</b> — {_e(moves)}</p>")
    if ctx.alarm:
        out.append(f"<p class='ctx'><b>News alarm</b> — {_e(ctx.alarm)}</p>")
    return "".join(out)


def _rated(x: Rated, fmt: str) -> str:
    if x.value is None:
        return "—"
    rank = f" <span class='rk'>#{x.rank}</span>" if x.rank is not None else ""
    return fmt.format(x.value) + rank


def _team_stats(game: GameCard) -> dict[str, TeamStats]:
    return {t: s for t, s in zip((game.away, game.home), game.context.teams, strict=False) if s is not None}


_TN = " class='tn'"


def _tape_cell(x: Rated, fmt: str, *, tonight: bool = False) -> str:
    mark = _TN if tonight else ""
    return f"<td{mark}>{_rated(x, fmt)}</td>"


def _efficiency(game: GameCard, stats: dict[str, TeamStats]) -> str:
    """Off, def and net rating per 100 possessions, each team at home and on the road."""
    cols: list[tuple[Split, bool]] = []
    for team, tonight in ((game.away, "away"), (game.home, "home")):
        s = stats.get(team)
        for where in ("home", "away"):
            cols.append((getattr(s, where) if s else Split(), where == tonight))
    teams = "".join(f"<th colspan='2'>{_e(t)}</th>" for t in (game.away, game.home))
    wheres = "".join(
        f"<th{_TN if on else ''}>{w}</th>"
        for (_, on), w in zip(cols, ("Home", "Road") * 2, strict=True)
    )
    body = "".join(
        f"<tr><td class='mk'>{label}</td>"
        + "".join(_tape_cell(getattr(sp, key), fmt, tonight=on) for sp, on in cols)
        + "</tr>"
        for label, key, fmt in (
            ("Off rtg", "off", "{:.1f}"),
            ("Def rtg", "dfn", "{:.1f}"),
            ("Net", "net", "{:+.1f}"),
        )
    )
    games = "".join(f"<td{_TN if on else ''}>{sp.games}</td>" for sp, on in cols)
    return (
        "<table class='tape'><thead><tr><th class='cap'>Per 100 poss</th>"
        f"{teams}</tr><tr><th></th>{wheres}</tr></thead><tbody>{body}"
        f"<tr class='g'><td class='mk'>Games</td>{games}</tr></tbody></table>"
    )


def _factors(game: GameCard, stats: dict[str, TeamStats]) -> str:
    """Each offense beside the defense it meets tonight."""
    units: list[tuple[str, str]] = [
        (game.away, "offense"),
        (game.home, "defense"),
        (game.home, "offense"),
        (game.away, "defense"),
    ]
    cols = [getattr(stats[t], u) if t in stats else Factors() for t, u in units]
    pairs = "".join(
        f"<th colspan='2'>{_e(o)} O vs {_e(d)} D</th>"
        for o, d in ((game.away, game.home), (game.home, game.away))
    )
    heads = "".join(f"<th>{_e(t)} {u[0].upper()}</th>" for t, u in units)
    body = "".join(
        f"<tr><td class='mk'>{label}</td>"
        + "".join(_tape_cell(getattr(f, key), fmt) for f in cols)
        + "</tr>"
        for label, key, fmt in (
            ("eFG%", "efg", "{:.1%}"),
            ("TOV%", "tov", "{:.1%}"),
            ("OREB%", "orb", "{:.1%}"),
            ("FT/FGA", "ftr", "{:.3f}"),
        )
    )
    return (
        "<table class='tape'><thead><tr><th class='cap'>Four factors</th>"
        f"{pairs}</tr><tr><th></th>{heads}</tr></thead><tbody>{body}</tbody></table>"
    )


def _team_panel(game: GameCard) -> str:
    """Ratings at home and on the road beside the four factors, one panel per game."""
    stats = _team_stats(game)
    if not stats:
        return "<p class='ctx'><b>Team ratings</b> — no regular-season box scores archived yet</p>"
    return (
        "<table class='panel'><tr>"
        f"<td>{_efficiency(game, stats)}</td><td>{_factors(game, stats)}</td>"
        "</tr></table><p class='tapenote'>Season to date, grey # = rank among teams with"
        " games; tonight's venue shaded. Early splits rest on few games, so ranks are noisy.</p>"
    )


def _others_line(game: GameCard) -> str:
    others = game.others()
    if not others:
        return ""
    buys = sum(1 for r in others if r.is_buy)
    model = any(r.model_prob is not None for r in others)
    tail = "" if model else f" — {NO_MODEL_TEXT}"
    return (
        f"<p class='muted'>First half and props: {len(others)} row(s) priced, {buys} buy(s)"
        f"{tail}. Every row is in the workbook.</p>"
    )


def _gate_text(gates: Counter[str]) -> str:
    return ", ".join(f"{g} ×{n}" for g, n in sorted(gates.items(), key=lambda kv: (-kv[1], kv[0])))


def _bet_item(r: LedgerRow, *, with_matchup: bool = False) -> str:
    where = f" ({_e(r.matchup)}, {_e(et_clock(r.tip_utc))})" if with_matchup else ""
    line = f" {line_text(r)}" if r.line is not None else ""
    kel = row_kelly(r)
    return (
        f"<li><b>{_e(side_label(r))}{_e(line)} ({american_text(r.american)},"
        f" {_e(BOOK_SHORT.get(r.book, r.book))})</b> — {_e(MARKET_LABEL.get(r.market, r.market))}"
        f"{where}, model {pct(r.model_prob)}, fair {pct(r.fair)}, EV {signed_pct(r.ev)},"
        f" Kelly {pct(kel)} · <i>{_e(r.tier)}</i></li>"
    )


def _game_bets(game: GameCard) -> str:
    buys = game.buys()
    if buys:
        items = "".join(_bet_item(r) for r in buys)
        return f"<p class='bets'><b>Best bets</b></p><ul class='bets'>{items}</ul>"
    gates = game.gates()
    refused = f" Refused: {_e(_gate_text(gates))}." if gates else ""
    return (
        "<p class='bets'><b>Best bets:</b> none — no row clears the gates.</p>"
        f"<p class='veto'>{len(game.rows)} row(s) priced.{refused}</p>"
    )


def _game_section(game: GameCard, podcast: PodcastView | None = None) -> str:
    return (
        f"<div class='game'><div class='keep'><h2>{_e(game.title)}<span class='kick'>{_e(game.tip_et)}</span></h2>"
        f"{_market_line(game)}{_board_table(game)}</div>{_team_panel(game)}"
        f"{_context(game)}<div class='keep'>{_others_line(game)}"
        f"{_game_bets(game)}</div>{podcast_game_block(podcast, game.matchup)}</div>"
    )


def _slate_bets(card: SlateCard) -> str:
    buys = card.buys()
    if not buys:
        gates = card.gates()
        why = f" Gates: {_e(_gate_text(gates))}." if gates else ""
        return (
            "<div class='slatebets'><h2>Today's best bets</h2>"
            f"<p>No buys today: every one of the {len(card.rows)} rows of record was refused.{why}</p>"
            "</div>"
        )
    items = "".join(_bet_item(r, with_matchup=True) for r in buys)
    return (
        "<div class='slatebets'><h2>Today's best bets</h2>"
        f"<p class='sbnote'>{len(buys)} buy(s), in tip-off order, strongest first within a game:</p>"
        f"<ul class='bets big'>{items}</ul></div>"
    )


def _summary_table(card: SlateCard) -> str:
    rows = []
    for g in card.games:
        cells = []
        for market in MAIN_MARKETS:
            main = g.main(market)
            pick = next(
                (r for r in main if r.side in (g.home, "over")),
                main[0] if main else None,
            )
            if pick is None:
                cells.append("—")
            elif market == "game_ml":
                cells.append(f"{pick.side} {pct(pick.fair)}")
            else:
                cells.append(f"{side_label(pick)} {line_text(pick)}")
        rows.append(
            f"<tr><td>{_e(g.tip_et)}</td><td>{_e(g.matchup)}</td>"
            + "".join(f"<td>{_e(c)}</td>" for c in cells)
            + f"<td>{len(g.rows)}</td><td>{len(g.buys())}</td></tr>"
        )
    return (
        "<h2>The board in tip-off order</h2>"
        "<table class='record'><thead><tr><th>Tip</th><th>Game</th><th>Home ML fair</th>"
        "<th>Spread</th><th>Total</th><th>Rows</th><th>Buys</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _ci(m: scorecard.Metrics) -> str:
    return f"{signed_pct(m.roi)} ({signed_pct(m.roi_lo)}, {signed_pct(m.roi_hi)})"


def _record_table(card: SlateCard) -> str:
    if not card.record:
        return (
            "<h2>Record to date</h2><p class='muted'>No graded ledger rows yet: the record starts"
            " the morning after the first priced slate is graded.</p>"
        )
    metrics = [*card.record.get("buys", []), *card.record.get("board", [])]
    rows = "".join(
        f"<tr><td>{_e(m.label)}</td><td>{m.n}</td><td>{m.wins}-{m.losses}-{m.pushes}</td>"
        f"<td>{pct(m.win_pct)}</td><td>{pct(m.required_win_pct)}</td><td>{m.units:+.2f}</td>"
        f"<td>{_ci(m)}</td><td>{m.clv_n}</td><td>{m.mean_clv * 100:+.2f}%</td></tr>"
        for m in metrics
    )
    return (
        "<h2>Record to date</h2>"
        f"<p class='muted'>{card.graded_rows} graded row(s). ROI with its 95% interval,"
        " bootstrapped by game; board rows are the market's base rate, not bets.</p>"
        "<table class='record'><thead><tr><th>Cut</th><th>n</th><th>W-L-P</th><th>Win%</th>"
        "<th>Need</th><th>Units</th><th>ROI (95% CI)</th><th>CLV n</th><th>CLV</th></tr>"
        f"</thead><tbody>{rows}</tbody></table>"
    )


def _integrity_line(card: SlateCard) -> str:
    """One line on the graded ledger's plumbing; silent when nothing is graded."""
    i = card.integrity
    if i is None:
        return ""
    flags = [
        (i.duplicates, "duplicate"),
        (i.priced_after_tip, "priced after tip"),
        (i.close_before_price, "close before price"),
        (i.ungraded, "ungraded"),
        (i.no_close, "no close"),
    ]
    bad = ", ".join(f"{n} {label}" for n, label in flags if n)
    ok = "no duplicates, nothing priced after tip, every priced row has a close record"
    return (
        f"<p class='muted'>Ledger integrity: {i.rows} rows, {i.positions} positions; "
        f"{bad or ok}; {i.voids} void, {i.stale_close} stale close, {i.pulled} pulled, "
        f"{i.moved} moved.</p>"
    )


def _bands_table(card: SlateCard) -> str:
    """The schedule bands, graded only: flat 1u on the main line at the card's price."""
    if not card.bands:
        return ""
    rows = "".join(
        f"<tr><td>{_e(b.name)}</td><td>{b.n}</td><td>{_e(b.record)}</td><td>{b.units:+.1f}</td>"
        f"<td>{signed_pct(b.roi)}</td><td>{f'{b.se:.1f}' if b.n > 1 else 'n/a'}</td>"
        f"<td>{b.halves[0]:+.1f} / {b.halves[1]:+.1f}</td><td>{_e(b.verdict)}</td></tr>"
        for b in card.bands
    )
    return (
        "<h2>Schedule bands</h2>"
        "<p class='muted'>Graded only, never a bet: flat 1u on the main line at the card's"
        f" price; a call needs {MIN_N}+ games and {CALL_SE:g} SE with both halves agreeing.</p>"
        "<table class='record'><thead><tr><th>Band</th><th>Games</th><th>W-L</th><th>Units</th>"
        "<th>ROI</th><th>SE (u)</th><th>Halves</th><th>Verdict</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html(card: SlateCard, podcast: PodcastView | None = None) -> str:
    """The card as HTML. ``podcast`` adds the shows' bets under each game and their
    records at the back; it never changes a row."""
    n_games, n_buys = len(card.games), len(card.buys())
    season = "preseason · " if card.preseason else ""
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Hardwood Slate</div>"
        f"<h1>{_e(card.title())}</h1>"
        "<p class='sub'>The market's number vs. ours — moneyline, spread and total,"
        " every game on the board.</p>"
        f"<div class='dateline'>Daily card · {n_games} games · {season}paper only</div></div>"
    )
    lead = (
        f"Here's the {n_games}-game NBA board in tip-off order. Each game shows the consensus"
        " no-vig probability across every book beside the engine's own, the DraftKings and"
        " BetMGM price a buy would be taken at, and the gates that refused a row."
        f" <b>{n_buys}</b> row{'s' if n_buys != 1 else ''} cleared the gates from"
        f" {len(card.rows)} priced."
    )
    model_wired = any(r.model_prob is not None for r in card.rows)
    notes = [f"<p class='muted'>{_e(PAPER_NOTE)}</p>"]
    if not model_wired:
        notes.append(
            "<p class='muted'>The market-anchored model is not wired into pricing yet, so"
            f" every market reads \u201c{NO_MODEL_TEXT}\u201d and every row is refused"
            " (no_model). Market numbers and EV against the consensus fair are shown as"
            " recorded.</p>"
        )
    body = "".join(_game_section(g, podcast) for g in card.games)
    passes = ", ".join(card.passes) or "none"
    fine = (
        "<p class='fine'>Methodology: the consensus fair is the median no-vig probability across"
        " every book that pairs the line, with the fitted Over bias removed; a buy is priced only"
        " at DraftKings or BetMGM, straight bets only. Model rows are market-anchored,"
        " p = w·p_model + (1−w)·p_fair, with per-market weights from the versioned params."
        " EV is per unit staked under the model (or, on a market-only row, under the fair);"
        f" Kelly is full Kelly on the model's probability. {_e(CONTEXT_NOTE)}"
        f" Ledger passes of record: {_e(passes)}. Params: {_e('; '.join(card.versions()))}."
        f" {_e(PAPER_NOTE)}</p>"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_STYLE}{PODCAST_CSS if podcast else ''}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{''.join(notes)}{body}"
        f"{_slate_bets(card)}{_summary_table(card)}{_record_table(card)}"
        f"{_integrity_line(card)}{_bands_table(card)}"
        f"{podcast_records_block(podcast, [g.matchup for g in card.games])}{fine}</body></html>"
    )


def render_text(card: SlateCard) -> str:
    """Plain-text card: the email's text part."""
    lines = [card.title(), PAPER_NOTE, ""]
    for g in card.games:
        lines.append(f"{g.tip_et}  {g.title}")
        for market in MAIN_MARKETS:
            for r in g.main(market):
                model = pct(r.model_prob) if r.model_prob is not None else NO_MODEL_TEXT
                lines.append(
                    f"  {MARKET_LABEL[market]:9} {side_label(r)} {line_text(r)}: market"
                    f" {pct(r.fair)}, model {model}, {BOOK_SHORT.get(r.book, r.book) or '—'}"
                    f" {american_text(r.american)}, EV {ev_text(r)}, {r.tier}"
                    f"{', gates ' + r.gates if r.gates else ''}"
                )
        lines.append("")
    buys = card.buys()
    lines.append(f"Best bets: {len(buys)}")
    for r in buys:
        lines.append(
            f"  {et_clock(r.tip_utc)} {r.matchup}: {side_label(r)} {line_text(r)}"
            f" {american_text(r.american)} {r.book}, EV {signed_pct(r.ev)}"
        )
    return "\n".join(lines) + "\n"


def render_pdf(html_body: str) -> bytes:
    """Render the card HTML to PDF. Imported here because WeasyPrint needs system
    libraries a headless box may not have; a missing PDF costs only the PDF.
    """
    from weasyprint import HTML

    return bytes(HTML(string=html_body).write_pdf())


__all__ = [
    "MAIN_MARKETS",
    "MARKET_LABEL",
    "NO_MODEL_TEXT",
    "PAPER_NOTE",
    "GameCard",
    "SlateCard",
    "board_order",
    "build_card",
    "kelly",
    "main_line",
    "render_html",
    "render_pdf",
    "render_text",
    "row_kelly",
    "strength",
]
