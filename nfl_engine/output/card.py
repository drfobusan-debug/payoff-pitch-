"""The week as something readable: one section per game, Markdown/HTML/PDF.

Built from the ledger rather than from a live pricing run, so a card can be
regenerated for any week that was priced without re-simulating anything and
without spending an Odds API credit. That also fixes what gets shown: the price
and book actually recorded, not whatever the board says now.

The card reports and nothing else. It reads ``tier`` and ``screens`` as written
by the market layer, and both the plays and the vetoes are shown -- a card that
only listed the bets would hide the rejections the record is diagnosed with.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

from nfl_engine.audit.availability import Observation
from nfl_engine.audit.availability import note as absence_note
from nfl_engine.audit.ledger import (
    ENGINE,
    LedgerEntry,
    Metrics,
    market_metrics,
    metrics,
    tier_metrics,
)
from nfl_engine.audit.outside import HeadToHead, benchmark_metrics, head_to_head
from nfl_engine.data.teamnames import canonical
from nfl_engine.market.screens import Tier
from nfl_engine.output.brief import GameBrief, TeamBrief, team_name

BUY_TIERS = (Tier.STRONG.value, Tier.MODERATE.value)
PAPER_NOTE = "Paper only: no stake is placed and no bankroll exists in this engine."


@dataclass
class Play:
    """One bought row, as the card states it."""

    matchup: str
    market: str
    side: str
    line: float | None
    book: str
    odds: float | None
    model_prob: float
    fair_prob: float | None
    ev_fair: float | None
    tier: str
    clv: float | None
    result: str

    def label(self) -> str:
        if self.line is None:
            return f"{self.side} ML"
        if self.market == "total":
            return f"{self.side.upper()} {self.line:g}"
        return f"{self.side} {self.line:+g}"

    def price(self) -> str:
        return "n/a" if self.odds is None else f"{self.odds:+.0f}"


@dataclass(frozen=True)
class MarketRead:
    """One market's consensus rung, market's fair probability beside the model's."""

    market: str
    side: str
    line: float | None
    fair_prob: float | None
    model_prob: float

    def selection(self) -> str:
        if self.market == "total":
            return f"{self.side} {self.line:g}" if self.line is not None else self.side
        if self.line is None:
            return self.side
        return f"{self.side} {self.line:+g}"


@dataclass
class GameSection:
    matchup: str
    kickoff: str
    plays: list[Play] = field(default_factory=list)
    # The market's no-vig number beside ours on the consensus rung of each market,
    # read from the same ledger rows. Display only.
    reads: list[MarketRead] = field(default_factory=list)
    # Records, ratings, starters, injuries, venue, forecast, storylines: the context
    # a reader wants, gathered after pricing and never fed back into it.
    brief: GameBrief | None = None
    # Vetoes fired on this game, and how many rows each one stopped.
    vetoes: dict[str, int] = field(default_factory=dict)
    # The outside forecast on this game, read from its own ledger rows. Display
    # only: it has moved no probability, no tier and no screen, and a game shows
    # the same play whether the benchmark is present, absent or against us.
    benchmark: HeadToHead | None = None
    # Who is out on either side, read from the availability log on disk. Display
    # only for the same reason the benchmark is: no screen, tier or probability
    # has seen it, and the timing evidence that would justify pricing it is still
    # being collected.
    absences: str = ""

    def benchmark_note(self) -> str:
        if self.benchmark is None or not self.benchmark.theirs:
            return ""
        bench = self.benchmark
        prob = f"{bench.their_prob * 100:.1f}%" if bench.their_prob is not None else "n/a"
        return f"FPI: {bench.theirs} {prob} ({bench.their_margin}) -- {bench.mark()}"


@dataclass
class WeekCard:
    season: int
    week: int
    games: list[GameSection]
    selections: int
    record: list[Metrics]
    # What the prices on this card were calibrated through, printed so a reader can
    # tell a corrected market from an uncorrected one. Display only: the card never
    # calibrates anything, it says what pricing already did.
    calibration: str = ""
    # Games where the outside forecast backed the other side of one of our plays.
    # Counted, not acted on: a disagreement is something to read afterwards.
    contested: int = 0

    def plays(self) -> list[Play]:
        return [play for game in self.games for play in game.plays]

    def title(self) -> str:
        return f"NFL {self.season} Week {self.week}"


def build_card(
    entries: list[LedgerEntry],
    *,
    season: int,
    week: int,
    calibration: str = "",
    absences: list[Observation] | None = None,
    briefs: dict[str, GameBrief] | None = None,
) -> WeekCard:
    """Group one week's engine rows into game sections, best execution edge first.

    Outside sources are excluded: a benchmark's row is graded in the same ledger
    so it can be measured, never so it can be presented as one of our plays.
    """
    scope = [e for e in entries if e.season == season and e.week == week and e.source == ENGINE]
    outside = {h.matchup: h for h in head_to_head(entries, season=season, week=week)}
    by_game: dict[str, list[LedgerEntry]] = {}
    for entry in scope:
        by_game.setdefault(entry.matchup, []).append(entry)
    sections: dict[str, GameSection] = {}
    for entry in sorted(scope, key=lambda e: -(e.ev_fair or 0.0)):
        section = sections.setdefault(
            entry.matchup,
            GameSection(
                matchup=entry.matchup,
                kickoff=entry.kickoff_utc or entry.date,
                reads=market_reads(by_game[entry.matchup]),
                brief=(briefs or {}).get(entry.matchup),
                benchmark=outside.get(entry.matchup),
                absences=absence_note(absences or [], entry.matchup),
            ),
        )
        if entry.screens:
            for name in entry.screens.split(";"):
                if name:
                    section.vetoes[name] = section.vetoes.get(name, 0) + 1
            continue
        section.plays.append(
            Play(
                matchup=entry.matchup,
                market=entry.market,
                side=entry.side,
                line=entry.line,
                book=entry.book,
                odds=entry.odds,
                model_prob=entry.model_prob,
                fair_prob=entry.fair_prob,
                ev_fair=entry.ev_fair,
                tier=entry.tier,
                clv=entry.clv,
                result=entry.result,
            )
        )
    games = sorted(sections.values(), key=lambda s: (not s.plays, s.kickoff, s.matchup))
    return WeekCard(
        season=season,
        week=week,
        games=games,
        selections=len(scope),
        record=_record(entries),
        calibration=calibration,
        contested=sum(1 for h in outside.values() if h.contested),
    )


def _home_of(matchup: str) -> str:
    return canonical(matchup.partition(" @ ")[2].strip())


def market_reads(rows: list[LedgerEntry]) -> list[MarketRead]:
    """Home moneyline, home spread and the over, each on the market's consensus rung.

    The consensus rung is the line whose no-vig probability sits nearest even
    money: the number the market is actually quoting, not the alternate the
    ladder happens to reach. Model and market are then read off the same row.
    """
    if not rows:
        return []
    home = _home_of(rows[0].matchup)
    picks: list[tuple[str, list[LedgerEntry]]] = [
        ("moneyline", [e for e in rows if e.market == "moneyline" and canonical(e.side) == home]),
        ("spread", [e for e in rows if e.market == "spread" and canonical(e.side) == home]),
        ("total", [e for e in rows if e.market == "total" and e.side == "over"]),
    ]
    out: list[MarketRead] = []
    for market, candidates in picks:
        priced = [e for e in candidates if e.fair_prob is not None]
        if not priced:
            continue
        row = min(priced, key=lambda e: abs((e.fair_prob or 0.5) - 0.5))
        out.append(
            MarketRead(
                market=market,
                side=row.side,
                line=row.line,
                fair_prob=row.fair_prob,
                model_prob=row.model_prob,
            )
        )
    return out


def _record(entries: list[LedgerEntry]) -> list[Metrics]:
    """The record to date, over every graded row in the ledger.

    Season-to-date rather than this week: a week is at most sixteen games, and a
    record quoted off it says nothing except which way the variance fell.
    """
    graded = [e for e in entries if e.result]
    if not graded:
        return []
    bench = benchmark_metrics(graded)
    return [
        *tier_metrics(graded),
        *market_metrics(graded),
        metrics([e for e in graded if e.source == ENGINE], lambda e: True, "ALL"),
        # The benchmark's own hit rate, on its own row, last. Its units are 0 by
        # construction: it publishes no price and stakes nothing.
        *([bench] if bench is not None else []),
    ]


def render_markdown(card: WeekCard) -> str:
    lines = [f"# {card.title()}", "", f"_{PAPER_NOTE}_", ""]
    if card.calibration:
        lines.extend([f"_{card.calibration}_", ""])
    bought = card.plays()
    lines.append(
        f"{card.selections} selections priced, {len(bought)} survive the screens"
        f" across {sum(1 for g in card.games if g.plays)} games."
    )
    if card.contested:
        lines.append("")
        lines.append(
            f"_FPI backs the other side on {card.contested} of them. Shown, not acted on._"
        )
    lines.append("")
    for game in card.games:
        lines.append(f"## {game.matchup}")
        lines.append("")
        reads = " | ".join(
            f"{r.market} {r.selection()} market {_pct(r.fair_prob)} · model {_pct(r.model_prob)}"
            for r in game.reads
        )
        if reads:
            lines.append(f"_{reads}_")
            lines.append("")
        if game.brief is not None:
            for team in (game.brief.away, game.brief.home):
                bits = [team.name, team.record or "n/a"]
                if team.streak and team.record not in ("", "0-0"):
                    bits.append(team.streak)
                if team.rated and team.net_rank is not None:
                    bits.append(f"rated #{team.net_rank}")
                if team.qb:
                    bits.append(f"QB {team.qb}")
                if team.out:
                    bits.append("out: " + ", ".join(team.out[:6]))
                lines.append("- " + " · ".join(bits))
            lines.append("")
        if game.plays:
            lines.append("| Play | Price | Book | Model | Fair | Exec EV | Tier |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for play in game.plays:
                lines.append(
                    f"| {play.label()} | {play.price()} | {play.book} |"
                    f" {play.model_prob:.3f} | {play.fair_prob or 0.0:.3f} |"
                    f" {play.ev_fair or 0.0:+.3f} | {play.tier} |"
                )
        else:
            lines.append("No play: every price on this game was vetoed.")
        if game.vetoes:
            named = ", ".join(f"{name} x{count}" for name, count in sorted(game.vetoes.items()))
            lines.append("")
            lines.append(f"Vetoed: {named}")
        for note in (game.absences, game.benchmark_note()):
            if note:
                lines.append("")
                lines.append(note)
        lines.append("")
    if card.record:
        lines.append("## Record to date")
        lines.append("")
        lines.append("| Split | n | Win% | Need | ROI | Units | CLV |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in card.record:
            lines.append(
                f"| {row.label} | {row.n} | {row.win_pct:.4f} | {row.required_win_pct:.4f} |"
                f" {row.roi:+.4f} | {row.units:+.2f} | {row.mean_clv:+.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


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
.game{page-break-inside:avoid;border-bottom:2px solid #eceef1;padding-bottom:10px;margin-bottom:6px;}
.shape{background:#eef2f6;border-left:4px solid #16324f;padding:6px 10px;margin:8px 0;font-size:9.8pt;}
.shape .tag{display:inline-block;background:#16324f;color:#fff;padding:1px 8px;border-radius:10px;font-size:8.4pt;font-family:'DejaVu Sans',sans-serif;margin-right:6px;}
.mkt{margin:-2px 0 4px;font-size:9.4pt;color:#4b5563;}
table.teams{width:100%;border-collapse:collapse;font-size:8.9pt;font-family:'DejaVu Sans',sans-serif;margin:6px 0 4px;}
table.teams th{text-align:left;font-weight:normal;color:#6b7280;border-bottom:1px solid #d7dbe0;padding:2px 6px;font-size:7.8pt;text-transform:uppercase;letter-spacing:.5px;}
table.teams td{padding:3px 6px;border-bottom:1px solid #eceef1;vertical-align:top;}
table.teams td.tn{font-weight:bold;color:#16324f;}
.rk{color:#6b7280;font-size:7.8pt;}.streak{color:#16324f;font-weight:bold;}.muted{color:#6b7280;font-style:italic;font-size:8.6pt;}
.take{font-size:10.3pt;margin:6px 0;}
.ctx{font-size:9.2pt;margin:3px 0;color:#2b2f36;}
.story{font-size:9.4pt;margin:6px 0;background:#fbf7ec;border-left:3px solid #c8a951;padding:4px 8px;}
.chip{display:inline-block;background:#c8102e;color:#fff;padding:0 7px;border-radius:9px;font-size:7.6pt;font-family:'DejaVu Sans',sans-serif;margin-right:4px;text-transform:uppercase;letter-spacing:.4px;}
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


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _rank(n: int | None) -> str:
    return "—" if n is None else f"#{n}"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _poss(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _tier_word(tier: str) -> str:
    return tier.replace(" buy", "").lower()


def _market_word(market: str) -> str:
    return {"moneyline": "moneyline", "spread": "spread", "total": "total"}.get(market, market)


def _market_line(game: GameSection) -> str:
    """The market's no-vig probabilities beside the model's, one italic line."""
    labels = {"moneyline": "ML", "spread": "ATS", "total": "Total"}
    bits = [
        f"{labels[r.market]} {html.escape(r.selection())} market {_pct(r.fair_prob)}"
        f" · model {_pct(r.model_prob)}"
        for r in game.reads
        if r.market in labels
    ]
    if not bits:
        return ""
    return f"<p class='mkt'><i>{' &nbsp;|&nbsp; '.join(bits)}</i></p>"


def _epa(x: float | None, rank: int | None) -> str:
    if x is None:
        return "—"
    return f"{x:+.3f} <span class='rk'>{_rank(rank)}</span>"


def _team_row(t: TeamBrief, *, ats: bool, qbs: bool) -> str:
    form = html.escape(t.record or "—")
    if t.streak and t.record not in ("", "0-0"):
        form += f" <span class='streak'>{html.escape(t.streak)}</span>"
    net = "<span class='muted'>unrated</span>"
    if t.rated:
        net = _rank(t.net_rank)
        if t.net_rating is not None:
            net += f" <span class='rk'>({t.net_rating:+.1f})</span>"
    qb_cell = f"<td>{html.escape(t.qb or '—')}</td>" if qbs else ""
    ats_cell = f"<td>{html.escape(t.ats or '—')}</td>" if ats else ""
    return (
        f"<tr><td class='tn'>{html.escape(t.name)}</td><td>{form}</td>{qb_cell}<td>{net}</td>"
        f"<td>{_epa(t.off_epa, t.off_rank)}</td><td>{_epa(t.def_epa, t.def_rank)}</td>{ats_cell}</tr>"
    )


def _team_table(b: GameBrief) -> str:
    ats = bool(b.away.ats or b.home.ats)
    qbs = bool(b.away.qb or b.home.qb)
    return (
        "<table class='teams'><thead><tr><th></th><th>Record</th>"
        f"{'<th>QB</th>' if qbs else ''}<th>Rating</th>"
        f"<th>Off EPA/play</th><th>Def EPA/play</th>{'<th>ATS</th>' if ats else ''}</tr></thead><tbody>"
        f"{_team_row(b.away, ats=ats, qbs=qbs)}{_team_row(b.home, ats=ats, qbs=qbs)}</tbody></table>"
    )


def _players_line(b: GameBrief) -> str:
    parts = [
        f"<b>{html.escape(t.name)}</b>: {html.escape('; '.join(t.leaders[:3]))}"
        for t in (b.away, b.home)
        if t.leaders
    ]
    if not parts:
        return ""
    return f"<p class='ctx'><b>Who matters</b> <span class='muted'>(ESPN season leaders)</span> — {' · '.join(parts)}</p>"


def _out_line(b: GameBrief, absences: str) -> str:
    parts = [
        f"<b>{html.escape(t.name)}</b>: {html.escape(', '.join(t.out[:6]))}"
        for t in (b.away, b.home)
        if t.out
    ]
    quest = [
        f"<b>{html.escape(t.name)}</b>: {html.escape(', '.join(t.questionable[:4]))}"
        for t in (b.away, b.home)
        if t.questionable
    ]
    lines: list[str] = []
    if parts:
        lines.append(
            f"<p class='ctx'><b>Out</b> <span class='muted'>(ESPN report; reported, not priced)</span>"
            f" — {' · '.join(parts)}</p>"
        )
    elif absences:
        lines.append(f"<p class='ctx'><b>Out</b> — {html.escape(absences)}</p>")
    if quest:
        lines.append(f"<p class='ctx'><b>Questionable</b> — {' · '.join(quest)}</p>")
    return "".join(lines)


def _venue_line(b: GameBrief) -> str:
    bits: list[str] = []
    where = html.escape(b.venue) if b.venue else None
    if where and b.city:
        where += f", {html.escape(b.city)}"
    if where:
        if b.surface:
            where += f" ({html.escape(b.surface)})"
        elif b.grass is True:
            where += " (grass)"
        elif b.grass is False:
            where += " (turf)"
        bits.append(where)
    if b.neutral_site:
        bits.append("neutral site")
    if b.indoors() or (
        b.roof is None and b.grass is None and b.condition is None and b.temperature_f is None
    ):
        if b.indoors():
            bits.append("indoors")
    else:
        wx = []
        if b.condition:
            wx.append(html.escape(b.condition.lower()))
        if b.temperature_f is not None:
            wx.append(f"{b.temperature_f:.0f}°F")
        if b.gust_mph is not None and b.gust_mph >= 10:
            wx.append(f"gusts to {b.gust_mph:.0f} mph")
        if b.precip_pct is not None and b.precip_pct > 0:
            wx.append(f"{b.precip_pct:.0f}% rain chance")
        if wx:
            bits.append("kickoff forecast " + ", ".join(wx))
    if b.home.rest is not None and b.away.rest is not None and b.home.rest != b.away.rest:
        bits.append(f"rest {b.home.rest}d home vs {b.away.rest}d away")
    if b.div_game:
        bits.append("division game")
    if b.broadcast:
        bits.append(f"on {html.escape(b.broadcast)}")
    if not bits:
        return ""
    return f"<p class='ctx'><b>Venue &amp; weather</b> — {'; '.join(bits)}.</p>"


def _story_line(b: GameBrief) -> str:
    if not b.headline and not b.story and not b.tags:
        return ""
    chips = "".join(f"<span class='chip'>{html.escape(t)}</span>" for t in b.tags)
    text = html.escape(b.headline or "")
    if b.story:
        text = f"<b>{text}</b> {html.escape(b.story)}" if text else html.escape(b.story)
    return f"<p class='story'>{chips}{text} <span class='muted'>— ESPN/AP preview</span></p>"


def _take(game: GameSection) -> str:
    """The casual read, sentence by sentence, each one only if the data is there."""
    b = game.brief
    if b is None:
        return ""
    parts: list[str] = []

    def tag(t: TeamBrief) -> str:
        bits = [t.record] if t.record else []
        if t.streak and t.record not in ("", "0-0"):
            bits.append(t.streak)
        if t.rated and t.net_rank is not None:
            bits.append(f"rated {_rank(t.net_rank)}")
        return f"{t.name} ({', '.join(bits)})" if bits else t.name

    where = f"in {b.city.split(',')[0]}" if b.city else "on the road"
    if b.neutral_site:
        where = "at a neutral site"
    parts.append(f"{tag(b.away)} visits {tag(b.home)} {where}.")
    if b.away.last or b.home.last:
        lasts = [f"{t.name} {t.last}" for t in (b.away, b.home) if t.last]
        parts.append("Last time out: " + "; ".join(lasts) + ".")
    if b.market_spread is not None:
        fav, dog = (b.home, b.away) if b.market_spread < 0 else (b.away, b.home)
        if b.market_spread == 0:
            parts.append("The market has this a pick'em")
        else:
            parts.append(f"The market has {fav.name} by {abs(b.market_spread):g}")
        if b.market_total is not None:
            parts[-1] += f" with the total at {b.market_total:g}"
        parts[-1] += "."
        if fav.off_rank is not None and dog.def_rank is not None:
            parts.append(
                f"The matchup to watch is {_poss(fav.name)} {_ordinal(fav.off_rank)}-ranked offense "
                f"(EPA/play) against a {dog.name} defense we rate {_ordinal(dog.def_rank)}"
                + (" — a mismatch on paper." if fav.off_rank + 12 < dog.def_rank else ".")
            )
    if b.model_spread is not None:
        who = b.home if b.model_spread <= 0 else b.away
        parts.append(
            f"Our number is {who.name} by {abs(b.model_spread):g}"
            + (f", about {b.model_total:g} total points" if b.model_total is not None else "")
            + (
                f"; ESPN's FPI gives the home side {b.fpi_home:.0f}%."
                if b.fpi_home is not None
                else "."
            )
        )
    elif b.fpi_home is not None:
        parts.append(f"ESPN's FPI gives the home side {b.fpi_home:.0f}%.")
    ml = next((r for r in game.reads if r.market == "moneyline"), None)
    if ml is not None and ml.fair_prob is not None:
        gap = (ml.model_prob - ml.fair_prob) * 100
        if gap >= 3:
            parts.append(
                f"We like {team_name(ml.side)} more than the market does ({gap:+.1f} pts)."
            )
        elif gap <= -3:
            parts.append(
                f"The market is higher on {team_name(ml.side)} than we are ({gap:+.1f} pts), so no moneyline play."
            )
        else:
            parts.append("Model and market are within a few points on the moneyline.")
    if b.away.qb and b.home.qb:
        parts.append(f"Under center: {b.away.qb} for {b.away.name}, {b.home.qb} for {b.home.name}.")
    return f"<p class='take'>{html.escape(' '.join(parts))}</p>"


def _shape(game: GameSection) -> str:
    n = len(game.plays)
    if n == 0:
        headline, desc = "Pass", "every price on this game was vetoed; nothing to bet."
    else:
        strong = sum(1 for p in game.plays if p.tier == Tier.STRONG.value)
        headline = "Strong buy" if strong else "Lean"
        best = game.plays[0]
        desc = (
            f"{n} price{'s' if n > 1 else ''} survive the screens, led by "
            f"{html.escape(best.label())} at {best.price()} ({html.escape(best.book)})."
        )
    return f"<div class='shape'><span class='tag'>{headline}</span> {desc}</div>"


def _play_item(play: Play, *, with_matchup: bool = False) -> str:
    where = f" ({html.escape(play.matchup)})" if with_matchup else ""
    clv = f", CLV {play.clv * 100:+.1f}%" if play.clv is not None else ""
    res = f" · {html.escape(play.result)}" if play.result else ""
    return (
        f"<li><b>{html.escape(play.label())} ({play.price()}, {html.escape(play.book)})</b> —"
        f" {_market_word(play.market)}{where}, model {play.model_prob * 100:.0f}%,"
        f" fair {_pct(play.fair_prob)}, exec EV {(play.ev_fair or 0.0) * 100:+.1f}%{clv}"
        f" · <i>{html.escape(_tier_word(play.tier))}</i>{res}</li>"
    )


def _game_best_block(game: GameSection) -> str:
    if not game.plays:
        return "<p class='bets'><b>Best bets:</b> none clear the screens — model passes.</p>"
    items = "".join(_play_item(p) for p in game.plays)
    return f"<p class='bets'><b>Best bets</b></p><ul class='bets'>{items}</ul>"


def _veto_line(game: GameSection) -> str:
    if not game.vetoes:
        return ""
    named = ", ".join(f"{name} x{count}" for name, count in sorted(game.vetoes.items()))
    return f"<p class='veto'>Vetoed: {html.escape(named)}</p>"


def _kickoff(game: GameSection) -> str:
    if game.brief is not None and game.brief.kickoff_local:
        return html.escape(game.brief.kickoff_local)
    return html.escape(game.kickoff)


def _game_section(game: GameSection) -> str:
    b = game.brief
    title = game.matchup
    if b is not None:
        title = f"{b.away.name} at {b.home.name}"
    context = ""
    if b is not None:
        context = (
            f"{_team_table(b)}{_take(game)}{_story_line(b)}{_players_line(b)}"
            f"{_out_line(b, game.absences)}{_venue_line(b)}"
        )
    elif game.absences:
        context = f"<p class='ctx'><b>Out</b> — {html.escape(game.absences)}</p>"
    bench = game.benchmark_note()
    bench_html = (
        f"<p class='ctx'><b>Outside benchmark</b> — {html.escape(bench)}</p>" if bench else ""
    )
    return (
        f"<div class='game'><h2>{html.escape(title)}<span class='kick'>{_kickoff(game)}</span></h2>"
        f"{_market_line(game)}"
        f"{_shape(game)}"
        f"{context}{bench_html}"
        f"{_game_best_block(game)}{_veto_line(game)}</div>"
    )


def _slate_best_block(card: WeekCard) -> str:
    plays = sorted(card.plays(), key=lambda p: -(p.ev_fair or 0.0))
    if not plays:
        return (
            "<div class='slatebets'><h2>Week's best bets</h2>"
            "<p>The screens pass the entire board this week.</p></div>"
        )
    items = "".join(_play_item(p, with_matchup=True) for p in plays)
    return (
        "<div class='slatebets'><h2>Week's best bets</h2>"
        f"<p class='sbnote'>{len(plays)} plays survive the screens, best execution edge first:</p>"
        f"<ul class='bets big'>{items}</ul></div>"
    )


def _record_table(card: WeekCard) -> str:
    if not card.record:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(row.label)}</td><td>{row.n}</td>"
        f"<td>{row.win_pct * 100:.1f}%</td><td>{row.required_win_pct * 100:.1f}%</td>"
        f"<td>{row.roi * 100:+.1f}%</td><td>{row.units:+.2f}</td>"
        f"<td>{row.mean_clv * 100:+.2f}%</td></tr>"
        for row in card.record
    )
    return (
        "<h2>Record to date</h2>"
        "<table class='record'><thead><tr><th>Split</th><th>n</th><th>Win%</th><th>Need</th>"
        f"<th>ROI</th><th>Units</th><th>CLV</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def render_html(card: WeekCard) -> str:
    n_games = len(card.games)
    n_plays = len(card.plays())
    with_plays = sum(1 for g in card.games if g.plays)
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Sunday Slate</div>"
        f"<h1>{html.escape(card.title())}</h1>"
        "<p class='sub'>The market's number vs. ours — moneyline, spread and totals, every game on the board.</p>"
        f"<div class='dateline'>Weekly card · {n_games} games · paper only</div></div>"
    )
    lead = (
        f"Good morning — here's the {n_games}-game NFL board for Week {card.week}. For every matchup "
        "we read the market's consensus line, price each rung of the ladder off one score "
        "distribution, and set our probability beside the no-vig number in italics under the "
        f"title. The screens kept <b>{n_plays}</b> price{'s' if n_plays != 1 else ''} across "
        f"{with_plays} game{'s' if with_plays != 1 else ''} from {card.selections} priced — bold "
        "under each game and gathered at the bottom. Records, ratings, injuries, venue and "
        "the week's storylines are context for the reader; none of it moves a price."
    )
    notes = [f"<p class='muted'>{html.escape(PAPER_NOTE)}</p>"]
    if card.calibration:
        notes.append(f"<p class='muted'>{html.escape(card.calibration)}</p>")
    if card.contested:
        notes.append(
            f"<p class='muted'>FPI backs the other side on {card.contested} of our plays."
            " Shown, not acted on.</p>"
        )
    body = "".join(_game_section(g) for g in card.games)
    fine = (
        "<p class='fine'>Methodology: the consensus line is de-vigged across paired books and "
        "priced through a drive-level score distribution; every rung and price on the ladder is "
        "screened, and a rung is bought only where the model and the market agree the number is "
        "wrong and the best available price pays for it. Team ratings are opponent-adjusted EPA "
        "and success rate from play-by-play, ranked across the league. ESPN FPI, records, "
        "injuries, venue, forecast and previews are shown for the reader and never priced. "
        f"{html.escape(PAPER_NOTE)}</p>"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_STYLE}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{''.join(notes)}{body}"
        f"{_slate_best_block(card)}{_record_table(card)}{fine}</body></html>"
    )


def render_pdf(html_body: str) -> bytes:
    """Render the card HTML to PDF. Imported here because WeasyPrint needs system
    libraries that a headless box may not have, and a missing PDF must cost the
    caller the PDF only -- never the workbook or the email.
    """
    from weasyprint import HTML

    return bytes(HTML(string=html_body).write_pdf())
