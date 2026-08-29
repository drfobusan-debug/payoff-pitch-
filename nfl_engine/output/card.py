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
from nfl_engine.market.screens import Tier

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


@dataclass
class GameSection:
    matchup: str
    kickoff: str
    plays: list[Play] = field(default_factory=list)
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
) -> WeekCard:
    """Group one week's engine rows into game sections, best execution edge first.

    Outside sources are excluded: a benchmark's row is graded in the same ledger
    so it can be measured, never so it can be presented as one of our plays.
    """
    scope = [e for e in entries if e.season == season and e.week == week and e.source == ENGINE]
    outside = {h.matchup: h for h in head_to_head(entries, season=season, week=week)}
    sections: dict[str, GameSection] = {}
    for entry in sorted(scope, key=lambda e: -(e.ev_fair or 0.0)):
        section = sections.setdefault(
            entry.matchup,
            GameSection(
                matchup=entry.matchup,
                kickoff=entry.kickoff_utc or entry.date,
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
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 13px;
       color: #16181d; margin: 24px; }
h1 { font-size: 20px; margin-bottom: 2px; }
h2 { font-size: 15px; margin: 18px 0 6px; border-bottom: 1px solid #d8dbe0;
     padding-bottom: 3px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 6px; }
th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #e6e8eb; }
th { background: #f4f5f7; font-weight: 600; }
.note { color: #6b7280; font-size: 11px; }
.veto { color: #6b7280; font-size: 11px; margin: 0 0 8px; }
"""


def render_html(card: WeekCard) -> str:
    parts = [
        "<html><head><meta charset='utf-8'>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>{html.escape(card.title())}</h1>",
        f"<p class='note'>{html.escape(PAPER_NOTE)}</p>",
        *([f"<p class='note'>{html.escape(card.calibration)}</p>"] if card.calibration else []),
        f"<p>{card.selections} selections priced, {len(card.plays())} survive the screens.</p>",
        *(
            [
                f"<p class='note'>FPI backs the other side on {card.contested} of them."
                " Shown, not acted on.</p>"
            ]
            if card.contested
            else []
        ),
    ]
    for game in card.games:
        parts.append(f"<h2>{html.escape(game.matchup)}</h2>")
        if game.plays:
            parts.append(
                "<table><tr><th>Play</th><th>Price</th><th>Book</th><th>Model</th>"
                "<th>Fair</th><th>Exec EV</th><th>Tier</th></tr>"
            )
            for play in game.plays:
                parts.append(
                    f"<tr><td>{html.escape(play.label())}</td><td>{play.price()}</td>"
                    f"<td>{html.escape(play.book)}</td><td>{play.model_prob:.3f}</td>"
                    f"<td>{play.fair_prob or 0.0:.3f}</td><td>{play.ev_fair or 0.0:+.3f}</td>"
                    f"<td>{html.escape(play.tier)}</td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append("<p class='veto'>No play: every price on this game was vetoed.</p>")
        if game.vetoes:
            named = ", ".join(f"{name} x{count}" for name, count in sorted(game.vetoes.items()))
            parts.append(f"<p class='veto'>Vetoed: {html.escape(named)}</p>")
        for note in (game.absences, game.benchmark_note()):
            if note:
                parts.append(f"<p class='note'>{html.escape(note)}</p>")
    if card.record:
        parts.append("<h2>Record to date</h2>")
        parts.append(
            "<table><tr><th>Split</th><th>n</th><th>Win%</th><th>Need</th>"
            "<th>ROI</th><th>Units</th><th>CLV</th></tr>"
        )
        for row in card.record:
            parts.append(
                f"<tr><td>{html.escape(row.label)}</td><td>{row.n}</td>"
                f"<td>{row.win_pct:.4f}</td><td>{row.required_win_pct:.4f}</td>"
                f"<td>{row.roi:+.4f}</td><td>{row.units:+.2f}</td>"
                f"<td>{row.mean_clv:+.4f}</td></tr>"
            )
        parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts)


def render_pdf(html_body: str) -> bytes:
    """Render the card HTML to PDF. Imported here because WeasyPrint needs system
    libraries that a headless box may not have, and a missing PDF must cost the
    caller the PDF only -- never the workbook or the email.
    """
    from weasyprint import HTML

    return bytes(HTML(string=html_body).write_pdf())
