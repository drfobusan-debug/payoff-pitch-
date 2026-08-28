"""Render the morning power screen as a research note (HTML, and PDF via WeasyPrint).

House style is a sell-side equity note rather than a tout sheet: a thesis that
states the position and the risk to it, the provenance of every window, the
analysis, and a rated recommendation table at the end. The prose that carries
judgement is generated from the numbers -- what makes an arsenal dangerous is a
comparison, and a comparison is a sentence, not a cell.

Ratings describe conviction in the *matchup* and read no price. A priced board
is appended when one is supplied (see ``power_board``, which reads the card's own
devigged rows off disk rather than fetching anything); it sits beside the ratings
and feeds none of them, so a bet still has to clear the matchup and the number
separately.
"""

from __future__ import annotations

import html
import math
from datetime import date as Date

from mlb_engine.audit.power_ledger import GradedPosition, Record, Scorecard
from mlb_engine.output.power_board import ROWS_PER_BATTER, Board, BoardRow
from mlb_engine.output.power_screen import (
    MIN_STARTER_BF,
    MIN_STARTER_PITCHES,
    SCORED,
    STARTER_TOP_N,
    TOP_K,
    WORK_DAYS,
    ContactLine,
    HitterView,
    MatchupSection,
    ScreenResult,
    StarterCard,
    StarterMetric,
    StarterSplit,
)

CUT_LOG_ROWS = 12  # near misses printed in the appendix
CUT_LOG_MIN_PA = 60  # a hitter cut on fewer plate appearances is not a near miss

STYLE = """
@page{size:letter;margin:16mm 14mm 18mm;
  @bottom-center{content:"Payoff Pitch United - power screen - " counter(page) " of "
    counter(pages);font-family:Helvetica,Arial,sans-serif;font-size:8pt;color:#777}}
body{font-family:Georgia,'Times New Roman',serif;font-size:9.5pt;line-height:1.45;
  color:#151515;max-width:820px;margin:0 auto}
h1{font-family:Helvetica,Arial,sans-serif;font-size:19pt;margin:0 0 2mm;
  border-bottom:2.5pt solid #8c0000;padding-bottom:2mm}
h2{font-family:Helvetica,Arial,sans-serif;font-size:12pt;margin:7mm 0 2mm;color:#8c0000;
  border-bottom:.6pt solid #bbb;padding-bottom:1mm;break-after:avoid}
h3{font-family:Helvetica,Arial,sans-serif;font-size:10pt;margin:5mm 0 1.5mm;break-after:avoid}
p{margin:0 0 2.5mm;text-align:justify}
table{border-collapse:collapse;width:100%;margin:2mm 0 4mm;
  font-family:Helvetica,Arial,sans-serif;font-size:7.6pt}
thead{display:table-header-group}
tr{break-inside:avoid}
th{background:#efefef;text-align:left;padding:1.6mm 1.8mm;border-bottom:1pt solid #888;
  font-size:7.4pt;letter-spacing:.02em}
td{padding:1.3mm 1.8mm;border-bottom:.4pt solid #e2e2e2;vertical-align:top}
tbody tr:nth-child(even){background:#fafafa}
td.n,th.n{text-align:right}
.sub{color:#555;font-size:8.5pt;margin-top:-1mm}
.caveat{background:#fbf6e8;border-left:2.5pt solid #b8860b;padding:2mm 3mm;margin:3mm 0}
.buy{color:#0a6000;font-weight:bold}.hold{color:#8a6d00;font-weight:bold}
.avoid{color:#8c0000;font-weight:bold}
tbody tr.top,tbody tr.top:nth-child(even){background:#fdeeee}
tbody tr.top td{font-weight:bold;border-bottom:.4pt solid #e8c9c9}
tbody tr.top td:first-child{border-left:2.5pt solid #8c0000}
"""


def _f3(x: float) -> str:
    """A rate on the .300 scale, the way a baseball reader expects to see it."""
    if x is None or math.isnan(x):
        return "&mdash;"
    return f".{round(x * 1000):03d}"


def _pc(x: float, digits: int = 1) -> str:
    if x is None or math.isnan(x):
        return "&mdash;"
    return f"{x * 100:.{digits}f}%"


def _num(x: float, digits: int = 1, signed: bool = False) -> str:
    if x is None or math.isnan(x):
        return "&mdash;"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _table(
    headers: list[str],
    rows: list[list[str]],
    numeric_from: int = 1,
    row_classes: list[str] | None = None,
) -> str:
    head = "".join(
        f"<th class='{'n' if i >= numeric_from else ''}'>{h}</th>" for i, h in enumerate(headers)
    )
    classes = row_classes or [""] * len(rows)
    body = "".join(
        f"<tr class='{cls}'>"
        + "".join(
            f"<td class='{'n' if i >= numeric_from else ''}'>{c}</td>" for i, c in enumerate(r)
        )
        + "</tr>"
        for r, cls in zip(rows, classes, strict=True)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _contact_row(label: str, line: ContactLine, usage: float | None) -> list[str]:
    return [
        label,
        _pc(usage) if usage is not None else "&mdash;",
        str(line.pitches),
        str(line.bbe),
        _num(line.rv100, 2, signed=True),
        _f3(line.xba),
        _f3(line.xwoba),
        _pc(line.hh),
        _pc(line.brl),
        _pc(line.whiff),
    ]


_CONTACT_HEAD = ["pitch", "usage", "n", "bbe", "RV/100", "xBA", "xwOBA", "HH%", "Brl%", "whiff%"]


# --- prose ---------------------------------------------------------------


def _worst_pitch(section: MatchupSection) -> tuple[str, ContactLine, float] | None:
    """The most damaging pitch the starter throws in volume, by xwOBA allowed."""
    candidates = [
        (name, line, section.starter.usage.get(name, 0.0))
        for name, line in section.starter.arsenal.items()
        if not math.isnan(line.xwoba) and section.starter.usage.get(name, 0.0) >= 0.10
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[1].xwoba)


def _best_pitch(section: MatchupSection) -> tuple[str, ContactLine, float] | None:
    candidates = [
        (name, line, section.starter.usage.get(name, 0.0))
        for name, line in section.starter.arsenal.items()
        if not math.isnan(line.xwoba)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[1].xwoba)


def _starter_prose(section: MatchupSection) -> str:
    s = section.starter
    bits = [
        f"<strong>{html.escape(s.name)} ({s.throws}HP) vs {html.escape(s.opponent)}</strong> "
        f"allows a {_pc(s.brl_pct)} barrel rate and {_pc(s.hh_pct)} hard contact on a "
        f"{_pc(s.fb_pct)} fly-ball rate, with {_f3(s.xwobacon)} xwOBA on contact and "
        f"{_pc(s.hr_per_bf, 2)} of batters faced leaving the yard. "
        f"He misses bats at {_pc(s.k_bb_pct)} K-BB%."
    ]
    worst = _worst_pitch(section)
    best = _best_pitch(section)
    if worst:
        name, line, usage = worst
        bits.append(
            f"<strong>His problem pitch is the {name.lower()}, {_pc(usage)} of the mix at "
            f"{_f3(line.xwoba)} xwOBA and {_f3(line.xba)} xBA allowed on "
            f"{_num(line.rv100, 2, signed=True)} run value per 100.</strong>"
        )
    if best and worst and best[0] != worst[0]:
        name, line, usage = best
        bits.append(
            f"The {name.lower()} is the pitch that saves him &mdash; {_f3(line.xwoba)} allowed on "
            f"a {_pc(line.whiff)} whiff rate &mdash; and it is only {_pc(usage)} of what he throws."
        )
    bits.append(
        f"The engine projects <strong>{section.starter_bf:.1f} batters faced</strong> "
        f"(sd {section.starter_bf_sd:.1f}): a {section.pitch_cap}-pitch ceiling at "
        f"{section.pitches_per_pa:.2f} pitches per plate appearance, against a workload cap of "
        f"{section.starter_bf_cap}."
    )
    return " ".join(bits)


def _hitter_prose(view: HitterView, section: MatchupSection) -> str:
    h = view.line
    bits: list[str] = []
    delta = view.fit_delta
    if not math.isnan(delta):
        direction = "adds" if delta > 0 else "takes away"
        bits.append(
            f"Weighting his own splits by this mix gives a fit xwOBA of {_f3(view.fit_xwoba)} "
            f"against his overall {_f3(view.overall.xwoba)} &mdash; the arsenal {direction} "
            f"{abs(delta) * 1000:.0f} points."
        )
    readable = {k: v for k, v in view.per_pitch.items() if not math.isnan(v.xwoba)}
    if readable:
        best = max(readable.items(), key=lambda kv: kv[1].xwoba)
        worst = min(readable.items(), key=lambda kv: kv[1].xwoba)
        usage_best = section.starter.usage.get(best[0], math.nan)
        bits.append(
            f"He does his damage against the {best[0].lower()} "
            f"({_f3(best[1].xwoba)} xwOBA, {_pc(best[1].brl)} barrels), which is "
            f"{_pc(usage_best)} of the mix."
        )
        if worst[0] != best[0]:
            usage_worst = section.starter.usage.get(worst[0], math.nan)
            bits.append(
                f"<strong>The pitch that beats him is the {worst[0].lower()}</strong> "
                f"({_f3(worst[1].xwoba)}, {_pc(worst[1].hh)} hard-hit), {_pc(usage_worst)} of it."
            )
    if view.exposure is not None:
        e = view.exposure
        bits.append(
            f"Batting {h.slot if h.slot else '?'} he gets <strong>{e.pa_vs_starter:.2f} of "
            f"{e.pa_total:.2f} plate appearances</strong> against the starter "
            f"({_pc(e.share_vs_starter, 0)}), a {_pc(e.third_look, 0)} chance of a third look, "
            f"and {e.pa_vs_pen:.2f} against the bullpen."
        )
    if h.power_exception:
        bits.append(
            f"<strong>He is here as a power exception</strong>: a {h.wrc:.0f} wRC+ fails the "
            f"screen, and a {_f3(h.xwoba_con)} xwOBA on contact with a {_pc(h.k)} strikeout rate "
            f"is a home-run profile rather than a counting-stat one."
        )
    if h.luck_gap > 0.030:
        bits.append(
            f"His wOBA outruns his expected mark by {h.luck_gap * 1000:+.0f} points, so some of "
            f"the line above is luck the market may already have taken back."
        )
    return " ".join(bits)


def _rating(view: HitterView) -> tuple[str, str]:
    """Conviction in the matchup, and the reason, from the assembled evidence.

    Deliberately mechanical: exposure to the starter, whether the arsenal adds or
    subtracts, contact quality, and whether the bullpen gives it back. A rating
    is about the matchup only -- there is no price in this module.
    """
    h = view.line
    e = view.exposure
    share = e.share_vs_starter if e else math.nan
    opp = e.opponent_xwoba if e else math.nan
    delta = view.fit_delta
    reasons: list[str] = []
    score = 0
    if not math.isnan(share):
        if share >= 0.58:
            score += 1
            reasons.append(f"{share * 100:.0f}% of his game is the matchup")
        elif share < 0.45:
            score -= 1
            reasons.append(f"only {share * 100:.0f}% of his game is the matchup")
    if not math.isnan(delta):
        if delta >= 0.030:
            score += 1
            reasons.append(f"arsenal fit {delta * 1000:+.0f} points")
        elif delta <= -0.020:
            score -= 1
            reasons.append(f"arsenal fit {delta * 1000:+.0f} points")
    if not math.isnan(opp):
        if opp >= 0.340:
            score += 1
            reasons.append(f"full-game opponent {_f3(opp)}")
        elif opp <= 0.300:
            score -= 1
            reasons.append(f"bullpen pulls the full-game opponent to {_f3(opp)}")
    if h.xwoba_con >= 0.440:
        score += 1
        reasons.append(f"{_f3(h.xwoba_con)} xwOBA on contact")
    if h.k >= 0.30:
        score -= 1
        reasons.append(f"{_pc(h.k)} strikeout rate")
    if h.power_exception:
        # He is here on contact quality with a rate line the wRC+ cut rejected, so
        # the damage markets are the only ones the evidence covers -- a strikeout
        # that ends the plate appearance pays nothing on hits or on H+R+RBI.
        reasons.append("kept on power alone: home runs and total bases only")
    if score >= 3:
        return "BUY", "; ".join(reasons)
    if score >= 1:
        return "HOLD", "; ".join(reasons)
    return "AVOID", "; ".join(reasons)


# --- sections ------------------------------------------------------------


def _thesis(result: ScreenResult) -> str:
    if not result.sections:
        return (
            "<p>No starter on the slate cleared the readability floor, so the screen has "
            "no position today.</p>"
        )
    lead = result.sections[0]
    kept = sum(len(s.hitters) for s in result.sections)
    live = sum(1 for s in result.sections if s.hitters)
    worst = _worst_pitch(lead)
    paras = [
        f"<p><strong>{kept} hitters survive the screen across "
        f"{live} of the day's matchups.</strong> The chain ranks every probable "
        f"starter on the damage he allows in the air, keeps the softest few, scores the lineups "
        f"facing them on eleven contact and discipline metrics split by hand, cuts on plate "
        f"appearances, then wRC+, then expected contact, and finally tests each survivor against "
        f"the arsenal he will actually see and the number of turns he will actually get.</p>"
    ]
    lead_bit = (
        f"<p><strong>The lead position is {html.escape(lead.starter.opponent)} against "
        f"{html.escape(lead.starter.name)}</strong>, the most exposed arm on the board: "
        f"{_pc(lead.starter.brl_pct)} barrels and {_pc(lead.starter.hh_pct)} hard contact allowed "
        f"on a {_pc(lead.starter.fb_pct)} fly-ball rate."
    )
    if worst:
        name, line, usage = worst
        lead_bit += (
            f" His {name.lower()} is {_pc(usage)} of the mix at {_f3(line.xwoba)} xwOBA allowed "
            f"and {_num(line.rv100, 2, signed=True)} run value per 100, and it is the pitch the "
            f"surviving bats are being asked to hunt."
        )
    lead_bit += "</p>"
    paras.append(lead_bit)
    downgrades = [
        (s, v) for s in result.sections for v in s.hitters
        if v.exposure and not math.isnan(v.exposure.opponent_xwoba)
        and v.exposure.opponent_xwoba <= 0.300
    ]
    if downgrades:
        names = ", ".join(html.escape(v.line.name) for _s, v in downgrades[:4])
        paras.append(
            f"<p><strong>Exposure is where the screen changes its mind.</strong> {names} clear "
            f"every form filter and then lose most of the case to what follows the starter: a "
            f"full-game opponent xwOBA at or below .300 once the bullpen is weighted in. Read the "
            f"exposure table before the form table.</p>"
        )
    return "".join(paras)


def _price(american: float | None) -> str:
    if american is None:
        return "&mdash;"
    return f"{american:+.0f}"


def _board_section(board: Board) -> str:
    """The survivors on the card's own board, best expected value first."""
    rows = []
    for r in board.rows:
        fair = _pc(r.fair_prob) if r.fair_prob is not None else "one-way"
        rows.append([
            html.escape(r.batter),
            r.label,
            _price(r.american),
            html.escape(r.book or "&mdash;"),
            _pc(r.model_prob),
            fair + ("" if r.devigged else "*"),
            _num((r.edge or 0.0) * 100, 1, signed=True) if r.edge is not None else "&mdash;",
            _pc(r.ev, 1) if r.ev is not None else "&mdash;",
            r.tier,
        ])
    out = [
        "<h2>The board</h2>",
        f"<p><strong>{len(board.priced)} of "
        f"{len(board.priced) + len(board.unpriced)} survivors have a price</strong>, and "
        f"{len(board.buys)} of their rows cleared the card's buy tiers. Every figure below is "
        f"the nightly run's own: the model probability it simulated, the best price it found, "
        f"and the two-sided no-vig mark it measured the edge against. Nothing was re-priced or "
        f"re-simulated for this note, so a row here is the bet the engine actually made.</p>",
    ]
    if rows:
        out.append(
            _table(
                ["batter", "market", "price", "book", "model", "no-vig", "edge", "EV", "tier"],
                rows,
                numeric_from=2,
            )
        )
        out.append(
            "<p class='sub'>Edge is model minus no-vig, in points. A no-vig mark starred is "
            "one-sided at the book, so its vig could not be stripped and the edge beside it is "
            "overstated by roughly half the hold.</p>"
        )
    if board.dropped:
        out.append(
            f"<p class='sub'>{board.dropped} further priced rows on these hitters are not shown; "
            f"each keeps his homer and his H+R+RBI where both were quoted, then fills to "
            f"{ROWS_PER_BATTER} rows by expected value, one quote per bet.</p>"
        )
    if board.unpriced:
        names = ", ".join(html.escape(n) for n in board.unpriced)
        out.append(
            f"<div class='caveat'><strong>Priced by nobody: {names}.</strong> The pipeline "
            f"declines a game whose lineup is not posted, and this screen runs before that on "
            f"purpose. These are matchup opinions with no bet attached yet &mdash; re-read the "
            f"card once the lineups land.</div>"
        )
    against = [
        r for r in board.rows
        if r.side == "over" and r.edge is not None and r.edge <= -0.03
    ]
    if against:
        worst = min(against, key=lambda r: r.edge or 0.0)
        out.append(
            f"<p><strong>The market disagrees hardest on {html.escape(worst.batter)} "
            f"{worst.label}</strong>: {_pc(worst.model_prob)} modelled against a "
            f"{_pc(worst.fair_prob) if worst.fair_prob is not None else 'one-way'} no-vig line. "
            f"The screen reads form and exposure; the price reads everything, including the "
            f"lineup card this note is guessing at.</p>"
        )
    return "".join(out)


def ratings(result: ScreenResult) -> dict[str, str]:
    """Each survivor's BUY/HOLD/AVOID, for anything recording what the note said."""
    return {v.line.name: _rating(v)[0] for s in result.sections for v in s.hitters}


def _wl(rec: Record) -> str:
    out = f"{rec.wins}-{rec.losses}"
    return out + (f"-{rec.pushes}" if rec.pushes else "")


def _scorecard_section(card: Scorecard, graded: list[GradedPosition]) -> str:
    """How the previous board actually did, before this one asks to be believed.

    Printed first among the priced sections deliberately: the receipt for the
    last claim belongs above the next one.
    """
    o = card.overall
    if not o.n:
        return (
            "<h2>Yesterday's board, graded</h2>"
            f"<p>Nothing from {html.escape(card.day)} could be graded"
            + (f" &mdash; {card.voided} rows voided" if card.voided else "")
            + ". A row is voided when the game did not finish or the hitter never batted, "
            "which is a book's answer too, not a loss.</p>"
        )
    rows = [
        [
            html.escape(g.position.batter),
            g.position.label,
            _price(g.position.odds),
            _pc(g.position.model_prob),
            _pc(g.position.fair_prob) if g.position.fair_prob is not None else "one-way",
            str(g.actual),
            g.result,
            _num(g.units, 2, signed=True),
        ]
        for g in sorted(graded, key=lambda g: -g.units)
    ]
    out = [
        "<h2>Yesterday's board, graded</h2>",
        f"<p><strong>{html.escape(card.day)}: {_wl(o)}, "
        f"{_num(o.units, 2, signed=True)} units at the prices shown"
        + (f", {card.voided} voided" if card.voided else "")
        + ".</strong> Every row the note printed that day is graded here off the box score, "
        "flat one unit apiece at the price it was shown at &mdash; not at a better one found "
        "later, and not only on the rows that worked.</p>",
        _table(
            ["batter", "market", "price", "model", "no-vig", "actual", "result", "units"],
            rows,
            numeric_from=2,
        ),
    ]
    if card.model_brier is not None and card.market_brier is not None:
        verdict = "the model" if card.model_beat_market else "the price"
        out.append(
            f"<p><strong>{verdict.capitalize()} was closer.</strong> Brier score "
            f"{card.model_brier:.3f} for the model against {card.market_brier:.3f} for the "
            f"two-sided no-vig line, over the {card.scored_probs} rows where the hold could be "
            f"stripped; the model averaged "
            f"{_pc(card.mean_model_prob) if card.mean_model_prob is not None else '&mdash;'} "
            f"against the market's "
            f"{_pc(card.mean_market_prob) if card.mean_market_prob is not None else '&mdash;'} "
            f"and the rows went {o.wins} of {o.decided}. One slate settles nothing; the column "
            f"that matters is this line repeated, which is what the ledger accumulates.</p>"
        )
    splits = [
        ("card tier", card.by_tier),
        ("screen rating", card.by_rating),
        ("market", card.by_market),
    ]
    split_rows = [
        [html.escape(label), html.escape(rec.label), _wl(rec), _num(rec.units, 2, signed=True)]
        for label, recs in splits
        for rec in recs
    ]
    if split_rows:
        out.append(_table(["cut", "bucket", "W-L", "units"], split_rows, numeric_from=2))
        out.append(
            "<p class='sub'>The tier cut says whether the card's buys beat the rows it passed "
            "on; the rating cut says whether the matchup read discriminated at all. They are "
            "separate because a rating carries no price and can be right about the hitter while "
            "the number was wrong.</p>"
        )
    return "".join(out)


def _provenance(result: ScreenResult, board: Board | None = None) -> str:
    rows = [
        ["Hitter form, hand-split", "Statcast pitch-level",
         f"{result.window_start:%-m/%d}&ndash;{result.window_end:%-m/%d}", "Observed"],
        ["Pitch-type splits, both sides", "Statcast pitch-level",
         f"{result.window_start:%-m/%d}&ndash;{result.window_end:%-m/%d}", "Observed"],
        ["Starter exit point", "features.workload + features.efficiency",
         f"{result.form_days}d form", "Modelled"],
        ["Bullpen quality", "features.rolling.build_bullpen_profile",
         "21d relief from the 6th", "Observed"],
        ["Lineup slots, projected PA", "Rotowire expected lineups", "Today", "Projected"],
    ]
    if board is not None:
        rows.append([
            "Prices, model probabilities, no-vig",
            f"the card's own run ({html.escape(board.source or 'predictions file')})",
            "Today",
            "Market",
        ])
    out = [
        "<h2>Data basis</h2>",
        _table(["Layer", "Source", "Window", "Status"], rows, numeric_from=4),
        "<p>Run value is quoted <strong>from the hitter's side throughout</strong> &mdash; "
        "positive means the offence gained runs. Savant prints a pitcher's version with the "
        "opposite sign. Expected SLG is modelled from the expected bases implied by each ball "
        "in play; Statcast publishes no xSLG.</p>",
    ]
    caveats = list(result.notes)
    if not result.has_run_value:
        caveats.append(
            "Run value is unavailable: the cached Statcast frame predates the "
            "<code>delta_run_exp</code> column. Refresh the cache to populate the RV/100 columns."
        )
    if any(s.lineup_projected for s in result.sections):
        caveats.append(
            "Lineup slots are projections, not posted lineups. Every plate-appearance split below "
            "moves if the order does, and the top recommendation usually rests on one slot."
        )
    if board is None:
        caveats.append(
            "This note reads no market. A matchup rating is not a bet; size it against a price."
        )
    else:
        caveats.append(
            "A matchup rating still contains no price: the board is shown beside the ratings and "
            "is an input to none of them, so agreement between the two is evidence and "
            "disagreement is not resolved for you."
        )
        if board.unpriced:
            caveats.append(
                f"{len(board.unpriced)} of the survivors were never priced by the engine, whose "
                f"games had no posted lineup at the time of the run. Their ratings stand; their "
                f"bets do not exist yet."
            )
    out.append(
        "<div class='caveat'><strong>Carried forward, unresolved.</strong><ul>"
        + "".join(f"<li>{c}</li>" for c in caveats)
        + "</ul></div>"
    )
    return "".join(out)


def _starter_gate(result: ScreenResult) -> str:
    """Stage 0: which arms were eligible to be ranked at all, and why the rest were not."""
    kept = len(result.starters_ranked)
    work_cuts = [c for c in result.starter_cuts if c.stage == "work"]
    siera_cuts = [c for c in result.starter_cuts if c.stage == "siera"]
    if result.siera_floor <= 0:
        out = [
            "<h2>Stage 0 &mdash; the SIERA gate, disabled</h2>",
            "<p class='sub'>This note was run with the gate off, so the ranking below "
            "may contain arms that prevent runs perfectly well. The work floor still "
            "applies: a metric measured on nothing is not a metric.</p>",
        ]
    else:
        out = [
            "<h2>Stage 0 &mdash; who is eligible to be ranked</h2>",
            f"<p class='sub'>Two cuts, in order. First the work floor: an arm needs "
            f"{MIN_STARTER_BF} batters faced and {MIN_STARTER_PITCHES} pitches in the last "
            f"{WORK_DAYS} days before his numbers are measurements rather than small samples, "
            "which is what removes a call-up, an opener and a starter two outings back from "
            f"the injured list. Then SIERA: only arms above {result.siera_floor:.2f} are "
            f"eligible. {kept} of {kept + len(result.starter_cuts)} probables cleared both. "
            "The metrics below say how an arm is hit; SIERA says whether he prevents runs, "
            "and screening on the former alone has repeatedly nominated aces. An arm without "
            "enough work to carry a trusted SIERA is ineligible, not assumed soft.</p>",
        ]
    for label, cuts in (("too little work", work_cuts), ("prevents runs", siera_cuts)):
        if not cuts:
            continue
        rows = [
            [
                html.escape(c.card.name),
                f"{c.card.throws}HP",
                html.escape(c.card.opponent),
                _num(c.card.siera, 2) if c.card.siera is not None else "&mdash;",
                str(c.card.siera_pa),
                html.escape(c.reason),
            ]
            for c in cuts
        ]
        out.append(
            _table([f"gated out &mdash; {label}", "hand", "vs", "SIERA", "PA", "why"],
                   rows, numeric_from=3)
        )
    return "".join(out)


def _metric_cell(metric: StarterMetric, card: StarterCard, split: str) -> str:
    """One arm's number in one metric, or an em dash when he is unrated in it."""
    line = card.lines.get(split)
    if line is None:
        return "&mdash;"
    value = line.value(metric)
    if math.isnan(value):
        return "&mdash;"
    if metric.attr in ("xera", "xfip", "siera"):
        return _num(value, 2)
    if metric.attr == "stuff_plus":
        return _num(value, 0, signed=True)
    if metric.attr == "hr_per_bf":
        return _pc(value, 2)
    return _pc(value)


def _starter_ranking(result: ScreenResult) -> str:
    """Stage 1: the point total, where each point came from, and the numbers behind it."""
    ranked = result.starters_ranked
    splits = result.splits
    pool = result.starters_scored or ranked
    out = [
        "<h2>Stage 1 &mdash; the arms, ranked on eleven metrics from ten angles</h2>",
        "<p class='sub'>Every eligible arm is ranked in each metric he has the sample for: "
        f"one point for being rated in it and two more for a top-{STARTER_TOP_N} finish, so a "
        f"top-{STARTER_TOP_N} metric is worth three points and "
        "a metric he is short of sample for is worth none. That runs ten times &mdash; overall, "
        "innings 1-3, innings 1-5, each time through the order, each batter hand, and home runs "
        "allowed by hand &mdash; and the total is the sum. Each metric is read over the window it "
        "stabilizes in, not one shared frame: whiff, shape and swing rates over three weeks, "
        "strikeout and walk rates over six, batted-ball and run-estimator rates over three "
        "months, xERA season-to-date from Savant. Split floors are scaled by the pool's own "
        "median share of that split, so a third time through the order is judged on a third "
        "time through the order's worth of evidence. The index is the old air-contact z-sum, "
        "kept only to break ties.</p>",
    ]
    if not result.has_xera:
        out.append(
            "<p class='sub'>Savant's expected-statistics board did not answer this morning, so "
            "xERA is unrated for every arm and the overall ranking ran on ten metrics.</p>"
        )
    out.extend(_split_table(sp, pool) for sp in splits)
    out.append(_final_table(pool, splits))
    unrated = [
        (s, sorted({f"{sp.label}: {', '.join(s.scores[sp.key].unrated)}"
                    for sp in splits if sp.key in s.scores and s.scores[sp.key].unrated}))
        for s in pool
    ]
    lines = [f"<li>{html.escape(s.name)} &mdash; {'; '.join(u)}</li>" for s, u in unrated if u]

    if lines:
        out.append(
            "<p class='sub'>Where an arm was unrated, so a low total is not read as a good "
            "pitcher:</p><ul class='sub'>" + "".join(lines) + "</ul>"
        )
    return "".join(out)


def _split_table(split: StarterSplit, pool: list[StarterCard]) -> str:
    """One ranking, in its own order, with the worst ``STARTER_TOP_N`` arms highlighted.

    The point column alone hides what the splits are for: an arm can be sixth
    overall and the worst on the slate the third time through the order, and that
    is the note's whole thesis for a fifth-inning bet. So every ranking prints in
    its own order with its own numbers, and the top three are marked -- worst
    three in the first three innings, worst three against left-handed hitters,
    and so on.
    """
    scored = [s for s in pool if split.key in s.scores]
    if not scored:
        return ""
    scored = sorted(scored, key=lambda s: s.scores[split.key].rank)
    rows = []
    classes = []
    for s in scored:
        score = s.scores[split.key]
        rows.append([
            f"{score.rank}. {html.escape(s.name)}",
            f"{s.throws}HP",
            html.escape(s.opponent),
            str(score.points),
            *[_metric_cell(m, s, split.key) for m in split.metrics],
            ", ".join(score.top_in) or "&mdash;",
        ])
        classes.append("top" if score.rank <= STARTER_TOP_N else "")
    return (
        f"<h3>Worst {STARTER_TOP_N} &mdash; {html.escape(split.label)}</h3>"
        + _table(
            ["starter", "hand", "vs", "pts", *[m.label for m in split.metrics],
             f"top {STARTER_TOP_N} in"],
            rows, numeric_from=3, row_classes=classes,
        )
    )


def _final_table(pool: list[StarterCard], splits: tuple[StarterSplit, ...]) -> str:
    """The one ranking the rest add up to: every eligible arm, worst first.

    Each column is what a single ranking gave him, so an arm carried by one split
    can be told apart from one the whole slate agrees is soft, and the total is
    the order the screen actually acts on. The worst three are marked; the index
    column is the old air-contact z-sum, printed because it breaks ties.
    """
    if not pool:
        return ""
    rows = []
    for i, s in enumerate(pool, 1):
        rows.append([
            f"{i}. {html.escape(s.name)}",
            f"{s.throws}HP",
            html.escape(s.opponent),
            f"<b>{s.points}</b>",
            *[str(s.scores[sp.key].points) if sp.key in s.scores else "&mdash;" for sp in splits],
            _num(s.index, 2, signed=True),
        ])
    return (
        f"<h3>Final ranking &mdash; every ranking added up, worst {STARTER_TOP_N} marked</h3>"
        + _table(
            ["starter", "hand", "vs", "pts", *[sp.label for sp in splits], "index"],
            rows, numeric_from=3,
            row_classes=["top" if i <= STARTER_TOP_N else "" for i in range(1, len(rows) + 1)],
        )
    )


def _pool_table(section: MatchupSection) -> str:
    rows = []
    for v in section.hitters:
        h = v.line
        rows.append([
            html.escape(h.name) + (" *" if h.power_exception else ""),
            str(h.slot or "&mdash;"),
            str(int(h.pa)),
            _num(h.wrc, 0),
            str(h.points),
            str(len(h.top_in)),
            _f3(h.xwoba_pa),
            _f3(h.xwoba_con),
            _pc(h.brl),
            _pc(h.hh),
            _num(h.ev90, 1),
            _pc(h.osw),
        ])
    return _table(
        ["batter", "LP", "PA", "wRC+", "pts", "top5", "xwOBA", "xwOBAcon", "Brl%", "HH%", "EV90",
         "O-Sw%"],
        rows, numeric_from=1,
    )


def _exposure_table(section: MatchupSection) -> str:
    rows = []
    for v in section.hitters:
        e = v.exposure
        if e is None:
            continue
        rows.append([
            html.escape(v.line.name),
            str(v.line.slot or "&mdash;"),
            _num(e.pa_vs_starter, 2),
            _num(e.pa_total, 2),
            _num(e.pa_vs_pen, 2),
            _pc(e.share_vs_starter, 0),
            _pc(e.third_look, 0),
            _f3(v.fit_xwoba),
            _num(v.fit_delta * 1000, 0, signed=True),
            _f3(e.opponent_xwoba),
        ])
    if not rows:
        return ""
    return _table(
        ["batter", "LP", "PA vs SP", "PA total", "vs pen", "% vs SP", "P(3rd look)", "fit xwOBA",
         "fit &Delta;", "full-game opp"],
        rows, numeric_from=1,
    )


def _section_html(section: MatchupSection, index: int) -> str:
    s = section.starter
    out = [
        f"<h2>{index}. {html.escape(s.opponent)} vs {html.escape(s.name)}</h2>",
        f"<p>{_starter_prose(section)}</p>",
        "<h3>His arsenal</h3>",
        _table(
            _CONTACT_HEAD,
            [
                _contact_row(name, line, s.usage.get(name))
                for name, line in sorted(
                    s.arsenal.items(), key=lambda kv: -s.usage.get(kv[0], 0.0)
                )
            ],
        ),
    ]
    if section.bullpen:
        b = section.bullpen
        out.append(
            f"<p><strong>Behind him: the {html.escape(b.team)} bullpen, ranked "
            f"{b.rank} of {b.of_n} softest.</strong> {_f3(b.xwoba)} xwOBA allowed in relief over "
            f"{b.relief_pa} plate appearances, {_pc(b.k_pct)} strikeouts, {_pc(b.bb_pct)} walks, "
            f"{_pc(b.hr_pct, 2)} home runs, {_pc(b.late_k_pct)} strikeouts from the 8th on.</p>"
        )
    out.append("<h3>The bats that survived</h3>")
    out.append(_pool_table(section))
    for v in section.hitters:
        out.append(f"<h3>{html.escape(v.line.name)}</h3>")
        out.append(f"<p>{_hitter_prose(v, section)}</p>")
        if v.per_pitch:
            rows = [
                _contact_row(name, line, s.usage.get(name))
                for name, line in sorted(
                    v.per_pitch.items(), key=lambda kv: -s.usage.get(kv[0], 0.0)
                )
            ]
            rows.append(_contact_row("all, vs hand", v.overall, None))
            out.append(_table(_CONTACT_HEAD, rows))
    exposure = _exposure_table(section)
    if exposure:
        out.append("<h3>Exposure</h3>")
        out.append(exposure)
    return "".join(out)


def _best_price_cell(row: BoardRow | None) -> str:
    if row is None:
        return "not priced"
    ev = _pc(row.ev, 1) if row.ev is not None else "&mdash;"
    return f"{row.label} {_price(row.american)} ({ev})"


def _recommendations(result: ScreenResult, board: Board | None = None) -> str:
    graded = [
        (v, s) for s in result.sections for v in s.hitters
    ]
    order = {"BUY": 0, "HOLD": 1, "AVOID": 2}
    rated = [(*_rating(v), v, s) for v, s in graded]
    rated.sort(key=lambda t: (order[t[0]], -(t[2].line.points)))
    rows = []
    for rating, reason, view, section in rated:
        css = rating.lower()
        row = [
            f"<span class='{css}'>{rating}</span>",
            html.escape(view.line.name),
            html.escape(section.starter.name),
        ]
        if board is not None:
            row.append(_best_price_cell(board.best_for_batter(view.line.name)))
        row.append(reason or "&mdash;")
        rows.append(row)
    buys = [r for r in rated if r[0] == "BUY"]
    lead = (
        f"<p><strong>{len(buys)} of {len(rated)} survivors rate a buy on the matchup.</strong> "
        f"Ratings weigh how much of the game is the matchup, whether the arsenal adds or "
        f"subtracts, contact quality, strikeout risk, and whether the bullpen gives the edge "
        f"back. They contain no price.</p>"
    )
    if board is not None:
        agreed = [
            t for t in rated
            if t[0] == "BUY" and (b := board.best_for_batter(t[2].line.name)) is not None
            and b.is_buy
        ]
        lead += (
            f"<p><strong>{len(agreed)} of those the card also bought at a price.</strong> The "
            f"best-price column is his highest-EV row from the board above, and it is the only "
            f"column here that knows what anything costs: a rating without a price is a matchup "
            f"waiting for a number, and a price without a rating is the market's opinion, not "
            f"ours.</p>"
        )
    tail = (
        "<p><strong>Re-check before first pitch.</strong> Lineup slots here are projections; the "
        "plate-appearance split, and with it every rating, moves if the order does.</p>"
    )
    headers = ["rating", "batter", "vs"]
    if board is not None:
        headers.append("best price (EV)")
    headers.append("basis")
    return (
        "<h2>Recommendations</h2>" + lead
        + _table(headers, rows, numeric_from=len(headers) - 1)
        + tail
    )


def render_html(
    result: ScreenResult,
    *,
    prepared_for: str | None = None,
    board: Board | None = None,
    review: tuple[Scorecard, list[GradedPosition]] | None = None,
) -> str:
    """The full note as a standalone HTML document."""
    subtitle = f"Power screen &middot; {result.as_of:%A, %-d %B %Y}"
    if prepared_for:
        subtitle += f" &middot; prepared for {html.escape(prepared_for)}"
    body = [
        "<h1>Payoff Pitch United &mdash; Morning Power Screen</h1>",
        f"<p class='sub'>{subtitle}</p>",
        "<h2>Thesis</h2>",
        _thesis(result),
        _provenance(result, board),
        _starter_gate(result),
        _starter_ranking(result),
    ]
    for i, section in enumerate(result.sections, 1):
        body.append(_section_html(section, i))
    if result.cut_log:
        # Only the near misses are worth printing: a hitter cut on 14 plate
        # appearances says nothing, and a full cut list on a four-game screen runs
        # to thirty rows that push the recommendations off the page.
        near = sorted(
            (h for h in result.cut_log if h.pa >= CUT_LOG_MIN_PA),
            key=lambda h: -h.wrc,
        )[:CUT_LOG_ROWS]
        rows = [
            [html.escape(h.name), str(int(h.pa)), _num(h.wrc, 0), _f3(h.xwoba_pa), h.cut_reason]
            for h in near
        ]
        body.append("<h2>Appendix &mdash; the closest misses</h2>")
        body.append(
            f"<p>The {len(near)} strongest of {len(result.cut_log)} hitters the cuts removed, "
            f"by window wRC+, among those clearing {CUT_LOG_MIN_PA} plate appearances.</p>"
        )
        body.append(_table(["batter", "PA", "wRC+", "xwOBA", "cut at"], rows, numeric_from=1))
    body.append(
        "<h2>Appendix &mdash; scored metrics</h2><p>"
        + ", ".join(f"{label} ({'high' if hi else 'low'} is better)" for _a, label, hi in SCORED)
        + f". One point apiece, a second for a top-{TOP_K} finish within the surviving pool.</p>"
    )
    if review is not None:
        body.append(_scorecard_section(*review))
    if board is not None:
        body.append(_board_section(board))
    body.append(_recommendations(result, board))
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Power screen {result.as_of.isoformat()}</title>"
        f"<style>{STYLE}</style></head><body>{''.join(body)}</body></html>"
    )


def render_pdf(
    result: ScreenResult,
    *,
    prepared_for: str | None = None,
    board: Board | None = None,
    review: tuple[Scorecard, list[GradedPosition]] | None = None,
) -> bytes:
    """The note as a PDF, through the same WeasyPrint path as the nightly card."""
    from mlb_engine.output.card import render_pdf as _pdf

    return _pdf(render_html(result, prepared_for=prepared_for, board=board, review=review))


def default_filename(as_of: Date, suffix: str = "pdf") -> str:
    return f"power_screen_{as_of.isoformat()}.{suffix}"
