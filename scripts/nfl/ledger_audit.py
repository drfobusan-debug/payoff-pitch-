"""Season audit of the NFL game ledger: buys, one model lean per game-market, screens, CLV, underdogs.

Read-only. Rows the ledger has not graded yet are settled in memory against nflverse
finals, so the audit is current even when ``nfl-engine grade`` lags the workbook;
nothing is written back.

The ledger holds both sides of every quote at every book, so raw Pass W-L is the
vig. The model is judged on one lean per game-market: the side it rates above the
de-vigged fair, at the quote with the most paired books (for spreads and totals, the
rung nearest a coin flip), scored by record and Brier against fair. A game priced on
two weekly cards (a Thursday game on both) counts once, off its latest capture.

``--email`` sends the built-in ``nfl-engine report`` table and this audit as one
message, with the text and the ledger CSV attached (the Sunday-night daemon).
Without ``--season`` it audits the current season and sends nothing when that
season has no rows.

Usage::

    python -m scripts.nfl.ledger_audit [--season 2026] [--email [--to ADDRESS]]
"""

from __future__ import annotations

import argparse
import html
import math
import statistics as st
from collections import defaultdict
from collections.abc import Callable

from engine_runtime import faststart
from nfl_engine.audit.graduation import (
    CLOSE_WINDOW,
    MIN_N,
    graduation_lines,
    has_true_close,
    priced_before_kickoff,
)
from nfl_engine.audit.ledger import ENGINE, LOSS, PUSH, WIN, LedgerEntry, grade, load_ledger
from nfl_engine.audit.tracked import tracked_lines
from nfl_engine.cli import _final_scores, current_week, ledger_path, report_lines
from nfl_engine.config import load_config
from nfl_engine.data import nflverse
from nfl_engine.market.odds import american_to_prob
from nfl_engine.market.screens import Tier

MARKETS = ("moneyline", "spread", "total")
GAP_BANDS = ((0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 1.0))


def _mean(values: list[float]) -> float:
    return st.mean(values) if values else float("nan")


def _wins(entries: list[LedgerEntry]) -> int:
    return sum(e.result == WIN for e in entries)


def record_line(label: str, entries: list[LedgerEntry]) -> str:
    wins = _wins(entries)
    losses = sum(e.result == LOSS for e in entries)
    pushes = sum(e.result == PUSH for e in entries)
    n = wins + losses + pushes
    units = sum(e.pnl for e in entries)
    roi = units / n if n else 0.0
    se = st.pstdev([e.pnl for e in entries]) / math.sqrt(n) if n > 1 else float("nan")
    clv = [e.clv for e in entries if e.clv is not None and has_true_close(e)]
    drift = [e.drift for e in entries if e.drift is not None]
    return (
        f"  {label:<34} n {n:>5}  {wins}-{losses}-{pushes}  units {units:+7.2f}"
        f"  ROI {roi:+.3f} (se {se:.3f})  CLV {_mean(clv):+.4f} (n {len(clv)})"
        f"  drift {_mean(drift):+.4f} (n {len(drift)})"
    )


def has_paired_close(entry: LedgerEntry) -> bool:
    """The close was de-vigged against its opposite side; an unpaired close keeps the raw implied."""
    if entry.close_odds is None or entry.close_prob is None:
        return False
    return abs(entry.close_prob - american_to_prob(entry.close_odds)) > 1e-6


def close_ev_line(label: str, entries: list[LedgerEntry]) -> str:
    """EV of the price taken under the no-vig close: above 0 means the lean beat the vig."""
    values = [
        e.clv_ev
        for e in entries
        if e.clv_ev is not None and has_true_close(e) and has_paired_close(e)
    ]
    n = len(values)
    if n < 2:
        return f"  close EV {label:<26} n {n:>4}  n/a"
    mean, se = st.mean(values), st.stdev(values) / math.sqrt(n)
    if mean - 2 * se > 0:
        verdict = "above 0 by 2 se"
    elif mean + 2 * se < 0:
        verdict = "below 0 by 2 se"
    else:
        verdict = "within 2 se of 0: underpowered"
    return f"  close EV {label:<26} n {n:>4}  {mean:+.4f} (se {se:.4f})  {verdict}"


def settle_missing(entries: list[LedgerEntry], season: int | None) -> int:
    scores = _final_scores(season)
    filled = 0
    for entry in entries:
        if entry.result:
            continue
        final = scores.get((entry.matchup, entry.date))
        if final is None:
            continue
        grade(entry, final[0], final[1], home=entry.matchup.split(" @ ")[-1])
        filled += 1
    return filled


def _quote_rank(entry: LedgerEntry) -> tuple[int, float]:
    fair = entry.fair_prob if entry.fair_prob is not None else 0.0
    rung = 0.0 if entry.market == "moneyline" else -abs(fair - 0.5)
    return entry.paired_books, rung


def in_game(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Rows priced after kickoff: live lines, not prices anyone could have bet into."""
    return [e for e in entries if e.captured_at and e.kickoff_utc and not priced_before_kickoff(e)]


def model_leans(graded: list[LedgerEntry]) -> list[LedgerEntry]:
    """One lean per game-market, from rows priced before kickoff only."""
    live = set(map(id, in_game(graded)))
    groups: dict[tuple[int, str, str, str], list[LedgerEntry]] = defaultdict(list)
    for e in graded:
        if e.fair_prob is not None and e.result != PUSH and id(e) not in live:
            groups[(e.season, e.matchup, e.date, e.market)].append(e)
    leans = []
    for rows in groups.values():
        latest = max((e.week, e.captured_at) for e in rows)
        card = [e for e in rows if (e.week, e.captured_at) == latest]
        positive = [e for e in card if e.fair_prob is not None and e.model_prob > e.fair_prob]
        if positive:
            leans.append(max(positive, key=_quote_rank))
    return leans


def _fair(entry: LedgerEntry) -> float:
    return entry.fair_prob if entry.fair_prob is not None else float("nan")


def brier(entries: list[LedgerEntry], prob: Callable[[LedgerEntry], float]) -> float:
    return _mean([(prob(e) - (e.result == WIN)) ** 2 for e in entries])


def _model(entry: LedgerEntry) -> float:
    return entry.model_prob


def lean_lines(leans: list[LedgerEntry]) -> list[str]:
    out = []
    for market in MARKETS:
        rows = [e for e in leans if e.market == market]
        if not rows:
            continue
        wins = _wins(rows)
        out.append(
            f"  {market:<9} n {len(rows):>3}  {wins}-{len(rows) - wins}"
            f"  win {wins / len(rows):.3f}  avg fair {_mean([_fair(e) for e in rows]):.3f}"
            f"  avg model {_mean([e.model_prob for e in rows]):.3f}"
            f"  Brier model {brier(rows, _model):.4f} fair {brier(rows, _fair):.4f}"
            f"  units@taken price {sum(e.pnl for e in rows):+.2f}"
        )
        for lo, hi in GAP_BANDS:
            band = [e for e in rows if lo <= e.model_prob - _fair(e) < hi]
            if band:
                bw = _wins(band)
                out.append(
                    f"      gap {lo:.2f}-{hi:.2f}: n {len(band):>3}  {bw}-{len(band) - bw}"
                    f"  win {bw / len(band):.3f} vs fair {_mean([_fair(e) for e in band]):.3f}"
                    f"  units {sum(e.pnl for e in band):+.2f}"
                )
        if market == "total":
            overs = [e for e in rows if e.side == "over"]
            ow = _wins(overs)
            out.append(
                f"      model leans over {len(overs)}/{len(rows)}; overs {ow}-{len(overs) - ow}"
            )
    out.append("  by week:")
    for week in sorted({(e.season, e.week) for e in leans}):
        rows = [e for e in leans if (e.season, e.week) == week]
        wins = _wins(rows)
        out.append(
            f"    {week[0]} wk{week[1]}: {wins}-{len(rows) - wins}"
            f"  Brier model {brier(rows, _model):.4f} fair {brier(rows, _fair):.4f}"
        )
    return out


def screen_lines(leans: list[LedgerEntry]) -> list[str]:
    by: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in leans:
        for screen in filter(None, (s.strip() for s in (e.screens or "").split(";"))):
            by[screen].append(e)
    out = []
    for screen, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        wins = _wins(rows)
        out.append(
            f"  {screen:<20} n {len(rows):>3}  {wins}-{len(rows) - wins}"
            f"  win {wins / len(rows):.3f} vs fair {_mean([_fair(e) for e in rows]):.3f}"
            f"  units if bet {sum(e.pnl for e in rows):+.2f}"
        )
    return out


def close_coverage_lines(entries: list[LedgerEntry]) -> list[str]:
    """How many graded rows carry a close seen within the window, per week."""
    minutes = int(CLOSE_WINDOW.total_seconds() // 60)
    out = [f"  CLV counts only closes captured within {minutes} min of kickoff; coverage by week:"]
    for season, week in sorted({(e.season, e.week) for e in entries}):
        rows = [e for e in entries if (e.season, e.week) == (season, week)]
        timely = sum(has_true_close(e) for e in rows)
        stamped = sum(e.clv is not None for e in rows)
        out.append(
            f"    {season} wk{week}: {timely} of {len(rows)} rows have a timely close"
            f" ({stamped} stamped at any age)"
        )
    return out


def is_underdog(entry: LedgerEntry) -> bool:
    if entry.market == "spread":
        return entry.line is not None and entry.line > 0
    return entry.market == "moneyline" and _fair(entry) < 0.5


def underdog_lines(leans: list[LedgerEntry]) -> list[str]:
    out = []
    for market in ("moneyline", "spread"):
        rows = [e for e in leans if e.market == market]
        dogs = [e for e in rows if is_underdog(e)]
        out.append(record_line(f"{market} underdog leans", dogs))
        out.append(record_line(f"{market} favourite leans", [e for e in rows if e not in dogs]))
    out.append("  underdog leans that won money:")
    winners = [e for e in leans if is_underdog(e) and e.pnl > 0]
    for e in sorted(winners, key=lambda e: (e.season, e.week, e.market, e.matchup)):
        rung = "" if e.line is None else f" {e.line:+g}"
        price = "" if e.odds is None else f" {e.odds:+.0f}"
        out.append(
            f"    wk{e.week} {e.matchup:<11} {e.market:<9} {e.side}{rung}{price} {e.book}"
            f"  model {e.model_prob:.3f} fair {_fair(e):.3f}  {e.pnl:+.2f}u"
            f"  screens {e.screens}"
        )
    return out


def report(entries: list[LedgerEntry], filled: int) -> list[str]:
    graded = [e for e in entries if e.result in (WIN, LOSS, PUSH)]
    buys = [e for e in graded if e.tier != Tier.PASS.value]
    out = [
        f"rows {len(entries)}  graded {len(graded)} (graded here off nflverse: {filled})"
        f"  ungraded {len(entries) - len(graded)}",
        f"weeks: {sorted({(e.season, e.week) for e in entries})}",
        "",
        "1) Every buy the engine made",
    ]
    for e in sorted(buys, key=lambda e: (e.season, e.week)):
        rung = "" if e.line is None else f" {e.line:+g}"
        price = "" if e.odds is None else f" {e.odds:+.0f}"
        out.append(
            f"  wk{e.week} {e.matchup:<11} {e.market:<9} {e.side}{rung}{price} {e.book}"
            f"  {e.tier}  model {e.model_prob:.3f} fair {_fair(e):.3f}"
            f"  -> {e.result} {e.pnl:+.2f}  clv {e.clv}"
        )
    for tier in (Tier.STRONG, Tier.MODERATE):
        out.append(record_line(tier.value, [e for e in buys if e.tier == tier.value]))
    out.append(record_line("Buys", buys))
    out.append(
        record_line(
            "Pass rows (both sides, all books)", [e for e in graded if e.tier == Tier.PASS.value]
        )
    )
    leans = model_leans(graded)
    live = in_game(graded)
    pregame = [e for e in graded if id(e) not in set(map(id, live))]
    out += [
        "",
        "2) One model lean per game-market (side with model > fair)",
        f"  {len(live)} graded row(s) priced after kickoff are excluded from every lean figure"
        " below: in-game lines are not prices anyone could have bet",
        *lean_lines(leans),
    ]
    out += [
        "",
        "3) What each screen refused, graded on the model leans it blocked",
        *screen_lines(leans),
    ]
    out += ["", "4) CLV / drift on the leans", *close_coverage_lines(pregame)]
    out.append(record_line("all leans", leans))
    out += [record_line(f"leans {m}", [e for e in leans if e.market == m]) for m in MARKETS]
    out.append(
        "  close EV = the price taken, valued at the no-vig close (timely, two-sided closes"
        " only); it must be above 0 to beat the vig"
    )
    out.append(close_ev_line("all leans", leans))
    out += [close_ev_line(f"leans {m}", [e for e in leans if e.market == m]) for m in MARKETS]
    out += ["", "5) Underdog vs favourite leans (ML: fair < .5; spread: getting points)"]
    out += underdog_lines(leans)
    out += ["", f"6) Graduation bar (all four: {MIN_N}+ buys, ROI and CLV > 1 se, both halves up)"]
    out += graduation_lines(entries)
    out += ["", "7) Graded-only rules on the model leans (not bets; every screen still applies)"]
    out += tracked_lines(leans)
    return out


NFL_CUTS = {"end of Q1": 1, "half": 2, "end of Q3": 3}


def close_totals(entries: list[LedgerEntry]) -> dict[tuple[str, str], float]:
    """(matchup, date) -> the timely close total: per book, the Over rung whose two-sided
    close is nearest a coin flip; the median of those across books."""
    rungs: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    for e in entries:
        if e.market != "total" or e.side != "over" or e.line is None or e.close_prob is None:
            continue
        if not (has_true_close(e) and has_paired_close(e)):
            continue
        near = abs(e.close_prob - 0.5)
        held = rungs[(e.matchup, e.date)].get(e.book)
        if held is None or near < held[0]:
            rungs[(e.matchup, e.date)][e.book] = (near, e.line)
    return {k: st.median(line for _, line in by_book.values()) for k, by_book in rungs.items()}


def quarter_points(pbp: object) -> dict[tuple[str, str], list[int]]:
    """(``AWAY @ HOME``, game date) -> both teams' points in each of the four quarters."""
    out: dict[tuple[str, str], list[int]] = {}
    plays = pbp[
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "qtr",
            "total_home_score",
            "total_away_score",
        ]
    ]  # type: ignore[index]
    plays = plays.dropna(subset=["total_home_score", "total_away_score"])
    for (_gid, day, home, away), g in plays.groupby(
        ["game_id", "game_date", "home_team", "away_team"]
    ):
        pts = g.total_home_score + g.total_away_score
        upto = [int(pts[g.qtr <= q].max()) if (g.qtr <= q).any() else 0 for q in range(1, 5)]
        out[(f"{away} @ {home}", str(day))] = [
            upto[0],
            *(b - a for a, b in zip(upto, upto[1:], strict=False)),
        ]
    return out


def fast_start_reads(
    entries: list[LedgerEntry], quarters: dict[tuple[str, str], list[int]]
) -> list[faststart.Read]:
    finals = {
        (e.matchup, e.date): float(e.home_score + e.away_score)
        for e in entries
        if e.home_score is not None and e.away_score is not None
    }
    by_cut: dict[str, list[faststart.Obs]] = {c: [] for c in NFL_CUTS}
    for key, total in close_totals(entries).items():
        if key not in finals or key not in quarters:
            continue
        gid = f"{key[1]} {key[0]}"
        for cut, o in faststart.checkpoints(
            gid, key[1], total, quarters[key], finals[key], NFL_CUTS
        ).items():
            by_cut[cut].append(o)
    return [r for c, obs in by_cut.items() if (r := faststart.read(c, obs)) is not None]


def fast_start_lines(entries: list[LedgerEntry]) -> list[str]:
    out = [
        "",
        "8) Fast starts: does the early score stay in the final? (one row per game, timely close)",
    ]
    try:
        quarters: dict[tuple[str, str], list[int]] = {}
        for season in sorted({e.season for e in entries}):
            quarters.update(quarter_points(nflverse.play_by_play(season)))
        reads = fast_start_reads(entries, quarters)
    except Exception as exc:  # noqa: BLE001 - a study table must not stop the audit
        return [*out, f"  unavailable ({exc})"]
    out.append("  " + " | ".join(faststart.HEAD))
    out += ["  " + " | ".join(r) for r in faststart.table(reads)]
    if not reads:
        out.append(
            f"  needs {faststart.MIN_PRIOR}+ earlier games with a timely close and a final first"
        )
    return [*out, "  " + faststart.NOTE]


def email_audit(text: str, *, season: int | None, ledger_csv: bytes, to: str | None) -> str:
    """Send the audit text as the body, with it and the ledger CSV attached."""
    from nfl_engine.output.email import send_package

    label = "all seasons" if season is None else str(season)
    return send_package(
        load_config(),
        subject=f"NFL ledger audit -- {label} [paper]",
        html_body=f'<pre style="font-family:monospace">{html.escape(text)}</pre>',
        text_body=text,
        to=to,
        attachments=[
            (f"nfl_audit_{label.replace(' ', '_')}.txt", text.encode("utf-8")),
            ("nfl_ledger.csv", ledger_csv),
        ],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--email", action="store_true", help="email the report and the audit")
    parser.add_argument("--to", default=None, help="override the recipient")
    args = parser.parse_args(argv)
    season = args.season
    if season is None and args.email:
        season = current_week()[0]
    path = ledger_path()
    entries = [
        e
        for e in load_ledger(path)
        if e.source == ENGINE and (season is None or e.season == season)
    ]
    filled = settle_missing(entries, season)
    lines = [f"ledger {path}"]
    if args.email:
        lines += ["", "nfl-engine report --all", *report_lines(entries), ""]
    lines += report(entries, filled)
    lines += fast_start_lines(entries)
    text = "\n".join(lines)
    print(text)
    if not args.email:
        return
    if not entries:
        print(f"no {season} rows; nothing emailed")
        return
    from nfl_engine.output.email import EmailNotConfigured

    try:
        recipient = email_audit(text, season=season, ledger_csv=path.read_bytes(), to=args.to)
    except EmailNotConfigured as exc:
        print(f"audit not emailed ({exc})")
        return
    print(f"audit emailed to {recipient}")


if __name__ == "__main__":
    main()
