"""Do the two underdog zones the 2026 screens refused hold up on closing prices?

Replays every played week of each season through the live pricing path (the
same functions ``nfl-engine replay`` calls), keeps the rows in memory, and grades
the graded-only rules in :mod:`nfl_engine.audit.tracked`, applied to the same one
lean per game-market the weekly audit grades, with the de-vigged close as fair.

The replayed board has one book, so these rows test the model's lean only, not an
execution edge. Usage::

    python -m scripts.nfl.dog_zone_study --first 2006 --last 2025 --cache rows.pkl
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import replace
from pathlib import Path

from nfl_engine import calibration
from nfl_engine import replay as replay_mod
from nfl_engine.audit.ledger import LOSS, WIN, LedgerEntry, apply_close, entry_from_bet, grade
from nfl_engine.audit.tracked import RULES
from nfl_engine.cli import _closing_quote
from nfl_engine.features import books as books_mod
from nfl_engine.models.drives import DriveSim
from nfl_engine.pipeline import price_slate
from scripts.nfl.ledger_audit import model_leans


def season_rows(season: int, sims: int, ratings: bool) -> list[LedgerEntry]:
    sim = DriveSim(n_sims=sims, seed=17)
    maps = calibration.load()
    out: list[LedgerEntry] = []
    for week in replay_mod.played_weeks(season):
        book = books_mod.as_of(season, week.week) if ratings else books_mod.Books()
        pricings = price_slate(
            week.games,
            week.board,
            book=book.ratings,
            starters=book.starters,
            sim=sim,
            calibrator=maps,
        )
        for pricing in pricings:
            for bet in pricing.bets:
                entry = entry_from_bet(
                    bet,
                    season=season,
                    week=week.week,
                    date=pricing.game.game_date.isoformat(),
                    captured_at="",
                    kickoff_utc=pricing.game.kickoff_utc or "",
                )
                quote = _closing_quote(week.board, entry)
                if quote is not None:
                    apply_close(entry, quote[0], quote[1], captured_at="")
                final = week.finals.get(entry.matchup)
                if final is not None:
                    grade(entry, final[0], final[1], home=entry.matchup.split(" @ ")[-1])
                out.append(entry)
    return out


def summary(label: str, rows: list[LedgerEntry]) -> str:
    decided = [e for e in rows if e.result in (WIN, LOSS) and e.odds is not None]
    n = len(decided)
    if not n:
        return f"  {label:<44} n   0"
    wins = sum(e.result == WIN for e in decided)
    pnl = [e.pnl for e in decided]
    roi = sum(pnl) / n
    se = math.sqrt(sum((p - roi) ** 2 for p in pnl) / (n - 1) / n) if n > 1 else float("nan")
    need = sum(e.fair_prob or 0.0 for e in decided) / n
    return (
        f"  {label:<44} n {n:>4}  {wins}-{n - wins}  win {wins / n:.1%} (fair {need:.1%})"
        f"  units {sum(pnl):+.1f}  ROI {roi:+.1%} ± {se:.1%}  t {roi / se:+.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2006)
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--sims", type=int, default=5000)
    parser.add_argument("--no-ratings", dest="ratings", action="store_false", default=True)
    parser.add_argument("--cache", type=Path, default=None, help="reuse or write the rows here")
    args = parser.parse_args()

    if args.cache and args.cache.exists():
        rows: list[LedgerEntry] = pickle.loads(args.cache.read_bytes())
    else:
        rows = []
        for season in range(args.first, args.last + 1):
            got = season_rows(season, args.sims, args.ratings)
            print(f"{season}: {len(got)} rows", flush=True)
            rows += got
        if args.cache:
            args.cache.write_bytes(pickle.dumps(rows))

    # One book: the de-vigged close is the fair, exactly as the live audit uses consensus.
    closed = [replace(e, fair_prob=e.close_prob) for e in rows if e.close_prob is not None]
    leans = [e for e in model_leans(closed) if args.first <= e.season <= args.last]
    print(f"\n{args.first}-{args.last}: {len(leans)} model leans, graded at the closing price")
    for _, label, rule in RULES:
        print(summary(label, [e for e in leans if rule(e)]))
        mid = (args.first + args.last) // 2
        for lo, hi in ((args.first, mid), (mid + 1, args.last)):
            print(summary(f"  {lo}-{hi}", [e for e in leans if rule(e) and lo <= e.season <= hi]))
    print(
        summary(
            "every spread dog lean",
            [e for e in leans if e.market == "spread" and (e.line or 0) > 0],
        )
    )
    print(
        summary(
            "every ML dog lean",
            [e for e in leans if e.market == "moneyline" and (e.fair_prob or 1) < 0.5],
        )
    )


if __name__ == "__main__":
    main()
