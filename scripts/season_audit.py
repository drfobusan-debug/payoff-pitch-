"""Audit a season of the totals sheet and the daily worksheet, and file it for next year.

The morning audits grade one day and print a running tally that is gone the next
morning. This script reads both ledgers, ``totals_ledger.csv`` and
``worksheet_ledger.csv``, grades any finished game still pending (in memory
only), and writes a dated record under ``docs/mlb_audits/<season>/``:

* ``totals_worksheet_audit.md`` -- the season's records with 95% ranges and a
  verdict against the price each tool has to beat;
* ``totals_ledger.csv`` and ``worksheet_ledger.csv`` -- the season's rows as
  graded, so the record survives whatever happens to the state branch.

The totals sheet is judged against -110 both ways and against simply taking the
Over on every game; the worksheet's favoured side against the no-vig market
probability the ledger recorded for it.

The source ledgers are never written. Nothing here feeds a price.

Usage:
    python -m scripts.season_audit                       # this season, local audit dir
    python -m scripts.season_audit --engine-state        # read origin/engine-state
    python -m scripts.season_audit --season 2026 --no-grade
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb_engine.config import load_config  # noqa: E402
from mlb_engine.output import daily_worksheet as ws  # noqa: E402
from mlb_engine.output import totals_audit as ta  # noqa: E402

log = logging.getLogger("season_audit")

REPO = Path(__file__).resolve().parents[1]
STATE_PREFIX = "mlb"
Z = 1.96
POWER_Z = 0.84  # 80% power


# --- shared arithmetic --------------------------------------------------------------


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def rng(hits: int, n: int) -> str:
    if not n:
        return "-"
    lo, hi = ta.wilson(hits, n)
    return f"{100 * lo:.0f}-{100 * hi:.0f}%"


def payout(american: float) -> float:
    """Profit on a 1u stake that wins at ``american``."""
    return american / 100 if american > 0 else 100 / -american


@dataclass
class MarketTest:
    """Wins on priced games against the no-vig probabilities the market gave them."""

    n: int = 0
    wins: int = 0
    expected: float = 0.0
    variance: float = 0.0
    units: float = 0.0
    staked: int = 0

    def add(self, won: bool, p: float, american: float | None) -> None:
        self.n += 1
        self.wins += won
        self.expected += p
        self.variance += p * (1 - p)
        if american is not None:
            self.staked += 1
            self.units += payout(american) if won else -1.0

    @property
    def market(self) -> float | None:
        return self.expected / self.n if self.n else None

    @property
    def rate(self) -> float | None:
        return self.wins / self.n if self.n else None

    @property
    def z(self) -> float | None:
        return (self.wins - self.expected) / math.sqrt(self.variance) if self.variance else None

    def verdict(self) -> str:
        z = self.z
        if self.n < ta.MIN_GRADED or z is None:
            return f"unproven (n<{ta.MIN_GRADED})"
        if z >= Z:
            return "beats market"
        if z <= -Z:
            return "trails market"
        return "unproven"

    def games_needed(self) -> int | None:
        """Priced games for the current gap over the market to clear 95% at 80% power."""
        r, m = self.rate, self.market
        if r is None or m is None or r <= m:
            return None
        return math.ceil((Z + POWER_Z) ** 2 * m * (1 - m) / (r - m) ** 2)


# --- totals sheet -------------------------------------------------------------------


def lean_tally(rows: list[ta.LedgerRow], lean: str) -> ta.Tally:
    t = ta.Tally()
    for r in rows:
        if r.graded and r.current and r.lean == lean:
            t.add(r.hit, r.result == ta.PUSH)
    return t


def _trow(name: str, t: ta.Tally, verdict: bool = True) -> str:
    v = f" {ta.verdict(t)} |" if verdict else " |"
    return f"| {name} | {t.text()} | {rng(t.hits, t.n)} |{v}"


def totals_section(rows: list[ta.LedgerRow]) -> list[str]:
    s = ta.summarize(rows)
    pending = sum(r.pending and r.current and r.line is not None for r in rows)
    lineless = sum(r.pending and r.current and r.line is None for r in rows)
    out = [
        "## Totals sheet",
        "",
        f"{s.days} days, {s.games} graded games on bands `{ta.BANDS}`"
        + (f"; {s.legacy} older-band rows on record, not counted" if s.legacy else "")
        + (f"; {pending} still ungraded" if pending else "")
        + (f"; {lineless} with no line on record, never graded" if lineless else "")
        + f". Break-even at -110 is {100 * ta.BREAK_EVEN:.1f}%; a slice is an edge only when its whole"
        f" 95% range clears it on {ta.MIN_GRADED}+ decided games.",
        "",
        "| Slice | Record | 95% range | Verdict |",
        "|---|---|---|---|",
        _trow("SUM sign as the call", s.sign),
        _trow("Always the Over, same games", s.over_rate),
        _trow("SUM says Over", lean_tally(rows, ta.OVER)),
        _trow("SUM says Under", lean_tally(rows, ta.UNDER)),
        _trow(f"Top-{ta.RANK_N} Overs each day", s.rank_over),
        _trow(f"Bottom-{ta.RANK_N} Unders each day", s.rank_under),
    ]
    out += [_trow(f"\\|SUM\\| {name}", t) for name, t in s.by_mag.items()]
    out += [
        _trow(f"Watch Over (SUM +{ta.FLAG_OVER_SUM[0]}..{ta.FLAG_OVER_SUM[1]}, total <= {ta.FLAG_OVER_MAX_LINE:g})", s.flag_over),
        _trow(f"Watch Under (SUM {ta.FLAG_UNDER_SUM[0]}..{ta.FLAG_UNDER_SUM[1]}, total <= {ta.FLAG_UNDER_MAX_LINE:g})", s.flag_under),
        _trow("Engine's side of the sheet's line", s.engine),
        _trow("Engine's side vs the market", s.edge),
        _trow("Sheet call, engine agrees", s.agree),
        _trow("Sheet call, engine disagrees", s.split_sheet),
        _trow("Engine call, engine disagrees", s.split_engine),
        "",
        "Over rate by SUM (the slate's own base rate inside each band):",
        "",
        "| SUM | Overs-Unders-Pushes | Over rate |",
        "|---|---|---|",
    ]
    out += [f"| {name} | {t.text()} | {rng(t.hits, t.n)} |" for name, t in s.by_bucket.items() if t.n or t.pushes]
    out += ["", "By month:", "", "| Month | Games | SUM sign | Over rate |", "|---|---|---|---|"]
    for month in sorted({r.date[:7] for r in rows if r.graded and r.current}):
        m = ta.summarize([r for r in rows if r.date.startswith(month)])
        out.append(f"| {month} | {m.games} | {m.sign.text()} | {m.over_rate.text()} |")
    return out


# --- daily worksheet ----------------------------------------------------------------


def fav_price(r: ws.LedgerRow) -> float | None:
    return r.away_ml if r.fav == r.away else r.home_ml


def market_test(rows: list[ws.LedgerRow]) -> MarketTest:
    t = MarketTest()
    for r in rows:
        if r.fav_implied is not None:
            t.add(r.result == "fav", r.fav_implied, fav_price(r))
    return t


def counted(rows: list[ws.LedgerRow]) -> list[ws.LedgerRow]:
    """The rows the worksheet's own Audit tab counts."""
    return [r for r in rows if r.result in ("fav", "dog") and r.complete and r.weights == ws.WEIGHTS_VERSION]


def worksheet_section(rows: list[ws.LedgerRow]) -> list[str]:
    cur = counted(rows)
    older = sum(r.result in ("fav", "dog") and r.weights != ws.WEIGHTS_VERSION for r in rows)
    pending = sum(not r.graded and r.complete for r in rows)
    span = f"{min(r.date for r in cur)} to {max(r.date for r in cur)}" if cur else "no graded games"
    out = [
        "## Daily worksheet",
        "",
        f"Weights `{ws.WEIGHTS_VERSION}` ({span}): {len(cur)} graded games with both starters scored"
        + (f"; {older} graded rows on older weights, not counted" if older else "")
        + (f"; {pending} still ungraded" if pending else "")
        + ". The call is the side with the higher wTOTAL; priced games are judged against the no-vig"
        " probability the market gave that side when the worksheet was written.",
        "",
        "| Gap band | Fav W-L | Win % (95%) | Priced fav W-L | Priced win % (95%) | Market | Flat 1u ML | Fav run line | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    bands = [(name, [r for r in cur if ws.band_of(r.gap) == name]) for name, _, _ in ws.GAP_BANDS]
    for name, sub in bands + [("**All**", cur)]:
        t = ws.tally(sub)[-1]
        m = market_test(sub)
        units = f"{m.units:+.2f}u on {m.staked}" if m.staked else "-"
        out.append(
            f"| {name} | {t.fav_wins}-{t.n - t.fav_wins} | {pct(t.win_rate)} ({rng(t.fav_wins, t.n)}) | "
            f"{m.wins}-{m.n - m.wins} | {pct(m.rate)} ({rng(m.wins, m.n)}) | {pct(m.market)} | {units} | "
            f"{t.rl_fav_covers}-{t.rl_n - t.rl_fav_covers} | {m.verdict()} |"
        )
    m = market_test(cur)
    if m.z is not None:
        need = m.games_needed()
        out += [
            "",
            f"Priced favourites won {m.wins} against {m.expected:.1f} expected at the market's prices"
            f" (z = {m.z:+.2f})."
            + (f" Confirming the current gap at 95% with 80% power takes about {need} priced games." if need else ""),
        ]
    out += ["", "By month:", "", "| Month | Fav W-L | Priced W-L vs market |", "|---|---|---|"]
    for month in sorted({r.date[:7] for r in cur}):
        sub = [r for r in cur if r.date.startswith(month)]
        w = sum(r.result == "fav" for r in sub)
        mm = market_test(sub)
        out.append(f"| {month} | {w}-{len(sub) - w} | {mm.wins}-{mm.n - mm.wins} vs {pct(mm.market)} |")
    return out


# --- io -----------------------------------------------------------------------------


def from_engine_state(dest: Path, branch: str) -> Path:
    """Copy both ledgers off ``origin/<branch>`` without checking it out."""
    subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin", branch], check=True)
    for name in (ta.LEDGER_NAME, ws.LEDGER_NAME):
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"origin/{branch}:{STATE_PREFIX}/{name}"],
            check=True, capture_output=True,
        ).stdout
        (dest / name).write_bytes(blob)
    return dest


def grade_totals(rows: list[ta.LedgerRow], today: Date) -> int:
    n = 0
    for day in sorted({Date.fromisoformat(r.date) for r in rows if r.pending and r.date < today.isoformat()}):
        try:
            n += ta.grade(rows, ta.finals(day))
        except Exception as exc:  # noqa: BLE001
            log.warning("totals: finals for %s unavailable: %s", day, exc)
    return n


def report(season: int, totals: list[ta.LedgerRow], sheet: list[ws.LedgerRow], source: str) -> str:
    days = [r.date for r in totals] + [r.date for r in sheet]
    through = max(days) if days else "-"
    lines = [
        f"# MLB {season}: totals sheet and daily worksheet audit",
        "",
        f"Through {through}, from {source}. Regenerate with `python -m scripts.season_audit --season {season}"
        " --engine-state`; the graded ledgers beside this file are the rows it counted.",
        "",
        *totals_section(totals),
        "",
        *worksheet_section(sheet),
        "",
    ]
    return "\n".join(lines)


def run(
    season: int, ledger_dir: Path, out_dir: Path, source: str, grade: bool = True, today: Date | None = None
) -> Path:
    today = today or Date.today()
    prefix = f"{season}-"
    totals = [r for r in ta.read_ledger(ledger_dir / ta.LEDGER_NAME) if r.date.startswith(prefix)]
    sheet = [r for r in ws.clean_ledger(ws.load_ledger(ledger_dir / ws.LEDGER_NAME)) if r.date.startswith(prefix)]
    if grade:
        log.info("graded %d totals and %d worksheet rows off the finals", grade_totals(totals, today), ws.grade_pending(sheet, today))
    out_dir.mkdir(parents=True, exist_ok=True)
    ta.write_ledger(out_dir / ta.LEDGER_NAME, totals)
    ws.save_ledger(out_dir / ws.LEDGER_NAME, sheet)
    path = out_dir / "totals_worksheet_audit.md"
    path.write_text(report(season, totals, sheet, source))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=Date.today().year)
    ap.add_argument("--ledger-dir", type=Path, default=None, help="directory holding both ledgers (default: audit dir)")
    ap.add_argument("--engine-state", action="store_true", help="read the ledgers off origin/engine-state")
    ap.add_argument("--out", type=Path, default=None, help="default: docs/mlb_audits/<season>")
    ap.add_argument("--no-grade", action="store_true", help="do not fill pending rows from the finals")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = args.out or REPO / "docs" / "mlb_audits" / str(args.season)
    with tempfile.TemporaryDirectory() as tmp:
        if args.engine_state:
            branch = load_config().state_branch
            src, label = from_engine_state(Path(tmp), branch), f"`origin/{branch}`"
        else:
            src = args.ledger_dir or load_config().audit_dir
            label = "the local audit ledgers"
        print(run(args.season, src, out, label, grade=not args.no_grade))
    return 0


if __name__ == "__main__":
    sys.exit(main())
