"""Graded-only schedule bands: VSiN's two rest spots that reproduced, read at the card's price.

Steve Makinen's 2026-27 scheduling list was rebuilt on 2023-26 (3,701 games,
ESPN BET closes). Two spots held at about 2 SE and have a mechanism; the rest
were n<20, undefined, or came out the other way. They are read here as bands,
flat 1u on the main line the card had, one row per game: nothing is priced,
gated or tiered by them.

* ``rest_host_ats``: visitor on one day's rest, host on 3+ days' rest -> home
  side of the spread (rebuilt 34-21 ATS).
* ``rested_visitor_under``: visitor on two days' rest, host's 3rd home game in
  4 days -> Under the total (rebuilt 26-10).

Rest is counted from the schedule the engine already archives (ESPN finals
under ``results/``) plus the slates' own graded games, so a team's first game
of the season has no rest figure and never qualifies. ``results/`` is written
by grading on the production Mac and is not state-synced; on a machine without
it, unknown rest keeps a game out of a band rather than guessing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from engine_common.bands import CALL_SE, MIN_N, BandRead
from nba_engine.audit.ledger import LedgerRow
from nba_engine.data.boxes import read_results

LOOKBACK = timedelta(days=30)  # covers any rest gap a band asks about, All-Star break included


@dataclass
class Schedule:
    """Each team's game days (and home days), season to date."""

    days: dict[str, set[Date]] = field(default_factory=lambda: defaultdict(set))
    home_days: dict[str, set[Date]] = field(default_factory=lambda: defaultdict(set))

    def add(self, day: Date, away: str, home: str) -> None:
        self.days[away].add(day)
        self.days[home].add(day)
        self.home_days[home].add(day)

    def rest(self, team: str, day: Date) -> int | None:
        """Days off before ``day``: 0 on a back-to-back; ``None`` with no prior game known."""
        prior = [d for d in self.days.get(team, ()) if d < day]
        return (day - max(prior)).days - 1 if prior else None

    def nth_home_in(self, team: str, day: Date, window: int) -> int:
        """How many home games, this one included, fall in the trailing ``window`` days."""
        return sum(1 for d in self.home_days.get(team, ()) if d <= day and (day - d).days < window)


def schedule_from(data_dir: Path, rows: Iterable[LedgerRow]) -> Schedule:
    """The archived results before each slate plus the slates' own matchups."""
    sched = Schedule()
    slates = sorted({Date.fromisoformat(r.slate_date) for r in rows})
    if not slates:
        return sched
    day = slates[0] - LOOKBACK
    while day <= slates[-1]:
        for g in read_results(data_dir, day) or []:
            if g.season_type in (0, 2):
                sched.add(g.game_date, g.away, g.home)
        day += timedelta(days=1)
    for r in rows:
        if not r.graded:  # a postponed or voided game was not played
            continue
        away, _, home = r.matchup.partition(" @ ")
        sched.add(Date.fromisoformat(r.slate_date), away, home)
    return sched


def _main(rows: list[LedgerRow]) -> LedgerRow | None:
    """The graded row at the line whose no-vig is nearest a coin flip."""
    priced = [r for r in rows if r.graded and r.pnl is not None and r.fair is not None]
    return min(priced, key=lambda r: abs((r.fair or 0.5) - 0.5), default=None)


def _rest_host_ats(sched: Schedule, day: Date, away: str, home: str) -> bool:
    h, a = sched.rest(home, day), sched.rest(away, day)
    return a == 1 and h is not None and h >= 3


def _rested_visitor_under(sched: Schedule, day: Date, away: str, home: str) -> bool:
    return sched.rest(away, day) == 2 and sched.nth_home_in(home, day, 4) >= 3


BANDS = (
    (
        "rest_host_ats: visitor 1 day rest vs host 3+ -> home ATS",
        "game_ats",
        "home",
        _rest_host_ats,
    ),
    (
        "rested_visitor_under: visitor 2 days rest vs host 3rd home in 4 -> Under",
        "game_total",
        "under",
        _rested_visitor_under,
    ),
)


def read_bands(rows: Iterable[LedgerRow], sched: Schedule) -> list[BandRead]:
    rows = [r for r in rows if not r.entity]
    by_game: dict[tuple[str, str, str, str], list[LedgerRow]] = defaultdict(list)
    for r in rows:
        by_game[(r.slate_date, r.matchup, r.market, r.side)].append(r)
    out: list[BandRead] = []
    for name, market, want, hit in BANDS:
        read = BandRead(name)
        for (slate, matchup, mkt, side), rs in sorted(by_game.items()):
            if mkt != market:
                continue
            away, _, home = matchup.partition(" @ ")
            if side != (home if want == "home" else want):
                continue
            day = Date.fromisoformat(slate)
            if not hit(sched, day, away, home):
                continue
            main = _main(rs)
            if main is not None and main.pnl is not None:
                read.pnls.append(main.pnl)
                read.games.append(main.pnl)
        out.append(read)
    return out


def lines(reads: list[BandRead]) -> list[str]:
    out = [
        f"[schedule bands: graded only, flat 1u at the card's price; call at {MIN_N}+ games, {CALL_SE:g} SE]"
    ]
    for b in reads:
        se = f"{b.se:.1f}" if b.n > 1 else "n/a"
        out.append(
            f"  {b.name}\n    n {b.n:4d}  {b.record:>7}  units {b.units:+.1f}  ROI {100 * b.roi:+.1f}%"
            f"  se {se}  halves {b.halves[0]:+.1f} / {b.halves[1]:+.1f}  -> {b.verdict}"
        )
    return out


__all__ = ["BANDS", "Schedule", "lines", "read_bands", "schedule_from"]
