"""Graded-only research bands: VSiN's two rest spots that reproduced, plus one exploratory
shooting spot, read at the card's price.

Steve Makinen's 2026-27 scheduling list was rebuilt on 2023-26 (3,701 games,
ESPN BET closes). Two spots held at about 2 SE and have a mechanism; the rest
were n<20, undefined, or came out the other way. They are read here as bands,
flat 1u on the main line the card had, one row per game: nothing is priced,
gated or tiered by them.

* ``rest_host_ats``: visitor on one day's rest, host on 3+ days' rest -> home
  side of the spread (rebuilt 34-21 ATS).
* ``rested_visitor_under``: visitor on two days' rest, host's 3rd home game in
  4 days -> Under the total (rebuilt 26-10).

* ``exploratory``: the ATS underdog whose 2P% gap (2P% made less 2P% allowed)
  beats the favourite's while its 3P% gap trails, once both teams have
  ``SHOOT_MIN_GAMES`` finals this season. Found post hoc (CBB, then NBA
  2023-26: 253-216, +1.7 SE, +2.9% at one book's close), so it is a lead, not a
  rule. Shooting is pooled over attempts from finals dated before the slate.

Rest is counted from the schedule the engine already archives (ESPN finals
under ``results/``) plus the slate's own games, so a team's first game of the
season has no rest figure and never qualifies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from engine_common.bands import CALL_SE, MIN_N, BandRead
from engine_common.promotion import close_roi
from nba_engine.audit.ledger import LedgerRow, timely_close
from nba_engine.data.boxes import read_results
from nba_engine.models.schedule import season_of
from nba_engine.schemas import GameResult

LOOKBACK = timedelta(days=10)  # the longest window any rest band asks about
SHOOT_MIN_GAMES = 10


@dataclass
class Schedule:
    """Each team's game days (and home days), season to date."""

    days: dict[str, set[Date]] = field(default_factory=lambda: defaultdict(set))
    home_days: dict[str, set[Date]] = field(default_factory=lambda: defaultdict(set))
    # (season, team) -> per final: (day, 2PM, 2PA, 3PM, 3PA, then the same allowed)
    shots: dict[tuple[int, str], list[tuple[Date, tuple[int, ...]]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, day: Date, away: str, home: str) -> None:
        self.days[away].add(day)
        self.days[home].add(day)
        self.home_days[home].add(day)

    def add_box(self, g: GameResult) -> None:
        if g.away_box is None or g.home_box is None:
            return
        season = season_of(g.game_date)
        for mine, theirs in ((g.away_box, g.home_box), (g.home_box, g.away_box)):
            counts = tuple(
                v for b in (mine, theirs) for v in (b.fgm - b.fg3m, b.fga - b.fg3a, b.fg3m, b.fg3a)
            )
            self.shots[(season, mine.team)].append((g.game_date, counts))

    def shooting(self, team: str, day: Date) -> tuple[float, float] | None:
        """(2P% gap, 3P% gap), each made less allowed, pooled over this season's finals before ``day``;
        ``None`` under ``SHOOT_MIN_GAMES``."""
        prior = [c for d, c in self.shots.get((season_of(day), team), ()) if d < day]
        if len(prior) < SHOOT_MIN_GAMES:
            return None
        m2, a2, m3, a3, om2, oa2, om3, oa3 = (sum(col) for col in zip(*prior, strict=True))
        if not (a2 and a3 and oa2 and oa3):
            return None
        return m2 / a2 - om2 / oa2, m3 / a3 - om3 / oa3

    def rest(self, team: str, day: Date) -> int | None:
        """Days off before ``day``: 0 on a back-to-back; ``None`` with no prior game known."""
        prior = [d for d in self.days.get(team, ()) if d < day]
        return (day - max(prior)).days - 1 if prior else None

    def nth_home_in(self, team: str, day: Date, window: int) -> int:
        """How many home games, this one included, fall in the trailing ``window`` days."""
        return sum(1 for d in self.home_days.get(team, ()) if d <= day and (day - d).days < window)


def schedule_from(data_dir: Path, rows: Iterable[LedgerRow]) -> Schedule:
    """The archived results before each slate (from its season's start) plus the slates' own matchups."""
    sched = Schedule()
    slates = sorted({Date.fromisoformat(r.slate_date) for r in rows})
    if not slates:
        return sched
    day = min(slates[0] - LOOKBACK, Date(season_of(slates[0]), 10, 1))
    while day <= slates[-1]:
        for g in read_results(data_dir, day) or []:
            if g.season_type in (0, 2):
                sched.add(g.game_date, g.away, g.home)
            if g.is_final and g.season_type != 1:
                sched.add_box(g)
        day += timedelta(days=1)
    for r in rows:
        away, _, home = r.matchup.partition(" @ ")
        sched.add(Date.fromisoformat(r.slate_date), away, home)
    return sched


def _main(rows: list[LedgerRow]) -> LedgerRow | None:
    """The graded row at the line whose no-vig is nearest a coin flip."""
    priced = [r for r in rows if r.graded and r.pnl is not None and r.fair is not None]
    return min(priced, key=lambda r: abs((r.fair or 0.5) - 0.5), default=None)


def _rest_host_ats(sched: Schedule, day: Date, away: str, home: str, side: str) -> bool:
    h, a = sched.rest(home, day), sched.rest(away, day)
    return a == 1 and h is not None and h >= 3


def _rested_visitor_under(sched: Schedule, day: Date, away: str, home: str, side: str) -> bool:
    return sched.rest(away, day) == 2 and sched.nth_home_in(home, day, 4) >= 3


def _inside_not_outside(sched: Schedule, day: Date, away: str, home: str, side: str) -> bool:
    """``side``'s 2P% gap beats the opponent's while its 3P% gap trails."""
    me, them = sched.shooting(side, day), sched.shooting(home if side == away else away, day)
    if me is None or them is None:
        return False
    return me[0] > them[0] and me[1] < them[1]


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
    (
        f"exploratory: underdog better 2P%, worse 3P% ({SHOOT_MIN_GAMES}+ games) ATS",
        "game_ats",
        "dog",
        _inside_not_outside,
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
            if want == "dog":
                if side not in (away, home):
                    continue
            elif side != (home if want == "home" else want):
                continue
            main = _main(rs)
            if want == "dog" and (main is None or main.line is None or main.line <= 0):
                continue
            if not hit(sched, Date.fromisoformat(slate), away, home, side):
                continue
            if main is not None and main.pnl is not None:
                read.pnls.append(main.pnl)
                read.games.append(main.pnl)
                if (
                    main.close_fair is not None
                    and main.american is not None
                    and timely_close(main)
                    and main.pnl != 0
                ):
                    read.close_rois.append(close_roi(main.close_fair, main.american))
        out.append(read)
    return out


def lines(reads: list[BandRead]) -> list[str]:
    out = [
        f"[schedule and shooting bands: graded only, flat 1u at the card's price; call at {MIN_N}+ games, {CALL_SE:g} SE]"
    ]
    for b in reads:
        se = f"{b.se:.1f}" if b.n > 1 else "n/a"
        out.append(
            f"  {b.name}\n    n {b.n:4d}  {b.record:>7}  units {b.units:+.1f}  ROI {100 * b.roi:+.1f}%"
            f"  se {se}  halves {b.halves[0]:+.1f} / {b.halves[1]:+.1f}  {b.price_read}  -> {b.verdict}"
        )
    return out


__all__ = ["BANDS", "Schedule", "lines", "read_bands", "schedule_from"]
