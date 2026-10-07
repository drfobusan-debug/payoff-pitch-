"""Who the power screen kept, and which arms it rated, whether or not a price existed.

The ledger (:mod:`mlb_engine.audit.power_ledger`) holds priced rows only, so a
survivor with no prop posted at screen time -- most of a lineup that has not
been announced -- left no trace, and "how did the screen's hitters do?" could
only be answered for the ones a book happened to quote. This file is the rest
of the receipt: one row per screened bat and per probable starter, per run,
with the bucket, the side, the arm he was screened against and the lineup slot
the screen assumed. It carries no price and is never graded for money.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROSTER_NAME = "power_screen_roster.csv"

BAT = "bat"
ARM = "arm"

#: Bat statuses.
HELD = "held"
DROPPED = "dropped"
LATE_CUT = "late cut"
#: Arm statuses.
SCREENED = "screened"
NOT_SCREENED = "not screened"
GATED = "gated"


@dataclass(frozen=True)
class RosterRow:
    date: str
    run_id: str
    #: ``soft`` or ``elite``: the pass that wrote the row.
    arm_tier: str
    #: ``bat`` or ``arm``.
    kind: str
    name: str
    player_id: int
    team: str
    game_pk: int | None = None
    #: A bat's opposing starter as the screen had him (an arm's row repeats itself).
    versus: str = ""
    versus_id: int | None = None
    #: The lineup slot the screen read, and whether that lineup was projected.
    slot: int | None = None
    projected: bool = False
    #: A bat's gate bucket (``power_report`` key) and the side it holds, if any.
    bucket: str = ""
    side: str = ""
    status: str = ""
    #: Why an arm was gated, or a bat cut.
    reason: str = ""
    siera: float | None = None

    @property
    def is_bat(self) -> bool:
        return self.kind == BAT


FIELDS = tuple(f.name for f in fields(RosterRow))


def _int(v: str) -> int | None:
    try:
        return int(float(v)) if v != "" else None
    except ValueError:
        return None


def _float(v: str) -> float | None:
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


def load(path: Path) -> list[RosterRow]:
    if not path.exists():
        return []
    out: list[RosterRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pid = _int(r.get("player_id", ""))
            if pid is None:
                continue
            out.append(
                RosterRow(
                    date=r.get("date", ""),
                    run_id=r.get("run_id", ""),
                    arm_tier=r.get("arm_tier", ""),
                    kind=r.get("kind", ""),
                    name=r.get("name", ""),
                    player_id=pid,
                    team=r.get("team", ""),
                    game_pk=_int(r.get("game_pk", "")),
                    versus=r.get("versus", ""),
                    versus_id=_int(r.get("versus_id", "")),
                    slot=_int(r.get("slot", "")),
                    projected=str(r.get("projected", "")).lower() in ("true", "1", "yes"),
                    bucket=r.get("bucket", ""),
                    side=r.get("side", ""),
                    status=r.get("status", ""),
                    reason=r.get("reason", ""),
                    siera=_float(r.get("siera", "")),
                )
            )
    return out


def record(path: Path, rows: list[RosterRow], day: str, run_id: str) -> None:
    """Append one run's rows, replacing only what the same run wrote before."""
    kept = [r for r in load(path) if not (r.date == day and r.run_id == run_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(FIELDS))
        w.writeheader()
        for r in kept + rows:
            w.writerow({k: "" if v is None else v for k, v in asdict(r).items()})


def for_day(path: Path, day: str, run_id: str | None = None) -> list[RosterRow]:
    """One run's roster for the day: the last run unless ``run_id`` pins one."""
    rows = [r for r in load(path) if r.date == day]
    if not rows:
        return []
    want = max(r.run_id for r in rows) if run_id is None else run_id
    return [r for r in rows if r.run_id == want]
