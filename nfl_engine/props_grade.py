"""Grade the prop research rows against the box score, and say what it means.

The props layer prices archived quotes and stamps every row ``research_only``;
this module is the other half of the bargain -- the rows are graded against what
the player actually did, from nflverse weekly stats, so the question the
pseudo-line study could not ask ("can the number a book hung be beaten?") starts
accruing an answer. Nothing here touches the ledger: the graded file sits beside
the research file it was cut from, and a row's ``basis`` travels with it, so two
projection bases are two records.

Two things the summary keeps apart on purpose. *Probability quality* -- the Brier
of the model's probability and of the de-vigged fair probability against the
outcome, on every row with a line, beside the base rate of the same rows -- is
what says whether the projection knows anything the book does not. *Shadow
return* -- flat one unit at the captured price on the rows only ``research_only``
stopped -- is what a bettor would have seen, and is quoted with its count because
a week of it is noise. Neither is a reason to open a market on its own; the bar
in :mod:`nfl_engine.props` moves on the accumulated file, not on one week.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl_engine.config import data_dir
from nfl_engine.data import nflverse
from nfl_engine.features.usage import REGULAR, normalise
from nfl_engine.market.board import OVER
from nfl_engine.market.odds import american_to_decimal
from nfl_engine.models.player import STATS
from nfl_engine.props import FIELDS as RESEARCH_FIELDS
from nfl_engine.props import RESEARCH_ONLY, research_path

log = logging.getLogger(__name__)

WIN, LOSS, PUSH, VOID = "win", "loss", "push", "void"
NO_BOX_SCORE = "no_box_score"
NO_LINE = "no_line"


@dataclass(frozen=True)
class GradedProp:
    """One research row with the outcome attached."""

    captured_at: str
    season: int
    week: int
    matchup: str
    market: str
    player: str
    side: str
    line: float | None
    book: str
    american: float
    stat: str
    projection: float | None
    model_prob: float
    fair_prob: float | None
    ev_fair: float | None
    screens: str
    basis: str
    actual: float | None
    result: str
    reason: str
    pnl: float  # flat one unit at the captured price; 0 on push and void
    graded_at: str

    @property
    def shadow_bet(self) -> bool:
        """Only ``research_only`` stopped it: what a bettor would have taken."""
        return all(r == RESEARCH_ONLY for r in self.screens.split(";") if r)


GRADED_FIELDS = [f.name for f in fields(GradedProp)]


def graded_path(season: int, week: int, *, root: Path | None = None) -> Path:
    return (root or data_dir()) / "props" / f"graded_{season}_wk{week:02d}.csv"


def _float(value: str | None) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    return float(value)


def read_research(path: Path) -> list[dict[str, str]]:
    """The research rows, each run's exact duplicates collapsed to one."""
    if not path.exists():
        return []
    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row.get(name, "") for name in RESEARCH_FIELDS)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def box_scores(season: int, week: int) -> dict[tuple[str, str, str], float]:
    """``(normalised name, team, stat) -> actual`` for the week's played games."""
    frame = nflverse.player_week(season)
    if frame.empty:
        return {}
    if "season_type" in frame.columns:
        frame = frame[frame.season_type == REGULAR]
    frame = frame[frame.week == week]
    if frame.empty:
        return {}
    name_col = "player_display_name" if "player_display_name" in frame.columns else "player_name"
    stats = [stat for stat in STATS if stat in frame.columns]
    out: dict[tuple[str, str, str], float] = {}
    for row in frame[[name_col, "team", *stats]].to_dict("records"):
        name = normalise(str(row[name_col]))
        team = str(row["team"])
        for stat in stats:
            value = row[stat]
            if pd.notna(value):
                out[(name, team, stat)] = float(value)
    return out


def _actual(
    scores: dict[tuple[str, str, str], float], player: str, matchup: str, stat: str
) -> float | None:
    name = normalise(player)
    teams = [part.strip() for part in matchup.split("@")]
    for team in teams:
        value = scores.get((name, team, stat))
        if value is not None:
            return value
    return None


def grade_row(
    row: dict[str, str], scores: dict[tuple[str, str, str], float], now: str
) -> GradedProp:
    line = _float(row.get("line"))
    stat = row["stat"]
    actual = _actual(scores, row["player"], row["matchup"], stat)
    american = float(row["american"])
    result, reason, pnl = VOID, "", 0.0
    if line is None:
        reason = NO_LINE
    elif actual is None:
        reason = NO_BOX_SCORE
    elif actual == line:
        result = PUSH
    else:
        over_hit = actual > line
        won = over_hit if row["side"] == OVER else not over_hit
        result = WIN if won else LOSS
        pnl = american_to_decimal(american) - 1.0 if won else -1.0
    return GradedProp(
        captured_at=row["captured_at"],
        season=int(row["season"]),
        week=int(row["week"]),
        matchup=row["matchup"],
        market=row["market"],
        player=row["player"],
        side=row["side"],
        line=line,
        book=row["book"],
        american=american,
        stat=stat,
        projection=_float(row.get("projection")),
        model_prob=float(row["model_prob"]),
        fair_prob=_float(row.get("fair_prob")),
        ev_fair=_float(row.get("ev_fair")),
        screens=row.get("screens", ""),
        basis=row.get("basis", ""),
        actual=actual,
        result=result,
        reason=reason,
        pnl=round(pnl, 4),
        graded_at=now,
    )


def grade_week(
    season: int, week: int, *, root: Path | None = None, write: bool = True
) -> list[GradedProp]:
    """Grade the week's research file; an ungraded week (no box scores yet) is empty."""
    rows = read_research(research_path(season, week, root=root))
    if not rows:
        return []
    scores = box_scores(season, week)
    if not scores:
        log.info("props grade: no box scores yet for %d week %d", season, week)
        return []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    graded = [grade_row(row, scores, now) for row in rows]
    if write:
        path = graded_path(season, week, root=root)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=GRADED_FIELDS)
                writer.writeheader()
                for item in graded:
                    writer.writerow(asdict(item))
        except OSError as exc:
            log.warning("could not write graded props to %s: %s", path, exc)
    return graded


def pending_weeks(season: int, *, root: Path | None = None) -> list[int]:
    """Weeks with a research file and no graded file, oldest first."""
    folder = (root or data_dir()) / "props"
    if not folder.exists():
        return []
    weeks = []
    for path in sorted(folder.glob(f"research_{season}_wk*.csv")):
        week = int(path.stem.rsplit("wk", 1)[1])
        if not graded_path(season, week, root=root).exists():
            weeks.append(week)
    return weeks


def read_graded(season: int, *, root: Path | None = None) -> list[GradedProp]:
    folder = (root or data_dir()) / "props"
    out: list[GradedProp] = []
    for path in sorted(folder.glob(f"graded_{season}_wk*.csv")) if folder.exists() else []:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out.append(
                    GradedProp(
                        captured_at=row["captured_at"],
                        season=int(row["season"]),
                        week=int(row["week"]),
                        matchup=row["matchup"],
                        market=row["market"],
                        player=row["player"],
                        side=row["side"],
                        line=_float(row["line"]),
                        book=row["book"],
                        american=float(row["american"]),
                        stat=row["stat"],
                        projection=_float(row["projection"]),
                        model_prob=float(row["model_prob"]),
                        fair_prob=_float(row["fair_prob"]),
                        ev_fair=_float(row["ev_fair"]),
                        screens=row["screens"],
                        basis=row["basis"],
                        actual=_float(row["actual"]),
                        result=row["result"],
                        reason=row["reason"],
                        pnl=float(row["pnl"]),
                        graded_at=row["graded_at"],
                    )
                )
    return out


def _brier(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    return sum((p - hit) ** 2 for p, hit in pairs) / len(pairs)


def summary(graded: list[GradedProp]) -> list[str]:
    """Per basis and market: settled count, Brier of model / fair / base rate, and
    the shadow return on the rows only ``research_only`` stopped."""
    settled = [g for g in graded if g.result in (WIN, LOSS)]
    voids = sum(1 for g in graded if g.result == VOID)
    if not settled:
        return [f"props grade: nothing settled ({len(graded)} rows, {voids} void)"]
    lines = [
        f"props grade: {len(settled)} settled, {sum(1 for g in graded if g.result == PUSH)} push,"
        f" {voids} void -- research, not a record"
    ]
    keys = sorted({(g.basis, g.market) for g in settled})
    for basis, market in keys:
        rows = [g for g in settled if g.basis == basis and g.market == market]
        hits = [1 if g.result == WIN else 0 for g in rows]
        base = sum(hits) / len(hits)
        model = _brier(
            [(g.model_prob, h) for g, h in zip(rows, hits, strict=True) if g.projection is not None]
        )
        fair = _brier(
            [(g.fair_prob, h) for g, h in zip(rows, hits, strict=True) if g.fair_prob is not None]
        )
        base_brier = _brier([(base, h) for h in hits])
        shadow = [g for g in rows if g.shadow_bet]
        roi = sum(g.pnl for g in shadow) / len(shadow) if shadow else None
        lines.append(
            f"  {basis:44s} {market:26s} n={len(rows):4d}"
            f" brier model {model if model is not None else float('nan'):.4f}"
            f" fair {fair if fair is not None else float('nan'):.4f}"
            f" base {base_brier if base_brier is not None else float('nan'):.4f}"
            f" | shadow n={len(shadow):3d}"
            f" roi {roi if roi is not None else float('nan'):+.3f}"
        )
    return lines


__all__ = [
    "GRADED_FIELDS",
    "LOSS",
    "NO_BOX_SCORE",
    "PUSH",
    "VOID",
    "WIN",
    "GradedProp",
    "box_scores",
    "grade_row",
    "grade_week",
    "graded_path",
    "pending_weeks",
    "read_graded",
    "read_research",
    "summary",
]
