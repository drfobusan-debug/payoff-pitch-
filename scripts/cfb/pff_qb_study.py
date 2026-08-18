"""Does a PFF passing grade tell the closing spread anything it does not know?

The case for buying PFF was never team quality -- SP+ and the number already hold
that -- it was pricing an absence: charging the actual gap between the starter and
the man who replaces him instead of a flat points hit. This grades that claim on
exported ``passing_summary`` files, using the *prior* season's grade only, so a
grade never prices the games it was computed from.

Two questions, both null on 2022-2024 grades against 2023-2025 games:

A. Starter's prior-season grade (z-scored against FBS passers with 100+
   dropbacks) vs the closing spread's residual, over 1,899 team-games::

       all rows            r=-0.0187  t=-0.81   -0.25 pts per z
       weeks 1-4           r=-0.0626  t=-1.31   -0.84 pts per z
       200+ dropbacks      r=-0.0031  t=-0.11   -0.05 pts per z
       back a +1z starter  48.51% ATS (n=404)
       fade a -1z starter  52.68% ATS (n=317)

   The sign is backwards and the magnitude is inside the noise; at this n a real
   one-point-per-z effect (r~.07) would have shown.

B. The absence gap, on the 173 absences in that window::

       both men graded            n= 61   gap vs residual r=+0.045  t=+0.34
       gap >= 10 grade points     n= 37   mean residual -0.92   fade 48.65%
       gap < 10                   n= 24   mean residual -3.44   fade 62.50%
       replacement never graded   n=112   mean residual -3.97   fade 60.36%

   Also backwards: the team falls *further* short of the number when the drop in
   grade is small, and the absences where the replacement has no grade at all are
   the worst of the three -- which is a statement about playing an unknown, not
   about the grade of the man who is out.

The structural problem is visible in those counts. Only 35% of absences have a
prior-season grade for the replacement, because a backup by definition has not
played -- so the input is missing precisely where the question is asked. A
season-grade export cannot fix that; it would need the grade to be attached to a
player before he has snaps, which is what recruiting and portal ratings claim to
do and this file does not contain.

The join is bootstrapped rather than hand-written: PFF spells teams as broadcast
abbreviations ("BOWL GREEN") and names with nicknames and suffixes ("Cam Ward",
"Billy Edwards Jr."), so each PFF team is mapped to the CFBD team whose passer
surnames overlap it most, and players are matched inside the team on surname plus
first initial. That lands 99% of FBS passers with 10+ attempts (the residual
misses are FCS teams, which PFF's FBS export does not cover).

Usage, with one ``passing_<season>.csv`` per season in the directory::

    CFBD_API_KEY=... CFBE_PFF_DIR=~/pff python scripts/cfb/pff_qb_study.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import statistics
import sys

from scripts.cfb.qb_absence_study import frame

PFF_DIR = pathlib.Path(os.getenv("CFBE_PFF_DIR", pathlib.Path.home() / "pff"))
SEASONS = (2023, 2024, 2025)
MIN_DROPBACKS = 100  # the pool the z-score is taken against
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# -- the join --------------------------------------------------------------
def _norm(name: str) -> list[str]:
    cleaned = name.replace(".", " ").replace("'", "").replace("-", " ").lower()
    return [part for part in cleaned.split() if part not in SUFFIXES]


def _handle(name: str) -> tuple[str, str]:
    """Surname and first initial -- the most a nickname leaves intact."""
    parts = _norm(name)
    return (parts[-1], parts[0][0]) if parts else ("", "")


def _pff(season: int) -> list[dict[str, str]]:
    path = PFF_DIR / f"passing_{season}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; export PFF passing_summary for {season}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _team_map(season: int, cfbd_names: dict[str, set[str]]) -> dict[str, str]:
    """PFF ``team_name`` -> CFBD team key, by surname overlap."""
    pff_names: dict[str, set[str]] = {}
    for row in _pff(season):
        pff_names.setdefault(row["team_name"], set()).add(_handle(row["player"])[0])
    ranked = sorted(
        (
            (len(names & cfbd), pff_team, cfbd_team)
            for pff_team, names in pff_names.items()
            for cfbd_team, cfbd in cfbd_names.items()
            if names & cfbd
        ),
        reverse=True,
    )
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for _overlap, pff_team, cfbd_team in ranked:
        if pff_team not in mapping and cfbd_team not in taken:
            mapping[pff_team] = cfbd_team
            taken.add(cfbd_team)
    return mapping


class GradeBook:
    """One season of PFF passing grades, keyed the way CFBD names passers."""

    def __init__(self, season: int, cfbd_names: dict[str, set[str]]) -> None:
        mapping = _team_map(season, cfbd_names)
        self.rows: dict[tuple[str, str, str], dict[str, str]] = {}
        pool: list[float] = []
        for row in _pff(season):
            team = mapping.get(row["team_name"])
            grade = row["grades_pass"]
            if team is None or not grade:
                continue
            surname, initial = _handle(row["player"])
            self.rows[(team, surname, initial)] = row
            if float(row["dropbacks"] or 0) >= MIN_DROPBACKS:
                pool.append(float(grade))
        self.mean = statistics.fmean(pool)
        self.sigma = statistics.pstdev(pool)

    def get(self, team: str, name: str) -> dict[str, str] | None:
        surname, initial = _handle(name)
        return self.rows.get((team, surname, initial))

    def z(self, grade: float) -> float:
        return (grade - self.mean) / self.sigma


# -- statistics ------------------------------------------------------------
def _corr(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Pearson r, its t statistic, and the slope of y on x."""
    if len(xs) < 10:
        return (0.0, 0.0, 0.0)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    r = cov / math.sqrt(vx * vy) if vx and vy else 0.0
    t = r * math.sqrt((len(xs) - 2) / max(1e-12, 1 - r * r))
    return (r, t, cov / vx if vx else 0.0)


def _fade(rows: list[dict], side: float) -> tuple[float, int]:
    wins = decisions = 0
    for row in rows:
        cover = row["residual"] * side
        if abs(cover) < 1e-9:
            continue
        decisions += 1
        wins += cover > 0
    return (wins / decisions if decisions else 0.0, decisions)


# -- the study -------------------------------------------------------------
def rows_for(season: int) -> list[dict]:
    """Team-games whose established starter carries a prior-season grade."""
    raw = frame(season)
    cfbd_names: dict[str, set[str]] = {}
    for row in raw:
        for name in (row["starter"], row["replacement"]):
            if name:
                cfbd_names.setdefault(row["team"], set()).add(_handle(name)[0])
    book = GradeBook(season - 1, cfbd_names)

    out: list[dict] = []
    for row in raw:
        starter = book.get(row["team"], row["starter"])
        if starter is None:
            continue
        replacement = book.get(row["team"], row["replacement"]) if row["absent"] else None
        out.append(
            {
                **row,
                "residual": row["margin"] - row["spread"],
                "grade": float(starter["grades_pass"]),
                "grade_z": book.z(float(starter["grades_pass"])),
                "dropbacks": float(starter["dropbacks"] or 0),
                "repl_grade": (
                    float(replacement["grades_pass"])
                    if replacement and replacement["grades_pass"]
                    else None
                ),
            }
        )
    return out


def question_a(rows: list[dict]) -> None:
    print("\n=== A. prior-season grade vs the closing spread residual ===")
    for label, subset in (
        ("all rows", rows),
        ("weeks 1-4", [r for r in rows if r["week"] <= 4]),
        ("200+ prior dropbacks", [r for r in rows if r["dropbacks"] >= 200]),
    ):
        r, t, b = _corr([x["grade_z"] for x in subset], [x["residual"] for x in subset])
        print(f"{label:<24}n={len(subset):>5}  r={r:+.4f}  t={t:+.2f}  {b:+.2f} pts per z")
    for label, subset, side in (
        ("back a +1z starter", [r for r in rows if r["grade_z"] >= 1.0], 1.0),
        ("fade a -1z starter", [r for r in rows if r["grade_z"] <= -1.0], -1.0),
    ):
        rate, n = _fade(subset, side)
        print(f"{label:<24}n={n:>5}  {rate:.2%} ATS")


def question_b(rows: list[dict]) -> None:
    absent = [r for r in rows if r["absent"]]
    graded = [r for r in absent if r["repl_grade"] is not None]
    print(f"\n=== B. the absence gap ({len(graded)} of {len(absent)} have a graded backup) ===")
    if graded:
        gaps = [r["grade"] - (r["repl_grade"] or 0.0) for r in graded]
        r, t, b = _corr(gaps, [x["residual"] for x in graded])
        print(
            f"{'gap vs residual':<28}n={len(graded):>5}  r={r:+.4f}  t={t:+.2f}  {b:+.3f} pts/grade"
        )
    for label, subset in (
        (
            "gap >= 10 grade points",
            [r for r in graded if r["grade"] - (r["repl_grade"] or 0) >= 10],
        ),
        ("gap < 10", [r for r in graded if r["grade"] - (r["repl_grade"] or 0) < 10]),
        ("replacement never graded", [r for r in absent if r["repl_grade"] is None]),
    ):
        if not subset:
            continue
        rate, n = _fade(subset, -1.0)
        miss = statistics.fmean(x["residual"] for x in subset)
        print(f"{label:<28}n={len(subset):>5}  mean residual {miss:+6.2f}  fade {rate:.2%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, action="append", help="default 2023-2025")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for season in args.season or SEASONS:
        got = rows_for(season)
        print(f"{season}: {len(got):>5} team-games with a graded prior-season starter")
        rows.extend(got)
    print(f"total {len(rows)}, absences {sum(1 for r in rows if r['absent'])}")
    question_a(rows)
    question_b(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
