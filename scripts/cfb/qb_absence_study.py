"""What a missing quarterback is worth, and when we are allowed to know it.

:mod:`cfb_engine.data.injuries` records a 55.3% fade on teams that lost a 75%+
starter, with the edge concentrated in the first game of an absence (56.2%, and
59.9% in a 2024-25 holdout) and gone once the backup is common knowledge. That
result -- and every version of it in this repo -- defines the absence from the
box score of the game being graded: the established passer took no dropbacks.

A Saturday morning does not have that box score. This script separates the two
questions the single number was answering, and they come apart:

    definition of "starter is out"          n      fade ATS        95% CI
    this week's box score (as measured)   1382      55.81%    [53.18, 58.44]
      first game of the absence            998      56.32%    [53.23, 59.41]
      he had already missed a week         384      54.47%    [49.47, 59.48]
    last week's box score (knowable)       717      51.41%    [47.73, 55.09]
      still out this week                  384      54.47%    [49.47, 59.48]
      back this week                       333      47.87%    [42.46, 53.27]

The first block is not a forecast. Roughly three quarters of its absences are
single-week gaps, and a passer who threw a quarter of his team's attempts is as
often a starter hooked in a blowout as one who never dressed -- so the flag is
partly *caused by* the margin it is used to predict. Restrict the condition to
the previous week's box score, which is genuinely available before kickoff, and
the fade lands at 51.41% against a 52.38% break-even: null, and null in each
era (47.9% / 54.4% / 51.5% across 2014-17, 2018-21, 2022-25). The mean miss
against the number collapses with it, -2.03 points to -0.60.

Two things follow. Retrospectively, the absence term has no measured edge, which
is why ``CFBE_INJURY_QB_PTS`` stays 0.0. And per-player quality data -- PFF
grades, a starter-minus-backup gap, anything that refines *how much* the absence
is worth -- cannot be graded on this frame, because the frame's own binary
version does not survive the leak check. What it needs instead is the
prospective log: a real-time designation, timestamped, against the line at that
moment.

The absence set itself is still worth having, and this script writes it out for
that purpose: 14,766 team-games from 2014-2025 with an established starter, of
which 1,382 are absences, joined to the median closing spread and the result.

Usage::

    CFBD_API_KEY=... python scripts/cfb/qb_absence_study.py
    CFBD_API_KEY=... python scripts/cfb/qb_absence_study.py --dump frame.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time

import requests

from cfb_engine.data.cfbd import _passer_games
from cfb_engine.data.starters import MIN_PRIOR_ATTEMPTS, STARTER_SHARE
from cfb_engine.data.teamnames import school_key

CACHE = pathlib.Path(os.getenv("CFBE_STUDY_CACHE", pathlib.Path.home() / "cfb_cache"))
CFBD = "https://api.collegefootballdata.com"
SEASONS = list(range(2014, 2026))
BREAK_EVEN = 0.5238  # -110 both ways
NOT_A_BOOK = {"teamrankings"}
OUT_SHARE = 0.25  # under a quarter of the week's attempts is "did not play"


# -- data ------------------------------------------------------------------
def _passers(season: int) -> list[dict]:
    """Every passer's weekly attempts for ``season``, cached on disk."""
    dest = CACHE / f"players_{season}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    key = os.environ["CFBD_API_KEY"]
    rows: list[dict] = []
    for week in range(1, 17):
        resp = requests.get(
            f"{CFBD}/games/players",
            params={"year": season, "week": week},
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            rows.extend(
                {
                    "week": game.week,
                    "team": game.team,
                    "player_id": game.player_id,
                    "name": game.name,
                    "attempts": game.attempts,
                }
                for game in _passer_games(payload, week)
            )
        time.sleep(0.2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows))
    return rows


def _closes(season: int) -> list[dict]:
    """Graded games with a median closing spread on the home-margin axis."""
    path = CACHE / f"lines_{season}.json"
    if not path.exists():
        return []
    out: list[dict] = []
    for game in json.loads(path.read_text()):
        home_score, away_score = game.get("homeScore"), game.get("awayScore")
        if home_score is None or away_score is None:
            continue
        spreads = [
            float(line["spread"])
            for line in game.get("lines") or []
            if line.get("spread") is not None
            and str(line.get("provider", "")).lower() not in NOT_A_BOOK
        ]
        if not spreads:
            continue
        out.append(
            {
                "week": int(game["week"]),
                "home": school_key(str(game["homeTeam"])),
                "away": school_key(str(game["awayTeam"])),
                # CFBD signs the spread from the bettor's view of the home team;
                # the margin axis here is home - away, so the handicap flips.
                "spread": -statistics.median(spreads),
                "margin": float(home_score) - float(away_score),
            }
        )
    return out


def _week_share(games: list[dict], week: int, player_id: str) -> float:
    """Fraction of a team's attempts in ``week`` thrown by ``player_id``."""
    total = sum(g["attempts"] for g in games if g["week"] == week)
    if total <= 0:
        return 0.0
    own = sum(g["attempts"] for g in games if g["week"] == week and g["player_id"] == player_id)
    return own / total


def _took_snaps(games: list[dict], week: int) -> tuple[str, int]:
    """Who threw the most in ``week``, and how much -- the man in the slot."""
    week_games = [g for g in games if g["week"] == week]
    if not week_games:
        return ("", 0)
    top = max(week_games, key=lambda g: g["attempts"])
    return (str(top["name"]), int(top["attempts"]))


def frame(season: int) -> list[dict]:
    """Team-games with an established starter, flagged absent and priced."""
    by_team: dict[str, list[dict]] = {}
    for row in _passers(season):
        if row["attempts"] > 0:
            by_team.setdefault(row["team"], []).append(row)
    closes = _closes(season)
    at_home = {(g["week"], g["home"]): g for g in closes}
    away = {(g["week"], g["away"]): g for g in closes}

    rows: list[dict] = []
    for team, games in by_team.items():
        weeks = sorted({g["week"] for g in games})
        for week in weeks:
            prior = [g for g in games if g["week"] < week]
            if not prior:
                continue
            totals: dict[str, int] = {}
            names: dict[str, str] = {}
            for game in prior:
                totals[game["player_id"]] = totals.get(game["player_id"], 0) + game["attempts"]
                names[game["player_id"]] = game["name"]
            overall = sum(totals.values())
            starter = max(totals, key=lambda pid: totals[pid])
            share = totals[starter] / overall
            if totals[starter] < MIN_PRIOR_ATTEMPTS or share < STARTER_SHARE:
                continue  # no established starter to be missing
            game_row = at_home.get((week, team))
            side = 1.0
            if game_row is None:
                game_row = away.get((week, team))
                side = -1.0
            if game_row is None:
                continue
            replacement, replacement_attempts = _took_snaps(games, week)
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": team,
                    "spread": side * game_row["spread"],
                    "margin": side * game_row["margin"],
                    "starter_id": starter,
                    "starter": names[starter],
                    "starter_share": share,
                    "starter_attempts": totals[starter],
                    "absent": _week_share(games, week, starter) < OUT_SHARE,
                    "replacement": replacement,
                    "replacement_attempts": replacement_attempts,
                    "out_prior_week": any(
                        _week_share(games, back, starter) < OUT_SHARE
                        for back in weeks
                        if week - 3 < back < week
                    ),
                }
            )
    return rows


# -- grading ---------------------------------------------------------------
def _fade(rows: list[dict]) -> tuple[int, int]:
    """Wins and decisions from betting against each listed team at the close."""
    wins = decisions = 0
    for row in rows:
        cover = -(row["margin"] - row["spread"])
        if abs(cover) < 1e-9:
            continue
        decisions += 1
        wins += cover > 0
    return wins, decisions


def _report(label: str, rows: list[dict]) -> None:
    wins, decisions = _fade(rows)
    if not decisions:
        print(f"{label:<40}{'(empty)':>12}")
        return
    rate = wins / decisions
    err = 1.96 * math.sqrt(rate * (1 - rate) / decisions)
    miss = statistics.fmean(row["margin"] - row["spread"] for row in rows)
    verdict = "clears" if rate - err > BREAK_EVEN else "null"
    print(
        f"{label:<40}n={len(rows):>5}  fade {rate * 100:>6.2f}% "
        f"[{(rate - err) * 100:>5.2f}, {(rate + err) * 100:>5.2f}]"
        f"  spread miss {miss:>+6.2f}  {verdict}"
    )


def grade(rows: list[dict]) -> None:
    absent = [r for r in rows if r["absent"]]
    print("\n=== measured off this week's box score (not available before kickoff) ===")
    _report("starter played", [r for r in rows if not r["absent"]])
    _report("starter absent", absent)
    _report("  first game of the absence", [r for r in absent if not r["out_prior_week"]])
    _report("  had already missed a week", [r for r in absent if r["out_prior_week"]])
    _report("  share of prior attempts >= 0.95", [r for r in absent if r["starter_share"] >= 0.95])

    knowable = [r for r in rows if r["out_prior_week"]]
    print("\n=== measured off last week's box score (knowable) ===")
    _report("starter missed the previous week", knowable)
    _report("  still out this week", [r for r in knowable if r["absent"]])
    _report("  back this week", [r for r in knowable if not r["absent"]])
    _report("  laying points", [r for r in knowable if r["spread"] > 0])
    _report("  taking points", [r for r in knowable if r["spread"] < 0])
    print("\n=== the knowable rule out of time ===")
    for low, high in ((2014, 2017), (2018, 2021), (2022, 2025)):
        _report(f"{low}-{high}", [r for r in knowable if low <= r["season"] <= high])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=pathlib.Path, help="write the joined frame as JSON")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    for season in SEASONS:
        season_rows = frame(season)
        absences = sum(1 for row in season_rows if row["absent"])
        print(f"{season}: {len(season_rows):>5} team-games with a starter, {absences:>4} absences")
        rows.extend(season_rows)
    print(f"total {len(rows)} team-games, {sum(1 for r in rows if r['absent'])} absences")
    grade(rows)
    if args.dump:
        args.dump.write_text(json.dumps(rows))
        print(f"\nwrote {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
