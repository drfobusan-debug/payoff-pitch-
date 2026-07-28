"""Season team xRD baseline for the run-line luck-gap signal.

The framework: a team whose **actual** run differential far outruns its
**expected** run differential (from underlying contact quality) is overperforming
on sequencing luck and is a fade candidate; a team whose actual RD lags its
strong xRD is a buy-low. We proxy expected run differential with season team
xwOBA **for** minus **against** (Baseball Savant ``estimated_woba``), and read
actual RD/G from the StatsAPI standings.

This aggregation is season-long and stabilizes slowly, so it is built **once per
day** by the ``team-form`` batch CLI and cached to JSON. The pipeline only reads
the cache -- it never re-aggregates per card.

The final ``luck_gap`` per team is league-relative: ``z(actual_rd_g) - z(xrd
proxy)``. Positive => actual outruns expected (lucky -> fade); negative =>
expected outruns actual (unlucky -> buy-low). The magnitude threshold that turns
this into a tier nudge is deliberately left to the graded-data backtest; the
signal ships off-by-default.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Teams whose Statcast and StatsAPI abbreviations differ, normalized to a single
# canonical token so the xwOBA and run-differential sides join cleanly.
_ABBR_ALIASES = {"OAK": "ATH"}

MIN_BATTED_BALLS = 200  # per side: below this a team's season xwOBA is too thin


def _canon(abbr: str) -> str:
    return _ABBR_ALIASES.get(abbr, abbr)


@dataclass(frozen=True)
class TeamForm:
    """One team's season xRD baseline inputs."""

    team: str
    xwoba_for: float
    xwoba_against: float
    actual_rd_g: float | None  # season runs scored - allowed, per game
    games: int

    @property
    def xrd_proxy(self) -> float:
        """Contact-quality expected run-differential proxy (for - against)."""
        return self.xwoba_for - self.xwoba_against


def _side_xwoba(df: pd.DataFrame, team: str, *, batting: bool) -> tuple[float, int]:
    """Mean ``estimated_woba`` (and batted-ball count) for a team's PAs.

    ``batting=True`` selects the team's own hitters; ``batting=False`` selects the
    opponents its pitchers faced.
    """
    top, bot = df["inning_topbot"] == "Top", df["inning_topbot"] == "Bot"
    away, home = df["away_team"] == team, df["home_team"] == team
    if batting:
        mask = (top & away) | (bot & home)
    else:
        mask = (bot & away) | (top & home)
    xw = pd.to_numeric(df.loc[mask, "estimated_woba_using_speedangle"], errors="coerce").dropna()
    return (float(xw.mean()), int(len(xw))) if len(xw) else (float("nan"), 0)


def build_team_forms(
    statcast: pd.DataFrame, run_diffs: dict[str, tuple[float, int]]
) -> dict[str, TeamForm]:
    """Build the per-team season baseline from a season Statcast frame + standings."""
    if statcast.empty:
        return {}
    teams = {
        _canon(t)
        for col in ("home_team", "away_team")
        for t in statcast[col].dropna().unique()
    }
    rd_canon = {_canon(k): v for k, v in run_diffs.items()}
    df = statcast.assign(
        home_team=statcast["home_team"].map(_canon),
        away_team=statcast["away_team"].map(_canon),
    )
    forms: dict[str, TeamForm] = {}
    for team in sorted(teams):
        xw_for, n_for = _side_xwoba(df, team, batting=True)
        xw_against, n_against = _side_xwoba(df, team, batting=False)
        if min(n_for, n_against) < MIN_BATTED_BALLS:
            continue
        rd = rd_canon.get(team)
        forms[team] = TeamForm(
            team=team,
            xwoba_for=xw_for,
            xwoba_against=xw_against,
            actual_rd_g=rd[0] if rd else None,
            games=rd[1] if rd else 0,
        )
    return forms


def compute_luck_gaps(forms: dict[str, TeamForm]) -> dict[str, float]:
    """League-relative luck gap per team: ``z(actual_rd_g) - z(xrd_proxy)``.

    Only teams with an actual RD/G are scored (needs both sides). Returns an empty
    dict if fewer than two such teams exist (a z-score needs spread).
    """
    scored = [f for f in forms.values() if f.actual_rd_g is not None]
    if len(scored) < 2:
        return {}
    actual = [f.actual_rd_g for f in scored if f.actual_rd_g is not None]
    proxy = [f.xrd_proxy for f in scored]
    a_mean, a_sd = statistics.mean(actual), statistics.pstdev(actual)
    p_mean, p_sd = statistics.mean(proxy), statistics.pstdev(proxy)
    if a_sd == 0 or p_sd == 0:
        return {}
    gaps: dict[str, float] = {}
    for f in scored:
        assert f.actual_rd_g is not None
        z_actual = (f.actual_rd_g - a_mean) / a_sd
        z_proxy = (f.xrd_proxy - p_mean) / p_sd
        gaps[f.team] = z_actual - z_proxy
    return gaps


def save_team_forms(forms: dict[str, TeamForm], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {t: asdict(f) for t, f in forms.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_team_forms(path: Path) -> dict[str, TeamForm]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {t: TeamForm(**d) for t, d in raw.items()}


def luck_gap_for(team_abbrev: str, gaps: dict[str, float]) -> float | None:
    return gaps.get(_canon(team_abbrev))
