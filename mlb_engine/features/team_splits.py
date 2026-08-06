"""League-wide team offensive splits, so a lineup line can be *ranked*.

A lineup's .318 xwOBA means nothing to a reader on its own — the question is
whether that is a good offense against the hand it draws tonight, and whether it
travels. This module reads every team's batting from the same Statcast frame the
simulator uses and reports, per team:

* wOBA against right-handed and against left-handed pitching, each with its rank
  among the 30 clubs and a top/middle/bottom-third label,
* wOBA at home and on the road.

Ranks are over teams with enough plate appearances in the split, so an early-
season or injury-thinned split cannot manufacture a rank-1 offense.

It also measures the league's *own* batted-ball xwOBA the two ways the preview
reports it — averaged per hitter and per pitcher — because a lineup's .367 and a
starter's .326 are not comparable until each is read against the baseline of the
same statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import pandas as pd

MIN_SPLIT_PA = 150  # PA in a split before a team is ranked in it
MIN_VENUE_PA = 100  # a venue split is roughly half a season's PA, so a lower floor
MIN_BASELINE_BBE = 15  # batted balls before a player joins a league baseline

XWOBA_COL = "estimated_woba_using_speedangle"


@dataclass(frozen=True)
class LeagueContact:
    """League batted-ball xwOBA, averaged the same two ways the preview is."""

    batter: float | None  # mean of per-hitter xwOBA
    pitcher: float | None  # mean of per-pitcher xwOBA allowed


@dataclass(frozen=True)
class SplitRank:
    """A team's wOBA in one split, with its place among the ranked clubs."""

    woba: float
    pa: int
    rank: int  # 1 = best offense in the split
    of: int  # number of teams ranked

    @property
    def bucket(self) -> str:
        """``top`` / ``middle`` / ``bottom`` third of the ranked clubs."""
        third = self.of / 3.0
        if self.rank <= third:
            return "top"
        if self.rank <= 2 * third:
            return "middle"
        return "bottom"


@dataclass(frozen=True)
class TeamSplits:
    """One club's overall, platoon and home/road offense over the window."""

    team: str
    vs_rhp: SplitRank | None
    vs_lhp: SplitRank | None
    at_home: SplitRank | None
    on_road: SplitRank | None
    overall: SplitRank | None = None

    @property
    def home_woba(self) -> float | None:
        return None if self.at_home is None else self.at_home.woba

    @property
    def away_woba(self) -> float | None:
        return None if self.on_road is None else self.on_road.woba

    def vs_hand(self, hand: str | None) -> SplitRank | None:
        if hand == "L":
            return self.vs_lhp
        if hand == "R":
            return self.vs_rhp
        return None

    def at_venue(self, is_home: bool | None) -> SplitRank | None:
        if is_home is None:
            return None
        return self.at_home if is_home else self.on_road


def _player_mean_xwoba(df: pd.DataFrame, by: str) -> float | None:
    """Average of each player's mean batted-ball xwOBA (thin samples dropped).

    Restricted to balls in play, matching how the regression layer reads a hitter
    or a starter, so the baseline is on the same scale as the lines it centres.
    """
    if XWOBA_COL not in df or "bb_type" not in df:
        return None
    batted = df[df["bb_type"].notna() & df[XWOBA_COL].notna()]
    if not len(batted):
        return None
    per_player = batted.groupby(by)[XWOBA_COL].agg(["mean", "size"])
    keep = per_player[per_player["size"] >= MIN_BASELINE_BBE]["mean"]
    return round(float(keep.mean()), 3) if len(keep) else None


def league_contact(df: pd.DataFrame, as_of: Date, days: int) -> LeagueContact:
    """League baselines for the lineup and starter xwOBA lines in the preview."""
    window = _window(df, as_of, days)
    if not len(window):
        return LeagueContact(batter=None, pitcher=None)
    return LeagueContact(
        batter=_player_mean_xwoba(window, "batter"),
        pitcher=_player_mean_xwoba(window, "pitcher"),
    )


def _window(df: pd.DataFrame, as_of: Date, days: int) -> pd.DataFrame:
    if not len(df):
        return df
    dates = pd.to_datetime(df["game_date"]).dt.date
    return df[(dates > as_of - timedelta(days=days)) & (dates <= as_of)]


def _batting_team(df: pd.DataFrame) -> pd.Series:
    """The hitting side of each pitch: home club bats in the bottom half."""
    return df["home_team"].where(df["inning_topbot"] == "Bot", df["away_team"])


def _woba(df: pd.DataFrame) -> tuple[float | None, int]:
    denom = float(df["woba_denom"].fillna(0.0).sum())
    if denom <= 0:
        return None, 0
    return float(df["woba_value"].fillna(0.0).sum()) / denom, int(round(denom))


def _rank_split(
    by_team: dict[str, tuple[float | None, int]], min_pa: int = MIN_SPLIT_PA
) -> dict[str, SplitRank]:
    ranked = [(t, w, pa) for t, (w, pa) in by_team.items() if w is not None and pa >= min_pa]
    ranked.sort(key=lambda r: -r[1])
    return {
        t: SplitRank(woba=round(w, 3), pa=pa, rank=i + 1, of=len(ranked))
        for i, (t, w, pa) in enumerate(ranked)
    }


def build_team_splits(df: pd.DataFrame, as_of: Date, days: int) -> dict[str, TeamSplits]:
    """Platoon and home/road wOBA for all 30 clubs over the trailing ``days``."""
    window = _window(df, as_of, days).copy()
    if not len(window):
        return {}
    window["bat_team"] = _batting_team(window)
    pa_rows = window[window["woba_denom"].notna()]

    vs_r: dict[str, tuple[float | None, int]] = {}
    vs_l: dict[str, tuple[float | None, int]] = {}
    home: dict[str, tuple[float | None, int]] = {}
    away: dict[str, tuple[float | None, int]] = {}
    overall: dict[str, tuple[float | None, int]] = {}
    for team, rows in pa_rows.groupby("bat_team"):
        team = str(team)
        vs_r[team] = _woba(rows[rows["p_throws"] == "R"])
        vs_l[team] = _woba(rows[rows["p_throws"] == "L"])
        home[team] = _woba(rows[rows["inning_topbot"] == "Bot"])
        away[team] = _woba(rows[rows["inning_topbot"] == "Top"])
        overall[team] = _woba(rows)

    r_ranked = _rank_split(vs_r)
    l_ranked = _rank_split(vs_l)
    home_ranked = _rank_split(home, min_pa=MIN_VENUE_PA)
    away_ranked = _rank_split(away, min_pa=MIN_VENUE_PA)
    all_ranked = _rank_split(overall)

    return {
        team: TeamSplits(
            team=team,
            vs_rhp=r_ranked.get(team),
            vs_lhp=l_ranked.get(team),
            at_home=home_ranked.get(team),
            on_road=away_ranked.get(team),
            overall=all_ranked.get(team),
        )
        for team in vs_r
    }
