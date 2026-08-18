"""League-wide team offensive splits, so a lineup line can be *ranked*.

A lineup's .318 xwOBA means nothing to a reader on its own — the question is
whether that is a good offense against the hand it draws tonight, and whether it
travels. This module reads every team's batting from the same Statcast frame the
simulator uses and reports, per team:

* xwOBA against right-handed and against left-handed pitching, each with its rank
  among the 30 clubs and a top/middle/bottom-third label,
* xwOBA at home and on the road.

It also measures the league's *own* batted-ball xwOBA the two ways the preview
reports it — averaged per hitter and per pitcher — because a lineup's .367 and a
starter's .326 are not comparable until each is read against the baseline of the
same statistic.

**The splits are expected outcomes, shrunk, and they have to be all three.**
Measured on 2026 to 07-22, the naive version — actual wOBA, six weeks, ranked
from 150 PA — carried almost nothing:

    split         spread across clubs   sampling noise   real
    vs LHP            33 points             27              34%
    home              29                    21              50%
    road              25                    20              31%

and none of it survived contact with a second window. A club's split did not
predict its own next split (overall r = -0.06, vs LHP r = -0.15), and it did not
even agree with itself when the *same* six weeks were dealt into odd and even
dates (r = +0.12). Three separate things were wrong:

1. **The metric.** wOBA counts whether a ball found a glove. Dealing the same
   games into halves 60 times, full-sample reliability goes 0.31 -> 0.68 for a
   hitter and 0.45 -> 0.65 for a pitcher when the outcome is xwOBA instead, so
   the results version was throwing away most of what was measurable.
2. **The sample.** A club is nine hitters averaged, so team spread is small next
   to per-PA variance and 400 PA cannot resolve it. Hence ``MIN_SPLIT_PA``.
3. **The absence of a prior.** Even at 500 PA a platoon split is only about a
   quarter signal, so a raw number claims a separation it has not earned.

The venue read was the worst of the three and is worth stating plainly: the
whole league hits .3364 at home against .3265 on the road, a ten-point edge,
while the noise on a single club's home-road gap is +-29 points. Five of thirty
clubs cleared two standard errors where chance alone gives about 1.4.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from mlb_engine.features.rolling import woba_from_rates

MIN_SPLIT_PA = 500  # PA in a split before a team is ranked in it
MIN_BASELINE_BBE = 15  # batted balls before a player joins a league baseline

XWOBA_COL = "estimated_woba_using_speedangle"

# The four splits, as masks on priced plate appearances. The home club bats in
# the bottom half of an inning, which is what makes a venue split readable off
# pitch data with no schedule attached.
SPLITS: dict[str, Callable[[pd.DataFrame], pd.Series[bool]]] = {
    "vs_rhp": lambda d: d["p_throws"] == "R",
    "vs_lhp": lambda d: d["p_throws"] == "L",
    "at_home": lambda d: d["inning_topbot"] == "Bot",
    "on_road": lambda d: d["inning_topbot"] == "Top",
}

# Equivalent PA of prior each split deserves, k = sigma_pa^2 / var(talent), so a
# split measured over n PA keeps n / (n + k) of its distance from the baseline.
# Both terms measured by ``scripts.team_split_shrink`` on 2026 to 07-22: the
# per-PA spread of xwOBA within the league, and the spread across the thirty
# clubs with that noise subtracted.
#
# Measured over 115,008 priced PA (2026-03-25..07-22), per-PA xwOBA sd 0.3729:
#
#     split      median PA   observed sd   noise sd   talent sd      k
#     overall       3821       0.0088       0.0060      0.0064     3396
#     vs RHP        2764       0.0098       0.0071      0.0068     3007
#     vs LHP        1130       0.0151       0.0111      0.0102     1342
#     at home       1886       0.0108       0.0086      0.0065     3284
#     on road       1956       0.0122       0.0084      0.0088     1802
#
# Read the talent column rather than the k column: **the whole league fits
# inside 6 to 10 points of true xwOBA**, so a printed 40-point gap between two
# clubs was never a real gap. The platoon split is the one with the most in it,
# which is the shape you would expect -- a club's handedness profile is a roster
# fact, while its venue split is mostly which parks it happened to visit. At 500
# PA the weights come out at 27% for vs LHP and 13% at home, so a 49-point
# home/road swing prints as seven and the preview calls it the wash it is.
SPLIT_PRIOR_PA = {
    "vs_rhp": 3007.0,
    "vs_lhp": 1342.0,
    "at_home": 3284.0,
    "on_road": 1802.0,
}
OVERALL_PRIOR_PA = 3396.0


@dataclass(frozen=True)
class LeagueContact:
    """League batted-ball xwOBA, averaged the same two ways the preview is."""

    batter: float | None  # mean of per-hitter xwOBA
    pitcher: float | None  # mean of per-pitcher xwOBA allowed


@dataclass(frozen=True)
class SplitRank:
    """A team's xwOBA in one split, with its place among the ranked clubs.

    ``woba`` is the shrunk number, which is the one to print and the one the rank
    is taken on; ``raw`` is what the window actually showed, kept so the gap
    between them can be inspected rather than taken on trust.
    """

    woba: float
    raw: float
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


def _batting_team(df: pd.DataFrame) -> pd.Series[str]:
    """The hitting side of each pitch: home club bats in the bottom half."""
    return df["home_team"].where(df["inning_topbot"] == "Bot", df["away_team"])


def _priced_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Plate appearances Statcast puts an expected value on, tagged by batting club.

    An intentional walk, a bunt and a catcher's interference arrive without an
    xwOBA and are dropped rather than scored zero: none of the three is the
    hitter describing his own offense, and together they are about 1% of PA.
    """
    if not len(df) or XWOBA_COL not in df:
        return df.iloc[:0]
    d = df[df["woba_denom"].notna() & df[XWOBA_COL].notna()].copy()
    d["bat_team"] = _batting_team(d)
    d[XWOBA_COL] = d[XWOBA_COL].astype(float)
    return d


def projected_offsets(
    df: pd.DataFrame, ros_priors: Mapping[int, Mapping[str, float]]
) -> dict[str, float]:
    """Each club's projected offense relative to the league, in wOBA points.

    The shrinkage target has to be somewhere *true*, not merely somewhere
    average: pulling every club to the league mean would flatten the offenses
    that are genuinely good, which is the one thing a ranking cannot afford. A
    club's hitters carry rest-of-season projections already (the same ones the
    batter prior uses), so their PA-weighted wOBA says where this offense should
    sit before tonight, independent of the window being shrunk.

    Returned as an *offset* rather than a level because a projection is on the
    wOBA scale and the splits are on the xwOBA scale; only the club's distance
    from its own league is transferable between the two.
    """
    if not len(df) or not ros_priors:
        return {}
    weight = df.groupby(["bat_team", "batter"]).size()
    proj: dict[str, float] = {}
    for team in {str(t) for t, _ in weight.index}:
        num = den = 0.0
        for (t, batter), pa in weight.items():
            if str(t) != team:
                continue
            vec = ros_priors.get(int(batter))
            if vec is None:
                continue
            num += woba_from_rates(dict(vec)) * float(pa)
            den += float(pa)
        if den:
            proj[team] = num / den
    if len(proj) < 2:
        return {}
    league = sum(proj.values()) / len(proj)
    return {team: value - league for team, value in proj.items()}


def _shrink(raw: float, pa: int, baseline: float, k: float) -> float:
    """Pull a split toward its baseline by how little of it the sample resolves."""
    return baseline + (pa / (pa + k)) * (raw - baseline)


def _rank(
    measured: dict[str, tuple[float, int]],
    offsets: Mapping[str, float],
    k: float,
    min_pa: int = MIN_SPLIT_PA,
) -> dict[str, SplitRank]:
    eligible = {t: (w, pa) for t, (w, pa) in measured.items() if pa >= min_pa}
    if not eligible:
        return {}
    league = sum(w for w, _ in eligible.values()) / len(eligible)
    shrunk = [
        (t, _shrink(w, pa, league + offsets.get(t, 0.0), k), w, pa)
        for t, (w, pa) in eligible.items()
    ]
    shrunk.sort(key=lambda r: -r[1])
    return {
        t: SplitRank(woba=round(s, 3), raw=round(w, 3), pa=pa, rank=i + 1, of=len(shrunk))
        for i, (t, s, w, pa) in enumerate(shrunk)
    }


def build_team_splits(
    df: pd.DataFrame,
    as_of: Date,
    days: int,
    ros_priors: Mapping[int, Mapping[str, float]] | None = None,
) -> dict[str, TeamSplits]:
    """Platoon and home/road xwOBA for all 30 clubs over the trailing ``days``."""
    d = _priced_pa(_window(df, as_of, days))
    if not len(d):
        return {}
    offsets = projected_offsets(d, ros_priors or {})

    measured: dict[str, dict[str, tuple[float, int]]] = {}
    for key, mask in SPLITS.items():
        g = d[mask(d)].groupby("bat_team")[XWOBA_COL].agg(["mean", "size"])
        measured[key] = {str(t): (float(r["mean"]), int(r["size"])) for t, r in g.iterrows()}
    overall = d.groupby("bat_team")[XWOBA_COL].agg(["mean", "size"])
    measured["overall"] = {
        str(t): (float(r["mean"]), int(r["size"])) for t, r in overall.iterrows()
    }

    ranked = {
        key: _rank(measured[key], offsets, SPLIT_PRIOR_PA[key]) for key in SPLITS
    }
    ranked["overall"] = _rank(measured["overall"], offsets, OVERALL_PRIOR_PA)

    return {
        team: TeamSplits(
            team=team,
            vs_rhp=ranked["vs_rhp"].get(team),
            vs_lhp=ranked["vs_lhp"].get(team),
            at_home=ranked["at_home"].get(team),
            on_road=ranked["on_road"].get(team),
            overall=ranked["overall"].get(team),
        )
        for team in measured["overall"]
    }
