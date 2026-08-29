"""Morning power screen: the softest arms on a slate, and who to hunt them with.

This is the hand method Franz has been running for years, written down. It runs
beside the nightly card rather than inside it, because it answers a narrower
question than the engine does and it answers it before lineups are posted, when
the engine declines to price a game at all.

The chain is five stages behind two gates, each of which throws work away:

0. **Gate the arms.** Two conditions, in order. An arm needs enough recent work
   for his numbers to be measurements rather than a rounding of one outing --
   which is what removes a call-up, an opener and an arm back from the injured
   list -- and then his SIERA has to be above the scrub ceiling (4.40, the
   engine's own ``singles_siera_bad``). A contact score alone will nominate an
   ace who happens to give up loud air contact; the gate says the arm has to be
   bad at run prevention *first*, and an arm with too little work to have a
   trusted SIERA is not eligible rather than assumed soft.
1. **Rank the arms.** The eligible starters are scored on eleven metrics --
   xERA, xFIP, SIERA, K%, CSW%, K-BB%, FB%, Stuff+, O-Swing%, HH% and SwStr% --
   one point apiece and two more for a top-three finish, the same shape as the
   hitter pool score in stage 2. **Each metric is read over its own window**:
   whiff and shape signals stabilize in a few hundred pitches and are read
   short, batted-ball and run-prevention signals need months and are read long,
   and a metric a starter lacks the sample for scores nothing for him rather
   than scoring average. The old air-contact index is kept as the tiebreak, and
   the top few games are the only ones that proceed.
2. **Score the lineups.** The hitters facing those arms are ranked on eleven
   contact and discipline metrics, split against the hand they will face, one
   point apiece and a second for a top-five finish in the pool. Ties are broken
   by nothing: the score exists to sort, not to price.
3. **Cut.** A plate-appearance floor first (a hot fortnight outranks a good six
   weeks otherwise), then at least one top-five finish, then wRC+, then an
   expected-contact floor that removes the hitters whose results have outrun
   their batted balls.
4. **Read the arsenal.** Per-pitch-type run value, xBA, xwOBA, hard-hit and
   barrel rates on both sides, and the hitter's own splits weighted by the mix
   he will actually see. A hitter who is helpless against the one pitch his
   opponent throws 15% of the time is a different bet than one who is not.
5. **Scale by exposure.** How many turns he gets against that starter given the
   engine's exit-point model and his slot, and who covers the rest of the game.
   A short starter in front of the league's softest bullpen is not the matchup
   the screen selected, and it can be better.
6. **Score both halves of the game.** Innings 1-6 and 7+ are scored as separate
   pools on nine stabilized metrics, each half shrunk toward the hitter's own
   all-innings rate rather than toward the league. The screen is looking for
   hitters who play all game, so a bat carried by one half is visible as such.
7. **Adjust for context.** Eight signed terms: the luck gap between xwOBA and
   wOBA, the three-week direction of bat speed, chase and EV90, the park, the
   forecast, one point for facing a bottom-three arm, and the hitter's run value
   on the starter's three most-thrown pitches. Each is ``+1`` toward the hitter
   or ``-1`` toward the pitcher, and ``0`` inside the metric's own noise band --
   missing evidence is neutral, not negative. The run-value term trebles past
   two runs per 100, which is the only place the layer pays for size.
8. **Score the arsenal edge.** The hitter's per-pitch run value, xwOBA, whiff,
   hard-hit and barrel marks and the starter's allowed marks on the same
   families, both weighted by the mix he will actually throw.

Every figure is observed or modelled, never priced: this module reads no market.
Run value is quoted **from the hitter's side throughout**, so both halves of a
matchup share an axis; Savant prints a pitcher's version with the opposite sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as Date
from typing import TYPE_CHECKING

import pandas as pd

from mlb_engine.data.statcast import batted_balls
from mlb_engine.features import stuff
from mlb_engine.features.siera import MIN_SIERA_PA, SIERA_LEAGUE_ANCHOR, pitcher_siera

if TYPE_CHECKING:  # the bet card reads a ScreenResult, so the import is one-way
    from mlb_engine.output.power_bets import BetCard

# --- pitch classification -------------------------------------------------
# Savant's codes collapsed to what a hitter actually distinguishes. Kept finer
# than the engine's four arsenal classes: a sweeper and a slider are the same
# class to the matchup multiplier and are not the same pitch to a hitter.
PITCH_FAMILY = {
    "FF": "4-Seam", "FA": "4-Seam", "SI": "Sinker", "FT": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "SV": "Slurve", "CU": "Curveball", "KC": "Curveball",
    "CS": "Curveball", "CH": "Changeup", "FS": "Splitter", "FO": "Splitter",
    "KN": "Knuckleball", "SC": "Screwball",
}

# 2026 linear weights on the Statcast scale, with HBP separated (the engine's
# WOBA_WEIGHTS folds it into BB, which is right for an outcome model and wrong
# for reconstructing an observed line).
WOBA_W = {
    "walk": 0.69, "hit_by_pitch": 0.72, "single": 0.89,
    "double": 1.27, "triple": 1.62, "home_run": 2.10,
}
WOBA_SCALE = 1.15
LG_R_PER_PA = 0.115  # runs per PA, for the wRC+ denominator

K_EVENTS = ("strikeout", "strikeout_double_play")
SWING_DESC = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
NO_AB = {
    "walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf",
    "sac_fly_double_play", "sac_bunt_double_play",
}

# Fly balls, on the FanGraphs definition xFIP's HR/FB is built on: infield flies
# are a subset of fly balls, not a fourth batted-ball class.
FB_TYPES = {"fly_ball", "popup"}

#: Outs an event records, for the innings xFIP divides by. Absent events record
#: none; a triple play is not folded into the double-play entry.
OUT_EVENTS = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "fielders_choice_out": 1,
    "sac_fly": 1, "sac_bunt": 1, "other_out": 1, "caught_stealing_2b": 1,
    "caught_stealing_3b": 1, "caught_stealing_home": 1, "pickoff_1b": 1,
    "pickoff_2b": 1, "pickoff_3b": 1, "pickoff_caught_stealing_2b": 1,
    "pickoff_caught_stealing_3b": 1, "pickoff_caught_stealing_home": 1,
    "strikeout_double_play": 2, "grounded_into_double_play": 2,
    "double_play": 2, "sac_fly_double_play": 2, "sac_bunt_double_play": 2,
    "triple_play": 3,
}

#: What identifies one starter's work in one game, for the time-through-order label.
TTO_KEYS = ["game_date", "home_team", "away_team", "pitcher", "batter"]

# xFIP is put on the same run scale SIERA is recentred to, so the two estimators
# in the ranking cannot disagree about what an average arm costs.
XFIP_LEAGUE_ANCHOR = SIERA_LEAGUE_ANCHOR

# Screen thresholds. Defaults are the ones the 8/17 hand run settled on.
SIERA_FLOOR = 4.4  # stage 0: a starter is eligible only above this SIERA
MIN_STARTER_BF = 120  # batters faced in the window before an arm is readable
MIN_STARTER_PITCHES = 400  # pitches in the same window before the shape reads mean anything
STARTER_TOP_N = 3  # a top-three finish in a stage-1 metric earns two more points
MIN_BATTER_PA = 60  # hand-split PA floor
MIN_WRC = 120  # window wRC+ floor, before the power exception
MIN_XWOBA_EDGE = 0.020  # xwOBA/PA must clear league by this much
MAX_LUCK_GAP = 0.050  # wOBA minus xwOBA/PA above this is unearned
POWER_XWOBACON = 0.440  # contact good enough to keep a hitter the wRC+ cut drops
MIN_PITCHER_PITCHES = 15  # per-pitch-type floor, pitcher side
MIN_BATTER_PITCHES = 25  # per-pitch-type floor, hitter side
TOP_K = 5  # a top-K finish in a scored metric earns the second point


@dataclass(frozen=True)
class StarterMetric:
    """One stage-1 metric: how it is measured, which way it points, and its sample.

    ``days`` is the window the metric is read over and ``min_sample`` the
    denominator it needs inside that window -- both set by how fast the metric
    stabilizes rather than by whichever frame was already loaded. Reading a fast
    signal long buries tonight's arm under June's; reading a slow one short
    ranks noise. ``sample`` names the denominator (``bf``, ``pitches``,
    ``oz_pitches``, ``bbe``, ``siera_pa`` or the external ``xera_pa``) so a
    starter short of it is unrated *in that metric alone*, rather than dropped
    from the ranking or handed a league-average stand-in.

    ``soft_high`` is true when a *higher* number means an easier arm to hit,
    which is why the directions are written down rather than assumed: xERA,
    xFIP, SIERA, fly-ball rate, hard-hit rate and home runs allowed all rise as
    an arm gets worse, while K%, CSW%, K-BB%, Stuff+, O-Swing% and SwStr% fall.
    """

    attr: str
    label: str
    soft_high: bool
    days: int  # 0 = season to date
    min_sample: int
    sample: str


#: The eleven metrics stage 1 ranks eligible starters on.
#:
#: Windows and floors follow how fast each metric settles. Pitch-level whiff and
#: shape signals are the fast half -- swinging-strike and called-plus-swinging
#: rates are near their season shape within a few hundred pitches, and the shape
#: grade holds start to start at r=+0.99 (``features/stuff``) -- so they are read
#: over three weeks, where they describe the arm taking the mound tonight. Plate
#: appearance rates settle next (K% by ~70 batters faced, K-BB% by ~200) and are
#: read over six. Batted-ball and run-prevention rates settle last: hard-hit rate
#: by ~50 batted balls, fly-ball rate and the run estimators by months, so those
#: are read over three months, SIERA keeps its own 80-PA floor, and xERA is taken
#: season-to-date from Savant's own board rather than reconstructed here.
STARTER_SCORED: tuple[StarterMetric, ...] = (
    StarterMetric("xera", "xERA", True, 0, 150, "xera_pa"),
    StarterMetric("xfip", "xFIP", True, 90, 200, "bf"),
    StarterMetric("siera", "SIERA", True, 90, MIN_SIERA_PA, "siera_pa"),
    StarterMetric("k_pct", "K%", False, 45, 150, "bf"),
    StarterMetric("csw_pct", "CSW%", False, 21, 400, "pitches"),
    StarterMetric("k_bb_pct", "K-BB%", False, 45, 200, "bf"),
    StarterMetric("fb_pct", "FB%", True, 90, 80, "bbe"),
    StarterMetric("stuff_plus", "Stuff+", False, 21, stuff.MIN_PITCHES, "pitches"),
    StarterMetric("osw_pct", "O-Swing%", False, 21, 200, "oz_pitches"),
    StarterMetric("hh_pct", "HH%", True, 90, 50, "bbe"),
    StarterMetric("swstr_pct", "SwStr%", False, 21, 400, "pitches"),
)

#: Home runs allowed per batter faced: the one metric the two HR splits rank on.
HR_METRIC = StarterMetric("hr_per_bf", "HR/BF", True, 90, 200, "bf")

#: The metrics a *split* can be scored on. Savant's xERA is a season total for
#: the whole arm and cannot be cut by inning, hand or time through the order, so
#: it is absent here rather than repeated unchanged into nine rankings where it
#: would quietly vote nine more times.
SPLIT_SCORED: tuple[StarterMetric, ...] = tuple(
    m for m in STARTER_SCORED if m.attr != "xera"
)


@dataclass(frozen=True)
class StarterSplit:
    """One perspective the arms are ranked from, and the metrics it ranks on.

    The overall ranking answers "who is worst"; these answer "worst at what".
    An arm who is average for five innings and collapses the third time through
    the order is a different bet than one who is hit from the first pitch, and a
    reverse-split arm is a different bet again -- the split rankings are what
    separate them, and the final order is the sum of every ranking's points.
    """

    key: str
    label: str
    metrics: tuple[StarterMetric, ...]


#: The overall ranking plus the nine split perspectives, in report order.
STARTER_SPLITS: tuple[StarterSplit, ...] = (
    StarterSplit("overall", "overall", STARTER_SCORED),
    StarterSplit("inn13", "innings 1-3", SPLIT_SCORED),
    StarterSplit("inn15", "innings 1-5", SPLIT_SCORED),
    StarterSplit("tto1", "1st time through", SPLIT_SCORED),
    StarterSplit("tto2", "2nd time through", SPLIT_SCORED),
    StarterSplit("tto3", "3rd time through", SPLIT_SCORED),
    StarterSplit("vs_l", "vs LHH", SPLIT_SCORED),
    StarterSplit("vs_r", "vs RHH", SPLIT_SCORED),
    StarterSplit("hr_l", "HR vs LHH", (HR_METRIC,)),
    StarterSplit("hr_r", "HR vs RHH", (HR_METRIC,)),
)

#: The windows every metric above is measured over (season-to-date excluded).
STARTER_WINDOWS: tuple[int, ...] = tuple(
    sorted({m.days for m in (*STARTER_SCORED, HR_METRIC) if m.days})
)

#: The window the work gate reads, which is the longest any metric is read over:
#: an arm is judged on his season's body of work, not on two good weeks.
WORK_DAYS = max(STARTER_WINDOWS)

#: Scored metrics: attribute, label, and whether more is better.
SCORED: tuple[tuple[str, str, bool], ...] = (
    ("wrc", "wRC+", True),
    ("ops", "OPS", True),
    ("ba", "BA", True),
    ("xba", "xBA", True),
    ("slg", "SLG", True),
    ("xslg", "xSLG", True),
    ("xwoba_con", "xwOBAcon", True),
    ("brl", "Brl%", True),
    ("hh", "HH%", True),
    ("ev90", "EV90", True),
    ("osw", "O-Swing%", False),
)


def pitch_family(code: object) -> str | None:
    """Savant pitch code to the family a hitter reads it as."""
    if not isinstance(code, str):
        return None
    return PITCH_FAMILY.get(code.upper())


def _rv_per_100(df: pd.DataFrame) -> float:
    """Run value per 100 pitches, from the hitter's side.

    ``delta_run_exp`` is already signed for the batting team in the feed -- a home
    run reads +1.53 and a called strike -0.07 -- so it is summed as it comes and
    the sign needs no flipping. Savant's pitcher-facing displays negate it.

    The denominator is every pitch in the bucket, not every pitch with a value,
    so a run value is diluted rather than inflated by pitches the feed did not
    score. Absent from frames cached before the column was requested, in which
    case the figure reads as unavailable rather than as zero.
    """
    if "delta_run_exp" not in df or df.empty:
        return math.nan
    rv = df["delta_run_exp"].dropna()
    if rv.empty:
        return math.nan
    return float(rv.sum() / len(df) * 100)


def _whiff(df: pd.DataFrame) -> float:
    if "description" not in df or df.empty:
        return math.nan
    swings = int(df["description"].isin(SWING_DESC).sum())
    if not swings:
        return math.nan
    return float(df["description"].isin(WHIFF_DESC).sum() / swings)


@dataclass(frozen=True)
class ContactLine:
    """Contact quality for one bucket of pitches, either side of the matchup."""

    pitches: int
    bbe: int
    rv100: float
    xba: float
    xwoba: float
    hh: float
    brl: float
    whiff: float

    @property
    def thin(self) -> bool:
        return self.bbe < 10


def contact_line(df: pd.DataFrame) -> ContactLine:
    """xBA, xwOBA, hard-hit, barrel and run value for a set of pitches.

    xBA and xwOBA are per ball in play, not per PA, so they describe the damage
    on contact and are directly comparable between a hitter and the pitcher who
    threw to him.
    """
    bip = batted_balls(df)
    xba = bip["estimated_ba_using_speedangle"].dropna() if len(bip) else pd.Series(dtype=float)
    xw = bip["estimated_woba_using_speedangle"].dropna() if len(bip) else pd.Series(dtype=float)
    ls = bip["launch_speed"].dropna() if len(bip) else pd.Series(dtype=float)
    lsa = bip["launch_speed_angle"].dropna() if len(bip) else pd.Series(dtype=float)
    return ContactLine(
        pitches=int(len(df)),
        bbe=int(len(bip)),
        rv100=_rv_per_100(df),
        xba=float(xba.mean()) if len(xba) else math.nan,
        xwoba=float(xw.mean()) if len(xw) else math.nan,
        hh=float((ls >= 95).mean()) if len(ls) else math.nan,
        brl=float((lsa == 6).mean()) if len(lsa) else math.nan,
        whiff=_whiff(df),
    )


# --- stage 1: rank the arms ----------------------------------------------


@dataclass
class StarterCard:
    """One probable starter's damage profile over the form window."""

    name: str
    mlbam_id: int
    team: str
    opponent: str
    throws: str
    bf: int
    fb_pct: float
    brl_pct: float
    hh_pct: float
    xwobacon: float
    hr_per_bf: float
    k_bb_pct: float
    csw_pct: float
    index: float = 0.0
    siera: float | None = None  # None when the window is too thin to trust one
    siera_pa: int = 0
    work_bf: int = 0  # batters faced over ``WORK_DAYS``, for the work gate
    work_pitches: int = 0
    points: int = 0  # stage-1 points, summed over every ranking
    lines: dict[str, MetricLine] = field(default_factory=dict)
    scores: dict[str, SplitScore] = field(default_factory=dict)
    arsenal: dict[str, ContactLine] = field(default_factory=dict)
    usage: dict[str, float] = field(default_factory=dict)


def _pa_events(df: pd.DataFrame) -> pd.Series:
    return df["events"].dropna() if "events" in df else pd.Series(dtype=object)


def starter_damage(
    rows: pd.DataFrame, *, name: str, mlbam_id: int, team: str, opponent: str, throws: str
) -> StarterCard:
    """Profile one starter on the damage he allows in the air."""
    ev = _pa_events(rows)
    bf = int(len(ev))
    bip = batted_balls(rows)
    la = bip["launch_angle"].dropna() if len(bip) else pd.Series(dtype=float)
    ls = bip["launch_speed"].dropna() if len(bip) else pd.Series(dtype=float)
    lsa = bip["launch_speed_angle"].dropna() if len(bip) else pd.Series(dtype=float)
    xw = bip["estimated_woba_using_speedangle"].dropna() if len(bip) else pd.Series(dtype=float)
    k = float(ev.isin(K_EVENTS).sum()) / bf if bf else math.nan
    bb = float(ev.eq("walk").sum()) / bf if bf else math.nan
    called_or_swinging = (
        float(rows["description"].isin(WHIFF_DESC | {"called_strike"}).mean())
        if "description" in rows and len(rows)
        else math.nan
    )
    s = pitcher_siera(rows)
    return StarterCard(
        name=name,
        mlbam_id=mlbam_id,
        team=team,
        opponent=opponent,
        throws=throws,
        bf=bf,
        fb_pct=float(la.between(25, 50).mean()) if len(la) else math.nan,
        brl_pct=float((lsa == 6).mean()) if len(lsa) else math.nan,
        hh_pct=float((ls >= 95).mean()) if len(ls) else math.nan,
        xwobacon=float(xw.mean()) if len(xw) else math.nan,
        hr_per_bf=float(ev.eq("home_run").sum()) / bf if bf else math.nan,
        k_bb_pct=k - bb if not (math.isnan(k) or math.isnan(bb)) else math.nan,
        csw_pct=called_or_swinging,
        siera=s.siera if s.has_data else None,
        siera_pa=s.pa,
    )


def with_tto(frame: pd.DataFrame) -> pd.DataFrame:
    """Label every pitch with the time through the order it was thrown in.

    A hitter's first meeting with a starter in a game is that starter's first
    time through the order, his second meeting the second, and so on -- so the
    label is the dense rank of the inning among the innings in which that batter
    faced that pitcher in that game. Built that way it needs no pitch-sequence
    column, which the cached frames do not carry. Two plate appearances in one
    inning (a nine-run rally) collapse to a single number; the alternative is a
    sequence the frame cannot supply.
    """
    if not set([*TTO_KEYS, "inning"]).issubset(frame.columns):
        return frame
    out = frame.copy()
    out["tto"] = out.groupby(TTO_KEYS, sort=False)["inning"].rank(method="dense")
    return out


def split_rows(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    """The subset of a starter's pitches one split ranking is measured on."""
    empty = rows.iloc[0:0]
    if key == "overall":
        return rows
    if key in ("inn13", "inn15"):
        if "inning" not in rows:
            return empty
        return rows[rows["inning"] <= (3 if key == "inn13" else 5)]
    if key in ("tto1", "tto2", "tto3"):
        if "tto" not in rows:
            return empty
        return rows[rows["tto"] == float(key[-1])]
    if key in ("vs_l", "hr_l", "vs_r", "hr_r"):
        if "stand" not in rows:
            return empty
        return rows[rows["stand"] == ("L" if key.endswith("_l") else "R")]
    return empty


@dataclass(frozen=True)
class LeagueArms:
    """The league numbers xFIP needs, measured over the frame it is scored on.

    ``hr_per_fb`` is the rate xFIP substitutes for a pitcher's own home runs --
    the whole point of the estimator -- and ``constant`` puts the result on the
    league's run scale, fitted so a league-average arm reads ``anchor`` rather
    than being carried over from another season's published constant.
    """

    hr_per_fb: float
    constant: float


def league_arms(window: pd.DataFrame, anchor: float = XFIP_LEAGUE_ANCHOR) -> LeagueArms:
    """Fit the league's HR-per-fly-ball rate and xFIP constant on one window."""
    ev = _pa_events(window)
    bip = batted_balls(window)
    fb = int(bip["bb_type"].isin(FB_TYPES).sum()) if len(bip) and "bb_type" in bip else 0
    outs = _outs(ev)
    if not fb or outs < 3:
        return LeagueArms(math.nan, math.nan)
    hr_per_fb = float(ev.eq("home_run").sum()) / fb
    raw = _fip_core(ev, fb, hr_per_fb, outs)
    return LeagueArms(hr_per_fb, anchor - raw)


def _outs(events: pd.Series) -> int:
    return int(sum(OUT_EVENTS.get(str(e), 0) for e in events))


def _fip_core(events: pd.Series, fb: int, hr_per_fb: float, outs: int) -> float:
    """The xFIP numerator over innings, before the league constant."""
    k = float(events.isin(K_EVENTS).sum())
    bb = float(events.eq("walk").sum())
    hbp = float(events.eq("hit_by_pitch").sum())
    return (13.0 * fb * hr_per_fb + 3.0 * (bb + hbp) - 2.0 * k) / (outs / 3.0)


def _window_metrics(
    rows: pd.DataFrame, league: LeagueArms
) -> tuple[dict[str, float], dict[str, int]]:
    """Every stage-1 metric and every denominator behind it, for one frame.

    Computed for the whole frame at once rather than per metric because the
    metrics that share a window share their samples, and a metric with nothing
    behind it returns ``nan`` here and is refused by its sample floor later --
    never zero, which would read as the softest arm on the slate.
    """
    ev = _pa_events(rows)
    bf = int(len(ev))
    bip = batted_balls(rows)
    bbt = (
        bip["bb_type"].dropna() if len(bip) and "bb_type" in bip else pd.Series(dtype=object)
    )
    ls = bip["launch_speed"].dropna() if len(bip) else pd.Series(dtype=float)
    desc = rows["description"] if "description" in rows else pd.Series(dtype=object)
    oz = rows[rows["zone"] > 9] if "zone" in rows else rows.iloc[0:0]
    pitches = int(len(rows))
    s = pitcher_siera(rows)
    k = float(ev.isin(K_EVENTS).sum()) / bf if bf else math.nan
    bb = float(ev.eq("walk").sum()) / bf if bf else math.nan
    fb = int(bbt.isin(FB_TYPES).sum()) if len(bbt) else 0
    outs = _outs(ev)
    values = {
        "k_pct": k,
        "k_bb_pct": k - bb if not (math.isnan(k) or math.isnan(bb)) else math.nan,
        "csw_pct": (
            float(desc.isin(WHIFF_DESC | {"called_strike"}).mean()) if pitches else math.nan
        ),
        "swstr_pct": float(desc.isin(WHIFF_DESC).mean()) if pitches else math.nan,
        "osw_pct": float(oz["description"].isin(SWING_DESC).mean()) if len(oz) else math.nan,
        "fb_pct": float(bbt.isin(FB_TYPES).mean()) if len(bbt) else math.nan,
        "hh_pct": float((ls >= 95).mean()) if len(ls) else math.nan,
        "hr_per_bf": float(ev.eq("home_run").sum()) / bf if bf else math.nan,
        "siera": s.siera,
        "stuff_plus": (
            stuff.shape_plus(rows) * 100.0 if pitches >= stuff.MIN_PITCHES else math.nan
        ),
        "xfip": (
            _fip_core(ev, fb, league.hr_per_fb, outs) + league.constant
            if fb and outs >= 3 and not math.isnan(league.hr_per_fb)
            else math.nan
        ),
        "xera": math.nan,  # season-to-date and external; never derived here
    }
    samples = {
        "bf": bf,
        "pitches": pitches,
        "oz_pitches": int(len(oz)),
        "bbe": int(len(bip)),
        "siera_pa": s.pa,
        "xera_pa": 0,
    }
    return values, samples


@dataclass(frozen=True)
class MetricLine:
    """One arm's stage-1 metrics for one split, and the sample behind each.

    Samples are keyed by denominator *and* window, because the same arm has a
    different number of batters faced in three weeks than in three months and
    each metric is held to the sample of the window it was read over.
    """

    values: dict[str, float]
    samples: dict[str, int]

    def value(self, metric: StarterMetric) -> float:
        return self.values.get(metric.attr, math.nan)

    def sample(self, metric: StarterMetric) -> int:
        return self.sample_of(metric.sample, metric.days)

    def sample_of(self, kind: str, days: int) -> int:
        return self.samples.get(f"{kind}:{days}", 0)


def starter_lines(
    frames: dict[int, pd.DataFrame],
    league: LeagueArms,
    *,
    xera: float | None = None,
    xera_pa: int = 0,
    splits: tuple[StarterSplit, ...] = STARTER_SPLITS,
) -> dict[str, MetricLine]:
    """One arm's metric line for every ranking, each read over its own window.

    ``frames`` maps a window in days to that starter's pitches over it, so a
    fast metric is never read long and a slow one never short. ``xera`` is
    Savant's season figure and its ``xera_pa`` the sample behind it; passing
    ``None`` leaves xERA unavailable rather than substituting a number.
    """
    out: dict[str, MetricLine] = {}
    for split in splits:
        values: dict[str, float] = {}
        samples: dict[str, int] = {}
        cache: dict[int, tuple[dict[str, float], dict[str, int]]] = {}
        for metric in split.metrics:
            if metric.attr == "xera":
                values["xera"] = math.nan if xera is None else xera
                samples[f"xera_pa:{metric.days}"] = xera_pa
                continue
            rows = frames.get(metric.days)
            if rows is None:
                continue
            if metric.days not in cache:
                cache[metric.days] = _window_metrics(split_rows(rows, split.key), league)
            window_values, window_samples = cache[metric.days]
            values[metric.attr] = window_values[metric.attr]
            for kind, n in window_samples.items():
                samples[f"{kind}:{metric.days}"] = n
        out[split.key] = MetricLine(values, samples)
    return out


@dataclass
class SplitScore:
    """What one ranking made of one arm: his place, his points, and what went unrated.

    ``rank`` is his place *in that ranking alone* -- 1 is the arm the split says is
    worst -- which is what the note highlights, so "worst three the third time
    through the order" is a fact on the page and not something a reader has to
    reconstruct from a points column.
    """

    key: str
    label: str
    points: int = 0
    rank: int = 0
    places: int = 0
    top_in: tuple[str, ...] = ()
    unrated: tuple[str, ...] = ()


@dataclass(frozen=True)
class StarterCut:
    """A starter a stage-0 gate removed, and the number that removed him."""

    card: StarterCard
    reason: str
    stage: str  # "work" or "siera"


def gate_starters(
    cards: list[StarterCard],
    *,
    siera_floor: float = SIERA_FLOOR,
    min_bf: int = MIN_STARTER_BF,
    min_pitches: int = MIN_STARTER_PITCHES,
) -> tuple[list[StarterCard], list[StarterCut]]:
    """Stage 0: the arms whose numbers are trustworthy *and* bad, and who left.

    Two conditions in order, because they fail for different reasons. The work
    floor removes the arm nobody has seen enough of -- a call-up, an opener, a
    starter three outings back from the injured list -- whose eleven metrics
    would otherwise be eleven small samples ranked as if they were skills.
    Then the SIERA floor removes the arms who prevent runs: a contact index says
    how loudly an arm is hit, which a good pitcher survives, and on its own it
    has repeatedly nominated sub-3.4 SIERA aces.

    Both are measured over the longest stage-1 window rather than the form
    window, so a starter is judged on his season's body of work and not on two
    good weeks. ``siera_floor <= 0`` turns the second gate off, which is a
    debugging switch and not a mode the note is written in.
    """
    kept: list[StarterCard] = []
    cuts: list[StarterCut] = []
    for card in cards:
        bf = card.work_bf
        pitches = card.work_pitches
        if bf < min_bf or pitches < min_pitches:
            cuts.append(
                StarterCut(
                    card,
                    f"{bf} BF and {pitches} pitches in {WORK_DAYS}d "
                    f"(needs {min_bf} and {min_pitches})",
                    "work",
                )
            )
        elif siera_floor <= 0:
            kept.append(card)
        elif card.siera is None:
            cuts.append(
                StarterCut(card, f"no SIERA ({card.siera_pa} PA < {MIN_SIERA_PA})", "siera")
            )
        elif card.siera <= siera_floor:
            cuts.append(
                StarterCut(card, f"SIERA {card.siera:.2f} \u2264 {siera_floor:.2f}", "siera")
            )
        else:
            kept.append(card)
    cuts.sort(key=lambda c: (c.stage != "work", c.card.siera is None, -(c.card.siera or 0.0)))
    return kept, cuts


def _split_scale(cards: list[StarterCard], split: StarterSplit, metric: StarterMetric) -> float:
    """How much of a metric's sample a split holds, measured on the pool itself.

    A third time through the order is a fifth of a start, so demanding the full
    overall floor inside it would leave every arm unrated and the ranking empty.
    The floor is scaled by the *pool's own median share* of that denominator in
    that split -- as much of the split as a normal starter has -- so the
    standard is still a sample standard rather than a guess, while staying
    honest that a split ranking rests on less evidence than the overall one.
    """
    if split.key == "overall":
        return 1.0
    ratios = [
        card.lines[split.key].sample(metric) / card.lines["overall"].sample_of(
            metric.sample, metric.days
        )
        for card in cards
        if split.key in card.lines
        and "overall" in card.lines
        and card.lines["overall"].sample_of(metric.sample, metric.days) > 0
    ]
    if not ratios:
        return 1.0
    ratios.sort()
    mid = len(ratios) // 2
    median = ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    return min(1.0, max(0.05, median))


def score_starters(
    cards: list[StarterCard],
    *,
    splits: tuple[StarterSplit, ...] = STARTER_SPLITS,
    top_n: int = STARTER_TOP_N,
) -> None:
    """One point per rated metric in every ranking, two more for a top-``top_n``.

    Ten rankings run: the overall one on all eleven metrics, seven splits on the
    ten that can be cut (Savant's xERA cannot), and the two home-run splits on
    HR/BF alone. Within each, every metric an arm has the sample for scores him a
    point and a top-three finish in it scores two more, so a top-three metric is
    worth three points and a metric he is short of sample for is worth none. His
    stage-1 total is the sum across all ten, which is what the final order sorts
    on.

    The unconditional point is kept for the same reason stage 2 keeps it: it
    makes the total readable next to the count of top-three finishes. It also
    means the total can only be compared between arms rated in the same metrics,
    which is why ``SplitScore.unrated`` is carried into the note instead of a
    low total being left to look like a good pitcher.

    Mutates the cards.
    """
    for card in cards:
        card.scores = {s.key: SplitScore(s.key, s.label) for s in splits}
        card.points = 0
    for split in splits:
        for metric in split.metrics:
            scale = _split_scale(cards, split, metric)
            floor = max(1, round(metric.min_sample * scale))
            rated = [
                card
                for card in cards
                if split.key in card.lines
                and not math.isnan(card.lines[split.key].value(metric))
                and card.lines[split.key].sample(metric) >= floor
            ]
            # Worst first in the metric's own direction, then by name, so a tie
            # between two identical lines does not depend on the order the slate
            # happened to arrive in.
            rated.sort(
                key=lambda c: (
                    -c.lines[split.key].value(metric)
                    if metric.soft_high
                    else c.lines[split.key].value(metric),
                    c.name,
                )
            )
            rated_ids = {id(card) for card in rated}
            for i, card in enumerate(rated):
                score = card.scores[split.key]
                score.points += 1
                score.places += i + 1
                if i < top_n:
                    score.points += 2
                    score.top_in = (*score.top_in, metric.label)
            for card in cards:
                if id(card) not in rated_ids:
                    score = card.scores[split.key]
                    score.unrated = (*score.unrated, metric.label)
        # Place inside this ranking, worst arm first. Points come first, then the
        # sum of his places in the split's metrics, so three arms who are all
        # top-three everywhere -- and therefore hold the same points -- still print
        # in the order the metrics put them rather than alphabetically. Then the
        # count of metrics the split could not rate him in, so an arm scored on
        # nine does not outrank one scored on four by having had more chances, and
        # finally the name, so two identical lines always print the same way.
        standing = sorted(
            cards,
            key=lambda c: (
                -c.scores[split.key].points,
                c.scores[split.key].places,
                len(c.scores[split.key].unrated),
                c.name,
            ),
        )
        for place, card in enumerate(standing, 1):
            card.scores[split.key].rank = place
    for card in cards:
        card.points = sum(score.points for score in card.scores.values())


def _z(values: list[float]) -> list[float]:
    """Z-scores, with NaNs held at the mean so a missing metric cannot vote."""
    real = [v for v in values if not math.isnan(v)]
    if len(real) < 2:
        return [0.0 for _ in values]
    mean = sum(real) / len(real)
    var = sum((v - mean) ** 2 for v in real) / (len(real) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [0.0 if math.isnan(v) else (v - mean) / sd for v in values]


def rank_starters(cards: list[StarterCard], top_n: int = 4) -> list[StarterCard]:
    """Order the eligible arms by stage-1 points and return the worst ``top_n``.

    The sort key is the total across all ten rankings, which is the number Franz
    asked for; the air-contact index below is computed on the same pool and kept
    as the tiebreak, because two arms can hold the same points for different
    reasons and the home-run screen should prefer the one hit harder in the air.

    Arms are expected to have cleared ``gate_starters`` already -- that is where
    a thin sample is refused. Cards that never went through it are ranked on
    whatever they have, so the caller keeps the gate, not this function.
    """
    readable = list(cards)
    if not readable:
        return []
    parts = {
        "brl": _z([c.brl_pct for c in readable]),
        "hh": _z([c.hh_pct for c in readable]),
        "fb": _z([c.fb_pct for c in readable]),
        "xwobacon": _z([c.xwobacon for c in readable]),
        "hr": _z([c.hr_per_bf for c in readable]),
        "kbb": _z([c.k_bb_pct for c in readable]),
        "csw": _z([c.csw_pct for c in readable]),
    }
    for i, card in enumerate(readable):
        card.index = (
            parts["brl"][i] + parts["hh"][i] + parts["fb"][i]
            + parts["xwobacon"][i] + parts["hr"][i]
            - parts["kbb"][i] - parts["csw"][i]
        )
    return sorted(readable, key=lambda c: (-c.points, -c.index))[:top_n]


# --- stage 2 and 3: score and cut the lineups ----------------------------


@dataclass
class HitterLine:
    """One hitter's hand-split window line, plus his screen result."""

    name: str
    mlbam_id: int
    team: str
    slot: int | None
    bats: str | None
    versus: str  # starter name
    pa: int
    wrc: float
    woba: float
    obp: float
    slg: float
    ops: float
    ba: float
    xba: float
    xslg: float
    xwoba_pa: float
    xwoba_con: float
    k: float
    bb: float
    brl: float
    hh: float
    ev90: float
    osw: float
    points: int = 0
    top_in: tuple[str, ...] = ()
    kept: bool = False
    cut_reason: str = ""
    power_exception: bool = False

    @property
    def luck_gap(self) -> float:
        return self.woba - self.xwoba_pa


def batter_window_line(rows: pd.DataFrame) -> dict[str, float]:
    """Reconstruct an observed rate line from a hitter's pitch-level rows."""
    ev = _pa_events(rows)
    pa = int(len(ev))
    if not pa:
        return {}
    ab = float(sum(1 for e in ev if e not in NO_AB))
    singles = float(ev.eq("single").sum())
    doubles = float(ev.eq("double").sum())
    triples = float(ev.eq("triple").sum())
    hr = float(ev.eq("home_run").sum())
    bb = float(ev.eq("walk").sum())
    hbp = float(ev.eq("hit_by_pitch").sum())
    hits = singles + doubles + triples + hr
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    woba = (
        WOBA_W["walk"] * bb + WOBA_W["hit_by_pitch"] * hbp + WOBA_W["single"] * singles
        + WOBA_W["double"] * doubles + WOBA_W["triple"] * triples + WOBA_W["home_run"] * hr
    ) / pa
    bip = batted_balls(rows)
    xba = bip["estimated_ba_using_speedangle"].dropna() if len(bip) else pd.Series(dtype=float)
    xw = bip["estimated_woba_using_speedangle"].dropna() if len(bip) else pd.Series(dtype=float)
    ls = bip["launch_speed"].dropna() if len(bip) else pd.Series(dtype=float)
    lsa = bip["launch_speed_angle"].dropna() if len(bip) else pd.Series(dtype=float)
    # Statcast publishes expected BA and expected wOBA per ball in play but no
    # expected SLG, so expected bases are implied from each ball's xwOBA net of
    # the on-base component. Directional, not Savant's xSLG.
    xslg = float(xw.sum() / WOBA_W["single"] * 1.55 / ab) if ab and len(xw) else math.nan
    out_of_zone = rows[rows["zone"] > 9] if "zone" in rows else rows.iloc[0:0]
    return {
        "pa": float(pa),
        "woba": woba,
        "obp": (hits + bb + hbp) / pa,
        "slg": tb / ab if ab else math.nan,
        "ba": hits / ab if ab else math.nan,
        "xba": float(xba.sum() / ab) if ab and len(xba) else math.nan,
        "xslg": xslg,
        "xwoba_pa": (
            WOBA_W["walk"] * bb + WOBA_W["hit_by_pitch"] * hbp + float(xw.sum())
        ) / pa,
        "xwoba_con": float(xw.mean()) if len(xw) else math.nan,
        "k": float(ev.isin(K_EVENTS).mean()),
        "bb": bb / pa,
        "brl": float((lsa == 6).mean()) if len(lsa) else math.nan,
        "hh": float((ls >= 95).mean()) if len(ls) else math.nan,
        "ev90": float(ls.quantile(0.90)) if len(ls) >= 10 else math.nan,
        "osw": (
            float(out_of_zone["description"].isin(SWING_DESC).mean())
            if len(out_of_zone) else math.nan
        ),
    }


def wrc_plus(woba: float, league_woba: float) -> float:
    """Window wRC+ against the same window's league line for that hand.

    100 is league *over these weeks against this hand*, not the season figure a
    leaderboard would show.
    """
    if math.isnan(woba) or not league_woba:
        return math.nan
    return 100.0 + (woba - league_woba) / WOBA_SCALE / LG_R_PER_PA * 100.0


def score_pool(pool: list[HitterLine], top_k: int = TOP_K) -> None:
    """One point per scored metric, two for a top-``top_k`` finish in the pool.

    Mutates ``pool``. The base point is unconditional, so the score is really
    ``len(SCORED) + top-K count``; the flat part is kept because it makes the
    number readable next to the count of top finishes rather than because it
    discriminates.
    """
    for h in pool:
        h.points = 0
        h.top_in = ()
    for attr, label, higher_better in SCORED:
        ranked = sorted(
            (h for h in pool if not math.isnan(getattr(h, attr))),
            key=lambda h: getattr(h, attr),
            reverse=higher_better,
        )
        for i, h in enumerate(ranked):
            h.points += 1
            if i < top_k:
                h.points += 1
                h.top_in = (*h.top_in, label)


def apply_cuts(
    pool: list[HitterLine],
    league_xwoba: float,
    *,
    min_pa: int = MIN_BATTER_PA,
    min_wrc: float = MIN_WRC,
    keep_power: bool = True,
) -> list[HitterLine]:
    """Run the four cuts in order and return the survivors.

    Order matters: the PA floor comes first so the top-K bonuses are recomputed
    within a pool that no longer contains a two-week hot streak, and the
    expected-contact cut comes last so a hitter is only dropped for unearned
    results after he has proved he belongs on every other axis.

    ``keep_power`` restores a hitter the wRC+ cut drops when his contact quality
    is high enough to matter for home runs specifically -- the Riley case. He is
    flagged, not silently promoted.
    """
    survivors = [h for h in pool if h.pa >= min_pa]
    for h in pool:
        if h.pa < min_pa:
            h.cut_reason = f"under {min_pa} PA"
    score_pool(survivors)

    stage = []
    for h in survivors:
        if not h.top_in:
            h.cut_reason = f"no top-{TOP_K} finish"
        else:
            stage.append(h)

    kept = []
    for h in stage:
        if h.wrc >= min_wrc:
            kept.append(h)
        elif keep_power and h.xwoba_con >= POWER_XWOBACON:
            h.power_exception = True
            kept.append(h)
        else:
            h.cut_reason = f"wRC+ {h.wrc:.0f} under {min_wrc:.0f}"

    final = []
    for h in kept:
        if h.xwoba_pa < league_xwoba + MIN_XWOBA_EDGE and not h.power_exception:
            h.cut_reason = f"xwOBA {h.xwoba_pa:.3f} at league"
        elif h.luck_gap > MAX_LUCK_GAP and not h.power_exception:
            h.cut_reason = f"wOBA outruns xwOBA by {h.luck_gap:+.3f}"
        else:
            h.kept = True
            final.append(h)
    return sorted(final, key=lambda h: (-h.points, -h.xwoba_con))


# --- stage 4: the arsenal ------------------------------------------------


def arsenal(rows: pd.DataFrame, *, min_pitches: int = MIN_PITCHER_PITCHES) -> tuple[
    dict[str, ContactLine], dict[str, float]
]:
    """Per-family contact lines and usage share for one pitcher."""
    if rows.empty or "pitch_type" not in rows:
        return {}, {}
    fam = rows.assign(_fam=rows["pitch_type"].map(pitch_family))
    total = int(len(fam))
    lines: dict[str, ContactLine] = {}
    usage: dict[str, float] = {}
    for name, group in fam.groupby("_fam"):
        if len(group) < min_pitches:
            continue
        lines[str(name)] = contact_line(group)
        usage[str(name)] = len(group) / total
    return lines, usage


def batter_arsenal(
    rows: pd.DataFrame, families: list[str], *, min_pitches: int = MIN_BATTER_PITCHES
) -> dict[str, ContactLine]:
    """The hitter's own line against each family the starter throws."""
    if rows.empty or "pitch_type" not in rows:
        return {}
    fam = rows.assign(_fam=rows["pitch_type"].map(pitch_family))
    out: dict[str, ContactLine] = {}
    for name in families:
        group = fam[fam["_fam"] == name]
        if len(group) >= min_pitches:
            out[name] = contact_line(group)
    return out


def arsenal_fit(
    hitter: dict[str, ContactLine], overall: ContactLine, usage: dict[str, float]
) -> tuple[float, float, float]:
    """(fit xwOBA, fit xBA, share falling back) for one hitter/arsenal pair.

    The hitter's own per-pitch marks weighted by the starter's usage: what his
    line becomes if every pitch he sees comes out of this arsenal. Families he
    has no readable sample against fall back to his overall mark, so the weights
    still sum to one and the fallback share is reported rather than hidden.
    """
    if not usage:
        return math.nan, math.nan, 1.0
    total = sum(usage.values())
    fit_w = fit_b = fallback = 0.0
    for name, share in usage.items():
        weight = share / total
        line = hitter.get(name)
        if line is None or math.isnan(line.xwoba):
            fallback += weight
            fit_w += weight * overall.xwoba
            fit_b += weight * overall.xba
        else:
            fit_w += weight * line.xwoba
            fit_b += weight * (line.xba if not math.isnan(line.xba) else overall.xba)
    return fit_w, fit_b, fallback


# --- stage 5: exposure ---------------------------------------------------


@dataclass(frozen=True)
class Exposure:
    """How much of a hitter's game is the matchup the screen selected."""

    pa_vs_starter: float
    pa_total: float
    pa_vs_pen: float
    third_look: float
    opponent_xwoba: float

    @property
    def share_vs_starter(self) -> float:
        return self.pa_vs_starter / self.pa_total if self.pa_total else math.nan


@dataclass
class BullpenCard:
    """The relief corps that covers whatever the starter does not."""

    team: str
    rank: int  # 1 = softest of the pens profiled
    of_n: int
    relief_pa: int
    xwoba: float
    k_pct: float
    bb_pct: float
    hr_pct: float
    zone_pct: float
    late_k_pct: float


@dataclass
class HitterView:
    """A surviving hitter with everything the last stages added."""

    line: HitterLine
    per_pitch: dict[str, ContactLine]
    overall: ContactLine
    fit_xwoba: float
    fit_xba: float
    fallback_share: float
    exposure: Exposure | None = None
    #: Stages 6-8. Absent when the hitter's season rows were not loaded, in which
    #: case the composite ranking simply does not include him.
    early: HalfLine | None = None
    late: HalfLine | None = None
    context: ContextTerms | None = None
    trends: TrendDeltas | None = None
    edge: ArsenalEdge | None = None

    @property
    def fit_delta(self) -> float:
        return self.fit_xwoba - self.overall.xwoba


@dataclass
class MatchupSection:
    """One starter, his arsenal, the pen behind him, and the bats that survived."""

    starter: StarterCard
    bullpen: BullpenCard | None
    hitters: list[HitterView]
    starter_bf: float
    starter_bf_sd: float
    starter_bf_cap: int
    pitches_per_pa: float
    pitch_cap: int
    discipline: float
    lineup_projected: bool


@dataclass
class ScreenResult:
    """Everything the report needs, with the provenance of each window."""

    as_of: Date
    form_days: int
    window_start: Date
    window_end: Date
    league_woba: dict[str, float]  # hand -> league wOBA over the window
    league_xwoba: dict[str, float]
    starters_ranked: list[StarterCard]
    sections: list[MatchupSection]
    #: Every eligible arm, scored, worst first. The per-split rankings read this
    #: rather than ``starters_ranked``: an arm can miss the note's shortlist on his
    #: total and still be the worst on the slate the third time through the order,
    #: and a ranking that silently drops him is not that ranking.
    starters_scored: list[StarterCard] = field(default_factory=list)
    #: Stages 6-8 pooled across the slate and ranked best first: the two game
    #: halves, the context points and the arsenal fit. Empty when the season rows
    #: the halves need were not loaded.
    final: list[FinalScore] = field(default_factory=list)
    cut_log: list[HitterLine] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    has_run_value: bool = True
    siera_floor: float = SIERA_FLOOR
    starter_cuts: list[StarterCut] = field(default_factory=list)
    splits: tuple[StarterSplit, ...] = STARTER_SPLITS
    has_xera: bool = True  # false when Savant's board could not be read
    #: Stage 9: the engine's projection and surviving prices for the hitters the
    #: screen kept and the arms they face. ``None`` when the screen ran without
    #: pricing, which is the cheap path -- it costs a slate of odds credits.
    bets: BetCard | None = None


def bf_pmf(mean: float, sd: float, cap: int, limit: int = 45) -> list[float]:
    """P(batters faced = k), normal about the model's mean, truncated at the cap.

    The starter's exit point is a managerial decision with a fat tail on the
    short side, but the engine's own model gives a point estimate and his game
    log gives a spread; a truncated normal is the least that does not pretend
    the point estimate is certain.
    """
    sd = max(sd, 1.0)
    weights = [
        math.exp(-0.5 * ((k - mean) / sd) ** 2) if k <= cap else 0.0
        for k in range(limit + 1)
    ]
    total = sum(weights)
    if total <= 0:
        return [0.0] * (limit + 1)
    return [w / total for w in weights]


def pa_vs_starter(slot: int, pmf: list[float]) -> float:
    """Expected turns against the starter for a hitter batting ``slot``.

    Slot ``i`` gets a ``t``-th look whenever batters faced reaches ``9(t-1)+i``,
    so the leadoff man's third look arrives at 19 and the nine-hole's at 27.
    """
    survival = [sum(pmf[k:]) for k in range(len(pmf))]
    total = 0.0
    for turn in range(1, 6):
        idx = 9 * (turn - 1) + slot
        if idx < len(survival):
            total += survival[idx]
    return total


def third_look_prob(slot: int, pmf: list[float]) -> float:
    idx = 18 + slot
    return sum(pmf[idx:]) if idx < len(pmf) else 0.0


def exposure(
    slot: int,
    pa_total: float,
    pmf: list[float],
    starter_xwoba: float,
    pen_xwoba: float | None,
) -> Exposure:
    """Split a projected game into starter and bullpen, and weight the opponent."""
    vs_sp = min(pa_vs_starter(slot, pmf), pa_total)
    vs_pen = max(pa_total - vs_sp, 0.0)
    pen = starter_xwoba if pen_xwoba is None else pen_xwoba
    weighted = (
        (vs_sp * starter_xwoba + vs_pen * pen) / pa_total
        if pa_total and not math.isnan(starter_xwoba)
        else math.nan
    )
    return Exposure(
        pa_vs_starter=vs_sp,
        pa_total=pa_total,
        pa_vs_pen=vs_pen,
        third_look=third_look_prob(slot, pmf),
        opponent_xwoba=weighted,
    )


# --- stage 6: both halves of the game ------------------------------------


@dataclass(frozen=True)
class HalfMetric:
    """One metric in the game-half score, with the sample it stabilizes on.

    ``k`` is the denominator at which the split rate is trusted half as much as
    the hitter's own all-innings rate. It is the stabilization point of the
    metric, not a floor: a thin cell is shrunk rather than dropped, because a
    hitter who has only 40 late plate appearances still has a late half and the
    honest estimate of it is what he does the rest of the game.
    """

    attr: str
    label: str
    higher_better: bool
    k: float
    denom: str


#: Barrel rate is deliberately absent. It needs roughly 200 batted balls and a
#: game-half is a third of a hitter's work, so a season of it does not reach the
#: sample -- sweet-spot rate and exit velocity on air contact carry the same
#: information at a fifth of the sample and are scored in its place.
HALF_SCORED: tuple[HalfMetric, ...] = (
    HalfMetric("k", "K%", False, 60, "pa"),
    HalfMetric("swstr", "SwStr%", False, 250, "pitches"),
    HalfMetric("zcon", "Z-Con%", True, 80, "z_swings"),
    HalfMetric("bb", "BB%", True, 120, "pa"),
    HalfMetric("osw", "O-Swing%", False, 80, "oz_pitches"),
    HalfMetric("sweet", "Sweet%", True, 60, "bbe"),
    HalfMetric("ev_fbld", "EV FB/LD", True, 40, "fbld"),
    HalfMetric("hh", "HH%", True, 50, "bbe"),
    HalfMetric("ev90", "EV90", True, 40, "bbe"),
)

#: The inning the bullpen's half begins. Innings 1-6 are the starter's half; the
#: seventh is where the screen stops measuring the arm it faded and starts
#: measuring the relief corps behind him.
SPLIT_INNING = 7

#: Launch angles that produce line drives and fly balls -- the sweet spot.
SWEET_ANGLES = (8.0, 32.0)


@dataclass
class HalfLine:
    """One hitter's line over one half of the game, shrunk toward himself."""

    half: str  # "early" or "late"
    values: dict[str, float] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    pa: int = 0
    bbe: int = 0
    points: int = 0
    top_in: tuple[str, ...] = ()

    def value(self, metric: HalfMetric) -> float:
        return self.values.get(metric.attr, math.nan)

    def sample(self, metric: HalfMetric) -> int:
        return self.samples.get(metric.denom, 0)


def half_rates(rows: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    """The nine half-score metrics and their denominators for one set of rows."""
    if rows.empty:
        return {}, {}
    desc = rows["description"] if "description" in rows else pd.Series(dtype=object)
    swings = desc.isin(SWING_DESC)
    in_zone = rows["zone"] <= 9 if "zone" in rows else pd.Series(False, index=rows.index)
    z_swings = int((swings & in_zone).sum())
    oz = rows[~in_zone] if "zone" in rows else rows.iloc[0:0]
    ev = _pa_events(rows)
    pa = int(len(ev))
    bip = batted_balls(rows)
    ls = bip["launch_speed"].dropna() if len(bip) else pd.Series(dtype=float)
    la = bip["launch_angle"].dropna() if len(bip) else pd.Series(dtype=float)
    lo, hi = SWEET_ANGLES
    air = bip[(bip["launch_angle"] >= lo)] if len(bip) else bip
    air_ev = air["launch_speed"].dropna() if len(air) else pd.Series(dtype=float)
    samples = {
        "pa": pa,
        "pitches": int(len(rows)),
        "z_swings": z_swings,
        "oz_pitches": int(len(oz)),
        "bbe": int(len(bip)),
        "fbld": int(len(air_ev)),
    }
    values = {
        "k": float(ev.isin(K_EVENTS).mean()) if pa else math.nan,
        "bb": float(ev.eq("walk").mean()) if pa else math.nan,
        "swstr": float(desc.isin(WHIFF_DESC).mean()) if len(desc) else math.nan,
        "zcon": (
            float((desc.eq("hit_into_play") | desc.eq("foul"))[swings & in_zone].mean())
            if z_swings else math.nan
        ),
        "osw": float(oz["description"].isin(SWING_DESC).mean()) if len(oz) else math.nan,
        "sweet": float(((la >= lo) & (la <= hi)).mean()) if len(la) else math.nan,
        "ev_fbld": float(air_ev.mean()) if len(air_ev) else math.nan,
        "hh": float((ls >= 95).mean()) if len(ls) else math.nan,
        "ev90": float(ls.quantile(0.90)) if len(ls) >= 10 else math.nan,
    }
    return values, samples


def _shrink(split: float, whole: float, n: float, k: float) -> float:
    """The split rate pulled toward the hitter's own all-innings rate."""
    if math.isnan(split):
        return math.nan
    if math.isnan(whole):
        return split
    w = n / (n + k) if n + k else 0.0
    return w * split + (1 - w) * whole


def half_lines(rows: pd.DataFrame, *, split_at: int = SPLIT_INNING) -> tuple[HalfLine, HalfLine]:
    """(innings 1-6, innings ``split_at``+) for one hitter, each shrunk to himself.

    The two halves are measured against everyone, not against the starter's hand:
    after the sixth a hitter is facing the bullpen, so a hand split of the late
    half describes a matchup that will not happen.
    """
    whole, _ = half_rates(rows)
    inning = rows["inning"] if "inning" in rows else pd.Series(dtype=float)
    out: list[HalfLine] = []
    for name, mask in (("early", inning < split_at), ("late", inning >= split_at)):
        part = rows[mask] if len(inning) else rows.iloc[0:0]
        values, samples = half_rates(part)
        line = HalfLine(half=name, samples=samples, pa=samples.get("pa", 0),
                        bbe=samples.get("bbe", 0))
        for metric in HALF_SCORED:
            line.values[metric.attr] = _shrink(
                values.get(metric.attr, math.nan),
                whole.get(metric.attr, math.nan),
                samples.get(metric.denom, 0),
                metric.k,
            )
        out.append(line)
    return out[0], out[1]


def score_halves(pool: list[HalfLine], *, top_n: int = STARTER_TOP_N) -> None:
    """One point per rated metric, two more for a top-``top_n`` finish.

    Each half is scored as its own pool, so a hitter's late points are earned
    against the other hitters' late lines rather than against his own early one.
    """
    for line in pool:
        line.points = 0
        line.top_in = ()
    for metric in HALF_SCORED:
        ranked = sorted(
            (line for line in pool if not math.isnan(line.value(metric))),
            key=lambda line: line.value(metric),
            reverse=metric.higher_better,
        )
        for i, line in enumerate(ranked):
            line.points += 1
            if i < top_n:
                line.points += 2
                line.top_in = (*line.top_in, metric.label)


# --- stage 7: regression, park and weather -------------------------------

#: Noise bands. A term is zero inside its band, because the point of the layer is
#: to catch a hitter who has changed and 0.2 mph of bat speed is not a change.
LUCK_BAND = 0.015  # wOBA points between xwOBA and wOBA
BAT_SPEED_BAND = 0.7  # mph, recent against season
CHASE_BAND = 0.030  # 3 points of out-of-zone swing rate
EV90_BAND = 1.0  # mph
PARK_BAND = 1.5  # park-factor points either side of 100
WEATHER_BAND = 0.02  # 2% on the forecast's home-run multiplier
#: How many of the starter's families the run-value term reads, and the run value
#: per 100 pitches at which it doubles. Three families is most of a start -- the
#: fourth and fifth pitches are shown situationally -- and 2 runs per 100 is a
#: large enough edge on them to be worth more than a sign.
TOP_PITCHES = 3
BIG_RV = 2.0
TREND_DAYS = 21  # the recent window the trends are read over
MIN_TREND_SWINGS = 60
MIN_TREND_BBE = 25
MIN_TREND_OZ = 40


@dataclass(frozen=True)
class ContextTerms:
    """Six signed points of context, the worst-arm point and the run-value point.

    Every term is ``+1`` when it favours the hitter, ``-1`` when it favours the
    pitcher and ``0`` when the evidence does not clear the metric's noise band or
    is missing altogether. Missing is neutral: a hitter whose bat speed was never
    tracked has not slowed down. ``top_rv`` is the one term that can be worth
    more than a point, because a hitter two runs per 100 to the good on the
    pitches he will see most is not marginally better off.
    """

    luck: int = 0
    bat_speed: int = 0
    chase: int = 0
    ev90: int = 0
    park: int = 0
    weather: int = 0
    worst_arm: int = 0
    top_rv: int = 0

    @property
    def regression(self) -> int:
        """The four form terms, which is what 'positive regression' means here."""
        return self.luck + self.bat_speed + self.chase + self.ev90

    @property
    def total(self) -> int:
        return (
            self.regression + self.park + self.weather + self.worst_arm + self.top_rv
        )


@dataclass(frozen=True)
class TrendDeltas:
    """Recent-window minus season, for the three metrics that carry direction."""

    bat_speed: float = math.nan
    chase: float = math.nan
    ev90: float = math.nan
    swings: int = 0
    oz_pitches: int = 0
    bbe: int = 0


def rv_term(rv100: float, *, big: float = BIG_RV) -> int:
    """The run-value point on the starter's most-thrown pitches: ``+-1`` or ``+-3``.

    A hitter who is above water on the three pitches he will see most gets the
    sign, and a hitter who is a long way above water gets two more -- the size of
    the edge is information the sign throws away. Exactly zero run value, and an
    unmeasured one, are both no point.
    """
    if math.isnan(rv100) or rv100 == 0:
        return 0
    point = 1 if rv100 > 0 else -1
    return point * 3 if abs(rv100) >= big else point


def signed_term(delta: float, band: float, *, higher_helps_hitter: bool) -> int:
    """``+1``/``-1``/``0`` for a move, signed toward whoever it helps."""
    if math.isnan(delta) or abs(delta) < band:
        return 0
    up = 1 if delta > 0 else -1
    return up if higher_helps_hitter else -up


def trend_deltas(recent: pd.DataFrame, season: pd.DataFrame) -> TrendDeltas:
    """Bat speed, chase and EV90 over the last few weeks against the season.

    A delta is reported only when the recent window carries enough of the
    metric's own denominator to be a reading; otherwise it is unavailable, which
    the terms treat as neutral rather than as a decline.
    """

    def _bat(df: pd.DataFrame) -> tuple[float, int]:
        if "bat_speed" not in df or df.empty:
            return math.nan, 0
        swung = df[df["description"].isin(SWING_DESC)]["bat_speed"].dropna()
        return (float(swung.mean()) if len(swung) else math.nan), int(len(swung))

    def _chase(df: pd.DataFrame) -> tuple[float, int]:
        if "zone" not in df or df.empty:
            return math.nan, 0
        oz = df[df["zone"] > 9]
        return (
            float(oz["description"].isin(SWING_DESC).mean()) if len(oz) else math.nan,
            int(len(oz)),
        )

    def _ev90(df: pd.DataFrame) -> tuple[float, int]:
        bip = batted_balls(df) if not df.empty else df
        ls = bip["launch_speed"].dropna() if len(bip) else pd.Series(dtype=float)
        return (float(ls.quantile(0.90)) if len(ls) >= 10 else math.nan), int(len(ls))

    bat_now, swings = _bat(recent)
    bat_szn, _ = _bat(season)
    chase_now, oz = _chase(recent)
    chase_szn, _ = _chase(season)
    ev_now, bbe = _ev90(recent)
    ev_szn, _ = _ev90(season)
    return TrendDeltas(
        bat_speed=bat_now - bat_szn if swings >= MIN_TREND_SWINGS else math.nan,
        chase=chase_now - chase_szn if oz >= MIN_TREND_OZ else math.nan,
        ev90=ev_now - ev_szn if bbe >= MIN_TREND_BBE else math.nan,
        swings=swings,
        oz_pitches=oz,
        bbe=bbe,
    )


def build_context(
    *,
    woba: float,
    xwoba: float,
    trends: TrendDeltas,
    park_factor: float = math.nan,
    weather_hr_mult: float = math.nan,
    worst_arm: bool = False,
    top_pitch_rv: float = math.nan,
) -> ContextTerms:
    """The seven context points for one hitter in one game.

    The luck term is signed on ``xwOBA - wOBA``: a hitter whose expected line is
    above his actual one has been unlucky and regresses up, which is the only
    direction the word "positive" can mean here. It is the opposite sign to the
    stage-3 luck cut, which removes a hitter for the same gap in the other
    direction, and the two are consistent: results ahead of contact quality are
    not evidence.
    """
    return ContextTerms(
        luck=signed_term(xwoba - woba, LUCK_BAND, higher_helps_hitter=True),
        bat_speed=signed_term(trends.bat_speed, BAT_SPEED_BAND, higher_helps_hitter=True),
        chase=signed_term(trends.chase, CHASE_BAND, higher_helps_hitter=False),
        ev90=signed_term(trends.ev90, EV90_BAND, higher_helps_hitter=True),
        park=signed_term(park_factor - 100.0, PARK_BAND, higher_helps_hitter=True),
        weather=signed_term(weather_hr_mult - 1.0, WEATHER_BAND, higher_helps_hitter=True),
        worst_arm=1 if worst_arm else 0,
        top_rv=rv_term(top_pitch_rv),
    )


# --- stage 8: the arsenal edge -------------------------------------------


@dataclass(frozen=True)
class FitMetric:
    attr: str
    label: str
    higher_better: bool  # for the hitter


#: The five marks the fit is scored on, on both sides of the matchup. Run value
#: is the summary and the other four are the mechanism, so all five are scored
#: rather than collapsed: a hitter can beat an arsenal on contact quality and not
#: on run value when the pitches he handles are not the ones thrown for strikes.
FIT_SCORED: tuple[FitMetric, ...] = (
    FitMetric("rv100", "RV/100", True),
    FitMetric("xwoba", "xwOBA", True),
    FitMetric("whiff", "Whiff%", False),
    FitMetric("hh", "HH%", True),
    FitMetric("brl", "Brl%", True),
)


@dataclass
class ArsenalEdge:
    """One hitter against one arsenal, usage-weighted on five marks.

    ``hitter`` is what his own splits become when every pitch comes out of this
    mix. ``allowed`` is what the starter has given up on the same pitches with
    the same weights, so the two are on one axis and can be read together:
    ``pair`` is their mean, the level of the matchup rather than either side of
    it, which is what a fit is. Both sides carry the same direction on all five
    marks -- a hitter who does not whiff meeting a pitcher who does not miss bats
    is a low pair and good for the hitter -- so the mean needs no re-signing.
    """

    hitter: dict[str, float] = field(default_factory=dict)
    allowed: dict[str, float] = field(default_factory=dict)
    pair: dict[str, float] = field(default_factory=dict)
    fallback_share: float = 1.0
    points: int = 0
    top_in: tuple[str, ...] = ()
    #: The starter's most-thrown families, and the hitter's run value per 100
    #: pitches on them alone. Kept apart from the weighted marks above because it
    #: is scored as a context point rather than ranked against the slate: what it
    #: says about a hitter does not depend on who else is playing.
    top_families: tuple[str, ...] = ()
    top_rv: float = math.nan

    def value(self, metric: FitMetric) -> float:
        return self.pair.get(metric.attr, math.nan)


def contact_mark(line: ContactLine, attr: str) -> float:
    """One named mark off a contact line, without reaching for the attribute."""
    marks = {
        "rv100": line.rv100, "xba": line.xba, "xwoba": line.xwoba,
        "hh": line.hh, "brl": line.brl, "whiff": line.whiff,
    }
    return marks[attr]


def _weighted(
    lines: dict[str, ContactLine],
    usage: dict[str, float],
    attr: str,
    fallback: float,
) -> tuple[float, float]:
    """(usage-weighted mark, share of the weight that fell back)."""
    total = sum(usage.values())
    if not total:
        return math.nan, 1.0
    value = covered = missing = 0.0
    for name, share in usage.items():
        weight = share / total
        line = lines.get(name)
        mark = contact_mark(line, attr) if line is not None else math.nan
        if math.isnan(mark):
            missing += weight
            mark = fallback
            if math.isnan(mark):
                continue
        value += weight * mark
        covered += weight
    if not covered:
        return math.nan, 1.0
    # Renormalized over the families that were readable, so a mark is an average
    # of what is known rather than an average diluted toward zero by what is not.
    return value / covered, missing


def arsenal_edge(
    hitter: dict[str, ContactLine],
    overall: ContactLine,
    starter: dict[str, ContactLine],
    usage: dict[str, float],
) -> ArsenalEdge:
    """Weight both sides of the matchup by the mix the starter actually throws."""
    edge = ArsenalEdge()
    if not usage:
        return edge
    fallbacks = 0.0
    for metric in FIT_SCORED:
        own = contact_mark(overall, metric.attr)
        h_val, missing = _weighted(hitter, usage, metric.attr, own)
        a_val, _ = _weighted(starter, usage, metric.attr, math.nan)
        edge.hitter[metric.attr] = h_val
        edge.allowed[metric.attr] = a_val
        if math.isnan(h_val) and math.isnan(a_val):
            edge.pair[metric.attr] = math.nan
        elif math.isnan(a_val):
            edge.pair[metric.attr] = h_val
        elif math.isnan(h_val):
            edge.pair[metric.attr] = a_val
        else:
            edge.pair[metric.attr] = (h_val + a_val) / 2
        fallbacks += missing
    edge.fallback_share = fallbacks / len(FIT_SCORED)
    edge.top_families, edge.top_rv = top_pitch_rv(hitter, usage)
    return edge


def top_pitch_rv(
    hitter: dict[str, ContactLine],
    usage: dict[str, float],
    *,
    top_n: int = TOP_PITCHES,
) -> tuple[tuple[str, ...], float]:
    """(the starter's ``top_n`` families, the hitter's run value per 100 on them).

    Weighted by usage inside those families only, so the pitch he throws a third
    of the time counts a third of the time. No fallback to the hitter's overall
    line here: this term is about these pitches, and if he has not seen them the
    honest answer is that it does not apply.
    """
    top = sorted(usage.items(), key=lambda kv: -kv[1])[:top_n]
    families = tuple(name for name, _ in top)
    rv, _ = _weighted(hitter, dict(top), "rv100", math.nan)
    return families, rv


def score_edges(pool: list[ArsenalEdge], *, top_n: int = STARTER_TOP_N) -> None:
    """One point per rated mark, two more for a top-``top_n`` fit on the slate."""
    for edge in pool:
        edge.points = 0
        edge.top_in = ()
    for metric in FIT_SCORED:
        ranked = sorted(
            (e for e in pool if not math.isnan(e.value(metric))),
            key=lambda e: e.value(metric),
            reverse=metric.higher_better,
        )
        for i, edge in enumerate(ranked):
            edge.points += 1
            if i < top_n:
                edge.points += 2
                edge.top_in = (*edge.top_in, metric.label)


# --- the composite ranking ----------------------------------------------


@dataclass
class FinalScore:
    """What the whole chain says about one hitter, and where each point came from.

    Kept as its own object rather than a single number because the stages are not
    interchangeable: a hitter carried by the arsenal fit and a hitter carried by
    the park are the same total and not the same bet.
    """

    name: str
    team: str
    versus: str
    slot: int | None
    early: HalfLine
    late: HalfLine
    context: ContextTerms
    edge: ArsenalEdge
    pen_rank: int | None = None

    @property
    def halves(self) -> int:
        return self.early.points + self.late.points

    @property
    def total(self) -> int:
        return self.halves + self.edge.points + self.context.total

    @property
    def weakest_half(self) -> int:
        return min(self.early.points, self.late.points)


def rank_final(scores: list[FinalScore]) -> list[FinalScore]:
    """Worst-to-best is meaningless here, so: best total first.

    Ties break on the weaker of the two halves rather than on the total again --
    the screen is looking for hitters who play all game, so between two equal
    totals the one with no bad half is the one it wants.
    """
    return sorted(
        scores,
        key=lambda s: (-s.total, -s.weakest_half, -s.halves, s.name),
    )
