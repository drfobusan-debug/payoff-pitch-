"""Morning power screen: the softest arms on a slate, and who to hunt them with.

This is the hand method Franz has been running for years, written down. It runs
beside the nightly card rather than inside it, because it answers a narrower
question than the engine does and it answers it before lineups are posted, when
the engine declines to price a game at all.

The chain is five stages, each of which throws work away:

1. **Rank the arms.** Every probable starter is scored on how much damage he
   allows in the air -- barrel and hard-hit rates allowed, fly-ball rate,
   xwOBA-on-contact, home runs per batter faced -- against how little he misses
   bats (K-BB%) and how long he lasts. The top few are the only games that
   proceed.
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

Every figure is observed or modelled, never priced: this module reads no market.
Run value is quoted **from the hitter's side throughout**, so both halves of a
matchup share an axis; Savant prints a pitcher's version with the opposite sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as Date

import pandas as pd

from mlb_engine.data.statcast import batted_balls

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

# Screen thresholds. Defaults are the ones the 8/17 hand run settled on.
MIN_STARTER_BF = 120  # batters faced in the window before an arm is readable
MIN_BATTER_PA = 60  # hand-split PA floor
MIN_WRC = 120  # window wRC+ floor, before the power exception
MIN_XWOBA_EDGE = 0.020  # xwOBA/PA must clear league by this much
MAX_LUCK_GAP = 0.050  # wOBA minus xwOBA/PA above this is unearned
POWER_XWOBACON = 0.440  # contact good enough to keep a hitter the wRC+ cut drops
MIN_PITCHER_PITCHES = 15  # per-pitch-type floor, pitcher side
MIN_BATTER_PITCHES = 25  # per-pitch-type floor, hitter side
TOP_K = 5  # a top-K finish in a scored metric earns the second point

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
    )


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
    """Score every arm on home-run exposure and return the worst ``top_n``.

    The index is deliberately blunt -- an equal-weight z-sum of the five damage
    signals minus the two that describe an arm that can defend itself. It sorts
    a slate; it does not price one.
    """
    readable = [c for c in cards if c.bf >= MIN_STARTER_BF]
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
    return sorted(readable, key=lambda c: -c.index)[:top_n]


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
    """A surviving hitter with everything the last two stages added."""

    line: HitterLine
    per_pitch: dict[str, ContactLine]
    overall: ContactLine
    fit_xwoba: float
    fit_xba: float
    fallback_share: float
    exposure: Exposure | None = None

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
    cut_log: list[HitterLine] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    has_run_value: bool = True


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
