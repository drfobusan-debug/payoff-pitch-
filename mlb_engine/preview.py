"""Per-game preview context emitted by the pipeline for the daily slate report.

The daily ``mlb-engine run`` prices every market, but the reader-facing slate
preview needs the *matchup story* behind those prices: how each starter's stuff
and command stacks up against the lineup he faces, the bullpen each offense will
see late, who is regressing toward (or away from) their Statcast expectation, the
shape of game the simulator expects (blowout vs. coin-flip, low- vs. high-run),
the weather, and the moneyline's implied probability and edge.

The pipeline already computes all of that internally; this module is the small,
JSON-serializable record that carries it out to the report generator so the
article and audio can be built without re-running the simulation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class StarterLine:
    """A starting pitcher's skill line (stuff / command / contact allowed)."""

    name: str
    pitches: int
    k_pct: float
    xk_pct: float  # stuff-based expected K% (CSW/SwStr driven)
    bb_pct: float
    xbb_pct: float  # command-based expected BB% (Zone/chase/F-strike driven)
    csw: float
    whiff: float
    swstr: float
    zone_pct: float
    xwoba_allowed: float
    barrel_allowed: float
    dxwoba: float  # xwOBA - wOBA allowed: positive => bailed out, hits coming
    spin: float | None = None
    hard_hit_allowed: float | None = None
    babip_allowed: float | None = None
    siera: float | None = None
    # Form direction inside the same window: recent half minus earlier half.
    # SIERA and CSW% in rate points, velocity in mph. None => too thin to read.
    siera_trend: float | None = None
    stuff_trend: float | None = None  # CSW%
    vfa_trend: float | None = None  # mph
    # League mean of per-pitcher xwOBA allowed, so this arm can be read as better
    # or worse than average rather than against the hitters' own scale.
    league_xwoba_allowed: float | None = None


@dataclass
class BullpenLine:
    xwoba_allowed: float | None = None
    k_pct: float | None = None
    zone_pct: float | None = None
    recent_load: float | None = None  # >1 => heavier 3-day workload than baseline
    fatigue: float | None = None  # 0-100 StatsAPI workload proxy


@dataclass
class RegFlag:
    name: str
    points: float  # |xwOBA - wOBA| in points (1000*dxwoba)


@dataclass
class LineupLine:
    n: int
    woba: float
    xwoba: float
    dxwoba: float
    xslg: float
    barrel: float
    # regression leaders: `hot` = overperforming (actual > expected, due to cool
    # off); `cold` = underperforming (expected > actual, buy-low / due to heat up)
    hot: list[RegFlag] = field(default_factory=list)
    cold: list[RegFlag] = field(default_factory=list)
    # Team-level context for the matchup verdict: how this offense hits the hand
    # it draws tonight (with its league rank) and how it hits home vs. away.
    vs_hand: str | None = None  # the opposing starter's throwing hand
    split_woba: float | None = None  # team wOBA vs that hand
    split_rank: int | None = None  # 1 = best offense vs that hand
    split_of: int | None = None  # teams ranked in the split
    split_bucket: str | None = None  # top / middle / bottom third
    home_woba: float | None = None
    away_woba: float | None = None
    is_home: bool | None = None
    league_xwoba: float | None = None  # league mean of per-hitter xwOBA
    # Log5 matchup projections from the simulator's own per-hitter rates, so the
    # article can say what this order does against *this* arm and against an
    # average one. Platoon and home/road context are already inside them.
    proj_woba: float | None = None
    proj_woba_vs_league: float | None = None


@dataclass
class BestBet:
    selection: str
    market: str
    odds: float | None
    model_prob: float
    edge: float | None
    ev: float | None
    tier: str


@dataclass
class GamePreview:
    game_date: str
    game_pk: int
    matchup: str
    home: str  # abbrev
    away: str  # abbrev
    home_starter: StarterLine
    away_starter: StarterLine
    home_lineup: LineupLine
    away_lineup: LineupLine
    home_pen: BullpenLine
    away_pen: BullpenLine
    # game shape (home perspective for xrd)
    xrd: float
    xrd_sd: float
    total_mean: float
    p_home_win: float
    p_blowout: float  # P(|margin| >= 4)
    p_close: float  # P(|margin| <= 1)
    # environment
    park_name: str | None = None
    park_factor: float | None = None
    roof: str | None = None
    wx_summary: str | None = None
    wx_hr_mult: float | None = None
    # moneyline market
    home_ml_prob: float = 0.0
    away_ml_prob: float = 0.0
    fav_side: str = "home"
    fav_team: str = ""
    fav_odds: float | None = None
    fav_implied: float | None = None
    fav_edge: float | None = None
    best_bets: list[BestBet] = field(default_factory=list)


def save_previews(previews: list[GamePreview], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in previews], indent=2))


def _starter(d: dict) -> StarterLine:
    return StarterLine(**d)


def _lineup(d: dict) -> LineupLine:
    d = dict(d)
    d["hot"] = [RegFlag(**r) for r in d.get("hot", [])]
    d["cold"] = [RegFlag(**r) for r in d.get("cold", [])]
    return LineupLine(**d)


def load_previews(path: Path) -> list[GamePreview]:
    raw = json.loads(path.read_text())
    out: list[GamePreview] = []
    for d in raw:
        d = dict(d)
        d["home_starter"] = _starter(d["home_starter"])
        d["away_starter"] = _starter(d["away_starter"])
        d["home_lineup"] = _lineup(d["home_lineup"])
        d["away_lineup"] = _lineup(d["away_lineup"])
        d["home_pen"] = BullpenLine(**d["home_pen"])
        d["away_pen"] = BullpenLine(**d["away_pen"])
        d["best_bets"] = [BestBet(**b) for b in d.get("best_bets", [])]
        out.append(GamePreview(**d))
    return out
