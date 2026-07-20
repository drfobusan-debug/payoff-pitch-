"""Distribution-tail bonus/penalty layer.

Players who sit in the tails of the league distribution (>= 2 SD from the mean)
on stable skill metrics get an extra bounded kick beyond the normal regression
signal -- rewarding true outliers and penalizing the bottom tail. Typical
players (|z| < 2) get exactly nothing, so this never double-counts the graded
regression already applied.

Distributions are computed over the same rolling Statcast window already loaded,
across qualified players only:

- Pitchers: K-BB%, CSW%, hard-hit% allowed, barrel% allowed (higher K-BB/CSW and
  lower hard-hit/barrel allowed = better -> suppresses the opposing offense).
- Batters: xwOBA, hard-hit%, barrel% (higher = better -> boosts that offense).

Extra metrics fold into the same z-composite via ``extra_batter_z`` /
``extra_pitcher_z`` (id-keyed directional z's): batter xSLG from the public
Savant leaderboard, and SIERA & Stuff+ (pitchers) + wRC+ & xSLG (batters) from a
FanGraphs custom-report CSV drop-in (see ``data/fangraphs.py``). Anything not
supplied is simply absent -- never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MIN_BBE = 15
MIN_PITCHES = 100
Z_THRESHOLD = 2.0
_PER_METRIC = 0.02
_MAX_EFFECT = 0.05

_CALLED_OR_WHIFF = {"called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"}
_K_EVENTS = ["strikeout", "strikeout_double_play"]
_BB_EVENTS = ["walk", "hit_by_pitch"]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _zmap(series: pd.Series) -> dict[int, float]:
    """Z-score each qualified player's value against the population."""
    s = series.dropna()
    if len(s) < 2:
        return {}
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    if std <= 0:
        return {}
    return {int(k): (float(v) - mean) / std for k, v in s.items()}


@dataclass
class TailAdjuster:
    # player_id -> metric -> directional z (positive z always = "better")
    batter_z: dict[int, dict[str, float]] = field(default_factory=dict)
    pitcher_z: dict[int, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        batter_xslg: dict[int, float] | None = None,
        extra_batter_z: dict[int, dict[str, float]] | None = None,
        extra_pitcher_z: dict[int, dict[str, float]] | None = None,
    ) -> TailAdjuster:
        if df is None or df.empty:
            return cls()
        batter_z = _batter_z(df)
        if batter_xslg:
            for pid, z in _zmap(pd.Series(batter_xslg)).items():
                batter_z.setdefault(pid, {})["xslg"] = z
        pitcher_z = _pitcher_z(df)
        _merge_z(batter_z, extra_batter_z)
        _merge_z(pitcher_z, extra_pitcher_z)
        return cls(batter_z=batter_z, pitcher_z=pitcher_z)

    def _net(self, z: dict[str, float]) -> float:
        elite = sum(1 for v in z.values() if v >= Z_THRESHOLD)
        poor = sum(1 for v in z.values() if v <= -Z_THRESHOLD)
        return _clip((elite - poor) * _PER_METRIC, -_MAX_EFFECT, _MAX_EFFECT)

    def batter_multiplier(self, batter_id: int) -> dict[str, float]:
        z = self.batter_z.get(batter_id)
        if not z:
            return {}
        b = self._net(z)
        if b == 0.0:
            return {}
        m = 1.0 + b
        return {"1B": m, "2B": m, "3B": m, "HR": m}

    def pitcher_multiplier(self, pitcher_id: int) -> dict[str, float]:
        """Applied to the OPPOSING offense: elite arm suppresses hits, adds Ks."""
        z = self.pitcher_z.get(pitcher_id)
        if not z:
            return {}
        b = self._net(z)
        if b == 0.0:
            return {}
        return {"1B": 1.0 - b, "2B": 1.0 - b, "3B": 1.0 - b, "HR": 1.0 - b, "K": 1.0 + b}


def _batter_z(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    batted = df[df["launch_speed"].notna()].copy()
    if batted.empty:
        return {}
    batted["_hard"] = (batted["launch_speed"] >= 95).astype(float)
    batted["_barrel"] = (batted["launch_speed_angle"] == 6).astype(float)
    agg = batted.groupby("batter").agg(
        bbe=("launch_speed", "size"),
        xwoba=("estimated_woba_using_speedangle", "mean"),
        hard_hit=("_hard", "mean"),
        barrel=("_barrel", "mean"),
    )
    agg = agg[agg["bbe"] >= MIN_BBE]
    return _combine_z(
        {
            "xwoba": _zmap(agg["xwoba"]),
            "hard_hit": _zmap(agg["hard_hit"]),
            "barrel": _zmap(agg["barrel"]),
        }
    )


def _pitcher_z(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    work = df.copy()
    work["_csw"] = work["description"].isin(_CALLED_OR_WHIFF).astype(float)
    pg = work.groupby("pitcher").agg(pitches=("_csw", "size"), csw=("_csw", "mean"))
    pg = pg[pg["pitches"] >= MIN_PITCHES]

    ev = work.dropna(subset=["events"]).copy()
    ev["_k"] = ev["events"].isin(_K_EVENTS).astype(float)
    ev["_bb"] = ev["events"].isin(_BB_EVENTS).astype(float)
    eg = ev.groupby("pitcher").agg(k=("_k", "mean"), bb=("_bb", "mean"))
    k_bb = (eg["k"] - eg["bb"]).reindex(pg.index)

    batted = work[work["launch_speed"].notna()].copy()
    batted["_hard"] = (batted["launch_speed"] >= 95).astype(float)
    batted["_barrel"] = (batted["launch_speed_angle"] == 6).astype(float)
    bg = batted.groupby("pitcher").agg(
        bbe=("launch_speed", "size"),
        hard_hit=("_hard", "mean"),
        barrel=("_barrel", "mean"),
    )
    bg = bg[bg["bbe"] >= MIN_BBE].reindex(pg.index)

    # Allowed metrics: lower is better -> negate so positive z = better.
    return _combine_z(
        {
            "k_bb": _zmap(k_bb),
            "csw": _zmap(pg["csw"]),
            "hard_hit_allowed": {k: -v for k, v in _zmap(bg["hard_hit"]).items()},
            "barrel_allowed": {k: -v for k, v in _zmap(bg["barrel"]).items()},
        }
    )


def _merge_z(
    dest: dict[int, dict[str, float]], extra: dict[int, dict[str, float]] | None
) -> None:
    """Merge id-keyed metric->z contributions into ``dest`` in place."""
    if not extra:
        return
    for pid, zmap in extra.items():
        dest.setdefault(pid, {}).update(zmap)


def _combine_z(metric_maps: dict[str, dict[int, float]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for metric, zmap in metric_maps.items():
        for pid, z in zmap.items():
            out.setdefault(pid, {})[metric] = z
    return out
