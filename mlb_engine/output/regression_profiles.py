"""Starter and hitter regression profiles: the luck/level read the reports rank.

Moved out of ``scripts/`` so the daily run can build the regression article
itself. The scripts that render the stat cards import these same builders, so
the article the engine emails and the cards a study renders are read off one
implementation rather than two.

A profile carries the *level* a player is (SIERA, Stuff, vFA, xSLG) and the
*luck* term the results have added on top (BABIP, the wOBA-minus-xwOBA gap).
Only the second is due to move; ranking uses it alone.
"""

from __future__ import annotations

from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.audit.ledger import prop_subject
from mlb_engine.features.arm import ArmProfile, build_arm_profile
from mlb_engine.features.arm import stage_two as arm_stage_two
from mlb_engine.features.arm import velo_trend as arm_velo_trend
from mlb_engine.features.power_change import PowerChange, build_power_change
from mlb_engine.features.regression import (
    BL_BABIP,
    BatterRegression,
    build_batter_regression,
    build_pitcher_regression,
)
from mlb_engine.features.siera import pitcher_siera
from mlb_engine.features.swing import SwingProfile, build_swing_profile, stage_two
from mlb_engine.market.ranking import price_rank

FB = ("FF", "SI")
RECENT_DAYS = 21  # "3-week" window for vFA + trend split
MIN_BBE = 25  # a hitter's batted balls before his luck term is worth ranking
TOPN = 10  # hitters printed per direction


def _pitcher_id_map(preds: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in preds:
        if r["market"].startswith("pitcher_") and r.get("player_id"):
            nm = (
                r["selection"]
                .split(" Ks")[0]
                .split(" Outs")[0]
                .split(" Hits")[0]
                .split(" Walks")[0]
                .split(" ER")[0]
            )
            out[nm] = r["player_id"]
    return out


def _starter_games(previews: list[dict]) -> dict[str, dict]:
    """Map each starter -> game context (opponent lineup, park, weather, matchup)."""
    ctx: dict[str, dict] = {}
    for g in previews:
        for side, opp in (("home", "away"), ("away", "home")):
            st = g[f"{side}_starter"]["name"]
            ctx[st] = {
                "matchup": g["matchup"],
                "team": g[side],
                "opp": g[opp],
                "home_away": "home" if side == "home" else "away",
                "opp_lineup_xwoba": g[f"{opp}_lineup"]["xwoba"],
                "park_name": g.get("park_name"),
                "park_factor": g.get("park_factor"),
                "wx_summary": g.get("wx_summary"),
                "wx_hr_mult": g.get("wx_hr_mult"),
                "total_mean": g["total_mean"],
                "fav_team": g.get("fav_team"),
            }
    return ctx


def _vfa(slice_df: pd.DataFrame) -> float:
    if "release_speed" not in slice_df:
        return float("nan")
    fb = slice_df[slice_df["pitch_type"].isin(FB)]
    return _fmean(fb["release_speed"]) if len(fb) else float("nan")


def _stuff_xk(slice_df: pd.DataFrame) -> float:
    if slice_df.empty:
        return float("nan")
    return float(build_pitcher_regression(slice_df).expected_k_pct())


def _siera_val(slice_df: pd.DataFrame) -> float:
    return float(pitcher_siera(slice_df).siera)


def _arr(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _fmean(s: pd.Series) -> float:
    a = _arr(s)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _biomech(slice_df: pd.DataFrame) -> dict[str, float]:
    """The release biomechanics the stat cards print, off the shared arm model.

    Read on the arm's last ``arm.WINDOW`` fastballs rather than over the whole
    slice, so a level is the sample the measure was validated on and a thin arm
    reads as unmeasured instead of averaging two starts with twenty.
    """
    prof = build_arm_profile(slice_df)
    return {"ext": prof.ext, "ivb": prof.ivb, "spin": prof.spin, "scatter": prof.scatter}


def _arm_fields(prof: ArmProfile, dxwoba: float) -> dict[str, float | int | str]:
    """The second stage of the starter read: the delivery, and whether it agrees.

    ``dxwoba`` is xwOBA-allowed minus wOBA-allowed, which is the luck term stage
    two is crossed against directly. The verdict is a level; the trend rides
    along beside it because it earns something on the fade side of the flag and
    nothing on the other, so it qualifies a fade rather than voting on one.
    """
    return {
        "arm_pitches": prof.pitches,
        "arm_velo": prof.velo,
        "arm_pvelo": prof.pvelo,
        "arm_ext": prof.ext,
        "arm_rel_x": prof.rel_x,
        "arm_rel_z": prof.rel_z,
        "arm_spin": prof.spin,
        "arm_ivb": prof.ivb,
        "arm_hb": prof.hb,
        "arm_scatter": prof.scatter,
        "stuff_z": prof.stuff_z,
        "ride_z": prof.ride_z,
        "arm_d_pvelo": prof.d_pvelo,
        "trend_z": prof.trend_z,
        "arm_stage2": arm_stage_two(dxwoba, prof),
        "arm_trend": arm_velo_trend(prof),
    }


def analyze(name: str, pid: int, df: pd.DataFrame, cutoff: Date) -> dict:
    sl = df[df["pitcher"] == pid]
    reg = build_pitcher_regression(sl)
    prof = build_arm_profile(sl)
    sr = pitcher_siera(sl)
    recent = sl[pd.to_datetime(sl["game_date"]).dt.date > cutoff]
    prior = sl[pd.to_datetime(sl["game_date"]).dt.date <= cutoff]
    unlucky_babip = reg.babip_allowed - BL_BABIP  # + => unlucky => positive regression
    unlucky_xwoba = -reg.dxwoba  # dxwoba<0 (woba>xwoba) => unlucky => positive
    return {
        "name": name,
        "pitches": int(len(sl)),
        "siera": sr.siera,
        "siera_pa": sr.pa,
        "babip": reg.babip_allowed,
        "dxwoba": reg.dxwoba,
        "xwoba": reg.xwoba_allowed,
        "woba": reg.woba_allowed,
        "csw": reg.csw,
        "xk": reg.expected_k_pct(),
        "k_pct": reg.k_pct,
        "bb_pct": reg.bb_pct,
        "barrel": reg.barrel_allowed,
        "fb": reg.fb_allowed,
        "gb": reg.gb_allowed,
        "vfa": _vfa(sl),
        "biomech": {
            "ext": prof.ext,
            "ivb": prof.ivb,
            "spin": prof.spin,
            "scatter": prof.scatter,
        },
        **_arm_fields(prof, reg.dxwoba),
        "unlucky_babip": unlucky_babip,
        "unlucky_xwoba": unlucky_xwoba,
        # recent-vs-prior trends (recent minus prior)
        "d_siera": _siera_val(recent) - _siera_val(prior),
        "d_xk": _stuff_xk(recent) - _stuff_xk(prior),
        "d_vfa": _vfa(recent) - _vfa(prior),
    }


def _rank_of(row: dict) -> float:
    """Rank a persisted prediction row the way the card ranks a live one.

    A stored row's ``fair_prob`` is absent or blank on a market nothing could be
    devigged, which :func:`price_rank` reads as "raw price only" rather than as
    a probability of zero.
    """

    def num(key: str) -> float | None:
        v = row.get(key)
        return float(v) if isinstance(v, int | float) else None

    return price_rank(num("market_american"), num("fair_prob"), num("ev"))


def _bets_for(pid: int, preds: list[dict]) -> list[dict]:
    out = []
    for r in preds:
        if r.get("player_id") == pid and r["market"].startswith("pitcher_"):
            out.append(r)
    # buys first, then by the devigged price on them
    tier_rank = {"Strong buy": 0, "Moderate buy": 1, "Pass": 2}
    out.sort(key=lambda r: (tier_rank.get(r["tier"], 3), _rank_of(r)))
    return out


def build_profiles(previews: list[dict], preds: list[dict], df: pd.DataFrame):
    """Return (pos, neg, ctxs) starter regression profiles for the slate."""
    idmap = _pitcher_id_map(preds)
    ctxs = _starter_games(previews)
    maxd = pd.to_datetime(df["game_date"]).dt.date.max()
    cutoff = maxd - pd.Timedelta(days=RECENT_DAYS)
    cutoff = cutoff if isinstance(cutoff, Date) else cutoff.date()

    profiles = []
    for name in sorted(ctxs):
        pid = idmap.get(name)
        if pid is None:
            continue
        p = analyze(name, pid, df, cutoff)
        if p["pitches"] < 150:  # too thin to trust the luck read
            continue
        profiles.append(p)

    # z-score the two luck components across the slate, then combine.
    for key in ("unlucky_babip", "unlucky_xwoba"):
        vals = np.array([p[key] for p in profiles])
        mu, sd = vals.mean(), vals.std() or 1.0
        for p in profiles:
            p[f"z_{key}"] = (p[key] - mu) / sd
    for p in profiles:
        p["reg_index"] = p["z_unlucky_babip"] + p["z_unlucky_xwoba"]

    profiles.sort(key=lambda p: -p["reg_index"])
    pos = [p for p in profiles if p["reg_index"] > 0]
    neg = [p for p in profiles if p["reg_index"] <= 0]
    neg.sort(key=lambda p: p["reg_index"])  # most negative first
    return pos, neg, ctxs


# --- hitters ---------------------------------------------------------------
# selection is "{name} {stat} {side}{line}" ("Matt McLain 1B o0.5", "Carlos
# Narvaez H+R+RBI u1.5"); the hitter is what is left once the stat and the side
# marker are dropped. A local pattern for the marker is what put ten cards of one
# man in a top ten -- it knew only the over, so every under kept the market glued
# to the name and read as a separate hitter carrying identical contact. The
# ledger already parses the marker to grade with, so read it there rather than
# keep a copy that has to learn every new market and side by hand.


def _batter_name(sel: str) -> str:
    subject = prop_subject(sel)
    if subject == sel:  # no side marker: not a prop selection, so chop nothing
        return sel
    return subject.rpartition(" ")[0] or subject


def _batter_id_map(preds: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in preds:
        if r["market"].startswith("batter_") and r.get("player_id"):
            out[_batter_name(r["selection"])] = r["player_id"]
    return out


def _batter_ctx(preds: list[dict], pv_by_pk: dict[int, dict]) -> dict[str, dict]:
    ctx: dict[str, dict] = {}
    for r in preds:
        if not r["market"].startswith("batter_"):
            continue
        sel = _batter_name(r["selection"])
        if sel in ctx:
            continue
        pk = r.get("game_pk")
        g = pv_by_pk.get(int(pk), {}) if pk is not None else {}
        ctx[sel] = {
            "matchup": r.get("matchup", ""),
            "park_factor": g.get("park_factor"),
            "wx_hr_mult": g.get("wx_hr_mult"),
        }
    return ctx


def _woba(slice_df: pd.DataFrame) -> float:
    if slice_df.empty:
        return float("nan")
    return build_batter_regression(slice_df).woba


def _fb_rate(reg: BatterRegression) -> float:
    """Fly balls (with pop-ups) as a share of a hitter's batted balls.

    Statcast's four batted-ball classes exhaust the slice, so the air share is
    what the ground balls and line drives leave behind. NaN when either half is
    unmeasurable.
    """
    gb, ld = reg.gb_pct, reg.ld_pct
    if gb != gb or ld != ld:
        return float("nan")
    return max(0.0, 1.0 - gb - ld)


def _swing_fields(prof: SwingProfile, dxwoba: float) -> dict[str, float | int | str]:
    """The second stage of the hitter read: the swing, and whether it agrees.

    ``dxwoba`` is xwOBA minus wOBA, so the luck gap stage two is crossed against
    is its negative. Levels only -- the recent-versus-prior move in these same
    measures adds nothing out of time (bat speed t +1.4, blast t -0.3 on 3,175
    batter-windows), the same verdict PR #109 reached on the barrel trend, so no
    swing trend is computed for the article to print.
    """
    return {
        "swings": prof.swings,
        "bat_speed": prof.bat_speed,
        "fast": prof.fast,
        "squared_up": prof.squared_up,
        "blast": prof.blast,
        "swing_length": prof.swing_length,
        "attack_angle": prof.attack_angle,
        "power_z": prof.power_z,
        "contact_z": prof.contact_z,
        "lift_z": prof.lift_z,
        "stage2": stage_two(-dxwoba, prof),
    }


def _power_fields(pc: PowerChange) -> dict[str, float | int | bool]:
    """Peak exit velocity and the fastball whiff, each over its own window.

    Both levels forecast (t +8 to +16 on a held-out block); neither *move* does,
    so the move is carried alongside the band a hitter who did not change would
    still produce, and the article prints it as a diagnostic rather than reading
    a direction off it. See :mod:`mlb_engine.features.power_change`.
    """
    return {
        "max_ev": pc.max_ev,
        "max_ev_pa": pc.max_ev_pa,
        "d_max_ev": pc.d_max_ev,
        "max_ev_moved": pc.moved("max_ev"),
        "fb_whiff": pc.fb_whiff,
        "fb_whiff_pa": pc.fb_whiff_pa,
        "fb_swings": pc.fb_swings,
        "d_fb_whiff": pc.d_fb_whiff,
        "fb_whiff_moved": pc.moved("fb_whiff"),
        "power_block_pa": pc.block_pa,
        "power_pa": pc.pa,
    }


def analyze_batter(name: str, pid: int, df: pd.DataFrame, cutoff: Date) -> dict:
    sl = df[df["batter"] == pid]
    reg = build_batter_regression(sl)
    recent = sl[pd.to_datetime(sl["game_date"]).dt.date > cutoff]
    return {
        "name": name,
        "bbe": reg.bbe,
        "woba": reg.woba,
        "xwoba": reg.xwoba,
        "dxwoba": reg.dxwoba,  # xwoba - woba: + => underperforming (heat up)
        "xslg": reg.xslg,
        "barrel": reg.barrel_rate,
        "babip": reg.babip,
        "hard_hit": reg.hard_hit,
        "fb": _fb_rate(reg),
        "gb": reg.gb_pct,
        "iffb": reg.iffb_pct,
        "woba6": reg.woba,
        "woba3": _woba(recent),
        **_swing_fields(build_swing_profile(sl), reg.dxwoba),
        **_power_fields(build_power_change(sl)),
    }


def _best_batter_bet(pid: int, preds: list[dict]) -> dict | None:
    cands = [r for r in preds if r.get("player_id") == pid and r["market"].startswith("batter_")]
    if not cands:
        return None
    tier_rank = {"Strong buy": 0, "Moderate buy": 1, "Pass": 2}
    cands.sort(key=lambda r: (tier_rank.get(r["tier"], 3), _rank_of(r)))
    return cands[0]


def build_batter_profiles(preds: list[dict], df: pd.DataFrame):
    idmap = _batter_id_map(preds)
    maxd = pd.to_datetime(df["game_date"]).dt.date.max()
    cutoff = maxd - pd.Timedelta(days=RECENT_DAYS)
    cutoff = cutoff if isinstance(cutoff, Date) else cutoff.date()
    profs = []
    seen: set[int] = set()
    for name, pid in idmap.items():
        # A hitter is one hitter however his name reaches the sheet: two spellings
        # of the same id must not both be ranked.
        if pid in seen:
            continue
        seen.add(pid)
        p = analyze_batter(name, pid, df, cutoff)
        if p["bbe"] < MIN_BBE:
            continue
        p["pid"] = pid
        profs.append(p)
    pos = sorted([p for p in profs if p["dxwoba"] > 0], key=lambda p: -p["dxwoba"])[:TOPN]
    neg = sorted([p for p in profs if p["dxwoba"] < 0], key=lambda p: p["dxwoba"])[:TOPN]
    return pos, neg
