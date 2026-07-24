"""Statcast regression signals mapped to per-outcome multipliers.

Implements the sensitivity / PPV / NPV framework the user specified:

  HR   : bat speed & max EV (sensitive), barrel rate (PPV), hard-hit% / LA (NPV)
  XBH  : sweet-spot% (sensitive), xSLG (PPV), blast/bat-speed (NPV)
  1B   : whiff% / zone-contact% (sensitive), xBA + sprint speed (PPV),
         pull% grounders (NPV)
  All  : BABIP and dxwOBA (xwOBA - wOBA) luck-regression signals.

Each raw metric is compared to a league baseline and squashed into a bounded
multiplier so no single signal dominates. Multipliers are applied to the matchup
outcome probabilities before simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# League baselines (approximate, recalibratable).
BL_BARREL = 0.080
BL_HARD_HIT = 0.400
BL_SWEET_SPOT = 0.330
BL_BAT_SPEED = 71.5
BL_MAX_EV = 108.0
BL_WHIFF = 0.240
BL_ZONE_CONTACT = 0.820
BL_XBA = 0.250
BL_XSLG = 0.400
BL_BABIP = 0.290
BL_SPRINT = 27.0

MIN_BBE = 15  # minimum batted-ball events for a stable signal


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class BatterRegression:
    bbe: int
    barrel_rate: float
    hard_hit: float
    sweet_spot: float
    bat_speed: float
    max_ev: float
    whiff: float
    zone_contact: float
    xba: float
    xslg: float
    babip: float
    woba: float
    xwoba: float
    sprint_speed: float = BL_SPRINT

    @property
    def dxwoba(self) -> float:
        return self.xwoba - self.woba

    def multipliers(self) -> dict[str, float]:
        """Return bounded outcome multipliers for {1B,2B,3B,HR}."""
        if self.bbe < MIN_BBE:
            return {}

        # --- Home runs ---
        hr = 1.0
        hr *= 1.0 + _clip((self.barrel_rate - BL_BARREL) * 2.5, -0.12, 0.15)  # PPV
        hr *= 1.0 + _clip((self.bat_speed - BL_BAT_SPEED) * 0.010, -0.06, 0.06)  # sensitive
        hr *= 1.0 + _clip((self.max_ev - BL_MAX_EV) * 0.006, -0.05, 0.06)  # sensitive
        if self.hard_hit < 0.30:  # NPV
            hr *= 0.80
        hr = _clip(hr, 0.75, 1.30)

        # --- Extra-base hits (2B/3B) ---
        xbh = 1.0
        xbh *= 1.0 + _clip((self.xslg - BL_XSLG) * 0.30, -0.10, 0.12)  # PPV
        xbh *= 1.0 + _clip((self.sweet_spot - BL_SWEET_SPOT) * 0.60, -0.08, 0.08)  # sensitive
        xbh *= 1.0 + _clip((self.bat_speed - BL_BAT_SPEED) * 0.006, -0.05, 0.05)  # NPV (blast)
        xbh = _clip(xbh, 0.82, 1.20)

        # --- Singles ---
        one = 1.0
        one *= 1.0 + _clip((self.xba - BL_XBA) * 0.60, -0.08, 0.10)  # PPV
        one *= 1.0 + _clip((BL_WHIFF - self.whiff) * 0.30, -0.06, 0.06)  # sensitive
        one *= 1.0 + _clip((self.zone_contact - BL_ZONE_CONTACT) * 0.30, -0.05, 0.05)  # sensitive
        one *= 1.0 + _clip((self.sprint_speed - BL_SPRINT) * 0.010, -0.04, 0.05)  # PPV speed
        one = _clip(one, 0.85, 1.18)

        # --- BABIP / dxwOBA luck regression (nudges contact outcomes) ---
        # Positive dxwOBA (unlucky) -> nudge up; high BABIP + neg dxwOBA -> down.
        luck = _clip(self.dxwoba * 1.5, -0.06, 0.06)
        if self.babip > 0.330 and self.dxwoba < 0:
            luck -= 0.03
        elif self.babip < 0.260 and self.dxwoba > 0:
            luck += 0.03
        contact_luck = 1.0 + luck

        return {
            "1B": one * contact_luck,
            "2B": xbh * contact_luck,
            "3B": xbh * contact_luck,
            "HR": hr,
        }


def _rate(series: pd.Series, cond) -> float:
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float(cond(s).mean())


def build_batter_regression(
    bdf: pd.DataFrame, sprint_speed: float = BL_SPRINT
) -> BatterRegression:
    """Compute regression metrics from a batter's pitch-level Statcast slice."""
    batted = bdf[bdf["launch_speed"].notna()]
    swings = bdf[
        bdf["description"].isin(
            ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"]
        )
    ]
    whiffs = bdf["description"].isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])

    n_bbe = int(len(batted))
    lsa = batted["launch_speed_angle"].dropna() if "launch_speed_angle" in batted else pd.Series([])
    barrel = float((lsa == 6).mean()) if len(lsa) else 0.0
    hard_hit = float((batted["launch_speed"] >= 95).mean()) if n_bbe else 0.0
    la = batted["launch_angle"].dropna() if "launch_angle" in batted else pd.Series([])
    sweet = float(la.between(8, 32).mean()) if len(la) else 0.0
    bat_speed = (
        float(bdf["bat_speed"].dropna().mean()) if bdf["bat_speed"].notna().any() else BL_BAT_SPEED
    )
    max_ev = float(batted["launch_speed"].max()) if n_bbe else BL_MAX_EV
    n_sw = int(len(swings))
    whiff = float(whiffs.sum() / n_sw) if n_sw else BL_WHIFF
    zc_swings = swings["zone"].between(1, 9) if "zone" in swings else pd.Series(dtype=bool)
    zc_contact = swings["description"].eq("hit_into_play") | swings["description"].isin(
        ["foul", "foul_tip"]
    )
    n_zsw = int(zc_swings.sum()) if len(zc_swings) else 0
    zone_contact = (
        float((zc_swings & zc_contact).sum() / n_zsw) if n_zsw else BL_ZONE_CONTACT
    )
    xba = (
        float(batted["estimated_ba_using_speedangle"].dropna().mean())
        if n_bbe and "estimated_ba_using_speedangle" in batted
        else BL_XBA
    )
    xwoba = (
        float(batted["estimated_woba_using_speedangle"].dropna().mean())
        if n_bbe and "estimated_woba_using_speedangle" in batted
        else float("nan")
    )
    # actual wOBA over the same batted balls (contact-only comparison for dxwOBA)
    woba = (
        float(batted["woba_value"].dropna().mean())
        if n_bbe and "woba_value" in batted
        else float("nan")
    )
    xslg = _estimate_xslg(batted)
    babip = _babip(bdf)

    if np.isnan(xwoba):
        xwoba = BL_XBA
    if np.isnan(woba):
        woba = xwoba

    return BatterRegression(
        bbe=n_bbe,
        barrel_rate=barrel,
        hard_hit=hard_hit,
        sweet_spot=sweet,
        bat_speed=bat_speed,
        max_ev=max_ev,
        whiff=whiff,
        zone_contact=zone_contact,
        xba=xba,
        xslg=xslg,
        babip=babip,
        woba=woba,
        xwoba=xwoba,
        sprint_speed=sprint_speed,
    )


def _estimate_xslg(batted: pd.DataFrame) -> float:
    """Approximate xSLG from launch parameters (no clean per-pitch column).

    Uses estimated wOBA-on-contact scaled to a slugging-like range as a proxy.
    """
    if len(batted) == 0 or "estimated_woba_using_speedangle" not in batted:
        return BL_XSLG
    ewoba = batted["estimated_woba_using_speedangle"].dropna()
    if ewoba.empty:
        return BL_XSLG
    # rough affine map from xwOBAcon to xSLG scale
    return float(_clip(ewoba.mean() * 1.35, 0.10, 1.20))


def _babip(bdf: pd.DataFrame) -> float:
    ev = bdf["events"].dropna()
    if ev.empty:
        return BL_BABIP
    hits = ev.isin(["single", "double", "triple"]).sum()
    hr = ev.eq("home_run").sum()
    k = ev.isin(["strikeout", "strikeout_double_play"]).sum()
    bb = ev.isin(["walk", "hit_by_pitch"]).sum()
    sf = ev.eq("sac_fly").sum()
    ab_in_play = len(ev) - bb - hr - k  # balls in play (excl HR)
    denom = ab_in_play + sf
    if denom <= 0:
        return BL_BABIP
    return float((hits) / denom)


# Pitcher baselines.
BL_CSW = 0.280
BL_K_PCT = 0.220
BL_BB_PCT = 0.080
BL_K_MINUS_BB = 0.140
BL_BARREL_ALLOWED = 0.080
BL_TWO_STRIKE_WHIFF = 0.280
BL_STUFF_PLUS = 100.0
BL_LOCATION_PLUS = 100.0
BL_SWSTR = 0.110  # swinging strikes / pitches
BL_WHIFF_PITCHER = 0.240  # swinging strikes / swings induced

# Stuff-based expected-K% fit: xK% is a linear function of the two fastest-
# stabilizing whiff signals (CSW% and SwStr%), anchored so a league-average arm
# (CSW .280, SwStr .110) maps to ~.220 K%. Used as the small-sample K prior so a
# pitcher regresses toward his stuff, not the flat league mean.
XK_CSW_COEF = 2.6
XK_SWSTR_COEF = 1.4
XK_INTERCEPT = BL_K_PCT - XK_CSW_COEF * BL_CSW - XK_SWSTR_COEF * BL_SWSTR
MIN_SPLIT_PA = 25  # min PA vs a handedness before trusting a pitcher's platoon K%

# Walk (plate-discipline) baselines: Zone%, chase (O-Swing%), first-pitch strike%.
BL_ZONE = 0.480
BL_CHASE = 0.310
BL_FSTRIKE = 0.605

# Stuff-based expected-BB% fit: walks rise when a pitcher throws fewer strikes,
# induces fewer chases, and falls behind more often — these command signals
# stabilize far faster than observed BB%. Each term is (baseline - value), so a
# league-average arm maps to ~.085 and a high-Zone/high-chase arm (NPV screen)
# maps below it. Used as the small-sample BB prior.
XBB_ZONE_COEF = 0.50
XBB_CHASE_COEF = 0.40
XBB_FSTRIKE_COEF = 0.30

CALLED_OR_WHIFF = {"called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING_DESC = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
K_EVENTS_P = ["strikeout", "strikeout_double_play"]


@dataclass
class PitcherRegression:
    bbe: int
    pitches: int
    babip_allowed: float
    woba_allowed: float
    xwoba_allowed: float
    hard_hit_allowed: float
    barrel_allowed: float
    csw: float
    k_pct: float
    bb_pct: float
    two_strike_whiff: float
    swstr: float = BL_SWSTR
    whiff: float = BL_WHIFF_PITCHER
    zone_pct: float = BL_ZONE
    chase: float = BL_CHASE
    fstrike: float = BL_FSTRIKE
    k_pct_vs_l: float = float("nan")
    k_pct_vs_r: float = float("nan")
    ivb: float = float("nan")
    extension: float = float("nan")
    release_var: float = float("nan")
    spin: float = float("nan")
    # Optional FanGraphs pitch-modeling metrics (None if no subscription data).
    stuff_plus: float | None = None
    location_plus: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def dxwoba(self) -> float:
        return self.xwoba_allowed - self.woba_allowed

    @property
    def k_minus_bb(self) -> float:
        return self.k_pct - self.bb_pct

    def expected_k_pct(self) -> float:
        """Stuff-based expected K% (xK%) from CSW% and SwStr%.

        These whiff signals stabilize in far fewer pitches than observed K%, so
        xK% is the right small-sample prior: a hard-to-hit arm with a thin PA
        sample should regress toward his stuff, not the flat league mean.
        """
        xk = XK_INTERCEPT + XK_CSW_COEF * self.csw + XK_SWSTR_COEF * self.swstr
        return _clip(xk, 0.08, 0.42)

    def expected_bb_pct(self) -> float:
        """Command-based expected BB% from Zone%, chase (O-Swing%), and F-strike%.

        These discipline signals stabilize far faster than observed BB%, so they
        are the right small-sample prior: fewer strikes / fewer chases / more
        first-pitch balls all push walks up; the reverse (the NPV screen) pushes
        them below the league mean.
        """
        xbb = (
            BL_BB_PCT
            + XBB_ZONE_COEF * (BL_ZONE - self.zone_pct)
            + XBB_CHASE_COEF * (BL_CHASE - self.chase)
            + XBB_FSTRIKE_COEF * (BL_FSTRIKE - self.fstrike)
        )
        return _clip(xbb, 0.02, 0.20)

    def platoon_k_multiplier(self, bats: str | None) -> float:
        """K-rate multiplier for a batter of a given handedness (vs-L / vs-R split).

        Returns 1.0 when handedness is unknown or the split sample is too thin.
        """
        if bats not in ("L", "R") or self.k_pct <= 0:
            return 1.0
        split = self.k_pct_vs_l if bats == "L" else self.k_pct_vs_r
        if split != split:  # NaN -> insufficient split sample
            return 1.0
        return _clip(split / self.k_pct, 0.85, 1.18)

    def k_multiplier(self) -> float:
        """Multiplier on the pitcher's projected strikeout rate.

        Driven by CSW% and K-BB% (fast-stabilizing K predictors), the 2-strike
        put-away whiff rate, and Stuff+ when a FanGraphs feed is present.
        """
        if self.pitches < 100:
            return 1.0
        m = 1.0
        m *= 1.0 + _clip((self.csw - BL_CSW) * 2.5, -0.15, 0.20)  # highest baseline PPV
        m *= 1.0 + _clip((self.k_minus_bb - BL_K_MINUS_BB) * 1.5, -0.12, 0.15)
        m *= 1.0 + _clip((self.two_strike_whiff - BL_TWO_STRIKE_WHIFF) * 0.8, -0.06, 0.08)
        if self.stuff_plus is not None:
            m *= 1.0 + _clip((self.stuff_plus - BL_STUFF_PLUS) * 0.004, -0.10, 0.15)
        return _clip(m, 0.75, 1.30)

    def allowed_multipliers(self) -> dict[str, float]:
        """Multipliers on outcomes the pitcher ALLOWS (hits/xbh/hr)."""
        if self.bbe < MIN_BBE:
            return {}
        base = 1.0
        # High BABIP allowed -> positive regression (fewer hits going forward).
        base *= 1.0 + _clip((BL_BABIP - self.babip_allowed) * 0.6, -0.08, 0.08)
        # Positive dxwOBA allowed (getting bailed out) -> more hits coming.
        base *= 1.0 + _clip(self.dxwoba * 1.2, -0.06, 0.08)
        base = _clip(base, 0.88, 1.14)

        # Barrel rate allowed drives HR specifically (highest PPV for HR/9).
        hr = base * (1.0 + _clip((self.barrel_allowed - BL_BARREL_ALLOWED) * 2.0, -0.10, 0.18))
        hr = _clip(hr, 0.85, 1.35)
        return {"1B": base, "2B": base, "3B": base, "HR": hr}


def build_pitcher_regression(
    pdf: pd.DataFrame,
    stuff_plus: float | None = None,
    location_plus: float | None = None,
) -> PitcherRegression:
    batted = pdf[pdf["launch_speed"].notna()]
    n_bbe = int(len(batted))
    n_pitches = int(len(pdf))
    babip = _babip(pdf)
    hard_hit = float((batted["launch_speed"] >= 95).mean()) if n_bbe else BL_HARD_HIT
    barrel = (
        float((batted["launch_speed_angle"] == 6).mean())
        if n_bbe and "launch_speed_angle" in batted
        else BL_BARREL_ALLOWED
    )
    xwoba = (
        float(batted["estimated_woba_using_speedangle"].dropna().mean())
        if n_bbe and "estimated_woba_using_speedangle" in batted
        else BL_XBA
    )
    woba = (
        float(batted["woba_value"].dropna().mean())
        if n_bbe and "woba_value" in batted
        else xwoba
    )

    # CSW% = (called strikes + whiffs) / pitches
    if n_pitches and "description" in pdf:
        csw = float(pdf["description"].isin(CALLED_OR_WHIFF).mean())
    else:
        csw = BL_CSW

    # SwStr% (whiffs / pitches) and whiff-per-swing: fastest-stabilizing K signals.
    if n_pitches and "description" in pdf:
        n_whiff = int(pdf["description"].isin(WHIFF_DESC).sum())
        n_swings = int(pdf["description"].isin(SWING_DESC).sum())
        swstr = n_whiff / n_pitches
        whiff = n_whiff / n_swings if n_swings else BL_WHIFF_PITCHER
    else:
        swstr, whiff = BL_SWSTR, BL_WHIFF_PITCHER

    # Command signals for the expected-BB% prior: Zone%, chase (O-Swing%), F-strike%.
    if n_pitches and "zone" in pdf and pdf["zone"].notna().any():
        in_zone = pdf["zone"].between(1, 9)
        zone_pct = float(in_zone.mean())
        out_zone = pdf[~in_zone]
        n_out = len(out_zone)
        chase = (
            float(out_zone["description"].isin(SWING_DESC).sum() / n_out)
            if n_out
            else BL_CHASE
        )
    else:
        zone_pct, chase = BL_ZONE, BL_CHASE

    # First-pitch strike% = share of 0-0 pitches that are not balls (type != "B").
    if n_pitches and "balls" in pdf and "strikes" in pdf and "type" in pdf:
        first = pdf[(pdf["balls"] == 0) & (pdf["strikes"] == 0)]
        fstrike = float((first["type"] != "B").mean()) if len(first) else BL_FSTRIKE
    else:
        fstrike = BL_FSTRIKE

    # K% / BB% over plate appearances the pitcher ended.
    ev = pdf["events"].dropna()
    pa = len(ev)
    if pa:
        k_pct = float(ev.isin(K_EVENTS_P).sum() / pa)
        bb_pct = float(ev.isin(["walk", "hit_by_pitch"]).sum() / pa)
    else:
        k_pct, bb_pct = BL_K_PCT, BL_BB_PCT

    # Platoon K% split (vs LHB / vs RHB) from the batter's stance on each PA.
    k_pct_vs_l = k_pct_vs_r = float("nan")
    if pa and "stand" in pdf:
        pa_rows = pdf[pdf["events"].notna()]
        for hand, target in (("L", "l"), ("R", "r")):
            side = pa_rows[pa_rows["stand"] == hand]["events"]
            if len(side) >= MIN_SPLIT_PA:
                rate = float(side.isin(K_EVENTS_P).sum() / len(side))
                if target == "l":
                    k_pct_vs_l = rate
                else:
                    k_pct_vs_r = rate

    # 2-strike put-away whiff rate.
    if "strikes" in pdf and n_pitches:
        two_strike = pdf[pdf["strikes"] == 2]
        n2 = len(two_strike)
        two_strike_whiff = (
            float(two_strike["description"].isin(WHIFF_DESC).sum() / n2)
            if n2
            else BL_TWO_STRIKE_WHIFF
        )
    else:
        two_strike_whiff = BL_TWO_STRIKE_WHIFF

    # Induced vertical break of fastball-ish pitches (pfx_z in feet -> inches).
    ivb = (
        float(pdf["pfx_z"].dropna().mean() * 12.0)
        if "pfx_z" in pdf and pdf["pfx_z"].notna().any()
        else float("nan")
    )

    ext = float(pdf["release_extension"].dropna().mean()) if pdf["release_extension"].notna().any() else float("nan")
    rel_var = (
        float(np.sqrt(pdf["release_pos_x"].dropna().var() + pdf["release_pos_z"].dropna().var()))
        if pdf["release_pos_x"].notna().any()
        else float("nan")
    )
    spin = float(pdf["release_spin_rate"].dropna().mean()) if pdf["release_spin_rate"].notna().any() else float("nan")

    return PitcherRegression(
        bbe=n_bbe,
        pitches=n_pitches,
        babip_allowed=babip,
        woba_allowed=woba,
        xwoba_allowed=xwoba,
        hard_hit_allowed=hard_hit,
        barrel_allowed=barrel,
        csw=csw,
        k_pct=k_pct,
        bb_pct=bb_pct,
        two_strike_whiff=two_strike_whiff,
        swstr=swstr,
        whiff=whiff,
        zone_pct=zone_pct,
        chase=chase,
        fstrike=fstrike,
        k_pct_vs_l=k_pct_vs_l,
        k_pct_vs_r=k_pct_vs_r,
        ivb=ivb,
        extension=ext,
        release_var=rel_var,
        spin=spin,
        stuff_plus=stuff_plus,
        location_plus=location_plus,
    )
