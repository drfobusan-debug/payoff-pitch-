"""Unit tests for the model, market, and audit layers (no network)."""

from __future__ import annotations

from datetime import date

import numpy as np

from mlb_engine.audit.grade import LOSS, PUSH, WIN, grade
from mlb_engine.audit.scorecard import build_scorecard
from mlb_engine.config import EVThresholds
from mlb_engine.data.results import GameResult, PlayerLine, _ip_to_outs
from mlb_engine.features.rolling import LEAGUE_RATES, OutcomeRates, rates_from_events
from mlb_engine.market.ev import MarketQuote, ev_per_dollar, evaluate
from mlb_engine.market.odds import (
    american_to_decimal,
    american_to_prob,
    no_vig_two_way,
    prob_to_american,
)
from mlb_engine.market.tiers import Tier, classify
from mlb_engine.models.markov_f5 import (
    f5_from_lineups,
    f5_from_rates,
    team_f5_distribution,
)
from mlb_engine.models.matchup import apply_multipliers, combine
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.rbi_rule import evaluate_lineup, rbi_multiplier
from mlb_engine.recommendations import Recommendation


def _league_rates() -> OutcomeRates:
    r = LEAGUE_RATES
    return OutcomeRates(500, r["1B"], r["2B"], r["3B"], r["HR"], r["BB"], r["K"], r["OUT"])


# ---- odds math ----
def test_odds_roundtrip():
    for a in (-200, -110, 100, 150, 250):
        p = american_to_prob(a)
        assert 0 < p < 1
    assert abs(american_to_decimal(100) - 2.0) < 1e-9
    assert abs(american_to_decimal(-200) - 1.5) < 1e-9


def test_prob_to_american_inverse():
    for p in (0.4, 0.55, 0.7):
        a = prob_to_american(p)
        assert abs(american_to_prob(a) - p) < 1e-6


def test_no_vig_sums_to_one():
    a, b = no_vig_two_way(-110, -110)
    assert abs(a + b - 1.0) < 1e-9
    assert abs(a - 0.5) < 1e-9


# ---- matchup ----
def test_combine_normalizes():
    lg = _league_rates()
    out = combine(lg, lg)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    # combining league vs league should return ~league rates
    assert abs(out["HR"] - LEAGUE_RATES["HR"]) < 1e-3


def test_apply_multipliers_renormalizes():
    lg = _league_rates()
    base = combine(lg, lg)
    out = apply_multipliers(base, {"HR": 2.0})
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["HR"] > base["HR"]


# ---- markov F5 ----
def test_f5_reasonable():
    lg = LEAGUE_RATES
    r = f5_from_rates(dict(lg), dict(lg))
    mean = sum(i * p for i, p in enumerate(r.home_dist))
    assert 1.5 < mean < 3.5  # ~2.4 runs/team through 5
    assert abs(r.p_home_ml + r.p_away_ml + r.p_tie - 1.0) < 1e-6
    assert abs(sum(r.total_dist) - 1.0) < 1e-6


def test_f5_lineup_matches_convolution_without_tto():
    lg = dict(LEAGUE_RATES)
    conv = f5_from_rates(lg, lg)
    dp = team_f5_distribution([lg] * 9, tto_factors=(1.0, 1.0, 1.0, 1.0))
    m_conv = sum(i * p for i, p in enumerate(conv.home_dist))
    m_dp = sum(i * p for i, p in enumerate(dp))
    assert abs(m_conv - m_dp) < 1e-4


def test_f5_tto_raises_scoring():
    lg = dict(LEAGUE_RATES)
    base = sum(i * p for i, p in enumerate(team_f5_distribution([lg] * 9, (1.0, 1.0, 1.0, 1.0))))
    tto = sum(i * p for i, p in enumerate(team_f5_distribution([lg] * 9)))
    assert tto > base
    r = f5_from_lineups([lg] * 9, [lg] * 9)
    assert abs(r.p_home_ml + r.p_away_ml + r.p_tie - 1.0) < 1e-6


# ---- weather (WAM park-config filter) ----
def test_weather_park_config_gates_wind():
    from mlb_engine.data.parks import get_park
    from mlb_engine.filters.weather import WeatherConditions, _effect

    out = WeatherConditions(85, 50, 15, 0, 15)  # 15 mph straight out to CF
    wrigley, _ = _effect(out, get_park(17))  # open bowl, wind-receptive
    oracle, _ = _effect(out, get_park(2395))  # shielded
    assert wrigley > 1.15  # wind reaches the field -> big HR boost
    assert oracle < wrigley  # architecture suppresses the same wind

    blow_in = WeatherConditions(85, 50, 15, 180, -15)
    wrigley_in, _ = _effect(blow_in, get_park(17))
    assert wrigley_in < 1.0  # in-from-CF suppresses power


# ---- monte carlo ----
def test_montecarlo_runs():
    lg = LEAGUE_RATES
    bat = [dict(lg) for _ in range(9)]
    cfg = TeamSimConfig(bat_vs_starter=bat, bat_vs_pen=bat)
    res = MonteCarlo(400, seed=1).simulate(cfg, cfg)
    assert res.home_runs_full.shape == (400,)
    assert 3.0 < res.home_runs_full.mean() < 6.5
    # home never trails-and-bats: full >= f5
    assert (res.home_runs_full >= res.home_runs_f5).all()


# ---- rbi rule ----
def _profile(rates: OutcomeRates):
    from mlb_engine.features.rolling import BatterProfile

    return BatterProfile(
        mlbam_id=1, home=rates, away=rates, vs_rhp=rates, vs_lhp=rates, overall=rates
    )


def test_rbi_rule_flags_high_obp():
    hi = OutcomeRates(200, 0.18, 0.06, 0.005, 0.05, 0.12, 0.18, 0.40)  # obp high
    lo = OutcomeRates(200, 0.10, 0.02, 0.0, 0.01, 0.04, 0.30, 0.53)  # obp low
    flags = evaluate_lineup([_profile(hi)] * 9)
    assert all(f.flagged for f in flags)
    assert rbi_multiplier(flags[0]) > 1.0
    flags2 = evaluate_lineup([_profile(lo)] * 9)
    assert not any(f.flagged for f in flags2)
    assert rbi_multiplier(flags2[0]) == 1.0


def _reg(xslg: float, zone_contact: float):
    from mlb_engine.features.regression import BatterRegression

    return BatterRegression(
        bbe=40, barrel_rate=0.08, hard_hit=0.40, sweet_spot=0.33, bat_speed=72.0,
        max_ev=108.0, whiff=0.24, zone_contact=zone_contact, xba=0.25, xslg=xslg,
        babip=0.29, woba=0.32, xwoba=0.32,
    )


def test_rbi_ppv_npv_tiers():
    hi = OutcomeRates(200, 0.18, 0.06, 0.005, 0.05, 0.12, 0.18, 0.40)
    profs = [_profile(hi)] * 9
    # PPV: elite xSLG with runners on boosts RBI above the volume-only boost.
    elite = evaluate_lineup(profs, regressions=[_reg(0.520, 0.82)] * 9)
    vol_only = evaluate_lineup(profs)
    assert rbi_multiplier(elite[0]) > rbi_multiplier(vol_only[0])
    # NPV: in-zone contact collapse caps RBI despite big opportunity.
    collapse = evaluate_lineup(profs, regressions=[_reg(0.400, 0.60)] * 9)
    assert rbi_multiplier(collapse[0]) < rbi_multiplier(vol_only[0])


# ---- bullpen vs batter ----
def test_bullpen_profile_excludes_starter_and_uses_late_innings():
    import pandas as pd

    from mlb_engine.features.rolling import build_bullpen_profile

    gd = date(2024, 7, 10)
    rows = [
        # starter (NYY, pitching in Top as home team): 1st + carries into 6th
        (gd, 100, 1, "Top", "single"),
        (gd, 100, 6, "Top", "single"),
        # relievers in late innings -> heavy strikeouts
        (gd, 200, 7, "Top", "strikeout"),
        (gd, 200, 8, "Top", "strikeout"),
        (gd, 201, 9, "Top", "strikeout"),
        # noise: another team's pitcher in NYY's batting half (Bot, home=NYY) - excluded
        (gd, 300, 8, "Bot", "single"),
    ]
    df = pd.DataFrame(
        rows, columns=["game_date", "pitcher", "inning", "inning_topbot", "events"]
    )
    df["batter"] = 1
    df["home_team"] = "NYY"
    df["away_team"] = "BOS"

    pen = build_bullpen_profile(df, "NYY", date(2024, 7, 19), 21, min_inning=6)
    # relievers were all strikeouts -> pen K rate well above league (0.225),
    # and the starter's late single must not leak in.
    assert pen.allowed.p_k > LEAGUE_RATES["K"]


def test_bullpen_npv_walk_trap_and_fatigue():
    import pandas as pd

    from mlb_engine.features.rolling import BullpenProfile

    lg = rates_from_events(pd.Series(dtype=object))
    empty = pd.DataFrame()
    # Walk trap: zone% below .40 -> BB boosted.
    trap = BullpenProfile(lg, empty, zone_pct=0.34, recent_load=1.0)
    assert trap.npv_multipliers().get("BB", 1.0) > 1.0
    # Fatigue: heavy recent workload -> HR/hits boosted.
    tired = BullpenProfile(lg, empty, zone_pct=0.50, recent_load=1.4)
    assert tired.npv_multipliers().get("HR", 1.0) > 1.0
    # Rotowire availability override (rested) suppresses the fatigue penalty.
    assert "HR" not in tired.npv_multipliers(availability=1.0)


# ---- fielding defense ----
def test_defense_hit_multiplier_bounds_and_direction():
    from mlb_engine.filters.defense import defense_hit_multiplier

    # Elite defense (+FRV) suppresses BIP hits (<1); poor defense inflates (>1).
    assert defense_hit_multiplier(20.0) < 1.0
    assert defense_hit_multiplier(-20.0) > 1.0
    assert defense_hit_multiplier(0.0) == 1.0
    # Bounded to +/-6%.
    assert defense_hit_multiplier(9999.0) >= 0.94
    assert defense_hit_multiplier(-9999.0) <= 1.06


def test_team_defense_positional_and_npv():
    from mlb_engine.filters.defense import TeamDefense

    # Team FRV fallback suppresses both singles and XBH.
    good = TeamDefense(frv=40.0).bip_multipliers()
    assert good["1B"] < 1.0 and good["2B"] < 1.0 and good["3B"] < 1.0
    assert set(good) == {"1B", "2B", "3B"}  # never touches K/BB/HR

    # Per-position OAA: elite OF range hits XBH harder than singles.
    of = TeamDefense(outfield_oaa=15.0, infield_oaa=0.0).bip_multipliers()
    assert of["2B"] < of["1B"]

    # NPV: broken middle infield (SS+2B OAA < -8) leaks grounder singles.
    broken = TeamDefense(middle_if_oaa=-15.0).bip_multipliers()
    assert broken["1B"] > 1.0

    # Neutral profile -> no-op multipliers.
    neutral = TeamDefense().bip_multipliers()
    assert neutral["1B"] == 1.0 and neutral["2B"] == 1.0


# ---- human element + umpire ----
def test_human_divisional_and_umpire_zone():
    from mlb_engine.filters.human import HumanFactors
    from mlb_engine.filters.umpire import zone_runs_for_name

    # Divisional familiarity -> fewer Ks, a touch more contact.
    div = HumanFactors(divisional=True).offense_multipliers()
    assert div["K"] < 1.0
    assert div.get("1B", 1.0) > 1.0

    # Tight-zone "Over" ump (Moscoso) -> fewer Ks, more walks.
    over = HumanFactors(umpire_zone_runs=zone_runs_for_name("Edwin Moscoso")).offense_multipliers()
    assert over["K"] < 1.0
    assert over["BB"] > 1.0

    # Wide-zone "Under" ump (Barrett) -> more Ks, fewer walks.
    under = HumanFactors(umpire_zone_runs=zone_runs_for_name("Lance Barrett")).offense_multipliers()
    assert under["K"] > 1.0
    assert under["BB"] < 1.0

    # Unknown umpire -> neutral.
    assert zone_runs_for_name("Nobody McUnknown") is None


# ---- manager tendencies ----
def test_manager_hook_platoon_and_speed():
    from mlb_engine.data.managers import get_manager

    # Quick-hook (Cash, TB=139) caps starter well below the long-leash default;
    # long leash (Francona, CIN=113) extends it.
    assert get_manager(139).starter_bf_cap < 24
    assert get_manager(113).starter_bf_cap > 24
    # Unknown team -> neutral default, no tilts.
    neutral = get_manager(999999)
    assert neutral.starter_bf_cap == 24
    assert neutral.offense_multipliers() == {}
    assert neutral.pen_multipliers() == {}
    # Platoon maximizer (Baldelli, MIN=142) tilts only the bullpen matchup.
    assert get_manager(142).pen_multipliers().get("K", 1.0) < 1.0
    # Speed engine (Vogt, CLE=114) boosts advancement + lowers K.
    speed = get_manager(114).offense_multipliers()
    assert speed.get("2B", 1.0) > 1.0 and speed.get("K", 1.0) < 1.0


# ---- VSIN public splits -> moneyline quotes ----
def test_vsin_fetch_quotes_maps_to_slate():
    import datetime

    from mlb_engine.config import Credentials
    from mlb_engine.data.vsin import VSINClient
    from mlb_engine.schemas import Game, Slate, TeamGameInfo, Venue

    def team(tid, name, ab, home):
        return TeamGameInfo(team_id=tid, name=name, abbrev=ab, is_home=home)

    game = Game(
        game_pk=1, game_date=datetime.date(2026, 7, 20), status="Preview",
        venue=Venue(venue_id=1, name="x"),
        home=team(114, "Cleveland Guardians", "CLE", True),
        away=team(142, "Minnesota Twins", "MIN", False),
    )
    slate = Slate(slate_date=datetime.date(2026, 7, 20), games=[game])

    client = VSINClient(Credentials())

    def fake_fetch_book(src):
        # name-normalization also covers punctuation differences like St. Louis.
        return [
            ("Minnesota Twins", -131.0, 84.0, 62.0),
            ("Cleveland Guardians", 109.0, 16.0, 38.0),
            ("Unlisted Team", -120.0, 50.0, 50.0),  # dropped: not on the slate
        ]

    client._fetch_book = fake_fetch_book  # type: ignore[method-assign]
    q = client.fetch_quotes(slate)
    # two books -> two quotes per matched selection; unlisted team dropped.
    assert set(q) == {
        ("MIN @ CLE", "game_ml", "MIN ML"),
        ("MIN @ CLE", "game_ml", "CLE ML"),
    }
    min_q = q[("MIN @ CLE", "game_ml", "MIN ML")]
    assert len(min_q) == 2  # DK + circa
    assert min_q[0].american == -131.0
    assert min_q[0].handle_pct == 84.0 and min_q[0].bets_pct == 62.0


def test_vsin_parse_helpers():
    from mlb_engine.data.vsin import _american, _norm_name, _pct

    assert _norm_name("St. Louis Cardinals") == _norm_name("ST Louis Cardinals")
    assert _pct("84% \u25b2") == 84.0
    assert _pct("nan") is None
    assert _american("-131") == -131.0
    assert _american("109") == 109.0


# ---- bullpen fatigue feed (StatsAPI parsing) ----
def test_bullpen_fatigue_from_boxscores():
    import datetime

    from mlb_engine.data.mlb_statsapi import MLBStatsClient

    tid = 100
    sched = {"dates": [
        {"date": "2026-07-17", "games": [
            {"gamePk": 1, "status": {"abstractGameState": "Final"}}]},
        {"date": "2026-07-18", "games": [
            {"gamePk": 2, "status": {"abstractGameState": "Final"}}]},
    ]}

    def _box(relievers):
        players = {}
        pitchers = []
        for pid, gs, npitch in relievers:
            pitchers.append(pid)
            players[f"ID{pid}"] = {"stats": {"pitching": {
                "gamesStarted": gs, "numberOfPitches": npitch}}}
        return {"teams": {
            "home": {"team": {"id": tid}, "pitchers": pitchers, "players": players},
            "away": {"team": {"id": 999}, "pitchers": [], "players": {}}}}

    # Day 1: relievers 10,11 throw; Day 2: reliever 10 again (back-to-back) + 12.
    boxes = {
        1: _box([(5, 1, 95), (10, 0, 20), (11, 0, 18)]),  # 5 is the starter
        2: _box([(6, 1, 90), (10, 0, 15), (12, 0, 35)]),
    }

    client = MLBStatsClient()

    def fake_get(path, **params):
        if path == "schedule":
            return sched
        pk = int(path.split("/")[1])
        return boxes[pk]

    client._get = fake_get  # type: ignore[method-assign]
    score = client.bullpen_fatigue(tid, datetime.date(2026, 7, 19))
    # reliever 10 = back-to-back, reliever 12 = 35 pitches yesterday -> 2 gassed.
    assert score == 40.0


# ---- comeback resilience ----
def test_comeback_flag_from_signals():
    from mlb_engine.models.comeback import ComebackSignal, evaluate

    # Strong contact edge + high OBP + long-leash opp starter -> resilient.
    strong = evaluate(ComebackSignal(
        xwoba_diff=0.045, team_obp=0.345, opp_starter_bf_cap=29,
    ))
    assert strong.resilient and strong.score >= 0.60 and strong.reasons

    # Outclassed, low-OBP team facing a quick hook -> not resilient.
    weak = evaluate(ComebackSignal(
        xwoba_diff=-0.050, team_obp=0.290, opp_starter_bf_cap=19,
    ))
    assert not weak.resilient and weak.score < strong.score

    # Bullpen-fatigue hook only fires when supplied.
    base = evaluate(ComebackSignal(xwoba_diff=0.0, team_obp=0.320, opp_starter_bf_cap=24))
    fatigued = evaluate(ComebackSignal(
        xwoba_diff=0.0, team_obp=0.320, opp_starter_bf_cap=24, opp_bullpen_fatigue=90,
    ))
    assert fatigued.score > base.score
    assert any("fatigued" in r for r in fatigued.reasons)


# ---- run-line PPV layer ----
def test_runline_xwoba_confirms_and_contradicts():
    from mlb_engine.market.runline import RunLineSignal, runline_adjustment

    # Home owns a strong xwOBA edge.
    sig = RunLineSignal(xwoba_diff=0.040)
    # Home -1.5 confirmed; away -1.5 (against the edge) flagged; home +1.5 outclass-safe.
    assert runline_adjustment("home", -1.5, sig)[0] == 1
    assert runline_adjustment("away", -1.5, sig)[0] == -1
    # Away is the outclassed dog -> +1.5 risk (downgrade).
    assert runline_adjustment("away", 1.5, sig)[0] == -1

    # Near-even by xwOBA -> both +1.5 dogs gain value.
    even = RunLineSignal(xwoba_diff=0.005)
    assert runline_adjustment("home", 1.5, even)[0] == 1
    assert runline_adjustment("away", 1.5, even)[0] == 1

    # Hooks: cold-but-sound + depleted favorite bullpen stack onto opponent +1.5.
    hook = RunLineSignal(xwoba_diff=None, cold_sound_side="away", fav_pen_depleted_side="home")
    steps, reasons = runline_adjustment("away", 1.5, hook)
    assert steps == 2 and len(reasons) == 2

    # Non-run-line selection -> untouched.
    assert runline_adjustment("home", None, sig) == (0, [])


# ---- distribution tails ----
def test_tail_bonus_and_penalty_symmetric():
    import numpy as np
    import pandas as pd

    from mlb_engine.features.tails import TailAdjuster

    rng = np.random.default_rng(3)
    rows = []
    # A population of average batters (ids 1000..1039) with ~15 BBE each.
    for pid in range(1000, 1040):
        for _ in range(16):
            rows.append(
                {
                    "batter": pid,
                    "pitcher": 1,
                    "launch_speed": float(rng.normal(88, 3)),
                    "launch_speed_angle": 1,
                    "estimated_woba_using_speedangle": float(rng.normal(0.32, 0.02)),
                    "description": "hit_into_play",
                    "events": "single",
                }
            )
    # An elite outlier (id 9001) and a bottom-tail hitter (id 9002).
    for _ in range(16):
        rows.append({"batter": 9001, "pitcher": 1, "launch_speed": 106.0,
                     "launch_speed_angle": 6, "estimated_woba_using_speedangle": 0.55,
                     "description": "hit_into_play", "events": "home_run"})
        rows.append({"batter": 9002, "pitcher": 1, "launch_speed": 72.0,
                     "launch_speed_angle": 1, "estimated_woba_using_speedangle": 0.18,
                     "description": "hit_into_play", "events": "field_out"})
    df = pd.DataFrame(rows)

    tails = TailAdjuster.build(df)
    elite = tails.batter_multiplier(9001)
    poor = tails.batter_multiplier(9002)
    assert elite and elite["HR"] > 1.0  # >2 SD above -> bonus
    assert poor and poor["HR"] < 1.0    # >2 SD below -> penalty
    assert tails.batter_multiplier(1000) == {}  # average -> neutral


# ---- arsenal matching (pitch mix) ----
def test_arsenal_matchup_high_whiff_vs_slider():
    import pandas as pd

    from mlb_engine.features.pitch_mix import (
        arsenal_matchup_multiplier,
        build_arsenal,
        build_batter_pitch_profile,
    )

    # Slider-heavy pitcher with lots of swinging strikes.
    prows = pd.DataFrame(
        {
            "pitch_type": ["SL"] * 60 + ["FF"] * 40,
            "description": (["swinging_strike"] * 30 + ["foul"] * 30) + ["hit_into_play"] * 40,
            "estimated_woba_using_speedangle": [None] * 100,
        }
    )
    arsenal = build_arsenal(prows)
    assert arsenal.usage["BRK"] > arsenal.usage["FB"]
    assert arsenal.swstr["BRK"] > 0.0

    # Batter who whiffs badly on breaking balls.
    brows = pd.DataFrame(
        {
            "pitch_type": ["SL"] * 40,
            "description": ["swinging_strike"] * 28 + ["hit_into_play"] * 12,
            "estimated_woba_using_speedangle": [0.2] * 40,
        }
    )
    bpp = build_batter_pitch_profile(brows)
    mult = arsenal_matchup_multiplier(arsenal, bpp)
    assert mult["K"] > 1.0  # high-whiff hitter vs. whiff-heavy slider -> more Ks

    # Empty inputs -> neutral (no fabricated edge).
    assert arsenal_matchup_multiplier(build_arsenal(pd.DataFrame()), bpp) == {}


# ---- schedule pacing (DGANG) ----
def test_dgang_tax_only_night_then_day():
    from mlb_engine.filters.schedule import dgang_multipliers, is_day, is_night

    # A ~7pm local night game -> next-day ~1pm day game (rest=1) taxes offense.
    prev_night = 19.0
    today_day = 13.0
    assert is_night(prev_night) and is_day(today_day)
    tax = dgang_multipliers(prev_night, today_day, rest_days=1)
    assert tax and tax["1B"] < 1.0 and tax["K"] > 1.0

    # Day-after-day, or two days rest, or missing times -> no tax.
    assert dgang_multipliers(13.0, 13.0, 1) == {}
    assert dgang_multipliers(prev_night, today_day, 2) == {}
    assert dgang_multipliers(None, today_day, 1) == {}


# ---- travel / circadian ----
def test_travel_eastward_worse_than_west_and_boosts_opponent_hr():
    from datetime import timedelta

    from mlb_engine.data.parks import Park
    from mlb_engine.filters.travel_rest import PrevGame, compute

    east_park = Park(1, "East", 40.8, -73.9, 0.0, "open", 100.0)  # NYC-ish
    west_lon = -122.4  # SF-ish
    today = date(2024, 7, 19)
    yday = today - timedelta(days=1)

    # Team flew EAST (from west coast to NYC) with 1 day rest.
    east_trip = compute(PrevGame(yday, 37.8, west_lon), east_park, today)
    # Team flew WEST the same distance/rest.
    west_park = Park(2, "West", 37.8, west_lon, 0.0, "open", 100.0)
    west_trip = compute(PrevGame(yday, 40.8, -73.9), west_park, today)

    assert east_trip.east and not west_trip.east
    assert east_trip.offense_mult < west_trip.offense_mult  # eastward hurts more
    assert east_trip.hr_allowed_mult > 1.0  # tired staff -> opponent HR spike

    # Mid-trip (same city as prior game) -> fully adapted, no penalty.
    mid = compute(PrevGame(yday, 40.8, -73.9), east_park, today)
    assert mid.offense_mult == 1.0
    assert mid.hr_allowed_mult == 1.0


# ---- ev + tiers ----
def test_ev_positive_when_underpriced():
    # model 60%, priced at +100 (fair 50%) -> positive EV
    ev = ev_per_dollar(0.60, 100)
    assert ev > 0.15


def test_classify_tiers():
    thr = EVThresholds(strong_buy=0.08, moderate_buy=0.03)
    q = [MarketQuote("draftkings", 120, handle_pct=70, bets_pct=45)]
    res = evaluate(0.60, q)
    tier, reasons = classify(res, thr)
    assert tier in (Tier.STRONG, Tier.MODERATE)
    # negative edge -> pass
    res2 = evaluate(0.30, q)
    tier2, _ = classify(res2, thr)
    assert tier2 == Tier.PASS


# ---- grading ----
def _rec(**kw) -> Recommendation:
    base = dict(
        game_date=date(2024, 7, 19),
        game_pk=1,
        matchup="AAA @ BBB",
        category="game",
        market="game_ml",
        selection="x",
        model_prob=0.5,
    )
    base.update(kw)
    return Recommendation(**base)


def test_grade_ml_and_total():
    res = GameResult(1, True, 5, 3, 3, 1)
    assert grade(_rec(market="game_ml", team_side="home", side="win"), res) == WIN
    assert grade(_rec(market="game_ml", team_side="away", side="win"), res) == LOSS
    over = _rec(category="game", market="game_total", line=7.5, side="over")
    assert grade(over, res) == WIN  # total 8 > 7.5
    under = _rec(category="game", market="game_total", line=8.5, side="under")
    assert grade(under, res) == WIN  # total 8 < 8.5


def test_grade_batter_prop():
    res = GameResult(
        1, True, 5, 3, 3, 1, players={99: PlayerLine(batting={"H": 2, "HR": 1, "R": 1, "RBI": 2})}
    )
    r = _rec(category="batter", market="batter_h", player_id=99, stat="H", line=1.5, side="over")
    assert grade(r, res) == WIN
    r2 = _rec(category="batter", market="batter_hr", player_id=99, stat="HR", line=1.5, side="over")
    assert grade(r2, res) == LOSS


def test_grade_push():
    res = GameResult(1, True, 4, 4, 2, 2)
    r = _rec(market="game_ml", team_side="home", side="win")
    assert grade(r, res) == PUSH


def test_ip_to_outs():
    assert _ip_to_outs("5.2") == 17
    assert _ip_to_outs("6.0") == 18


# ---- scorecard ----
def test_scorecard_metrics():
    graded = [
        (_rec(tier=Tier.STRONG), WIN),
        (_rec(tier=Tier.STRONG), WIN),
        (_rec(tier=Tier.STRONG), LOSS),
        (_rec(tier=Tier.PASS), LOSS),
        (_rec(tier=Tier.PASS), LOSS),
    ]
    rows = build_scorecard(graded, date(2024, 7, 19))
    strong = next(r for r in rows if r.tier == Tier.STRONG.value)
    assert strong.n == 3
    assert strong.wins == 2
    assert abs(strong.ppv - 2 / 3) < 1e-3


def test_rates_from_events_sums_to_one():
    import pandas as pd

    ev = pd.Series(["single", "home_run", "strikeout", "walk", "field_out"] * 20)
    r = rates_from_events(ev)
    total = sum(r.as_dict().values())
    assert abs(total - 1.0) < 1e-9
    assert 0 < r.obp < 1


def test_np_import_available():
    assert np.array([1, 2, 3]).sum() == 6
