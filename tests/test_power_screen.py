"""The morning power screen: the SIERA gate, the five stages, and the note.

The screen's job is to reproduce a hand process, so the tests pin the parts of it
that a refactor could quietly invert -- the SIERA gate, the run-value sign, the
order of the cuts, the power exception, and which turns a lineup slot actually
gets.
"""

from __future__ import annotations

import math
from datetime import date as Date

import pandas as pd

from mlb_engine.output import power_report
from mlb_engine.output.power_screen import (
    HR_METRIC,
    MIN_BATTER_PA,
    MIN_STARTER_BF,
    MIN_STARTER_PITCHES,
    POWER_XWOBACON,
    SIERA_FLOOR,
    STARTER_SCORED,
    STARTER_SPLITS,
    XFIP_LEAGUE_ANCHOR,
    HitterLine,
    HitterView,
    MatchupSection,
    MetricLine,
    ScreenResult,
    StarterCard,
    StarterSplit,
    apply_cuts,
    arsenal,
    arsenal_fit,
    batter_arsenal,
    batter_window_line,
    bf_pmf,
    contact_line,
    exposure,
    gate_starters,
    league_arms,
    pa_vs_starter,
    pitch_family,
    rank_starters,
    score_starters,
    split_rows,
    starter_damage,
    starter_lines,
    third_look_prob,
    with_tto,
    wrc_plus,
)

OVERALL = STARTER_SPLITS[0]


def _pitch(
    *,
    pitch_type: str = "FF",
    description: str = "hit_into_play",
    events: str | None = "single",
    ev: float = 100.0,
    la: float = 20.0,
    lsa: int | None = 4,
    xba: float = 0.500,
    xwoba: float = 0.600,
    rv: float = 0.45,
    zone: int = 5,
) -> dict[str, object]:
    in_play = description == "hit_into_play"
    return {
        "batter": 1,
        "pitcher": 2,
        "pitch_type": pitch_type,
        "description": description,
        "type": "X" if in_play else "S",
        "events": events,
        "launch_speed": ev if in_play else None,
        "launch_angle": la if in_play else None,
        "launch_speed_angle": lsa if in_play else None,
        "estimated_ba_using_speedangle": xba if in_play else None,
        "estimated_woba_using_speedangle": xwoba if in_play else None,
        "delta_run_exp": rv,
        "zone": zone,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- run value ------------------------------------------------------------


def test_run_value_reads_from_the_hitters_side() -> None:
    """A home run is a gain for the offence, so its run value must be positive.

    ``delta_run_exp`` arrives signed for the batting team; negating it would
    invert every pitch-type conclusion in the note while still looking plausible.
    """
    hr = _frame([_pitch(events="home_run", rv=1.53) for _ in range(4)])
    assert contact_line(hr).rv100 > 0
    strikes = _frame(
        [_pitch(description="called_strike", events=None, rv=-0.07) for _ in range(10)]
    )
    assert contact_line(strikes).rv100 < 0


def test_run_value_is_unavailable_rather_than_zero_without_the_column() -> None:
    df = _frame([_pitch() for _ in range(5)]).drop(columns=["delta_run_exp"])
    assert math.isnan(contact_line(df).rv100)


def test_fouls_are_not_batted_balls_in_a_contact_line() -> None:
    rows = [_pitch(ev=105.0) for _ in range(5)]
    rows += [_pitch(description="foul", events=None, ev=60.0) for _ in range(5)]
    line = contact_line(_frame(rows))
    assert line.pitches == 10
    assert line.bbe == 5
    assert line.hh == 1.0  # the 60 mph fouls would have halved this


# --- stage 0: the work floor and the SIERA gate ---------------------------


def _starter_rows(n: int, *, brl: bool, k: bool = False) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        if k:
            rows.append(_pitch(description="swinging_strike", events="strikeout", lsa=None))
        else:
            rows.append(_pitch(events="double", la=28.0, lsa=6 if brl else 3))
    return _frame(rows)


def _arm(name: str, rows: pd.DataFrame) -> StarterCard:
    """A measured arm that has cleared the work floor, so the SIERA gate is next."""
    card = starter_damage(
        rows, name=name, mlbam_id=len(name), team="AAA", opponent="BBB", throws="R"
    )
    card.work_bf = MIN_STARTER_BF
    card.work_pitches = MIN_STARTER_PITCHES
    return card


def test_the_work_floor_removes_an_arm_before_his_numbers_are_ranked() -> None:
    """A call-up or a rehab start is eleven small samples, not eleven skills."""
    scrub = _arm("Scrub", _starter_rows(130, brl=True))
    green = _arm("Green", _starter_rows(130, brl=True))
    green.name = "Green"
    green.work_bf = MIN_STARTER_BF - 1
    eligible, cuts = gate_starters([green, scrub])
    assert [c.name for c in eligible] == ["Scrub"]
    assert [(c.card.name, c.stage) for c in cuts] == [("Green", "work")]
    assert str(MIN_STARTER_BF) in cuts[0].reason


def test_only_the_arms_above_the_siera_floor_are_eligible() -> None:
    """A soft-contact ace must not reach the ranking at all.

    The contact index measures how loudly an arm is hit, which a good pitcher can
    survive; SIERA measures whether he prevents runs. Ranking on contact alone
    nominated sub-3.4 SIERA arms, so the gate runs before it.
    """
    scrub = _arm("Scrub", _starter_rows(130, brl=True))
    ace = _arm("Ace", _starter_rows(130, brl=False, k=True))
    assert scrub.siera is not None and scrub.siera > SIERA_FLOOR
    assert ace.siera is not None and ace.siera < SIERA_FLOOR

    eligible, cuts = gate_starters([ace, scrub])
    assert [c.name for c in eligible] == ["Scrub"]
    assert [(c.card.name, c.stage) for c in cuts] == [("Ace", "siera")]
    assert f"{SIERA_FLOOR:.2f}" in cuts[0].reason


def test_the_floor_itself_is_not_above_the_floor() -> None:
    """``> 4.4`` is the instruction, so an arm sitting exactly on it is cut."""
    on_it = _arm("On It", _starter_rows(130, brl=True))
    on_it.siera = SIERA_FLOOR
    eligible, cuts = gate_starters([on_it])
    assert eligible == []
    assert cuts[0].stage == "siera"
    above = _arm("Above", _starter_rows(130, brl=True))
    above.siera = SIERA_FLOOR + 0.01
    assert [c.name for c in gate_starters([above])[0]] == ["Above"]


def test_an_arm_without_a_trusted_siera_is_ineligible_not_assumed_soft() -> None:
    thin = _arm("Thin", _starter_rows(30, brl=True))
    assert thin.siera is None and thin.siera_pa == 30
    eligible, cuts = gate_starters([thin])
    assert eligible == []
    assert "no SIERA" in cuts[0].reason


def test_the_siera_gate_switches_off_but_the_work_floor_does_not() -> None:
    ace = _arm("Ace", _starter_rows(130, brl=False, k=True))
    eligible, cuts = gate_starters([ace], siera_floor=0.0)
    assert [c.name for c in eligible] == ["Ace"]
    assert cuts == []
    ace.work_pitches = 0
    assert gate_starters([ace], siera_floor=0.0)[0] == []


# --- stage 1: the eleven metrics, ten rankings ----------------------------


def _line(**values: float) -> MetricLine:
    """A metric line whose samples clear every floor, so only the values rank."""
    samples = {
        f"{m.sample}:{m.days}": m.min_sample * 10 for m in (*STARTER_SCORED, HR_METRIC)
    }
    return MetricLine(dict(values), samples)


def _card(name: str, *, lines: dict[str, MetricLine] | None = None, **values: float) -> StarterCard:
    card = StarterCard(
        name=name,
        mlbam_id=abs(hash(name)) % 9999,
        team="AAA",
        opponent="BBB",
        throws="R",
        bf=500,
        fb_pct=0.30,
        brl_pct=0.07,
        hh_pct=0.40,
        xwobacon=0.350,
        hr_per_bf=0.030,
        k_bb_pct=0.150,
        csw_pct=0.280,
    )
    card.siera = 4.60
    card.lines = lines if lines is not None else {"overall": _line(**values)}
    return card


def test_each_metric_ranks_in_the_direction_that_makes_an_arm_hittable() -> None:
    """High xFIP is bad pitching; high strikeout rate is good, so it must invert.

    A single sign error here would rank the slate's best arm as its most
    vulnerable while every table still looked plausible.
    """
    for attr, hittable, sturdy in (
        ("xfip", 5.40, 3.10),
        ("xera", 5.60, 3.20),
        ("siera", 5.10, 3.30),
        ("fb_pct", 0.44, 0.24),
        ("hh_pct", 0.48, 0.32),
        ("k_pct", 0.14, 0.31),
        ("csw_pct", 0.24, 0.33),
        ("k_bb_pct", 0.05, 0.24),
        ("stuff_plus", 88.0, 118.0),
        ("osw_pct", 0.24, 0.36),
        ("swstr_pct", 0.07, 0.15),
    ):
        soft = _card("Soft", **{attr: hittable})
        tough = _card("Tough", **{attr: sturdy})
        metric = next(m for m in STARTER_SCORED if m.attr == attr)
        split = StarterSplit("overall", "overall", (metric,))
        score_starters([tough, soft], splits=(split,), top_n=1)
        assert soft.points > tough.points, attr
        assert soft.scores["overall"].rank == 1, attr


def test_a_top_three_metric_is_worth_three_points_and_the_rest_one() -> None:
    cards = [_card(f"Arm {i}", xfip=5.5 - i * 0.1) for i in range(5)]
    split = StarterSplit("overall", "overall", (next(
        m for m in STARTER_SCORED if m.attr == "xfip"
    ),))
    score_starters(cards, splits=(split,), top_n=3)
    assert [c.points for c in cards] == [3, 3, 3, 1, 1]
    assert cards[0].scores["overall"].top_in == ("xFIP",)
    assert cards[4].scores["overall"].top_in == ()


def test_a_metric_short_of_sample_scores_nothing_and_says_so() -> None:
    """An unrated metric must cost points, never be filled in at league average."""
    metric = next(m for m in STARTER_SCORED if m.attr == "xfip")
    split = StarterSplit("overall", "overall", (metric,))
    thin = _card("Thin")
    thin.lines = {"overall": MetricLine({"xfip": 6.20}, {f"{metric.sample}:{metric.days}": 1})}
    fat = _card("Fat", xfip=4.10)
    score_starters([thin, fat], splits=(split,), top_n=3)
    assert thin.points == 0
    assert thin.scores["overall"].unrated == ("xFIP",)
    assert fat.points == 3  # the soft arm's 6.20 did not outrank him from nowhere


def test_identical_lines_rank_in_a_stable_order() -> None:
    a = _card("Aaron", xfip=4.50)
    b = _card("Zeke", xfip=4.50)
    split = StarterSplit("overall", "overall", (next(
        m for m in STARTER_SCORED if m.attr == "xfip"
    ),))
    score_starters([b, a], splits=(split,), top_n=1)
    first = (a.scores["overall"].rank, b.scores["overall"].rank)
    score_starters([a, b], splits=(split,), top_n=1)
    assert (a.scores["overall"].rank, b.scores["overall"].rank) == first == (1, 2)


def test_the_final_order_is_the_sum_of_every_ranking() -> None:
    """An arm worst in one split and mid in another must beat one mid in both."""
    xfip = next(m for m in STARTER_SCORED if m.attr == "xfip")
    splits = (
        StarterSplit("overall", "overall", (xfip,)),
        StarterSplit("vs_l", "vs LHH", (xfip,)),
    )
    both = _card("Both", lines={"overall": _line(xfip=5.90), "vs_l": _line(xfip=6.40)})
    lefties_only = _card("Lefties", lines={"overall": _line(xfip=4.20), "vs_l": _line(xfip=6.30)})
    neither = _card("Neither", lines={"overall": _line(xfip=4.10), "vs_l": _line(xfip=4.30)})
    score_starters([neither, lefties_only, both], splits=splits, top_n=1)
    assert both.points == 3 + 3
    assert lefties_only.points == 1 + 1  # second in a split is not a top-three finish
    assert neither.points == 1 + 1
    assert both.scores["vs_l"].rank == 1 and lefties_only.scores["vs_l"].rank == 2
    assert [c.name for c in rank_starters([neither, lefties_only, both], top_n=1)] == ["Both"]


def test_the_time_through_the_order_follows_a_batters_own_meetings() -> None:
    """TTO is a hitter's second look at the starter, not the game's second inning."""
    rows = _frame([
        {"game_date": Date(2026, 8, 1), "home_team": "AAA", "away_team": "BBB",
         "pitcher": 2, "batter": batter, "inning": inning}
        for batter, inning in ((10, 1), (11, 1), (10, 4), (11, 5), (10, 7))
    ])
    out = with_tto(rows)
    assert list(out["tto"]) == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert list(split_rows(out, "tto3")["batter"]) == [10]
    assert len(split_rows(out, "tto2")) == 2


def test_a_split_reads_only_its_own_rows() -> None:
    rows = _frame([
        {"inning": 2, "stand": "L", "tto": 1.0},
        {"inning": 4, "stand": "R", "tto": 2.0},
        {"inning": 7, "stand": "L", "tto": 3.0},
    ])
    assert len(split_rows(rows, "overall")) == 3
    assert list(split_rows(rows, "inn13")["inning"]) == [2]
    assert list(split_rows(rows, "inn15")["inning"]) == [2, 4]
    assert list(split_rows(rows, "vs_l")["inning"]) == [2, 7]
    assert list(split_rows(rows, "hr_r")["inning"]) == [4]
    # a frame without the column is empty, not everything
    assert len(split_rows(rows.drop(columns=["stand"]), "vs_l")) == 0


def test_the_home_run_splits_rank_on_one_rate_and_say_so() -> None:
    split = next(s for s in STARTER_SPLITS if s.key == "hr_l")
    assert [m.label for m in split.metrics] == ["HR/BF"]
    homers = _card("Homers", lines={"overall": _line(), "hr_l": _line(hr_per_bf=0.055)})
    grounders = _card("Worms", lines={"overall": _line(), "hr_l": _line(hr_per_bf=0.012)})
    score_starters([grounders, homers], splits=(split,), top_n=1)
    assert homers.scores["hr_l"].rank == 1
    assert homers.points == 3 and grounders.points == 1


def test_the_league_constant_puts_an_average_arm_on_the_anchor() -> None:
    """xFIP is reconstructed here, so the recentring has to be pinned."""
    rows = []
    for _ in range(40):
        rows.append(_pitch(events="field_out", la=28.0, lsa=3))
    for _ in range(20):
        rows.append(_pitch(description="swinging_strike", events="strikeout", lsa=None))
    for _ in range(8):
        rows.append(_pitch(description="ball", events="walk", lsa=None))
    for _ in range(4):
        rows.append(_pitch(events="home_run", la=28.0, lsa=6))
    frame = _frame(rows)
    frame["bb_type"] = ["fly_ball"] * 40 + [None] * 28 + ["fly_ball"] * 4
    league = league_arms(frame)
    assert league.hr_per_fb == 4 / 44
    # an arm whose line *is* the league's reads the anchor back, which is what
    # makes the reconstructed xFIP comparable to the SIERA in the same table
    xfip = next(m for m in STARTER_SCORED if m.attr == "xfip")
    lines = starter_lines(
        {xfip.days: frame},
        league,
        splits=(StarterSplit("overall", "overall", (xfip,)),),
    )
    assert abs(lines["overall"].values["xfip"] - XFIP_LEAGUE_ANCHOR) < 1e-9


def test_a_thin_league_window_leaves_xfip_unavailable_not_zero() -> None:
    thin = _frame([_pitch(events="single", lsa=3)])
    thin["bb_type"] = ["ground_ball"]
    league = league_arms(thin)
    assert math.isnan(league.hr_per_fb) and math.isnan(league.constant)


# --- stage 2 and 3 --------------------------------------------------------


def test_a_window_line_is_rebuilt_from_the_pitch_rows() -> None:
    rows = [_pitch(events="home_run", ev=105.0, la=28.0, lsa=6) for _ in range(2)]
    rows += [_pitch(events="strikeout", description="swinging_strike", lsa=None) for _ in range(2)]
    line = batter_window_line(_frame(rows))
    assert line["pa"] == 4
    assert line["ba"] == 0.5
    assert line["slg"] == 2.0
    assert line["k"] == 0.5
    assert line["brl"] == 1.0
    assert line["woba"] > 1.0


def test_wrc_plus_is_league_relative() -> None:
    assert wrc_plus(0.315, 0.315) == 100.0
    assert wrc_plus(0.400, 0.315) > 100.0
    assert wrc_plus(0.250, 0.315) < 100.0


def _hitter(name: str, **kw: float) -> HitterLine:
    base = dict(
        pa=100.0,
        wrc=150.0,
        woba=0.380,
        obp=0.380,
        slg=0.520,
        ops=0.900,
        ba=0.290,
        xba=0.290,
        xslg=0.520,
        xwoba_pa=0.370,
        xwoba_con=0.450,
        k=0.220,
        bb=0.090,
        brl=0.150,
        hh=0.500,
        ev90=106.0,
        osw=0.250,
    )
    base.update(kw)
    return HitterLine(
        name=name,
        mlbam_id=abs(hash(name)) % 10**6,
        team="AAA",
        slot=3,
        bats="L",
        versus="Some Arm",
        **{k: float(v) for k, v in base.items()},  # type: ignore[arg-type]
    )


def test_the_cuts_run_in_order_and_say_why_each_hitter_left() -> None:
    strong = _hitter("Strong")
    thin = _hitter("Thin", pa=MIN_BATTER_PA - 1, wrc=300.0)
    weak = _hitter("Weak", wrc=90.0, xwoba_con=0.300)
    lucky = _hitter("Lucky", woba=0.470, xwoba_pa=0.360, xwoba_con=0.400)
    at_league = _hitter("League", wrc=130.0, woba=0.330, xwoba_pa=0.310, xwoba_con=0.330)
    pool = [strong, thin, weak, lucky, at_league]

    kept = apply_cuts(pool, league_xwoba=0.305)

    assert [h.name for h in kept] == ["Strong"]
    assert thin.cut_reason == f"under {MIN_BATTER_PA} PA"
    assert "under 120" in weak.cut_reason
    assert "outruns" in lucky.cut_reason
    assert "at league" in at_league.cut_reason
    assert strong.kept and strong.points > 0


def test_a_power_bat_survives_the_wrc_cut_and_is_flagged() -> None:
    """The Riley case: elite contact, ordinary rate line, kept for home runs only."""
    riley = _hitter("Riley", wrc=104.0, xwoba_pa=0.302, xwoba_con=POWER_XWOBACON + 0.02)
    olson = _hitter("Olson")
    kept = apply_cuts([riley, olson], league_xwoba=0.305)
    assert {h.name for h in kept} == {"Riley", "Olson"}
    assert riley.power_exception
    assert not olson.power_exception


def test_the_power_exception_can_be_switched_off() -> None:
    riley = _hitter("Riley", wrc=104.0, xwoba_pa=0.302, xwoba_con=POWER_XWOBACON + 0.02)
    kept = apply_cuts([riley, _hitter("Olson")], league_xwoba=0.305, keep_power=False)
    assert [h.name for h in kept] == ["Olson"]


# --- stage 4 --------------------------------------------------------------


def test_pitch_families_collapse_savant_codes() -> None:
    assert pitch_family("FF") == pitch_family("FA") == "4-Seam"
    assert pitch_family("ST") == "Sweeper" and pitch_family("SL") == "Slider"
    assert pitch_family(None) is None and pitch_family("ZZ") is None


def test_usage_shares_sum_to_one_and_thin_families_are_dropped() -> None:
    rows = [_pitch(pitch_type="FF") for _ in range(40)]
    rows += [_pitch(pitch_type="CH") for _ in range(30)]
    rows += [_pitch(pitch_type="KN") for _ in range(3)]  # under the floor
    lines, usage = arsenal(_frame(rows))
    assert set(usage) == {"4-Seam", "Changeup"}
    assert usage["4-Seam"] > usage["Changeup"]
    assert abs(sum(usage.values()) - 70 / 73) < 1e-9  # the dropped family still counted as thrown
    assert lines["4-Seam"].pitches == 40


def test_the_fit_weights_the_hitter_by_what_he_will_actually_see() -> None:
    rows = [_pitch(pitch_type="FF", xwoba=0.700, xba=0.500) for _ in range(30)]
    rows += [_pitch(pitch_type="ST", xwoba=0.100, xba=0.150) for _ in range(30)]
    hitter = _frame(rows)
    overall = contact_line(hitter)
    per_pitch = batter_arsenal(hitter, ["4-Seam", "Sweeper"])

    heavy_heat, _b, fallback = arsenal_fit(per_pitch, overall, {"4-Seam": 0.8, "Sweeper": 0.2})
    heavy_sweep, _b2, _f2 = arsenal_fit(per_pitch, overall, {"4-Seam": 0.2, "Sweeper": 0.8})
    assert heavy_heat > overall.xwoba > heavy_sweep
    assert fallback == 0.0


def test_a_family_the_hitter_has_not_seen_falls_back_and_says_so() -> None:
    rows = [_pitch(pitch_type="FF") for _ in range(30)]
    hitter = _frame(rows)
    overall = contact_line(hitter)
    per_pitch = batter_arsenal(hitter, ["4-Seam", "Splitter"])
    fit, _b, fallback = arsenal_fit(per_pitch, overall, {"4-Seam": 0.6, "Splitter": 0.4})
    assert abs(fallback - 0.4) < 1e-9
    assert abs(fit - overall.xwoba) < 1e-9


# --- stage 5 --------------------------------------------------------------


def test_a_short_start_takes_turns_away_from_the_bottom_of_the_order() -> None:
    long_start = bf_pmf(24.0, 2.5, cap=24)
    short_start = bf_pmf(14.8, 7.2, cap=18)
    assert abs(sum(long_start) - 1.0) < 1e-9
    assert sum(long_start[25:]) == 0.0  # the cap binds

    lead_long = pa_vs_starter(1, long_start)
    lead_short = pa_vs_starter(1, short_start)
    eight_long = pa_vs_starter(8, long_start)
    assert lead_long > lead_short
    assert lead_long > eight_long
    assert third_look_prob(1, long_start) > third_look_prob(8, long_start)
    assert third_look_prob(8, short_start) == 0.0


def test_exposure_blends_the_starter_and_the_pen_by_turns() -> None:
    pmf = bf_pmf(24.0, 2.5, cap=24)
    soft_pen = exposure(1, 4.7, pmf, starter_xwoba=0.380, pen_xwoba=0.356)
    hard_pen = exposure(1, 4.7, pmf, starter_xwoba=0.380, pen_xwoba=0.248)
    assert soft_pen.pa_vs_starter + soft_pen.pa_vs_pen == soft_pen.pa_total
    assert 0.0 < soft_pen.share_vs_starter < 1.0
    assert soft_pen.opponent_xwoba > hard_pen.opponent_xwoba
    assert hard_pen.opponent_xwoba < 0.380  # the pen gives the edge back


def test_an_unknown_pen_leaves_the_starters_mark_standing() -> None:
    pmf = bf_pmf(24.0, 2.5, cap=24)
    e = exposure(3, 4.5, pmf, starter_xwoba=0.380, pen_xwoba=None)
    assert abs(e.opponent_xwoba - 0.380) < 1e-9


# --- the note -------------------------------------------------------------


def _split_lines(edge: float) -> dict[str, MetricLine]:
    """A metric line for every ranking, ``edge`` worse than a neutral arm's."""
    values = {
        "xera": 4.00 + edge * 40,
        "xfip": 4.10 + edge * 40,
        "siera": 4.50 + edge * 40,
        "k_pct": 0.22 - edge,
        "csw_pct": 0.28 - edge,
        "k_bb_pct": 0.14 - edge,
        "fb_pct": 0.30 + edge,
        "stuff_plus": 100.0 - edge * 100,
        "osw_pct": 0.31 - edge,
        "hh_pct": 0.38 + edge,
        "swstr_pct": 0.11 - edge,
        "hr_per_bf": 0.030 + edge,
    }
    return {split.key: _line(**values) for split in STARTER_SPLITS}


def _result() -> ScreenResult:
    rows = [_pitch(pitch_type="FF") for _ in range(40)] + [
        _pitch(pitch_type="CH", xwoba=0.480) for _ in range(35)
    ]
    frame = _frame(rows)
    lines, usage = arsenal(frame)
    card = starter_damage(
        _starter_rows(130, brl=True),
        name="Bailey Ober",
        mlbam_id=641927,
        team="MIN",
        opponent="ATL",
        throws="R",
    )
    card.arsenal = lines
    card.usage = usage
    card.work_bf = MIN_STARTER_BF
    card.work_pitches = MIN_STARTER_PITCHES
    others = [
        _card("Arm Two", lines=_split_lines(0.02)),
        _card("Arm Three", lines=_split_lines(0.01)),
        _card("Arm Four", lines=_split_lines(0.00)),
    ]
    card.lines = _split_lines(0.03)
    score_starters([card, *others])
    ranked = rank_starters([card, *others], top_n=4)

    hitter = _hitter("Matt Olson")
    hitter.kept = True
    overall = contact_line(frame)
    per_pitch = batter_arsenal(frame, list(usage))
    fit_w, fit_b, fallback = arsenal_fit(per_pitch, overall, usage)
    view = HitterView(
        line=hitter,
        per_pitch=per_pitch,
        overall=overall,
        fit_xwoba=fit_w,
        fit_xba=fit_b,
        fallback_share=fallback,
        exposure=exposure(3, 4.56, bf_pmf(24.0, 2.5, cap=24), card.xwobacon, 0.303),
    )
    section = MatchupSection(
        starter=card,
        bullpen=None,
        hitters=[view],
        starter_bf=24.0,
        starter_bf_sd=2.5,
        starter_bf_cap=24,
        pitches_per_pa=3.76,
        pitch_cap=92,
        discipline=0.99,
        lineup_projected=True,
    )
    cut = _hitter("Cut Bat", wrc=90.0, xwoba_con=0.300)
    cut.cut_reason = "wRC+ 90 under 120"
    return ScreenResult(
        as_of=Date(2026, 8, 17),
        form_days=42,
        window_start=Date(2026, 7, 6),
        window_end=Date(2026, 8, 16),
        league_woba={"R": 0.313},
        league_xwoba={"R": 0.305},
        starters_ranked=ranked,
        sections=[section],
        cut_log=[cut],
        starter_cuts=gate_starters([_arm("Ace", _starter_rows(130, brl=False, k=True))])[1],
    )


def test_the_note_carries_every_section_and_ends_on_the_recommendations() -> None:
    html = power_report.render_html(_result(), prepared_for="Franz")
    for heading in (
        "Thesis",
        "Data basis",
        "who is eligible to be ranked",
        "the arms, ranked",
        "Ace",
        "Bailey Ober",
        "Matt Olson",
        "Recommendations",
    ):
        assert heading in html
    assert html.index("Recommendations") > html.index("Thesis")
    assert "Franz" in html
    # a rating is not a price, and the note has to keep saying so
    assert "no price" in html
    assert "projections" in html


def test_the_note_prints_every_ranking_and_highlights_the_worst_three() -> None:
    """A points column hides the splits; the note has to show each one's own order.

    The point of ranking a starter nine ways is that the answers differ, so every
    ranking gets its own table and the three arms it says are worst are marked --
    worst three against left-handed hitters, worst three the first time through,
    and so on.
    """
    result = _result()
    html = power_report.render_html(result)
    for split in result.splits:
        assert f"Worst 3 &mdash; {split.label}" in html
    # one highlighted row per arm per ranking, plus the aggregate table's own three
    assert html.count("<tr class='top'>") == 3 * (len(result.splits) + 1)
    worst = result.starters_ranked[0]
    assert worst.scores["vs_l"].rank == 1
    assert "HR/BF" in html  # the home-run splits rank on one rate and say which


def test_an_arm_outside_the_worst_three_of_a_split_is_not_highlighted() -> None:
    result = _result()
    fifth = _card("Arm Five", lines=_split_lines(-0.01))
    score_starters([*result.starters_ranked, fifth])
    result.starters_ranked = rank_starters([*result.starters_ranked, fifth], top_n=5)
    html = power_report.render_html(result)
    assert fifth.scores["overall"].rank == 5
    assert html.count("<tr class='top'>") == 3 * (len(result.splits) + 1)


def test_a_missing_run_value_column_is_disclosed_not_hidden() -> None:
    result = _result()
    result.has_run_value = False
    assert "delta_run_exp" in power_report.render_html(result)
    result.has_run_value = True
    assert "delta_run_exp" not in power_report.render_html(result)


def test_the_cut_appendix_prints_the_near_misses_only() -> None:
    result = _result()
    noise = _hitter("Two PA Wonder", pa=2.0, wrc=400.0)
    noise.cut_reason = "under 60 PA"
    result.cut_log.append(noise)
    html = power_report.render_html(result)
    assert "Cut Bat" in html
    assert "Two PA Wonder" not in html


def test_the_filename_is_dated() -> None:
    assert power_report.default_filename(Date(2026, 8, 17)) == "power_screen_2026-08-17.pdf"
    assert power_report.default_filename(Date(2026, 8, 17), "html").endswith(".html")


def test_a_power_exception_is_disclosed_in_the_recommendation() -> None:
    """A bat kept on contact alone must not read as a hits or H+R+RBI play."""
    result = _result()
    result.sections[0].hitters[0].line.power_exception = True
    html = power_report.render_html(result)
    assert "home runs and total bases only" in html
