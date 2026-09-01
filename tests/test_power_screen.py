"""The morning power screen: the SIERA gate, the five stages, and the note.

The screen's job is to reproduce a hand process, so the tests pin the parts of it
that a refactor could quietly invert -- the SIERA gate, the run-value sign, the
order of the cuts, the power exception, and which turns a lineup slot actually
gets.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date as Date

import pandas as pd
import pytest

from mlb_engine.features import arm as arm_model
from mlb_engine.features.swing import LEAGUE, WINDOW, SwingProfile
from mlb_engine.output import power_report
from mlb_engine.output.power_screen import (
    BAT_SPEED_BAND,
    FIT_SCORED,
    HALF_FLOOR,
    HALF_SCORED,
    HR_METRIC,
    MIN_BATTER_PA,
    MIN_STARTER_BF,
    MIN_STARTER_PITCHES,
    POWER_XWOBACON,
    RESCUE_POWER_Z,
    SIERA_FLOOR,
    STARTER_SCORED,
    STARTER_SPLITS,
    TOP_PITCHES,
    XFIP_LEAGUE_ANCHOR,
    ArsenalEdge,
    ContactLine,
    ContextTerms,
    FinalScore,
    HalfLine,
    HitterLine,
    HitterView,
    MatchupSection,
    MetricLine,
    PoolBatter,
    ScreenResult,
    StarterCard,
    StarterSplit,
    TrendDeltas,
    apply_cuts,
    arsenal,
    arsenal_edge,
    arsenal_fit,
    batter_arsenal,
    batter_window_line,
    bf_pmf,
    build_context,
    contact_line,
    contact_mark,
    exposure,
    gate_starters,
    half_lines,
    hitter_pool,
    keep_arms,
    league_arms,
    pa_vs_starter,
    pitch_family,
    rank_final,
    rank_starters,
    rv_term,
    score_edges,
    score_halves,
    score_starters,
    signed_term,
    split_rows,
    starter_damage,
    starter_lines,
    third_look_prob,
    top_pitch_rv,
    trend_deltas,
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
    cards = [_card(f"Arm {i}", xfip=5.5 - i * 0.1) for i in range(6)]
    split = StarterSplit("overall", "overall", (next(
        m for m in STARTER_SCORED if m.attr == "xfip"
    ),))
    score_starters(cards, splits=(split,), top_n=3)
    assert [c.points for c in cards] == [3, 3, 3, 1, 1, 1]
    assert cards[0].scores["overall"].top_in == ("xFIP",)
    assert cards[5].scores["overall"].top_in == ()


def test_the_bonus_never_goes_to_more_than_half_the_field() -> None:
    """Top three of five is a top three of nothing: five arms, two bonuses."""
    cards = [_card(f"Arm {i}", xfip=5.5 - i * 0.1) for i in range(5)]
    split = StarterSplit("overall", "overall", (next(
        m for m in STARTER_SCORED if m.attr == "xfip"
    ),))
    score_starters(cards, splits=(split,), top_n=3)
    assert [c.points for c in cards] == [3, 3, 1, 1, 1]


def test_the_flat_cutoff_comes_back_when_the_cap_is_switched_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLBE_POWER_BONUS_HALF", "0")
    cards = [_card(f"Arm {i}", xfip=5.5 - i * 0.1) for i in range(5)]
    split = StarterSplit("overall", "overall", (next(
        m for m in STARTER_SCORED if m.attr == "xfip"
    ),))
    score_starters(cards, splits=(split,), top_n=3)
    assert [c.points for c in cards] == [3, 3, 3, 1, 1]


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
    # The soft arm's 6.20 did not outrank him from nowhere: he is the only rated
    # arm, which is a point for the metric and no top-half finish in a field of one.
    assert fat.points == 1


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


# --- which arms are screened ----------------------------------------------


def _ranked(*points: int) -> list[StarterCard]:
    """A ranked pool carrying nothing but its stage-1 points, worst first."""
    cards = []
    for i, pts in enumerate(points):
        card = _card(f"Arm {i}")
        card.points = pts
        cards.append(card)
    return cards


def test_the_headcount_alone_screens_an_arm_the_ranking_already_disowned() -> None:
    """8/30: the fourth arm held 46 points against 94 and kept his lineup in."""
    assert [c.points for c in keep_arms(_ranked(94, 93, 83, 46), keep=4)] == [94, 93, 83, 46]


def test_the_gap_keeps_the_close_arms_and_drops_the_one_off_the_pack() -> None:
    kept = keep_arms(_ranked(94, 93, 83, 46), keep=4, gap=25)
    assert [c.points for c in kept] == [94, 93, 83]


def test_an_arm_exactly_on_the_bar_is_still_hunted() -> None:
    assert [c.points for c in keep_arms(_ranked(94, 69), keep=4, gap=25)] == [94, 69]


def test_the_gap_thins_a_screen_and_never_empties_one() -> None:
    """Every arm behind a runaway leader is still one lineup worth screening."""
    assert [c.points for c in keep_arms(_ranked(94, 10, 4), keep=4, gap=25)] == [94]
    assert keep_arms([], keep=4, gap=25) == []


def test_the_headcount_still_binds_under_the_gap() -> None:
    kept = keep_arms(_ranked(94, 93, 92, 91, 90), keep=3, gap=25)
    assert [c.points for c in kept] == [94, 93, 92]


def test_a_pool_shorter_than_the_headcount_is_kept_whole() -> None:
    assert [c.points for c in keep_arms(_ranked(94, 90), keep=4, gap=25)] == [94, 90]


def test_tied_arms_keep_the_rankings_order() -> None:
    kept = keep_arms(_ranked(94, 94, 94), keep=4, gap=25)
    assert [c.name for c in kept] == ["Arm 0", "Arm 1", "Arm 2"]


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


def test_the_cuts_can_be_run_under_a_scoring_rule_the_screen_no_longer_uses() -> None:
    """The replay's seam: whoever scores, the cuts read that scorer's own top-K."""

    def score_nobody(pool: list[HitterLine]) -> None:
        for h in pool:
            h.points = 0
            h.top_in = ()

    assert apply_cuts([_hitter("Strong")], league_xwoba=0.305, scorer=score_nobody) == []


def test_a_pool_is_built_from_the_rows_a_hitter_has_against_that_hand() -> None:
    rows = pd.DataFrame(
        [
            {**_pitch(events="single"), "batter": 10, "p_throws": "R"},
            {**_pitch(events="home_run"), "batter": 10, "p_throws": "R"},
            {
                **_pitch(events="strikeout", description="swinging_strike"),
                "batter": 11,
                "p_throws": "L",
            },
        ]
    )
    pool = hitter_pool(
        rows,
        [
            PoolBatter(mlbam_id=10, name="Faces RHP", slot=2, bats="L"),
            PoolBatter(mlbam_id=11, name="Faces LHP Only", slot=3, bats="R"),
            PoolBatter(mlbam_id=12, name="No Rows", slot=4, bats="R"),
        ],
        hand="R",
        team="AAA",
        versus="Some Arm",
        league_woba=0.315,
    )

    assert [h.name for h in pool] == ["Faces RHP"]
    assert pool[0].pa == 2 and pool[0].slot == 2 and pool[0].versus == "Some Arm"


def test_a_power_bat_survives_the_wrc_cut_and_is_flagged() -> None:
    """The Riley case: elite contact, ordinary rate line, kept for home runs only."""
    riley = _hitter("Riley", wrc=104.0, xwoba_pa=0.302, xwoba_con=POWER_XWOBACON + 0.02)
    olson = _hitter("Olson")
    kept = apply_cuts([riley, olson], league_xwoba=0.305)
    assert {h.name for h in kept} == {"Riley", "Olson"}
    assert riley.power_exception
    assert not olson.power_exception


def _swing(power: float) -> SwingProfile:
    """A readable swing whose bat speed and blast rate sit ``power`` SD from league."""
    bmu, bsd = LEAGUE["bat_speed"]
    zmu, zsd = LEAGUE["blast"]
    return SwingProfile(swings=400, bat_speed=bmu + power * bsd, blast=zmu + power * zsd)


def test_the_swing_keeps_a_hitter_the_luck_gap_wants_cut() -> None:
    """Stage two: the gap says the results outran the contact, the swing disagrees.

    Of the 471 windows this cut removes out of time, the better-swinging half went
    on to .3801 TB/PA against .3355 for the worse half -- so a flagged hitter whose
    power swing is above league is kept, and flagged as kept on the swing rather
    than on the rate line.
    """
    lucky = _hitter("Lucky", woba=0.470, xwoba_pa=0.360, xwoba_con=0.400)
    lucky.swing = _swing(1.0)
    kept = apply_cuts([lucky], league_xwoba=0.305)
    assert [h.name for h in kept] == ["Lucky"]
    assert lucky.swing_rescue and lucky.kept and not lucky.cut_reason
    assert not lucky.power_exception  # a different rescue, kept distinguishable


def test_a_barely_above_league_swing_does_not_clear_the_fitted_bar() -> None:
    """The rescue bar is fitted, not league average.

    Scanning the threshold, relief peaks at +0.375 in both window sizes and that
    is the lowest value whose rescued rows beat the kept ones in both seasons; at
    league average the 2026 rescues went .3619 TB/PA against .3674 for the hitters
    kept. So a swing this side of the bar leaves the cut standing.
    """
    assert RESCUE_POWER_Z > 0.0
    marginal = _hitter("Marginal", woba=0.470, xwoba_pa=0.360, xwoba_con=0.400)
    marginal.swing = _swing(RESCUE_POWER_Z - 0.1)
    assert apply_cuts([marginal], league_xwoba=0.305) == []
    assert "outruns" in marginal.cut_reason and not marginal.swing_rescue


def test_a_below_league_swing_confirms_the_cut() -> None:
    lucky = _hitter("Lucky", woba=0.470, xwoba_pa=0.360, xwoba_con=0.400)
    lucky.swing = _swing(-1.0)
    assert apply_cuts([lucky], league_xwoba=0.305) == []
    assert "outruns" in lucky.cut_reason and not lucky.swing_rescue


def test_an_unreadable_swing_leaves_the_cut_standing() -> None:
    """The default is stage one. Too few tracked swings must not become a rescue."""
    lucky = _hitter("Lucky", woba=0.470, xwoba_pa=0.360, xwoba_con=0.400)
    lucky.swing = SwingProfile(swings=8, bat_speed=80.0)  # blast rate unreadable
    assert apply_cuts([lucky], league_xwoba=0.305) == []
    assert "outruns" in lucky.cut_reason


def test_the_swing_does_not_rescue_a_hitter_the_earlier_cuts_removed() -> None:
    """The rescue answers the luck gap only; the sample and league floors stand."""
    thin = _hitter("Thin", pa=MIN_BATTER_PA - 1, woba=0.470, xwoba_pa=0.360)
    thin.swing = _swing(2.0)
    at_league = _hitter("League", woba=0.390, xwoba_pa=0.310, xwoba_con=0.330)
    at_league.swing = _swing(2.0)
    assert apply_cuts([thin, at_league], league_xwoba=0.305) == []
    assert thin.cut_reason and at_league.cut_reason
    assert not (thin.swing_rescue or at_league.swing_rescue)


def test_a_steep_attack_angle_does_not_rescue_on_its_own() -> None:
    """Attack angle sorts the home-run line and not the rows this cut removes.

    Inside the flagged group its coefficient on the next fortnight's total bases
    is t -0.07 (steeper half .3684, flatter .3880), so it is printed for the
    market it points at and left out of the rescue.
    """
    mu, sd = LEAGUE["attack_angle"]
    lucky = _hitter("Steep", woba=0.470, xwoba_pa=0.360, xwoba_con=0.400)
    lucky.swing = SwingProfile(swings=400, attack_angle=mu + 3 * sd)
    assert apply_cuts([lucky], league_xwoba=0.305) == []
    assert "outruns" in lucky.cut_reason and not lucky.swing_rescue


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


def test_a_family_read_off_two_batted_balls_falls_back_to_the_overall_line() -> None:
    """The per-family floor counts pitches; xwOBA is measured on balls in play.

    Raleigh's curveball on 8/30 cleared the 25-pitch floor with two batted balls,
    one of them a home run, and its 1.342 xwOBA carried his fit 115 points past
    his own overall mark.
    """
    rows = [_pitch(pitch_type="FF", xwoba=0.400, xba=0.300) for _ in range(30)]
    rows += [_pitch(pitch_type="CU", xwoba=1.342, xba=0.900) for _ in range(2)]
    rows += [
        _pitch(pitch_type="CU", description="called_strike", events=None) for _ in range(25)
    ]
    hitter = _frame(rows)
    overall = contact_line(hitter)
    per_pitch = batter_arsenal(hitter, ["4-Seam", "Curveball"])
    assert per_pitch["Curveball"].pitches == 27 and per_pitch["Curveball"].bbe == 2

    fit, _b, fallback = arsenal_fit(per_pitch, overall, {"4-Seam": 0.6, "Curveball": 0.4})
    assert abs(fallback - 0.4) < 1e-9
    assert fit < per_pitch["Curveball"].xwoba
    assert abs(fit - (0.6 * 0.400 + 0.4 * overall.xwoba)) < 1e-9


def test_run_value_survives_the_batted_ball_floor_because_it_is_per_pitch() -> None:
    thin = contact_line(
        _frame(
            [_pitch(pitch_type="CU", xwoba=1.342) for _ in range(2)]
            + [_pitch(pitch_type="CU", description="swinging_strike", events=None)]
        )
    )
    assert math.isnan(contact_mark(thin, "xwoba")) and math.isnan(contact_mark(thin, "brl"))
    assert not math.isnan(contact_mark(thin, "rv100"))
    assert not math.isnan(contact_mark(thin, "whiff"))


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


def test_the_lead_position_is_the_worst_arm_a_hitter_survived_against() -> None:
    """The softest arm on the board is not a position if nobody hunts him.

    On 8/30 the note opened on Robbie Ray's four-seam as the pitch the surviving
    bats were being asked to hunt, and neither survivor was facing him.
    """
    result = _result()
    live = result.sections[0]
    empty = replace(live, starter=_card("Robbie Ray", lines=_split_lines(0.04)), hitters=[])
    result.sections = [empty, live]
    html = power_report.render_html(result)
    lead = html[html.index("The lead position is"):]
    assert live.starter.name in lead[:200]
    assert "Robbie Ray" not in lead[:200]
    assert "the most exposed arm the screen kept a hitter against" in html


def test_a_slate_with_no_survivor_anywhere_claims_no_lead_position() -> None:
    result = _result()
    result.sections = [replace(result.sections[0], hitters=[])]
    html = power_report.render_html(result)
    assert "The lead position is" not in html
    assert "there is no position today" in html


def test_the_prose_does_not_name_a_two_batted_ball_family_as_his_damage() -> None:
    result = _result()
    section = result.sections[0]
    view = section.hitters[0]
    view.per_pitch["Curveball"] = contact_line(
        _frame([_pitch(pitch_type="CU", xwoba=1.342) for _ in range(2)])
    )
    prose = power_report._hitter_prose(view, section)
    assert "curveball" not in prose


def test_a_rate_at_or_above_one_keeps_its_leading_digit() -> None:
    """.1342 read as .134; the xwOBA it stood for was 1.342."""
    assert power_report._f3(0.447) == ".447"
    assert power_report._f3(1.342) == "1.342"
    assert power_report._f3(-0.031) == "-.031"
    assert power_report._f3(math.nan) == "&mdash;"


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


def test_a_swing_rescue_is_disclosed_as_a_cut_being_overruled() -> None:
    """A hitter here on his swing must not read as a clean survivor of the cuts."""
    result = _result()
    line = result.sections[0].hitters[0].line
    line.swing_rescue = True
    line.swing = _swing(1.0)
    html = power_report.render_html(result)
    assert "Kept on the swing after the luck gap flagged them" in html
    assert "\u2021" in html  # the row is marked as well as footnoted
    assert "does not rescue" in html  # squared-up rate is reported, not priced
    assert "Attack angle" in html
    assert f"{WINDOW['blast']} for blast" in html  # its own window, not a round six weeks


def test_the_swing_columns_are_sourced_even_when_no_cut_was_overruled() -> None:
    """A bat-speed figure without its window is unreadable, rescue or no rescue."""
    result = _result()
    result.sections[0].hitters[0].line.swing = _swing(0.0)
    html = power_report.render_html(result)
    assert "Kept on the swing" not in html  # nothing was overruled
    assert "The swing columns" in html
    assert "too few tracked swings to read, not an average one" in html


def test_the_swing_note_is_absent_when_no_swing_was_read_at_all() -> None:
    assert "The swing columns" not in power_report.render_html(_result())


# --- the delivery behind the ranking -------------------------------------


def _arm_rows(n: int, *, velo: float, throws: str = "R") -> pd.DataFrame:
    """Tracked fastballs, on the columns the ingestion now keeps."""
    return pd.DataFrame(
        {
            "pitcher": [2] * n,
            "pitch_type": ["FF"] * n,
            "p_throws": [throws] * n,
            "game_date": [Date(2026, 8, 1)] * n,
            "release_speed": [velo] * n,
            "release_extension": [6.6] * n,
            "release_pos_x": [-1.9] * n,
            "release_pos_z": [5.9] * n,
            "release_spin_rate": [2300.0] * n,
            "pfx_z": [1.25] * n,
            "pfx_x": [-0.85] * n,
        }
    )


def _armed_card(velo: float) -> StarterCard:
    rows = pd.concat([_starter_rows(130, brl=True), _arm_rows(60, velo=velo)], ignore_index=True)
    return starter_damage(
        rows, name="Bailey Ober", mlbam_id=641927, team="MIN", opponent="ATL", throws="R"
    )


def test_the_screen_reads_the_arm_off_the_same_slice_it_ranks_on() -> None:
    card = _armed_card(97.0)
    assert card.arm is not None
    assert math.isclose(card.arm.velo, 97.0)


def test_a_slice_with_no_tracked_delivery_leaves_the_arm_unmeasured() -> None:
    """The screen's own fixtures carry no release columns, and must not raise."""
    card = starter_damage(
        _starter_rows(130, brl=True),
        name="Thin",
        mlbam_id=1,
        team="A",
        opponent="B",
        throws="R",
    )
    assert card.arm is not None
    assert all(v != v for v in card.arm.levels().values())  # counted, never guessed
    assert card.arm_verdict == arm_model.UNMEASURED


def test_the_arm_does_not_reorder_or_gate_the_screen() -> None:
    """Out of time a good arm is better everywhere, not a rescue -- so it cannot cut.

    A hard-throwing starter whose batted balls got hit still ranks on the damage:
    the delivery is disclosure beside the index, never a filter on it.
    """
    soft, hard = _armed_card(89.0), _armed_card(99.0)
    hard.name, hard.mlbam_id = "Hard", 2
    ranked = rank_starters([soft, hard], top_n=2)
    assert [c.mlbam_id for c in ranked] == [641927, 2]
    assert math.isclose(soft.index, hard.index)
    assert hard.arm_verdict == arm_model.CONTRADICTED  # disclosed, still ranked


def test_the_delivery_reaches_the_note_with_its_window() -> None:
    result = _result()
    result.starters_ranked[0].arm = result.sections[0].starter.arm = _armed_card(97.0).arm
    html = power_report.render_html(result)
    assert "mph perceived" in html
    assert "pVelo" in html and "IVB" in html  # the starter table columns
    assert f"last {arm_model.WINDOW} four-seams" in html
    assert str(arm_model.MIN_LEVEL_PITCHES) in html
    assert "Nothing here gates" in html


def test_an_unreadable_delivery_is_stated_rather_than_left_blank() -> None:
    html = power_report.render_html(_result())  # fixtures carry no release columns
    assert "unreadable at this sample" in html
    assert "\u2020" not in html  # nothing to disagree with


def test_a_contradicted_delivery_is_marked_in_the_row_and_the_prose() -> None:
    result = _result()
    result.starters_ranked[0].arm = result.sections[0].starter.arm = _armed_card(99.0).arm
    html = power_report.render_html(result)
    assert "The delivery disagrees" in html
    assert "\u2020" in html
    assert "less certain than the index reads" in html


def _moving_arm(d_pvelo: float) -> arm_model.ArmProfile:
    prof = _armed_card(93.0).arm
    assert prof is not None
    return replace(prof, d_pvelo=d_pvelo)


def test_a_shedding_delivery_sharpens_the_selection_in_the_prose() -> None:
    """The screen only ever selects the fade side, which is where the trend graded."""
    result = _result()
    moving = _moving_arm(-0.9)
    result.starters_ranked[0].arm = result.sections[0].starter.arm = moving
    html = power_report.render_html(result)
    assert "He is also shedding it: 0.9 mph" in html
    assert "worth another .026 of wOBA inside a fade" in html


def test_a_held_delivery_reads_as_the_softer_half_of_the_same_fade() -> None:
    result = _result()
    result.starters_ranked[0].arm = result.sections[0].starter.arm = _moving_arm(+0.6)
    html = power_report.render_html(result)
    assert "He is holding the delivery, +0.6 mph" in html


def test_an_unread_or_flat_trend_is_left_out_of_the_starter_paragraph() -> None:
    for d_pvelo in (float("nan"), -0.01):
        result = _result()
        result.starters_ranked[0].arm = result.sections[0].starter.arm = _moving_arm(d_pvelo)
        html = power_report.render_html(result)
        assert "block before this one" not in html
        assert "mph perceived" in html  # the level still prints


def test_a_power_exception_is_disclosed_in_the_recommendation() -> None:
    """A bat kept on contact alone must not read as a hits or H+R+RBI play."""
    result = _result()
    result.sections[0].hitters[0].line.power_exception = True
    html = power_report.render_html(result)
    assert "home runs and total bases only" in html


# --- stage 6: both halves of the game ------------------------------------


def _half_pitch(inning: int, **kw: object) -> dict[str, object]:
    """One pitch in a known inning, with the columns the half score reads."""
    row = _pitch(**kw)  # type: ignore[arg-type]
    row["inning"] = inning
    return row


def test_the_halves_split_on_the_seventh_inning() -> None:
    """Innings 1-6 are the starter's half; the seventh is the bullpen's."""
    rows = [_half_pitch(i) for i in (1, 3, 6, 6)] + [_half_pitch(i) for i in (7, 8, 9)]
    early, late = half_lines(_frame(rows))
    assert (early.half, late.half) == ("early", "late")
    assert early.samples["pitches"] == 4
    assert late.samples["pitches"] == 3


def test_a_thin_half_is_shrunk_toward_the_hitter_himself_not_the_league() -> None:
    """The null for a missing late sample is what he does earlier in the game.

    A hitter with 200 early plate appearances at a 10% strikeout rate and four
    late ones at 100% is not a 100% strikeout hitter after the sixth, and the
    league's 22% is not the right prior either -- his own 10% is.
    """
    early_rows = [
        _half_pitch(2, description="called_strike",
                    events="strikeout" if i % 10 == 0 else "field_out")
        for i in range(200)
    ]
    late_rows = [
        _half_pitch(8, description="swinging_strike", events="strikeout") for _ in range(4)
    ]
    early, late = half_lines(_frame(early_rows + late_rows))
    assert late.values["k"] < 0.5  # raw split is 1.00
    assert late.values["k"] > early.values["k"]  # but still worse than his own half
    assert late.samples["pa"] == 4


def test_a_half_with_no_rows_at_all_is_unavailable_rather_than_zero() -> None:
    early, late = half_lines(_frame([_half_pitch(i) for i in (1, 2, 3)]))
    assert late.pa == 0
    assert all(math.isnan(v) for v in late.values.values())
    assert late.points == 0


def _half(**values: float) -> HalfLine:
    line = HalfLine(half="late")
    for metric in HALF_SCORED:
        line.values[metric.attr] = values.get(metric.attr, math.nan)
    return line


def test_a_half_carries_the_floor_and_earns_only_the_top_three_bonus() -> None:
    best = _half(k=0.10)
    rest = [_half(k=0.20 + i / 100) for i in range(5)]
    score_halves([best, *rest])
    assert best.points == HALF_FLOOR + 2  # the floor, plus the best K% of six
    assert best.earned == 2
    assert rest[0].points == HALF_FLOOR + 2
    assert rest[3].points == HALF_FLOOR  # rated, outside the top three
    assert rest[3].earned == 0
    assert best.top_in == ("K%",)


def test_a_metric_that_could_not_be_read_costs_the_hitter_nothing() -> None:
    """A hitter with no late batted balls has no late exit velocity, not a bad one.

    The floor is the same for both, so the unmeasured hitter is level with the
    measured one that finished outside the top three rather than behind him.
    """
    measured = _half(k=0.20, ev90=108.0)
    unmeasured = _half(k=0.20)
    beaten = _half(k=0.20, ev90=88.0)
    score_halves([measured, unmeasured, beaten], top_n=1)
    assert measured.points > unmeasured.points
    assert unmeasured.points == beaten.points == HALF_FLOOR
    assert "EV90" in measured.top_in


def test_two_hitters_do_not_both_finish_in_the_top_three() -> None:
    """8/30 kept two bats and scored both 27 of 27, which separated neither."""
    better = _half(k=0.12, ev90=108.0)
    worse = _half(k=0.34, ev90=88.0)
    score_halves([better, worse])
    assert better.earned == 4  # two metrics, one bonus apiece
    assert worse.earned == 0
    assert worse.points == HALF_FLOOR


def test_the_strikeout_rate_is_scored_low_is_better() -> None:
    quiet = _half(k=0.12)
    loud = _half(k=0.34)
    score_halves([quiet, loud], top_n=1)
    assert quiet.points > loud.points


# --- stage 7: regression, park and weather -------------------------------


def test_a_term_is_zero_inside_its_noise_band() -> None:
    assert signed_term(0.4, BAT_SPEED_BAND, higher_helps_hitter=True) == 0
    assert signed_term(1.4, BAT_SPEED_BAND, higher_helps_hitter=True) == 1
    assert signed_term(-1.4, BAT_SPEED_BAND, higher_helps_hitter=True) == -1


def test_a_missing_reading_is_neutral_rather_than_negative() -> None:
    """A hitter whose bat speed was never tracked has not slowed down."""
    assert signed_term(math.nan, BAT_SPEED_BAND, higher_helps_hitter=True) == 0
    terms = build_context(woba=math.nan, xwoba=math.nan, trends=TrendDeltas())
    assert terms.total == 0


def test_the_luck_term_pays_the_hitter_whose_contact_beats_his_results() -> None:
    """xwOBA above wOBA is the only thing 'positive regression' can mean.

    The same gap in the other direction is what the stage-3 luck cut removes, so
    the sign here has to agree with it: results ahead of contact quality are not
    evidence.
    """
    unlucky = build_context(woba=0.300, xwoba=0.340, trends=TrendDeltas())
    lucky = build_context(woba=0.380, xwoba=0.330, trends=TrendDeltas())
    assert unlucky.luck == 1
    assert lucky.luck == -1
    assert build_context(woba=0.330, xwoba=0.340, trends=TrendDeltas()).luck == 0


def test_chasing_more_is_a_point_against_the_hitter() -> None:
    """Bat speed and exit velocity go up for the hitter; chase goes up against him."""
    chasing = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(chase=0.06, oz_pitches=100)
    )
    disciplined = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(chase=-0.06, oz_pitches=100)
    )
    assert chasing.chase == -1
    assert disciplined.chase == 1


def test_the_park_and_the_forecast_are_signed_toward_the_hitter() -> None:
    hot = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(),
        park_factor=108.0, weather_hr_mult=1.08,
    )
    cold = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(),
        park_factor=92.0, weather_hr_mult=0.92,
    )
    neutral = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(),
        park_factor=100.5, weather_hr_mult=1.005,
    )
    assert (hot.park, hot.weather) == (1, 1)
    assert (cold.park, cold.weather) == (-1, -1)
    assert (neutral.park, neutral.weather) == (0, 0)


def test_facing_one_of_the_slates_worst_arms_is_worth_a_point() -> None:
    terms = build_context(woba=0.330, xwoba=0.330, trends=TrendDeltas(), worst_arm=True)
    assert terms.worst_arm == 1
    assert terms.regression == 0  # the arm is context, not form
    assert terms.total == 1


def test_the_regression_property_is_the_four_form_terms_only() -> None:
    terms = build_context(
        woba=0.300, xwoba=0.340,
        trends=TrendDeltas(bat_speed=1.5, chase=-0.06, ev90=2.0,
                           swings=100, oz_pitches=100, bbe=50),
        park_factor=108.0, weather_hr_mult=1.08, worst_arm=True,
    )
    assert terms.regression == 4
    assert terms.total == 7


def test_a_trend_needs_its_own_denominator_before_it_is_a_reading() -> None:
    """Three weeks of a metric is a reading only if the sample got there."""
    def rows(n: int, *, bat: float, ev: float) -> pd.DataFrame:
        out = []
        for _ in range(n):
            row = _pitch(ev=ev)
            row["bat_speed"] = bat
            out.append(row)
        return _frame(out)

    season = rows(400, bat=72.0, ev=100.0)
    thin = trend_deltas(rows(5, bat=68.0, ev=90.0), season)
    assert math.isnan(thin.bat_speed)
    assert math.isnan(thin.ev90)
    assert thin.swings == 5

    read = trend_deltas(rows(80, bat=68.0, ev=90.0), season)
    assert read.bat_speed < -3
    assert read.ev90 < -9
    assert build_context(woba=0.330, xwoba=0.330, trends=read).regression == -2


# --- stage 8: the arsenal edge -------------------------------------------


def _mix(usage: dict[str, float], **marks: float) -> dict[str, ContactLine]:
    return {
        name: ContactLine(
            pitches=200, bbe=60, rv100=marks.get(f"{name}_rv", 1.0), xba=0.260,
            xwoba=marks.get(f"{name}_xwoba", 0.320), hh=0.400, brl=0.080,
            whiff=marks.get(f"{name}_whiff", 0.250),
        )
        for name in usage
    }


def test_the_fit_is_weighted_by_the_mix_the_starter_actually_throws() -> None:
    """A pitch he crushes counts as much as the starter throws it, and no more."""
    usage = {"4-Seam": 0.90, "Curveball": 0.10}
    hitter = _mix(usage, **{"4-Seam_xwoba": 0.200, "Curveball_xwoba": 0.900})
    starter = _mix(usage)
    overall = ContactLine(1000, 300, 0.5, 0.250, 0.310, 0.38, 0.07, 0.24)
    edge = arsenal_edge(hitter, overall, starter, usage)
    # 0.9*.200 + 0.1*.900 = .270, not the .550 an unweighted mean would give
    assert abs(edge.hitter["xwoba"] - 0.270) < 1e-9
    assert edge.fallback_share == 0.0


def test_a_pitch_the_hitter_has_not_seen_falls_back_and_says_how_much() -> None:
    usage = {"4-Seam": 0.60, "Splitter": 0.40}
    hitter = {"4-Seam": _mix({"4-Seam": 1.0})["4-Seam"]}
    overall = ContactLine(1000, 300, 0.5, 0.250, 0.310, 0.38, 0.07, 0.24)
    edge = arsenal_edge(hitter, overall, _mix(usage), usage)
    assert abs(edge.fallback_share - 0.40) < 1e-9
    assert abs(edge.hitter["xwoba"] - (0.6 * 0.320 + 0.4 * 0.310)) < 1e-9


def test_an_arsenal_with_no_usage_is_unrated_rather_than_scored() -> None:
    edge = arsenal_edge({}, ContactLine(0, 0, math.nan, math.nan, math.nan,
                                        math.nan, math.nan, math.nan), {}, {})
    score_edges([edge])
    assert edge.points == 0
    assert edge.fallback_share == 1.0


def test_the_fit_scores_whiff_low_is_better_and_the_rest_high() -> None:
    usage = {"4-Seam": 1.0}
    overall = ContactLine(1000, 300, 0.5, 0.250, 0.310, 0.38, 0.07, 0.24)
    contactful = arsenal_edge(
        _mix(usage, **{"4-Seam_whiff": 0.15}), overall, _mix(usage), usage
    )
    swinging = arsenal_edge(
        _mix(usage, **{"4-Seam_whiff": 0.40}), overall, _mix(usage), usage
    )
    score_edges([contactful, swinging], top_n=1)
    assert contactful.points > swinging.points
    assert "Whiff%" in contactful.top_in


def test_the_five_scored_fit_marks_are_the_ones_asked_for() -> None:
    assert [m.label for m in FIT_SCORED] == ["RV/100", "xwOBA", "Whiff%", "HH%", "Brl%"]


# --- the composite ranking ----------------------------------------------


def _final(name: str, *, early: int, late: int, fit: int = 0, ctx: int = 0) -> FinalScore:
    e, latest = HalfLine(half="early", points=early), HalfLine(half="late", points=late)
    return FinalScore(
        name=name, team="AAA", versus="Some Arm", slot=3,
        early=e, late=latest,
        context=ContextTerms(luck=ctx),
        edge=ArsenalEdge(points=fit),
    )


def test_the_total_adds_every_stage_and_keeps_them_apart() -> None:
    score = _final("Both Halves", early=15, late=13, fit=7, ctx=1)
    assert score.halves == 28
    assert score.total == 36
    assert score.weakest_half == 13


def test_a_tie_breaks_toward_the_hitter_with_no_bad_half() -> None:
    """The screen wants hitters who play all game, so the weaker half decides."""
    steady = _final("Steady", early=14, late=14)
    lopsided = _final("Lopsided", early=22, late=6)
    order = rank_final([lopsided, steady])
    assert [s.name for s in order] == ["Steady", "Lopsided"]


def test_the_composite_prints_both_halves_the_context_and_the_fit() -> None:
    result = _result()
    view = result.sections[0].hitters[0]
    early, late = half_lines(
        _frame([_half_pitch(i % 9 + 1) for i in range(60)])
    )
    view.early, view.late = early, late
    view.context = build_context(
        woba=0.300, xwoba=0.340, trends=TrendDeltas(), park_factor=104.0, worst_arm=True
    )
    view.edge = arsenal_edge(
        view.per_pitch, view.overall, result.sections[0].starter.arsenal,
        result.sections[0].starter.usage,
    )
    score_halves([early])
    score_halves([late])
    score_edges([view.edge])
    result.final = rank_final([
        FinalScore(
            name=view.line.name, team=view.line.team, versus=view.line.versus,
            slot=view.line.slot, early=early, late=late,
            context=view.context, edge=view.edge, pen_rank=None,
        )
    ])
    html = power_report.render_html(result)
    assert "The composite" in html
    assert "The arsenal fit" in html
    assert f"RV{TOP_PITCHES}" in html  # the run-value point has its own column
    assert f"RV/100 on {TOP_PITCHES}" in html
    assert "the bullpen&#x27;s half" in html or "bullpen&#39;s half" in html or (
        "bullpen's half" in html
    )
    assert "Matt Olson" in html


def test_the_composite_section_is_absent_when_the_season_rows_were_not_loaded() -> None:
    result = _result()
    assert not result.final
    assert "The composite" not in power_report.render_html(result)


# --- the run-value point on the pitches he will see most ------------------


def test_the_run_value_point_reads_the_starters_three_most_thrown_pitches() -> None:
    usage = {"4-Seam": 0.45, "Slider": 0.25, "Changeup": 0.20, "Curveball": 0.10}
    hitter = _mix(usage, **{
        "4-Seam_rv": 1.0, "Slider_rv": 1.0, "Changeup_rv": 1.0, "Curveball_rv": -50.0,
    })
    families, rv = top_pitch_rv(hitter, usage)
    assert families == ("4-Seam", "Slider", "Changeup")
    assert abs(rv - 1.0) < 1e-9  # the fourth pitch is not in the term


def test_the_run_value_point_is_weighted_inside_the_top_three() -> None:
    """The pitch thrown half the time counts half, not a third."""
    usage = {"4-Seam": 0.50, "Slider": 0.25, "Changeup": 0.25}
    hitter = _mix(usage, **{"4-Seam_rv": 4.0, "Slider_rv": 0.0, "Changeup_rv": 0.0})
    _, rv = top_pitch_rv(hitter, usage)
    assert abs(rv - 2.0) < 1e-9


def test_a_pitch_he_has_never_seen_does_not_borrow_his_overall_run_value() -> None:
    """This term is about these pitches; his line on other pitches is not a read."""
    usage = {"4-Seam": 0.60, "Splitter": 0.40}
    seen = _mix({"4-Seam": 1.0}, **{"4-Seam_rv": 3.0})
    _, rv = top_pitch_rv(seen, usage)
    assert abs(rv - 3.0) < 1e-9  # renormalized over what is readable
    _, none = top_pitch_rv({}, usage)
    assert math.isnan(none)
    assert rv_term(none) == 0


def test_the_run_value_point_signs_on_the_hitters_side_and_trebles_when_large() -> None:
    assert rv_term(0.4) == 1
    assert rv_term(-0.4) == -1
    assert rv_term(2.0) == 3
    assert rv_term(-2.5) == -3
    assert rv_term(0.0) == 0
    assert rv_term(math.nan) == 0


def test_the_run_value_point_enters_the_total_without_becoming_regression() -> None:
    """It is a matchup fact known before first pitch, not a form trend."""
    ahead = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(), top_pitch_rv=2.4
    )
    behind = build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(), top_pitch_rv=-2.4
    )
    assert (ahead.top_rv, ahead.total, ahead.regression) == (3, 3, 0)
    assert (behind.top_rv, behind.total) == (-3, -3)


def test_the_edge_carries_the_top_three_run_value_for_the_context_layer() -> None:
    usage = {"4-Seam": 0.50, "Slider": 0.30, "Changeup": 0.20}
    overall = ContactLine(1000, 300, 0.5, 0.250, 0.310, 0.38, 0.07, 0.24)
    edge = arsenal_edge(_mix(usage, **{"4-Seam_rv": 3.0, "Slider_rv": 3.0,
                                      "Changeup_rv": 3.0}), overall, _mix(usage), usage)
    assert edge.top_families == ("4-Seam", "Slider", "Changeup")
    assert abs(edge.top_rv - 3.0) < 1e-9
    assert build_context(
        woba=0.330, xwoba=0.330, trends=TrendDeltas(), top_pitch_rv=edge.top_rv
    ).top_rv == 3


def test_the_run_value_point_reorders_the_composite() -> None:
    """Two hitters level on skill are separated by the mix they are about to see."""
    fits = _final("Fits The Mix", early=14, late=14)
    fits.context = ContextTerms(top_rv=3)
    misfits = _final("Wrong Mix", early=14, late=14)
    misfits.context = ContextTerms(top_rv=-3)
    order = rank_final([misfits, fits])
    assert [s.name for s in order] == ["Fits The Mix", "Wrong Mix"]
    assert order[0].total - order[1].total == 6
