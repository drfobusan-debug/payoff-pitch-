"""The morning power screen: the five stages, and the note it writes.

The screen's job is to reproduce a hand process, so the tests pin the parts of it
that a refactor could quietly invert -- the run-value sign, the order of the cuts,
the power exception, and which turns a lineup slot actually gets.
"""

from __future__ import annotations

import math
from datetime import date as Date

import pandas as pd

from mlb_engine.features.swing import LEAGUE, WINDOW, SwingProfile
from mlb_engine.output import power_report
from mlb_engine.output.power_screen import (
    MIN_BATTER_PA,
    POWER_XWOBACON,
    HitterLine,
    HitterView,
    MatchupSection,
    PoolBatter,
    ScreenResult,
    apply_cuts,
    arsenal,
    arsenal_fit,
    batter_arsenal,
    batter_window_line,
    bf_pmf,
    contact_line,
    exposure,
    hitter_pool,
    pa_vs_starter,
    pitch_family,
    rank_starters,
    starter_damage,
    third_look_prob,
    wrc_plus,
)


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


# --- stage 1 --------------------------------------------------------------


def _starter_rows(n: int, *, brl: bool, k: bool = False) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        if k:
            rows.append(_pitch(description="swinging_strike", events="strikeout", lsa=None))
        else:
            rows.append(_pitch(events="double", la=28.0, lsa=6 if brl else 3))
    return _frame(rows)


def test_the_softer_arm_ranks_first_and_a_thin_sample_is_unreadable() -> None:
    soft = starter_damage(
        _starter_rows(130, brl=True),
        name="Soft",
        mlbam_id=1,
        team="AAA",
        opponent="BBB",
        throws="R",
    )
    tough = starter_damage(
        _starter_rows(130, brl=False, k=True),
        name="Tough",
        mlbam_id=2,
        team="BBB",
        opponent="AAA",
        throws="L",
    )
    thin = starter_damage(
        _starter_rows(20, brl=True),
        name="Thin",
        mlbam_id=3,
        team="CCC",
        opponent="DDD",
        throws="R",
    )
    ranked = rank_starters([tough, soft, thin])
    assert [c.name for c in ranked] == ["Soft", "Tough"]
    assert ranked[0].index > ranked[1].index
    assert soft.brl_pct == 1.0
    assert tough.k_bb_pct == 1.0


def test_no_readable_arm_is_an_empty_ranking_not_a_guess() -> None:
    thin = starter_damage(
        _starter_rows(10, brl=True), name="Thin", mlbam_id=1, team="A", opponent="B", throws="R"
    )
    assert rank_starters([thin]) == []


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
            {**_pitch(events="strikeout", description="swinging_strike"),
             "batter": 11, "p_throws": "L"},
        ]
    )
    pool = hitter_pool(
        rows,
        [PoolBatter(mlbam_id=10, name="Faces RHP", slot=2, bats="L"),
         PoolBatter(mlbam_id=11, name="Faces LHP Only", slot=3, bats="R"),
         PoolBatter(mlbam_id=12, name="No Rows", slot=4, bats="R")],
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
    rank_starters([card], top_n=1)

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
        starters_ranked=[card],
        sections=[section],
        cut_log=[cut],
    )


def test_the_note_carries_every_section_and_ends_on_the_recommendations() -> None:
    html = power_report.render_html(_result(), prepared_for="Franz")
    for heading in (
        "Thesis",
        "Data basis",
        "the arms, ranked",
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


def test_a_power_exception_is_disclosed_in_the_recommendation() -> None:
    """A bat kept on contact alone must not read as a hits or H+R+RBI play."""
    result = _result()
    result.sections[0].hitters[0].line.power_exception = True
    html = power_report.render_html(result)
    assert "home runs and total bases only" in html
