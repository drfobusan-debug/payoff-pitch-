"""The slate article's regression prose, split ranks and pitcher comparison."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from mlb_engine.features.rolling import pen_arm_spread, woba_from_rates
from mlb_engine.features.team_splits import build_team_splits, league_contact
from mlb_engine.features.trend import pitcher_trends
from mlb_engine.output.daily_preview import (
    build_preview_report,
    bullpen_verdict,
    edge_side,
    lineup_profile,
    matchup_gap,
    matchup_verdict,
    starter_duel,
    starter_trend_sentence,
)
from mlb_engine.preview import (
    BullpenLine,
    LineupLine,
    StarterLine,
    load_previews,
    save_previews,
)
from tests.test_preview import _preview


def _starter(**over) -> StarterLine:
    base = dict(
        name="Casey Mize",
        pitches=900,
        k_pct=0.25,
        xk_pct=0.27,
        bb_pct=0.05,
        xbb_pct=0.06,
        csw=0.29,
        whiff=0.11,
        swstr=0.13,
        zone_pct=0.49,
        xwoba_allowed=0.326,
        barrel_allowed=0.04,
        dxwoba=0.049,
        spin=2178.0,
        hard_hit_allowed=0.22,
        babip_allowed=0.228,
        siera=2.98,
        siera_trend=-0.60,
        stuff_trend=-0.106,
        vfa_trend=-0.54,
        league_xwoba_allowed=0.370,
    )
    base.update(over)
    return StarterLine(**base)


def _lineup(**over) -> LineupLine:
    base = dict(
        n=9,
        woba=0.340,
        xwoba=0.389,
        dxwoba=0.049,
        xslg=0.450,
        barrel=0.09,
        vs_hand="R",
        split_woba=0.336,
        split_rank=12,
        split_of=30,
        split_bucket="middle",
        home_woba=0.337,
        away_woba=0.361,
        is_home=False,
        league_xwoba=0.372,
    )
    base.update(over)
    return LineupLine(**base)


def test_starter_row_shows_swstr_and_hard_hit_without_zone():
    gp = _preview(home_starter=_starter(), away_starter=_starter(name="Mitch Bratt"))
    html, _ = build_preview_report(dt.date(2026, 8, 5), [gp])

    assert "SwStr%" in html and "Hard-hit%" in html
    assert "Zone%" not in html


def test_trend_sentence_reads_direction_from_the_pitchers_side():
    txt = starter_trend_sentence("SD", _starter())

    assert "SIERA 2.98" in txt
    assert "improving" in txt  # SIERA fell
    assert txt.count("slipping") == 2  # CSW% and velocity both fell
    assert "-10.6%" in txt and "-0.5" in txt


def test_trend_sentence_flags_a_thin_half_window_instead_of_guessing():
    txt = starter_trend_sentence("SD", _starter(siera_trend=None, stuff_trend=None))

    assert txt.count("too thin to read") == 2


def test_trend_sentence_reads_the_babip_xwoba_gap():
    owed = starter_trend_sentence("SD", _starter())
    unlucky = starter_trend_sentence("AZ", _starter(dxwoba=-0.090, babip_allowed=0.330))

    assert "0.277 wOBA on 0.326 xwOBA" in owed  # xwOBA minus the gap
    assert "the hits are owed" in owed
    assert "49 points more damage" in owed
    assert "lucky 0.228 BABIP" in owed
    assert "hit harder than the contact deserved" in unlucky
    assert "unlucky 0.330 BABIP" in unlucky


def test_verdict_reads_the_simulators_own_matchup_projection():
    lu = _lineup(proj_woba=0.305, proj_woba_vs_league=0.333)
    assert round(matchup_gap(lu, _starter()), 3) == -0.028
    assert edge_side(lu, _starter()) == "arm"

    txt = matchup_verdict("SD", "AZ", lu, _starter())
    assert "Edge: Casey Mize (AZ)" in txt
    assert "projects SD's order at a 0.305 wOBA against Casey Mize" in txt
    assert "28 points below the 0.333" in txt


def test_verdict_names_the_bats_when_the_projection_beats_an_average_arm():
    lu = _lineup(proj_woba=0.350, proj_woba_vs_league=0.333)
    assert edge_side(lu, _starter()) == "bats"
    assert "Edge: SD's bats" in matchup_verdict("SD", "AZ", lu, _starter())


def test_verdict_calls_a_wash_when_the_starter_projects_as_average():
    lu = _lineup(proj_woba=0.336, proj_woba_vs_league=0.333)
    assert edge_side(lu, _starter()) == "wash"
    assert "Wash" in matchup_verdict("SD", "AZ", lu, _starter())


def test_verdict_falls_back_to_centred_xwoba_without_a_projection():
    # Raw numbers favour the bats (.389 vs .326 allowed); against their own
    # baselines the starter is 44 points better than league and the bats 17.
    lu, sl = _lineup(), _starter()
    assert lu.proj_woba is None
    assert edge_side(lu, sl) == "arm"

    txt = matchup_verdict("SD", "AZ", lu, sl)
    assert "44 points better than league" in txt
    assert "17 points better than league" in txt


def test_woba_from_rates_puts_matchup_probabilities_on_the_woba_scale():
    league = {"1B": 0.140, "2B": 0.045, "3B": 0.004, "HR": 0.032, "BB": 0.085, "K": 0.225, "OUT": 0.469}
    assert 0.30 < woba_from_rates(league) < 0.34
    assert woba_from_rates({"OUT": 1.0}) == 0.0


def test_verdict_reports_split_rank_bucket_and_venue_form():
    txt = matchup_verdict("SD", "AZ", _lineup(), _starter())

    assert "an order that hits right-handers at a 0.336 wOBA" in txt
    assert "12 of 30" in txt
    assert "middle third" in txt
    assert "on the road tonight" in txt
    assert "their better half" in txt


def test_verdict_says_so_when_the_split_is_too_thin_to_rank():
    txt = matchup_verdict("SD", "AZ", _lineup(split_woba=None, split_rank=None), _starter())
    assert "too thin a sample against right-handers to rank" in txt


def test_narration_speaks_the_edge_and_the_split_rank():
    gp = _preview(
        home_starter=_starter(name="Mitch Bratt"),
        away_starter=_starter(),
        home_lineup=_lineup(vs_hand="R"),
        away_lineup=_lineup(vs_hand="L", split_rank=21, split_bucket="bottom"),
    )
    _, narr = build_preview_report(dt.date(2026, 8, 5), [gp])

    assert "has the edge on" in narr
    assert "21 of 30 against lefties, bottom third" in narr


def test_previews_written_before_the_new_fields_still_load(tmp_path):
    path = tmp_path / "previews.json"
    save_previews([_preview()], path)
    raw = path.read_text()
    assert "siera_trend" in raw

    legacy = raw.replace('"siera_trend": null, ', "").replace('"league_xwoba": null, ', "")
    path.write_text(legacy)

    loaded = load_previews(path)
    assert loaded[0].home_starter.siera_trend is None
    assert loaded[0].home_lineup.league_xwoba is None


# --- feature layer ---------------------------------------------------------
def _pitch(day: str, *, csw: bool, velo: float, pa: bool) -> dict:
    return {
        "game_date": day,
        "description": "swinging_strike" if csw else "foul",
        "pitch_type": "FF",
        "release_speed": velo,
        "events": "strikeout" if pa else None,
        "woba_denom": 1.0 if pa else None,
        "woba_value": 0.0,
        "bb_type": None,
    }


def test_pitcher_trend_needs_both_halves_before_it_reports_a_change():
    recent = pd.DataFrame(
        [_pitch("2026-08-01", csw=True, velo=94.0, pa=i % 4 == 0) for i in range(200)]
    )
    trends = pitcher_trends(recent, dt.date(2026, 8, 5), 42)
    assert trends.vfa.recent is not None
    assert trends.vfa.prior is None
    assert trends.vfa.delta is None

    both = pd.concat(
        [recent, pd.DataFrame([_pitch("2026-07-05", csw=True, velo=96.0, pa=i % 4 == 0) for i in range(200)])]
    )
    trends = pitcher_trends(both, dt.date(2026, 8, 5), 42)
    assert trends.vfa.delta is not None
    assert round(trends.vfa.delta, 1) == -2.0


def _bat(day: str, team: str, opp: str, *, topbot: str, hand: str, woba: float) -> dict:
    return {
        "game_date": day,
        "home_team": team if topbot == "Bot" else opp,
        "away_team": opp if topbot == "Bot" else team,
        "inning_topbot": topbot,
        "p_throws": hand,
        "woba_value": woba,
        "woba_denom": 1.0,
        "batter": 1,
        "pitcher": 2,
        "bb_type": "line_drive",
        "estimated_woba_using_speedangle": woba,
    }


def _split_frame() -> pd.DataFrame:
    rows = []
    for i, (team, woba) in enumerate([("SD", 0.5), ("AZ", 0.3), ("LAD", 0.1)]):
        for n in range(200):
            topbot = "Bot" if n % 2 else "Top"
            rows.append(_bat("2026-08-01", team, f"OP{i}", topbot=topbot, hand="R", woba=woba))
    return pd.DataFrame(rows)


def test_team_splits_rank_and_bucket_the_platoon_offenses():
    splits = build_team_splits(_split_frame(), dt.date(2026, 8, 5), 42)

    sd = splits["SD"].vs_hand("R")
    lad = splits["LAD"].vs_hand("R")
    assert sd is not None and lad is not None
    assert (sd.rank, sd.of, sd.bucket) == (1, 3, "top")
    assert (lad.rank, lad.bucket) == (3, "bottom")
    assert splits["SD"].vs_hand("L") is None  # no LHP faced
    assert splits["SD"].home_woba == 0.5 and splits["SD"].away_woba == 0.5


def test_team_splits_skip_offenses_with_too_few_plate_appearances():
    thin = _split_frame().head(10)
    splits = build_team_splits(thin, dt.date(2026, 8, 5), 42)
    assert splits["SD"].vs_hand("R") is None


def test_league_contact_baselines_average_per_player():
    league = league_contact(_split_frame(), dt.date(2026, 8, 5), 42)
    assert league.batter == 0.3  # mean of the three clubs' hitters
    assert league.pitcher == 0.3


def test_league_contact_is_empty_without_data():
    league = league_contact(pd.DataFrame(), dt.date(2026, 8, 5), 42)
    assert league.batter is None and league.pitcher is None


def _pen(**over) -> BullpenLine:
    base = dict(
        xwoba_allowed=0.31,
        k_pct=0.24,
        zone_pct=0.45,
        recent_load=0.90,
        fatigue=0.0,
        proj_woba=0.320,
        proj_woba_close=0.305,
        arm_spread=0.028,
        arms=7,
    )
    base.update(over)
    return BullpenLine(**base)


def test_starter_duel_names_the_better_arm_by_siera():
    gp = _preview(
        home_starter=_starter(),
        away_starter=_starter(name="Mitch Bratt", siera=5.94, swstr=0.09, hard_hit_allowed=0.44),
    )
    txt = starter_duel(gp)

    assert "Better pitcher: Casey Mize (BBB)" in txt
    assert "by a wide margin" in txt
    assert "2.98 SIERA to Mitch Bratt's 5.94" in txt
    assert "13% SwStr" in txt and "44% hard-hit" in txt


def test_starter_duel_says_so_when_siera_is_missing():
    gp = _preview(home_starter=_starter(siera=None), away_starter=_starter(name="Mitch Bratt"))

    assert "No SIERA read on both arms" in starter_duel(gp)


def test_bullpen_verdict_reads_freshness_projection_and_volatility():
    txt = bullpen_verdict("SD", "AZ", _pen())

    assert "AZ's pen" in txt
    assert "Fresh" in txt and "nobody on back-to-back days" in txt
    assert "three-day workload 0.90× normal" in txt
    assert "projects 0.320 wOBA against SD's order" in txt
    assert "0.305 once the 8th-inning arms take it" in txt
    assert "normal spread" in txt and "across 7 arms" in txt


def test_bullpen_verdict_flags_a_worked_volatile_walk_prone_pen():
    txt = bullpen_verdict("SD", "AZ", _pen(fatigue=80.0, arm_spread=0.061, zone_pct=0.37))

    assert "Worked" in txt and "4 arms gassed" in txt
    assert "volatile" in txt and "0.061 wOBA spread" in txt
    assert "walk trap at 37% zone" in txt


def test_bullpen_verdict_admits_a_thin_relief_sample():
    txt = bullpen_verdict("SD", "AZ", _pen(proj_woba=None, arm_spread=None, arms=1, fatigue=None))

    assert "no projection against this order" in txt
    assert "too few arms" in txt
    assert "Workload unknown" in txt


def test_article_says_which_pen_holds_late():
    gp = _preview(home_pen=_pen(proj_woba=0.300), away_pen=_pen(proj_woba=0.360))
    html, narr = build_preview_report(dt.date(2026, 8, 5), [gp])

    assert "Late-inning edge: <b>BBB's pen</b>, by 60 points" in html
    assert "late innings favor the BBB pen" in narr


def test_article_calls_the_pens_even_when_they_project_together():
    gp = _preview(home_pen=_pen(proj_woba=0.320), away_pen=_pen(proj_woba=0.323))
    html, _ = build_preview_report(dt.date(2026, 8, 5), [gp])

    assert "Late-inning edge: <b>even</b>" in html


def test_lineup_profile_gives_general_form_then_tonights_situation():
    lu = _lineup(team_woba=0.331, team_rank=6, team_of=30, venue_rank=4, venue_of=30)
    txt = lineup_profile("SD", lu)

    assert "0.331 wOBA club overall" in txt
    assert "<b>6 of 30</b> (top third)" in txt
    assert "hits right-handers at a 0.336 wOBA" in txt  # tonight's platoon split
    assert "on the road they hit 0.361, 4 of 30 in that split" in txt
    assert "24 points better than the 0.337 they hit at home" in txt


def test_lineup_profile_falls_back_to_the_lineup_line_without_club_ranks():
    txt = lineup_profile("SD", _lineup())

    assert "0.340 wOBA / 0.389 xwOBA batting order" in txt


def test_pen_arm_spread_measures_the_gap_between_a_pens_arms():
    rows = []
    for arm, ev in ((1, "home_run"), (2, "strikeout")):
        rows.extend({"pitcher": arm, "events": ev} for _ in range(30))
    spread, arms = pen_arm_spread(pd.DataFrame(rows))

    assert arms == 2
    assert spread is not None and spread > 0.05  # a slugger's pen next to a shutdown arm

    tight, _ = pen_arm_spread(pd.DataFrame([{"pitcher": 1, "events": "strikeout"}] * 30))
    assert tight is None  # one arm can't have a spread


def test_team_splits_rank_the_venue_and_overall_offenses():
    splits = build_team_splits(_split_frame(), dt.date(2026, 8, 5), 42)

    sd, lad = splits["SD"], splits["LAD"]
    assert sd.overall is not None and sd.overall.rank == 1
    assert lad.overall is not None and lad.overall.rank == 3
    home = sd.at_venue(True)
    road = sd.at_venue(False)
    assert home is not None and road is not None
    assert home.rank == 1 and road.rank == 1
    assert sd.at_venue(None) is None
