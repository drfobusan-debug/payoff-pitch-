"""The slate article's regression prose, split ranks and pitcher comparison."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from mlb_engine.features.team_splits import build_team_splits, league_contact
from mlb_engine.features.trend import pitcher_trends
from mlb_engine.output.daily_preview import (
    build_preview_report,
    edge_side,
    matchup_verdict,
    starter_trend_sentence,
)
from mlb_engine.preview import LineupLine, StarterLine, load_previews, save_previews
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


def test_verdict_centres_each_side_on_its_own_league_baseline():
    # Raw numbers favour the bats (.389 vs .326 allowed); against their own
    # baselines the starter is 44 points better than league and the bats 17.
    lu, sl = _lineup(), _starter()
    assert edge_side(lu, sl) == "arm"

    txt = matchup_verdict("SD", "AZ", lu, sl)
    assert "Edge: Casey Mize (AZ)" in txt
    assert "44 points better than league" in txt
    assert "17 points better than league" in txt


def test_verdict_names_the_bats_when_the_lineup_is_the_better_side():
    lu = _lineup(xwoba=0.440)
    assert edge_side(lu, _starter()) == "bats"
    assert "Edge: SD's bats" in matchup_verdict("SD", "AZ", lu, _starter())


def test_verdict_calls_a_wash_when_neither_side_is_ahead():
    lu = _lineup(xwoba=0.372 + 0.044)
    assert edge_side(lu, _starter()) == "wash"
    assert "Wash" in matchup_verdict("SD", "AZ", lu, _starter())


def test_verdict_reports_split_rank_bucket_and_venue_form():
    txt = matchup_verdict("SD", "AZ", _lineup(), _starter())

    assert "they hit right-handers at a 0.336 wOBA" in txt
    assert "12 of 30" in txt
    assert "middle third" in txt
    assert "on the road tonight" in txt
    assert "their better half" in txt


def test_verdict_says_so_when_the_split_is_too_thin_to_rank():
    txt = matchup_verdict("SD", "AZ", _lineup(split_woba=None, split_rank=None), _starter())
    assert "too thin to rank" in txt


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
