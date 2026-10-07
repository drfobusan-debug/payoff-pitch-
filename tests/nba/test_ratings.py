from datetime import date, timedelta

from nba_engine import cli
from nba_engine.data import arenas, boxes
from nba_engine.data.espn import parse_team_boxes, with_summary
from nba_engine.market import params
from nba_engine.models import rating_fit, ratings
from nba_engine.models.schedule import TeamSchedule, schedule
from nba_engine.schemas import GameResult, PlayerLine, TeamBox


def box(team, fga=88, oreb=10, tov=13, fta=22):
    return TeamBox(team, 42, fga, 13, 36, 17, fta, oreb, 33, tov, 19)


def final(eid, day, away, home, away_pts=110, home_pts=110, **kw):
    return GameResult(
        eid,
        day,
        away,
        home,
        "STATUS_FINAL",
        away_q=(30, 30, 25, away_pts - 85),
        home_q=(30, 30, 25, home_pts - 85),
        players=(PlayerLine("1", home, "A B", 30, 20, 5, 5, 2),),
        away_box=box(away),
        home_box=box(home),
        **kw,
    )


SUMMARY = {
    "boxscore": {
        "teams": [
            {
                "team": {"abbreviation": "SA"},
                "statistics": [
                    {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "44-95"},
                    {
                        "name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                        "displayValue": "14-34",
                    },
                    {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "18-20"},
                    {"name": "offensiveRebounds", "displayValue": "12"},
                    {"name": "defensiveRebounds", "displayValue": "35"},
                    {"name": "totalTurnovers", "displayValue": "14"},
                    {"name": "fouls", "displayValue": "17"},
                ],
            },
            {"team": {"abbreviation": "LAL"}, "statistics": []},
        ]
    },
    "header": {"competitions": [{"neutralSite": True}], "season": {"type": 2}},
    "gameInfo": {"venue": {"fullName": "T-Mobile Arena", "address": {"city": "Las Vegas"}}},
}


def test_team_box_parses_and_estimates_possessions():
    teams = parse_team_boxes(SUMMARY)
    assert set(teams) == {"SAS"}  # the team with no shooting lines is left out
    t = teams["SAS"]
    assert (t.fgm, t.fga, t.fg3m, t.fg3a, t.ftm, t.fta) == (44, 95, 14, 34, 18, 20)
    assert (t.oreb, t.dreb, t.tov, t.fouls) == (12, 35, 14, 17)
    assert t.possessions == 95 - 12 + 14 + 0.44 * 20


def test_summary_attaches_venue_and_a_missing_box_leaves_possessions_unknown():
    bare = GameResult("9", date(2025, 12, 16), "SAS", "LAL", "STATUS_FINAL")
    g = with_summary(bare, SUMMARY)
    assert g.neutral and g.city == "Las Vegas" and g.season_type == 2
    assert g.away_box is not None and g.home_box is None
    assert g.possessions is None


def test_team_boxes_round_trip_and_old_archives_still_load(tmp_path):
    day = date(2025, 12, 10)
    g = final("1", day, "SAS", "LAL", neutral=True, city="Las Vegas", season_type=2)
    boxes.write_results(tmp_path, day, [g])
    assert boxes.read_results(tmp_path, day) == [g]
    old = GameResult("2", day, "SAS", "LAL", "STATUS_FINAL", away_q=(30,), home_q=(25,))
    boxes.write_results(tmp_path, day, [old])
    (held,) = boxes.read_results(tmp_path, day) or []
    assert held.away_box is None and held.possessions is None and not held.neutral


class _SummaryClient:
    def results(self, day, on_summary=None):
        if on_summary is not None:
            on_summary("401", {"plays": [{"id": "1"}]})
        return [final("401", day, "SAS", "LAL")]


def test_each_summary_is_kept_raw_alongside_the_results(tmp_path):
    day = date(2025, 12, 10)
    boxes.ensure_results(tmp_path, day, _SummaryClient())
    assert boxes.read_raw_summary(tmp_path, day, "401") == {"plays": [{"id": "1"}]}
    assert boxes.read_results(tmp_path, day) is not None


def test_schedule_reads_rest_trips_and_altitude():
    d = date(2025, 11, 1)
    games = [
        final("1", d, "BOS", "NYK"),
        final("2", d + timedelta(days=1), "BOS", "DEN"),
        final("3", d + timedelta(days=3), "BOS", "UTA"),
        final("4", d + timedelta(days=4), "MIA", "BOS"),
    ]
    s = schedule(games)
    first, b2b, three, home = (s[(str(i), "BOS")] for i in range(1, 5))
    assert first.rest_days == 4 and not first.b2b and first.road_game == 1
    assert b2b.b2b and b2b.altitude and b2b.road_game == 2
    assert round(b2b.miles or 0) == round(arenas.miles(arenas.HOME["NYK"], arenas.HOME["DEN"]))
    assert b2b.tz_shift == -2.0
    assert three.three_in_four and three.road_game == 3
    assert home.road_game == 0
    assert not s[("2", "DEN")].altitude  # the home team lives up there


def test_a_new_season_resets_rest_and_an_unknown_neutral_site_has_no_miles():
    s = schedule(
        [
            final("1", date(2025, 4, 12), "BOS", "NYK"),
            final("2", date(2025, 10, 22), "BOS", "NYK", neutral=True, city="Nowhere"),
        ]
    )
    assert s[("2", "BOS")].rest_days == 4
    assert s[("2", "BOS")].miles is None and s[("2", "BOS")].tz_shift is None


def _season(n=40, start=date(2024, 11, 1), boost=20):
    """MIA beats everyone by ``boost``; the rest are even."""
    teams = ["MIA", "BOS", "NYK", "CHI", "DAL", "PHX"]
    out = []
    for i in range(n):
        a, h = teams[i % 6], teams[(i + 1 + i // 6) % 6]
        if a == h:
            continue
        ap = 110 + (boost if a == "MIA" else 0)
        hp = 110 + (boost if h == "MIA" else 0)
        out.append(final(str(i), start + timedelta(days=i), a, h, ap, hp))
    return out


def test_a_team_that_keeps_outscoring_its_rating_rises():
    preds, book = ratings.replay(_season())
    mia = [p for p in preds if "MIA" in (p.home, p.away)]
    edge = [p.margin if p.home == "MIA" else -p.margin for p in mia]
    assert edge[-1] > edge[0] + 5
    assert book.table()[0]["team"] == "MIA"


def test_replay_never_sees_a_later_game():
    games = _season()
    base, _ = ratings.replay(games)
    last = games[-1]
    changed = games[:-1] + [final(last.espn_id, last.game_date, last.away, last.home, 60, 150)]
    later, book = ratings.replay(changed)
    assert base == later  # only the last score changed, and it is predicted before it is folded in
    assert book.table() != ratings.replay(games)[1].table()


def test_preseason_and_all_star_games_are_not_rated():
    d = date(2025, 10, 5)
    pre = final("1", d, "BOS", "NYK", season_type=1)
    star = final("2", d, "LBN", "GIA", season_type=2)
    assert not ratings.rateable(pre) and not ratings.rateable(star)
    assert ratings.replay([pre, star])[0] == []


def test_b2b_cost_is_measured_and_applied_only_when_it_clears_zero():
    games, d = [], date(2024, 11, 1)
    teams = ["BOS", "NYK", "CHI", "DAL"]
    for i in range(400):
        day = d + timedelta(days=i)
        a, h = teams[i % 4], teams[(i + 1) % 4]
        tired = i % 2 == 0
        games.append(final(f"{i}", day, a, h, 110 - (6 if tired else 0), 110))
    sched = schedule(games)
    for i, g in enumerate(games):
        b2b = i % 2 == 0
        sched[(g.espn_id, g.away)] = TeamSchedule(1 if b2b else 2, b2b, False, 0.0, 0.0, False, 1)
    finals = {g.espn_id: g for g in games}
    flat = ratings.RatingParams(drift_sd=0.0, prior_sd=0.5, hca_gain=0.0)
    fit = rating_fit.fit_b2b(ratings.replay(games, flat, sched)[0], finals, sched, {2024})
    assert 4 < fit.cost < 8 and fit.applied == fit.cost
    priced = ratings.RatingParams(b2b_margin=fit.applied)
    book = ratings.RatingBook(priced)
    tired = book.predict(games[0], away_b2b=True)
    rested = book.predict(games[0])
    assert round(rested.margin - tired.margin + fit.applied, 6) == 0.0
    assert round(tired.total - rested.total, 6) == 0.0
    noise = rating_fit.B2BFit(100, 1.0, -0.5, 2.5)
    assert noise.applied == 0.0


def test_close_grade_lines_up_the_model_with_the_close():
    games = _season()
    preds, _ = ratings.replay(games)
    finals = {g.espn_id: g for g in games}
    lines = {(p.game_date, f"{p.away} @ {p.home}"): (-p.margin, p.total) for p in preds}
    grades = {g.market: g for g in rating_fit.vs_close(preds, finals, lines, {2024}, draws=50)}
    assert grades["margin"].n == len(preds)
    assert round(grades["margin"].model_mae - grades["margin"].close_mae, 9) == 0.0


def test_fit_ratings_with_no_finals_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    argv = ["fit-ratings", "--season", "2025-26", "--train", "2025", "--until", "2025-10-30"]
    assert cli.main(argv) == 1
    assert params.latest(tmp_path, ratings.NAME) is None
