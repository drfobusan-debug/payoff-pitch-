"""The books a card gets: prior games only, and a graceful nothing when thin."""

from __future__ import annotations

import argparse
from datetime import date as Date

import pandas as pd
import pytest

from nfl_engine import cli
from nfl_engine.features import books as books_mod
from nfl_engine.features import panel as panel_mod
from nfl_engine.features import quarterback as qb_mod
from nfl_engine.features import ratings as ratings_mod
from nfl_engine.schemas import Game, TeamGameInfo

TEAMS = ("AAA", "BBB", "CCC", "DDD")
TAKEN = "2025-09-18T18:00:00Z"


def _panel(seasons: tuple[int, ...] = (2024, 2025), weeks: int = 6) -> pd.DataFrame:
    """A toy panel wide enough to be week-indexed and joined like the real one."""
    rows = []
    game = 0
    for season in seasons:
        for week in range(1, weeks + 1):
            pairs = (
                (("AAA", "BBB"), ("CCC", "DDD")) if week % 2 else (("AAA", "CCC"), ("BBB", "DDD"))
            )
            for home, away in pairs:
                game += 1
                gid = f"{season}_{week}_{game}"
                for off, dfn, is_home in ((home, away, True), (away, home, False)):
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "game_id": gid,
                            "posteam": off,
                            "defteam": dfn,
                            "is_home": is_home,
                            "epa": 0.05 if off == "AAA" else 0.0,
                            "success": 0.45,
                            "drives": 11.0,
                        }
                    )
    return pd.DataFrame(rows)


def _games(panel: pd.DataFrame) -> pd.DataFrame:
    """The schedule rows ``with_results`` needs, in nflverse's shape."""
    home = panel[panel.is_home]
    return pd.DataFrame(
        {
            "game_id": home.game_id,
            "season": home.season,
            "week": home.week,
            "gameday": "2025-09-07",
            "home_team": home.posteam,
            "away_team": home.defteam,
            "home_score": 24,
            "away_score": 17,
            "spread_line": -3.0,
            "total_line": 44.0,
            "home_qb_id": "QB_" + home.posteam,
            "away_qb_id": "QB_" + home.defteam,
        }
    ).reset_index(drop=True)


def test_history_stops_at_the_week_being_priced():
    panel = _panel()
    joined = panel_mod.with_results(panel, _games(panel))
    history = books_mod._before(joined, 2025, 3)
    assert set(history.season.unique()) == {2024, 2025}
    assert history[history.season == 2025].week.max() == 2
    # The panel on disk holds later weeks; a card must not have read them.
    assert not ((history.season == 2025) & (history.week >= 3)).any()


def test_week_one_carries_the_prior_season_and_none_of_its_own():
    panel = _panel()
    joined = panel_mod.with_results(panel, _games(panel))
    history = books_mod._before(joined, 2025, 1)
    assert set(history.season.unique()) == {2024}


def test_the_earliest_week_of_the_earliest_season_has_no_history():
    panel = _panel(seasons=(2025,))
    joined = panel_mod.with_results(panel, _games(panel))
    assert books_mod._before(joined, 2025, 1).empty


def test_no_panel_still_returns_starters_and_says_why(monkeypatch: pytest.MonkeyPatch):
    panel = _panel()
    games = _games(panel)
    monkeypatch.setattr(books_mod.nflverse, "games", lambda: games)
    monkeypatch.setattr(books_mod.panel_mod, "panel", lambda *a, **k: pd.DataFrame())
    out = books_mod.as_of(2025, 3)
    assert out.notes == ("no_panel",)
    assert not out.ratings.is_usable()
    assert out.starters.prior  # the schedule alone is enough for the QB book
    assert "market is the mean" in out.summary()


def test_a_thin_book_is_flagged_rather_than_trusted(monkeypatch: pytest.MonkeyPatch):
    panel = _panel()
    games = _games(panel)
    monkeypatch.setattr(books_mod.nflverse, "games", lambda: games)
    monkeypatch.setattr(books_mod.panel_mod, "panel", lambda *a, **k: panel)
    out = books_mod.as_of(2025, 3)
    # 4 toy teams and 24 games is far under MIN_HISTORY_GAMES, and must say so
    # rather than price a card off a rating fitted on nothing.
    assert out.ratings.games_used < ratings_mod.MIN_HISTORY_GAMES
    assert out.notes == ("ratings_thin",)
    assert out.ratings.teams  # still fitted, and still reported
    assert out.ratings.rating("AAA").off_epa > out.ratings.rating("BBB").off_epa


def _game(home: str, away: str, season: int = 2025, week: int = 3) -> Game:
    return Game(
        game_id=f"{away}@{home}",
        season=season,
        week=week,
        game_date=Date(2025, 9, 21),
        home=TeamGameInfo(name=home, abbrev=home, is_home=True),
        away=TeamGameInfo(name=away, abbrev=away, is_home=False),
    )


def test_a_live_slate_is_named_from_the_schedule(monkeypatch: pytest.MonkeyPatch):
    panel = _panel()
    monkeypatch.setattr(books_mod.nflverse, "games", lambda: _games(panel))
    games = [_game("AAA", "BBB")]
    assert books_mod.attach_qbs(games, 2025, 3) == 2
    assert games[0].home_qb_id == "QB_AAA"
    assert games[0].away_qb_id == "QB_BBB"


def test_an_unnamed_starter_stays_unknown(monkeypatch: pytest.MonkeyPatch):
    panel = _panel()
    schedule = _games(panel)
    schedule["home_qb_id"] = None
    schedule["away_qb_id"] = ""
    monkeypatch.setattr(books_mod.nflverse, "games", lambda: schedule)
    games = [_game("AAA", "BBB")]
    assert books_mod.attach_qbs(games, 2025, 3) == 0
    assert games[0].home_qb_id is None
    assert games[0].away_qb_id is None


def test_a_game_the_schedule_does_not_have_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
):
    panel = _panel()
    monkeypatch.setattr(books_mod.nflverse, "games", lambda: _games(panel))
    games = [_game("ZZZ", "YYY")]
    assert books_mod.attach_qbs(games, 2025, 3) == 0
    assert games[0].home_qb_id is None


def test_the_card_prices_with_the_books_it_built(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys):
    """The regression this whole module exists for: the CLI priced with no book."""
    panel = _panel()
    schedule = _games(panel)
    monkeypatch.setattr(books_mod.nflverse, "games", lambda: schedule)
    monkeypatch.setattr(books_mod.panel_mod, "panel", lambda *a, **k: panel)
    monkeypatch.setattr(
        cli, "_fetch", lambda *a, **k: cli.Fetched(2025, 3, TAKEN, [_game("AAA", "BBB")], {})
    )
    monkeypatch.setattr(cli, "ledger_path", lambda root=None: tmp_path / "l.csv")
    seen: dict[str, object] = {}

    def spy(games, board, **kwargs):
        seen.update(kwargs)
        seen["games"] = games
        return []

    monkeypatch.setattr(cli, "price_slate", spy)
    args = argparse.Namespace(days=7, sims=100, top=5, write=False, ratings=True)
    assert cli.cmd_price(args) == 0
    assert isinstance(seen["book"], ratings_mod.RatingBook)
    assert seen["book"].teams  # a real fit, not the empty default
    assert isinstance(seen["starters"], qb_mod.StarterBook)
    priced = seen["games"]
    assert isinstance(priced, list)
    assert priced[0].home_qb_id == "QB_AAA"
    assert "ratings" in capsys.readouterr().out


def test_no_ratings_prices_the_way_it_used_to(monkeypatch: pytest.MonkeyPatch):
    out = cli._books(2025, 3, ratings=False)
    assert out.ratings.games_used == 0
    assert not out.starters.prior


def test_the_default_books_are_the_old_behaviour():
    """An empty book rates every team average, which is what the CLI had before."""
    out = books_mod.Books()
    assert out.ratings.games_used == 0
    assert out.ratings.rating("AAA").off_epa == 0.0
    assert out.starters.status(2025, 6, "AAA", "QB_AAA") == "unknown"
