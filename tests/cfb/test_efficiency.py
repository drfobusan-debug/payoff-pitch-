"""Opponent-adjusted efficiency ridge, the returning-production term, and the
leak-free week cut that keeps a game out of its own rating."""

from __future__ import annotations

import math

from cfb_engine.config import Config
from cfb_engine.data.cfbd import RatingBook, TeamGamePPA, TeamRating
from cfb_engine.data.efficiency import (
    POINTS_PER_PPA,
    EfficiencyBook,
    EfficiencyProvider,
    TeamEfficiency,
    blend_efficiency,
    fit_efficiency,
)
from cfb_engine.data.returning import ReturningBook

# Four teams whose true offensive strengths are 0.4/0.2/0.0/-0.2 and whose
# defences suppress 0.1/0.0/-0.1/-0.2 PPA per play (higher = stingier). A double
# round robin gives the ridge a connected graph, which the opponent adjustment
# needs to separate offence from defence at all.
TRUE_OFF = {"Alpha": 0.40, "Bravo": 0.20, "Charlie": 0.00, "Delta": -0.20}
TRUE_DEF = {"Alpha": 0.10, "Bravo": 0.00, "Charlie": -0.10, "Delta": -0.20}


def _round_robin() -> list[tuple[str, str, float, float]]:
    rows = []
    for home in TRUE_OFF:
        for away in TRUE_OFF:
            if home == away:
                continue
            for team, opp, site in ((home, away, 1.0), (away, home, -1.0)):
                ppa = TRUE_OFF[team] - TRUE_DEF[opp] + 0.05 * site
                rows.append((team, opp, site, ppa))
    return rows


def test_ridge_recovers_offence_defence_and_home_field():
    book = fit_efficiency(_round_robin(), alpha=1e-6)
    assert math.isclose(book.hfa, 0.05, abs_tol=1e-3)
    # Only differences are identified (the intercept absorbs a constant), so
    # compare each team against the field rather than to the absolute truth.
    for team in TRUE_OFF:
        want = (TRUE_OFF[team] - sum(TRUE_OFF.values()) / 4) + (
            TRUE_DEF[team] - sum(TRUE_DEF.values()) / 4
        )
        got = book.ratings[team.lower()].net
        assert math.isclose(got, want, abs_tol=1e-3), team


def test_net_ordering_is_not_sign_flipped():
    """A stingy defence must help, not hurt.

    The design matrix enters a defence with -1, so its coefficient is already
    sign-flipped and net is off + def. Getting it backwards is silent -- the
    ratings still look like numbers -- so it is pinned here.
    """
    book = fit_efficiency(_round_robin(), alpha=1e-6)
    order = sorted(book.ratings.values(), key=lambda e: -e.net)
    assert [e.team for e in order] == ["Alpha", "Bravo", "Charlie", "Delta"]


def test_min_games_hides_a_thin_rating():
    book = EfficiencyBook(
        ratings={
            "alpha": TeamEfficiency("Alpha", 0.3, 0.1, games=5),
            "bravo": TeamEfficiency("Bravo", 0.0, 0.0, games=1),
        }
    )
    assert book.get("Alpha") is not None
    assert book.get("Bravo") is None
    assert book.net_gap_points("Alpha", "Bravo") is None


def test_net_gap_converts_to_points():
    book = EfficiencyBook(
        ratings={
            "alpha": TeamEfficiency("Alpha", 0.2, 0.1, games=6),
            "bravo": TeamEfficiency("Bravo", 0.0, 0.0, games=6),
        }
    )
    gap = book.net_gap_points("Alpha", "Bravo")
    assert gap is not None
    assert math.isclose(gap, 0.3 * POINTS_PER_PPA, rel_tol=1e-9)


def _base_book() -> RatingBook:
    return RatingBook(
        ratings={"alpha": TeamRating("Alpha", 34.0, 20.0)},  # net +14
        league_avg=27.0,
    )


def test_blend_at_zero_is_a_no_op():
    eff = EfficiencyBook(ratings={"alpha": TeamEfficiency("Alpha", 0.0, 0.0, games=8)})
    out = blend_efficiency(_base_book(), eff, blend=0.0, league_avg=27.0)
    assert out is not None
    assert out.ratings["alpha"] == _base_book().ratings["alpha"]


def test_blend_moves_the_net_but_preserves_the_total():
    eff = EfficiencyBook(ratings={"alpha": TeamEfficiency("Alpha", 0.0, 0.0, games=8)})
    out = blend_efficiency(_base_book(), eff, blend=0.5, league_avg=27.0)
    assert out is not None
    rating = out.ratings["alpha"]
    # efficiency net 0 -> halfway between +14 and 0
    assert math.isclose(rating.offense - rating.defense, 7.0, abs_tol=1e-9)
    assert math.isclose(rating.offense + rating.defense, 54.0, abs_tol=1e-9)


def test_efficiency_becomes_the_book_when_sp_plus_is_missing():
    eff = EfficiencyBook(ratings={"alpha": TeamEfficiency("Alpha", 0.2, 0.0, games=8)})
    out = blend_efficiency(None, eff, blend=0.0, league_avg=27.0)
    assert out is not None
    rating = out.ratings["alpha"]
    assert math.isclose(rating.offense - rating.defense, 0.2 * POINTS_PER_PPA, rel_tol=1e-9)
    assert math.isclose(rating.offense + rating.defense, 54.0, abs_tol=1e-9)


def test_no_efficiency_book_leaves_ratings_untouched():
    base = _base_book()
    assert blend_efficiency(base, None, blend=0.9, league_avg=27.0) is base


class _FakeCFBD:
    """Two seasons of PPA: the current one thin, the prior one complete."""

    def __init__(self, rows: list[TeamGamePPA], neutral: set[int] | None = None) -> None:
        self.rows = rows
        self.neutral = neutral or set()

    def fetch_team_game_ppa(self, season: int) -> list[TeamGamePPA]:
        return [r for r in self.rows if r.season == season]

    def neutral_game_ids(self, season: int) -> set[int]:
        return self.neutral


def _ppa_rows(season: int, week_span: range) -> list[TeamGamePPA]:
    rows: list[TeamGamePPA] = []
    gid = season * 1000
    for week in week_span:
        for home in TRUE_OFF:
            for away in TRUE_OFF:
                if home == away:
                    continue
                gid += 1
                for team, opp, is_home in ((home, away, True), (away, home, False)):
                    rows.append(
                        TeamGamePPA(
                            game_id=gid,
                            season=season,
                            week=week,
                            season_type="regular",
                            team=team,
                            opponent=opp,
                            offence_ppa=TRUE_OFF[team] - TRUE_DEF[opp],
                            home=is_home,
                        )
                    )
    return rows


def test_book_only_reads_weeks_before_the_slate():
    rows = _ppa_rows(2025, range(1, 6))
    provider = EfficiencyProvider(_FakeCFBD(rows))  # type: ignore[arg-type]
    book = provider.book(2025, before_week=3)
    assert book is not None
    # 12 games a week, each team in 6 of them: weeks 1-2 only, so 12 team-games
    assert book.ratings["alpha"].games == 12
    # and week 3+ is genuinely excluded rather than merely down-weighted
    assert provider.book(2025, before_week=6).ratings["alpha"].games == 30  # type: ignore[union-attr]


def test_book_falls_back_to_the_prior_season_when_the_current_one_is_thin():
    rows = _ppa_rows(2024, range(1, 6)) + _ppa_rows(2025, range(1, 2))
    provider = EfficiencyProvider(_FakeCFBD(rows))  # type: ignore[arg-type]
    book = provider.book(2025, before_week=1)
    assert book is not None
    assert book.ratings["alpha"].games == 30  # the whole 2024 season


def test_returning_margin_delta_signs_and_cap():
    book = ReturningBook(shares={"alpha": 0.80, "bravo": 0.40})
    # gap +0.40 x 2.5 pts = +1.0 to the home side
    assert math.isclose(book.margin_delta("Alpha", "Bravo", 2.5, 3.0), 1.0, abs_tol=1e-9)
    assert math.isclose(book.margin_delta("Bravo", "Alpha", 2.5, 3.0), -1.0, abs_tol=1e-9)
    # capped
    assert math.isclose(book.margin_delta("Alpha", "Bravo", 50.0, 3.0), 3.0, abs_tol=1e-9)
    # disabled, and unknown teams
    assert book.margin_delta("Alpha", "Bravo", 0.0, 3.0) == 0.0
    assert book.margin_delta("Alpha", "Nobody", 2.5, 3.0) == 0.0


def test_defaults_keep_both_new_features_out_of_the_price():
    cfg = Config()
    assert cfg.efficiency_blend == 0.0
    assert cfg.returning_pts == 0.0
