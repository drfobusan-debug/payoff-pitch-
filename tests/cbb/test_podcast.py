"""CBB podcast picks: placed on the card's games, graded on ESPN finals, display only."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from cbb_engine import podcast
from cbb_engine.data.espn import Game as EspnGame
from cbb_engine.output import pdf
from engine_common.podcasts import picks as core
from engine_common.podcasts.extract import PICKS_DIR, Extraction, Pick
from engine_common.podcasts.shows import BY_KEY, CBB
from engine_runtime.ledger import Row

DAY = date(2026, 11, 10)
TIP = "2026-11-11T00:30:00Z"
SHOW = BY_KEY["c2c_hoops"]
TEAMS = {"e1": ("Duke Blue Devils", "Army Black Knights")}


def _row(market: str, side: str, line: float | None, american: float = -110.0, **kw: object) -> Row:
    base: dict[str, object] = dict(
        slate_date=DAY.isoformat(),
        event_id="e1",
        tip_utc=TIP,
        market=market,
        side=side,
        line=line,
        book="draftkings",
        pass_id="p",
        pass_kind="featured",
        bet_of_record=True,
        status="live",
        american=american,
        tier="Pass",
    )
    base.update(kw)
    return Row(**base)  # type: ignore[arg-type]


ROWS = [
    _row("game_ats", "home", -20.5),
    _row("game_ats", "away", 20.5),
    _row("game_ml", "home", None, -5000.0),
    _row("game_total", "over", 140.5),
    _row("game_total", "under", 140.5, tier="Moderate buy"),
]


def _pick(**kw: object) -> Pick:
    base: dict[str, object] = dict(
        pick_id="p1",
        show=SHOW.key,
        show_name=SHOW.name,
        host="Greg Peterson",
        episode_guid="g1",
        episode_title="11/10/26-Coast To Coast Hoops",
        published="2026-11-10T08:00:00+00:00",
        kind="official",
        market="game_ats",
        team="Army",
        opponent="Duke",
        side=None,
        line=20.5,
        price=None,
        units=None,
        seconds=60,
        quote="Army plus twenty and a half",
        reason="",
        description="Army +20.5",
        league=CBB,
        edge="Duke plays slow",
        fair_line=None,
    )
    base.update(kw)
    return Pick(**base)  # type: ignore[arg-type]


def _store(tmp_path: Path, *picks: Pick) -> Path:
    store = tmp_path / "store"
    (store / PICKS_DIR).mkdir(parents=True)
    ex = Extraction("g1", "m", "picks-v2", "sha", list(picks), [], [CBB])
    (store / PICKS_DIR / "c2c_hoops__g1.json").write_text(ex.to_json())
    return store


def test_the_shows_are_read_for_cbb() -> None:
    assert CBB in BY_KEY["c2c_hoops"].leagues and CBB in BY_KEY["cbb_experience"].leagues


def test_a_spoken_school_matches_its_own_team_only() -> None:
    assert podcast.same_team("Duke Blue Devils", "Duke")
    assert podcast.same_team("Iowa Hawkeyes", "Iowa")
    assert not podcast.same_team("Iowa State Cyclones", "Iowa")
    assert podcast.same_team("Iowa State Cyclones", "Iowa State")
    assert podcast.same_team("Iowa State Cyclones", "Iowa St.")
    assert not podcast.same_team("Texas A&M Aggies", "Texas")
    assert not podcast.same_team("Kansas City Roos", "Kansas")
    assert podcast.same_team("UConn Huskies", "UConn")
    assert not podcast.same_team("Duke Blue Devils", "Army")


def test_the_pick_prints_under_its_game_and_grades_on_the_final(tmp_path: Path) -> None:
    total = _pick(pick_id="p2", market="game_total", team="Duke", side="under", line=140.5, edge="")
    store = _store(tmp_path, _pick(), total)
    games = podcast.to_games(ROWS, TEAMS, DAY)
    view = podcast.view(store, games, DAY, ledger=tmp_path / "cbb.csv")
    assert view is not None and len(view.slate.placed) == 2
    page = pdf.build_html(DAY, ROWS, TEAMS, {}, podcast=view)
    game = page.index("Army Black Knights at Duke Blue Devils")
    assert page.index("Podcast picks", game) > game
    assert "Army Black Knights +20.5" in page and "(Duke plays slow)" in page

    finals = [
        EspnGame(
            "x",
            TIP,
            "Army Black Knights",
            "Duke Blue Devils",
            "STATUS_FINAL",
            False,
            False,
            (30, 25),
            (40, 35),
        )
    ]
    out = podcast.grade_day(store, DAY, ROWS, TEAMS, finals, ledger=tmp_path / "cbb.csv")
    by_market = {r["market"]: r for r in out}
    assert by_market["game_ats"]["result"] == "win"  # Army lost by 20 with +20.5
    assert (
        by_market["game_ats"]["price"] == "-110"
        and by_market["game_ats"]["price_source"] == "board"
    )
    assert by_market["game_total"]["result"] == "win"  # 130 under 140.5
    assert by_market["game_total"]["engine"] == "engine agrees"
    assert len(core.load_ledger(tmp_path / "cbb.csv")) == 2


def test_no_final_grades_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path, _pick())
    assert podcast.grade_day(store, DAY, ROWS, TEAMS, [], ledger=tmp_path / "cbb.csv") == []


def test_no_view_leaves_the_card_unchanged() -> None:
    page = pdf.build_html(DAY, ROWS, TEAMS, {})
    assert "Podcast picks" not in page and ".pod{" not in page


def test_a_pick_held_by_an_unpriced_earlier_game_is_not_placed_later(tmp_path: Path) -> None:
    store = _store(tmp_path, _pick(published="2026-11-08T08:00:00+00:00"))
    earlier = podcast.espn_games(
        [
            EspnGame(
                "y",
                "2026-11-09T00:00:00Z",
                "Army Black Knights",
                "Navy Midshipmen",
                "STATUS_FINAL",
                False,
                False,
                (30, 30),
                (35, 35),
            )
        ],
        date(2026, 11, 8),
    )
    games = podcast.to_games(ROWS, TEAMS, DAY)
    view = podcast.view(store, games, DAY, ledger=tmp_path / "cbb.csv", earlier=earlier)
    assert view is not None and view.slate.placed == []
