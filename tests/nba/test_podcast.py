from dataclasses import replace
from pathlib import Path

from engine_common.podcasts import picks as core
from engine_common.podcasts.extract import PICKS_DIR, Extraction, Pick
from engine_common.podcasts.shows import NBA, SHOWS
from nba_engine import podcast
from nba_engine.output.card import build_card, render_html
from tests.nba.test_output import DAY, LATE, priced

SHOW = next(s for s in SHOWS if s.key == "buckets")
PUB = "2025-12-10T15:00:00+00:00"


def _pick(**kw: object) -> Pick:
    base: dict[str, object] = dict(
        pick_id="p1",
        show=SHOW.key,
        show_name=SHOW.name,
        host="Matt Moore",
        episode_guid="g1",
        episode_title="Wednesday card",
        published=PUB,
        kind="official",
        market="game_ml",
        team="Knicks",
        opponent="Celtics",
        side=None,
        line=None,
        price=-150.0,
        units=None,
        seconds=300,
        quote="give me the Knicks on the moneyline at minus one fifty",
        reason="Boston on a back-to-back",
        description="Knicks ML",
        league=NBA,
        edge="",
        fair_line=None,
    )
    base.update(kw)
    return Pick(**base)  # type: ignore[arg-type]


def _store(tmp_path: Path, *picks: Pick) -> Path:
    store = tmp_path / "store"
    (store / PICKS_DIR).mkdir(parents=True)
    ex = Extraction("g1", "m", "picks-v2", "sha", list(picks), [], list(SHOW.leagues))
    (store / PICKS_DIR / "buckets__g1.json").write_text(ex.to_json())
    return store


def test_spoken_teams_map_to_one_code() -> None:
    assert podcast.team_code("Knicks") == "NYK"
    assert podcast.team_code("the Sixers") == "PHI"
    assert podcast.team_code("Golden State") == "GSW"
    assert podcast.team_code("Trail Blazers") == "POR"
    assert podcast.team_code("Los Angeles") is None


def test_card_prints_the_bet_under_its_game_without_the_quote(tmp_path: Path) -> None:
    rows = priced()
    total = _pick(
        pick_id="p2",
        market="game_total",
        side="under",
        line=220.5,
        price=None,
        edge="my number is 216",
    )
    store = _store(tmp_path, _pick(), total)
    card = build_card(rows, day=DAY)
    before = [(r.model_prob, r.edge, r.tier, r.gates) for r in card.rows]
    view = podcast.view(store, card, ledger=tmp_path / "pod.csv")
    assert view is not None and len(view.slate.placed) == 2
    page = render_html(card, view)
    game = page.index("Boston Celtics at New York Knicks")
    block = page.index("Podcast picks", game)
    assert "<b>Matt Moore · BUCKETS (Action Network)</b> (no graded picks yet)." in page[block:]
    assert "<b>NYK ML -150</b> (Boston on a back-to-back)" in page
    assert "<b>Under 220.5</b> (my number is 216)" in page
    assert "minus one fifty" not in page and "engine" not in page[block : block + 2000]
    assert "Podcast records" in page
    assert [(r.model_prob, r.edge, r.tier, r.gates) for r in card.rows] == before
    assert "Podcast picks" not in render_html(card)


def test_grade_day_settles_off_the_graded_rows(tmp_path: Path) -> None:
    rows = [r for r in priced() if r.matchup == LATE]
    graded = [replace(r, away_score=101, home_score=110) for r in rows]
    store = _store(tmp_path, _pick())
    ledger = tmp_path / "pod.csv"
    out = podcast.grade_day(store, DAY, graded, ledger=ledger)
    assert [(r["selection"], r["result"], r["price_source"]) for r in out] == [
        ("NYK ML", "win", "stated")
    ]
    rec = core.records(core.load_ledger(ledger)).official_for(SHOW.name, "Matt Moore")
    assert rec is not None and rec.wlp == "1-0"


def _placed(tmp_path: Path, pick: Pick, **kw: object) -> core.SlatePicks[podcast.Rows]:
    card = build_card(priced(), day=DAY)
    sub = tmp_path / str(len(list(tmp_path.iterdir())))
    view = podcast.view(_store(sub, pick), card, ledger=sub / "pod.csv", **kw)  # type: ignore[arg-type]
    assert view is not None
    return view.slate


def _game(matchup: str, hour: int, day: int = 10) -> core.Game[podcast.Rows]:
    from datetime import datetime, timezone

    away, _, home = matchup.partition(" @ ")
    return core.Game(matchup, home, away, datetime(2025, 12, day, hour, tzinfo=timezone.utc), DAY)


def test_a_pick_against_the_teams_earlier_opponent_is_that_games(tmp_path: Path) -> None:
    # Published at 15:00 after a 14:00 Knicks-Wizards tip: the bet was that game's.
    pick = _pick(opponent="Wizards")
    assert _placed(tmp_path, pick, earlier=[_game("NYK @ WAS", 14)]).placed == []
    # A misheard opponent still never vetoes the one game the team fits.
    assert len(_placed(tmp_path, pick).placed) == 1
    assert len(_placed(tmp_path, pick, earlier=[_game("NYK @ MIA", 14)]).placed) == 1


def test_a_pick_an_earlier_slate_game_fits_is_not_placed_again(tmp_path: Path) -> None:
    pick = _pick(opponent=None)
    assert _placed(tmp_path, pick, earlier=[_game("NYK @ MIA", 23)]).placed == []
    assert len(_placed(tmp_path, pick, earlier=[_game("NYK @ MIA", 14)]).placed) == 1


def test_passes_turned_down_sides_team_totals_and_player_lines_are_no_picks(
    tmp_path: Path,
) -> None:
    for quote in (
        "Yeah, it's definitely Knicks or pass",
        "It's definitely Knicks are passed",
        "I don't want the Knicks in this game at all",
        "the Knicks team total over 117 and a half, I'm playing that",
        "Tyler Herro over 22 and a half points at minus 104.",
    ):
        slate = _placed(tmp_path, _pick(quote=quote))
        assert slate.placed == [] and slate.unmatched == [], quote
    assert core.refusal(_pick(quote="I didn't mind the Knicks here")) == ""
    total = _pick(market="game_total", side="over", line=220.5, quote="over 220 and a half points")
    assert core.refusal(total) == ""


def test_a_regrade_drops_a_pick_no_longer_placed(tmp_path: Path) -> None:
    rows = [r for r in priced() if r.matchup == LATE]
    graded = [replace(r, away_score=101, home_score=110) for r in rows]
    ledger = tmp_path / "pod.csv"
    assert len(podcast.grade_day(_store(tmp_path / "a", _pick()), DAY, graded, ledger=ledger)) == 1
    passed = _store(tmp_path / "b", _pick(quote="Knicks or pass for me tonight"))
    assert podcast.grade_day(passed, DAY, graded, ledger=ledger) == []
    assert core.load_ledger(ledger) == []


def test_a_spelled_out_null_host_reads_as_none() -> None:
    ex = Extraction("g1", "m", "picks-v2", "sha", [_pick(host="null")], [], [NBA])
    assert Extraction.from_json(ex.to_json()).picks[0].host is None
