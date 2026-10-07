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
    rows = [
        replace(r, edge=0.031) if (r.matchup, r.side) == (LATE, "under") else r for r in priced()
    ]
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
    assert "<b>NYK ML -150</b>" in page and "<b>Under 220.5</b> (my number is 216; " in page
    assert "back-to-back" not in page and "minus one fifty" not in page
    assert "(my number is 216; engine +3.1% at 221.5)" in page
    assert "<b>NYK ML -150</b> (no engine edge)" in page
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
