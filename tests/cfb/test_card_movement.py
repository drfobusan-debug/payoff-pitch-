"""The slate card orders games by kickoff and prints open -> now line movement."""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path

from cfb_engine.output.card import _movement_line, _slate_order, _spoken_movement, build_article
from cfb_engine.recommendations import Recommendation, Tier, load_json, save_json

DAY = date(2026, 9, 19)


def _rec(
    game_id: str,
    market: str,
    selection: str,
    side: str,
    *,
    line: float | None = None,
    american: float = -110,
    open_line: float | None = None,
    open_american: float | None = None,
    kickoff: str | None = None,
) -> Recommendation:
    return Recommendation(
        game_date=DAY,
        game_id=game_id,
        matchup=f"AWAY{game_id} @ HOME{game_id}",
        market=market,
        selection=selection,
        line=line,
        model_prob=0.55,
        market_american=american,
        ev=0.02,
        edge=0.02,
        fair_prob=0.52,
        tier=Tier.PASS,
        home_abbrev=f"HOME{game_id}",
        away_abbrev=f"AWAY{game_id}",
        team_side="home",
        side=side,
        open_line=open_line,
        open_american=open_american,
        kickoff_utc=kickoff,
    )


def _game(game_id: str, kickoff: str | None, *, moved: bool = True) -> list[Recommendation]:
    h = f"HOME{game_id}"
    return [
        _rec(
            game_id,
            "game_ml",
            f"{h} ML",
            "win",
            american=-160,
            open_american=-150 if moved else None,
            kickoff=kickoff,
        ),
        _rec(
            game_id,
            "game_ats",
            f"{h} -3.5",
            "cover",
            line=-3.5,
            open_line=-3.0 if moved else None,
            kickoff=kickoff,
        ),
        _rec(
            game_id,
            "game_total",
            "Over 51.5",
            "over",
            line=51.5,
            open_line=52.5 if moved else None,
            kickoff=kickoff,
        ),
    ]


def test_movement_line_prints_open_then_now():
    html = _movement_line(_game("1", "2026-09-19T16:00:00Z"))
    assert "ML HOME1 -150 → <b>-160</b>" in html
    assert "Spread HOME1 -3 → <b>-3.5</b>" in html
    assert "Total 52.5 → <b>51.5</b>" in html


def test_movement_line_marks_unmoved_and_hides_when_no_open_board():
    recs = _game("1", None)
    same = [dataclasses.replace(r, open_line=r.line, open_american=r.market_american) for r in recs]
    assert "(unmoved)" in _movement_line(same)
    assert _movement_line(_game("1", None, moved=False)) == ""


def test_spoken_movement_only_names_what_moved():
    spoken = _spoken_movement(_game("1", "2026-09-20T00:00:00Z"))
    assert "Kickoff 8:00 PM Eastern" in spoken
    assert "moneyline opened -150, now -160" in spoken
    assert "total opened 52.5 and sits at 51.5" in spoken
    assert _spoken_movement(_game("2", None, moved=False)) == ""


def test_slate_orders_early_to_late_with_missing_kickoff_last():
    night = _game("n", "2026-09-20T02:30:00Z")
    noon = _game("a", "2026-09-19T16:00:00Z")
    late = _game("l", "2026-09-19T23:30:00Z")
    blank = _game("z", None)
    groups = {g[0].game_id: g for g in (night, blank, late, noon)}
    assert [g[0].game_id for g in _slate_order(groups)] == ["a", "l", "n", "z"]

    html, _ = build_article(DAY, night + noon + late)
    assert html.index("AWAYa @ HOMEa") < html.index("AWAYl @ HOMEl") < html.index("AWAYn @ HOMEn")
    assert "12:00 PM ET" in html and "10:30 PM ET" in html


def test_new_fields_round_trip_json(tmp_path: Path):
    recs = _game("1", "2026-09-19T16:00:00Z")
    save_json(recs, tmp_path / "p.json")
    back = load_json(tmp_path / "p.json")
    assert (back[0].open_american, back[0].kickoff_utc) == (-150, "2026-09-19T16:00:00Z")
    assert back[2].open_line == 52.5
