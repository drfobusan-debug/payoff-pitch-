from datetime import datetime, timezone
from pathlib import Path

from mlb_engine.audit.lineups import (
    LineupCapture,
    LineupPlayer,
    captures_from_slate,
    load_lineups,
    merge_lineups,
    save_lineups,
)
from mlb_engine.schemas import BatterSlot, Game, Hand, Pitcher, Player, Slate, TeamGameInfo, Venue


def _team(abbrev: str, ids: list[int], home: bool, sp: str | None = "Ace") -> TeamGameInfo:
    return TeamGameInfo(
        team_id=1 if home else 2,
        name=abbrev,
        abbrev=abbrev,
        is_home=home,
        probable_pitcher=Pitcher(mlbam_id=99, name=sp, throws=Hand.R) if sp else None,
        lineup=[
            BatterSlot(order=i, player=Player(mlbam_id=pid, name=f"P{pid}", bats=Hand.L))
            for i, pid in enumerate(ids, start=1)
        ],
    )


def _slate(home_ids: list[int], away_ids: list[int]) -> Slate:
    game = Game(
        game_pk=700,
        game_date=datetime(2026, 9, 21).date(),
        game_datetime_utc="2026-09-21T23:10:00Z",
        status="Scheduled",
        venue=Venue(venue_id=1, name="Park"),
        home=_team("CLE", home_ids, True),
        away=_team("MIN", away_ids, False, sp=None),
    )
    return Slate(slate_date=game.game_date, games=[game])


NINE = list(range(1, 10))


def test_only_confirmed_lineups_are_captured_with_lead_and_opposing_starter() -> None:
    now = datetime(2026, 9, 21, 20, 10, tzinfo=timezone.utc)
    caps = captures_from_slate(_slate(NINE, [1, 2, 3]), now=now)
    assert [c.side for c in caps] == ["home"]
    cap = caps[0]
    assert cap.team == "CLE" and cap.opponent == "MIN"
    assert cap.lead_hours == 3.0
    assert cap.captured_at == "2026-09-21T20:10:00+00:00"
    assert cap.starter is None  # the opposing (MIN) starter is not yet listed
    assert cap.player_ids == tuple(NINE)
    assert caps == captures_from_slate(_slate(NINE, [1, 2, 3]), now=now)


def _cap(ids: list[int], at: str) -> LineupCapture:
    return LineupCapture(
        game_pk=700,
        side="home",
        team="CLE",
        opponent="MIN",
        first_pitch_utc="2026-09-21T23:10:00Z",
        captured_at=at,
        lead_hours=None,
        starter="Ace",
        starter_id=99,
        starter_throws="R",
        players=[LineupPlayer(order=i, mlbam_id=p, name=f"P{p}") for i, p in enumerate(ids, 1)],
    )


def test_the_first_sighting_stays_and_a_scratch_becomes_a_revision() -> None:
    first = _cap(NINE, "2026-09-21T15:00:00+00:00")
    same_later = _cap(NINE, "2026-09-21T18:00:00+00:00")
    scratched = _cap([*NINE[:8], 10], "2026-09-21T22:30:00+00:00")

    merged = merge_lineups({}, [first])
    merged = merge_lineups(merged, [same_later])
    assert merged[first.key].captured_at == first.captured_at
    assert merged[first.key].revisions == []

    merged = merge_lineups(merged, [scratched])
    cap = merged[first.key]
    assert cap.captured_at == first.captured_at
    assert cap.player_ids == tuple(NINE)
    assert [r.captured_at for r in cap.revisions] == [scratched.captured_at]
    assert [p.mlbam_id for p in cap.final_players][-1] == 10


def test_captures_arriving_out_of_order_merge_to_the_same_record() -> None:
    first = _cap(NINE, "2026-09-21T15:00:00+00:00")
    scratched = _cap([*NINE[:8], 10], "2026-09-21T22:30:00+00:00")
    forward = merge_lineups(merge_lineups({}, [first]), [scratched])
    backward = merge_lineups(merge_lineups({}, [scratched]), [first])
    assert forward == backward
    assert backward[first.key].captured_at == first.captured_at


def test_lineup_files_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "lineups_2026-09-21.json"
    first = _cap(NINE, "2026-09-21T15:00:00+00:00")
    scratched = _cap([*NINE[:8], 10], "2026-09-21T22:30:00+00:00")
    merged = merge_lineups({}, [first, scratched])
    save_lineups(path, merged)
    assert load_lineups(path) == merged
    assert load_lineups(tmp_path / "missing.json") == {}
