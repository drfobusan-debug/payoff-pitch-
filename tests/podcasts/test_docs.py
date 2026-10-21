"""Doc's free-play box: game picks parsed, halves skipped, nothing filed after kickoff."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from engine_common.podcasts import docs, run
from engine_common.podcasts.extract import PICKS_DIR, Extraction
from engine_common.podcasts.shows import CFB, DOCS_SHOW, NFL


def _page(head: str, pick: str, start: str, expert: str, analysis: str) -> str:
    return f"""
<div class="featured-pick"><div class="white-transparent">
<span class="no-mar white-transparent-dougan">{head}</span></div>
<table class="table table-striped">
<tr><td><span class="strong">Pick:</span></td><td class="txtu">{pick}</td></tr>
<tr><td><span class="strong">Start:</span></td><td>{start}</td></tr>
<tr><td><span class="strong">Expert:</span></td>
<td><a href="//www.docsports.com/cappers.html?cap_id=36" class="site-color strong">{expert}</a></td></tr>
<tr><td colspan="2" align="center"><span class="strong">Analysis:</span></td></tr>
<tr><td colspan="2"><span class="analysis-content">{analysis}</span></td></tr>
</table></div>
<div class="featured-pick"><div class="white-transparent">
<span class="no-mar white-transparent-dougan">**7-3 +$1,820 NFL RUN**</span></div>
<table class="table table-striped"><tr><td>Join today</td></tr></table></div>
"""


SPREAD = _page(
    "MICHIGAN STATE @ UCLA",
    "MICHIGAN STATE +14",
    "October 24, 2026 @ 3:30 PM",
    "Scott Rickenbach",
    "Take #387 Michigan State Spartans +14 over UCLA Bruins. The Big Ten favourite is laying too much.",
)
HALF = _page(
    "NFL: Patriots at Bears (1H)",
    "Patriots +2.5 (1H)",
    "October 22, 2026 @ 8:15 PM",
    "Scott Rickenbach",
    "Take FIRST HALF New England Patriots +2.5 at Chicago",
)
TOTAL = _page(
    "NFL: Packers at Cardinals",
    "Over 44.5 (-110)",
    "October 25, 2026 @ 4:05 PM",
    "Scott Rickenbach",
    "Both offences are healthy.",
)
READ_AT = datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)


def test_spread_play_parsed_with_league_from_the_write_up() -> None:
    (play,) = docs.plays_in(SPREAD, 36)
    assert (play.expert, play.pick) == ("Scott Rickenbach", "MICHIGAN STATE +14")
    assert play.start == datetime(2026, 10, 24, 19, 30, tzinfo=timezone.utc)
    pick = docs.pick_of(play, READ_AT)
    assert pick is not None
    assert (pick.league, pick.market, pick.team, pick.opponent, pick.line, pick.price) == (
        CFB,
        "game_ats",
        "MICHIGAN STATE",
        "UCLA",
        14.0,
        None,
    )
    assert pick.show == DOCS_SHOW and pick.host == "Scott Rickenbach" and pick.kind == "official"
    assert pick.published == READ_AT.isoformat()


def test_total_with_price_and_league_from_the_head() -> None:
    (play,) = docs.plays_in(TOTAL, 36)
    pick = docs.pick_of(play, READ_AT)
    assert pick is not None
    assert (pick.league, pick.market, pick.side, pick.line, pick.price) == (
        NFL,
        "game_total",
        "over",
        44.5,
        -110.0,
    )
    assert (pick.team, pick.opponent) == ("Packers", "Cardinals")


def test_first_half_and_started_plays_are_dropped() -> None:
    (play,) = docs.plays_in(HALF, 36)
    assert docs.parse_play(play) is None
    now = datetime.now(timezone.utc)
    for head, pick in (
        ("NHL: Rangers at Devils (1P)", "Over 1.5"),
        ("NBA: Heat at Knicks 1st Quarter", "Over 54.5"),
    ):
        assert docs.parse_play(docs.Play(36, "x", head, pick, now, "")) is None
    ml = docs.parse_play(docs.Play(36, "x", "MLB: Phillies at Mets", "Phillies +140", now, ""))
    assert ml is not None and (ml.market, ml.line, ml.price) == ("game_ml", None, 140.0)
    (spread,) = docs.plays_in(SPREAD, 36)
    assert docs.pick_of(spread, datetime(2026, 10, 24, 20, 0, tzinfo=timezone.utc)) is None


def test_read_docs_files_once_and_keeps_the_first_release_time(tmp_path: Path) -> None:
    kept = docs.read_docs(tmp_path, READ_AT, pages={36: SPREAD})
    assert len(kept) == 1
    (path,) = (tmp_path / PICKS_DIR).glob(f"{DOCS_SHOW}__*.json")
    later = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)
    assert docs.read_docs(tmp_path, later, pages={36: SPREAD}) == []
    (pick,) = Extraction.from_json(path.read_text()).picks
    assert pick.published == READ_AT.isoformat()
    assert docs.read_docs(tmp_path, READ_AT, [NFL], pages={36: SPREAD + TOTAL}) != []
    assert len(list((tmp_path / PICKS_DIR).glob(f"{DOCS_SHOW}__*.json"))) == 2


def test_read_feeds_counts_docs_picks_and_survives_the_site_down(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    with patch.object(
        run, "read_docs", lambda root, at, lg: docs.read_docs(root, READ_AT, lg, pages={36: SPREAD})
    ):
        rep = run.read_feeds(tmp_path, now, now, shows=[], api_key="", log=False)
    assert rep.by_league[CFB] == 1

    def down(root: Path, at: datetime, lg: object) -> list[Extraction]:
        raise OSError("dns")

    with patch.object(run, "read_docs", down):
        rep = run.read_feeds(tmp_path, now, now, shows=[], api_key="", log=False)
    assert any("Doc's" in f for f in rep.failed)
