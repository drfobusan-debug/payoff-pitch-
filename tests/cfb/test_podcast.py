"""CFB podcast picks: feed selection, transcript cache, checked extraction, game
placement, grading, the separate ledger, the PDF blocks, and that none of it
reaches a price, a probability or a tier."""

from __future__ import annotations

import copy
import json
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cfb_engine.audit.grade import LOSS, PUSH, WIN, build_result_index
from cfb_engine.audit.snapshot import SideQuote
from cfb_engine.data.cfbd import GameResult
from cfb_engine.market.tiers import Tier
from cfb_engine.output.card import build_article
from cfb_engine.output.podcast import game_block, podcast_view, records_block
from cfb_engine.podcast import episodes as eps
from cfb_engine.podcast.episodes import Episode, ensure_transcript, select, transcript_path
from cfb_engine.podcast.extract import Extraction, Pick, check, extract, number_said, quote_found
from cfb_engine.podcast.picks import (
    by_game,
    grade_row,
    load_ledger,
    place,
    records,
    same_school,
    slate_picks,
    update_ledger,
)
from cfb_engine.podcast.shows import BY_KEY
from cfb_engine.recommendations import Recommendation
from cfb_engine.state import fill_in_file
from engine_common.podcasts.feed import FeedItem, parse_items
from engine_common.podcasts.transcript import Line, write_transcript

DAY = Date(2026, 10, 3)
PUB = "2026-10-01T15:00:00+00:00"
SHOW = BY_KEY["vsin_cfb"]

LINES = [
    Line(130, "And as we record this right now, Ohio State is a 14 and a half point favorite"),
    Line(134, "with a total of 45 and a half."),
    Line(136, "I've already bet this plus 14 and a half."),
    Line(200, "I'm going to play the Hawkeyes here again, I bet it plus 14 and a half."),
    Line(731, "Contemplated giving out Alabama team total over 33 and a half here."),
    Line(900, "I'll take the under 45 and a half, it's a lean for me."),
]


def _item(**kw: str) -> FeedItem:
    base = dict(
        guid="g1",
        title="Week 5 Betting Preview & Best Bets",
        published=PUB,
        audio_url="https://x/ep.mp3",
        duration="01:00:00",
    )
    base.update(kw)
    return FeedItem(**base)  # type: ignore[arg-type]


EP = Episode(SHOW, _item())


def _game(
    home: str, away: str, kick: str = "2026-10-03T19:30:00Z", gid: str = "g1"
) -> list[Recommendation]:
    m = f"{away} @ {home}"
    common = dict(
        game_date=DAY,
        game_id=gid,
        matchup=m,
        home_abbrev=home,
        away_abbrev=away,
        kickoff_utc=kick,
        tier=Tier.PASS,
        model_prob=0.5,
    )
    return [
        Recommendation(
            market="game_ml",
            selection=f"{home} ML",
            team_side="home",
            side="win",
            market_american=400,
            **common,
        ),  # type: ignore[arg-type]
        Recommendation(
            market="game_ml",
            selection=f"{away} ML",
            team_side="away",
            side="win",
            market_american=-550,
            **common,
        ),  # type: ignore[arg-type]
        Recommendation(
            market="game_ats",
            selection=f"{home} +14.5",
            line=14.5,
            team_side="home",
            side="cover",
            market_american=-110,
            **common,
        ),  # type: ignore[arg-type]
        Recommendation(
            market="game_ats",
            selection=f"{away} -14.5",
            line=-14.5,
            team_side="away",
            side="cover",
            market_american=-110,
            **common,
        ),  # type: ignore[arg-type]
        Recommendation(
            market="game_total",
            selection="Over 45.5",
            line=45.5,
            side="over",
            market_american=-110,
            **common,
        ),  # type: ignore[arg-type]
        Recommendation(
            market="game_total",
            selection="Under 45.5",
            line=45.5,
            side="under",
            market_american=-110,
            **common,
        ),  # type: ignore[arg-type]
    ]


def _pick(**kw: object) -> Pick:
    base: dict[str, object] = dict(
        pick_id="p1",
        show=SHOW.key,
        show_name=SHOW.name,
        host=None,
        episode_guid="g1",
        episode_title="Week 5",
        published=PUB,
        kind="official",
        market="game_ats",
        team="Iowa",
        opponent="Ohio State",
        side=None,
        line=14.5,
        price=None,
        units=None,
        seconds=136,
        quote="I've already bet this plus 14 and a half",
        reason="home dog",
        description="Iowa +14.5",
    )
    base.update(kw)
    return Pick(**base)  # type: ignore[arg-type]


def _raw(**kw: object) -> dict[str, object]:
    base: dict[str, object] = dict(
        kind="official",
        host=None,
        team="Iowa",
        opponent="Ohio State",
        market="game_ats",
        side=None,
        line=14.5,
        price=None,
        units=None,
        stamp="02:16",
        quote="I've already bet this plus 14 and a half",
        reason="home dog",
        description="Iowa +14.5",
    )
    base.update(kw)
    return base


# ------------------------------------------------------------------ feeds


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Week 5 Betting Preview</title><guid>a</guid>
<pubDate>Wed, 30 Sep 2026 15:00:00 +0000</pubDate><enclosure url="https://x/a.mp3"/></item>
<item><title>Week 4 Recap</title><guid>b</guid>
<pubDate>Mon, 28 Sep 2026 15:00:00 +0000</pubDate><enclosure url="https://x/b.mp3"/></item>
<item><title>Week 1 Preview</title><guid>c</guid>
<pubDate>Wed, 02 Sep 2026 15:00:00 +0000</pubDate><enclosure url="https://x/c.mp3"/></item>
<item><title>No audio</title><guid>d</guid><pubDate>Wed, 30 Sep 2026 15:00:00 +0000</pubDate></item>
</channel></rss>"""


def test_feed_selection_keeps_the_window_and_skips_recaps() -> None:
    items = parse_items(RSS)
    assert [i.guid for i in items] == ["a", "b", "c"]
    got = select(
        SHOW,
        items,
        datetime(2026, 9, 25, tzinfo=timezone.utc),
        datetime(2026, 10, 3, tzinfo=timezone.utc),
    )
    assert [e.item.guid for e in got] == ["a"]


def test_multi_sport_shows_read_only_cfb_titles() -> None:
    wt = BY_KEY["wagertalk"]
    assert wt.wants("College Football Week 5 Best Bets", "")
    assert not wt.wants("NFL Week 4 Picks", "Sunday's slate")
    assert BY_KEY["ftm"].wants("Follow the Money Hour 2", "")
    assert not BY_KEY["ftm"].wants("Best of Follow the Money", "")


def test_a_saved_transcript_is_reused_without_download(tmp_path: Path, monkeypatch) -> None:
    dest = transcript_path(tmp_path, EP)
    write_transcript(dest, LINES)
    monkeypatch.setattr(eps, "download", MagicMock(side_effect=AssertionError("downloaded")))
    assert ensure_transcript(EP, tmp_path) == dest
    other = Episode(SHOW, _item(guid="new"))
    assert ensure_transcript(other, tmp_path, transcribe_missing=False) is None


# ------------------------------------------------------------- extraction


def test_quote_must_be_near_its_stamp() -> None:
    assert quote_found("I've already bet this plus 14 and a half", LINES, 136)
    assert not quote_found("I've already bet this plus 14 and a half", LINES, 900)
    assert not quote_found("Clemson moneyline is my favorite bet", LINES, 136)


def test_numbers_must_have_been_said() -> None:
    assert number_said(14.5, LINES, 136)
    assert number_said(-14.5, LINES, 136)
    assert not number_said(17.0, LINES, 136)


def test_check_keeps_honest_picks_and_refuses_invented_ones() -> None:
    pick, _ = check(_raw(), EP, LINES)
    assert pick is not None and pick.line == 14.5 and pick.price is None
    bad, why = check(_raw(quote="Clemson ML is the bet of the week", stamp="02:16"), EP, LINES)
    assert bad is None and "quote" in why
    unsaid, _ = check(_raw(line=17.5, price=-115), EP, LINES)
    assert unsaid is not None and unsaid.line is None and unsaid.price is None
    total, why = check(_raw(market="game_total", side=None), EP, LINES)
    assert total is None and "over/under" in why


def test_official_and_lean_are_kept_apart() -> None:
    off, _ = check(_raw(), EP, LINES)
    lean, _ = check(
        _raw(
            kind="lean",
            market="game_total",
            side="under",
            line=45.5,
            stamp="15:00",
            quote="I'll take the under 45 and a half",
        ),
        EP,
        LINES,
    )
    assert off is not None and lean is not None
    assert (off.kind, lean.kind) == ("official", "lean")
    assert off.pick_id != lean.pick_id


def test_extraction_is_cached_per_transcript(tmp_path: Path) -> None:
    calls = {"n": 0}

    def post(url: str, **kw: object) -> MagicMock:
        calls["n"] += 1
        r = MagicMock()
        r.raise_for_status = lambda: None
        content = json.dumps({"picks": [_raw()] if calls["n"] == 1 else []})
        r.json = lambda: {"choices": [{"message": {"content": content}}]}
        return r

    sess = MagicMock()
    sess.post = post
    got = extract(EP, LINES, tmp_path, api_key="k", session=sess)
    n = calls["n"]
    assert [p.description for p in got.picks] == ["Iowa +14.5"]
    again = extract(EP, LINES, tmp_path, api_key="k", session=sess)
    assert calls["n"] == n and again.picks == got.picks
    assert Extraction.from_json(got.to_json()).picks == got.picks


# --------------------------------------------------------------- matching


def test_georgia_is_not_georgia_state() -> None:
    assert same_school("Georgia", "georgia")
    assert not same_school("Georgia", "Georgia State")
    assert not same_school("Georgia State", "Georgia")


def test_pick_lands_on_its_game_or_nowhere() -> None:
    games = by_game(_game("Iowa", "Ohio State") + _game("Georgia State", "Troy", gid="g2"))
    pl = place(_pick(), games)
    assert pl is not None and pl.matchup == "Ohio State @ Iowa" and pl.label == "Iowa +14.5"
    assert place(_pick(team="Georgia"), games) is None
    assert place(_pick(team="Clemson"), games) is None
    total = place(_pick(market="game_total", side="under", line=45.5), games)
    assert total is not None and total.label == "Under 45.5"


def test_a_team_on_two_games_needs_its_opponent() -> None:
    games = by_game(
        _game("Iowa", "Ohio State")
        + _game("Iowa", "Nebraska", kick="2026-10-04T00:00:00Z", gid="g2")
    )
    assert place(_pick(opponent=None), games) is None
    pl = place(_pick(opponent="Nebraska"), games)
    assert pl is not None and pl.matchup == "Nebraska @ Iowa"


def test_the_card_sets_the_sign_of_a_spread() -> None:
    games = by_game(_game("Iowa", "Ohio State"))
    pl = place(_pick(line=-14.5), games)
    assert pl is not None and pl.line == 14.5


def test_futures_and_other_slates_are_unmatched_and_repeats_count_once() -> None:
    recs = _game("Iowa", "Ohio State")
    later = _pick(pick_id="p2", published="2026-10-02T15:00:00+00:00", seconds=200)
    lean = _pick(pick_id="p3", kind="lean")
    future = _pick(
        pick_id="p4", team="Indiana", market="other", description="Indiana to win the Big Ten"
    )
    ex = Extraction("g1", "m", "v", "sha", [_pick(), later, lean, future])
    sp = slate_picks([ex], recs, DAY)
    assert [pl.pick.pick_id for pl in sp.placed] == ["p1"]
    assert [p.pick_id for p in sp.unmatched] == ["p4"]


# ---------------------------------------------------------------- grading


def _index(home: int, away: int):
    return build_result_index([GameResult("Iowa", "Ohio State", home, away)])


@pytest.mark.parametrize(
    ("pick", "home", "away", "want"),
    [
        (dict(), 20, 31, WIN),  # Iowa +14.5 loses by 11
        (dict(), 10, 31, LOSS),
        (dict(market="game_ml", line=None, price=400.0), 24, 21, WIN),
        (dict(market="game_total", side="over", line=45.5), 24, 21, LOSS),
        (dict(line=14.0), 10, 24, PUSH),
    ],
)
def test_grading_reuses_the_engine_rules(pick: dict, home: int, away: int, want: str) -> None:
    games = by_game(_game("Iowa", "Ohio State"))
    pl = place(_pick(**pick), games)
    assert pl is not None
    row = grade_row(pl, _index(home, away), {}, {})
    assert row["result"] == want


def test_units_only_with_a_real_price() -> None:
    games = by_game(_game("Iowa", "Ohio State"))
    pl = place(_pick(), games)
    assert pl is not None
    bare = grade_row(pl, _index(20, 31), {}, {})
    assert bare["units"] == "" and bare["price"] == ""
    key = "Ohio State @ Iowa|game_ats|Iowa"
    board = {key: SideQuote(american=-105, no_vig_prob=0.5, line=14.5)}
    off_number = {key: SideQuote(american=-105, no_vig_prob=0.5, line=13.5)}
    assert grade_row(pl, _index(20, 31), off_number, {})["units"] == ""
    priced = grade_row(pl, _index(20, 31), board, {key: SideQuote(-110, 0.5, 13.5)})
    assert priced["price_source"] == "board" and float(priced["units"]) == pytest.approx(
        100 / 105, abs=1e-3
    )
    assert priced["close_line"] == "13.5" and float(priced["clv_pts"]) > 0
    stated = place(_pick(price=-120.0, units=2.0), games)
    assert stated is not None
    row = grade_row(stated, _index(10, 31), board, {})
    assert row["price_source"] == "stated" and float(row["units"]) == -2.0


def test_ledger_is_its_own_file_and_regrading_a_day_replaces_it(tmp_path: Path) -> None:
    path = tmp_path / "podcast_ledger.csv"
    games = by_game(_game("Iowa", "Ohio State"))
    pl = place(_pick(price=-110.0), games)
    assert pl is not None
    row = grade_row(pl, _index(20, 31), {}, {})
    update_ledger(path, [row], DAY)
    update_ledger(path, [row], DAY)
    led = load_ledger(path)
    assert len(led) == 1 and led[0]["result"] == WIN
    rec = records(led)
    off, lean = rec.shows[0]
    assert off.wlp == "1-0" and off.underpowered and lean.decided == 0


def test_a_show_counts_one_opinion_however_many_hosts_share_it(tmp_path: Path) -> None:
    games = by_game(_game("Iowa", "Ohio State"))
    rows = []
    for i, host in enumerate(("Tim Murray", "Wes Reynolds")):
        pl = place(_pick(pick_id=f"h{i}", host=host, price=-110.0), games)
        assert pl is not None
        rows.append(grade_row(pl, _index(20, 31), {}, {}))
    rec = records(rows)
    assert rec.shows[0][0].wlp == "1-0"
    assert [h[0].wlp for h in rec.hosts] == ["1-0", "1-0"]


# -------------------------------------------------------------------- PDF


def _view(tmp_path: Path, recs: list[Recommendation]):
    ex = Extraction("g1", "m", "v", "sha", [_pick(host="Tim Murray")])
    (tmp_path / "podcast_picks").mkdir()
    (tmp_path / "podcast_picks" / "x.json").write_text(ex.to_json())
    return podcast_view(tmp_path, tmp_path / "podcast_ledger.csv", recs, DAY)


def test_pdf_prints_picks_under_the_game_and_records_at_the_back(tmp_path: Path) -> None:
    recs = _game("Iowa", "Ohio State")
    view = _view(tmp_path, recs)
    block = game_block(view, "Ohio State @ Iowa")
    assert (
        "<b>Tim Murray · VSiN College Football Betting Podcast</b> (no graded picks yet)." in block
    )
    assert "<b>Iowa +14.5</b>" in block and "[02:16]" not in block
    assert game_block(view, "Troy @ Georgia State") == ""
    back = records_block(view, [recs])
    assert "Podcast records" in back and "Consensus" in back
    assert podcast_view(tmp_path / "none", tmp_path / "x.csv", recs, DAY) is None


def test_podcasts_never_touch_the_card(tmp_path: Path) -> None:
    recs = _game("Iowa", "Ohio State")
    before = copy.deepcopy(recs)
    plain, narr_plain = build_article(DAY, recs)
    with_pod, narr_pod = build_article(DAY, recs, None, _view(tmp_path, recs))
    assert recs == before
    assert narr_plain == narr_pod
    assert "Podcast picks" not in plain and "Podcast picks" in with_pod


def test_state_fills_in_extractions_without_overwriting(tmp_path: Path) -> None:
    remote, local = tmp_path / "r.json", tmp_path / "l" / "r.json"
    remote.write_text("remote")
    assert fill_in_file(remote, local) and local.read_text() == "remote"
    remote.write_text("changed")
    assert not fill_in_file(remote, local) and local.read_text() == "remote"
