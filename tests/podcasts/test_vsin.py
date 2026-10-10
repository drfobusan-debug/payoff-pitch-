"""VSiN's written best bets: parsed as written, placed on the card, graded on their own."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine_common.podcasts import picks as core
from engine_common.podcasts import vsin
from engine_common.podcasts.render import PodcastView, game_block
from engine_common.podcasts.shows import CFB, MLB, NHL
from mlb_engine.data import vsin_bets
from mlb_engine.data.results import GameResult
from mlb_engine.output.card import GameCard, render_html
from mlb_engine.schemas import Game, Slate, TeamGameInfo, Venue
from nhl_engine.audit import vsin_picks

CFB_HTML = (
    "<h3>Indiana at Nebraska (+7.5)</h3><p>I can't pass up the points here.</p>"
    "<p>Subscribe to VSiN Pro and save 50%!</p><p>Last week's record: 5-3</p>"
    "<p><strong>College Football Best Bet:</strong> Nebraska +7.5</p>"
)
MLB_HTML = (
    "<h2>White Sox vs. Guardians (-137, 7) Prediction</h2>"
    "<p>Game 5 for all the proverbial marbles.</p><p>Pick: White Sox +114</p>"
)


def _post(content: str, league: str = CFB, pid: int = 7, published: str = "") -> vsin.Post:
    return vsin.Post(
        pid,
        published or "2026-10-10T13:00:00+00:00",
        f"https://vsin.com/post-{pid}/",
        "Tuley's Takes",
        "Dave Tuley",
        league,
        content,
    )


@pytest.mark.parametrize(
    ("text", "market", "team", "side", "line", "price", "units"),
    [
        ("Nebraska +7.5", "game_ats", "Nebraska", None, 7.5, None, None),
        ("Eagles +8 (-110 - 1.5 units)", "game_ats", "Eagles", None, 8.0, -110.0, 1.5),
        ("Nebraska ML (+250 - 0.5 units)", "game_ml", "Nebraska", None, None, 250.0, 0.5),
        (
            "Seattle/Detroit Over 6 (-110) - 2 units",
            "game_total",
            "Seattle",
            "over",
            6.0,
            -110.0,
            2.0,
        ),
        ("Falcons -3 (-118)", "game_ats", "Falcons", None, -3.0, -118.0, None),
    ],
)
def test_parse_bet_reads_the_number_as_written(
    text: str,
    market: str,
    team: str,
    side: str | None,
    line: float | None,
    price: float | None,
    units: float | None,
) -> None:
    got = vsin.parse_bet(text, None)
    assert got is not None
    assert (got.market, got.team, got.side, got.line, got.price, got.units) == (
        market,
        team,
        side,
        line,
        price,
        units,
    )
    assert got.description == text


def test_parse_bet_takes_the_matchup_from_the_heading_and_invents_no_price() -> None:
    got = vsin.parse_bet("White Sox +114", ("White Sox", "Guardians"))
    assert got is not None
    assert (got.market, got.team, got.opponent, got.price) == (
        "game_ml",
        "White Sox",
        "Guardians",
        114.0,
    )
    spread = vsin.parse_bet("Nebraska +7.5", ("Indiana", "Nebraska"))
    assert spread is not None and spread.price is None and spread.opponent == "Indiana"


def test_a_place_is_not_an_opponent() -> None:
    got = vsin.parse_bet("Indiana -7.5 in Lincoln", None)
    assert got is not None and got.team == "Indiana" and got.opponent is None


@pytest.mark.parametrize(
    "text",
    [
        "Bobby McMann Over 2.5 shots on goal (+115) - 1 unit",
        "Not sure about this one, I'll take the +2.5",
    ],
)
def test_props_and_vague_lines_are_kept_but_not_game_bets(text: str) -> None:
    got = vsin.parse_bet(text, None)
    assert got is not None and got.market == "other" and got.team is None


def test_bets_in_skips_promotions_and_records() -> None:
    got = list(vsin.bets_in(CFB_HTML + MLB_HTML))
    assert got == [
        ("Nebraska +7.5", ("Indiana", "Nebraska")),
        ("White Sox +114", ("White Sox", "Guardians")),
    ]


def test_picks_in_keeps_writer_article_and_link_and_dedupes() -> None:
    post = _post(CFB_HTML + CFB_HTML)
    got = vsin.picks_in(post)
    assert len(got) == 1
    p = got[0]
    assert (p.show, p.show_name, p.host, p.league) == (
        vsin.SHOW_KEY,
        vsin.SHOW_NAME,
        "Dave Tuley",
        CFB,
    )
    assert (p.episode_guid, p.episode_title, p.reason) == ("vsin-7", "Tuley's Takes", post.link)
    assert p.kind == "official" and p.published == post.published


class _Resp:
    def __init__(self, data: object, status: int = 200) -> None:
        self._data = data
        self.status_code = status

    def json(self) -> object:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(self.status_code)


class _Session:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.params: list[dict[str, object]] = []

    def get(self, url: str, params: dict[str, object], headers: object, timeout: int) -> _Resp:
        if "/users/" in url:
            return _Resp({"name": "Adam Burke"})
        self.params.append(params)
        return _Resp(self.rows)


def _row(pid: int, when: str) -> dict[str, object]:
    return {
        "id": pid,
        "date_gmt": when,
        "link": f"https://vsin.com/p{pid}/",
        "title": {"rendered": "White Sox vs. Guardians Game 5 Prediction"},
        "author": 3,
        "content": {"rendered": MLB_HTML},
    }


def test_fetch_posts_filters_by_category_and_window(tmp_path: Path) -> None:
    session = _Session([_row(1, "2026-10-10T05:27:05"), _row(2, "2026-10-01T05:00:00")])
    since = datetime(2026, 10, 9, 4, tzinfo=timezone.utc)
    until = datetime(2026, 10, 11, 4, tzinfo=timezone.utc)
    posts = vsin.fetch_posts(MLB, since, until, session=session)  # type: ignore[arg-type]
    assert [p.id for p in posts] == [1]
    assert session.params[0]["categories"] == vsin.CATEGORY[MLB]
    assert posts[0].author == "Adam Burke" and posts[0].link == "https://vsin.com/p1/"
    got = vsin.read_articles(tmp_path, since, until, [MLB], session=session)  # type: ignore[arg-type]
    assert len(got) == 1 and got[0].picks[0].description == "White Sox +114"
    assert vsin.extraction_path(tmp_path, posts[0]).exists()


def _slate(final: bool = True) -> tuple[Slate, dict[int, GameResult]]:
    venue = Venue(venue_id=1, name="Progressive Field")
    game = Game(
        game_pk=99,
        game_date=date(2026, 10, 10),
        game_datetime_utc="2026-10-10T23:08:00Z",
        status="Final",
        venue=venue,
        home=TeamGameInfo(team_id=114, name="Cleveland Guardians", abbrev="CLE", is_home=True),
        away=TeamGameInfo(team_id=145, name="Chicago White Sox", abbrev="CWS", is_home=False),
    )
    res = GameResult(game_pk=99, final=final, home_runs=2, away_runs=4, f5_home=0, f5_away=1)
    return Slate(slate_date=date(2026, 10, 10), games=[game]), {99: res}


def test_mlb_grades_the_writer_at_his_own_price_in_its_own_ledger(tmp_path: Path) -> None:
    slate, results = _slate()
    picks = vsin.picks_in(_post(MLB_HTML, MLB, published="2026-10-10T05:27:05+00:00"))
    rows = vsin_bets.grade(date(2026, 10, 10), picks, slate, results.get)
    assert len(rows) == 1
    r = rows[0]
    assert (r["matchup"], r["result"], r["price"], r["units"]) == (
        "CWS @ CLE",
        core.WIN,
        "+114",
        "1.140",
    )
    ledger = vsin_bets.update(tmp_path, date(2026, 10, 10), rows)
    assert (tmp_path / vsin_bets.LEDGER).exists() and not (tmp_path / "ledger.csv").exists()
    assert "Dave Tuley" in vsin_bets.render_text(ledger) and "1-0" in vsin_bets.render_text(ledger)


def test_mlb_pick_is_not_placed_on_the_next_game_of_the_series() -> None:
    slate, results = _slate()
    old = vsin.picks_in(_post(MLB_HTML, MLB, published="2026-10-08T15:00:00+00:00"))
    assert vsin_bets.grade(date(2026, 10, 10), old, slate, results.get) == []


def test_mlb_unfinished_game_is_left_ungraded() -> None:
    slate, results = _slate(final=False)
    picks = vsin.picks_in(_post(MLB_HTML, MLB, published="2026-10-10T05:27:05+00:00"))
    assert vsin_bets.grade(date(2026, 10, 10), picks, slate, results.get)[0]["result"] == ""


def test_mlb_card_prints_the_bet_inside_its_game_apart_from_the_plays() -> None:
    slate, _ = _slate()
    picks = vsin.picks_in(_post(MLB_HTML, MLB, published="2026-10-10T05:27:05+00:00"))
    games, rest = vsin_bets.html_sections(date(2026, 10, 10), picks, slate, [], ["CWS @ CLE"])
    assert rest == ""
    assert "White Sox +114" in games["CWS @ CLE"] and "not a model input" in games["CWS @ CLE"]
    card = GameCard(matchup="CWS @ CLE", plays=[])
    other = GameCard(matchup="NYY @ BOS", plays=[])
    page = render_html([card, other], date(2026, 10, 10), vsin_games=games)
    at = page.index("<h2>CWS @ CLE</h2>")
    assert at < page.index("White Sox +114") < page.index("<h2>NYY @ BOS</h2>")
    assert "VSiN best bets" not in render_html([card], date(2026, 10, 10))


def test_mlb_bet_off_the_card_prints_in_the_closing_section() -> None:
    slate, _ = _slate()
    picks = vsin.picks_in(_post(MLB_HTML, MLB, published="2026-10-10T05:27:05+00:00"))
    games, rest = vsin_bets.html_sections(date(2026, 10, 10), picks, slate, [])
    assert games == {}
    assert "CWS @ CLE: <b>White Sox +114</b>" in rest and "100+ graded bets" in rest
    page = render_html([], date(2026, 10, 10), vsin=rest)
    assert rest in page


def test_mlb_capture_files_merge_by_pick(tmp_path: Path) -> None:
    picks = vsin.picks_in(_post(MLB_HTML, MLB))
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    vsin_bets.save(a, picks)
    vsin_bets.save(b, picks)
    assert vsin_bets.merge_files(a, b)
    assert len(json.loads(b.read_text())) == 1


def test_nhl_places_game_bets_and_leaves_props_ungraded() -> None:
    post = _post(
        "<p><strong>NHL Predictions and Best Bets:</strong> <strong><br></strong>"
        "<strong>Seattle/Detroit Over 6 (-110) – 2 units</strong><strong><br></strong>"
        "<strong>Bobby McMann Over 2.5 shots on goal (+115) – 1 unit</strong></p>",
        NHL,
    )
    articles = vsin.picks_in(post)
    games = {"SEA@DET": ("SEA", "DET"), "NYR@WSH": ("NYR", "WSH")}
    placed = [p for a in articles if (p := vsin_picks.to_pick(a, date(2026, 10, 9), games))]
    assert len(placed) == 1
    p = placed[0]
    assert (p.matchup, p.market, p.side, p.line, p.american, p.stake, p.host) == (
        "SEA@DET",
        "game_total",
        "over",
        6.0,
        -110,
        2.0,
        "Dave Tuley",
    )
    assert [a.market for a in vsin_picks.unplaced(articles)] == ["other"]


def _placed(show: str, show_name: str) -> core.Placed[None]:
    pick = vsin.picks_in(_post(CFB_HTML))[0]
    pick = core.Pick(**{**pick.__dict__, "show": show, "show_name": show_name})
    g: core.Game[None] = core.Game("IND @ NEB", "Nebraska", "Indiana", None, date(2026, 10, 10))
    return core.Placed(pick, g, "home", 7.5)


def test_vsin_prints_in_its_own_block_beside_the_podcasts() -> None:
    view = PodcastView(
        core.SlatePicks(
            [_placed(vsin.SHOW_KEY, vsin.SHOW_NAME), _placed("cfb_pod", "CFB Pod")], []
        ),
        core.records([]),
    )
    out = game_block(view, "IND @ NEB")
    assert out.count("<div class='pod'>") == 2
    assert out.index("Podcast picks") < out.index("VSiN best bets")
