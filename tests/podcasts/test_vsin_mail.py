"""VSiN's picks email: game picks parsed as listed, props/parlays left out, stored per source."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engine_common.podcasts import vsin_mail
from engine_common.podcasts.extract import PICKS_DIR, Extraction
from engine_common.podcasts.shows import CIRCA_SHOW, NFL, NHL, VSIN_PRO_SHOW

BODY = """
Sunday picks at a glance
Best Bets Columns
VSiN Pro Picks by Game Last 24 hours
Last Updated on 10/18/2026 at 04:25:21 AM Eastern
NFL - Sun, Oct 18 @ 9:30 AM ET - Houston at Jacksonville
Aaron Moore : Marvin Harrison JR (Texans) OVER 10.5 [ Receiving Yards ] (-118) [ 0.5 units ]
Gill Alexander : Jaguars (-2.5) (-119) vs Texans
Dustin Swedelson : Texans (+3) (-120) at Jaguars
Jensen Lewis : Texans at Jaguars - UNDER (43.5) (-145) [ 2.5 units ]
John Hansen : Trevor Lawrence (Jaguars) [YES] (+400) Anytime TD
VSiN BOLT : Xavier Hutchinson (Texans) OVER 14.5 [ Receiving Yards ] (-108) | 20.4% Edge

NFL - Sun, Oct 18 @ 1:00 PM ET - New Orleans at NY Giants
Wes Reynolds : Saints (PK) (-110) at Giants
Wes Reynolds : Money Line - Saints (-112) at Giants

NHL - Sun, Oct 18 @ 8:07 PM ET - Carolina at Calgary
Jensen Lewis : Money Line - Hurricanes (-208) at Flames [ 2.5 units ]

NFL - Sun, Nov 1 @ 1:00 PM ET - Arizona at Dallas
Mitch Moss : Cardinals at Cowboys - OVER (49.5) (-110)

Tennis / UFC / Soccer
EPL - Sun, Oct 18 @ 9:00 AM - Bournemouth vs Sunderland
Wes Reynolds : EPL - Bournemouth vs. Sunderland Under 2.5 Goals (-112)

VSiN Pro Picks by OTHER
Last Updated on 10/18/2026 at 04:25:21 AM Eastern
NFL
Sun, Oct 18 : Wes Reynolds : 2-Leg 6-Point Teaser (-110) - [ Texans +8.5 ] / [ Buccaneers +8.5 ]

Circa Football Invitational - BY GAME
Last Updated on 10/18/2026 at 04:31:01 AM Eastern
NFL - Sun, Oct 18 at 9:30 AM ET - Texans at Jaguars
Jeff Petch : Texans (+2.5) at Jaguars
Jason Jarvis : Texans (+2.5) at Jaguars BEST BET
Matt Ste. Marie : Texans at Jaguars - OVER (42.5) BEST BET

NFL Week 6: Best Bets & Circa Contest Picks
View Article
"""

MAIL = vsin_mail.Mail("4242", "2026-10-18T11:30:00+00:00", "Sunday picks at a glance", BODY)


def test_game_picks_parsed_props_and_parlays_skipped() -> None:
    picks = vsin_mail.picks_in(MAIL)
    pro = {(p.host, p.market): p for p in picks[VSIN_PRO_SHOW]}
    assert set(pro) == {
        ("Gill Alexander", "game_ats"),
        ("Dustin Swedelson", "game_ats"),
        ("Jensen Lewis", "game_total"),
        ("Wes Reynolds", "game_ats"),
        ("Wes Reynolds", "game_ml"),
        ("Jensen Lewis", "game_ml"),
    }
    jax = pro[("Gill Alexander", "game_ats")]
    assert (jax.team, jax.opponent, jax.line, jax.price, jax.league) == (
        "Jaguars",
        "Houston",
        -2.5,
        -119.0,
        NFL,
    )
    assert jax.kind == "official" and jax.units is None
    tot = pro[("Jensen Lewis", "game_total")]
    assert (tot.side, tot.line, tot.price, tot.units) == ("under", 43.5, -145.0, 2.5)
    pk = pro[("Wes Reynolds", "game_ats")]
    assert pk.line == 0.0 and pk.price == -110.0
    car = pro[("Jensen Lewis", "game_ml")]
    assert (car.team, car.league, car.price, car.units) == ("Hurricanes", NHL, -208.0, 2.5)
    # Nov 1 is another slate; EPL and the teaser section are not game picks.
    assert all("Cardinals" not in (p.team or "") for p in picks[VSIN_PRO_SHOW])


def test_circa_best_bet_is_official_rest_are_leans() -> None:
    circa = {p.host: p for p in vsin_mail.picks_in(MAIL)[CIRCA_SHOW]}
    assert set(circa) == {"Jeff Petch", "Jason Jarvis", "Matt Ste. Marie"}
    assert circa["Jeff Petch"].kind == "lean" and circa["Jeff Petch"].price is None
    assert circa["Jason Jarvis"].kind == "official" and circa["Jason Jarvis"].line == 2.5
    assert circa["Matt Ste. Marie"].market == "game_total"
    assert circa["Matt Ste. Marie"].side == "over"


def test_read_mail_stores_one_file_per_source(tmp_path: Path) -> None:
    kept = vsin_mail.read_mail(
        tmp_path,
        datetime(2026, 10, 17, tzinfo=timezone.utc),
        datetime(2026, 10, 19, tzinfo=timezone.utc),
        mails=[MAIL],
    )
    files = sorted(p.name for p in (tmp_path / PICKS_DIR).glob("*.json"))
    assert files == ["circa_fi__4242.json", "vsin_pro__4242.json"]
    assert {e.leagues[0] for e in kept} >= {NFL}
    back = Extraction.from_json((tmp_path / PICKS_DIR / "vsin_pro__4242.json").read_text())
    assert sorted(back.leagues) == [NFL, NHL]
    # An NFL-only pass keeps only NFL picks.
    nfl_only = vsin_mail.read_mail(
        tmp_path,
        datetime(2026, 10, 17, tzinfo=timezone.utc),
        datetime(2026, 10, 19, tzinfo=timezone.utc),
        [NFL],
        mails=[MAIL],
    )
    assert all(p.league == NFL for e in nfl_only for p in e.picks)
