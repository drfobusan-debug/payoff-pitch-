"""PDF card: form/streak maths, league ranks, rendering, and email attachment."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from nhl_engine import email, report
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.config import Config, Credentials, Delivery
from nhl_engine.data.availability import Availability, append
from nhl_engine.features.props import GoalieProjection, SkaterProjection
from nhl_engine.features.starters import Starter
from nhl_engine.pipeline import GameCard, PropContext, SlateCard, TeamInputs
from tests.nhl.test_phase2 import LEAGUE, STRONG, WEAK, _inputs, _sim

SLATE = Date(2026, 10, 10)


def _games(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    recs = []
    for i, (d, ha, gf, ga) in enumerate(rows):
        for sit in ("all", "5on5"):
            recs.append(
                {
                    "gameId": i,
                    "gameDate": Date.fromisoformat(d),
                    "season": 2026,
                    "situation": sit,
                    "home_or_away": ha,
                    "goalsFor": gf,
                    "goalsAgainst": ga,
                }
            )
    return pd.DataFrame(recs)


def test_team_form_streak_and_splits():
    mp = MagicMock()
    mp.team_games.return_value = _games(
        [
            ("2026-10-01", "HOME", 4, 1),
            ("2026-10-02", "AWAY", 2, 3),
            ("2026-10-04", "HOME", 3, 3),  # goal tie -> shootout, not a W/L
            ("2026-10-05", "AWAY", 5, 2),
            ("2026-10-07", "HOME", 2, 1),
            ("2026-10-08", "AWAY", 1, 0),
            ("2026-10-09", "HOME", 6, 2),
            ("2026-10-10", "HOME", 0, 9),  # slate day -> excluded
        ]
    )
    f = report.team_form(mp, "X", SLATE, season=2026)
    assert f.games == 7 and f.record == "5-1"  # situation rows not double counted
    assert f.streak == "W4"
    assert f.last5 == "4-0 (1 SO)"
    assert f.last5_home == "3-0 (1 SO)" and f.last5_away == "2-1"
    assert f.gf5 == 3.4 and f.ga5 == 1.6


def test_team_form_empty_and_losing_streak():
    mp = MagicMock()
    mp.team_games.return_value = _games([])
    f = report.team_form(mp, "X", SLATE, season=2026)
    assert f.record == "0 GP" and f.streak == "--" and f.last5 == "--"
    mp.team_games.return_value = _games(
        [("2026-10-01", "HOME", 1, 2), ("2026-10-03", "AWAY", 0, 1)]
    )
    f = report.team_form(mp, "X", SLATE, season=2026)
    assert f.streak == "L2" and f.last5_home == "0-1" and f.last5_away == "0-1"


def test_league_ranks_direction():
    ranks = report.league_ranks({"S": STRONG, "W": WEAK, "M": LEAGUE})
    assert ranks["S"] == {"xgf60_5v5": 1, "xga60_5v5": 1, "pp_xgf60": 1, "pk_xga60": 1}
    assert ranks["W"]["xgf60_5v5"] == 3 and ranks["W"]["xga60_5v5"] == 3
    assert ranks["M"]["pk_xga60"] == 2


def _row(market: str, side: str, edge: float, *, buy: bool, entity: str = "") -> LedgerRow:
    return LedgerRow(
        SLATE.isoformat(),
        "A @ H",
        "H",
        "A",
        market,
        side,
        entity,
        6.5,
        "incl_ot_so",
        "dk",
        -110,
        3,
        0.5,
        0.5 + edge,
        0.0,
        edge,
        edge * 1.9,
        "Strong" if buy else "Lean",
        [] if buy else ["probation"],
        buy,
    )


def _card(tmp_path: Path) -> tuple[SlateCard, report.ReportContext]:
    home = _inputs("H", STRONG, "confirmed")
    away = TeamInputs(
        "A",
        dict(WEAK),
        {k: 0.5 for k in WEAK},
        0,
        "team_rate",
        Starter("A", 2, "A G", "expected", "rotowire", None),
    )
    sim = _sim(STRONG, WEAK, seed=1)
    from nhl_engine.models import markets

    game = GameCard("A @ H", "H", "A", "ev1", home, away, MagicMock(), sim, markets.summary(sim))
    game.rows = [
        _row("game_ml", "H", 0.07, buy=True),
        _row("game_total", "over", 0.03, buy=False),
        _row("sk_sog", "over", 0.12, buy=False, entity="Star Winger"),
    ]
    props = PropContext()
    props.skaters[("H", "star winger")] = SkaterProjection(
        9, "Star Winger", "H", "F", 19 * 60, 10.0, 1.2, 1.5, 2.7, 0.4, 5, 82
    )
    props.skaters[("H", "fourth liner")] = SkaterProjection(
        10, "Fourth Liner", "H", "F", 9 * 60, 5.0, 0.3, 0.3, 0.6, 0.0, 5, 82
    )
    props.goalies[("H", "h g")] = GoalieProjection(1, "H G", "H", 0.912, 300, 5)
    card = SlateCard(SLATE, 2026, "t", "pdf", "v1", [game], ["X @ Y: no quotes"], props=props)

    append(
        tmp_path,
        [
            Availability(
                SLATE.isoformat(),
                "A",
                77,
                "Hurt Guy",
                "out",
                "rotowire",
                "",
                "2026-10-10T10:00",
                role="D",
            )
        ],
    )
    append(
        tmp_path,
        [
            Availability(
                SLATE.isoformat(), "A", 78, "Back Guy", "out", "rotowire", "", "2026-10-10T09:00"
            )
        ],
    )
    append(
        tmp_path,
        [
            Availability(
                SLATE.isoformat(), "A", 78, "Back Guy", "in", "rotowire", "", "2026-10-10T11:00"
            )
        ],
    )

    mp = MagicMock()
    mp.team_games.return_value = _games(
        [("2026-10-08", "HOME", 3, 2), ("2026-10-09", "AWAY", 1, 4)]
    )
    ctx = report.build_context(
        card, mp=mp, data_dir=tmp_path, all_rates={"H": STRONG, "A": WEAK, "M": LEAGUE}
    )
    return card, ctx


def test_build_context_and_render(tmp_path: Path):
    card, ctx = _card(tmp_path)
    assert ctx.n_ranked == 3 and ctx.ranks["H"]["xgf60_5v5"] == 1
    assert ctx.out["A"] == ["Hurt Guy (out, D)"] and ctx.out["H"] == []  # Back Guy cleared
    assert [p.name for p in ctx.skaters["H"]] == ["Star Winger", "Fourth Liner"]
    assert ctx.goalies["H"] is not None and ctx.goalies["H"].name == "H G"
    assert ctx.form["H"].record == "1-1" and ctx.form["H"].streak == "L1"

    html = report.render_html(card, ctx)
    for needle in (
        "Tonight's buys",
        "game_ml",
        "H G",
        "Hurt Guy (out, D)",
        "Star Winger (F, 19 min, 0.85 pts/g, 3.2 SOG/g)",
        "H G (G, .912 SV% on 300 shots)",
        "#1/3",
        "#3/3",
        "L5 1-1",
        "home 1-0",
        "away 0-1",
        "L1",
        "research only",
        "X @ Y: no quotes",
        "A G",
        "expected",
    ):
        assert needle in html, needle
    assert "Back Guy" not in html and "weather" not in html.lower()


def test_write_pdf_and_email_attaches_it(tmp_path: Path):
    card, ctx = _card(tmp_path)
    path = report.write_pdf(card, ctx, report.pdf_path(tmp_path / "out", SLATE, "pdf"))
    assert path.name == "card_2026-10-10_pdf.pdf" and path.read_bytes()[:5] == b"%PDF-"

    txt = tmp_path / "out" / "card.txt"
    txt.write_text("card")
    cfg = Config(
        creds=Credentials(gmail_user="u@x", gmail_app_password="p"),
        delivery=Delivery(email_to="to@x"),
    )
    with patch("nhl_engine.email.smtplib.SMTP_SSL") as smtp:
        email.send_card(cfg, card, {"txt": txt, "pdf": path})
    msg = smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
    atts = {a.get_filename(): a.get_content_type() for a in msg.iter_attachments()}
    assert atts == {"card.txt": "text/plain", "card_2026-10-10_pdf.pdf": "application/pdf"}
