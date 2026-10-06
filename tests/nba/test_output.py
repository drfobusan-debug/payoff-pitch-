import smtplib
from dataclasses import asdict, replace
from datetime import date, timedelta
from io import BytesIO

import pytest
from openpyxl import load_workbook

from nba_engine import cli
from nba_engine.audit import ledger
from nba_engine.config import load_config
from nba_engine.data.capture import QuoteRow
from nba_engine.market.board import selections
from nba_engine.output import card as card_mod
from nba_engine.output import context
from nba_engine.output import email as email_mod
from nba_engine.output.card import NO_MODEL_TEXT, PAPER_NOTE, build_card, render_html
from nba_engine.output.excel import SELECTION_HEADER, build_workbook
from nba_engine.schemas import GameResult

DAY = date(2025, 12, 10)
LATE, EARLY = "BOS @ NYK", "PHX @ DET"  # alphabetical order is the reverse of tip order
TIPS = {"late": "2025-12-11T03:00:00Z", "early": "2025-12-11T00:00:00Z"}
EVENTS = {"late": LATE, "early": EARLY}


def q(eid, market, side, line, book, am, opp, at="2025-12-10T22:00:00Z"):
    return QuoteRow(at, DAY.isoformat(), EVENTS[eid], eid, market, side, "", line, book, am, opp)


def board(at="2025-12-10T22:00:00Z"):
    out = []
    for eid in EVENTS:
        away, _, home = EVENTS[eid].partition(" @ ")
        for book, fav, dog in (
            ("draftkings", -150, 130),
            ("betmgm", -145, 125),
            ("fanduel", -155, 135),
        ):
            out += [
                q(eid, "game_ml", home, None, book, fav, dog, at),
                q(eid, "game_ml", away, None, book, dog, fav, at),
                q(eid, "game_ats", home, -3.5, book, -110, -110, at),
                q(eid, "game_ats", away, 3.5, book, -110, -110, at),
                q(eid, "game_total", "over", 221.5, book, -108, -112, at),
                q(eid, "game_total", "under", 221.5, book, -112, -108, at),
            ]
    return out


def priced(tag="t1", at="2025-12-10T22:00:00Z"):
    pre = ledger.pregame(board(at), TIPS)
    return ledger.price(
        selections(pre),
        ledger.quote_index(pre),
        tips=TIPS,
        shift={},
        versions="over_bias=v1",
        pass_tag=tag,
    )


def as_buy(r, model_prob=0.66):
    return replace(r, model_prob=model_prob, ev=0.05, tier="Strong buy", gates="")


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    monkeypatch.delenv("NBAE_PRESEASON", raising=False)
    for var in (
        "GMAIL_APP_PASSWORD",
        "GMAIL_USER",
        "EMAIL_ADDRESS",
        "NBAE_EMAIL_TO",
        "MLBE_EMAIL_TO",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli, "render_pdf", lambda html: b"%PDF-1.4 fake")
    rows = priced()
    path = ledger.pass_path(tmp_path, DAY, rows[0].priced_at, "t1")
    ledger.write_once(rows, path)
    return tmp_path, path


def run_card(*extra):
    return cli.main(["card", "--date", DAY.isoformat(), "--no-sync", "--offline", *extra])


# -- order -------------------------------------------------------------------
def test_games_in_tip_order_with_et_tip_in_each_header():
    card = build_card(priced(), day=DAY)
    assert [g.matchup for g in card.games] == [EARLY, LATE]
    page = render_html(card)
    assert page.index("Phoenix Suns at Detroit Pistons") < page.index(
        "Boston Celtics at New York Knicks"
    )
    assert "7:00 PM ET" in page and "10:00 PM ET" in page


def test_best_bets_in_tip_order_strongest_first_within_a_game():
    rows = priced()
    late_ml = next(r for r in rows if r.event_id == "late" and r.market == "game_ml")
    late_tot = next(r for r in rows if r.event_id == "late" and r.market == "game_total")
    early = next(r for r in rows if r.event_id == "early" and r.market == "game_ats")
    picks = {id(late_ml): as_buy(late_ml), id(late_tot): replace(as_buy(late_tot), ev=0.09)}
    picks[id(early)] = replace(as_buy(early), ev=0.01)
    card = build_card([picks.get(id(r), r) for r in rows], day=DAY)
    assert [(b.event_id, b.market) for b in card.buys()] == [
        ("early", "game_ats"),
        ("late", "game_total"),
        ("late", "game_ml"),
    ]


# -- card --------------------------------------------------------------------
def test_card_from_fixture_ledger_says_market_only_and_shows_gates():
    page = render_html(build_card(priced(), day=DAY))
    assert NO_MODEL_TEXT in page
    assert PAPER_NOTE in page
    assert "no_model" in page
    assert "DK -150" in page and "MGM" in page  # execution prices beside the fair
    assert "Moneyline" in page and "Spread" in page and "Total" in page


def test_kelly_only_on_the_model_probability():
    assert card_mod.kelly(0.6, 100) == pytest.approx(0.2)
    assert card_mod.kelly(0.4, 100) == 0.0
    r = priced()[0]
    assert card_mod.row_kelly(r) is None  # market only: no Kelly invented
    assert card_mod.row_kelly(as_buy(r)) is not None


def test_cli_card_writes_artifacts_from_the_ledger(scratch, capsys):
    root, _ = scratch
    assert run_card() == 0
    out = root / "out"
    names = sorted(p.name for p in out.iterdir())
    assert names == [f"PayoffPitch_NBA_{DAY}.{ext}" for ext in ("html", "pdf", "xlsx")]
    assert "2 games" in capsys.readouterr().out


def test_no_ledger_pass_is_an_error(scratch):
    root, path = scratch
    path.unlink()
    assert run_card() == 1


def test_pdf_failure_costs_only_the_pdf(scratch, monkeypatch):
    root, _ = scratch

    def boom(html):
        raise OSError("no pango")

    monkeypatch.setattr(cli, "render_pdf", boom)
    assert run_card() == 0
    out = root / "out"
    assert (out / f"PayoffPitch_NBA_{DAY}.xlsx").exists()
    assert not (out / f"PayoffPitch_NBA_{DAY}.pdf").exists()


def test_pricing_neutrality_card_never_changes_ledger_rows(scratch):
    root, path = scratch
    before_bytes = path.read_bytes()
    before = [asdict(r) for r in ledger.passes(root, DAY)]
    assert run_card() == 0
    assert run_card() == 0
    assert path.read_bytes() == before_bytes
    assert [asdict(r) for r in ledger.passes(root, DAY)] == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_preseason_card_reads_its_own_tree(monkeypatch, tmp_path):
    monkeypatch.setenv("NBAE_PRESEASON", "1")
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    monkeypatch.delenv("NBAE_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    monkeypatch.setattr(cli, "render_pdf", lambda html: b"%PDF")
    rows = priced()
    ledger.write_once(rows, ledger.pass_path(tmp_path / "preseason", DAY, rows[0].priced_at, "t"))
    assert run_card() == 0
    out = tmp_path / "preseason" / "output"
    assert (out / f"PayoffPitch_NBA_Preseason_{DAY}.xlsx").exists()
    assert "NBA Preseason Slate" in (out / f"PayoffPitch_NBA_Preseason_{DAY}.html").read_text()


# -- workbook ----------------------------------------------------------------
def test_workbook_sheets_columns_frozen_and_filtered():
    rows = priced()
    rows[0] = as_buy(rows[0])
    wb = load_workbook(BytesIO(build_workbook(build_card(rows, day=DAY))))
    assert wb.sheetnames == [
        "Buys",
        "Selections",
        "Games",
        "Injuries",
        "Record",
        "Integrity",
        "Legend",
    ]
    sel = wb["Selections"]
    header = [c.value for c in sel[1]]
    assert header == list(SELECTION_HEADER)
    for col in (
        "Matchup",
        "Tip (ET)",
        "Market",
        "Side",
        "Line",
        "Book",
        "Price",
        "Fair prob",
        "Model prob",
        "EV",
        "Kelly",
        "Tier",
        "Gates",
        "Params version",
    ):
        assert col in header
    assert sel.max_row == len(rows) + 1
    assert sel.freeze_panes == "A2" and sel.auto_filter.ref.startswith("A1:")
    tips = [r[0] for r in sel.iter_rows(min_row=2, values_only=True)]
    assert tips == sorted(tips)
    buys = wb["Buys"]
    assert buys.max_row == 2 and buys["Q2"].value == "Strong buy"  # one buy
    assert buys.freeze_panes == "A2"


def test_record_sheet_carries_n_and_intervals():
    rows = [
        replace(r, outcome="win" if i % 2 else "loss", pnl=1.0 if i % 2 else -1.0)
        for i, r in enumerate(priced())
    ]
    wb = load_workbook(BytesIO(build_workbook(build_card(rows, day=DAY, graded=rows, draws=20))))
    rec = wb["Record"]
    header = [c.value for c in rec[1]]
    assert {"n", "ROI", "ROI 95% lo", "ROI 95% hi", "CLV EV 95% lo"} <= set(header)
    cuts = {r[0] for r in rec.iter_rows(min_row=2, values_only=True)}
    assert {"buys", "board"} <= cuts


# -- email -------------------------------------------------------------------
class FakeSMTP:
    sent: list = []

    def __init__(self, host, port, context=None, timeout=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        self.user, self.password = user, password

    def send_message(self, msg):
        FakeSMTP.sent.append((self, msg))


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.sent = []
    monkeypatch.setattr(email_mod.smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP", None)  # any plain-SMTP path would blow up
    return FakeSMTP.sent


def test_email_without_config_keeps_files_and_returns_zero(scratch, fake_smtp):
    root, _ = scratch
    assert run_card("--email") == 0
    assert (root / "out" / f"PayoffPitch_NBA_{DAY}.xlsx").exists()
    assert fake_smtp == []


def test_email_sends_pdf_and_workbook_via_fake_smtp(scratch, fake_smtp, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "me@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
    assert run_card("--email", "--to", "you@example.com") == 0
    ((server, msg),) = fake_smtp
    assert server.host == "smtp.gmail.com" and server.port == 465
    assert server.password == "abcdefghijklmnop"
    assert msg["To"] == "you@example.com"
    names = [p.get_filename() for p in msg.iter_attachments()]
    assert names == [f"PayoffPitch_NBA_{DAY}.pdf", f"PayoffPitch_NBA_{DAY}.xlsx"]


def test_size_guard_drops_attachments_measured_after_base64(monkeypatch, fake_smtp):
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setenv("GMAIL_USER", "me@example.com")
    big, small = b"x" * 60_000, b"y" * 1_000
    sent = email_mod.send_package(
        load_config(),
        subject="s",
        html_body="<html><body>card</body></html>",
        text_body="card",
        attachments=[("card.pdf", small), ("book.xlsx", big)],
        max_bytes=70_000,  # 60 kB raw is ~82 kB once base64'd
    )
    assert sent.attached == ("card.pdf",) and sent.left_out == ("book.xlsx",)
    assert sent.size <= 70_000
    ((_, msg),) = fake_smtp
    assert "book.xlsx" in msg.get_body(("plain",)).get_content()
    budget = email_mod.attachment_budget(email_mod.GMAIL_LIMIT_BYTES)
    assert budget / email_mod.GMAIL_LIMIT_BYTES == pytest.approx(1 / 1.37, abs=0.01)


def test_email_needs_a_recipient(monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    for var in ("GMAIL_USER", "EMAIL_ADDRESS", "NBAE_EMAIL_TO", "MLBE_EMAIL_TO"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(email_mod.EmailNotConfigured):
        email_mod.send_package(load_config(), subject="s", html_body="", text_body="")


# -- context -----------------------------------------------------------------
def test_rest_and_travel_from_prior_finals_and_unknown_when_a_day_is_missing():
    games = [context.Matchup("late", "BOS", "NYK", TIPS["late"])]
    prior = [GameResult("1", date(2025, 12, 9), "NYK", "BOS", "STATUS_FINAL")]
    days = {DAY - timedelta(days=k) for k in range(1, context.LOOKBACK_DAYS + 1)}
    rest = context.rest(DAY, games, prior, days)
    bos, nyk = rest[("late", "BOS")], rest[("late", "NYK")]
    assert bos.known and bos.b2b and nyk.b2b
    assert "back-to-back" in bos.text() and "mi travel" in nyk.text()
    # A team idle in the window can't be called rested if a day in it went unread.
    idle = [context.Matchup("x", "PHX", "DET", TIPS["early"])]
    assert not context.rest(DAY, idle, prior, days - {DAY - timedelta(days=2)})[("x", "PHX")].known
    assert context.rest(DAY, idle, prior, days)[("x", "PHX")].known


def test_line_move_open_to_now_from_archived_boards(scratch):
    root, _ = scratch
    from nba_engine.data import capture

    early = board("2025-12-10T16:00:00Z")
    late = [
        replace(r, american=-200, opposite_american=170)
        if r.market == "game_ml" and r.side == "DET"
        else replace(r, american=170, opposite_american=-200)
        if r.market == "game_ml"
        else r
        for r in board("2025-12-10T22:00:00Z")
    ]
    for rows in (early, late):
        capture.write_snapshot(rows, root, DAY, label="board", captured_at=rows[0].captured_at)
    games = context.games_of(priced())
    moves, opened, now = context.moves(root, DAY, games)["early"]
    ml = next(m for m in moves if m.key == "ml")
    assert ml.now > ml.open
    assert opened < now
