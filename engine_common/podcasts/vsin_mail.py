"""VSiN's "picks at a glance" email, read off Gmail into the podcast store.

Bill at VSiN mails a list of the day's picks: every VSiN Pro handicapper's bets
by game, with price and units, and the Circa Football Invitational contestants'
spread and total picks. Each email becomes two :class:`Extraction` files, one
per source, with the handicapper as the host, so every engine prints, grades
and audits these like a podcast pick: beside the card, in their own ledger,
never in a price, gate or stake.

Only game markets are kept (spread, moneyline, total); props, teasers and
parlays are not something the cards place. A Circa pick is an official bet
when marked BEST BET and a lean otherwise; it carries no price.
"""

from __future__ import annotations

import email
import hashlib
import imaplib
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup

from engine_common.podcasts.extract import PICKS_DIR, Extraction, Pick
from engine_common.podcasts.shows import (
    CBB,
    CFB,
    CIRCA_NAME,
    CIRCA_SHOW,
    MLB,
    NBA,
    NFL,
    NHL,
    VSIN_PRO_NAME,
    VSIN_PRO_SHOW,
)

SENDER = "bill@mail.vsin.com"
SUBJECT = "picks at a glance"
PARSER_VERSION = "vsin-mail-1"
IMAP_HOST = "imap.gmail.com"
# Games the email lists more than this far past its own date are other slates.
LOOKAHEAD = timedelta(days=2)

LEAGUE_OF = {
    "NFL": NFL,
    "NHL": NHL,
    "NBA": NBA,
    "MLB": MLB,
    "CFB": CFB,
    "NCAAF": CFB,
    "COLLEGE FOOTBALL": CFB,
    "CBB": CBB,
    "NCAAB": CBB,
    "COLLEGE BASKETBALL": CBB,
}

_PRO_HEAD = re.compile(r"^VSiN Pro Picks by Game", re.I)
_CIRCA_HEAD = re.compile(r"^Circa Football Invitational", re.I)
_END_HEAD = re.compile(r"^(?:VSiN Pro Picks by OTHER|Tennis / UFC|.*View Article$)", re.I)
_GAME = re.compile(
    r"^(?P<league>[A-Z][A-Z ]+?)\s+-\s+(?P<day>\w{3}),\s+(?P<mon>\w{3})\s+(?P<dom>\d{1,2})"
    r"\s+(?:@|at)\s+(?P<time>[\d:]+\s*[AP]M)\s*ET\s+-\s+(?P<away>.+?)\s+at\s+(?P<home>.+)$",
    re.I,
)
_LINE = re.compile(r"^(?P<host>[^:]+?)\s*:\s*(?P<bet>.+)$")
_UNITS = re.compile(r"\[\s*([\d.]+)\s*units?\s*\]", re.I)
_PRICE = re.compile(r"\(\s*([+-]\d{3,4})\s*\)")
_TOTAL = re.compile(
    r"^(?P<away>.+?)\s+at\s+(?P<home>.+?)\s+-\s+(?P<side>OVER|UNDER)\s+\(\s*(?P<num>[\d.]+)\s*\)"
)
_ML = re.compile(r"^Money Line\s+-\s+(?P<team>.+?)\s+(?:\([+-]\d+\)\s+)?(?:at|vs)\s+", re.I)
_ATS = re.compile(
    r"^(?P<team>[A-Z][\w.' ]+?)\s+\((?P<num>[+-]?[\d.]+|PK)\)\s+(?:\([+-]\d+\)\s+)?(?:at|vs)\s+",
    re.I,
)
_BEST = re.compile(r"\s*BEST BET\s*$", re.I)
_DOW = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


@dataclass(frozen=True)
class Mail:
    uid: str
    sent: str  # ISO8601 UTC
    subject: str
    text: str


@dataclass(frozen=True)
class GameHead:
    league: str
    date: datetime  # the game's date at midnight UTC (day granularity)
    away: str
    home: str


@dataclass(frozen=True)
class Parsed:
    market: str
    team: str
    opponent: str
    side: str | None
    line: float | None
    price: float | None
    units: float | None
    best: bool
    description: str


def _clean(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"[ \t\xa0]+", " ", text).strip()


def _game_head(line: str, year: int) -> GameHead | None:
    m = _GAME.match(line)
    if m is None:
        return None
    league = LEAGUE_OF.get(m.group("league").upper().strip())
    mon = _DOW.get(m.group("mon").title())
    if league is None or mon is None:
        return None
    return GameHead(
        league,
        datetime(year, mon, int(m.group("dom")), tzinfo=timezone.utc),
        _clean(m.group("away")),
        _clean(m.group("home")),
    )


def parse_bet(text: str, game: GameHead) -> Parsed | None:
    """One listed bet as a pick's fields; ``None`` for a prop, parlay or teaser."""
    text = _clean(text)
    best = bool(_BEST.search(text))
    body = _BEST.sub("", text)
    units_m = _UNITS.search(body)
    units = float(units_m.group(1)) if units_m else None
    body_wo_units = _UNITS.sub("", body).strip()
    desc = re.sub(r"\s*\|.*$", "", body_wo_units)
    if "[" in desc or "]" in desc:  # props carry "[ Receiving Yards ]"; parlays "[ leg ]"
        return None
    price_m = _PRICE.search(desc)
    price = float(price_m.group(1)) if price_m else None
    tot = _TOTAL.match(desc)
    if tot:
        return Parsed(
            "game_total",
            game.away,
            game.home,
            tot.group("side").lower(),
            float(tot.group("num")),
            price,
            units,
            best,
            desc,
        )
    ml = _ML.match(desc)
    if ml:
        team = _clean(ml.group("team"))
        return Parsed("game_ml", team, _other(team, game), None, None, price, units, best, desc)
    ats = _ATS.match(desc)
    if ats:
        team = _clean(ats.group("team"))
        tok = ats.group("num").upper()
        line = 0.0 if tok == "PK" else float(tok)
        return Parsed("game_ats", team, _other(team, game), None, line, price, units, best, desc)
    return None


def _other(team: str, game: GameHead) -> str:
    return game.home if team.lower() == game.away.lower() else game.away


def bets_in(text: str, year: int) -> Iterator[tuple[str, str, str, GameHead]]:
    """``(source show, host, bet text, game)`` for every listed bet in the email body."""
    show = ""
    game: GameHead | None = None
    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        if _PRO_HEAD.match(line):
            show, game = VSIN_PRO_SHOW, None
            continue
        if _CIRCA_HEAD.match(line):
            show, game = CIRCA_SHOW, None
            continue
        if _END_HEAD.match(line):
            show, game = "", None
            continue
        if not show:
            continue
        head = _game_head(line, year)
        if head is not None:
            game = head
            continue
        if game is None or line.lower().startswith("last updated"):
            continue
        m = _LINE.match(line)
        if m:
            yield show, _clean(m.group("host")), m.group("bet"), game


def picks_in(mail: Mail) -> dict[str, list[Pick]]:
    sent = datetime.fromisoformat(mail.sent)
    out: dict[str, dict[str, Pick]] = {VSIN_PRO_SHOW: {}, CIRCA_SHOW: {}}
    names = {VSIN_PRO_SHOW: VSIN_PRO_NAME, CIRCA_SHOW: CIRCA_NAME}
    for n, (show, host, bet, game) in enumerate(bets_in(mail.text, sent.year)):
        if game.date > sent + LOOKAHEAD:
            continue
        parsed = parse_bet(bet, game)
        if parsed is None:
            continue
        raw = "|".join((show, mail.uid, host, parsed.market, parsed.description, game.league))
        pid = hashlib.sha1(raw.encode()).hexdigest()[:12]
        official = show == VSIN_PRO_SHOW or parsed.best
        out[show].setdefault(
            pid,
            Pick(
                pick_id=pid,
                show=show,
                show_name=names[show],
                host=host or None,
                episode_guid=f"vsinmail-{mail.uid}",
                episode_title=mail.subject,
                published=mail.sent,
                kind="official" if official else "lean",
                market=parsed.market,
                team=parsed.team,
                opponent=parsed.opponent,
                side=parsed.side,
                line=parsed.line,
                price=parsed.price,
                units=parsed.units,
                seconds=n,
                quote=bet[:300],
                reason=f"{game.away} at {game.home}, {game.date.date()}",
                description=parsed.description,
                league=game.league,
                edge="",
            ),
        )
    return {k: list(v.values()) for k, v in out.items()}


def extractions(mail: Mail) -> list[tuple[str, Extraction]]:
    sha = hashlib.sha1(mail.text.encode()).hexdigest()
    out: list[tuple[str, Extraction]] = []
    for show, picks in picks_in(mail).items():
        if picks:
            leagues = sorted({p.league for p in picks})
            ex = Extraction(
                f"vsinmail-{show}-{mail.uid}", "rules", PARSER_VERSION, sha, picks, [], leagues
            )
            out.append((show, ex))
    return out


def extraction_path(root: Path, show: str, mail: Mail) -> Path:
    return root / PICKS_DIR / f"{show}__{mail.uid}.json"


# ------------------------------------------------------------------ Gmail


def _body_text(msg: Message) -> str:
    plain, html_ = "", ""
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        if ctype == "text/plain" and not plain:
            plain = text
        elif ctype == "text/html" and not html_:
            html_ = text
    if plain.strip():
        return plain
    return BeautifulSoup(html_, "html.parser").get_text("\n") if html_ else ""


def _mail(uid: str, raw: bytes) -> Mail | None:
    msg = email.message_from_bytes(raw)
    subject = _clean(str(msg.get("Subject", "")))
    if SUBJECT not in subject.lower():
        return None
    try:
        sent = parsedate_to_datetime(str(msg.get("Date", ""))).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
    return Mail(uid, sent.isoformat(), subject, _body_text(msg))


def fetch_mail(
    since: datetime,
    until: datetime,
    *,
    user: str | None = None,
    password: str | None = None,
    sender: str = SENDER,
) -> list[Mail]:
    """VSiN's picks emails received in ``[since, until)`` from the Gmail inbox."""
    user = user or os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    password = password or os.getenv("GMAIL_APP_PASSWORD")
    if not user or not password:
        raise RuntimeError("GMAIL_USER/GMAIL_APP_PASSWORD not set")
    out: list[Mail] = []
    box = imaplib.IMAP4_SSL(IMAP_HOST)
    try:
        box.login(user, password)
        box.select('"[Gmail]/All Mail"', readonly=True)
        day = (since - timedelta(days=1)).strftime("%d-%b-%Y")
        status, data = box.uid("search", "FROM", f'"{sender}"', "SINCE", day)
        if status != "OK" or not data or not data[0]:
            return out
        for uid in data[0].split():
            status, parts = box.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not parts:
                continue
            raw = next((p[1] for p in parts if isinstance(p, tuple) and len(p) > 1), None)
            if not isinstance(raw, bytes):
                continue
            mail = _mail(uid.decode(), raw)
            if mail and since <= datetime.fromisoformat(mail.sent) < until:
                out.append(mail)
    finally:
        try:
            box.logout()
        except Exception:  # noqa: BLE001 - logout failing loses nothing
            pass
    return out


def read_mail(
    root: Path,
    since: datetime,
    until: datetime,
    leagues: Iterable[str] | None = None,
    *,
    mails: list[Mail] | None = None,
) -> list[Extraction]:
    """Store each email's game picks; the same email re-read replaces its files."""
    wanted = set(leagues) if leagues is not None else None
    kept: list[Extraction] = []
    for mail in mails if mails is not None else fetch_mail(since, until):
        for show, ex in extractions(mail):
            if wanted is not None:
                ex.picks = [p for p in ex.picks if p.league in wanted]
                if not ex.picks:
                    continue
            path = extraction_path(root, show, mail)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ex.to_json())
            kept.append(ex)
    return kept


__all__ = [
    "GameHead",
    "Mail",
    "PARSER_VERSION",
    "Parsed",
    "SENDER",
    "SUBJECT",
    "bets_in",
    "extractions",
    "fetch_mail",
    "parse_bet",
    "picks_in",
    "read_mail",
]
