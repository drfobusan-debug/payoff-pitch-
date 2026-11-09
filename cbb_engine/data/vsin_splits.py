"""VSiN public betting splits for college basketball, captured as-of and archived.

VSiN publishes each book's handle share and ticket share per side at
``data.vsin.com/betting-splits/?sport=CBB``. Since November 2026 the full table
is behind VSiN Pro: an anonymous request gets one preview game and a "30 more
games for subscribers" banner. The page unlocks on the Piano session cookie
``__utp`` of a logged-in Pro account, read here from ``VSIN_UTP``; the login
itself is Turnstile-gated, so it is done once in a browser and the cookie (a
year's life) is what the machines carry. A page that still shows the preview
banner is a *gated* read and is dropped whole -- one game stored as a board
would look like a slate with no public money anywhere else.

Every board pass stores the rows it read under ``splits/cbb/<day>/<pass_id>.jsonl``
with the capture time, the book and VSiN's game code, so the public side can
later be read as of any moment before tip and never from a later page. Nothing
here prices: the only consumer is the graded-only reverse-line-move bands.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

import requests

from cbb_engine.config import SPORT
from cbb_engine.output import teamstats
from mlb_engine.data import http

log = logging.getLogger(__name__)

SPLITS_URL = "https://data.vsin.com/betting-splits/?source={source}&sport=CBB"
_HEADERS = {"User-Agent": "Mozilla/5.0 (payoff-pitch-cbb/1.0)"}
# VSiN's ``source`` code -> book label, sharpest first.
SOURCES: tuple[tuple[str, str], ...] = (("circa", "circa"), ("DK", "draftkings"))
TOKEN_ENV = "VSIN_UTP"
GATED_MARK = "sp-preview-more"

# VSiN spells schools the way a betting screen does; these are the spellings the token
# match cannot reach from the Odds API's "School Mascot" names. Keys are lower-case with
# hyphens as spaces, after the directional initial is expanded.
_ALIASES: dict[str, str] = {
    "central conn st": "Central Connecticut St",
    "texas arlington": "UT Arlington",
    "texas el paso": "UTEP",
    "southern u": "Southern",
    "uw green bay": "Green Bay",
    "uw milwaukee": "Milwaukee",
    "sf austin": "Stephen F. Austin",
    "miami fl": "Miami Hurricanes",
    "miami oh": "Miami (OH)",
    "loyola chi": "Loyola Chicago",
    "loyola mary": "Loyola Marymount",
    "ul monroe": "Louisiana Monroe",
    "ul lafayette": "Louisiana",
}
_DIRECTIONS = {"n": "North", "s": "South", "e": "East", "w": "Western", "c": "Central"}

_ROW_RE = re.compile(r'<tr class="sp-row[^"]*">(.*?)</tr>', re.S)
_GAME_RE = re.compile(r'data-gamecode="((\d{4})(\d{2})(\d{2})CBB\d+)"')
_TEAM_RE = re.compile(r'class="sp-team-link"[^>]*>([^<]+)<')
_BADGE_RE = re.compile(r'<span class="sp-badge[^"]*">(.*?)</span>', re.S)


@dataclass(frozen=True)
class SplitRow:
    """One team's line of the splits table at one book: the number and its shares."""

    game_date: str
    gamecode: str
    team: str
    book: str
    spread_line: float | None
    spread_handle: float | None
    spread_bets: float | None
    total_line: float | None
    total_handle: float | None
    total_bets: float | None
    ml_american: float | None
    ml_handle: float | None
    ml_bets: float | None
    captured_at: str = ""
    pass_id: str = ""
    source: str = "vsin"


@dataclass(frozen=True)
class Public:
    """The public's shares on one side, as read before tip."""

    bets_pct: float | None
    handle_pct: float | None
    book: str
    captured_at: str

    @property
    def majority(self) -> bool | None:
        """Whether the tickets lean to this side; ``None`` on a dead heat or a blank."""
        if self.bets_pct is None or self.bets_pct == 50.0:
            return None
        return self.bets_pct > 50.0


def vsin_name(raw: str) -> str:
    """A VSiN spelling in the engine's shape ("C Conn ST" -> "Central Connecticut St")."""
    words = html_mod.unescape(raw).replace("-", " ").split()
    if words and words[0].lower() in _DIRECTIONS:
        words[0] = _DIRECTIONS[words[0].lower()]
    joined = " ".join(words)
    return _ALIASES.get(joined.lower(), joined)


def token() -> str | None:
    tok = os.getenv(TOKEN_ENV, "").strip()
    return tok or None


def gated(html: str) -> bool:
    """True when the page is the anonymous preview (one game and a subscribe banner)."""
    return GATED_MARK in html


def parse_splits(html: str, book: str = "") -> list[SplitRow]:
    """Team rows (two per game, visitor first) from a splits page; the preview page parses to nothing."""
    if gated(html):
        return []
    rows: list[SplitRow] = []
    for chunk in _ROW_RE.findall(html):
        stamp = _GAME_RE.search(chunk)
        team = _TEAM_RE.search(chunk)
        badges = _BADGE_RE.findall(chunk)
        if stamp is None or team is None or len(badges) < 9:
            continue
        code, year, month, day = stamp.groups()
        rows.append(
            SplitRow(
                game_date=Date(int(year), int(month), int(day)).isoformat(),
                gamecode=code,
                team=team.group(1).strip(),
                book=book,
                spread_line=_num(badges[0]),
                spread_handle=_pct(badges[1]),
                spread_bets=_pct(badges[2]),
                total_line=_num(badges[3]),
                total_handle=_pct(badges[4]),
                total_bets=_pct(badges[5]),
                ml_american=_num(badges[6]),
                ml_handle=_pct(badges[7]),
                ml_bets=_pct(badges[8]),
            )
        )
    return rows


def fetch(day: Date, *, captured_at: str, pass_id: str, timeout: int = 25) -> list[SplitRow]:
    """Today's rows from each book, stamped with the pass; empty (and logged) without a token or when gated."""
    tok = token()
    if tok is None:
        log.warning("VSiN splits skipped: %s is not set", TOKEN_ENV)
        return []
    out: list[SplitRow] = []
    for source, label in SOURCES:
        try:
            resp = http.get(
                SPLITS_URL.format(source=source), headers={**_HEADERS, "Cookie": f"__utp={tok}"}, timeout=timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("VSiN splits fetch failed (%s): %s", source, exc)
            continue
        if gated(resp.text):
            log.warning("VSiN splits gated (%s): %s is missing or expired; nothing stored", source, TOKEN_ENV)
            continue
        rows = [r for r in parse_splits(resp.text, label) if r.game_date == day.isoformat()]
        out += [SplitRow(**{**asdict(r), "captured_at": captured_at, "pass_id": pass_id}) for r in rows]
    return out


def day_dir(root: Path, game_date: str) -> Path:
    return root / "splits" / SPORT / game_date


def write_capture(root: Path, game_date: str, pass_id: str, rows: Sequence[SplitRow]) -> Path | None:
    """One immutable JSONL per pass; nothing written for an empty read."""
    if not rows:
        return None
    path = day_dir(root, game_date) / f"{pass_id}.jsonl"
    if path.exists():
        raise FileExistsError(f"splits already written: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in rows))
    return path


def read_day(root: Path, game_date: str) -> list[SplitRow]:
    out: list[SplitRow] = []
    d = day_dir(root, game_date)
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                out.append(SplitRow(**json.loads(line)))
    return out


def public(
    rows: Iterable[SplitRow],
    teams: Mapping[str, tuple[str, str]],
    tips: Mapping[str, str],
) -> dict[str, dict[tuple[str, str], Public]]:
    """Each event's public shares per (market, side) from the latest capture before its tip.

    Rows are matched by date-stamped game code and team (``teamstats.match_team`` against the
    VSiN spellings on the page), never by team alone. The sharper book wins a tie in time;
    a capture at or after the tip is not pre-tip and is ignored.
    """
    by_cap: dict[str, list[SplitRow]] = {}
    for r in rows:
        by_cap.setdefault(r.captured_at, []).append(r)
    rank = {label: i for i, (_, label) in enumerate(SOURCES)}
    out: dict[str, dict[tuple[str, str], Public]] = {}
    for eid, (home, away) in teams.items():
        tip = tips.get(eid, "")
        best: tuple[str, dict[tuple[str, str], Public]] | None = None
        for at, group in by_cap.items():
            if tip and at >= tip:
                continue
            sides = _event_sides(group, home, away, rank)
            if sides and (best is None or at > best[0]):
                best = (at, sides)
        if best:
            out[eid] = best[1]
    return out


def _event_sides(
    group: Sequence[SplitRow], home: str, away: str, rank: Mapping[str, int]
) -> dict[tuple[str, str], Public]:
    names = {vsin_name(r.team): r.team for r in group}
    h, a = teamstats.match_team(home, names), teamstats.match_team(away, names)
    if not h or not a or h == a:
        return {}
    h, a = names[h], names[a]
    codes = {r.gamecode for r in group if r.team == h} & {r.gamecode for r in group if r.team == a}
    if not codes:
        return {}
    code = min(codes)
    out: dict[tuple[str, str], Public] = {}
    for r in sorted((r for r in group if r.gamecode == code), key=lambda r: rank.get(r.book, 99)):
        side = "home" if r.team == h else "away"
        over_under = "under" if side == "home" else "over"
        for key, bets, handle in (
            (("game_ats", side), r.spread_bets, r.spread_handle),
            (("game_ml", side), r.ml_bets, r.ml_handle),
            (("game_total", over_under), r.total_bets, r.total_handle),
        ):
            if bets is None or bets == 50.0 or bets == handle:
                # A market the book has no ticket count for prints a placeholder (Circa's
                # 50/50 tickets beside a real handle split, a 0/0 or 100/100 against a blank
                # side); none can be told from a dead heat, and a dead heat has no majority
                # anyway, so the side is left for a book that did publish a count.
                continue
            out.setdefault(key, Public(bets, handle, r.book, r.captured_at))
    return out


def _pct(v: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", v)
    return float(m.group()) if m else None


def _num(v: str) -> float | None:
    m = re.search(r"[+-]?\d+(?:\.\d+)?", v.replace(",", "").replace(" ", ""))
    return float(m.group()) if m else None


__all__ = [
    "SOURCES", "TOKEN_ENV", "Public", "SplitRow", "day_dir", "fetch", "gated", "parse_splits", "public",
    "read_day", "token", "write_capture",
]
