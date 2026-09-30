"""Players worth watching in a game: Heisman money and NFL draft stock.

Two live, machine-readable sources; both reported on the card, neither priced.

* **Heisman odds** -- RotoWire's futures table (the JSON behind its Heisman-odds
  page), every book it carries. The card quotes the best price and only names
  players inside :data:`HEISMAN_MAX_ODDS` (about 30 in August); a +50000 flier
  is not a watch.
* **NFL draft big board** -- Tankathon's consensus board for the coming draft.
  Anyone inside the first two rounds (:data:`DRAFT_MAX_RANK`) is named with his
  overall rank.

The position awards (Butkus, Outland, Bednarik, Thorpe, Biletnikoff...) have no
priced futures market and their watch lists are prose press releases, so they
are read from an optional hand-kept file instead: ``awards_watch.json`` in the
cache directory, ``{"Butkus": ["Player Name, School", ...], ...}``. Absent file,
no award tags -- the card never guesses a watch list.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path

from cfb_engine.data.depth import name_key
from cfb_engine.data.teamnames import school_key
from mlb_engine.data import http

log = logging.getLogger(__name__)

_HEISMAN = "https://www.rotowire.com/betting/college-football/tables/all-player-futures.php"
_BIG_BOARD = "https://www.tankathon.com/nfl/big-board"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    "Referer": "https://www.rotowire.com/betting/college-football/heisman-odds.php",
}
HEISMAN_MAX_ODDS = 10000  # longest best price still called a Heisman contender
DRAFT_MAX_RANK = 64  # first two rounds


@dataclass
class WatchPlayer:
    name: str
    team: str  # school_key
    position: str = ""
    heisman_odds: int | None = None  # best American price across books
    draft_rank: int | None = None  # overall big-board rank
    awards: list[str] = field(default_factory=list)

    def tags(self) -> list[str]:
        out: list[str] = []
        if self.heisman_odds is not None:
            out.append(f"Heisman {self.heisman_odds:+d}")
        if self.draft_rank is not None:
            rnd = "1st" if self.draft_rank <= 32 else "2nd"
            out.append(f"{rnd}-round draft stock (#{self.draft_rank} big board)")
        out.extend(f"{a} watch list" for a in self.awards)
        return out


# school_key -> name_key -> player
WatchBook = dict[str, dict[str, WatchPlayer]]


def _american(text: object) -> int | None:
    if not isinstance(text, str):
        return None
    s = text.strip().replace("+", "")
    if not s or not s.lstrip("-").isdigit():
        return None
    return int(s)


def parse_heisman(data: object) -> list[tuple[str, str, int]]:
    """``(name, team, best odds)`` per RotoWire futures row with any priced book."""
    out: list[tuple[str, str, int]] = []
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        name, team = row.get("name"), row.get("team")
        if not (isinstance(name, str) and isinstance(team, str)):
            continue
        prices = [
            odds
            for key, val in row.items()
            if key.endswith("_odds") and (odds := _american(val)) is not None
        ]
        if prices:
            out.append((name, team, max(prices)))
    return out


_ROW = re.compile(
    r'<div class="mock-row-pick-number">(\d+)</div>.*?'
    r'<div class="mock-row-name">([^<]+)</div>'
    r'<div class="mock-row-school-position">([^<|]+)\|([^<]+)</div>',
    re.S,
)


def parse_big_board(page: str) -> list[tuple[int, str, str, str]]:
    """``(rank, name, position, school)`` per Tankathon board row."""
    out: list[tuple[int, str, str, str]] = []
    for rank, name, pos, school in _ROW.findall(page):
        out.append((int(rank), unescape(name).strip(), pos.strip(), unescape(school).strip()))
    return out


def load_awards(path: Path | None) -> dict[str, list[tuple[str, str]]]:
    """``{award: [(player, school), ...]}`` from the optional hand-kept file."""
    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("awards watch file unreadable (%s): %s", path, exc)
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    if not isinstance(raw, dict):
        return out
    for award, names in raw.items():
        if not isinstance(names, list):
            continue
        for entry in names:
            if isinstance(entry, str) and "," in entry:
                player, school = entry.rsplit(",", 1)
                out.setdefault(str(award), []).append((player.strip(), school.strip()))
    return out


def build_watch_book(
    heisman: list[tuple[str, str, int]],
    board: list[tuple[int, str, str, str]],
    awards: dict[str, list[tuple[str, str]]],
) -> WatchBook:
    book: WatchBook = {}

    def slot(name: str, team: str) -> WatchPlayer:
        return book.setdefault(school_key(team), {}).setdefault(
            name_key(name), WatchPlayer(name=name, team=school_key(team))
        )

    for name, team, odds in heisman:
        if odds <= HEISMAN_MAX_ODDS:
            p = slot(name, team)
            p.heisman_odds = odds if p.heisman_odds is None else max(p.heisman_odds, odds)
    for rank, name, pos, school in board:
        if rank <= DRAFT_MAX_RANK:
            p = slot(name, school)
            p.draft_rank = rank
            p.position = p.position or pos
    for award, entries in awards.items():
        for name, school in entries:
            slot(name, school).awards.append(award)
    return book


def fetch_watch_book(*, awards_file: Path | None = None, timeout: float = 20.0) -> WatchBook:
    heisman: list[tuple[str, str, int]] = []
    board: list[tuple[int, str, str, str]] = []
    try:
        resp = http.get(_HEISMAN, params={"future": "Heisman"}, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        heisman = parse_heisman(resp.json())
    except Exception as exc:  # noqa: BLE001 - the watch line is a nicety, not the price
        log.warning("Heisman odds unavailable: %s", exc)
    try:
        resp = http.get(_BIG_BOARD, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        board = parse_big_board(resp.text)
    except Exception as exc:  # noqa: BLE001
        log.warning("draft big board unavailable: %s", exc)
    return build_watch_book(heisman, board, load_awards(awards_file))


def watch_for(book: WatchBook, team: str, *, limit: int = 3) -> list[WatchPlayer]:
    """A team's watch players, Heisman money first, then draft rank."""
    players = list(book.get(school_key(team), {}).values())
    players.sort(
        key=lambda p: (
            p.heisman_odds if p.heisman_odds is not None else 10**6,
            p.draft_rank if p.draft_rank is not None else 10**6,
            p.name,
        )
    )
    return players[:limit]
