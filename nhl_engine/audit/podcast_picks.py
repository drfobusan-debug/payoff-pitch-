"""Grade the Hockey Gambling Podcast's own picks against official results.

Deterministic, like the read it feeds on: a *pick* is a transcript line inside
a game segment that commits to a side ("give me the Habs", "25 puck bucks on
the over", "I'm taking Rangers on the puck line +170"). Lines that pass, lean
or say "no bet" are not picks. Each pick keeps the price quoted on that line
or, for a moneyline, the price the hosts read in the game intro; a pick with
no spoken price is graded W/L but carries no P&L -- a board price is never
substituted.

Attribution: Whisper has no speaker labels, so a pick is credited to a host
only when the transcript names one -- on the pick line itself or in the
hand-off just before it ("Chase, what you got?"). Everything else is
``unattributed``. The hosts also read their own puck-bucks records on air;
those are parsed and shown as *self-reported*, separate from our grading.

Nothing here reaches a model price, gate or the engine ledger.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

from engine_common.odds import american_to_prob
from nhl_engine.audit.grade import PUSH, WIN, pnl, settle
from nhl_engine.data.capture import QuoteRow, pregame
from nhl_engine.data.podcast import podcast_dir
from nhl_engine.features.podcast_read import (
    _PRICE,
    _STAKE,
    _TOTAL,
    _WORDNUM,
    NICKNAMES,
    GameRead,
    PodcastRead,
    _american,
    _norm,
    teams_in,
)
from nhl_engine.market.board import selections
from nhl_engine.schemas import GameResult

UNATTRIBUTED = "unattributed"
HOSTS: dict[str, tuple[str, ...]] = {
    "Ryan Gilbert": ("ryan", "gilbert"),
    "Joel Meyer": ("joel", "jol", "meyer", "myer"),
    "Sean Marciano": ("sean", "shawn", "archer", "marciano"),
    "Chase Saul": ("chase", "saul"),
}

_COMMIT = re.compile(
    r"\b(give me|i('| a| wi)?(m|ll) tak(e|ing)|i took|i bet|i('| )?ve got|i got|i have|"
    r"puck bucks|puck box|bucks on|on the (over|under)|sprinkle)\b",
    re.I,
)
_DECLINE = re.compile(
    r"\b(no bet|pass(ing)?( on)?|stay(ing)? away|not betting|wouldn'?t bet|lean(ing)?|"
    r"i got nothing|got no|no puck bucks|wait for|they|he('| i)?s|you('| wi)?(re|ll))\b",
    re.I,
)
_HANDOFF = re.compile(
    r"\b(what (do )?you got|what('| ha)?ve you got|you want this|close it out|how many|"
    r"what about you|your turn|go ahead|take us|you('| a)?re up)\b",
    re.I,
)
_THE_OU = re.compile(r"\bthe (over|under)\b", re.I)
_PUCKLINE = re.compile(r"\bpuck ?line\b|\bminus one and a half\b|\bplus one and a half\b", re.I)
_EACH = re.compile(r"\b(\d{1,3})\s+each\b", re.I)
_RECORD = re.compile(
    r"\b(?P<who>[a-z]+)\s+(?P<w>\w+)\s+(?:and\s+|-\s*)?(?P<l>\w+)\s*,?\s*(?:is\s+)?"
    r"(?P<dir>up|down)\s*(?:with\s+)?(?P<amt>\d+|[a-z]+(?:\s+[a-z]+)?)",
    re.I,
)
_NUMWORDS = {
    **_WORDNUM,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


@dataclass
class Pick:
    slate_date: str
    matchup: str
    market: str  # game_ml | game_pl | game_total
    side: str  # team code or over/under
    line: float | None
    american: int | None  # spoken price; None -> graded W/L only, no P&L
    stake: float | None  # puck bucks when stated
    host: str  # HOSTS key or "unattributed"
    stamp: str
    text: str
    outcome: str | None = None
    pnl_flat: float | None = None  # 1 unit at the spoken price
    pnl_stake: float | None = None  # stated puck bucks at the spoken price
    close_consensus: float | None = None
    clv: float | None = None
    graded_at: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.matchup, self.market, self.side, self.host)


@dataclass
class StatedRecord:
    host: str
    wins: int
    losses: int
    units: float
    stamp: str


# ---------------------------------------------------------------- extraction


def _num(tok: str) -> int | None:
    tok = tok.lower().strip()
    if tok.isdigit():
        return int(tok)
    parts = tok.split()
    if all(p in _NUMWORDS for p in parts):
        return sum(_NUMWORDS[p] for p in parts)
    return None


def host_in(text: str) -> str | None:
    t = f" {_norm(text)} "
    for host, aliases in HOSTS.items():
        if any(f" {a} " in t for a in aliases):
            return host
    return None


def stated_records(lines: list[tuple[str, str]]) -> list[StatedRecord]:
    """``(stamp, text)`` lines -> the latest self-reported record per host."""
    out: dict[str, StatedRecord] = {}
    for stamp, text in lines:
        t = _norm(text)
        for m in _RECORD.finditer(t):
            host = host_in(m.group("who"))
            w, lo, amt = _num(m.group("w")), _num(m.group("l")), _num(m.group("amt"))
            if host is None or w is None or lo is None or amt is None:
                continue
            out[host] = StatedRecord(
                host, w, lo, float(amt if m.group("dir") == "up" else -amt), stamp
            )
    return list(out.values())


def _segment_lines(read: GameRead) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in read.excerpt.splitlines():
        m = re.match(r"\[(\d\d:\d\d(?::\d\d)?)\]\s*(.*)", raw)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _total_line(text: str) -> tuple[str, float | None] | None:
    m = _TOTAL.search(text)
    if not m:
        return None
    if text[m.end() : m.end() + 1].isdigit():
        return None
    n = _WORDNUM.get(m.group(2).lower()) or int(m.group(2))
    half = bool(re.search(r"half|\.5|nap", m.group(0), re.I))
    line = n + (0.5 if half else 0.0)
    return m.group(1).lower(), (line if 4.0 <= line <= 8.5 else None)


def _near_commit(text: str, team: str) -> bool:
    """The team is the object of the commitment, not scenery ("can't trust Buffalo's goalie")."""
    t = f" {_norm(text)} "
    for alias in NICKNAMES[team]:
        for m in re.finditer(f" {re.escape(alias)} ", t):
            before = t[max(0, m.start() - 40) : m.start()]
            if re.search(
                r"(give me|tak(e|ing)|took|bet|on|bucks|with|love|like)\s+(the\s*)?$", before
            ):
                return True
    return False


def _stake(text: str) -> float | None:
    m = _STAKE.search(text) or _EACH.search(text)
    return float(m.group(1)) if m else None


def _price(text: str) -> int | None:
    prices = [_american(s, n) for s, n in _PRICE.findall(text)]
    return prices[-1] if prices else None


def extract_picks(read: PodcastRead) -> list[Pick]:
    picks: list[Pick] = []
    for g in read.games.values():
        away, home = g.matchup.split(" @ ", 1)
        codes = {away, home}
        lines = _segment_lines(g)
        seen: set[tuple[str, str, str, str]] = set()
        speaker = UNATTRIBUTED
        for stamp, text in lines[1:]:
            if _HANDOFF.search(text) and (named := host_in(text)):
                speaker = named
                continue
            if _DECLINE.search(text) or not _COMMIT.search(text):
                continue
            stake, spoken = _stake(text), _price(text)
            teams = teams_in(text, codes)
            cands: list[tuple[str, str, float | None, int | None]] = []
            tot = _total_line(text)
            if tot and "team total" not in text.lower():
                cands.append(("game_total", tot[0], tot[1] or g.quoted_total, spoken))
            elif (m := _THE_OU.search(text)) and g.quoted_total:
                cands.append(("game_total", m.group(1).lower(), g.quoted_total, spoken))
            if len(teams) == 1 and (not cands or _near_commit(text, teams[0])):
                team = teams[0]
                if _PUCKLINE.search(text):
                    fav = g.quoted_ml.get(team, 0) < 0
                    cands.append(("game_pl", team, -1.5 if fav else 1.5, spoken))
                else:
                    cands.append(("game_ml", team, None, spoken or g.quoted_ml.get(team)))
            host = speaker
            for market, side, line, price in cands:
                p = Pick(
                    read.slate_date, g.matchup, market, side, line, price, stake, host, stamp, text
                )
                if p.key in seen:
                    continue
                seen.add(p.key)
                picks.append(p)
    return picks


# ---------------------------------------------------------------- grading


def grade_picks(
    picks: list[Pick],
    results: Mapping[str, GameResult],
    day_quotes: list[QuoteRow],
    *,
    graded_at: str,
    starts: Mapping[str, str] | None = None,
) -> list[Pick]:
    if starts is not None:
        day_quotes = pregame(day_quotes, starts)
    close = {s.key: s for s in selections(day_quotes)} if day_quotes else {}
    for p in picks:
        res = results.get(p.matchup)
        if res is None or not res.is_final:
            continue
        entity = ""
        p.outcome = settle(
            market=p.market, side=p.side, entity=entity, line=p.line, ot_rule="incl_ot_so", res=res
        )
        if p.american is not None:
            p.pnl_flat = pnl(p.outcome, p.american)
            p.pnl_stake = pnl(p.outcome, p.american, p.stake) if p.stake else None
            c = close.get((p.matchup, p.market, p.side, entity, p.line))
            if c is not None:
                p.close_consensus = c.consensus
                p.clv = c.consensus - american_to_prob(p.american)
        p.graded_at = graded_at
    return picks


# ---------------------------------------------------------------- tally


@dataclass
class Tally:
    n: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    priced: int = 0
    pnl_flat: float = 0.0
    clv_sum: float = 0.0
    clv_n: int = 0

    def add(self, p: Pick) -> None:
        if p.outcome is None:
            return
        self.n += 1
        if p.outcome == WIN:
            self.wins += 1
        elif p.outcome == PUSH:
            self.pushes += 1
        else:
            self.losses += 1
        if p.pnl_flat is not None:
            self.priced += 1
            self.pnl_flat += p.pnl_flat
        if p.clv is not None:
            self.clv_sum += p.clv
            self.clv_n += 1

    def record(self) -> str:
        s = f"{self.wins}-{self.losses}"
        return s + (f"-{self.pushes}" if self.pushes else "")

    def line(self) -> str:
        if not self.n:
            return "no graded picks yet"
        parts = [self.record()]
        if self.priced:
            parts.append(f"{self.pnl_flat:+.1f}u flat on {self.priced} priced")
        if self.clv_n:
            parts.append(f"CLV {100 * self.clv_sum / self.clv_n:+.1f} pts (n={self.clv_n})")
        return " · ".join(parts)


def tally(picks: list[Pick], by: str) -> dict[str, Tally]:
    out: dict[str, Tally] = {}
    for p in picks:
        key = p.host if by == "host" else p.market
        out.setdefault(key, Tally()).add(p)
    return {k: v for k, v in out.items() if v.n}


def render(picks: list[Pick]) -> str:
    total = Tally()
    for p in picks:
        total.add(p)
    lines = [f"Podcast picks (graded by us, incl. OT/SO): {total.line()}"]
    for k, t in sorted(tally(picks, "market").items()):
        lines.append(f"  {k:<11} {t.line()}")
    for k, t in sorted(tally(picks, "host").items(), key=lambda kv: -kv[1].n):
        lines.append(f"  {k:<15} {t.line()}")
    unp = sum(1 for p in picks if p.outcome is not None and p.american is None)
    if unp:
        lines.append(f"  ({unp} picks had no spoken price: W/L only, no P&L)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- storage


def picks_path(data_dir: Path, slate: Date) -> Path:
    return podcast_dir(data_dir) / f"picks_{slate.isoformat()}.json"


def records_path(data_dir: Path, slate: Date) -> Path:
    return podcast_dir(data_dir) / f"stated_{slate.isoformat()}.json"


def save_picks(picks: list[Pick], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in picks], indent=1))
    return path


def load_picks(path: Path) -> list[Pick]:
    if not path.exists():
        return []
    return [Pick(**d) for d in json.loads(path.read_text())]


def load_all_picks(data_dir: Path) -> list[Pick]:
    return [p for f in sorted(podcast_dir(data_dir).glob("picks_*.json")) for p in load_picks(f)]


def save_records(recs: list[StatedRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in recs], indent=1))
    return path


def load_records(path: Path) -> list[StatedRecord]:
    if not path.exists():
        return []
    return [StatedRecord(**d) for d in json.loads(path.read_text())]


__all__ = [
    "HOSTS",
    "UNATTRIBUTED",
    "Pick",
    "StatedRecord",
    "Tally",
    "extract_picks",
    "grade_picks",
    "load_all_picks",
    "load_picks",
    "load_records",
    "picks_path",
    "records_path",
    "render",
    "save_picks",
    "save_records",
    "stated_records",
    "tally",
]
