"""Pull each college-football bet out of a transcript, and check it was said.

A language model reads the transcript in overlapping windows and returns every
CFB bet or lean as structured JSON with a verbatim quote and its timestamp.
Nothing it returns is trusted on its own: a pick survives only if its quote is
found in the transcript within a few minutes of the stamp, and a line or price
is kept only if that number is also spoken there. A dropped number leaves the
pick ungraded on that market rather than graded at a number nobody said.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path

import requests

from cfb_engine.podcast.episodes import Episode
from engine_common.podcasts.transcript import Line, stamp_seconds

log = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "cfb-picks-v1"
WINDOW_SECONDS = 20 * 60
OVERLAP_SECONDS = 2 * 60
QUOTE_SLACK_SECONDS = 180
QUOTE_MIN_RATIO = 0.8

MARKETS = ("game_ml", "game_ats", "game_total", "other")
KINDS = ("official", "lean")

SYSTEM_PROMPT = """You read sports-betting podcast transcripts and list the COLLEGE FOOTBALL bets the hosts make.
Each transcript line starts with its [mm:ss] or [h:mm:ss] timestamp. There is no speaker labelling.

Return every college-football pick in this window:
- kind "official": the speaker says they are betting it, it is a best bet, a play, "I'm on", "I bet", "give me", a pick they make for the record, or units are stated.
- kind "lean": a softer opinion on a side or total -- "I lean", "I like but not betting", "if I had to pick", lookahead thoughts.
Do NOT list: passes, games they discuss without taking a side, futures/awards/season win totals (unless market "other"), NFL/NBA/MLB/NHL or any non-college sport, a recap of a bet already graded.

Fields:
- host: the person making the pick, ONLY if the transcript makes it explicit (named handoff "Colby, what do you have?" or self-identification); otherwise null. Use a name from the provided host list when it matches.
- team: for moneyline/spread, the school picked; for a total, either school in the game. Use the school's name, not a mascot or abbreviation ("Southern Miss" not "USM", "Miami (OH)" not "the RedHawks", "Ole Miss" is fine).
- opponent: the other school in the game when said or obvious from the discussion; null otherwise.
- market: "game_ml" (moneyline / to win), "game_ats" (spread), "game_total" (full-game over/under), "other" (team totals, halves, props, parlays, futures).
- side: "over" or "under" for totals; null otherwise.
- line: the number as spoken, from the picked team's side for spreads (dog +7 -> 7, favorite -3.5 -> -3.5); the total for totals; null if no number is said. Never supply a number the speaker did not say.
- price: American odds if said (-110, +150); null otherwise.
- units: stake in units if said; null otherwise.
- stamp: the timestamp of the line where the pick is made, copied from the transcript.
- quote: an exact contiguous excerpt (8-40 words) copied from the transcript where the pick is made.
- reason: one sentence on why, in your words.
- description: the pick as a bettor would write it ("Ole Miss -3.5", "Under 54.5 Troy-Southern Miss").
If there are no picks, return an empty list."""

_NULL_STR = {"type": ["string", "null"]}
_NULL_NUM = {"type": ["number", "null"]}
SCHEMA = {
    "name": "cfb_picks",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["picks"],
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "kind",
                        "host",
                        "team",
                        "opponent",
                        "market",
                        "side",
                        "line",
                        "price",
                        "units",
                        "stamp",
                        "quote",
                        "reason",
                        "description",
                    ],
                    "properties": {
                        "kind": {"type": "string", "enum": list(KINDS)},
                        "host": _NULL_STR,
                        "team": _NULL_STR,
                        "opponent": _NULL_STR,
                        "market": {"type": "string", "enum": list(MARKETS)},
                        "side": {"type": ["string", "null"], "enum": ["over", "under", None]},
                        "line": _NULL_NUM,
                        "price": _NULL_NUM,
                        "units": _NULL_NUM,
                        "stamp": {"type": "string"},
                        "quote": {"type": "string"},
                        "reason": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            }
        },
    },
}


@dataclass(frozen=True)
class Pick:
    """One bet or lean a podcast made, as it was said."""

    pick_id: str
    show: str  # Show.key
    show_name: str
    host: str | None
    episode_guid: str
    episode_title: str
    published: str  # ISO8601 with offset
    kind: str  # "official" | "lean"
    market: str  # "game_ml" | "game_ats" | "game_total" | "other"
    team: str | None
    opponent: str | None
    side: str | None
    line: float | None
    price: float | None
    units: float | None
    seconds: int
    quote: str
    reason: str
    description: str

    @property
    def stamp(self) -> str:
        return _stamp(self.seconds)


@dataclass
class Extraction:
    """What one episode yielded: kept picks, and the ones the checks refused."""

    episode_guid: str
    model: str
    prompt_version: str
    transcript_sha: str
    picks: list[Pick] = field(default_factory=list)
    rejected: list[dict[str, object]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "episode_guid": self.episode_guid,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "transcript_sha": self.transcript_sha,
                "picks": [asdict(p) for p in self.picks],
                "rejected": self.rejected,
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> Extraction:
        d = json.loads(text)
        return cls(
            episode_guid=d["episode_guid"],
            model=d["model"],
            prompt_version=d["prompt_version"],
            transcript_sha=d["transcript_sha"],
            picks=[Pick(**p) for p in d["picks"]],
            rejected=list(d.get("rejected", [])),
        )


# --------------------------------------------------------------- windows


def _stamp(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def windows(lines: list[Line]) -> list[list[Line]]:
    """Overlapping ~20-minute slices, so a pick at a boundary is seen whole."""
    if not lines:
        return []
    out: list[list[Line]] = []
    start = lines[0].seconds
    end = lines[-1].seconds
    while start <= end:
        stop = start + WINDOW_SECONDS
        chunk = [ln for ln in lines if start - OVERLAP_SECONDS <= ln.seconds < stop]
        if chunk:
            out.append(chunk)
        start = stop
    return out


def render_window(lines: list[Line]) -> str:
    return "\n".join(f"[{_stamp(ln.seconds)}] {ln.text}" for ln in lines)


# --------------------------------------------------------------- checks

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("½", " and a half"))


def _near(lines: list[Line], seconds: int, slack: int = QUOTE_SLACK_SECONDS) -> list[Line]:
    return [ln for ln in lines if abs(ln.seconds - seconds) <= slack]


def quote_found(quote: str, lines: list[Line], seconds: int) -> bool:
    """Whether ``quote`` was said within a few minutes of ``seconds``.

    Word-level, so punctuation and casing the model tidied do not matter; at
    least 80% of the quote's words must appear in order in one stretch.
    """
    want = _words(quote)
    if len(want) < 4:
        return False
    have = [w for ln in _near(lines, seconds) for w in _words(ln.text)]
    if not have:
        return False
    sm = SequenceMatcher(None, want, have, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / len(want) >= QUOTE_MIN_RATIO


_UNITS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


def _spoken(n: int) -> list[str]:
    """English words for ``0 <= n < 100`` (Whisper writes some numbers out)."""
    if n in _UNITS:
        return [_UNITS[n]]
    tens, ones = divmod(n, 10)
    if tens in _TENS:
        return [_TENS[tens]] if ones == 0 else [_TENS[tens], _UNITS[ones]]
    return [str(n)]


def _number_forms(value: float) -> list[list[str]]:
    """Word sequences a transcript may use for a spread, total or price."""
    mag = abs(value)
    whole = int(mag)
    half = abs(mag - whole - 0.5) < 1e-9
    forms: list[list[str]] = []
    if half:
        forms += [[str(whole), "5"], [str(whole), "and", "a", "half"]]
        forms += [_spoken(whole) + ["and", "a", "half"], _spoken(whole) + ["point", "five"]]
        if whole == 0:
            forms.append(["pick", "em"])
    elif mag == whole:
        forms += [[str(whole)], _spoken(whole)] if whole < 100 else [[str(whole)]]
        if whole >= 100:
            hundreds, rest = divmod(whole, 100)
            forms.append(_spoken(hundreds) + ["hundred"] + (_spoken(rest) if rest else []))
            forms.append(_spoken(hundreds) + _spoken(rest) if rest else _spoken(hundreds))
        if whole == 0:
            forms += [["pick", "em"], ["pickem"], ["pk"]]
    else:
        forms.append(_words(f"{mag:g}"))
    return forms


def number_said(value: float, lines: list[Line], seconds: int) -> bool:
    """Whether ``value`` is spoken near ``seconds`` in any usual form."""
    have = [w for ln in _near(lines, seconds) for w in _words(ln.text)]
    for form in _number_forms(value):
        n = len(form)
        if any(have[i : i + n] == form for i in range(len(have) - n + 1)):
            return True
    return False


# --------------------------------------------------------------- model call


def _pick_id(show: str, guid: str, host: str | None, market: str, who: str, kind: str) -> str:
    raw = "|".join((show, guid, host or "", market, who.lower(), kind))
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def call_model(
    window_text: str,
    ep: Episode,
    *,
    api_key: str,
    model: str,
    session: requests.Session | None = None,
) -> list[dict[str, object]]:
    sess = session or requests.Session()
    user = (
        f"Show: {ep.show.name}\nHosts: {', '.join(ep.show.hosts)}\n"
        f"Episode: {ep.item.title}\nPublished: {ep.item.published}\n\nTranscript:\n{window_text}"
    )
    resp = sess.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": model,
                "temperature": 0,
                "response_format": {"type": "json_schema", "json_schema": SCHEMA},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }
        ),
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    picks = json.loads(content)["picks"]
    return [p for p in picks if isinstance(p, dict)]


def _float(v: object) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _str(v: object) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def check(raw: dict[str, object], ep: Episode, lines: list[Line]) -> tuple[Pick | None, str]:
    """A raw model pick as a :class:`Pick`, or ``None`` and why it was refused."""
    kind, market = raw.get("kind"), raw.get("market")
    if kind not in KINDS or market not in MARKETS:
        return None, "bad kind/market"
    seconds = stamp_seconds(str(raw.get("stamp", "")))
    if seconds is None:
        return None, "bad stamp"
    quote = _str(raw.get("quote")) or ""
    if not quote_found(quote, lines, seconds):
        return None, "quote not in transcript near stamp"
    team, side = _str(raw.get("team")), _str(raw.get("side"))
    if market != "other" and team is None:
        return None, "no team"
    if market == "game_total" and side not in ("over", "under"):
        return None, "total without over/under"
    line, price = _float(raw.get("line")), _float(raw.get("price"))
    if line is not None and not number_said(line, lines, seconds):
        line = None
    if price is not None and (abs(price) < 100 or not number_said(price, lines, seconds)):
        price = None
    host = _str(raw.get("host"))
    if host is not None:
        known = [h for h in ep.show.hosts if host.lower() in h.lower() or h.lower() in host.lower()]
        host = known[0] if len(known) == 1 else host
    who = f"{team}:{side}" if market == "game_total" else str(team)
    pick = Pick(
        pick_id=_pick_id(ep.show.key, ep.item.guid, host, str(market), who, str(kind)),
        show=ep.show.key,
        show_name=ep.show.name,
        host=host,
        episode_guid=ep.item.guid,
        episode_title=ep.item.title,
        published=ep.item.published,
        kind=str(kind),
        market=str(market),
        team=team,
        opponent=_str(raw.get("opponent")),
        side=side if market == "game_total" else None,
        line=line,
        price=price,
        units=_float(raw.get("units")),
        seconds=seconds,
        quote=quote,
        reason=_str(raw.get("reason")) or "",
        description=_str(raw.get("description")) or "",
    )
    return pick, ""


def _merge(picks: list[Pick]) -> list[Pick]:
    """One pick per id (overlapping windows repeat one), keeping the most specific."""
    best: dict[str, Pick] = {}
    for p in picks:
        held = best.get(p.pick_id)
        if held is None:
            best[p.pick_id] = p
            continue
        filled = replace(
            held,
            line=held.line if held.line is not None else p.line,
            price=held.price if held.price is not None else p.price,
            units=held.units if held.units is not None else p.units,
            opponent=held.opponent or p.opponent,
        )
        best[p.pick_id] = filled
    out = sorted(best.values(), key=lambda p: p.seconds)
    official = {(p.show, p.host, p.market, p.team, p.side) for p in out if p.kind == "official"}
    return [
        p
        for p in out
        if p.kind == "official" or (p.show, p.host, p.market, p.team, p.side) not in official
    ]


def picks_path(audit_dir: Path, ep: Episode) -> Path:
    return audit_dir / "podcast_picks" / f"{ep.show.key}__{ep.key}.json"


def extract(
    ep: Episode,
    lines: list[Line],
    audit_dir: Path,
    *,
    api_key: str,
    model: str | None = None,
    session: requests.Session | None = None,
) -> Extraction:
    """The episode's checked picks, from cache when the transcript and prompt are unchanged."""
    use_model = model or os.getenv("CFBE_PODCAST_MODEL") or DEFAULT_MODEL
    sha = hashlib.sha1(render_window(lines).encode()).hexdigest()
    path = picks_path(audit_dir, ep)
    if path.exists():
        cached = Extraction.from_json(path.read_text())
        if cached.transcript_sha == sha and cached.prompt_version == PROMPT_VERSION:
            return cached
    kept: list[Pick] = []
    rejected: list[dict[str, object]] = []
    for chunk in windows(lines):
        for raw in call_model(
            render_window(chunk), ep, api_key=api_key, model=use_model, session=session
        ):
            pick, why = check(raw, ep, lines)
            if pick is None:
                rejected.append({**raw, "why": why})
            else:
                kept.append(pick)
    result = Extraction(ep.item.guid, use_model, PROMPT_VERSION, sha, _merge(kept), rejected)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json())
    return result


def load_extractions(audit_dir: Path) -> list[Extraction]:
    folder = audit_dir / "podcast_picks"
    return [Extraction.from_json(p.read_text()) for p in sorted(folder.glob("*.json"))]


__all__ = [
    "DEFAULT_MODEL",
    "Extraction",
    "Pick",
    "PROMPT_VERSION",
    "call_model",
    "check",
    "extract",
    "load_extractions",
    "number_said",
    "picks_path",
    "quote_found",
    "render_window",
    "windows",
]
