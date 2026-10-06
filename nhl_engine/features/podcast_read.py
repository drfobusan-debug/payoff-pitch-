"""Per-game read of a Hockey Gambling Podcast transcript.

Two layers, kept apart on the page:

* **deterministic** -- the hosts open every game with a fixed intro
  ("down to seven o'clock, the Utah Mammoth +114 at the New Jersey Devils
  -135 ... over 5.5 -135 under 5.5 +114"); that anchor splits the transcript
  into game segments, gives the lines they were quoting, and every stake /
  price / side mention inside the segment is kept verbatim with its timestamp.
  Team names are matched on nicknames with a fuzzy fallback restricted to the
  slate's teams, so a Whisper "Buffalo Savers" still lands on BUF.
* **summary** -- optional: an OpenAI call turns a segment into "bet · who/stake
  · reasoning" lines. Absent key or a failed call leaves the summary empty; the
  deterministic layer is always shown. Summaries are cached per segment so a
  re-priced card never pays twice.

Nothing here touches a price or gate: the read is context under the game.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date as Date
from pathlib import Path

import requests

from nhl_engine.data.podcast import Episode, Line, podcast_dir
from nhl_engine.data.teamnames import CODES

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
MAX_MENTIONS = 10

NICKNAMES: dict[str, tuple[str, ...]] = {
    "ANA": ("ducks", "anaheim"),
    "BOS": ("bruins", "boston"),
    "BUF": ("sabres", "buffalo", "savers", "sabers"),
    "CGY": ("flames", "calgary"),
    "CAR": ("hurricanes", "canes", "carolina"),
    "CHI": ("blackhawks", "black hawks", "hawks", "chicago", "black ox"),
    "COL": ("avalanche", "avs", "colorado"),
    "CBJ": ("blue jackets", "jackets", "columbus"),
    "DAL": ("stars", "dallas"),
    "DET": ("red wings", "wings", "detroit"),
    "EDM": ("oilers", "edmonton"),
    "FLA": ("panthers", "florida"),
    "LAK": ("kings", "los angeles"),
    "MIN": ("wild", "minnesota"),
    "MTL": ("canadiens", "canadians", "habs", "montreal"),
    "NSH": ("predators", "preds", "nashville"),
    "NJD": ("devils", "new jersey"),
    "NYI": ("islanders", "isles"),
    "NYR": ("rangers",),
    "OTT": ("senators", "sens", "ottawa"),
    "PHI": ("flyers", "philadelphia", "philly"),
    "PIT": ("penguins", "pens", "pittsburgh"),
    "SJS": ("sharks", "san jose"),
    "SEA": ("kraken", "seattle"),
    "STL": ("blues", "st louis", "saint louis"),
    "TBL": ("lightning", "bolts", "tampa"),
    "TOR": ("maple leafs", "leafs", "leaves", "maple leaves", "toronto"),
    "UTA": ("mammoth", "utah"),
    "VAN": ("canucks", "vancouver"),
    "VGK": ("golden knights", "knights", "vegas"),
    "WSH": ("capitals", "caps", "washington"),
    "WPG": ("jets", "winnipeg"),
}
assert set(NICKNAMES) == set(CODES)
# Short/common words that must match exactly, never fuzzily.
_EXACT_ONLY = frozenset({"wild", "stars", "jets", "kings", "wings", "avs", "caps", "pens", "sens"})
FUZZY_CUTOFF = 0.84

_PRICE = re.compile(r"\b(plus|minus)\s*(\d{3})\b", re.I)
_TOTAL = re.compile(
    r"\b(over|under)\s+(four|five|six|seven|eight|\d)(?:\s+(?:and\s+a\s+)?half|\.5|\s+nap)?", re.I
)
_STAKE = re.compile(r"\b(\d{1,3})\s*(?:puck|buck|pup)\s*(?:bucks|box|books|bux)\b", re.I)
_BET_WORDS = re.compile(
    r"\b(puck line|team total|moneyline|money line|i('| a)?m taking|i took|i bet|give me|"
    r"i('| wi)?ll take|units?|sprinkle|lean|pass on|no bet|stay away)\b",
    re.I,
)
_WORDNUM = {"four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", "")).strip()


def _american(sign: str, num: str) -> int:
    return int(num) if sign.lower() == "plus" else -int(num)


def teams_in(text: str, slate_codes: set[str]) -> list[str]:
    """Slate teams named in ``text`` (exact nickname/city first, fuzzy single word second)."""
    t = f" {_norm(text)} "
    hits: list[str] = []
    for code in slate_codes:
        if any(f" {alias} " in t for alias in NICKNAMES[code]):
            hits.append(code)
    if len(hits) >= 2:
        return hits
    singles = {
        alias: code
        for code in slate_codes
        for alias in NICKNAMES[code]
        if " " not in alias and alias not in _EXACT_ONLY and code not in hits
    }
    for word in t.split():
        if len(word) < 5:
            continue
        close = difflib.get_close_matches(word, list(singles), n=1, cutoff=FUZZY_CUTOFF)
        if close and singles[close[0]] not in hits:
            hits.append(singles[close[0]])
    return hits


@dataclass(frozen=True)
class Mention:
    stamp: str
    text: str


@dataclass
class GameRead:
    matchup: str
    anchor: str  # mm:ss of the intro line
    quoted_ml: dict[str, int] = field(default_factory=dict)  # code -> american
    quoted_total: float | None = None
    mentions: list[Mention] = field(default_factory=list)
    excerpt: str = ""  # full segment text (LLM input; not rendered)
    summary: list[str] = field(default_factory=list)
    summary_source: str = ""  # "openai:<model>" | "" when none


@dataclass
class PodcastRead:
    slate_date: str
    status: str  # ok | no_episode | no_transcript
    detail: str = ""
    episode_title: str = ""
    published: str = ""
    games: dict[str, GameRead] = field(default_factory=dict)
    not_on_slate: list[str] = field(default_factory=list)  # anchors for games we don't price

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, raw: str) -> PodcastRead:
        d = json.loads(raw)
        games = {
            k: GameRead(**{**g, "mentions": [Mention(**m) for m in g["mentions"]]})
            for k, g in d.pop("games").items()
        }
        return cls(**d, games=games)


def _slate_games(matchups: list[tuple[str, str]]) -> dict[frozenset[str], str]:
    return {frozenset((a, h)): f"{a} @ {h}" for a, h in matchups}


def _anchor(line: Line, games: dict[frozenset[str], str], codes: set[str]) -> str | None:
    """A game intro: both teams named and a price quoted on the same line."""
    if not _PRICE.search(line.text):
        return None
    found = teams_in(line.text, codes)
    if len(found) < 2:
        return None
    return games.get(frozenset(found[:2]))


def _quoted_ml(line: Line, codes: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    t = _norm(line.text)
    for m in _PRICE.finditer(t):
        before = t[: m.start()]
        who = teams_in(before[-40:], codes)
        if who and who[-1] not in out:
            out[who[-1]] = _american(m.group(1), m.group(2))
    return out


def _quoted_total(lines: list[Line]) -> float | None:
    for ln in lines:
        m = _TOTAL.search(ln.text)
        if m:
            n = _WORDNUM.get(m.group(2).lower()) or int(m.group(2))
            half = bool(re.search(r"half|\.5|nap", m.group(0), re.I))
            return n + (0.5 if half else 0.0)
    return None


def _is_mention(text: str) -> bool:
    return bool(_STAKE.search(text) or _PRICE.search(text) or _BET_WORDS.search(text))


def segment(
    lines: list[Line], matchups: list[tuple[str, str]]
) -> tuple[dict[str, GameRead], list[str]]:
    """Split the transcript at game intros; returns slate reads and intro lines for
    games the slate does not carry (an episode covering two nights)."""
    games = _slate_games(matchups)
    codes = {c for pair in matchups for c in pair}
    anchors: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        hit = _anchor(ln, games, codes)
        if hit and (not anchors or anchors[-1][1] != hit):
            anchors.append((i, hit))
    reads: dict[str, GameRead] = {}
    for n, (start, matchup) in enumerate(anchors):
        end = anchors[n + 1][0] if n + 1 < len(anchors) else len(lines)
        seg = lines[start:end]
        read = reads.get(matchup) or GameRead(matchup=matchup, anchor=lines[start].stamp)
        read.quoted_ml = read.quoted_ml or _quoted_ml(lines[start], codes)
        read.quoted_total = read.quoted_total or _quoted_total(seg[:3])
        read.mentions.extend(Mention(ln.stamp, ln.text) for ln in seg[1:] if _is_mention(ln.text))
        read.mentions = read.mentions[:MAX_MENTIONS]
        read.excerpt = (read.excerpt + "\n" if read.excerpt else "") + "\n".join(
            f"[{ln.stamp}] {ln.text}" for ln in seg
        )
        reads[matchup] = read
    # Intros that name two NHL teams with a price but are not a slate game.
    others: list[str] = []
    all_codes = set(CODES)
    for ln in lines:
        if _PRICE.search(ln.text) and "oclock" in _norm(ln.text).replace(" ", ""):
            found = teams_in(ln.text, all_codes)
            if len(found) >= 2 and frozenset(found[:2]) not in games:
                others.append(f"{found[0]}/{found[1]} @ {ln.stamp}")
    return reads, others


# ---------------------------------------------------------------- summary

SYSTEM_PROMPT = (
    "You summarise a hockey betting podcast transcript segment for one NHL game. "
    "Whisper transcription: team and player names may be garbled; prices are spoken "
    "as 'plus 114' / 'minus 135'; stakes are 'puck bucks'. Speakers are not labelled -- "
    "name a host only when the text itself does. Output 2-6 short lines, each "
    "'BET — stake/who if stated — reasoning as given'. Include passes/leans as such. "
    "Only report what is actually said; if no bet is stated, output exactly 'no bet stated'. "
    "No preamble, no advice of your own."
)


def _summary_cache(data_dir: Path) -> Path:
    return podcast_dir(data_dir) / "summaries.json"


def _load_cache(path: Path) -> dict[str, list[str]]:
    if path.exists():
        data = json.loads(path.read_text())
        return {k: list(v) for k, v in data.items()}
    return {}


def summarize_segment(
    excerpt: str,
    matchup: str,
    *,
    api_key: str,
    model: str = OPENAI_MODEL,
    session: requests.Session | None = None,
) -> list[str]:
    sess = session or requests.Session()
    resp = sess.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Game: {matchup}\n\nTranscript:\n{excerpt}"},
            ],
        },
        timeout=90,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return [ln.strip(" -•*") for ln in text.splitlines() if ln.strip()]


def add_summaries(
    read: PodcastRead,
    data_dir: Path,
    *,
    api_key: str | None,
    model: str = OPENAI_MODEL,
    session: requests.Session | None = None,
) -> int:
    """Fill ``summary`` for every game read (cached by excerpt hash). Returns new calls."""
    if not api_key:
        return 0
    path = _summary_cache(data_dir)
    cache = _load_cache(path)
    calls = 0
    for g in read.games.values():
        if not g.excerpt:
            continue
        key = f"{model}:{hashlib.sha1(g.excerpt.encode()).hexdigest()}"
        if key not in cache:
            try:
                cache[key] = summarize_segment(
                    g.excerpt, g.matchup, api_key=api_key, model=model, session=session
                )
            except (requests.RequestException, KeyError, ValueError):
                continue
            calls += 1
        g.summary = cache[key]
        g.summary_source = f"openai:{model}"
    if calls:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    return calls


# ---------------------------------------------------------------- assembly


def build_read(
    ep: Episode, lines: list[Line], slate: Date, matchups: list[tuple[str, str]]
) -> PodcastRead:
    games, others = segment(lines, matchups)
    return PodcastRead(
        slate_date=slate.isoformat(),
        status="ok",
        episode_title=ep.title,
        published=ep.published,
        games=games,
        not_on_slate=others,
    )


def read_path(data_dir: Path, slate: Date) -> Path:
    return podcast_dir(data_dir) / f"read_{slate.isoformat()}.json"


def save_read(read: PodcastRead, data_dir: Path) -> Path:
    p = read_path(data_dir, Date.fromisoformat(read.slate_date))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(read.to_json())
    return p


def load_read(data_dir: Path, slate: Date) -> PodcastRead | None:
    p = read_path(data_dir, slate)
    return PodcastRead.from_json(p.read_text()) if p.exists() else None


__all__ = [
    "GameRead",
    "Mention",
    "NICKNAMES",
    "PodcastRead",
    "add_summaries",
    "build_read",
    "load_read",
    "read_path",
    "save_read",
    "segment",
    "summarize_segment",
    "teams_in",
]
