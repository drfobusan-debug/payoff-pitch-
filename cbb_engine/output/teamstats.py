"""Season-to-date team ratings and four factors from Torvik's game logs (display only).

Each row of ``getgamestats.php?csv=1`` is one team-game. Its box-score array holds
both teams' counts (FGM, FGA, 3PM, 3PA, FTM, FTA, OREB, DREB, REB, AST, STL, BLK,
TO, PF, PTS), so totals are pooled over possessions and shots rather than
averaging per-game rates. Off/def rating = points scored/allowed per 100
possessions, raw (not opponent-adjusted), split by Torvik's venue flag (H/A;
neutral games count only in the season columns). Four factors are over every
game: eFG% = (FGM + 0.5·3PM)/FGA, TOV% = TO/poss, OREB% = OREB/(OREB + opp DREB),
FT rate = FTA/FGA.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from pathlib import Path

from cbb_engine.history import _FOLD

_BOX = (
    "fgm",
    "fga",
    "tpm",
    "tpa",
    "ftm",
    "fta",
    "orb",
    "drb",
    "reb",
    "ast",
    "stl",
    "blk",
    "to",
    "pf",
    "pts",
)


@dataclass(frozen=True)
class Rated:
    value: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class Split:
    games: int = 0
    off: Rated = Rated()
    dfn: Rated = Rated()
    net: Rated = Rated()


@dataclass(frozen=True)
class Factors:
    efg: Rated = Rated()
    tov: Rated = Rated()
    orb: Rated = Rated()
    ftr: Rated = Rated()


@dataclass(frozen=True)
class TeamStats:
    team: str
    games: int
    home: Split
    away: Split
    offense: Factors
    defense: Factors
    adj_off: Rated = Rated()
    adj_def: Rated = Rated()
    last_game: Date | None = None


@dataclass
class _Acc:
    games: int = 0
    poss: float = 0.0
    own: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_BOX, 0.0))
    opp: dict[str, float] = field(default_factory=lambda: dict.fromkeys(_BOX, 0.0))

    def add(self, poss: float, own: Mapping[str, float], opp: Mapping[str, float]) -> None:
        self.games += 1
        self.poss += poss
        for k in _BOX:
            self.own[k] += own[k]
            self.opp[k] += opp[k]


def _box(raw: str, team: str) -> tuple[dict[str, float], dict[str, float], float] | None:
    try:
        b = json.loads(raw)
        first = dict(zip(_BOX, (float(x) for x in b[4:19]), strict=True))
        second = dict(zip(_BOX, (float(x) for x in b[19:34]), strict=True))
        poss = float(b[34])
    except (ValueError, TypeError, IndexError):
        return None
    if poss <= 0:
        return None
    if b[2] == team:
        return first, second, poss
    if b[3] == team:
        return second, first, poss
    return None


def _date(raw: str) -> Date | None:
    try:
        return datetime.strptime(raw, "%m/%d/%y").date()
    except ValueError:
        return None


def _ranks(values: Mapping[str, float | None], *, high_is_good: bool) -> dict[str, Rated]:
    """Competition ranks (1, 1, 3) among teams with a value."""
    have = sorted((v for v in values.values() if v is not None), reverse=high_is_good)
    first: dict[float, int] = {}
    for i, v in enumerate(have, 1):
        first.setdefault(v, i)
    return {t: Rated(v, first[v] if v is not None else None) for t, v in values.items()}


def _rating(a: _Acc) -> tuple[float | None, float | None]:
    if not a.poss:
        return None, None
    return 100 * a.own["pts"] / a.poss, 100 * a.opp["pts"] / a.poss


def _four(
    own: Mapping[str, float], opp: Mapping[str, float], poss: float
) -> dict[str, float | None]:
    fga, reb_chances = own["fga"], own["orb"] + opp["drb"]
    return {
        "efg": (own["fgm"] + 0.5 * own["tpm"]) / fga if fga else None,
        "tov": own["to"] / poss if poss else None,
        "orb": own["orb"] / reb_chances if reb_chances else None,
        "ftr": own["fta"] / fga if fga else None,
    }


def season(
    logs: Iterable[list[str]], before: Date, adjusted: Iterable[Mapping[str, str]] = ()
) -> dict[str, TeamStats]:
    """Every team's stats from games dated before ``before``; ranks are among all D-I teams."""
    acc: dict[str, dict[str, _Acc]] = defaultdict(lambda: {"all": _Acc(), "H": _Acc(), "A": _Acc()})
    last: dict[str, Date] = {}
    for row in logs:
        if len(row) < 30:
            continue
        day = _date(row[0])
        if day is None or day >= before:
            continue
        team, venue = row[2], row[5]
        parsed = _box(row[29], team)
        if parsed is None:
            continue
        own, opp, poss = parsed
        acc[team]["all"].add(poss, own, opp)
        if venue in ("H", "A"):
            acc[team][venue].add(poss, own, opp)
        last[team] = max(last.get(team, day), day)
    teams = sorted(acc)
    splits: dict[str, dict[str, dict[str, Rated]]] = {}
    for where in ("H", "A"):
        rt = {t: _rating(acc[t][where]) for t in teams}
        off = _ranks({t: r[0] for t, r in rt.items()}, high_is_good=True)
        dfn = _ranks({t: r[1] for t, r in rt.items()}, high_is_good=False)
        net = _ranks(
            {
                t: (r[0] - r[1]) if r[0] is not None and r[1] is not None else None
                for t, r in rt.items()
            },
            high_is_good=True,
        )
        splits[where] = {t: {"off": off[t], "dfn": dfn[t], "net": net[t]} for t in teams}
    off4 = {t: _four(acc[t]["all"].own, acc[t]["all"].opp, acc[t]["all"].poss) for t in teams}
    def4 = {t: _four(acc[t]["all"].opp, acc[t]["all"].own, acc[t]["all"].poss) for t in teams}
    good = {"efg": True, "tov": False, "orb": True, "ftr": True}
    o_rk = {k: _ranks({t: off4[t][k] for t in teams}, high_is_good=g) for k, g in good.items()}
    d_rk = {k: _ranks({t: def4[t][k] for t in teams}, high_is_good=not g) for k, g in good.items()}
    adj = {r.get("team", ""): r for r in adjusted}
    a_off = _ranks({t: _num(adj.get(t, {}).get("adjoe")) for t in teams}, high_is_good=True)
    a_def = _ranks({t: _num(adj.get(t, {}).get("adjde")) for t in teams}, high_is_good=False)
    return {
        t: TeamStats(
            team=t,
            games=acc[t]["all"].games,
            home=Split(acc[t]["H"].games, **splits["H"][t]),
            away=Split(acc[t]["A"].games, **splits["A"][t]),
            offense=Factors(**{k: o_rk[k][t] for k in good}),
            defense=Factors(**{k: d_rk[k][t] for k in good}),
            adj_off=a_off[t],
            adj_def=a_def[t],
            last_game=last.get(t),
        )
        for t in teams
    }


def _num(raw: str | None) -> float | None:
    try:
        return float(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def read_logs(path: Path | None) -> list[list[str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.reader(fh))


def _tokens(name: str) -> frozenset[str]:
    out = set()
    for raw in (
        name.lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace(".", " ")
        .replace("-", " ")
        .replace("&", " ")
        .replace("'", "")
        .split()
    ):
        tok = _FOLD.get(raw, raw)
        tok = "".join(ch for ch in tok if ch.isalnum())
        if tok:
            out.add(tok)
    return frozenset(out)


# Full names (Odds API / ESPN "School Mascot") whose school Torvik spells differently, or
# whose school name contains another school's ("Kansas City" ⊃ "Kansas"): matched on
# folded tokens, checked before the subset rule.
ALIASES: dict[str, str] = {
    "App State Mountaineers": "Appalachian St.",
    "Grambling Tigers": "Grambling St.",
    "IU Indianapolis Jaguars": "IU Indy",
    "Long Island University Sharks": "LIU",
    "McNeese Cowboys": "McNeese St.",
    "Miami Hurricanes": "Miami FL",
    "Miami (OH) RedHawks": "Miami OH",
    "NC State Wolfpack": "N.C. State",
    "Nicholls Colonels": "Nicholls St.",
    "Ole Miss Rebels": "Mississippi",
    "Omaha Mavericks": "Nebraska Omaha",
    "Pennsylvania Quakers": "Penn",
    "San José State Spartans": "San Jose St.",
    "San Jose St Spartans": "San Jose St.",
    "UAlbany Great Danes": "Albany",
    "UConn Huskies": "Connecticut",
    "UIC Flames": "Illinois Chicago",
    "UL Monroe Warhawks": "Louisiana Monroe",
    "UT Martin Skyhawks": "Tennessee Martin",
    "California Baptist Lancers": "Cal Baptist",
    "Florida International Panthers": "FIU",
    "West Florida Argonauts": "West Florida",
    "Sam Houston Bearkats": "Sam Houston St.",
    "Kansas City Roos": "UMKC",
    "SE Louisiana Lions": "Southeastern Louisiana",
    "Loyola Maryland Greyhounds": "Loyola MD",
    "South Carolina Upstate Spartans": "USC Upstate",
    "Texas A&M-Corpus Christi Islanders": "Texas A&M Corpus Chris",
    "Texas A&M-CC Islanders": "Texas A&M Corpus Chris",
}


def match_team(name: str, torvik_teams: Iterable[str]) -> str | None:
    """Torvik's name for an Odds API / ESPN name ("Michigan St Spartans" -> "Michigan St.").

    ``ALIASES`` first; otherwise the Torvik name must be a whole-token subset of the
    full name, the longest such name wins and a tie is no match.
    """
    full = _tokens(name)
    teams = list(torvik_teams)
    alias = next((t for k, t in ALIASES.items() if _tokens(k) == full), None)
    if alias is not None:
        return alias if alias in teams else None
    torvik_teams = teams
    hits = [(len(tt), t) for t in torvik_teams if (tt := _tokens(t)) and tt <= full]
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


__all__ = ["Factors", "Rated", "Split", "TeamStats", "match_team", "read_logs", "season"]
