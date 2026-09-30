"""Preseason prior asset (§5.1): ``preseason_prior_<season>.json``.

On opening night every team rate has zero exposure, so the EB posterior *is*
the prior. This module builds it from what is measurable today and records what
is not, so the card can say which:

1. **Prior-season rate regressed to league** (``prior_season_regressed``) --
   ``league + r_yy * (rate - league)`` per metric, where ``r_yy`` is the fitted
   year-to-year correlation from ``scripts/nhl/reliability_study.py`` (quoted in
   ``config.PriorParams``). A relocated franchise reads its predecessor
   (UTA <- ARI).
2. **Roster carry-over** (``roster_carryover``) -- *deferred*: needs the §5.7
   isolated-impact lineup rebuild (``features/lineup.py``). Recorded as
   ``not_fitted``; the prior is the regressed rate alone until it exists.
3. **Futures-implied strength** (``futures``) -- Cup-winner outrights are
   fetched (1 credit), devigged per book and averaged, and *stored* in the
   asset. They do not move the prior yet: mapping a title probability to a goal
   differential needs a fit against historical outrights, which we do not have
   (the Odds API historical endpoint is a paid add-on; Phase 1 open decision).
   Weight is 0 and the asset says so.

The asset is versioned (``version`` = build timestamp) so every downstream
record can carry the prior it was priced from. Refresh weekly through October,
then freeze: ``refresh_if_stale`` only rebuilds before ``freeze_after``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine_common.odds import devig
from nhl_engine.data.moneypuck import FRANCHISE_PREDECESSOR, MoneyPuckClient
from nhl_engine.data.teamnames import CODES
from nhl_engine.features.strength import METRICS, game_table, league_rates, rate

log = logging.getLogger(__name__)

SCHEMA = 1


@dataclass(frozen=True)
class TeamPrior:
    code: str
    source_code: str
    source_season: int
    source_games: int
    rates: dict[str, float]
    raw_prior_season: dict[str, float]
    cup_prob: float | None = None


@dataclass(frozen=True)
class PreseasonPrior:
    season: int
    version: str
    built_on: str
    league: dict[str, float]
    r_yy: dict[str, float]
    teams: dict[str, TeamPrior]
    components: dict[str, str] = field(default_factory=dict)
    futures_books: int = 0
    schema: int = SCHEMA

    def rates_for(self, code: str) -> dict[str, float]:
        """The prior for one team, or the league rate when the team is unknown."""
        team = self.teams.get(code)
        return dict(team.rates) if team is not None else dict(self.league)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True, default=float)

    @classmethod
    def from_json(cls, text: str) -> PreseasonPrior:
        raw = json.loads(text)
        teams = {k: TeamPrior(**v) for k, v in raw.pop("teams").items()}
        return cls(teams=teams, **raw)


def asset_path(folder: Path, season: int) -> Path:
    return folder / f"preseason_prior_{season}.json"


def build(
    mp: MoneyPuckClient,
    season: int,
    *,
    r_yy: dict[str, float],
    futures: dict[str, dict[str, float]] | None = None,
    codes: frozenset[str] = CODES,
    today: Date | None = None,
) -> PreseasonPrior:
    """Assemble the prior for ``season`` from the previous season's MoneyPuck logs."""
    prev = season - 1
    tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for code in sorted(codes):
        src = code
        games = mp.team_games(code)
        g = games[games["season"] == prev] if not games.empty else games
        if g.empty and code in FRANCHISE_PREDECESSOR:
            src = FRANCHISE_PREDECESSOR[code]
            games = mp.team_games(src)
            g = games[games["season"] == prev] if not games.empty else games
        if g.empty:
            log.warning("preseason prior: no %d games for %s; league rate used", prev, code)
            continue
        tables[code] = game_table(g)
        sources[code] = src

    league = league_rates(tables)
    cup = futures_probs(futures) if futures else {}
    teams: dict[str, TeamPrior] = {}
    for code, table in tables.items():
        raw: dict[str, float] = {}
        regressed: dict[str, float] = {}
        for m in METRICS:
            observed, exposure = rate(table, m.key)
            lg = league[m.key]
            if exposure <= 0 or observed != observed:
                raw[m.key] = lg
                regressed[m.key] = lg
                continue
            raw[m.key] = observed
            w = r_yy.get(m.key, 0.0)
            regressed[m.key] = lg + w * (observed - lg)
        teams[code] = TeamPrior(
            code=code,
            source_code=sources[code],
            source_season=prev,
            source_games=int(len(table)),
            rates=regressed,
            raw_prior_season=raw,
            cup_prob=cup.get(code),
        )

    now = datetime.now(timezone.utc)
    return PreseasonPrior(
        season=season,
        version=now.strftime("%Y%m%dT%H%M%SZ"),
        built_on=(today or now.date()).isoformat(),
        league=league,
        r_yy={m.key: r_yy.get(m.key, 0.0) for m in METRICS},
        teams=teams,
        components={
            "prior_season_regressed": "applied (weight r_yy per metric)",
            "roster_carryover": "not_fitted (needs features/lineup.py, §5.7)",
            "futures": (
                "recorded, weight 0 (no historical outrights to fit the map)"
                if cup
                else "not_available"
            ),
        },
        futures_books=_book_count(futures),
    )


def futures_probs(futures: dict[str, dict[str, float]]) -> dict[str, float]:
    """Book-averaged devigged Cup probability per team.

    Each book is devigged over the teams *it* prices, then teams are averaged
    across books; a book missing a team simply does not vote for it.
    """
    by_book: dict[str, dict[str, float]] = {}
    for code, prices in futures.items():
        for book, american in prices.items():
            by_book.setdefault(book, {})[code] = american
    votes: dict[str, list[float]] = {}
    for prices in by_book.values():
        if len(prices) < 16:  # a partial board would be devigged against itself
            continue
        for code, p in devig(prices).items():
            votes.setdefault(code, []).append(p)
    return {code: sum(v) / len(v) for code, v in votes.items()}


def _book_count(futures: dict[str, dict[str, float]] | None) -> int:
    if not futures:
        return 0
    return len({book for prices in futures.values() for book in prices})


def save(prior: PreseasonPrior, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = asset_path(folder, prior.season)
    path.write_text(prior.to_json())
    return path


def load(folder: Path, season: int) -> PreseasonPrior | None:
    path = asset_path(folder, season)
    if not path.exists():
        return None
    try:
        return PreseasonPrior.from_json(path.read_text())
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("preseason prior %s unreadable: %s", path, exc)
        return None


def is_stale(
    prior: PreseasonPrior | None, today: Date, *, freeze_after: Date, max_age_days: int = 7
) -> bool:
    """Rebuild weekly until the freeze date; never after it."""
    if prior is None:
        return True
    if today > freeze_after:
        return False
    built = Date.fromisoformat(prior.built_on)
    return (today - built).days >= max_age_days


__all__ = [
    "PreseasonPrior",
    "TeamPrior",
    "asset_path",
    "build",
    "futures_probs",
    "is_stale",
    "load",
    "save",
]
