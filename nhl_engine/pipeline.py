"""Slate -> rates -> joint sim -> priced board -> ledger rows (master plan §6 Phase 2).

One call per card pass. For every game on the archived board:

1. team EB strength as of the slate (``features.strength``) on the preseason
   prior; league rates from the same tables;
2. tonight's 5v5 rates: a lineup rebuild if one was written for the date
   (``lineups/<date>.json`` from the pre-drop/roster pass), otherwise the team
   rate tagged ``team_rate [reported, not scored]``;
3. starting goalies (``features.starters``) -- GSAx/60 into the sim, status
   into the ``goalie_unconfirmed`` gate;
4. one seeded joint draw (``models.periods``) read by every market
   (``models.markets``); ``team_total`` rows resolve their ``ot_rule`` per book
   through ``BookRules`` and stamp ``settlement_unverified`` when unknown;
5. calibration, EV/tier/gates (``market.pricing``) -> ``LedgerRow``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

from nhl_engine.audit.ledger import LedgerRow, row_from
from nhl_engine.calibration import Calibrator
from nhl_engine.config import Config
from nhl_engine.data.book_rules import BookRules
from nhl_engine.data.capture import MARKET_MAP, QuoteRow
from nhl_engine.data.moneypuck import MoneyPuckClient, as_of
from nhl_engine.data.preseason import PreseasonPrior
from nhl_engine.data.teamnames import CODES
from nhl_engine.features import strength
from nhl_engine.features.starters import Starter, load_overrides, overrides_path, starter_for
from nhl_engine.market.board import Selection, matchups, selections
from nhl_engine.market.pricing import stamp
from nhl_engine.models import markets
from nhl_engine.models.goals import GameRates, game_rates
from nhl_engine.models.periods import SimResult, simulate

log = logging.getLogger("nhl_engine")

TEAM_RATE = "team_rate [reported, not scored]"
DEFAULT_OT_RULE: dict[str, str] = {mk: rule for mk, rule in MARKET_MAP.values()}


@dataclass(frozen=True)
class LineupRates:
    xgf60: float
    xga60: float
    source: str
    turnover: float = 0.0


def lineups_path(data_dir: Path, slate: Date) -> Path:
    return data_dir / "lineups" / f"{slate.isoformat()}.json"


def load_lineups(path: Path) -> dict[str, LineupRates]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, LineupRates] = {}
    if isinstance(raw, dict):
        for team, v in raw.items():
            if isinstance(v, dict) and "xgf60" in v and "xga60" in v:
                out[team] = LineupRates(
                    float(v["xgf60"]),
                    float(v["xga60"]),
                    str(v.get("source", "projected")),
                    float(v.get("turnover", 0.0)),
                )
    return out


@dataclass
class TeamInputs:
    code: str
    rates: dict[str, float]
    reliability: dict[str, float]
    games: int
    lineup_source: str
    starter: Starter


@dataclass
class GameCard:
    matchup: str
    home: str
    away: str
    event_id: str
    home_in: TeamInputs
    away_in: TeamInputs
    rates: GameRates
    sim: SimResult
    summary: dict[str, float]
    rows: list[LedgerRow] = field(default_factory=list)

    @property
    def goalie_status(self) -> str:
        return f"{self.away_in.starter.status}/{self.home_in.starter.status}"


@dataclass
class SlateCard:
    slate_date: Date
    season: int
    priced_at: str
    tag: str
    prior_version: str
    games: list[GameCard] = field(default_factory=list)
    unpriced: list[str] = field(default_factory=list)

    @property
    def rows(self) -> list[LedgerRow]:
        return [r for g in self.games for r in g.rows]


def _apply_thin_fallback(
    rates: dict[str, float], rel: dict[str, float], league: dict[str, float], floor: float
) -> dict[str, float]:
    """Special-teams sub-states below the reliability floor ship at the league rate."""
    out = dict(rates)
    for key in ("pp_xgf60", "pp_xga60", "pk_xga60", "sh_xgf60"):
        if rel.get(key, 0.0) < floor and key in league:
            out[key] = league[key]
    return out


def team_inputs(
    mp: MoneyPuckClient,
    code: str,
    slate: Date,
    *,
    season: int,
    cfg: Config,
    prior: PreseasonPrior,
    league: dict[str, float],
    lineups: dict[str, LineupRates],
    overrides: dict[str, dict[str, object]],
    st_floor: float = 0.2,
) -> TeamInputs:
    est = strength.team_strength(
        mp.team_games(code),
        slate,
        season=season,
        priors=prior.rates_for(code),
        ks=cfg.shrink.team_k,
    )
    rates = {k: e.posterior for k, e in est.items()}
    rel = {k: e.reliability for k, e in est.items()}
    rates = _apply_thin_fallback(rates, rel, league, st_floor)
    games = next(iter(est.values())).games if est else 0
    lu = lineups.get(code)
    if lu is not None:
        rates["xgf60_5v5"], rates["xga60_5v5"] = lu.xgf60, lu.xga60
        source = f"{lu.source} (turnover {lu.turnover:.2f})"
    else:
        source = TEAM_RATE
    starter = starter_for(mp, code, slate, season=season, cfg=cfg, overrides=overrides)
    return TeamInputs(code, rates, rel, games, source, starter)


def league_for(
    mp: MoneyPuckClient, slate: Date, season: int, prior: PreseasonPrior
) -> dict[str, float]:
    """League rate per metric as of the slate; the prior's league mean where no games yet."""
    tables = {c: strength.game_table(as_of(mp.team_games(c), slate, season=season)) for c in CODES}
    live = strength.league_rates(tables)
    out: dict[str, float] = {}
    for k, v in live.items():
        out[k] = v if v == v else prior.league.get(k, float("nan"))
    return out


def price_game(
    board: list[Selection],
    *,
    home_in: TeamInputs,
    away_in: TeamInputs,
    league: dict[str, float],
    cfg: Config,
    slate: Date,
    rules: BookRules,
    calib: Calibrator,
    now: datetime,
    priced_at: str,
    tag: str,
    seed: int,
) -> GameCard:
    home, away = home_in.code, away_in.code
    matchup = f"{away} @ {home}"
    rates = game_rates(
        home_in.rates,
        away_in.rates,
        league,
        home_goalie_gsax60=home_in.starter.gsax60,
        away_goalie_gsax60=away_in.starter.gsax60,
        params=cfg.sim,
    )
    sim = simulate(rates, cfg.sim, seed=seed)
    card = GameCard(matchup, home, away, "", home_in, away_in, rates, sim, markets.summary(sim))
    both_confirmed = home_in.starter.confirmed and away_in.starter.confirmed
    for sel in _per_book_rules(s for s in board if s.matchup == matchup):
        ot_rule = DEFAULT_OT_RULE.get(sel.market, "reg_only")
        settlement_gate: str | None = None
        if ot_rule == "book_rule":
            ot_rule = rules.resolve(sel.best_book, sel.market, slate)
            settlement_gate = rules.gate(sel.best_book, sel.market, slate)
            if settlement_gate:
                ot_rule = "incl_ot_so"
        try:
            raw = markets.price(
                sim,
                market=sel.market,
                side=sel.side,
                entity=sel.entity,
                line=sel.line,
                ot_rule=ot_rule,
                home=home,
                away=away,
            )
        except markets.UnpricedMarket:
            continue
        model, cal_reasons = calib.apply(sel.market, raw)
        priced = stamp(
            sel,
            model,
            ot_rule=ot_rule,
            thr=cfg.gates,
            quote_age_minutes=sel.age_minutes(now),
            goalie_confirmed=both_confirmed,
            settlement_gate=settlement_gate,
        )
        priced.reasons.extend(cal_reasons)
        card.rows.append(
            row_from(
                priced,
                slate_date=slate,
                home=home,
                away=away,
                home_goalie=home_in.starter.name,
                away_goalie=away_in.starter.name,
                goalie_status=card.goalie_status,
                lineup_source=f"{away_in.lineup_source} | {home_in.lineup_source}",
                priced_at=priced_at,
                card_tag=tag,
            )
        )
    return card


def _per_book_rules(board: Iterable[Selection]) -> Iterator[Selection]:
    """Split book-rule markets (team totals) into one selection per book.

    The settlement rule changes the probability itself, so the best price across
    books is not comparable; every book prices under its own rule.
    """
    for sel in board:
        if DEFAULT_OT_RULE.get(sel.market) != "book_rule" or len(sel.per_book) <= 1:
            yield sel
            continue
        for book, american in sorted(sel.per_book.items()):
            yield replace(sel, best_american=american, best_book=book)


def run_slate(
    quotes: list[QuoteRow],
    *,
    slate: Date,
    season: int,
    cfg: Config,
    mp: MoneyPuckClient,
    prior: PreseasonPrior,
    rules: BookRules,
    calib: Calibrator,
    data_dir: Path,
    tag: str,
    now: datetime | None = None,
    seed: int | None = None,
) -> SlateCard:
    now = now or datetime.now(timezone.utc)
    priced_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    board = selections(quotes)
    games = matchups(quotes)
    league = league_for(mp, slate, season, prior)
    lineups = load_lineups(lineups_path(data_dir, slate))
    overrides = load_overrides(overrides_path(data_dir, slate))
    base_seed = seed if seed is not None else int(slate.strftime("%Y%m%d"))
    out = SlateCard(slate, season, priced_at, tag, prior.version)
    cache: dict[str, TeamInputs] = {}

    def inputs(code: str) -> TeamInputs:
        if code not in cache:
            cache[code] = team_inputs(
                mp,
                code,
                slate,
                season=season,
                cfg=cfg,
                prior=prior,
                league=league,
                lineups=lineups,
                overrides=overrides,
            )
        return cache[code]

    for i, (matchup, (away, home, event_id)) in enumerate(sorted(games.items())):
        if home not in CODES or away not in CODES:
            out.unpriced.append(f"{matchup}: unknown team code")
            continue
        try:
            card = price_game(
                board,
                home_in=inputs(home),
                away_in=inputs(away),
                league=league,
                cfg=cfg,
                slate=slate,
                rules=rules,
                calib=calib,
                now=now,
                priced_at=priced_at,
                tag=tag,
                seed=base_seed * 100 + i,
            )
        except Exception as exc:  # one bad game must not sink the slate
            log.exception("pricing %s failed", matchup)
            out.unpriced.append(f"{matchup}: {exc}")
            continue
        card.event_id = event_id
        out.games.append(card)
    return out


__all__ = [
    "GameCard",
    "LineupRates",
    "SlateCard",
    "TeamInputs",
    "league_for",
    "lineups_path",
    "load_lineups",
    "price_game",
    "run_slate",
    "team_inputs",
]
