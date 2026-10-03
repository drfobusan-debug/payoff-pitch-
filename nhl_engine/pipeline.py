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
5. calibration, the market anchor on game markets, EV/tier/gates
   (``market.pricing``) -> ``LedgerRow``;
6. props (``models.props``) off the same draws when a ``PropContext`` holds a
   projection for the quoted player -- every prop row is stamped
   ``research_only`` (master plan §6 Phase 3).
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
from nhl_engine.data.nhlapi import NHLAPIClient, RosterSpot
from nhl_engine.data.preseason import PreseasonPrior
from nhl_engine.data.rotowire import norm_name
from nhl_engine.data.skaters import NameIndex, PlayerLogClient, SkaterGame
from nhl_engine.data.teamnames import CODES
from nhl_engine.features import props as propf
from nhl_engine.features import strength
from nhl_engine.features.starters import Starter, load_overrides, overrides_path, starter_for
from nhl_engine.market.board import Selection, matchups, selections
from nhl_engine.market.pricing import anchor, is_prop, stamp
from nhl_engine.models import markets
from nhl_engine.models import props as propm
from nhl_engine.models.goals import GameRates, game_rates, goalie_factor
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
    # Team SOG for / against per 60 (all situations) for SOG opponent factors and saves.
    sf60: float = float("nan")
    sa60: float = float("nan")


@dataclass
class PropContext:
    """Who the book's player names resolve to and what they project to tonight.

    ``skaters``/``goalies`` are keyed by ``(team, normalised name)`` as the book
    spells it; a quoted player with no projection is left unpriced, not guessed.
    """

    skaters: dict[tuple[str, str], propf.SkaterProjection] = field(default_factory=dict)
    goalies: dict[tuple[str, str], propf.GoalieProjection] = field(default_factory=dict)
    league_sf60: float = 29.5
    league_sa60: float = 29.5

    def skater(self, team: str, name: str) -> propf.SkaterProjection | None:
        return self.skaters.get((team, norm_name(name)))

    def goalie(self, team: str, name: str) -> propf.GoalieProjection | None:
        return self.goalies.get((team, norm_name(name)))


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
    """Shorthanded sub-states below the reliability floor ship at the league rate.

    Only the two thin SH-scoring states (§5.3) are floored. PP xGF and PK xGA
    keep their EB posterior at any reliability: at 0 GP that posterior *is* the
    preseason prior, which is exactly what the sim should run on.
    """
    out = dict(rates)
    for key in ("pp_xga60", "sh_xgf60"):
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
    sf60, sa60 = team_shot_rates(mp, code, slate, season=season)
    return TeamInputs(code, rates, rel, games, source, starter, sf60, sa60)


LEAGUE_SHOTS60 = 29.5
SHOT_RATE_K_GAMES = 10.0


def team_shot_rates(
    mp: MoneyPuckClient, code: str, slate: Date, *, season: int
) -> tuple[float, float]:
    """All-situation SOG for / against per 60, shrunk over 10 games to last season's rate.

    Last season's rate is itself shrunk to the league constant, so a team with
    no games yet carries its own shot profile rather than the league's.
    """
    games = mp.team_games(code)

    def _sums(df) -> tuple[float, float, float]:
        if "situation" in df.columns:
            df = df[df["situation"] == "all"]
        if "iceTime" not in df.columns or not len(df):
            return 0.0, 0.0, 0.0
        return (
            float(df["iceTime"].sum()),
            float(df["shotsOnGoalFor"].sum()),
            float(df["shotsOnGoalAgainst"].sum()),
        )

    k = SHOT_RATE_K_GAMES * 3600.0

    def _shrunk(secs: float, events: float, prior60: float) -> float:
        return (events + k * prior60 / 3600.0) / (secs + k) * 3600.0

    p_secs, p_sf, p_sa = _sums(as_of(games, slate, season=season - 1))
    prior_sf = _shrunk(p_secs, p_sf, LEAGUE_SHOTS60)
    prior_sa = _shrunk(p_secs, p_sa, LEAGUE_SHOTS60)
    secs, sf, sa = _sums(as_of(games, slate, season=season))
    return _shrunk(secs, sf, prior_sf), _shrunk(secs, sa, prior_sa)


def build_prop_context(
    quotes: list[QuoteRow],
    games: dict[str, tuple[str, str, str]],
    *,
    slate: Date,
    season: int,
    logs: PlayerLogClient,
    shrink: propf.PropShrink = propf.DEFAULT_SHRINK,
) -> PropContext:
    """Projections for every player a prop quote names, from rosters + game logs.

    Only quoted players are fetched (two seasons of game log each, cached); the
    league positional means are pooled from that same set once it is large
    enough, otherwise the constants in ``features.props`` stand in.
    """
    ctx = PropContext()
    teams = sorted({t for away, home, _ in games.values() for t in (away, home)})
    index = NameIndex([s for t in teams for s in logs.current_roster(t)])
    wanted: dict[tuple[str, str], str] = {}
    for q in quotes:
        if not is_prop(q.market) or not q.entity or q.matchup not in games:
            continue
        away, home, _ = games[q.matchup]
        for t in (away, home):
            wanted.setdefault((t, norm_name(q.entity)), q.entity)
    resolved: dict[tuple[str, str], tuple[str, RosterSpot, list[SkaterGame], list[SkaterGame]]] = {}
    pool: list[tuple[str, list[SkaterGame]]] = []
    for (team, key), raw in wanted.items():
        spot = index.find(team, raw)
        if spot is None:
            continue
        if spot.position == "G":
            cur = logs.goalie_log(spot.player_id, season)
            prev = logs.goalie_log(spot.player_id, season - 1, final=True)
            ctx.goalies[(team, key)] = propf.project_goalie(
                player_id=spot.player_id,
                name=spot.name,
                team=team,
                current=cur,
                previous=prev,
                slate=slate,
                shrink=shrink,
            )
            continue
        cur_s = logs.skater_log(spot.player_id, season)
        prev_s = logs.skater_log(spot.player_id, season - 1, final=True)
        pos = "D" if spot.position == "D" else "F"
        pool.append((pos, prev_s))
        resolved[(team, key)] = (pos, spot, cur_s, prev_s)
    league = propf.league_means(pool)
    for (team, key), (pos, spot, cur_s, prev_s) in resolved.items():
        ctx.skaters[(team, key)] = propf.project_skater(
            player_id=spot.player_id,
            name=spot.name,
            team=team,
            position=pos,
            current=cur_s,
            previous=prev_s,
            slate=slate,
            league=league[pos],
            shrink=shrink,
        )
    return ctx


def shrink_league(
    live: dict[str, float], prior: dict[str, float], n: float, k_games: dict[str, float]
) -> dict[str, float]:
    """``w x live + (1 - w) x prior``, ``w = n / (n + k)``; a missing side is the other."""
    out: dict[str, float] = {}
    for key in live.keys() | prior.keys():
        v, base = live.get(key, float("nan")), prior.get(key, float("nan"))
        if v != v or n <= 0:
            out[key] = base
        elif base != base:
            out[key] = v
        else:
            w = n / (n + k_games.get(key, 0.0))
            out[key] = w * v + (1.0 - w) * base
    return out


def league_for(
    mp: MoneyPuckClient,
    slate: Date,
    season: int,
    prior: PreseasonPrior,
    k_games: dict[str, float] | None = None,
) -> dict[str, float]:
    """League rate per metric as of the slate, shrunk toward last season's league.

    ``k`` per metric (team-games) from ``ShrinkParams.league_k_games``.
    """
    tables = {c: strength.game_table(as_of(mp.team_games(c), slate, season=season)) for c in CODES}
    n = float(sum(len(t) for t in tables.values()))
    ks = k_games if k_games is not None else Config().shrink.league_k_games
    return shrink_league(strength.league_rates(tables), prior.league, n, ks)


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
    props: PropContext | None = None,
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
        if is_prop(sel.market):
            if props is None:
                continue
            model_p = price_prop(sel, sim, home_in=home_in, away_in=away_in, props=props)
            if model_p is None:
                continue
            priced = stamp(
                sel,
                model_p,
                ot_rule=ot_rule,
                thr=cfg.gates,
                quote_age_minutes=sel.age_minutes(now),
                goalie_confirmed=both_confirmed,
                settlement_gate=None,
            )
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
            continue
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
        sim_prob: float | None = None
        if sel.market in cfg.gates.anchored_markets:
            sim_prob = model.win
            model = anchor(model, sel.consensus, cfg.gates.market_anchor)
            cal_reasons = [
                *cal_reasons,
                f"anchored {cfg.gates.market_anchor:.2f} to market: sim {sim_prob:.3f}"
                f" -> bet {model.win:.3f}",
            ]
        priced = stamp(
            sel,
            model,
            ot_rule=ot_rule,
            thr=cfg.gates,
            quote_age_minutes=sel.age_minutes(now),
            goalie_confirmed=both_confirmed,
            settlement_gate=settlement_gate,
            sim_prob=sim_prob,
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


def _side_of(
    sel: Selection, home_in: TeamInputs, away_in: TeamInputs, props: PropContext
) -> tuple[TeamInputs, TeamInputs, propf.SkaterProjection] | None:
    """``(team_inputs, opp_inputs, projection)`` for the quoted player, or None."""
    for own, opp in ((home_in, away_in), (away_in, home_in)):
        sk = props.skater(own.code, sel.entity)
        if sk is not None:
            return own, opp, sk
    return None


def price_prop(
    sel: Selection,
    sim: propm.SimResult,
    *,
    home_in: TeamInputs,
    away_in: TeamInputs,
    props: PropContext,
) -> markets.Prob | None:
    mk = sel.market
    if mk == "g_saves":
        for own, opp in ((home_in, away_in), (away_in, home_in)):
            g = props.goalie(own.code, sel.entity)
            if g is None or sel.line is None:
                continue
            if norm_name(own.starter.name) != norm_name(g.name):
                return None  # quoted goalie is not the projected starter
            sf = opp.sf60 if opp.sf60 == opp.sf60 else props.league_sf60
            sa = own.sa60 if own.sa60 == own.sa60 else props.league_sa60
            shots = sf * sa / props.league_sa60 * 60.6 / 60.0
            mean = propm.saves_mean(shots, propm.exp_goals_against(sim, home=own is home_in))
            return propm.over_under(
                propm.negbin_pmf(mean, propm.SAVES_VAR_SLOPE), sel.line, sel.side
            )
        return None
    hit = _side_of(sel, home_in, away_in, props)
    if hit is None:
        return None
    own, opp, sk = hit
    is_home = own is home_in
    if mk == "sk_sog":
        if sel.line is None:
            return None
        sa = opp.sa60 if opp.sa60 == opp.sa60 else props.league_sa60
        mean = propm.sog_mean(
            sk.sog60,
            sk.toi,
            shot_factor=propm.score_state(sim, home=is_home).shot_factor,
            opp_sa_factor=sa / props.league_sa60,
        )
        return propm.over_under(propm.negbin_pmf(mean), sel.line, sel.side)
    if mk in ("sk_g", "ags"):
        # Goals scale with the opposing goalie exactly as the sim's goal rates do.
        mean = sk.mean("g") * goalie_factor(opp.starter.gsax60)
        pmf = propm.poisson_pmf(mean)
        if mk == "ags":
            return propm.anytime(pmf, sel.side)
        return propm.over_under(pmf, sel.line, sel.side) if sel.line is not None else None
    if mk in ("sk_a", "sk_pts", "sk_ppp"):
        if sel.line is None:
            return None
        stat = {"sk_a": "a", "sk_pts": "pts", "sk_ppp": "ppp"}[mk]
        return propm.over_under(propm.poisson_pmf(sk.mean(stat)), sel.line, sel.side)
    return None


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
    logs: PlayerLogClient | None = None,
) -> SlateCard:
    now = now or datetime.now(timezone.utc)
    priced_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    board = selections(quotes)
    games = matchups(quotes)
    props: PropContext | None = None
    if logs is None:
        logs = PlayerLogClient(NHLAPIClient(cache_dir=data_dir / "cache" / "nhlapi"))
    try:
        props = build_prop_context(quotes, games, slate=slate, season=season, logs=logs)
    except Exception:  # props are research rows; never sink the card
        log.exception("prop projections failed; props unpriced")
    league = league_for(mp, slate, season, prior, cfg.shrink.league_k_games)
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
                props=props,
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
    "shrink_league",
    "team_inputs",
]
