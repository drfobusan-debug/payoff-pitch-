"""``nhl-engine`` command line.

Phase 0 commands:

* ``capture`` -- archive today's featured board plus per-event period/prop
  markets. Idempotent: an unchanged board writes nothing.
* ``results`` -- pull official finals (per-period, OT/SO, starters) for a date.
* ``archive`` -- summarise what has been captured for a date.
* ``book-rules`` -- list, add or check per-book settlement rules.

Phase 1 commands (features, no prices):

* ``prior`` -- build/refresh ``preseason_prior_<season>.json`` from last
  season's MoneyPuck logs (+ Cup futures recorded, weight 0).
* ``strength`` -- every team's as-of EB posterior for each §5.1 metric, with
  the prior weight still in it.

Phase 2 commands (priced card, ledger, grading):

* ``card`` -- price the archived board from one joint sim per game; writes
  the write-once ledger (``--tag initial``, and ``--tag early --before HH:MM``
  for afternoon games) plus txt/md/xlsx outputs.
* ``starter`` -- record a confirmed/probable goalie for a team on a date.
* ``lineups`` -- pull RotoWire's expected/confirmed goalies and injury list
  for today (or tomorrow) into the starter overrides and availability log.
* ``podcast`` -- find the Hockey Gambling Podcast episode for the slate,
  transcribe it locally and write the per-game read the PDF shows.
* ``audit`` -- grade a date's ledger against official finals (CLV, dual-rule
  flag) and print the running scorecard.
* ``calibrate`` -- refit per-market isotonic maps from every graded ledger.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from engine_runtime.odds_client import OddsClient
from nhl_engine import closetest, email, history, outputs, pipeline, report, state
from nhl_engine.audit import (
    bands,
    grade,
    ledger,
    ledger_report,
    podcast_picks,
    scorecard,
    screens,
    vsin_picks,
)
from nhl_engine.calibration import Calibrator, calibration_path
from nhl_engine.config import cache_dir, data_dir, load_config, output_dir, priors_dir
from nhl_engine.data import availability, capture, dailyfaceoff, podcast, preseason, rotowire
from nhl_engine.data.book_rules import BookRule, BookRules, rules_path
from nhl_engine.data.moneypuck import MoneyPuckClient, season_of
from nhl_engine.data.nhlapi import FOREVER, NHLAPIClient
from nhl_engine.data.oddsapi import SPORT_KEY, OddsAPIClient
from nhl_engine.data.skaters import PlayerLogClient
from nhl_engine.data.teamnames import CODES, canonical
from nhl_engine.features import lineup_feed, podcast_read, starters, strength
from nhl_engine.market import board
from nhl_engine.schemas import GameResult

log = logging.getLogger("nhl_engine")
SLATE_TZ = ZoneInfo("America/New_York")
# Cards whose games go into the write-once ledger: the early card for games that
# start before the 17:00 card, and the 17:00 card for the rest.
LEDGER_TAGS = frozenset({"early", "initial"})


def _today() -> Date:
    return datetime.now(SLATE_TZ).date()


def _parse_date(raw: str | None) -> Date:
    return Date.fromisoformat(raw) if raw else _today()


def cmd_capture(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate_date = _parse_date(args.date)
    root = data_dir()
    if cfg.state_sync and not args.no_sync:
        state.auto_pull(root)
    client = OddsAPIClient(
        cfg.creds.odds_api_key, cache_dir=cache_dir() / "oddsapi", cache_ttl=args.cache_ttl
    )
    if not client.available():
        print(
            "no Odds API key (THE_ODDS_API_KEY / ODDS_API_KEY); nothing captured", file=sys.stderr
        )
        return 2
    if args.if_drop_within:
        soon = _minutes_to_next_drop(slate_date)
        if soon is None or soon > args.if_drop_within:
            print(
                json.dumps(
                    {
                        "slate_date": slate_date.isoformat(),
                        "skipped": f"no puck drop within {args.if_drop_within} min",
                        "next_drop_min": soon,
                    }
                )
            )
            return 0
    taken = capture.now_utc()
    slate, rows = client.fetch_board(
        slate_date=slate_date, horizon_hours=cfg.capture.horizon_hours, captured_at=taken
    )
    written = [capture.write_snapshot(rows, root, slate_date, label="board", captured_at=taken)]
    if not args.board_only and slate.games:
        event_rows = client.fetch_event_markets(
            slate.games,
            captured_at=taken,
            max_events=min(cfg.capture.max_events, args.max_events),
        )
        written.append(
            capture.write_snapshot(event_rows, root, slate_date, label="events", captured_at=taken)
        )
    else:
        event_rows = []
    wrote = [p.name for p in written if p is not None]
    print(
        json.dumps(
            {
                "slate_date": slate_date.isoformat(),
                "games": len(slate.games),
                "board_rows": len(rows),
                "event_rows": len(event_rows),
                "written": wrote,
                "credits_remaining": client.credits_remaining,
            }
        )
    )
    if cfg.state_sync and not args.no_sync and wrote:
        state.auto_push(root, f"nhl: capture {slate_date} {taken}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    client = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")
    games = client.schedule(day)
    out = []
    for game in games:
        res = client.result(game)
        if res is None:
            continue
        out.append(
            {
                "game": game.matchup,
                "nhl_game_id": res.nhl_game_id,
                "state": res.state,
                "decided": res.decided,
                "periods": [asdict(p) for p in res.periods],
                "reg": [res.reg_away, res.reg_home],
                "final": [res.final_away, res.final_home],
                "starters": [res.away_starter, res.home_starter],
            }
        )
    print(json.dumps({"date": day.isoformat(), "games": out}, indent=None if args.compact else 2))
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    rows = capture.read_day(data_dir(), day)
    summary = capture.archive_summary(rows)
    summary["snapshots"] = [p.name for p in capture.snapshot_paths(data_dir(), day)]
    print(json.dumps(summary, indent=2))
    return 0


def cmd_book_rules(args: argparse.Namespace) -> int:
    path = rules_path(data_dir())
    rules = BookRules.load(path)
    today = _today()
    if args.add:
        book, market, ot_rule = args.add
        rules.add(
            BookRule(
                book=book,
                market=market,
                ot_rule=ot_rule,
                verified_on=today.isoformat(),
                source=args.source,
            )
        )
        rules.save(path)
        print(f"recorded {book} {market} -> {ot_rule} (verified {today})")
        return 0
    if args.check:
        book, market = args.check
        print(rules.resolve(book, market, today))
        return 0
    stale = {(r.book, r.market) for r in rules.stale(today)}
    for r in rules.all():
        flag = "  EXPIRED" if (r.book, r.market) in stale else ""
        print(f"{r.book:16} {r.market:16} {r.ot_rule:12} verified {r.verified_on}{flag}")
    if not len(rules):
        print(f"no book rules recorded ({path}); every team-total quote is settlement_unverified")
    return 0


def _prior_for(season: int, mp: MoneyPuckClient, today: Date) -> preseason.PreseasonPrior:
    """Load the asset, rebuilding on the weekly schedule until the freeze date."""
    cfg = load_config()
    prior = preseason.load(priors_dir(), season)
    freeze = Date(season, *cfg.prior.freeze_month_day)
    if preseason.is_stale(prior, today, freeze_after=freeze, max_age_days=cfg.prior.refresh_days):
        futures = None
        if cfg.creds.has_odds_api():
            client = OddsAPIClient(
                cfg.creds.odds_api_key, cache_dir=cache_dir() / "oddsapi", cache_ttl=6 * 3600
            )
            futures = client.fetch_outrights()
        prior = preseason.build(mp, season, r_yy=cfg.prior.r_yy, futures=futures, today=today)
        path = preseason.save(prior, priors_dir())
        log.info("preseason prior %d rebuilt -> %s (version %s)", season, path, prior.version)
    assert prior is not None
    return prior


def cmd_prior(args: argparse.Namespace) -> int:
    today = _parse_date(args.date)
    season = args.season or season_of(today)
    mp = MoneyPuckClient(cache_dir=cache_dir())
    if args.force:
        cfg = load_config()
        futures = None
        if cfg.creds.has_odds_api() and not args.no_futures:
            client = OddsAPIClient(
                cfg.creds.odds_api_key, cache_dir=cache_dir() / "oddsapi", cache_ttl=6 * 3600
            )
            futures = client.fetch_outrights()
        prior = preseason.build(mp, season, r_yy=cfg.prior.r_yy, futures=futures, today=today)
        preseason.save(prior, priors_dir())
    else:
        prior = _prior_for(season, mp, today)
    print(
        json.dumps(
            {
                "season": prior.season,
                "version": prior.version,
                "built_on": prior.built_on,
                "teams": len(prior.teams),
                "components": prior.components,
                "futures_books": prior.futures_books,
                "path": str(preseason.asset_path(priors_dir(), season)),
            },
            indent=2,
        )
    )
    return 0


def cmd_strength(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate = _parse_date(args.date)
    season = args.season or season_of(slate)
    mp = MoneyPuckClient(cache_dir=cache_dir())
    prior = _prior_for(season, mp, slate)
    keys = args.metrics.split(",") if args.metrics else [m.key for m in strength.METRICS]
    rows = []
    for code in sorted(CODES):
        est = strength.team_strength(
            mp.team_games(code),
            slate,
            season=season,
            priors=prior.rates_for(code),
            ks=cfg.shrink.team_k,
        )
        rows.append((code, est))
    if args.json:
        print(
            json.dumps(
                {
                    "slate": slate.isoformat(),
                    "season": season,
                    "prior_version": prior.version,
                    "teams": {
                        code: {k: asdict(e) for k, e in est.items() if k in keys}
                        for code, est in rows
                    },
                },
                default=float,
            )
        )
        return 0
    print(f"as of {slate}  season {season}  prior {prior.version}  (posterior [prior→data weight])")
    head = "team  gp  " + "  ".join(f"{k:>16s}" for k in keys)
    print(head)
    for code, est in rows:
        gp = next(iter(est.values())).games if est else 0
        cells = []
        for k in keys:
            e = est[k]
            cells.append(f"{e.posterior:9.3f} [{e.reliability:4.2f}]")
        print(f"{code:4s} {gp:3d}  " + "  ".join(f"{c:>16s}" for c in cells))
    return 0


def _local_cutoff(slate: Date, hhmm: str) -> datetime:
    """``HH:MM`` Eastern on the slate date."""
    hour, minute = (int(x) for x in hhmm.split(":"))
    return datetime(slate.year, slate.month, slate.day, hour, minute, tzinfo=SLATE_TZ)


def cmd_card(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate = _parse_date(args.date)
    season = args.season or season_of(slate)
    root = data_dir()
    quotes = capture.read_day(root, slate)
    if not quotes:
        print(f"no archived prices for {slate}; run `nhl-engine capture` first", file=sys.stderr)
        return 2
    mp = MoneyPuckClient(cache_dir=cache_dir())
    prior = _prior_for(season, mp, slate)
    try:
        starts = {g.matchup: g.start_utc for g in _nhlapi().schedule(slate)}
    except Exception as exc:  # no schedule: price every game rather than none
        print(f"NHL schedule unavailable, started games not skipped: {exc}", file=sys.stderr)
        starts = {}
    only: set[str] | None = None
    if args.before:
        cutoff = _local_cutoff(slate, args.before)
        only = pipeline.starting_before(starts, cutoff, datetime.now(SLATE_TZ))
        if not only:
            print(f"no {slate} games start before {args.before} ET; no {args.tag} card")
            return 0
    card = pipeline.run_slate(
        quotes,
        slate=slate,
        season=season,
        cfg=cfg,
        mp=mp,
        prior=prior,
        rules=BookRules.load(rules_path(root)),
        calib=Calibrator.load(calibration_path(root)),
        data_dir=root,
        tag=args.tag,
        seed=args.seed,
        starts=starts,
        only=only,
    )
    paths = outputs.write_all(card, output_dir())
    if not args.no_pdf:
        try:
            known = {t.code: t for g in card.games for t in (g.away_in, g.home_in)}
            rates = pipeline.league_inputs(
                mp, slate, season=season, cfg=cfg, prior=prior, known=known
            )
            ctx = report.build_context(card, mp=mp, data_dir=root, all_rates=rates)
            ctx.podcast = podcast_read.load_read(root, slate)
            ctx.pod_picks = podcast_picks.load_picks(podcast_picks.picks_path(root, slate))
            ctx.pod_history = podcast_picks.load_all_picks(root)
            ctx.pod_stated = podcast_picks.load_records(podcast_picks.records_path(root, slate))
            try:
                games = {g.matchup: (g.away, g.home) for g in card.games}
                ctx.vsin_picks, ctx.vsin_other = vsin_picks.read_day(root, slate, games)
            except Exception:  # noqa: BLE001 - VSiN's bets are a view; never lose the PDF
                log.warning("VSiN best bets unavailable", exc_info=True)
            ctx.vsin_history = vsin_picks.load_all(root)
            try:
                graded = [
                    r
                    for p in sorted((root / "ledger").glob("graded_*.json"))
                    for r in ledger.load_rows(p)
                    if not r.priced_late and r.slate_date < slate.isoformat()
                ]
                b2b, unread = _lean_b2b(sorted({r.slate_date for r in graded}), slate)
                sctx = screens.Context(b2b=b2b)
                reads = {b.name: b for b in bands.read_bands(graded, sctx)}
                ctx.leans = [reads[n] for n in bands.LEANS if n in reads]
                _backfill_timing(root, graded)
                ctx.promotions = scorecard.scorecard(graded).promotions()
                ctx.lean_tonight = bands.tonight(card.rows, sctx)
                if unread:
                    for name in [n for n in ctx.lean_tonight if "back-to-back" in n]:
                        del ctx.lean_tonight[name]
                    ctx.lean_note = (
                        "Tonight's back-to-backs are unread (NHL schedule unavailable), "
                        "so the back-to-back leans list no sides tonight."
                    )
            except Exception:  # noqa: BLE001 - the leans are a view; never lose the PDF
                log.warning("graded-only leans unavailable", exc_info=True)
            paths["pdf"] = report.write_pdf(
                card, ctx, report.pdf_path(output_dir(), slate, args.tag)
            )
        except Exception:  # the PDF is a view of the card; never lose the card over it
            log.exception("pdf card failed")
    ledger.save_rows(card.rows, ledger.card_path(root, slate, args.tag))
    written = ""
    if args.tag in LEDGER_TAGS or args.ledger:
        path, wrote = ledger.write_once(
            card.rows, ledger.predictions_path(root, slate), force=args.force
        )
        written = f"\nledger: {path} ({'written' if wrote else 'already existed, kept'})"
    print(outputs.render_card(card), end="")
    print("outputs: " + ", ".join(str(p) for p in paths.values()) + written)
    if args.email:
        paths.update(ledger_report.latest(output_dir()))
        try:
            to = email.send_card(cfg, card, paths, to=args.to)
            print(f"emailed {len(paths)} attachment(s) for {slate} -> {to}")
        except email.EmailNotConfigured as exc:
            print(f"email skipped: {exc}", file=sys.stderr)
    if cfg.state_sync and not args.no_sync:
        state.auto_push(root, f"nhl card {slate} {args.tag}")
    return 0


def cmd_starter(args: argparse.Namespace) -> int:
    slate = _parse_date(args.date)
    path = starters.overrides_path(data_dir(), slate)
    starters.save_override(path, canonical(args.team), args.player_id, args.status, args.source)
    print(f"{canonical(args.team)} {slate}: goalie {args.player_id} {args.status} -> {path}")
    return 0


def cmd_lineups(args: argparse.Namespace) -> int:
    """RotoWire goalies + injuries, then Daily Faceoff goalies, into the starter ladder."""
    root = data_dir()
    today = _today()
    slate = _parse_date(args.date)
    season = args.season or season_of(slate)
    mp = MoneyPuckClient(cache_dir=cache_dir())
    quotes = capture.read_day(root, slate)
    applied = 0
    try:
        roto = rotowire.fetch(slate, today=today, data_dir=root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # network / HTTP: the card still runs on projected goalies
        print(f"rotowire unavailable: {exc}", file=sys.stderr)
    else:
        if not roto.games:
            print(f"rotowire: no NHL games parsed for {slate} (raw: {roto.raw_path})")
        else:
            summary = lineup_feed.apply_slate(
                roto, mp=mp, data_dir=root, season=season, quotes=quotes
            )
            print(f"rotowire {slate}: {len(roto.games)} game(s) parsed -> {roto.raw_path}")
            for line in summary.lines():
                print(line)
            applied += 1
    try:
        dfo = dailyfaceoff.fetch(slate, today=today, data_dir=root)
    except Exception as exc:
        print(f"dailyfaceoff unavailable: {exc}", file=sys.stderr)
    else:
        if not dfo.games:
            print(f"dailyfaceoff: no NHL games parsed for {slate} (raw: {dfo.raw_path})")
        else:
            summary = lineup_feed.apply_slate(
                dfo, mp=mp, data_dir=root, season=season, quotes=quotes, source=dailyfaceoff.SOURCE
            )
            print(f"dailyfaceoff {slate}: {len(dfo.games)} game(s) parsed -> {dfo.raw_path}")
            for line in summary.lines():
                print(line)
            applied += 1
    if not applied:
        return 1
    if load_config().state_sync and not args.no_sync:
        state.auto_push(root, f"nhl lineups {slate}")
    return 0


def _nhlapi() -> NHLAPIClient:
    return NHLAPIClient(cache_dir=cache_dir() / "nhlapi")


def _lean_b2b(nights: list[str], slate: Date) -> tuple[frozenset[str], bool]:
    """Back-to-backs for the graded nights and the slate; True when tonight's are unread.

    A past night's schedule does not change, so it is cached for good rather than
    re-fetched on every card; only the slate and the day before it are read fresh.
    """
    fresh, kept = _nhlapi(), NHLAPIClient(cache_dir=cache_dir() / "nhlapi", cache_ttl=FOREVER)
    failed: set[Date] = set()

    def played(day: Date) -> set[str]:
        client = fresh if day >= slate - timedelta(days=1) else kept
        try:
            games = client.schedule_or_none(day)
        except Exception as exc:  # noqa: BLE001 - one bad night must not sink the card
            log.warning("schedule for %s unavailable: %s", day, exc)
            games = None
        if games is None or (day == slate and not games):
            failed.add(day)
            return set()
        return {t for g in games for t in (g.home, g.away)}

    b2b = screens.b2b_teams([*nights, slate.isoformat()], played)
    return b2b, bool(failed & {slate, slate - timedelta(days=1)})


def _minutes_to_next_drop(day: Date, now: datetime | None = None) -> float | None:
    """Minutes until the next puck drop still ahead on the slate (NHL API, free)."""
    now = now or datetime.now(timezone.utc)
    ahead: list[float] = []
    for g in _nhlapi().schedule(day):
        start = history._parse(g.start_utc) if g.start_utc else None
        if start is not None and start > now:
            ahead.append((start - now).total_seconds() / 60)
    return min(ahead, default=None)


def _ppp_lookup(logs: PlayerLogClient, season: int) -> grade.PPPLookup:
    """Power-play points per (player, game) off the skater game log, fetched once a player."""
    cache: dict[int, dict[int, int]] = {}

    def ppp_of(player_id: int, game_id: int) -> int | None:
        if player_id not in cache:
            try:
                cache[player_id] = {g.game_id: g.ppp for g in logs.skater_log(player_id, season)}
            except Exception:  # noqa: BLE001 - one missing log must not stop the grade
                log.exception("skater log %s", player_id)
                cache[player_id] = {}
        return cache[player_id].get(game_id)

    return ppp_of


def _scheduled_for(nights: list[str]) -> dict[str, list[str]]:
    """Matchups the NHL schedule lists per night (cached; nothing paid)."""
    out: dict[str, list[str]] = {}
    for night in nights:
        try:
            out[night] = [g.matchup for g in _nhlapi().schedule(Date.fromisoformat(night))]
        except Exception:  # noqa: BLE001 - a missing schedule just leaves the column blank
            log.exception("schedule %s", night)
    return out


def _results_for(day: Date) -> tuple[dict[str, GameResult], dict[str, str]]:
    """Finals and puck-drop times (UTC) by matchup."""
    client = _nhlapi()
    out: dict[str, GameResult] = {}
    starts: dict[str, str] = {}
    for game in client.schedule(day):
        starts[game.matchup] = game.start_utc
        res = client.result(game)
        if res is not None:
            out[game.matchup] = res
    return out, starts


def cmd_backfill_ppp(args: argparse.Namespace) -> int:
    """Grade ``sk_ppp`` rows on past graded nights that went out before PPP grading existed."""
    root = data_dir()
    sync = load_config().state_sync and not args.no_sync
    if sync:
        state.auto_pull(root)
    day, last = _parse_date(args.start), _parse_date(args.end)
    total = 0
    while day <= last:
        path = ledger.graded_path(root, day)
        rows = ledger.load_rows(path)
        if any(r.market == "sk_ppp" and r.outcome is None for r in rows):
            results, _ = _results_for(day)
            ppp_of = _ppp_lookup(PlayerLogClient(_nhlapi()), season_of(day))
            done = ledger.backfill_ppp(rows, results, ppp_of)
            if done:
                ledger.save_outcomes(rows, path)
            total += done
            print(f"{day}: graded {done} sk_ppp rows")
        day += timedelta(days=1)
    if sync and total:
        state.auto_push(root, f"nhl backfill sk_ppp: {total} graded")
    return 0


def cmd_podcast(args: argparse.Namespace) -> int:
    """Episode for the slate -> cached transcript -> per-game read JSON."""
    cfg = load_config()
    root = data_dir()
    slate = _parse_date(args.date)
    quotes = capture.read_day(root, slate)
    pairs = sorted((a, h) for a, h, _ in board.matchups(quotes).values())
    try:
        eps = podcast.episodes_for(podcast.fetch_feed(args.feed), slate)
    except Exception as exc:  # noqa: BLE001 - the feed is a convenience, not the card
        eps = []
        print(f"podcast feed unavailable: {exc}", file=sys.stderr)
    if not eps:
        read = podcast_read.PodcastRead(
            slate.isoformat(), "no_episode", detail="no episode titled for this slate in the feed"
        )
        print(f"podcast: no episode for {slate}; wrote {podcast_read.save_read(read, root)}")
        return 0
    ep = eps[0]
    try:
        tpath = podcast.ensure_transcript(
            ep, root, model=args.model, transcribe_missing=not args.no_transcribe
        )
    except podcast.PodcastUnavailable as exc:
        tpath = None
        print(f"podcast: {exc}", file=sys.stderr)
    if tpath is None:
        read = podcast_read.PodcastRead(
            slate.isoformat(),
            "no_transcript",
            detail="episode found, no transcript",
            episode_title=ep.title,
            published=ep.published,
        )
        print(
            f"podcast: {ep.title!r} found but not transcribed; wrote {podcast_read.save_read(read, root)}"
        )
        return 1
    if not pairs:
        print(f"no archived prices for {slate}; run `nhl-engine capture` first", file=sys.stderr)
        return 2
    lines = podcast.read_transcript(tpath)
    read = podcast_read.build_read(ep, lines, slate, pairs)
    calls = podcast_read.add_summaries(read, root, api_key=cfg.creds.openai_api_key)
    path = podcast_read.save_read(read, root)
    picks = podcast_picks.extract_picks(read)
    podcast_picks.save_picks(picks, podcast_picks.picks_path(root, slate))
    stated = podcast_picks.stated_records([(ln.stamp, ln.text) for ln in lines])
    podcast_picks.save_records(stated, podcast_picks.records_path(root, slate))
    mode = (
        f"openai summaries ({calls} new)"
        if cfg.creds.openai_api_key
        else "deterministic only (no OPENAI_API_KEY)"
    )
    print(
        f"podcast: {ep.title!r} -> {len(read.games)}/{len(pairs)} slate games found, "
        f"{len(read.not_on_slate)} other intros; {mode}; wrote {path}"
    )
    for g in read.games.values():
        print(
            f"  {g.matchup} @ {g.anchor}: ML {g.quoted_ml} total {g.quoted_total} ({len(g.mentions)} mentions)"
        )
    print(f"  {len(picks)} picks logged for grading; {len(stated)} self-reported records heard")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    root = data_dir()
    day = _parse_date(args.date) if args.date else _today() - timedelta(days=1)
    sync = load_config().state_sync and not args.no_sync
    if sync:
        state.auto_pull(root)
    rows = ledger.load_rows(ledger.predictions_path(root, day))
    if rows:
        results, starts = _results_for(day)
        graded = ledger.grade_rows(
            rows,
            results,
            capture.read_day(root, day),
            graded_at=capture.now_utc(),
            starts=starts,
            ppp_of=_ppp_lookup(PlayerLogClient(_nhlapi()), season_of(day)),
        )
        ledger.save_rows(graded, ledger.graded_path(root, day))
        done = sum(1 for r in graded if r.outcome is not None)
        print(f"{day}: graded {done}/{len(graded)} rows -> {ledger.graded_path(root, day)}")
        if sync:
            state.auto_push(root, f"nhl audit {day}: {done} graded")
    else:
        print(f"{day}: no ledger to grade")
    pod_path = podcast_picks.picks_path(root, day)
    picks = podcast_picks.load_picks(pod_path)
    if picks:
        if not rows:
            results, starts = _results_for(day)
        podcast_picks.grade_picks(
            picks, results, capture.read_day(root, day), graded_at=capture.now_utc(), starts=starts
        )
        podcast_picks.save_picks(picks, pod_path)
        print(f"{day}: graded {sum(1 for p in picks if p.outcome)}/{len(picks)} podcast picks")
    vsin_path = vsin_picks.picks_path(root, day)
    vpicks = podcast_picks.load_picks(vsin_path)
    if vpicks:
        if not rows and not picks:
            results, starts = _results_for(day)
        podcast_picks.grade_picks(
            vpicks, results, capture.read_day(root, day), graded_at=capture.now_utc(), starts=starts
        )
        podcast_picks.save_picks(vpicks, vsin_path)
        print(f"{day}: graded {sum(1 for p in vpicks if p.outcome)}/{len(vpicks)} VSiN best bets")
    all_rows = [
        r for p in sorted((root / "ledger").glob("graded_*.json")) for r in ledger.load_rows(p)
    ]
    _backfill_timing(root, all_rows)
    stale = ledger.unregradable(root)
    stale_note = (
        f"Not regradable (graded file but no predictions file): {', '.join(map(str, stale))}; "
        "their grades and CLV stay as first written."
    )
    if stale:
        print(stale_note, file=sys.stderr)
    print()
    print(scorecard.scorecard(all_rows).render(), end="")
    pod_all = podcast_picks.load_all_picks(root)
    if pod_all:
        print()
        print(podcast_picks.render(pod_all), end="")
    vsin_all = vsin_picks.load_all(root)
    if vsin_all:
        print()
        print(vsin_picks.render(vsin_all), end="")
    if not args.no_report:
        as_of = capture.now_utc()
        audit = ledger_report.build(
            all_rows,
            _starters_for(all_rows),
            _screen_context(root, all_rows),
            scheduled=_scheduled_for(sorted({r.slate_date for r in all_rows})),
        )
        if stale:
            audit.findings.append(stale_note)
        try:
            paths = ledger_report.write(audit, output_dir(), day, as_of=as_of)
        except Exception:  # the PDF is a view of the md; keep the md
            log.exception("audit pdf failed")
            paths = ledger_report.write(audit, output_dir(), day, as_of=as_of, pdf=False)
        print()
        print("\n".join(audit.findings or ["no findings yet"]))
        print("audit report: " + ", ".join(str(p) for p in paths.values()))
        if args.email:
            try:
                to = email.send_files(
                    load_config(),
                    subject=f"NHL ledger audit {day} -- {audit.buys.record} buys",
                    body=ledger_report.render_md(audit, as_of=as_of),
                    paths=paths,
                    to=args.to,
                )
                print(f"emailed audit -> {to}")
            except email.EmailNotConfigured as exc:
                print(f"email skipped: {exc}", file=sys.stderr)
    return 0


def _backfill_timing(root: Path, rows: list[ledger.LedgerRow]) -> None:
    """Puck drop and close time for rows graded before the ledger carried them (in memory only)."""
    nights: dict[str, list[ledger.LedgerRow]] = {}
    for r in rows:
        if r.outcome is not None and (not r.puck_drop or (r.clv is not None and not r.close_at)):
            nights.setdefault(r.slate_date, []).append(r)
    for night, missing in sorted(nights.items()):
        day = Date.fromisoformat(night)
        try:
            _, starts = _results_for(day)
        except Exception as exc:  # noqa: BLE001 - one bad night must not sink the report
            log.warning("puck drops for %s unavailable: %s", night, exc)
            continue
        ledger.stamp_timing(missing, starts, capture.read_day(root, day))


def _screen_context(root: Path, rows: list[ledger.LedgerRow]) -> screens.Context:
    """Back-to-backs off the NHL schedule and late skater news off the availability log."""
    nights = sorted({r.slate_date for r in rows})
    client = _nhlapi()

    def played(day: Date) -> set[str]:
        try:
            return {t for g in client.schedule(day) for t in (g.home, g.away)}
        except Exception as exc:  # noqa: BLE001 - one bad night must not sink the report
            log.warning("schedule for %s unavailable: %s", day, exc)
            return set()

    records = [a for n in nights for a in availability.read_log(root, Date.fromisoformat(n))]
    return screens.Context(
        b2b=screens.b2b_teams(nights, played),
        late_news=screens.late_news_teams(rows, records),
    )


def _starters_for(rows: list[ledger.LedgerRow]) -> dict[str, tuple[str, str]]:
    """Actual starters for every graded night (NHL API cache), for the goalie read."""
    out: dict[str, tuple[str, str]] = {}
    for night in sorted({r.slate_date for r in rows}):
        try:
            results, _ = _results_for(Date.fromisoformat(night))
        except Exception as exc:  # noqa: BLE001 - one bad night must not sink the report
            log.warning("starters for %s unavailable: %s", night, exc)
            continue
        for m, res in results.items():
            out[f"{night}|{m}"] = (res.away_starter, res.home_starter)
    return out


def cmd_calibrate(args: argparse.Namespace) -> int:
    root = data_dir()
    all_rows = [
        r for p in sorted((root / "ledger").glob("graded_*.json")) for r in ledger.load_rows(p)
    ]
    cal = Calibrator.fit(all_rows, min_samples=args.min_samples)
    cal.save(calibration_path(root))
    print(json.dumps({"graded": cal.counts, "fitted": sorted(cal.maps)}, indent=2))
    return 0


def _history_schedule(client: NHLAPIClient, start: Date, end: Date) -> dict[Date, list]:
    """Regular-season games per day from the (free, cached) NHL API schedule."""
    out: dict[Date, list] = {}
    for day in history.days_between(start, end):
        games = [g for g in client.schedule(day) if g.start_utc]
        if games:
            out[day] = games
    return out


def cmd_history(args: argparse.Namespace) -> int:
    """plan (free, NHL API) → print credits → pull only with --yes; `export` is offline."""
    cfg = load_config()
    root = data_dir()
    start, end = _parse_date(args.start), _parse_date(args.end)
    nhl = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")
    sched = _history_schedule(nhl, start, end)
    markets = tuple(args.markets.split(","))
    if args.action == "export":
        out = Path(args.out) if args.out else output_dir() / f"history_closes_{start}_{end}.csv"
        rows: list[dict[str, object]] = []
        missing_close = missing_final = 0
        for day, games in sorted(sched.items()):
            starts = {g.matchup: g.start_utc for g in games}
            results: dict[str, GameResult] = {}
            for g in games:
                res = nhl.result(g)
                if res is not None and res.decided:
                    results[g.matchup] = res
            day_rows = history.export_day(root, day, starts, results)
            missing_close += len(games) - len(day_rows)
            missing_final += sum(1 for r in day_rows if r["final_home"] == "")
            rows.extend(day_rows)
        n = history.write_csv(rows, out)
        print(f"{n} games with a close -> {out}")
        print(
            f"games without a close: {missing_close}; closes without an official final: {missing_final}"
        )
        return 0
    regions = len(args.regions.split(","))
    p = history.plan(
        {d.isoformat(): [g.start_utc for g in gs] for d, gs in sched.items()},
        markets=markets,
        regions=regions,
    )
    print(
        f"{start}..{end}: {p['games']} games on {len(sched)} days, "
        f"{p['snapshots']} snapshots x {p['credits_per_snapshot']} = {p['credits']} credits "
        f"(markets {','.join(markets)}, regions {args.regions})"
    )
    if args.action == "plan" or not args.yes:
        if args.action == "pull":
            print("nothing bought: re-run with --yes to spend the credits above", file=sys.stderr)
        return 0
    if not cfg.creds.has_odds_api():
        print("no Odds API key (THE_ODDS_API_KEY / ODDS_API_KEY)", file=sys.stderr)
        return 2
    client = OddsClient(
        "nhl",
        SPORT_KEY,
        cfg.creds.odds_api_key,
        regions=args.regions,
        cache_dir=cache_dir() / "oddsapi_hist",
        runtime_root=Path(args.runtime_root).expanduser() if args.runtime_root else None,
    )
    tot = {"snapshots": 0, "skipped": 0, "rows": 0, "games": 0}
    for day, games in sorted(sched.items()):
        c = history.pull_day(client, root, day, [g.start_utc for g in games], markets=markets)
        for k in tot:
            tot[k] += c[k]
        print(f"{day}: {c}")
    print(f"total: {tot}")
    return 0


def cmd_closetest(args: argparse.Namespace) -> int:
    """Rebuild the sim on the hist closes (offline, no credits) and print the fixed rule."""
    out = Path(args.out) if args.out else output_dir() / f"closetest_{args.start}_{args.end}.csv"
    if args.action == "report":
        print(closetest.report(closetest.read_csv(out)))
        return 0
    cfg = load_config()
    root = data_dir()
    start, end = _parse_date(args.start), _parse_date(args.end)
    nhl = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")
    mp = MoneyPuckClient(cache_dir=cache_dir())
    priors: dict[int, preseason.PreseasonPrior] = {}
    obs: list[closetest.Obs] = []
    done = (
        {o.game_date for o in closetest.read_csv(out)}
        if out.is_file() and not args.fresh
        else set()
    )
    if done:
        obs = closetest.read_csv(out)
    for day, games in sorted(_history_schedule(nhl, start, end).items()):
        if day.isoformat() in done or not history.hist_rows(root, day):
            continue
        season = season_of(day)
        if season not in priors:
            priors[season] = preseason.build(
                mp, season, r_yy=cfg.prior.r_yy, today=Date(season, 10, 1)
            )
        results = {g.matchup: r for g in games if (r := nhl.result(g)) is not None and r.decided}
        card = closetest.rebuild_day(
            root=root,
            day=day,
            season=season,
            cfg=cfg,
            mp=mp,
            prior=priors[season],
            rules=BookRules.load(rules_path(root)),
            games=games,
            results=results,
        )
        day_obs = closetest.observe(card, results)
        obs.extend(day_obs)
        closetest.write_csv(obs, out)
        print(f"{day}: {len(card.games)} priced, {len(day_obs)} obs, unpriced {len(card.unpriced)}")
    print(f"{len(obs)} obs -> {out}")
    print(closetest.report(obs))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nhl-engine")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="archive the current NHL board")
    cap.add_argument("--date", help="slate date (ET), default today")
    cap.add_argument("--board-only", action="store_true", help="skip per-event markets")
    cap.add_argument(
        "--if-drop-within",
        type=float,
        default=0,
        metavar="MIN",
        help="capture only when a puck drop is within MIN minutes (close pass); else exit free",
    )
    cap.add_argument("--max-events", type=int, default=64)
    cap.add_argument("--cache-ttl", type=int, default=0, help="seconds; 0 = always fetch fresh")
    cap.add_argument("--no-sync", action="store_true", help="do not pull/push engine-state")
    cap.set_defaults(func=cmd_capture)

    res = sub.add_parser("results", help="official finals from the NHL API")
    res.add_argument("--date")
    res.add_argument("--compact", action="store_true")
    res.set_defaults(func=cmd_results)

    arc = sub.add_parser("archive", help="summarise captured prices for a date")
    arc.add_argument("--date")
    arc.set_defaults(func=cmd_archive)

    br = sub.add_parser("book-rules", help="per-book settlement rules")
    br.add_argument("--add", nargs=3, metavar=("BOOK", "MARKET", "OT_RULE"))
    br.add_argument("--source", default="")
    br.add_argument("--check", nargs=2, metavar=("BOOK", "MARKET"))
    br.set_defaults(func=cmd_book_rules)

    pr = sub.add_parser("prior", help="build or refresh the preseason prior asset")
    pr.add_argument("--season", type=int, help="season start year, default from --date")
    pr.add_argument("--date", help="as-of date for the refresh schedule, default today")
    pr.add_argument("--force", action="store_true", help="rebuild even if fresh or frozen")
    pr.add_argument("--no-futures", action="store_true", help="skip the 1-credit outrights call")
    pr.set_defaults(func=cmd_prior)

    st = sub.add_parser("strength", help="as-of team-strength posteriors (no prices)")
    st.add_argument("--date", help="slate date, default today")
    st.add_argument("--season", type=int)
    st.add_argument("--metrics", help="comma-separated metric keys (default all)")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_strength)

    cd = sub.add_parser("card", help="price the archived board (joint sim) and write the card")
    cd.add_argument("--date", help="slate date, default today")
    cd.add_argument("--season", type=int)
    cd.add_argument("--tag", default="initial", help="early | initial | goalie | predrop | close")
    cd.add_argument(
        "--before",
        metavar="HH:MM",
        help="price only games starting before this time (ET); no card if there are none",
    )
    cd.add_argument("--ledger", action="store_true", help="also write the write-once ledger")
    cd.add_argument(
        "--force", action="store_true", help="write a versioned ledger beside an existing one"
    )
    cd.add_argument("--seed", type=int, help="sim seed (default derived from the date)")
    cd.add_argument("--no-sync", action="store_true")
    cd.add_argument("--no-pdf", action="store_true", help="skip the PDF card")
    cd.add_argument("--email", action="store_true", help="send the card (pdf/txt/md/xlsx attached)")
    cd.add_argument("--to", help="recipient (default NHLE_EMAIL_TO / MLBE_EMAIL_TO)")
    cd.set_defaults(func=cmd_card)

    sr = sub.add_parser("starter", help="record tonight's goalie for a team")
    sr.add_argument("team")
    sr.add_argument("player_id", type=int, help="NHL player id")
    sr.add_argument("--status", default="confirmed", choices=["confirmed", "probable", "projected"])
    sr.add_argument("--source", default="manual")
    sr.add_argument("--date")
    sr.set_defaults(func=cmd_starter)

    lu = sub.add_parser(
        "lineups",
        help="RotoWire + Daily Faceoff goalies, RotoWire injuries -> overrides/availability",
    )
    lu.add_argument("--date", help="today (default) or tomorrow")
    lu.add_argument("--season", type=int)
    lu.add_argument("--no-sync", action="store_true")
    lu.set_defaults(func=cmd_lineups)

    pc = sub.add_parser("podcast", help="Hockey Gambling Podcast read for the slate -> PDF")
    pc.add_argument("--date")
    pc.add_argument("--feed", default=podcast.FEED_URL)
    pc.add_argument("--model", default=podcast.WHISPER_MODEL, help="faster-whisper model")
    pc.add_argument("--no-transcribe", action="store_true", help="use a cached transcript only")
    pc.set_defaults(func=cmd_podcast)

    au = sub.add_parser("audit", help="grade a date's ledger and print the scorecard")
    au.add_argument("--date", help="default yesterday")
    au.add_argument("--no-sync", action="store_true", help="do not pull/push engine-state")
    au.add_argument("--no-report", action="store_true", help="skip the ledger audit md/pdf")
    au.add_argument("--email", action="store_true", help="email the ledger audit report")
    au.add_argument("--to", help="override recipient")
    au.set_defaults(func=cmd_audit)

    bp = sub.add_parser("backfill-ppp", help="grade sk_ppp rows on past graded nights")
    bp.add_argument("--start", required=True, help="first slate date")
    bp.add_argument("--end", required=True, help="last slate date")
    bp.add_argument("--no-sync", action="store_true", help="do not pull/push engine-state")
    bp.set_defaults(func=cmd_backfill_ppp)

    ca = sub.add_parser("calibrate", help="refit isotonic maps from graded ledgers")
    ca.add_argument("--min-samples", type=int, default=200)
    ca.set_defaults(func=cmd_calibrate)

    hi = sub.add_parser(
        "history",
        help="historical closes for the close test: plan (free) / pull (--yes spends) / export",
    )
    hi.add_argument("action", choices=["plan", "pull", "export"])
    hi.add_argument("--start", required=True, help="first game date (YYYY-MM-DD)")
    hi.add_argument("--end", required=True, help="last game date (YYYY-MM-DD)")
    hi.add_argument("--markets", default=",".join(history.CLOSE_MARKETS))
    hi.add_argument("--regions", default="us")
    hi.add_argument("--yes", action="store_true", help="pull: actually spend the credits printed")
    hi.add_argument("--runtime-root", help="engine_runtime dir for the shared credit ledger")
    hi.add_argument("--out", help="export: CSV path")
    hi.set_defaults(func=cmd_history)

    ct = sub.add_parser("closetest", help="preregistered close test on the hist closes (offline)")
    ct.add_argument("action", choices=["run", "report"])
    ct.add_argument("--start", required=True)
    ct.add_argument("--end", required=True)
    ct.add_argument("--out", help="observations CSV (resumes per day unless --fresh)")
    ct.add_argument("--fresh", action="store_true")
    ct.set_defaults(func=cmd_closetest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
