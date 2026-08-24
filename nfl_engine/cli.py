"""The week, as commands: `capture`, `price`, `close`, `grade`, `report`, plus
`replay` and the unattended `job`.

In the order a week actually happens: archive the board; price it and write every
selection *and rejection* to the ledger; archive and stamp the closing number so
CLV can be scored; settle against final scores; read the record back with PPV/NPV
against the base rate.

**Nothing here can stake money.** There is no stake, bankroll or Kelly argument
anywhere in the engine, and every ledger row is written ``mode=paper``. That is
not a flag that could be flipped by accident -- placing a bet would require code
that does not exist. `price` prints the fact on every run, so a screenshot of a
run can never be mistaken for a bet slip.

Two properties the dry run needs that the phase-5 commands did not have:

**Pricing appends; it never rewrites.** Re-running on a moved board adds only
new positions and keeps the first price seen, because keeping the latest number
would hand the paper record the best price of the week in hindsight.

**What gets priced is what got archived.** Every fetch writes a timestamped
snapshot before anything is priced, so a recommendation can be checked against the
board that existed at the time rather than against a memory of it.

`replay` runs a played week at its closing prices through these same functions,
which is the only way to exercise grading, CLV and the report out of season.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

from nfl_engine import calibration
from nfl_engine import replay as replay_mod
from nfl_engine.audit import availability, outside
from nfl_engine.audit.ledger import (
    PAPER,
    LedgerEntry,
    apply_close,
    close_is_final,
    entry_from_bet,
    grade,
    load_ledger,
    market_metrics,
    merge_ledger,
    metrics,
    screen_metrics,
    tier_metrics,
    update_ledger,
)
from nfl_engine.config import data_dir, load_config, output_dir
from nfl_engine.data import capture, espn, injuries, nflverse
from nfl_engine.data.oddsapi import Board, OddsAPIClient
from nfl_engine.features import books as books_mod
from nfl_engine.market.screens import tier_of
from nfl_engine.models.drives import DriveSim
from nfl_engine.output.card import build_card, render_html, render_markdown, render_pdf
from nfl_engine.output.email import EmailNotConfigured, send_package
from nfl_engine.output.excel import build_workbook
from nfl_engine.pipeline import price_slate, slate_buys
from nfl_engine.schemas import Game

log = logging.getLogger(__name__)

LEDGER_NAME = "nfl_ledger.csv"
PAPER_BANNER = "paper only: no stake is placed and no bankroll exists in this engine"
# Points of rating-versus-line disagreement worth printing. Reported only: the
# rating's disagreement explains none of the line's error (t +0.25), so this is a
# thing to look at, never a thing to bet.
RATING_EDGE_NOTE = 3.0


def ledger_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / LEDGER_NAME


def _client() -> OddsAPIClient:
    config = load_config()
    return OddsAPIClient(config.creds.odds_api_key, cache_dir=None)


def current_week(today: Date | None = None) -> tuple[int, int, Date]:
    """Season, week and that week's first kickoff date, from the schedule itself.

    Taken from nflverse rather than from the calendar because the NFL week is not
    a fixed seven days: week 1 starts on a Thursday, the international games move
    the window, and a hard-coded offset silently prices the wrong week.
    """
    day = today or Date.today()
    games = nflverse.games()
    upcoming = games[games.gameday.astype(str) >= day.isoformat()]
    frame = upcoming if len(upcoming) else games
    row = frame.sort_values("gameday").iloc[0]
    season, week = int(row.season), int(row.week)
    week_games = games[(games.season == season) & (games.week == week)]
    first = min(str(value) for value in week_games.gameday)
    return season, week, Date.fromisoformat(first)


@dataclass
class Fetched:
    season: int
    week: int
    captured_at: str
    games: list[Game]
    board: Board
    archived: Path | None = None


def _fetch(days: int, *, kind: str = capture.GAME_KIND, archive: bool = True) -> Fetched:
    """Pull the board and archive it before a probability is formed."""
    client = _client()
    season, week, first_day = current_week()
    taken = capture.stamp()
    if not client.available():
        log.warning("no Odds API key: nothing to fetch")
        return Fetched(season, week, taken, [], {})
    slate, board = client.fetch_board(season=season, week=week, first_day=first_day, days=days)
    games = list(slate.games)
    rows = capture.rows_from_board(
        board,
        season=season,
        week=week,
        captured_at=taken,
        dates={game.matchup(): game.game_date.isoformat() for game in games},
        event_ids={game.matchup(): game.game_id for game in games},
    )
    written = capture.write_snapshot(rows, season=season, week=week, kind=kind) if archive else None
    if rows:
        print(f"{capture.archive_summary(rows)}")
        print(f"  archived {written.name if written else '(unchanged: board had not moved)'}")
    if client.credits_remaining is not None:
        print(f"  Odds API credits remaining: {client.credits_remaining}")
    return Fetched(season, week, taken, games, board, written)


def cmd_capture(args: argparse.Namespace) -> int:
    """Archive prices and nothing else -- the command that has to run from now on.

    Game prices are recoverable from nflverse afterwards; prop prices are not
    recoverable from anywhere, at any price, which is why this runs through the
    preseason even though the props layer is blocked and nothing is bet.
    """
    fetched = _fetch(args.days)
    if args.props and fetched.games:
        client = _client()
        rows = client.fetch_props(
            fetched.games, captured_at=fetched.captured_at, max_events=args.max_events
        )
        path = capture.write_snapshot(
            rows, season=fetched.season, week=fetched.week, kind=capture.PROP_KIND
        )
        players = len({row.player for row in rows})
        markets = len({row.market for row in rows})
        print(f"props: {len(rows)} quotes, {players} players, {markets} markets")
        print(f"  archived {path.name if path else '(unchanged)'}")
    return 0


def _ledger_rows(pricings: list, captured_at: str) -> list[LedgerEntry]:
    return [
        entry_from_bet(
            bet,
            season=pricing.game.season,
            week=pricing.game.week,
            date=pricing.game.game_date.isoformat(),
            captured_at=captured_at,
            kickoff_utc=pricing.game.kickoff_utc or "",
            mode=PAPER,
        )
        for pricing in pricings
        for bet in pricing.bets
    ]


def _books(season: int, week: int, *, ratings: bool) -> books_mod.Books:
    """The books the slate is priced with, announced so a run says which it had."""
    if not ratings:
        print("  ratings off: pricing on the market alone")
        return books_mod.Books()
    book = books_mod.as_of(season, week)
    print(f"  {book.summary()}")
    return book


def _print_rating_notes(pricings: list) -> None:
    """The quarterback charge, and how far the rating is from the line.

    Printed rather than screened: with the market at weight 1.0 neither number
    moved a price, and a run that shows them is how a disagreement gets noticed
    before it is ever trusted.
    """
    for pricing in pricings:
        shot = pricing.forecast
        if shot is None:
            continue
        edge, line = shot.rating_edge_margin(), shot.market_margin
        for note in shot.qb_notes:
            print(f"  {pricing.game.matchup():12s} {note}")
        if edge is not None and line is not None and abs(edge) >= RATING_EDGE_NOTE:
            print(
                f"  {pricing.game.matchup():12s} rating disagrees by {edge:+.1f}"
                f" (rating {shot.rating_margin:+.1f}, line {line:+.1f})"
            )


def cmd_price(args: argparse.Namespace) -> int:
    fetched = _fetch(args.days)
    if not fetched.games:
        return 0
    book = _books(fetched.season, fetched.week, ratings=args.ratings)
    named = books_mod.attach_qbs(fetched.games, fetched.season, fetched.week)
    print(f"  {named} of {2 * len(fetched.games)} starting quarterbacks named")
    maps = calibration.load()
    print(f"  {maps.stamp()}")
    pricings = price_slate(
        fetched.games,
        fetched.board,
        book=book.ratings,
        starters=book.starters,
        sim=DriveSim(n_sims=args.sims),
        calibrator=maps,
    )
    entries = _ledger_rows(pricings, fetched.captured_at)
    added = merge_ledger(ledger_path(), entries) if args.write else []
    _print_rating_notes(pricings)
    buys = slate_buys(pricings)
    print(f"{len(entries)} selections, {len(buys)} survive the screens [{PAPER_BANNER}]")
    if args.write:
        print(f"  {len(added)} new ledger rows ({len(entries) - len(added)} already held)")
    for bet in buys[: args.top]:
        fair_ev = bet.ev_fair or 0.0
        print(
            f"  {bet.matchup:12s} {bet.label():14s} {bet.american:+6.0f} {bet.book:14s}"
            f" model {bet.model_prob:.3f} fair {bet.fair_prob or 0.0:.3f}"
            f" exec EV {fair_ev:+.3f} [{tier_of(bet).value}]"
        )
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    """Re-fetch the board and re-stamp the closing number until kickoff.

    Run as often as convenient: while a game is unstarted the stamp is only the
    best estimate of its close so far, and each run replaces it with a later
    price. At kickoff the number in the ledger becomes the close and this command
    stops touching the row -- an in-play price is not a closing price, and neither
    is the Wednesday price that `job` used to freeze on its first run of the week.
    """
    path = ledger_path()
    entries = load_ledger(path)
    if not entries:
        print("empty ledger")
        return 0
    fetched = _fetch(args.days, kind=capture.CLOSE_KIND)
    now = datetime.now(tz=timezone.utc)
    stamped = restamped = frozen = missing = 0
    for entry in entries:
        if entry.result:
            continue
        if close_is_final(entry, now=now):
            if entry.close_odds is None:
                missing += 1
            else:
                frozen += 1
            continue
        quote = _closing_quote(fetched.board, entry)
        if quote is None:
            continue
        had = entry.close_odds is not None
        apply_close(entry, quote[0], quote[1], captured_at=fetched.captured_at)
        restamped += 1 if had else 0
        stamped += 0 if had else 1
    if args.write:
        update_ledger(path, entries)
    print(
        f"closing prices stamped on {stamped} rows, {restamped} re-stamped nearer"
        f" kickoff, {frozen} already final"
    )
    if missing:
        print(f"  {missing} started rows never got a closing price: CLV unscorable")
    return 0


def _closing_quote(board: dict, entry: LedgerEntry) -> tuple[float, float | None] | None:
    odds = board.get(entry.matchup)
    if odds is None:
        return None
    if entry.market == "moneyline":
        quotes = odds.ml.get(entry.side, [])
    elif entry.market == "spread" and entry.line is not None:
        home_point = entry.line if entry.side == entry.matchup.split(" @ ")[-1] else -entry.line
        quotes = odds.spreads.get(home_point, {}).get(entry.side, [])
    elif entry.market == "total" and entry.line is not None:
        quotes = odds.totals.get(entry.line, {}).get(entry.side, [])
    else:
        return None
    same_book = [q for q in quotes if q.book == entry.book] or quotes
    if not same_book:
        return None
    quote = same_book[0]
    return (quote.american, quote.opposite_american)


def cmd_grade(args: argparse.Namespace) -> int:
    path = ledger_path()
    entries = load_ledger(path)
    if not entries:
        print("empty ledger")
        return 0
    scores = _final_scores(args.season)
    graded = 0
    for entry in entries:
        if entry.result:
            continue
        key = (entry.matchup, entry.date)
        final = scores.get(key)
        if final is None:
            continue
        home = entry.matchup.split(" @ ")[-1]
        grade(entry, final[0], final[1], home=home)
        graded += 1
    if args.write:
        update_ledger(path, entries)
    print(f"graded {graded} rows")
    return 0


def _final_scores(season: int | None) -> dict[tuple[str, str], tuple[int, int]]:
    games = nflverse.games()
    games = games[games.home_score.notna() & games.away_score.notna()]
    if season:
        games = games[games.season == season]
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for row in games.itertuples():
        matchup = f"{row.away_team} @ {row.home_team}"
        out[(matchup, str(row.gameday))] = (int(row.home_score), int(row.away_score))
    return out


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit the per-market maps on history and print what the holdout measured.

    Fitted on seasons through ``--cutoff`` and scored on the seasons after it, so
    the number that decides whether a map ships was never trained on. ``--write``
    stores every market's measurement, applied or not, which is what makes the next
    refit comparable to this one.
    """
    rows = calibration.observations(first=args.first, sims=args.sims)
    fits = calibration.fit(rows, cutoff=args.cutoff)
    for line in calibration.report_lines(fits):
        print(line)
    if args.write:
        path = calibration.shipped_path()
        calibration.write_maps(path, fits)
        print(f"  wrote {path}")
    print(f"  {calibration.Calibrator.from_fits(fits).stamp()}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Capture ESPN's FPI call on a week and grade it into the ledger, beside ours.

    Free and keyless, so it spends no Odds API credit. The rows are written
    ``source=fpi``: they are graded against the same final score and read beside
    our plays, and every measurement of the engine filters them out. Nothing here
    is an input -- no probability, price, screen or tier of ours is touched, and a
    week prices identically whether the benchmark was captured or not.
    """
    season = args.season
    week = args.week
    if season is None or week is None:
        season, week, _ = current_week()
    games = espn.projections(season, week)
    if not games:
        print(f"no FPI projections for {season} week {week}")
        return 0
    scores = _final_scores(season)
    finals = {
        game.matchup: scores[(game.matchup, game.date)]
        for game in games
        if (game.matchup, game.date) in scores
    }
    rows = outside.entries_from_fpi(games, finals, captured_at=capture.stamp())
    graded = sum(1 for row in rows if row.result)
    print(
        f"FPI: {len(games)} games read, {len(rows)} calls written"
        f" ({graded} already final) [benchmark: display only, never an input]"
    )
    for row in sorted(rows, key=lambda e: -e.model_prob):
        print(f"  {row.matchup:14s} {row.side:4s} {row.model_prob:.3f} {row.tier:10s} {row.result}")
    if args.write:
        # Appended, never rewritten, exactly as our own prices are: the read of
        # record is the first one taken, so a Sunday capture cannot replace the
        # Wednesday projection it was supposed to be judged beside. Rows captured
        # before the game settle later through `grade`, which is source-agnostic.
        added = merge_ledger(ledger_path(), rows)
        print(f"  {len(added)} new benchmark rows ({len(rows) - len(added)} already held)")
    return 0


def _week_matchups(season: int, week: int) -> list[tuple[str, str, str]]:
    """``(matchup, away, home)`` for one week's schedule."""
    games = nflverse.games()
    games = games[(games.season == season) & (games.week == week)]
    return [
        (f"{row.away_team} @ {row.home_team}", str(row.away_team), str(row.home_team))
        for row in games.itertuples()
    ]


def cmd_injuries(args: argparse.Namespace) -> int:
    """Record who is out, and what the number did around the news.

    Keyless and free. Every observation is stamped with the market at the last
    archived capture before the item was posted and the first one after it, which
    is the measurement the whole feature turns on: an absence the market has
    already priced is worth nothing, and only our own timestamped archive can say
    which kind we are looking at.

    Nothing written here reaches a price. The card reports the absences and the
    log accumulates the timing evidence; if that evidence ever justifies an input,
    that is a separate change with the numbers attached.
    """
    season, week = args.season, args.week
    if season is None or week is None:
        current_season, current, _ = current_week()
        season, week = season or current_season, week or current
    book = injuries.fetch_report()
    if not book:
        print("no injury report available; nothing recorded")
        return 0
    news = injuries.fetch_news(cache=data_dir() / "cache" / "injury_news.json")
    observed = capture.now_utc()
    rows: list[availability.Observation] = []
    for matchup, away, home in _week_matchups(season, week):
        for team in (away, home):
            for row in injuries.watched_for(book, team):
                rows.append(
                    availability.observe(
                        row,
                        season=season,
                        week=week,
                        matchup=matchup,
                        news=news.get(row.player_id),
                        observed=observed,
                    )
                )
    print(
        f"availability: {len(rows)} absences on {season} week {week}'s watched groups"
        " [reported, never priced]"
    )
    for obs in rows:
        move = "n/a" if obs.spread_move is None else f"{obs.spread_move:+.1f}"
        print(
            f"  {obs.matchup:14s} {obs.team:3s} {obs.group:5s} {obs.position:2s}"
            f" {obs.player:22s} {obs.designation:12s} move {move:>5s} {obs.timing}"
        )
    if args.write and availability.append(availability.log_path(), rows):
        counts = availability.timing_counts(availability.read_log(availability.log_path()))
        totals = " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if "/" not in k)
        print(f"  log now holds {totals or 'nothing'}")
    return 0


def _injury_step(args: argparse.Namespace) -> int:
    """The availability read inside the weekly job, where a free feed may not answer."""
    try:
        return cmd_injuries(args)
    except Exception:
        log.warning("injuries step failed; the rest of the week still ran", exc_info=True)
        print("  injury report unavailable (see log); no engine output depends on it")
        return 0


def _benchmark_step(args: argparse.Namespace) -> int:
    """The benchmark inside the weekly job, where a free outside feed may not answer.

    ESPN being down, or slow, or having renamed a stat is not a reason to lose the
    week's close, grade and card, so this step reports and moves on. It is the only
    step allowed to: everything else in the job is ours and a failure there is real.
    """
    try:
        return cmd_benchmark(args)
    except Exception:
        log.warning("benchmark step failed; the rest of the week still ran", exc_info=True)
        print("  FPI unavailable (see log); no engine output depends on it")
        return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Run played weeks at their closing prices through the live functions.

    The board has one book, so the de-vigged consensus is the price taken and the
    execution edge is zero: every replayed row is a Pass, and that is the right
    answer rather than a bug. What is being tested is the machinery -- append-once
    ledger writes, grading on the side taken, CLV against the price struck, and the
    PPV/NPV report over real outcomes.
    """
    weeks = replay_mod.played_weeks(args.season, args.weeks)
    if not weeks:
        print(f"no played games for {args.season}")
        return 0
    path = ledger_path()
    sim = DriveSim(n_sims=args.sims)
    maps = calibration.load()
    print(f"  {maps.stamp()}")
    priced = added = graded = closed = 0
    for week in weeks:
        taken = capture.stamp()
        if args.archive:
            capture.write_snapshot(
                week.quote_rows(taken),
                season=week.season,
                week=week.week,
                kind=capture.CLOSE_KIND,
            )
        book = _books(week.season, week.week, ratings=args.ratings)
        pricings = price_slate(
            week.games,
            week.board,
            book=book.ratings,
            starters=book.starters,
            sim=sim,
            calibrator=maps,
        )
        entries = _ledger_rows(pricings, taken)
        priced += len(entries)
        for entry in entries:
            quote = _closing_quote(week.board, entry)
            if quote is not None:
                apply_close(entry, quote[0], quote[1], captured_at=taken)
                closed += 1
            final = week.finals.get(entry.matchup)
            if final is not None:
                grade(entry, final[0], final[1], home=entry.matchup.split(" @ ")[-1])
                graded += 1
        if args.write:
            added += len(merge_ledger(path, entries))
    print(
        f"{args.season}: {len(weeks)} weeks, {priced} selections priced, "
        f"{closed} closed, {graded} graded [{PAPER_BANNER}]"
    )
    if args.write:
        print(f"  {added} new ledger rows ({priced - added} already held)")
    return 0


def cmd_job(args: argparse.Namespace) -> int:
    """The unattended weekly run: capture, price, close, grade, report.

    One entry point for cron or a double-click, so the archive accrues and the
    ledger settles without anyone remembering the order. Each step is the same
    function the individual command calls, and a step with nothing to do is not an
    error -- in the off-season the whole job is a no-op that still exits 0.
    """
    steps = [
        ("capture", cmd_capture),
        ("price", cmd_price),
        # After pricing, and best-effort: a benchmark that fails, or disagrees with
        # every play, can neither stop the week nor reach anything that formed a
        # price. It only writes its own rows.
        ("benchmark", _benchmark_step),
        ("injuries", _injury_step),
        ("close", cmd_close),
        ("grade", cmd_grade),
        ("report", cmd_report),
    ]
    if args.card or args.email:
        steps.append(("card", cmd_card))
    for name, func in steps:
        print(f"== {name}")
        func(args)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    entries = load_ledger(ledger_path())
    if not entries:
        print("empty ledger")
        return 0
    rows = [
        *tier_metrics(entries),
        *market_metrics(entries),
        *screen_metrics(entries),
    ]
    if args.all:
        rows.append(metrics(entries, lambda e: True, "ALL"))
    header = (
        f"{'':26s} {'n':>5s} {'win%':>7s} {'need':>7s} {'PPV+':>7s} {'NPV+':>7s}"
        f" {'ROI':>7s} {'units':>8s} {'CLV':>8s}"
    )
    print(header)
    for row in rows:
        print(
            f"{row.label:26s} {row.n:5d} {row.win_pct:7.4f} {row.required_win_pct:7.4f}"
            f" {row.ppv_lift:+7.4f} {row.npv_lift:+7.4f} {row.roi:+7.4f}"
            f" {row.units:+8.2f} {row.mean_clv:+8.4f}"
        )
    return 0


def _write(path: Path, data: bytes) -> bool:
    """Write one artifact, reporting failure instead of raising it.

    Each file in the package stands alone: a read-only directory, a pre-existing
    file owned by root, a full disk -- none of those may cost the caller the other
    artifacts, and none may surface as a traceback.
    """
    try:
        path.write_bytes(data)
    except OSError as exc:
        print(f"  {path.name} not written ({exc})")
        return False
    return True


def cmd_card(args: argparse.Namespace) -> int:
    """Write the week's package -- card, workbook, PDF -- and optionally email it.

    Built from the ledger, so it costs no Odds API credit and can be re-run for any
    week already priced. Each artifact is guarded on its own: a box without
    WeasyPrint's system libraries loses the PDF and still gets the workbook, and a
    machine without SMTP credentials keeps everything on disk. Losing the whole
    package to one optional attachment is the MLB failure mode this avoids.
    """
    entries = load_ledger(ledger_path())
    if not entries:
        print("empty ledger")
        return 0
    season, week = args.season, args.week
    if season is None or week is None:
        current_season, current, _ = current_week()
        season, week = season or current_season, week or current
    card = build_card(
        entries,
        season=season,
        week=week,
        calibration=calibration.load().stamp(),
        # Read off disk, from what `injuries` already recorded: the card makes no
        # network call, so a week's absences are shown exactly as they were known
        # when they were captured.
        absences=availability.read_log(availability.log_path(), season=season, week=week),
    )
    if not card.games:
        print(f"no priced rows for {season} week {week}")
        return 0
    text, page = render_markdown(card), render_html(card)
    out = output_dir()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"card not written ({exc}); {out} is not a writable directory")
        return 1
    stem = f"NFL_{season}_Week{week:02d}"
    attachments: list[tuple[str, bytes]] = []

    # The workbook is built and written first because it is the artifact that must
    # survive: everything else is a rendering of what it already holds, and a
    # failure on a later file must not be able to take it with it.
    try:
        workbook = build_workbook(card, entries)
    except Exception as exc:  # noqa: BLE001 - report the failure, keep the card
        print(f"  workbook not built ({exc})")
    else:
        if _write(out / f"{stem}.xlsx", workbook):
            attachments.append((f"{stem}.xlsx", workbook))

    if _write(out / f"{stem}.md", text.encode("utf-8")):
        attachments.insert(0, (f"{stem}.md", text.encode("utf-8")))
    _write(out / f"{stem}.html", page.encode("utf-8"))

    try:
        pdf = render_pdf(page)
    except Exception as exc:  # noqa: BLE001 - the PDF is the optional artifact
        print(f"  card PDF not rendered ({exc}); markdown attached instead")
    else:
        if _write(out / f"{stem}.pdf", pdf):
            attachments.append((f"{stem}.pdf", pdf))

    if not attachments:
        print(f"card: nothing could be written to {out}")
        return 1
    print(f"card: {len(card.plays())} plays over {len(card.games)} games -> {out / stem}.*")
    if not args.email:
        return 0
    try:
        recipient = send_package(
            load_config(),
            subject=f"{card.title()} -- {len(card.plays())} plays [paper]",
            html_body=page,
            text_body=text,
            to=args.to,
            attachments=attachments,
        )
    except EmailNotConfigured as exc:
        print(f"  email not sent ({exc}); artifacts are in {out}")
        return 0
    print(f"  emailed {len(attachments)} attachments to {recipient}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nfl-engine", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    price = sub.add_parser("price", help="price the board and write the ledger")
    price.add_argument("--days", type=int, default=8)
    price.add_argument("--sims", type=int, default=40000)
    price.add_argument("--top", type=int, default=25)
    price.add_argument("--write", action="store_true", default=True)
    price.add_argument("--no-write", dest="write", action="store_false")
    price.add_argument(
        "--no-ratings",
        dest="ratings",
        action="store_false",
        default=True,
        help="price on the market alone, without reading the play-by-play panel",
    )
    price.set_defaults(func=cmd_price)

    close = sub.add_parser("close", help="stamp the closing number for CLV")
    close.add_argument("--days", type=int, default=2)
    close.add_argument("--write", action="store_true", default=True)
    close.add_argument("--no-write", dest="write", action="store_false")
    close.set_defaults(func=cmd_close)

    grade_cmd = sub.add_parser("grade", help="settle rows against final scores")
    grade_cmd.add_argument("--season", type=int, default=None)
    grade_cmd.add_argument("--write", action="store_true", default=True)
    grade_cmd.add_argument("--no-write", dest="write", action="store_false")
    grade_cmd.set_defaults(func=cmd_grade)

    card_cmd = sub.add_parser("card", help="write (and optionally email) the week's package")
    card_cmd.add_argument("--season", type=int, default=None)
    card_cmd.add_argument("--week", type=int, default=None)
    card_cmd.add_argument("--email", action="store_true")
    card_cmd.add_argument("--to", default=None, help="override the recipient")
    card_cmd.set_defaults(func=cmd_card)

    calibrate = sub.add_parser(
        "calibrate", help="fit and judge the per-market maps on historical closing lines"
    )
    calibrate.add_argument("--first", type=int, default=2007)
    calibrate.add_argument("--cutoff", type=int, default=2019, help="last training season")
    calibrate.add_argument("--sims", type=int, default=20000)
    calibrate.add_argument(
        "--write",
        action="store_true",
        help="replace the shipped map file with this fit and its measurements",
    )
    calibrate.set_defaults(func=cmd_calibrate)

    report = sub.add_parser("report", help="tier, market and screen records")
    report.add_argument("--all", action="store_true")
    report.set_defaults(func=cmd_report)

    capture_cmd = sub.add_parser("capture", help="archive prices, price nothing")
    capture_cmd.add_argument("--days", type=int, default=8)
    capture_cmd.add_argument("--props", action="store_true", help="also archive player-prop prices")
    capture_cmd.add_argument("--max-events", type=int, default=32)
    capture_cmd.set_defaults(func=cmd_capture)

    bench_cmd = sub.add_parser(
        "benchmark",
        help="capture ESPN's FPI call beside ours (free; display only, never an input)",
    )
    bench_cmd.add_argument("--season", type=int, default=None)
    bench_cmd.add_argument("--week", type=int, default=None)
    bench_cmd.add_argument("--write", action="store_true", default=True)
    bench_cmd.add_argument("--no-write", dest="write", action="store_false")
    bench_cmd.set_defaults(func=cmd_benchmark)

    inj_cmd = sub.add_parser(
        "injuries",
        help="record who is out and what the number did around the news (free; reported only)",
    )
    inj_cmd.add_argument("--season", type=int, default=None)
    inj_cmd.add_argument("--week", type=int, default=None)
    inj_cmd.add_argument("--write", action="store_true", default=True)
    inj_cmd.add_argument("--no-write", dest="write", action="store_false")
    inj_cmd.set_defaults(func=cmd_injuries)

    replay_cmd = sub.add_parser("replay", help="run played weeks at their closing prices")
    replay_cmd.add_argument("--season", type=int, required=True)
    replay_cmd.add_argument("--weeks", type=int, nargs="*", default=None)
    replay_cmd.add_argument("--sims", type=int, default=20000)
    replay_cmd.add_argument("--archive", action="store_true", default=False)
    replay_cmd.add_argument("--write", action="store_true", default=True)
    replay_cmd.add_argument("--no-write", dest="write", action="store_false")
    replay_cmd.add_argument("--no-ratings", dest="ratings", action="store_false", default=True)
    replay_cmd.set_defaults(func=cmd_replay)

    job = sub.add_parser("job", help="capture, price, close, grade and report in order")
    job.add_argument("--days", type=int, default=8)
    job.add_argument("--sims", type=int, default=40000)
    job.add_argument("--top", type=int, default=25)
    job.add_argument("--props", action="store_true")
    job.add_argument("--max-events", type=int, default=32)
    job.add_argument("--season", type=int, default=None)
    job.add_argument("--week", type=int, default=None)
    job.add_argument("--all", action="store_true")
    job.add_argument("--card", action="store_true", help="also write the reader-facing package")
    job.add_argument("--email", action="store_true", help="email the package (implies --card)")
    job.add_argument("--to", default=None)
    job.add_argument("--write", action="store_true", default=True)
    job.add_argument("--no-write", dest="write", action="store_false")
    job.add_argument("--no-ratings", dest="ratings", action="store_false", default=True)
    job.set_defaults(func=cmd_job)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
