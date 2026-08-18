"""Morning power screen: rank the slate's softest arms, then who to hunt them with.

Runs beside the nightly card, not inside it. The engine declines to price a game
whose lineup is not posted; this reads Rotowire's expected order instead, which is
what makes it a *morning* job -- the answer is wanted before the board fills, not
after.

    python scripts/power_screen.py --date 2026-08-17 --arms 4 --keep 3
    python scripts/power_screen.py --date 2026-08-17 --email      # PDF to the audit inbox

Writes ``power_screen_<date>.{html,pdf}`` to the engine's output directory. See
``mlb_engine/output/power_screen.py`` for the five stages and every threshold, and
``--help`` for the knobs worth moving (``--min-pa``, ``--min-wrc``, ``--arms``).

The screen fetches no market and spends no Odds API credit. It does read the
card's own board when the nightly run has already written one for the same day
(``predictions_<date>.json`` in the audit directory), and appends the survivors'
priced rows as a section; ``--no-prices`` suppresses that, and nothing about the
screen or its ratings changes either way.

Every priced row it prints is recorded to ``power_screen_ledger.csv``, and the
previous day's rows are graded off the box score at the top of the next note, so
the screen carries its own record instead of reading the same each morning
regardless of what happened. ``--no-grade`` skips both; ``--grade-date`` picks the
day to grade.
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.audit import power_ledger
from mlb_engine.config import Config, RollingWindows, load_config
from mlb_engine.data.managers import DEFAULT_BF_CAP
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.results import GameResult, fetch_result
from mlb_engine.data.rotowire import RotowireClient
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import VSINClient
from mlb_engine.features.efficiency import build_pitcher_efficiency, opponent_discipline_factor
from mlb_engine.features.rolling import build_bullpen_profile
from mlb_engine.features.workload import _bf_per_start, expected_bf_cap
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.output import power_board, power_report
from mlb_engine.output.email import send_card_email
from mlb_engine.output.power_board import Board
from mlb_engine.output.power_screen import (
    MIN_BATTER_PA,
    MIN_WRC,
    BullpenCard,
    HitterLine,
    HitterView,
    MatchupSection,
    ScreenResult,
    StarterCard,
    apply_cuts,
    arsenal,
    arsenal_fit,
    batter_arsenal,
    batter_window_line,
    bf_pmf,
    contact_line,
    exposure,
    pa_vs_starter,
    rank_starters,
    starter_damage,
    wrc_plus,
)
from mlb_engine.recommendations import load_json
from mlb_engine.schemas import Slate, TeamGameInfo

log = logging.getLogger("power_screen")

# A team's plate appearances in a nine-inning game, for splitting a projected
# game by slot. Measured off the window rather than assumed, with this as the
# fallback when the frame carries no game keys.
FALLBACK_TEAM_PA = 38.6
TEAM_PA_SD = 4.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", type=Date.fromisoformat, default=Date.today())
    p.add_argument("--arms", type=int, default=8, help="starters to rank (all are listed)")
    p.add_argument("--keep", type=int, default=4, help="softest arms whose lineups are screened")
    p.add_argument("--min-pa", type=int, default=None, help="hand-split PA floor")
    p.add_argument("--min-wrc", type=float, default=None, help="window wRC+ floor")
    p.add_argument(
        "--no-power-exception",
        action="store_true",
        help="drop hitters the wRC+ cut removes even when their contact is elite",
    )
    p.add_argument("--refresh", action="store_true", help="re-download the Statcast window")
    p.add_argument(
        "--predictions",
        default=None,
        help="the card's predictions JSON to price the survivors off "
        "(default: the audit directory's file for --date)",
    )
    p.add_argument(
        "--no-prices",
        action="store_true",
        help="leave the board out even when the card has already priced the slate",
    )
    p.add_argument(
        "--grade-date",
        type=Date.fromisoformat,
        default=None,
        help="the recorded board to grade in this note (default: the day before --date)",
    )
    p.add_argument(
        "--no-grade",
        action="store_true",
        help="neither record this board nor grade an earlier one",
    )
    p.add_argument("--email", action="store_true", help="email the PDF")
    p.add_argument("--to", default=None, help="override the recipient")
    p.add_argument("--prepared-for", default=None, help="name in the note's subtitle")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _league_lines(window: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """League wOBA and xwOBA/PA over the window, split by the hand on the mound."""
    woba: dict[str, float] = {}
    xwoba: dict[str, float] = {}
    for hand in ("R", "L"):
        rows = window[window["p_throws"] == hand] if "p_throws" in window else window
        line = batter_window_line(rows)
        if line:
            woba[hand] = line["woba"]
            xwoba[hand] = line["xwoba_pa"]
    return woba, xwoba


def _team_pa_per_game(window: pd.DataFrame) -> float:
    """Average plate appearances a team gets, from the window's own games."""
    keys = ["game_date", "home_team", "away_team"]
    if not all(k in window for k in keys) or "events" not in window:
        return FALLBACK_TEAM_PA
    pa_rows = window[window["events"].notna()]
    if pa_rows.empty:
        return FALLBACK_TEAM_PA
    per_game = pa_rows.groupby(keys).size()
    if per_game.empty:
        return FALLBACK_TEAM_PA
    return float(per_game.mean()) / 2.0


def _projected_pa(slot: int, team_pa: float) -> float:
    """Expected PA for a lineup slot, from the same survival machinery as the split.

    Slot ``i`` bats a ``t``-th time whenever the team reaches ``9(t-1)+i`` plate
    appearances, so this and the starter split are built the same way and cannot
    disagree about what a turn is.
    """
    return pa_vs_starter(slot, bf_pmf(team_pa, TEAM_PA_SD, cap=60, limit=60))


def _bullpen_cards(
    frame: pd.DataFrame, teams: list[str], w: RollingWindows, as_of: Date
) -> dict[str, BullpenCard]:
    """Profile and rank every pen on the slate, so a rank means something.

    Ranked within the slate's own pens rather than all 30: the note only ever
    quotes a pen that appears in it, and a rank out of 30 built from a frame that
    may not cover every club would be a worse figure dressed as a better one.
    """
    profiles = []
    for team in teams:
        profile = build_bullpen_profile(
            frame,
            team,
            as_of,
            w.bullpen_days,
            w.bullpen_min_inning,
            skill_days=w.bullpen_skill_days,
            xwoba_shrink=w.bullpen_xwoba_shrink,
        )
        if profile.xwoba_allowed is None:
            continue
        profiles.append((team, profile))
    profiles.sort(key=lambda t: -(t[1].xwoba_allowed or 0.0))
    out: dict[str, BullpenCard] = {}
    for rank, (team, profile) in enumerate(profiles, 1):
        allowed = profile.allowed
        out[team] = BullpenCard(
            team=team,
            rank=rank,
            of_n=len(profiles),
            relief_pa=int(allowed.pa),
            xwoba=float(profile.xwoba_allowed or math.nan),
            k_pct=allowed.p_k,
            bb_pct=allowed.p_bb,
            hr_pct=allowed.p_hr,
            zone_pct=profile.zone_pct,
            late_k_pct=profile.allowed_leverage.p_k,
        )
    return out


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    day: Date = args.date
    form = cfg.windows.pitcher_form_days

    stats = MLBStatsClient()
    slate = stats.get_slate(day)
    repo = StatcastRepository(cfg.cache_dir)
    projected = _fill_expected_lineups(cfg, slate, day, stats, repo)

    frame = repo.max_window(
        day,
        [form, cfg.windows.batter_vs_rhp_days, cfg.windows.batter_vs_lhp_days,
         cfg.windows.bullpen_skill_days],
        refresh=args.refresh,
    )
    end = day - timedelta(days=1)
    start = end - timedelta(days=form - 1)
    window = frame[(frame["game_date"] >= start) & (frame["game_date"] <= end)]
    league_woba, league_xwoba = _league_lines(window)
    team_pa = _team_pa_per_game(window)
    log.info("window %s..%s, %d pitches, %.1f PA per team-game", start, end, len(window), team_pa)

    # --- stage 1: rank the arms
    cards: list[StarterCard] = []
    context: dict[int, tuple[TeamGameInfo, TeamGameInfo]] = {}  # starter id -> (lineup team, his own team)
    for game in slate.games:
        for team, opp in ((game.home, game.away), (game.away, game.home)):
            pitcher = team.probable_pitcher
            if pitcher is None or not pitcher.mlbam_id:
                continue
            rows = window[window["pitcher"] == pitcher.mlbam_id]
            throws = getattr(pitcher.throws, "value", pitcher.throws) or "R"
            cards.append(
                starter_damage(
                    rows,
                    name=pitcher.name,
                    mlbam_id=int(pitcher.mlbam_id),
                    team=team.abbrev,
                    opponent=opp.abbrev,
                    throws=str(throws),
                )
            )
            context[int(pitcher.mlbam_id)] = (opp, team)
    ranked = rank_starters(cards, top_n=max(args.arms, args.keep))
    if not ranked:
        log.warning("no starter cleared the readability floor on %s", day)
    targets = ranked[: args.keep]

    pens = _bullpen_cards(
        frame,
        sorted({t.abbrev for g in slate.games for t in (g.home, g.away)}),
        cfg.windows,
        day,
    )

    sections: list[MatchupSection] = []
    cut_log: list[HitterLine] = []
    for card in targets:
        lineup_team, pen_team = context[card.mlbam_id]
        section = _build_section(
            card=card,
            lineup_team=lineup_team,
            pen=pens.get(pen_team.abbrev),
            window=window,
            frame=frame,
            as_of=day,
            form=form,
            league_woba=league_woba,
            league_xwoba=league_xwoba,
            team_pa=team_pa,
            min_pa=args.min_pa,
            min_wrc=args.min_wrc,
            keep_power=not args.no_power_exception,
            projected=projected,
            cut_log=cut_log,
        )
        if section is not None:
            sections.append(section)

    result = ScreenResult(
        as_of=day,
        form_days=form,
        window_start=start,
        window_end=end,
        league_woba=league_woba,
        league_xwoba=league_xwoba,
        starters_ranked=ranked,
        sections=sections,
        cut_log=cut_log,
        has_run_value="delta_run_exp" in window.columns,
    )
    _write(result, cfg, args)


def _fill_expected_lineups(
    cfg: Config, slate: Slate, day: Date, stats: MLBStatsClient, repo: StatcastRepository
) -> bool:
    """Fill unposted orders from Rotowire. True when any order is a projection.

    The engine's own enrichment, because a morning screen must run before lineups
    post and the MLB feed carries an order only once it has. Returns whether any
    of the orders used are projections, which the note has to disclose.
    """
    from mlb_engine.pipeline import Pipeline, PipelineDeps

    unposted = [
        t.abbrev for g in slate.games for t in (g.home, g.away) if not t.lineup_confirmed()
    ]
    if not unposted:
        return False
    deps = PipelineDeps(
        stats=stats,
        statcast=repo,
        weather=WeatherProvider(),
        vsin=VSINClient(cfg.creds),
        rotowire=RotowireClient(cfg.creds),
    )
    try:
        Pipeline(cfg, deps)._enrich_expected_lineups(slate, day)
    except Exception as exc:  # pragma: no cover - a scraper outage must not kill the note
        log.warning("expected lineups unavailable (%s); using posted orders only", exc)
        return False
    return True


def _build_section(
    *,
    card: StarterCard,
    lineup_team: TeamGameInfo,
    pen: BullpenCard | None,
    window: pd.DataFrame,
    frame: pd.DataFrame,
    as_of: Date,
    form: int,
    league_woba: dict[str, float],
    league_xwoba: dict[str, float],
    team_pa: float,
    min_pa: int | None,
    min_wrc: float | None,
    keep_power: bool,
    projected: bool,
    cut_log: list[HitterLine],
) -> MatchupSection | None:
    """Stages 2-5 for one starter: score, cut, read the arsenal, scale by exposure."""
    hand = card.throws
    lg_woba = league_woba.get(hand, league_woba.get("R", 0.315))
    lg_xwoba = league_xwoba.get(hand, league_xwoba.get("R", 0.305))

    pool: list[HitterLine] = []
    slots = getattr(lineup_team, "lineup", []) or []
    for slot in slots:
        player = slot.player
        if not player.mlbam_id:
            continue
        rows = window[
            (window["batter"] == player.mlbam_id) & (window["p_throws"] == hand)
        ]
        line = batter_window_line(rows)
        if not line:
            continue
        pool.append(
            HitterLine(
                name=player.name,
                mlbam_id=int(player.mlbam_id),
                team=getattr(lineup_team, "abbrev", "UNK"),
                slot=slot.order,
                bats=getattr(player.bats, "value", player.bats),
                versus=card.name,
                pa=int(line["pa"]),
                wrc=wrc_plus(line["woba"], lg_woba),
                woba=line["woba"],
                obp=line["obp"],
                slg=line["slg"],
                ops=line["obp"] + line["slg"],
                ba=line["ba"],
                xba=line["xba"],
                xslg=line["xslg"],
                xwoba_pa=line["xwoba_pa"],
                xwoba_con=line["xwoba_con"],
                k=line["k"],
                bb=line["bb"],
                brl=line["brl"],
                hh=line["hh"],
                ev90=line["ev90"],
                osw=line["osw"],
            )
        )
    if not pool:
        log.warning("no readable hitters vs %s", card.name)
        return None
    kept = apply_cuts(
        pool,
        lg_xwoba,
        min_pa=min_pa if min_pa is not None else MIN_BATTER_PA,
        min_wrc=min_wrc if min_wrc is not None else MIN_WRC,
        keep_power=keep_power,
    )
    cut_log.extend(h for h in pool if not h.kept and h.cut_reason)

    pitcher_rows = window[window["pitcher"] == card.mlbam_id]
    lines, usage = arsenal(pitcher_rows)
    card.arsenal = lines
    card.usage = usage
    families = sorted(usage, key=lambda k: -usage[k])

    # Starter exit point, from the engine's own two constraints.
    all_rows = frame[frame["pitcher"] == card.mlbam_id]
    eff = build_pitcher_efficiency(all_rows, as_of, form)
    bf_cap = expected_bf_cap(all_rows, as_of, form, manager_cap=DEFAULT_BF_CAP)
    disc = opponent_discipline_factor(
        frame, [h.mlbam_id for h in pool], as_of, form
    )
    ppa = eff.blended_pitches_per_pa() / disc
    bf_mean = min(eff.pitch_cap / ppa, bf_cap) if ppa else float(bf_cap)
    log_bf = _bf_per_start(all_rows, as_of, form)
    bf_sd = (
        float(pd.Series(log_bf).std(ddof=1)) if len(log_bf) > 1 and not pd.isna(
            pd.Series(log_bf).std(ddof=1)
        ) else 4.0
    )
    pmf = bf_pmf(bf_mean, bf_sd, bf_cap)

    views: list[HitterView] = []
    for h in kept:
        rows = window[(window["batter"] == h.mlbam_id) & (window["p_throws"] == hand)]
        per_pitch = batter_arsenal(rows, families)
        overall = contact_line(rows)
        fit_w, fit_b, fallback = arsenal_fit(per_pitch, overall, usage)
        view = HitterView(
            line=h,
            per_pitch=per_pitch,
            overall=overall,
            fit_xwoba=fit_w,
            fit_xba=fit_b,
            fallback_share=fallback,
        )
        if h.slot:
            view.exposure = exposure(
                h.slot,
                _projected_pa(h.slot, team_pa),
                pmf,
                card.xwobacon,
                pen.xwoba if pen else None,
            )
        views.append(view)

    return MatchupSection(
        starter=card,
        bullpen=pen,
        hitters=views,
        starter_bf=bf_mean,
        starter_bf_sd=bf_sd,
        starter_bf_cap=bf_cap,
        pitches_per_pa=eff.blended_pitches_per_pa(),
        pitch_cap=eff.pitch_cap,
        discipline=disc,
        lineup_projected=projected,
    )


def _board(result: ScreenResult, cfg: Config, args: argparse.Namespace) -> Board | None:
    """The survivors' rows off the card's own run, if it has already priced today.

    Missing is the normal case in the morning -- the screen exists to run before
    the engine can price anything -- so a missing or unreadable file is a note,
    never an error.
    """
    if args.no_prices:
        return None
    path = (
        Path(args.predictions).expanduser()
        if args.predictions
        else power_board.default_predictions_path(cfg.audit_dir, result.as_of)
    )
    if not path.exists():
        log.info("no priced board at %s; the note will carry no prices", path)
        return None
    try:
        recs = load_json(path)
    except Exception as exc:  # pragma: no cover - a malformed ledger must not kill the note
        log.warning("could not read %s (%s); the note will carry no prices", path, exc)
        return None
    board = power_board.build(result, recs, source=path.name)
    log.info(
        "board: %d priced rows on %d of %d survivors",
        len(board.rows),
        len(board.priced),
        len(board.priced) + len(board.unpriced),
    )
    return board


def _ledger_path(cfg: Config) -> Path:
    return cfg.audit_dir / power_ledger.LEDGER_NAME


def _record(
    result: ScreenResult, board: Board | None, cfg: Config, args: argparse.Namespace
) -> None:
    """Write today's priced rows to the ledger so tomorrow's note can grade them.

    Only priced rows are recorded. A rating with no number beside it is not a
    position, and grading one would have to invent the price it never had.
    """
    if args.no_grade or board is None or not board.rows:
        return
    positions = power_ledger.positions_from_board(
        board, result.as_of, power_report.ratings(result)
    )
    try:
        power_ledger.record(_ledger_path(cfg), positions, result.as_of)
    except OSError as exc:  # pragma: no cover - a ledger write must not cost the note
        log.warning("could not record %d positions: %s", len(positions), exc)
        return
    log.info("recorded %d positions to %s", len(positions), _ledger_path(cfg))


def _review(
    cfg: Config, args: argparse.Namespace, day: Date
) -> tuple[power_ledger.Scorecard, list[power_ledger.GradedPosition]] | None:
    """Grade an earlier day's recorded board off the box scores.

    Best-effort throughout: the scorecard is the note's memory, not its subject,
    so a Stats API outage costs the reader the receipt rather than the note.
    """
    if args.no_grade:
        return None
    graded_day = args.grade_date or (day - timedelta(days=1))
    positions = power_ledger.positions_for(_ledger_path(cfg), graded_day)
    if not positions:
        log.info("no recorded board for %s; the note carries no scorecard", graded_day)
        return None
    results: dict[int, GameResult] = {}
    for pk in {p.game_pk for p in positions if p.game_pk is not None}:
        try:
            results[pk] = fetch_result(pk, cache_dir=cfg.cache_dir)
        except Exception as exc:  # noqa: BLE001 - one missing box score voids one game
            log.warning("could not fetch the box score for %s: %s", pk, exc)
    graded, voided = power_ledger.grade_positions(positions, results)
    return power_ledger.scorecard(graded_day, graded, voided), graded


def _print_review(card: power_ledger.Scorecard) -> None:
    o = card.overall
    if not o.n:
        print(f"{card.day}: nothing gradeable ({card.voided} voided)")
        return
    line = (
        f"{card.day}: {o.wins}-{o.losses}"
        + (f"-{o.pushes}" if o.pushes else "")
        + f", {o.units:+.2f}u"
        + (f", {card.voided} voided" if card.voided else "")
    )
    if card.model_brier is not None and card.market_brier is not None:
        closer = "model" if card.model_beat_market else "market"
        line += (
            f", Brier {card.model_brier:.3f} model / {card.market_brier:.3f} no-vig"
            f" ({closer} closer, n={card.scored_probs})"
        )
    print(line)


def _write(result: ScreenResult, cfg: Config, args: argparse.Namespace) -> None:
    """Write the HTML and PDF, print the one-line-per-survivor summary, maybe email."""
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    board = _board(result, cfg, args)
    # Grade before recording: an earlier day is never this one, but a --grade-date
    # pointing at today should read what the ledger held when the note was asked.
    review = _review(cfg, args, result.as_of)
    _record(result, board, cfg, args)
    html_path = out_dir / power_report.default_filename(result.as_of, "html")
    pdf_path = out_dir / power_report.default_filename(result.as_of, "pdf")
    html_path.write_text(
        power_report.render_html(
            result, prepared_for=args.prepared_for, board=board, review=review
        ),
        encoding="utf-8",
    )
    pdf = power_report.render_pdf(
        result, prepared_for=args.prepared_for, board=board, review=review
    )
    pdf_path.write_bytes(pdf)
    kept = sum(len(s.hitters) for s in result.sections)
    print(f"{html_path}\n{pdf_path}")
    print(
        f"{len(result.starters_ranked)} arms ranked, {len(result.sections)} screened, "
        f"{kept} hitters kept, {len(result.cut_log)} cut"
    )
    for section in result.sections:
        for view in section.hitters:
            e = view.exposure
            tail = f"  PA vs SP {e.pa_vs_starter:.2f}" if e else ""
            best = board.best_for_batter(view.line.name) if board else None
            if best is not None:
                tail += f"  {best.label} {best.american:+.0f}"
                if best.ev is not None:
                    tail += f" EV {best.ev * 100:+.1f}%"
            elif board is not None:
                tail += "  not priced"
            print(
                f"  {view.line.name:<20} vs {section.starter.name:<18} "
                f"wRC+ {view.line.wrc:>4.0f}  fit {view.fit_delta * 1000:+4.0f}{tail}"
            )
    if review is not None:
        _print_review(review[0])
    if args.email:
        to = send_card_email(
            cfg,
            subject=f"Power screen - {result.as_of:%a %-m/%d}",
            html_body=power_report.render_html(
                result, prepared_for=args.prepared_for, board=board, review=review
            ),
            text_body=f"Power screen for {result.as_of.isoformat()} attached.",
            to=args.to,
            attachments=[(pdf_path.name, pdf)],
        )
        print(f"emailed to {to}")


if __name__ == "__main__":
    main()
