"""Fit isolated 5v5 skater impacts (RAPM) and measure how much to trust them.

    python scripts/nhl/rapm_fit.py --from 2022 --to 2024 --out ~/.nhl_engine/studies/rapm.json

Per season (needs ``scripts/nhl/fetch_shifts.py`` to have warmed the cache):

1. Build 5v5 stints from shift charts + MoneyPuck shot xG.
2. Choose ridge ``lam``: fit on odd game ids, score weighted MSE of stint xG/60
   on even game ids over a grid; take the minimum. Reported per season.
3. Fit on the full season at that ``lam``; write every skater's off/def impact.
4. Reliability of the impacts: fit odd-game and even-game halves separately,
   correlate ``off`` and ``dfn`` across skaters with >= 300 min in each half,
   Spearman-Brown to the full season, EB ``k`` (seconds) for the extra shrink
   in ``features.lineup.rebuild``. Year-to-year r of impacts is reported too.

The numbers to paste into ``config.LineupParams`` are ``lam`` (median across
seasons) and ``k`` (median). Nothing is applied by the engine until then.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from engine_common.shrink import k_from_split_half, pearson, spearman_brown
from nhl_engine.config import cache_dir
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.nhlapi import NHLAPIClient, regular_season_game_ids
from nhl_engine.features.lineup import PER_60, RapmFit, Stint, build_stints, fit_rapm

LAM_GRID = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0)
MIN_HALF_MIN = 300.0


def season_stints(season: int, nhl: NHLAPIClient, mp: MoneyPuckClient) -> list[Stint]:
    shots = mp.shots(season)
    by_game = {int(g): df for g, df in shots.groupby("game_id")}
    out: list[Stint] = []
    for gid in regular_season_game_ids(season):
        pbp = nhl.play_by_play(gid)
        if not pbp:
            continue
        shifts = nhl.shifts(gid)
        roster = nhl.roster(gid)
        home = str(pbp.get("homeTeam", {}).get("abbrev", ""))
        away = str(pbp.get("awayTeam", {}).get("abbrev", ""))
        game_shots = by_game.get(gid % 1_000_000)
        if not shifts or not roster or game_shots is None:
            continue
        out.extend(build_stints(gid, shifts, roster, game_shots, home=home, away=away))
    return out


def predict(fit: RapmFit, stints: list[Stint]) -> tuple[float, float]:
    """Weighted MSE and weighted total of stint xG/60 under ``fit``."""
    sse = wsum = 0.0
    for s in stints:
        w = float(s.seconds)
        for attackers, defenders, xg, is_home in (
            (s.home, s.away, s.home_xg, 1.0),
            (s.away, s.home, s.away_xg, 0.0),
        ):
            pred = fit.intercept + fit.home_ice * is_home + fit.venues.get(s.venue, 0.0)
            pred += sum(fit.impacts[p].off for p in attackers if p in fit.impacts)
            pred += sum(fit.impacts[p].dfn for p in defenders if p in fit.impacts)
            y = xg / w * PER_60
            sse += w * (y - pred) ** 2
            wsum += w
    return sse / wsum if wsum else float("nan"), wsum


def choose_lam(
    odd: list[Stint], even: list[Stint], *, venue: bool
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for lam in LAM_GRID:
        fit = fit_rapm(odd, lam=lam, venue_effects=venue)
        mse, _ = predict(fit, even)
        scores[str(int(lam))] = mse
    best = min(scores, key=lambda k: scores[k])
    return float(best), scores


def half_reliability(odd_fit: RapmFit, even_fit: RapmFit) -> dict[str, float]:
    ids = [
        p
        for p in odd_fit.impacts
        if p in even_fit.impacts
        and odd_fit.impacts[p].toi >= MIN_HALF_MIN * 60
        and even_fit.impacts[p].toi >= MIN_HALF_MIN * 60
    ]
    out: dict[str, float] = {"players": float(len(ids))}
    n_half = float(np.mean([odd_fit.impacts[p].toi for p in ids])) if ids else float("nan")
    for key in ("off", "dfn"):
        xs = [getattr(odd_fit.impacts[p], key) for p in ids]
        ys = [getattr(even_fit.impacts[p], key) for p in ids]
        r = pearson(xs, ys)
        out[f"r_half_{key}"] = r
        out[f"r_season_{key}"] = spearman_brown(r)
        out[f"k_sec_{key}"] = k_from_split_half(r, n_half)
    out["n_half_sec"] = n_half
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2022)
    ap.add_argument("--to", dest="end", type=int, default=2024)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-venue", action="store_true", help="drop the per-arena columns")
    args = ap.parse_args(argv)
    venue = not args.no_venue

    nhl = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")
    mp = MoneyPuckClient(cache_dir=cache_dir())
    report: dict[str, object] = {"seasons": {}, "lam_grid": list(LAM_GRID)}
    fits: dict[int, RapmFit] = {}
    lams: list[float] = []
    ks: list[float] = []
    for season in range(args.start, args.end + 1):
        t0 = time.time()
        stints = season_stints(season, nhl, mp)
        if not stints:
            print(f"{season}: no stints (run fetch_shifts.py first)")
            continue
        odd = [s for s in stints if s.game_id % 2 == 1]
        even = [s for s in stints if s.game_id % 2 == 0]
        lam, scores = choose_lam(odd, even, venue=venue)
        _, scores_other = choose_lam(odd, even, venue=not venue)
        full = fit_rapm(stints, lam=lam, venue_effects=venue)
        odd_fit = fit_rapm(odd, lam=lam, venue_effects=venue)
        even_fit = fit_rapm(even, lam=lam, venue_effects=venue)
        rel = half_reliability(odd_fit, even_fit)
        fits[season] = full
        lams.append(lam)
        ks.append(float(np.mean([rel["k_sec_off"], rel["k_sec_dfn"]])))
        top = sorted(full.impacts.values(), key=lambda i: -i.net)[:5]
        report["seasons"][str(season)] = {  # type: ignore[index]
            "stints": full.stints,
            "minutes_5v5": full.seconds / 60,
            "games": len({s.game_id for s in stints}),
            "lam": lam,
            "venue_effects": venue,
            "cv_mse": scores,
            "cv_mse_without_venue" if venue else "cv_mse_with_venue": scores_other,
            "venues_xg60": {v: round(x, 4) for v, x in sorted(full.venues.items())},
            "intercept_xg60": full.intercept,
            "home_ice_xg60": full.home_ice,
            "reliability": rel,
            "top_net": [(i.player_id, round(i.off, 3), round(i.dfn, 3)) for i in top],
            "impacts": {
                str(p): [round(i.off, 4), round(i.dfn, 4), i.toi] for p, i in full.impacts.items()
            },
        }
        print(
            f"{season}: {full.stints} stints, {full.seconds / 3600:.0f} h 5v5, lam={lam:.0f}, "
            f"intercept {full.intercept:.3f} home +{full.home_ice:.3f}, "
            f"r_half off {rel['r_half_off']:.2f} def {rel['r_half_dfn']:.2f}, "
            f"k_sec off {rel['k_sec_off']:.0f} def {rel['k_sec_dfn']:.0f} "
            f"(n={rel['players']:.0f}, {time.time() - t0:.0f}s)"
        )

    yy: dict[str, float] = {}
    seasons = sorted(fits)
    for a, b in zip(seasons, seasons[1:], strict=False):
        common = [
            p
            for p in fits[a].impacts
            if p in fits[b].impacts
            and fits[a].impacts[p].toi >= 2 * MIN_HALF_MIN * 60
            and fits[b].impacts[p].toi >= 2 * MIN_HALF_MIN * 60
        ]
        for key in ("off", "dfn"):
            yy[f"{a}->{b}_{key}"] = pearson(
                [getattr(fits[a].impacts[p], key) for p in common],
                [getattr(fits[b].impacts[p], key) for p in common],
            )
        yy[f"{a}->{b}_n"] = float(len(common))
    report["year_to_year"] = yy
    report["recommend"] = {
        "lam": statistics.median(lams) if lams else None,
        "k_sec": statistics.median(ks) if ks else None,
        "venue_effects": venue,
    }
    print("year-to-year:", {k: round(v, 3) for k, v in yy.items()})
    print("recommend:", report["recommend"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
