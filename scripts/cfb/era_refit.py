"""Does what the CFB engine learned from 2014-2025 still hold after the portal and NIL?

Read-only measurement; nothing here feeds the live price. Since 2021 (one-time
transfers, then NIL) last season's rating carries less into this one: SP+
year-to-year r fell from ~0.83 (2014-19) to ~0.76 (2021-25), and roster talent
predicts less. Three shipped numbers were chosen on the pooled 2014-2025 history:

* ``regression_factor`` 0.90 -- how much of a rating gap to keep;
* ``market_blend`` 0.35 -- the weight on the market margin, set by hand;
* the preseason prior -- last season's rating standing in for this one early on.

Each is re-measured by era (2014-19 vs 2021-25; 2020 dropped) and walk-forward
on 2022-2025, fitted either on all prior seasons or on post-portal seasons only.

Instruments, both point-in-time:

* **TeamRankings predictive** snapshots (an ensemble member), the last Saturday
  before kickoff -- reused from :mod:`scripts.cfb.stability_study`;
* **last season's final SP+** for the preseason question.

The market weight comes from the NHL engine's close test
(:mod:`nhl_engine.closetest`): ``margin - close = k * (model - close)``. ``k`` is
the share of the model's disagreement with the close that shows up in results,
so the weight the data supports on the market is ``1 - k`` (clipped to [0, 1]).
Portal churn (players in + out before Aug 1, z-scored within the season) enters
as ``k`` varying with churn, 2021 on only, because CFBD's portal feed starts there.

Result (6,409 non-neutral FBS games, TR snapshots and CFBD median closes)
------------------------------------------------------------------------
* **Shrink.** b = 0.98 (2014-19) vs 0.88 (2021-25), lower in every week bucket:
  ratings now overstate gaps by about 12%, so 0.90 fits the post-portal era
  better than the pooled 0.94-0.96 the old seasons pull it toward.
* **Market weight.** k = +0.12 +/- 0.06 pre-portal (weight 0.88) and
  +0.01 +/- 0.08 post-portal (weight ~1.0). After 2021 the rating's disagreement
  with the close is noise; the shipped 0.35 (k = 0.65) is ~8 SE too much model.
* **Transfer churn.** k1 = -0.01 +/- 0.10 per churn SD: no interaction. High-churn
  games are not where the rating goes wrong relative to the close.
* **Preseason prior.** Last season's SP+ in weeks 1-4: b ~0.81 in both eras and
  k <= 0 against the close -- the close already holds all of it.
* **Walk-forward 2022-2025** (n 2,345), mean |blend - final| minus the close's:
  shipped 0.90/0.35 +0.199 +/- 0.047 (worse by ~4 SE); fit on all prior +0.014;
  fit on 2021+ (and with churn) +0.000. No policy beats the close.

Caveats: TR is an ensemble member, not the engine's SP+/log5 rating, so the shrink
is on TR's scale; the engine blends with the bet-time market, this with the close.
Nothing here changes ``market_blend`` or ``regression_factor``.

    CFBD_API_KEY=... python -m scripts.cfb.era_refit
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable

import numpy as np

from cfb_engine.data.portal import build_portal_book
from cfb_engine.data.teamnames import school_key
from scripts.cfb import stability_study as S

POST = 2021
COVID = 2020
ERAS: dict[str, tuple[int, int]] = {"pre 2014-19": (2014, 2019), "post 2021-25": (2021, 2025)}
WEEKS: tuple[tuple[int, int], ...] = ((1, 4), (5, 9), (10, 15))
TEST_SEASONS = (2022, 2023, 2024, 2025)
SHIP_SHRINK = 0.90  # cfb_engine.config.FeatureParams.regression_factor
SHIP_BLEND = 0.35  # cfb_engine.config.ModelParams.market_blend


# -- data ------------------------------------------------------------------
def sp_final(season: int) -> dict[str, float]:
    rows = S._cfbd("/ratings/sp", f"sp_{season}", year=season)
    return {
        school_key(str(r["team"])): float(r["rating"])
        for r in rows
        if r.get("team") and r.get("team") != "nationalAverages" and r.get("rating") is not None
    }


def churn(season: int) -> dict[str, float]:
    if season < POST:
        return {}
    book = build_portal_book(S._cfbd("/player/portal", f"portal_{season}", year=season), season)
    return {team: float(p.players_in + p.players_out) for team, p in book.items()}


def enrich(
    rows: list[dict],
    prev_sp: Callable[[int], dict[str, float]] = lambda s: sp_final(s - 1),
    churn_of: Callable[[int], dict[str, float]] = churn,
) -> list[dict]:
    """Add ``market``, ``prev_gap`` (last season's SP+) and ``churn`` (z within season)."""
    out = []
    for season in sorted({r["season"] for r in rows}):
        season_rows = [r for r in rows if r["season"] == season]
        prev = prev_sp(season)
        raw = churn_of(season)
        teams = {t for r in season_rows for t in (r["home"], r["away"]) if t in raw}
        vals = np.array([raw[t] for t in teams]) if teams else np.array([])
        mu, sd = (float(vals.mean()), float(vals.std())) if len(vals) > 1 else (0.0, 0.0)
        for r in season_rows:
            if r.get("spread") is None:
                continue
            ph, pa = prev.get(r["home"]), prev.get(r["away"])
            ch, ca = raw.get(r["home"]), raw.get(r["away"])
            out.append(
                {
                    **r,
                    "market": -float(r["spread"]),
                    "prev_gap": None if ph is None or pa is None else ph - pa,
                    "churn": None
                    if ch is None or ca is None or sd == 0
                    else ((ch - mu) / sd + (ca - mu) / sd) / 2,
                }
            )
    return out


# -- fitting ---------------------------------------------------------------
def line(rows: list[dict], key: str = "gap") -> tuple[float, float, float, int]:
    """``margin ~ a + b * x``: slope b, intercept a, t(b), n."""
    sel = [r for r in rows if r.get(key) is not None]
    return S.ols(np.array([r[key] for r in sel]), np.array([r["margin"] for r in sel]))


def loso_pred(rows: list[dict], key: str = "gap") -> list[dict]:
    """Model margin per game from a fit that never saw that game's season."""
    out = []
    for season in sorted({r["season"] for r in rows}):
        b, a, _, n = line([r for r in rows if r["season"] != season], key)
        if n < 30:
            continue
        out += [
            {**r, "pred": b * r[key] + a}
            for r in rows
            if r["season"] == season and r.get(key) is not None
        ]
    return out


def close_test(rows: Iterable[dict]) -> tuple[float, float, int]:
    """``margin - market = k * (pred - market)``: k, SE(k), n."""
    sel = list(rows)
    d = np.array([r["pred"] - r["market"] for r in sel])
    y = np.array([r["margin"] - r["market"] for r in sel])
    if len(sel) < 30 or float(d @ d) == 0:
        return float("nan"), float("nan"), len(sel)
    k = float(d @ y / (d @ d))
    resid = y - k * d
    return k, float(np.sqrt(resid @ resid / (len(sel) - 1) / (d @ d))), len(sel)


def close_test_churn(rows: Iterable[dict]) -> tuple[tuple[float, float], tuple[float, float], int]:
    """``margin - market = (k0 + k1 * churn) * (pred - market)``: (k0, se), (k1, se), n."""
    sel = [r for r in rows if r.get("churn") is not None]
    nan = (float("nan"), float("nan"))
    if len(sel) < 60:
        return nan, nan, len(sel)
    d = np.array([r["pred"] - r["market"] for r in sel])
    c = np.array([r["churn"] for r in sel])
    x = np.column_stack([d, d * c])
    y = np.array([r["margin"] - r["market"] for r in sel])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    cov = float(resid @ resid) / (len(sel) - 2) * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    return (float(coef[0]), float(se[0])), (float(coef[1]), float(se[1])), len(sel)


def weight(k: float) -> float:
    """Market weight the close test supports: ``1 - k`` clipped to [0, 1]."""
    return 1.0 if not np.isfinite(k) else float(min(1.0, max(0.0, 1.0 - k)))


def in_era(rows: list[dict], era: str) -> list[dict]:
    lo, hi = ERAS[era]
    return [r for r in rows if lo <= r["season"] <= hi]


def in_weeks(rows: list[dict], lo: int, hi: int) -> list[dict]:
    return [r for r in rows if lo <= r["week"] <= hi]


# -- reports ---------------------------------------------------------------
def _k(label: str, k: tuple[float, float, int]) -> str:
    return (
        f"{label:<22} n={k[2]:5d}  k={k[0]:+.3f} ± {k[1]:.3f}  -> market weight {weight(k[0]):.2f}"
    )


def shrink_by_era(rows: list[dict]) -> None:
    print("\n=== 1. Shrink: margin ~ a + b*gap (b = share of the TR gap the data supports)")
    print(f"    shipped regression_factor {SHIP_SHRINK}")
    for era in ERAS:
        sel = in_era(rows, era)
        b, a, t, n = line(sel)
        parts = [f"{era:<14} all n={n:5d} b={b:.3f} (t {t:+.0f}) hfa={a:+.2f}"]
        for lo, hi in WEEKS:
            bw, _, _, nw = line(in_weeks(sel, lo, hi))
            parts.append(f"wk{lo}-{hi} {bw:.3f} (n{nw})")
        print("  ".join(parts))


def market_weight_by_era(rows: list[dict]) -> None:
    print("\n=== 2. Market weight: margin - close = k*(model - close), model = TR fit")
    print(f"    shipped market_blend {SHIP_BLEND} (i.e. k = {1 - SHIP_BLEND:.2f})")
    for era in ERAS:
        pred = loso_pred(in_era(rows, era))
        print(_k(era, close_test(pred)))
        for lo, hi in WEEKS:
            print(_k(f"  weeks {lo}-{hi}", close_test(in_weeks(pred, lo, hi))))
    post = loso_pred(in_era(rows, "post 2021-25"))
    (k0, s0), (k1, s1), n = close_test_churn(post)
    print(
        f"  post 2021-25 x churn   n={n:5d}  k0={k0:+.3f} ± {s0:.3f}  "
        f"k1 per churn SD={k1:+.3f} ± {s1:.3f}  (k1 < 0: high-churn games -> trust the model less)"
    )


def preseason_by_era(rows: list[dict]) -> None:
    print("\n=== 3. Preseason prior: last season's final SP+ gap, weeks 1-4")
    for era in ERAS:
        early = in_weeks(in_era(rows, era), 1, 4)
        b, a, t, n = line(early, "prev_gap")
        k = close_test(loso_pred(early, "prev_gap"))
        print(f"{era:<14} margin ~ prev SP+ gap: b={b:.3f} (t {t:+.0f}) n={n}")
        print(_k("  close test", k))


def _cluster_se(diff: np.ndarray, keys: list[tuple]) -> float:
    groups: dict[tuple, float] = {}
    for d, key in zip(diff, keys, strict=True):
        groups[key] = groups.get(key, 0.0) + float(d)
    n, g = len(diff), len(groups)
    mean = float(diff.mean())
    centred = [s - mean * sum(1 for k in keys if k == key) for key, s in groups.items()]
    return float(np.sqrt(sum(x * x for x in centred) / n**2 * g / max(1, g - 1)))


def fit_policy(train: list[dict], with_churn: bool) -> dict[str, float]:
    b, a, _, _ = line(train)
    pred = loso_pred(train)
    if with_churn:
        (k0, _), (k1, _), n = close_test_churn(pred)
        if n >= 60 and np.isfinite(k1):
            return {"b": b, "a": a, "k0": k0, "k1": k1}
    k, _, _ = close_test(pred)
    return {"b": b, "a": a, "k0": k, "k1": 0.0}


def walk_forward(rows: list[dict]) -> None:
    print("\n=== 4. Walk-forward 2022-2025: mean |blend - final| vs the close (lower is better)")
    pool = [r for r in rows if r["season"] != COVID]
    policies = ("shipped 0.90 / 0.35", "fit on all prior", "fit on 2021+", "fit on 2021+ x churn")
    err: dict[str, list[float]] = {p: [] for p in policies}
    mkt: list[float] = []
    keys: list[tuple] = []
    for season in TEST_SEASONS:
        test = [r for r in pool if r["season"] == season]
        prior = [r for r in pool if r["season"] < season]
        recent = [r for r in prior if r["season"] >= POST]
        hfa = line(prior)[1]
        fits = {
            "fit on all prior": fit_policy(prior, False),
            "fit on 2021+": fit_policy(recent, False),
            "fit on 2021+ x churn": fit_policy(recent, True),
        }
        picked = "  ".join(
            f"{p.removeprefix('fit on ')}: b {f['b']:.2f} w {weight(f['k0']):.2f}"
            for p, f in fits.items()
        )
        print(f"  {season} fitted -> {picked}")
        for r in test:
            mkt.append(abs(r["market"] - r["margin"]))
            keys.append((r["season"], r["week"]))
            ship = (1 - SHIP_BLEND) * (SHIP_SHRINK * r["gap"] + hfa) + SHIP_BLEND * r["market"]
            err["shipped 0.90 / 0.35"].append(abs(ship - r["margin"]))
            for p, f in fits.items():
                k = f["k0"] + f["k1"] * (r["churn"] or 0.0) if p.endswith("churn") else f["k0"]
                w = weight(k)
                blend = (1 - w) * (f["b"] * r["gap"] + f["a"]) + w * r["market"]
                err[p].append(abs(blend - r["margin"]))
    base = np.array(mkt)
    print(f"  {'market only (close)':<24} MAE {base.mean():.3f}  n={len(base)}")
    for p in policies:
        e = np.array(err[p])
        diff = e - base
        print(
            f"  {p:<24} MAE {e.mean():.3f}  vs close {diff.mean():+.3f} ± {_cluster_se(diff, keys):.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    rows = enrich([r for r in S.tr_rows() if not r["neutral"]])
    rows = [r for r in rows if r["season"] != COVID]
    print(
        f"{len(rows)} non-neutral FBS games with a point-in-time TR rating and a close (2020 out)"
    )
    shrink_by_era(rows)
    market_weight_by_era(rows)
    preseason_by_era(rows)
    walk_forward(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
