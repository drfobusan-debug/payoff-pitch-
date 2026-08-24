"""Per-market probability calibration, fitted on history and only applied on evidence.

The MLB engine paid for this lesson twice. A model whose "62%" wins 57% of the
time manufactures an edge out of arithmetic, and every EV, tier and veto
downstream inherits the fabrication -- so it fitted an isotonic map per market.
Then it learned the second half: a map is a statement about the probabilities one
particular engine produced, so a feature change silently re-imposes the bias the
map was built to remove (a corrected home-run probability moved 11.76% -> 11.62%,
i.e. nothing). Hence the basis stamp, and hence retirement *per market* rather
than all-or-nothing (#230/#232).

Two things are deliberately different here.

**The fit is out of time on history, not on the ledger.** MLB had to calibrate on
its own graded slates because nothing else existed; the NFL's closing lines and
finals go back to 2007, so a map can be fitted on seasons the holdout never sees
before a single Week 1 bet is placed. ``observations`` anchors the simulator to
each game's closing spread and total -- the same anchor
:mod:`nfl_engine.pipeline` uses live -- and reads the realised outcome off the
final score.

**A market is only corrected if the correction measurably helps.** ``fit`` scores
every candidate map against the identity map on the holdout seasons and accepts it
only when it wins by more than the noise in that comparison. That matters because
the mean here is the market's own closing number rather than a rating, so the
prior is that little correction is warranted -- and a curve fitted on noise would
add error while looking like diligence.

It currently accepts nothing. ``nfl-engine calibrate`` on 2007-2019, scored on
2020-2025 at 20,000 sims:

    market      n_fit   n_hold   identity    fitted      gain        se
    moneyline    6,924    3,376    0.21057   0.21024  +0.00033   0.00044
    spread       6,752    3,320    0.24988   0.24969  +0.00019   0.00047
    total        6,846    3,350    0.25016   0.24998  +0.00017   0.00032

Every gain is an order of magnitude below ``MIN_GAIN`` and smaller than its own
standard error -- three markets' worth of "the correction is noise" -- which is
consistent with the rung study's finding that the anchored distribution already
sits within 0.6pp of realised at every rung. So the shipped map file records the
measurement and corrects nothing, and the layer is here for the day a change to
the distribution makes that untrue: at which point the evidence, not a hunch,
turns it on, one market at a time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from nfl_engine.data import nflverse
from nfl_engine.market.ev import MONEYLINE, SPREAD, TOTAL
from nfl_engine.models.distribution import ScoreDistribution
from nfl_engine.models.drives import DriveSim, ExpectedGame

log = logging.getLogger(__name__)

# Which probabilities a fitted map is valid for. Bump this whenever a change
# moves the simulator's probabilities systematically; that retires every stored
# curve until it is refitted, which is the only thing standing between a fixed
# bias and a map that quietly re-applies it.
BASIS = "market-anchored-drivesim-2026.08"

MARKETS = (MONEYLINE, SPREAD, TOTAL)

# A map has to beat doing nothing by this much Brier on the holdout seasons, and
# by more than twice the standard error of that comparison, before it is applied.
# Both bars exist for the same reason: at ~3,300 holdout rows a curve fitted on
# noise wins by a few ten-thousandths about as often as it loses, and correcting
# a well-calibrated market on that basis adds error to every price on the slate.
MIN_GAIN = 0.0020
MIN_FIT_ROWS = 2000
MIN_HOLDOUT_ROWS = 1000

_N_BINS = 40


def _pav(y: list[float], w: list[float]) -> list[float]:
    """Pool-adjacent-violators: least-squares monotone (non-decreasing) fit."""
    vals = list(y)
    wts = list(w)
    blocks: list[list[float]] = [[vals[i], wts[i], 1.0] for i in range(len(vals))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0]:
            i += 1
            continue
        m0, w0, c0 = blocks[i]
        m1, w1, c1 = blocks[i + 1]
        tw = w0 + w1
        merged = [(m0 * w0 + m1 * w1) / tw if tw else 0.0, tw, c0 + c1]
        blocks[i : i + 2] = [merged]
        if i > 0:
            i -= 1
    out: list[float] = []
    for mean, _tw, count in blocks:
        out.extend([mean] * int(count))
    return out


@dataclass(frozen=True)
class IsotonicMap:
    """Piecewise-linear monotone map raw probability -> calibrated probability."""

    x: list[float]
    y: list[float]

    def apply(self, p: float) -> float:
        if not self.x:
            return p
        if p <= self.x[0]:
            cal = self.y[0]
        elif p >= self.x[-1]:
            cal = self.y[-1]
        else:
            cal = self.y[-1]
            for i in range(1, len(self.x)):
                if p <= self.x[i]:
                    x0, x1 = self.x[i - 1], self.x[i]
                    y0, y1 = self.y[i - 1], self.y[i]
                    frac = (p - x0) / (x1 - x0) if x1 > x0 else 0.0
                    cal = y0 + frac * (y1 - y0)
                    break
        return min(max(cal, 1e-6), 1 - 1e-6)

    @classmethod
    def fit(cls, pairs: list[tuple[float, int]], n_bins: int = _N_BINS) -> IsotonicMap:
        """Fit from ``(model_prob, outcome)`` pairs by binning, then PAV."""
        if not pairs:
            return cls([], [])
        buckets: dict[int, list[tuple[float, int]]] = {}
        for prob, won in pairs:
            idx = min(int(prob * n_bins), n_bins - 1)
            buckets.setdefault(idx, []).append((prob, won))
        xs: list[float] = []
        raw_y: list[float] = []
        wts: list[float] = []
        for idx in sorted(buckets):
            items = buckets[idx]
            xs.append(sum(p for p, _ in items) / len(items))
            raw_y.append(sum(w for _, w in items) / len(items))
            wts.append(float(len(items)))
        return cls(xs, _pav(raw_y, wts))


@dataclass(frozen=True)
class MarketFit:
    """One market's candidate curve and the out-of-time evidence about it.

    Stored whether or not it was accepted: a rejected curve with its numbers is
    what lets the next refit see that the market was measured and found already
    calibrated, rather than never looked at.
    """

    market: str
    curve: IsotonicMap
    accepted: bool
    n_fit: int
    n_holdout: int
    brier_identity: float
    brier_fitted: float
    gain: float
    gain_se: float
    basis: str = BASIS
    fit_seasons: str = ""
    holdout_seasons: str = ""

    def verdict(self) -> str:
        if self.accepted:
            return "applied"
        if self.basis != BASIS:
            return "retired (basis)"
        if self.n_fit < MIN_FIT_ROWS or self.n_holdout < MIN_HOLDOUT_ROWS:
            return "not fitted (thin)"
        return "no correction (identity wins)"


@dataclass
class Calibrator:
    """The maps a slate prices through: only the accepted, current-basis ones.

    Absence is the point. A market with no accepted map takes **no** correction --
    not a pooled one fitted across the other markets, which is how MLB's early
    version applied a spread's bias to a total.
    """

    maps: dict[str, IsotonicMap] = field(default_factory=dict)
    fits: dict[str, MarketFit] = field(default_factory=dict)

    @classmethod
    def from_fits(cls, fits: dict[str, MarketFit]) -> Calibrator:
        maps = {
            market: f.curve
            for market, f in fits.items()
            if f.accepted and f.basis == BASIS and f.curve.x
        }
        return cls(maps=maps, fits=dict(fits))

    def apply(self, market: str, prob: float) -> float:
        curve = self.maps.get(market)
        return prob if curve is None else curve.apply(prob)

    def applied_markets(self) -> tuple[str, ...]:
        return tuple(sorted(self.maps))

    def stamp(self) -> str:
        """One line for a card or a workbook: what is being corrected, and on what."""
        applied = self.applied_markets()
        if not applied:
            return f"calibration: none applied (basis {BASIS})"
        return f"calibration: {', '.join(applied)} on basis {BASIS}"


def observations(*, first: int = 2007, sims: int = 20000) -> list[tuple[int, str, float, int]]:
    """``(season, market, model probability, outcome)`` for every played game.

    Anchored to each game's closing spread and total, which is the same anchor
    :func:`nfl_engine.pipeline.price_slate` uses live, so the probabilities being
    calibrated are the ones the engine actually produces. Distributions are cached
    by rounded (margin, total) because a decade of closing numbers only holds a
    few hundred distinct pairs.

    Both sides of every line are kept: a map is fitted on the probability axis, and
    one side alone only ever populates half of it. Pushes are dropped -- a game
    that landed exactly on the number resolves as no bet, so it is not evidence
    about a probability.
    """
    games = nflverse.games()
    played = games[
        games.result.notna()
        & games.spread_line.notna()
        & games.total_line.notna()
        & (games.season >= first)
    ]
    sim = DriveSim(n_sims=sims, seed=17)
    cache: dict[tuple[float, float], ScoreDistribution] = {}
    rows: list[tuple[int, str, float, int]] = []
    for row in played.itertuples():
        margin, total_line = float(row.spread_line), float(row.total_line)
        key = (round(margin * 2) / 2, round(total_line * 2) / 2)
        if key not in cache:
            cache[key] = sim.simulate(
                ExpectedGame(
                    home_points=(key[1] + key[0]) / 2.0,
                    away_points=(key[1] - key[0]) / 2.0,
                )
            )
        dist = cache[key]
        season, home_margin, points = int(row.season), float(row.result), float(row.total)
        if home_margin != 0.0:
            rows.append(
                (season, MONEYLINE, dist.moneyline(home=True).conditional, int(home_margin > 0))
            )
            rows.append(
                (season, MONEYLINE, dist.moneyline(home=False).conditional, int(home_margin < 0))
            )
        # The home side lays the closing number, the away side takes it, and each
        # side's cover is read off its own handicap: negating the point does not
        # give the other side's probability.
        if home_margin != margin:
            rows.append(
                (season, SPREAD, dist.spread(-margin).conditional, int(home_margin > margin))
            )
            rows.append(
                (
                    season,
                    SPREAD,
                    dist.spread(margin, home=False).conditional,
                    int(home_margin < margin),
                )
            )
        if points != total_line:
            rows.append(
                (season, TOTAL, dist.total(total_line, over=True).conditional, int(points > total_line))
            )
            rows.append(
                (
                    season,
                    TOTAL,
                    dist.total(total_line, over=False).conditional,
                    int(points < total_line),
                )
            )
    return rows


def fit(
    rows: list[tuple[int, str, float, int]],
    *,
    cutoff: int,
) -> dict[str, MarketFit]:
    """Fit and judge one curve per market from ``(season, market, prob, outcome)``.

    Seasons up to ``cutoff`` train; everything after scores. The split is by
    season rather than at random because a random split leaks: two rows of the
    same game (both sides of one line) would land on either side of it, and a
    curve would be graded on games it had already seen.
    """
    out: dict[str, MarketFit] = {}
    seasons = [season for season, _m, _p, _y in rows]
    fit_span = _span([s for s in seasons if s <= cutoff])
    hold_span = _span([s for s in seasons if s > cutoff])
    for market in sorted({market for _s, market, _p, _y in rows}):
        train = [(p, y) for s, m, p, y in rows if m == market and s <= cutoff]
        holdout = [(p, y) for s, m, p, y in rows if m == market and s > cutoff]
        curve = IsotonicMap([], [])
        identity = brier = gain = gain_se = 0.0
        accepted = False
        if len(train) >= MIN_FIT_ROWS and len(holdout) >= MIN_HOLDOUT_ROWS:
            curve = IsotonicMap.fit(train)
            diffs = [
                (p - y) ** 2 - (curve.apply(p) - y) ** 2 for p, y in holdout
            ]
            identity = sum((p - y) ** 2 for p, y in holdout) / len(holdout)
            brier = sum((curve.apply(p) - y) ** 2 for p, y in holdout) / len(holdout)
            gain = sum(diffs) / len(diffs)
            gain_se = _se(diffs)
            accepted = gain >= MIN_GAIN and gain >= 2.0 * gain_se
        out[market] = MarketFit(
            market=market,
            curve=curve,
            accepted=accepted,
            n_fit=len(train),
            n_holdout=len(holdout),
            brier_identity=identity,
            brier_fitted=brier,
            gain=gain,
            gain_se=gain_se,
            fit_seasons=fit_span,
            holdout_seasons=hold_span,
        )
    return out


def _span(seasons: list[int]) -> str:
    return f"{min(seasons)}-{max(seasons)}" if seasons else ""


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (var / n) ** 0.5


def report_lines(fits: dict[str, MarketFit]) -> list[str]:
    """The measurement, market by market, as a table a refit can be judged on."""
    head = (
        f"{'market':10s} {'n_fit':>7s} {'n_hold':>7s} {'identity':>9s} "
        f"{'fitted':>9s} {'gain':>9s} {'se':>8s}  verdict"
    )
    lines = [head]
    for market, f in sorted(fits.items()):
        lines.append(
            f"{market:10s} {f.n_fit:7d} {f.n_holdout:7d} {f.brier_identity:9.5f} "
            f"{f.brier_fitted:9.5f} {f.gain:+9.5f} {f.gain_se:8.5f}  {f.verdict()}"
        )
    return lines


def write_maps(path: Path, fits: dict[str, MarketFit]) -> None:
    """Persist every measured market, accepted or not, with its own basis stamp."""
    payload = {
        "basis": BASIS,
        "markets": {
            market: {
                "basis": f.basis,
                "accepted": f.accepted,
                "x": f.curve.x,
                "y": f.curve.y,
                "n_fit": f.n_fit,
                "n_holdout": f.n_holdout,
                "brier_identity": round(f.brier_identity, 6),
                "brier_fitted": round(f.brier_fitted, 6),
                "gain": round(f.gain, 6),
                "gain_se": round(f.gain_se, 6),
                "fit_seasons": f.fit_seasons,
                "holdout_seasons": f.holdout_seasons,
            }
            for market, f in sorted(fits.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_maps(path: Path) -> Calibrator:
    """Load a map file, applying only the markets that earned it on this basis.

    Three ways a stored curve is refused, and each is per market: it was fitted on
    another basis, it was measured and rejected, or its recorded gain no longer
    clears the current bar (which is how tightening ``MIN_GAIN`` retires a curve
    without a refit). Everything refused is still loaded into ``fits`` so a report
    can say *why* nothing is being corrected.
    """
    fits: dict[str, MarketFit] = {}
    maps: dict[str, IsotonicMap] = {}
    data = json.loads(path.read_text(encoding="utf-8"))
    pooled = str(data.get("basis", ""))
    stale: list[str] = []
    for market, raw in data.get("markets", {}).items():
        basis = str(raw.get("basis", pooled))
        curve = IsotonicMap(list(raw.get("x", [])), list(raw.get("y", [])))
        gain = float(raw.get("gain", 0.0))
        gain_se = float(raw.get("gain_se", 0.0))
        accepted = (
            bool(raw.get("accepted", False))
            and basis == BASIS
            and bool(curve.x)
            and gain >= MIN_GAIN
            and gain >= 2.0 * gain_se
        )
        fits[market] = MarketFit(
            market=market,
            curve=curve,
            accepted=accepted,
            n_fit=int(raw.get("n_fit", 0)),
            n_holdout=int(raw.get("n_holdout", 0)),
            brier_identity=float(raw.get("brier_identity", 0.0)),
            brier_fitted=float(raw.get("brier_fitted", 0.0)),
            gain=gain,
            gain_se=gain_se,
            basis=basis,
            fit_seasons=str(raw.get("fit_seasons", "")),
            holdout_seasons=str(raw.get("holdout_seasons", "")),
        )
        if accepted:
            maps[market] = curve
        elif basis != BASIS:
            stale.append(market)
    if stale:
        log.warning(
            "calibration map %s: %s fitted on another basis than %r, ignored until refit",
            path.name,
            ", ".join(sorted(stale)),
            BASIS,
        )
    return Calibrator(maps=maps, fits=fits)


def shipped_path() -> Path:
    return Path(__file__).with_name("data") / "calibration.json"


def load() -> Calibrator:
    """The calibrator a live slate prices through, or an empty one."""
    path = shipped_path()
    if not path.exists():
        return Calibrator()
    try:
        return read_maps(path)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("calibration map %s unreadable (%s); pricing uncalibrated", path, exc)
        return Calibrator()
