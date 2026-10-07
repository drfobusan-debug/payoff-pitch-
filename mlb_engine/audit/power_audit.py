"""The power screen's audit article: yesterday graded, the whole ledger graded, and what it means.

Three parts, in this order:

1. **The day.** The arms the screen gated and what they did; every bat it kept,
   priced or not, against the starter it was screened on and over the full game
   (off the play-by-play, so "vs the starter" is literal); and the recorded
   positions against their prices.
2. **The whole record.** Every day the ledger holds, last run per day, cut by
   half, arm tier, market, bucket, tier and gate, with the model's and the
   printed probability's Brier beside the no-vig price's on the same rows.
3. **Commentary.** Problems the audit found in the receipt itself, what the
   evidence does and does not say, and what to do about it -- written from rules
   with the sample size in every sentence, never from a single day's luck.

Nothing here writes a price, a probability, a tier or a rating. It reads the
ledger (:mod:`power_ledger`) and the roster (:mod:`power_roster`) and the box
scores, and it says so when one of them is missing.
"""

from __future__ import annotations

import html
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.audit.power_ledger import (
    GradedPosition,
    Position,
    Record,
    bucket,
    grade_positions,
    name_key,
    summarize,
)
from mlb_engine.audit.power_roster import ARM, BAT, DROPPED, GATED, HELD, SCREENED, RosterRow
from mlb_engine.data.plays import PlateAppearance, Tally, starters, tally
from mlb_engine.data.results import GameResult
from mlb_engine.output.power_report import LABEL_EARN_ROWS, RATING_DISPLAY, earned_label

#: Standard errors a difference must clear before the commentary calls it one.
Z = 2.0
#: Fewer rows than this and a comparison is "underpowered" in so many words.
MIN_COMPARE = 100
#: One-way quotes above this share of the record and the Brier sample is thin.
ONE_WAY_WARN = 0.25
#: Held bats with no price above this share and the screen is mostly ungraded.
UNPRICED_WARN = 0.4
#: Voided rows above this share of the recorded ones is a data problem, not luck.
VOID_WARN = 0.10


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class BatResult:
    """One kept (or dropped) bat, against the arm and over the game."""

    row: RosterRow
    played: bool
    started: bool
    slot: int | None
    #: The starter who actually faced his side, and whether it was the one screened.
    starter_id: int | None
    probable_started: bool | None
    vs_starter: Tally
    game: Tally
    priced: bool

    @property
    def slot_moved(self) -> bool:
        return (
            self.started
            and self.row.slot is not None
            and self.slot is not None
            and self.row.slot != self.slot
        )


@dataclass(frozen=True)
class ArmResult:
    name: str
    player_id: int
    team: str
    #: Tier -> what that pass did with him (screened / not screened / gated: ...).
    calls: dict[str, str]
    siera: float | None
    started: bool | None
    line: dict[str, int]


@dataclass(frozen=True)
class Comparison:
    """Brier of the printed probability and the no-vig price, paired row by row."""

    n: int
    model: float
    shown: float
    market: float
    #: mean(shown error - market error) and its standard error; negative favours us.
    diff: float
    diff_se: float | None

    @property
    def verdict(self) -> str:
        if self.n < MIN_COMPARE:
            return "underpowered"
        if self.diff_se is None or self.diff_se == 0:
            return "no spread to judge"
        if self.diff <= -Z * self.diff_se:
            return "the screen's number beat the price"
        if self.diff >= Z * self.diff_se:
            return "the price beat the screen's number"
        return "no difference from the price"


@dataclass(frozen=True)
class Decision:
    base_rate: float
    bets: int
    ppv: float | None
    passes: int
    npv: float | None


@dataclass
class DayAudit:
    day: str
    run_id: str
    recorded: int
    graded: list[GradedPosition]
    voided: int
    roster_recorded: bool
    bats: list[BatResult] = field(default_factory=list)
    arms: list[ArmResult] = field(default_factory=list)
    #: Ledger rows on a bat the screen had dropped, which it should not hold.
    dropped_positions: list[str] = field(default_factory=list)


@dataclass
class TotalAudit:
    days: list[str]
    recorded: int
    graded: list[GradedPosition]
    voided: int
    by_day: list[Record]
    #: Days with a recorded roster, and every bat on them.
    roster_days: list[str] = field(default_factory=list)
    bats: list[BatResult] = field(default_factory=list)


@dataclass
class Commentary:
    problems: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- cuts


def display_bucket(position: Position) -> str:
    if position.category == "pitcher":
        return "arm"
    if not position.rating:
        return "(no rating)"
    b = bucket(position)
    return RATING_DISPLAY.get(b, b)


def _prop_key(p: Position) -> tuple[str, str, str, str, float | None]:
    who = str(p.player_id) if p.player_id is not None else name_key(p.batter)
    return (p.date, p.run_id, who, p.stat, p.line)


def two_sided(positions: Iterable[Position]) -> set[tuple[str, str, str, str, float | None]]:
    """Props recorded on both sides in one run: a pair that can only lose the vig."""
    sides: dict[tuple[str, str, str, str, float | None], set[str]] = defaultdict(set)
    for p in positions:
        sides[_prop_key(p)].add(p.side)
    return {k for k, v in sides.items() if {"over", "under"} <= v}


def without_pairs(graded: list[GradedPosition]) -> list[GradedPosition]:
    pairs = two_sided(g.position for g in graded)
    return [g for g in graded if _prop_key(g.position) not in pairs]


def group(graded: list[GradedPosition], key: str) -> list[Record]:
    """Records by ``half`` / ``arm tier`` / ``market`` / ``bucket`` / ``tier`` / ``gate``."""
    buckets: dict[str, list[GradedPosition]] = defaultdict(list)
    for g in graded:
        p = g.position
        if key == "half":
            k = "arms" if p.category == "pitcher" else "hitters"
        elif key == "arm tier":
            k = f"{p.arm_tier or 'soft'} arms"
        elif key == "market":
            k = p.market
        elif key == "tier":
            k = p.tier or "(not recorded)"
        elif key == "gate":
            k = "bought" if p.is_buy else (p.gate or "(not recorded)")
        else:
            k = display_bucket(p)
        buckets[k].append(g)
    return [summarize(k, v) for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))]


def bucket_records(graded: list[GradedPosition]) -> dict[str, Record]:
    """Raw bucket key -> record, for the word the record has earned."""
    out: dict[str, list[GradedPosition]] = defaultdict(list)
    for g in graded:
        if g.position.rating and g.position.category != "pitcher":
            out[bucket(g.position)].append(g)
    return {k: summarize(k, v) for k, v in out.items()}


def compare(graded: list[GradedPosition]) -> Comparison | None:
    """The three Briers on the decided two-sided rows, and the paired difference."""
    rows = [
        (g.position, 1 if g.result == WIN else 0)
        for g in graded
        if g.result != PUSH and g.position.devigged and g.position.fair_prob is not None
    ]
    if not rows:
        return None
    n = len(rows)
    model = sum((p.model_prob - o) ** 2 for p, o in rows) / n
    shown = sum((p.shown_prob - o) ** 2 for p, o in rows) / n
    market = sum(((p.fair_prob or 0.0) - o) ** 2 for p, o in rows) / n
    d = [(p.shown_prob - o) ** 2 - ((p.fair_prob or 0.0) - o) ** 2 for p, o in rows]
    mean = sum(d) / n
    se = (
        math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else None
    )
    return Comparison(n, round(model, 4), round(shown, 4), round(market, 4), mean, se)


def decision(graded: list[GradedPosition]) -> Decision | None:
    decided = [g for g in graded if g.result != PUSH]
    if not decided:
        return None
    tp = sum(1 for g in decided if g.position.is_buy and g.result == WIN)
    fp = sum(1 for g in decided if g.position.is_buy and g.result == LOSS)
    fn = sum(1 for g in decided if not g.position.is_buy and g.result == WIN)
    tn = sum(1 for g in decided if not g.position.is_buy and g.result == LOSS)
    return Decision(
        base_rate=(tp + fn) / len(decided),
        bets=tp + fp,
        ppv=tp / (tp + fp) if tp + fp else None,
        passes=tn + fn,
        npv=tn / (tn + fn) if tn + fn else None,
    )


def rank_quartiles(graded: list[GradedPosition]) -> list[Record]:
    ranked = [g for g in graded if g.position.rank is not None]
    ranks = sorted({g.position.rank for g in ranked if g.position.rank is not None})
    if not ranks:
        return []
    if len(ranks) < 4:
        return [
            summarize(f"rank {r}", [g for g in ranked if g.position.rank == r]) for r in ranks
        ]
    step = len(ranks) / 4
    out = []
    for q in range(4):
        lo = ranks[int(q * step)]
        hi = ranks[min(int((q + 1) * step) - 1, len(ranks) - 1)]
        out.append(
            summarize(
                f"rank {lo}-{hi}",
                [g for g in ranked if g.position.rank is not None and lo <= g.position.rank <= hi],
            )
        )
    return out


# --------------------------------------------------------------------------- outcomes


def bat_results(
    roster: list[RosterRow],
    results: dict[int, GameResult],
    plays: dict[int, list[PlateAppearance]],
    priced_ids: set[int],
) -> list[BatResult]:
    """Each bat on the roster against the starter his side actually faced."""
    out: list[BatResult] = []
    seen: set[tuple[str, int]] = set()
    for row in roster:
        if row.kind != BAT or (row.date, row.player_id) in seen:
            continue
        seen.add((row.date, row.player_id))
        res = results.get(row.game_pk) if row.game_pk is not None else None
        pas = plays.get(row.game_pk, []) if row.game_pk is not None else []
        mine = [pa for pa in pas if pa.batter == row.player_id]
        line = res.players.get(row.player_id) if res is not None else None
        sps = starters(pas)
        sp: int | None = None
        if mine:
            sp = sps.get("home" if mine[0].half == "top" else "away")
        elif row.versus_id in sps.values():
            sp = row.versus_id
        out.append(
            BatResult(
                row=row,
                played=bool(res and res.batted(row.player_id)),
                started=bool(line and line.started),
                slot=line.slot if line else None,
                starter_id=sp,
                probable_started=(
                    None if sp is None or row.versus_id is None else sp == row.versus_id
                ),
                vs_starter=tally([pa for pa in mine if sp is not None and pa.pitcher == sp]),
                game=tally(mine),
                priced=row.player_id in priced_ids,
            )
        )
    return out


def arm_results(
    roster: list[RosterRow],
    results: dict[int, GameResult],
    plays: dict[int, list[PlateAppearance]],
) -> list[ArmResult]:
    by_id: dict[int, list[RosterRow]] = defaultdict(list)
    for row in roster:
        if row.kind == ARM:
            by_id[row.player_id].append(row)
    out: list[ArmResult] = []
    for pid, rows in by_id.items():
        first = rows[0]
        calls = {
            r.arm_tier: (r.status if r.status != GATED else f"gated ({r.reason})") for r in rows
        }
        res = results.get(first.game_pk) if first.game_pk is not None else None
        pas = plays.get(first.game_pk, []) if first.game_pk is not None else []
        out.append(
            ArmResult(
                name=first.name,
                player_id=pid,
                team=first.team,
                calls=calls,
                siera=next((r.siera for r in rows if r.siera is not None), None),
                started=(pid in starters(pas).values()) if pas else None,
                line=res.pitcher(pid) if res is not None else {},
            )
        )
    out.sort(key=lambda a: (not any(c == SCREENED for c in a.calls.values()), a.name))
    return out


def dropped_positions(roster: list[RosterRow], positions: list[Position]) -> list[str]:
    dropped = {r.player_id for r in roster if r.kind == BAT and r.status == DROPPED}
    held = {r.player_id for r in roster if r.kind == BAT and r.status == HELD}
    return sorted(
        {p.batter for p in positions if p.player_id in dropped and p.player_id not in held}
    )


# --------------------------------------------------------------------------- commentary


def _rec(r: Record) -> str:
    w = f"{r.wins}-{r.losses}" + (f"-{r.pushes}" if r.pushes else "")
    roi = "" if r.roi is None else f", {r.roi:+.1%} ROI"
    se = "" if r.roi_se is None else f" (±{r.roi_se:.1%})"
    return f"{w}, {r.units:+.2f}u{roi}{se} on n={r.n}"


def _signal(r: Record) -> str:
    """Does a record say anything? The same bar the screen's words are held to."""
    if r.n < LABEL_EARN_ROWS or r.roi is None or r.roi_se is None or r.roi_se <= 0:
        return "underpowered"
    if r.roi >= Z * r.roi_se:
        return "positive beyond 2 SE"
    if r.roi <= -Z * r.roi_se:
        return "negative beyond 2 SE"
    return "within noise"


def _names(items: Iterable[str], limit: int = 8) -> str:
    xs = list(items)
    head = ", ".join(xs[:limit])
    return head + (f" and {len(xs) - limit} more" if len(xs) > limit else "")


def _vs_line(t: Tally) -> str:
    if not t.pa:
        return "no PA"
    return f"{t.h}-for-{t.ab}, {t.hr} HR, {t.bb} BB, {t.k} K ({t.pa} PA)"


def commentary(day: DayAudit, total: TotalAudit) -> Commentary:
    c = Commentary()
    held = [b for b in day.bats if b.row.status == HELD]

    # ---- problems in the receipt
    day_pairs = two_sided(g.position for g in day.graded)
    all_pairs = two_sided(g.position for g in total.graded)
    if all_pairs:
        names = sorted({f"{k[2]}" for k in day_pairs})
        who = {
            str(g.position.player_id): g.position.batter
            for g in day.graded
            if g.position.player_id is not None
        }
        today = (
            f" {len(day_pairs)} of them on {day.day} ({_names(who.get(n, n) for n in names)})."
            if day_pairs
            else ""
        )
        c.problems.append(
            f"{len(all_pairs)} props in the ledger are recorded on both sides in the same run."
            f"{today} A pair settles one win and one loss and pays the vig, so it puts a "
            "guaranteed loss into its bucket's record. The bucket tables below also show "
            "each record with the pairs removed."
        )
    if day.roster_recorded:
        dnp = [b.row.name for b in held if b.row.game_pk is not None and not b.played]
        bench = [b.row.name for b in held if b.played and not b.started]
        moved = [f"{b.row.name} ({b.row.slot}→{b.slot})" for b in held if b.slot_moved]
        if dnp or bench or moved:
            parts = []
            if dnp:
                parts.append(f"did not play: {_names(dnp)}")
            if bench:
                parts.append(f"came off the bench: {_names(bench)}")
            if moved:
                parts.append(f"batted in a different slot: {_names(moved)}")
            projected = sum(1 for b in held if b.row.projected)
            c.problems.append(
                f"The lineup the screen read was wrong for {len(dnp) + len(bench) + len(moved)} "
                f"of {len(held)} kept bats ({projected} were on a projected lineup). "
                + "; ".join(parts)
                + "."
            )
        unpriced = [b.row.name for b in held if not b.priced]
        if held and len(unpriced) / len(held) > UNPRICED_WARN:
            c.problems.append(
                f"{len(unpriced)} of {len(held)} kept bats had no recorded price "
                f"({_names(unpriced)}). Their games are graded below for the matchup, "
                "but they are not bets and carry no units."
            )
        wrong_sp = sorted(
            {b.row.versus for b in day.bats if b.probable_started is False and b.row.versus}
        )
        no_start = [a.name for a in day.arms if a.started is False]
        if wrong_sp or no_start:
            c.problems.append(
                "The screen rated an arm who did not start: "
                f"{_names(sorted(set(wrong_sp) | set(no_start)))}. Every bat screened "
                "against him was graded against a pitcher nobody rated."
            )
    else:
        c.problems.append(
            f"No roster was recorded for {day.day}, so only the priced positions can be "
            "graded. Bats the screen kept without a price, and the arms it gated, are "
            "missing from this audit, not counted as failures."
        )
    if day.dropped_positions:
        c.problems.append(
            f"The ledger holds positions on bats the screen dropped on production: "
            f"{_names(day.dropped_positions)}. Those rows are graded as recorded, but "
            "they are not the screen's call."
        )
    if day.voided:
        c.problems.append(
            f"{day.voided} of {day.recorded} recorded rows on {day.day} were voided (player "
            "did not appear or game not final). They are left out, not counted as losses."
        )
    if total.recorded and total.voided / total.recorded > VOID_WARN:
        c.problems.append(
            f"{total.voided} of {total.recorded} recorded rows ({total.voided / total.recorded:.0%}) "
            "are voided across the ledger, more than a scratch rate explains."
        )
    decided = [g for g in total.graded if g.result != PUSH]
    one_way = [g for g in decided if not (g.position.devigged and g.position.fair_prob)]
    if decided and len(one_way) / len(decided) > ONE_WAY_WARN:
        c.problems.append(
            f"{len(one_way)} of {len(decided)} decided rows had a one-way quote, so they have "
            "no no-vig price and are left out of the Brier comparison."
        )

    # ---- insights
    if day.graded:
        r = summarize("day", day.graded)
        buys = [g for g in day.graded if g.position.is_buy]
        c.insights.append(
            f"{day.day}: the recorded positions went {_rec(r)}. The card bought "
            + (f"{len(buys)} of them." if buys else "none of them; every row was a Pass.")
            + " One day is an anecdote, not a test."
        )
    for side, word in (("under", "under"), ("over", "over")):
        group_ = [b for b in held if b.row.side == side and b.vs_starter.pa]
        if group_:
            t = sum((b.vs_starter for b in group_), Tally())
            c.insights.append(
                f"Kept bats held on the {word} went {_vs_line(t)} against the starter."
            )
    for a in day.arms:
        if SCREENED in a.calls.values() and a.line:
            ln = a.line
            ip = f"{ln.get('outs', 0) // 3}.{ln.get('outs', 0) % 3}"
            c.insights.append(
                f"{a.name} ({', '.join(f'{k}: {v}' for k, v in a.calls.items())}) threw "
                f"{ip} IP, {ln.get('K', 0)} K, {ln.get('BB', 0)} BB, "
                f"{ln.get('H', 0)} H, {ln.get('ER', 0)} ER."
            )
    for label, cmp_ in (("on the day", compare(day.graded)), ("on the ledger", compare(total.graded))):
        if cmp_ is None:
            continue
        se = "" if cmp_.diff_se is None else f" ± {cmp_.diff_se:.4f}"
        c.insights.append(
            f"Brier {label} (n={cmp_.n} two-sided rows): printed {cmp_.shown:.4f}, model "
            f"{cmp_.model:.4f}, no-vig price {cmp_.market:.4f}; paired difference "
            f"{cmp_.diff:+.4f}{se}: {cmp_.verdict}."
        )
    tot = summarize("all", total.graded)
    if total.graded:
        c.insights.append(
            f"Whole ledger, {len(total.days)} days ({total.days[0]} to {total.days[-1]}): "
            f"{_rec(tot)}. Rows are not independent bets: one hitter carries several "
            "markets, and both sides of a prop can be recorded."
        )
    records = bucket_records(total.graded)
    clean = bucket_records(without_pairs(total.graded))
    for key, rec in sorted(records.items(), key=lambda kv: -kv[1].n):
        name = RATING_DISPLAY.get(key, key)
        extra = ""
        if key in clean and clean[key].n != rec.n:
            extra = f"; without both-sided pairs {_rec(clean[key])}"
        c.insights.append(
            f"Bucket '{name}': {_rec(rec)}{extra}. {_signal(rec).capitalize()}; "
            f"word earned: {earned_label(rec)}."
        )
    for rec in group(total.graded, "arm tier"):
        c.insights.append(f"Against {rec.label}: {_rec(rec)}. {_signal(rec).capitalize()}.")
    quart = rank_quartiles(total.graded)
    if len(quart) >= 2 and quart[0].roi is not None and quart[-1].roi is not None:
        c.insights.append(
            f"Composite order: top quartile ({quart[0].label}) {_rec(quart[0])}; bottom "
            f"({quart[-1].label}) {_rec(quart[-1])}."
        )
    dec = decision(total.graded)
    if dec is not None:
        if dec.ppv is None:
            c.insights.append(
                f"The card bought nothing gradeable on the ledger, so the buy decision has no "
                f"PPV; the NPV on {dec.passes} passes is free when the screen passes on "
                "everything and is not evidence of skill."
            )
        else:
            c.insights.append(
                f"Buy decision: PPV {dec.ppv:.1%} on {dec.bets} bets against a "
                f"{dec.base_rate:.1%} base rate."
            )

    # ---- recommendations
    if all_pairs:
        c.recommendations.append(
            "Record one side per prop for the buckets that hold no side (RV<0 watch, "
            "production watch), or record them as display-only. Until then, read those "
            "buckets with the pairs removed."
        )
    if any("lineup the screen read" in p for p in c.problems):
        c.recommendations.append(
            "Grade the run captured after lineups post (pass --run to this audit, or make "
            "the post-lineup capture the last run of the day); a projected lineup is "
            "grading players who did not play."
        )
    if any("did not start" in p for p in c.problems):
        c.recommendations.append(
            "Gate out any arm who is not the confirmed probable, and treat a missing "
            "strikeout market on a probable as a warning that he may not start."
        )
    if any("no recorded price" in p for p in c.problems):
        c.recommendations.append(
            "Re-run the screen once the props post. A survivor with no price is a "
            "matchup opinion that cannot be scored against the market."
        )
    cmp_all = compare(total.graded)
    if cmp_all is None or cmp_all.verdict in ("underpowered", "no spread to judge"):
        n = 0 if cmp_all is None else cmp_all.n
        c.recommendations.append(
            f"Make no pricing change from this ledger: {n} two-sided rows is below the "
            f"{MIN_COMPARE} needed to compare against the price."
        )
    elif cmp_all.verdict == "the price beat the screen's number":
        c.recommendations.append(
            "Keep every screen row at Pass and shrink the screen's probability toward the "
            "no-vig price before it can tier a buy. The price is beating it on "
            f"n={cmp_all.n}."
        )
    elif cmp_all.verdict == "the screen's number beat the price":
        c.recommendations.append(
            f"The screen's number beats the price on n={cmp_all.n}. Test that on a "
            "walk-forward holdout before it changes a tier; do not change one on this audit."
        )
    else:
        c.recommendations.append(
            f"On n={cmp_all.n} the screen's number is indistinguishable from the no-vig "
            "price. Leave tiers as they are; there is no independent information to bet yet."
        )
    earned = [
        RATING_DISPLAY.get(k, k)
        for k, r in records.items()
        if _signal(r) in ("positive beyond 2 SE", "negative beyond 2 SE")
    ]
    if earned:
        c.recommendations.append(
            f"Buckets with a record beyond 2 SE on {LABEL_EARN_ROWS}+ rows: "
            f"{_names(earned)}. Confirm on a holdout before changing a word."
        )
    else:
        c.recommendations.append(
            f"No bucket has {LABEL_EARN_ROWS}+ rows and an ROI more than 2 SE from zero, so "
            "no word changes. Keep logging."
        )
    c.recommendations.append(
        "This audit changes nothing in production: no price, tier, gate or rating."
    )
    return c


# --------------------------------------------------------------------------- render

_CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#1d1d1f;max-width:900px;margin:0 auto;
padding:8mm;font-size:10.5pt;line-height:1.4}
h1{font-size:19pt;margin:0 0 1mm} h2{font-size:14pt;border-bottom:1px solid #999;
margin-top:7mm} h3{font-size:11.5pt;margin:4mm 0 1mm}
.sub{color:#666;font-size:9pt}
table{border-collapse:collapse;width:100%;margin:2mm 0 4mm;font-family:Helvetica,Arial,
sans-serif;font-size:8.5pt}
th,td{border-bottom:1px solid #ddd;padding:2px 4px;text-align:left}
th{background:#f2f2f2} td.n{text-align:right;font-variant-numeric:tabular-nums}
.win{color:#1a7f37} .loss{color:#b42318}
ul{margin:1mm 0 3mm 5mm;padding-left:4mm} li{margin:1mm 0}
"""


def _e(x: object) -> str:
    return html.escape(str(x))


def _table(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    if not rows:
        return "<p class='sub'>(none)</p>"
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td class='n'>{c}</td>" if i >= numeric_from else f"<td>{c}</td>"
            for i, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    return f"<table><tr>{th}</tr>{body}</table>"


def _rec_cells(r: Record) -> list[str]:
    record = f"{r.wins}-{r.losses}" + (f"-{r.pushes}" if r.pushes else "")
    return [
        _e(r.label),
        str(r.n),
        record,
        "-" if r.win_pct is None else f"{r.win_pct:.1%}",
        f"{r.units:+.2f}",
        "-" if r.roi is None else f"{r.roi:+.1%}",
        "-" if r.roi_se is None else f"±{r.roi_se:.1%}",
    ]


_REC_HEAD = ["", "n", "W-L", "win%", "units", "ROI", "1 SE"]


def _records_table(records: list[Record]) -> str:
    return _table(_REC_HEAD, [_rec_cells(r) for r in records])


def _avg(x: float | None) -> str:
    if x is None:
        return "-"
    return f".{round(x * 1000):03d}" if x < 1 else f"{x:.3f}"


def _tally_cells(t: Tally) -> list[str]:
    return [str(t.pa), f"{t.h}-{t.ab}", str(t.tb), str(t.hr), str(t.bb), str(t.k),
            _avg(t.avg), _avg(t.slg)]


_TALLY_HEAD = ["PA", "H-AB", "TB", "HR", "BB", "K", "AVG", "SLG"]


def _bucket_of_bat(row: RosterRow) -> str:
    return RATING_DISPLAY.get(row.bucket, row.bucket or "-")


def _bat_groups(bats: list[BatResult]) -> list[list[str]]:
    groups: dict[tuple[str, str, str], list[BatResult]] = defaultdict(list)
    for b in bats:
        groups[(b.row.arm_tier, _bucket_of_bat(b.row), b.row.status)].append(b)
    rows = []
    for (tier, bk, status), bs in sorted(groups.items()):
        vs = sum((b.vs_starter for b in bs), Tally())
        game = sum((b.game for b in bs), Tally())
        rows.append(
            [_e(f"{tier} / {bk} / {status}"), str(len(bs))]
            + _tally_cells(vs)[:2] + [_avg(vs.slg), str(vs.k)]
            + _tally_cells(game)[:2] + [_avg(game.slg), str(game.k)]
        )
    return rows


_GROUP_HEAD = ["arm tier / bucket / status", "bats", "PA vs SP", "H-AB vs SP", "SLG vs SP",
               "K vs SP", "PA game", "H-AB game", "SLG game", "K game"]


def _positions_rows(graded: list[GradedPosition]) -> list[list[str]]:
    out = []
    for g in sorted(graded, key=lambda x: (x.position.batter, x.position.stat, x.position.side)):
        p = g.position
        css = "win" if g.result == WIN else "loss" if g.result == LOSS else ""
        out.append(
            [
                _e(p.batter),
                _e(p.label),
                _e(display_bucket(p)),
                "-" if p.odds is None else f"{p.odds:+.0f}",
                f"{p.shown_prob:.3f}",
                "-" if p.fair_prob is None or not p.devigged else f"{p.fair_prob:.3f}",
                _e(p.tier),
                str(g.actual),
                f"<span class='{css}'>{_e(g.result)}</span>",
                f"{g.units:+.2f}",
            ]
        )
    return out


def _compare_block(cmp_: Comparison | None) -> str:
    if cmp_ is None:
        return "<p class='sub'>No decided row had a two-sided price to strip the vig from.</p>"
    se = "-" if cmp_.diff_se is None else f"{cmp_.diff_se:.4f}"
    return _table(
        ["two-sided rows", "model", "printed", "no-vig price", "printed − price", "1 SE",
         "verdict"],
        [[str(cmp_.n), f"{cmp_.model:.4f}", f"{cmp_.shown:.4f}", f"{cmp_.market:.4f}",
          f"{cmp_.diff:+.4f}", se, _e(cmp_.verdict)]],
        numeric_from=0,
    )


def _day_section(day: DayAudit) -> str:
    parts = [f"<h2>1. The day: {day.day}</h2>"]
    run = f"run {day.run_id}" if day.run_id else "no run id"
    parts.append(
        f"<p class='sub'>Graded {run}: {day.recorded} recorded rows, {len(day.graded)} "
        f"graded, {day.voided} voided.</p>"
    )
    if day.roster_recorded:
        parts.append("<h3>The arms</h3>")
        parts.append(
            _table(
                ["arm", "team", "SIERA", "what the screen did", "started", "IP", "K", "BB",
                 "H", "ER"],
                [
                    [
                        _e(a.name), _e(a.team),
                        "-" if a.siera is None else f"{a.siera:.2f}",
                        _e("; ".join(f"{k}: {v}" for k, v in sorted(a.calls.items()))),
                        "-" if a.started is None else ("yes" if a.started else "NO"),
                        f"{a.line.get('outs', 0) // 3}.{a.line.get('outs', 0) % 3}"
                        if a.line else "-",
                        str(a.line.get("K", "-")), str(a.line.get("BB", "-")),
                        str(a.line.get("H", "-")), str(a.line.get("ER", "-")),
                    ]
                    for a in day.arms
                ],
                numeric_from=5,
            )
        )
        parts.append("<h3>The bats, by group</h3>")
        parts.append(_table(_GROUP_HEAD, _bat_groups(day.bats)))
        parts.append("<h3>Every bat</h3>")
        parts.append(
            _table(
                ["bat", "vs", "bucket", "side", "status", "priced", "slot (screen→game)",
                 "started"] + [f"{h} vs SP" for h in ("PA", "H-AB", "HR", "K")]
                + ["H-AB game", "TB game"],
                [
                    [
                        _e(b.row.name), _e(b.row.versus), _e(_bucket_of_bat(b.row)),
                        _e(b.row.side or "-"), _e(b.row.status),
                        "yes" if b.priced else "no",
                        f"{b.row.slot or '-'}→{b.slot or '-'}",
                        "DNP" if not b.played else ("yes" if b.started else "bench"),
                        str(b.vs_starter.pa), f"{b.vs_starter.h}-{b.vs_starter.ab}",
                        str(b.vs_starter.hr), str(b.vs_starter.k),
                        f"{b.game.h}-{b.game.ab}", str(b.game.tb),
                    ]
                    for b in sorted(day.bats, key=lambda x: (x.row.versus, x.row.status, x.row.name))
                ],
                numeric_from=8,
            )
        )
        parts.append(
            "<p class='sub'>\"vs SP\" is every plate appearance against the starter his side "
            "actually faced, off the play-by-play. A bat who did not play is shown as DNP and "
            "is not a failure of the screen.</p>"
        )
    else:
        parts.append(
            "<p class='sub'>No roster was recorded for this day (the screen started writing "
            "one with this audit), so only the priced positions below can be graded.</p>"
        )
    parts.append("<h3>The positions, against their recorded prices</h3>")
    if day.graded:
        parts.append(_records_table([summarize("all positions", day.graded)]
                                    + group(day.graded, "half")
                                    + group(day.graded, "bucket")))
        parts.append(
            _table(
                ["player", "prop", "bucket", "odds", "printed p", "no-vig p", "tier", "actual",
                 "result", "units"],
                _positions_rows(day.graded),
                numeric_from=3,
            )
        )
        parts.append(_compare_block(compare(day.graded)))
    else:
        parts.append("<p class='sub'>No recorded position on this day could be graded.</p>")
    return "".join(parts)


def _total_section(total: TotalAudit) -> str:
    parts = ["<h2>2. The whole record</h2>"]
    if not total.graded:
        return "".join(parts) + "<p class='sub'>Nothing in the ledger is gradeable yet.</p>"
    parts.append(
        f"<p class='sub'>{len(total.days)} days, {total.days[0]} to {total.days[-1]}, last run "
        f"of each day: {total.recorded} recorded rows, {len(total.graded)} graded, "
        f"{total.voided} voided. Units are at the recorded price, one unit a row, pushes at "
        "zero.</p>"
    )
    g = total.graded
    parts.append(
        _records_table(
            [
                summarize("all positions", g),
                summarize("both-sided pairs removed", without_pairs(g)),
                summarize("the card bet", [x for x in g if x.position.is_buy]),
                summarize("the card passed", [x for x in g if not x.position.is_buy]),
            ]
        )
    )
    for title, key in (
        ("By half", "half"),
        ("By arm tier", "arm tier"),
        ("By market", "market"),
        ("By tier", "tier"),
        ("By gate (what refused the row)", "gate"),
    ):
        parts.append(f"<h3>{title}</h3>")
        parts.append(_records_table(group(g, key)))
    parts.append("<h3>By bucket, and the word each record has earned</h3>")
    recs = bucket_records(g)
    clean = bucket_records(without_pairs(g))
    parts.append(
        _table(
            _REC_HEAD + ["pairs removed", "signal", "word"],
            [
                [_e(RATING_DISPLAY.get(k, k))]
                + _rec_cells(r)[1:]
                + [
                    _rec(clean[k]) if k in clean else "-",
                    _e(_signal(r)),
                    _e(earned_label(r)),
                ]
                for k, r in sorted(recs.items(), key=lambda kv: -kv[1].n)
            ],
        )
    )
    parts.append("<h3>Was the number better than the price?</h3>")
    parts.append(_compare_block(compare(g)))
    parts.append("<h3>Did the buy decision discriminate?</h3>")
    dec = decision(g)
    if dec is not None:
        parts.append(
            _table(
                ["base rate", "bets", "PPV", "passes", "NPV"],
                [[f"{dec.base_rate:.1%}", str(dec.bets),
                  "-" if dec.ppv is None else f"{dec.ppv:.1%}", str(dec.passes),
                  "-" if dec.npv is None else f"{dec.npv:.1%}"]],
                numeric_from=0,
            )
        )
    parts.append("<h3>Did the composite's order discriminate?</h3>")
    quart = rank_quartiles(g)
    parts.append(
        _records_table(quart) if quart
        else "<p class='sub'>No row carries a rank.</p>"
    )
    if total.bats:
        parts.append(
            f"<h3>Every kept bat against the starter ({len(total.roster_days)} days with a "
            "roster)</h3>"
        )
        parts.append(_table(_GROUP_HEAD, _bat_groups(total.bats)))
    parts.append("<h3>Day by day</h3>")
    parts.append(_records_table(total.by_day))
    return "".join(parts)


def _list(items: list[str]) -> str:
    if not items:
        return "<p class='sub'>(none)</p>"
    return "<ul>" + "".join(f"<li>{_e(i)}</li>" for i in items) + "</ul>"


def render_html(day: DayAudit, total: TotalAudit, notes: Commentary) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Power screen audit {day.day}</title><style>{_CSS}</style></head><body>"
        f"<h1>Power screen audit: {day.day}</h1>"
        "<p class='sub'>Yesterday's screen graded against the box scores and the prices it "
        "recorded, then the whole ledger, then what it means. Missing players and unpriced "
        "rows are reported as missing, not as losses.</p>"
        + _day_section(day)
        + _total_section(total)
        + "<h2>3. Commentary</h2>"
        + "<h3>Problems</h3>" + _list(notes.problems)
        + "<h3>Insights</h3>" + _list(notes.insights)
        + "<h3>Recommendations</h3>" + _list(notes.recommendations)
        + "</body></html>"
    )


def render_text(day: DayAudit, total: TotalAudit, notes: Commentary) -> str:
    """The plain-text email body: the verdicts, not the tables."""
    lines = [f"Power screen audit {day.day}", ""]
    if day.graded:
        lines.append(f"Day: {_rec(summarize('day', day.graded))}")
    if total.graded:
        lines.append(f"Ledger ({len(total.days)} days): {_rec(summarize('all', total.graded))}")
    for title, items in (
        ("Problems", notes.problems),
        ("Insights", notes.insights),
        ("Recommendations", notes.recommendations),
    ):
        lines += ["", title + ":"] + [f"- {i}" for i in items]
    return "\n".join(lines)


# --------------------------------------------------------------------------- assembly

ResultFetcher = Callable[[int], GameResult | None]
PlaysFetcher = Callable[[int, bool], list[PlateAppearance]]


def build(
    day: str,
    positions: list[Position],
    roster: list[RosterRow],
    fetch_result: ResultFetcher,
    fetch_plays: PlaysFetcher,
    run_id: str | None = None,
) -> tuple[DayAudit, TotalAudit]:
    """Grade ``day`` (its last run, or ``run_id``) and every ledger day up to it.

    ``positions`` is the whole ledger, display-only rows already removed;
    ``roster`` the whole roster file. Fetchers are injected so a test (or a
    scratch rerun) never touches the network.
    """
    upto = [p for p in positions if p.date <= day]
    total_rows = _last_runs_except(upto, day, run_id)
    want = run_id
    if want is None:
        runs = {p.run_id for p in upto if p.date == day}
        want = max(runs) if runs else None
    day_rows = [p for p in total_rows if p.date == day]
    day_roster = [r for r in roster if r.date == day]
    if day_roster:
        r_runs = {r.run_id for r in day_roster}
        pick = want if want in r_runs else max(r_runs)
        day_roster = [r for r in day_roster if r.run_id == pick]

    games = {p.game_pk for p in total_rows if p.game_pk is not None}
    roster_by_day = _last_roster_per_day(roster, day)
    games |= {r.game_pk for r in roster_by_day if r.game_pk is not None}
    results: dict[int, GameResult] = {}
    for pk in sorted(games):
        res = fetch_result(pk)
        if res is not None:
            results[pk] = res
    plays: dict[int, list[PlateAppearance]] = {}
    for pk in sorted({r.game_pk for r in roster_by_day if r.game_pk is not None}):
        res = results.get(pk)
        if res is not None:
            plays[pk] = fetch_plays(pk, res.final)

    graded_all: list[GradedPosition] = []
    voided_all = 0
    by_day: list[Record] = []
    days = sorted({p.date for p in total_rows})
    for d in days:
        rows = [p for p in total_rows if p.date == d]
        g, v = grade_positions(rows, results)
        graded_all.extend(g)
        voided_all += v
        by_day.append(summarize(d, g))
    day_graded = [g for g in graded_all if g.position.date == day]
    day_voided = len(day_rows) - len(day_graded)

    def priced(d: str) -> set[int]:
        return {p.player_id for p in total_rows if p.date == d and p.player_id is not None}

    total_bats: list[BatResult] = []
    roster_days = sorted({r.date for r in roster_by_day})
    for d in roster_days:
        total_bats.extend(
            bat_results([r for r in roster_by_day if r.date == d], results, plays, priced(d))
        )
    day_audit = DayAudit(
        day=day,
        run_id=want or "",
        recorded=len(day_rows),
        graded=day_graded,
        voided=day_voided,
        roster_recorded=bool(day_roster),
        bats=bat_results(day_roster, results, plays, priced(day)),
        arms=arm_results(day_roster, results, plays),
        dropped_positions=dropped_positions(day_roster, day_rows),
    )
    total = TotalAudit(
        days=days,
        recorded=len(total_rows),
        graded=graded_all,
        voided=voided_all,
        by_day=by_day,
        roster_days=roster_days,
        bats=[b for b in total_bats if b.row.status == HELD],
    )
    return day_audit, total


def _last_runs_except(
    positions: list[Position], day: str, run_id: str | None
) -> list[Position]:
    """Last run per day, except ``day`` itself when a run was pinned for it."""
    keep: dict[str, str] = {}
    for p in positions:
        keep[p.date] = max(keep.get(p.date, ""), p.run_id)
    if run_id is not None:
        keep[day] = run_id
    return [p for p in positions if p.run_id == keep[p.date]]


def _last_roster_per_day(roster: list[RosterRow], upto: str) -> list[RosterRow]:
    keep: dict[str, str] = {}
    for r in roster:
        if r.date <= upto:
            keep[r.date] = max(keep.get(r.date, ""), r.run_id)
    return [r for r in roster if r.date in keep and r.run_id == keep[r.date]]
