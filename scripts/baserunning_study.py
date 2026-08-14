"""Measure how runners actually advance, against what the simulator assumes.

The Monte Carlo advances runners deterministically and maximally: every single
sends the runner from first to third and scores anyone from second, every double
scores the runner from first, and no out ever drives a run in. Real baseball does
none of those things reliably. Since ``batter_r``, ``batter_rbi`` and
``batter_hrr`` are priced off exactly this conversion of hits into runs, the
assumption is not a detail -- it decides which lineup slots look profitable.

Run it:

    python -m scripts.baserunning_study [--start 2026-06-01] [--end 2026-07-27]

It reads the MLB play-by-play feed (cached under ``~/.mlb_engine/cache/pbp``) and
reports, for each situation the simulator hard-codes, the rate real runners
managed -- plus where RBI actually come from, which is the other half of the
story: the sim can only produce an RBI on a hit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import requests

from mlb_engine.data import http
from mlb_engine.data.collapse import BASE, _fetch_pbp

CACHE = Path.home() / ".mlb_engine" / "cache"

# What montecarlo._apply_pa does, as rates, for comparison.
SIM_ASSUMES = {
    "1st->3rd on a single": 1.0,
    "scores from 2nd on a single": 1.0,
    "scores from 1st on a double": 1.0,
    "scores from 3rd on an out": 0.0,
}


@dataclass
class Tally:
    """One situation, split by the out count the batter came to the plate with.

    Pooling across outs hides the dominant conditioner. A man on second scores on
    a single 40% of the time with nobody out and 79% with two: with two outs he
    leaves on contact, with none he cannot afford to be erased. One pooled rate is
    wrong in both directions at once, so the model indexes these by outs and so
    does this study.
    """

    did: list[int] = field(default_factory=lambda: [0, 0, 0])
    chances: list[int] = field(default_factory=lambda: [0, 0, 0])

    def add(self, outs: int, happened: bool) -> None:
        self.chances[outs] += 1
        self.did[outs] += int(happened)

    def rate(self, outs: int) -> float:
        n = self.chances[outs]
        return self.did[outs] / n if n else 0.0

    @property
    def pooled(self) -> float:
        n = sum(self.chances)
        return sum(self.did) / n if n else 0.0


@dataclass
class Study:
    first_to_third: Tally = field(default_factory=Tally)  # single, runner on 1st
    score_from_second: Tally = field(default_factory=Tally)  # single, runner on 2nd
    score_from_first: Tally = field(default_factory=Tally)  # double, runner on 1st
    # Ball in play for an out with a runner on 3rd and fewer than 2 outs: the sac
    # fly and the run-scoring groundout, neither of which the sim can produce.
    score_from_third_on_out: Tally = field(default_factory=Tally)
    # Two outs on one ball in play: the force at second, available only with a man
    # on first. It was in the simulator and not in the Markov chain.
    double_play: Tally = field(default_factory=Tally)
    # The productive out: the runner takes the next base while the batter is retired.
    second_to_third_on_out: Tally = field(default_factory=Tally)
    first_to_second_on_out: Tally = field(default_factory=Tally)
    rbi_by_event: Counter[str] = field(default_factory=Counter)
    runs: int = 0
    rbi: int = 0
    plays: int = 0
    games: int = 0
    # PA outcome counts, so the simulator can be fed the same league it is being
    # validated against: a run-scoring model can only be judged on inputs that
    # reproduce the league's hits and walks.
    outcomes: Counter[str] = field(default_factory=Counter)

    def league_rates(self) -> dict[str, float]:
        total = sum(self.outcomes.values())
        return {k: self.outcomes[k] / total for k in ("1B", "2B", "3B", "HR", "BB", "K", "OUT")}


def schedule_game_pks(start: Date, end: Date, session: requests.Session) -> list[int]:
    url = (
        f"{BASE}/schedule?sportId=1&startDate={start.isoformat()}"
        f"&endDate={end.isoformat()}&gameType=R"
    )
    data = session.get(url, timeout=30).json()
    pks: list[int] = []
    for day in data.get("dates", []) or []:
        for g in day.get("games", []) or []:
            if str(g.get("status", {}).get("abstractGameState")) == "Final":
                pks.append(int(g["gamePk"]))
    return pks


def cached_game_pks() -> list[int]:
    return sorted(int(p.stem) for p in (CACHE / "pbp").glob("*.json") if p.stem.isdigit())


def _origin(runner: dict) -> str | None:
    mv = runner.get("movement", {}) or {}
    return mv.get("originBase") or mv.get("start")


def _end(runner: dict) -> str | None:
    return (runner.get("movement", {}) or {}).get("end")


def _scored(runner: dict) -> bool:
    return _end(runner) == "score"


@dataclass
class Move:
    """One runner's whole journey on one play."""

    origin: str | None
    end: str | None

    @property
    def scored(self) -> bool:
        return self.end == "score"


def _moves(runners: list[dict]) -> dict[int, Move]:
    """Collapse the feed's per-leg runner entries into one move per runner.

    The feed splits a single advance into legs -- a runner going first to third
    appears as 1B->2B and then 2B->3B -- so reading only the first entry scores
    him as having stopped at second. Origin is the first leg's origin and the
    destination is the last leg's end.
    """
    out: dict[int, Move] = {}
    for r in runners:
        rid = ((r.get("details", {}) or {}).get("runner", {}) or {}).get("id")
        if rid is None:
            continue
        rid = int(rid)
        if rid in out:
            out[rid].end = _end(r) or out[rid].end
        else:
            out[rid] = Move(origin=_origin(r), end=_end(r))
    return out


PA_BUCKET = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "BB",
    "strikeout": "K",
    "strikeout_double_play": "K",
}


def _pa_bucket(event: str) -> str | None:
    """The simulator's seven-outcome bucket for a feed event, or None if it is
    not a plate appearance at all (a steal, a pickoff, a balk)."""
    if event in PA_BUCKET:
        return PA_BUCKET[event]
    if event in BALL_IN_PLAY_OUTS or event in ("field_error", "fielders_choice"):
        return "OUT"
    return None


def _post_state(play: dict) -> tuple[bool, bool, bool]:
    """Occupancy of (1B, 2B, 3B) *after* the play, from the feed's matchup."""
    m = play.get("matchup", {}) or {}
    return (
        m.get("postOnFirst") is not None,
        m.get("postOnSecond") is not None,
        m.get("postOnThird") is not None,
    )


def read_game(pbp: dict, st: Study) -> None:
    st.games += 1
    # A runner who holds his base generates no movement entry, so occupancy has to
    # come from the previous play's post-state or every "did he advance?" rate is
    # measured only over the runners who did move.
    pre: tuple[bool, bool, bool] = (False, False, False)
    # ``count.outs`` is the out total *after* the play, so the situation a batter
    # walked into is the previous play's figure.
    outs_after = 0
    half_key: tuple[int, str] | None = None
    for play in pbp.get("allPlays", []) or []:
        about = play.get("about", {}) or {}
        key = (int(about.get("inning", 0) or 0), str(about.get("halfInning", "")))
        if key != half_key:
            half_key, pre, outs_after = key, (False, False, False), 0
        on_first, on_second, on_third = pre
        pre = _post_state(play)
        outs_at_bat = outs_after
        outs_after = int((play.get("count", {}) or {}).get("outs", outs_after) or 0)
        result = play.get("result", {}) or {}
        event = str(result.get("eventType", ""))
        runners = play.get("runners", []) or []
        rbi = int(result.get("rbi", 0) or 0)
        moves = _moves(runners)
        st.plays += 1
        bucket = _pa_bucket(event)
        if bucket is not None:
            st.outcomes[bucket] += 1
        st.rbi += rbi
        st.runs += sum(1 for m in moves.values() if m.scored)
        if rbi:
            st.rbi_by_event[event] += rbi

        from_first = next((m for m in moves.values() if m.origin == "1B"), None)
        from_second = next((m for m in moves.values() if m.origin == "2B"), None)
        from_third = next((m for m in moves.values() if m.origin == "3B"), None)

        if outs_at_bat > 2:
            continue

        if event == "single":
            if on_first:
                took_third = from_first is not None and from_first.end in ("3B", "score")
                st.first_to_third.add(outs_at_bat, took_third)
            if on_second:
                st.score_from_second.add(
                    outs_at_bat, from_second is not None and from_second.scored
                )
        elif event == "double" and on_first:
            st.score_from_first.add(outs_at_bat, from_first is not None and from_first.scored)

        if event in BALL_IN_PLAY_OUTS and outs_at_bat < 2:
            if on_third:
                st.score_from_third_on_out.add(
                    outs_at_bat, from_third is not None and from_third.scored
                )
            if on_second:
                st.second_to_third_on_out.add(
                    outs_at_bat, from_second is not None and from_second.end == "3B"
                )
            if on_first:
                # The feed's out total is the figure after the play, so two outs
                # recorded on one ball in play is the double play by definition --
                # more robust than trusting the event name, which misses the
                # fielder's-choice-plus-tag and the lineout doubling off.
                turned_two = outs_after - outs_at_bat >= 2
                st.double_play.add(outs_at_bat, turned_two)
                if not turned_two:
                    st.first_to_second_on_out.add(
                        outs_at_bat, from_first is not None and from_first.end == "2B"
                    )


# Outs on a ball in play: the chance to trade an out for a run. Strikeouts are
# excluded because they offer no such chance.
BALL_IN_PLAY_OUTS = {
    "field_out",
    "sac_fly",
    "sac_bunt",
    "grounded_into_double_play",
    "double_play",
    "force_out",
    "fielders_choice_out",
    "fielders_choice",
    "sac_fly_double_play",
}

OUT_EVENTS = {
    "field_out",
    "sac_fly",
    "sac_bunt",
    "grounded_into_double_play",
    "double_play",
    "force_out",
    "fielders_choice_out",
    "sac_fly_double_play",
    "strikeout",
    "strikeout_double_play",
}


def report(st: Study) -> str:
    L: list[str] = []
    L.append(
        f"{st.games} games, {st.plays} plays, {st.runs} runs "
        f"({st.runs / st.games:.2f} per game), {st.rbi} RBI\n"
    )
    L.append("Runner advancement, by the out count the batter came up with")
    L.append(f"{'situation':<32}{'0 outs':>16}{'1 out':>16}{'2 outs':>16}{'pooled':>9}")
    for label, tally in (
        ("1st->3rd on a single", st.first_to_third),
        ("scores from 2nd on a single", st.score_from_second),
        ("scores from 1st on a double", st.score_from_first),
        ("scores from 3rd on an out", st.score_from_third_on_out),
        ("2nd->3rd on an out", st.second_to_third_on_out),
        ("1st->2nd on an out (no DP)", st.first_to_second_on_out),
        ("two outs on one ball in play", st.double_play),
    ):
        cells = ""
        for outs in (0, 1, 2):
            n = tally.chances[outs]
            cells += f"{f'{tally.rate(outs):.3f} (n={n:,})' if n >= 30 else (f'n={n}'):>16}"
        L.append(f"{label:<32}{cells}{tally.pooled:9.3f}")
    L.append("")
    L.append("The rates the sim used to hold at certainty, for comparison:")
    for label, assumed in SIM_ASSUMES.items():
        L.append(f"  {label:<32}{assumed:.2f}")

    out_rbi = sum(n for ev, n in st.rbi_by_event.items() if ev in OUT_EVENTS)
    L.append("")
    L.append("Where RBI come from")
    for ev, n in st.rbi_by_event.most_common(10):
        flag = "  <- impossible in the sim" if ev in OUT_EVENTS else ""
        L.append(f"  {ev:<28}{n:6d}  {n / st.rbi * 100:5.1f}%{flag}")
    L.append("")
    L.append(
        f"RBI on outs: {out_rbi} ({out_rbi / st.rbi * 100:.1f}% of all RBI) -- the "
        f"simulator scores none of these, since only a hit can drive a run in."
    )
    L.append(f"RBI per run: {st.rbi / st.runs:.3f} (the sim produces exactly 1.000)")
    L.append("")
    L.append("League PA rates over the same games (feed these to the sim to validate it)")
    L.append("  " + "  ".join(f"{k} {v:.4f}" for k, v in st.league_rates().items()))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="ISO date; omit to use cached games")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap games (0 = no cap)")
    args = ap.parse_args()

    session = http.session()
    if args.start and args.end:
        pks = schedule_game_pks(Date.fromisoformat(args.start), Date.fromisoformat(args.end), session)
    else:
        pks = cached_game_pks()
    if args.limit:
        pks = pks[: args.limit]

    st = Study()
    read = 0
    for pk in pks:
        path = CACHE / "pbp" / f"{pk}.json"
        if path.exists():
            try:
                pbp = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
        else:
            pbp = _fetch_pbp(pk, session, CACHE, 30)
            if pbp is None:
                continue
        read += 1
        read_game(pbp, st)
    print(f"{read} games\n")
    print(report(st))


if __name__ == "__main__":
    main()
