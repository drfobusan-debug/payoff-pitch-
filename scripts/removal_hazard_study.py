"""Empirical batter-removal hazard from play-by-play.

Who actually left the game, and when. The Monte Carlo bats nine fixed slots to
the end of the game, so a pinch hitter's plate appearances are credited to the
starter's profile. This measures the thing the simulator would need: the chance
that the original occupant of a lineup slot is already gone when that slot comes
up again, by inning, slot, whether the opposing starter has exited, and the
platoon state of the matchup.

Batting order is preserved through substitutions, so the k-th plate appearance a
team takes belongs to slot k % 9 whatever the book says: the slot's original
occupant is whoever batted there the first time through. Which plays count as
plate appearances is the whole ballgame here -- see ``data.pbp``, since the feed
types a caught stealing as an at-bat and counting one shifts every slot after it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from mlb_engine.data.pbp import is_plate_appearance

PBP = Path.home() / ".mlb_engine" / "cache" / "pbp"


def slot_appearances() -> list[dict]:
    """One row per slot appearance: who batted, and the state it happened in."""
    rows: list[dict] = []
    for path in sorted(PBP.glob("*.json")):
        try:
            plays = json.loads(path.read_text()).get("allPlays") or []
        except (json.JSONDecodeError, OSError):
            continue
        if not plays:
            continue
        # A game the feed only half-carries is not evidence about removals.
        if max(int(p["about"]["inning"]) for p in plays) < 8:
            continue

        n_pa: dict[str, int] = defaultdict(int)  # plate appearances per team
        slot_of: dict[tuple[str, int], int] = {}  # (team, slot) -> original batter
        hand_of: dict[tuple[str, int], str] = {}  # (team, slot) -> his batting hand
        starter: dict[str, int] = {}  # pitching team -> starting pitcher
        game_rows: list[dict] = []

        for play in plays:
            if not is_plate_appearance(play):
                continue
            about, mu = play["about"], play["matchup"]
            team = "away" if about["isTopInning"] else "home"
            pit_team = "home" if about["isTopInning"] else "away"
            pitcher = int(mu["pitcher"]["id"])
            starter.setdefault(pit_team, pitcher)

            slot = n_pa[team] % 9
            n_pa[team] += 1
            batter = int(mu["batter"]["id"])
            first = slot_of.setdefault((team, slot), batter)

            game_rows.append(
                {
                    "game": path.stem,
                    "team": team,
                    "slot": slot + 1,
                    "inning": int(about["inning"]),
                    "batter": batter,
                    "removed": int(batter != first),
                    "bat_hand": mu["batSide"]["code"],
                    "orig_hand": hand_of.setdefault(
                        (team, slot), mu["batSide"]["code"]
                    ),
                    "pit_hand": mu["pitchHand"]["code"],
                    "sp_out": int(pitcher != starter[pit_team]),
                    "tto": n_pa[team] // 9,
                }
            )
        rows.extend(game_rows)
    return rows


def rate(rows: list[dict], *keys: str) -> list[tuple[tuple, int, float]]:
    """Removal share of slot appearances, grouped by ``keys``."""
    agg: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        cell = agg[tuple(r[k] for k in keys)]
        cell[0] += 1
        cell[1] += r["removed"]
    return [(k, n, d / n) for k, (n, d) in sorted(agg.items())]


def show(title: str, table: list[tuple[tuple, int, float]], floor: int = 200) -> None:
    print(f"\n{title}")
    for key, n, p in table:
        if n < floor:
            continue
        label = " ".join(str(x) for x in key)
        print(f"  {label:<24} n={n:<7} removed {p:6.2%}")


def hazard(rows: list[dict], *keys: str) -> list[tuple[tuple, int, float]]:
    """Conditional hazard: he was still in, and this appearance he is gone.

    The share above counts a slot as removed for every later appearance, which
    over-weights early exits. This is the number a simulation needs: given the
    original occupant is still batting, the chance the *next* time the slot comes
    up it belongs to someone else.
    """
    return rate([r for r in rows if not r["prev_removed"]], *keys)


def with_history(rows: list[dict]) -> list[dict]:
    """Tag each appearance with whether the slot was already gone before it."""
    seen: dict[tuple[str, str], int] = {}
    out = []
    for r in rows:
        key = (r["game"], f"{r['team']}{r['slot']}")
        out.append(dict(r, prev_removed=seen.get(key, 0)))
        seen[key] = r["removed"]
    return out


def pa_loss(rows: list[dict]) -> None:
    """What the fixed-lineup assumption actually costs the original hitter."""
    tot: dict[tuple[str, str], int] = defaultdict(int)
    his: dict[tuple[str, str], int] = defaultdict(int)
    slot: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["game"], f"{r['team']}{r['slot']}")
        tot[key] += 1
        his[key] += 1 - r["removed"]
        slot[key] = r["slot"]
    by_slot: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for key, n in tot.items():
        cell = by_slot[slot[key]]
        cell[0] += 1
        cell[1] += n
        cell[2] += his[key]
        cell[3] += int(his[key] < n)
    print("\nplate appearances the slot took vs the original hitter took")
    for s in sorted(by_slot):
        games, slot_pa, own_pa, lost = by_slot[s]
        print(
            f"  slot {s}  games={games:<5} slot PA {slot_pa / games:.2f}"
            f"  his PA {own_pa / games:.2f}"
            f"  lost {(slot_pa - own_pa) / games:.2f}"
            f"  ({lost / games:5.1%} of games he is lifted)"
        )


def main() -> None:
    rows = with_history(slot_appearances())
    games = len({r["game"] for r in rows})
    print(f"{len(rows):,} slot appearances, {games} games")
    print(f"overall removal share {sum(r['removed'] for r in rows) / len(rows):.2%}")

    show("by inning", rate(rows, "inning"))
    show("by slot", rate(rows, "slot"))
    show("by starter state (opposing SP still in / out)", rate(rows, "sp_out"))
    late = [r for r in rows if r["inning"] >= 7]
    print(f"\ninning 7+: n={len(late):,}")
    show("  inning 7+ by slot", rate(late, "slot"), floor=100)
    show("  inning 7+ by starter state", rate(late, "sp_out"), floor=100)
    show("  inning 7+, SP out, by inning x slot",
         rate([r for r in late if r["sp_out"]], "inning", "slot"), floor=60)

    # Platoon state of the matchup the slot is walking into. A removal is a
    # decision made *before* the plate appearance, so the hand shown here is the
    # hand of whoever ended up batting -- the honest read is on the pitcher.
    plat = [
        dict(r, opp_hand=r["pit_hand"])
        for r in late
        if r["sp_out"] and r["pit_hand"] in ("L", "R")
    ]
    show("  inning 7+, SP out, by pitcher hand", rate(plat, "opp_hand"), floor=100)
    show("  inning 7+, SP out, by pitcher hand x slot",
         rate(plat, "opp_hand", "slot"), floor=60)

    print("\n--- conditional hazard (he was still in as of this appearance) ---")
    show("by inning", hazard(rows, "inning"), floor=100)
    show("by starter state", hazard(rows, "sp_out"), floor=100)
    show("inning 7+, SP out, by slot",
         hazard([r for r in rows if r["inning"] >= 7 and r["sp_out"]], "slot"),
         floor=100)

    # The platoon question, on the original occupant's hand rather than the
    # substitute's: is the hitter on the wrong side of the matchup the one lifted?
    late_out = [
        r for r in rows
        if r["inning"] >= 7 and r["sp_out"]
        and r["orig_hand"] in ("L", "R") and r["pit_hand"] in ("L", "R")
    ]
    show("inning 7+, SP out, original hand vs pitcher hand",
         hazard(late_out, "orig_hand", "pit_hand"), floor=100)
    disadv = [dict(r, bad=int(r["orig_hand"] == r["pit_hand"])) for r in late_out]
    show("inning 7+, SP out, platoon disadvantage (same hand)",
         hazard(disadv, "bad"), floor=100)
    show("inning 7+, SP out, platoon disadvantage x slot",
         hazard(disadv, "bad", "slot"), floor=60)

    pa_loss(rows)


if __name__ == "__main__":
    main()
