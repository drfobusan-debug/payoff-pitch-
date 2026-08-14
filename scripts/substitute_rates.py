"""What a substitute actually hits, measured off the same play-by-play.

The removal branch needs a rate vector for whoever takes the slot. A league
average bat is the wrong guess in principle -- pinch hitters and defensive subs
are bench players facing high-leverage relief -- so measure it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PBP = Path.home() / ".mlb_engine" / "cache" / "pbp"

EVENT = {
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


def main() -> None:
    counts: dict[str, dict[str, int]] = {
        "starter": defaultdict(int),
        "substitute": defaultdict(int),
        "late_all": defaultdict(int),
    }
    for path in sorted(PBP.glob("*.json")):
        try:
            plays = json.loads(path.read_text()).get("allPlays") or []
        except (json.JSONDecodeError, OSError):
            continue
        plays = [p for p in plays if (p.get("result") or {}).get("type") == "atBat"]
        if not plays or max(int(p["about"]["inning"]) for p in plays) < 8:
            continue
        n_pa: dict[str, int] = defaultdict(int)
        first: dict[tuple[str, int], int] = {}
        for play in plays:
            about, mu = play["about"], play["matchup"]
            team = "away" if about["isTopInning"] else "home"
            slot = n_pa[team] % 9
            n_pa[team] += 1
            batter = int(mu["batter"]["id"])
            orig = first.setdefault((team, slot), batter)
            oc = EVENT.get(str(play["result"].get("eventType", "")), "OUT")
            group = "substitute" if batter != orig else "starter"
            counts[group][oc] += 1
            if int(about["inning"]) >= 7:
                counts["late_all"][oc] += 1

    for group, c in counts.items():
        n = sum(c.values())
        line = "  ".join(
            f"{k} {c[k] / n:.4f}" for k in ("1B", "2B", "3B", "HR", "BB", "K", "OUT")
        )
        print(f"{group:<11} n={n:<7} {line}")


if __name__ == "__main__":
    main()
