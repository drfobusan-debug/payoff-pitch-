"""Does the game a player is in move his prop, once his own usage is known?

:mod:`scripts.nfl.props_study` established that usage is projectable and that a
projection made of usage alone has no edge at the line: the book knows his usage
too. This study asks what is left after usage -- the things the usage model does
not see and a book does -- and measures each one out of time, on the same rows,
at the same pseudo-line, so a gain here is a gain against the baseline the props
layer ships rather than against the position mean.

Four blocks, each built only from weeks before the one being projected:

*share*    the player's share of his team's volume (targets per team pass attempt,
           carries per team carry), shrunk, times the team's projected volume.
           A share is stabler than a count, and the volume term is where a game
           script can enter.
*opponent* what the defence has allowed per game to his position group, as a
           ratio to the league, shrunk toward one.
*script*   the closing spread from his team's side and the closing total, from
           nflverse ``games`` -- the market's own read of how the game goes.
*all*      the three together.

Each block is a ridge correction to the baseline on the training seasons; every
term is an interaction with the baseline so the effect scales with the role, and
the design is :func:`nfl_engine.features.context.features`, so what is measured
here is what ships. The holdout is scored at the pseudo-line drawn on the
*baseline* projection for every model alike, so what is compared is probability
quality at one fixed line -- of our own, not a book's.

    python -m scripts.nfl.props_context_study --cutoff 2021 --coefficients
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features.context import (
    OPP_SHRINK,
    VOLUME_OF,
    Terms,
    features,
    team_volume,
)
from nfl_engine.features.usage import POSITIONS, REGULAR
from nfl_engine.models.player import (
    MIN_GAMES,
    SHRINK,
    STATS,
    USAGE_FLOOR,
    Spread,
    prob_over,
)
from scripts.nfl.props_study import MIN_GAIN, pseudo_line, spread_fit

# Which columns of the full design each block keeps: intercept and baseline
# always, then share / opponent / spread+total.
BLOCKS = {
    "share": (0, 1, 2),
    "opponent": (0, 1, 3),
    "script": (0, 1, 4, 5),
    "all": (0, 1, 2, 3, 4, 5),
}


def weekly_with_games(seasons: list[int]) -> pd.DataFrame:
    """The study frame, with the team, opponent and game each row belongs to."""
    frames = []
    for season in seasons:
        frame = nflverse.player_week(season)
        if frame.empty:
            continue
        if "season_type" in frame.columns:
            frame = frame[frame.season_type == REGULAR]
        keep = ["player_id", "position", "season", "week", "team", "opponent_team", "game_id"]
        columns = [c for c in keep + list(STATS) if c in frame.columns]
        frames.append(frame[columns])
    rows = pd.concat(frames, ignore_index=True)
    return rows.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def _prior_shrunk(
    frame: pd.DataFrame, by: list[str], value: str, anchor: pd.Series, shrink: float
) -> pd.Series:
    """The shrunk mean of ``value`` over the group's earlier rows this season."""
    grouped = frame.groupby(by)[value]
    prior_sum = grouped.cumsum() - frame[value]
    prior_n = grouped.cumcount()
    return (prior_sum + shrink * anchor) / (prior_n + shrink)


def _prior_league_mean(frame: pd.DataFrame, value: str) -> pd.Series:
    """The league mean of ``value`` over the season's earlier weeks, last season's
    mean before week one -- never the week itself."""
    weekly = frame.groupby(["season", "week"])[value].agg(["sum", "count"]).reset_index()
    weekly = weekly.sort_values(["season", "week"])
    by_season = weekly.groupby("season")
    prior_sum = by_season["sum"].cumsum() - weekly["sum"]
    prior_n = by_season["count"].cumsum() - weekly["count"]
    last = weekly.groupby("season")["sum"].sum() / weekly.groupby("season")["count"].sum()
    fallback = weekly.season.map(last.shift(1)).fillna(last.iloc[0])
    weekly["league"] = np.where(prior_n > 0, prior_sum / prior_n.clip(lower=1), fallback)
    merged = frame.merge(weekly[["season", "week", "league"]], how="left", on=["season", "week"])
    return pd.Series(merged.league.to_numpy(), index=frame.index)


def projected_volume(vol: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's projected volume from its earlier games this season."""
    vol = vol.sort_values(["team", "season", "week"]).reset_index(drop=True)
    for col in ("team_attempts", "team_carries", "team_plays"):
        season_mean = vol.groupby(["team", "season"])[col].mean().rename("prev")
        prev = season_mean.reset_index()
        prev["season"] = prev.season + 1
        merged = vol.merge(prev, how="left", on=["team", "season"])
        anchor = merged.prev.fillna(_prior_league_mean(vol, col)).to_numpy()
        vol[f"{col}_proj"] = _prior_shrunk(vol, ["team", "season"], col, anchor, SHRINK)
    return vol


def opponent_factor(rows: pd.DataFrame, stat: str) -> pd.DataFrame:
    """What each defence allowed per game to the stat's position group, as a
    shrunk ratio to the league, from its earlier games this season."""
    pos = rows[rows.position.isin(POSITIONS[stat])]
    allowed = (
        pos.groupby(["season", "week", "game_id", "opponent_team"], as_index=False)[stat]
        .sum()
        .rename(columns={"opponent_team": "defence", stat: "allowed"})
    )
    allowed["ratio"] = allowed.allowed / _prior_league_mean(allowed, "allowed")
    allowed = allowed.sort_values(["defence", "season", "week"]).reset_index(drop=True)
    anchor = pd.Series(np.ones(len(allowed)), index=allowed.index)
    allowed["opp_factor"] = _prior_shrunk(
        allowed, ["defence", "season"], "ratio", anchor, OPP_SHRINK
    )
    return allowed[["game_id", "defence", "opp_factor"]]


def baseline(rows: pd.DataFrame, stat: str) -> pd.DataFrame:
    """The props layer's own projection, and the player's shrunk share."""
    out = rows[rows.position.isin(POSITIONS[stat]) & rows[stat].notna()].copy()
    volume = VOLUME_OF[stat]
    out["share"] = out[stat] / out[volume].clip(lower=1.0)

    for col, target in ((stat, "proj"), ("share", "share_proj")):
        season_mean = out.groupby(["player_id", "season"])[col].mean().rename("prev")
        prev = season_mean.reset_index()
        prev["season"] = prev.season + 1
        merged = out.merge(prev, how="left", on=["player_id", "season"])
        pos_mean = out.groupby("position")[col].transform("mean")
        anchor = merged.prev.fillna(pos_mean).to_numpy()
        out[target] = _prior_shrunk(out, ["player_id", "season"], col, anchor, SHRINK)
    out["prior_n"] = out.groupby(["player_id", "season"]).cumcount()
    out["share_volume"] = out.share_proj * out[f"{volume}_proj"]
    return out[out.prior_n >= MIN_GAMES]


def design(frame: pd.DataFrame, block: str) -> np.ndarray:
    """The shipped design, restricted to one block's columns."""
    full = features(
        frame.proj.to_numpy(),
        frame.share_volume.to_numpy(),
        frame.opp_factor.to_numpy(),
        frame.team_spread.to_numpy(),
        frame.total_line.to_numpy(),
    )
    return full[:, list(BLOCKS[block])]


def fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ y)


def score(
    hold: pd.DataFrame, stat: str, mean: np.ndarray, spread: Spread, lines: np.ndarray
) -> tuple[float, float]:
    probs = np.array(
        [
            prob_over(stat, m, line, sd=spread.sd(m)).conditional
            for m, line in zip(mean, lines, strict=True)
        ]
    )
    hit = (hold[stat].to_numpy() > lines).astype(int)
    mae = float(np.abs(hold[stat].to_numpy() - mean).mean())
    return mae, float(((probs - hit) ** 2).mean())


def measure(rows: pd.DataFrame, stat: str, cutoff: int) -> tuple[dict[str, float], Terms | None]:
    """Score every block out of time; return the gains and the ``all`` terms."""
    frame = baseline(rows, stat).dropna(
        subset=["team_spread", "total_line", "opp_factor", "share_volume"]
    )
    train = frame[frame.season <= cutoff]
    hold = frame[(frame.season > cutoff) & (frame.proj >= USAGE_FLOOR[stat])]
    if train.empty or hold.empty:
        print(f"{stat:18s} not enough rows")
        return {}, None
    lines = np.array([pseudo_line(stat, p) for p in hold.proj])
    hit = (hold[stat].to_numpy() > lines).astype(int)
    base_brier = float(((hit.mean() - hit) ** 2).mean())

    spread = spread_fit(frame, stat, cutoff)
    mae0, brier0 = score(hold, stat, hold.proj.to_numpy(), spread, lines)
    print(
        f"{stat:18s} n={len(hold):6d} over={hit.mean():.3f}"
        f"  baseline MAE={mae0:6.2f} Brier={brier0:.4f} (base rate {base_brier:.4f})"
    )
    gains: dict[str, float] = {}
    terms: Terms | None = None
    for block in BLOCKS:
        beta = fit(design(train, block), train[stat].to_numpy())
        fitted = train.assign(proj=design(train, block) @ beta)
        sd = spread_fit(fitted, stat, cutoff)
        mean = np.clip(design(hold, block) @ beta, 0.0, None)
        mae, brier = score(hold, stat, mean, sd, lines)
        gain = brier0 - brier
        gains[block] = gain
        verdict = "gain" if gain >= MIN_GAIN else ("loss" if gain <= -MIN_GAIN else "tie")
        shown = " ".join(f"{b:+.3f}" for b in beta[2:])
        print(
            f"    {block:9s} MAE={mae:6.2f} ({mae - mae0:+.2f})"
            f" Brier={brier:.4f} ({gain:+.4f}) {verdict:4s}  terms {shown}"
        )
        if block == "all":
            terms = Terms(*(float(b) for b in beta))
    return gains, terms


def build(seasons: list[int]) -> pd.DataFrame:
    rows = weekly_with_games(seasons)
    vol = team_volume(rows)
    vol = projected_volume(vol)
    rows = rows.merge(vol, how="left", on=["season", "week", "team"])
    games = nflverse.games()[["game_id", "home_team", "spread_line", "total_line"]]
    rows = rows.merge(games, how="left", on="game_id")
    home = rows.team == rows.home_team
    rows["team_spread"] = np.where(home, rows.spread_line, -rows.spread_line)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2016)
    parser.add_argument("--cutoff", type=int, default=2021, help="last training season")
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument(
        "--coefficients",
        action="store_true",
        help="print the `all` terms per stat in the form nfl_engine.features.context.TERMS takes",
    )
    args = parser.parse_args()

    rows = build(list(range(args.first, args.last + 1)))
    fitted: dict[str, Terms] = {}
    for stat in STATS:
        opp = opponent_factor(rows, stat).rename(columns={"defence": "opponent_team"})
        _, terms = measure(
            rows.merge(opp, how="left", on=["game_id", "opponent_team"]), stat, args.cutoff
        )
        if terms is not None:
            fitted[stat] = terms
    if args.coefficients:
        print("\nTERMS = {")
        for stat, terms in fitted.items():
            print(
                f"    {stat.upper()}: Terms({terms.intercept:.4f}, {terms.usage:.4f},"
                f" {terms.share:.4f}, {terms.opponent:.4f}, {terms.spread:.4f}, {terms.total:.4f}),"
            )
        print("}")


if __name__ == "__main__":
    main()
