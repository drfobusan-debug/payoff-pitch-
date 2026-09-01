"""The morning brief: each game written out, in the order a reader thinks.

The card's tables answer "what did we buy and at what price". They do not answer
the question a person actually asks on a Sunday morning -- *why this game, and
what is it about it that moved the number* -- and a table of five probabilities
per row is a poor place to look for it.

So this is prose, generated from the same ledger rows the tables are built from,
plus the schedule context the pricing run already fetched. Nothing here computes
a probability, chooses a side or changes a tier: every number quoted is one the
market layer wrote, and the brief's only job is to say what it means.

The line it will not cross is asserting that something moves a price when the
engine's own evidence says it does not. Rest, travel, cold and injury are in here
because they are the first things a reader asks about, and each one is written as
what it is -- context the market has already priced, carried at zero weight by
:mod:`nfl_engine.features.adjustments`, and labelled *reported, not priced* every
time it appears. Wind and the divisional flag are the two that survived their own
test, and only those two are ever described as having moved the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from mlb_engine.data.openmeteo import VenueWeather
from nfl_engine.audit.ledger import LedgerEntry
from nfl_engine.data import weather
from nfl_engine.data.schedule import ScheduleContext, week_context
from nfl_engine.features.adjustments import (
    Situation,
    div_total_delta,
    unpriced_notes,
    wind_total_delta,
)
from nfl_engine.market.ev import MONEYLINE, SPREAD, TOTAL
from nfl_engine.market.screens import Tier

# Rest days beyond which a team is off a bye, and at or below which it is on a
# short week. Both mirror :mod:`nfl_engine.features.adjustments`, where both
# measured as nothing.
BYE_REST = 11
SHORT_REST = 5


@dataclass(frozen=True)
class GameContext:
    """What the schedule and the sky said about one game.

    Both halves are optional and independently so: nflverse can be unreachable,
    a forecast can fail, and a brief written without either is a shorter brief
    rather than a missing one.
    """

    schedule: ScheduleContext | None = None
    weather: VenueWeather | None = None

    def situation(self) -> Situation:
        sched, sky = self.schedule, self.weather
        return Situation(
            roof=sched.roof if sched else None,
            wind_mph=sky.wind_mph if sky else None,
            temp_f=sky.temperature_f if sky else None,
            home_rest=sched.home_rest if sched else None,
            away_rest=sched.away_rest if sched else None,
            neutral_site=bool(sched.neutral_site) if sched else False,
            div_game=sched.div_game if sched else None,
        )


@dataclass
class GameBrief:
    """One game, in paragraphs. Empty means there was nothing worth saying."""

    matchup: str
    headline: str = ""
    paragraphs: list[str] = field(default_factory=list)


def context_for(season: int, week: int, kickoffs: dict[str, str]) -> dict[str, GameContext]:
    """Schedule and weather for the games named in ``matchup -> kickoff stamp``.

    Rebuilt here rather than persisted with the bet because a card is written
    from the ledger and may be written weeks later: the schedule is a local file
    and Open-Meteo answers for a past kickoff out of its archive, so the same
    call serves a Sunday morning and a post-mortem.
    """
    contexts = {
        matchup: context
        for matchup, context in week_context(season, week).items()
        if matchup in kickoffs
    }
    points: dict[str, tuple[str, datetime]] = {}
    for matchup, context in contexts.items():
        kickoff = weather.parse_kickoff(kickoffs[matchup])
        if context.stadium_id and kickoff and weather.is_outdoors(context.roof):
            points[matchup] = (context.stadium_id, kickoff)
    skies = weather.readings(points) if points else {}
    return {
        matchup: GameContext(schedule=context, weather=skies.get(matchup))
        for matchup, context in contexts.items()
    }


def _teams(matchup: str) -> tuple[str, str]:
    """``"NYJ @ BUF"`` -> ``("NYJ", "BUF")``."""
    away, _, home = matchup.partition(" @ ")
    return away.strip(), home.strip()


def _pct(prob: float | None) -> str:
    return "n/a" if prob is None else f"{prob * 100:.0f}%"


def _points(value: float) -> str:
    return f"{value:+.1f}"


@dataclass(frozen=True)
class MarketView:
    """The market's own statement about the game, off the priced rows."""

    home_spread: float | None = None
    total: float | None = None
    home_prob: float | None = None
    away_prob: float | None = None

    def favourite(self, home: str, away: str) -> tuple[str, float] | None:
        if self.home_spread is None or self.home_spread == 0:
            return None
        return (home, -self.home_spread) if self.home_spread < 0 else (away, self.home_spread)


def market_view(entries: list[LedgerEntry], home: str, away: str) -> MarketView:
    """What the board said, taken from the rows rather than re-fetched.

    The moneyline probabilities are the de-vigged ones the pricing run computed,
    so the brief quotes the same fair number the edge was measured against, not a
    raw price with the hold still in it.
    """
    home_spread = next(
        (e.line for e in entries if e.market == SPREAD and e.side == home and e.line is not None),
        None,
    )
    if home_spread is None:
        away_line = next(
            (
                e.line
                for e in entries
                if e.market == SPREAD and e.side == away and e.line is not None
            ),
            None,
        )
        home_spread = None if away_line is None else -away_line
    total = next((e.line for e in entries if e.market == TOTAL and e.line is not None), None)
    probs = {
        e.side: e.fair_prob for e in entries if e.market == MONEYLINE and e.fair_prob is not None
    }
    return MarketView(
        home_spread=home_spread,
        total=total,
        home_prob=probs.get(home),
        away_prob=probs.get(away),
    )


def _headline(matchup: str, market: MarketView, context: GameContext) -> str:
    away, home = _teams(matchup)
    favourite = market.favourite(home, away)
    if favourite is None:
        price = "a pick'em" if market.home_spread == 0 else "no posted spread yet"
    else:
        team, points = favourite
        price = f"{team} by {points:g}"
    total = "" if market.total is None else f", total {market.total:g}"
    sched = context.schedule
    if sched is not None and sched.div_game:
        kind = "Divisional"
    elif sched is not None and sched.div_game is False:
        kind = "Non-divisional"
    else:
        kind = "Matchup"
    return f"{kind}: {price}{total}."


def _stakes(matchup: str, market: MarketView, context: GameContext, week: int) -> str:
    """Why the game matters, and what the market thinks of it."""
    away, home = _teams(matchup)
    bits: list[str] = []
    sched = context.schedule
    if sched is not None and sched.div_game:
        bits.append(
            "A divisional game carries the tiebreaker as well as the win, which is the"
            " leverage nobody outside the two buildings sees in the standings"
        )
    elif sched is not None and sched.div_game is False:
        bits.append("Out of division, so this is a game with no tiebreaker attached")
    if market.home_prob is not None and market.away_prob is not None:
        bits.append(
            f"the market makes it {home} {_pct(market.home_prob)} / {away}"
            f" {_pct(market.away_prob)} once the hold is taken out"
        )
    if week >= 15:
        bits.append(f"and week {week} is where a game like this decides seeding")
    if not bits:
        return ""
    return ". ".join(bits).replace(". and ", ", and ") + "."


def _edge(entries: list[LedgerEntry]) -> str:
    """What the engine disagreed with the market about, and by how much."""
    bought = [
        e for e in entries if not e.screens and e.tier in (Tier.STRONG.value, Tier.MODERATE.value)
    ]
    if not bought:
        vetoed = sorted({name for e in entries for name in e.screens.split(";") if name})
        if not vetoed:
            return "No disagreement worth a bet: the model and the price agree here."
        return (
            "Nothing survives on this game. The model's number was inside the price on"
            f" every market, or a screen refused it ({', '.join(vetoed)})."
        )
    parts: list[str] = []
    for bet in sorted(bought, key=lambda e: -(e.ev_fair or 0.0)):
        if bet.line is None:
            line = ""
        else:
            line = f" {bet.line:g}" if bet.market == TOTAL else f" {bet.line:+g}"
        price = "n/a" if bet.odds is None else f"{bet.odds:+.0f}"
        gap = None if bet.fair_prob is None else (bet.model_prob - bet.fair_prob) * 100
        edge = "" if gap is None else f", {gap:+.1f} points of edge over the fair price"
        parts.append(
            f"{bet.tier} on {bet.side}{line} at {price} ({bet.book}): the simulation"
            f" wins it {_pct(bet.model_prob)} against a fair {_pct(bet.fair_prob)}{edge}"
        )
    return ". ".join(parts) + "."


def _conditions(context: GameContext) -> str:
    """Weather, and specifically whether it was allowed to move anything."""
    situation = context.situation()
    if situation.indoors():
        return "Indoors, so the weather never enters it."
    sky = context.weather
    if sky is None or sky.wind_mph is None:
        if context.schedule is None or context.schedule.roof is None:
            return (
                "No roof on file for the venue, so the engine declined to guess at the"
                " weather rather than price a room as a field."
            )
        return "No kickoff forecast came back, so nothing weather-related touched the total."
    wind = wind_total_delta(situation)
    temp = "" if sky.temperature_f is None else f" at {sky.temperature_f:.0f}F"
    if abs(wind) < 0.05:
        body = (
            f"{sky.wind_mph:.0f} mph{temp} is the league's average outdoor game, and worth nothing"
        )
    else:
        direction = "off" if wind < 0 else "onto"
        body = (
            f"{sky.wind_mph:.0f} mph{temp} takes {abs(wind):.1f} {direction} the total"
            " -- the one weather term that beat the closing line in the fit (-0.20 per"
            " mph over an 8.5 mph average)"
        )
    gust = "" if sky.gust_mph is None else f" Gusts to {sky.gust_mph:.0f}."
    rain = (
        " Rain in the kickoff hour, which measured as nothing on its own."
        if sky.precipitation is not None and sky.precipitation > 0.2
        else ""
    )
    return f"Wind: {body}.{gust}{rain}"


def _travel(matchup: str, context: GameContext) -> str:
    """Rest, the bye and the flight -- every one of them reported, not priced."""
    away, home = _teams(matchup)
    sched = context.schedule
    if sched is None:
        return ""
    bits: list[str] = []
    if sched.home_rest is not None and sched.away_rest is not None:
        gap = sched.home_rest - sched.away_rest
        rested = f"{home} on {sched.home_rest} days' rest to {away}'s {sched.away_rest}"
        if abs(gap) >= 3:
            bits.append(f"{rested} -- a {abs(gap)}-day edge to {home if gap > 0 else away}")
        else:
            bits.append(f"{rested}, effectively level")
    for note in unpriced_notes(context.situation()):
        bits.append(note)
    if not bits:
        return ""
    return (
        "Rest and travel: "
        + "; ".join(bits)
        + ". Every one of those is reported, not priced: rest differential fitted at"
        " +0.06 points a day (t +0.87) against the closing line, which is another way"
        " of saying the market already has it."
    )


def _clv(entries: list[LedgerEntry]) -> str:
    """Did we beat the close -- the only leading indicator the engine trusts."""
    closed = [e for e in entries if not e.screens and e.clv is not None]
    if not closed:
        return ""
    parts: list[str] = []
    for bet in closed:
        clv = bet.clv or 0.0
        close = "" if bet.close_odds is None else f" (closed {bet.close_odds:+.0f})"
        verdict = "the market came to us" if clv > 0 else "the market went the other way"
        parts.append(f"{bet.side} {clv * 100:+.1f} pts{close}, {verdict}")
    mean = sum(e.clv or 0.0 for e in closed) / len(closed)
    return (
        "Closing line value: "
        + "; ".join(parts)
        + f". Mean {mean * 100:+.1f} points, and this is the column that says whether the"
        " selection was any good long before the record does."
    )


def _availability(absences: str) -> str:
    if not absences:
        return ""
    return (
        absences + " The absence itself is not charged to the price: the edge in an injury is"
        " hearing it before the number moves, and that timing is what the availability"
        " log is measuring."
    )


def write_brief(
    matchup: str,
    entries: list[LedgerEntry],
    *,
    week: int,
    context: GameContext | None = None,
    absences: str = "",
) -> GameBrief:
    """One game's brief, from its own ledger rows and whatever context exists."""
    ctx = context or GameContext()
    away, home = _teams(matchup)
    market = market_view(entries, home, away)
    paragraphs = [
        _stakes(matchup, market, ctx, week),
        _edge(entries),
        _conditions(ctx),
        _travel(matchup, ctx),
        _clv(entries),
        _availability(absences),
    ]
    return GameBrief(
        matchup=matchup,
        headline=_headline(matchup, market, ctx),
        paragraphs=[text for text in paragraphs if text],
    )


def priced_context(context: GameContext) -> str:
    """The situational points actually applied, for the card's own footnote."""
    situation = context.situation()
    wind, div = wind_total_delta(situation), div_total_delta(situation)
    bits: list[str] = []
    if wind:
        bits.append(f"wind {_points(wind)} total")
    if div:
        label = "divisional" if situation.div_game else "non-divisional"
        bits.append(f"{label} {_points(div)} total")
    return ", ".join(bits)
