"""Feature layer: team-strength and goalie rates, as-of a slate date, EB-shrunk.

Nothing here prices a market. ``strength.py`` and ``goalie.py`` turn MoneyPuck
game logs into shrunk rates with their exposure and reliability attached, so
Phase 2 can build λ from them and the card can say how much of each number is
data versus prior.
"""
