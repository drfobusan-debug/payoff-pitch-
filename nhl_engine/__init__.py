"""NHL betting engine.

Phase 0: the price archive and the official results feed.
Phase 1: MoneyPuck ingestion with as-of slicing, EB-shrunk team-strength and
goalie (GSAx/60) features with study-fitted ``k`` per metric, and the preseason
prior asset. Nothing here forms a game probability or recommends a bet; see
``docs/nhl/master_plan.md`` §6 for what each phase ships.
"""
