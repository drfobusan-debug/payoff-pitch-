"""NFL prediction engine: moneyline, ATS, totals and player props.

Third sibling to :mod:`mlb_engine` and :mod:`cfb_engine`, and deliberately the
one built in the opposite order. The MLB engine learned every lesson on graded
bets one slate at a time; the NFL has 272 games a season, so a ledger takes a
whole year to reach the size at which the MLB ledger was still misleading us.
What the NFL has instead is history: 7,276 completed games with closing spreads,
totals and moneylines back to 1999, free. So the rule here is that nothing enters
a price off this engine's own ledger -- every term is fitted on the historical
panel first, and the ledger's job is grading and closing-line value.
"""
