"""Pure, dependency-free primitives shared across engines.

Rule (docs/nhl/master_plan.md §4): only code that can be shared *without
altering any existing engine* lives here -- math on numbers, no sport-specific
thresholds, no I/O. Each engine imports what it needs; nothing here imports an
engine.
"""
