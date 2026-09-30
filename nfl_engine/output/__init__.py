"""Reader-facing output: the weekly card, the workbook, and email delivery.

Everything here reads the ledger and writes a document. Nothing in this package
forms a probability, chooses a side or moves a price -- the MLB engine's report
layer earned that rule the hard way, and keeping it means a formatting change can
never be mistaken for a modelling one.
"""
