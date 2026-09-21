"""In-play edge detector for the CFB and NFL engines.

Rolls a pregame prior (the closing consensus, or the engine's own number) forward
through the live score and clock, compares the resulting win / cover / over
probabilities with devigged in-play prices, and writes watch-only flags with a
tape of every snapshot so each flag can be graded and the rule calibrated.
"""
