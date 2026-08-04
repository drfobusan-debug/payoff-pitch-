"""Situational feature filters layered on top of team ratings.

The classic handicapping angles a betting guide leans on -- home field, rest and
fatigue, travel, weather, and mean-reversion regression -- expressed as point
adjustments to the model's expected margin and total. Every adjustment is a
no-op when its input data is unavailable, so the engine still runs on ratings
and the market alone.
"""
