"""Does the anchored distribution need a calibration map, per market?

Thin driver over :mod:`nfl_engine.calibration`: build the historical observations,
fit on seasons up to ``--cutoff``, score on everything after, and print what the
holdout says about each market. ``nfl-engine calibrate`` runs the same thing and
can write the map file; this exists so a study can sweep the cutoff without
touching the shipped artifact.

Usage::

    python scripts/nfl/calibration_study.py --first 2007 --cutoff 2019 --sims 20000
"""

from __future__ import annotations

import argparse

from nfl_engine import calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2007)
    parser.add_argument("--cutoff", type=int, default=2019, help="last training season")
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()

    rows = calibration.observations(first=args.first, sims=args.sims)
    for line in calibration.report_lines(calibration.fit(rows, cutoff=args.cutoff)):
        print(line)


if __name__ == "__main__":
    main()
