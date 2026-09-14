"""
battery_chemistry.py
====================

Which battery chemistry the cars of a segment group carry, in a given year,
under a given scenario -- and how much of that has no composition behind it.

Shares are stated percentages at the anchor years, interpolated between them,
held flat outside, and renormalised per group and year. They are NOT drawn: a
scenario saying 30% LFP is an assumption about a real fleet mix, and drawing it
would turn a stated input into a spread it never had.

Lives in src/ rather than in the stage because the stage and its figures both
need it, and a shared definition cannot drift apart the way two copies can.
"""

from __future__ import annotations

import numpy as np


def chemistry_share(params, scenario: str, group: str, chemistry: str,
                    year: float) -> float:
    """One chemistry's share of a segment group in a year, renormalised."""
    materials = params.materials
    anchors = np.asarray(materials.battery_chemistry_anchor_years, dtype=float)
    definition = materials.battery_chemistry_scenarios[scenario][group]
    total = sum(np.interp(year, anchors, np.asarray(v, dtype=float))
                for v in definition.values())
    if total <= 0:
        return 0.0
    here = np.interp(year, anchors, np.asarray(definition[chemistry], dtype=float))
    return float(here / total)


def uncovered_share(params, scenario: str, group: str, year: float) -> float:
    """
    The share of a group with no composition, SUMMED over the chemistries that
    lack one -- sodium-ion and solid-state are both missing under S3, and the
    hole they leave is the two together.
    """
    materials = params.materials
    named = materials.battery_chemistry_file_names
    return sum(chemistry_share(params, scenario, group, chemistry, year)
               for chemistry in materials.battery_chemistry_scenarios[scenario][group]
               if chemistry not in named)
