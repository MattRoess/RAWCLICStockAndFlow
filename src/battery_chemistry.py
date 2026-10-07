"""
battery_chemistry.py
====================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Which battery chemistry the cars of a segment group carry, in a given year,
under a given scenario.

Shares are stated percentages at the anchor years, interpolated between them,
held flat outside, and renormalised per group and year.

THE SHARES ARE DRAWN, AND THE DRAW IS A STATEMENT ABOUT DISTANCE, NOT ABOUT THE
SCENARIO. A scenario is a stated assumption and `chemistry_share` returns it
unchanged. But a share stated for 2070 is a guess made forty-five years early,
and `chemistry_share_draws` says so: a triangular multiplier whose half-width
grows from nothing in 2020 to the parameter's full value in 2070.

NOTHING IS A RESIDUAL. Every chemistry is perturbed, and the group is then
renormalised to one. The renormalisation IS the correlation -- one chemistry
gaining means the others give way, in that draw -- rather than a bookkeeping
trick that dumps the imbalance on whichever chemistry was listed last.

ONE DRAW PER CHEMISTRY, SHARED ACROSS GROUPS AND HELD ACROSS YEARS. If a
chemistry beats expectations it beats them in every group that carries it, so the
multiplier is keyed on the chemistry alone. Redrawing it per year would make a
trajectory that jitters, which is noise and not uncertainty; redrawing it per
group would let a chemistry win in one segment and lose in the next for no reason.
The two sodium cells are two chemistries and so draw independently: the uncertainty
of sodium as a whole is narrower than one multiplier on all of it would give.

The seed is offset with `zlib.crc32`, never `hash()`: Python salts `str` hashing
per process, which cost this project its reproducibility once already.

Lives in src/ rather than in the stage because the stage and its figures both
need it, and a shared definition cannot drift apart the way two copies can.
"""

from __future__ import annotations

import zlib

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


def _share_multiplier(params, chemistry: str, years, n_draws: int,
                      seed: int) -> np.ndarray:
    """
    (n_draws, n_years) multiplier on one chemistry's stated share.

    One triangular deviate per chemistry, held across every year and group, with
    its distance from 1 scaled by how far the year is from the present.
    """
    materials = params.materials
    spread = materials.battery_chemistry_share_spread
    first, last = materials.battery_chemistry_share_spread_years
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), zlib.crc32(chemistry.encode())]))
    drawn = rng.triangular(spread["min"], spread["mode"], spread["max"], n_draws)
    ramp = np.clip((np.asarray(years, dtype=float) - first) / (last - first), 0.0, 1.0)
    # float32 throughout: five chemistries over sixty-seven build years at
    # 200,000 draws is half a gigabyte in float64 and half of that here, and a
    # share carries nowhere near seven significant digits of real information.
    return (1.0 + (drawn - 1.0)[:, None] * ramp[None, :]).astype(np.float32)


def chemistry_share_draws(params, scenario: str, group: str, years,
                          n_draws: int, seed: int = 0) -> dict[str, np.ndarray]:
    """
    Every chemistry of one segment group, as (n_draws, n_years) shares summing
    to one across the group in every draw and every year.

    Returned for the WHOLE group at once because the renormalisation needs all
    of them: a share cannot be drawn on its own without deciding what gives way.
    """
    materials = params.materials
    definition = materials.battery_chemistry_scenarios[scenario][group]
    stated = {chemistry: np.array(
        [chemistry_share(params, scenario, group, chemistry, year)
         for year in years], dtype=np.float32)
        for chemistry in definition}

    drawn = {chemistry: values[None, :] * _share_multiplier(
        params, chemistry, years, n_draws, seed)
        for chemistry, values in stated.items()}
    total = sum(drawn.values())
    return {chemistry: np.divide(values, total, out=np.zeros_like(values),
                                 where=total > 0).astype(np.float32)
            for chemistry, values in drawn.items()}
