"""
src/battery_capacity.py
=======================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

What a BEV of a given segment carries, in kWh, per Monte Carlo draw and year.

    from src.battery_capacity import capacity_draws
    capacity_draws(params, "JC", years, n_draws=200_000, seed=7)

Stage 04_04 multiplies these against the composition files, which are per
capacity, so this is where the fleet's own knowledge enters the battery
material calculation. The reasoning, the evidence and the alternatives that
were rejected are in `documentation/DESIGN_bev_capacity_for_04_04.md`.

WHAT IS DRAWN, AND AT WHICH LEVEL
----------------------------------
Three things, and which of them is shared matters as much as their values:

  the pack size      ONE LEVEL PER SEGMENT PER DRAW, from that segment's own
                     discrete mixture. Segments are independent of each other:
                     a market where JC happens to sell its big pack says
                     nothing about which pack F sells.

  the growth rate    ONE PER DRAW, SHARED BY EVERY SEGMENT. It is a statement
  the plateau year   about the technology and the market, not about a segment.
                     Drawn per segment they would partly cancel, and a fleet
                     total would come out falsely certain about a trajectory
                     nobody can date.

All three are held ACROSS YEARS within a draw, so a draw is one coherent world
rather than a fresh guess each year. Redrawing per year would make the series
jagged, and any sum over years would average the mixture away -- which is
exactly the collapse that drawing a level was chosen to avoid.

⚠️ THE BAND THIS PRODUCES IS A MIXTURE, NOT AN ERROR. Its width says "a JC is
sold with 65, 70, 80 or 85 kWh and we do not know which one this is", not "we
are unsure how much battery a JC has". Anything reporting the band should say
so, or a reader will take it for composition uncertainty.
"""

from __future__ import annotations

import zlib

import numpy as np


def _triangular(rng: np.random.Generator, band: dict, size: int) -> np.ndarray:
    """A band given as {'min','mode','max'}, drawn. Degenerate bands pass through."""
    low, mode, high = float(band["min"]), float(band["mode"]), float(band["max"])
    if high <= low:                       # a band with no width is a constant
        return np.full(size, mode, dtype=float)
    return rng.triangular(low, mode, high, size=size)


def growth_draws(params, n_draws: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The market-wide trajectory: (rate per decade, plateau year), one per draw.

    Shared by every segment on purpose -- see the module docstring.
    """
    materials = params.materials
    # EVERY STREAM IN THE BATTERY MODULES HAS ITS OWN TAG: it is seeded
    # `[seed, crc32(<its name>), ...]` and never with the bare seed. Two streams seeded
    # alike are ONE stream. This growth rate, the voltage band (`battery_voltage`) and the
    # composition's extrapolation factor were all `default_rng(404)` and had rank
    # correlation +1.0000, so the world in which capacity grows fastest was, exactly,
    # the one in which 800 V arrives earliest (found 2026-10-08, fixed 2026-10-09).
    # Shared by every segment on purpose -- see the module docstring -- so no segment here.
    rng = np.random.default_rng([seed, zlib.crc32(b"battery_capacity.growth")])
    rate = _triangular(rng, materials.battery_capacity_growth_per_decade, n_draws)
    plateau = _triangular(rng, materials.battery_capacity_plateau_year, n_draws)
    return rate, plateau


def level_draws(params, segment: str, n_draws: int, *, seed: int) -> np.ndarray:
    """One pack size per draw, from the segment's discrete mixture."""
    entry = params.materials.battery_capacity_levels.get(segment)
    if entry is None:
        raise KeyError(
            f"no materials.battery_capacity_levels entry for segment {segment!r}. "
            f"Known: {sorted(params.materials.battery_capacity_levels)}")
    levels = np.asarray(entry["levels_kwh"], dtype=float)
    weights = np.asarray(entry["weights"], dtype=float)
    weights = weights / weights.sum()          # renormalised, as the parameter says
    # A stream per segment, spawned from the caller's seed: segments are
    # independent of each other but reproducible from one number.
    #
    # THE SEGMENT ENTERS THROUGH `zlib.crc32`, NEVER `hash()`. Python salts the hash of
    # a `str` per process, so `abs(hash(segment))` drew a different pack size in every
    # run of 04_04 and an export could not be reproduced (found 2026-10-08, fixed
    # 2026-10-09; it takes effect with the next run of 04_04).
    #
    # AND THE STREAM HAS ITS OWN TAG. This and the voltage's adoption order were both
    # `[seed, segment]`, so they were one stream: the uniform that picked a segment's
    # pack size also placed the car in the 800 V adoption order, and a small pack was
    # almost always 800 V and a large one almost never (segment A, 2050: 100 %, 63 %,
    # 0.6 % for 25, 30 and 35 kWh). See `growth_draws` for the rule.
    rng = np.random.default_rng([seed, zlib.crc32(b"battery_capacity.level"),
                                 zlib.crc32(segment.encode())])
    return rng.choice(levels, size=n_draws, p=weights)


def capacity_draws(params, segment: str, years, *, n_draws: int,
                   seed: int) -> np.ndarray:
    """
    Nominal capacity in kWh, shape (n_draws, n_years).

    capacity = level x (1 + rate) ** ((min(year, plateau) - levels_year) / 10)

    The growth runs FORWARD from `battery_capacity_levels_year` until the drawn
    plateau, and BACKWARD before it at the same rate -- which is how a car
    scrapped in 2040 is given the capacity of the year it was built rather than
    of the year it died. That backward arm is an extrapolation: measured
    capacity was 68 kWh in 2018-21 against 82 now, a steeper rise than 10% a
    decade, so the early years here are somewhat too high. Stated rather than
    hidden; see the design note.
    """
    years = np.asarray(years, dtype=float)
    levels = level_draws(params, segment, n_draws, seed=seed)
    rate, plateau = growth_draws(params, n_draws, seed=seed)
    anchor = float(params.materials.battery_capacity_levels_year)

    # (n_draws, n_years): each draw's own plateau caps its own year axis.
    effective = np.minimum(years[None, :], plateau[:, None])
    factor = (1.0 + rate[:, None]) ** ((effective - anchor) / 10.0)
    return levels[:, None] * factor


def summary(params, segments, years, *, n_draws: int = 20_000, seed: int = 0):
    """Mean and 2.5-97.5 band per segment and year -- what the numbers look like."""
    import pandas as pd

    rows = []
    for segment in segments:
        draws = capacity_draws(params, segment, years, n_draws=n_draws, seed=seed)
        for index, year in enumerate(years):
            column = draws[:, index]
            rows.append({
                "segment": segment, "year": int(year),
                "mean_kwh": float(column.mean()),
                "p2.5": float(np.percentile(column, 2.5)),
                "p97.5": float(np.percentile(column, 97.5)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pandas as pd
    from src.params_schema import Params

    params = Params()
    segments = list(params.materials.battery_capacity_levels)
    years = [2020, 2030, 2040, 2050, 2070]
    table = summary(params, segments, years)

    print("BEV capacity per segment and year, kWh -- mean [2.5-97.5 band]\n")
    wide = table.pivot(index="segment", columns="year", values="mean_kwh").round(1)
    band = table.assign(w=table["p97.5"] - table["p2.5"]).pivot(
        index="segment", columns="year", values="w").round(1)
    print("mean:"); print(wide.to_string())
    print("\nband width (p97.5 - p2.5):"); print(band.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
