"""
src/battery_composition.py
==========================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

What a battery of a drawn capacity, voltage and chemistry is made of, per draw.

    from src.battery_composition import CompositionAtCapacity
    comp = CompositionAtCapacity(params)
    comp.masses(chemistry="battLiFP_subsub", capacity_kwh=cap, voltage_v=volt,
                year=2040)                       # -> (n_draws, n_elements) kg

The composition itself comes from RAWCLICVehicleBattery, which answers "what is
in a battery of capacity X" and deliberately does not answer "what capacity does
a segment carry". This module joins the two: the capacity and voltage arrive as
per-draw arrays from `battery_capacity` and `battery_voltage`, and the element
masses are read off the composition files at each draw's own point.

DRAW i MEETS DRAW i
-------------------
Both sides run at the same number of draws, so draw i's vehicle count, capacity,
voltage and composition are one coherent world. Nothing here averages across
draws; that is the whole reason the battery project persists its draws rather
than percentiles -- the 97.5th percentile of a product is not the product of the
97.5th percentiles.

THREE THINGS VARY PER DRAW, AND EACH IS HANDLED WHERE IT BELONGS
----------------------------------------------------------------
  capacity   interpolated between the files' anchors (25/45/60/80/100 kWh),
             linearly extrapolated above the top one -- with extra uncertainty,
             see `_extrapolation_factor`
  voltage    selects between the 400 V and 800 V arrays, per draw. It is a
             binary state, never blended: blending would give every car a third
             of the copper saving, which no car has
  year       multiplies by the improvement factor draws the battery project
             wrote, which is one scalar per draw shared by every component
"""

from __future__ import annotations

import re
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np


# The two levels the battery project writes, and the files each lives in.
# The elements do NOT add up to the pack: the cell casing and the separator have
# no element rows and the electrolyte's cover 1% of its mass, so 7-11% of a pack
# -- plastics, separator, electrolyte -- exists only at the component level.
LEVEL_FILES = {
    "element": ("_mass_draws.npy", "_elements.txt"),
    "component": ("_component_mass_draws.npy", "_components.txt"),
    # The cross of the two, named "<element>|<component>": the mass of an
    # element WITHIN a component, which neither level on its own can answer and
    # which is what a recovery model needs -- copper in a cable is recovered by
    # a different process from copper in an electrode foil.
    "pair": ("_pair_mass_draws.npy", "_pairs.txt"),
}


class CompositionError(ValueError):
    """Raised when the composition files cannot answer what is being asked."""


class CompositionAtCapacity:
    """The composition files, read once and answerable at any drawn capacity."""

    def __init__(self, params):
        self.params = params
        self.directory = Path(params.materials.battery_composition_dir)
        if not self.directory.is_dir():
            raise CompositionError(
                f"battery composition directory not found: {self.directory}\n"
                "It is written by RAWCLICVehicleBattery's 05_composition.py and is "
                "not tracked in git -- run that project, or point "
                "materials.battery_composition_dir somewhere else.")

    # ------------------------------------------------------------- loading
    @lru_cache(maxsize=None)
    def _anchors(self, chemistry: str, voltage: int, level: str = "element") -> tuple:
        """(capacities, names, masses) with masses (n_anchors, n_draws, n_names)."""
        if level not in LEVEL_FILES:
            raise CompositionError(
                f"unknown level {level!r}; the files carry {sorted(LEVEL_FILES)}.")
        mass_suffix, names_suffix = LEVEL_FILES[level]
        found = sorted(self.directory.glob(
            f"{chemistry}_*kWh_{voltage}V{mass_suffix}"))
        if not found:
            raise CompositionError(
                f"no {level} mass arrays for {chemistry!r} at {voltage} V in "
                f"{self.directory}. "
                "The name must be a file stem the battery project writes "
                "(materials.battery_chemistry_file_names), and that project's "
                "05_composition.py must have been run.")
        capacities, arrays, elements = [], [], None
        for path in found:
            # Matched rather than positioned: chemistry names carry a varying
            # number of underscores (battLiFP_subsub against battLiNMC_highNi),
            # so counting tokens from either end is fragile.
            match = re.search(r"_(\d+(?:\.\d+)?)kWh_", path.name)
            if match is None:
                raise CompositionError(
                    f"cannot read a capacity out of {path.name!r}.")
            capacity = float(match.group(1))
            here = (path.parent / path.name.replace(mass_suffix, names_suffix)
                    ).read_text().split()
            if elements is None:
                elements = here
            elif here != elements:
                raise CompositionError(
                    f"{chemistry} {voltage}V: {level} order differs between anchors "
                    f"({capacity} kWh). The arrays cannot be stacked.")
            capacities.append(capacity)
            arrays.append(np.load(path))
        order = np.argsort(capacities)
        return (np.array(capacities)[order], tuple(elements),
                np.stack([arrays[i] for i in order]))

    def names(self, chemistry: str, level: str = "element",
              voltage: int = 400) -> tuple:
        """The elements, or the components, in the order the arrays carry them."""
        return self._anchors(chemistry, voltage, level)[1]

    def elements(self, chemistry: str, voltage: int = 400) -> tuple:
        return self.names(chemistry, "element", voltage)

    @lru_cache(maxsize=None)
    def _improvement_years(self) -> tuple[int, ...]:
        """The years the battery project wrote an improvement factor for."""
        found = sorted(int(re.search(r"_(\d{4})\.npy$", path.name).group(1))
                       for path in self.directory.glob("improvement_factor_draws_*.npy"))
        if not found:
            raise CompositionError(
                f"no improvement_factor_draws_*.npy in {self.directory}. Run "
                "RAWCLICVehicleBattery's 05_composition.py first -- carrying on "
                "without them would quietly drop the mass improvement over time.")
        return tuple(found)

    @lru_cache(maxsize=None)
    def _improvement(self, year: int) -> np.ndarray:
        """
        The per-draw mass improvement factor for any year, not only the ones on
        file. Those are written every five years; a two-year reporting grid asks
        for the ones in between, and returning nothing for them would silently
        drop the improvement in every second year -- about 20% by 2070.

        Interpolated PER DRAW between the surrounding years, so draw i's factor
        in 2022 lies between draw i's own 2020 and 2025 factors rather than
        between two population averages. Held flat outside the range.
        """
        year = int(year)
        anchors = self._improvement_years()
        path = self.directory / f"improvement_factor_draws_{year}.npy"
        if path.exists():
            return np.load(path)
        if year <= anchors[0]:
            return self._improvement(anchors[0])
        if year >= anchors[-1]:
            return self._improvement(anchors[-1])
        lower = max(a for a in anchors if a < year)
        upper = min(a for a in anchors if a > year)
        weight = (year - lower) / (upper - lower)
        return (self._improvement(lower) * (1.0 - weight)
                + self._improvement(upper) * weight)

    # ------------------------------------------------------- interpolation
    def _at_capacity(self, chemistry: str, voltage: int,
                     capacity_kwh: np.ndarray, level: str = "element") -> np.ndarray:
        """
        (n_draws, n_names) at each draw's own capacity.

        Linear in capacity, between anchors and beyond the top one. Linear
        rather than shape-preserving on purpose: mass runs about a fixed
        intercept plus a constant per kWh -- 125 kg + 4.9 kg/kWh for LFP -- so
        there is almost no curvature to preserve, and a cubic continued past the
        last anchor diverges.
        """
        anchors, _, masses = self._anchors(chemistry, voltage, level)
        capacity = np.asarray(capacity_kwh, dtype=float)

        # Which pair of anchors each draw sits between; the top pair carries
        # every draw above the last anchor, which is the extrapolation.
        upper = np.clip(np.searchsorted(anchors, capacity), 1, len(anchors) - 1)
        lower = upper - 1
        span = anchors[upper] - anchors[lower]
        weight = ((capacity - anchors[lower]) / span)[:, None]

        draw_index = np.arange(capacity.size)
        low = masses[lower, draw_index, :]
        high = masses[upper, draw_index, :]
        return low + (high - low) * weight

    def _extrapolation_factor(self, capacity_kwh: np.ndarray,
                              top_anchor: float, seed: int) -> np.ndarray:
        """
        Extra uncertainty for capacities past the last anchor, (n_draws, 1).

        A triangular centred on 1 whose half-width grows with distance: nothing
        at the anchor, the parameter's full value 100 kWh beyond it. Below the
        anchor it is exactly 1, so an interpolated draw is untouched.
        """
        rate = float(self.params.materials.battery_extrapolation_uncertainty_per_100kwh)
        capacity = np.asarray(capacity_kwh, dtype=float)
        beyond = np.clip(capacity - top_anchor, 0.0, None)
        half_width = rate * beyond / 100.0
        if not np.any(half_width > 0):
            return np.ones((capacity.size, 1))
        # ITS OWN TAG: seeded with the bare seed this was the same stream as the capacity
        # growth and the voltage band (rank correlation +1.0000; found 2026-10-08, fixed
        # 2026-10-09). The tag does not change what is shared ON PURPOSE: every call with
        # the same seed -- both levels, every chemistry -- still gets the same uniform
        # for draw i. See `battery_capacity.growth_draws`.
        rng = np.random.default_rng([seed, zlib.crc32(b"battery_composition.extrapolation")])
        # One uniform per draw, turned into a symmetric triangular of the right
        # width -- so the same draw is equally optimistic about every element of
        # the same pack rather than each element wandering on its own.
        u = rng.random(capacity.size)
        symmetric = np.where(u < 0.5, np.sqrt(2 * u) - 1.0, 1.0 - np.sqrt(2 * (1 - u)))
        return (1.0 + symmetric * half_width)[:, None]

    # -------------------------------------------------------------- public
    def masses(self, *, chemistry: str, capacity_kwh: np.ndarray,
               voltage_v: np.ndarray, year: int, seed: int = 0,
               level: str = "element") -> np.ndarray:
        """
        Masses in kg, (n_draws, n_names), for one chemistry, year and level.

        `capacity_kwh` and `voltage_v` are per-draw arrays of the same length.
        The two levels share the seed, so draw i's extrapolation factor is the
        same number at both -- the elements and the components of one car have
        to be one car.
        """
        capacity = np.asarray(capacity_kwh, dtype=float)
        voltage = np.asarray(voltage_v)
        if capacity.shape != voltage.shape:
            raise CompositionError(
                f"capacity and voltage must have one value per draw: "
                f"{capacity.shape} against {voltage.shape}")

        out = None
        for pack_voltage in np.unique(voltage):
            here = voltage == pack_voltage
            values = self._at_capacity(chemistry, int(pack_voltage),
                                       capacity[here], level)
            if out is None:
                out = np.zeros((capacity.size, values.shape[1]), dtype=float)
            out[here] = values

        top_anchor = float(self._anchors(chemistry, int(voltage.flat[0]), level)[0][-1])
        out *= self._extrapolation_factor(capacity, top_anchor, seed)

        improvement = self._improvement(int(year))
        if improvement.size != capacity.size:
            raise CompositionError(
                f"the improvement factors for {year} have {improvement.size} "
                f"draws against {capacity.size} here. Both sides must run at "
                "the same number of draws for draw i to mean one world.")
        return out * improvement[:, None]
