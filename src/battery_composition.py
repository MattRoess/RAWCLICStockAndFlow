"""
src/battery_composition.py
==========================

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
from functools import lru_cache
from pathlib import Path

import numpy as np


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
    def _anchors(self, chemistry: str, voltage: int) -> tuple:
        """(capacities, elements, masses) with masses (n_anchors, n_draws, n_elements)."""
        found = sorted(self.directory.glob(f"{chemistry}_*kWh_{voltage}V_mass_draws.npy"))
        if not found:
            raise CompositionError(
                f"no mass arrays for {chemistry!r} at {voltage} V in {self.directory}. "
                "Sodium-ion and solid-state have none -- they have no composition of "
                "their own, and a caller must handle that rather than be handed zeros.")
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
            here = (path.parent / path.name.replace("_mass_draws.npy", "_elements.txt")
                    ).read_text().split()
            if elements is None:
                elements = here
            elif here != elements:
                raise CompositionError(
                    f"{chemistry} {voltage}V: element order differs between anchors "
                    f"({capacity} kWh). The arrays cannot be stacked.")
            capacities.append(capacity)
            arrays.append(np.load(path))
        order = np.argsort(capacities)
        return (np.array(capacities)[order], tuple(elements),
                np.stack([arrays[i] for i in order]))

    def elements(self, chemistry: str, voltage: int = 400) -> tuple:
        return self._anchors(chemistry, voltage)[1]

    @lru_cache(maxsize=None)
    def _improvement(self, year: int) -> np.ndarray | None:
        path = self.directory / f"improvement_factor_draws_{int(year)}.npy"
        return np.load(path) if path.exists() else None

    # ------------------------------------------------------- interpolation
    def _at_capacity(self, chemistry: str, voltage: int,
                     capacity_kwh: np.ndarray) -> np.ndarray:
        """
        (n_draws, n_elements) at each draw's own capacity.

        Linear in capacity, between anchors and beyond the top one. Linear
        rather than shape-preserving on purpose: mass runs about a fixed
        intercept plus a constant per kWh -- 125 kg + 4.9 kg/kWh for LFP -- so
        there is almost no curvature to preserve, and a cubic continued past the
        last anchor diverges.
        """
        anchors, _, masses = self._anchors(chemistry, voltage)
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
        rng = np.random.default_rng(seed)
        # One uniform per draw, turned into a symmetric triangular of the right
        # width -- so the same draw is equally optimistic about every element of
        # the same pack rather than each element wandering on its own.
        u = rng.random(capacity.size)
        symmetric = np.where(u < 0.5, np.sqrt(2 * u) - 1.0, 1.0 - np.sqrt(2 * (1 - u)))
        return (1.0 + symmetric * half_width)[:, None]

    # -------------------------------------------------------------- public
    def masses(self, *, chemistry: str, capacity_kwh: np.ndarray,
               voltage_v: np.ndarray, year: int, seed: int = 0) -> np.ndarray:
        """
        Element masses in kg, (n_draws, n_elements), for one chemistry and year.

        `capacity_kwh` and `voltage_v` are per-draw arrays of the same length.
        """
        capacity = np.asarray(capacity_kwh, dtype=float)
        voltage = np.asarray(voltage_v)
        if capacity.shape != voltage.shape:
            raise CompositionError(
                f"capacity and voltage must have one value per draw: "
                f"{capacity.shape} against {voltage.shape}")

        out = None
        for level in np.unique(voltage):
            anchors, _, _ = self._anchors(chemistry, int(level))
            here = voltage == level
            values = self._at_capacity(chemistry, int(level), capacity[here])
            if out is None:
                out = np.zeros((capacity.size, values.shape[1]), dtype=float)
            out[here] = values

        top_anchor = float(self._anchors(chemistry, int(voltage.flat[0]))[0][-1])
        out *= self._extrapolation_factor(capacity, top_anchor, seed)

        improvement = self._improvement(int(year))
        if improvement is not None:
            if improvement.size != capacity.size:
                raise CompositionError(
                    f"the improvement factors for {year} have {improvement.size} "
                    f"draws against {capacity.size} here. Both sides must run at "
                    "the same number of draws for draw i to mean one world.")
            out *= improvement[:, None]
        return out
