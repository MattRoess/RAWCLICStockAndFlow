"""
battery_vintage.py
==================

WHERE THE CARS LEAVING THE FLEET WERE BUILT.

A battery scrapped in 2050 was built around 2034 and carries the chemistry and
the pack size of 2034, not of 2050. Stage 04_04 gave it the scrap year's, and
the consequence was measurable: the chemistry mix then multiplied the inflow
and the outflow by exactly the same factor, so it cancelled out of every
outflow-over-inflow ratio and all three scenarios produced one curve.

This spreads each year's outflow back over the build years that could have
produced it:

    weight(build year b -> scrap year t)  =  inflow(b) x f(t - b)

f is the Weibull density of the age at scrapping -- the derivative of the
survival curve stage 02 and 03_02 already use -- and the inflow is THE DRAW'S
OWN inflow history. Draw i's vintages therefore come from draw i's fleet, and
nothing is averaged before the weighting.

WHAT THIS IS AND IS NOT
------------------------
It is not exact cohort accounting. 03_02 reports a year's outflow as one
number, not as a matrix by build year, so the vintage mixture is reconstructed
from the inflow history and the lifetime curve rather than read off one. Two
consequences, both stated rather than hidden:

  - the lifetime SCALE is drawn per draw in 03_02; only its central value
    enters here, so the vintage weights carry less spread than the flows they
    are applied to.
  - the composition files start in 2020. Build years before that are clamped
    onto 2020, which overstates those cars' packs slightly. `clamped_share`
    reports how much weight that touches, so the size of the approximation is
    measured and not assumed.
"""

from __future__ import annotations

import numpy as np


def scrapping_density(ages: np.ndarray, shape_k: float,
                      scale_lambda: float) -> np.ndarray:
    """
    Weibull density of the age at scrapping, zero for ages below zero.

    The survival curve is exp(-(a/lambda)^k); this is its derivative, which is
    the share of a cohort that dies at each age.
    """
    age = np.asarray(ages, dtype=float)
    safe = np.clip(age, 0.0, None)
    density = ((shape_k / scale_lambda) * (safe / scale_lambda) ** (shape_k - 1.0)
               * np.exp(-(safe / scale_lambda) ** shape_k))
    return np.where(age > 0, density, 0.0)


def vintage_weights(inflow: np.ndarray, build_years: np.ndarray,
                    output_years: list[int], vintages: list[int],
                    shape_k: float, scale_lambda: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    (n_draws, n_output_years, n_vintages) and the clamped share per output year.

    Each row over the vintage axis sums to one: it is where the cars leaving in
    that year were built. `inflow` is (n_draws, n_build_years).
    """
    build_years = np.asarray(build_years, dtype=int)
    vintage_array = np.asarray(vintages, dtype=int)
    n_vintages, n_output = len(vintages), len(output_years)

    # Each build year lands on its nearest vintage, and anything earlier than
    # the first one lands on the first -- there is no composition before it.
    clamped = np.clip(build_years, vintage_array[0], vintage_array[-1])
    vintage_index = np.abs(clamped[:, None] - vintage_array[None, :]).argmin(axis=1)
    was_clamped = build_years < vintage_array[0]

    kernel = np.zeros((len(build_years), n_output * n_vintages), dtype=np.float32)
    clamp_kernel = np.zeros((len(build_years), n_output), dtype=np.float32)
    for build_position, build_year in enumerate(build_years):
        for output_position, output_year in enumerate(output_years):
            density = scrapping_density(output_year - build_year, shape_k, scale_lambda)
            if density <= 0:
                continue
            kernel[build_position,
                   output_position * n_vintages + vintage_index[build_position]] += density
            if was_clamped[build_position]:
                clamp_kernel[build_position, output_position] += density

    drawn = np.asarray(inflow, dtype=np.float32)
    weights = (drawn @ kernel).reshape(-1, n_output, n_vintages)
    clamped_mass = drawn @ clamp_kernel

    totals = weights.sum(axis=2, keepdims=True)
    clamped_share = np.divide(clamped_mass, totals[:, :, 0],
                              out=np.zeros_like(clamped_mass),
                              where=totals[:, :, 0] > 0).mean(axis=0)
    np.divide(weights, totals, out=weights, where=totals > 0)
    return weights, clamped_share
