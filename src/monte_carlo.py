"""
monte_carlo.py
================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

What the stages share for a Monte Carlo: the one probability distribution they draw from
(`Triangular`, used by `cohort_flow_mc.py`), the summary of a distribution of results
(`summarize_distribution`, `sum_by_period`), the rank correlations behind the tornado
figures (`sensitivity_correlations`) and the tornado itself (`plot_tornado`).

[2026-10-09] This module used to be a generic toolkit: a `Distribution` family (`Fixed`,
`Uniform`, `Normal`, `Empirical`), a path-based sampler for a whole frozen `Params` tree
(`sample_params`, `set_path`, `get_path`, `run_monte_carlo`, `sample_scalars`) and a
histogram plot. No stage called any of it -- the stages sample their parameters inside
`cohort_flow_mc.py`, which needs only `Triangular` -- and it was removed as dead code.
`git log` has it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
class Distribution:
    """Base class for a scalar probability distribution used to perturb a parameter."""

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        raise NotImplementedError

    def point(self) -> float:
        """A single deterministic representative value (mean/mode) -- used wherever a
        distribution needs to resolve to one number outside a sampling loop (e.g. for
        printing a summary, or as a non-Monte-Carlo fallback)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Triangular(Distribution):
    """Triangular(low, mode, high) -- a common choice when you have an expert's best
    guess (mode) plus a plausible range (low, high), without more detailed data to
    justify a Normal's specific standard deviation."""
    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(
                f"Triangular: require low <= mode <= high, got "
                f"({self.low}, {self.mode}, {self.high})."
            )

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, size=n)

    def point(self) -> float:
        return self.mode


# ---------------------------------------------------------------------------
# Summary statistics and histogram persistence
# ---------------------------------------------------------------------------
def summarize_distribution(values: np.ndarray, bins: int = 50) -> dict[str, Any]:
    """
    Standard summary of a Monte Carlo output distribution: mean, median, mode (the
    center of the highest-frequency bin in a `bins`-bin histogram -- there's no single
    universally-agreed "mode" for continuous data), the 95% interval (P2.5, P97.5),
    plus the histogram itself (`bin_edges`, `frequencies`).

    [FIXED, real crash reproduced and traced to numpy itself] `np.histogram` can raise
    `ValueError: Too many bins for data range. Cannot create N finite-sized bins.` --
    NOT specific to this project's data, this is a float64 precision limit in how
    numpy computes `bins+1` bin edges (see `numpy.lib._histograms_impl._get_bin_
    edges`): it happens whenever the values' MAGNITUDE is large enough (empirically,
    around 1e15 and up) that adding numpy's own zero-range padding doesn't change the
    floating-point value at all, so consecutive bin edges round to the exact same
    float and linspace can't produce `bins+1` strictly-increasing edges. Reproduces
    even for a perfectly constant array at that magnitude -- it is NOT a sign the
    array has "bad" or "wrong" values by itself, just that they're numerically huge.
    In this project's "million vehicles" units, a real value anywhere near 1e15 would
    itself be a strong sign of an upstream unit/aggregation bug -- so rather than
    silently swallowing this, we fall back to a single degenerate bin AND print a
    warning with the actual range/magnitude, so an unexpectedly huge value doesn't
    just vanish into "well the histogram still rendered."
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]  # was np.isnan only -- inf triggers a related,
                                            # differently-worded numpy crash; both are
                                            # "not a real value to histogram," filtered the same way
    if values.size == 0:
        raise ValueError("summarize_distribution: no valid (finite, non-NaN) values to summarize.")

    try:
        freqs, edges = np.histogram(values, bins=bins)
    except ValueError as e:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        magnitude = max(abs(vmin), abs(vmax), 1.0)
        value_range = vmax - vmin
        # Two GENUINELY DIFFERENT causes hit this same numpy error, and they mean
        # opposite things -- conflating them into one message was itself a bug
        # (found after a real run flooded the console with a scary-sounding warning
        # for the harmless case):
        #   (a) HUGE absolute magnitude (empirically >~1e15) -- numpy's own zero-range
        #       padding can't change the float at that scale. Worth investigating in
        #       this project's "million vehicles" units, where nothing should
        #       plausibly approach that.
        #   (b) TINY range relative to a perfectly normal magnitude (e.g. a value
        #       around 500 differing by ~1e-12 across draws) -- this means the
        #       quantity has essentially NO real uncertainty for this particular
        #       period/scope/metric (e.g. the very first simulated year, before much
        #       attrition has occurred, or a segment with near-zero inflow): the true
        #       answer is the same regardless of the draw, and what little "range"
        #       exists is floating-point noise from running the real per-draw
        #       calculation, not a sign anything is wrong. Silent -- printing a loud
        #       warning for what is actually an unremarkable, expected result would
        #       bury the genuinely-worth-investigating case (a) in noise.
        is_implausibly_large = magnitude > 1e6  # generous vs. this project's actual scale
        if is_implausibly_large:
            print(
                f"summarize_distribution: WARNING -- np.histogram could not build {bins} bins "
                f"({e}). Value range=[{vmin:.6g}, {vmax:.6g}] (magnitude ~{magnitude:.3g}). "
                f"Falling back to a single degenerate bin covering the full range. This "
                f"project's values are in 'million vehicles' units -- a magnitude this large "
                f"is a strong signal of an upstream bug (unit mismatch, runaway accumulation, "
                f"etc.), worth checking where this value came from."
            )
        pad = magnitude * 1e-9 if vmax > vmin else max(magnitude * 1e-9, 1e-9)
        edges = np.array([vmin - pad, vmax + pad])
        freqs = np.array([values.size])

    mode_bin = int(np.argmax(freqs))
    mode = float((edges[mode_bin] + edges[mode_bin + 1]) / 2.0)

    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mode": mode,
        "std": float(np.std(values)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "bin_edges": edges.tolist(),
        "frequencies": freqs.tolist(),
    }


def sum_by_period(
    per_year_array: np.ndarray, years: np.ndarray, periods: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Sum a `(n_years, n_draws)` per-year Monte Carlo array over each requested
    `(start_year, end_year)` INCLUSIVE period, returning `{period: (n_draws,) array}`.

    This is the shared building block behind "cumulative output between year X and
    year Y" (a single year is just `start == end`) wherever a model already tracks
    per-year, per-draw totals cheaply (shape `(n_years, n_draws)`, NOT a full
    per-cohort history) -- `stockflow_model.py`'s `inflow_by_year`/
    `out_survival_by_year` are exactly this shape, for instance. Combine with
    `summarize_distribution()` on each returned array to get the full mean/median/
    mode/std/P2.5/P97.5/histogram summary for that period.

    For a quantity that's actually DETERMINISTIC (identical across every draw --
    e.g. stock in a stock-driven model, where stock is pinned to a prescribed target
    regardless of lifetime uncertainty; or inflow in a flow-driven model with no
    inflow uncertainty), pass a `per_year_array` where every draw's column is
    identical (broadcast a `(n_years,)` array to `(n_years, n_draws)` via
    `np.tile(x[:, None], (1, n_draws))` or similar) -- `summarize_distribution()`
    will then correctly report `std=0` and a degenerate single-bin-mass histogram,
    rather than silently mixing a bare scalar into code paths that expect an array.

    `years` may be shorter than the range implied by `periods` (e.g. a synthetic test
    with a narrower simulated horizon than a period request) -- years outside
    `[years.min(), years.max()]` are simply not summed over (no error), so the
    returned sum reflects whatever years were actually simulated within the
    requested window.
    """
    years = np.asarray(years)
    out: dict[tuple[int, int], np.ndarray] = {}
    for start, end in periods:
        mask = (years >= start) & (years <= end)
        out[(int(start), int(end))] = per_year_array[mask, :].sum(axis=0)
    return out


def sensitivity_correlations(
    input_draws: dict[str, np.ndarray],
    output_values: np.ndarray,
) -> "pd.DataFrame":
    """
    Spearman RANK correlation between each sampled input parameter and one output
    metric, across all draws -- the data a tornado plot needs. Rank correlation (not
    Pearson) because the relationship between an input like `scale_lambda` and an
    output like cumulative inflow need not be linear, only monotonic, for rank
    correlation to sensibly capture "this parameter matters."

    Returns a DataFrame with columns [parameter, spearman_r, abs_r], sorted by
    `abs_r` descending -- the most influential parameters first, ready to plot as a
    tornado chart directly.
    """
    import pandas as pd
    from scipy import stats

    output_values = np.asarray(output_values, dtype=float)
    rows = []
    for label, draws in input_draws.items():
        draws = np.asarray(draws, dtype=float)
        r, _p = stats.spearmanr(draws, output_values)
        rows.append({"parameter": label, "spearman_r": float(r), "abs_r": abs(float(r))})

    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


def plot_tornado(sensitivity_df, title: str, top_n: int = 20):
    """Tornado chart from sensitivity_correlations()'s output. Returns (fig, ax)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = sensitivity_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(df))))
    colors = ["#c0392b" if r < 0 else "#2b6cb0" for r in df["spearman_r"]]
    ax.barh(df["parameter"], df["spearman_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Spearman rank correlation with output")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, ax