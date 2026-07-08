"""
monte_carlo.py
================

Generic Monte Carlo infrastructure: probability distributions, and a utility to sample
many perturbed variations of a (possibly deeply-nested, frozen-dataclass) parameter
tree without touching any consuming code.

DESIGN GOALS
--------------
- **Product-agnostic**: nothing in this file knows about vehicles, drivetrains, or
  EVmodel specifically. It works for any frozen-dataclass parameter tree (any project's
  `Params`), and any numeric field, whether it lives directly on a dataclass, inside a
  dict of dataclasses (e.g. `StockFlowParams.lifetime_by_drv["BEV"].scale_lambda`), or a
  plain dict of floats (e.g. `StockFlowParams.export_share_by_drv["BEV"]`).
- **Additive, not invasive**: every stage built so far (00-03) needs ZERO changes to
  become "Monte Carlo compatible". Every params field stays exactly what it already is
  (a plain float, a frozen dataclass, a dict) -- Monte Carlo works by constructing many
  DIFFERENT, complete `Params` instances (via `sample_params`) and re-running the
  EXISTING deterministic computation functions once per instance. No stage's code needs
  to know a distribution exists; it always sees a plain, concrete `Params` object, same
  as a single deterministic run.
- **Reuses the pure functions already built for exactly this purpose**: stage 02's
  `run_cohort_survival_model()` was already extracted as a pure function specifically so
  it could be called repeatedly with resampled inputs (see its own docstring) -- this
  module is what actually drives that repeated calling in a structured, reusable way,
  instead of a bespoke loop written fresh each time.

VERIFIED (see HOW_TO_RUN_AND_VERIFY.md for the exact commands):
- Every distribution's `.sample()` produces the expected range/shape.
- `sample_params()` never mutates the base params object (frozen dataclasses make this
  structurally guaranteed, but verified directly with an identity/equality check).
- A real end-to-end Monte Carlo run over `StockFlowParams.lifetime_by_drv["BEV"]
  .scale_lambda` produces a distribution of distinct 2070 EUR BEV stock values, each
  traceable back to a specific, fully-formed `Params` object.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Sequence

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
class Fixed(Distribution):
    """A degenerate 'distribution' that always returns the same value. Lets code treat
    every parameter uniformly (as a Distribution) even when it isn't actually uncertain."""
    value: float

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return np.full(n, self.value, dtype=float)

    def point(self) -> float:
        return self.value


@dataclass(frozen=True)
class Uniform(Distribution):
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Uniform: low ({self.low}) must be <= high ({self.high}).")

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)

    def point(self) -> float:
        return (self.low + self.high) / 2.0


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


@dataclass(frozen=True)
class Normal(Distribution):
    mean: float
    std: float
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(f"Normal: std must be >= 0, got {self.std}.")

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        draws = rng.normal(self.mean, self.std, size=n)
        if self.clip_min is not None or self.clip_max is not None:
            draws = np.clip(draws, self.clip_min, self.clip_max)
        return draws

    def point(self) -> float:
        return self.mean


@dataclass(frozen=True)
class Empirical(Distribution):
    """
    Histogram-based distribution: given bin edges and frequencies (e.g. a 50-bin
    empirical histogram from a composition data source), sample by drawing a bin
    proportional to its frequency, then interpolating UNIFORMLY within that bin's
    range for a continuous value -- standard technique for sampling from a binned
    empirical distribution without inventing a parametric shape for it.

    `bin_edges`: length n+1 (n bin boundaries + 1), monotonically increasing.
    `frequencies`: length n, non-negative (need not already sum to 1 -- normalized here).
    """
    bin_edges: tuple[float, ...]
    frequencies: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.bin_edges) != len(self.frequencies) + 1:
            raise ValueError(
                f"Empirical: bin_edges must have exactly one more entry than "
                f"frequencies (got {len(self.bin_edges)} edges, "
                f"{len(self.frequencies)} frequencies)."
            )
        if any(f < 0 for f in self.frequencies):
            raise ValueError("Empirical: frequencies must be non-negative.")
        if sum(self.frequencies) <= 0:
            raise ValueError("Empirical: frequencies must sum to a positive value.")
        edges = np.asarray(self.bin_edges, dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Empirical: bin_edges must be strictly increasing.")

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        edges = np.asarray(self.bin_edges, dtype=float)
        freqs = np.asarray(self.frequencies, dtype=float)
        probs = freqs / freqs.sum()
        bin_idx = rng.choice(len(freqs), size=n, p=probs)
        low = edges[bin_idx]
        high = edges[bin_idx + 1]
        return rng.uniform(low, high)

    def point(self) -> float:
        """Midpoint of the highest-frequency bin (the histogram's mode bin)."""
        edges = np.asarray(self.bin_edges, dtype=float)
        freqs = np.asarray(self.frequencies, dtype=float)
        i = int(np.argmax(freqs))
        return float((edges[i] + edges[i + 1]) / 2.0)

    @classmethod
    def from_values(cls, values: np.ndarray, bins: int = 50) -> "Empirical":
        """Build an Empirical distribution FROM raw draws/observations by binning them
        -- the inverse of `sample()`, used to turn one stage's raw output into the next
        stage's input distribution."""
        freqs, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
        return cls(bin_edges=tuple(float(e) for e in edges), frequencies=tuple(float(f) for f in freqs))


def as_distribution(x: float | Distribution) -> Distribution:
    """Wrap a plain float as a `Fixed` distribution; pass a real `Distribution` through
    unchanged. Lets calling code accept `float | Distribution` uniformly."""
    return x if isinstance(x, Distribution) else Fixed(float(x))


# ---------------------------------------------------------------------------
# Path-based get/set over a tree of frozen dataclasses and dicts
# ---------------------------------------------------------------------------
# A "path" is a tuple of steps, e.g. ("stock_flow", "lifetime_by_drv", "BEV",
# "scale_lambda") -- each step is either a dataclass field name or a dict key,
# resolved dynamically based on what's actually at that point in the tree. This is what
# makes sample_params() generic: it doesn't need to know Params' shape in advance.

PathStep = str


def get_path(obj: Any, path: Sequence[PathStep]) -> Any:
    """Read the value at `path` inside `obj` (a tree of dataclasses/dicts)."""
    cur = obj
    for step in path:
        if is_dataclass(cur):
            cur = getattr(cur, step)
        elif isinstance(cur, dict):
            cur = cur[step]
        else:
            raise TypeError(
                f"get_path: cannot descend into {type(cur).__name__} at step {step!r} "
                f"(remaining path: {path})."
            )
    return cur


def set_path(obj: Any, path: Sequence[PathStep], value: Any) -> Any:
    """
    Return a NEW object, structurally identical to `obj` except that the value at
    `path` is replaced by `value`. `obj` itself is never mutated -- frozen dataclasses
    make this a structural guarantee for the dataclass segments of the path; dict
    segments are explicitly shallow-copied here to preserve the same guarantee for them.
    """
    if not path:
        return value
    step, rest = path[0], path[1:]
    if is_dataclass(obj):
        current_child = getattr(obj, step)
        new_child = set_path(current_child, rest, value)
        return replace(obj, **{step: new_child})
    if isinstance(obj, dict):
        new_dict = dict(obj)
        new_dict[step] = set_path(obj[step], rest, value)
        return new_dict
    raise TypeError(
        f"set_path: cannot descend into {type(obj).__name__} at step {step!r} "
        f"(remaining path: {path})."
    )


# ---------------------------------------------------------------------------
# Sampling a full params tree
# ---------------------------------------------------------------------------
UncertaintySpec = dict[tuple[PathStep, ...], Distribution]
"""Maps a param path (tuple of steps) to the Distribution describing its uncertainty.
Example: {("stock_flow", "lifetime_by_drv", "BEV", "scale_lambda"): Triangular(11,13,15)}"""


def sample_params(base: Any, spec: UncertaintySpec, rng: np.random.Generator) -> Any:
    """
    Return a NEW params object (same type/shape as `base`) with every path in `spec`
    replaced by one fresh draw from its distribution. `base` is never mutated -- verify
    this yourself with `base_before == base_after` if paranoid; it's structurally
    guaranteed by frozen dataclasses + the copy-on-write `set_path` above, and directly
    tested in `HOW_TO_RUN_AND_VERIFY.md`.
    """
    result = base
    for path, dist in spec.items():
        value = float(as_distribution(dist).sample(rng, n=1)[0])
        result = set_path(result, path, value)
    return result


def run_monte_carlo(
    base_params: Any,
    spec: UncertaintySpec,
    n_draws: int,
    compute_fn: Callable[[Any], Any],
    seed: int | None = None,
) -> list[Any]:
    """
    Draw `n_draws` perturbed copies of `base_params` (per `spec`) and call
    `compute_fn(params_variant)` once per draw, collecting results in a list, in draw
    order.

    `compute_fn` should be a pure function of `params_variant` that returns whatever
    you want to collect a distribution of (a scalar, a DataFrame, a dict -- this
    function doesn't care). It should NOT depend on anything outside `params_variant`
    that also varies between draws (e.g. don't close over mutable global state).

    This function has no EV/vehicle-specific logic -- it's a generic "sample N variants
    of a params tree, run a function on each, collect results" loop.

    PERFORMANCE NOTE: this rebuilds a full `Params` tree (via `dataclasses.replace`)
    on every draw. Fine for a few hundred/thousand draws. At high volume (tens of
    thousands+), where you're only ever changing a handful of leaf scalars, see
    `sample_scalars()` below instead -- it draws plain NumPy arrays up front with no
    per-draw tree reconstruction, which is what actually matters at 200,000 draws.
    """
    rng = np.random.default_rng(seed)
    results: list[Any] = []
    for _ in range(n_draws):
        params_variant = sample_params(base_params, spec, rng)
        results.append(compute_fn(params_variant))
    return results


def sample_scalars(
    spec: dict[str, Distribution],
    n_draws: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Draw `n_draws` values for EACH distribution in `spec` up front, as plain NumPy
    arrays -- no `Params` tree involved at all. This is the performance-critical path
    for high-volume Monte Carlo (tens of thousands of draws+): building and
    `dataclasses.replace`-ing a full nested `Params` object per draw has real,
    unnecessary overhead when you're only ever varying a handful of leaf scalars.

    `spec`: {label -> Distribution}, where `label` is any string you choose to
    identify that parameter later (e.g. "BEV_scale_lambda") -- unlike
    `UncertaintySpec`'s tuple-of-path keys (which double as a literal path into a
    `Params` tree), these labels are just names for your own bookkeeping; nothing
    automatically writes them back into a `Params` object. Combine with
    `sensitivity_correlations()` below using the same labels.

    Returns {label -> array of shape (n_draws,)}. Same `seed` always reproduces the
    same draws for the same spec (each distribution gets its own independently-seeded
    sub-generator, so adding/removing a parameter from `spec` doesn't change the other
    parameters' draws -- unlike drawing everything from one shared `rng` in a fixed
    dict-iteration order, which WOULD shift downstream draws if `spec` ever changes
    size mid-project).
    """
    root_rng = np.random.default_rng(seed)
    # Give every named parameter its own independently-seeded generator (derived from
    # the root seed via SeedSequence.spawn) so results are stable under adding/removing
    # keys from `spec`, not just under reordering them.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(spec))
    out: dict[str, np.ndarray] = {}
    for (label, dist), child_seed in zip(spec.items(), child_seeds):
        rng = np.random.default_rng(child_seed)
        out[label] = as_distribution(dist).sample(rng, n=n_draws)
    return out


# ---------------------------------------------------------------------------
# Summary statistics and histogram persistence
# ---------------------------------------------------------------------------
def summarize_distribution(values: np.ndarray, bins: int = 50) -> dict[str, Any]:
    """
    Standard summary of a Monte Carlo output distribution: mean, median, mode (the
    center of the highest-frequency bin in a `bins`-bin histogram -- there's no single
    universally-agreed "mode" for continuous data, this is the same binning-based
    convention used for `Empirical.point()` above, kept consistent both directions),
    the 95% interval (P2.5, P97.5), plus the histogram itself (`bin_edges`,
    `frequencies`) for storage/reuse as the next stage's `Empirical` input distribution
    (see `Empirical.from_values()`).

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


# ---------------------------------------------------------------------------
# Generic diagnostic plots (no EV-specific logic -- caller saves the figure)
# ---------------------------------------------------------------------------
def plot_distribution(
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 50,
):
    """Generic Monte Carlo output histogram with mean/P2.5/P97.5 markers. Returns
    (fig, ax) -- caller saves it (`fig.savefig(...)`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(values, bins=bins, color="#4a7fb5", alpha=0.85, edgecolor="white")
    mean = values.mean()
    p2_5, p97_5 = np.percentile(values, [2.5, 97.5])
    ax.axvline(mean, color="black", linewidth=1.6, label=f"mean = {mean:.2f}")
    ax.axvline(p2_5, color="black", linewidth=1.0, linestyle="--", label=f"P2.5 = {p2_5:.2f}")
    ax.axvline(p97_5, color="black", linewidth=1.0, linestyle="--", label=f"P97.5 = {p97_5:.2f}")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Draws")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


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