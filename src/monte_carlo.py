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
    of a params tree, run a function on each, collect results" loop. See
    `02_stockdriven.py`'s Monte Carlo block for a concrete, EV-specific usage example
    (varying `StockFlowParams.lifetime_by_drv["BEV"].scale_lambda` and collecting final
    stock).
    """
    rng = np.random.default_rng(seed)
    results: list[Any] = []
    for _ in range(n_draws):
        params_variant = sample_params(base_params, spec, rng)
        results.append(compute_fn(params_variant))
    return results
