"""
cohort_flow_mc.py
====================
Generic, vectorized Monte Carlo cohort-flow engine.

WHAT THIS MODELS
------------------
A set of independent cohort populations (one per unique combination of
`group_cols`, e.g. Region x Product x Segment), each subject to:
  - Weibull survival/hazard-driven attrition (age-dependent retirement of
    cohort members each period),
  - a FLAT, age-independent split of each period's attrition into N named
    destination flows (e.g. "collected" / "export" / "unknown"), governed by
    per-entity destination shares,
  - optional periodic inflow (new cohort births) added at either the start or
    end of each period ("pre_inflow" / "post_inflow" timing).

Vectorized across a `draws` axis: any of the per-entity parameters (lifetime
scale, destination shares) can be sampled from a distribution instead of held
at a point estimate, giving full Monte Carlo uncertainty propagation without
duplicating the recurrence itself for each new source of uncertainty.

DESIGNED TO BE PRODUCT-AGNOSTIC
----------------------------------
Nothing in this module hardcodes vehicle / drivetrain / EV terminology.
Callers supply:
  - `group_cols`: the DataFrame columns that jointly define ONE independent
    cohort population, e.g. `["Region", "Drive Train", "Segment"]` for
    vehicles, or `["Facility", "Robot Type"]` for a robots pipeline.
  - `entity_key_col`: WHICH SINGLE column in `group_cols` selects the
    per-entity parameter dicts below (e.g. `"Drive Train"`, `"Robot Type"`).
    Every group sharing the same value in this column gets the same
    lifetime/share parameters (and the same per-draw sampled values, when
    `lifetime_scale_lambda_relative_spread_by_entity` etc. are provided) --
    but each group still runs its OWN independent cohort recurrence (own
    inflow history, own starting stock).
  - per-entity parameter dicts, keyed by whatever values that column takes.

This is the ONE place the recurrence, the Weibull vectorization, the
chunking-for-memory logic, and the uncertainty-sampling conventions live.
Product-specific pipeline stages (e.g. `03_02_adjustedflows.py` in
RAWCLICStockAndFlow) should be thin wrappers around this engine -- resolving
their own product-specific parameter names/dicts into the generic shape this
module expects, then calling it. See `flowdriven_model.py`'s
`run_flow_driven_model_monte_carlo` for an example of such a wrapper.

UNCERTAINTY CONVENTION
-------------------------
  - `shape_k` (Weibull shape) is NOT made uncertain by this module -- callers
    that want shape uncertainty would need to add an analogous shape-sampling
    helper; not needed by any pipeline yet.
  - `scale_lambda` (Weibull scale, i.e. characteristic lifetime) varies per
    draw via a RELATIVE spread, sampled from `monte_carlo.Triangular`:
        scale_lambda_draw ~ Triangular(
            point * (1 - lower_spread), point, point * (1 + upper_spread)
        )
    matching this project's established convention (`params_schema.py`'s
    `lifetime_scale_lambda_relative_spread`) -- GENERALIZED to allow
    `lower_spread != upper_spread` (an asymmetric belief, e.g. "10% shorter is
    plausible, but up to 20% longer is also plausible" -- a flat +/-15% would
    misrepresent that either way). Passing a single float still means
    symmetric (`lower_spread = upper_spread = that float`), so every existing
    caller is unaffected. NOTE: an asymmetric Triangular's MEAN is `(low + mode
    + high) / 3`, not `point` -- it shifts toward whichever side has the wider
    spread. This is correct distribution behavior, not a bug: if the belief is
    "more likely to run longer than shorter", the sampled mean SHOULD sit above
    `point`, not be forced back onto it. The SAME per-draw multiplier
    (`scale_lambda_draw / point`) is reapplied if a `lifetime_change_by_entity`
    override changes the point value partway through the horizon, so a draw
    that samples "10% longer-lived" stays 10% longer-lived across the
    override boundary rather than resampling independently.
    `relative_spread` (and any other "*_by_entity" parameter) can be passed
    EITHER as a single scalar/asymmetric-spread (applied uniformly to every
    entity -- a "general" scenario, e.g. "assume 15% lifetime uncertainty
    across the board") OR as a dict keyed by entity (a "specific" scenario,
    e.g. "BEV lifetime is far more uncertain than Diesel's"). See
    `resolve_lifetime_spread`/`as_spread_pair`.
  - Destination shares vary per draw via `monte_carlo.Normal(point, std)`
    clipped to [0, 1] -- matches `params_schema.py`'s `unknown_whereabouts_
    share_std` / `export_share_std` convention. If two shares for the same
    entity would sum to more than 1 for a given draw, both are rescaled down
    proportionally for that draw only (preserves their ratio; only meaningful
    for the 2-share case used today -- generalizing to N shares that must
    jointly sum to <=1 would need a different sampling scheme, e.g. a
    Dirichlet, if a future product needs more than 2 named destination flows).
  - Both distributions are this project's OWN `monte_carlo.Triangular` /
    `monte_carlo.Normal` classes (imported, not reimplemented) -- `monte_carlo.
    py` is itself explicitly product-agnostic, so reusing it here doesn't
    couple this module to anything vehicle-specific, and keeps exactly one
    implementation of each distribution's sampling math in the codebase.
    NOTE: `monte_carlo.sample_scalars()` is NOT used here even though it looks
    like the obvious fit -- it takes a plain `int` seed internally (builds its
    own `np.random.SeedSequence(seed)`), which can't accept an already-spawned
    `SeedSequence` as input. This module needs exactly that: a multi-level
    spawn hierarchy (scenario -> group -> entity, established by callers like
    `03_02_adjustedflows.py`) so every level's draws are independently seeded
    without correlation. `Triangular`/`Normal(...).sample(rng, n=...)` are
    called directly instead, with `rng` built from a spawned child
    `SeedSequence` -- same distribution classes, just composed into the
    existing seed hierarchy rather than through the top-level convenience
    wrapper that assumes it owns the whole seed tree itself.

PER-DRAW INFLOW OVERRIDE (opt-in, off by default)
---------------------------------------------------
Inflow is DETERMINISTIC by default -- the same scalar value from `df`/`inflow_col`
is broadcast identically to every draw, exactly as before this feature existed.
Callers that need a specific (group, year) to genuinely vary per draw instead
(e.g. the vehicle pipeline's future inflow-SEGMENT-mix uncertainty, sampled and
renormalized upstream in `03_02_adjustedflows.py`) pass `inflow_draws_by_group`:
`{group_key: {year: np.ndarray of shape (n_draws,)}}`. Any (group, year) NOT
present in this dict keeps using the plain scalar from `df` -- this is a strict,
backward-compatible ADDITION: passing `None`/`{}` (the default) reproduces prior
output byte-for-byte (verified by regression test). This engine itself does no
sampling, ramping, or renormalizing of its own -- it only accepts a precomputed
per-draw array and uses it in place of the scalar wherever supplied, so it stays
product-agnostic (no "segment," "ramp," or "Triangular" concept here).

MEMORY
------
The cohort-level per-draw state (`stock_prev`, shape (n_draws, n_cohorts)) is
transient per group and never retained across periods or groups. Per-period
aggregated flows (summed over cohorts, shape (n_draws,)) are what get
accumulated into running cumulative totals -- NOT a full
(n_draws, n_cohorts, n_periods) history, which is what causes OOM at scale.
Draws are additionally processed in chunks (`chunk_size`) so peak memory is
bounded by `chunk_size x n_cohorts`, not `n_draws x n_cohorts`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

# Reuses this project's own generic distribution primitives (`Triangular`,
# `Normal`) instead of reimplementing the sampling math here -- `monte_carlo.py`
# is itself explicitly product-agnostic (see its own module docstring), so this
# is "product-agnostic library A depends on product-agnostic library B for its
# distribution primitives", not a coupling to anything vehicle-specific.
import src.monte_carlo as _monte_carlo

# A lifetime relative-spread specification: a single float (symmetric), a
# (lower, upper) tuple/list, any object exposing `.lower`/`.upper` (e.g.
# `params_schema.AsymmetricSpread`), a dict of any of those keyed by entity, or
# None (no uncertainty). Left as `Any`-ish rather than a precise Union since
# the "object with .lower/.upper" branch is intentionally duck-typed -- this
# module deliberately doesn't import params_schema.AsymmetricSpread itself
# (stays product-agnostic). See `resolve_lifetime_spread`/`as_spread_pair`.
_SpreadSpec = Any


def _weibull_survival_vec(shape_k: np.ndarray, scale_lambda: np.ndarray, max_age: int) -> np.ndarray:
    """Vectorized survival curve across draws. shape_k, scale_lambda: (n_draws,). Returns (n_draws, max_age+1)."""
    ages = np.arange(max_age + 1, dtype=float)
    ratio = ages[None, :] / scale_lambda[:, None]
    survival = np.exp(-(ratio ** shape_k[:, None]))
    return np.clip(survival, 0.0, 1.0)


def _weibull_hazard_vec(shape_k: np.ndarray, scale_lambda: np.ndarray, max_age: int) -> np.ndarray:
    """Vectorized hazard curve across draws, shape (n_draws, max_age+1)."""
    survival = _weibull_survival_vec(shape_k, scale_lambda, max_age)
    hazard = np.zeros_like(survival)
    valid = survival[:, :-1] > 0
    ratio = np.zeros_like(survival[:, :-1])
    np.divide(survival[:, 1:], survival[:, :-1], out=ratio, where=valid)
    hazard[:, :-1] = np.where(valid, 1.0 - ratio, 0.0)
    hazard[:, -1] = 1.0
    return np.clip(hazard, 0.0, 1.0)


def _resolve_entity_param(
    value: float | dict[str, float] | None, entity: str, default: float = 0.0
) -> float:
    """
    Resolve a parameter that may be given EITHER as a single scalar (applied
    uniformly to every entity -- a "general" scenario, e.g. "assume 15%
    lifetime uncertainty across the board") OR as a dict keyed by entity (a
    per-entity value -- a "specific" scenario, e.g. "BEV lifetime is much more
    uncertain than Diesel's"). A dict missing some entities falls back to
    `default` for those. This lets callers switch between a quick general-
    uncertainty run and a detailed per-entity one without changing which
    sampling code path runs -- only which value(s) they pass in.

    Used for SYMMETRIC parameters only (e.g. share stds). For the lifetime
    spread, which can be asymmetric, see `resolve_lifetime_spread` below.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        return float(value.get(entity, default))
    return float(value)


def as_spread_pair(value) -> tuple[float, float]:
    """
    [PUBLIC, shared] Promoted from a private (`_`-prefixed) helper to a public one:
    originally only used internally by this module's own `run_cohort_flow_monte_carlo`,
    now ALSO imported directly by `02_stockdriven.py` for stage 02's lifetime sampling
    -- one shared implementation of "how do we turn a spread spec into Triangular
    bounds", rather than stage 02 re-deriving the same symmetric-only formula
    separately (which is what it did before this change, and which silently couldn't
    handle `AsymmetricSpread` -- see fix log).

    Normalize a single spread value into a (lower, upper) pair.

    A symmetric spread was never actually required by the Triangular
    distribution -- `Triangular(low, mode, high)` doesn't need `mode-low ==
    high-mode`. This accepts, by duck-typing (no import of any particular type,
    so callers can pass this project's `params_schema.AsymmetricSpread`, a
    plain `(lower, upper)` tuple/list, or nothing fancier than a single float):
      - an object with `.lower`/`.upper` attributes (e.g. `AsymmetricSpread`)
      - a 2-element tuple/list, read as `(lower, upper)`
      - a single float/int -> symmetric, `(value, value)` (the old behavior,
        still the default and still correct when uncertainty genuinely is
        symmetric)
    """
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return (float(value.lower), float(value.upper))
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"A (lower, upper) spread tuple/list must have exactly 2 elements; got {value!r}."
            )
        return (float(value[0]), float(value[1]))
    return (float(value), float(value))


def resolve_lifetime_spread(
    value, entity: str, default: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    """
    Resolve the lifetime relative-spread specification for one entity into a
    `(lower_spread, upper_spread)` pair. Same general/per-entity dual mode as
    `_resolve_entity_param`, but each resolved value can ALSO be asymmetric
    (see `as_spread_pair`):
      - `None` -> `default` (no uncertainty)
      - a single float `s`, or an asymmetric spread object/tuple, applied
        uniformly to every entity (a "general" scenario)
      - a dict keyed by entity, each value itself either of the above (a
        "specific" scenario) -- entities missing from the dict fall back to
        `default`.
    """
    if value is None:
        return default
    if isinstance(value, dict):
        if entity not in value:
            return default
        return as_spread_pair(value[entity])
    return as_spread_pair(value)


def sample_relative_triangular_scale(
    point: float, lower_spread: float, upper_spread: float, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Sample a positive scale parameter (e.g. Weibull scale_lambda) from
    Triangular(point*(1-lower_spread), point, point*(1+upper_spread)), via
    this project's own `monte_carlo.Triangular` class -- matches the
    established convention for lifetime uncertainty (`params_schema.py`'s
    `lifetime_scale_lambda_relative_spread`), now generalized to ASYMMETRIC
    spreads: `lower_spread` and `upper_spread` need not be equal (e.g. "10%
    shorter-lived is plausible, but up to 20% longer-lived is also plausible"
    is a real, non-symmetric belief a flat +/-15% would misrepresent either
    way). Passing equal values reproduces the old symmetric behavior exactly.

    NOTE on the mean: an asymmetric Triangular's mean is `(low + mode + high)
    / 3`, NOT `mode` -- it shifts toward whichever side has the wider spread.
    This is correct Triangular-distribution behavior, not a bug: if you
    believe "more likely to be longer-lived than shorter-lived", the sampled
    distribgution's mean SHOULD sit above `point`, not be forced back onto it.

    The lower bound is floored at 1% of `point` as a safety net against a
    misconfigured `lower_spread >= 1` producing a non-positive lower bound
    (`Triangular` itself would otherwise raise on low > mode) -- this should
    never bind in practice (spreads are meant to stay well under 1).

    `n_draws == 1` (or both spreads <= 0) short-circuits to an exact point
    mass rather than a single random draw -- this is what lets `n_draws=1`
    reproduce the scalar deterministic reference exactly (see the regression
    tests), which a genuine one-sample-from-Triangular call would not guarantee.
    """
    if (lower_spread <= 0 and upper_spread <= 0) or n_draws == 1:
        return np.full(n_draws, float(point), dtype=float)
    low = max(float(point) * (1.0 - lower_spread), float(point) * 0.01)
    high = float(point) * (1.0 + upper_spread)
    return _monte_carlo.Triangular(low=low, mode=float(point), high=high).sample(rng, n=n_draws)


def normalize_three_shares(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [NEW] Normalizes three non-negative arrays to sum to exactly 1, elementwise.
    Generic three-way analog of the vehicle pipeline's `disaggregation.
    compute_collected_export_unknown_shares` -- deliberately duplicated rather
    than imported, since THIS module is product-agnostic (see module docstring)
    and `disaggregation.py` is vehicle-pipeline-specific; this is the shared
    engine's own version of the same idea; replaces the old two-share-plus-
    remainder approach (`share_a`/`share_b` sampled, third bucket = `1 - a - b`),
    which gave `share_c`/`"collected"` no uncertainty of its own.

    Each input clipped to >= 0 first (guards a stray negative Triangular tail
    draw). If all three are <= 0 for a given entry, that entry is forced to
    `(0, 0, 0)` instead of dividing by zero, with a count printed.
    """
    a = np.clip(np.asarray(a, dtype=float), 0.0, None)
    b = np.clip(np.asarray(b, dtype=float), 0.0, None)
    c = np.clip(np.asarray(c, dtype=float), 0.0, None)
    total = a + b + c
    zero_total = total <= 0.0
    if np.any(zero_total):
        print(
            f"NOTE: {int(np.sum(zero_total))}/{np.size(total)} entries had all three "
            f"shares <= 0 after clipping -- set to (0, 0, 0) instead of dividing by zero."
        )
    safe_total = np.where(zero_total, 1.0, total)
    a_out = np.where(zero_total, 0.0, a / safe_total)
    b_out = np.where(zero_total, 0.0, b / safe_total)
    c_out = np.where(zero_total, 0.0, c / safe_total)
    return a_out, b_out, c_out


def sample_relative_triangular_scale_varying_center(
    rng: np.random.Generator, center_draws: np.ndarray, lower_spread: float, upper_spread: float,
) -> np.ndarray:
    """
    [NEW, promoted from 03_01_flowdriven.py's local `_sample_coupled_residual_
    triangular` -- now shared, since this engine also needs it] Per-draw-VARYING-
    center Triangular sampling. Unlike `sample_relative_triangular_scale` (one
    fixed point shared by every draw), each draw here gets its OWN center
    (`center_draws[i]`), with residual Triangular noise layered around THAT
    value -- used for a share whose central tendency is itself a function of
    another per-draw quantity (e.g. `share_b`'s coupling to that entity's own
    `scale_lambda` draw below), not a single flat point estimate.

    `center_draws` should already be elementwise positive; floored defensively
    at 1% of each draw's own center as a safety net, same convention as
    `sample_relative_triangular_scale`.

    `numpy.random.Generator.triangular(left, mode, right)` broadcasts array
    arguments natively -- one call, not a per-draw Python loop.
    """
    center = np.asarray(center_draws, dtype=float)
    if lower_spread <= 0 and upper_spread <= 0:
        return center.copy()
    low = np.maximum(center * (1.0 - lower_spread), center * 0.01)
    high = center * (1.0 + upper_spread)
    return rng.triangular(low, center, high)


def run_cohort_flow_monte_carlo(
    df: pd.DataFrame,
    years: np.ndarray,
    t_end: int,
    *,
    group_cols: list[str],
    entity_key_col: str,
    lifetime_by_entity: dict[str, dict[str, float]],
    share_a_by_entity: dict[str, float],
    share_b_by_entity: dict[str, float],
    share_c_by_entity: dict[str, float],
    share_a_name: str = "export",
    share_b_name: str = "unknown",
    share_c_name: str = "collected",
    starting_stock_by_cohort_lookup: dict[tuple, dict[int, float]] | None = None,
    n_draws: int = 1,
    # Each value (top-level or per-entity within the dict) may be a plain float
    # (symmetric spread), a (lower, upper) tuple/list, or any object exposing
    # `.lower`/`.upper` (e.g. `params_schema.AsymmetricSpread`) -- see
    # `resolve_lifetime_spread`/`as_spread_pair`.
    lifetime_scale_lambda_relative_spread_by_entity: _SpreadSpec = None,
    # [FIXED, replaces share_a_std_by_entity/share_b_std_by_entity] Was: two
    # independent Normal(point, std, clip 0..1) spreads for share_a/share_b, with
    # share_c ("collected" by default) computed as whatever's left over
    # (`1 - a - b`) -- meaning share_c had NO uncertainty of its own. Now all
    # THREE shares get their own Triangular spread (same `_SpreadSpec` convention
    # as the lifetime spread above), sampled independently, then NORMALIZED per
    # draw so they sum to exactly 1 (see `normalize_three_shares`) -- no
    # privileged remainder share.
    share_a_relative_spread_by_entity: _SpreadSpec = None,
    share_b_relative_spread_by_entity: _SpreadSpec = None,
    share_c_relative_spread_by_entity: _SpreadSpec = None,
    # [NEW] Optional coupling: shifts share_b's CENTER for a given draw based on
    # THAT SAME draw's scale_lambda outcome, before the residual Triangular noise
    # above is layered on top. `{entity: k}` (or a single float applied to every
    # entity, or missing/None -- default 0.0 = no coupling, fully backward
    # compatible with share_b sampled around its own flat point estimate).
    # Formula: `rel_dev = (scale_lambda_draw - scale_lambda_point) /
    # scale_lambda_point; share_b_center = share_b_point * (1 - k * rel_dev)`.
    # This is deliberately NOT product-specific despite the vehicle pipeline's
    # own "shorter lifetime -> more unexplained outflow" motivation for it --
    # any product where one share's magnitude is believed to move with the
    # entity's own lifetime uncertainty can use it, hence living in the generic
    # engine rather than only in `flowdriven_model.py`'s wrapper.
    share_b_lifetime_coupling_k_by_entity: dict[str, float] | float | None = None,
    year_col: str = "year",
    inflow_col: str = "value",
    outflow_timing: str = "post_inflow",
    lifetime_change_by_entity: dict[str, dict[str, float]] | None = None,
    period_inflow_multiplier: dict[int, float] | None = None,
    # [NEW] Opt-in per-draw-varying inflow -- see module docstring's "PER-DRAW INFLOW
    # OVERRIDE" section. `{group_key: {year: np.ndarray shape (n_draws,)}}`. Default
    # None/{} preserves the fully-deterministic-inflow behavior exactly.
    inflow_draws_by_group: dict[tuple, dict[int, np.ndarray]] | None = None,
    output_periods: list[tuple[int, int]] | None = None,
    seed: int | np.random.SeedSequence | None = None,
    chunk_size: int = 20_000,
    collect_per_year: bool = False,
    verbose: bool = True,
    progress_label: str = "",
) -> dict:
    """
    Vectorized, many-draws cohort-flow Monte Carlo engine. See module
    docstring for the full model description and uncertainty conventions.

    Each period's attrition ("out_survival" = total retiring cohort members)
    is split into three named outcomes: `share_a_name` (default "export"),
    `share_b_name` (default "unknown"), and `share_c_name` (default
    "collected") -- ALL THREE explicitly sampled and independently uncertain
    (see `share_a_relative_spread_by_entity`/etc. below), normalized to sum to
    1 per draw via `normalize_three_shares`. [CHANGED] `share_c`/"collected"
    used to be an IMPLICIT remainder (`1 - share_a - share_b`, no uncertainty
    of its own) -- now a first-class sampled input like the other two. This
    matches the 3-way collected/export/unknown-whereabouts split used by the
    vehicle pipeline today; a product with a different residual-flow name
    should still pass its own name via `share_c_name`.

    `period_inflow_multiplier`: optional {year: multiplier} dict applied
    flatly to that year's inflow for every entity/group (e.g. the vehicle
    pipeline's `stock_modifier_2027`, expressed generically as
    `{2027: 0.8, 2028: 0.8, ...}` rather than a single hardcoded year). NOTE:
    this is a SINGLE-YEAR-keyed dict (an existing, unrelated parameter) --
    not to be confused with `output_periods` below, which is a list of
    (start, end) YEAR RANGES.

    `output_periods`: list of `(start_year, end_year)` INCLUSIVE ranges to
    report results for (e.g. `[(2030, 2030), (2030, 2040), (2005, 2070)]` --
    a single year is just a range where start == end). Specified UP FRONT
    (before running), not queryable after the fact -- this is what keeps
    memory bounded at real scale: results for each requested range are
    accumulated DURING the one simulation pass, so nothing resembling a full
    `(n_draws, n_cohorts, n_years)` history is ever materialized. Defaults to
    a single implicit period covering the whole `years` range (`years.min()`
    to `years.max()`) -- this reproduces the pre-`output_periods` behavior
    exactly, including the flat `"cumulative_survival"`/etc. keys described
    below (only present when the whole-horizon period is among the requested
    ones, which it is by default).

    For EACH requested period, four flow metrics (survival/export/unknown/
    collected -- all summed over years in that period) AND three stock
    metrics are reported:
      - `"stock_end_of_period"`: stock at the period's LAST year (a snapshot;
        for a single-year period this is just that year's stock).
      - `"stock_sum_over_period"`: SUM of stock across every year in the
        period ("stock-years", e.g. total vehicle-years in service).
      - `"stock_per_year"`: `{year: array}` for every year in the period --
        the full per-year series, not just start/end.
    `"cumulative_inflow"` is also reported per period -- inflow is DETERMINISTIC
    (not resampled per draw) UNLESS `inflow_draws_by_group` supplies a per-draw
    override for a given (group, year), see module docstring's "PER-DRAW INFLOW
    OVERRIDE" section. In the default (no override) case this is a `(n_draws,)`
    array of an IDENTICAL value repeated -- kept as a full array (not a bare
    float) so `monte_carlo.summarize_distribution()` can be applied uniformly to
    every metric without special-casing (it will report `std=0`, a
    degenerate-but-correct histogram, etc., for this one). When an override IS
    supplied, this array genuinely varies per draw like any other metric.

    Returns:
        {
          "by_group": {group_key: {
              # Present only if the whole-horizon (years.min(), years.max())
              # period is among `output_periods` (true by default):
              "cumulative_survival":  np.ndarray (n_draws,),
              "cumulative_a":         np.ndarray (n_draws,),
              "cumulative_b":         np.ndarray (n_draws,),
              "cumulative_collected": np.ndarray (n_draws,),
              "periods": {
                  (start_year, end_year): {
                      "cumulative_survival":    np.ndarray (n_draws,),
                      "cumulative_a":           np.ndarray (n_draws,),
                      "cumulative_b":           np.ndarray (n_draws,),
                      "cumulative_collected":   np.ndarray (n_draws,),
                      "cumulative_inflow":      np.ndarray (n_draws,),  # deterministic, see above
                      "stock_end_of_period":    np.ndarray (n_draws,),
                      "stock_sum_over_period":  np.ndarray (n_draws,),
                      "stock_per_year": {year: np.ndarray (n_draws,), ...},
                  }, ...
              },
              # only if collect_per_year=True (small n_draws -- validation/plotting):
              "years": np.ndarray (n_years,),
              "per_year_survival":  np.ndarray (n_draws, n_years),
              "per_year_a":         np.ndarray (n_draws, n_years),
              "per_year_b":         np.ndarray (n_draws, n_years),
              "per_year_collected": np.ndarray (n_draws, n_years),
          }, ...},
          "total": {  # summed across all group keys, same shape as one group's result
              "cumulative_survival": np.ndarray (n_draws,), ...,
              "periods": {(start, end): {...}, ...},
          },
          "entity_draws": {entity: {  # the sampled INPUT draws themselves, for
                                       # sensitivity analysis (see monte_carlo.
                                       # sensitivity_correlations) -- one entry per
                                       # entity (not per group; groups sharing an
                                       # entity share these same draws)
              "scale_lambda": np.ndarray (n_draws,),
              "share_a": np.ndarray (n_draws,),
              "share_b": np.ndarray (n_draws,),
              "share_c": np.ndarray (n_draws,),
          }, ...},
        }
    """
    if entity_key_col not in group_cols:
        raise ValueError(f"entity_key_col={entity_key_col!r} must be one of group_cols={group_cols!r}")

    if starting_stock_by_cohort_lookup is None:
        starting_stock_by_cohort_lookup = {}
    if share_a_relative_spread_by_entity is None:
        share_a_relative_spread_by_entity = {}
    if share_b_relative_spread_by_entity is None:
        share_b_relative_spread_by_entity = {}
    if share_c_relative_spread_by_entity is None:
        share_c_relative_spread_by_entity = {}
    if share_b_lifetime_coupling_k_by_entity is None:
        share_b_lifetime_coupling_k_by_entity = {}
    if period_inflow_multiplier is None:
        period_inflow_multiplier = {}
    if inflow_draws_by_group is None:
        inflow_draws_by_group = {}

    entity_idx_in_group = group_cols.index(entity_key_col)

    whole_horizon_period = (int(years.min()), int(years.max()))
    if output_periods is None:
        output_periods = [whole_horizon_period]
    output_periods = [(int(s), int(e)) for s, e in output_periods]
    for s, e in output_periods:
        if s > e:
            raise ValueError(f"output_periods entry ({s}, {e}) has start > end.")
    # De-duplicate while preserving order (a caller might list the same period twice,
    # e.g. once explicitly and once as the implicit default).
    seen = set()
    output_periods = [p for p in output_periods if not (p in seen or seen.add(p))]

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    n_years = len(years)

    by_group: dict[tuple, dict] = {}
    # Per-entity draw streams are cached so every group sharing the same
    # entity value gets the SAME sampled lifetime/share draws (not resampled
    # per group) -- consistent with "one entity, one set of uncertain
    # parameters, applied identically wherever that entity appears".
    entity_draw_cache: dict[str, dict] = {}

    group_keys_all = list(df.groupby(group_cols, dropna=False).groups.keys())
    n_groups = len(group_keys_all)
    label_prefix = f"[{progress_label}] " if progress_label else ""
    if verbose:
        print(
            f"{label_prefix}cohort_flow_mc: {n_groups} group(s) x {n_draws:,} draw(s) x "
            f"{n_years} year(s) -- starting"
        )
    t_start_all = time.time()

    for group_idx, (group_key, sub) in enumerate(df.groupby(group_cols, dropna=False), start=1):
        t_start_group = time.time()
        sub = sub.sort_values(year_col).reset_index(drop=True)

        if isinstance(group_key, tuple):
            lookup_key = group_key
            entity = group_key[entity_idx_in_group]
        else:
            lookup_key = group_key
            entity = group_key

        if entity not in entity_draw_cache:
            shape_k_point = float(lifetime_by_entity[entity]["shape_k"])
            scale_lambda_point = float(lifetime_by_entity[entity]["scale_lambda"])
            lower_spread, upper_spread = resolve_lifetime_spread(
                lifetime_scale_lambda_relative_spread_by_entity, entity, default=(0.0, 0.0)
            )

            entity_seed = ss.spawn(1)[0]
            rng = np.random.default_rng(entity_seed)

            scale_lambda_draws_full = sample_relative_triangular_scale(
                scale_lambda_point, lower_spread, upper_spread, n_draws, rng
            )

            # [FIXED, replaces two independent _sample_clipped_normal calls +
            # implicit "c = 1 - a - b" remainder] share_a and share_c are sampled
            # independently via Triangular, each around its OWN point estimate.
            # share_b is coupled to THIS entity's own scale_lambda draw (see
            # share_b_lifetime_coupling_k_by_entity docstring above) before its
            # own residual Triangular noise is layered on top -- k=0 (the default
            # when not supplied) recovers a flat-centered Triangular exactly like
            # share_a/share_c. All three are then normalized to sum to 1.
            a_point = float(share_a_by_entity[entity])
            a_lower, a_upper = resolve_lifetime_spread(share_a_relative_spread_by_entity, entity, default=(0.0, 0.0))
            a_draws_raw = sample_relative_triangular_scale(a_point, a_lower, a_upper, n_draws, rng)

            c_point = float(share_c_by_entity[entity])
            c_lower, c_upper = resolve_lifetime_spread(share_c_relative_spread_by_entity, entity, default=(0.0, 0.0))
            c_draws_raw = sample_relative_triangular_scale(c_point, c_lower, c_upper, n_draws, rng)

            b_point = float(share_b_by_entity[entity])
            b_lower, b_upper = resolve_lifetime_spread(share_b_relative_spread_by_entity, entity, default=(0.0, 0.0))
            if isinstance(share_b_lifetime_coupling_k_by_entity, dict):
                coupling_k = float(share_b_lifetime_coupling_k_by_entity.get(entity, 0.0))
            else:
                coupling_k = float(share_b_lifetime_coupling_k_by_entity or 0.0)
            rel_dev = (scale_lambda_draws_full - scale_lambda_point) / scale_lambda_point
            b_center_draws = np.clip(b_point * (1.0 - coupling_k * rel_dev), 1e-6, None)
            b_draws_raw = sample_relative_triangular_scale_varying_center(rng, b_center_draws, b_lower, b_upper)

            a_draws_full, b_draws_full, c_draws_full = normalize_three_shares(
                a_draws_raw, b_draws_raw, c_draws_raw
            )

            override = None
            if (
                lifetime_change_by_entity
                and entity in lifetime_change_by_entity
                and lifetime_change_by_entity[entity] is not None
            ):
                ov = lifetime_change_by_entity[entity]
                override_scale_lambda_point = float(ov["scale_lambda"])
                # Same relative multiplier reapplied to the override's point value --
                # see "UNCERTAINTY CONVENTION" in the module docstring.
                rel_multiplier = scale_lambda_draws_full / scale_lambda_point
                override_scale_lambda_draws_full = override_scale_lambda_point * rel_multiplier
                override = {
                    "start_year": int(ov["start_year"]),
                    "shape_k": float(ov["shape_k"]),
                    "scale_lambda_draws_full": override_scale_lambda_draws_full,
                }

            # [REMOVED] Old "rescale a+b if sum > 1" safety clamp -- no longer
            # needed, `normalize_three_shares` above already guarantees all
            # three (not just two) sum to exactly 1 for every draw.

            entity_draw_cache[entity] = {
                "shape_k_point": shape_k_point,
                "scale_lambda_draws_full": scale_lambda_draws_full,
                "a_draws_full": a_draws_full,
                "b_draws_full": b_draws_full,
                "c_draws_full": c_draws_full,
                "override": override,
            }

        ec = entity_draw_cache[entity]
        shape_k_point = ec["shape_k_point"]
        scale_lambda_draws_full = ec["scale_lambda_draws_full"]
        a_draws_full = ec["a_draws_full"]
        b_draws_full = ec["b_draws_full"]
        c_draws_full = ec["c_draws_full"]
        override = ec["override"]

        year_inflow_map = dict(zip(sub[year_col], sub[inflow_col]))
        cohort_stock_map = starting_stock_by_cohort_lookup.get(lookup_key, {})

        # [NEW] Per-draw inflow override for this group, if any -- see module
        # docstring's "PER-DRAW INFLOW OVERRIDE" section. Validated once per group
        # (cheap: at most n_years entries) rather than silently accepting a
        # mis-shaped array, which would otherwise fail confusingly (or not at all,
        # via broadcasting) deep inside the chunked draws loop below.
        group_inflow_draws_raw = inflow_draws_by_group.get(lookup_key, {})
        group_inflow_draws: dict[int, np.ndarray] = {}
        for yr, arr in group_inflow_draws_raw.items():
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n_draws,):
                raise ValueError(
                    f"inflow_draws_by_group[{lookup_key!r}][{yr}] has shape {arr.shape}, "
                    f"expected ({n_draws},) to match n_draws."
                )
            group_inflow_draws[int(yr)] = arr

        # [FIXED, resolves M33 -- same fix as, and cross-validated against,
        # flowdriven_model.py's scalar function; see that fix's comment for the full
        # explanation] `cohort_years` now includes any vintage in `cohort_stock_map`
        # older than `years.min()`, instead of silently dropping that starting stock.
        extra_vintages = sorted(
            int(tau) for tau in cohort_stock_map if int(tau) < int(years.min())
        )
        if extra_vintages:
            cohort_years = np.concatenate([np.array(extra_vintages, dtype=int), years.copy()])
        else:
            cohort_years = years.copy()
        max_age_group = int(t_end - int(cohort_years.min()))

        stock_prev_point = np.array(
            [float(cohort_stock_map.get(int(tau), 0.0)) for tau in cohort_years],
            dtype=float,
        )

        # Per-period cumulative inflow -- an accumulator like every other per-period
        # metric below (NOT precomputed as a single deterministic float up front
        # anymore): in the common case (no `inflow_draws_by_group` override for this
        # group) every draw accumulates the identical scalar each year, so the
        # resulting (n_draws,) array is still uniform -- byte-identical to the old
        # precomputed-scalar-broadcast-to-n_draws behavior (verified by regression
        # test). When an override IS present for a given year, this genuinely varies
        # per draw like any other sampled metric.
        cum_inflow_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}

        cum_survival_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}
        cum_a_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}
        cum_b_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}
        cum_collected_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}
        stock_end_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}
        stock_sum_by_period = {p: np.zeros(n_draws, dtype=float) for p in output_periods}
        stock_per_year_by_period = {
            p: {t: np.zeros(n_draws, dtype=float) for t in years if p[0] <= t <= p[1]}
            for p in output_periods
        }

        if collect_per_year:
            per_year_survival = np.zeros((n_draws, n_years), dtype=float)
            per_year_a = np.zeros((n_draws, n_years), dtype=float)
            per_year_b = np.zeros((n_draws, n_years), dtype=float)
            per_year_collected = np.zeros((n_draws, n_years), dtype=float)

        n_chunks = -(-n_draws // chunk_size)  # ceil division

        for chunk_idx, chunk_start in enumerate(range(0, n_draws, chunk_size), start=1):
            chunk_end = min(chunk_start + chunk_size, n_draws)
            d = chunk_end - chunk_start

            if verbose and n_chunks > 1:
                print(
                    f"{label_prefix}  group {group_idx}/{n_groups} {lookup_key}: "
                    f"chunk {chunk_idx}/{n_chunks} (draws {chunk_start:,}-{chunk_end:,})"
                )

            scale_lambda_draws = scale_lambda_draws_full[chunk_start:chunk_end]
            a_draws = a_draws_full[chunk_start:chunk_end]
            b_draws = b_draws_full[chunk_start:chunk_end]
            c_draws = c_draws_full[chunk_start:chunk_end]
            shape_k_draws = np.full(d, shape_k_point, dtype=float)

            if override is not None:
                override_lam_draws = override["scale_lambda_draws_full"][chunk_start:chunk_end]
                override_k_draws = np.full(d, override["shape_k"], dtype=float)

            stock_prev = np.tile(stock_prev_point, (d, 1))  # (d, n_cohorts)

            for yi, t in enumerate(years):
                if override is not None and t >= override["start_year"]:
                    k_eff = override_k_draws
                    lam_eff = override_lam_draws
                else:
                    k_eff = shape_k_draws
                    lam_eff = scale_lambda_draws

                h = _weibull_hazard_vec(k_eff, lam_eff, max_age_group)  # (d, max_age+1)

                stock_start = stock_prev  # (d, n_cohorts)

                # [NEW] Per-draw-varying inflow when `group_inflow_draws` has an
                # override for this year; otherwise the original scalar, broadcast to
                # every draw in this chunk -- identical to pre-this-feature behavior.
                if t in group_inflow_draws:
                    inflow_t_draws = group_inflow_draws[t][chunk_start:chunk_end]
                else:
                    inflow_t_scalar = float(year_inflow_map.get(t, 0.0))
                    inflow_t_draws = np.full(d, inflow_t_scalar, dtype=float)
                if t in period_inflow_multiplier:
                    inflow_t_draws = inflow_t_draws * float(period_inflow_multiplier[t])
                # [FIXED] `year_inflow_map` is sourced from stage 02's raw, possibly-
                # negative residual (`matrices_by_key`'s flows_df["inflow"] -- see
                # stockflow_model.py's `_run_cohort_recurrence`, which deliberately
                # keeps the raw residual visible for diagnostics near a drivetrain's
                # hard_zero_inflow_from_year cutoff). Stage 02's OWN recurrence always
                # floors this at 0 before adding to cohort state; this engine was
                # missing that same floor, so a negative-residual year got SUBTRACTED
                # from a single vintage cohort's stock here too -- same root cause as
                # the scalar `run_flow_driven_model_with_outflow_disaggregation`,
                # fixed there for the same reason. Floored here, once, before either
                # the stock update below OR `cum_inflow_by_period` accumulation, so
                # both are consistent (unlike the scalar function, this engine has no
                # separate raw-residual diagnostic field to preserve).
                inflow_t_draws = np.maximum(inflow_t_draws, 0.0)

                if outflow_timing == "post_inflow":
                    stock_base = stock_start.copy()
                    newborn_mask = cohort_years == t
                    if newborn_mask.any():
                        stock_base[:, newborn_mask] += inflow_t_draws[:, None]
                    ages_base = (t - cohort_years).astype(int)
                    active = cohort_years <= t
                else:
                    stock_base = stock_start.copy()
                    ages_base = (t - 1 - cohort_years).astype(int)
                    active = (cohort_years <= (t - 1)) & (ages_base >= 0)

                age_idx = np.clip(ages_base, 0, max_age_group)
                h_gathered = h[:, age_idx]  # (d, n_cohorts)
                active_f = active.astype(float)[None, :]  # (1, n_cohorts)

                out_surv = stock_base * h_gathered * active_f
                out_surv = np.minimum(out_surv, stock_base)
                stock_end = stock_base - out_surv

                if outflow_timing != "post_inflow":
                    newborn_mask = cohort_years == t
                    if newborn_mask.any():
                        stock_end[:, newborn_mask] += inflow_t_draws[:, None]

                out_a = out_surv * a_draws[:, None]
                out_b = out_surv * b_draws[:, None]
                out_collected = out_surv * c_draws[:, None]

                out_surv_sum = out_surv.sum(axis=1)
                out_a_sum = out_a.sum(axis=1)
                out_b_sum = out_b.sum(axis=1)
                out_collected_sum = out_collected.sum(axis=1)
                stock_sum_this_year = stock_end.sum(axis=1)

                for p in output_periods:
                    start, end = p
                    if not (start <= t <= end):
                        continue
                    cum_survival_by_period[p][chunk_start:chunk_end] += out_surv_sum
                    cum_a_by_period[p][chunk_start:chunk_end] += out_a_sum
                    cum_b_by_period[p][chunk_start:chunk_end] += out_b_sum
                    cum_collected_by_period[p][chunk_start:chunk_end] += out_collected_sum
                    cum_inflow_by_period[p][chunk_start:chunk_end] += inflow_t_draws
                    stock_sum_by_period[p][chunk_start:chunk_end] += stock_sum_this_year
                    stock_per_year_by_period[p][t][chunk_start:chunk_end] = stock_sum_this_year
                    if t == end:
                        stock_end_by_period[p][chunk_start:chunk_end] = stock_sum_this_year

                if collect_per_year:
                    per_year_survival[chunk_start:chunk_end, yi] = out_surv_sum
                    per_year_a[chunk_start:chunk_end, yi] = out_a_sum
                    per_year_b[chunk_start:chunk_end, yi] = out_b_sum
                    per_year_collected[chunk_start:chunk_end, yi] = out_collected_sum

                stock_prev = stock_end

        group_result = {
            "periods": {
                p: {
                    "cumulative_survival": cum_survival_by_period[p],
                    "cumulative_a": cum_a_by_period[p],
                    "cumulative_b": cum_b_by_period[p],
                    "cumulative_collected": cum_collected_by_period[p],
                    "cumulative_inflow": cum_inflow_by_period[p],
                    "stock_end_of_period": stock_end_by_period[p],
                    "stock_sum_over_period": stock_sum_by_period[p],
                    "stock_per_year": stock_per_year_by_period[p],
                }
                for p in output_periods
            }
        }
        if whole_horizon_period in group_result["periods"]:
            whp = group_result["periods"][whole_horizon_period]
            group_result["cumulative_survival"] = whp["cumulative_survival"]
            group_result["cumulative_a"] = whp["cumulative_a"]
            group_result["cumulative_b"] = whp["cumulative_b"]
            group_result["cumulative_collected"] = whp["cumulative_collected"]

        if collect_per_year:
            group_result.update(
                {
                    "years": years.copy(),
                    "per_year_survival": per_year_survival,
                    "per_year_a": per_year_a,
                    "per_year_b": per_year_b,
                    "per_year_collected": per_year_collected,
                }
            )
        by_group[lookup_key] = group_result

        if verbose:
            dt_group = time.time() - t_start_group
            dt_elapsed = time.time() - t_start_all
            print(
                f"{label_prefix}  group {group_idx}/{n_groups} {lookup_key} done in "
                f"{dt_group:.1f}s (elapsed {dt_elapsed:.1f}s total)"
            )

    per_period_metrics = [
        "cumulative_survival", "cumulative_a", "cumulative_b", "cumulative_collected",
        "cumulative_inflow", "stock_end_of_period", "stock_sum_over_period",
    ]
    total: dict = {"periods": {}}
    for p in output_periods:
        total["periods"][p] = {
            metric: sum(g["periods"][p][metric] for g in by_group.values())
            for metric in per_period_metrics
        }
        # stock_per_year: sum across groups, per year in this period.
        years_in_period = list(next(iter(by_group.values()))["periods"][p]["stock_per_year"].keys()) if by_group else []
        total["periods"][p]["stock_per_year"] = {
            t: sum(g["periods"][p]["stock_per_year"][t] for g in by_group.values())
            for t in years_in_period
        }
    if whole_horizon_period in total["periods"]:
        whp = total["periods"][whole_horizon_period]
        total["cumulative_survival"] = whp["cumulative_survival"]
        total["cumulative_a"] = whp["cumulative_a"]
        total["cumulative_b"] = whp["cumulative_b"]
        total["cumulative_collected"] = whp["cumulative_collected"]

    if verbose:
        print(f"{label_prefix}cohort_flow_mc: all {n_groups} group(s) done in {time.time() - t_start_all:.1f}s")

    # [NEW] Expose the per-entity sampled INPUT draws (scale_lambda, share_a,
    # share_b) -- previously computed internally (cached in `entity_draw_cache`,
    # keyed by entity so groups sharing an entity share draws) but never returned.
    # Needed for sensitivity analysis (correlating each input's draws against an
    # output metric's draws, e.g. via `monte_carlo.sensitivity_correlations()`) --
    # without this, there was no way to ask "which entity's lifetime uncertainty
    # actually drives the output uncertainty" from outside this function.
    entity_draws = {
        entity: {
            "scale_lambda": ec["scale_lambda_draws_full"],
            "share_a": ec["a_draws_full"],
            "share_b": ec["b_draws_full"],
            "share_c": ec["c_draws_full"],
        }
        for entity, ec in entity_draw_cache.items()
    }

    return {
        "by_group": by_group,
        "total": total,
        "share_a_name": share_a_name,
        "share_b_name": share_b_name,
        "share_c_name": share_c_name,
        "entity_draws": entity_draws,
    }