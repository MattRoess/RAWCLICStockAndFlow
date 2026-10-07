"""
stockflow_model.py
=====================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

THE cohort-survival stock-driven model -- the actual math. Moved here from
02_stockdriven.py so that stage 03 (and any future stage) can import and reuse it
properly, the same way every other pipeline stage already imports `data_prep.py`,
`disaggregation.py`, `stock_flow.py`, `materials.py` as `src/` libraries.

WHY THIS MOVE, this round: stage 03's own Monte Carlo extension needed to regenerate
stage 02's per-year (not just cumulative) outflow uncertainty, using the SAME sampled
lifetime draws stage 02 already produced. Before this move, that math only existed
inside `02_stockdriven.py` -- a pipeline STAGE SCRIPT, not a library -- so stage 03
would have had to either duplicate the cohort-survival implementation a second time
(the exact anti-pattern already corrected once this session) or reach into another
stage's script file, which isn't a clean import relationship. Moving the model itself
here, and leaving `02_stockdriven.py` as a thin orchestration script that imports from
here, fixes the actual structural gap rather than working around it again.

`02_stockdriven.py` and `03_01_flowdriven.py` both import from this module. There is
still exactly ONE implementation of the cohort-survival math in the whole codebase --
its location changed, not its uniqueness.

CONTENTS
---------
- `BackcastState`, `build_backcast_state()` -- pre-t0 cohort age-structure setup
  (thin wrapper around `stock_flow.prepare_backcasting_state`).
- `_run_cohort_recurrence()` -- the ONLY implementation of the actual year-by-year
  cohort-survival recurrence. Always array-shaped internally (`shape_k`/`scale_lambda`
  as arrays of length `n_draws`) -- a normal single run is the `n_draws=1` case.
- `run_cohort_survival_model()` -- thin wrapper, `n_draws=1`, rich per-year DataFrame
  output. Used by a normal deterministic run.
- `run_cohort_survival_monte_carlo()` -- thin wrapper, array-valued `n_draws`,
  cumulative + per-year-total array output (not a full per-cohort history, which
  would need tens of GB at high draw counts). Used by Monte Carlo.

REMOVED (dead code, confirmed unused after the vectorization refactor that introduced
`_run_cohort_recurrence`): `weibull_hazard_lookup`, `get_effective_lifetime_params`,
`apply_negative_inflow_policy` -- all three were scalar, per-year helper functions
from the pre-vectorization implementation; their logic is now inlined (and
array-vectorized) directly inside `_run_cohort_recurrence`. Grepped for call sites
before removing -- none existed outside their own definitions and docstring mentions.

[NEW] `hard_zero_inflow_from_year` -- a hard policy override, threaded through
`_run_cohort_recurrence`/`run_cohort_survival_model`/`run_cohort_survival_monte_carlo`.
NOT a mathematical/statistical correction of the residual-inflow formula's output --
a deterministic if/else tied to a real, external, documented fact (e.g. a legal
ICE-sales ban), applied identically to every draw. `None` (the default everywhere)
is byte-identical to before this parameter existed -- verified via regression test.
See `params_schema.py`'s `StockFlowParams.hard_zero_inflow_from_year_by_drv` for the
full rationale and where the real value (drivetrain, year) is actually configured --
nothing is hardcoded in this file. The pre-override raw residual is never discarded:
it's always tracked separately (`inflow_pre_hard_zero_override_by_year`/`_t`/
`diag_df["inflow_pre_hard_zero_override"]`), whether or not the override is active.

[NEW, step 3 of the agreed plan] `hard_zero_inflow_until_year` -- the MIRROR IMAGE
of `hard_zero_inflow_from_year`: forces inflow to exactly 0 for every year STRICTLY
BEFORE `hard_zero_inflow_until_year`, instead of every year at/after some year. Same
"real, external, documented fact" treatment -- here, a real-world introduction year
(e.g. BEV first sold in Europe in 2011: REMIND's own target_stock trajectory implies
a nonzero BEV inflow before 2011, which is a modeling artifact, not a real fact --
the residual formula has no notion of "this drivetrain didn't exist yet"). Both
parameters can be set independently for the same drivetrain (a "born in year X, phased
out at year Y" window) or independently for different drivetrains -- the two
conditions are combined with OR: `hard_zero_active = (from_year condition) OR
(until_year condition)`. `None` (the default everywhere) is byte-identical to before
this parameter existed. Same "raw residual always tracked separately" guarantee as
`hard_zero_inflow_from_year` -- see `params_schema.py`'s
`StockFlowParams.hard_zero_inflow_until_year_by_drv`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from src.stock_flow import prepare_backcasting_state  # type: ignore
from src.params_schema import LifetimeOverride  # type: ignore


class BackcastState(NamedTuple):
    """Lightweight wrapper around prepare_backcasting_state()'s return, for readability."""
    tau_back: np.ndarray
    stock0_by_cohort: np.ndarray
    h_out_life_age: np.ndarray
    max_age: int


def resolve_inflow_mode_settings(
    drivetrain: str,
    mode_by_drv: dict[str, str],
    default_negative_inflow_policy: str,
    hard_zero_from_by_drv: dict[str, int],
    hard_zero_until_by_drv: dict[str, int],
) -> tuple[str, str, "int | None", "int | None"]:
    """
    Resolve `inflow_mode_by_drv`'s 3-way choice for one drivetrain into the concrete
    (negative_inflow_policy, hard_zero_from, hard_zero_until) settings for PASS 1.
    See `StockFlowParams.inflow_mode_by_drv` for the rationale behind each mode.

    "inflow_phaseout" is deliberately treated identically to "remind_soft" for pass 1
    -- pass 1 exists ONLY to learn every drivetrain's own NATURAL (uncapped) inflow,
    including other phase-out drivetrains, so the cap computation (done afterward, from
    pass-1 results) never depends on which phase-out drivetrain is visited first.

    Lives here rather than in `02_stockdriven.py` because the deterministic loop, that
    stage's Monte Carlo block, AND `03_01_flowdriven.py`'s reconstruction all need the
    identical resolution. Returns
    `(mode, negative_inflow_policy, hard_zero_from_year, hard_zero_until_year)`.
    """
    mode = mode_by_drv.get(drivetrain, "remind_soft")
    if mode == "remind_literal":
        return mode, "clip_to_target", None, None
    return (
        mode,
        default_negative_inflow_policy,
        hard_zero_from_by_drv.get(drivetrain),
        hard_zero_until_by_drv.get(drivetrain),
    )


class Stage02MonteCarlo(NamedTuple):
    """
    The complete result of `run_stage02_cohort_monte_carlo()`.

    `results[drv]` is exactly what `run_cohort_survival_monte_carlo` returned for that
    drivetrain AFTER the phase-out cap pass, i.e. the final simulated outcome -- the
    thing both stage 02 and stage 03_01 must agree on, draw for draw.
    """
    results: dict[str, dict]
    scale_lambda_draws: dict[str, np.ndarray]
    max_share_draws: dict[str, "np.ndarray | None"]
    backcast: dict[str, BackcastState]
    targets: "StockTargetDraws"


class StockTargetDraws(NamedTuple):
    """
    Everything `sample_stock_target_draws()` produces, keyed by drivetrain.

    `stock_target_draws[drv]` is the `(n_years, n_draws)` array to hand straight to
    `run_cohort_survival_monte_carlo(stock_target_draws=...)`. The rest is bookkeeping
    the callers need anyway: the stock series and year axis each drivetrain's array was
    built on, the deterministic (unperturbed) target, and the raw per-drivetrain and
    shared-fleet multipliers for sensitivity reporting.
    """
    stock_target_draws: dict[str, np.ndarray]
    stock_target_mult_draws: dict[str, np.ndarray]
    stock_series: dict[str, pd.Series]
    t_years: dict[str, np.ndarray]
    stock_t_det: dict[str, np.ndarray]
    fleet_mult_draws: np.ndarray | None


def sample_stock_target_draws(
    *,
    stock_dict: dict,
    drivetrains: "list[str] | tuple[str, ...]",
    model_end_year: int,
    n_draws: int,
    seed: int | None,
    stock_target_relative_spread: Any,
    stock_target_uncertainty_start_year: int,
    stock_target_ramp_max_rate_per_year: float,
    correlated_mix: bool,
    total_fleet_relative_spread: Any,
    region: str = "EUR",
) -> StockTargetDraws:
    """
    THE ONLY PLACE THE PER-DRAW STOCK TARGET IS BUILT -- same principle as
    `_run_cohort_recurrence` being the only place the cohort math lives. Both
    `02_stockdriven.py` (which simulates with these targets) and
    `03_01_flowdriven.py` (which RE-DERIVES stage 02's per-year results from the same
    saved `scale_lambda` draws, and so must reconstruct the identical targets) call
    this. Two independent implementations of this sampling would silently disagree,
    which is exactly the bug this function was extracted to remove: 03_01 previously
    omitted `stock_target_draws` from its re-run entirely, so its per-year bands
    carried no stock-target uncertainty at all while stage 02's did.

    WHAT IT BUILDS, per drivetrain:
      1. ONE relative multiplier per Monte Carlo trial, from
         `Triangular(1-lower, 1, 1+upper)` per `stock_target_relative_spread`. Held
         constant across years within a trial (not resampled per year) -- total
         vehicle stock moves smoothly in reality.
      2. That multiplier's deviation from 1.0 ramped in LINEARLY from 0 at
         `stock_target_uncertainty_start_year`, at a rate capped at
         `stock_target_ramp_max_rate_per_year` percentage points/year. Smaller
         deviations therefore complete sooner; every draw shares the same maximum
         RATE, not the same ramp DURATION. At the cutoff year itself every draw's
         multiplier is exactly 1.0, so there is no seam-year discontinuity against
         the deterministic pre-cutoff target.
      3. If `correlated_mix`, a cross-drivetrain renormalization onto the simplex
         plus one shared total-fleet multiplier -- see
         `StockFlowParams.stock_target_correlated_mix` for the full rationale.

    SEEDING -- load-bearing, read before changing. The caller's own lifetime
    (`scale_lambda`) axis takes the FIRST spawn off a `SeedSequence(seed)`; this
    function needs the SECOND (per-drivetrain targets) and the THIRD (shared fleet).
    Rather than have callers pass pre-spawned seeds in the right order -- which is
    exactly the kind of coupling that drifts apart between two call sites -- this
    function rebuilds the sequence from `seed` itself and spawns in that same fixed
    order, discarding the first block. `SeedSequence.spawn` is a pure function of
    (entropy, spawn_key, children-already-spawned), so a fresh `SeedSequence(seed)`
    spawning `n, n, 1` yields byte-identical children to the caller's own object
    doing the same -- the two agree by construction, with no shared mutable state.
    """
    # Imported here rather than at module scope purely to keep this module's import
    # surface unchanged for its existing callers; `cohort_flow_mc` imports only
    # `src.monte_carlo`, so there is no cycle either way.
    from src.cohort_flow_mc import resolve_lifetime_spread, sample_relative_triangular_scale

    drivetrains = list(drivetrains)
    cutoff_year = int(stock_target_uncertainty_start_year)
    max_rate = float(stock_target_ramp_max_rate_per_year)
    if max_rate <= 0:
        raise ValueError(
            f"stock_target_ramp_max_rate_per_year must be > 0, got {max_rate} "
            f"(0 would mean the multiplier never finishes ramping in)."
        )

    seed_seq = np.random.SeedSequence(seed)
    seed_seq.spawn(len(drivetrains))  # block 1: the CALLER's lifetime axis -- not ours
    target_child_seeds = seed_seq.spawn(len(drivetrains))

    out_draws: dict[str, np.ndarray] = {}
    out_mult: dict[str, np.ndarray] = {}
    out_series: dict[str, pd.Series] = {}
    out_years: dict[str, np.ndarray] = {}
    out_det: dict[str, np.ndarray] = {}

    for drivetrain, child_seed in zip(drivetrains, target_child_seeds):
        stock_series = pd.to_numeric(
            stock_dict[(region, drivetrain)]["stock"], errors="coerce"
        ).fillna(0.0)
        stock_series.index = stock_series.index.astype(int)
        stock_series = stock_series.sort_index()

        # Replicates EXACTLY the reindex/ffill/bfill `_run_cohort_recurrence` uses
        # internally to build its own `stock_t` -- deliberately kept in sync, since
        # the draws array has to exist before that function is called.
        t0 = int(stock_series.index.min())
        t_years = np.arange(t0, model_end_year + 1, dtype=int)
        stock_t_det = stock_series.reindex(t_years).ffill().bfill().to_numpy(dtype=float)

        rng = np.random.default_rng(child_seed)
        lower, upper = resolve_lifetime_spread(
            stock_target_relative_spread, drivetrain, default=(0.0, 0.0)
        )
        mult_draws = sample_relative_triangular_scale(1.0, lower, upper, n_draws, rng)

        uncertain_mask = t_years >= cutoff_year
        dev_draws = mult_draws - 1.0
        # `np.where` guards the near-zero-deviation division only: a draw that landed
        # essentially on 1.0 needs no real ramp, and the guard value cannot affect the
        # result because `dev_draws` is ~0 there anyway.
        ramp_years_needed = np.where(
            np.abs(dev_draws) < 1e-12, 1.0, np.abs(dev_draws) / max_rate
        )
        years_since = (t_years[uncertain_mask] - cutoff_year).astype(float)
        ramp_fraction = np.clip(
            years_since[:, None] / ramp_years_needed[None, :], 0.0, 1.0
        )
        effective_mult = 1.0 + dev_draws[None, :] * ramp_fraction

        target_draws = np.tile(stock_t_det[:, None], (1, n_draws))
        target_draws[uncertain_mask, :] = stock_t_det[uncertain_mask, None] * effective_mult

        out_draws[drivetrain] = target_draws
        out_mult[drivetrain] = mult_draws
        out_series[drivetrain] = stock_series
        out_years[drivetrain] = t_years
        out_det[drivetrain] = stock_t_det

    fleet_mult_draws: np.ndarray | None = None
    if correlated_mix:
        # Block 3 of the spawn order documented above. Spawned inside this branch so
        # the uncorrelated path consumes no randomness for it and stays byte-identical
        # to the behavior that predates the correlated mix.
        fleet_rng = np.random.default_rng(seed_seq.spawn(1)[0])
        lower_tf, upper_tf = resolve_lifetime_spread(
            total_fleet_relative_spread, "__total_fleet__", default=(0.0, 0.0)
        )
        fleet_mult_draws = sample_relative_triangular_scale(
            1.0, lower_tf, upper_tf, n_draws, fleet_rng
        )
        fleet_dev = fleet_mult_draws - 1.0
        fleet_ramp_years = np.where(
            np.abs(fleet_dev) < 1e-12, 1.0, np.abs(fleet_dev) / max_rate
        )

        # Drivetrains need not share a year axis (each starts at its own stock series'
        # first year), so the composition is assembled year by year over the UNION of
        # years, matching rows by actual year rather than assuming a common index.
        row_of_year = {
            d: {int(y): i for i, y in enumerate(out_years[d])} for d in drivetrains
        }
        all_years = sorted({int(y) for d in drivetrains for y in out_years[d]})

        for year in all_years:
            # Pre-cutoff years are never visited, which makes the historic period a
            # STRUCTURAL guarantee rather than a float-arithmetic coincidence.
            if year < cutoff_year:
                continue
            present = [d for d in drivetrains if year in row_of_year[d]]
            if not present:
                continue

            total_det = float(sum(out_det[d][row_of_year[d][year]] for d in present))
            total_draw = np.zeros(n_draws, dtype=float)
            for d in present:
                total_draw += out_draws[d][row_of_year[d][year]]

            fleet_fraction = np.clip((year - cutoff_year) / fleet_ramp_years, 0.0, 1.0)
            intended_total = total_det * (1.0 + fleet_dev * fleet_fraction)

            # `total_draw == 0` means every drivetrain's target is zero that year, so
            # there is nothing to redistribute; scale 1.0 leaves the zeros alone rather
            # than producing 0/0. A drivetrain individually zero in a year where others
            # are not stays exactly zero automatically, since the scale is a common
            # multiplicative factor -- BEV before it existed, and Liquids after the
            # phase-out, must never have stock conjured into them by the constraint.
            scale = np.divide(
                intended_total, total_draw,
                out=np.ones_like(total_draw), where=total_draw > 0,
            )
            for d in present:
                out_draws[d][row_of_year[d][year]] *= scale

    return StockTargetDraws(
        stock_target_draws=out_draws,
        stock_target_mult_draws=out_mult,
        stock_series=out_series,
        t_years=out_years,
        stock_t_det=out_det,
        fleet_mult_draws=fleet_mult_draws,
    )


def run_stage02_cohort_monte_carlo(
    *,
    stock_dict: dict,
    drivetrains: "list[str] | tuple[str, ...]",
    model_end_year: int,
    init_max_age: Any,
    n_draws: int,
    seed: int | None,
    lifetime_by_drv: dict,
    lifetime_override_by_drv: dict,
    lifetime_scale_lambda_relative_spread: Any,
    negative_inflow_policy: str,
    hard_zero_inflow_from_year_by_drv: dict,
    hard_zero_inflow_until_year_by_drv: dict,
    inflow_mode_by_drv: dict,
    inflow_phaseout_by_drv: dict,
    inflow_phaseout_max_share_triangular_by_drv: dict,
    stock_target_relative_spread: Any,
    stock_target_uncertainty_start_year: int,
    stock_target_ramp_max_rate_per_year: float,
    stock_target_correlated_mix: bool,
    total_fleet_relative_spread: Any,
    region: str = "EUR",
) -> Stage02MonteCarlo:
    """
    THE ONLY PLACE STAGE 02'S MONTE CARLO COHORT RUN IS ORCHESTRATED. Runs the whole
    thing -- target sampling, lifetime sampling, the natural pass, and the phase-out cap
    pass -- and returns each drivetrain's FINAL per-draw result.

    WHY THIS EXISTS AS ONE FUNCTION. Two stages need these results: `02_stockdriven.py`,
    which summarizes and persists them, and `03_01_flowdriven.py`, which must reproduce
    them exactly in order to layer its own collected/export/unknown split onto the SAME
    draws (stage 02 cannot persist the full per-year-per-draw arrays -- they run to
    hundreds of MB per drivetrain at production draw counts, so 03_01 regenerates them
    instead). While 03_01 rebuilt that call itself, the two drifted apart repeatedly and
    silently, because a re-run that is missing an argument still produces perfectly
    plausible-looking output:

      - the hard-zero overrides were omitted, so Liquids/Hybrid post-2050 and BEV
        pre-2011 diverged;
      - `stock_target_draws` was omitted, so NONE of the stock-target uncertainty was
        present -- measured at up to 20.8% error on per-year inflow;
      - the `inflow_mode_by_drv` resolution and the phase-out cap pass were never
        replicated at all, which was latent only because no drivetrain was configured
        to use them.

    Each was a separate silent divergence with the same root cause: the sequence was
    written down twice. Now it is written down once, and 03_01 gets stage 02's results
    by calling the same function rather than by re-deriving them. Adding a new
    uncertainty axis or policy here reaches both stages automatically -- which is the
    entire point, and the reason not to inline any part of this back into a stage script.

    THE PASSES:
      0. Per-draw stock targets (`sample_stock_target_draws`; includes the correlated
         drivetrain mix and the shared total-fleet axis).
      1. Per-drivetrain lifetime `scale_lambda` and, for phase-out drivetrains, the
         cap's own `max_share`; then the natural (uncapped) cohort run. Phase-out
         drivetrains run exactly like "remind_soft" here, so the cap below never depends
         on visit order.
      2. For each "inflow_phaseout" drivetrain, cap its inflow at
         `other_drivetrains_inflow * max_share / (1 - max_share)` and re-run it with
         that override. Drivetrains needing no cap in any year are left untouched.

    SEEDING: spawn block 1 off `SeedSequence(seed)` is the lifetime axis (and the cap's
    max_share, drawn from the same per-drivetrain generator immediately after
    scale_lambda); blocks 2 and 3 belong to `sample_stock_target_draws`. Both callers
    reach identical draws because this single function owns the whole order.
    """
    from src.cohort_flow_mc import resolve_lifetime_spread, sample_relative_triangular_scale

    drivetrains = list(drivetrains)

    # Block 1 -- this function's lifetime axis. `sample_stock_target_draws` rebuilds the
    # same sequence internally and takes blocks 2 and 3; see its docstring.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(drivetrains))

    targets = sample_stock_target_draws(
        stock_dict=stock_dict,
        drivetrains=drivetrains,
        model_end_year=model_end_year,
        n_draws=n_draws,
        seed=seed,
        stock_target_relative_spread=stock_target_relative_spread,
        stock_target_uncertainty_start_year=stock_target_uncertainty_start_year,
        stock_target_ramp_max_rate_per_year=stock_target_ramp_max_rate_per_year,
        correlated_mix=stock_target_correlated_mix,
        total_fleet_relative_spread=total_fleet_relative_spread,
        region=region,
    )

    results: dict[str, dict] = {}
    scale_lambda_by_drv: dict[str, np.ndarray] = {}
    max_share_by_drv: dict[str, np.ndarray | None] = {}
    backcast_by_drv: dict[str, BackcastState] = {}

    # ---- PASS 1: natural run ----------------------------------------------------
    for drivetrain, child_seed in zip(drivetrains, child_seeds):
        rng = np.random.default_rng(child_seed)
        base = lifetime_by_drv[drivetrain]
        stock_series = targets.stock_series[drivetrain]

        lower_spread, upper_spread = resolve_lifetime_spread(
            lifetime_scale_lambda_relative_spread, drivetrain, default=(0.0, 0.0)
        )
        scale_lambda_draws = sample_relative_triangular_scale(
            base.scale_lambda, lower_spread, upper_spread, n_draws, rng
        )

        backcast = build_backcast_state(
            stock_series=stock_series, model_end_year=model_end_year,
            # Point-estimate backcast, deliberately shared across draws.
            shape_k=base.shape_k, scale_lambda=base.scale_lambda,
            init_max_age=init_max_age,
        )

        mode, pass1_policy, pass1_from, pass1_until = resolve_inflow_mode_settings(
            drivetrain, inflow_mode_by_drv, negative_inflow_policy,
            hard_zero_inflow_from_year_by_drv, hard_zero_inflow_until_year_by_drv,
        )

        # Drawn from the SAME generator immediately after scale_lambda, so the seed
        # reproduces both; a non-phase-out drivetrain consumes no extra randomness at
        # all, keeping the default configuration regression-safe.
        if mode == "inflow_phaseout":
            point_max_share = inflow_phaseout_by_drv[drivetrain][1]
            cap_low, cap_mode, cap_high = inflow_phaseout_max_share_triangular_by_drv.get(
                drivetrain, (point_max_share, point_max_share, point_max_share)
            )
            if cap_mode > 0:
                lower_cap = max(0.0, (cap_mode - cap_low) / cap_mode)
                upper_cap = max(0.0, (cap_high - cap_mode) / cap_mode)
            else:
                lower_cap = upper_cap = 0.0
            max_share_draws = sample_relative_triangular_scale(
                cap_mode, lower_cap, upper_cap, n_draws, rng
            )
        else:
            max_share_draws = None

        results[drivetrain] = run_cohort_survival_monte_carlo(
            stock_series=stock_series, model_end_year=model_end_year, drivetrain=drivetrain,
            shape_k_draws=np.full(n_draws, base.shape_k),
            scale_lambda_draws=scale_lambda_draws,
            lifetime_override=lifetime_override_by_drv.get(drivetrain), backcast=backcast,
            negative_inflow_policy=pass1_policy,
            hard_zero_inflow_from_year=pass1_from,
            hard_zero_inflow_until_year=pass1_until,
            stock_target_draws=targets.stock_target_draws[drivetrain],
        )
        scale_lambda_by_drv[drivetrain] = scale_lambda_draws
        max_share_by_drv[drivetrain] = max_share_draws
        backcast_by_drv[drivetrain] = backcast

    # ---- PASS 2: phase-out cap --------------------------------------------------
    for drivetrain in drivetrains:
        if inflow_mode_by_drv.get(drivetrain, "remind_soft") != "inflow_phaseout":
            continue
        start_year, _point = inflow_phaseout_by_drv[drivetrain]
        hard_zero_year = hard_zero_inflow_from_year_by_drv.get(drivetrain)
        own_result = results[drivetrain]
        own_years = list(own_result["t"])
        # Already floored >= 0 per draw -- the Monte Carlo counterpart of the
        # deterministic loop's `flows_df["inflow"].clip(lower=0.0)`.
        own_inflow = own_result["inflow_applied_by_year"]
        max_share_draws = max_share_by_drv[drivetrain]

        others = [d for d in drivetrains if d != drivetrain]
        override_by_year: dict[int, np.ndarray] = {}
        for i_year, year in enumerate(own_years):
            if year < start_year:
                continue
            if hard_zero_year is not None and year >= hard_zero_year:
                continue
            other_total = np.zeros(n_draws, dtype=float)
            for other in others:
                other_years = list(results[other]["t"])
                if year in other_years:
                    other_total += results[other]["inflow_applied_by_year"][
                        other_years.index(year)
                    ]
            cap = other_total * max_share_draws / (1.0 - max_share_draws)
            natural = own_inflow[i_year]
            needs_cap = natural > cap
            if np.any(needs_cap):
                override_by_year[year] = np.where(needs_cap, cap, natural)

        if override_by_year:
            base = lifetime_by_drv[drivetrain]
            results[drivetrain] = run_cohort_survival_monte_carlo(
                stock_series=targets.stock_series[drivetrain],
                model_end_year=model_end_year, drivetrain=drivetrain,
                shape_k_draws=np.full(n_draws, base.shape_k),
                scale_lambda_draws=scale_lambda_by_drv[drivetrain],
                lifetime_override=lifetime_override_by_drv.get(drivetrain),
                backcast=backcast_by_drv[drivetrain],
                negative_inflow_policy=negative_inflow_policy,
                hard_zero_inflow_from_year=hard_zero_inflow_from_year_by_drv.get(drivetrain),
                hard_zero_inflow_until_year=hard_zero_inflow_until_year_by_drv.get(drivetrain),
                stock_target_draws=targets.stock_target_draws[drivetrain],
                inflow_override_by_year=override_by_year,
            )

    return Stage02MonteCarlo(
        results=results,
        scale_lambda_draws=scale_lambda_by_drv,
        max_share_draws=max_share_by_drv,
        backcast=backcast_by_drv,
        targets=targets,
    )


def build_backcast_state(
    stock_series: pd.Series,
    model_end_year: int,
    shape_k: float,
    scale_lambda: float,
    init_max_age: int = 50,
) -> BackcastState:
    """
    Reconstruct the pre-t0 cohort age structure via `src.stock_flow.prepare_backcasting_state`
    (contract, per that function's usage here): given `stock0` at `t0` and a Weibull
    survival curve, returns the cohort "birth years" tracked (`tau_back`), stock0 split
    by cohort (`stock0_by_cohort`), a precomputed hazard-by-age lookup
    (`h_out_life_age`), and the truncation age (`max_age`).
    """
    t0 = int(stock_series.index.min())
    stock0 = float(stock_series.loc[t0])

    tau_back, stock0_by_cohort, h_out_life_age, max_age = prepare_backcasting_state(
        t0=t0,
        t_end=int(model_end_year),
        stock0=stock0,
        init_max_age=init_max_age,
        shape_k=shape_k,
        scale_lambda=scale_lambda,
    )
    return BackcastState(tau_back, stock0_by_cohort, h_out_life_age, max_age)


def _run_cohort_recurrence(
    stock_series: pd.Series,
    model_end_year: int,
    drivetrain: str,
    shape_k: np.ndarray,
    scale_lambda: np.ndarray,
    lifetime_override: LifetimeOverride | None,
    backcast: BackcastState,
    negative_inflow_policy: str,
    keep_full_history: bool,
    hard_zero_inflow_from_year: int | None = None,
    hard_zero_inflow_until_year: int | None = None,
    # [NEW] Optional per-draw-varying stock TARGET -- shape `(n_t, n_draws)`,
    # aligned 1:1 with `t` (the year axis this function builds internally from
    # `stock_series`/`model_end_year`). When `None` (default), behavior is
    # BYTE-IDENTICAL to before this parameter existed: `target` stays the plain
    # scalar `float(stock_t[i_t])`, shared by every draw. When supplied, `target`
    # becomes `stock_target_draws[i_t]` (a `(n_draws,)` array) instead -- the
    # `inflow_raw = target - remaining_total` line below already works
    # unchanged either way, since numpy broadcasts a scalar against `remaining_
    # total` exactly the same way it broadcasts a same-shape array. Built by the
    # CALLER (`02_stockdriven.py`'s Monte Carlo block), not sampled in here --
    # same "engine takes pre-realized draws, caller does the sampling from
    # params" convention as `shape_k`/`scale_lambda` above.
    stock_target_draws: np.ndarray | None = None,
    # [NEW] Optional per-year forced inflow override -- {year: value}, where
    # `value` is either a plain float (broadcast to every draw) or an
    # `(n_draws,)` array (a genuinely different forced value per draw, e.g. a
    # Monte Carlo phase-out cap that itself varies draw-to-draw). For a year
    # present as a key here, `inflow_applied` is forced to exactly that value
    # instead of whatever `inflow(t) = target(t) - remaining_total(t)` (plus
    # `negative_inflow_policy`) would otherwise have computed. `None` (default)
    # is byte-identical to before this parameter existed. If BOTH this and a
    # hard_zero override apply to the same year, hard_zero wins (0 always beats
    # a computed cap) -- see the precedence check in the loop below. The raw
    # residual is still tracked separately regardless (inflow_pre_hard_zero_
    # override_by_year/_t), same guarantee as the hard_zero overrides.
    inflow_override_by_year: dict[int, float | np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    THE ONLY PLACE THE COHORT-SURVIVAL MATH IS IMPLEMENTED. `shape_k`/`scale_lambda`
    are always arrays here, shape `(n_draws,)` -- a normal single deterministic run is
    simply the `n_draws == 1` case. `run_cohort_survival_model()` (below, single-run,
    rich per-year output) and `run_cohort_survival_monte_carlo()` (below, many-draw,
    cumulative-totals-only output) are both thin wrappers around this function. There
    is exactly one implementation of the model -- a normal run and a Monte Carlo run
    call the identical code, just with a different number of draws and a different
    amount of output detail kept.

    `keep_full_history`: if True, also builds and returns full per-year, per-draw,
    per-cohort arrays (only sensible for a small `n_draws`, e.g. 1, for a normal run).
    If False, only cumulative totals per draw are returned -- what a large-`n_draws`
    Monte Carlo run needs (a full history at 200,000 draws would need tens of GB).

    Lifetime overrides (`lifetime_override_by_drv`, currently `None` for every
    drivetrain as shipped) are supported for BOTH modes: the override itself is not
    treated as uncertain (there's currently no mechanism for that), so when its window
    is active, every draw uses the override's own (shape_k, scale_lambda) for that
    year, replacing whatever that draw's own sampled/point values would have given.

    `hard_zero_inflow_from_year`: [NEW] a hard policy override, NOT a mathematical
    correction -- see `params_schema.py`'s `StockFlowParams.hard_zero_inflow_from_
    year_by_drv` for the full rationale. `None` (default) means no override, byte-
    identical behavior to before this parameter existed. When set, for every year
    `>= hard_zero_inflow_from_year`, `inflow_applied` is forced to exactly 0 --
    overriding whatever `negative_inflow_policy` would otherwise have computed --
    and the cohort simply decays via ordinary Weibull attrition from then on. The
    raw, would-have-been residual is NEVER discarded: it's always tracked separately
    (`inflow_pre_hard_zero_override_by_year`/`_t`), regardless of whether the
    override is active, so nothing is hidden.

    `hard_zero_inflow_until_year`: [NEW, step 3] the mirror image -- when set, for
    every year `< hard_zero_inflow_until_year`, `inflow_applied` is forced to exactly
    0 instead (a drivetrain that had not been introduced yet). Combines with
    `hard_zero_inflow_from_year` via OR if both are set for the same drivetrain. Same
    "raw residual always tracked separately, `None` is byte-identical" guarantees.
    """
    n_draws = int(shape_k.shape[0])
    tau_back = backcast.tau_back
    max_age = backcast.max_age

    t0 = int(stock_series.index.min())
    t = np.arange(t0, model_end_year + 1, dtype=int)
    n_t = len(t)
    stock_t = stock_series.reindex(t).ffill().bfill().to_numpy(dtype=float)

    # [NEW] `stock_target_draws`, if supplied, must align exactly with the (n_t,
    # n_draws) shape this function just derived -- validated once, up front,
    # rather than failing confusingly deep inside the per-year loop below (same
    # "validate the whole array once, cheap" pattern `cohort_flow_mc.py` uses for
    # `inflow_draws_by_group`).
    if stock_target_draws is not None:
        stock_target_draws = np.asarray(stock_target_draws, dtype=float)
        if stock_target_draws.shape != (n_t, n_draws):
            raise ValueError(
                f"stock_target_draws has shape {stock_target_draws.shape}, expected "
                f"({n_t}, {n_draws}) to match (n_years, n_draws) for this run."
            )

    n_cohorts = tau_back.size
    cohort_state = np.tile(backcast.stock0_by_cohort, (n_draws, 1))  # (n_draws, n_cohorts)

    cumulative_inflow = np.zeros(n_draws, dtype=float)
    cumulative_out_survival = np.zeros(n_draws, dtype=float)

    # [NEW] Per-YEAR totals (shape (n_t, n_draws)) -- ALWAYS tracked, regardless of
    # `keep_full_history`. This is cheap (n_t x n_draws floats, e.g. 65 x 200,000 x 8
    # bytes =~ 100MB) even at high draw counts, UNLIKE the full per-COHORT history
    # below (n_t x n_draws x n_cohorts, which would be tens of GB at 200,000 draws --
    # that's what `keep_full_history` actually guards against). Without this, a Monte
    # Carlo run could only ever report ONE cumulative number per drivetrain (the
    # 2070 total) -- not how the uncertainty band evolves year by year, which is what
    # actually lets a flows-over-time chart show uncertainty instead of just an
    # isolated endpoint histogram.
    inflow_by_year = np.zeros((n_t, n_draws), dtype=float)
    out_survival_by_year = np.zeros((n_t, n_draws), dtype=float)
    # [NEW] Per-draw stock TARGET actually used each year -- identical to
    # `stock_t` (broadcast) for every year before `stock_target_draws` was given,
    # or before its uncertainty window starts (the caller is expected to have
    # already filled `stock_target_draws` with the plain point value for years
    # outside its own uncertainty window, so this array is simply whatever
    # `target` resolved to each year, per draw). Same cost class as `inflow_by_
    # year` above. This is what makes it possible to report a genuine per-draw
    # STOCK band for post-uncertainty-start years -- previously stock was only
    # ever `stock_t` (n_t,), deterministic, identical regardless of any draw.
    stock_target_by_year = np.zeros((n_t, n_draws), dtype=float)
    # [NEW] Always tracked, same cost class as inflow_by_year above -- the raw
    # residual BEFORE any hard_zero_inflow_from_year override is applied. Equal to
    # inflow_by_year whenever no override is active (or none is configured for this
    # drivetrain); diverges only in override years, where inflow_by_year becomes 0
    # but this array still shows what the residual formula would have implied.
    inflow_pre_hard_zero_override_by_year = np.zeros((n_t, n_draws), dtype=float)
    # [NEW] The actually-applied per-draw inflow -- always >= 0 by construction
    # (`inflow_applied = max(inflow_raw, 0)` under "report_only", or the
    # clip_to_target/hard_zero equivalent). Distinct from `inflow_by_year` above,
    # which deliberately keeps showing the RAW (possibly negative) residual so a
    # negative-inflow year is never hidden -- this array is for callers that want
    # "what the model actually simulated" (e.g. a Monte Carlo uncertainty band over
    # time, which should never show negative inflow since no draw's simulated inflow
    # ever went negative) rather than the raw diagnostic signal. `cumulative_inflow`
    # already sums exactly this per-year quantity -- this array is what makes that
    # consistent with a per-year breakdown, instead of only being derivable from the
    # (raw-residual) `inflow_by_year`.
    inflow_applied_by_year = np.zeros((n_t, n_draws), dtype=float)

    if negative_inflow_policy not in ("report_only", "clip_to_target"):
        raise ValueError(
            f"negative_inflow_policy={negative_inflow_policy!r} is not one of "
            f"['report_only', 'clip_to_target']."
        )

    if keep_full_history:
        stock_t_tau = np.zeros((n_t, n_draws, n_cohorts), dtype=float)
        stock_t_tau[0] = cohort_state
        outflow_surv_t_tau = np.zeros_like(stock_t_tau)
        inflow_t = np.zeros((n_t, n_draws), dtype=float)
        inflow_pre_hard_zero_override_t = np.zeros((n_t, n_draws), dtype=float)
        inflow_applied_t = np.zeros((n_t, n_draws), dtype=float)
        outflow_surv_t = np.zeros((n_t, n_draws), dtype=float)
        outflow_excess_t = np.zeros((n_t, n_draws), dtype=float)
        outflow_total_t = np.zeros((n_t, n_draws), dtype=float)

    # Precompute the hazard-by-age table ONCE per draw (shape (n_draws, max_age+1)),
    # instead of recomputing exp() every single simulated year -- same idea as the
    # original per-draw hazard cache, generalized across draws at once. Without an
    # active override, every year of the loop below is then just an array-index
    # lookup, not a fresh Weibull evaluation.
    ages_all = np.arange(max_age + 2, dtype=float)
    lam_col = scale_lambda[:, None]
    k_col = shape_k[:, None]
    survival_table = np.exp(-((ages_all[None, :] / lam_col) ** k_col))
    base_hazard_table = np.zeros((n_draws, max_age + 1), dtype=float)
    S_a_all, S_a1_all = survival_table[:, :-1], survival_table[:, 1:]
    nz = S_a_all > 0
    base_hazard_table[nz] = 1.0 - (S_a1_all[nz] / S_a_all[nz])
    base_hazard_table = np.clip(base_hazard_table, 0.0, 1.0)
    base_hazard_table[:, -1] = 1.0  # truncation at max_age

    # [NEW] t0's own target -- the loop below starts at i_t=1 (t0 is the given
    # starting point, not something inflow is solved for), so this year is never
    # touched by the `stock_target_by_year[i_t] = target` assignment inside the
    # loop. Set explicitly here so `stock_target_by_year` reports the REAL known
    # starting stock at every draw for t0, not a stale `0.0` from the initial
    # `np.zeros(...)` allocation. `stock_target_draws[0]` (if supplied) should
    # equal `stock_t[0]` broadcast anyway (t0 is always before any uncertainty
    # window), so this is consistent with the caller's own array either way.
    stock_target_by_year[0] = stock_target_draws[0] if stock_target_draws is not None else stock_t[0]

    for i_t in range(1, n_t):
        year = int(t[i_t])
        prev_year = int(t[i_t - 1])

        ages_prev = (prev_year - tau_back).astype(int)
        valid = (ages_prev >= 0) & (ages_prev <= max_age)
        ages_valid = ages_prev[valid]

        override_active = (
            lifetime_override is not None
            and lifetime_override.start_year <= year <= lifetime_override.end_year
        )
        if override_active:
            # Override replaces every draw's lifetime for this year -- not itself
            # treated as uncertain (see docstring). Computed fresh only for this one
            # year (cheap; overrides are rare/inactive as shipped), not cached, since
            # a bounded window is by design a departure from the base hazard table.
            ages_f = ages_valid.astype(float)
            S_a_o = np.exp(-((ages_f / lifetime_override.scale_lambda) ** lifetime_override.shape_k))
            S_a1_o = np.exp(-(((ages_f + 1) / lifetime_override.scale_lambda) ** lifetime_override.shape_k))
            hz = np.zeros_like(ages_f)
            nz_o = S_a_o > 0
            hz[nz_o] = 1.0 - (S_a1_o[nz_o] / S_a_o[nz_o])
            hz[ages_valid == max_age] = 1.0
            hazard = np.tile(np.clip(hz, 0.0, 1.0), (n_draws, 1))
        else:
            hazard = base_hazard_table[:, ages_valid]

        prev_stock_valid = cohort_state[:, valid]
        out_surv = prev_stock_valid * hazard
        out_surv = np.minimum(out_surv, prev_stock_valid)
        cohort_state[:, valid] = prev_stock_valid - out_surv

        remaining_total = cohort_state.sum(axis=1)
        # [NEW] `target` becomes a per-draw `(n_draws,)` array when `stock_target_
        # draws` was supplied, instead of the plain scalar `float(stock_t[i_t])` --
        # `inflow_raw = target - remaining_total` below is UNCHANGED either way,
        # since numpy broadcasts a scalar against `remaining_total` exactly the
        # same way it broadcasts a same-shape (n_draws,) array. `None` (default)
        # is byte-identical to before this parameter existed.
        if stock_target_draws is not None:
            target = stock_target_draws[i_t]
        else:
            target = float(stock_t[i_t])
        inflow_raw = target - remaining_total
        # inflow_raw < 0: prescribed stock declined faster than natural attrition
        # explains. What happens next is determined entirely by `negative_inflow_policy`
        # -- see apply_negative_inflow_policy() and MATH_MODELS.md §2.3.

        if negative_inflow_policy == "report_only":
            inflow_applied = np.maximum(inflow_raw, 0.0)
            excess_outflow = np.zeros(n_draws, dtype=float)
        else:  # clip_to_target
            inflow_applied = np.maximum(inflow_raw, 0.0)
            need_clip = inflow_raw < 0
            excess = np.maximum(-inflow_raw, 0.0)
            scale = np.ones(n_draws, dtype=float)
            safe = need_clip & (remaining_total > 0)
            scale[safe] = np.clip((remaining_total[safe] - excess[safe]) / remaining_total[safe], 0.0, None)
            cohort_state[safe, :] = cohort_state[safe, :] * scale[safe, None]
            excess_outflow = np.where(safe, remaining_total - cohort_state.sum(axis=1), 0.0)

        # [NEW] Hard-zero policy override -- applied AFTER negative_inflow_policy,
        # overriding whatever that policy computed. Not a mathematical correction of
        # inflow_raw: a deterministic if/else tied to an external fact (e.g. a real
        # ICE-sales ban), same treatment for every draw. See docstring above and
        # params_schema.py's StockFlowParams.hard_zero_inflow_from_year_by_drv for
        # the full rationale. `inflow_raw` itself is untouched either way -- only
        # what gets ADDED to the cohort (inflow_applied) and what gets RECORDED as
        # "the" inflow for this year changes; the raw residual stays separately
        # visible in inflow_pre_hard_zero_override_by_year/_t regardless.
        # [NEW, step 3] hard_zero_inflow_until_year is the mirror image: forces the
        # same zero-inflow override for every year BEFORE a drivetrain's real
        # introduction year, instead of every year at/after some policy year. The two
        # conditions are independent and combined via OR, so a drivetrain could in
        # principle have both (a "born in year X, phased out in year Y" window),
        # though as configured today only one or the other is ever set per drivetrain.
        hard_zero_active = (
            (hard_zero_inflow_from_year is not None and year >= hard_zero_inflow_from_year)
            or (hard_zero_inflow_until_year is not None and year < hard_zero_inflow_until_year)
        )
        if hard_zero_active:
            inflow_applied = np.zeros(n_draws, dtype=float)

        # [NEW] Generic per-year forced-inflow override -- checked AFTER hard_zero
        # so hard_zero always wins if both apply to the same year (defensive; the
        # two are not expected to overlap in practice, but 0 should always beat a
        # computed cap if they ever do). `override_value` may be a plain float
        # (broadcasts to every draw) or an (n_draws,) array (already per-draw).
        override_active = (
            not hard_zero_active
            and inflow_override_by_year is not None
            and year in inflow_override_by_year
        )
        if override_active:
            override_value = np.asarray(inflow_override_by_year[year], dtype=float)
            inflow_applied = np.broadcast_to(override_value, (n_draws,)).astype(float).copy()

        j_new = np.where(tau_back == year)[0]
        if j_new.size != 1:
            raise ValueError(f"Year {year} not found in tau_back range.")
        cohort_state[:, j_new[0]] += inflow_applied
        # NOTE: absent a hard_zero override, `inflow_raw` (not `inflow_applied`) is
        # what gets recorded below -- the raw, possibly-negative value stays visible
        # regardless of negative_inflow_policy, so switching policies never hides
        # that a negative-inflow year occurred. With an ACTIVE hard_zero override,
        # the recorded value becomes `inflow_applied` (== 0) instead, since for that
        # year there is no real residual-inflow question left to report -- inflow is
        # unconditionally 0 by policy. The pre-override raw residual is preserved
        # separately either way (see inflow_pre_hard_zero_override_by_year/_t).

        cumulative_inflow += inflow_applied
        this_year_out_survival = out_surv.sum(axis=1)
        this_year_total_outflow = this_year_out_survival + excess_outflow
        cumulative_out_survival += this_year_total_outflow

        inflow_by_year[i_t] = inflow_applied if (hard_zero_active or override_active) else inflow_raw
        inflow_pre_hard_zero_override_by_year[i_t] = inflow_raw
        inflow_applied_by_year[i_t] = inflow_applied
        out_survival_by_year[i_t] = this_year_total_outflow
        stock_target_by_year[i_t] = target

        if keep_full_history:
            stock_t_tau[i_t] = cohort_state
            year_out_tau = np.zeros((n_draws, n_cohorts), dtype=float)
            year_out_tau[:, valid] = out_surv
            outflow_surv_t_tau[i_t] = year_out_tau
            inflow_t[i_t] = inflow_applied if (hard_zero_active or override_active) else inflow_raw
            inflow_pre_hard_zero_override_t[i_t] = inflow_raw
            inflow_applied_t[i_t] = inflow_applied
            outflow_surv_t[i_t] = this_year_out_survival
            outflow_excess_t[i_t] = excess_outflow
            outflow_total_t[i_t] = this_year_total_outflow

    result: dict[str, Any] = {
        "cumulative_inflow": cumulative_inflow,
        "cumulative_out_survival": cumulative_out_survival,
        "t": t,
        "inflow_by_year": inflow_by_year,
        "inflow_pre_hard_zero_override_by_year": inflow_pre_hard_zero_override_by_year,
        "inflow_applied_by_year": inflow_applied_by_year,
        "out_survival_by_year": out_survival_by_year,
        # [FIXED] `stock_t` (the prescribed target -- deterministic, identical for
        # every draw regardless of lifetime uncertainty; see module docstring in
        # `02_stockdriven.py`: "modeled stock is FORCED to exactly equal the
        # REMIND-prescribed target every year") used to only be included when
        # `keep_full_history=True`, so `run_cohort_survival_monte_carlo` (which always
        # calls with `keep_full_history=False`) had no way to report "cumulative
        # stock" for a period at all. Cheap to always include -- it's a `(n_t,)`
        # array, not per-draw.
        "stock_t": stock_t,
        # [NEW] The per-draw target ACTUALLY used each year -- `(n_t, n_draws)`.
        # Identical to `stock_t` broadcast across every draw when `stock_target_
        # draws` was `None` (the note above about `stock_t` being "deterministic,
        # identical regardless of draw" is UNCHANGED in that case). Genuinely
        # varies per draw for years within `stock_target_draws`'s own uncertainty
        # window when supplied -- this IS the per-draw stock trajectory itself
        # (not just inflow/outflow), letting a caller report a real Monte Carlo
        # stock band for post-uncertainty-start years, something `stock_t` alone
        # could never express.
        "stock_target_by_year": stock_target_by_year,
    }
    if keep_full_history:
        result.update({
            "tau_back": tau_back,
            "stock_t_tau": stock_t_tau, "outflow_surv_t_tau": outflow_surv_t_tau,
            "inflow_t": inflow_t, "inflow_pre_hard_zero_override_t": inflow_pre_hard_zero_override_t,
            "inflow_applied_t": inflow_applied_t,
            "outflow_surv_t": outflow_surv_t,
            "outflow_excess_t": outflow_excess_t, "outflow_total_t": outflow_total_t,
        })
    return result


def run_cohort_survival_model(
    stock_series: pd.Series,
    model_end_year: int,
    drivetrain: str,
    base_shape_k: float,
    base_scale_lambda: float,
    lifetime_override: LifetimeOverride | None,
    backcast: BackcastState,
    negative_inflow_policy: str = "report_only",
    hard_zero_inflow_from_year: int | None = None,
    hard_zero_inflow_until_year: int | None = None,
    # [NEW] Passed straight through to `_run_cohort_recurrence`'s parameter of the
    # same name. `None` (default) preserves byte-identical behavior.
    inflow_override_by_year: dict[int, float | np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Single deterministic run for ONE (region, drivetrain) key -- a thin wrapper
    around `_run_cohort_recurrence()` with `n_draws=1`, unpacking its array outputs
    back into the same per-year DataFrames this function has always returned. No
    caller needs to change; this is a REFACTOR, not a behavior change (verified
    byte-identical against the pre-refactor implementation -- see
    HOW_TO_RUN_AND_VERIFY.md).

    `negative_inflow_policy`: "report_only" (default, original behavior) or
    "clip_to_target" -- see `apply_negative_inflow_policy()` above and
    `MATH_MODELS.md` §2.3 for the exact mechanics of both.

    `hard_zero_inflow_from_year`: [NEW] `None` (default) preserves byte-identical
    behavior. See `_run_cohort_recurrence`'s docstring and `params_schema.py`'s
    `StockFlowParams.hard_zero_inflow_from_year_by_drv` for the full rationale --
    forces `flows_df["inflow"]` to exactly 0 from this year onward, while
    `diag_df["inflow_pre_hard_zero_override"]` preserves what the raw residual
    would have been, for every year, regardless of whether the override is active.

    `hard_zero_inflow_until_year`: [NEW, step 3] the mirror image -- forces
    `flows_df["inflow"]` to exactly 0 for every year BEFORE this one (a drivetrain
    not yet introduced). `None` (default) preserves byte-identical behavior. See
    `params_schema.py`'s `StockFlowParams.hard_zero_inflow_until_year_by_drv`.

    Returns a dict with keys: "stock_t_tau_df", "outflow_surv_df", "flows_df",
    "diag_df", "results_df".
    """
    core = _run_cohort_recurrence(
        stock_series=stock_series, model_end_year=model_end_year, drivetrain=drivetrain,
        shape_k=np.array([base_shape_k], dtype=float), scale_lambda=np.array([base_scale_lambda], dtype=float),
        lifetime_override=lifetime_override, backcast=backcast,
        negative_inflow_policy=negative_inflow_policy, keep_full_history=True,
        hard_zero_inflow_from_year=hard_zero_inflow_from_year,
        hard_zero_inflow_until_year=hard_zero_inflow_until_year,
        inflow_override_by_year=inflow_override_by_year,
    )
    t, tau_back, stock_t = core["t"], core["tau_back"], core["stock_t"]
    stock_t_tau = core["stock_t_tau"][:, 0, :]
    outflow_surv_t_tau = core["outflow_surv_t_tau"][:, 0, :]
    inflow_t = core["inflow_t"][:, 0]
    inflow_pre_hard_zero_override_t = core["inflow_pre_hard_zero_override_t"][:, 0]
    outflow_surv_t = core["outflow_surv_t"][:, 0]
    outflow_excess_t = core["outflow_excess_t"][:, 0]
    outflow_total_t = core["outflow_total_t"][:, 0]

    stock_df = pd.DataFrame({"year": t, "stock_prescribed": stock_t}).set_index("year")

    stock_t_tau_df = pd.DataFrame(stock_t_tau, index=t, columns=tau_back)
    stock_t_tau_df.index.name = "year"
    stock_t_tau_df.columns.name = "cohort_year"

    outflow_surv_df = pd.DataFrame(outflow_surv_t_tau, index=t, columns=tau_back)
    outflow_surv_df.index.name = "year"
    outflow_surv_df.columns.name = "cohort_year"

    flows_df = pd.DataFrame({
        "year": t, "inflow": inflow_t, "out_survival": outflow_surv_t,
        "out_excess": outflow_excess_t, "out_total": outflow_total_t,
    }).set_index("year")

    # diag_df reconstructed from the history arrays -- prev_stock_total(year i) is
    # exactly stock_t_tau's row for year i-1 (the state BEFORE that year's hazard was
    # applied), matching the original per-year dict-building exactly. The first year
    # (t0) never had a diag row in the original either (loop started at i_t=1) -- same
    # here via the `.iloc[1:]` slice below.
    prev_stock_total = np.concatenate([[np.nan], stock_t_tau[:-1].sum(axis=1)])
    diag_df = pd.DataFrame({
        "year": t, "prev_stock_total": prev_stock_total, "target_stock": stock_t,
        "nas": stock_t - prev_stock_total, "out_survival": outflow_surv_t,
        "out_excess": outflow_excess_t, "out_total": outflow_total_t,
        "inflow_residual": inflow_t,
        # [NEW] Always present, regardless of whether hard_zero_inflow_from_year is
        # set for this drivetrain -- equal to "inflow_residual" in every year where
        # no override is active. See module/StockFlowParams docstrings.
        "inflow_pre_hard_zero_override": inflow_pre_hard_zero_override_t,
    }).set_index("year").iloc[1:]

    results_df = stock_df.join(diag_df, how="left")

    return {
        "stock_t_tau_df": stock_t_tau_df,
        "outflow_surv_df": outflow_surv_df,
        "flows_df": flows_df,
        "diag_df": diag_df,
        "results_df": results_df,
    }


def run_cohort_survival_monte_carlo(
    stock_series: pd.Series,
    model_end_year: int,
    drivetrain: str,
    shape_k_draws: np.ndarray,
    scale_lambda_draws: np.ndarray,
    lifetime_override: LifetimeOverride | None,
    backcast: BackcastState,
    negative_inflow_policy: str = "report_only",
    hard_zero_inflow_from_year: int | None = None,
    hard_zero_inflow_until_year: int | None = None,
    # [NEW] Passed straight through to `_run_cohort_recurrence`'s parameter of
    # the same name -- see that function's docstring for the full explanation.
    # `(n_years, n_draws)`, built by the CALLER (`02_stockdriven.py`'s Monte
    # Carlo block) from `StockFlowParams.stock_target_relative_spread` /
    # `.stock_target_uncertainty_start_year`. `None` (default) preserves byte-
    # identical behavior to before this parameter existed -- `target` stays the
    # plain deterministic scalar for every year, every draw.
    stock_target_draws: np.ndarray | None = None,
    # [NEW] Passed straight through to `_run_cohort_recurrence`'s parameter of the
    # same name -- {year: value}, value a plain float (every draw) or an
    # (n_draws,) array (genuinely per-draw, e.g. a Monte Carlo phase-out cap).
    # `None` (default) preserves byte-identical behavior.
    inflow_override_by_year: dict[int, float | np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Many-draw run for ONE (region, drivetrain) key -- a thin wrapper around the SAME
    `_run_cohort_recurrence()` used by `run_cohort_survival_model()` above, just with
    array-valued `shape_k`/`scale_lambda` and `keep_full_history=False` (cumulative
    totals only -- a full per-year history at high draw counts would need tens of GB).

    `backcast` is built ONCE, from a single (point-estimate) lifetime, and shared
    across every draw -- the pre-t0 age structure is not itself resampled per draw.
    This is a documented simplification, not an oversight: it only affects the small
    initial stock at `t0`, not the forward-simulated majority of the horizon.

    `hard_zero_inflow_from_year`: [NEW] `None` (default) preserves byte-identical
    behavior. See `_run_cohort_recurrence`'s docstring for the full rationale --
    applies identically to every draw (the override is not itself uncertain).

    `hard_zero_inflow_until_year`: [NEW, step 3] the mirror image -- forces inflow
    to 0 for every year before this one, instead of after. `None` (default)
    preserves byte-identical behavior. See `params_schema.py`'s
    `StockFlowParams.hard_zero_inflow_until_year_by_drv`.

    `stock_target_draws`: [NEW] `None` (default) preserves byte-identical behavior
    -- the REMIND stock target stays deterministic, identical for every draw, same
    as before this parameter existed. When supplied, the target itself becomes
    per-draw-varying for whichever years the caller filled with something other
    than the plain point value (see `StockFlowParams.stock_target_uncertainty_
    start_year` -- typically every year before it stays the deterministic value,
    every year from it onward gets a per-draw multiplier). This means modeled
    STOCK, not just inflow/outflow, can now genuinely vary per draw for those
    years -- see `stock_target_by_year` in the return value below, and
    `02_stockdriven.py`'s module docstring for how this changes that stage's
    previously-unconditional "modeled stock always exactly equals target"
    guarantee (still exactly true before this parameter's uncertainty window,
    no longer true within it).

    Returns {"cumulative_inflow": (n_draws,), "cumulative_out_survival": (n_draws,),
    "t": (n_years,), "inflow_by_year": (n_years, n_draws),
    "inflow_pre_hard_zero_override_by_year": (n_years, n_draws) -- [NEW] the raw
    residual before any override, always present,
    "inflow_applied_by_year": (n_years, n_draws) -- [NEW] the actually-simulated
    per-draw inflow, always >= 0 (this is what "cumulative_inflow" sums over years).
    Prefer THIS over "inflow_by_year" for anything that should never show a
    negative value (e.g. a per-year Monte Carlo uncertainty band) -- "inflow_by_year"
    deliberately still shows the raw, possibly-negative residual in years where no
    hard_zero override is active, so a negative-inflow year is never silently hidden
    from diagnostics; it is NOT the same thing as "what the model actually did".
    "out_survival_by_year": (n_years, n_draws), "stock_t": (n_years,) -- the
    prescribed target, deterministic, identical regardless of draw, "stock_target_
    by_year": (n_years, n_draws) -- [NEW] the target ACTUALLY used each year, per
    draw; identical to "stock_t" broadcast when `stock_target_draws` was `None`}.
    See `monte_carlo.sum_by_period()` for turning "inflow_applied_by_year"/
    "out_survival_by_year" into cumulative sums over an arbitrary (start_year,
    end_year) window without needing a full per-cohort history.
    """
    return _run_cohort_recurrence(
        stock_series=stock_series, model_end_year=model_end_year, drivetrain=drivetrain,
        shape_k=np.asarray(shape_k_draws, dtype=float), scale_lambda=np.asarray(scale_lambda_draws, dtype=float),
        lifetime_override=lifetime_override, backcast=backcast,
        negative_inflow_policy=negative_inflow_policy, keep_full_history=False,
        hard_zero_inflow_from_year=hard_zero_inflow_from_year,
        hard_zero_inflow_until_year=hard_zero_inflow_until_year,
        stock_target_draws=stock_target_draws,
        inflow_override_by_year=inflow_override_by_year,
    )


def build_inflow_draws_by_drivetrain(
    *,
    stock_dict: dict,
    params,
    parent_by_drv: "tuple[tuple[str, str], ...]",
    years: "np.ndarray",
) -> "dict[str, np.ndarray]":
    """
    Stage 02's per-draw INFLOW, in absolute vehicles, ready for a later stage to use.

    WHY THIS EXISTS. Stage 02 samples how large the fleet is, and therefore how many
    vehicles are bought each year. Stage 03_02 never saw any of it: the artifact
    between them holds one number per row and has no draw dimension, so total BEV
    inflow arrived with a coefficient of variation of 0.000001% against stage 02's
    9.6%. Every downstream inflow band was too narrow as a result. See
    `documentation/DESIGN_inflow_uncertainty_propagation.md` for the full record.

    WHY ABSOLUTE VALUES AND NOT A RATIO. Inflow is not sampled; it is a residual,
    `stock target - survivors`, and while a drivetrain is phased out that residual
    approaches and crosses zero. Its RELATIVE spread then diverges -- Liquids reaches
    a CV of 1775% in 2040 on an absolute spread of 0.005 million vehicles, and the
    downstream table carries genuinely negative values (Liquids is -4.655 in 2040).
    A multiplier is meaningless there. Absolute values stay finite and correct
    through zero and below it. An earlier ratio-based design was measured, rejected
    for exactly this, and deleted; do not reintroduce it.

    WHY NO CORRECTION FACTOR. Stage 03_01 passes stage 02's inflow through untouched
    -- measured ratio 1.000 to three decimals for every drivetrain and year tested.
    The two stages hold the same quantity in the same units, so a draw transplants
    as-is.

    COARSE TO FINE. Stage 02 models Liquids and Hybrid; later stages split them into
    Petrol/Diesel and HEV/PHEV. `parent_by_drv` maps each fine drivetrain to its
    parent, and both children receive the parent's per-draw values. The caller then
    applies its own share to divide them. This keeps the three effects separate and
    composable:
        volume  from stage 02, where it is sampled
        split   from stage 03_01, untouched, with its own uncertainty
        segment from stage 03_02, untouched, renormalised to sum to one

    RAW, NOT FLOORED. The RAW residual is returned, negatives included, because
    flooring must happen per draw at the point of use and not here. Flooring is
    nonlinear -- `mean(max(x,0)) >= max(mean(x),0)` -- so flooring each draw gives a
    slightly higher mean than today's single floored trajectory: +0.228 million
    vehicles for Liquids in 2035, +0.066 for Hybrid in 2040, zero elsewhere and zero
    for BEV throughout. That difference is the correct Monte Carlo answer, not an
    artefact: in a draw where the fleet target lands higher, liquid-fuel inflow
    really is still positive that year, and the single-trajectory pipeline had no way
    to represent it.

    Returns `{fine_drivetrain: (n_years, n_draws) float32}` in millions of vehicles,
    aligned to `years`. Years stage 02 does not cover are filled with NaN so a caller
    cannot silently treat them as zero.
    """
    p02 = params.stock_flow
    drivetrains = sorted({drv for (_, drv) in stock_dict.keys()})
    mc = run_stage02_cohort_monte_carlo(
        stock_dict=stock_dict,
        drivetrains=drivetrains,
        model_end_year=int(p02.model_end_year),
        init_max_age=p02.init_max_age,
        n_draws=int(params.monte_carlo.n_draws),
        seed=params.monte_carlo.seed,
        lifetime_by_drv=p02.lifetime_by_drv,
        lifetime_override_by_drv=p02.lifetime_override_by_drv,
        lifetime_scale_lambda_relative_spread=p02.lifetime_scale_lambda_relative_spread,
        negative_inflow_policy=p02.negative_inflow_policy,
        hard_zero_inflow_from_year_by_drv=p02.hard_zero_inflow_from_year_by_drv,
        hard_zero_inflow_until_year_by_drv=p02.hard_zero_inflow_until_year_by_drv,
        inflow_mode_by_drv=p02.inflow_mode_by_drv,
        inflow_phaseout_by_drv=p02.inflow_phaseout_by_drv,
        inflow_phaseout_max_share_triangular_by_drv=getattr(
            p02, "inflow_phaseout_max_share_triangular_by_drv", {}
        ),
        stock_target_relative_spread=p02.stock_target_relative_spread,
        stock_target_uncertainty_start_year=p02.stock_target_uncertainty_start_year,
        stock_target_ramp_max_rate_per_year=p02.stock_target_ramp_max_rate_per_year,
        stock_target_correlated_mix=p02.stock_target_correlated_mix,
        total_fleet_relative_spread=p02.total_fleet_relative_spread,
    )

    # MEMORY. The stage-02 run above holds five per-year arrays for every drivetrain
    # and is the peak of this whole function -- at 200,000 draws roughly 6.6 GB, and
    # it grows in proportion to the draw count. Everything except the one array we
    # need is dropped immediately below, so what this function RETAINS is only
    # `n_parents x n_years x n_draws` in float32: 254 MB at 200,000 draws, 1.3 GB at
    # a million. If a run ever exceeds available memory it will be the stage-02 pass
    # that does it, not this.
    by_parent: dict[str, np.ndarray] = {}
    for coarse in list(mc.results):
        result = mc.results[coarse]
        t = np.asarray(result["t"], dtype=int)
        # The RAW residual. `inflow_applied_by_year` is already floored and would
        # hide the negatives the caller has to floor per draw.
        raw = np.asarray(result["inflow_by_year"], dtype=np.float32)
        idx = {int(y): i for i, y in enumerate(t)}
        aligned = np.full((len(years), raw.shape[1]), np.nan, dtype=np.float32)
        for j, y in enumerate(years):
            i = idx.get(int(y))
            if i is not None:
                aligned[j] = raw[i]
        by_parent[coarse] = aligned
        mc.results[coarse] = None       # release this drivetrain's other arrays now
        del result, raw
    del mc

    # Fine drivetrains sharing a parent get the SAME array object, not a copy --
    # Petrol and Diesel are one 154 MB array between them, not two. Callers must
    # treat these as read-only.
    out: dict[str, np.ndarray] = {}
    for fine, parent in parent_by_drv:
        if parent not in by_parent:
            raise KeyError(
                f"inflow_uncertainty_parent_by_drv maps {fine!r} to {parent!r}, but "
                f"stage 02 does not model {parent!r}. It models {sorted(by_parent)}."
            )
        out[fine] = by_parent[parent]
    return out
