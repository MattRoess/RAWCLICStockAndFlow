"""
stockflow_model.py
=====================

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

        inflow_by_year[i_t] = inflow_applied if hard_zero_active else inflow_raw
        inflow_pre_hard_zero_override_by_year[i_t] = inflow_raw
        inflow_applied_by_year[i_t] = inflow_applied
        out_survival_by_year[i_t] = this_year_total_outflow

        if keep_full_history:
            stock_t_tau[i_t] = cohort_state
            year_out_tau = np.zeros((n_draws, n_cohorts), dtype=float)
            year_out_tau[:, valid] = out_surv
            outflow_surv_t_tau[i_t] = year_out_tau
            inflow_t[i_t] = inflow_applied if hard_zero_active else inflow_raw
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
    prescribed target, deterministic, identical regardless of draw}. See
    `monte_carlo.sum_by_period()` for turning "inflow_applied_by_year"/
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
    )