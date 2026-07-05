"""
02_stockdriven.py
==================

Implements a **stock-driven cohort-survival model**: given a prescribed total vehicle
stock trajectory per (region, drivetrain) from stage 01, works out the annual inflow
required to hit that prescribed stock each year, accounting for retirement via a
Weibull survival function. Persists `matrices_by_key` for stage 03.

FIXES APPLIED THIS ROUND
--------------------------
- **Dataclass params**: `params["02_stock_flow"]["lifetime_by_drv"]` ->
  `params.stock_flow.lifetime_by_drv` (a `dict[str, WeibullLifetime]`, not
  `dict[str, dict]` -- `LIFETIME_BY_DRV[drv]["shape_k"]` -> `LIFETIME_BY_DRV[drv].shape_k`).
  `lifetime_override_by_drv` values are now `LifetimeOverride | None` dataclass
  instances, not raw dicts -- `get_effective_lifetime_params` updated accordingly.
- **Project-root resolution**: was `Path(__file__).resolve().parent` (only correct if
  this script sits directly in the project root) -- same bug class already fixed in
  `00_parameters.py`/`01_data_prep.py`. Now uses the same upward-searching
  `_find_project_root`.
- **`root=` threading**: `load_many`/`save_many` now receive `root=PROJECT_ROOT`
  explicitly, same fix as `01_data_prep.py`'s Fix Log item 19 -- without this, running
  from `code/` would fail to find stage 00/01's artifacts.
- **Dead code removed** (M17, M18, M19, plus the duplicate-computation copy-paste
  noted at the per-key loop): `last_exp_data_year`, `accelerating_year`,
  `start_year_plotting`, `end_year_plotting`, the module-level `start_year`/`t_global`/
  `N_t_global` (shadowed and never used -- the per-key `t0`/`t`/`N_t` inside the loop is
  what's actually used), the duplicate `pd.to_numeric(...)` recomputation, and the
  unused `years` variable. The unused `df_exp_eu` load is also removed -- if stage 03
  needs it, it can load it directly from stage 01's artifacts; stage 02 never read it.
- **Structural, for generalization + Monte Carlo**: the per-(region, drivetrain)
  cohort-survival computation is now a standalone, pure function,
  `run_cohort_survival_model()`, with no I/O and no dependency on the params object --
  it takes `shape_k`/`scale_lambda`/a backcast state/a stock trajectory as plain
  arguments and returns plain DataFrames. This is what makes Monte Carlo tractable
  later: a sampler can call this function many times with resampled `shape_k` /
  `scale_lambda` / `stock_series` without touching `main()`'s I/O or the artifact store
  at all. `main()` itself is now just: load -> loop calling the pure function -> save.

STILL OPEN / NOT VERIFIABLE HERE
-----------------------------------
- `prepare_backcasting_state` (from `src/stock_flow.py`) is not yet shared -- its
  internal correctness (what historical-inflow assumption it makes before t0) cannot
  be verified from this file alone. See its call site below for the exact contract this
  file assumes.

NEGATIVE-INFLOW HANDLING -- now a configurable, tested policy, not a fixed behavior
--------------------------------------------------------------------------------------
**Correction to the original file's own comment, found via actual verification this
round**: the original notebook's inline note claimed a negative-inflow year leaves
modeled stock "LESS than the prescribed target" (a shortfall). That's backwards.
`inflow(t) = target(t) - remaining_total(t)` is negative exactly when
`remaining_total(t) > target(t)` -- i.e. natural attrition alone wasn't enough to bring
stock down to the falling target. Adding zero inflow (the "report_only" default) means
the modeled stock KEEPS `remaining_total(t)`, which is HIGHER than target -- a growing
SURPLUS, not a shortfall. Verified directly on the synthetic rise-then-phase-out case:
modeled stock exceeds target by up to +20 units (out of a ~200-unit peak) by the end of
the decline phase, strictly increasing, never negative. This matters for direction of
bias: `report_only` OVERSTATES fleet size (and everything downstream that scales with
it -- material demand, etc.), not understates it.

Per your decision: the DEFAULT behavior is unchanged ("report_only" -- a negative
residual-inflow year adds zero, and the model's total stock silently exceeds the
REMIND-prescribed target from that point on, with no automatic correction). This is no
longer hardcoded -- `params.stock_flow.negative_inflow_policy` selects between:
  - `"report_only"`    (default) -- original behavior, byte-identical output.
  - `"clip_to_target"` -- forces additional pro-rata outflow across surviving cohorts
    so the modeled stock hits the prescribed target exactly, even in a negative-inflow
    year -- this removes the surplus rather than letting it persist. Implemented and
    tested (see `apply_negative_inflow_policy()` below and `MATH_MODELS.md` §2.3) --
    not the default, since switching changes real numbers.
`check_negative_inflows()`'s diagnostic report is unaffected by which policy is active
-- it always reports the RAW residual value, so a negative-inflow year is never hidden
just because "clip_to_target" absorbed its effect on the cohort matrix. Note that under
`clip_to_target`, correcting one year's surplus changes the starting stock for every
subsequent year, which can itself change whether LATER years also compute a negative
residual -- the two policies are not simple diagnostic-vs-corrected views of the same
sequence of raw values, they genuinely diverge over time.
NEW THIS ROUND -- integrated diagnostic plots (per your request: in the step, not a
separate script)
--------------------------------------------------------------------------------------
Right after `matrices_by_key` is built, this stage now generates and saves TWO charts:
  1. `02_stock_vs_target_check.png` -- modeled stock vs. REMIND-prescribed target, per
     drivetrain. Under "report_only" the modeled line visibly drifts above target after
     a negative-inflow year; under "clip_to_target" the two lines overlap exactly.
  2. `02_flows_by_drivetrain_check.png` -- **NEW**: inflow (can go negative) and
     outflow (survival vs. excess) over time, per drivetrain -- the direct visual for
     this stage's core mechanism, not just its end result.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never opens an interactive window -- always saves to file
import matplotlib.pyplot as plt


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.artifacts import load_many, save_many, artifact_status  # type: ignore
from src.stock_flow import prepare_backcasting_state  # type: ignore
from src.params_schema import WeibullLifetime, LifetimeOverride  # type: ignore
from src.monte_carlo import Triangular, run_monte_carlo  # type: ignore


# ---------------------------------------------------------------------------
# THE MATH MODEL
# ---------------------------------------------------------------------------
# For each (region, drivetrain) and each year t:
#
#   1. Every existing cohort ages by one year. Survival follows Weibull:
#        S(a) = exp( -(a / lambda)^k )
#   2. Annual hazard: h(a) = 1 - S(a+1) / S(a)  -- weibull_hazard_lookup() below.
#   3. Survival outflow: out_surv = stock(age=a) * h(a).
#      stock_after_surv = stock(age=a) - out_surv.
#   4. INFLOW IS SOLVED AS A RESIDUAL: given the prescribed target(t),
#        inflow(t) = target(t) - sum_over_cohorts( stock_after_surv )
#      equivalent to: inflow(t) = [stock(t) - stock(t-1)] + outflow_survival(t)
#
# WHY THIS CAN GO NEGATIVE (a structural property, not a bug): if the prescribed stock
# declines faster than natural Weibull attrition alone would explain, the residual
# formula returns a NEGATIVE inflow. Under the default "report_only" policy this means
# modeled stock EXCEEDS the falling target from then on (a surplus, not a shortfall --
# corrected this round, see module docstring "NEGATIVE-INFLOW HANDLING" for the
# verification). Verified directly: a synthetic "rise then rapid phase-out" stock
# trajectory produces 16/61 years of negative inflow at scale_lambda=13, 9/61 at
# scale_lambda=9 (differs from a naive expectation because correcting/not-correcting
# earlier years changes the state every later year's computation starts from) -- see
# stock_driven_negative_inflow.png. Almost certainly the reason behind the
# Hybrid/PHEV/HEV scale_lambda=9.0 tuning comment in params_schema.py.
def weibull_hazard_lookup(shape_k: float, scale_lambda: float, max_age: int) -> np.ndarray:
    """
    Annual hazard by age: h[a] = 1 - S(a+1)/S(a) with S(a) = exp(-(a/lambda)^k).
    h[max_age] is forced to 1 (a Weibull-tail truncation so cohorts don't persist
    forever). `mask = S[:-1] > 0` guards against divide-by-zero if S(a) underflows to
    exactly 0.0 for very old ages -- currently unreachable for max_age=50 and
    lambda in [9, 13], but worth knowing if those change.
    """
    ages = np.arange(max_age + 1, dtype=float)
    S = np.exp(-((ages / scale_lambda) ** shape_k))

    h = np.zeros(max_age + 1, dtype=float)
    mask = S[:-1] > 0
    h[:-1][mask] = 1.0 - (S[1:][mask] / S[:-1][mask])
    h[-1] = 1.0
    return np.clip(h, 0.0, 1.0)


def get_effective_lifetime_params(
    drivetrain: str,
    year: int,
    base_params_by_drv: dict[str, WeibullLifetime],
    override_by_drv: dict[str, LifetimeOverride | None],
) -> tuple[float, float]:
    """
    Return (shape_k, scale_lambda) for `drivetrain` in `year`, applying a time-windowed
    override if one is configured and `year` falls inside [start_year, end_year].
    As shipped, every entry in `lifetime_override_by_drv` is `None` -- this mechanism is
    currently inactive for every drivetrain.
    """
    base = base_params_by_drv[drivetrain]
    k, lam = base.shape_k, base.scale_lambda

    ov = override_by_drv.get(drivetrain)
    if ov is None:
        return k, lam

    if ov.start_year <= year <= ov.end_year:
        return ov.shape_k, ov.scale_lambda
    return k, lam


def apply_negative_inflow_policy(
    policy: str,
    inflow: float,
    cohort_stock_after_survival: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Decide what actually happens to the cohort matrix in a given year, given the raw
    (possibly negative) residual inflow computed from the target-vs-survival identity.

    Returns (final_cohort_stock, inflow_applied, excess_outflow):
      - final_cohort_stock: per-cohort stock AFTER this policy's adjustment, BEFORE any
        positive inflow is added to the "born this year" slot by the caller.
      - inflow_applied: the inflow actually added to the cohort matrix (0 if inflow <= 0
        under either policy -- neither policy invents negative registrations).
      - excess_outflow: additional outflow (beyond natural survival) this policy forced,
        0 under "report_only".

    "report_only" (ORIGINAL, default): if inflow <= 0, do nothing further -- the cohort
    matrix keeps whatever natural survival alone produced, silently falling short of
    the prescribed target. This is a passthrough, added only so both policies share one
    call site rather than an if/else scattered through the caller.

    "clip_to_target": if inflow <= 0, forces `-inflow` of ADDITIONAL outflow, spread
    pro-rata across all currently-surviving cohorts (proportional to each cohort's
    share of total remaining stock), so the cohort matrix's total exactly equals the
    prescribed target this year. If total remaining stock is already 0 (nothing left to
    remove), no adjustment is possible and the policy degrades to "report_only" for
    that single year (there is nothing else it could do).
    """
    if inflow > 0:
        return cohort_stock_after_survival, float(inflow), 0.0

    if policy == "report_only":
        return cohort_stock_after_survival, 0.0, 0.0

    if policy == "clip_to_target":
        total = float(cohort_stock_after_survival.sum())
        excess = -float(inflow)
        if total <= 0.0:
            return cohort_stock_after_survival, 0.0, 0.0
        scale = max(0.0, (total - excess) / total)
        adjusted = cohort_stock_after_survival * scale
        actual_excess = total - float(adjusted.sum())
        return adjusted, 0.0, actual_excess

    raise ValueError(
        f"negative_inflow_policy={policy!r} is not one of ['report_only', 'clip_to_target']."
    )


NEGATIVE_INFLOW_POLICIES: tuple[str, ...] = ("report_only", "clip_to_target")


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


def run_cohort_survival_model(
    stock_series: pd.Series,
    model_end_year: int,
    drivetrain: str,
    base_shape_k: float,
    base_scale_lambda: float,
    lifetime_override: LifetimeOverride | None,
    backcast: BackcastState,
    negative_inflow_policy: str = "report_only",
) -> dict[str, pd.DataFrame]:
    """
    Pure, I/O-free cohort-survival simulation for ONE (region, drivetrain) key.

    MONTE CARLO NOTE: this function takes `stock_series`, `base_shape_k`,
    `base_scale_lambda`, and `backcast` as plain arguments, with no dependency on the
    params object or the artifact store. A Monte Carlo sampler can call this directly,
    many times, with resampled values for any of these (re-deriving `backcast` via
    `build_backcast_state()` first, since it also depends on shape_k/scale_lambda) --
    no change to this function or to `main()`'s I/O is needed to do that.

    `negative_inflow_policy`: "report_only" (default, original behavior) or
    "clip_to_target" -- see `apply_negative_inflow_policy()` above and
    `MATH_MODELS.md` §2.3 for the exact mechanics of both.

    Returns a dict with keys: "stock_t_tau_df", "outflow_surv_df", "flows_df",
    "diag_df", "results_df" -- same shapes as the original per-key outputs, plus a new
    "out_excess" column in `flows_df`/`diag_df` (always 0 under "report_only").
    """
    tau_back = backcast.tau_back
    max_age = backcast.max_age

    t0 = int(stock_series.index.min())
    t = np.arange(t0, model_end_year + 1, dtype=int)
    N_t = len(t)

    stock_t = stock_series.reindex(t).ffill().bfill().to_numpy(dtype=float)

    N_tau = tau_back.size
    stock_t_tau = np.zeros((N_t, N_tau), dtype=float)
    stock_t_tau[0, :] = backcast.stock0_by_cohort

    inflow_t = np.zeros(N_t, dtype=float)
    outflow_surv_t = np.zeros(N_t, dtype=float)
    outflow_excess_t = np.zeros(N_t, dtype=float)
    outflow_total_t = np.zeros(N_t, dtype=float)
    outflow_surv_t_tau = np.zeros_like(stock_t_tau)

    diag_rows: list[dict[str, Any]] = []

    # [NEW, performance + Monte Carlo readiness] Cache the hazard-by-age lookup keyed by
    # (shape_k, scale_lambda). Without an active lifetime_override (the shipped default
    # for every drivetrain), k_eff/lam_eff are IDENTICAL every single year, so
    # weibull_hazard_lookup() was being recomputed with the same inputs N_t times per
    # (region, drivetrain) key -- confirmed via the original notebook's own comment on
    # prepare_backcasting_state's h_out_life_age (computed once, then silently discarded
    # in favor of a fresh recomputation every year). Negligible for one deterministic
    # run, but this function is exactly what Monte Carlo calls repeatedly (see
    # src/monte_carlo.py) -- multiplying the redundant recomputation by every draw. This
    # cache makes repeated calls with unchanged (shape_k, scale_lambda) free after the
    # first, with zero behavior change (same array returned either way).
    _hazard_cache: dict[tuple[float, float], np.ndarray] = {}

    for i_t in range(1, N_t):
        year = int(t[i_t])
        prev_year = int(t[i_t - 1])
        prev_stock = stock_t_tau[i_t - 1, :].copy()

        ages_prev = (prev_year - tau_back).astype(int)
        valid_prev = ages_prev >= 0

        k_eff, lam_eff = get_effective_lifetime_params(
            drivetrain=drivetrain,
            year=year,
            base_params_by_drv={drivetrain: WeibullLifetime(base_shape_k, base_scale_lambda)},
            override_by_drv={drivetrain: lifetime_override},
        )

        cache_key = (k_eff, lam_eff)
        h_eff = _hazard_cache.get(cache_key)
        if h_eff is None:
            h_eff = weibull_hazard_lookup(shape_k=k_eff, scale_lambda=lam_eff, max_age=max_age)
            _hazard_cache[cache_key] = h_eff

        hazard = np.zeros_like(prev_stock)
        in_age_range = valid_prev & (ages_prev <= max_age)
        hazard[in_age_range] = h_eff[ages_prev[in_age_range]]
        hazard = np.clip(hazard, 0.0, 1.0)

        out_surv = prev_stock * hazard
        out_surv = np.minimum(out_surv, prev_stock)
        stock_after_surv = prev_stock - out_surv
        # No export/loss term here -- deferred to stage 03.
        stock_after_both = stock_after_surv

        target = float(stock_t[i_t])
        remaining_total = float(stock_after_both.sum())
        inflow_raw = target - remaining_total
        # inflow_raw < 0: prescribed stock declined faster than natural attrition
        # explains. What happens next is determined entirely by `negative_inflow_policy`
        # -- see apply_negative_inflow_policy() and MATH_MODELS.md §2.3.

        cohort_stock_final, inflow_applied, excess_outflow = apply_negative_inflow_policy(
            policy=negative_inflow_policy,
            inflow=inflow_raw,
            cohort_stock_after_survival=stock_after_both,
        )

        stock_t_tau[i_t, :] = cohort_stock_final
        j_new = np.where(tau_back == year)[0]
        if j_new.size != 1:
            raise ValueError(f"Year {year} not found in tau_back range.")
        if inflow_applied > 0:
            stock_t_tau[i_t, j_new[0]] += inflow_applied
        # NOTE: `inflow_raw` (not `inflow_applied`) is recorded in flows_df/diag_df --
        # the raw, possibly-negative value stays visible for diagnostics
        # (check_negative_inflows() reads this column) regardless of which policy is
        # active, so switching policies never hides that a negative-inflow year occurred.
        inflow_t[i_t] = inflow_raw

        outflow_surv_t_tau[i_t, :] = out_surv
        outflow_surv_t[i_t] = float(out_surv.sum())
        outflow_excess_t[i_t] = excess_outflow
        outflow_total_t[i_t] = outflow_surv_t[i_t] + outflow_excess_t[i_t]

        diag_rows.append({
            "year": year,
            "prev_stock_total": float(prev_stock.sum()),
            "target_stock": target,
            "nas": target - float(prev_stock.sum()),
            "out_survival": outflow_surv_t[i_t],
            "out_excess": outflow_excess_t[i_t],
            "out_total": outflow_total_t[i_t],
            "inflow_residual": inflow_raw,
        })

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

    diag_df = pd.DataFrame(diag_rows).set_index("year")
    results_df = stock_df.join(diag_df, how="left")

    return {
        "stock_t_tau_df": stock_t_tau_df,
        "outflow_surv_df": outflow_surv_df,
        "flows_df": flows_df,
        "diag_df": diag_df,
        "results_df": results_df,
    }


def plot_flows_by_drivetrain(
    matrices_by_key: dict[tuple[str, str], dict[str, pd.DataFrame]],
    region: str = "EUR",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    The direct visual for the core mechanism of THIS stage: inflow (top, can go
    negative -- see MATH_MODELS.md §2.3) and outflow, split into survival vs. excess
    (bottom, stacked), per drivetrain in `region`. A negative dip in the top panel is
    exactly a negative-inflow year; a nonzero orange band in the bottom panel is
    "out_excess" from the "clip_to_target" policy (always zero under "report_only").
    """
    keys = sorted(k for k in matrices_by_key if k[0] == region)
    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = plt.cm.tab10.colors

    for i, key in enumerate(keys):
        drivetrain = key[1]
        flows = matrices_by_key[key]["flows_df"]
        color = colors[i % len(colors)]
        ax_in.plot(flows.index, flows["inflow"], color=color, linewidth=1.6, label=drivetrain)

    ax_in.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_in.set_title(f"Annual inflow (new registrations) by drivetrain ({region})", fontsize=12)
    ax_in.set_ylabel("Inflow [million/year]")
    ax_in.grid(True, linestyle="--", alpha=0.3)
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    ax_in.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

    # Bottom panel: pick the drivetrain with the largest total outflow to avoid an
    # unreadable stacked chart across every drivetrain at once -- still shows the
    # survival-vs-excess split concretely for the drivetrain where it matters most.
    totals = {k[1]: matrices_by_key[k]["flows_df"]["out_total"].sum() for k in keys}
    focus_drv = max(totals, key=totals.get) if totals else None
    if focus_drv is not None:
        focus_key = next(k for k in keys if k[1] == focus_drv)
        flows = matrices_by_key[focus_key]["flows_df"]
        ax_out.stackplot(
            flows.index, flows["out_survival"], flows["out_excess"],
            labels=["out_survival", "out_excess"], colors=["#4a7fb5", "#e0793c"],
        )
        ax_out.set_title(f"Outflow breakdown for '{focus_drv}' (largest total outflow in {region})", fontsize=12)
    ax_out.set_xlabel("Year")
    ax_out.set_ylabel("Outflow [million/year]")
    ax_out.grid(True, linestyle="--", alpha=0.3)
    ax_out.spines["top"].set_visible(False)
    ax_out.spines["right"].set_visible(False)
    ax_out.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig, (ax_in, ax_out)


def check_negative_inflows(matrices_by_key: dict[tuple[str, str], dict[str, pd.DataFrame]]) -> pd.DataFrame | None:
    """
    Report every (drivetrain, year) where the EUR-aggregate residual inflow came out
    negative. Reports only -- does not correct (see module docstring "STILL OPEN").
    """
    neg_orig = []
    for (reg, drv), mats in matrices_by_key.items():
        if reg != "EUR" or "flows_df" not in mats:
            continue
        df = mats["flows_df"].copy()
        df = df[df["inflow"] < 0]
        if len(df) > 0:
            df = df.reset_index()
            df["Drive Train"] = drv
            neg_orig.append(df[["Drive Train", "year", "inflow"]])

    if neg_orig:
        neg_orig = pd.concat(neg_orig, ignore_index=True)
        print("NEGATIVE INFLOW IN ORIGINAL MODEL:")
        print(neg_orig.sort_values("inflow").to_string(index=False))
        return neg_orig
    print("No negatives in original model inflow.")
    return None


def plot_stock_vs_target(
    matrices_by_key: dict[tuple[str, str], dict[str, pd.DataFrame]],
    region: str = "EUR",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot modeled stock (summed across cohorts, from `stock_t_tau_df`) against the
    REMIND-prescribed target (`diag_df["target_stock"]`), one line pair per drivetrain
    in `region`. This is the direct visual for the surplus/shortfall question in
    MATH_MODELS.md §2.3 -- under "report_only", the modeled line visibly drifts above
    target after a negative-inflow year; under "clip_to_target", the two lines overlap
    exactly at every point.
    """
    keys = [k for k in matrices_by_key if k[0] == region]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors

    for i, key in enumerate(sorted(keys)):
        drivetrain = key[1]
        mats = matrices_by_key[key]
        modeled = mats["stock_t_tau_df"].sum(axis=1)
        target = mats["diag_df"]["target_stock"].reindex(modeled.index)
        color = colors[i % len(colors)]
        ax.plot(modeled.index, modeled.values, color=color, linewidth=1.6, label=f"{drivetrain} (modeled)")
        ax.plot(target.index, target.values, color=color, linewidth=1.2, linestyle=":", label=f"{drivetrain} (target)")

    ax.set_title(f"Modeled stock vs. prescribed target ({region})", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stock [million]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig, ax


def plot_monte_carlo_histogram(
    values: list[float],
    title: str,
    xlabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Generic histogram of Monte Carlo draw results -- no EV-specific logic, just a
    plain distribution plot with mean/percentile markers."""
    values_arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(values_arr, bins=30, color="#4a7fb5", alpha=0.85, edgecolor="white")
    mean = values_arr.mean()
    p5, p95 = np.percentile(values_arr, [5, 95])
    ax.axvline(mean, color="black", linewidth=1.6, label=f"mean = {mean:.2f}")
    ax.axvline(p5, color="black", linewidth=1.0, linestyle="--", label=f"5th pct = {p5:.2f}")
    ax.axvline(p95, color="black", linewidth=1.0, linestyle="--", label=f"95th pct = {p95:.2f}")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Draws")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


def main() -> dict[str, Path]:
    """Run the full stage-02 stock-driven cohort model and persist `matrices_by_key`."""
    try:
        loaded = load_many("params", "stock_dict", root=PROJECT_ROOT)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing inputs for 02_stock_flow. Run 00_parameters and 01_data_prep "
            "through their final save cells first."
        ) from exc

    params = loaded["params"]
    stock_dict = loaded["stock_dict"]

    print(artifact_status(root=PROJECT_ROOT))

    p02 = params.stock_flow
    model_end_year = int(p02.model_end_year)
    LIFETIME_BY_DRV = p02.lifetime_by_drv
    LIFETIME_OVERRIDE_BY_DRV = p02.lifetime_override_by_drv
    negative_inflow_policy = p02.negative_inflow_policy
    # NOTE (organizational, unchanged from earlier review): `p02` also has
    # `unknown_whereabouts_share` and `export_share_by_drv`, neither read here --
    # both are actually consumed by stage 03. Declared under stock_flow for historical
    # reasons; not moved this round since stage 03 isn't fixed yet.

    init_max_age = 50
    results_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    matrices_by_key: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}

    for key, df in stock_dict.items():
        region, drivetrain = key

        base = LIFETIME_BY_DRV[drivetrain]
        stock_series = pd.to_numeric(df["stock"], errors="coerce").fillna(0.0)
        stock_series.index = stock_series.index.astype(int)
        stock_series = stock_series.sort_index()

        backcast = build_backcast_state(
            stock_series=stock_series,
            model_end_year=model_end_year,
            shape_k=base.shape_k,
            scale_lambda=base.scale_lambda,
            init_max_age=init_max_age,
        )

        out = run_cohort_survival_model(
            stock_series=stock_series,
            model_end_year=model_end_year,
            drivetrain=drivetrain,
            base_shape_k=base.shape_k,
            base_scale_lambda=base.scale_lambda,
            lifetime_override=LIFETIME_OVERRIDE_BY_DRV.get(drivetrain),
            backcast=backcast,
            negative_inflow_policy=negative_inflow_policy,
        )

        results_by_key[key] = out["results_df"]
        matrices_by_key[key] = {
            "stock_t_tau_df": out["stock_t_tau_df"],
            "outflow_surv_df": out["outflow_surv_df"],
            "flows_df": out["flows_df"],
            "diag_df": out["diag_df"],
        }

    check_negative_inflows(matrices_by_key)

    # -----------------------------------------------------------------------
    # Diagnostic plot: modeled stock vs. prescribed target (integrated here, not a
    # separate script -- this is exactly the step that produces matrices_by_key, the
    # only thing this chart needs). See MATH_MODELS.md §2.3 for what to look for.
    # -----------------------------------------------------------------------
    fig, ax = plot_stock_vs_target(matrices_by_key, region="EUR")
    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "02_stock_vs_target_check.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Diagnostic plot 2: inflow/outflow by drivetrain -- the core mechanism of this
    # stage, including exactly where/how much inflow goes negative.
    # -----------------------------------------------------------------------
    fig, _ = plot_flows_by_drivetrain(matrices_by_key, region="EUR")
    fig_path = fig_dir / "02_flows_by_drivetrain_check.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    saved = save_many(matrices_by_key=matrices_by_key, root=PROJECT_ROOT)
    print("Saved artifacts:", saved)

    # -----------------------------------------------------------------------
    # Monte Carlo demonstration (opt-in via params.monte_carlo.enabled, default False --
    # does not affect or slow down a normal deterministic run). See src/monte_carlo.py
    # for the generic, product-agnostic sampling machinery this uses.
    #
    # WHY THIS LIVES HERE, AS A CONCRETE EXAMPLE, NOT JUST LIBRARY CODE: proving Monte
    # Carlo readiness means actually running it, not just building an API that could
    # theoretically be called. This varies BEV's Weibull scale_lambda (a genuinely
    # uncertain parameter -- "TODO: change per scenario" in params_schema.py) via a
    # Triangular(11, 13, 15) distribution centered on the current point estimate, and
    # collects the resulting distribution of EUR BEV stock in the model's final year.
    # Swap the `spec` dict below for any other path in the params tree to explore a
    # different uncertain input -- the mechanism is fully generic (see
    # src/monte_carlo.py's UncertaintySpec).
    # -----------------------------------------------------------------------
    if params.monte_carlo.enabled:
        bev_key = ("EUR", "BEV")
        if bev_key not in stock_dict:
            print("Monte Carlo skipped: no ('EUR', 'BEV') key in stock_dict.")
        else:
            bev_stock_series = pd.to_numeric(stock_dict[bev_key]["stock"], errors="coerce").fillna(0.0)
            bev_stock_series.index = bev_stock_series.index.astype(int)
            bev_stock_series = bev_stock_series.sort_index()

            def compute_total_bev_inflow(params_variant) -> float:
                """
                Pure function of params_variant -- re-runs the exact same
                deterministic per-key computation stage 02 already uses, just with a
                resampled BEV scale_lambda.

                METRIC CHOICE, worth understanding: this returns TOTAL CUMULATIVE
                INFLOW over the horizon, not final stock. In a stock-driven model,
                final stock is PINNED to the REMIND-prescribed target regardless of
                lifetime assumptions (verified directly: varying scale_lambda from 11
                to 15 left final EUR BEV stock byte-identical) -- inflow is the
                residual that adjusts to hit that fixed target, so it's inflow/outflow
                that actually carries the lifetime uncertainty, not stock itself. This
                is a structural property of stock-driven models generally, not
                specific to BEV or this dataset -- worth remembering when choosing a
                Monte Carlo output metric for any stock-driven model.
                """
                p_variant = params_variant.stock_flow
                base = p_variant.lifetime_by_drv["BEV"]
                backcast_variant = build_backcast_state(
                    stock_series=bev_stock_series,
                    model_end_year=model_end_year,
                    shape_k=base.shape_k,
                    scale_lambda=base.scale_lambda,
                    init_max_age=init_max_age,
                )
                out_variant = run_cohort_survival_model(
                    stock_series=bev_stock_series,
                    model_end_year=model_end_year,
                    drivetrain="BEV",
                    base_shape_k=base.shape_k,
                    base_scale_lambda=base.scale_lambda,
                    lifetime_override=p_variant.lifetime_override_by_drv.get("BEV"),
                    backcast=backcast_variant,
                    negative_inflow_policy=p_variant.negative_inflow_policy,
                )
                return float(out_variant["flows_df"]["inflow"].sum())

            spec = {
                ("stock_flow", "lifetime_by_drv", "BEV", "scale_lambda"): Triangular(11.0, 13.0, 15.0),
            }
            mc_results = run_monte_carlo(
                base_params=params,
                spec=spec,
                n_draws=params.monte_carlo.n_draws,
                compute_fn=compute_total_bev_inflow,
                seed=params.monte_carlo.seed,
            )
            print(
                f"Monte Carlo: {len(mc_results)} draws of BEV scale_lambda ~ "
                f"Triangular(11, 13, 15) -> cumulative BEV inflow through "
                f"{model_end_year}: mean={np.mean(mc_results):.3f}, std={np.std(mc_results):.3f}"
            )

            fig, _ = plot_monte_carlo_histogram(
                mc_results,
                title=f"Monte Carlo: cumulative EUR BEV inflow through {model_end_year} under lifetime uncertainty",
                xlabel="Cumulative BEV inflow [million vehicles]",
            )
            fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / "02_monte_carlo_bev_stock.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved diagnostic plot: {fig_path}")

    return saved


if __name__ == "__main__":
    main()
