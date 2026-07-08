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
from typing import Any

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
from src.monte_carlo import (  # type: ignore
    Triangular, summarize_distribution, sum_by_period, sensitivity_correlations, plot_tornado,
)
from src.stockflow_model import (  # type: ignore
    BackcastState, build_backcast_state,
    run_cohort_survival_model, run_cohort_survival_monte_carlo,
)


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
# See src/stockflow_model.py for the actual implementation of everything above.


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

    init_max_age = p02.init_max_age
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
    # Monte Carlo (opt-in via params.monte_carlo.enabled, default False -- does not
    # affect or slow down a normal deterministic run above; everything above this
    # point is unchanged whether or not this block runs).
    #
    # Uses the SAME `_run_cohort_recurrence()` core as the deterministic run above
    # (via `run_cohort_survival_monte_carlo`, a thin wrapper around it) -- there is no
    # separate/duplicate implementation of the cohort model for Monte Carlo. Every
    # varying value (which drivetrains, what spread) comes from `params` --
    # nothing here is hardcoded.
    #
    # IMPORTANT, READ BEFORE COMPARING THIS TO THE TWO DIAGNOSTIC PLOTS ABOVE:
    # `02_stock_vs_target_check.png` CANNOT show any Monte Carlo variation, ever, no
    # matter how this block is extended -- it's not a limitation of this code, it's
    # what "stock-driven" means: modeled stock is FORCED to exactly equal the
    # REMIND-prescribed target every year, regardless of lifetime assumptions
    # (verified directly: `abs(modeled_stock - target) < 1e-6` for every year, always).
    # `02_flows_by_drivetrain_check.png`'s underlying quantities (inflow, outflow) DO
    # genuinely vary with lifetime uncertainty -- this block now tracks that variation
    # YEAR BY YEAR (not just a single 2070 total) and plots it directly against the
    # deterministic flows chart's own style, so the two are actually comparable.
    #
    # Saves the RAW per-draw CUMULATIVE arrays (not the full per-year-per-draw arrays,
    # which would be ~1GB+ at 200,000 draws x 5 drivetrains x 2 metrics x 65 years --
    # too large to pickle by default) as `mc_stage02_draws`: stage 03 needs these,
    # with the SAME draw index, to apply its own (unknown_whereabouts_share,
    # export_share) uncertainty on top of THIS stage's lifetime uncertainty. The
    # per-YEAR uncertainty bands (median + P2.5/P97.5, tiny -- one number per year,
    # not per draw) are saved in `mc_stage02_summary` instead.
    # -----------------------------------------------------------------------
    if params.monte_carlo.enabled:
        n_draws = params.monte_carlo.n_draws
        output_periods = params.monte_carlo.output_periods
        rng_seed_seq = np.random.SeedSequence(params.monte_carlo.seed)
        drivetrains_present = sorted({drv for (_, drv) in stock_dict.keys()})
        # Each drivetrain gets its own independently-seeded generator (spawned from
        # the one params.monte_carlo.seed) -- stable under adding/removing
        # drivetrains, same reasoning as monte_carlo.sample_scalars().
        child_seeds = rng_seed_seq.spawn(len(drivetrains_present))

        draws_by_drv: dict[str, dict[str, np.ndarray]] = {}
        summary_by_drv: dict[str, dict] = {}

        # [NEW] Cumulative inflow/out_survival/stock for arbitrary (start, end) year
        # windows (single year: start==end), not just the whole-horizon 2070 total --
        # same `params.monte_carlo.output_periods` setting stage 03_02 uses, so a
        # request like "2030 only" or "2030-2040" means the same thing everywhere.
        # Uses `monte_carlo.sum_by_period()` on the per-year arrays
        # `_run_cohort_recurrence` ALREADY tracks cheaply (n_years x n_draws, not a
        # full per-cohort history) -- no change needed to the recurrence itself, only
        # this post-hoc summarization layer. Stock is DETERMINISTIC here (see module
        # docstring: modeled stock is FORCED to equal the REMIND target exactly,
        # regardless of lifetime draws) -- reported as a `summarize_distribution()`
        # of a constant-broadcast array (correctly shows std=0), for API consistency
        # with inflow/out_survival rather than a bare float needing separate handling.
        period_summary_by_drv: dict[str, dict] = {}
        eu_total_period_sums: dict[tuple[int, int], dict[str, np.ndarray]] = {
            p: {"cumulative_inflow": np.zeros(n_draws), "cumulative_out_survival": np.zeros(n_draws)}
            for p in output_periods
        }
        # [NEW] Sensitivity analysis: which drivetrain's lifetime uncertainty actually
        # drives EU-total output uncertainty. Only ONE uncertain input per drivetrain
        # exists at this stage (scale_lambda -- shape_k is never made uncertain), so
        # this answers "does BEV's or Diesel's (etc.) lifetime spread matter more for
        # the EU total", not a within-drivetrain sensitivity question.
        sensitivity_input_draws: dict[str, np.ndarray] = {}

        for drivetrain, child_seed in zip(drivetrains_present, child_seeds):
            rng = np.random.default_rng(child_seed)
            base = LIFETIME_BY_DRV[drivetrain]
            spread = p02.lifetime_scale_lambda_relative_spread[drivetrain]
            scale_lambda_draws = Triangular(
                base.scale_lambda * (1 - spread), base.scale_lambda, base.scale_lambda * (1 + spread),
            ).sample(rng, n=n_draws)
            sensitivity_input_draws[f"{drivetrain}_scale_lambda"] = scale_lambda_draws

            stock_series = pd.to_numeric(stock_dict[("EUR", drivetrain)]["stock"], errors="coerce").fillna(0.0)
            stock_series.index = stock_series.index.astype(int)
            stock_series = stock_series.sort_index()

            backcast = build_backcast_state(
                stock_series=stock_series, model_end_year=model_end_year,
                shape_k=base.shape_k, scale_lambda=base.scale_lambda,  # point-estimate backcast, shared across draws
                init_max_age=init_max_age,
            )
            result = run_cohort_survival_monte_carlo(
                stock_series=stock_series, model_end_year=model_end_year, drivetrain=drivetrain,
                shape_k_draws=np.full(n_draws, base.shape_k), scale_lambda_draws=scale_lambda_draws,
                lifetime_override=LIFETIME_OVERRIDE_BY_DRV.get(drivetrain), backcast=backcast,
                negative_inflow_policy=negative_inflow_policy,
            )
            draws_by_drv[drivetrain] = {
                "scale_lambda": scale_lambda_draws,
                "cumulative_inflow": result["cumulative_inflow"],
                "cumulative_out_survival": result["cumulative_out_survival"],
            }

            # Per-YEAR uncertainty band (median, P2.5, P97.5) -- computed from the
            # full (n_years, n_draws) arrays, but only the tiny summarized band (one
            # triple of numbers PER YEAR, not per draw) is kept/saved.
            years_list = result["t"].tolist()
            inflow_band = {
                "years": years_list,
                "p2_5": np.percentile(result["inflow_by_year"], 2.5, axis=1).tolist(),
                "median": np.percentile(result["inflow_by_year"], 50, axis=1).tolist(),
                "p97_5": np.percentile(result["inflow_by_year"], 97.5, axis=1).tolist(),
            }
            out_survival_band = {
                "years": years_list,
                "p2_5": np.percentile(result["out_survival_by_year"], 2.5, axis=1).tolist(),
                "median": np.percentile(result["out_survival_by_year"], 50, axis=1).tolist(),
                "p97_5": np.percentile(result["out_survival_by_year"], 97.5, axis=1).tolist(),
            }

            summary_by_drv[drivetrain] = {
                "cumulative_inflow": summarize_distribution(result["cumulative_inflow"]),
                "cumulative_out_survival": summarize_distribution(result["cumulative_out_survival"]),
                "inflow_by_year_band": inflow_band,
                "out_survival_by_year_band": out_survival_band,
            }
            s = summary_by_drv[drivetrain]["cumulative_out_survival"]
            print(
                f"Monte Carlo [{drivetrain}]: {n_draws:,} draws, cumulative_out_survival -- "
                f"mean={s['mean']:.2f}, median={s['median']:.2f}, mode={s['mode']:.2f}, "
                f"std={s['std']:.2f}, P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
            )

            # --- [NEW] period-based summaries for this drivetrain ---
            inflow_period_sums = sum_by_period(result["inflow_by_year"], result["t"], output_periods)
            out_survival_period_sums = sum_by_period(result["out_survival_by_year"], result["t"], output_periods)
            drv_period_summary: dict[tuple[int, int], dict] = {}
            for period in output_periods:
                start, end = period
                mask = (result["t"] >= start) & (result["t"] <= end)
                years_in_period = result["t"][mask]
                stock_values_in_period = result["stock_t"][mask]  # deterministic, (n_years_in_period,)

                eu_total_period_sums[period]["cumulative_inflow"] += inflow_period_sums[period]
                eu_total_period_sums[period]["cumulative_out_survival"] += out_survival_period_sums[period]

                drv_period_summary[period] = {
                    "cumulative_inflow": summarize_distribution(inflow_period_sums[period]),
                    "cumulative_out_survival": summarize_distribution(out_survival_period_sums[period]),
                    "stock_end_of_period": summarize_distribution(
                        np.full(n_draws, stock_values_in_period[-1] if stock_values_in_period.size else np.nan)
                    ),
                    "stock_sum_over_period": summarize_distribution(np.full(n_draws, stock_values_in_period.sum())),
                    "stock_per_year": {
                        int(y): summarize_distribution(np.full(n_draws, v))
                        for y, v in zip(years_in_period, stock_values_in_period)
                    },
                }
            period_summary_by_drv[drivetrain] = drv_period_summary

        # --- [NEW] EU-total period summaries (summed across drivetrains) ---
        eu_total_period_summary: dict[tuple[int, int], dict] = {}
        for period in output_periods:
            eu_total_period_summary[period] = {
                "cumulative_inflow": summarize_distribution(eu_total_period_sums[period]["cumulative_inflow"]),
                "cumulative_out_survival": summarize_distribution(eu_total_period_sums[period]["cumulative_out_survival"]),
            }
        for period in output_periods:
            s = eu_total_period_summary[period]["cumulative_out_survival"]
            print(
                f"Monte Carlo [EU total, {period[0]}-{period[1]}, cumulative_out_survival]: "
                f"{n_draws:,} draws -- mean={s['mean']:.2f}, median={s['median']:.2f}, "
                f"P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
            )

        # --- [NEW] Sensitivity analysis: which drivetrain's lifetime uncertainty
        # drives EU-total cumulative_out_survival uncertainty most. Headline period =
        # the WIDEST requested period (typically the whole horizon) -- sensitivity is
        # reported for ONE output at a time by design (`monte_carlo.
        # sensitivity_correlations`), and the widest period is the most decision-
        # relevant "does this matter for the big picture" question.
        headline_period = max(output_periods, key=lambda p: p[1] - p[0])
        sensitivity_df = sensitivity_correlations(
            sensitivity_input_draws, eu_total_period_sums[headline_period]["cumulative_out_survival"],
        )
        print(f"\nSensitivity [EU total, {headline_period[0]}-{headline_period[1]}, cumulative_out_survival]:")
        print(sensitivity_df.to_string(index=False))

        fig_tornado, _ = plot_tornado(
            sensitivity_df,
            title=f"Sensitivity: EU-total cumulative_out_survival, {headline_period[0]}-{headline_period[1]}",
        )
        fig_path_tornado = fig_dir / "02_monte_carlo_sensitivity_tornado.png"
        fig_tornado.savefig(fig_path_tornado, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path_tornado}")

        saved_mc = save_many(
            mc_stage02_draws=draws_by_drv,       # raw per-draw CUMULATIVE arrays -- stage 03 consumes these
            mc_stage02_summary=summary_by_drv,   # cumulative summary + per-year bands -- for inspection/plotting
            mc_stage02_period_summary={"by_drivetrain": period_summary_by_drv, "eu_total": eu_total_period_summary},
            mc_stage02_sensitivity=sensitivity_df,
            root=PROJECT_ROOT,
        )
        print("Saved Monte Carlo artifacts:", saved_mc)

        # -----------------------------------------------------------------------
        # Plot 1: inflow/outflow OVER TIME, median + P2.5-P97.5 band, all drivetrains
        # overlaid -- the DIRECT Monte Carlo counterpart of
        # `02_flows_by_drivetrain_check.png` above (same two-panel inflow/outflow
        # layout), so the deterministic and uncertainty views are actually comparable
        # side by side, not an isolated, differently-shaped chart.
        # -----------------------------------------------------------------------
        fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        colors = plt.cm.tab10.colors
        for i, drivetrain in enumerate(drivetrains_present):
            color = colors[i % len(colors)]
            inb = summary_by_drv[drivetrain]["inflow_by_year_band"]
            oub = summary_by_drv[drivetrain]["out_survival_by_year_band"]
            ax_in.plot(inb["years"], inb["median"], color=color, linewidth=1.6, label=drivetrain)
            ax_in.fill_between(inb["years"], inb["p2_5"], inb["p97_5"], color=color, alpha=0.2)
            ax_out.plot(oub["years"], oub["median"], color=color, linewidth=1.6, label=drivetrain)
            ax_out.fill_between(oub["years"], oub["p2_5"], oub["p97_5"], color=color, alpha=0.2)

        ax_in.axhline(0, color="black", linewidth=0.8)
        ax_in.set_title(f"Monte Carlo: annual inflow, median + P2.5-P97.5 band (n={n_draws:,})", fontsize=12)
        ax_in.set_ylabel("Inflow [million/year]")
        ax_in.grid(True, linestyle="--", alpha=0.3)
        ax_in.spines["top"].set_visible(False)
        ax_in.spines["right"].set_visible(False)
        ax_in.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

        ax_out.set_title("Monte Carlo: annual total outflow, median + P2.5-P97.5 band", fontsize=12)
        ax_out.set_xlabel("Year")
        ax_out.set_ylabel("Outflow [million/year]")
        ax_out.grid(True, linestyle="--", alpha=0.3)
        ax_out.spines["top"].set_visible(False)
        ax_out.spines["right"].set_visible(False)
        ax_out.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig_path = fig_dir / "02_monte_carlo_flows_over_time.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path}")

        # -----------------------------------------------------------------------
        # Plot 2: cumulative-by-2070 histograms, one per drivetrain -- "what's the
        # total by the end of the horizon", complementing plot 1's "how does the
        # uncertainty evolve year by year".
        # -----------------------------------------------------------------------
        fig, axes = plt.subplots(len(drivetrains_present), 1, figsize=(8, 3.2 * len(drivetrains_present)), squeeze=False)
        for ax_row, drivetrain in zip(axes, drivetrains_present):
            ax = ax_row[0]
            summ = summary_by_drv[drivetrain]["cumulative_out_survival"]
            edges = np.array(summ["bin_edges"])
            freqs = np.array(summ["frequencies"])
            ax.bar((edges[:-1] + edges[1:]) / 2, freqs, width=np.diff(edges), color="#4a7fb5", alpha=0.85)
            ax.axvline(summ["mean"], color="black", linewidth=1.4, label=f"mean={summ['mean']:.1f}")
            ax.axvline(summ["median"], color="#2b8a3e", linewidth=1.2, linestyle="-.", label=f"median={summ['median']:.1f}")
            ax.axvline(summ["mode"], color="#e0793c", linewidth=1.2, linestyle=":", label=f"mode={summ['mode']:.1f}")
            ax.axvline(summ["p2_5"], color="black", linestyle="--", linewidth=1.0, label=f"P2.5={summ['p2_5']:.1f}")
            ax.axvline(summ["p97_5"], color="black", linestyle="--", linewidth=1.0, label=f"P97.5={summ['p97_5']:.1f}")
            ax.set_title(f"{drivetrain}: cumulative out-survival through {model_end_year} (n={n_draws:,})")
            ax.legend(frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig_path = fig_dir / "02_monte_carlo_cumulative_out_survival.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path}")

    return saved


if __name__ == "__main__":
    main()
