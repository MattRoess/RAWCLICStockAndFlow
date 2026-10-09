"""
02_stockdriven.py
==================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

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

[DISPLAY-ONLY FIX, this round] Both `02_flows_by_drivetrain_check.png` and
`02_monte_carlo_flows_over_time.png` (the Monte Carlo counterpart) now start plotting
one year AFTER t0 instead of AT t0. Root cause, confirmed directly in
`src/stockflow_model.py`'s `_run_cohort_recurrence`: its loop is `for i_t in
range(1, n_t)`, so t0 (the backcast base year) never gets a computed inflow/outflow
value -- every drivetrain's flows_df row 0 is exactly 0.0 by construction (the
initial cohort-by-age snapshot, not a simulated flow). This was visually glaring for
"Liquids" (its real t0 magnitude is large, ~20M/year) and invisible for drivetrains
whose real t0 magnitude happens to already be near zero (e.g. BEV, Hybrid) -- same
underlying cause everywhere, just only visible where the true value is large.
Per explicit instruction, the model itself and `matrices_by_key`/`flows_df`/
`summary_by_drv` (the actual saved artifacts, consumed by stage 03) are completely
UNCHANGED -- t0's row is still there. Only `plot_flows_by_drivetrain()` and the
Monte Carlo "Plot 1" block below crop their own x-range by one year at display time.
Whether t0's true value is genuinely unknown (REMIND's first reported year) or
recoverable was explicitly left open -- nothing is invented here either way.

[NEW, this round] hard_zero_inflow_from_year_by_drv -- a real policy override, not a
mathematical correction
--------------------------------------------------------------------------------------
Verified directly against real data (2026-07-11): for "Liquids", the residual-inflow
formula manufactures a genuine positive "phantom" inflow from ~2047 onward (peaking
~0.45M/year around 2054) -- confirmed to NOT be an interpolation artifact (the raw
REMIND target_stock itself is smooth and monotonically declining throughout) and NOT
tied to a preceding negative-inflow year (none occurred). It's a mechanical
consequence of the model always being forced to hit target_stock exactly: once the
aging survivor fleet's natural attrition outpaces the target's own decline rate near
a low floor, the model invents new registrations to hold the target, with zero basis
in reality. Per an actual documented external fact (a real ICE-sales ban), real
inflow should be a hard 0 from the ban year on, and modeled stock should be allowed
to fall below REMIND's target from that point.

`params.stock_flow.hard_zero_inflow_from_year_by_drv` (default `{"Liquids": 2050}`)
is threaded into both the deterministic call and the Monte Carlo call below. This is
a single point of truth: 03_01_flowdriven.py and 03_02_adjustedflows.py read this
stage's `flows_df["inflow"]` and only DISAGGREGATE it (Liquids -> Diesel/Petrol, then
into segments) -- they never recompute it -- so a 0 here becomes a genuine 0
downstream automatically, with no override logic duplicated in those files. Nothing
is hidden: `diag_df` gains a new `"inflow_pre_hard_zero_override"` column showing
what the raw residual would have been every year, whether or not the override is
active for that drivetrain/year. See `stockflow_model.py` and `params_schema.py`'s
`StockFlowParams.hard_zero_inflow_from_year_by_drv` for the full mechanism and
rationale, and `diagnose_02_liquids_2050_bump.py` for the diagnostic that found this.

[NEW, step 3 of the agreed plan] hard_zero_inflow_until_year_by_drv -- the mirror
image, for real introduction years
--------------------------------------------------------------------------------------
Same principle, opposite direction: BEV was first sold in Europe in 2011 -- there
were no BEV registrations before then. REMIND's own target_stock trajectory for BEV
nonetheless implies a nonzero residual inflow before 2011 (the model has no notion
of "this drivetrain didn't exist yet"), which is exactly the same kind of modeling
artifact as the Liquids 2050+ phantom inflow, just at the START of a drivetrain's
history instead of after a real-world cutoff.

`params.stock_flow.hard_zero_inflow_until_year_by_drv` (default `{"BEV": 2011}`) is
threaded into both the deterministic call and the Monte Carlo call below, alongside
`hard_zero_inflow_from_year_by_drv` -- the two are independent and OR'd together
inside `_run_cohort_recurrence`, so a drivetrain could in principle have both a start
and an end cutoff. Same single-point-of-truth propagation to stage 03 as
`hard_zero_inflow_from_year_by_drv` (03_01/03_02 only disaggregate this stage's
`flows_df["inflow"]`, never recompute it). See `stockflow_model.py` and
`params_schema.py`'s `StockFlowParams.hard_zero_inflow_until_year_by_drv` for the
full mechanism.

[NEW, this round] inflow_mode_by_drv / inflow_phaseout_by_drv -- 3-way inflow
resolution policy, and the new "inflow_phaseout" mechanism
--------------------------------------------------------------------------------------
`params.stock_flow.inflow_mode_by_drv` selects, per drivetrain, one of three ways
this stage resolves that drivetrain's annual inflow (see `StockFlowParams.
inflow_mode_by_drv`'s own docstring in params_schema.py for the full rationale):
  - "remind_literal"  -- track REMIND's target as closely as possible: forces
                         negative_inflow_policy="clip_to_target" and disables both
                         hard_zero overrides for this drivetrain.
  - "remind_soft"     -- (default for every drivetrain not listed) today's existing
                         behavior, unchanged: the global negative_inflow_policy plus
                         this drivetrain's own hard_zero_inflow_from_year_by_drv /
                         hard_zero_inflow_until_year_by_drv overrides.
  - "inflow_phaseout"  -- [NEW MECHANISM] from `inflow_phaseout_by_drv[drivetrain]`'s
                         start year onward (until that drivetrain's own hard-zero
                         cutoff, if any), inflow is capped at a maximum SHARE of that
                         year's TOTAL EU inflow across every other drivetrain, instead
                         of being solved as REMIND's residual -- stock becomes a pure
                         OUTPUT of natural Weibull attrition on however much inflow the
                         cap allowed in. Implemented below as a genuine TWO-PASS
                         computation per section (deterministic and Monte Carlo each
                         have their own two/three-pass block): pass 1 learns every
                         drivetrain's own NATURAL (uncapped) inflow; a later pass
                         re-runs only the "inflow_phaseout" drivetrains with the
                         computed cap applied. This guarantees the cap computation
                         never depends on the order drivetrains happen to be visited
                         in, even when two phaseout drivetrains are active at once.
Default `{"Liquids": "remind_soft"}` (in params_schema.py) is BYTE-IDENTICAL to
today's actual behavior -- nothing changes for any drivetrain unless its
`inflow_mode_by_drv` entry is edited to `"remind_literal"` or `"inflow_phaseout"`.
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
from matplotlib.ticker import MaxNLocator


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.artifacts import load_many, save_many, artifact_status  # type: ignore
from src.config import get_paths  # type: ignore
from src.monte_carlo import (  # type: ignore
    summarize_distribution, sum_by_period, sensitivity_correlations, plot_tornado,
)
from src.stockflow_model import (  # type: ignore
    build_backcast_state, run_cohort_survival_model,
    # [NEW] The whole Monte Carlo cohort orchestration (target sampling + natural run
    # + phase-out cap) lives in src/ so stage 03_01 reproduces it by CALLING it rather
    # than re-deriving it -- three silent divergences came from the latter. Same reason
    # `resolve_inflow_mode_settings` moved out of this file: the deterministic loop
    # below and both Monte Carlo consumers must resolve the modes identically.
    resolve_inflow_mode_settings, run_stage02_cohort_monte_carlo,
)
# (The asymmetric-spread resolution and Triangular sampling this stage used to do
# inline now happen inside `run_stage02_cohort_monte_carlo`, so `cohort_flow_mc` is no
# longer imported here.)


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
    label_by_drv: dict[str, str] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    The direct visual for the core mechanism of THIS stage: inflow (top, can go
    negative -- see MATH_MODELS.md §2.3) and outflow, split into survival vs. excess
    (bottom, stacked), per drivetrain in `region`. A negative dip in the top panel is
    exactly a negative-inflow year; a nonzero orange band in the bottom panel is
    "out_excess" from the "clip_to_target" policy (always zero under "report_only").

    `label_by_drv`: [NEW] optional {drivetrain: plain-language phrase describing
    what's actually governing its inflow this run}, built by `main()` via
    `_describe_inflow_behavior()`. When supplied, each legend entry reads
    "{drivetrain} -- {phrase}" (e.g. "Liquids -- capped at 10% of everyone else's
    inflow, starting 2035, forced to zero from 2050") instead of just the bare
    drivetrain name -- so which real-world rule is active for a given line is
    readable directly off the chart, with no need to check params_schema.py.
    `None` (default) preserves the old bare-name-only labels.

    [DISPLAY-ONLY FIX, this round] The first plotted year is t0+1, not t0. Confirmed
    directly in `src/stockflow_model.py`'s `_run_cohort_recurrence`: its loop is
    `for i_t in range(1, n_t)`, so row 0 (year == t0, the backcast base year) of
    `flows_df["inflow"]`/`["out_survival"]`/`["out_excess"]` is never computed -- it
    stays at its zero-initialized value for EVERY drivetrain. This is the model
    treating t0 as the initial cohort-by-age snapshot, not a simulated flow year --
    not a data problem specific to any one drivetrain (it's just invisible for
    drivetrains whose real t0 magnitude happens to already be near zero, and glaring
    for one whose t0 magnitude is large, e.g. Liquids). Per explicit instruction: the
    model and `matrices_by_key`/`flows_df` themselves are NOT touched -- t0's row is
    still there, unchanged, in the underlying data and in every saved artifact. Only
    this chart's x-range is cropped to start one year later, so the zero-by-
    construction point is never drawn as if it were a real value.
    """
    keys = sorted(k for k in matrices_by_key if k[0] == region)
    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    colors = plt.cm.tab10.colors

    for i, key in enumerate(keys):
        drivetrain = key[1]
        flows = matrices_by_key[key]["flows_df"].iloc[1:]  # drop t0 -- see docstring
        color = colors[i % len(colors)]
        tag = label_by_drv.get(drivetrain, "") if label_by_drv else ""
        label = f"{drivetrain} ({tag})" if tag else drivetrain
        # [FIXED, this round] Plot the FLOORED value (never below 0 -- real new
        # registrations can't be negative), not the raw residual. The raw
        # residual can dip negative before a hard-zero year kicks in (see
        # stockflow_model.py's docstring) -- that's a real, deliberate part of
        # the underlying data/model and is NOT changed here, only what this one
        # display draws. `matrices_by_key["flows_df"]["inflow"]` itself, the
        # saved artifact stage 03 reads, is untouched.
        ax_in.plot(flows.index, flows["inflow"].clip(lower=0.0), color=color, linewidth=1.6, label=label)

    ax_in.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_in.set_title(f"Annual inflow by drivetrain ({region})", fontsize=12)
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
        flows = matrices_by_key[focus_key]["flows_df"].iloc[1:]  # drop t0 -- see docstring
        # [FIXED, this round] "out_excess" is exactly 0 for every year whenever
        # negative_inflow_policy="report_only" (the default) -- it only becomes
        # nonzero under "clip_to_target". Previously always listed in the
        # legend regardless, so the legend showed an orange entry that could
        # never actually appear on the chart -- confusing ("where is the
        # orange?"). Now: only plot/label out_excess as its own stacked layer
        # if it's ever actually nonzero this run; otherwise just show
        # out_survival alone, with no leftover legend entry for a layer that
        # isn't there.
        has_excess = bool((flows["out_excess"].abs() > 1e-9).any())
        if has_excess:
            ax_out.stackplot(
                flows.index, flows["out_survival"], flows["out_excess"],
                labels=["out_survival", "out_excess"], colors=["#4a7fb5", "#e0793c"],
            )
            title_note = ""
        else:
            ax_out.stackplot(
                flows.index, flows["out_survival"],
                labels=["out_survival"], colors=["#4a7fb5"],
            )
            title_note = " (out_excess=0)"
        ax_out.set_title(f"Outflow breakdown: {focus_drv}{title_note}", fontsize=12)
    ax_out.set_xlabel("Year")
    ax_out.set_ylabel("Outflow [million/year]")
    ax_out.grid(True, linestyle="--", alpha=0.3)
    ax_out.spines["top"].set_visible(False)
    ax_out.spines["right"].set_visible(False)
    ax_out.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
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

    [SHORTENED, this round -- the previous version repeated a full sentence for
    every single drivetrain, making the legend unreadably long] Solid = what the
    simulation actually uses. Dotted = REMIND's original number, reference only,
    never used once a hard-zero/phase-out override is active. Said ONCE, in the
    title, instead of twice per drivetrain -- the legend itself just lists
    drivetrain names (solid) plus one shared "REMIND reference" entry for every
    dotted line, so it stays short no matter how many drivetrains are plotted.
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
        ax.plot(modeled.index, modeled.values, color=color, linewidth=1.6, label=drivetrain)
        ax.plot(target.index, target.values, color=color, linewidth=1.2, linestyle=":")

    # One shared proxy entry for every dotted "reference" line, instead of
    # repeating "REMIND's original number" once per drivetrain.
    ax.plot([], [], color="gray", linewidth=1.2, linestyle=":", label="REMIND reference (not used)")

    ax.set_title(f"Modeled stock vs. target ({region}): solid = used, dotted = REMIND reference", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stock [million]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    return fig, ax


def _describe_inflow_behavior(
    drivetrain: str,
    mode_by_drv: dict[str, str],
    hard_zero_from_by_drv: dict[str, int],
    hard_zero_until_by_drv: dict[str, int],
    phaseout_by_drv: dict[str, tuple[int, float]],
) -> str:
    """
    [SHORTENED, this round -- the original full-sentence version made legends
    unreadably long and was cut] Returns a SHORT tag for one drivetrain's
    legend entry, or "" if nothing special is happening to it (a drivetrain
    with no tag just follows REMIND's target directly -- the ordinary case,
    not worth calling out on every single line). Only drivetrains with an
    actual exception (a hard-zero override or a phase-out cap) get a tag at
    all -- this is what keeps the legend short: most drivetrains show up as a
    bare name, only the ones that matter get annotated.
    """
    mode = mode_by_drv.get(drivetrain, "remind_soft")
    hz_from = hard_zero_from_by_drv.get(drivetrain)
    hz_until = hard_zero_until_by_drv.get(drivetrain)

    # [REWORDED, this round] "0 from {year}" read as "the line visibly drops to
    # 0 exactly at {year}" -- misleading whenever the floored value already sat
    # at 0 earlier for other reasons (the ordinary case). "guaranteed" makes
    # clear this is a floor/rule that holds from that year on, not necessarily
    # a visible change at that exact year.
    if mode == "inflow_phaseout" and drivetrain in phaseout_by_drv:
        start_year, max_share = phaseout_by_drv[drivetrain]
        tag = f"cap {max_share:.0%}→{start_year}"
        if hz_from is not None:
            tag += f", guaranteed 0 from {hz_from}"
        return tag

    if hz_from is not None and hz_until is not None:
        return f"guaranteed 0 before {hz_until} & from {hz_from}"
    if hz_from is not None:
        return f"guaranteed 0 from {hz_from}"
    if hz_until is not None:
        return f"guaranteed 0 before {hz_until}"
    return ""


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
    HARD_ZERO_INFLOW_FROM_YEAR_BY_DRV = p02.hard_zero_inflow_from_year_by_drv
    # [NEW] Real, external policy override (e.g. an actual ICE-sales ban) -- NOT a
    # mathematical correction of the residual-inflow formula. See
    # params_schema.py's StockFlowParams.hard_zero_inflow_from_year_by_drv for the
    # full rationale. `.get(drivetrain)` returns None (no override) for every
    # drivetrain not explicitly listed there -- byte-identical to before this
    # mechanism existed.
    HARD_ZERO_INFLOW_UNTIL_YEAR_BY_DRV = p02.hard_zero_inflow_until_year_by_drv
    # [NEW, step 3 of the agreed plan] The mirror image, e.g. BEV first sold in
    # Europe in 2011 -- REMIND's target_stock trajectory implies a nonzero BEV
    # inflow before then, a modeling artifact with no basis in reality. See
    # params_schema.py's StockFlowParams.hard_zero_inflow_until_year_by_drv.
    # NOTE (organizational, unchanged from earlier review): `p02` also has
    # `unknown_whereabouts_share` and `export_share_by_drv`, neither read here --
    # both are actually consumed by stage 03. Declared under stock_flow for historical
    # reasons; not moved this round since stage 03 isn't fixed yet.

    # [NEW] 3-way per-drivetrain inflow resolution policy -- see
    # `StockFlowParams.inflow_mode_by_drv`'s own docstring for the full rationale,
    # and `stockflow_model.resolve_inflow_mode_settings()` for how it's applied below.
    # `.get(drivetrain, "remind_soft")` at every call site means a drivetrain not
    # listed here behaves exactly as it did before this field existed.
    INFLOW_MODE_BY_DRV = p02.inflow_mode_by_drv
    # [NEW] Only consulted for a drivetrain whose INFLOW_MODE_BY_DRV entry is
    # "inflow_phaseout" -- {drivetrain: (phaseout_start_year, max_share_of_total)}.
    # See StockFlowParams.inflow_phaseout_by_drv's docstring for the cap derivation.
    INFLOW_PHASEOUT_BY_DRV = p02.inflow_phaseout_by_drv
    # [NEW] Optional per-drivetrain Monte-Carlo uncertainty on the phase-out cap's
    # own max_share (deterministic run always uses the point value from
    # INFLOW_PHASEOUT_BY_DRV -- no uncertainty there, consistent with every other
    # point estimate in the deterministic path). `getattr` with a `{}` default so
    # this works whether or not the field exists yet in params_schema.py.
    INFLOW_PHASEOUT_MAX_SHARE_TRIANGULAR_BY_DRV = getattr(
        p02, "inflow_phaseout_max_share_triangular_by_drv", {}
    )

    init_max_age = p02.init_max_age
    results_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    matrices_by_key: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}

    # -------------------------------------------------------------------------
    # [NEW] Two-pass computation, replacing the old single pass, needed to support
    # "inflow_phaseout" mode. Pass 1 runs EVERY (region, drivetrain) key with its
    # resolved pass-1 settings (`resolve_inflow_mode_settings`) -- for
    # "remind_soft" and "remind_literal" keys this IS the final result, byte-
    # identical to the old single-pass loop. Pass 2 re-runs ONLY "inflow_phaseout"
    # keys, with a computed per-year cap applied, using every OTHER same-region
    # key's pass-1 NATURAL inflow as the "everyone else" total in the cap formula.
    # -------------------------------------------------------------------------
    pass1_state_by_key: dict[tuple[str, str], dict] = {}
    natural_inflow_by_key: dict[tuple[str, str], dict[int, float]] = {}

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

        _mode, pass1_policy, pass1_from, pass1_until = resolve_inflow_mode_settings(
            drivetrain, INFLOW_MODE_BY_DRV, negative_inflow_policy,
            HARD_ZERO_INFLOW_FROM_YEAR_BY_DRV, HARD_ZERO_INFLOW_UNTIL_YEAR_BY_DRV,
        )
        out = run_cohort_survival_model(
            stock_series=stock_series,
            model_end_year=model_end_year,
            drivetrain=drivetrain,
            base_shape_k=base.shape_k,
            base_scale_lambda=base.scale_lambda,
            lifetime_override=LIFETIME_OVERRIDE_BY_DRV.get(drivetrain),
            backcast=backcast,
            negative_inflow_policy=pass1_policy,
            hard_zero_inflow_from_year=pass1_from,
            hard_zero_inflow_until_year=pass1_until,
        )
        pass1_state_by_key[key] = {
            "out": out, "backcast": backcast, "base": base, "stock_series": stock_series,
        }
        # Natural inflow for the cap formula below -- floored at 0 (a negative
        # residual isn't a meaningful "share of total inflow" input; this mirrors
        # the same floor `run_cohort_survival_monte_carlo`'s own
        # `inflow_applied_by_year` already applies per draw).
        natural_inflow_by_key[key] = out["flows_df"]["inflow"].clip(lower=0.0).to_dict()

    # --- Pass 2: apply the phase-out cap, only for "inflow_phaseout" keys.
    for key, df in stock_dict.items():
        region, drivetrain = key
        mode = INFLOW_MODE_BY_DRV.get(drivetrain, "remind_soft")
        if mode != "inflow_phaseout":
            continue
        start_year, max_share = INFLOW_PHASEOUT_BY_DRV[drivetrain]
        hard_zero_year = HARD_ZERO_INFLOW_FROM_YEAR_BY_DRV.get(drivetrain)
        own_natural = natural_inflow_by_key[key]
        other_keys = [k for k in stock_dict if k != key and k[0] == region]

        # cap(t) = other_total(t) * max_share / (1 - max_share) -- the value that
        # makes this drivetrain's SELF-INCLUSIVE share of the true combined total
        # equal exactly `max_share`. See StockFlowParams.inflow_phaseout_by_drv's
        # docstring in params_schema.py for the derivation.
        override_by_year: dict[int, float] = {}
        for year, natural_value in own_natural.items():
            if year < start_year:
                continue
            if hard_zero_year is not None and year >= hard_zero_year:
                continue  # hard_zero wins downstream regardless -- no cap needed
            other_total = sum(natural_inflow_by_key[ok].get(year, 0.0) for ok in other_keys)
            cap = other_total * max_share / (1.0 - max_share)
            if natural_value > cap:
                override_by_year[year] = cap

        if override_by_year:
            saved = pass1_state_by_key[key]
            out = run_cohort_survival_model(
                stock_series=saved["stock_series"],
                model_end_year=model_end_year,
                drivetrain=drivetrain,
                base_shape_k=saved["base"].shape_k,
                base_scale_lambda=saved["base"].scale_lambda,
                lifetime_override=LIFETIME_OVERRIDE_BY_DRV.get(drivetrain),
                backcast=saved["backcast"],
                negative_inflow_policy=negative_inflow_policy,
                hard_zero_inflow_from_year=HARD_ZERO_INFLOW_FROM_YEAR_BY_DRV.get(drivetrain),
                hard_zero_inflow_until_year=HARD_ZERO_INFLOW_UNTIL_YEAR_BY_DRV.get(drivetrain),
                inflow_override_by_year=override_by_year,
            )
            pass1_state_by_key[key]["out"] = out

    for key in stock_dict:
        out = pass1_state_by_key[key]["out"]
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
    fig_dir = get_paths(start=PROJECT_ROOT).figures      # the one folder, src/config.py
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "02_1_stock_vs_target.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Diagnostic plot 2: inflow/outflow by drivetrain -- the core mechanism of this
    # stage, including exactly where/how much inflow goes negative.
    # -----------------------------------------------------------------------
    # [NEW] Plain-language legend labels -- see _describe_inflow_behavior() above.
    label_by_drv = {
        drv: _describe_inflow_behavior(
            drv, INFLOW_MODE_BY_DRV, HARD_ZERO_INFLOW_FROM_YEAR_BY_DRV,
            HARD_ZERO_INFLOW_UNTIL_YEAR_BY_DRV, INFLOW_PHASEOUT_BY_DRV,
        )
        for drv in {k[1] for k in matrices_by_key}
    }
    fig, _ = plot_flows_by_drivetrain(matrices_by_key, region="EUR", label_by_drv=label_by_drv)
    fig_path = fig_dir / "02_2_inflow_outflow_by_drivetrain.png"
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
    # `02_stock_vs_target_check.png` (the DETERMINISTIC plot, built from the single
    # point-estimate run above, never from this MC block) always shows modeled stock
    # exactly equal to the REMIND-prescribed target -- that's what "stock-driven"
    # means for the deterministic run, unaffected by anything below.
    # [FIXED] This comment previously claimed that invariant held "regardless of
    # lifetime assumptions" and "always", full stop -- true when it was written
    # (stock had no uncertainty mechanism of any kind), no longer true
    # unconditionally now that `StockFlowParams.stock_target_relative_spread` /
    # `.stock_target_uncertainty_start_year` exist: for years BEFORE the cutoff
    # year, modeled stock still equals target exactly, every draw (verified:
    # `abs(modeled_stock - target) < 1e-6`, same as before). For years AT/AFTER
    # the cutoff, the target itself is now genuinely per-draw-varying (see
    # `stock_target_draws`/`result["stock_target_by_year"]` below), so modeled
    # stock varies across draws too, by design -- that's the whole point of the
    # feature. `summary_by_drv[drv]["stock_by_year_band"]` is where that shows up.
    # `02_flows_by_drivetrain_check.png`'s underlying quantities (inflow, outflow) DO
    # genuinely vary with lifetime uncertainty -- this block now tracks that variation
    # YEAR BY YEAR (not just a single 2070 total) and plots it directly against the
    # deterministic flows chart's own style, so the two are actually comparable.
    #
    # [NEW, this round] The same "inflow_phaseout" two-pass idea from the
    # deterministic loop above applies here too, restructured as THREE passes
    # (see the block below): pass 1 (natural run, every drivetrain, plus per-draw
    # cap-uncertainty sampling for phaseout drivetrains), pass 2 (cap + re-run,
    # phaseout drivetrains only), pass 3 (finalize -- IDENTICAL summarization logic
    # to the single-pass loop this replaces, just reading from the dict pass 1/2
    # populated instead of a freshly-computed `result`).
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
        drivetrains_present = sorted({drv for (_, drv) in stock_dict.keys()})
        # All Monte Carlo seeding now happens inside
        # `run_stage02_cohort_monte_carlo` (it owns the whole spawn order, which is
        # what lets stage 03_01 land on identical draws) -- nothing to seed here.

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

        # (Every `mc_*_by_drv` name below is bound from
        # `run_stage02_cohort_monte_carlo`'s return value, so none is pre-declared.)

        # =====================================================================
        # PASSES 0-2: target sampling, the natural cohort run, and the phase-out cap.
        #
        # All of it lives in `stockflow_model.run_stage02_cohort_monte_carlo` rather
        # than here, because `03_01_flowdriven.py` has to reproduce these results
        # EXACTLY (it layers its own collected/export/unknown split onto these same
        # draws, and regenerates the per-year arrays that are too large to persist).
        # While that sequence was written down in both places it drifted apart three
        # separate times, silently -- see that function's docstring for the specifics.
        # It is now written down once. Do not inline any of it back here.
        # =====================================================================
        _mc = run_stage02_cohort_monte_carlo(
            stock_dict=stock_dict,
            drivetrains=drivetrains_present,
            model_end_year=model_end_year,
            init_max_age=init_max_age,
            n_draws=n_draws,
            seed=params.monte_carlo.seed,
            lifetime_by_drv=LIFETIME_BY_DRV,
            lifetime_override_by_drv=LIFETIME_OVERRIDE_BY_DRV,
            lifetime_scale_lambda_relative_spread=p02.lifetime_scale_lambda_relative_spread,
            negative_inflow_policy=negative_inflow_policy,
            hard_zero_inflow_from_year_by_drv=HARD_ZERO_INFLOW_FROM_YEAR_BY_DRV,
            hard_zero_inflow_until_year_by_drv=HARD_ZERO_INFLOW_UNTIL_YEAR_BY_DRV,
            inflow_mode_by_drv=INFLOW_MODE_BY_DRV,
            inflow_phaseout_by_drv=INFLOW_PHASEOUT_BY_DRV,
            inflow_phaseout_max_share_triangular_by_drv=INFLOW_PHASEOUT_MAX_SHARE_TRIANGULAR_BY_DRV,
            stock_target_relative_spread=p02.stock_target_relative_spread,
            stock_target_uncertainty_start_year=p02.stock_target_uncertainty_start_year,
            stock_target_ramp_max_rate_per_year=p02.stock_target_ramp_max_rate_per_year,
            stock_target_correlated_mix=p02.stock_target_correlated_mix,
            total_fleet_relative_spread=p02.total_fleet_relative_spread,
        )
        mc_result_by_drv = _mc.results
        mc_scale_lambda_draws_by_drv = _mc.scale_lambda_draws
        mc_stock_target_draws_by_drv = _mc.targets.stock_target_draws
        mc_stock_target_mult_draws_by_drv = _mc.targets.stock_target_mult_draws

        for _drv in drivetrains_present:
            sensitivity_input_draws[f"{_drv}_scale_lambda"] = _mc.scale_lambda_draws[_drv]
            sensitivity_input_draws[f"{_drv}_stock_target_mult"] = (
                _mc.targets.stock_target_mult_draws[_drv]
            )
        if _mc.targets.fleet_mult_draws is not None:
            sensitivity_input_draws["TOTAL_FLEET_mult"] = _mc.targets.fleet_mult_draws

        # =====================================================================
        # PASS 3: finalize -- IDENTICAL summarization logic to the single-pass
        # loop this replaces (regression-safe: for every drivetrain in
        # "remind_soft"/"remind_literal" mode, `mc_result_by_drv[drivetrain]` is
        # exactly what a single-pass loop would have produced). Only change from
        # the original: reads `result`/`scale_lambda_draws`/`stock_target_mult_
        # draws` from the dicts pass 1/2 populated, instead of computing them
        # inline in this same loop.
        # =====================================================================
        for drivetrain in drivetrains_present:
            result = mc_result_by_drv[drivetrain]
            scale_lambda_draws = mc_scale_lambda_draws_by_drv[drivetrain]
            stock_target_mult_draws = mc_stock_target_mult_draws_by_drv[drivetrain]

            draws_by_drv[drivetrain] = {
                "scale_lambda": scale_lambda_draws,
                "stock_target_mult": stock_target_mult_draws,
                "cumulative_inflow": result["cumulative_inflow"],
                "cumulative_out_survival": result["cumulative_out_survival"],
            }

            # Per-YEAR uncertainty band (median, P2.5, P97.5) -- computed from the
            # full (n_years, n_draws) arrays, but only the tiny summarized band (one
            # triple of numbers PER YEAR, not per draw) is kept/saved.
            #
            # [FIXED, renamed] TWO bands are now kept, not one:
            #   - "inflow_applied_by_year_band": from `inflow_applied_by_year`
            #     (always >= 0 per draw, floored BEFORE the percentile is taken --
            #     the mathematically correct order). This is what actually got
            #     simulated -- used for the main flows-over-time plot and period
            #     sums (matches `cumulative_inflow`'s own convention).
            #   - "inflow_raw_by_year_band": from `inflow_by_year` (the unfloored
            #     residual, can be negative -- deliberately preserved, same
            #     diagnostic role as `02_flows_by_drivetrain_check.png`'s
            #     deterministic counterpart: a negative-residual year should never
            #     be silently invisible). Was previously computed under the OLD key
            #     name `inflow_by_year_band` and used (incorrectly, pre-fix) for the
            #     main plot -- now kept only as the explicit diagnostic band.
            years_list = result["t"].tolist()
            inflow_applied_band = {
                "years": years_list,
                "p2_5": np.percentile(result["inflow_applied_by_year"], 2.5, axis=1).tolist(),
                "median": np.percentile(result["inflow_applied_by_year"], 50, axis=1).tolist(),
                "p97_5": np.percentile(result["inflow_applied_by_year"], 97.5, axis=1).tolist(),
            }
            inflow_raw_band = {
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
            # [NEW] Per-year STOCK band -- previously impossible to report ("stock
            # is deterministic here" was a true statement before this round's
            # stock-target uncertainty feature). Degenerates to a flat
            # p2_5==median==p97_5 line for every year before `stock_target_
            # uncertainty_start_year` (byte-identical values across draws there,
            # same as `stock_t` always was), genuinely widens from that year on.
            stock_band = {
                "years": years_list,
                "p2_5": np.percentile(result["stock_target_by_year"], 2.5, axis=1).tolist(),
                "median": np.percentile(result["stock_target_by_year"], 50, axis=1).tolist(),
                "p97_5": np.percentile(result["stock_target_by_year"], 97.5, axis=1).tolist(),
            }

            summary_by_drv[drivetrain] = {
                "cumulative_inflow": summarize_distribution(result["cumulative_inflow"]),
                "cumulative_out_survival": summarize_distribution(result["cumulative_out_survival"]),
                "inflow_applied_by_year_band": inflow_applied_band,
                "inflow_raw_by_year_band": inflow_raw_band,
                "out_survival_by_year_band": out_survival_band,
                "stock_by_year_band": stock_band,
            }
            s = summary_by_drv[drivetrain]["cumulative_out_survival"]
            print(
                f"Monte Carlo [{drivetrain}]: {n_draws:,} draws, cumulative_out_survival -- "
                f"mean={s['mean']:.2f}, median={s['median']:.2f}, mode={s['mode']:.2f}, "
                f"std={s['std']:.2f}, P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
            )

            # --- [NEW] period-based summaries for this drivetrain ---
            # [FIXED] Was `result["inflow_by_year"]` (raw residual) -- inconsistent
            # with `cumulative_inflow` above, which sums `inflow_applied` (>= 0 per
            # draw). A negative-residual year would previously make a period's
            # summed inflow LOWER than what was actually simulated for that period,
            # and could even make a period sum negative. `inflow_applied_by_year`
            # matches `cumulative_inflow`'s own convention exactly.
            inflow_period_sums = sum_by_period(result["inflow_applied_by_year"], result["t"], output_periods)
            out_survival_period_sums = sum_by_period(result["out_survival_by_year"], result["t"], output_periods)
            drv_period_summary: dict[tuple[int, int], dict] = {}
            for period in output_periods:
                start, end = period
                mask = (result["t"] >= start) & (result["t"] <= end)
                years_in_period = result["t"][mask]
                # [FIXED] Was `result["stock_t"][mask]` (deterministic, `(n_years_
                # in_period,)`) wrapped in `np.full(n_draws, scalar)` below to fake
                # a per-draw shape for `summarize_distribution` -- that broadcast
                # trick was ALWAYS there specifically because stock genuinely was
                # identical across every draw before this round's stock-target
                # uncertainty feature. `stock_target_by_year` is the real per-draw
                # array now (still byte-identical across draws for any year before
                # `stock_target_uncertainty_start_year`, genuinely varying from it
                # on) -- already the right `(n_years_in_period, n_draws)` shape, no
                # broadcast trick needed.
                stock_values_in_period = result["stock_target_by_year"][mask]  # (n_years_in_period, n_draws)

                eu_total_period_sums[period]["cumulative_inflow"] += inflow_period_sums[period]
                eu_total_period_sums[period]["cumulative_out_survival"] += out_survival_period_sums[period]

                drv_period_summary[period] = {
                    "cumulative_inflow": summarize_distribution(inflow_period_sums[period]),
                    "cumulative_out_survival": summarize_distribution(out_survival_period_sums[period]),
                    "stock_end_of_period": summarize_distribution(
                        stock_values_in_period[-1] if stock_values_in_period.size else np.full(n_draws, np.nan)
                    ),
                    "stock_sum_over_period": summarize_distribution(stock_values_in_period.sum(axis=0)),
                    "stock_per_year": {
                        int(y): summarize_distribution(stock_values_in_period[i])
                        for i, y in enumerate(years_in_period)
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
        fig_path_tornado = fig_dir / "02_5_montecarlo_sensitivity.png"
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
            inb = summary_by_drv[drivetrain]["inflow_applied_by_year_band"]
            oub = summary_by_drv[drivetrain]["out_survival_by_year_band"]
            # [DISPLAY-ONLY FIX, this round -- same cause as plot_flows_by_drivetrain()
            # above] inb["years"][0]/oub["years"][0] is t0; inflow_by_year/
            # out_survival_by_year are always exactly 0.0 at t0 for every draw (same
            # _run_cohort_recurrence loop, starts at i_t=1 -- see stockflow_model.py).
            # `summary_by_drv` (the saved artifact, used by stage 03) is NOT touched --
            # only the slice used for THIS plot skips index 0.
            ax_in.plot(inb["years"][1:], inb["median"][1:], color=color, linewidth=1.6, label=drivetrain)
            ax_in.fill_between(inb["years"][1:], inb["p2_5"][1:], inb["p97_5"][1:], color=color, alpha=0.2)
            ax_out.plot(oub["years"][1:], oub["median"][1:], color=color, linewidth=1.6, label=drivetrain)
            ax_out.fill_between(oub["years"][1:], oub["p2_5"][1:], oub["p97_5"][1:], color=color, alpha=0.2)

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
        fig_path = fig_dir / "02_3_montecarlo_inflow_outflow.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path}")

        # -----------------------------------------------------------------------
        # Plot 1b: [REORDERED + RETITLED, this round] SECONDARY/reference check
        # only -- the primary Monte Carlo result is the chart above (Plot 1,
        # 02_3_montecarlo_inflow_outflow.png). This chart exists ONLY to double-
        # check the primary result against REMIND's own untouched number, so the
        # ACTUALLY-USED panel is now shown FIRST (top), and the untouched
        # REMIND reference is shown SECOND (bottom) -- reading top-to-bottom
        # matches "what matters most" first, "for reference" second. For most
        # drivetrains the two panels look identical (no negative-residual years
        # ever occur for them); they only diverge for a drivetrain that actually
        # hit a negative-residual year (Liquids, Hybrid here) -- exactly the
        # years where natural attrition alone outpaces the falling REMIND target.
        # -----------------------------------------------------------------------
        fig, (ax_used, ax_ref) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        for i, drivetrain in enumerate(drivetrains_present):
            color = colors[i % len(colors)]
            rb = summary_by_drv[drivetrain]["inflow_raw_by_year_band"]
            ab = summary_by_drv[drivetrain]["inflow_applied_by_year_band"]
            # Same t0-skip display fix as Plot 1 above -- row 0 is always exactly 0.0.
            ax_used.plot(ab["years"][1:], ab["median"][1:], color=color, linewidth=1.6, label=drivetrain)
            ax_used.fill_between(ab["years"][1:], ab["p2_5"][1:], ab["p97_5"][1:], color=color, alpha=0.2)
            ax_ref.plot(rb["years"][1:], rb["median"][1:], color=color, linewidth=1.6, label=drivetrain)
            ax_ref.fill_between(rb["years"][1:], rb["p2_5"][1:], rb["p97_5"][1:], color=color, alpha=0.2)

        ax_used.axhline(0, color="black", linewidth=0.8)
        ax_used.set_title(
            f"Adjusted inflow: floored at 0, starting in 2050 -- used downstream (n={n_draws:,})",
            fontsize=11,
        )
        ax_used.set_ylabel("Inflow [million/year]")
        ax_used.grid(True, linestyle="--", alpha=0.3)
        ax_used.spines["top"].set_visible(False)
        ax_used.spines["right"].set_visible(False)
        ax_used.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

        ax_ref.axhline(0, color="black", linewidth=0.8)
        ax_ref.set_title(
            "REMIND literal: negative where target falls faster than scrappage",
            fontsize=11,
        )
        ax_ref.set_xlabel("Year")
        ax_ref.set_ylabel("Inflow [million/year]")
        ax_ref.grid(True, linestyle="--", alpha=0.3)
        ax_ref.spines["top"].set_visible(False)
        ax_ref.spines["right"].set_visible(False)
        ax_ref.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fig_path_raw_vs_applied = fig_dir / "02_6_inflow_adjusted_vs_remind_literal.png"
        fig.savefig(fig_path_raw_vs_applied, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path_raw_vs_applied}")

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
            ax.set_xlabel("Cumulative out-survival [million vehicles]")
            ax.set_ylabel("Number of draws")
            # Draw counts are read at a glance -- a dense tick ladder adds nothing.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.legend(frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig_path = fig_dir / "02_4_montecarlo_cumulative_totals.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path}")

    return saved


if __name__ == "__main__":
    main()