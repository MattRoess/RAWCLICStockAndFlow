"""
03_02_adjustedflows.py
========================

Stage 03, part 2: takes `flows_03` (03_01's baseline flow-driven output) and re-runs the
flow-driven model under a series of ALTERNATIVE inflow-composition scenarios (different
drivetrain mixes, different BEV segment-size profiles), to explore sensitivity of the
downstream materials tracker to those assumptions.

======================================================================
MONTE CARLO -- GENUINE lifetime + share uncertainty, fully vectorized. Same
clean pattern as stages 02/03_01 (one shared function, params-driven, no
duplicate implementation) -- and now also product-agnostic underneath.
======================================================================
`flowdriven_model.py`'s own recurrence is now vectorized across a `draws` axis
(`run_flow_driven_model_monte_carlo`, cross-validated byte-for-byte against the
scalar `run_flow_driven_model_with_outflow_disaggregation` function it wraps --
see `test_regression.py`/`test_regression_generic.py`). The actual math lives
in `cohort_flow_mc.py`, a product-agnostic vectorized cohort-flow engine (no
vehicle/drivetrain terminology hardcoded there) that other pipelines/products
can build their own thin wrappers around, the same way this one does.

EACH of the 11 scenarios re-simulates its OWN lifetime and share assumptions
per draw -- not a post-hoc re-split of a fixed deterministic total. This
happens INSIDE `run_adjusted_scenario` itself (see its `monte_carlo_enabled`
parameter): the exact same `inflow_df`, `lifetime_change_by_drv`,
`stock_modifier_2027`, and share dicts used for that scenario's deterministic
run are reused for its Monte Carlo run, just with `scale_lambda`,
`unknown_whereabouts_share`, and `export_share` resampled per draw. This
stage's final block only AGGREGATES the 11 already-computed per-scenario
results into summary stats and the comparison plot.

Uncertainty spreads come from `params.stock_flow` (`lifetime_scale_lambda_
relative_spread`, `unknown_whereabouts_share_std`, `export_share_std`) --
the SAME fields stage 02 already uses, nothing hardcoded here. `scale_lambda`
is sampled from a TRIANGULAR distribution, `Triangular(point*(1-spread),
point, point*(1+spread))` -- matching `params_schema.py`'s own documented
convention for lifetime uncertainty (NOT Normal, unlike the two share
parameters below). The SAME per-draw multiplier carries through a scenario's
`lifetime_change_by_drv` override (e.g. BEV_longer's scale_lambda=17 from 2027
onward), so a draw that samples "10% longer-lived" stays 10% longer-lived
across that boundary rather than resampling independently -- see
`cohort_flow_mc.py`'s module docstring for the full uncertainty convention.

`lifetime_scale_lambda_relative_spread` can be passed as EITHER a single
float (a "general" scenario -- the same uncertainty spread applied to every
drivetrain, e.g. `params.monte_carlo.stockflow_lifetime_spread`'s 0.15) OR a
dict keyed by drivetrain (a "specific" scenario -- e.g. "BEV's lifetime is
far less certain than Diesel's"), so both kinds of scenario can be modeled
without touching the engine. This stage defaults to the per-drivetrain dict
(`p02.lifetime_scale_lambda_relative_spread`, already keyed by drivetrain in
`params_schema.py`); pass a single float instead at the `run_adjusted_scenario`
/ `main()` level if a "general uncertainty" run is wanted for some analysis.

Saves `mc_stage03_02_summary` and a scenario-comparison box plot
(`03_02_monte_carlo_scenario_comparison.png`).

======================================================================
C6 -- RESOLVED AND VERIFIED THIS ROUND (previously the single most consequential
finding in the entire review).
======================================================================
Previously, five "sensitivity scenario" trackers (`tracker_keyed_BEV_longer`,
`tracker_keyed_ICEV_shorter`, `tracker_keyed_stock_lower`, `tracker_keyed_losses_zero`,
`tracker_keyed_losses_high`) were each a literal, unmodified copy of `tracker_keyed_BAU`
-- the comments describing what SHOULD change were never wired into actual code. The
target values were, however, already fully specified in those same comments:
  - `ICEV_shorter`: Diesel/Petrol `scale_lambda` -> 9.0 (from 2027 onward)
  - `BEV_longer`:   BEV `scale_lambda` -> 17.0 (from 2027 onward)
  - `stock_lower`:  `stock_modifier_2027` -> 0.8
  - `losses_zero`:  BEV `export_share` -> 0, BEV `unknown_whereabouts_share` -> 0
  - `losses_high`:  BEV `unknown_whereabouts_share` -> 0.43, BEV `export_share` -> 0.08

Implemented as five genuine `run_adjusted_scenario(...)` calls, replacing the five
`tracker_keyed_BAU.copy()` lines. `run_adjusted_scenario` was parametrized to accept an
explicit `lifetime_change_by_drv` override (previously silently closed over a
module-level global) so the five scenarios don't step on each other via shared mutable
state.

**VERIFIED, with real numbers, now that `flowdriven_model.py`/`disaggregation.py` are
available** -- this was run end-to-end against synthetic data, not just implemented and
hoped-correct:
  1. All five scenario trackers are confirmed structurally different from
     `tracker_keyed_BAU` (`.equals()` returns `False` for every one).
  2. `losses_zero` specifically checked at the number level: BEV export outflow went
     from 11.583 (BAU) to exactly 0.0; BEV unknown-whereabouts outflow went from 57.916
     (BAU) to exactly 0.0 -- precisely the intended effect, not just "some difference".
  3. The two schema questions from the previous round are now resolved by reading
     `flowdriven_model.py`'s actual source (not guessed): `lifetime_change_by_drv` IS
     open-ended from `start_year` (no `end_year` needed -- confirmed in
     `run_flow_driven_model_with_outflow_disaggregation`'s own docstring), and
     `export_share_by_drivetrain`/`unknown_whereabouts_share` ARE accessed via direct
     indexing (`dict[drivetrain]`, not `.get()`) -- meaning `losses_zero`/`losses_high`
     correctly copy the FULL baseline dict before overriding just BEV, which is exactly
     what this implementation already does.

See `HOW_TO_RUN_AND_VERIFY.md` for the exact verification commands.

======================================================================
THE BIG CAVEAT, still partially relevant: everything is now tested with SYNTHETIC data
======================================================================
Stage 03 has now been run end-to-end (00->01->02->03_01->03_02) and produces correct,
sensible-looking output -- but against synthetic REMIND/export/EEA data, not your real
data. Structural correctness (does the code run, do the scenarios diverge as intended)
is verified; real-world numerical plausibility (do the actual magnitudes make sense for
the EU vehicle fleet) still needs a run against your real data files.

FIXES APPLIED THIS ROUND
--------------------------
- Dataclass params access throughout (`params["02_stock_flow"]` -> `params.stock_flow`,
  etc.), same as `03_01_flowdriven.py`.
- Project-root resolution: upward-searching `_find_project_root`.
- `root=PROJECT_ROOT` threaded into `load_many`/`save_many`.
- C6 (see above): five genuine sensitivity scenarios implemented AND verified,
  replacing literal BAU copies.
- `fdm.build_p02_mapped_inputs` and `run_adjusted_scenario`'s `p02` parameter receive a
  DICT view (`params.to_nested_dict()["02_stock_flow"]`), not the `StockFlowParams`
  dataclass directly -- confirmed necessary by reading `flowdriven_model.py`'s source.
- **[NEW] Integrated diagnostic plot**: `plot_flows_split_collected_unknown_all_trackers`
  (from `src/plotting.py`, now fixed to save instead of forcing `plt.show()`) is called
  automatically at the end of `main()`, saving
  `data/processed/figures/03_02_flows_all_scenarios.png` -- inflow/outflow split for
  all 11 scenarios side by side. This is the most direct visual confirmation that the
  C6 fix produced genuinely different scenarios, not just five more silent BAU copies.

See "EXACT DIFFERENCES FROM 03_01" further down for a full comparison of the two
notebooks' methodology.
"""

from __future__ import annotations

import sys
import time
import math
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never opens an interactive window -- always saves to file
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.flowdriven_model as fdm  # type: ignore
import src.artifacts as artifacts  # type: ignore
import src.plotting as plotting  # type: ignore
from src.disaggregation import (  # type: ignore
    build_tracker_from_disaggregated, add_keys_to_tracker_dict,
)
from src.monte_carlo import summarize_distribution  # type: ignore

load_many = artifacts.load_many
save_many = artifacts.save_many
plot_flows_split_collected_unknown_all_trackers = plotting.plot_flows_split_collected_unknown_all_trackers


# ---------------------------------------------------------------------------
# THE MATH MODEL: inflow-composition "tweak" transforms
# ---------------------------------------------------------------------------
def tweak_inflow_drivetrain_shares(
    inflow_df: pd.DataFrame, *, region: str = "EUR", year_col: str = "year", value_col: str = "value",
    drivetrain_col: str = "Drive Train", segment_col: str = "Segment", region_col: str = "Region",
    scenario_start_year: int = 2026, target_shares_by_year: dict[int, dict[str, float]] | None = None,
    target_shares_final: dict[str, float] | None = None, ramp_end_year: int | None = None,
    preserve_other_regions: bool = True, strict: bool = True,
) -> pd.DataFrame:
    """
    Re-weight inflow across DRIVETRAINS for a given region, holding each year's TOTAL
    inflow fixed, and holding each drivetrain's SEGMENT mix (relative shares within that
    drivetrain) fixed at baseline.

    THE MATH: for year y >= scenario_start_year, either use an explicit
    `target_shares_by_year[y]` (holding the last-specified year's shares for any year
    after it), or linearly interpolate from the baseline share at `scenario_start_year`
    to `target_shares_final` by `ramp_end_year`:

        alpha(y) = (y - scenario_start_year) / (ramp_end_year - scenario_start_year)
        target_share[drv](y) = (1-alpha)*baseline_share[drv](scenario_start_year) + alpha*target_shares_final[drv]

    Each drivetrain's new absolute inflow is `total_inflow(y) * target_share[drv](y)`,
    then redistributed across segments using that drivetrain's ORIGINAL (baseline)
    within-drivetrain segment mix, renormalized to sum to 1 -- i.e. segment composition
    within a drivetrain is untouched by this function; only the drivetrain mix changes.
    `strict=True` requires every target share dict to sum to exactly 1.0 (within 1e-8) and
    forbids negative shares, raising `ValueError` otherwise; `strict=False` renormalizes
    instead of raising.
    """
    required_cols = {region_col, drivetrain_col, segment_col, year_col, value_col}
    missing = required_cols.difference(inflow_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if (target_shares_by_year is None) == (target_shares_final is None):
        raise ValueError("Provide exactly one of target_shares_by_year or target_shares_final.")
    if target_shares_final is not None and ramp_end_year is None:
        raise ValueError("ramp_end_year is required when using target_shares_final.")

    df = inflow_df.copy()
    df[year_col] = df[year_col].astype(int)
    df[value_col] = df[value_col].astype(float)

    mask_region = df[region_col].eq(region)
    df_region = df.loc[mask_region].copy()
    df_other = df.loc[~mask_region].copy()
    if df_region.empty:
        raise ValueError(f"No rows found for region={region!r}.")

    df_region = (df_region.groupby([region_col, drivetrain_col, segment_col, year_col], as_index=False)[value_col]
                 .sum().sort_values([year_col, drivetrain_col, segment_col]).reset_index(drop=True))

    total_by_year = df_region.groupby(year_col, as_index=False)[value_col].sum().rename(columns={value_col: "total_inflow_year"})
    drv_totals = df_region.groupby([year_col, drivetrain_col], as_index=False)[value_col].sum().rename(columns={value_col: "drv_total"})

    seg_mix = df_region.merge(drv_totals, on=[year_col, drivetrain_col], how="left")
    seg_mix["seg_share_within_drv"] = np.where(seg_mix["drv_total"] > 0, seg_mix[value_col] / seg_mix["drv_total"], 0.0)

    baseline_drv_shares = drv_totals.merge(total_by_year, on=year_col, how="left")
    baseline_drv_shares["drv_share"] = np.where(baseline_drv_shares["total_inflow_year"] > 0, baseline_drv_shares["drv_total"] / baseline_drv_shares["total_inflow_year"], 0.0)

    drivetrains = sorted(df_region[drivetrain_col].dropna().unique().tolist())
    years = sorted(df_region[year_col].unique().tolist())

    def _validate_or_normalize_share_dict(share_dict: dict[str, float]) -> dict[str, float]:
        out = {drv: float(share_dict.get(drv, 0.0)) for drv in drivetrains}
        total = sum(out.values())
        if strict:
            if any(v < 0 for v in out.values()):
                raise ValueError(f"Negative share found: {out}")
            if not np.isclose(total, 1.0, atol=1e-8):
                raise ValueError(f"Shares must sum to 1. Got {total:.12f} for {out}")
            return out
        if total <= 0:
            raise ValueError("Share total must be positive.")
        return {k: v / total for k, v in out.items()}

    baseline_share_map: dict[int, dict[str, float]] = {}
    for y in years:
        tmp = baseline_drv_shares[baseline_drv_shares[year_col] == y]
        baseline_share_map[y] = {drv: float(tmp.loc[tmp[drivetrain_col] == drv, "drv_share"].sum()) for drv in drivetrains}

    target_share_map: dict[int, dict[str, float]] = {}
    if target_shares_by_year is not None:
        for y in years:
            if y < scenario_start_year:
                target_share_map[y] = baseline_share_map[y]
            elif y in target_shares_by_year:
                target_share_map[y] = _validate_or_normalize_share_dict(target_shares_by_year[y])
            else:
                specified_years = sorted(k for k in target_shares_by_year if k <= y)
                target_share_map[y] = _validate_or_normalize_share_dict(target_shares_by_year[specified_years[-1]]) if specified_years else baseline_share_map[y]
    else:
        target_final = _validate_or_normalize_share_dict(target_shares_final)
        for y in years:
            if y < scenario_start_year:
                target_share_map[y] = baseline_share_map[y]
                continue
            if y >= ramp_end_year:
                target_share_map[y] = target_final
                continue
            base_start = baseline_share_map.get(scenario_start_year)
            if base_start is None:
                raise ValueError(f"scenario_start_year={scenario_start_year} not found in inflow_df.")
            alpha = (y - scenario_start_year) / (ramp_end_year - scenario_start_year)
            target_share_map[y] = {drv: (1.0 - alpha) * base_start.get(drv, 0.0) + alpha * target_final.get(drv, 0.0) for drv in drivetrains}
            if not strict:
                s = sum(target_share_map[y].values())
                target_share_map[y] = {k: v / s for k, v in target_share_map[y].items()}

    rows = []
    seg_mix_small = seg_mix[[region_col, drivetrain_col, segment_col, year_col, "seg_share_within_drv"]].copy()
    totals_map = dict(zip(total_by_year[year_col], total_by_year["total_inflow_year"]))
    for y in years:
        total_y = float(totals_map.get(y, 0.0))
        target_shares_y = target_share_map[y]
        for drv in drivetrains:
            drv_target_total = total_y * float(target_shares_y.get(drv, 0.0))
            drv_seg = seg_mix_small[(seg_mix_small[year_col] == y) & (seg_mix_small[drivetrain_col] == drv)].copy()
            if drv_seg.empty:
                continue
            seg_sum = float(drv_seg["seg_share_within_drv"].sum())
            if seg_sum <= 0:
                continue
            drv_seg["seg_share_within_drv"] = drv_seg["seg_share_within_drv"] / seg_sum
            drv_seg[value_col] = drv_target_total * drv_seg["seg_share_within_drv"]
            rows.append(drv_seg[[region_col, drivetrain_col, segment_col, year_col, value_col]])

    scenario_region = pd.concat(rows, ignore_index=True)
    out = pd.concat([scenario_region, df_other], ignore_index=True) if preserve_other_regions else scenario_region
    return (out.groupby([region_col, drivetrain_col, segment_col, year_col], as_index=False)[value_col].sum()
            .sort_values([region_col, drivetrain_col, segment_col, year_col]).reset_index(drop=True))


def tweak_inflow_segment_shares_within_drivetrain(
    inflow_df: pd.DataFrame, *, region: str = "EUR", drivetrain: str | None = None,
    scenario_start_year: int = 2026, target_segment_shares_by_year: dict[int, dict[str, float]] | None = None,
    target_segment_shares_final: dict[str, float] | None = None, ramp_end_year: int | None = None,
    region_col: str = "Region", drivetrain_col: str = "Drive Train", segment_col: str = "Segment",
    year_col: str = "year", value_col: str = "value", strict: bool = True,
) -> pd.DataFrame:
    """
    Mirror image of `tweak_inflow_drivetrain_shares`: re-weight inflow across SEGMENTS
    within one (or all) drivetrain(s), holding both total inflow per year AND each
    drivetrain's total inflow per year fixed -- only the segment mix within the
    drivetrain(s) changes. Same ramp/exact-year targeting and strict/lenient validation
    as the sibling function above; see its docstring for the shared math.
    """
    required_cols = {region_col, drivetrain_col, segment_col, year_col, value_col}
    missing = required_cols.difference(inflow_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if (target_segment_shares_by_year is None) == (target_segment_shares_final is None):
        raise ValueError("Provide exactly one of target_segment_shares_by_year or target_segment_shares_final.")
    if target_segment_shares_final is not None and ramp_end_year is None:
        raise ValueError("ramp_end_year is required when using target_segment_shares_final.")

    df = inflow_df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    mask = df[region_col].eq(region)
    if drivetrain is not None:
        mask &= df[drivetrain_col].eq(drivetrain)
    df_target = df.loc[mask].copy()
    df_other = df.loc[~mask].copy()
    if df_target.empty:
        raise ValueError(f"No rows found for region={region!r}, drivetrain={drivetrain!r}.")

    df_target = (df_target.groupby([region_col, drivetrain_col, segment_col, year_col], as_index=False)[value_col]
                 .sum().sort_values([year_col, drivetrain_col, segment_col]).reset_index(drop=True))

    drv_totals = df_target.groupby([year_col, drivetrain_col], as_index=False)[value_col].sum().rename(columns={value_col: "drv_total"})
    seg_totals = df_target.groupby([year_col, drivetrain_col, segment_col], as_index=False)[value_col].sum().rename(columns={value_col: "seg_total"})
    baseline_seg_shares = seg_totals.merge(drv_totals, on=[year_col, drivetrain_col], how="left")
    baseline_seg_shares["seg_share"] = np.where(baseline_seg_shares["drv_total"] > 0, baseline_seg_shares["seg_total"] / baseline_seg_shares["drv_total"], 0.0)

    segments = sorted(df_target[segment_col].dropna().unique().tolist())
    drivetrains = sorted(df_target[drivetrain_col].dropna().unique().tolist())
    years = sorted(df_target[year_col].unique().tolist())

    def _validate_or_normalize(share_dict: dict[str, float]) -> dict[str, float]:
        out = {seg: float(share_dict.get(seg, 0.0)) for seg in segments}
        total = sum(out.values())
        if strict:
            if any(v < 0 for v in out.values()):
                raise ValueError(f"Negative segment share found: {out}")
            if not np.isclose(total, 1.0, atol=1e-8):
                raise ValueError(f"Segment shares must sum to 1. Got {total:.12f} for {out}")
            return out
        if total <= 0:
            raise ValueError("Segment share total must be positive.")
        return {k: v / total for k, v in out.items()}

    baseline_share_map: dict[tuple[int, str], dict[str, float]] = {}
    for y in years:
        for drv in drivetrains:
            tmp = baseline_seg_shares[(baseline_seg_shares[year_col] == y) & (baseline_seg_shares[drivetrain_col] == drv)]
            baseline_share_map[(y, drv)] = {seg: float(tmp.loc[tmp[segment_col] == seg, "seg_share"].sum()) for seg in segments}

    target_share_map: dict[tuple[int, str], dict[str, float]] = {}
    if target_segment_shares_by_year is not None:
        for y in years:
            for drv in drivetrains:
                if y < scenario_start_year:
                    target_share_map[(y, drv)] = baseline_share_map[(y, drv)]
                elif y in target_segment_shares_by_year:
                    target_share_map[(y, drv)] = _validate_or_normalize(target_segment_shares_by_year[y])
                else:
                    specified_years = sorted(k for k in target_segment_shares_by_year if k <= y)
                    target_share_map[(y, drv)] = _validate_or_normalize(target_segment_shares_by_year[specified_years[-1]]) if specified_years else baseline_share_map[(y, drv)]
    else:
        target_final = _validate_or_normalize(target_segment_shares_final)
        for y in years:
            for drv in drivetrains:
                if y < scenario_start_year:
                    target_share_map[(y, drv)] = baseline_share_map[(y, drv)]
                    continue
                if y >= ramp_end_year:
                    target_share_map[(y, drv)] = target_final
                    continue
                base_start = baseline_share_map.get((scenario_start_year, drv))
                if base_start is None:
                    raise ValueError(f"scenario_start_year={scenario_start_year} not found for drivetrain={drv!r}.")
                alpha = (y - scenario_start_year) / (ramp_end_year - scenario_start_year)
                target_share_map[(y, drv)] = {seg: (1.0 - alpha) * base_start.get(seg, 0.0) + alpha * target_final.get(seg, 0.0) for seg in segments}
                if not strict:
                    s = sum(target_share_map[(y, drv)].values())
                    target_share_map[(y, drv)] = {k: v / s for k, v in target_share_map[(y, drv)].items()}

    rows = []
    for y in years:
        for drv in drivetrains:
            drv_total = float(drv_totals.loc[(drv_totals[year_col] == y) & (drv_totals[drivetrain_col] == drv), "drv_total"].sum())
            shares = target_share_map[(y, drv)]
            for seg in segments:
                rows.append({region_col: region, drivetrain_col: drv, segment_col: seg, year_col: y, value_col: drv_total * float(shares.get(seg, 0.0))})

    df_new = pd.DataFrame(rows)
    out = pd.concat([df_other, df_new], ignore_index=True)
    return (out.groupby([region_col, drivetrain_col, segment_col, year_col], as_index=False)[value_col].sum()
            .sort_values([region_col, drivetrain_col, segment_col, year_col]).reset_index(drop=True))


def run_adjusted_scenario(
    *, scenario_name: str, inflow_df: pd.DataFrame, matrices_by_key: dict, seg_share_by_drv: dict,
    flows_03: pd.DataFrame, p02: dict, unknown_whereabouts_share: dict[str, float], segment_map: dict,
    drv_prefix_map: dict, materials_region: str, materials_drivetrains: tuple[str, ...], base_year: int = 2005,
    segment_shares_by_drv: dict, allowed_export_segments: dict[str, list[str]] | None = None,
    export_share_by_drivetrain: dict[str, float] = None, stock_modifier_2027: float = 1.0,
    lifetime_change_by_drv: dict | None = None,
    monte_carlo_enabled: bool = False,
    n_draws: int = 0,
    lifetime_scale_lambda_relative_spread: dict[str, float] | float | None = None,
    unknown_whereabouts_share_std: dict[str, float] | None = None,
    export_share_std: dict[str, float] | None = None,
    mc_seed: int | np.random.SeedSequence | None = None,
    mc_chunk_size: int = 20_000,
) -> dict[str, Any]:
    """
    Run one adjusted-inflow scenario end to end: rebuild the segment-level cohort
    starting point from stage 02's stock (NOT a fresh 1975 backcast -- see "EXACT
    DIFFERENCES FROM 03_01" below), re-run the flow-driven model with `outflow_timing=
    "post_inflow"`, and build the resulting materials tracker.

    [NO HARDCODED SCENARIO PARAMETERS]: `stock_modifier_2027` and `lifetime_change_by_
    drv` are plain function parameters with no model-specific defaults baked in here
    (`stock_modifier_2027=1.0` and `lifetime_change_by_drv=None` are generic "no
    override" fallbacks, the same category of default as `outflow_timing="post_inflow"`
    below -- not a scenario definition). Every REAL scenario value (BEV_longer's
    scale_lambda=17, stock_lower's modifier=0.8, etc.) lives in `params_schema.py`'s
    `AdjustedFlowsParams.scenarios` and is resolved by `main()` before calling this
    function -- see `ScenarioSpec`'s docstring there. This is what lets `ICEV_shorter`
    and `BEV_longer` each pass their own override dict without a shared mutable global
    being reassigned between scenario runs (a real, order-dependent bug the pre-refactor
    module-level-constant version would have had if attempted that way).

    `p02`: pass the DICT view (`params.to_nested_dict()["02_stock_flow"]`), not the
    `StockFlowParams` dataclass -- see module docstring for why.

    [NEW] Monte Carlo, genuine lifetime + share uncertainty: when
    `monte_carlo_enabled=True`, this ALSO calls `fdm.run_flow_driven_model_
    monte_carlo` -- the vectorized engine -- using the EXACT SAME `inflow_df`,
    `years`, `t_end`, `mapped_inputs["lifetime_by_drv"]`,
    `starting_stock_by_cohort_lookup`, `lifetime_change_by_drv`,
    `unknown_whereabouts_share`, `export_share_by_drivetrain`, and
    `stock_modifier_2027` that the deterministic call directly above uses -- not
    a re-derivation, not a post-hoc re-split of the deterministic total. This is
    what makes the MC result a genuine re-simulation of THIS scenario's own
    assumptions (including its lifetime overrides, e.g. BEV_longer's
    scale_lambda=17), not shares-only uncertainty layered on top of a fixed
    deterministic number. Returned under the `"mc"` key (`None` if disabled).
    `mc_chunk_size` normally comes from `params.monte_carlo.chunk_size`, resolved by
    `main()` -- the literal default here is only a fallback for direct/standalone calls.
    """
    if lifetime_change_by_drv is None:
        lifetime_change_by_drv = {}

    stock_by_segment_base = fdm.build_stock_by_segment_at_base_year(
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv, region="EUR",
        base_year=base_year, allowed_drivetrains=("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
    )
    starting_stock_by_cohort_lookup = fdm.build_starting_stock_by_cohort_lookup(stock_by_segment_base)

    outflow_exp_segments = (
        flows_03[["Region", "Drive Train", "Segment", "year", "out_export"]].rename(columns={"out_export": "value"}).copy()
    )
    outflow_exp_segments["year"] = pd.to_numeric(outflow_exp_segments["year"], errors="coerce").astype(int)
    outflow_exp_segments["value"] = pd.to_numeric(outflow_exp_segments["value"], errors="coerce").fillna(0.0)
    outflow_exp_segments = (outflow_exp_segments.groupby(["Region", "Drive Train", "Segment", "year"], as_index=False)["value"]
                             .sum().sort_values(["Region", "Drive Train", "Segment", "year"]).reset_index(drop=True))

    inflow_df = inflow_df.copy()
    inflow_df["year"] = pd.to_numeric(inflow_df["year"], errors="coerce").astype(int)
    inflow_df["value"] = pd.to_numeric(inflow_df["value"], errors="coerce").fillna(0.0)
    inflow_df = (inflow_df.groupby(["Region", "Drive Train", "Segment", "year"], as_index=False)["value"]
                 .sum().sort_values(["Region", "Drive Train", "Segment", "year"]).reset_index(drop=True))

    mapped_inputs = fdm.build_p02_mapped_inputs(p02, drivetrains=("BEV", "HEV", "PHEV", "Diesel", "Petrol"))
    years = np.arange(int(inflow_df["year"].min()), int(inflow_df["year"].max()) + 1, dtype=int)

    print(f"[{scenario_name}] deterministic run: {len(years)} years ({years.min()}-{years.max()})...")
    t_start_det = time.time()
    results = fdm.run_flow_driven_model_with_outflow_disaggregation(
        df=inflow_df, years=years, t_end=int(years.max()),
        lifetime_by_drv=mapped_inputs["lifetime_by_drv"], export_r_by_drv=mapped_inputs["export_r_by_drv"],
        age_bins=mapped_inputs["age_bins"], unknown_whereabouts_share=unknown_whereabouts_share,
        starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
        outflow_value_col="value", year_col="year", inflow_col="value",
        group_cols=["Region", "Drive Train", "Segment"], outflow_timing="post_inflow",
        lifetime_change_by_drv=lifetime_change_by_drv, segment_shares_by_drv=segment_shares_by_drv,
        export_share_by_drivetrain=export_share_by_drivetrain, stock_modifier_2027=stock_modifier_2027,
    )
    print(f"[{scenario_name}] deterministic run done in {time.time() - t_start_det:.1f}s")

    mc_result = None
    if monte_carlo_enabled:
        print(f"[{scenario_name}] Monte Carlo run: {n_draws:,} draws...")
        t_start_mc = time.time()
        # Same inputs as the deterministic call directly above -- genuine
        # re-simulation of THIS scenario's own lifetime/share assumptions, not a
        # post-hoc re-split of the deterministic total. See docstring.
        mc_result = fdm.run_flow_driven_model_monte_carlo(
            df=inflow_df, years=years, t_end=int(years.max()),
            lifetime_by_drv=mapped_inputs["lifetime_by_drv"],
            unknown_whereabouts_share=unknown_whereabouts_share,
            export_share_by_drivetrain=export_share_by_drivetrain,
            starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
            n_draws=n_draws,
            lifetime_scale_lambda_relative_spread=lifetime_scale_lambda_relative_spread,
            unknown_whereabouts_share_std=unknown_whereabouts_share_std,
            export_share_std=export_share_std,
            group_cols=["Region", "Drive Train", "Segment"], outflow_timing="post_inflow",
            lifetime_change_by_drv=lifetime_change_by_drv, stock_modifier_2027=stock_modifier_2027,
            seed=mc_seed, chunk_size=mc_chunk_size, collect_per_year=False,
            verbose=True, progress_label=scenario_name,
        )
        print(f"[{scenario_name}] Monte Carlo run done in {time.time() - t_start_mc:.1f}s")

    flows_new = results["flows_df"].copy()
    outflow_surv_new = results["outflow_surv_df"].copy()
    outflow_exp_new = results["outflow_exp_df"].copy()
    outflow_unknown_new = results["outflow_unknown_df"].copy()

    disaggregated_new = {
        "inflow_segments": inflow_df.copy(),
        "outflow_coll_segments": (
            outflow_surv_new.copy().assign(
                value=lambda d: d["out_survival"] * (1.0 - d["Drive Train"].map(unknown_whereabouts_share).fillna(0.0).clip(0.0, 1.0))
            )[["Region", "Drive Train", "Segment", "year", "cohort_year", "value"]]
        ),
        "outflow_exp_segments": outflow_exp_new.copy().rename(columns={"out_export": "value"})[["Region", "Drive Train", "Segment", "year", "cohort_year", "value"]],
        "outflow_unk_segments": outflow_unknown_new.copy().rename(columns={"out_unknown": "value"})[["Region", "Drive Train", "Segment", "year", "cohort_year", "value"]],
    }
    tracker_new = build_tracker_from_disaggregated(disaggregated_new, region=materials_region, drivetrains=materials_drivetrains, include_zero=False)
    tracker_keyed_new, missing_new = add_keys_to_tracker_dict(tracker_new, segment_map=segment_map, drv_prefix_map=drv_prefix_map)

    return {
        "scenario_name": scenario_name, "inflow_df": inflow_df, "flows_df": flows_new,
        "tracker": tracker_new, "tracker_keyed": tracker_keyed_new, "missing_keys": missing_new,
        "mc": mc_result,
    }


def main() -> dict[str, Any]:
    loaded = load_many(
        "params", "matrices_by_key", "df_exp_eu", "synthetic_pre_2005_inflows",
        "starting_stock_2005_segments", "tracker_keyed", "flows_03", "seg_share_by_drv",
        root=PROJECT_ROOT,
    )
    params = loaded["params"]
    matrices_by_key = loaded["matrices_by_key"]
    tracker_keyed = loaded["tracker_keyed"]
    flows_03 = loaded["flows_03"]
    seg_share_by_drv = loaded["seg_share_by_drv"]

    p02 = params.stock_flow
    p03_02 = params.adjusted_flows
    p04 = params.materials

    # [IMPORTANT] dict view for fdm.py calls -- see module docstring for why (fdm.py's
    # internals are unverified; the ORIGINAL code passed a raw dict here).
    p02_dict = params.to_nested_dict()["02_stock_flow"]

    export_share_by_drivetrain_base = dict(p02.export_share_by_drv)
    unknown_whereabouts_share_base = dict(p02.unknown_whereabouts_share)
    # NOTE: unlike 03_01 (which pulls `unknown_whereabouts_share` from
    # `fdm.build_p02_mapped_inputs(...)["unknown_whereabouts_share"]`), this notebook
    # reads it DIRECTLY off `p02` with no mapping step. If `build_p02_mapped_inputs`
    # applies any transformation (e.g. filling in defaults for missing drivetrains,
    # renaming), these two could differ subtly -- not verifiable without `fdm.py`.
    segment_map = p04.segment_map
    drv_prefix_map = p04.drv_prefix_map
    materials_region = p04.region
    materials_drivetrains = tuple(p04.drivetrains)

    # -----------------------------------------------------------------------
    # [NEW] Monte Carlo setup -- genuine lifetime + share uncertainty, re-simulated
    # per scenario via the vectorized engine (see `run_adjusted_scenario`'s
    # `monte_carlo_enabled` docstring). Uncertainty spreads come from
    # `p02.lifetime_scale_lambda_relative_spread` / `p02.unknown_whereabouts_
    # share_std` / `p02.export_share_std` -- same `params_schema.py` fields
    # stage 02 already uses, nothing hardcoded here.
    # -----------------------------------------------------------------------
    monte_carlo_enabled = params.monte_carlo.enabled
    n_draws_mc = params.monte_carlo.n_draws if monte_carlo_enabled else 0
    lifetime_scale_lambda_relative_spread = dict(p02.lifetime_scale_lambda_relative_spread)
    unknown_whereabouts_share_std_mc = dict(p02.unknown_whereabouts_share_std)
    export_share_std_mc = dict(p02.export_share_std)
    mc_chunk_size = params.monte_carlo.chunk_size

    # Independent seed stream per scenario, spawn_key=(2,) -- matches this stage's
    # existing convention (differs from stage 02's implicit 0 and 03_01's (1,), so
    # this stage's draws aren't correlated with either by accident of sharing a
    # raw seed stream). Scenario NAMES themselves now come from `p03_02.scenarios`
    # (params-driven) rather than a hardcoded list -- adding a 12th scenario in
    # `AdjustedFlowsParams` is picked up here automatically.
    scenario_names_all = list(p03_02.scenarios.keys())
    mc_seed_seq = np.random.SeedSequence(params.monte_carlo.seed, spawn_key=(2,))
    mc_scenario_seeds = dict(zip(scenario_names_all, mc_seed_seq.spawn(len(scenario_names_all))))

    # -----------------------------------------------------------------------
    # Take 03_01's baseline inflow (by DRIVETRAIN + SEGMENT) as this notebook's starting point
    # -----------------------------------------------------------------------
    inflow_segments_scenario = (
        flows_03[["year", "Segment", "inflow", "Region", "Drive Train"]].rename(columns={"inflow": "value"})
        .groupby(["year", "Segment", "Region", "Drive Train"], as_index=False)["value"].sum()
        .sort_values(["Region", "Drive Train", "Segment", "year"]).reset_index(drop=True)
    )

    # =========================================================================
    # Resolve each scenario's inflow, share overrides, and lifetime override from
    # its `ScenarioSpec` (see params_schema.py), then run it. ONE loop for all 11
    # scenarios -- adding a 12th means adding one `ScenarioSpec` entry in
    # `AdjustedFlowsParams.scenarios`, not a new block of code here.
    #
    # Scenarios with their own inflow transform (`inflow_drivetrain_shares_final` or
    # `inflow_segment_shares_final` set -- BAU's own segment-mix baseline, BEV_only,
    # and the four BEV segment-profile scenarios) get it applied to the raw stage-03
    # inflow. Scenarios with neither set (BEV_longer, ICEV_shorter, stock_lower,
    # losses_zero, losses_high) reuse BAU's ALREADY-RESOLVED inflow -- BAU must
    # therefore be processed first, which `AdjustedFlowsParams.scenarios` guarantees
    # by construction (BAU is always its first entry; `validate()` requires it exist).
    # =========================================================================
    inflow_by_scenario: dict[str, pd.DataFrame] = {}
    for name, spec in p03_02.scenarios.items():
        if spec.inflow_drivetrain_shares_final is not None:
            inflow_by_scenario[name] = tweak_inflow_drivetrain_shares(
                inflow_df=inflow_segments_scenario, region="EUR",
                scenario_start_year=p03_02.scenario_start_year, ramp_end_year=p03_02.scenario_ramp_end_year,
                target_shares_final=spec.inflow_drivetrain_shares_final,
            )
        elif spec.inflow_segment_shares_final is not None:
            inflow_by_scenario[name] = tweak_inflow_segment_shares_within_drivetrain(
                inflow_df=inflow_segments_scenario, region="EUR", drivetrain=spec.inflow_segment_shares_drivetrain,
                scenario_start_year=p03_02.scenario_start_year, ramp_end_year=p03_02.scenario_ramp_end_year,
                target_segment_shares_final=spec.inflow_segment_shares_final,
            )
        else:
            inflow_by_scenario[name] = inflow_by_scenario["BAU"]

    scenario_results_all: dict[str, dict[str, Any]] = {}
    n_scenarios = len(p03_02.scenarios)
    t_start_all_scenarios = time.time()
    for scenario_idx, (name, spec) in enumerate(p03_02.scenarios.items(), start=1):
        elapsed = time.time() - t_start_all_scenarios
        eta_str = ""
        if scenario_idx > 1:
            avg_per_scenario = elapsed / (scenario_idx - 1)
            eta = avg_per_scenario * (n_scenarios - scenario_idx + 1)
            eta_str = f", ~{eta / 60:.1f} min remaining (est.)"
        print(
            f"\n=== Scenario {scenario_idx}/{n_scenarios}: '{name}' "
            f"(elapsed {elapsed / 60:.1f} min{eta_str}) ==="
        )

        export_share_scn = {**export_share_by_drivetrain_base, **spec.export_share_overrides}
        unknown_whereabouts_scn = {**unknown_whereabouts_share_base, **spec.unknown_whereabouts_share_overrides}
        lifetime_change_scn = {
            drv: {"start_year": c.start_year, "shape_k": c.shape_k, "scale_lambda": c.scale_lambda}
            for drv, c in spec.lifetime_change_by_drv.items()
        } or None

        scenario_results_all[name] = run_adjusted_scenario(
            scenario_name=name, inflow_df=inflow_by_scenario[name],
            matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
            flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_scn,
            segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
            materials_drivetrains=materials_drivetrains, base_year=2005,
            # `segment_shares_by_drv`: confirmed DEAD inside `flowdriven_model.py`
            # itself (finding C10 -- accepted but never read), kept only because the
            # scalar function's signature requires the argument. Nothing to source
            # from params here since it has no numerical effect.
            segment_shares_by_drv={},
            export_share_by_drivetrain=export_share_scn,
            stock_modifier_2027=spec.stock_modifier,
            lifetime_change_by_drv=lifetime_change_scn,
            monte_carlo_enabled=monte_carlo_enabled, n_draws=n_draws_mc,
            lifetime_scale_lambda_relative_spread=lifetime_scale_lambda_relative_spread,
            unknown_whereabouts_share_std=unknown_whereabouts_share_std_mc,
            export_share_std=export_share_std_mc, mc_seed=mc_scenario_seeds[name],
            mc_chunk_size=mc_chunk_size,
        )

    print(
        f"\n=== All {n_scenarios} scenarios done in "
        f"{(time.time() - t_start_all_scenarios) / 60:.1f} min total ===\n"
    )

    tracker_keyed_by_scenario = {name: r["tracker_keyed"] for name, r in scenario_results_all.items()}

    # -----------------------------------------------------------------------
    # Persist -- same two artifact groups as before (matches `artifacts.py`'s
    # existing `ARTIFACT_FILES` naming), now derived from each scenario's OWN
    # `ScenarioSpec` (has an inflow transform, or doesn't) instead of two
    # hardcoded name lists.
    # -----------------------------------------------------------------------
    inflow_mix_names = [
        name for name, spec in p03_02.scenarios.items()
        if spec.inflow_drivetrain_shares_final is not None or spec.inflow_segment_shares_final is not None
    ]
    other_names = [name for name in scenario_names_all if name not in inflow_mix_names]

    saved_inflow_mix_scenarios = save_many(
        **{f"tracker_keyed_{name}": tracker_keyed_by_scenario[name] for name in inflow_mix_names},
        root=PROJECT_ROOT,
    )
    saved_other_scenarios = save_many(
        **{f"tracker_keyed_{name}": tracker_keyed_by_scenario[name] for name in other_names},
        root=PROJECT_ROOT,
    )
    print("Saved inflow-mix scenario artifacts:", saved_inflow_mix_scenarios)
    print("Saved lifetime/loss/stock scenario artifacts (C6 fix):", saved_other_scenarios)

    # -----------------------------------------------------------------------
    # Diagnostic plot: inflow/outflow split, ALL 11 scenarios at once -- integrated
    # here (not a separate script), since this is exactly the step where every
    # scenario's tracker is available. This is also the most direct visual
    # confirmation that the C6 fix actually produced 5 genuinely different scenarios,
    # not 5 more silent copies of BAU.
    # -----------------------------------------------------------------------
    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "03_02_flows_all_scenarios.png"
    plot_flows_split_collected_unknown_all_trackers(
        tracker_keyed_by_scenario=tracker_keyed_by_scenario,
        region=materials_region,
        show=False,
        save_path=fig_path,
    )
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Monte Carlo (opt-in via params.monte_carlo.enabled, default False -- does not
    # affect or slow down a normal deterministic run above; everything above this
    # point is unchanged whether or not this block runs).
    #
    # [REPLACES the old share-only post-hoc MC] Each scenario's Monte Carlo result
    # was already computed inside `run_adjusted_scenario` itself (see its
    # `monte_carlo_enabled` docstring) via `fdm.run_flow_driven_model_monte_carlo`
    # -- the vectorized engine, itself a thin wrapper around the product-agnostic
    # `cohort_flow_mc.run_cohort_flow_monte_carlo` -- using THIS scenario's own
    # `inflow_df`, `lifetime_change_by_drv`, `stock_modifier_2027`, and share
    # dicts. This block only AGGREGATES those already-computed per-scenario
    # results (`scenario_results_all[name]["mc"]`) into summary stats and the
    # comparison plot; it performs no simulation of its own.
    #
    # GENUINE LIFETIME UNCERTAINTY, not just shares: `scale_lambda` itself is
    # resampled per draw (relative spread from `p02.lifetime_scale_lambda_
    # relative_spread`) and the SAME per-draw multiplier is carried through a
    # scenario's `lifetime_change_by_drv` override (e.g. BEV_longer's
    # scale_lambda=17 from 2027) -- see `cohort_flow_mc`'s "UNCERTAINTY
    # CONVENTION" docstring. This is what makes BEV_longer's uncertainty band
    # reflect genuine "how much longer, exactly, do these vehicles last"
    # uncertainty, not just downstream share noise on a fixed deterministic total.
    # -----------------------------------------------------------------------
    if monte_carlo_enabled:
        n_draws = n_draws_mc
        eu_total_cumulative_collected_by_scenario: dict[str, np.ndarray] = {}
        summary_mc: dict[str, dict] = {}

        for scenario_name in scenario_names_all:
            mc = scenario_results_all[scenario_name]["mc"]
            if mc is None:
                continue  # should not happen when monte_carlo_enabled, but fail soft

            eu_total_cumulative_collected_by_scenario[scenario_name] = mc["eu_total"]["cumulative_collected"]

            drivetrains_present = sorted({group_key[1] for group_key in mc["by_group"].keys()})
            for drivetrain in drivetrains_present:
                per_drv_groups = [g for gk, g in mc["by_group"].items() if gk[1] == drivetrain]
                for metric in ["cumulative_collected", "cumulative_export", "cumulative_unknown"]:
                    values = sum(g[metric] for g in per_drv_groups)
                    summary_mc[f"{scenario_name}__{drivetrain}__{metric}"] = summarize_distribution(values)

            for metric in ["cumulative_collected", "cumulative_export", "cumulative_unknown"]:
                summary_mc[f"{scenario_name}__EU_total__{metric}"] = summarize_distribution(mc["eu_total"][metric])

        saved_mc = save_many(mc_stage03_02_summary=summary_mc, root=PROJECT_ROOT)
        print("Saved Monte Carlo artifacts:", saved_mc)

        for scenario_name in scenario_names_all:
            s = summary_mc[f"{scenario_name}__EU_total__cumulative_collected"]
            print(
                f"Monte Carlo [{scenario_name}, EU total collected]: {n_draws:,} draws -- "
                f"mean={s['mean']:.2f}, median={s['median']:.2f}, P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
            )

        # -----------------------------------------------------------------------
        # Comparison figure: EU-total cumulative collected volume, ALL 11 scenarios
        # side by side -- the direct "how does uncertainty differ across scenarios"
        # visual, extending the existing all-scenarios flows chart above with an
        # uncertainty view of the same comparison.
        # -----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6))
        box_data = [eu_total_cumulative_collected_by_scenario[name] for name in scenario_names_all]
        bp = ax.boxplot(box_data, tick_labels=scenario_names_all, showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4a7fb5")
            patch.set_alpha(0.6)
        ax.set_title(f"Monte Carlo: EU-total cumulative collected volume by scenario (n={n_draws:,} draws each)", fontsize=12)
        ax.set_ylabel("Cumulative collected [million vehicles]")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        fig_path_mc = fig_dir / "03_02_monte_carlo_scenario_comparison.png"
        fig.savefig(fig_path_mc, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path_mc}")

    return {**saved_inflow_mix_scenarios, **saved_other_scenarios}


if __name__ == "__main__":
    main()
