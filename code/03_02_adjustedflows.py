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
relative_spread`, `collected_share_relative_spread`, `export_share_relative_
spread`, `unknown_whereabouts_share_relative_spread`, plus `unknown_share_
lifetime_coupling_k`) -- the SAME fields stage 02 already uses, nothing
hardcoded here. All four are sampled from TRIANGULAR distributions,
`Triangular(point*(1-lower), point, point*(1+upper))` -- matching
`params_schema.py`'s own documented convention (NOT Normal). Unlike lifetime,
the three shares (collected/export/unknown) are each independently sampled
with their own spread, then normalized per draw to sum to exactly 1; unknown's
center is additionally shifted per draw by that draw's own scale_lambda
outcome before its residual spread is applied (see `params_schema.py`'s
`unknown_share_lifetime_coupling_k` docstring). The SAME per-draw multiplier carries through a scenario's
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
available** -- run end-to-end, not just implemented and hoped-correct. (The checks below
were first made against synthetic inputs; the pipeline has since run on the real data --
see the input note above.):
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
INPUT DATA: REAL, as of 2026-08-20. This note previously said otherwise.
======================================================================
It used to read "everything is now tested with SYNTHETIC data ... against synthetic
REMIND/export/EEA data, not your real data". That was true when written and had gone
stale, which mattered: it cast doubt on every number the pipeline produced, and it was
still there long after the real files were in place.

Checked on 2026-08-20, all three sources are real:

  EEA registrations  data/raw/EEA_final_data.csv -- 24,610 rows, 30 country codes,
                     2010-2023, 177 M registrations, no synthetic marker.
  REMIND             data/raw/REMIND/*.mif -- five scenario files, 33-47 MB each.
  Used-vehicle       data/raw/usedvehicles_v1.2.xlsx.
  export

The only synthetic-input path in the codebase is
`params.disaggregation.use_synthetic_eea_fallback`, which is False and writes a
loudly-labelled placeholder when it is True. It is not in use.

STILL NOT A SUBSTITUTE FOR REVIEW. That the inputs are real says nothing about whether
the OUTPUT magnitudes are sensible for the EU fleet. That is a judgement about the
world, not a property of the code, and it needs a domain expert reading the numbers.

Not to be confused with `build_synthetic_pre_baseyear_inflows`: the pre-2005 inflows
are reconstructed from the 2005 stock through the survival curves. That is ordinary
cohort back-casting, not stand-in data.

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
    compute_collected_export_unknown_shares,
)
from src.monte_carlo import summarize_distribution, sensitivity_correlations, plot_tornado  # type: ignore
from src.stockflow_model import build_inflow_draws_by_drivetrain  # type: ignore

load_many = artifacts.load_many
save_many = artifacts.save_many
plot_flows_split_collected_unknown_all_trackers = plotting.plot_flows_split_collected_unknown_all_trackers
plot_flows_by_drivetrain_single_scenario = plotting.plot_flows_by_drivetrain_single_scenario


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


def _as_lower_upper_spread(value) -> tuple[float, float]:
    """
    Duck-typed (lower, upper) resolution -- same convention as `cohort_flow_mc.py`'s
    `_as_spread_pair` (not imported from there to avoid this stage-script depending on
    the engine's private helpers): accepts an object with `.lower`/`.upper` attributes
    (e.g. `params_schema.AsymmetricSpread`, without importing that class itself -- this
    file doesn't otherwise import `params_schema`), or a single float/int (symmetric).
    """
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return (float(value.lower), float(value.upper))
    return (float(value), float(value))


def sample_future_segment_share_inflow_draws(
    inflow_df: pd.DataFrame, *, region: str = "EUR", drivetrain: str,
    scenario_start_year: int, ramp_end_year: int,
    target_segment_shares_final: dict[str, float],
    inflow_segment_share_spread,
    n_draws: int, rng: np.random.Generator, years: np.ndarray,
    region_col: str = "Region", drivetrain_col: str = "Drive Train", segment_col: str = "Segment",
    year_col: str = "year", value_col: str = "value",
) -> dict[tuple, dict[int, np.ndarray]]:
    """
    The Monte Carlo counterpart of `tweak_inflow_segment_shares_within_drivetrain`'s
    deterministic ramp: instead of a single point-estimate `target_segment_shares_final`,
    sample a `(n_draws, n_segments)` matrix of FUTURE (year >= scenario_start_year)
    segment-mix draws, ramped exactly the same way, and return them shaped for
    `cohort_flow_mc.run_cohort_flow_monte_carlo`'s `inflow_draws_by_group` parameter.
    Years before `scenario_start_year` are NOT included in the returned dict at all --
    real historic segment-mix data stays fully deterministic (see
    `ScenarioSpec.inflow_segment_share_spread`'s docstring in `params_schema.py`).

    THE SAMPLING: for each segment (independently), draw
    `Triangular(mode*(1-lower), mode, mode*(1+upper))` -- `mode` = that segment's
    `target_segment_shares_final` value, `(lower, upper)` from
    `inflow_segment_share_spread` (a plain float means symmetric). ONE draw per Monte
    Carlo trial, giving a `(n_draws, n_segments)` matrix. Each ROW (one trial's whole
    segment-mix vector) is then renormalized by dividing by its own sum, so it sums to
    exactly 1 -- every segment moves proportionally on every draw, not one fixed segment
    absorbing the gap. `mode == 0` (a segment with literally zero target share) is left
    at exactly 0 for every draw (a degenerate Triangular would otherwise be
    ill-defined) -- consistent with "this segment gets none of this drivetrain's future
    inflow" being a hard constraint of the scenario, not an uncertain quantity.

    THE RAMP (identical math to `tweak_inflow_segment_shares_within_drivetrain`, just
    with the sampled/renormalized matrix in place of the single point `target_final`):
    for `scenario_start_year <= y < ramp_end_year`,
        share_draws[y, seg] = (1 - alpha(y)) * base_start_share[seg] + alpha(y) * final_target_draws[:, seg]
    where `base_start_share` is the REAL, deterministic baseline segment mix at
    `scenario_start_year` (read directly from `inflow_df`, held fixed through the ramp
    -- exactly what the deterministic function's own `base_start` is); for
    `y >= ramp_end_year`, `share_draws[y, seg] = final_target_draws[:, seg]` directly.
    Each year's absolute inflow is that year's DETERMINISTIC drivetrain total (read
    from `inflow_df`, unaffected by this -- only how it's split across segments is
    uncertain) times `share_draws[y, seg]`.
    """
    df = inflow_df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    mask = df[region_col].eq(region) & df[drivetrain_col].eq(drivetrain)
    df_drv = df.loc[mask].copy()
    if df_drv.empty:
        raise ValueError(f"No rows found for region={region!r}, drivetrain={drivetrain!r}.")
    df_drv = df_drv.groupby([segment_col, year_col], as_index=False)[value_col].sum()

    segments = sorted(target_segment_shares_final.keys())
    n_segments = len(segments)

    drv_total_by_year = df_drv.groupby(year_col)[value_col].sum()

    baseline_at_start = df_drv[df_drv[year_col] == scenario_start_year].set_index(segment_col)[value_col]
    baseline_total_at_start = float(baseline_at_start.sum())
    if baseline_total_at_start <= 0:
        raise ValueError(
            f"No inflow found for drivetrain={drivetrain!r} at scenario_start_year="
            f"{scenario_start_year} -- cannot resolve the ramp's baseline segment mix."
        )
    base_start_share = np.array(
        [float(baseline_at_start.get(seg, 0.0)) / baseline_total_at_start for seg in segments]
    )

    lower, upper = _as_lower_upper_spread(inflow_segment_share_spread)

    final_target_draws = np.zeros((n_draws, n_segments), dtype=float)
    for j, seg in enumerate(segments):
        mode = float(target_segment_shares_final[seg])
        low = mode * (1.0 - lower)
        high = mode * (1.0 + upper)
        if mode <= 0.0 or low >= high:
            final_target_draws[:, j] = mode
        else:
            final_target_draws[:, j] = rng.triangular(low, mode, high, size=n_draws)
    row_sums = final_target_draws.sum(axis=1, keepdims=True)
    final_target_draws = np.divide(
        final_target_draws, row_sums, out=np.zeros_like(final_target_draws), where=row_sums > 0
    )

    inflow_draws_by_group: dict[tuple, dict[int, np.ndarray]] = {
        (region, drivetrain, seg): {} for seg in segments
    }
    for y in years:
        y = int(y)
        if y < scenario_start_year:
            continue
        drv_total_y = float(drv_total_by_year.get(y, 0.0))
        if y >= ramp_end_year:
            share_draws_y = final_target_draws  # (n_draws, n_segments)
        else:
            alpha = (y - scenario_start_year) / (ramp_end_year - scenario_start_year)
            share_draws_y = (1.0 - alpha) * base_start_share[None, :] + alpha * final_target_draws
        for j, seg in enumerate(segments):
            inflow_draws_by_group[(region, drivetrain, seg)][y] = drv_total_y * share_draws_y[:, j]

    return inflow_draws_by_group


def run_adjusted_scenario(
    *, scenario_name: str, inflow_df: pd.DataFrame, matrices_by_key: dict, seg_share_by_drv: dict,
    flows_03: pd.DataFrame, p02: dict, unknown_whereabouts_share: dict[str, float], segment_map: dict,
    drv_prefix_map: dict, materials_region: str, materials_drivetrains: tuple[str, ...], base_year: int = 2005,
    segment_shares_by_drv: dict, allowed_export_segments: dict[str, list[str]] | None = None,
    export_share_by_drivetrain: dict[str, float] = None, stock_modifier_2027: float = 1.0,
    stock_modifier_start_year: int = 2027,
    lifetime_change_by_drv: dict | None = None,
    monte_carlo_enabled: bool = False,
    n_draws: int = 0,
    lifetime_scale_lambda_relative_spread: dict[str, float] | float | None = None,
    # [FIXED, replaces unknown_whereabouts_share_std/export_share_std] Matches
    # `flowdriven_model.py`'s `run_flow_driven_model_monte_carlo` signature: three
    # Triangular relative spreads (own uncertainty for each of collected/export/
    # unknown, none privileged as a no-uncertainty remainder) plus the lifetime<->
    # unknown_share coupling strength. `collected_share_by_drivetrain` is now
    # required by that function (previously implicit as `1 - export - unknown`).
    collected_share_by_drivetrain: dict[str, float] | None = None,
    collected_share_relative_spread: dict[str, float] | None = None,
    export_share_relative_spread: dict[str, float] | None = None,
    unknown_whereabouts_share_relative_spread: dict[str, float] | None = None,
    unknown_share_lifetime_coupling_k: dict[str, float] | float | None = None,
    mc_seed: int | np.random.SeedSequence | None = None,
    mc_seed_by_drivetrain: int | np.random.SeedSequence | None = None,
    bev_draw_export_dir: str | Path | None = None,
    mc_chunk_size: int = 20_000,
    output_periods: list[tuple[int, int]] | None = None,
    # [NEW] Precomputed per-draw future segment-share inflow overrides for this
    # scenario, if any -- built by `main()` via `sample_future_segment_share_inflow_
    # draws` (only when `ScenarioSpec.inflow_segment_share_spread` is set, e.g. BAU)
    # and threaded straight through to `fdm.run_flow_driven_model_monte_carlo` /
    # `cohort_flow_mc.run_cohort_flow_monte_carlo`'s `inflow_draws_by_group`. `None`
    # (default, every other scenario) means fully deterministic inflow during MC,
    # unchanged from before this parameter existed. Same "resolved by main(), not
    # hardcoded here" convention as every other scenario-specific value above.
    inflow_draws_by_group: dict[tuple, dict[int, np.ndarray]] | None = None,
    # [NEW] Passed straight through to `fdm.run_flow_driven_model_monte_carlo`'s
    # parameter of the same name -- see that function's / `cohort_flow_mc.py`'s
    # docstring. `(2.5, 97.5)` for a 95% per-year, per-drivetrain uncertainty
    # band, computed from this same MC run (no separate smaller pass). Exposed
    # on the return dict via `result["mc"]["per_year_entity_bands"]` (only
    # populated when `monte_carlo_enabled=True` AND this is not `None`).
    per_year_entity_band_pct: tuple[float, float] | None = None,
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

    [FIXED, found on a later audit] `stock_modifier_start_year` (the year `stock_
    modifier_2027` starts applying) and `base_year` (the stock-flow cohort base year)
    used to be hardcoded literals -- `stock_modifier_start_year` as a bare `2027`
    inside `flowdriven_model.py` itself (both the scalar function and the Monte Carlo
    wrapper), `base_year` as a bare `2005` at this function's call site in `main()`.
    Both are now real parameters, sourced from `AdjustedFlowsParams.stock_modifier_
    start_year` / `.base_year`, defaulting to the same values (2027 / 2005) so this
    fix changes nothing numerically unless those params fields are edited.

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

    [NEW] `inflow_draws_by_group`, when supplied (e.g. for BAU), additionally makes
    this scenario's FUTURE (year >= scenario_start_year) inflow segment-mix genuinely
    uncertain per Monte Carlo draw -- not just resampled lifetime/shares layered on a
    deterministic inflow. See `sample_future_segment_share_inflow_draws`'s docstring
    for exactly how it's built, and `cohort_flow_mc.py`'s "PER-DRAW INFLOW OVERRIDE"
    module docstring section for how the engine consumes it.
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
        stock_modifier_start_year=stock_modifier_start_year,
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
            collected_share_by_drivetrain=collected_share_by_drivetrain,
            starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
            n_draws=n_draws,
            lifetime_scale_lambda_relative_spread=lifetime_scale_lambda_relative_spread,
            collected_share_relative_spread=collected_share_relative_spread,
            export_share_relative_spread=export_share_relative_spread,
            unknown_whereabouts_share_relative_spread=unknown_whereabouts_share_relative_spread,
            unknown_share_lifetime_coupling_k=unknown_share_lifetime_coupling_k,
            group_cols=["Region", "Drive Train", "Segment"], outflow_timing="post_inflow",
            lifetime_change_by_drv=lifetime_change_by_drv, stock_modifier_2027=stock_modifier_2027,
            stock_modifier_start_year=stock_modifier_start_year, output_periods=output_periods,
            inflow_draws_by_group=inflow_draws_by_group,
            seed=mc_seed, chunk_size=mc_chunk_size, collect_per_year=False,
            per_year_entity_band_pct=per_year_entity_band_pct,
            verbose=True, progress_label=scenario_name,
        )
        print(f"[{scenario_name}] Monte Carlo run done in {time.time() - t_start_mc:.1f}s")

        # ---------------------------------------------------------------------------
        # [NEW] Independent by-drivetrain re-simulation -- see docstring section above.
        # ---------------------------------------------------------------------------
        if mc_seed_by_drivetrain is None:
            raise ValueError(
                f"[{scenario_name}] monte_carlo_enabled=True but mc_seed_by_drivetrain "
                "was not supplied -- required for the independent by-drivetrain "
                "re-simulation (see run_adjusted_scenario docstring)."
            )

        inflow_df_drivetrain = (
            inflow_df.groupby(["Region", "Drive Train", "year"], as_index=False)["value"].sum()
        )

        stock_by_drivetrain_base = fdm.build_stock_by_drivetrain_at_base_year(
            matrices_by_key=matrices_by_key, region="EUR", base_year=base_year,
            allowed_drivetrains=("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
        )
        starting_stock_by_cohort_lookup_drivetrain = fdm.build_starting_stock_by_cohort_lookup(
            stock_by_drivetrain_base, group_cols=["Region", "Drive Train"],
        )

        # If this scenario carries per-draw future segment-share inflow overrides
        # (e.g. BAU's `inflow_segment_share_spread`), aggregate them to drivetrain
        # level too (sum the same per-draw arrays across a drivetrain's segments,
        # for the SAME draw index) -- so the independent re-simulation's inflow is
        # genuinely the same per-draw realization as the segment-level run's, not a
        # fixed/deterministic total.
        inflow_draws_by_group_drivetrain = None
        if inflow_draws_by_group is not None:
            inflow_draws_by_group_drivetrain = {}
            for (grp_region, grp_drv, _grp_seg), year_draws in inflow_draws_by_group.items():
                drv_key = (grp_region, grp_drv)
                acc = inflow_draws_by_group_drivetrain.setdefault(drv_key, {})
                for yr, arr in year_draws.items():
                    acc[yr] = acc[yr] + arr if yr in acc else arr.copy()

        print(f"[{scenario_name}] Monte Carlo run (by-drivetrain, independent re-simulation): {n_draws:,} draws...")
        t_start_mc_drv = time.time()
        mc_result_by_drivetrain = fdm.run_flow_driven_model_monte_carlo(
            df=inflow_df_drivetrain, years=years, t_end=int(years.max()),
            lifetime_by_drv=mapped_inputs["lifetime_by_drv"],
            unknown_whereabouts_share=unknown_whereabouts_share,
            export_share_by_drivetrain=export_share_by_drivetrain,
            collected_share_by_drivetrain=collected_share_by_drivetrain,
            starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup_drivetrain,
            n_draws=n_draws,
            lifetime_scale_lambda_relative_spread=lifetime_scale_lambda_relative_spread,
            collected_share_relative_spread=collected_share_relative_spread,
            export_share_relative_spread=export_share_relative_spread,
            unknown_whereabouts_share_relative_spread=unknown_whereabouts_share_relative_spread,
            unknown_share_lifetime_coupling_k=unknown_share_lifetime_coupling_k,
            group_cols=["Region", "Drive Train"], outflow_timing="post_inflow",
            lifetime_change_by_drv=lifetime_change_by_drv, stock_modifier_2027=stock_modifier_2027,
            stock_modifier_start_year=stock_modifier_start_year, output_periods=output_periods,
            inflow_draws_by_group=inflow_draws_by_group_drivetrain,
            # [CHANGED] collect_per_year=True -- this by-drivetrain re-simulation
            # is only 5 groups (not 12 segments x drivetrain), so the added memory
            # cost (n_draws x n_years x 5 metrics x 5 groups) is affordable even at
            # 200,000 draws. This is what makes a genuine per-year 95% uncertainty
            # band possible for inflow/outflow/export/unknown/collected, not just a
            # single cumulative-period number -- see the plotting section below.
            seed=mc_seed_by_drivetrain, chunk_size=mc_chunk_size, collect_per_year=True,
            verbose=True, progress_label=f"{scenario_name} [by_drivetrain]",
        )
        print(f"[{scenario_name}] Monte Carlo run (by-drivetrain) done in {time.time() - t_start_mc_drv:.1f}s")
        mc_result["by_drivetrain"] = mc_result_by_drivetrain["by_group"]

        # -------------------------------------------------------------------
        # [NEW] Per-YEAR, per-DRAW arrays for BEV, one segment at a time, saved
        # to disk for stage 04_02 (BEV electronics).
        #
        # WHY IT LIVES HERE AND NOT IN 04_02: the inflow these draws come from is
        # THIS scenario's resolved inflow, which exists only inside this function
        # and is never persisted. A stage that re-derived it would be reproducing
        # a long setup by hand -- exactly how 03_01 silently diverged from stage
        # 02 three separate times. Exporting from the one place that already has
        # the right numbers removes that whole failure mode.
        #
        # WHY ONE SEGMENT PER CALL: the engine returns (n_draws, n_years) arrays
        # per group when collect_per_year=True. At 200,000 draws that is 154 MB
        # per array, so all 12 BEV segments at once would need ~9 GB. One segment
        # at a time peaks at ~0.8 GB and is written out before the next starts.
        #
        # SEEDS STILL LINE UP: the engine draws lifetime and share values per
        # ENTITY (drivetrain), caching them the first time that entity is seen,
        # and spawns those from `seed` in group order. BEV sorts first either
        # way, so a BEV-only call with the same `mc_seed` consumes the same first
        # spawn and therefore the same BEV draws as the full run above. Stage
        # 04_02 re-checks this against the saved period summary rather than
        # trusting it.
        # -------------------------------------------------------------------
        if bev_draw_export_dir is not None:
            bev_dir = Path(bev_draw_export_dir) / scenario_name
            bev_dir.mkdir(parents=True, exist_ok=True)
            bev_segments = sorted(
                {g[2] for g in mc_result["by_group"] if g[1] == "BEV"}
            )
            print(f"[{scenario_name}] exporting BEV per-year draws for "
                  f"{len(bev_segments)} segments -> {bev_dir}")
            for seg in bev_segments:
                seg_df = inflow_df[
                    (inflow_df["Drive Train"] == "BEV") & (inflow_df["Segment"] == seg)
                ]
                if seg_df.empty:
                    continue
                seg_draws = None
                if inflow_draws_by_group:
                    seg_draws = {
                        k: v for k, v in inflow_draws_by_group.items()
                        if k[1] == "BEV" and k[2] == seg
                    }
                t_seg = time.time()
                seg_mc = fdm.run_flow_driven_model_monte_carlo(
                    df=seg_df, years=years, t_end=int(years.max()),
                    lifetime_by_drv=mapped_inputs["lifetime_by_drv"],
                    unknown_whereabouts_share=unknown_whereabouts_share,
                    export_share_by_drivetrain=export_share_by_drivetrain,
                    collected_share_by_drivetrain=collected_share_by_drivetrain,
                    starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
                    n_draws=n_draws,
                    lifetime_scale_lambda_relative_spread=lifetime_scale_lambda_relative_spread,
                    collected_share_relative_spread=collected_share_relative_spread,
                    export_share_relative_spread=export_share_relative_spread,
                    unknown_whereabouts_share_relative_spread=unknown_whereabouts_share_relative_spread,
                    unknown_share_lifetime_coupling_k=unknown_share_lifetime_coupling_k,
                    group_cols=["Region", "Drive Train", "Segment"], outflow_timing="post_inflow",
                    lifetime_change_by_drv=lifetime_change_by_drv,
                    stock_modifier_2027=stock_modifier_2027,
                    stock_modifier_start_year=stock_modifier_start_year,
                    output_periods=output_periods,
                    inflow_draws_by_group=seg_draws,
                    seed=mc_seed, chunk_size=mc_chunk_size, collect_per_year=True,
                    per_year_entity_band_pct=None,
                    verbose=False, progress_label=f"{scenario_name} [BEV {seg}]",
                )
                grp = seg_mc["by_group"][("EUR", "BEV", seg)]
                yrs = np.asarray(grp["years"], dtype=int)
                np.save(bev_dir / "years.npy", yrs)
                # float32: these are vehicle counts in millions, ~1e-3..1e1, and
                # 7 significant digits is far beyond the model's real precision.
                # Halves both the file size and stage 04_02's peak memory.
                for name, key in (("inflow", "per_year_inflow"),
                                  ("outflow", "per_year_survival"),
                                  ("collected", "per_year_collected")):
                    arr = grp.get(key)
                    if arr is None:
                        continue
                    np.save(bev_dir / f"BEV_{seg}_{name}.npy",
                            np.asarray(arr, dtype=np.float32))
                print(f"    BEV {seg}: {time.time() - t_seg:.1f}s")
                del seg_mc, grp

    flows_new = results["flows_df"].copy()
    outflow_surv_new = results["outflow_surv_df"].copy()
    outflow_exp_new = results["outflow_exp_df"].copy()
    outflow_unknown_new = results["outflow_unknown_df"].copy()

    # [FIXED, this round -- REAL BUG, found via cross-checking 04_01's "inflow: total
    # mass by year across scenarios" plot against mc_stage03_02_summary for the
    # stock_lower scenario] This USED to be `inflow_df.copy()` -- the scenario's RAW,
    # PRE-stock_modifier input. But `flowdriven_model.py`'s
    # `run_flow_driven_model_with_outflow_disaggregation` (the deterministic call just
    # above, whose output is `results`/`flows_new`) applies `stock_modifier_2027` AS AN
    # INFLOW MULTIPLIER internally (`inflow_t *= stock_modifier_2027` for
    # `t >= stock_modifier_start_year` -- see that function's own docstring, point 4,
    # and the line `flows_rows.append({..., "inflow": inflow_t, ...})`) -- that IS the
    # mechanism by which `stock_lower` (stock_modifier_2027=0.8) produces a lower
    # stock: fewer new vehicles registered from 2027 onward, not a post-hoc stock-only
    # adjustment. `flows_new` (`results["flows_df"]`) already correctly carries this
    # modifier-adjusted inflow in its own "inflow" column -- it was computed right
    # above, then simply never used for `inflow_segments`, while the RAW pre-modifier
    # `inflow_df` was used instead. That silently threw away the modifier's effect on
    # every downstream artifact built from `inflow_segments` (`tracker`/`tracker_keyed`,
    # and therefore 04_01's year-by-year inflow-mass plots), even though the actual
    # simulated stock/outflow (correctly using the modifier) and the separate Monte
    # Carlo summary (`mc_stage03_02_summary`, via `period_inflow_multiplier` in
    # `cohort_flow_mc.py` -- also correctly modifier-adjusted) both diverged from BAU
    # as intended. Confirmed with real data: `tracker_keyed_BAU` and
    # `tracker_keyed_stock_lower`'s inflow were byte-identical for EVERY drivetrain
    # (wrong -- stock_lower's stock_modifier_2027=0.8 should reduce inflow from 2027
    # onward), while `mc_stage03_02_summary`'s cumulative_inflow mean for stock_lower
    # was ~20% below BAU's for every drivetrain (correct). Building `inflow_segments`
    # from `flows_new` instead makes the deterministic/tracker path consistent with
    # both the model's own internal simulation AND the Monte Carlo summary -- the same
    # single source of truth (`results["flows_df"]`) every OTHER disaggregated_new
    # entry below already correctly uses (outflow_coll/exp/unk_segments all derive from
    # `results[...]`, never from a raw pre-simulation input) -- inflow_segments was the
    # one inconsistent entry, not by design, just an oversight.
    inflow_segments_new = flows_new.rename(columns={"inflow": "value"})[
        ["Region", "Drive Train", "Segment", "year", "value"]
    ].copy()

    # [FIXED] COLLECTED IS THE THREE-WAY SHARE, NOT "EVERYTHING THAT IS NOT UNKNOWN".
    #
    # This line used to read `out_survival * (1 - unknown_share)`, which drops export
    # entirely. Every other implementation in the codebase -- 03_01's tracker split
    # (`disaggregation.compute_collected_export_unknown_shares`), the Monte Carlo
    # engine (`cohort_flow_mc.normalize_three_shares`) and
    # `flowdriven_model.py:1149` -- treats collected/export/unknown as one partition
    # of the survival outflow. This path was the only one that did not, and it was
    # the path feeding `tracker_keyed`, hence 04_01 and the material stages.
    #
    # Two things were wrong with it. It OVERSTATED collected by the export share:
    # with the base parameters, 0.900 instead of 0.880 for BEV (+2.3%) and 0.570
    # instead of 0.490 for every other drivetrain (+16.3%). And it broke the
    # partition -- `outflow_exp_segments` below still reports export separately, so
    # collected + export + unknown came to 1.02 for BEV and 1.08 for the others, with
    # exported vehicles counted twice.
    #
    # Reported by Yousef and colleagues, 2026-08-19.
    #
    # Uses the shared function so there is one implementation rather than four. It
    # normalises the three shares to sum to 1, which is identical to subtracting when
    # the inputs already sum to 1 (they do, for every drivetrain in the base
    # parameters) and stays correct when a scenario override makes them not.
    _coll_share_by_drv: dict[str, float] = {}
    for _drv in set(outflow_surv_new["Drive Train"].unique()):
        _c, _e, _u = compute_collected_export_unknown_shares(
            (collected_share_by_drivetrain or {}).get(_drv, 0.0),
            (export_share_by_drivetrain or {}).get(_drv, 0.0),
            unknown_whereabouts_share.get(_drv, 0.0),
        )
        _coll_share_by_drv[_drv] = float(_c)

    disaggregated_new = {
        "inflow_segments": inflow_segments_new,
        "outflow_coll_segments": (
            outflow_surv_new.copy().assign(
                value=lambda d: d["out_survival"] * d["Drive Train"].map(_coll_share_by_drv).fillna(0.0)
            )[["Region", "Drive Train", "Segment", "year", "cohort_year", "value"]]
        ),
        "outflow_exp_segments": outflow_exp_new.copy().rename(columns={"out_export": "value"})[["Region", "Drive Train", "Segment", "year", "cohort_year", "value"]],
        "outflow_unk_segments": outflow_unknown_new.copy().rename(columns={"out_unknown": "value"})[["Region", "Drive Train", "Segment", "year", "cohort_year", "value"]],
    }
    # THE PARTITION MUST CLOSE. collected + export + unknown has to equal the
    # survival outflow it was split from -- checked here, on the actual frames that
    # go into the tracker, rather than trusting the three share dicts that produced
    # them. This is the check that would have caught the dropped export share above
    # at the point where it mattered, in the numbers 04_01 consumes.
    _parts = sum(float(disaggregated_new[k]["value"].sum())
                 for k in ("outflow_coll_segments", "outflow_exp_segments",
                           "outflow_unk_segments"))
    _surv = float(outflow_surv_new["out_survival"].sum())
    if _surv > 0 and abs(_parts / _surv - 1.0) > 1e-6:
        raise ValueError(
            f"[{scenario_name}] outflow partition does not close: collected + export "
            f"+ unknown = {_parts:,.6f} against a survival outflow of {_surv:,.6f} "
            f"({100 * (_parts / _surv - 1):+.3f}%). The three sub-flows must sum to "
            f"out_survival exactly -- a share is being dropped or double-counted."
        )

    tracker_new = build_tracker_from_disaggregated(disaggregated_new, region=materials_region, drivetrains=materials_drivetrains, include_zero=False)
    tracker_keyed_new, missing_new = add_keys_to_tracker_dict(tracker_new, segment_map=segment_map, drv_prefix_map=drv_prefix_map)

    return {
        "scenario_name": scenario_name, "inflow_df": inflow_df, "flows_df": flows_new,
        "tracker": tracker_new, "tracker_keyed": tracker_keyed_new, "missing_keys": missing_new,
        "mc": mc_result,
    }


# ---------------------------------------------------------------------------
# [NEW] Within-scenario detail figures: drivetrain comparison, and segment
# comparison within one drivetrain -- boxplot view of the raw Monte Carlo draws
# (the 11-scenario `03_02_monte_carlo_scenario_comparison_*.png` figure further
# down only compares EU-total collected ACROSS scenarios; these compare WITHIN
# one scenario, one level of detail down).
# [REMOVED, per user request] This block used to also render a Gaussian-KDE
# density-curve overlay per (scenario, period) via a `_gaussian_kde_curve`
# helper -- both the helper and the density/PDF output are gone. Only the
# boxplot half of `plot_group_comparison_boxplot_and_pdf` (now renamed
# `plot_group_comparison_boxplot`) remains.
# ---------------------------------------------------------------------------
def plot_group_comparison_boxplot(
    values_by_label: dict[str, np.ndarray], *, title: str, xlabel: str, ylabel: str,
    fig_path_boxplot: Path,
) -> None:
    """
    Save a boxplot comparison figure for a `{label: (n_draws,) array}` dict --
    median/IQR/whisker view, quick to read. [REMOVED, per user request] This
    function used to also save a second `_pdf.png` Gaussian-KDE density-curve
    figure alongside the boxplot (was `plot_group_comparison_boxplot_and_pdf`) --
    that half, and its `_gaussian_kde_curve` helper, are gone; only the boxplot
    remains. `xlabel` is kept in the signature for call-site compatibility even
    though only the boxplot (which doesn't use it) is drawn now.
    Shared helper for both the drivetrain-comparison and segment-comparison
    figures below -- same comparison pattern, just a different label set each
    time. Styling matches the existing `03_02_monte_carlo_scenario_comparison_*.
    png` boxplot (`#4a7fb5`, dashed gridlines, hidden top/right spines) for
    visual consistency across all of this stage's Monte Carlo figures.
    """
    labels = list(values_by_label.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    box_data = [values_by_label[label] for label in labels]
    bp = ax.boxplot(box_data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4a7fb5")
        patch.set_alpha(0.6)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(fig_path_boxplot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paired_group_comparison_boxplot(
    values_by_label_a: dict[str, np.ndarray], values_by_label_b: dict[str, np.ndarray], *,
    label_a: str, label_b: str, title: str, ylabel: str, fig_path_boxplot: Path,
) -> None:
    """
    ONE figure comparing two approaches per group, side by side: for every label
    present in BOTH dicts, two boxes are drawn next to each other (approach A then
    approach B) at the same x tick, so the pair can be read against each other
    directly instead of flipping between two separate figures.

    Used for segment-sum vs independent by-drivetrain re-simulation: same drivetrain,
    two ways of computing it. Only labels in both dicts are drawn -- a drivetrain
    missing from either side would make the pair meaningless.
    """
    labels = [k for k in values_by_label_a if k in values_by_label_b]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.32
    gap = 0.06          # small gap so the two boxes of a pair don't touch
    pos = np.arange(len(labels), dtype=float)
    for offset, data_by_label, color in (
        (-(width + gap) / 2, values_by_label_a, "#4a7fb5"),
        (+(width + gap) / 2, values_by_label_b, "#c8794a"),
    ):
        bp = ax.boxplot(
            [data_by_label[k] for k in labels], positions=pos + offset, widths=width,
            showfliers=False, patch_artist=True, manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for element in ("medians", "whiskers", "caps"):
            for artist in bp[element]:
                artist.set_color("#333333")

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#4a7fb5", alpha=0.6, label=label_a),
            plt.Rectangle((0, 0), 1, 1, facecolor="#c8794a", alpha=0.6, label=label_b),
        ],
        loc="upper right", frameon=False, fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(fig_path_boxplot, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    # [NEW] Scenario selection -- run a subset of the 11 instead of always all of
    # them, for focused research runs. Controlled by `p03_02.scenarios_to_run`
    # (`AdjustedFlowsParams` in `params_schema.py`; defaults to `("BAU",)` there, so
    # a run is fast unless you deliberately widen it, e.g. to `None` for the full
    # sweep). Resolved via `active_scenario_names()` -- see that method's docstring
    # for the exact None-vs-tuple semantics; not re-implemented here so there is
    # exactly one place this logic lives. Already validated against `p03_02.scenarios`
    # in `AdjustedFlowsParams.validate()`, so an unknown scenario name fails at
    # `00_parameters.py` time, not partway through this run.
    #
    # NOTE: BAU's INFLOW is always resolved further down regardless of selection
    # (scenarios with no inflow transform of their own reuse it), but BAU itself is
    # only actually SIMULATED (deterministic + Monte Carlo run) if "BAU" ends up in
    # `active_scenario_names` -- deselecting it is safe and saves time if you don't
    # need its own output.
    # -----------------------------------------------------------------------
    active_scenario_names = list(p03_02.active_scenario_names())
    print(f"Running {len(active_scenario_names)}/{len(p03_02.scenarios)} scenario(s): {active_scenario_names}")

    # -----------------------------------------------------------------------
    # Monte Carlo setup -- genuine lifetime + share uncertainty, re-simulated per
    # scenario via the vectorized engine (see `run_adjusted_scenario`'s
    # `monte_carlo_enabled` docstring). Uncertainty spreads come from
    # `p02.lifetime_scale_lambda_relative_spread` / `p02.collected_share_
    # relative_spread` / `p02.export_share_relative_spread` / `p02.unknown_
    # whereabouts_share_relative_spread` / `p02.unknown_share_lifetime_
    # coupling_k` -- same `params_schema.py` fields stage 02 already uses,
    # nothing hardcoded here. `params.monte_carlo.enabled` is
    # shared across stages 02/03_01/03_02 by design -- narrow a run with
    # `scenarios_to_run` above rather than toggling MC off here, so 02/03_01 keep
    # running Monte Carlo as usual.
    # -----------------------------------------------------------------------
    monte_carlo_enabled = params.monte_carlo.enabled
    n_draws_mc = params.monte_carlo.n_draws if monte_carlo_enabled else 0
    lifetime_scale_lambda_relative_spread = dict(p02.lifetime_scale_lambda_relative_spread)
    # [FIXED] Was `unknown_whereabouts_share_std` / `export_share_std` -- both
    # retired from `StockFlowParams` when the share-uncertainty model was
    # refactored (see `params_schema.py`'s `[FIXED, replaces
    # unknown_whereabouts_share_std/export_share_std below]` comment, and
    # `flowdriven_model.py`'s `run_flow_driven_model_monte_carlo`, which now
    # requires `collected_share_by_drivetrain` and the three `*_relative_spread`
    # dicts instead of `_std`). This stage was not updated to match at the time --
    # these five lines are the fix.
    collected_share_by_drivetrain_mc = dict(p02.collected_share_by_drv)
    collected_share_relative_spread_mc = dict(p02.collected_share_relative_spread)
    export_share_relative_spread_mc = dict(p02.export_share_relative_spread)
    unknown_whereabouts_share_relative_spread_mc = dict(p02.unknown_whereabouts_share_relative_spread)
    unknown_share_lifetime_coupling_k_mc = dict(p02.unknown_share_lifetime_coupling_k)
    mc_chunk_size = params.monte_carlo.chunk_size

    # [NEW] Where BEV per-year, per-draw arrays go for stage 04_02 (BEV
    # electronics). Set to None to skip the export entirely -- it adds roughly
    # 12 extra engine calls (one per BEV segment) to a Monte Carlo run, so it is
    # real time, not free. 04_02 cannot run without it.
    # -----------------------------------------------------------------------
    # [NEW] Stage 02's per-draw INFLOW, carried across the boundary that used to
    # lose it. Built once and reused by every scenario, because it describes how
    # many vehicles Europe buys, which no scenario in this stage changes.
    #
    # THE DEFECT THIS CLOSES. Stage 02 samples fleet size and therefore purchase
    # volume. None of it reached here: `flows_03` holds one number per row with no
    # draw dimension, so total BEV inflow arrived with a coefficient of variation
    # of 0.000001% against stage 02's 9.6%, and every inflow band produced
    # downstream was too narrow. See documentation/DESIGN_inflow_uncertainty_
    # propagation.md for the measurements and the rejected alternatives.
    # -----------------------------------------------------------------------
    stage02_inflow_draws = None
    _s02_year_index: dict[int, int] = {}
    if params.monte_carlo.enabled and p02.propagate_stage02_inflow_uncertainty:
        _s02_years = np.arange(
            int(flows_03["year"].min()), int(flows_03["year"].max()) + 1, dtype=int
        )
        print(f"[03_02] carrying stage 02's per-draw inflow across the boundary "
              f"({params.monte_carlo.n_draws:,} draws)...")
        stage02_inflow_draws = build_inflow_draws_by_drivetrain(
            stock_dict=load_many("stock_dict", root=PROJECT_ROOT)["stock_dict"],
            params=params,
            parent_by_drv=p02.inflow_uncertainty_parent_by_drv,
            years=_s02_years,
        )
        _s02_year_index = {int(y): i for i, y in enumerate(_s02_years)}

    bev_draw_export_dir_resolved = (
        PROJECT_ROOT / "data" / "processed" / "bev_draws"
        if params.monte_carlo.enabled and p04.bev_electronics_export_draws
        else None
    )

    # Independent seed stream per scenario, spawn_key=(2,) -- matches this stage's
    # existing convention (differs from stage 02's implicit 0 and 03_01's (1,), so
    # this stage's draws aren't correlated with either by accident of sharing a
    # raw seed stream). Scenario NAMES now come from the (possibly filtered)
    # `active_scenario_names` rather than a hardcoded list -- adding a 12th
    # scenario in `AdjustedFlowsParams` is still picked up here automatically.
    scenario_names_all = active_scenario_names
    mc_seed_seq = np.random.SeedSequence(params.monte_carlo.seed, spawn_key=(2,))
    mc_scenario_seeds = dict(zip(scenario_names_all, mc_seed_seq.spawn(len(scenario_names_all))))

    mc_scenario_seeds_drivetrain = {
        name: np.random.SeedSequence(entropy=seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size)
        for name, seed in mc_scenario_seeds.items()
    }

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
    n_scenarios = len(active_scenario_names)
    t_start_all_scenarios = time.time()
    for scenario_idx, name in enumerate(active_scenario_names, start=1):
        spec = p03_02.scenarios[name]
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

        # -----------------------------------------------------------------------
        # [NEW] Future inflow segment-share Monte Carlo uncertainty (pilot: BAU only,
        # via `ScenarioSpec.inflow_segment_share_spread` -- None for every other
        # scenario today). Sampled here in `main()`, not inside `run_adjusted_scenario`
        # -- same "resolved by main(), not hardcoded in the scenario runner" convention
        # every other scenario-specific value on this call already follows.
        #
        # Seed: spawned from this scenario's OWN seed (`mc_scenario_seeds[name]`)
        # BEFORE that same SeedSequence object is handed to `run_adjusted_scenario` as
        # `mc_seed` below. `SeedSequence.spawn()` is stateful (advances an internal
        # spawn counter) -- spawning once here first means the engine's own later
        # internal per-entity spawns (inside `cohort_flow_mc.run_cohort_flow_monte_
        # carlo`) start from the NEXT child onward, so this segment-share sampling
        # stream and the engine's lifetime/share entity-draw streams are independently
        # seeded and never overlap, same multi-level spawn-hierarchy convention this
        # stage already uses everywhere else (scenario -> group -> entity).
        #
        # `years` recomputed identically to how `run_adjusted_scenario` derives it
        # internally from the same `inflow_by_scenario[name]` -- so the year keys in
        # the returned draws dict line up exactly with what the engine iterates over.
        # -----------------------------------------------------------------------
        inflow_draws_for_scenario = None
        if (
            monte_carlo_enabled
            and spec.inflow_segment_share_spread is not None
            and spec.inflow_segment_shares_final is not None
        ):
            segment_share_seed = mc_scenario_seeds[name].spawn(1)[0]
            _ = mc_scenario_seeds_drivetrain[name].spawn(1)[0]
            segment_share_rng = np.random.default_rng(segment_share_seed)
            scenario_inflow_years = np.arange(
                int(inflow_by_scenario[name]["year"].min()),
                int(inflow_by_scenario[name]["year"].max()) + 1,
                dtype=int,
            )
            inflow_draws_for_scenario = sample_future_segment_share_inflow_draws(
                inflow_df=inflow_segments_scenario, region="EUR",
                drivetrain=spec.inflow_segment_shares_drivetrain,
                scenario_start_year=p03_02.scenario_start_year,
                ramp_end_year=p03_02.scenario_ramp_end_year,
                target_segment_shares_final=spec.inflow_segment_shares_final,
                inflow_segment_share_spread=spec.inflow_segment_share_spread,
                n_draws=n_draws_mc, rng=segment_share_rng, years=scenario_inflow_years,
            )

        # -------------------------------------------------------------------
        # Compose the three independent effects into the per-draw inflow the
        # engine consumes. None of them replaces another:
        #
        #   volume   stage 02's per-draw purchase volume, per coarse drivetrain
        #   split    stage 03_01's Petrol/Diesel and HEV/PHEV split, untouched
        #   segment  this stage's own segment mix, renormalised to sum to one
        #
        # A group with no segment-share draws still receives the volume, applied
        # through its deterministic share. Groups whose share draws already exist
        # keep them and are scaled, so both effects vary together.
        #
        # FLOORING HAPPENS HERE, PER DRAW, and not in the shared builder, which
        # deliberately hands back the raw residual including negatives. Flooring
        # is nonlinear, so flooring each draw is not the same as flooring one
        # average trajectory -- it lifts the Liquids mean by about 0.228 million
        # vehicles around 2035. That is the correct Monte Carlo answer: in a draw
        # where the fleet target lands higher, liquid-fuel inflow really is still
        # positive that year.
        # -------------------------------------------------------------------
        # ===================================================================
        # DO NOT MODIFY THIS BLOCK WITHOUT RUNNING code/test_stage03_inflow.py
        # ===================================================================
        # Settled 2026-08-20 after this block produced inflow figures that were
        # wrong by a factor of two for four years without anything detecting it.
        # It is validated against real registration statistics, and the test file
        # named above re-checks every claim below. Change the rule and the tests
        # fail; that is deliberate.
        #
        # WHAT WENT WRONG, so nobody reintroduces it:
        #
        # 1. THE PARENT VOLUME WAS NEVER SPLIT BETWEEN ITS CHILDREN.
        #    `build_inflow_draws_by_drivetrain` hands BOTH children of a coarse
        #    group the parent's array -- its docstring says so, and says "the
        #    caller then applies its own share to divide them". This caller did
        #    not. `parent_total` was grouped by ["Drive Train", "year"], which is
        #    a drivetrain's own total across SEGMENTS. So Diesel's segments summed
        #    to the entire Liquids volume, and so did Petrol's. Measured against
        #    real EEA registrations, both came out at 2.2x reality for every year
        #    from 2010 to 2019, and 15.2 million diesel cars in 2018 against an
        #    actual EU diesel market of ~5.6 million. The variable being named
        #    `parent_total` while holding the drivetrain total is what hid it.
        #
        # 2. THE BASE YEAR IS IDENTICALLY ZERO. Stage 02's first modelled year is
        #    2005, where inflow = stock_target - survivors and the target IS the
        #    initial stock, so the residual is exactly 0.00e+00 for every
        #    drivetrain and every draw. Structural, not sampling noise. Writing
        #    that over the deterministic value produced a spike to zero at 2005.
        #
        # 3. THE TWO STAGES DO NOT SHARE A LEVEL. Stage 02's MC mean sits below
        #    stage 03's deterministic value -- ratio 0.954 / 0.939 / 0.967 /
        #    0.982 / 0.868 at 2006 / 2010 / 2015 / 2020 / 2030 -- because inflow
        #    is a nonlinear function of the sampled lifetime, so
        #    E[inflow] != inflow(E[lambda]). That is not a defect, but it means
        #    SUBSTITUTING stage 02's level for stage 03's can never line up at the
        #    boundary, however the shares are computed.
        #
        # THE RULE. Stage 03 owns the volume; stage 02 contributes only its
        # deviation from its own mean:
        #
        #     share = det_value / parent_group_total     (share of the PARENT)
        #     dev   = draws_parent(y) - mean(draws_parent(y))
        #     value = max(det_value + share * dev, 0)
        #
        # BEFORE FLOORING the mean over draws is exactly the deterministic value
        # (measured worst deviation 3.3e-16), so there is no step at the boundary by
        # construction rather than by tuning. At 2005 every draw is zero, so dev is
        # zero and the deterministic value survives with a zero-width band -- the
        # honest answer, since stage 02 carries no information about that year.
        #
        # AFTER FLOORING the mean can only be LIFTED, never lowered, and in a
        # near-zero year the lift is large. Measured: exactly 1.00x wherever there is
        # real volume, but 10.35x at HEV 2040, where the deterministic value is
        # 0.0128 million (12,800 cars, the tail of the phase-out) and 36.6% of draws
        # fall below zero. That is the correct Monte Carlo answer -- in a draw where
        # the fleet target lands higher, hybrid inflow really is still positive that
        # year -- but a result read out of a phase-out tail carries it and should be
        # quoted with that in mind. `test_stage03_inflow.py` asserts both halves:
        # exact equality before flooring, and never below afterwards.
        #
        # ADDITIVE, NOT THE RATIO FORM `draw / mean`. That was rejected earlier in
        # this project because the residual passes through zero as Liquids phases
        # out and the relative spread diverges (CV reached 1775%).
        #
        # VALIDATED against real data, all 14 years available:
        #     Diesel 2010-2019   2.21x -> 1.11x of EEA registrations
        #     Petrol 2010-2019   2.20x -> 1.13x
        #     BEV                1.04x -> 1.05x (parent is itself; never affected)
        #     pre-2005 level     13.12 M/yr vs 13.08 M/yr EU long-run average
        #
        # KNOWN AND NOT FIXED HERE: the model's own hybrid volume is 0.66x (2019)
        # to 0.36x (2023) of real HEV+PHEV registrations, and 2020-2023 runs high
        # because a stock-driven scenario model does not reproduce the COVID and
        # chip-shortage collapse. Neither is caused by this block.
        #
        # FLOORING STILL HAPPENS HERE, PER DRAW, for the reason given above.
        if stage02_inflow_draws is not None:
            _parent_of = dict(p02.inflow_uncertainty_parent_by_drv)
            det = inflow_by_scenario[name]
            det = det[det["Region"] == "EUR"].groupby(
                ["Region", "Drive Train", "Segment", "year"], as_index=False
            )["value"].sum()
            det["_parent"] = det["Drive Train"].map(_parent_of).fillna(det["Drive Train"])
            parent = (det.groupby(["_parent", "year"], as_index=False)["value"]
                        .sum().rename(columns={"value": "parent_total"}))
            det = det.merge(parent, on=["_parent", "year"], how="left")

            prior = inflow_draws_for_scenario or {}
            composed: dict = {}
            _share_sum: dict[tuple, float] = {}
            _dev_cache: dict[tuple, np.ndarray] = {}
            for region, drv, seg, yr, val, par, parent_total in det.itertuples(
                index=False, name=None
            ):
                vol = stage02_inflow_draws.get(drv)
                yi = _s02_year_index.get(int(yr))
                if vol is None or yi is None:
                    continue
                # Stage 02 begins in 2005; this stage begins in 1975. The builder
                # marks years it does not model with NaN rather than zero, so they
                # cannot be mistaken for "no vehicles". Those years get NO per-draw
                # override at all and keep their deterministic value, which is
                # correct: there is no stage-02 uncertainty for them to carry.
                # Passing the NaN through instead poisons every summary downstream.
                if not np.isfinite(vol[yi]).all():
                    continue

                ck = (str(par), int(yr))
                dev = _dev_cache.get(ck)
                if dev is None:
                    col = np.asarray(vol[yi], dtype=np.float64)
                    dev = col - col.mean()
                    _dev_cache[ck] = dev

                share = (float(val) / float(parent_total)) if parent_total else 0.0
                _share_sum[ck] = _share_sum.get(ck, 0.0) + share

                key = (region, drv, seg)
                existing = prior.get(key, {}).get(int(yr))
                base = (np.asarray(existing, dtype=np.float64)
                        if existing is not None else float(val))
                composed.setdefault(key, {})[int(yr)] = np.maximum(base + share * dev, 0.0)

            # THE CHILDREN MUST CONSUME EXACTLY ONE PARENT VOLUME. Every segment of
            # every drivetrain under a parent holds a share of that parent, and
            # those shares must sum to 1 for the year. This is the check that would
            # have caught defect 1: with the old grouping it summed to 1 per
            # DRIVETRAIN, so a parent with two children summed to 2.
            _bad = {k: s for k, s in _share_sum.items() if abs(s - 1.0) > 1e-6}
            if _bad:
                k, s = sorted(_bad.items())[0]
                raise ValueError(
                    f"[{name}] stage-02 inflow shares do not sum to 1 for "
                    f"parent={k[0]!r} year={k[1]}: got {s:.6f}, across {len(_bad)} "
                    f"(parent, year) pairs. Each parent's volume must be DIVIDED "
                    f"among its children, not handed to each of them. See the note "
                    f"above this block and code/test_stage03_inflow.py."
                )
            inflow_draws_for_scenario = composed

        scenario_results_all[name] = run_adjusted_scenario(
            scenario_name=name, inflow_df=inflow_by_scenario[name],
            matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
            flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_scn,
            segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
            materials_drivetrains=materials_drivetrains, base_year=p03_02.base_year,
            # `segment_shares_by_drv`: confirmed DEAD inside `flowdriven_model.py`
            # itself (finding C10 -- accepted but never read), kept only because the
            # scalar function's signature requires the argument. Nothing to source
            # from params here since it has no numerical effect.
            segment_shares_by_drv={},
            export_share_by_drivetrain=export_share_scn,
            stock_modifier_2027=spec.stock_modifier,
            stock_modifier_start_year=p03_02.stock_modifier_start_year,
            lifetime_change_by_drv=lifetime_change_scn,
            monte_carlo_enabled=monte_carlo_enabled, n_draws=n_draws_mc,
            lifetime_scale_lambda_relative_spread=lifetime_scale_lambda_relative_spread,
            collected_share_by_drivetrain=collected_share_by_drivetrain_mc,
            collected_share_relative_spread=collected_share_relative_spread_mc,
            export_share_relative_spread=export_share_relative_spread_mc,
            unknown_whereabouts_share_relative_spread=unknown_whereabouts_share_relative_spread_mc,
            unknown_share_lifetime_coupling_k=unknown_share_lifetime_coupling_k_mc,
            mc_seed=mc_scenario_seeds[name],
            mc_seed_by_drivetrain=mc_scenario_seeds_drivetrain[name],
            bev_draw_export_dir=bev_draw_export_dir_resolved,
            mc_chunk_size=mc_chunk_size, output_periods=params.monte_carlo.output_periods,
            inflow_draws_by_group=inflow_draws_for_scenario,
            # [NEW] 95% per-year, per-drivetrain uncertainty band for the flow
            # plots below -- computed from THIS scenario's actual MC run
            # (`n_draws_mc` draws, same as everything else this scenario
            # reports), no separate smaller pass. `None` when MC is disabled
            # (nothing to band around).
            per_year_entity_band_pct=(2.5, 97.5) if monte_carlo_enabled else None,
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
        name for name in active_scenario_names
        if p03_02.scenarios[name].inflow_drivetrain_shares_final is not None
        or p03_02.scenarios[name].inflow_segment_shares_final is not None
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
    # [MOVED] The "inflow/outflow split, all scenarios" diagnostic plot used to be
    # generated right here, before Monte Carlo even ran -- meaning it could only ever
    # show the deterministic lines, never an uncertainty band. It now happens further
    # down, after the Monte Carlo block, once `per_year_entity_bands` is available for
    # every scenario that had MC enabled. `fig_dir` stays defined here since the
    # MC-only figures below (boxplots, tornado, scenario comparison) also need it.
    # -----------------------------------------------------------------------
    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Monte Carlo (opt-in via params.monte_carlo.enabled, default False -- does not
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
    #
    # [NEW] OUTPUT GRANULARITY: for EACH scenario x EACH requested period
    # (`params.adjusted_flows.output_periods`, e.g. a single year `(2030, 2030)` or
    # a range `(2030, 2040)`) x EACH drivetrain (summed over its segments) x EACH
    # individual (drivetrain, segment) x EACH metric (survival/export/unknown/
    # collected/inflow/stock), a full distribution summary (mean/median/mode/std/
    # P2.5/P97.5 + 50-bin histogram, via `monte_carlo.summarize_distribution`) is
    # computed and saved -- not just the single EU-total-collected-at-2070 number
    # this block used to produce. Stock additionally gets a per-YEAR summary within
    # each period (`stock_per_year`), not just the period's end/sum. This is a
    # genuinely large output for wide periods/many scenarios -- see the printed
    # summary count below; narrow `output_periods` (fewer, shorter ranges) if the
    # output size becomes unwieldy.
    # -----------------------------------------------------------------------
    def _summarize_period_result(period_result: dict, prefix: str, out: dict[str, dict]) -> None:
        """
        Given one period's result dict (matching cohort_flow_mc's per-period shape),
        compute summarize_distribution() for every flow/inflow/stock metric, plus a
        per-year summary for every year in `stock_per_year`, storing into `out` under
        keys prefixed with `prefix`.
        """
        for metric in [
            "cumulative_survival", "cumulative_export", "cumulative_unknown",
            "cumulative_collected", "cumulative_inflow",
            "stock_end_of_period", "stock_sum_over_period",
        ]:
            out[f"{prefix}__{metric}"] = summarize_distribution(period_result[metric])
        for year, values in period_result["stock_per_year"].items():
            out[f"{prefix}__stock_year_{int(year)}"] = summarize_distribution(values)

    def _sum_period_results(period_results: list[dict]) -> dict:
        """Sum several groups' same-period result dicts together (e.g. all segments
        of one drivetrain) -- same aggregation `cohort_flow_mc.py` itself uses for
        `total`, just applied to a caller-chosen subset of groups instead of all of them."""
        flat_metrics = [
            "cumulative_survival", "cumulative_a", "cumulative_b", "cumulative_collected",
            "cumulative_inflow", "stock_end_of_period", "stock_sum_over_period",
        ]
        # Include the export/unknown aliases too if present (added by the fdm wrapper).
        if "cumulative_export" in period_results[0]:
            flat_metrics += ["cumulative_export", "cumulative_unknown"]
        summed = {m: sum(pr[m] for pr in period_results) for m in flat_metrics}
        years_in_period = list(period_results[0]["stock_per_year"].keys())
        summed["stock_per_year"] = {
            y: sum(pr["stock_per_year"][y] for pr in period_results) for y in years_in_period
        }
        return summed

    if monte_carlo_enabled:
        n_draws = n_draws_mc
        eu_total_cumulative_collected_by_scenario_period: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
        drivetrain_collected_by_scenario_period: dict[tuple[str, str, tuple[int, int]], np.ndarray] = {}
        summary_mc: dict[str, dict] = {}
        sensitivity_by_scenario: dict[str, "pd.DataFrame"] = {}
        # Headline period for sensitivity analysis (one output at a time, by design
        # of `sensitivity_correlations`) -- the WIDEST requested period, not
        # necessarily assuming the whole-horizon period was requested at all.
        headline_period = max(params.monte_carlo.output_periods, key=lambda p: p[1] - p[0])

        for scenario_name in scenario_names_all:
            mc = scenario_results_all[scenario_name]["mc"]
            if mc is None:
                continue  # should not happen when monte_carlo_enabled, but fail soft

            drivetrains_present = sorted({group_key[1] for group_key in mc["by_group"].keys()})

            # --- [NEW] Sensitivity analysis: which entity's (drivetrain's)
            # scale_lambda / export_share / unknown_share uncertainty drives THIS
            # scenario's EU-total collected volume most. `mc["entity_draws"]` is the
            # per-entity sampled draws the vectorized engine used internally,
            # exposed specifically for this.
            input_draws = {}
            for entity, draws in mc.get("entity_draws", {}).items():
                input_draws[f"{entity}_scale_lambda"] = draws["scale_lambda"]
                input_draws[f"{entity}_export_share"] = draws["export"]
                input_draws[f"{entity}_unknown_share"] = draws["unknown"]
            output_for_sensitivity = mc["eu_total"]["periods"][headline_period]["cumulative_collected"]
            sensitivity_by_scenario[scenario_name] = sensitivity_correlations(input_draws, output_for_sensitivity)

            for (start, end) in params.monte_carlo.output_periods:
                period = (start, end)
                period_label = f"{start}-{end}"

                eu_period_result = mc["eu_total"]["periods"][period]
                eu_total_cumulative_collected_by_scenario_period[(scenario_name, period)] = (
                    eu_period_result["cumulative_collected"]
                )
                _summarize_period_result(
                    eu_period_result, f"{scenario_name}__{period_label}__EU_total", summary_mc
                )

                # [NEW] Accumulate raw draws (not just summarize_distribution's binned
                # histogram) for this period, so the within-scenario detail figures
                # below can build boxplots/KDE curves from the actual draws -- reusing
                # exactly the same `drv_period_result`/`g["periods"][period]` values
                # `_summarize_period_result` already computes on, not a re-derivation.
                collected_by_drivetrain: dict[str, np.ndarray] = {}
                collected_by_drivetrain_direct: dict[str, np.ndarray] = {}
                collected_by_segment_within_drivetrain: dict[str, dict[str, np.ndarray]] = {}

                for drivetrain in drivetrains_present:
                    per_drv_groups = [g for gk, g in mc["by_group"].items() if gk[1] == drivetrain]
                    drv_period_result = _sum_period_results(
                        [g["periods"][period] for g in per_drv_groups]
                    )
                    _summarize_period_result(
                        drv_period_result, f"{scenario_name}__{period_label}__{drivetrain}", summary_mc
                    )
                    collected_by_drivetrain[drivetrain] = drv_period_result["cumulative_collected"]
                    drivetrain_collected_by_scenario_period[(scenario_name, drivetrain, period)] = (
                        drv_period_result["cumulative_collected"]
                    )

                    # [RESTORED] Independent by-drivetrain re-simulation, same period,
                    # recorded side by side with the segment-sum above:
                    #   ..__{drivetrain}__cumulative_collected          (segment sum)
                    #   ..__{drivetrain}__direct__cumulative_collected  (independent)
                    # A mismatch beyond Monte Carlo noise flags that the base-year
                    # uniform segment-mix assumption has a real effect for that
                    # drivetrain/period.
                    by_drivetrain_direct = mc.get("by_drivetrain")
                    if by_drivetrain_direct is not None:
                        direct_group = by_drivetrain_direct.get(("EUR", drivetrain))
                        if direct_group is not None:
                            _summarize_period_result(
                                direct_group["periods"][period],
                                f"{scenario_name}__{period_label}__{drivetrain}__direct",
                                summary_mc,
                            )
                            collected_by_drivetrain_direct[drivetrain] = (
                                direct_group["periods"][period]["cumulative_collected"]
                            )

                    # Segment level: individual (drivetrain, segment) groups, no further summing.
                    segment_values: dict[str, np.ndarray] = {}
                    for group_key, g in mc["by_group"].items():
                        if group_key[1] != drivetrain:
                            continue
                        segment = group_key[2] if len(group_key) > 2 else None
                        if segment is None:
                            continue
                        _summarize_period_result(
                            g["periods"][period],
                            f"{scenario_name}__{period_label}__{drivetrain}__{segment}",
                            summary_mc,
                        )
                        segment_values[segment] = g["periods"][period]["cumulative_collected"]
                    collected_by_segment_within_drivetrain[drivetrain] = segment_values

                # -----------------------------------------------------------------------
                # [NEW] Within-scenario detail figures for THIS (scenario, period) --
                # generated for every scenario actually run (`scenario_names_all`, i.e.
                # `AdjustedFlowsParams.scenarios_to_run`'s resolution -- no separate
                # config knob needed: narrow scenarios_to_run and you narrow which
                # scenarios get these figures too). [REMOVED, per user request] Used to
                # also save a KDE density-curve `_pdf.png` alongside the boxplot --
                # only the boxplot is saved now.
                # -----------------------------------------------------------------------
                if len(collected_by_drivetrain) > 1:
                    plot_group_comparison_boxplot(
                        collected_by_drivetrain,
                        title=f"{scenario_name}: cumulative collected by drivetrain, {period_label}",
                        xlabel="Cumulative collected [million vehicles]",
                        ylabel="Cumulative collected [million vehicles]",
                        fig_path_boxplot=fig_dir / f"03_02_monte_carlo_drivetrain_comparison_{scenario_name}_{period_label}_boxplot.png",
                    )

                # [RESTORED] Same 5-drivetrain comparison, but from the independent
                # re-simulation instead of the 12-segment sum. Separate filename so the
                # two sit side by side rather than overwriting each other.
                if len(collected_by_drivetrain_direct) > 1:
                    plot_group_comparison_boxplot(
                        collected_by_drivetrain_direct,
                        title=f"{scenario_name}: cumulative collected by drivetrain "
                              f"(independent re-simulation, not segment-sum), {period_label}",
                        xlabel="Cumulative collected [million vehicles]",
                        ylabel="Cumulative collected [million vehicles]",
                        fig_path_boxplot=fig_dir / f"03_02_monte_carlo_drivetrain_comparison_direct_{scenario_name}_{period_label}_boxplot.png",
                    )

                # ONE figure, both approaches side by side per drivetrain.
                if len(collected_by_drivetrain_direct) > 1:
                    plot_paired_group_comparison_boxplot(
                        collected_by_drivetrain, collected_by_drivetrain_direct,
                        label_a="Sum of 12 segments",
                        label_b="Independent re-simulation (5 drivetrains)",
                        title=f"{scenario_name}: cumulative collected -- segment-sum vs "
                              f"independent re-simulation, {period_label}",
                        ylabel="Cumulative collected [million vehicles]",
                        fig_path_boxplot=fig_dir / f"03_02_monte_carlo_drivetrain_segmentsum_vs_direct_{scenario_name}_{period_label}_boxplot.png",
                    )

                for drivetrain, segment_values in collected_by_segment_within_drivetrain.items():
                    if len(segment_values) <= 1:
                        continue
                    # Conventional A..F, then JA..JF ordering (matches segment_map's
                    # own key order in params_schema.py) rather than plain alphabetical
                    # (which would put "A" before "B" but "JA" before "A" is wrong too).
                    ordered_segments = sorted(segment_values.keys(), key=lambda s: (s.startswith("J"), s))
                    plot_group_comparison_boxplot(
                        {seg: segment_values[seg] for seg in ordered_segments},
                        title=f"{scenario_name} / {drivetrain}: cumulative collected by segment, {period_label}",
                        xlabel="Cumulative collected [million vehicles]",
                        ylabel="Cumulative collected [million vehicles]",
                        fig_path_boxplot=fig_dir / f"03_02_monte_carlo_segment_comparison_{scenario_name}_{drivetrain}_{period_label}_boxplot.png",
                    )

                print(f"Saved within-scenario detail figures: {scenario_name}, {period_label}")

        saved_mc = save_many(
            mc_stage03_02_summary=summary_mc, mc_stage03_02_sensitivity=sensitivity_by_scenario, root=PROJECT_ROOT,
        )
        print(f"Saved Monte Carlo artifacts: {saved_mc} ({len(summary_mc):,} summary entries)")

        for scenario_name in scenario_names_all:
            sens_df = sensitivity_by_scenario.get(scenario_name)
            if sens_df is None or sens_df.empty:
                continue
            print(f"\nSensitivity [{scenario_name}, EU total collected, {headline_period[0]}-{headline_period[1]}]:")
            print(sens_df.to_string(index=False))
            fig_tornado, _ = plot_tornado(
                sens_df,
                title=f"Sensitivity: {scenario_name} EU-total collected, {headline_period[0]}-{headline_period[1]}",
            )
            fig_path_tornado = fig_dir / f"03_02_monte_carlo_sensitivity_tornado_{scenario_name}.png"
            fig_tornado.savefig(fig_path_tornado, dpi=150, bbox_inches="tight")
            fig_tornado.clf()
            print(f"Saved diagnostic plot: {fig_path_tornado}")

        for scenario_name in scenario_names_all:
            for (start, end) in params.monte_carlo.output_periods:
                s = summary_mc[f"{scenario_name}__{start}-{end}__EU_total__cumulative_collected"]
                print(
                    f"Monte Carlo [{scenario_name}, {start}-{end}, EU total collected]: {n_draws:,} draws -- "
                    f"mean={s['mean']:.2f}, median={s['median']:.2f}, P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
                )

        # -----------------------------------------------------------------------
        # Comparison figure: EU-total cumulative collected volume, ALL scenarios side
        # by side -- one figure PER requested period (was a single figure for the
        # implicit whole-horizon-only case before `output_periods` existed).
        # -----------------------------------------------------------------------
        for (start, end) in params.monte_carlo.output_periods:
            period = (start, end)
            period_label = f"{start}-{end}"
            fig, ax = plt.subplots(figsize=(11, 6))
            box_data = [
                eu_total_cumulative_collected_by_scenario_period[(name, period)] for name in scenario_names_all
            ]
            bp = ax.boxplot(box_data, tick_labels=scenario_names_all, showfliers=False, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#4a7fb5")
                patch.set_alpha(0.6)
            ax.set_title(
                f"Monte Carlo: EU-total cumulative collected volume by scenario, "
                f"{period_label} (n={n_draws:,} draws each)", fontsize=12,
            )
            ax.set_ylabel("Cumulative collected [million vehicles]")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            fig_path_mc = fig_dir / f"03_02_monte_carlo_scenario_comparison_{period_label}.png"
            fig.savefig(fig_path_mc, dpi=150, bbox_inches="tight")
            print(f"Saved diagnostic plot: {fig_path_mc}")

    # -----------------------------------------------------------------------
    # [MOVED + NEW] Diagnostic plots: inflow/outflow split, now WITH each
    # scenario's 95% Monte Carlo band (when MC was enabled) -- this is why
    # these plots had to move here, after the MC block above, rather than
    # staying where they used to be generated (right after `tracker_keyed_by_
    # scenario` was built, before Monte Carlo had even run). `entity_bands_
    # by_scenario` pulls straight from each scenario's own `mc` result --
    # genuinely re-simulated uncertainty for THAT scenario's own assumptions,
    # not a shared/generic band. A scenario with MC disabled (or that had no
    # `mc` result) simply gets an unshaded plot -- same lines as before this
    # feature existed, degrading gracefully rather than erroring.
    # -----------------------------------------------------------------------
    entity_bands_by_scenario = {
        name: (r["mc"]["per_year_entity_bands"] if r["mc"] is not None else None)
        for name, r in scenario_results_all.items()
    }

    fig_path = fig_dir / "03_02_flows_all_scenarios.png"
    plot_flows_split_collected_unknown_all_trackers(
        tracker_keyed_by_scenario=tracker_keyed_by_scenario,
        region=materials_region,
        show=False,
        save_path=fig_path,
        entity_bands_by_scenario=entity_bands_by_scenario,
    )
    print(f"Saved diagnostic plot: {fig_path}")

    # [NEW] One full-size, one-subplot-per-drivetrain plot PER SCENARIO --
    # every scenario actually run (`scenario_names_all`, i.e. `AdjustedFlows
    # Params.scenarios_to_run`'s resolution -- same "narrow scenarios_to_run,
    # narrow which scenarios get these figures" convention as the within-
    # scenario detail figures above), not hardcoded to BAU. Restores the
    # per-scenario flow plot that previously existed only for BAU, now for
    # every active scenario, each with its own real MC band.
    flows_drivetrains = ("BEV", "Diesel", "Petrol", "PHEV", "HEV")
    def _mc_per_year_band(mc_result, region, drv, field):
        """
        (years, median, p2_5, p97_5) for one field ("per_year_inflow" /
        "per_year_survival" / "per_year_a" / "per_year_b" / "per_year_collected")
        of one drivetrain's by-drivetrain MC re-simulation, or None if Monte
        Carlo wasn't run / this drivetrain has no entry (e.g. zero vehicles in
        this scenario).
        """
        if not mc_result or "by_drivetrain" not in mc_result:
            return None
        group = mc_result["by_drivetrain"].get((region, drv))
        if group is None or field not in group:
            return None
        arr = group[field]  # (n_draws, n_years)
        p2_5, median, p97_5 = np.percentile(arr, [2.5, 50, 97.5], axis=0)
        return group["years"], median, p2_5, p97_5

    def _plot_series_with_band(ax, years, median, p2_5, p97_5, color, label=None):
        ax.plot(years, median, color=color, linewidth=2, label=label)
        ax.fill_between(years, p2_5, p97_5, color=color, alpha=0.2, linewidth=0)


    for scenario_name in scenario_names_all:
        fig_path_scenario = fig_dir / f"03_02_flows_{scenario_name}.png"
        plot_flows_by_drivetrain_single_scenario(
            tracker_keyed=tracker_keyed_by_scenario[scenario_name],
            scenario_name=scenario_name,
            region=materials_region,
            entity_bands=entity_bands_by_scenario.get(scenario_name),
            show=False,
            save_path=fig_path_scenario,
        )
        print(f"Saved diagnostic plot: {fig_path_scenario}")

        flows_scn = scenario_results_all[scenario_name]["flows_df"]
        mc_result = scenario_results_all[scenario_name]["mc"]
        flows_by_year_drv = (
            flows_scn[flows_scn["Region"] == materials_region]
            .groupby(["Drive Train", "year"], as_index=False)[
                ["inflow", "out_survival", "out_collected", "out_export", "out_unknown"]
            ].sum()
        )
        for drv in flows_drivetrains:
            d = flows_by_year_drv[
                (flows_by_year_drv["Drive Train"] == drv)
                & (flows_by_year_drv["year"] >= 1975)
                & (flows_by_year_drv["year"] <= 2070)
            ].sort_values("year")

            # --- Inflow: unchanged, own figure ---
            inflow_band = _mc_per_year_band(mc_result, materials_region, drv, "per_year_inflow")
            if d.empty and inflow_band is None:
                print(f"Skipped inflow figure: no data for scenario={scenario_name!r}, drivetrain={drv!r}.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                color = plotting.DRIVETRAIN_LINE_COLORS.get(drv, "#333333")
                if inflow_band is not None:
                    years_b, median, p2_5, p97_5 = inflow_band
                    mask = (years_b >= 1975) & (years_b <= 2070)
                    _plot_series_with_band(ax, years_b[mask], median[mask], p2_5[mask], p97_5[mask], color)
                    n_draws_band = mc_result["by_drivetrain"][(materials_region, drv)]["per_year_inflow"].shape[0]
                    ax.set_title(f"{scenario_name} / {drv}: inflow, 1975-2070 (median + 95% MC band, n={n_draws_band:,})", fontsize=11)
                elif not d.empty:
                    ax.plot(d["year"], d["inflow"], color=color, linewidth=2)
                    ax.set_title(f"{scenario_name} / {drv}: inflow, 1975-2070", fontsize=12)
                ax.set_xlabel("Year")
                ax.set_ylabel("Inflow [million/year]")
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                fig_path = fig_dir / f"03_02_inflow_{scenario_name}_{drv}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved diagnostic plot: {fig_path}")

            # --- Outflow breakdown: ONE combined figure ---
            total_band = _mc_per_year_band(mc_result, materials_region, drv, "per_year_survival")
            collected_band = _mc_per_year_band(mc_result, materials_region, drv, "per_year_collected")
            export_band = _mc_per_year_band(mc_result, materials_region, drv, "per_year_a")
            unknown_band = _mc_per_year_band(mc_result, materials_region, drv, "per_year_b")

            if d.empty and total_band is None:
                print(f"Skipped outflow breakdown figure: no data for scenario={scenario_name!r}, drivetrain={drv!r}.")
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            color = plotting.DRIVETRAIN_LINE_COLORS.get(drv, "#333333")
            stack_colors = {"Collected": "#2E86AB", "Export": "#E67E22", "Unknown whereabouts": "#8E44AD"}

            if collected_band is not None and export_band is not None and unknown_band is not None:
                years_b = collected_band[0]
                mask = (years_b >= 1975) & (years_b <= 2070)
                years_plot = years_b[mask]
                collected_med = collected_band[1][mask]
                export_med = export_band[1][mask]
                unknown_med = unknown_band[1][mask]
                ax.stackplot(
                    years_plot, collected_med, export_med, unknown_med,
                    labels=["Collected", "Export", "Unknown whereabouts"],
                    colors=[stack_colors["Collected"], stack_colors["Export"], stack_colors["Unknown whereabouts"]],
                    alpha=0.55,
                )
            elif not d.empty:
                years_plot = d["year"].to_numpy()
                ax.stackplot(
                    years_plot, d["out_collected"], d["out_export"], d["out_unknown"],
                    labels=["Collected", "Export", "Unknown whereabouts"],
                    colors=[stack_colors["Collected"], stack_colors["Export"], stack_colors["Unknown whereabouts"]],
                    alpha=0.55,
                )

            if total_band is not None:
                years_b, median, p2_5, p97_5 = total_band
                mask = (years_b >= 1975) & (years_b <= 2070)
                ax.plot(years_b[mask], median[mask], color="black", linewidth=2.2, label="Total outflow (median)", zorder=5)
                ax.fill_between(years_b[mask], p2_5[mask], p97_5[mask], color="black", alpha=0.15, linewidth=0, zorder=4, label="Total outflow 95% MC band")
                n_draws_band = mc_result["by_drivetrain"][(materials_region, drv)]["per_year_survival"].shape[0]
                ax.set_title(f"{scenario_name} / {drv}: outflow breakdown, 1975-2070 (n={n_draws_band:,})", fontsize=11)
            elif not d.empty:
                ax.plot(d["year"], d["out_survival"], color="black", linewidth=2.2, label="Total outflow")
                ax.set_title(f"{scenario_name} / {drv}: outflow breakdown, 1975-2070", fontsize=12)

            ax.set_xlabel("Year")
            ax.set_ylabel("Outflow [million/year]")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
            plt.tight_layout(rect=[0, 0, 0.78, 1])
            fig_path = fig_dir / f"03_02_outflow_breakdown_{scenario_name}_{drv}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved diagnostic plot: {fig_path}")


        # [RESTORED] One figure PER DRIVETRAIN comparing scenarios side by side, so a
        # shift in the EU total can be attributed to a specific drivetrain. Only drawn
        # when more than one scenario actually ran for that drivetrain/period.
        for (start, end) in params.monte_carlo.output_periods:
            period = (start, end)
            period_label = f"{start}-{end}"
            for drivetrain in ("BEV", "HEV", "PHEV", "Diesel", "Petrol"):
                by_scenario = {
                    scenario_name: drivetrain_collected_by_scenario_period[(scenario_name, drivetrain, period)]
                    for scenario_name in scenario_names_all
                    if (scenario_name, drivetrain, period) in drivetrain_collected_by_scenario_period
                }
                if len(by_scenario) <= 1:
                    continue
                plot_group_comparison_boxplot(
                    by_scenario,
                    title=f"{drivetrain}: cumulative collected across scenarios, {period_label}",
                    xlabel="Cumulative collected [million vehicles]",
                    ylabel="Cumulative collected [million vehicles]",
                    fig_path_boxplot=fig_dir / f"03_02_monte_carlo_scenario_comparison_by_drivetrain_{drivetrain}_{period_label}_boxplot.png",
                )
            print(f"Saved per-drivetrain cross-scenario comparison figures: {period_label}")

    return {**saved_inflow_mix_scenarios, **saved_other_scenarios}


if __name__ == "__main__":
    main()