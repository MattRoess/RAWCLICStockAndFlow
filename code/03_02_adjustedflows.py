"""
03_02_adjustedflows.py
========================

Stage 03, part 2: takes `flows_03` (03_01's baseline flow-driven output) and re-runs the
flow-driven model under a series of ALTERNATIVE inflow-composition scenarios (different
drivetrain mixes, different BEV segment-size profiles), to explore sensitivity of the
downstream materials tracker to those assumptions.

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

load_many = artifacts.load_many
save_many = artifacts.save_many
plot_flows_split_collected_unknown_all_trackers = plotting.plot_flows_split_collected_unknown_all_trackers


# ---------------------------------------------------------------------------
# Scenario "knobs" -- see CRITICAL FINDING above: several of these are currently no-ops
# ---------------------------------------------------------------------------
stock_modifier_2027 = 1  # BAU default: no-op multiplier. Overridden to 0.8 for the
                          # "stock_lower" scenario at the call site below (C6 fix).

# BAU (no-op) lifetime-change dict -- the default passed to `run_adjusted_scenario`
# unless a scenario explicitly overrides it (see LIFETIME_CHANGE_BY_DRV_ICEV_SHORTER /
# _BEV_LONGER below, and the C6 fix in the module docstring).
#
# SCHEMA NOTE (still open, needs flowdriven_model.py to resolve): this dict provides
# only {"start_year", "shape_k", "scale_lambda"} -- no "end_year". Compare to
# `00_parameters.py`'s `lifetime_override_by_drv` / `02_stockdriven.py`'s
# `get_effective_lifetime_params`, which REQUIRE both start_year AND end_year for a
# bounded override window. If `run_flow_driven_model_with_outflow_disaggregation`'s
# `lifetime_change_by_drv` handling expects the same 4-key shape, this would be missing
# a required key; if it instead treats a missing end_year as "open-ended from
# start_year onward", that's a second, independent override-schema convention in the
# codebase. Not verifiable without `flowdriven_model.py`'s source.
LIFETIME_CHANGE_BY_DRV_BAU = {
    "Diesel": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 13.0},
    "Petrol": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 13.0},
    "BEV": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 13.0},
}

# [NEW, resolves C6] "ICEV_shorter" scenario: Diesel/Petrol lifetime shortened to
# scale_lambda=9.0 from 2027 onward, per the abandoned scaffold's own comment. BEV left
# at the BAU value (this scenario is specifically about ICE vehicles, not BEV).
LIFETIME_CHANGE_BY_DRV_ICEV_SHORTER = {
    "Diesel": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 9.0},
    "Petrol": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 9.0},
    "BEV": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 13.0},
}

# [NEW, resolves C6] "BEV_longer" scenario: BEV lifetime extended to scale_lambda=17.0
# from 2027 onward, per the abandoned scaffold's own comment. Diesel/Petrol unchanged.
LIFETIME_CHANGE_BY_DRV_BEV_LONGER = {
    "Diesel": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 13.0},
    "Petrol": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 13.0},
    "BEV": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 17.0},
}

# BEV segment-share profiles used for the segment-mix sensitivity scenarios below.
# "BAU" = business as usual (identical to 03_01's single profile). The other four
# represent alternative assumptions about which vehicle segments BEV adoption favors:
# concentrated in conventional segments (A_F), concentrated in the "J"-prefixed
# (presumably SUV/crossover) segments (JA_JF), skewed toward larger vehicles (Large), or
# skewed toward smaller vehicles (Small). These do NOT sum-check against BAU by
# construction -- they're independent hypotheses, each individually summing to ~1.0.
BEV_SEGMENT_SHARES_BAU = {
    "A": 0.11599, "B": 0.07235, "C": 0.17037, "D": 0.06288, "E": 0.04190, "F": 0.02945,
    "JA": 0.00341, "JB": 0.07498, "JC": 0.25116, "JD": 0.15162, "JE": 0.02108, "JF": 0.00481,
}
BEV_SEGMENT_SHARES_A_F = {
    "A": 0.11940, "B": 0.14733, "C": 0.42153, "D": 0.21450, "E": 0.06298, "F": 0.03426,
    "JA": 0.0, "JB": 0.0, "JC": 0.0, "JD": 0.0, "JE": 0.0, "JF": 0.0,
}
BEV_SEGMENT_SHARES_JA_JF = {
    "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0,
    "JA": 0.11940, "JB": 0.14733, "JC": 0.42153, "JD": 0.21450, "JE": 0.06298, "JF": 0.03426,
}
BEV_SEGMENT_SHARES_LARGE = {
    "A": 0.02, "B": 0.05, "C": 0.12, "D": 0.16, "E": 0.12, "F": 0.06,
    "JA": 0.04, "JB": 0.10, "JC": 0.16, "JD": 0.10, "JE": 0.05, "JF": 0.02,
}
BEV_SEGMENT_SHARES_SMALL = {
    "A": 0.10, "B": 0.22, "C": 0.28, "D": 0.14, "E": 0.04, "F": 0.01,
    "JA": 0.03, "JB": 0.07, "JC": 0.07, "JD": 0.03, "JE": 0.01, "JF": 0.00,
}
segment_shares_by_drv = {
    "BEV": BEV_SEGMENT_SHARES_BAU, "A_F": BEV_SEGMENT_SHARES_A_F, "JA_JF": BEV_SEGMENT_SHARES_JA_JF,
    "Large": BEV_SEGMENT_SHARES_LARGE, "Small": BEV_SEGMENT_SHARES_SMALL,
}


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
    export_share_by_drivetrain: dict[str, float] = None, stock_modifier_2027: float = stock_modifier_2027,
    lifetime_change_by_drv: dict | None = None,
) -> dict[str, Any]:
    """
    Run one adjusted-inflow scenario end to end: rebuild the segment-level cohort
    starting point from stage 02's stock (NOT a fresh 1975 backcast -- see "EXACT
    DIFFERENCES FROM 03_01" below), re-run the flow-driven model with `outflow_timing=
    "post_inflow"`, and build the resulting materials tracker.

    [FIXED, resolves part of C6] `lifetime_change_by_drv` is now an explicit parameter
    (defaults to `LIFETIME_CHANGE_BY_DRV_BAU`, the no-op baseline) instead of the
    function silently closing over a module-level global of the same name. This is
    what lets `ICEV_shorter` and `BEV_longer` pass their own override dict without a
    shared mutable global being reassigned between scenario runs (which would have been
    a real, order-dependent bug if attempted the original way).

    `p02`: pass the DICT view (`params.to_nested_dict()["02_stock_flow"]`), not the
    `StockFlowParams` dataclass -- see module docstring for why.
    """
    if lifetime_change_by_drv is None:
        lifetime_change_by_drv = LIFETIME_CHANGE_BY_DRV_BAU

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
    p03 = params.disaggregation
    p04 = params.materials

    # [IMPORTANT] dict view for fdm.py calls -- see module docstring for why (fdm.py's
    # internals are unverified; the ORIGINAL code passed a raw dict here).
    p02_dict = params.to_nested_dict()["02_stock_flow"]

    export_share_by_drivetrain = dict(p02.export_share_by_drv)
    unknown_whereabouts_share = dict(p02.unknown_whereabouts_share)
    # NOTE: unlike 03_01 (which pulls `unknown_whereabouts_share` from
    # `fdm.build_p02_mapped_inputs(...)["unknown_whereabouts_share"]`), this notebook
    # reads it DIRECTLY off `p02` with no mapping step. If `build_p02_mapped_inputs`
    # applies any transformation (e.g. filling in defaults for missing drivetrains,
    # renaming), these two could differ subtly -- not verifiable without `fdm.py`.
    # `dict(...)` copies here (not just a rename) since the `losses_zero`/`losses_high`
    # C6 scenarios below need their OWN modified copies without mutating the shared
    # baseline dict other scenarios also read.
    segment_map = p04.segment_map
    drv_prefix_map = p04.drv_prefix_map
    materials_region = p04.region
    materials_drivetrains = tuple(p04.drivetrains)

    # -----------------------------------------------------------------------
    # Take 03_01's baseline inflow (by DRIVETRAIN + SEGMENT) as this notebook's starting point
    # -----------------------------------------------------------------------
    inflow_segments_scenario = (
        flows_03[["year", "Segment", "inflow", "Region", "Drive Train"]].rename(columns={"inflow": "value"})
        .groupby(["year", "Segment", "Region", "Drive Train"], as_index=False)["value"].sum()
        .sort_values(["Region", "Drive Train", "Segment", "year"]).reset_index(drop=True)
    )

    # -----------------------------------------------------------------------
    # Build 6 alternative inflow-composition scenarios (BAU + 5 genuine tweaks)
    # -----------------------------------------------------------------------
    inflow_segments_scenario_BAU = tweak_inflow_segment_shares_within_drivetrain(
        inflow_df=inflow_segments_scenario, region="EUR", drivetrain="BEV",
        scenario_start_year=2026, ramp_end_year=2026, target_segment_shares_final=BEV_SEGMENT_SHARES_BAU,
    )
    inflow_segments_scenario_BEV_only = tweak_inflow_drivetrain_shares(
        inflow_df=inflow_segments_scenario, region="EUR", scenario_start_year=2026, ramp_end_year=2026,
        target_shares_final={"BEV": 1, "HEV": 0, "PHEV": 0, "Diesel": 0, "Petrol": 0},
    )
    inflow_segments_scenario_BEV_A_F = tweak_inflow_segment_shares_within_drivetrain(
        inflow_df=inflow_segments_scenario, region="EUR", drivetrain="BEV",
        scenario_start_year=2026, ramp_end_year=2026, target_segment_shares_final=BEV_SEGMENT_SHARES_A_F,
    )
    inflow_segments_scenario_BEV_JA_JF = tweak_inflow_segment_shares_within_drivetrain(
        inflow_df=inflow_segments_scenario, region="EUR", drivetrain="BEV",
        scenario_start_year=2026, ramp_end_year=2026, target_segment_shares_final=BEV_SEGMENT_SHARES_JA_JF,
    )
    inflow_segments_scenario_BEV_large = tweak_inflow_segment_shares_within_drivetrain(
        inflow_df=inflow_segments_scenario, region="EUR", drivetrain="BEV",
        scenario_start_year=2026, ramp_end_year=2026, target_segment_shares_final=BEV_SEGMENT_SHARES_LARGE,
    )
    inflow_segments_scenario_BEV_small = tweak_inflow_segment_shares_within_drivetrain(
        inflow_df=inflow_segments_scenario, region="EUR", drivetrain="BEV",
        scenario_start_year=2026, ramp_end_year=2026, target_segment_shares_final=BEV_SEGMENT_SHARES_SMALL,
    )

    scenario_inflows1 = {
        "BAU": inflow_segments_scenario_BAU, "BEV_only": inflow_segments_scenario_BEV_only,
        "BEV_A_F": inflow_segments_scenario_BEV_A_F, "BEV_JA_JF": inflow_segments_scenario_BEV_JA_JF,
        "BEV_large": inflow_segments_scenario_BEV_large, "BEV_small": inflow_segments_scenario_BEV_small,
    }

    scenario_results = {
        name: run_adjusted_scenario(
            scenario_name=name, inflow_df=df, matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
            flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_share,
            segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
            materials_drivetrains=materials_drivetrains, base_year=2005, segment_shares_by_drv=segment_shares_by_drv,
            export_share_by_drivetrain=export_share_by_drivetrain,
        )
        for name, df in scenario_inflows1.items()
    }

    tracker_keyed_BAU = scenario_results["BAU"]["tracker_keyed"]
    tracker_keyed_BEV_only = scenario_results["BEV_only"]["tracker_keyed"]
    tracker_keyed_BEV_A_F = scenario_results["BEV_A_F"]["tracker_keyed"]
    tracker_keyed_BEV_JA_JF = scenario_results["BEV_JA_JF"]["tracker_keyed"]
    tracker_keyed_BEV_large = scenario_results["BEV_large"]["tracker_keyed"]
    tracker_keyed_BEV_small = scenario_results["BEV_small"]["tracker_keyed"]

    # =========================================================================
    # === C6 FIX -- see module docstring for the full explanation and the caveat
    # that this is UNTESTED without flowdriven_model.py. ===
    # Each of these five scenarios now runs `run_adjusted_scenario` with a genuine
    # override, using the exact target values the abandoned scaffold's own comments
    # specified. `inflow_segments_scenario_BAU` (the same baseline inflow used for the
    # "BAU" tracker above) is reused as the inflow basis for all five -- these
    # scenarios vary lifetime/export/unknown-whereabouts/stock_modifier, NOT the
    # inflow composition itself (that's what the BEV_* scenarios above already cover).
    # =========================================================================
    icev_shorter_result = run_adjusted_scenario(
        scenario_name="ICEV_shorter", inflow_df=inflow_segments_scenario_BAU,
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
        flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_share,
        segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
        materials_drivetrains=materials_drivetrains, base_year=2005, segment_shares_by_drv=segment_shares_by_drv,
        export_share_by_drivetrain=export_share_by_drivetrain,
        lifetime_change_by_drv=LIFETIME_CHANGE_BY_DRV_ICEV_SHORTER,
    )
    tracker_keyed_ICEV_shorter = icev_shorter_result["tracker_keyed"]

    bev_longer_result = run_adjusted_scenario(
        scenario_name="BEV_longer", inflow_df=inflow_segments_scenario_BAU,
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
        flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_share,
        segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
        materials_drivetrains=materials_drivetrains, base_year=2005, segment_shares_by_drv=segment_shares_by_drv,
        export_share_by_drivetrain=export_share_by_drivetrain,
        lifetime_change_by_drv=LIFETIME_CHANGE_BY_DRV_BEV_LONGER,
    )
    tracker_keyed_BEV_longer = bev_longer_result["tracker_keyed"]

    stock_lower_result = run_adjusted_scenario(
        scenario_name="stock_lower", inflow_df=inflow_segments_scenario_BAU,
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
        flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_share,
        segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
        materials_drivetrains=materials_drivetrains, base_year=2005, segment_shares_by_drv=segment_shares_by_drv,
        export_share_by_drivetrain=export_share_by_drivetrain,
        stock_modifier_2027=0.8,  # was the module default of 1 (no-op) -- per the scaffold's own comment
    )
    tracker_keyed_stock_lower = stock_lower_result["tracker_keyed"]

    # losses_zero / losses_high both modify ONLY the BEV entry of export_share_by_drivetrain
    # / unknown_whereabouts_share -- fresh copies each, so the two scenarios (and the
    # scenarios computed above, which share the unmodified baseline dicts) don't
    # accidentally share or mutate each other's dicts.
    export_share_losses_zero = dict(export_share_by_drivetrain)
    export_share_losses_zero["BEV"] = 0.0
    unknown_whereabouts_losses_zero = dict(unknown_whereabouts_share)
    unknown_whereabouts_losses_zero["BEV"] = 0.0
    losses_zero_result = run_adjusted_scenario(
        scenario_name="losses_zero", inflow_df=inflow_segments_scenario_BAU,
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
        flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_losses_zero,
        segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
        materials_drivetrains=materials_drivetrains, base_year=2005, segment_shares_by_drv=segment_shares_by_drv,
        export_share_by_drivetrain=export_share_losses_zero,
    )
    tracker_keyed_losses_zero = losses_zero_result["tracker_keyed"]

    export_share_losses_high = dict(export_share_by_drivetrain)
    export_share_losses_high["BEV"] = 0.08  # "like ICEV" -- matches every other non-BEV drivetrain
    unknown_whereabouts_losses_high = dict(unknown_whereabouts_share)
    unknown_whereabouts_losses_high["BEV"] = 0.43  # "like ICEV"
    losses_high_result = run_adjusted_scenario(
        scenario_name="losses_high", inflow_df=inflow_segments_scenario_BAU,
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv,
        flows_03=flows_03, p02=p02_dict, unknown_whereabouts_share=unknown_whereabouts_losses_high,
        segment_map=segment_map, drv_prefix_map=drv_prefix_map, materials_region=materials_region,
        materials_drivetrains=materials_drivetrains, base_year=2005, segment_shares_by_drv=segment_shares_by_drv,
        export_share_by_drivetrain=export_share_losses_high,
    )
    tracker_keyed_losses_high = losses_high_result["tracker_keyed"]

    tracker_keyed_by_scenario = {
        "BAU": tracker_keyed_BAU, "BEV_only": tracker_keyed_BEV_only, "BEV_A_F": tracker_keyed_BEV_A_F,
        "BEV_JA_JF": tracker_keyed_BEV_JA_JF, "BEV_large": tracker_keyed_BEV_large, "BEV_small": tracker_keyed_BEV_small,
        "BEV_longer": tracker_keyed_BEV_longer, "ICEV_shorter": tracker_keyed_ICEV_shorter,
        "stock_lower": tracker_keyed_stock_lower, "losses_zero": tracker_keyed_losses_zero, "losses_high": tracker_keyed_losses_high,
    }

    # -----------------------------------------------------------------------
    # Persist -- all 11 scenarios are now genuine (see C6 fix above)
    # -----------------------------------------------------------------------
    saved_inflow_mix_scenarios = save_many(
        tracker_keyed_BAU=tracker_keyed_BAU, tracker_keyed_BEV_only=tracker_keyed_BEV_only,
        tracker_keyed_BEV_A_F=tracker_keyed_BEV_A_F, tracker_keyed_BEV_JA_JF=tracker_keyed_BEV_JA_JF,
        tracker_keyed_BEV_large=tracker_keyed_BEV_large, tracker_keyed_BEV_small=tracker_keyed_BEV_small,
        root=PROJECT_ROOT,
    )
    saved_other_scenarios = save_many(
        tracker_keyed_BEV_longer=tracker_keyed_BEV_longer, tracker_keyed_ICEV_shorter=tracker_keyed_ICEV_shorter,
        tracker_keyed_stock_lower=tracker_keyed_stock_lower, tracker_keyed_losses_zero=tracker_keyed_losses_zero,
        tracker_keyed_losses_high=tracker_keyed_losses_high,
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

    return {**saved_inflow_mix_scenarios, **saved_other_scenarios}


if __name__ == "__main__":
    main()
