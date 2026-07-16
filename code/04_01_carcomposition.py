"""
04_01_carcomposition.py
=========================

Stage 04, part 1: combines the "tracker" (per-flow, per-cohort vehicle/segment counts
from stage 03) with COMPONENT-MATERIAL (C-M) composition data for a given drivetrain and
segment, producing total material MASS by year/flow/drivetrain/material, WITH combined
Monte Carlo uncertainty (vehicle-count draws x composition draws) when MC is enabled.

*** REWRITE IN PROGRESS -- being rebuilt step by step, confirmed with testing at each
    step before moving to the next. This file is NOT yet complete. ***

[RENAMED, this round] Was 04_01_materials.py -- renamed to 04_01_carcomposition.py.
Purely a filename change: no code, function names, saved-artifact keys, or output
filenames were altered as part of the rename (see the file-level note near main()
for what a full rename would additionally involve, if wanted).

NEW DATA MODEL (replaces the old bulk composition Excel table entirely):
  1. `36_MonteCarlo_Summary.xlsx` -- summary statistics (mean/median/mode/std/P025/P975)
     per (component, material, drivetrain, segment, year). Used for the SCALAR
     (point-estimate) path. ~10 MB, read directly with pandas/openpyxl (no streaming
     needed).
  2. A histogram file (`SampleHistogram.xlsx` for now; growing to ~5x this size as
     resolution moves from every-5-years to annual) -- 50-bin histograms per
     (component, material, drivetrain, segment, year), used to BOOTSTRAP draws for the
     MC path. Too large to load wholesale -- streamed + filtered + cached to Parquet
     (step 3, not yet built).

SCOPE (confirmed with the user):
  - Only `active_scenario_names()` (i.e. `params.adjusted_flows.scenarios_to_run`,
    typically BAU + at most one comparison scenario) -- matches the same performance
    constraint already established for stage 03_02.
  - `componentCarOther` sheet / drivetrain "Other" is out of scope, ignored.
  - `segment == "standard"` rows are now KEPT (previously dropped, not one of the 12
    real segments) -- [NEW, this round] used for the drivetrain-level ("standard")
    mass path, combined with 03_02_adjustedflows.py's independent by-drivetrain
    re-simulation (the "__direct__" entries already saved in mc_stage03_02_summary)
    instead of the 12-segment tracker. See select_standard_composition,
    combine_scalar_mass_standard, and combine_flow_and_composition_draws(direct=True)
    below, plus the comparison plots this enables (04_01_standard_vs_segments_*.png).
  - For now, only the FIRST available year's composition data per drivetrain is used
    (temporal resolution is currently every 5 years; will become annual later). This is
    controlled by `params.materials.material_mc_time_resolution` (currently only
    "period" is implemented -- "annual"/"both" are placeholders for later).
  - Both a scalar path (point estimate, using `composition_scalar_statistic`, default
    "mean") and a fully vectorized Monte Carlo path (combining vehicle-count draws from
    stage 03_02 with composition draws bootstrapped from the histogram file) must work.

BUILD PROGRESS (step by step, per user's request to test alongside):
  [x] Step 1: `params_schema.py` -- `MaterialsParams` fields for the new data model
      (composition_summary_file_name, composition_scalar_statistic, histogram_file_name,
      histogram_sheet_names_by_drv, material_mc_time_resolution). DONE, verified.
  [x] Step 2: `load_composition_summary()` below -- tidy loader for the summary-stats
      file, used by the scalar path. DONE, verified against the real uploaded file.
  [x] Step 3: `load_histogram_data()` -- streaming + Parquet-cached histogram loader.
      DONE, verified (real sample file + synthetic multi-sheet file).
  [x] Step 4: `bootstrap_composition_draws()` / `_bootstrap_from_bins()` -- vectorized
      bootstrap sampler. DONE, verified (matches real file's own mean/std).
  [x] Step 5: `combine_flow_and_composition_draws()` (MC path, bootstraps vehicle-count
      draws from stage 03_02's SAVED `mc_stage03_02_summary` histograms -- raw draws are
      never persisted to disk, confirmed against the real project) and
      `combine_scalar_mass_from_tracker()` (scalar path, built on `tracker_keyed` --
      the only vehicle-count artifact actually saved). Both DONE, verified against
      synthetic data matching the real confirmed schemas.
  [ ] Step 6: full scalar + MC pipeline wired into `main()`, scoped to
      `scenarios_to_run`, with diagnostic outputs. IN PROGRESS -- an end-to-end real-data
      test script (`test_materials_pipeline_e2e.py`) comes first, to confirm every piece
      works together against the real project before `main()` is written.
      [STILL SHOWS "IN PROGRESS" as of this docstring, but `main()` is fully present
      and wired below -- flagged as a possibly-stale checklist entry, not silently
      marked done here since whether the real-data e2e test actually ran is not
      something this file can confirm on its own.]
  [x] Step 7 [NEW, this round]: drivetrain-level ("standard" composition) mass path,
      combined with 03_02_adjustedflows.py's independent by-drivetrain re-simulation
      (the "__direct__" entries already present in the saved `mc_stage03_02_summary`
      -- no changes needed to 03_02_adjustedflows.py itself), PLUS a "reasonable
      comparison" figure (`04_01_standard_vs_segments_<scenario>_<flow>.png`) against
      the existing 12-segment path's total mass, per drivetrain, scalar point estimate
      + 95% MC range side by side. New: `select_standard_composition`,
      `combine_scalar_mass_standard`, `combine_flow_and_composition_draws(direct=True)`,
      `mc_summary_key(direct=True)`, `plot_standard_vs_segments_comparison`. Changed:
      `load_composition_summary`/`_stream_histogram_sheets` no longer drop "standard"
      rows (histogram cache bumped to v2 so a pre-existing cache can't mask this);
      `select_first_year_composition` now explicitly excludes "standard" (own,
      independent first-year selection, since "standard" isn't guaranteed to share
      the 12 segments' year coverage). DONE, syntax-checked and every new call site
      cross-checked against its definition -- NOT yet run against real data (no
      access to the actual composition/histogram/mc_stage03_02_summary files).
  [x] Step 8 [NEW, this round]: composition data is now ANNUAL (through 2070, per the
      user) -- switched from "one representative first-available-year composition
      applied to every cohort" to matching each vehicle's OWN build year
      (`cohort_year`), clamped to the nearest available composition year when out of
      range (confirmed with the user), with a printed diagnostic of how many rows
      needed clamping. New: `match_cohort_year_to_composition` (the core per-row
      matching primitive -- uses `pd.merge_asof` for nearest-year lookup, THEN a
      regular exact merge for correct fan-out to every component/material; merge_asof
      alone was tried first and found, via testing, to silently drop materials --
      it only returns ONE matching right-side row per left row, not all of them),
      `compute_cohort_year_weights` + `bootstrap_mixed_composition_draws` (the fused,
      memory-safe MC-side per-year-mixture bootstrap -- see Step 9 below for why
      "fused" specifically). Changed: `select_first_year_composition`/
      `select_standard_composition` no longer restrict to one year at all (renamed in
      spirit, not in name, for minimal call-site disruption); `combine_scalar_mass_
      from_tracker`/`combine_scalar_mass_by_year_from_tracker` rewritten to join
      composition PER ROW before aggregating (not aggregate vehicle counts first
      against one flat composition value); `combine_flow_and_composition_draws`'s
      expected input key format changed (no more "year" dimension -- composition
      draws are pre-mixed by cohort-year weight before reaching it).
      TWO REAL LIMITATIONS DISCOVERED AND DOCUMENTED (not silently worked around):
      (1) the "standard" (drivetrain-level) path has NO per-cohort-year vehicle-count
      data anywhere (`mc_stage03_02_summary`'s "__direct__" entries are period-
      cumulative only) -- approximated with the period's own midpoint year instead
      of true per-cohort weighting (see `combine_scalar_mass_standard`'s LIMITATION
      note). (2) the SAME constraint applies to the MC vehicle-count side of the
      12-segment path too -- only the COMPOSITION side could get true per-cohort-year
      treatment; vehicle-count MC draws remain ONE period-cumulative bootstrap per
      (drivetrain, segment), combined against a cohort-year-WEIGHTED MIXTURE of
      composition draws (see `bootstrap_mixed_composition_draws`'s docstring).
      DONE (this step's own design was superseded by Step 9's fix below -- see that
      entry for what actually shipped), syntax-checked; the cohort-year matching + mass aggregation logic was
      ALSO verified numerically in isolation (synthetic data, confirms correct
      fan-out to every component/material and correct clamping) -- NOT yet run
      against real data (no access to the actual composition/histogram files).
  [x] Step 9 [NEW, this round -- fixes a REAL crash found by the user running Step 8
      against real data]: the process was killed by the OS (`zsh: killed`, an OOM
      kill, not a Python exception) partway through composition bootstrapping.
      MEASURED root cause: Step 8's design bootstrapped and CACHED a full
      `n_draws`-length array per (drivetrain, segment, YEAR) combination across the
      whole run -- with ~70 years of annual composition data, 91,882 such
      combinations x 200,000 draws x 8 bytes = 147 GB, before even finishing two
      drivetrains. Fixed by fusing the bootstrap and the cohort-year mixing into ONE
      step (`bootstrap_mixed_composition_draws`, replaces `mix_composition_draws_by_
      cohort_weight` + the old per-year `bootstrap_composition_draws` pre-loop):
      decide which year each of the n_draws belongs to FIRST (cheap -- n_draws small
      integers), then bootstrap only THAT MANY draws from each year directly into
      the final array. Peak memory per (drivetrain, segment, components, material)
      group is back to one (n_draws,) array (1.6 MB at n_draws=200,000), matching
      the pre-annual-composition-data design -- verified numerically (isolated test,
      confirms same statistical result as the old two-step design) and by direct
      memory-footprint calculation. The old cross-scenario lazy-caching machinery in
      main() (`composition_draws_cache`, `_ensure_years_bootstrapped`, etc.) is
      removed entirely -- no longer needed, since the fused function is called fresh
      per (scenario, flow, period) directly from the (cheap, bin-only) histogram
      data. DONE, syntax-checked and numerically verified in isolation -- STILL NOT
      run end-to-end against the real project (the user's own run is what surfaced
      Step 8's bug; Step 9's fix has not yet had the same real-data confirmation).
"""

from __future__ import annotations

import sys
import time
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

SCRIPT_DIR = Path(__file__).resolve().parent

# `src.artifacts` (load_many/save_many/artifact_status) and `src.materials`
# (save_unregistered_scenario_outputs -- same helper the OLD 04_01_materials.py used
# for its per-scenario outputs) -- used by `main()`, imported here (top-level, after
# PROJECT_ROOT/sys.path setup) for consistency with every other numbered stage script.
import src.artifacts as artifacts  # type: ignore
import src.materials as materials  # type: ignore

load_many = artifacts.load_many
save_unregistered_scenario_outputs = materials.save_unregistered_scenario_outputs

COMPOSITION_DIR_NAME = "composition"  # data/raw/composition/, per the user ("It will be
                                       # in a folder data/raw/composition")

# The six raw stat columns carried straight through from the summary file, unchanged.
COMPOSITION_SUMMARY_STAT_COLUMNS = ["mean", "median", "mode", "std", "P025", "P975"]


def load_composition_summary(p04, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """
    Load `36_MonteCarlo_Summary.xlsx` (or whatever `p04.composition_summary_file_name`
    points to) into one tidy long DataFrame, for the SCALAR (point-estimate) materials
    path.

    THE MATH MODEL: no computation here, just tidying. Per-sheet drivetrain tagging,
    then a straight concat.

    Parameters
    ----------
    p04 : params.materials (the `MaterialsParams` dataclass instance)
    project_root : defaults to this script's resolved PROJECT_ROOT

    Behavior confirmed against the real uploaded file:
      - Workbook has 6 sheets: componentCarPetrol, componentCarDiesel, componentCarBEV,
        componentCarHEV, componentCarPHEV, componentCarOther.
      - Each sheet has columns: components, segment, year, variable, mean, median, mode,
        std, P025, P975. No missing values in any of these columns.
      - `componentCarOther` is skipped entirely (out of scope, confirmed).
      - Drivetrain is derived purely from the sheet name by stripping the
        "componentCar" prefix (e.g. "componentCarBEV" -> "BEV") -- this exactly matches
        `p04.drivetrains` = ("BEV", "HEV", "PHEV", "Diesel", "Petrol") once "Other" is
        excluded, so no separate explicit sheet->drivetrain mapping is needed for this
        file (unlike the histogram file, where some drivetrains span >1 sheet).
      - `segment` includes the 12 real segments (A-F, JA-JF) plus a 13th value
        "standard" -- [CHANGED] KEPT here (previously dropped). "standard" is the
        composition source for the drivetrain-level ("standard") mass path, which
        combines it with 03_02_adjustedflows.py's independent by-drivetrain
        re-simulation instead of the 12-segment tracker (see select_standard_
        composition / combine_scalar_mass_standard / combine_flow_and_
        composition_draws(direct=True) below).
      - The "variable" column is the material name (e.g. "calmildSteel", "calAHSS",
        "battery", ...) -- renamed to "material" here for clarity downstream.

    Returns
    -------
    DataFrame with columns:
      components, material, segment, year, drivetrain, mean, median, mode, std, P025, P975
    One row per (component, material, drivetrain, segment, year) -- i.e. exactly the
    grain of the source file, just tidied and drivetrain-tagged.

    Raises
    ------
    FileNotFoundError if the file isn't where expected.
    ValueError if any drivetrain in `p04.drivetrains` has no matching sheet.
    """
    file_path = project_root / "data" / "raw" / COMPOSITION_DIR_NAME / p04.composition_summary_file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Composition summary file not found: {file_path}")

    xl = pd.ExcelFile(file_path)
    frames: list[pd.DataFrame] = []
    found_drivetrains: set[str] = set()

    for sheet_name in xl.sheet_names:
        if sheet_name == "componentCarOther":
            continue
        if not sheet_name.startswith("componentCar"):
            print(f"[load_composition_summary] WARNING: sheet {sheet_name!r} does not "
                  f"start with 'componentCar' -- skipping (not a recognized drivetrain sheet).")
            continue

        drivetrain = sheet_name[len("componentCar"):]
        if drivetrain not in p04.drivetrains:
            print(f"[load_composition_summary] WARNING: sheet {sheet_name!r} -> derived "
                  f"drivetrain {drivetrain!r} is not in params.materials.drivetrains "
                  f"{p04.drivetrains} -- skipping.")
            continue

        df = xl.parse(sheet_name)
        # [CHANGED] 'standard' rows are no longer dropped here -- they're the
        # composition source for the NEW drivetrain-level ("standard") mass path
        # (see select_standard_composition below). The existing 12-segment path
        # still only ever inner-joins against tracker_keyed's real segment codes
        # (A-F/JA-JF), which never include "standard", so it is unaffected by
        # this change without any code change on its side.
        df["drivetrain"] = drivetrain
        df = df.rename(columns={"variable": "material"})
        frames.append(df)
        found_drivetrains.add(drivetrain)

    missing_drvs = set(p04.drivetrains) - found_drivetrains
    if missing_drvs:
        raise ValueError(
            f"load_composition_summary: no sheet found for drivetrain(s) "
            f"{sorted(missing_drvs)} in {file_path.name} (expected a sheet named "
            f"'componentCar<drivetrain>')."
        )

    composition_summary = pd.concat(frames, ignore_index=True)
    ordered_cols = ["components", "material", "segment", "year", "drivetrain"] + COMPOSITION_SUMMARY_STAT_COLUMNS
    composition_summary = composition_summary[ordered_cols]
    return composition_summary


# -----------------------------------------------------------------------------------
# Step 3: streaming + Parquet-cached histogram loader.
# -----------------------------------------------------------------------------------
HISTOGRAM_CACHE_COLUMNS = [
    "components", "material", "segment", "year", "drivetrain",
    "bin_lower", "bin_upper", "count", "frequency",
]
HISTOGRAM_REQUIRED_SOURCE_COLUMNS = {
    "components", "segment", "year", "variable", "bin_lower", "bin_upper", "count", "frequency",
}


def _histogram_cache_path(drivetrain: str, project_root: Path) -> Path:
    cache_dir = project_root / "data" / "processed" / "intermediate"
    # [CHANGED] v2 -> v3: v2's sheet-resolution logic only auto-discovered sibling
    # numbered sheets when the EXACT requested name was entirely absent from the
    # workbook -- an explicitly (and, it turned out, incompletely) listed name like
    # "componentCarPetrol_1" was accepted as sufficient even when sibling sheets
    # "_2".."_5" existed too, silently under-reading real data. Fixed in this
    # version's _stream_histogram_sheets (always discovers the full sibling family,
    # regardless of what was explicitly listed) -- but a v2 cache was built under the
    # OLD, buggy logic and would keep serving that same incomplete result forever
    # (its freshness check only compares against the SOURCE FILE's mtime, which
    # hasn't changed -- fixing the code doesn't invalidate an existing cache). Same
    # versioned-filename approach as the v1->v2 bump: a v2 cache is simply never
    # looked up again, forcing a fresh, now-correct stream + cache-write.
    return cache_dir / f"04_01_histogram_cache_v3_{drivetrain}.parquet"


def _stream_histogram_sheets(
    file_path: Path,
    sheet_names: list[str],
    drivetrain: str,
    verbose: bool = True,
    progress_every: int = 200_000,
) -> pd.DataFrame:
    """
    Stream the given sheet(s) of the (potentially very large, ~260 MB now, growing
    further toward annual resolution) histogram workbook using openpyxl's read-only
    mode, so the full workbook is never materialized in memory -- only the sheet(s)
    mapped to this one drivetrain are touched, row by row.

    `segment == "standard"` rows are KEPT here (previously dropped, same change as
    `load_composition_summary` -- see that function's docstring). The "variable"
    column is renamed to "material" for consistency with the summary loader.

    [FIXED, this round] `sheet_names` entries not found literally in the workbook are
    auto-expanded to every numbered variant matching "{name}_<digits>" (e.g.
    "componentCarBEV" -> "componentCarBEV_1".."_4") -- the source workbook now splits
    each drivetrain across multiple sheets (annual composition data made a single
    sheet too large), and `params.materials.histogram_sheet_names_by_drv` may still
    list the old single-sheet name. See the implementation below for why this is
    auto-discovered rather than requiring an exact, fragile sheet count in
    params_schema.py.

    If `verbose` (default True -- this step can take a while on the real file and was
    otherwise silent, which looks like a hang), prints a line when each sheet starts,
    a running row count every `progress_every` rows read, and a per-sheet summary
    (rows kept, elapsed time) when each sheet finishes.
    """
    import openpyxl
    import re

    t_start_all = time.time()
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    try:
        available = set(wb.sheetnames)

        # [NEW, this round -- fixes a real crash] The source workbook now SPLITS each
        # drivetrain across multiple numbered sheets (e.g. "componentCarBEV_1",
        # "_2", "_3", "_4") instead of one sheet per drivetrain -- confirmed: annual
        # composition data made a single sheet too large. `p04.histogram_sheet_names_
        # by_drv` may still list the OLD literal single-sheet name (e.g.
        # "componentCarBEV"), which no longer exists as-is. For any requested name NOT
        # found literally, auto-discover every sheet matching "{name}_<digits>" and
        # expand to that (sorted numerically, not alphabetically -- "_2" must sort
        # before "_10" if it ever gets that large). This means params_schema.py does
        # NOT need to be updated with an exact, fragile sheet count that could break
        # again if the number of chunks changes.
        # [FIXED, this round -- a real bug, not a config problem] The OLD logic only
        # auto-discovered sibling sheets when the EXACT requested name was entirely
        # ABSENT from the workbook. If `histogram_sheet_names_by_drv` explicitly lists
        # a name that DOES exist (e.g. "componentCarPetrol_1"), the old code accepted
        # it and moved on -- even if that name is only ONE of several sibling sheets
        # (e.g. "_1".."_5") and the requested list is incomplete. This let a manually
        # -- or historically -- specified partial list (confirmed: params_schema.py
        # had `["componentCarPetrol_1", "componentCarPetrol_2"]` when the workbook
        # actually has 5 sheets, "_1".."_5") silently under-read real data on the
        # unlisted sheets, with no error, no warning -- nothing to indicate anything
        # was missing. FIXED: for EVERY requested name, always check whether it's
        # part of a numbered-sheet family (i.e. itself matches "{base}_<digits>", or a
        # sibling "{name}_<digits>" exists) and expand to the FULL family found in the
        # workbook, regardless of what was explicitly listed. This makes correctness
        # depend on what's actually IN the workbook, not on `params_schema.py` having
        # an exact, current sheet count -- the previous auto-discovery already had
        # this goal for the "name not found at all" case; this closes the gap for the
        # "name found, but incomplete" case too.
        def _discover_family(base: str) -> list[str] | None:
            """Every sheet in the workbook matching '{base}_<digits>', sorted
            numerically. None if no such sheet exists at all."""
            pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")
            matches = [(int(m.group(1)), s) for s in available if (m := pattern.match(s))]
            if not matches:
                return None
            matches.sort(key=lambda t: t[0])
            return [s for _, s in matches]

        expanded_sheet_names: list[str] = []
        seen: set[str] = set()
        for name in sheet_names:
            # Is `name` ITSELF one of the numbered siblings (e.g. requested list
            # contains "componentCarPetrol_1")? If so, its "base" is everything
            # before the trailing "_<digits>" -- check for siblings under that base.
            m = re.match(r"^(.*)_(\d+)$", name)
            base_from_numbered_name = m.group(1) if m else None

            family = None
            if base_from_numbered_name is not None:
                family = _discover_family(base_from_numbered_name)
            if family is None:
                # `name` is a bare prefix (or a numbered name with no discoverable
                # siblings, e.g. a single "_1" and nothing else) -- try treating IT as
                # the base directly (covers the original "bare name not found at all"
                # case from before).
                family = _discover_family(name)

            if family is not None:
                new_sheets = [s for s in family if s not in seen]
                if new_sheets:
                    if set(family) != {name} and name not in seen:
                        print(f"  [load_histogram_data:{drivetrain}] {name!r} requested -- "
                              f"discovered its FULL sibling family in the workbook: {family} "
                              f"(using all of them, not just what was explicitly listed).")
                    expanded_sheet_names.extend(new_sheets)
                    seen.update(new_sheets)
            elif name in available:
                if name not in seen:
                    expanded_sheet_names.append(name)
                    seen.add(name)
            else:
                expanded_sheet_names.append(name)  # let the missing-sheet check below report it
        sheet_names = expanded_sheet_names

        missing = [s for s in sheet_names if s not in available]
        if missing:
            raise KeyError(
                f"Histogram sheet(s) {missing} (mapped to drivetrain {drivetrain!r} via "
                f"params.materials.histogram_sheet_names_by_drv) not found in "
                f"{file_path.name}, and no auto-discoverable numbered variant (e.g. "
                f"'{missing[0]}_1') exists either. Available sheets: {wb.sheetnames}"
            )

        rows = []
        for sheet_idx, sheet_name in enumerate(sheet_names, start=1):
            if verbose:
                print(f"  [load_histogram_data:{drivetrain}] streaming sheet "
                      f"{sheet_idx}/{len(sheet_names)}: {sheet_name!r} ...", flush=True)
            t_start_sheet = time.time()
            ws = wb[sheet_name]
            row_iter = ws.iter_rows(values_only=True)
            header = next(row_iter)
            col_idx = {name: i for i, name in enumerate(header) if name is not None}
            missing_cols = HISTOGRAM_REQUIRED_SOURCE_COLUMNS - set(col_idx)
            if missing_cols:
                raise KeyError(
                    f"Histogram sheet {sheet_name!r} is missing required column(s) "
                    f"{sorted(missing_cols)} -- found columns: {list(col_idx)}"
                )

            n_read = 0
            n_kept_this_sheet = 0
            for row in row_iter:
                n_read += 1
                if verbose and n_read % progress_every == 0:
                    elapsed = time.time() - t_start_sheet
                    print(f"    ... {n_read:,} rows read so far in {sheet_name!r} "
                          f"({elapsed:.1f}s elapsed, {n_read / max(elapsed, 1e-9):,.0f} rows/s)", flush=True)
                if row is None or row[col_idx["components"]] is None:
                    continue  # skip fully-blank trailing rows (openpyxl read-only can emit these)
                segment = row[col_idx["segment"]]
                # [CHANGED] 'standard' rows are no longer dropped -- kept for the
                # new drivetrain-level ("standard") mass path. See load_composition_
                # summary's docstring for the same change and its rationale.
                rows.append((
                    row[col_idx["components"]],
                    row[col_idx["variable"]],
                    segment,
                    row[col_idx["year"]],
                    drivetrain,
                    row[col_idx["bin_lower"]],
                    row[col_idx["bin_upper"]],
                    row[col_idx["count"]],
                    row[col_idx["frequency"]],
                ))
                n_kept_this_sheet += 1

            if verbose:
                elapsed = time.time() - t_start_sheet
                print(f"  [load_histogram_data:{drivetrain}] sheet {sheet_name!r} done: "
                      f"{n_read:,} rows read, {n_kept_this_sheet:,} kept, {elapsed:.1f}s", flush=True)
    finally:
        wb.close()

    if verbose:
        print(f"  [load_histogram_data:{drivetrain}] all sheets streamed in "
              f"{time.time() - t_start_all:.1f}s, {len(rows):,} total rows kept -- "
              f"building DataFrame...", flush=True)

    return pd.DataFrame(rows, columns=HISTOGRAM_CACHE_COLUMNS)


def load_histogram_data(
    p04,
    drivetrain: str,
    project_root: Path = PROJECT_ROOT,
    years: set[int] | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the 50-bin histogram data for ONE drivetrain, used to bootstrap Monte Carlo
    composition draws.

    Strategy (confirmed with the user: "Stream + filter, cache to Parquet"):
      1. If a fresh Parquet cache exists for this drivetrain (mtime >= source file's
         mtime), read that instead -- fast, and avoids re-streaming a huge workbook on
         every run.
      2. Otherwise, stream ONLY the sheet(s) mapped to this drivetrain via
         `p04.histogram_sheet_names_by_drv[drivetrain]` (some drivetrains span more than
         one sheet, e.g. Petrol -> componentCarPetrol_1 + componentCarPetrol_2, because
         the source sheets got too long) using openpyxl read-only streaming -- the full
         workbook is never loaded into memory, and non-target sheets/drivetrains are
         never touched at all. Write the result to the Parquet cache for next time.
      3. The cache stores ALL years for this drivetrain (so it stays valid as the data
         densifies from every-5-years to annual); an optional `years` filter is applied
         AFTER loading from cache/stream, not baked into the cache itself.

    Parameters
    ----------
    p04 : params.materials (the `MaterialsParams` dataclass instance)
    drivetrain : one of `p04.drivetrains`, must be a key in `p04.histogram_sheet_names_by_drv`
    years : optional set of years to keep (e.g. {2000} for "first available year only");
        None keeps all years found for this drivetrain
    use_cache : set False to force a fresh stream + cache rewrite (e.g. for testing)
    verbose : default True -- prints progress (cache hit/miss, per-sheet streaming
        progress, parquet write timing). The first run against the real ~260MB+ file
        can take a while with nothing else printed, which looks like a hang -- this
        exists specifically so that doesn't happen. Set False to silence.

    Returns
    -------
    DataFrame with columns:
      components, material, segment, year, drivetrain, bin_lower, bin_upper, count, frequency
    One row per (component, material, segment, year, bin) -- 50 rows per
    (component, material, segment, year) group, matching the source file's grain.

    Raises
    ------
    KeyError if `drivetrain` has no entry in `p04.histogram_sheet_names_by_drv`, or a
        mapped sheet name doesn't exist in the workbook.
    FileNotFoundError if the histogram file isn't where expected.
    """
    if drivetrain not in p04.histogram_sheet_names_by_drv:
        raise KeyError(
            f"No histogram sheet mapping for drivetrain {drivetrain!r} in "
            f"params.materials.histogram_sheet_names_by_drv "
            f"({sorted(p04.histogram_sheet_names_by_drv)})."
        )
    sheet_names = p04.histogram_sheet_names_by_drv[drivetrain]

    file_path = project_root / "data" / "raw" / COMPOSITION_DIR_NAME / p04.histogram_file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Histogram file not found: {file_path}")

    cache_path = _histogram_cache_path(drivetrain, project_root)
    cache_fresh = (
        use_cache
        and cache_path.exists()
        and cache_path.stat().st_mtime >= file_path.stat().st_mtime
    )

    if cache_fresh:
        if verbose:
            print(f"[load_histogram_data:{drivetrain}] using fresh Parquet cache: {cache_path}", flush=True)
        histogram_df = pd.read_parquet(cache_path)
        if verbose:
            print(f"[load_histogram_data:{drivetrain}] loaded {len(histogram_df):,} rows from cache.", flush=True)
    else:
        if verbose:
            reason = "cache disabled" if not use_cache else ("no cache yet" if not cache_path.exists() else "source file is newer than cache")
            print(f"[load_histogram_data:{drivetrain}] {reason} -- streaming {file_path.name} "
                  f"(sheet(s): {sheet_names}). This can take a while on a large file...", flush=True)
        histogram_df = _stream_histogram_sheets(file_path, sheet_names, drivetrain, verbose=verbose)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[load_histogram_data:{drivetrain}] writing Parquet cache ({len(histogram_df):,} rows) "
                  f"to {cache_path} ...", flush=True)
        t_write = time.time()
        histogram_df.to_parquet(cache_path, index=False)
        if verbose:
            print(f"[load_histogram_data:{drivetrain}] cache written in {time.time() - t_write:.1f}s.", flush=True)

    if years is not None:
        histogram_df = histogram_df[histogram_df["year"].isin(years)].reset_index(drop=True)

    return histogram_df


# -----------------------------------------------------------------------------------
# Step 4: vectorized bootstrap sampler.
# -----------------------------------------------------------------------------------
def _bootstrap_from_bins(
    bin_lower: np.ndarray,
    bin_upper: np.ndarray,
    weights: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shared low-level bootstrap primitive, used by BOTH the composition-histogram
    bootstrap (`bootstrap_composition_draws`, weights = pre-normalized `frequency`
    values from the C-M histogram file) and the vehicle-count-histogram bootstrap
    (`bootstrap_vehicle_count_draws_from_summary`, weights = raw integer `frequencies`
    from stage 03_02's saved MC summary) -- same math, same technique, deliberately
    factored out so both sides of the eventual mass combination are bootstrapped
    identically.

    THE MATH MODEL:
      1. Bin selection: pick a bin index per draw, weighted by `weights` (need not be
         pre-normalized -- normalized here), via inverse-CDF sampling on the cumulative
         weight (`np.searchsorted`), vectorized across all `n_draws` at once.
      2. Within-bin position: for a draw that landed in bin i, sample uniformly within
         [bin_lower_i, bin_upper_i).

    Raises
    ------
    ValueError if `weights` sum to (near) zero.
    """
    weights = np.asarray(weights, dtype=float)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("_bootstrap_from_bins: weights sum to zero (or less) -- cannot bootstrap.")
    frequency = weights / weight_sum

    cumulative = np.cumsum(frequency)
    cumulative[-1] = 1.0  # guard against float round-off leaving cumulative[-1] < 1

    u = rng.random(n_draws)
    bin_idx = np.searchsorted(cumulative, u, side="right")
    bin_idx = np.clip(bin_idx, 0, len(weights) - 1)

    lows = np.asarray(bin_lower)[bin_idx]
    highs = np.asarray(bin_upper)[bin_idx]
    return lows + rng.random(n_draws) * (highs - lows)


def bootstrap_composition_draws(
    histogram_df: pd.DataFrame,
    n_draws: int,
    rng: np.random.Generator,
    group_cols: tuple[str, ...] = ("components", "material", "segment", "year", "drivetrain"),
    expected_n_bins: int | None = 50,
    verbose: bool = True,
    progress_every: int = 500,
) -> dict[tuple, np.ndarray]:
    """
    Vectorized bootstrap sampler: for each (component, material, segment, year,
    drivetrain) group in `histogram_df` (50 histogram-bin rows per group, as returned
    by `load_histogram_data`), draw `n_draws` samples matching that group's empirical
    binned distribution -- the composition-uncertainty half of the combined MC draws
    (the other half being stage 03_02's vehicle-count draws).

    THE MATH MODEL:
      1. Bin selection: pick a bin index per draw, weighted by that bin's `frequency`,
         via inverse-CDF sampling on the cumulative frequency (`np.searchsorted`) --
         vectorized across all `n_draws` at once per group (no per-draw Python loop).
      2. Within-bin position: for a draw that landed in bin i, sample uniformly within
         [bin_lower_i, bin_upper_i) -- reconstructs a continuous distribution matching
         the binned empirical histogram. Confirmed on the real sample file: bins are
         contiguous and equal-width within a group, and `frequency` sums to 1.0 exactly
         (`frequency == count / sum(count)`), so this is a faithful bootstrap of the
         underlying (unobserved) continuous draws that produced the histogram.

    One `rng.random(n_draws)` call per group for bin selection and one more for the
    within-bin position -- both fully vectorized (no Python-level loop over draws), only
    the loop over GROUPS is in Python, which is unavoidable since each group has its own
    bin edges/weights.

    Parameters
    ----------
    histogram_df : output of `load_histogram_data` (optionally for multiple drivetrains
        concatenated together -- `drivetrain` is part of `group_cols` so groups never
        cross drivetrains unless the caller explicitly drops that column)
    n_draws : number of bootstrap draws per group
    rng : numpy Generator (caller controls seeding -- same seed-spawning convention as
        the rest of the MC engine)
    group_cols : columns identifying one histogram (one material's mass distribution
        for one component/segment/year/drivetrain)
    expected_n_bins : if not None, raises if any group's bin count doesn't match this
        (malformed/truncated histogram data, not something to silently paper over);
        pass None to skip this check
    verbose : default True -- prints a running progress line every `progress_every`
        groups (a full drivetrain can have thousands of (component, material, segment,
        year) groups, so this loop is not always instant). Set False to silence.
    progress_every : how many groups between progress prints

    Returns
    -------
    dict mapping each `group_cols` tuple -> an (n_draws,) float array of bootstrapped
    values (kg of that material in that component, per vehicle).

    Raises
    ------
    ValueError if a group's bin count doesn't match `expected_n_bins`, or its
    frequencies sum to (near) zero.
    """
    t_start = time.time()
    n_groups_total = histogram_df[list(group_cols)].drop_duplicates().shape[0]
    if verbose:
        print(f"[bootstrap_composition_draws] {n_groups_total:,} groups to bootstrap "
              f"(n_draws={n_draws:,} each)...", flush=True)

    draws_by_group: dict[tuple, np.ndarray] = {}
    for i, (group_key, group_df) in enumerate(histogram_df.groupby(list(group_cols), sort=False), start=1):
        if verbose and i % progress_every == 0:
            elapsed = time.time() - t_start
            print(f"  ... {i:,}/{n_groups_total:,} groups bootstrapped "
                  f"({elapsed:.1f}s elapsed, {i / max(elapsed, 1e-9):,.0f} groups/s)", flush=True)
        group_df = group_df.sort_values("bin_lower")
        n_bins = len(group_df)
        if expected_n_bins is not None and n_bins != expected_n_bins:
            raise ValueError(
                f"bootstrap_composition_draws: group {group_key} has {n_bins} bins, "
                f"expected {expected_n_bins}."
            )

        frequency = group_df["frequency"].to_numpy(dtype=float)
        freq_sum = frequency.sum()
        if freq_sum <= 0:
            raise ValueError(f"bootstrap_composition_draws: group {group_key} has zero total frequency.")
        if not np.isclose(freq_sum, 1.0, atol=1e-6):
            # Defensive re-normalization -- warn rather than silently ignoring a data issue.
            print(f"[bootstrap_composition_draws] WARNING: group {group_key} frequencies "
                  f"sum to {freq_sum:.6f}, not 1.0 -- renormalizing.")

        bin_lower = group_df["bin_lower"].to_numpy(dtype=float)
        bin_upper = group_df["bin_upper"].to_numpy(dtype=float)

        draws_by_group[group_key] = _bootstrap_from_bins(bin_lower, bin_upper, frequency, n_draws, rng)

    if verbose:
        print(f"[bootstrap_composition_draws] done: {len(draws_by_group):,} groups in "
              f"{time.time() - t_start:.1f}s.", flush=True)

    return draws_by_group


# -----------------------------------------------------------------------------------
# Step 5, part A: bootstrap vehicle-count draws from stage 03_02's SAVED summary
# artifact (`mc_stage03_02_summary`) -- the disk-persisted counterpart to raw MC draws.
# -----------------------------------------------------------------------------------
# IMPORTANT ARCHITECTURE NOTE, confirmed against the real project via
# diagnose_tracker_schema.py / _2.py / _3.py:
#   - 03_02_adjustedflows.py's raw per-draw MC arrays (`mc["by_group"][...]["periods"]
#     [period]["cumulative_inflow"/"cumulative_collected"]`) exist ONLY in memory during
#     that script's own run -- they are NEVER written to disk.
#   - What IS persisted is `mc_stage03_02_summary`, a dict keyed by
#     f"{scenario_name}__{start}-{end}__{drivetrain}__{metric}" (drivetrain-level,
#     summed across segments) AND f"{scenario_name}__{start}-{end}__{drivetrain}__
#     {segment}__{metric}" (segment-level) -- confirmed exact key format against
#     `_summarize_period_result`'s call sites in 03_02_adjustedflows.py's main(), and
#     verified for real against the actual saved file (round 3: key
#     'BAU__1975-2070__BEV__cumulative_survival' -> dict with keys n, mean, median,
#     mode, std, p2_5, p97_5, min, max, bin_edges (51 values), frequencies (50 integer
#     counts)).
#   - Since 04_01 is a SEPARATE script that only reads saved artifacts (same pattern as
#     the original 04_01_materials.py, which only ever loaded `tracker_keyed*`), the MC
#     combine step bootstraps vehicle-count draws from this saved histogram -- the exact
#     same technique already used for the composition side -- rather than assuming
#     access to raw draws that were never saved. This requires NO changes to
#     03_02_adjustedflows.py.
#   - Side effect (a genuine improvement, not just a workaround): since BOTH sides of
#     the combination are now independent bootstraps from their own saved histograms,
#     `n_draws` for the materials-stage combination is fully decoupled from stage
#     03_02's original `n_draws_mc` -- no length-matching constraint remains.
# -----------------------------------------------------------------------------------
def mc_summary_key(
    scenario_name: str, period: tuple[int, int], drivetrain: str, metric: str,
    segment: str | None = None, direct: bool = False,
) -> str:
    """
    Build the exact key format `mc_stage03_02_summary` uses, confirmed against the
    real saved artifact:
      - segment-level:            f"{scenario}__{start}-{end}__{drivetrain}__{segment}__{metric}"
      - drivetrain-level (segment-sum):  f"{scenario}__{start}-{end}__{drivetrain}__{metric}"
      - drivetrain-level (INDEPENDENT re-simulation, `direct=True`) -- [NEW]:
            f"{scenario}__{start}-{end}__{drivetrain}__direct__{metric}"
        This is 03_02_adjustedflows.py's "by-drivetrain, independent re-simulation"
        (see that script's run_adjusted_scenario docstring) -- a genuinely SEPARATE
        simulation from the segment-sum, bypassing the base-year segment-share
        split entirely. Used here as the vehicle-count source for the drivetrain-
        level ("standard" composition) mass path. `direct=True` requires
        `segment=None` (drivetrain-level only, no segment concept in that
        re-simulation).
    """
    if direct and segment is not None:
        raise ValueError("mc_summary_key: direct=True requires segment=None (no segment concept in the independent by-drivetrain re-simulation).")
    start, end = period
    period_label = f"{start}-{end}"
    if direct:
        return f"{scenario_name}__{period_label}__{drivetrain}__direct__{metric}"
    if segment is None:
        return f"{scenario_name}__{period_label}__{drivetrain}__{metric}"
    return f"{scenario_name}__{period_label}__{drivetrain}__{segment}__{metric}"


def bootstrap_vehicle_count_draws_from_summary(
    summary: dict,
    key: str,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Bootstrap vehicle-count draws from ONE entry of the saved `mc_stage03_02_summary`
    dict (build the key with `mc_summary_key`), using the same
    `_bootstrap_from_bins` primitive as `bootstrap_composition_draws`.

    Parameters
    ----------
    summary : the loaded `mc_stage03_02_summary` dict (e.g.
        `load_many("mc_stage03_02_summary", root=PROJECT_ROOT)["mc_stage03_02_summary"]`)
    key : exact key string, e.g. "BAU__1975-2070__BEV__A__cumulative_inflow" -- see
        `mc_summary_key`
    n_draws, rng : as elsewhere (independent of stage 03_02's own n_draws -- see the
        architecture note above)

    Returns
    -------
    (n_draws,) float array.

    Raises
    ------
    KeyError if `key` is not present in `summary` (e.g. that scenario/period/
    drivetrain/segment combination wasn't actually run/summarized).
    """
    if key not in summary:
        raise KeyError(
            f"bootstrap_vehicle_count_draws_from_summary: key {key!r} not found in "
            f"mc_stage03_02_summary. Available keys containing the same drivetrain/"
            f"scenario prefix: "
            f"{[k for k in summary if k.startswith(key.rsplit('__', 1)[0])][:10]}"
        )
    entry = summary[key]
    bin_edges = np.asarray(entry["bin_edges"], dtype=float)
    frequencies = np.asarray(entry["frequencies"], dtype=float)
    bin_lower = bin_edges[:-1]
    bin_upper = bin_edges[1:]
    return _bootstrap_from_bins(bin_lower, bin_upper, frequencies, n_draws, rng)


# -----------------------------------------------------------------------------------
# Step 5, part B: combine stage 03_02's (bootstrapped) vehicle-count draws with
# composition MC draws.
# -----------------------------------------------------------------------------------
# `flow_metric` -> the real `flow` value used in `tracker_keyed` (for the scalar path)
# vs. the metric-name suffix used in `mc_stage03_02_summary` (for the MC path) --
# confirmed these are DIFFERENT strings for the same concept: tracker_keyed's `flow`
# column uses "collected"; mc_stage03_02_summary's metric suffix uses
# "cumulative_collected". Both mean the same thing (end-of-life vehicles). This map
# translates one flow_metric argument into whichever spelling each function needs.
FLOW_TO_TRACKER_VALUE = {"cumulative_inflow": "inflow", "cumulative_collected": "collected"}
# [NEW] The reverse mapping -- needed by combine_scalar_mass_standard, which starts
# from a `flow` value ("inflow"/"collected", matching FLOW_VALUES_IN_SCOPE) and needs
# the mc_stage03_02_summary metric-name spelling to look up, same two strings as
# FLOW_TO_TRACKER_VALUE above, just inverted rather than duplicated by hand.
FLOW_TO_MC_METRIC = {v: k for k, v in FLOW_TO_TRACKER_VALUE.items()}

# -----------------------------------------------------------------------------------
# Vehicle-count units. CONFIRMED via diagnose_vehicle_count_units.py against the real
# project: `tracker_keyed`'s `amount` column (and, by construction, `mc_stage03_02_
# summary`'s "cumulative_inflow"/"cumulative_collected" entries, which 03_02_
# adjustedflows.py builds from that same underlying vehicle-count data) is in MILLIONS
# of vehicles, not raw vehicle count. Real diagnostic output: raw-count interpretation
# implied ~23 new vehicles/year EU-wide (impossible); millions-of-vehicles
# interpretation implied ~22.8 million/year, which lines up with real EU-27 new-car
# registration rates (~10-15 million/year) to within the same order of magnitude. This
# also explains the "one car" scale bug the user spotted (HEV collected mass peaking
# under 1000 kg fleet-wide) -- vehicle counts were being multiplied into mass while
# still in millions-of-vehicles units instead of raw vehicle count.
# -----------------------------------------------------------------------------------
VEHICLE_COUNT_UNIT_SCALE = 1_000_000.0

# All mass CALCULATIONS stay in kg throughout (mass_by_year tables, MC draws, saved
# pickles) -- this constant is used ONLY at plot-render time, to display tonnes instead
# of kg on the mass-by-year figures (per the user's request), without touching any
# upstream numbers other code might depend on.
KG_PER_TONNE = 1_000.0


def _floor_amount_at_zero(df: pd.DataFrame, context_label: str = "") -> pd.DataFrame:
    """
    [NEW, this round -- fixes a real crash] Floor `amount` at 0, with a diagnostic
    print if anything was actually negative.

    WHY THIS IS NEEDED: `tracker_keyed`'s `amount` column, for `flow="inflow"` rows,
    deliberately carries the RAW residual (`target - remaining_total`), which CAN be
    negative in years where natural attrition outpaces a falling target -- the exact
    same "always show the raw residual, never hide a negative-inflow year" convention
    as stage 02's own `inflow_by_year` (see `stockflow_model.py`'s `_run_cohort_
    recurrence` docstring). The ACTUAL simulated inflow used elsewhere in the
    pipeline is `inflow_applied_by_year` (`max(raw, 0)`) -- but `tracker_keyed`
    itself only carries the raw diagnostic value, not both.

    This bit here specifically because a negative `amount` in ONE cohort_year, even
    with a POSITIVE total across the whole period, produces a NEGATIVE per-year
    weight in `compute_cohort_year_weights` -- and `rng.choice(p=...)` requires
    every probability to be non-negative, so it crashes rather than silently
    misbehaving. Flooring here matches what the rest of the pipeline already treats
    as "what actually happened" (`inflow_applied`, never negative) rather than the
    raw diagnostic value this table happens to carry.
    """
    negative_mask = df["amount"] < 0
    n_negative = int(negative_mask.sum())
    if n_negative > 0:
        label = f" [{context_label}]" if context_label else ""
        print(f"[_floor_amount_at_zero]{label} {n_negative:,}/{len(df):,} rows had negative "
              f"'amount' (raw negative-inflow residual, not a real vehicle count -- see "
              f"stockflow_model.py's inflow_by_year vs inflow_applied_by_year) -- floored to 0.")
        df = df.copy()
        df.loc[negative_mask, "amount"] = 0.0
    return df


def match_cohort_year_to_composition(
    rows: pd.DataFrame,
    composition: pd.DataFrame,
    row_year_col: str,
    exact_cols: list[str],
    context_label: str = "",
) -> pd.DataFrame:
    """
    [NEW, this round] Match each row in `rows` to the NEAREST available composition
    year, for the SAME `exact_cols` group (e.g. drivetrain, or drivetrain+segment) --
    the core operation behind the switch from "one representative composition year
    applied to every cohort" to "each vehicle's own build-year composition."

    WHY NEAREST-YEAR, NOT EXACT-YEAR: composition data may not cover every year a
    cohort could have been built in (e.g. a pre-2000 backcast cohort still being
    scrapped decades later, if composition data starts later than that). Confirmed
    with the user: clamp to the NEAREST available composition year in that case,
    with a printed diagnostic count of how many rows needed clamping (not silent).

    Uses `pd.merge_asof(..., direction="nearest")`, which naturally implements BOTH
    exact matches (when available) AND nearest-neighbor clamping at the boundaries
    (when a row's year falls entirely outside the composition data's range) in one
    operation -- no separate boundary-clamping code path needed.

    Parameters
    ----------
    rows : DataFrame with a `row_year_col` column (e.g. "cohort_year") and every
        column in `exact_cols`
    composition : DataFrame with a "year" column and every column in `exact_cols`
    row_year_col : which column in `rows` holds the year to match (typically
        "cohort_year" -- the vehicle's BUILD year, not its flow/scrap year -- see
        the module-level note on cohort-year vs. flow-year composition)
    exact_cols : columns that must match EXACTLY (not nearest) -- e.g.
        `["drivetrain", "segment"]` for the 12-segment path, `["drivetrain"]` for
        the standard path (no segment concept there)
    context_label : included in the printed diagnostic, for telling multiple calls
        apart in a busy log (e.g. "BEV/collected")

    Returns
    -------
    `rows` merged with `composition` (every composition column, e.g. components/
    material/the scalar statistic columns/bin data, fanned out across every
    composition row matching that (exact_cols, nearest year)), plus a `matched_year`
    column (the composition year actually used -- compare against `row_year_col` to
    see which rows were clamped).
    """
    if rows.empty:
        return rows.assign(matched_year=pd.Series(dtype="int64"))

    # [FIXED, found in testing] pd.merge_asof returns exactly ONE matching right-side
    # row per left row (the nearest one) -- it does NOT fan out to every right-side
    # row sharing that year, unlike a regular merge. Composition data has MANY rows
    # per (exact_cols, year) -- one per (components, material) -- so merge_asof
    # directly against the full composition table was silently dropping materials
    # (which one "won" the asof match was essentially arbitrary tie-breaking, not a
    # real selection). Fixed with two steps: (1) merge_asof against a DEDUPLICATED
    # (exact_cols, year) table to find just the matched year for each row, (2) a
    # regular EXACT merge against the full composition table on (exact_cols,
    # matched_year) -- which correctly fans out to every component/material.
    year_lookup = composition[exact_cols + ["year"]].drop_duplicates().sort_values("year").reset_index(drop=True)
    rows_sorted = rows.sort_values(row_year_col).reset_index(drop=True)

    with_matched_year = pd.merge_asof(
        rows_sorted, year_lookup,
        left_on=row_year_col, right_on="year",
        by=exact_cols, direction="nearest",
    ).rename(columns={"year": "matched_year"})

    n_clamped = int((with_matched_year["matched_year"] != with_matched_year[row_year_col]).sum())
    if n_clamped > 0:
        label = f" [{context_label}]" if context_label else ""
        print(f"[match_cohort_year_to_composition]{label} {n_clamped:,}/{len(with_matched_year):,} rows "
              f"clamped to the nearest available composition year (their own {row_year_col} "
              f"had no exact composition-year match).")

    unmatched = with_matched_year["matched_year"].isna()
    if unmatched.any():
        missing_groups = with_matched_year.loc[unmatched, exact_cols].drop_duplicates()
        print(f"[match_cohort_year_to_composition] WARNING: {int(unmatched.sum()):,} rows had "
              f"NO composition match at all (not even a clamp) for these {exact_cols} "
              f"combinations -- dropped:\n{missing_groups.to_string(index=False)}")
        with_matched_year = with_matched_year.loc[~unmatched].copy()
    with_matched_year["matched_year"] = with_matched_year["matched_year"].astype(int)

    # Step 2: exact merge against the FULL composition table, on (exact_cols,
    # matched_year) -- this is a regular merge, so it correctly fans out to every
    # (components, material) row available for that (exact_cols, matched_year).
    merged = with_matched_year.merge(
        composition, left_on=exact_cols + ["matched_year"], right_on=exact_cols + ["year"],
        how="inner", suffixes=("", "_composition"),
    )
    return merged.drop(columns=["year"])


def bootstrap_mixed_composition_draws(
    histogram_df: pd.DataFrame,
    cohort_year_weights: dict[tuple[str, str], dict[int, float]],
    n_draws: int,
    rng: np.random.Generator,
    verbose: bool = True,
    progress_every: int = 500,
) -> dict[tuple, np.ndarray]:
    """
    [FIXED, this round -- replaces the old bootstrap_composition_draws + mix_
    composition_draws_by_cohort_weight two-step pipeline, which caused a real OOM
    crash] Bootstrap composition draws DIRECTLY as a cohort-year-weighted mixture,
    for one (drivetrain, segment, components, material) group at a time, WITHOUT
    ever materializing a full `n_draws` array per individual year first.

    THE BUG THIS FIXES: the old design called `bootstrap_composition_draws` for
    EVERY (drivetrain, segment, year) combination needed -- a FULL n_draws-length
    array per year -- then mixed them after the fact by masking. With annual
    composition data spanning ~70 years, that multiplies memory by ~70x compared to
    the old "one representative year" design. Measured on the real project: 91,882
    (group x year) combinations x 200,000 draws x 8 bytes = 147 GB -- the process was
    killed by the OS (`zsh: killed`) before finishing even two drivetrains.

    THE FIX: for a given (drivetrain, segment), the "which year does draw i belong
    to" ASSIGNMENT is decided ONCE (rng.choice over years, weighted -- cheap, just
    n_draws small integers), then SHARED across every (components, material) group
    in that (drivetrain, segment) -- exactly like the old design's year_choice_idx,
    but computed BEFORE bootstrapping rather than after. For each year, only the
    `count` draws actually assigned to it are bootstrapped (`_bootstrap_from_bins`
    already supports any draw count, not just the full n_draws), and written
    directly into their positions in the output array. Total draws bootstrapped per
    group across all its years sums to EXACTLY n_draws, not n_draws x n_years --
    peak memory per group is back to one (n_draws,) float64 array (1.6 MB at
    n_draws=200,000), same as the pre-annual-composition-data design.

    Parameters
    ----------
    histogram_df : RAW histogram data (50-bin rows), for every (drivetrain, segment,
        year) combination that could possibly be needed -- e.g. `pd.concat` of
        `histogram_segments_by_drivetrain[drv]` for every relevant drivetrain. Only
        cheap to hold in memory (bin edges/frequencies, not draws) -- this is NOT
        where the memory cost was.
    cohort_year_weights : {(drivetrain, segment): {matched_year: weight, ...}}, each
        inner dict's weights summing to ~1.0 -- see `compute_cohort_year_weights`.
        ONLY the (drivetrain, segment) combinations present here are processed --
        this is what scopes the work to exactly what one (scenario, flow, period)
        actually needs, not every combination that could ever exist.
    n_draws, rng : as elsewhere -- `rng` drives BOTH the year assignment and the
        within-year bootstrap (one shared stream, unlike the old two-function design
        which used a separate `mixing_rng` -- no longer needed since there's only
        one function now)
    verbose, progress_every : progress printing, same convention as `bootstrap_
        composition_draws`

    Returns
    -------
    dict keyed by (drivetrain, segment, components, material) -> (n_draws,) mixed
    composition-value array [kg/vehicle]
    """
    t_start = time.time()
    mixed: dict[tuple, np.ndarray] = {}
    n_dt_seg_total = len(cohort_year_weights)
    n_groups_done = 0

    for (drivetrain, segment), year_weights in cohort_year_weights.items():
        years = list(year_weights.keys())
        if not years:
            continue
        w = np.array([year_weights[y] for y in years], dtype=float)
        # [DEFENSIVE, this round] Floor any negative weight at 0 before normalizing --
        # rng.choice(p=...) requires non-negative probabilities. Weights should
        # already be non-negative by the time they get here (compute_cohort_year_
        # weights sources them from tracker amounts that _floor_amount_at_zero has
        # already floored) -- this is a second line of defense, not the primary fix,
        # in case a future caller builds cohort_year_weights some other way.
        if np.any(w < 0):
            print(f"[bootstrap_mixed_composition_draws] WARNING: negative weight(s) for "
                  f"(drivetrain={drivetrain!r}, segment={segment!r}) -- floored to 0 "
                  f"before normalizing (should not happen if weights came from "
                  f"compute_cohort_year_weights on already-floored tracker amounts).")
            w = np.clip(w, 0.0, None)
        weight_sum = w.sum()
        if weight_sum <= 0:
            continue  # every year's weight floored to 0 -- nothing to mix for this group
        w = w / weight_sum

        # ONE year assignment per draw, shared across every (components, material)
        # group in this (drivetrain, segment) -- this is the key memory fix: decided
        # BEFORE any bootstrapping happens, so each year's bootstrap only ever
        # produces exactly as many draws as were actually assigned to it.
        if len(years) == 1:
            year_of_draw = np.full(n_draws, years[0])
        else:
            year_choice_idx = rng.choice(len(years), size=n_draws, p=w)
            year_of_draw = np.asarray(years)[year_choice_idx]
        masks = {y: (year_of_draw == y) for y in years}
        counts = {y: int(masks[y].sum()) for y in years}

        sub = histogram_df[
            (histogram_df["drivetrain"] == drivetrain)
            & (histogram_df["segment"] == segment)
            & (histogram_df["year"].isin(years))
        ]
        if sub.empty:
            continue

        for (components, material), grp in sub.groupby(["components", "material"], sort=False):
            n_groups_done += 1
            if verbose and n_groups_done % progress_every == 0:
                elapsed = time.time() - t_start
                print(f"  ... {n_groups_done:,} groups bootstrapped ({elapsed:.1f}s elapsed, "
                      f"{n_groups_done / max(elapsed, 1e-9):,.0f} groups/s)", flush=True)

            out = np.empty(n_draws, dtype=float)
            for y in years:
                cnt = counts[y]
                if cnt == 0:
                    continue
                year_bins = grp[grp["year"] == y].sort_values("bin_lower")
                if year_bins.empty:
                    continue
                out[masks[y]] = _bootstrap_from_bins(
                    year_bins["bin_lower"].to_numpy(dtype=float),
                    year_bins["bin_upper"].to_numpy(dtype=float),
                    year_bins["frequency"].to_numpy(dtype=float),
                    cnt, rng,
                )
            mixed[(drivetrain, segment, components, material)] = out

    if verbose:
        print(f"[bootstrap_mixed_composition_draws] done: {len(mixed):,} groups across "
              f"{n_dt_seg_total:,} (drivetrain, segment) combinations in "
              f"{time.time() - t_start:.1f}s.", flush=True)

    return mixed


def compute_cohort_year_weights(
    tracker_keyed: dict[tuple[str, str], pd.DataFrame],
    flow: str,
    period: tuple[int, int],
    region: str,
    composition_years_by_drivetrain: dict[str, np.ndarray],
) -> dict[tuple[str, str], dict[int, float]]:
    """
    [NEW, this round] From the deterministic tracker, compute each matched composition
    year's REAL share of a (drivetrain, segment)'s total vehicle count within `period`
    -- the weights `bootstrap_mixed_composition_draws` needs. Each tracker row's
    OWN `cohort_year` is clamped to the nearest available composition year first (same
    rule as the scalar path -- see `match_cohort_year_to_composition`), then weights
    are the clamped-year-grouped share of total `amount`.

    Parameters
    ----------
    tracker_keyed, flow, period, region : same filtering as `combine_scalar_mass_
        from_tracker`
    composition_years_by_drivetrain : {drivetrain: sorted array of available
        composition years} -- used only to determine the nearest-year clamping,
        not to look up actual composition values (this function returns weights only)

    Returns
    -------
    {(drivetrain, segment): {matched_year: weight, ...}}, each inner dict's weights
    summing to 1.0 (drivetrain, segment combinations with zero vehicles in this
    period are simply absent, not present with weight 0)
    """
    start, end = period
    weights: dict[tuple[str, str], dict[int, float]] = {}
    for (grp_region, drivetrain), df in tracker_keyed.items():
        if grp_region != region:
            continue
        sub = df[(df["flow"] == flow) & (df["scrap_year"] >= start) & (df["scrap_year"] <= end)]
        if sub.empty:
            continue
        sub = _floor_amount_at_zero(sub, context_label=f"{drivetrain}/{flow}/weights")
        available_years = composition_years_by_drivetrain.get(drivetrain)
        if available_years is None or len(available_years) == 0:
            continue
        cohort_years = sub["cohort_year"].to_numpy()
        # Nearest-year clamp via searchsorted -- same "nearest" semantics as
        # match_cohort_year_to_composition's merge_asof, implemented directly here
        # since we only need the clamped year, not a full row-level merge.
        idx = np.searchsorted(available_years, cohort_years)
        idx = np.clip(idx, 0, len(available_years) - 1)
        idx_prev = np.clip(idx - 1, 0, len(available_years) - 1)
        # [FIXED, found via cross-check testing] `<=` not `<` -- must match
        # pd.merge_asof(direction="nearest")'s own tie-breaking EXACTLY (confirmed:
        # pandas prefers the earlier/lower year on an exact tie) for consistency with
        # match_cohort_year_to_composition, which is what actually determines the
        # composition VALUES used elsewhere. A `<` vs `<=` mismatch here would only
        # bite at an exact midpoint between two available composition years, but
        # would then silently disagree with the value-matching about which year
        # "won" -- weights computed for one year, values looked up for another.
        use_prev = np.abs(available_years[idx_prev] - cohort_years) <= np.abs(available_years[idx] - cohort_years)
        clamped_years = np.where(use_prev, available_years[idx_prev], available_years[idx])

        for segment in sub["Segment"].unique():
            seg_mask = sub["Segment"].to_numpy() == segment
            seg_amount = sub["amount"].to_numpy()[seg_mask]
            seg_years = clamped_years[seg_mask]
            total = seg_amount.sum()
            if total <= 0:
                continue
            year_totals: dict[int, float] = {}
            for y, a in zip(seg_years, seg_amount):
                year_totals[int(y)] = year_totals.get(int(y), 0.0) + float(a)
            weights[(drivetrain, segment)] = {y: v / total for y, v in year_totals.items()}

    return weights


def combine_flow_and_composition_draws(
    mc_summary: dict,
    scenario_name: str,
    period: tuple[int, int],
    flow_metric: str,
    composition_draws_by_group: dict[tuple, np.ndarray],
    n_draws: int,
    rng: np.random.Generator,
    direct: bool = False,
) -> dict[tuple, np.ndarray]:
    """
    Combine ONE scenario's vehicle-count MC uncertainty with per-vehicle composition MC
    draws into material MASS draws.

    ARCHITECTURE (see the note above `bootstrap_vehicle_count_draws_from_summary`):
    stage 03_02's raw per-draw arrays are never saved to disk -- only
    `mc_stage03_02_summary` (mean/median/mode/std/P2.5/P97.5 + a 50-bin histogram per
    scenario/period/drivetrain[/segment]/metric) is. So vehicle-count draws here are
    BOOTSTRAPPED from that saved histogram (`bootstrap_vehicle_count_draws_from_summary`,
    same technique as the composition side), not read from an in-memory MC result.

    THE MATH MODEL (confirmed with the user -- "independent random pairing"):
      mass_draws[i] = count_draws[i] * composition_draws[i]  for i in range(n_draws)
    Both draw arrays are independent bootstraps from their own saved histograms and are
    multiplied elementwise BY DRAW INDEX -- statistically valid because each is
    internally iid across the draw axis, and the two sources (vehicle-count history/
    lifetime uncertainty vs. C-M composition uncertainty) are independent by
    construction. Because both sides are bootstrapped fresh here, `n_draws` is a free
    choice for this materials-stage combination -- it does NOT need to match stage
    03_02's original `n_draws_mc`.

    Parameters
    ----------
    mc_summary : the loaded `mc_stage03_02_summary` dict
    scenario_name : e.g. "BAU" -- must match a scenario actually present in
        `mc_summary`'s keys (i.e. one of `AdjustedFlowsParams.active_scenario_names()`)
    period : (start, end), must match one of `params.monte_carlo.output_periods`
    flow_metric : "cumulative_inflow" or "cumulative_collected"
    composition_draws_by_group : [CHANGED, this round] output of `mix_composition_
        draws_by_cohort_weight` -- keyed by (drivetrain, segment, components,
        material), i.e. NO YEAR dimension any more (each array already reflects the
        real cohort-year mix for that (drivetrain, segment) -- see that function's
        docstring for why the year dimension is collapsed BEFORE this combine step,
        not inside it). When `direct=True`, every entry's `segment` is expected to be
        "standard" (see `select_standard_composition`) -- the segment component of
        the key is ignored for the `mc_summary` lookup (drivetrain-level keys have no
        segment), but still used for the OUTPUT key so callers can tell this mass
        came from the "standard" composition path.
    n_draws : number of combined draws to produce (independent of stage 03_02's own
        n_draws -- see architecture note)
    rng : numpy Generator for the vehicle-count bootstrap (composition draws are
        assumed already sampled -- this function does not draw them)
    direct : [NEW] False (default) = existing 12-segment path, vehicle-count draws
        bootstrapped from `mc_summary`'s SEGMENT-level entries (the scenario's
        segment-share-split simulation). True = NEW drivetrain-level ("standard")
        path, vehicle-count draws bootstrapped from `mc_summary`'s "__direct__"
        entries (03_02_adjustedflows.py's INDEPENDENT by-drivetrain re-simulation,
        which bypasses the segment-share split entirely) -- see `mc_summary_key`'s
        docstring for the exact key format difference.

    Returns
    -------
    dict keyed by (drivetrain, segment, components, material) -> (n_draws,) mass array
    [kg], for every (drivetrain, segment) present in `composition_draws_by_group` AND
    for which a matching `mc_summary` entry exists. (drivetrain, segment) combinations
    with NO matching `mc_summary` entry are SKIPPED (confirmed with the user -- this
    means that combination has zero/negligible vehicles in this scenario, so 03_02
    never created a group for it; skipping is equivalent to treating its mass as zero,
    not an error). A summary of skipped combinations is printed once per call, not
    once per (component, material) pair.

    Raises
    ------
    ValueError if `flow_metric` isn't recognized, or a composition group's draw-array
    length doesn't match `n_draws`.
    """
    if flow_metric not in {"cumulative_inflow", "cumulative_collected"}:
        raise ValueError(
            f"combine_flow_and_composition_draws: flow_metric={flow_metric!r} not "
            f"supported -- expected 'cumulative_inflow' or 'cumulative_collected'."
        )

    # Cache one vehicle-count bootstrap per (drivetrain, segment) -- reused across every
    # (components, material) sharing that group, rather than re-bootstrapping per material.
    # `None` marks a (drivetrain, segment) confirmed to have no matching mc_summary entry
    # (skipped, not re-attempted). For direct=True, every group shares one cache entry
    # per drivetrain (the "segment" in the cache key is always "standard" there, but the
    # mc_summary lookup itself is drivetrain-level, segment-independent).
    count_draws_cache: dict[tuple[str, str], np.ndarray | None] = {}
    mass_draws_by_group: dict[tuple, np.ndarray] = {}
    skipped_groups: set[tuple[str, str]] = set()

    for comp_key, comp_draws in composition_draws_by_group.items():
        drivetrain, segment, components, material = comp_key

        if (drivetrain, segment) not in count_draws_cache:
            key = mc_summary_key(scenario_name, period, drivetrain, flow_metric, segment=None if direct else segment, direct=direct)
            try:
                count_draws_cache[(drivetrain, segment)] = bootstrap_vehicle_count_draws_from_summary(
                    mc_summary, key, n_draws, rng
                ) * VEHICLE_COUNT_UNIT_SCALE
            except KeyError:
                count_draws_cache[(drivetrain, segment)] = None
                skipped_groups.add((drivetrain, segment))
        count_draws = count_draws_cache[(drivetrain, segment)]
        if count_draws is None:
            continue  # no matching mc_summary entry -- treated as zero vehicles, zero mass

        if len(comp_draws) != n_draws:
            raise ValueError(
                f"combine_flow_and_composition_draws: composition draws for {comp_key} "
                f"have length {len(comp_draws)}, expected n_draws={n_draws}."
            )

        mass_draws_by_group[(drivetrain, segment, components, material)] = count_draws * comp_draws

    if skipped_groups:
        print(f"[combine_flow_and_composition_draws] WARNING: {len(skipped_groups)} "
              f"(drivetrain, segment) combination(s) have no matching mc_stage03_02_summary "
              f"entry for scenario={scenario_name!r}, period={period}, flow_metric="
              f"{flow_metric!r} -- treated as zero vehicles/zero mass: "
              f"{sorted(skipped_groups)}")

    return mass_draws_by_group


# -----------------------------------------------------------------------------------
# Step 5, part C: scalar (point-estimate) counterpart, built on `tracker_keyed` -- the
# ONLY vehicle-count artifact 03_02_adjustedflows.py actually persists to disk per
# scenario (confirmed via diagnose_tracker_schema.py / _2.py against the real project:
# `tracker_keyed` is a dict keyed by (Region, Drive Train) -> DataFrame with columns
# Region, Drive Train, flow, scrap_year, cohort_year, Segment, amount, year, key).
#
# Confirmed real `flow` column values: "inflow", "collected", "export",
# "unknown_whereabouts" (round 2). Only "inflow" and "collected" are in scope for
# material mass. Confirmed: `scrap_year` is the right "calendar year of this flow
# event" column for BOTH flow types -- for flow="inflow", scrap_year == cohort_year
# always (both are the year the vehicle entered); for flow="collected", scrap_year is
# the year of scrapping and matches the separate `year` column exactly in every row
# checked (cohort_year differs -- it's the vehicle's original build year).
# -----------------------------------------------------------------------------------
FLOW_VALUES_IN_SCOPE = {"inflow", "collected"}


def select_first_year_composition(composition_summary: pd.DataFrame) -> pd.DataFrame:
    """
    [CHANGED, this round] Restrict a composition summary DataFrame (from
    `load_composition_summary`) to real-segment rows (excludes "standard"), for the
    12-SEGMENT path. NAME KEPT for minimal call-site disruption, but this NO LONGER
    restricts to one "first available year" -- composition data is now available
    annually (through 2070, confirmed by the user), so every available year is kept,
    and the actual year-matching now happens per tracker row, against the vehicle's
    own build year (`cohort_year`) -- see `match_cohort_year_to_composition`.

    SUPERSEDED behavior (composition only every 5 years, one representative year per
    drivetrain applied to every cohort regardless of age) is no longer applicable now
    that annual data exists for the years that actually matter.
    """
    return composition_summary[composition_summary["segment"] != "standard"].copy()


def select_standard_composition(composition_summary: pd.DataFrame) -> pd.DataFrame:
    """
    [CHANGED, this round] Restrict a composition summary DataFrame to
    `segment == "standard"` rows -- the composition source for the drivetrain-level
    ("standard") mass path. NO LONGER restricted to one "first available year" --
    same change as `select_first_year_composition` above, for the same reason.

    Raises
    ------
    ValueError if there are NO "standard" rows at all -- a genuinely missing data
    case, not something to silently skip (see original docstring reasoning, still
    applicable).
    """
    standard_only = composition_summary[composition_summary["segment"] == "standard"].copy()
    if standard_only.empty:
        raise ValueError(
            "select_standard_composition: no 'segment'==\"standard\" rows found in "
            "composition_summary -- either the composition summary file genuinely has "
            "no standard-car rows, or load_composition_summary is filtering them out "
            "upstream (it shouldn't -- see that function's docstring)."
        )
    return standard_only


def combine_scalar_mass_from_tracker(
    tracker_keyed: dict[tuple[str, str], pd.DataFrame],
    flow: str,
    period: tuple[int, int],
    composition_summary_first_year: pd.DataFrame,
    p04,
    region: str,
) -> pd.DataFrame:
    """
    Scalar (point-estimate) counterpart to `combine_flow_and_composition_draws`, built
    directly on `tracker_keyed` (see the architecture note above -- this is what's
    actually saved to disk per scenario, e.g.
    `load_many("tracker_keyed_BAU", root=PROJECT_ROOT)["tracker_keyed_BAU"]`).

    THE MATH MODEL: mass = vehicle_count * composition_value, where:
      - vehicle_count = sum of `amount` over rows where `flow == flow` and
        `scrap_year` falls within `period` (inclusive), grouped by (Drive Train,
        Segment), summed across ALL cohort_years within that window (i.e. NOT split by
        the vehicle's original build year -- see the note below).
      - composition_value = the `p04.composition_scalar_statistic` column (default
        "mean") of `composition_summary_first_year` for the matching (drivetrain,
        segment, components, material).

    [CHANGED, this round] Composition now matches each tracker row's OWN build year
    (`cohort_year`), not one flat representative year -- see `match_cohort_year_to_
    composition`. This means `mass` is computed PER ROW first (that row's own amount
    times that row's own cohort-year-matched composition value), THEN summed over the
    period -- NOT the old order (sum vehicle_count over the period first, multiply by
    one shared composition value once). The output's `vehicle_count` and
    `composition_value` columns are KEPT (same schema, for backward compatibility with
    everything that reads this table) but `composition_value` is now an EFFECTIVE,
    mass-weighted value backed out as `mass / vehicle_count` -- i.e. computed FROM the
    correctly-aggregated mass, not the other way around. For a period spanning cohorts
    with materially different compositions (e.g. an early-BEV vs. a late-BEV battery
    chemistry), this effective value will not equal any single year's actual
    composition -- that is expected, not a bug; it is a weighted average.

    Cohort years outside the composition data's own coverage are CLAMPED to the
    nearest available composition year (confirmed with the user), with a diagnostic
    printed showing how many rows this affected.

    Parameters
    ----------
    tracker_keyed : dict keyed by (Region, Drive Train) -> DataFrame, as persisted by
        03_02_adjustedflows.py (e.g. `tracker_keyed_BAU`)
    flow : must be "inflow" or "collected" (the only `flow` column values in scope for
        material mass -- export/unknown_whereabouts vehicles are not processed for
        material recovery)
    period : (start, end) year range, inclusive on both ends -- should match one of
        `params.monte_carlo.output_periods` for direct comparability with the MC path
    composition_summary_first_year : output of `select_first_year_composition`
    p04 : params.materials (the `MaterialsParams` dataclass instance)
    region : which Region to compute (composition data is not region-specific)

    Returns
    -------
    Tidy DataFrame, one row per (drivetrain, segment) present in `tracker_keyed` for
    `region`/`flow` crossed with every (components, material) available in
    `composition_summary_first_year` for that (drivetrain, segment), columns:
      region, flow, drivetrain, segment, components, material, vehicle_count,
      composition_value, mass

    Raises
    ------
    ValueError if `flow` is not one of the in-scope values.
    """
    if flow not in FLOW_VALUES_IN_SCOPE:
        raise ValueError(
            f"combine_scalar_mass_from_tracker: flow={flow!r} not supported -- "
            f"expected one of {sorted(FLOW_VALUES_IN_SCOPE)} (export/unknown_whereabouts "
            f"vehicles are out of scope for material mass)."
        )

    stat_col = p04.composition_scalar_statistic
    start, end = period

    frames = []
    for (grp_region, drivetrain), df in tracker_keyed.items():
        if grp_region != region:
            continue
        sub = df[(df["flow"] == flow) & (df["scrap_year"] >= start) & (df["scrap_year"] <= end)]
        if sub.empty:
            continue
        sub = _floor_amount_at_zero(sub, context_label=f"{drivetrain}/{flow}/scalar")
        rows = sub[["Segment", "cohort_year", "amount"]].rename(columns={"Segment": "segment"})
        rows["drivetrain"] = drivetrain

        comp_drv = composition_summary_first_year[composition_summary_first_year["drivetrain"] == drivetrain]
        if comp_drv.empty:
            print(f"[combine_scalar_mass_from_tracker] WARNING: no composition rows for "
                  f"drivetrain={drivetrain!r} -- skipped entirely.")
            continue

        matched = match_cohort_year_to_composition(
            rows, comp_drv, row_year_col="cohort_year", exact_cols=["drivetrain", "segment"],
            context_label=f"{drivetrain}/{flow}",
        )
        if matched.empty:
            continue

        matched["row_mass"] = matched["amount"] * matched[stat_col]
        frames.append(matched[["drivetrain", "segment", "components", "material", "amount", "row_mass"]])

    empty_cols = ["region", "flow", "drivetrain", "segment", "components", "material", "vehicle_count", "composition_value", "mass"]
    if not frames:
        return pd.DataFrame(columns=empty_cols)

    all_rows = pd.concat(frames, ignore_index=True)
    # `amount` (and therefore vehicle_count/mass) is in MILLIONS of vehicles --
    # confirmed via diagnose_vehicle_count_units.py. Scale AFTER the per-row multiply
    # above (row_mass is still in "millions of vehicles x kg/vehicle" at this point),
    # so one scaling step covers both the reported vehicle_count and mass correctly.
    grouped = all_rows.groupby(["drivetrain", "segment", "components", "material"], as_index=False).agg(
        vehicle_count=("amount", "sum"), mass=("row_mass", "sum"),
    )
    grouped["vehicle_count"] *= VEHICLE_COUNT_UNIT_SCALE
    grouped["mass"] *= VEHICLE_COUNT_UNIT_SCALE
    # Effective (mass-weighted) composition value, backed out from the correctly
    # computed mass -- see docstring. Guard divide-by-zero for a (drivetrain, segment,
    # components, material) combination with zero vehicles in this period.
    grouped["composition_value"] = np.where(
        grouped["vehicle_count"] > 0, grouped["mass"] / grouped["vehicle_count"], 0.0
    )
    grouped["region"] = region
    grouped["flow"] = flow

    return grouped[empty_cols].reset_index(drop=True)


def combine_scalar_mass_standard(
    mc_summary: dict,
    scenario_name: str,
    flow: str,
    period: tuple[int, int],
    composition_standard: pd.DataFrame,
    p04,
    region: str,
    point_estimate_stat: str = "mean",
) -> pd.DataFrame:
    """
    [NEW] Scalar (point-estimate) counterpart to `combine_flow_and_composition_draws
    (direct=True)` -- the drivetrain-level ("standard" composition) mass path.

    IMPORTANT ARCHITECTURE DIFFERENCE from `combine_scalar_mass_from_tracker`: there
    is NO deterministic, single-run, drivetrain-level (no-segment) vehicle-count
    artifact anywhere in this pipeline -- 03_02_adjustedflows.py's drivetrain-level
    computation ("by-drivetrain, independent re-simulation") only ever runs as part
    of its Monte Carlo pass (`monte_carlo_enabled=True`), never as a separate
    deterministic call. So the "point estimate" used here is `mc_stage03_02_summary`'s
    own `point_estimate_stat` (default "mean") of the "__direct__" entry for this
    (scenario, period, drivetrain, metric) -- i.e. the MEAN of that Monte Carlo run,
    not a genuinely separate deterministic simulation. This is a deliberate,
    documented choice, not a hidden assumption -- if 03_02_adjustedflows.py is ever
    run with `monte_carlo_enabled=False`, this function will raise (no "__direct__"
    entries exist in `mc_summary` at all in that case).

    [CHANGED, this round -- IMPORTANT LIMITATION, distinct from the 12-segment path]
    Composition data is now annual, and the 12-segment path (`combine_scalar_mass_
    from_tracker`) matches each vehicle's OWN build year (`cohort_year`) exactly --
    but that is NOT possible here. `mc_stage03_02_summary`'s "__direct__" entries are
    PERIOD-CUMULATIVE ONLY (confirmed: no per-year, let alone per-cohort-year, vehicle
    count is persisted anywhere for the by-drivetrain independent re-simulation --
    `03_02_adjustedflows.py` only ever summarizes it per requested period). There is
    therefore no per-row build-year to match against here, unlike the segment path
    (which has full per-row `cohort_year` data in `tracker_keyed`).

    BEST ACHIEVABLE APPROXIMATION: composition is matched to the PERIOD's OWN
    MIDPOINT YEAR (clamped to the nearest available composition year, same mechanism
    as the segment path -- `match_cohort_year_to_composition`), applied to the whole
    period's vehicle-count total. This is a genuine improvement over the old "always
    the first available year regardless of what period was requested" behavior (the
    value now actually responds to which period is being asked about), but it is NOT
    equivalent to the segment path's true per-cohort weighting -- closing that gap
    would require 03_02_adjustedflows.py to persist a per-year (not just per-period)
    "__direct__" summary, which it does not currently do.

    THE MATH MODEL: mass = vehicle_count * composition_value, where:
      - vehicle_count = `mc_summary[mc_summary_key(..., direct=True)][point_estimate_stat]`
        for the matching (scenario, period, drivetrain, flow_metric) -- already in
        raw vehicle count (mc_stage03_02_summary's own values are in MILLIONS of
        vehicles, same VEHICLE_COUNT_UNIT_SCALE conversion as the segment path).
      - composition_value = the `p04.composition_scalar_statistic` column of
        `composition_standard`, at the period's midpoint year (clamped to the
        nearest available composition year), for the matching drivetrain.

    Parameters
    ----------
    mc_summary : the loaded `mc_stage03_02_summary` dict
    scenario_name : e.g. "BAU"
    flow : must be "inflow" or "collected"
    period : (start, end), must match one of `params.monte_carlo.output_periods`
    composition_standard : output of `select_standard_composition`
    p04 : params.materials (the `MaterialsParams` dataclass instance)
    region : which Region to compute (kept for column-shape parity with the segment
        path; mc_stage03_02_summary's "__direct__" entries are not region-keyed
        beyond the pipeline's own single-region convention)
    point_estimate_stat : which `mc_summary` entry statistic to use as the vehicle-
        count point estimate -- "mean" (default), "median", or "mode"

    Returns
    -------
    Tidy DataFrame, one row per drivetrain present in both `mc_summary`'s "__direct__"
    entries AND `composition_standard`, crossed with every (components, material)
    available for that drivetrain, columns:
      region, flow, drivetrain, segment ("standard", constant), components, material,
      vehicle_count, composition_value, mass

    Raises
    ------
    ValueError if `flow` is not one of the in-scope values.
    """
    if flow not in FLOW_VALUES_IN_SCOPE:
        raise ValueError(
            f"combine_scalar_mass_standard: flow={flow!r} not supported -- "
            f"expected one of {sorted(FLOW_VALUES_IN_SCOPE)}."
        )

    flow_metric = FLOW_TO_MC_METRIC[flow]
    stat_col = p04.composition_scalar_statistic
    start, end = period
    midpoint_year = round((start + end) / 2)

    empty_cols = ["region", "flow", "drivetrain", "segment", "components", "material", "vehicle_count", "composition_value", "mass"]

    drivetrains = sorted(composition_standard["drivetrain"].unique())
    counts_rows = []
    skipped = []
    for drivetrain in drivetrains:
        key = mc_summary_key(scenario_name, period, drivetrain, flow_metric, direct=True)
        entry = mc_summary.get(key)
        if entry is None:
            skipped.append(drivetrain)
            continue
        counts_rows.append({
            "drivetrain": drivetrain,
            "vehicle_count": entry[point_estimate_stat] * VEHICLE_COUNT_UNIT_SCALE,
            # Synthetic "row year" for match_cohort_year_to_composition -- the
            # period's own midpoint, NOT a real per-vehicle build year (see the
            # docstring's LIMITATION note: no per-cohort data exists for this path).
            "period_midpoint_year": midpoint_year,
        })
    if skipped:
        print(f"[combine_scalar_mass_standard] WARNING: {len(skipped)} drivetrain(s) have no "
              f"matching mc_stage03_02_summary '__direct__' entry for scenario={scenario_name!r}, "
              f"period={period}, flow={flow!r} -- treated as zero vehicles/zero mass: {sorted(skipped)}")

    if not counts_rows:
        return pd.DataFrame(columns=empty_cols)

    counts_all = pd.DataFrame(counts_rows)
    matched = match_cohort_year_to_composition(
        counts_all, composition_standard, row_year_col="period_midpoint_year", exact_cols=["drivetrain"],
        context_label=f"standard/{scenario_name}/{flow}/{period}",
    )
    if matched.empty:
        return pd.DataFrame(columns=empty_cols)

    matched["mass"] = matched["vehicle_count"] * matched[stat_col]
    matched = matched.rename(columns={stat_col: "composition_value"})
    matched["region"] = region
    matched["flow"] = flow
    matched["segment"] = "standard"

    return matched[empty_cols].reset_index(drop=True)


def combine_scalar_mass_by_year_from_tracker(
    tracker_keyed: dict[tuple[str, str], pd.DataFrame],
    flow: str,
    year_range: tuple[int, int],
    composition_summary_first_year: pd.DataFrame,
    p04,
    region: str,
) -> pd.DataFrame:
    """
    Same as `combine_scalar_mass_from_tracker`, but preserves YEAR-BY-YEAR granularity
    instead of collapsing across the whole period -- for the "material mass by year"
    diagnostic plot, where a genuine year-over-year trend is wanted rather than one
    summed total. The "year" in the output is the FLOW year (`scrap_year` for
    "collected", trivially the same as cohort_year for "inflow") -- this is the natural
    x-axis for "how much mass flowed in this calendar year," not the vehicles' build
    year.

    [CHANGED, this round] Composition now matches each contributing tracker row's OWN
    build year (`cohort_year`), same as `combine_scalar_mass_from_tracker` -- see that
    function's docstring and `match_cohort_year_to_composition`. Vehicles scrapped in
    the SAME calendar year (`scrap_year`) can have been built in DIFFERENT years, so
    composition is matched PER ROW (per cohort_year) before aggregating to (segment,
    scrap_year, components, material) -- not one flat value applied across every
    scrap_year the way it was before annual composition data existed.

    Parameters
    ----------
    tracker_keyed, flow, composition_summary_first_year, p04, region : same as
        `combine_scalar_mass_from_tracker`
    year_range : (start, end) inclusive -- filters `scrap_year`, kept as its own column
        in the output (NOT summed away)

    Returns
    -------
    Tidy DataFrame, one row per (drivetrain, segment, scrap_year) present in
    `tracker_keyed` for `region`/`flow` crossed with every (components, material)
    available for that (drivetrain, segment) at the cohort years contributing to that
    scrap_year, columns:
      region, flow, drivetrain, segment, year, components, material,
      vehicle_count, composition_value, mass

    Raises
    ------
    ValueError if `flow` is not one of the in-scope values.
    """
    if flow not in FLOW_VALUES_IN_SCOPE:
        raise ValueError(
            f"combine_scalar_mass_by_year_from_tracker: flow={flow!r} not supported -- "
            f"expected one of {sorted(FLOW_VALUES_IN_SCOPE)}."
        )

    stat_col = p04.composition_scalar_statistic
    start, end = year_range

    frames = []
    for (grp_region, drivetrain), df in tracker_keyed.items():
        if grp_region != region:
            continue
        sub = df[(df["flow"] == flow) & (df["scrap_year"] >= start) & (df["scrap_year"] <= end)]
        if sub.empty:
            continue
        sub = _floor_amount_at_zero(sub, context_label=f"{drivetrain}/{flow}/scalar-by-year")
        rows = sub[["Segment", "cohort_year", "scrap_year", "amount"]].rename(columns={"Segment": "segment"})
        rows["drivetrain"] = drivetrain

        comp_drv = composition_summary_first_year[composition_summary_first_year["drivetrain"] == drivetrain]
        if comp_drv.empty:
            print(f"[combine_scalar_mass_by_year_from_tracker] WARNING: no composition rows for "
                  f"drivetrain={drivetrain!r} -- skipped entirely.")
            continue

        matched = match_cohort_year_to_composition(
            rows, comp_drv, row_year_col="cohort_year", exact_cols=["drivetrain", "segment"],
            context_label=f"{drivetrain}/{flow}/by-year",
        )
        if matched.empty:
            continue

        matched["row_mass"] = matched["amount"] * matched[stat_col]
        frames.append(matched[["drivetrain", "segment", "scrap_year", "components", "material", "amount", "row_mass"]])

    empty_cols = [
        "region", "flow", "drivetrain", "segment", "year", "components", "material",
        "vehicle_count", "composition_value", "mass",
    ]
    if not frames:
        return pd.DataFrame(columns=empty_cols)

    all_rows = pd.concat(frames, ignore_index=True)
    # `amount` (and therefore vehicle_count/mass) is in MILLIONS of vehicles --
    # confirmed via diagnose_vehicle_count_units.py. See VEHICLE_COUNT_UNIT_SCALE note.
    grouped = all_rows.groupby(
        ["drivetrain", "segment", "scrap_year", "components", "material"], as_index=False
    ).agg(vehicle_count=("amount", "sum"), mass=("row_mass", "sum"))
    grouped["vehicle_count"] *= VEHICLE_COUNT_UNIT_SCALE
    grouped["mass"] *= VEHICLE_COUNT_UNIT_SCALE
    grouped["composition_value"] = np.where(
        grouped["vehicle_count"] > 0, grouped["mass"] / grouped["vehicle_count"], 0.0
    )
    grouped = grouped.rename(columns={"scrap_year": "year"})
    grouped["region"] = region
    grouped["flow"] = flow

    return grouped[empty_cols].reset_index(drop=True)


# -----------------------------------------------------------------------------------
# Diagnostic plot: material mass by year, per drivetrain and per segment.
# -----------------------------------------------------------------------------------


def _order_segments(segments) -> list[str]:
    """Conventional A..F, then JA..JF ordering (matches segment_map's own key order in
    params_schema.py) rather than plain alphabetical (which would wrongly put "JA"
    before "A")."""
    return sorted(segments, key=lambda s: (s.startswith("J"), s))


def _gaussian_kde_curve(
    values: np.ndarray, n_grid: int = 200, bandwidth: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual Gaussian-kernel density estimate (numpy only, no scipy dependency) --
    reflects the actual (possibly asymmetric/skewed) shape of `values` rather than
    forcing a symmetric fit. Same technique already used in `03_02_adjustedflows.py`'s
    drivetrain/segment comparison diagnostic plots, reimplemented here so
    `04_01_carcomposition.py` has no cross-file dependency on that script.

    Bandwidth: Silverman's rule of thumb if not given --
      0.9 * min(std, IQR/1.34) * n**(-1/5)

    Returns
    -------
    (grid, density) -- grid is an (n_grid,) array of x-values spanning the data range
    (plus 10% padding on each side), density is the estimated PDF value at each grid
    point (integrates to ~1 over the grid).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    std = values.std(ddof=1) if n > 1 else 1.0
    if bandwidth is None:
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        spread = min(std, iqr / 1.34) if iqr > 0 else std
        spread = spread if spread > 0 else 1.0
        bandwidth = 0.9 * spread * n ** (-1 / 5)
        if bandwidth <= 0:
            value_range = values.max() - values.min()
            bandwidth = value_range / 50 if value_range > 0 else 1.0

    grid_min, grid_max = values.min(), values.max()
    pad = (grid_max - grid_min) * 0.1
    if pad <= 0:
        pad = abs(grid_min) * 0.1 if grid_min != 0 else 1.0
    grid = np.linspace(grid_min - pad, grid_max + pad, n_grid)

    diffs = (grid[:, None] - values[None, :]) / bandwidth
    kernel_vals = np.exp(-0.5 * diffs ** 2) / np.sqrt(2 * np.pi)
    density = kernel_vals.sum(axis=1) / (n * bandwidth)
    return grid, density


def _plot_mass_by_year_stacked(
    mass_by_year_df: pd.DataFrame,
    drivetrain: str,
    stack_col: str,
    mc_draws_by_category: dict[str, np.ndarray] | None,
    title: str,
    fig_path: Path,
    top_n: int | None = 8,
    always_include: set[str] | None = None,
) -> None:
    """
    Shared implementation for both the per-drivetrain (stack_col="material") and
    per-segment (stack_col="segment") mass-by-year plots.

    THE SCALE-MISMATCH CAVEAT (why the MC uncertainty is a SEPARATE panel, not an
    overlaid band on the by-year axis): the MC path only has PERIOD-level uncertainty
    (e.g. total mass summed over 1975-2070), which is on a completely different scale
    than any single year's mass -- overlaying it directly on the by-year stacked axis
    would be misleading (comparing a ~100-year cumulative total against one year's
    value). A separate boxplot panel keeps the two honest: "here's the year-by-year
    scalar trend" next to "here's the overall period total's uncertainty range".

    THE COMPARISON REQUIREMENT (confirmed with the user -- a single aggregate density
    doesn't let you compare categories against each other): the MC panel shows ONE
    DENSITY CURVE PER CATEGORY (per material, or per segment) -- the SAME categories as
    the stacked area's legend, color-matched to the stackplot's own fill colors
    (captured from `ax1.stackplot`'s return value and reused for each curve) -- so a
    curve's color ties directly back to its band in the stacked area above it. Each
    curve is a Gaussian KDE (`_gaussian_kde_curve`) fit to that category's own MC draws
    of the period-total mass -- i.e. a probability density function derived from the
    draws, per the user's explicit request ("I want probability density functions
    derived from the histograms").

    Parameters
    ----------
    mass_by_year_df : output of `combine_scalar_mass_by_year_from_tracker`
    drivetrain : which drivetrain to plot (function filters to this)
    stack_col : "material" (grouped into top_n + "Other") or "segment" (all 12 shown,
        conventional A-F/JA-JF ordering, no grouping)
    mc_draws_by_category : dict mapping EVERY category value appearing in `stack_col`
        (BEFORE any top_n grouping -- e.g. every individual material name, not yet
        grouped into "Other") to an (n_draws,) array of that category's own
        period-TOTAL mass (summed across every other dimension -- segments/components
        for material, materials/components for segment). Categories grouped into
        "Other" for the stacked area are summed together here too, so the "Other" curve
        matches the "Other" band. Pass None to skip the MC panel entirely (e.g. the MC
        path wasn't run/available). Categories with no entry are silently omitted from
        the density panel (their scalar band still shows in the stacked area -- e.g. a
        (drivetrain, segment) combination that was confirmed zero-vehicle and skipped
        by `combine_flow_and_composition_draws`).
    title : plot title
    fig_path : where to save the figure (PNG -- `fig.savefig` infers format from the
        extension, so pass a path ending in `.png`)
    top_n : only used when stack_col="material" -- how many materials to show
        individually before grouping the rest as "Other"; ignored for "segment"
    always_include : [NEW] only used when stack_col="material" -- material names in
        this set are ALWAYS shown individually, regardless of their rank by total
        mass, even if that means showing more than `top_n` categories total (this
        widens the shown set, it never displaces an always-included material to fit
        the original `top_n` count). Motivating case: catalytic-converter and
        electronics materials are individually much smaller than bulk structural
        materials (steel, aluminum) by mass, so a pure top-N-by-mass ranking swept
        them all into "Other" -- correct data, but not what's actually wanted when
        those specific materials are the ones being tracked. None (default) = old
        behavior, pure top-N by mass with no exceptions.
    """
    df = mass_by_year_df[mass_by_year_df["drivetrain"] == drivetrain]
    if df.empty:
        print(f"[_plot_mass_by_year_stacked] no data for drivetrain={drivetrain!r} -- skipping {fig_path.name}.")
        return

    pivot = df.groupby(["year", stack_col])["mass"].sum().unstack(stack_col, fill_value=0.0).sort_index()

    if stack_col == "material" and top_n is not None and pivot.shape[1] > top_n:
        totals = pivot.sum(axis=0).sort_values(ascending=False)
        always_present = [c for c in (always_include or set()) if c in totals.index]
        # Fill the remaining slots (top_n minus however many always-included
        # materials there are) with the highest-ranking materials NOT already
        # forced in -- always_include only ever WIDENS the shown set, never
        # displaces a forced-in material to keep the count at exactly top_n.
        remaining_slots = max(top_n - len(always_present), 0)
        ranked_excluding_forced = [c for c in totals.index if c not in always_present]
        top_cols = always_present + ranked_excluding_forced[:remaining_slots]
        other_cols = [c for c in totals.index if c not in top_cols]
        plot_df = pivot[top_cols].copy()
        if other_cols:
            plot_df["Other"] = pivot[other_cols].sum(axis=1)
        category_members = {c: [c] for c in top_cols}
        if other_cols:
            category_members["Other"] = other_cols
    elif stack_col == "segment":
        ordered = _order_segments(pivot.columns)
        plot_df = pivot[ordered]
        category_members = {c: [c] for c in ordered}
    else:
        plot_df = pivot
        category_members = {c: [c] for c in pivot.columns}

    # Build the per-category MC draws, respecting the SAME top_n/"Other" grouping as
    # the stacked area, so both panels show identical categories.
    density_categories: list[str] = []
    density_draws: list[np.ndarray] = []
    if mc_draws_by_category is not None:
        for category, members in category_members.items():
            member_draws = [mc_draws_by_category[m] for m in members if m in mc_draws_by_category]
            if not member_draws:
                continue  # no MC data for any member of this category -- omit the curve
            density_categories.append(category)
            density_draws.append(np.sum(member_draws, axis=0))

    show_mc_panel = bool(density_draws)

    # [CHANGED, this round -- confirmed with the user] Was ONE overlaid density panel,
    # every category's KDE curve sharing the same y-axis (probability density). A
    # material with a NARROW, low-variance distribution (a small, tightly-clustered
    # total mass) needs a very TALL spike to integrate to 1 over that small width --
    # that tall spike then flattens every wider-spread material's curve to near-
    # invisible on the same shared y-axis, regardless of that material's own actual
    # mass. This is NOT about which material has the largest total mass (a log-x-axis
    # fix would have targeted the wrong thing) -- it is about density HEIGHT varying
    # by orders of magnitude between narrow and wide distributions. Fixed with a
    # SMALL-MULTIPLES grid instead: one subplot per category, each with its OWN
    # independently-scaled x and y axes -- no material's curve can ever drown out
    # another's, regardless of scale or narrowness. Each subplot's own title labels
    # it (color-matched to its stackplot band), so this panel needs no separate
    # legend at all -- resolves the "don't show the legend twice" request too.
    if show_mc_panel:
        n_cat = len(density_categories)
        n_cols = min(4, n_cat)
        n_rows = -(-n_cat // n_cols)  # ceil division
        fig_width = 11 + 2.6 * n_cols
        fig_height = max(6, 1.9 * n_rows)
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs_outer = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.4 * n_cols / 2])
        ax1 = fig.add_subplot(gs_outer[0])
    else:
        fig, ax1 = plt.subplots(figsize=(11, 6))

    # Display in TONNES, not kg -- everything upstream of this function (mass_by_year_df,
    # mc_draws_by_category) is still in kg; only the plotted values are converted, right
    # here, right before rendering.
    plot_df_t = plot_df / KG_PER_TONNE
    stack_polys = ax1.stackplot(plot_df_t.index, plot_df_t.T.values, labels=plot_df_t.columns, alpha=0.85)
    ax1.set_title(title, fontsize=12)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mass [t]")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.15) if show_mc_panel else (1.02, 1),
        ncol=4 if show_mc_panel else 1,
        frameon=False, fontsize=8,
    )
    stack_colors = {label: poly.get_facecolor()[0] for label, poly in zip(plot_df_t.columns, stack_polys)}

    if show_mc_panel:
        gs_right = gs_outer[1].subgridspec(n_rows, n_cols, hspace=0.75, wspace=0.45)
        for i, (category, draws) in enumerate(zip(density_categories, density_draws)):
            ax_small = fig.add_subplot(gs_right[i // n_cols, i % n_cols])
            grid, density = _gaussian_kde_curve(draws / KG_PER_TONNE)
            color = stack_colors.get(category, "gray")
            ax_small.plot(grid, density, color=color, linewidth=1.3)
            ax_small.fill_between(grid, density, color=color, alpha=0.25)
            ax_small.set_title(category, fontsize=8, color=color, fontweight="bold")
            ax_small.tick_params(labelsize=6)
            ax_small.set_yticks([])  # density's absolute scale isn't comparable across
            # panels anyway (that was the whole problem with sharing one) -- omitting
            # the y-ticks avoids implying otherwise; the shape/spread is what matters.
            ax_small.spines["top"].set_visible(False)
            ax_small.spines["right"].set_visible(False)
            ax_small.spines["left"].set_visible(False)
            # Small tonnes-formatted x-axis so each panel's own scale is still legible.
            ax_small.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        fig.text(0.99, 0.5, "Period total (MC) -- one independently-scaled panel per "
                 "category, x-axis in tonnes", rotation=90, va="center", ha="right",
                 fontsize=8, color="#666666")

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


# [NEW] Materials that are individually small by mass compared to bulk structural
# materials (steel, aluminum), but specifically worth tracking on their own rather
# than being swept into "Other" by a pure top-N-by-mass ranking -- catalytic
# converter precious/rare-earth metals and electronics components.
MATERIALS_ALWAYS_SHOWN_INDIVIDUALLY = {
    "calculatePt", "calculatePd", "calculateRh", "calculateCe", "calculateLa",
    "powerElectronics", "actuators", "controllers", "light", "cableLike",
}


def plot_material_mass_by_year_per_drivetrain(
    mass_by_year_df: pd.DataFrame,
    drivetrain: str,
    mc_draws_by_material: dict[str, np.ndarray] | None,
    title_prefix: str,
    fig_path: Path,
    top_n: int | None = None,
    always_include: set[str] | None = MATERIALS_ALWAYS_SHOWN_INDIVIDUALLY,
) -> None:
    """
    For ONE drivetrain: stacked-area material mass by year (aggregated across all
    segments and components sharing the same material name), plus a period-total MC
    boxplot panel comparing materials against each other. See
    `_plot_mass_by_year_stacked`'s docstring for the panel-layout reasoning.

    mc_draws_by_material : dict mapping EVERY individual material name (not yet
        grouped into "Other") -> (n_draws,) period-total mass array for that material,
        summed across all segments/components for this drivetrain.
    top_n : [CHANGED, this round] default is now None -- EVERY material is shown
        individually, no "Other" grouping at all (confirmed with the user: an
        earlier "always show catalytic/electronics individually, but still group
        the rest into Other" compromise was NOT what was wanted -- "I want to see
        all the materials"). Pass an int to re-enable top-N grouping if the legend
        ever gets too large to be useful for a specific case.
    always_include : only has any effect when `top_n` is an int (grouping enabled) --
        see `_plot_mass_by_year_stacked`'s docstring. Irrelevant at the current
        top_n=None default, kept here so grouping can be re-enabled later without
        losing the "these specific materials are never grouped away" behavior.
    """
    _plot_mass_by_year_stacked(
        mass_by_year_df, drivetrain, "material", mc_draws_by_material,
        title=f"{title_prefix}: material mass by year -- {drivetrain}",
        fig_path=fig_path, top_n=top_n, always_include=always_include,
    )


def plot_material_mass_by_year_per_segment(
    mass_by_year_df: pd.DataFrame,
    drivetrain: str,
    mc_draws_by_segment: dict[str, np.ndarray] | None,
    title_prefix: str,
    fig_path: Path,
) -> None:
    """
    For ONE drivetrain: stacked-area mass by year, stacked by SEGMENT (materials
    aggregated away) -- shows which segments (A-F, JA-JF) drive mass within that
    drivetrain over time, plus a period-total MC boxplot panel comparing segments
    against each other. Same two-panel layout as
    `plot_material_mass_by_year_per_drivetrain`.

    mc_draws_by_segment : dict mapping EVERY segment ("A".."F", "JA".."JF") -> (n_draws,)
        period-total mass array for that segment, summed across all
        materials/components for this drivetrain.
    """
def _scenario_color_map(scenario_names: list[str]) -> dict[str, Any]:
    """[NEW] Consistent color per scenario across every cross-scenario comparison
    plot in this module (trajectory lines, boxplots, PDFs) -- the same scenario
    always gets the same color, using matplotlib's default 'tab10' cycle."""
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(scenario_names)}


def plot_scenario_comparison_boxplot_and_pdf(
    by_scenario_draws: dict[str, np.ndarray],
    title: str,
    xlabel: str,
    fig_path_boxplot: Path,
    fig_path_pdf: Path,
    colors: dict[str, Any] | None = None,
) -> None:
    """
    [NEW] Cross-scenario comparison of one (drivetrain, flow, headline period) total
    mass distribution -- one box (boxplot figure) / one KDE curve (PDF figure) per
    scenario, e.g. BAU vs stock_lower. Same two-file convention (boxplot PNG + PDF/
    density PNG) as 03_02_adjustedflows.py's cross-scenario comparison figures,
    reimplemented locally here (uses this module's own `_gaussian_kde_curve`, no
    cross-file dependency -- same convention already used for that function).

    Parameters
    ----------
    by_scenario_draws : scenario_name -> (n_draws,) mass array [kg]. n_draws may
        differ between scenarios (independent bootstraps) -- each scenario's own
        array length is used as-is.
    title : shared title for both figures
    xlabel : e.g. "Total mass [t]"
    fig_path_boxplot, fig_path_pdf : output paths
    colors : optional scenario -> color map (see `_scenario_color_map`); built
        fresh from `by_scenario_draws`'s keys if not given
    """
    scenario_names = list(by_scenario_draws.keys())
    if not scenario_names:
        print(f"[plot_scenario_comparison_boxplot_and_pdf] no scenarios to plot -- skipping {fig_path_boxplot.name}.")
        return
    colors = colors or _scenario_color_map(scenario_names)
    draws_t = {name: np.asarray(draws, dtype=float) / KG_PER_TONNE for name, draws in by_scenario_draws.items()}

    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(scenario_names)), 6))
    bp = ax.boxplot(
        [draws_t[name] for name in scenario_names],
        labels=scenario_names, showfliers=False, patch_artist=True, widths=0.5,
    )
    for patch, name in zip(bp["boxes"], scenario_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax.set_ylabel(xlabel)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_path_boxplot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path_boxplot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in scenario_names:
        grid, density = _gaussian_kde_curve(draws_t[name])
        ax.plot(grid, density, color=colors[name], linewidth=1.6, label=name)
        ax.fill_between(grid, density, color=colors[name], alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig_path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_total_mass_by_year_scenario_comparison(
    mass_by_year_by_scenario: dict[str, pd.DataFrame],
    headline_draws_by_scenario: dict[str, np.ndarray],
    drivetrain: str,
    title: str,
    fig_path: Path,
    colors: dict[str, Any] | None = None,
) -> None:
    """
    [NEW] Cross-scenario trajectory comparison: total mass by year (summed across
    every segment/material) for ONE drivetrain, one line per scenario (e.g. BAU vs
    stock_lower), plus a small KDE panel showing each scenario's headline-period
    total mass distribution -- same "separate panel, not overlaid band" reasoning
    as `_plot_mass_by_year_stacked` (year-by-year values and the period-cumulative
    MC total are on different scales).

    Parameters
    ----------
    mass_by_year_by_scenario : scenario_name -> that scenario's `mass_by_year_
        tables[flow]` DataFrame (columns include at least year, drivetrain, mass)
    headline_draws_by_scenario : scenario_name -> (n_draws,) headline-period total
        mass array [kg] for this drivetrain (summed across segment/component/
        material); scenarios with no data for this drivetrain may be omitted
    drivetrain, title, fig_path : as elsewhere
    colors : optional scenario -> color map; built fresh if not given
    """
    scenario_names = list(mass_by_year_by_scenario.keys())
    colors = colors or _scenario_color_map(scenario_names)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax1 = fig.add_subplot(gs[0])
    any_data = False
    for name in scenario_names:
        df = mass_by_year_by_scenario[name]
        df = df[df["drivetrain"] == drivetrain]
        if df.empty:
            continue
        by_year = df.groupby("year")["mass"].sum().sort_index() / KG_PER_TONNE
        ax1.plot(by_year.index, by_year.values, color=colors[name], linewidth=1.8, label=name)
        any_data = True
    if not any_data:
        print(f"[plot_total_mass_by_year_scenario_comparison] no data for drivetrain={drivetrain!r} -- skipping {fig_path.name}.")
        plt.close(fig)
        return
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mass [t]")
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[1])
    for name in scenario_names:
        draws = headline_draws_by_scenario.get(name)
        if draws is None or len(draws) == 0:
            continue
        grid, density = _gaussian_kde_curve(np.asarray(draws, dtype=float) / KG_PER_TONNE)
        ax2.plot(grid, density, color=colors[name], linewidth=1.4)
        ax2.fill_between(grid, density, color=colors[name], alpha=0.2)
    ax2.set_title("Headline-period total (MC)", fontsize=9)
    ax2.set_xlabel("Mass [t]")
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_material_breakdown_scenario_comparison(
    mc_draws_by_scenario: dict[str, dict[tuple, np.ndarray]],
    drivetrain: str,
    title: str,
    fig_path: Path,
    top_n: int = 10,
    colors: dict[str, Any] | None = None,
) -> None:
    """
    [NEW] Per-material mass breakdown, grouped bars: for ONE drivetrain, the top_n
    materials by mass (ranked by the largest scenario's median), one bar group per
    material, one bar per scenario within each group -- median as bar height, P2.5-
    P97.5 MC range as the error bar. Same visual convention as
    `plot_standard_vs_segments_comparison`, generalized from 2 fixed categories
    (segment-sum vs standard) to N scenarios and to materials instead of
    drivetrains on the x-axis.

    Parameters
    ----------
    mc_draws_by_scenario : scenario_name -> one (period, flow) entry of
        `mc_draws_tables`, i.e. dict keyed by (drivetrain, segment, components,
        material) -> (n_draws,) mass array [kg]
    drivetrain : which drivetrain to plot (function filters + aggregates to this)
    title, fig_path : as elsewhere
    top_n : how many materials to show, ranked by the largest scenario's median
        total mass for this drivetrain
    colors : optional scenario -> color map; built fresh if not given
    """
    scenario_names = list(mc_draws_by_scenario.keys())
    colors = colors or _scenario_color_map(scenario_names)

    by_scenario_material: dict[str, dict[str, np.ndarray]] = {name: {} for name in scenario_names}
    for name in scenario_names:
        for (drv, _seg, _comp, mat), draws in mc_draws_by_scenario[name].items():
            if drv != drivetrain:
                continue
            acc = by_scenario_material[name].get(mat)
            by_scenario_material[name][mat] = draws if acc is None else acc + draws

    all_materials: set[str] = set()
    for d in by_scenario_material.values():
        all_materials.update(d.keys())
    if not all_materials:
        print(f"[plot_material_breakdown_scenario_comparison] no data for drivetrain={drivetrain!r} -- skipping {fig_path.name}.")
        return

    def _material_rank_value(mat: str) -> float:
        medians = []
        for name in scenario_names:
            draws = by_scenario_material[name].get(mat)
            if draws is not None and len(draws) > 0:
                medians.append(np.median(draws))
        return max(medians) if medians else 0.0

    ranked_materials = sorted(all_materials, key=_material_rank_value, reverse=True)[:top_n]

    n_mat = len(ranked_materials)
    n_scen = len(scenario_names)
    x = np.arange(n_mat)
    width = 0.8 / max(n_scen, 1)

    fig, ax = plt.subplots(figsize=(max(9, 1.4 * n_mat), 6))
    for i, name in enumerate(scenario_names):
        medians, lo, hi = [], [], []
        for mat in ranked_materials:
            draws = by_scenario_material[name].get(mat)
            if draws is not None and len(draws) > 0:
                d_t = draws / KG_PER_TONNE
                p_lo, p_mid, p_hi = np.percentile(d_t, [2.5, 50, 97.5])
            else:
                p_lo = p_mid = p_hi = 0.0
            medians.append(p_mid); lo.append(p_lo); hi.append(p_hi)
        medians = np.array(medians); lo = np.array(lo); hi = np.array(hi)
        offset = (i - (n_scen - 1) / 2) * width
        ax.bar(
            x + offset, medians, width, label=name, color=colors[name],
            yerr=[np.clip(medians - lo, 0, None), np.clip(hi - medians, 0, None)],
            capsize=3, error_kw={"linewidth": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ranked_materials, rotation=45, ha="right")
    ax.set_ylabel("Total mass [t]")
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_delta_by_drivetrain_and_material(
    mc_draws_by_scenario: dict[str, dict[tuple, np.ndarray]],
    baseline_scenario: str,
    comparison_scenario: str,
    drivetrains: list[str],
    title: str,
    fig_path: Path,
    top_n_materials: int = 8,
) -> None:
    """
    [NEW] Single diverging bar chart: for each drivetrain, percent change in total
    mass from `baseline_scenario` to `comparison_scenario`
    ((comparison - baseline) / baseline), computed on Monte Carlo MEDIANS -- one bar
    per drivetrain (aggregated across every segment/material), plus, for the
    `top_n_materials` biggest materials overall (ranked by baseline total mass), one
    additional bar per (drivetrain, material) pair showing where within that
    drivetrain the change concentrates. Negative (reduction) and positive (increase)
    bars are colored differently -- makes "where does `comparison_scenario` actually
    save or add material relative to `baseline_scenario`" visible at a glance,
    rather than left implied by two separate bars/boxes elsewhere.

    Parameters
    ----------
    mc_draws_by_scenario : scenario_name -> one (period, flow) entry of
        `mc_draws_tables`, keyed by (drivetrain, segment, components, material) ->
        (n_draws,) mass array [kg]
    baseline_scenario, comparison_scenario : which two scenarios to compare (percent
        change is baseline -> comparison); must both be keys of
        `mc_draws_by_scenario`
    drivetrains : which drivetrains to include, in this order
    title, fig_path : as elsewhere
    top_n_materials : how many materials (ranked by baseline total mass, summed
        across all drivetrains) to break out individually; 0 disables the
        material-level bars, showing only the per-drivetrain totals
    """
    if baseline_scenario not in mc_draws_by_scenario or comparison_scenario not in mc_draws_by_scenario:
        print(f"[plot_scenario_delta_by_drivetrain_and_material] baseline "
              f"{baseline_scenario!r} or comparison {comparison_scenario!r} not in "
              f"mc_draws_by_scenario ({list(mc_draws_by_scenario)}) -- skipping {fig_path.name}.")
        return

    baseline_draws = mc_draws_by_scenario[baseline_scenario]
    comparison_draws = mc_draws_by_scenario[comparison_scenario]

    def _median_by(draws_dict: dict[tuple, np.ndarray], drivetrain: str, material: str | None = None) -> float:
        total = None
        for (drv, _seg, _comp, mat), draws in draws_dict.items():
            if drv != drivetrain:
                continue
            if material is not None and mat != material:
                continue
            total = draws.copy() if total is None else total + draws
        return float(np.median(total)) if total is not None else 0.0

    labels: list[str] = []
    pct_changes: list[float] = []

    for drv in drivetrains:
        base_val = _median_by(baseline_draws, drv)
        comp_val = _median_by(comparison_draws, drv)
        if base_val > 0:
            labels.append(f"{drv} (total)")
            pct_changes.append(100.0 * (comp_val - base_val) / base_val)

    if top_n_materials > 0:
        material_totals: dict[tuple[str, str], float] = {}
        for (drv, _seg, _comp, mat), draws in baseline_draws.items():
            if drv not in drivetrains:
                continue
            key = (drv, mat)
            material_totals[key] = material_totals.get(key, 0.0) + float(np.median(draws))
        ranked = sorted(material_totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n_materials]
        for (drv, mat), _base_total in ranked:
            base_val = _median_by(baseline_draws, drv, mat)
            comp_val = _median_by(comparison_draws, drv, mat)
            if base_val > 0:
                labels.append(f"{drv} / {mat}")
                pct_changes.append(100.0 * (comp_val - base_val) / base_val)

    if not labels:
        print(f"[plot_scenario_delta_by_drivetrain_and_material] nothing to plot -- skipping {fig_path.name}.")
        return

    order = np.argsort(pct_changes)
    labels = [labels[i] for i in order]
    pct_changes = [pct_changes[i] for i in order]
    colors_bar = ["#C0392B" if v >= 0 else "#2980B9" for v in pct_changes]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, pct_changes, color=colors_bar)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"% change, {baseline_scenario} -> {comparison_scenario} (MC median)")
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_standard_vs_segments_comparison(
    mc_draws_segments: dict[tuple, np.ndarray],
    mc_draws_standard: dict[tuple, np.ndarray],
    drivetrains: list[str],
    title: str,
    fig_path: Path,
) -> None:
    """
    "Reasonable comparison" figure: for each drivetrain, total mass (summed across
    every material, and -- for the segment path -- every segment too) from the
    12-SEGMENT path against the drivetrain-level ("standard" composition) path, at
    ONE period -- grouped bar chart, Monte Carlo median as the bar height, 95%
    (P2.5-P97.5) range as the error bar.

    [CHANGED, this round -- confirmed with the user] This is now a PURELY Monte
    Carlo comparison. An earlier version showed the deterministic/scalar estimate
    too (first as the bar height, which was a real inconsistency since the error
    bar came from a different, MC-based distribution entirely; then as a separate
    marker once that inconsistency was fixed) -- confirmed with the user that the
    deterministic estimate should not be part of this figure at all, in any form.
    `scalar_segments`/`scalar_standard` are no longer parameters here.

    A mismatch beyond Monte Carlo noise between the two bars for a given drivetrain
    means the 12-segment composition profile (weighted by that scenario's actual
    segment mix) gives a genuinely different total than applying one "standard" car
    composition to that drivetrain's whole fleet -- exactly the kind of check this
    figure exists to make visible, not something to explain away.

    Parameters
    ----------
    mc_draws_segments, mc_draws_standard : one (period, flow) entry each from
        `mc_draws_tables` / `mc_draws_tables_standard` -- dict keyed by
        (drivetrain, segment, components, material) -> (n_draws,) mass array [kg].
    drivetrains : which drivetrains to plot, in this order (x-axis order)
    title, fig_path : as elsewhere
    """
    n_draws_local = next(iter(mc_draws_segments.values())).shape[0] if mc_draws_segments else (
        next(iter(mc_draws_standard.values())).shape[0] if mc_draws_standard else 0
    )

    def _mc_total(draws_dict: dict[tuple, np.ndarray], drivetrain: str) -> np.ndarray:
        total = np.zeros(n_draws_local, dtype=float)
        for (drv, _seg, _comp, _mat), draws in draws_dict.items():
            if drv == drivetrain:
                total = total + draws
        return total

    segment_median, segment_p2_5, segment_p97_5 = [], [], []
    standard_median, standard_p2_5, standard_p97_5 = [], [], []
    for drv in drivetrains:
        seg_draws = _mc_total(mc_draws_segments, drv) / KG_PER_TONNE
        std_draws = _mc_total(mc_draws_standard, drv) / KG_PER_TONNE
        seg_lo, seg_mid, seg_hi = (np.percentile(seg_draws, [2.5, 50, 97.5]) if seg_draws.any() else (0.0, 0.0, 0.0))
        std_lo, std_mid, std_hi = (np.percentile(std_draws, [2.5, 50, 97.5]) if std_draws.any() else (0.0, 0.0, 0.0))
        segment_p2_5.append(seg_lo); segment_median.append(seg_mid); segment_p97_5.append(seg_hi)
        standard_p2_5.append(std_lo); standard_median.append(std_mid); standard_p97_5.append(std_hi)

    segment_median = np.array(segment_median)
    segment_p2_5 = np.array(segment_p2_5); segment_p97_5 = np.array(segment_p97_5)
    standard_median = np.array(standard_median)
    standard_p2_5 = np.array(standard_p2_5); standard_p97_5 = np.array(standard_p97_5)

    x = np.arange(len(drivetrains))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(drivetrains)), 6))

    ax.bar(
        x - width / 2, segment_median, width, label="12-segment (sum)", color="#2E86AB",
        yerr=[np.clip(segment_median - segment_p2_5, 0, None), np.clip(segment_p97_5 - segment_median, 0, None)],
        capsize=4, error_kw={"linewidth": 1.2},
    )
    ax.bar(
        x + width / 2, standard_median, width, label="standard (drivetrain-level)", color="#E67E22",
        yerr=[np.clip(standard_median - standard_p2_5, 0, None), np.clip(standard_p97_5 - standard_median, 0, None)],
        capsize=4, error_kw={"linewidth": 1.2},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(drivetrains)
    ax.set_ylabel("Total mass [t]")
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------------
# Step 6: main() -- ties everything together, scoped to `scenarios_to_run`, both
# scalar and MC paths, both flows (inflow + collected).
# -----------------------------------------------------------------------------------
def main() -> dict[str, Any]:
    """
    For every scenario in `params.adjusted_flows.active_scenario_names()` (i.e.
    `scenarios_to_run` -- typically BAU + at most one comparison scenario, the same
    performance constraint already established for stage 03_02) and both in-scope
    flows ("inflow", "collected"):
      - SCALAR path: point-estimate mass per (drivetrain, segment, components,
        material), one table per (scenario, period, flow)
        (`combine_scalar_mass_from_tracker`), plus a year-by-year version for the
        diagnostic plots (`combine_scalar_mass_by_year_from_tracker`).
      - MC path: mass draws per (drivetrain, segment, components, material), one dict
        per (scenario, period, flow) (`combine_flow_and_composition_draws`), bootstrapped
        from stage 03_02's saved `mc_stage03_02_summary` histograms combined with
        composition draws bootstrapped from the histogram file (`bootstrap_composition_
        draws`) -- composition draws are bootstrapped ONCE (composition doesn't depend
        on scenario) and shared across every scenario; vehicle-count draws are
        bootstrapped fresh per (scenario, period, flow) with an independent seed stream.

    Also generates, per scenario and per flow: a material-mass-by-year plot AND a
    mass-by-year-by-segment plot for EVERY drivetrain in `p04.drivetrains`, each with a
    period-total MC boxplot panel for context (see `_plot_mass_by_year_stacked`'s
    docstring for why that's a separate panel, not an overlaid band).

    Persistence: scalar tables and MC draws dicts are saved per-scenario via
    `save_unregistered_scenario_outputs` (same helper the OLD 04_01_materials.py used
    for its per-scenario outputs -- no `ARTIFACT_FILES` registry entry needed). Plots
    are saved as PNGs under `data/processed/figures/`.
    """
    t_start_main = time.time()
    print("=" * 76)
    print("04_01_carcomposition.py -- main()")
    print("=" * 76)

    loaded = load_many("params", root=PROJECT_ROOT)
    params = loaded["params"]
    p04 = params.materials
    region = p04.region
    periods = params.monte_carlo.output_periods
    headline_period = max(periods, key=lambda p: p[1] - p[0])
    active_scenario_names = params.adjusted_flows.active_scenario_names()
    n_draws = p04.materials_mc_n_draws

    if not active_scenario_names:
        raise RuntimeError("params.adjusted_flows.active_scenario_names() is empty -- nothing to process.")

    print(f"Active scenarios: {active_scenario_names}")
    print(f"Periods: {periods} (headline for MC-panel plots: {headline_period})")
    print(f"Region: {region!r}, materials MC n_draws: {n_draws:,}")

    master_seed_seq = np.random.SeedSequence(p04.materials_mc_seed)
    composition_seed, *scenario_seed_list = master_seed_seq.spawn(1 + len(active_scenario_names))
    scenario_seeds = dict(zip(active_scenario_names, scenario_seed_list))

    # -------------------------------------------------------------------------------
    # Shared, scenario-independent inputs: composition summary + composition MC draws.
    # Composition data does NOT depend on scenario, so this is computed ONCE and reused
    # for every scenario below.
    # -------------------------------------------------------------------------------
    print("\nLoading composition summary...")
    composition_summary = load_composition_summary(p04, project_root=PROJECT_ROOT)
    # [CHANGED, this round] No longer restricted to "first available year" -- composition
    # data is now annual (through 2070, per the user), so ALL years are kept here; actual
    # year-matching happens per tracker row against its own cohort_year, INSIDE the
    # scenario loop below (composition bootstrapping is now scenario-dependent -- see
    # the note there for why).
    composition_first_year = select_first_year_composition(composition_summary)
    composition_standard = select_standard_composition(composition_summary)

    composition_years_by_drivetrain = {
        drv: np.sort(composition_first_year.loc[composition_first_year["drivetrain"] == drv, "year"].unique())
        for drv in p04.drivetrains
    }
    composition_years_by_drivetrain_standard = {
        drv: np.sort(composition_standard.loc[composition_standard["drivetrain"] == drv, "year"].unique())
        for drv in p04.drivetrains
    }

    print("\nLoading composition histograms (bin data only -- cheap to hold in memory; "
          "the actual bootstrap happens per-scenario, per-period, directly into the "
          "final mixed array -- see the scenario loop below)...")
    histogram_segments_by_drivetrain: dict[str, pd.DataFrame] = {}
    histogram_standard_by_drivetrain: dict[str, pd.DataFrame] = {}
    for drivetrain in p04.drivetrains:
        histogram_df = load_histogram_data(p04, drivetrain, project_root=PROJECT_ROOT)
        histogram_segments_by_drivetrain[drivetrain] = histogram_df[histogram_df["segment"] != "standard"]
        histogram_standard_by_drivetrain[drivetrain] = histogram_df[histogram_df["segment"] == "standard"]
    # Concatenated once, reused (read-only) across every scenario/flow/period below --
    # bootstrap_mixed_composition_draws filters this itself per call.
    histogram_segments_all = pd.concat(histogram_segments_by_drivetrain.values(), ignore_index=True)
    histogram_standard_all = pd.concat(histogram_standard_by_drivetrain.values(), ignore_index=True)

    # [FIXED, this round -- see bootstrap_mixed_composition_draws's docstring for the
    # full story] The OLD design bootstrapped and CACHED a full n_draws array per
    # (drivetrain, segment, YEAR) combination across the whole run -- with ~70 years
    # of annual composition data, that was measured to reach 147 GB and get the
    # process killed by the OS before finishing even two drivetrains. Composition
    # bootstrapping now happens FRESH per (scenario, flow, period), directly into the
    # final cohort-year-mixed array, with no intermediate per-year caching layer --
    # peak memory per group is back to one (n_draws,) array, matching the pre-annual-
    # composition-data design.
    composition_rng = np.random.default_rng(composition_seed)

    mc_summary = load_many("mc_stage03_02_summary", root=PROJECT_ROOT)["mc_stage03_02_summary"]

    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = PROJECT_ROOT / "data" / "processed" / "intermediate"

    all_outputs: dict[str, Any] = {}
    saved_paths: dict[str, Any] = {}

    for scenario_name in active_scenario_names:
        print(f"\n{'-' * 76}\nScenario: {scenario_name}\n{'-' * 76}")
        t_start_scenario = time.time()

        tracker_artifact_name = f"tracker_keyed_{scenario_name}"
        tracker_keyed = load_many(tracker_artifact_name, root=PROJECT_ROOT)[tracker_artifact_name]

        scenario_rng = np.random.default_rng(scenario_seeds[scenario_name])

        scalar_tables: dict[tuple, pd.DataFrame] = {}
        mc_draws_tables: dict[tuple, dict[tuple, np.ndarray]] = {}
        # [NEW] Drivetrain-level ("standard") counterparts, same (period, flow) keying.
        scalar_tables_standard: dict[tuple, pd.DataFrame] = {}
        mc_draws_tables_standard: dict[tuple, dict[tuple, np.ndarray]] = {}
        year_span = (
            min(df["scrap_year"].min() for df in tracker_keyed.values()),
            max(df["scrap_year"].max() for df in tracker_keyed.values()),
        )
        mass_by_year_tables: dict[str, pd.DataFrame] = {}

        for flow in sorted(FLOW_VALUES_IN_SCOPE):
            flow_metric = f"cumulative_{flow}"

            print(f"  [{flow}] year-by-year scalar mass ({year_span[0]}-{year_span[1]})...")
            mass_by_year_tables[flow] = combine_scalar_mass_by_year_from_tracker(
                tracker_keyed, flow, year_span, composition_first_year, p04, region=region
            )

            for period in periods:
                print(f"  [{flow}] period {period}: scalar mass (12-segment)...")
                scalar_tables[(period, flow)] = combine_scalar_mass_from_tracker(
                    tracker_keyed, flow, period, composition_first_year, p04, region=region
                )

                # [NEW, this round] Cohort-year weights for THIS (scenario, flow, period)
                # -- from the deterministic tracker, each matched composition year's real
                # share of vehicles. Feeds directly into the fused bootstrap-and-mix
                # below -- no separate "which years need bootstrapping" bookkeeping any
                # more (that was the source of the OOM crash; see bootstrap_mixed_
                # composition_draws's docstring for the full story).
                cohort_year_weights = compute_cohort_year_weights(
                    tracker_keyed, flow, period, region, composition_years_by_drivetrain
                )

                print(f"  [{flow}] period {period}: bootstrapping composition draws "
                      f"(cohort-year-weighted mixture, {n_draws:,} draws)...")
                mixed_composition_draws = bootstrap_mixed_composition_draws(
                    histogram_segments_all, cohort_year_weights, n_draws, composition_rng
                )

                print(f"  [{flow}] period {period}: MC mass, 12-segment ({n_draws:,} draws)...")
                mc_draws_tables[(period, flow)] = combine_flow_and_composition_draws(
                    mc_summary, scenario_name, period, flow_metric, mixed_composition_draws,
                    n_draws, scenario_rng,
                )

                # [NEW] Drivetrain-level ("standard") scalar + MC mass, same period/flow.
                print(f"  [{flow}] period {period}: scalar mass (standard)...")
                scalar_tables_standard[(period, flow)] = combine_scalar_mass_standard(
                    mc_summary, scenario_name, flow, period, composition_standard, p04, region=region,
                    point_estimate_stat="mean",
                    # NOTE: point_estimate_stat is the VEHICLE-COUNT statistic (mean of the
                    # by-drivetrain MC run) -- independent of p04.composition_scalar_statistic,
                    # which is the COMPOSITION statistic. Left explicit ("mean") rather than
                    # reusing p04.composition_scalar_statistic here to avoid conflating two
                    # different "which statistic" choices that happen to share a similarly-
                    # named parameter; see combine_scalar_mass_standard's own docstring.
                )

                # [NEW, this round] Standard path: no per-cohort weighting is possible
                # (see combine_scalar_mass_standard's LIMITATION note -- mc_stage03_02_
                # summary's "__direct__" entries are period-cumulative only). Same
                # midpoint-year approximation as the scalar standard path: a trivial
                # single-year "mixture" (weight 1.0) into the SAME fused function --
                # degenerates to bootstrapping n_draws from one year, same cost as
                # before this round for this specific path (it was never the source of
                # the OOM -- only one year per drivetrain per period, always).
                midpoint_year = round((period[0] + period[1]) / 2)
                cohort_year_weights_standard: dict[tuple[str, str], dict[int, float]] = {}
                for drv in p04.drivetrains:
                    available_years = composition_years_by_drivetrain_standard.get(drv)
                    if available_years is None or len(available_years) == 0:
                        continue
                    clamped_year = int(available_years[np.argmin(np.abs(available_years - midpoint_year))])
                    cohort_year_weights_standard[(drv, "standard")] = {clamped_year: 1.0}

                mixed_composition_draws_standard = bootstrap_mixed_composition_draws(
                    histogram_standard_all, cohort_year_weights_standard, n_draws, composition_rng,
                    verbose=False,  # tiny (one year per drivetrain) -- per-group progress prints add noise, not signal
                )

                print(f"  [{flow}] period {period}: MC mass, standard ({n_draws:,} draws)...")
                mc_draws_tables_standard[(period, flow)] = combine_flow_and_composition_draws(
                    mc_summary, scenario_name, period, flow_metric, mixed_composition_draws_standard,
                    n_draws, scenario_rng, direct=True,
                )

        # ---------------------------------------------------------------------------
        # Diagnostic plots: material mass by year + mass by year by segment, for
        # EVERY drivetrain, for both flows. The MC panel now shows one box PER
        # CATEGORY (material, or segment) so categories can be compared against each
        # other -- not one aggregate box. Saved as PDF (vector), per the user's
        # preference.
        # ---------------------------------------------------------------------------
        print(f"  Generating diagnostic plots for {len(p04.drivetrains)} drivetrains x "
              f"{len(FLOW_VALUES_IN_SCOPE)} flows...")
        for flow in sorted(FLOW_VALUES_IN_SCOPE):
            headline_mc_draws = mc_draws_tables[(headline_period, flow)]
            for drivetrain in p04.drivetrains:
                # Aggregate this drivetrain's (segment, components, material) MC groups
                # down to per-MATERIAL and per-SEGMENT period-total draws -- each
                # summed across every other dimension -- for the two comparison plots.
                mc_by_material: dict[str, np.ndarray] = {}
                mc_by_segment: dict[str, np.ndarray] = {}
                for (drv, seg, _comp, mat), draws in headline_mc_draws.items():
                    if drv != drivetrain:
                        continue
                    mc_by_material[mat] = mc_by_material.get(mat, np.zeros(n_draws)) + draws
                    mc_by_segment[seg] = mc_by_segment.get(seg, np.zeros(n_draws)) + draws

                title_prefix = f"{scenario_name} / {flow}"
                plot_material_mass_by_year_per_drivetrain(
                    mass_by_year_tables[flow], drivetrain, mc_by_material or None, title_prefix,
                    fig_dir / f"04_01_mass_by_year_material_{scenario_name}_{flow}_{drivetrain}.png",
                )
                plot_material_mass_by_year_per_segment(
                    mass_by_year_tables[flow], drivetrain, mc_by_segment or None, title_prefix,
                    fig_dir / f"04_01_mass_by_year_segment_{scenario_name}_{flow}_{drivetrain}.png",
                )

        # ---------------------------------------------------------------------------
        # [NEW] "Reasonable comparison" figure: 12-segment total vs. drivetrain-level
        # ("standard") total, all 5 drivetrains side by side, at the HEADLINE period
        # (the widest requested period -- same convention as the other MC-panel
        # plots above). One figure per flow.
        # ---------------------------------------------------------------------------
        print(f"  Generating standard-vs-segments comparison plots ({len(FLOW_VALUES_IN_SCOPE)} flows)...")
        for flow in sorted(FLOW_VALUES_IN_SCOPE):
            plot_standard_vs_segments_comparison(
                mc_draws_tables[(headline_period, flow)],
                mc_draws_tables_standard[(headline_period, flow)],
                list(p04.drivetrains),
                title=f"{scenario_name} / {flow}: 12-segment vs. standard composition, "
                      f"{headline_period[0]}-{headline_period[1]}",
                fig_path=fig_dir / f"04_01_standard_vs_segments_{scenario_name}_{flow}.png",
            )
            print(f"  Saved diagnostic plot: {fig_dir}/04_01_standard_vs_segments_{scenario_name}_{flow}.png")

        # ---------------------------------------------------------------------------
        # Persist: same "unregistered per-scenario pickle" convention the OLD
        # 04_01_materials.py used for its per-scenario outputs.
        # ---------------------------------------------------------------------------
        saved_paths.update(save_unregistered_scenario_outputs(artifacts_dir, {
            f"04_01_scalar_mass_{scenario_name}.pkl": scalar_tables,
            f"04_01_mc_mass_draws_{scenario_name}.pkl": mc_draws_tables,
            f"04_01_mass_by_year_{scenario_name}.pkl": mass_by_year_tables,
            # [NEW] Drivetrain-level ("standard") counterparts.
            f"04_01_scalar_mass_standard_{scenario_name}.pkl": scalar_tables_standard,
            f"04_01_mc_mass_draws_standard_{scenario_name}.pkl": mc_draws_tables_standard,
        }))
        all_outputs[scenario_name] = {
            "scalar_tables": scalar_tables,
            "mc_draws_tables": mc_draws_tables,
            "mass_by_year_tables": mass_by_year_tables,
            "scalar_tables_standard": scalar_tables_standard,
            "mc_draws_tables_standard": mc_draws_tables_standard,
        }
        print(f"  Scenario {scenario_name} done in {time.time() - t_start_scenario:.1f}s.")


    # -------------------------------------------------------------------------------
    # [NEW] Cross-scenario comparison plots: only meaningful with 2+ active
    # scenarios (e.g. BAU + stock_lower). For each (flow, drivetrain): (a) total
    # mass-by-year trajectory, one line per scenario, with a headline-period MC
    # density panel; (b) a boxplot + PDF pair comparing headline-period total mass
    # distributions across scenarios.
    # -------------------------------------------------------------------------------
    if len(active_scenario_names) > 1:
        print(f"\n{'-' * 76}\nCross-scenario comparison plots ({len(active_scenario_names)} scenarios)\n{'-' * 76}")
        scenario_colors = _scenario_color_map(list(active_scenario_names))
        for flow in sorted(FLOW_VALUES_IN_SCOPE):
            mass_by_year_by_scenario = {
                name: all_outputs[name]["mass_by_year_tables"][flow] for name in active_scenario_names
            }
            for drivetrain in p04.drivetrains:
                headline_draws_by_scenario: dict[str, np.ndarray] = {}
                for name in active_scenario_names:
                    mc_draws = all_outputs[name]["mc_draws_tables"][(headline_period, flow)]
                    total = np.zeros(n_draws, dtype=float)
                    found = False
                    for (drv, _seg, _comp, _mat), draws in mc_draws.items():
                        if drv == drivetrain:
                            total = total + draws
                            found = True
                    if found:
                        headline_draws_by_scenario[name] = total

                plot_total_mass_by_year_scenario_comparison(
                    mass_by_year_by_scenario, headline_draws_by_scenario, drivetrain,
                    title=f"{flow}: total mass by year across scenarios -- {drivetrain}",
                    fig_path=fig_dir / f"04_01_scenario_comparison_mass_by_year_{flow}_{drivetrain}.png",
                    colors=scenario_colors,
                )

                if len(headline_draws_by_scenario) > 1:
                    plot_scenario_comparison_boxplot_and_pdf(
                        headline_draws_by_scenario,
                        title=f"{flow}: {drivetrain} total mass across scenarios, "
                              f"{headline_period[0]}-{headline_period[1]}",
                        xlabel="Total mass [t]",
                        fig_path_boxplot=fig_dir / f"04_01_scenario_comparison_boxplot_{flow}_{drivetrain}.png",
                        fig_path_pdf=fig_dir / f"04_01_scenario_comparison_pdf_{flow}_{drivetrain}.png",
                        colors=scenario_colors,
                    )

        # [NEW] Material breakdown (grouped bars) + percent-change delta plots.
        for flow in sorted(FLOW_VALUES_IN_SCOPE):
            mc_draws_by_scenario_flow = {
                name: all_outputs[name]["mc_draws_tables"][(headline_period, flow)] for name in active_scenario_names
            }
            for drivetrain in p04.drivetrains:
                plot_material_breakdown_scenario_comparison(
                    mc_draws_by_scenario_flow, drivetrain,
                    title=f"{flow}: material breakdown across scenarios -- {drivetrain}, "
                          f"{headline_period[0]}-{headline_period[1]}",
                    fig_path=fig_dir / f"04_01_scenario_comparison_materials_{flow}_{drivetrain}.png",
                    colors=scenario_colors,
                )

            baseline_scenario = "BAU" if "BAU" in active_scenario_names else active_scenario_names[0]
            for comparison_scenario in active_scenario_names:
                if comparison_scenario == baseline_scenario:
                    continue
                plot_scenario_delta_by_drivetrain_and_material(
                    mc_draws_by_scenario_flow, baseline_scenario, comparison_scenario,
                    list(p04.drivetrains),
                    title=f"{flow}: % change {baseline_scenario} -> {comparison_scenario}, "
                          f"{headline_period[0]}-{headline_period[1]}",
                    fig_path=fig_dir / f"04_01_scenario_comparison_delta_{flow}_{baseline_scenario}_vs_{comparison_scenario}.png",
                )
        print(f"  Saved material-breakdown and delta comparison plots to {fig_dir}")
        print(f"  Saved cross-scenario comparison plots to {fig_dir}")
    else:
        print("\nOnly one active scenario -- skipping cross-scenario comparison plots.")

    print(f"\n{'=' * 76}\nAll scenarios done in {time.time() - t_start_main:.1f}s.")
    print("Saved (per-scenario, unregistered):", saved_paths)
    return {"saved_paths": saved_paths, "outputs": all_outputs}


if __name__ == "__main__":
    main()