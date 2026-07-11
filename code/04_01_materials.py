"""
04_01_materials.py
====================

Stage 04, part 1: combines the "tracker" (per-flow, per-cohort vehicle/segment counts
from stage 03) with COMPONENT-MATERIAL (C-M) composition data for a given drivetrain and
segment, producing total material MASS by year/flow/drivetrain/material, WITH combined
Monte Carlo uncertainty (vehicle-count draws x composition draws) when MC is enabled.

*** REWRITE IN PROGRESS -- being rebuilt step by step, confirmed with testing at each
    step before moving to the next. This file is NOT yet complete. ***

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
  - `segment == "standard"` rows are dropped (not one of the 12 real segments).
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
        "standard", which is dropped here (out of scope, confirmed).
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
        df = df[df["segment"] != "standard"].copy()
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
    return cache_dir / f"04_01_histogram_cache_{drivetrain}.parquet"


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

    `segment == "standard"` rows are dropped here too (same out-of-scope rule as
    `load_composition_summary`). The "variable" column is renamed to "material" for
    consistency with the summary loader.

    If `verbose` (default True -- this step can take a while on the real file and was
    otherwise silent, which looks like a hang), prints a line when each sheet starts,
    a running row count every `progress_every` rows read, and a per-sheet summary
    (rows kept, elapsed time) when each sheet finishes.
    """
    import openpyxl

    t_start_all = time.time()
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    try:
        available = set(wb.sheetnames)
        missing = [s for s in sheet_names if s not in available]
        if missing:
            raise KeyError(
                f"Histogram sheet(s) {missing} (mapped to drivetrain {drivetrain!r} via "
                f"params.materials.histogram_sheet_names_by_drv) not found in "
                f"{file_path.name}. Available sheets: {wb.sheetnames}"
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
                if segment == "standard":
                    continue  # out of scope, same rule as load_composition_summary
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
def mc_summary_key(scenario_name: str, period: tuple[int, int], drivetrain: str, metric: str, segment: str | None = None) -> str:
    """
    Build the exact key format `mc_stage03_02_summary` uses, confirmed against the
    real saved artifact: f"{scenario}__{start}-{end}__{drivetrain}__{metric}" for
    drivetrain-level entries (segment=None), or
    f"{scenario}__{start}-{end}__{drivetrain}__{segment}__{metric}" for segment-level
    entries.
    """
    start, end = period
    period_label = f"{start}-{end}"
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


def combine_flow_and_composition_draws(
    mc_summary: dict,
    scenario_name: str,
    period: tuple[int, int],
    flow_metric: str,
    composition_draws_by_group: dict[tuple, np.ndarray],
    n_draws: int,
    rng: np.random.Generator,
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
    composition_draws_by_group : output of `bootstrap_composition_draws`, keyed by
        (components, material, segment, year, drivetrain)
    n_draws : number of combined draws to produce (independent of stage 03_02's own
        n_draws -- see architecture note)
    rng : numpy Generator for the vehicle-count bootstrap (composition draws are
        assumed already sampled -- this function does not draw them)

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
    # (skipped, not re-attempted).
    count_draws_cache: dict[tuple[str, str], np.ndarray | None] = {}
    mass_draws_by_group: dict[tuple, np.ndarray] = {}
    skipped_groups: set[tuple[str, str]] = set()

    for comp_key, comp_draws in composition_draws_by_group.items():
        components, material, segment, _year, drivetrain = comp_key

        if (drivetrain, segment) not in count_draws_cache:
            key = mc_summary_key(scenario_name, period, drivetrain, flow_metric, segment=segment)
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
    Restrict a composition summary DataFrame (from `load_composition_summary`) to only
    the FIRST available year PER DRIVETRAIN.

    Confirmed scope: composition data is currently only available every 5 years, and
    different drivetrains start at different years (Petrol/Diesel from 1990, BEV/HEV/
    PHEV from 2000 -- confirmed against the real uploaded file). Per the user ("at the
    moment just take the first entry" -- this restriction is expected to be lifted once
    the source data reaches annual resolution), this keeps only each drivetrain's own
    earliest year, not one single year applied uniformly across all drivetrains.
    """
    first_years = composition_summary.groupby("drivetrain")["year"].transform("min")
    return composition_summary[composition_summary["year"] == first_years].copy()


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

    NOTE on cohort-year vs. flow-year composition: strictly, a scrapped vehicle's
    material composition should reflect its BUILD year (`cohort_year`), not its scrap
    year (`scrap_year`) -- a car scrapped in 2050 was typically built well before 2050
    and contains whatever materials were used at build time. This distinction is
    deliberately NOT modeled yet, because composition data is currently restricted to
    one representative "first available year" per drivetrain regardless of vehicle age
    (see `select_first_year_composition`) -- there is currently no cohort-year-specific
    composition to look up even if we wanted to. This will need revisiting once
    composition data reaches annual resolution and per-cohort composition becomes
    available; flagged here, not solved now.

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
        counts = (
            sub.groupby("Segment", as_index=False)["amount"]
            .sum()
            .rename(columns={"Segment": "segment", "amount": "vehicle_count"})
        )
        counts["drivetrain"] = drivetrain
        frames.append(counts)

    empty_cols = ["region", "flow", "drivetrain", "segment", "components", "material", "vehicle_count", "composition_value", "mass"]
    if not frames:
        return pd.DataFrame(columns=empty_cols)

    counts_all = pd.concat(frames, ignore_index=True)
    # `amount` (and therefore `vehicle_count`) is in MILLIONS of vehicles -- confirmed
    # via diagnose_vehicle_count_units.py. Convert to raw vehicle count before
    # multiplying into per-vehicle composition mass (see VEHICLE_COUNT_UNIT_SCALE note).
    counts_all["vehicle_count"] *= VEHICLE_COUNT_UNIT_SCALE

    comp = composition_summary_first_year[["drivetrain", "segment", "components", "material", stat_col]].rename(
        columns={stat_col: "composition_value"}
    )

    merged = counts_all.merge(comp, on=["drivetrain", "segment"], how="inner")
    merged["region"] = region
    merged["flow"] = flow
    merged["mass"] = merged["vehicle_count"] * merged["composition_value"]

    return merged[empty_cols].reset_index(drop=True)


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
    summed total.

    IMPORTANT SIMPLIFICATION (confirmed with the user, not a bug): vehicle counts come
    from `tracker_keyed`, which genuinely varies year to year (`scrap_year`). The
    COMPOSITION side still only has each drivetrain's single "first available year"
    value (see `select_first_year_composition`) -- composition intensity (kg/vehicle)
    is therefore held CONSTANT across every plotted year; only the vehicle-count side
    produces the year-to-year trend. This will change once composition data reaches
    annual resolution and the "first year only" restriction is lifted.

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
    available for that (drivetrain, segment), columns:
      region, flow, drivetrain, segment, scrap_year, components, material,
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
        counts = (
            sub.groupby(["Segment", "scrap_year"], as_index=False)["amount"]
            .sum()
            .rename(columns={"Segment": "segment", "scrap_year": "year", "amount": "vehicle_count"})
        )
        counts["drivetrain"] = drivetrain
        frames.append(counts)

    empty_cols = [
        "region", "flow", "drivetrain", "segment", "year", "components", "material",
        "vehicle_count", "composition_value", "mass",
    ]
    if not frames:
        return pd.DataFrame(columns=empty_cols)

    counts_all = pd.concat(frames, ignore_index=True)
    # `amount` (and therefore `vehicle_count`) is in MILLIONS of vehicles -- confirmed
    # via diagnose_vehicle_count_units.py. See VEHICLE_COUNT_UNIT_SCALE note above.
    counts_all["vehicle_count"] *= VEHICLE_COUNT_UNIT_SCALE

    comp = composition_summary_first_year[["drivetrain", "segment", "components", "material", stat_col]].rename(
        columns={stat_col: "composition_value"}
    )

    merged = counts_all.merge(comp, on=["drivetrain", "segment"], how="inner")
    merged["region"] = region
    merged["flow"] = flow
    merged["mass"] = merged["vehicle_count"] * merged["composition_value"]

    return merged[empty_cols].reset_index(drop=True)


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
    `04_01_materials.py` has no cross-file dependency on that script.

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
    """
    df = mass_by_year_df[mass_by_year_df["drivetrain"] == drivetrain]
    if df.empty:
        print(f"[_plot_mass_by_year_stacked] no data for drivetrain={drivetrain!r} -- skipping {fig_path.name}.")
        return

    pivot = df.groupby(["year", stack_col])["mass"].sum().unstack(stack_col, fill_value=0.0).sort_index()

    if stack_col == "material" and top_n is not None and pivot.shape[1] > top_n:
        totals = pivot.sum(axis=0).sort_values(ascending=False)
        top_cols = totals.index[:top_n].tolist()
        other_cols = totals.index[top_n:].tolist()
        plot_df = pivot[top_cols].copy()
        plot_df["Other"] = pivot[other_cols].sum(axis=1)
        category_members = {c: [c] for c in top_cols}
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

    if show_mc_panel:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax2 = None

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
        bbox_to_anchor=(0, -0.15) if ax2 is not None else (1.02, 1),
        ncol=4 if ax2 is not None else 1,
        frameon=False, fontsize=8,
    )
    stack_colors = {label: poly.get_facecolor()[0] for label, poly in zip(plot_df_t.columns, stack_polys)}

    if ax2 is not None:
        # One overlaid Gaussian KDE density curve per category, color-matched to that
        # category's stackplot band -- lets you compare the SHAPE and spread of each
        # category's period-total-mass uncertainty directly against the others, all on
        # a shared x-axis (mass), rather than a single aggregate distribution.
        for category, draws in zip(density_categories, density_draws):
            grid, density = _gaussian_kde_curve(draws / KG_PER_TONNE)
            color = stack_colors.get(category, "gray")
            ax2.plot(grid, density, color=color, linewidth=1.6, label=category)
            ax2.fill_between(grid, density, color=color, alpha=0.15)
        ax2.set_title("Period total (MC), by category", fontsize=10)
        ax2.set_xlabel("Total mass, whole period [t]")
        ax2.set_ylabel("Probability density")
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(loc="upper right", frameon=False, fontsize=7)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def plot_material_mass_by_year_per_drivetrain(
    mass_by_year_df: pd.DataFrame,
    drivetrain: str,
    mc_draws_by_material: dict[str, np.ndarray] | None,
    title_prefix: str,
    fig_path: Path,
    top_n: int = 8,
) -> None:
    """
    For ONE drivetrain: stacked-area material mass by year (aggregated across all
    segments and components sharing the same material name), plus a period-total MC
    boxplot panel comparing materials against each other. See
    `_plot_mass_by_year_stacked`'s docstring for the panel-layout reasoning.

    mc_draws_by_material : dict mapping EVERY individual material name (not yet
        grouped into "Other") -> (n_draws,) period-total mass array for that material,
        summed across all segments/components for this drivetrain.
    """
    _plot_mass_by_year_stacked(
        mass_by_year_df, drivetrain, "material", mc_draws_by_material,
        title=f"{title_prefix}: material mass by year -- {drivetrain}",
        fig_path=fig_path, top_n=top_n,
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
    _plot_mass_by_year_stacked(
        mass_by_year_df, drivetrain, "segment", mc_draws_by_segment,
        title=f"{title_prefix}: mass by year by segment -- {drivetrain}",
        fig_path=fig_path, top_n=None,
    )


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
    print("04_01_materials.py -- main()")
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
    composition_first_year = select_first_year_composition(composition_summary)

    print("\nBootstrapping composition MC draws for all drivetrains...")
    composition_rng = np.random.default_rng(composition_seed)
    composition_draws_all: dict[tuple, np.ndarray] = {}
    for drivetrain in p04.drivetrains:
        histogram_df = load_histogram_data(p04, drivetrain, project_root=PROJECT_ROOT)
        first_year = int(histogram_df["year"].min())
        histogram_first_year = histogram_df[histogram_df["year"] == first_year]
        drv_draws = bootstrap_composition_draws(histogram_first_year, n_draws, composition_rng)
        composition_draws_all.update(drv_draws)
    print(f"Composition MC draws ready: {len(composition_draws_all):,} groups across "
          f"{len(p04.drivetrains)} drivetrains.")

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
                print(f"  [{flow}] period {period}: scalar mass...")
                scalar_tables[(period, flow)] = combine_scalar_mass_from_tracker(
                    tracker_keyed, flow, period, composition_first_year, p04, region=region
                )

                print(f"  [{flow}] period {period}: MC mass ({n_draws:,} draws)...")
                mc_draws_tables[(period, flow)] = combine_flow_and_composition_draws(
                    mc_summary, scenario_name, period, flow_metric, composition_draws_all,
                    n_draws, scenario_rng,
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
        # Persist: same "unregistered per-scenario pickle" convention the OLD
        # 04_01_materials.py used for its per-scenario outputs.
        # ---------------------------------------------------------------------------
        saved_paths.update(save_unregistered_scenario_outputs(artifacts_dir, {
            f"04_01_scalar_mass_{scenario_name}.pkl": scalar_tables,
            f"04_01_mc_mass_draws_{scenario_name}.pkl": mc_draws_tables,
            f"04_01_mass_by_year_{scenario_name}.pkl": mass_by_year_tables,
        }))
        all_outputs[scenario_name] = {
            "scalar_tables": scalar_tables,
            "mc_draws_tables": mc_draws_tables,
            "mass_by_year_tables": mass_by_year_tables,
        }
        print(f"  Scenario {scenario_name} done in {time.time() - t_start_scenario:.1f}s.")

    print(f"\n{'=' * 76}\nAll scenarios done in {time.time() - t_start_main:.1f}s.")
    print("Saved (per-scenario, unregistered):", saved_paths)
    return {"saved_paths": saved_paths, "outputs": all_outputs}


if __name__ == "__main__":
    main()