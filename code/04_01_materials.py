"""
04_01_materials.py
====================

Stage 04, part 1: merges the materials "tracker" (per-flow, per-cohort vehicle/segment
counts from stage 03) with a bulk MATERIAL composition table (steel, aluminum, plastics,
etc.), producing total material MASS by year/flow/drivetrain/material-level.

Runs this for the baseline tracker (`tracker_keyed`) AND all 10 named scenario trackers
from stage 03.

ESSENTIAL REQUIREMENT already satisfied: `MATERIAL_PARAMETER_CODE = "m-c"` (component +
material level, never decomposed into individual chemical elements) -- confirmed
correct, no change needed. See `params_schema.py`'s `MaterialsParams` for the
centralized setting this now reads from.

FIXES APPLIED THIS ROUND
--------------------------
- Dataclass params access throughout (`params["01_data_prep"]` -> `params.data_prep`,
  etc.), same pattern as every other stage.
- Upward-searching `_find_project_root` + `SCRIPT_DIR`-anchored `input_dir` resolution
  (same fix class as `01_data_prep.py`'s and `03_01_flowdriven.py`'s CWD-independence
  fixes this round -- composition file paths are no longer resolved against the
  caller's working directory).
- `root=PROJECT_ROOT` threaded into `load_many`/`save_many`.
- **CASCADE WARNING RESOLVED**: the original review flagged that this notebook
  processes all 11 scenario trackers including 5 that were, at the time, silent copies
  of BAU (stage 03's C6 finding). C6 is now fixed and verified (see
  `03_02_adjustedflows.py`) -- all 11 scenario trackers are genuinely different from
  each other where their underlying assumptions differ. This notebook's per-scenario
  loop now produces 11 genuinely distinct sets of outputs, not a mix of real and
  silently-duplicated ones.
- **Persistence cleaned up**: the original saved the SAME baseline data three times
  (once via `save_many` under legacy names, once via a second raw, unregistered pickle
  under different filenames). Reduced to one clean `save_many` call for the baseline,
  plus the shared `materials.save_unregistered_scenario_outputs()` helper (new this
  round, also used by `04_03_tractionmotors.py`/`04_04_batteries.py`) for the 11
  per-scenario outputs, which have no `ARTIFACT_FILES` registry entry.
- **New integrated diagnostic plot** (per standing practice): total BAU material mass
  by year, stacked by material category -- generated right after the baseline
  computation, the exact point where this data first exists.

MATERIAL_LEVEL_KEY changed from a locally hardcoded constant to
`params.materials.material_level_key` (default `"materialKeyLevel_highest"`,
unchanged value, now centralized).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

# Same fix as 01_data_prep.py / 03_01_flowdriven.py this round: resolve relative params
# (input_dir) against THIS script's own location, not the caller's cwd.
SCRIPT_DIR = Path(__file__).resolve().parent

import src.artifacts as artifacts  # type: ignore
import src.materials as materials  # type: ignore

load_many = artifacts.load_many
save_many = artifacts.save_many
artifact_status = artifacts.artifact_status

quantify_elements_from_tracker = materials.quantify_elements_from_tracker
combine_materials_by_flow = materials.combine_materials_by_flow
save_unregistered_scenario_outputs = materials.save_unregistered_scenario_outputs

MATERIAL_LEVELS = ["materialKeyLevel4", "materialKeyLevel3", "materialKeyLevel2", "materialKeyLevel1"]
MATERIAL_LEVEL_COLS = [
    "materialKeyLevel0", "materialKeyLevel1", "materialKeyLevel2",
    "materialKeyLevel3", "materialKeyLevel4", "materialKeyLevel_highest",
]
COLUMNS_TO_KEEP = [
    "productionYear", "productKeyLevel1", "productKeyLevel2", "componentKeyLevel0",
    "componentKeyLevel1", "materialKeyLevel0", "materialKeyLevel1", "materialKeyLevel2",
    "materialKeyLevel3", "materialKeyLevel4", "parameterCode", "element", "value",
]


def load_composition(p01, p04) -> pd.DataFrame:
    """
    Load and extend the bulk material composition table.

    THE MATH MODEL: static composition beyond `composition_extend_from_year`. Real
    Excel-sourced composition data presumably only covers years up to
    `composition_extend_from_year` (2050). Every year from 2051 through
    `composition_end_year` (`p04.material_level_key`'s horizon, from
    `visualization.year_plot_end`) is filled by literally COPYING the 2050 row set
    forward unchanged -- i.e. the model assumes material composition is FROZEN at its
    2050 value for the rest of the century. A common, defensible simplification absent
    a real composition forecast, but stated explicitly here as an assumption.
    """
    input_dir = SCRIPT_DIR / p01.input_dir
    composition_file = input_dir / p01.composition_file_name
    petrol_composition_file = input_dir / p01.petrol_composition_file_name
    sheets = p01.sheets
    composition_extend_from_year = int(p01.composition_extend_from_year)

    raw_sheets = {sheet: pd.read_excel(composition_file, sheet_name=sheet) for sheet in sheets}
    petrol_composition_df = pd.read_excel(petrol_composition_file)

    df_list = [df[COLUMNS_TO_KEEP] for df in raw_sheets.values()]
    composition_raw = pd.concat(df_list, ignore_index=True)

    # [STILL OPEN, unchanged -- flagged not fixed]: this filter is applied ONLY to
    # composition_raw (main-vehicle sheets), not to composition_petrol. If the petrol
    # workbook shares the same componentKeyLevel0 taxonomy and also has "elvEVspecific"
    # rows, those would slip through unfiltered. Not changed without seeing real data.
    composition_raw = composition_raw[composition_raw["componentKeyLevel0"] != "elvEVspecific"]

    composition_petrol = petrol_composition_df[COLUMNS_TO_KEEP]
    composition_wo_petrol = composition_raw[composition_raw["productKeyLevel1"] != "elvPetrol"]
    composition = pd.concat([composition_wo_petrol, composition_petrol], ignore_index=True)

    composition_end_year = int(p04.get("year_plot_end", composition_extend_from_year))

    rows_anchor = composition[composition["productionYear"] == composition_extend_from_year].copy()
    extended_rows = []
    for year in range(composition_extend_from_year + 1, composition_end_year + 1):
        temp = rows_anchor.copy()
        temp["productionYear"] = year
        extended_rows.append(temp)

    composition_extended = (
        pd.concat([composition] + extended_rows, ignore_index=True) if extended_rows else composition.copy()
    )

    available_parameter_codes = sorted(
        composition_extended["parameterCode"].dropna().astype(str).str.strip().unique().tolist()
    )
    print("Available parameterCode values:", available_parameter_codes)
    print("Selected MATERIAL_PARAMETER_CODE:", p04["composition_parameter_code"])
    print("Selected MATERIAL_LEVEL_KEY:", p04["material_level_key"])

    # THE MATH MODEL: hierarchical fallback to the deepest available material level.
    # `materialKeyLevel_highest` = the deepest non-null value among Level4 > Level3 >
    # Level2 > Level1, via bfill. materialKeyLevel0 is intentionally excluded from this
    # fallback chain -- a row with data ONLY at level 0 gets materialKeyLevel_highest =
    # NaN, which would render as the literal string "nan" downstream. [STILL OPEN,
    # unchanged]: worth checking whether any real composition rows actually hit this.
    for col in MATERIAL_LEVELS:
        if col in composition_extended.columns:
            composition_extended[col] = composition_extended[col].astype(str).str.strip()
            composition_extended.loc[composition_extended[col].isin(["", "nan", "None"]), col] = pd.NA

    composition_extended["materialKeyLevel_highest"] = (
        composition_extended[MATERIAL_LEVELS].bfill(axis=1).iloc[:, 0]
    )

    return composition_extended


def quantify_and_aggregate(tracker_keyed: dict, composition_extended: pd.DataFrame, p04) -> tuple[dict, dict]:
    """Quantify material mass from one tracker and aggregate to (Region, Drive Train, scrap_year, material level)."""
    material_level_key = p04["material_level_key"]

    quantified = quantify_elements_from_tracker(
        tracker_keyed=tracker_keyed,
        composition_extended=composition_extended,
        parameter_code=p04["composition_parameter_code"],
        material_level_key=material_level_key,
    )

    mass_by_year_materials_dict = {}
    for key, frame in quantified.items():
        if frame.empty:
            continue
        required_cols = {"scrap_year", "mass", material_level_key}
        missing_cols = required_cols.difference(frame.columns)
        if missing_cols:
            raise KeyError(f"Missing columns for material aggregation: {sorted(missing_cols)}")

        keep_cols = [c for c in MATERIAL_LEVEL_COLS if c in frame.columns]
        grouped = (
            frame.groupby(["Region", "Drive Train", "scrap_year"] + keep_cols, as_index=False, dropna=False)["mass"]
            .sum()
            .sort_values(["Region", "Drive Train", "scrap_year"] + keep_cols)
            .reset_index(drop=True)
        )
        grouped["material"] = grouped[material_level_key].astype(str)
        grouped["element"] = grouped["material"]  # kept for downstream-schema compatibility
        mass_by_year_materials_dict[key] = grouped

    combined_materials = combine_materials_by_flow(mass_by_year_materials_dict)
    return mass_by_year_materials_dict, combined_materials


def plot_material_mass_by_year(combined_materials: dict, region: str = "EUR") -> tuple[plt.Figure, plt.Axes]:
    """
    Total material mass by year for the "inflow" flow, stacked by material category --
    the most direct "does the materials stage look right" check: what's the model
    actually saying about material demand over time, broken down by material.
    """
    key = (region, "inflow")
    fig, ax = plt.subplots(figsize=(11, 6))
    if key not in combined_materials or combined_materials[key].empty:
        ax.text(0.5, 0.5, f"No inflow data for region={region!r}", ha="center", va="center")
        return fig, ax

    df = combined_materials[key]
    pivot = df.pivot_table(index="scrap_year", columns="material", values="mass", aggfunc="sum", fill_value=0.0)
    # Keep the plot readable: only the top N materials by total mass, rest grouped as "Other".
    top_n = 8
    totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_materials = totals.index[:top_n].tolist()
    other_materials = totals.index[top_n:].tolist()
    plot_df = pivot[top_materials].copy()
    if other_materials:
        plot_df["Other"] = pivot[other_materials].sum(axis=1)

    ax.stackplot(plot_df.index, plot_df.T.values, labels=plot_df.columns, alpha=0.85)
    ax.set_title(f"Material mass by year, inflow ({region}, BAU)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mass [kg]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig, ax


def main() -> dict[str, Any]:
    scenario_names = [
        "BAU", "BEV_only", "stock_lower", "BEV_A_F", "BEV_JA_JF", "BEV_large", "BEV_small",
        "BEV_longer", "ICEV_shorter", "losses_zero", "losses_high",
    ]
    loaded = load_many(
        "params", "tracker_keyed", *[f"tracker_keyed_{n}" for n in scenario_names],
        root=PROJECT_ROOT,
    )
    params = loaded["params"]
    p01 = params.data_prep
    p04_dict = params.to_nested_dict()["04_materials"]
    p04_dict["year_plot_end"] = params.visualization.year_plot_end
    tracker_keyed = loaded["tracker_keyed"]
    trackers = {n: loaded[f"tracker_keyed_{n}"] for n in scenario_names}

    print(artifact_status(root=PROJECT_ROOT))

    composition_extended = load_composition(p01, p04_dict)

    # -----------------------------------------------------------------------
    # Baseline (un-suffixed "tracker_keyed", i.e. 03_01's flow-driven baseline -- NOT
    # the same methodology as "tracker_keyed_BAU" from 03_02. Both get processed
    # through this stage, in parallel, under different names -- worth being explicit
    # about which one is "the" official baseline for reporting purposes.)
    # -----------------------------------------------------------------------
    mass_by_year_materials_dict, combined_materials = quantify_and_aggregate(tracker_keyed, composition_extended, p04_dict)

    # [FIXED] one clean, registered save for the baseline -- was three saves (one
    # registered, two more raw/unregistered duplicates of the SAME data) previously.
    saved = save_many(
        mass_by_year_elem_dict=mass_by_year_materials_dict,
        combined=combined_materials,
        composition_extended=composition_extended,
        root=PROJECT_ROOT,
    )

    # -----------------------------------------------------------------------
    # Diagnostic plot: material mass by year, BAU baseline -- integrated here, the
    # exact point this data first exists.
    # -----------------------------------------------------------------------
    fig, _ = plot_material_mass_by_year(combined_materials, region=params.materials.region)
    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "04_01_material_mass_by_year.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # All 11 named scenarios -- C6 is now fixed, so these are genuinely 11 distinct
    # sets of outputs, not a mix of real and silently-duplicated placeholders.
    # -----------------------------------------------------------------------
    artifacts_dir = PROJECT_ROOT / "data" / "processed" / "intermediate"
    all_outputs: dict[str, Any] = {}
    saved_paths: dict[str, Any] = {}
    for name, tracker in trackers.items():
        mass_dict, combined = quantify_and_aggregate(tracker, composition_extended, p04_dict)
        print(name, "| mass_dict:", len(mass_dict), "| combined_dict:", len(combined))

        saved_paths.update(save_unregistered_scenario_outputs(artifacts_dir, {
            f"04_mass_by_year_materials_dict_{name}.pkl": mass_dict,
            f"04_combined_materials_{name}.pkl": combined,
        }))
        all_outputs[name] = {"mass_by_year_materials_dict": mass_dict, "combined_materials": combined}

    print("Saved (per-scenario, unregistered -- see materials.py's helper docstring for why):", saved_paths)
    print("Saved (registered):", saved)
    return {"saved": saved, "saved_paths": saved_paths, "outputs": all_outputs}


if __name__ == "__main__":
    main()
