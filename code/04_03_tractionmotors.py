"""
04_03_tractionmotors.py
=========================

Stage 04, part 3: material-only quantification for traction motors, using a SEPARATE
composition workbook from the main `ELV_2010_2050.xlsx` used by 04_01.

ESSENTIAL REQUIREMENT already satisfied: `MATERIAL_PARAMETER_CODE = "m-c"` (component +
material level, never decomposed into individual chemical elements) -- confirmed
correct, no change needed.

FIXES APPLIED THIS ROUND
--------------------------
- Dataclass params access throughout.
- Upward-searching `_find_project_root` + `SCRIPT_DIR`-anchored `input_dir` resolution
  (same CWD-independence fix as every other stage this round).
- `root=PROJECT_ROOT` threaded into `load_many`.
- **Traction composition filename centralized**: was hardcoded directly in this file
  (`"20260309-Traction_motors_consolidated.xlsx"`, a date-stamped name) -- now read
  from `params.materials.traction_composition_file_name`.
- **Persistence cleaned up**: replaced the raw, ad hoc pickle-saving loops with the
  shared `materials.save_unregistered_scenario_outputs()` helper (new this round, also
  used by `04_01_materials.py`/`04_04_batteries.py`).
- **New integrated diagnostic plot**: the secondary-supply ratio
  (`build_material_ratio_df`, already computed by this file but never plotted) --
  collected-vs-inflow mass ratio over time, the direct visual for "what fraction of
  this year's traction-motor material demand could be met by end-of-life collection."

STILL OPEN, UNCHANGED (flagged, not fixed -- needs real data or your confirmation)
--------------------------------------------------------------------------------------
- The `.abs()` normalization on `productionYear` (fixing negative-year data-quality
  quirks) would silently collide a genuine "-2020" and genuine "2020" row if both
  existed. Not changed without seeing real data.
- The single-anchor-year, bidirectional composition extension is a materially stronger
  assumption than 04_01's forward-only extension (traction motor material intensity
  assumed CONSTANT across the entire vehicle history, not just frozen beyond 2050).
  Documented, not changed -- this is a modeling choice, not a bug.
- Whether the traction file provides distributional data (`meanValue` alongside
  min/max/std) that's being collapsed to a point estimate -- not verifiable without
  the real file.
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

SCRIPT_DIR = Path(__file__).resolve().parent

from src.artifacts import load_many, artifact_status  # type: ignore
import src.materials as materials  # type: ignore

quantify_elements_from_tracker = materials.quantify_elements_from_tracker
save_unregistered_scenario_outputs = materials.save_unregistered_scenario_outputs

MATERIAL_LEVELS = ["materialKeyLevel4", "materialKeyLevel3", "materialKeyLevel2", "materialKeyLevel1"]
MATERIAL_LEVEL_COLS = [
    "materialKeyLevel0", "materialKeyLevel1", "materialKeyLevel2",
    "materialKeyLevel3", "materialKeyLevel4", "materialKeyLevel_highest",
]


def load_composition(p01, p04, tracker_keyed: dict) -> pd.DataFrame:
    """
    Load the traction-motor composition workbook and extend it to every cohort year
    that appears anywhere in `tracker_keyed`.

    THE MATH MODEL: single-anchor-year, bidirectional extension -- see module docstring
    "STILL OPEN" for why this is a stronger assumption than 04_01's forward-only one.
    """
    input_dir = SCRIPT_DIR / p01.input_dir
    composition_file = input_dir / p04["traction_composition_file_name"]
    composition_extend_from_year = int(p01.composition_extend_from_year)
    material_parameter_code = p04["composition_parameter_code"]

    if not composition_file.exists():
        raise FileNotFoundError(
            f"04_03_tractionmotors.py: expected the traction-motor composition "
            f"workbook at {composition_file}, but it doesn't exist. Set "
            f"params.materials.traction_composition_file_name if your real filename "
            f"differs, or place the file at this path."
        )
    print("Using traction composition file:", composition_file)

    xlsx = pd.ExcelFile(composition_file)
    selected_sheets = ["Consolidated data"]
    columns_to_keep = [
        "productionYear", "productKeyLevel1", "productKeyLevel2", "componentKeyLevel0",
        "componentKeyLevel1", "materialKeyLevel0", "materialKeyLevel1", "materialKeyLevel2",
        "materialKeyLevel3", "materialKeyLevel4", "parameterCode", "value",
    ]

    raw_sheets = {}
    for sheet_name in selected_sheets:
        frame = pd.read_excel(composition_file, sheet_name=sheet_name)
        if frame.empty:
            continue
        frame = frame.rename(columns={col: str(col).strip() for col in frame.columns})
        if "value" not in frame.columns and "meanValue" in frame.columns:
            frame["value"] = frame["meanValue"]  # see module docstring: drops any uncertainty info
        raw_sheets[sheet_name] = frame[columns_to_keep].copy()

    if not raw_sheets:
        raise ValueError(
            f"No usable sheets found in traction composition file after schema "
            f"normalization. Available sheets: {xlsx.sheet_names}"
        )

    composition = pd.concat(raw_sheets.values(), ignore_index=True)
    composition["parameterCode"] = composition["parameterCode"].astype(str).str.strip()
    composition = composition[composition["parameterCode"].eq(material_parameter_code)].copy()

    composition["productionYear"] = pd.to_numeric(composition["productionYear"], errors="coerce").abs().astype("Int64")
    composition = composition[composition["productionYear"].notna()].copy()
    composition["productionYear"] = composition["productionYear"].astype(int)

    cohort_years = sorted({
        int(y) for frame in tracker_keyed.values() for y in frame["cohort_year"].dropna().astype(int).unique().tolist()
    })
    if not cohort_years:
        raise ValueError("No cohort years found in tracker_keyed.")

    if composition_extend_from_year in set(composition["productionYear"].unique()):
        base_year = composition_extend_from_year
    else:
        base_year = int(composition["productionYear"].mode().iloc[0])

    rows_anchor = composition[composition["productionYear"] == base_year].copy()
    if rows_anchor.empty:
        raise ValueError(f"No composition rows available for base_year={base_year}.")

    extended_rows = []
    for year in cohort_years:
        temp = rows_anchor.copy()
        temp["productionYear"] = year
        extended_rows.append(temp)
    composition_extended = pd.concat(extended_rows, ignore_index=True)

    print("Base composition year:", base_year)
    print("Extended to cohort year range:", min(cohort_years), "to", max(cohort_years))

    for col in MATERIAL_LEVELS:
        if col in composition_extended.columns:
            composition_extended[col] = composition_extended[col].astype(str).str.strip()
            composition_extended.loc[composition_extended[col].isin(["", "nan", "None"]), col] = pd.NA
    composition_extended["materialKeyLevel_highest"] = (
        composition_extended[MATERIAL_LEVELS].bfill(axis=1).iloc[:, 0]
    )

    return composition_extended


def quantify_by_key(
    tracker_keyed: dict, composition_extended: pd.DataFrame, p04, *, keep_flow_in_key: bool,
) -> dict:
    """Shared quantify+aggregate step, used both for the single-tracker run and the per-scenario loop."""
    material_parameter_code = p04["composition_parameter_code"]
    material_level_key = p04["material_level_key"]

    quantified = quantify_elements_from_tracker(
        tracker_keyed=tracker_keyed,
        composition_extended=composition_extended,
        parameter_code=material_parameter_code,
        material_level_key=material_level_key,
    )

    mass_by_year_materials_dict = {}
    for key, frame in quantified.items():
        if frame.empty:
            continue
        out = frame.copy()

        if keep_flow_in_key and len(key) == 3:
            region_key, drv_key, flow_key = key
            if "Region" not in out.columns:
                out["Region"] = region_key
            if "Drive Train" not in out.columns:
                out["Drive Train"] = drv_key
            if "flow" not in out.columns:
                out["flow"] = flow_key
            group_cols = ["Region", "Drive Train", "flow", "scrap_year"]
        else:
            group_cols = ["Region", "Drive Train", "scrap_year"]

        keep_cols = [c for c in MATERIAL_LEVEL_COLS if c in out.columns]
        required_cols = {"scrap_year", "mass"}.union(keep_cols).union(set(group_cols))
        missing_cols = required_cols.difference(out.columns)
        if missing_cols:
            raise KeyError(f"Missing columns for material aggregation in {key}: {sorted(missing_cols)}")

        grouped = (
            out.groupby(group_cols + keep_cols, as_index=False, dropna=False)["mass"]
            .sum()
            .sort_values(group_cols + keep_cols)
            .reset_index(drop=True)
        )
        grouped["material"] = grouped[material_level_key].astype(str)
        grouped["element"] = grouped["material"]
        mass_by_year_materials_dict[key] = grouped

    return mass_by_year_materials_dict


def combine_by_flow_drv(mass_by_year_materials_dict: dict, material_level_key: str) -> dict:
    """Combine per-(Region, Drive Train, flow) frames into per-(flow, Drive Train) views."""
    combined: dict = {}
    for (reg, drv, flow), frame in mass_by_year_materials_dict.items():
        if frame.empty:
            continue
        key = (flow, drv)
        combined[key] = pd.concat([combined[key], frame], ignore_index=True) if key in combined else frame.copy()

    for key, frame in combined.items():
        keep_cols = [c for c in MATERIAL_LEVEL_COLS if c in frame.columns]
        combined[key] = (
            frame.groupby(["Region", "Drive Train", "scrap_year"] + keep_cols, as_index=False)["mass"]
            .sum()
            .sort_values(["Region", "Drive Train", "scrap_year"] + keep_cols)
            .reset_index(drop=True)
        )
        combined[key]["material"] = combined[key][material_level_key].astype(str)
        combined[key]["element"] = combined[key]["material"]
    return combined


def build_material_ratio_df(combined_materials_tractionmotor: dict, material_level_key: str, region: str = "EUR") -> pd.DataFrame:
    """
    Same-calendar-year "secondary supply ratio": collected mass in year Y divided by
    new inflow mass in that SAME year Y -- "what fraction of this year's new material
    demand could in principle be met by this year's collected end-of-life material".
    """
    key_in = (region, "inflow")
    key_col = (region, "collected")
    if key_in not in combined_materials_tractionmotor or key_col not in combined_materials_tractionmotor:
        return pd.DataFrame(columns=["scrap_year", material_level_key, "inflow", "collected", "ratio"])

    inflow = (
        combined_materials_tractionmotor[key_in][["scrap_year", material_level_key, "mass"]]
        .rename(columns={"mass": "inflow"}).groupby(["scrap_year", material_level_key], as_index=False)["inflow"].sum()
    )
    collected = (
        combined_materials_tractionmotor[key_col][["scrap_year", material_level_key, "mass"]]
        .rename(columns={"mass": "collected"}).groupby(["scrap_year", material_level_key], as_index=False)["collected"].sum()
    )
    ratio_df = inflow.merge(collected, on=["scrap_year", material_level_key], how="inner")
    ratio_df["ratio"] = ratio_df["collected"] / ratio_df["inflow"].where(ratio_df["inflow"] > 0)
    return ratio_df


def plot_secondary_supply_ratio(ratio_df: pd.DataFrame, material_level_key: str) -> tuple[plt.Figure, plt.Axes]:
    """
    [NEW] Direct visual for build_material_ratio_df's output -- previously computed
    but never plotted. Shows, per material, what fraction of new inflow demand could
    be met by same-year collected end-of-life material.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if ratio_df.empty:
        ax.text(0.5, 0.5, "No overlapping inflow/collected data to compute a ratio", ha="center", va="center")
        return fig, ax

    for material, grp in ratio_df.groupby(material_level_key):
        grp = grp.sort_values("scrap_year")
        ax.plot(grp["scrap_year"], grp["ratio"], linewidth=1.8, label=str(material))

    ax.set_title("Traction motor secondary-supply ratio (collected / inflow, same year)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Ratio")
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
    material_level_key = p04_dict["material_level_key"]
    tracker_keyed = loaded["tracker_keyed"]

    print(artifact_status(root=PROJECT_ROOT))

    composition_extended = load_composition(p01, p04_dict, tracker_keyed)

    # -----------------------------------------------------------------------
    # Single-tracker (baseline) run
    # -----------------------------------------------------------------------
    mass_by_year_materials_dict_tractionmotor = quantify_by_key(
        tracker_keyed, composition_extended, p04_dict, keep_flow_in_key=False
    )
    combined_materials_tractionmotor = combine_by_flow_drv(mass_by_year_materials_dict_tractionmotor, material_level_key)

    artifacts_dir = PROJECT_ROOT / "data" / "processed" / "intermediate"
    saved = save_unregistered_scenario_outputs(artifacts_dir, {
        "04_mass_by_year_materials_dict_tractionmotor.pkl": mass_by_year_materials_dict_tractionmotor,
        "04_combined_materials_tractionmotor.pkl": combined_materials_tractionmotor,
    })

    ratio_df_material_tm = build_material_ratio_df(combined_materials_tractionmotor, material_level_key)

    # -----------------------------------------------------------------------
    # Diagnostic plot: secondary supply ratio -- integrated here, the exact point
    # this data first exists.
    # -----------------------------------------------------------------------
    fig, _ = plot_secondary_supply_ratio(ratio_df_material_tm, material_level_key)
    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "04_03_secondary_supply_ratio.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Per-scenario loop (C6 is fixed -- all 11 are genuinely distinct)
    # -----------------------------------------------------------------------
    saved_scenarios: dict[str, Any] = {}
    for scenario_name in scenario_names:
        tracker = loaded[f"tracker_keyed_{scenario_name}"]
        mass_dict = quantify_by_key(tracker, composition_extended, p04_dict, keep_flow_in_key=True)
        combined_dict = combine_by_flow_drv(mass_dict, material_level_key)
        print(scenario_name, "| mass_dict:", len(mass_dict), "| combined_dict:", len(combined_dict))

        saved_scenarios.update(save_unregistered_scenario_outputs(artifacts_dir, {
            f"04_mass_by_year_materials_dict_tractionmotor_{scenario_name}.pkl": mass_dict,
            f"04_combined_materials_tractionmotor_{scenario_name}.pkl": combined_dict,
        }))

    print("Saved (baseline):", saved)
    print("Saved (per-scenario):", saved_scenarios)
    return {"saved": saved, "saved_scenarios": saved_scenarios, "ratio_df": ratio_df_material_tm}


if __name__ == "__main__":
    main()
