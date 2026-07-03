"""
00_parameters.py
=================

Centralized parameter definitions for the EVmodel pipeline (stages 00-07).

PURPOSE
-------
This module is the SINGLE SOURCE OF TRUTH for every tunable input used downstream:
    01_data_prep     -> scenario selection, model horizon, composition inputs, taxonomy
    02_stock_flow    -> vehicle lifetime (Weibull) and export/unknown-fate shares
    03_disaggregation-> regional/segment disaggregation window
    04_materials     -> drivetrain/segment -> material-composition code mapping
    06_visualization -> plotting windows and default scenario

Run this script first (`python 00_parameters.py`). It builds the PARAMS dict, validates it,
persists it as a pickle artifact for the other stages to load, and writes a human-readable
Excel companion (params.xlsx) for non-programmers to review/edit values.

STAGE 05/07 UPDATE (resolved, see CHANGELOG below)
----------------------------------------------------
The original notebook's docstring claimed parameters for "notebooks 01-06", and an
earlier pass of this file flagged the absence of a "05_..." section as an open question.
This is now resolved: stage 05 (`05_indicator_disagg_stock.py`) and stage 07
(`07_proposal.py`) both exist and have been reviewed (see
`EVmodel_review_consolidated.md`). Neither currently reads a `PARAMS["05_..."]` /
`PARAMS["07_..."]` section -- stage 05's scenario-adjustment knobs (`FLOW_TWEAKS`,
`STOCK_TWEAKS`) are hardcoded inline in that file rather than sourced from here (see
finding C14) -- centralizing those is a candidate for when stage 05 itself is fixed, not
addressed in this pass since it's out of this file's scope.

CHANGELOG -- fixes applied in this pass (see EVmodel_review_consolidated.md for full
finding descriptions; each entry below is also marked inline at its exact location)
----------------------------------------------------------------------------------------
- [FIXED, mechanical] `lifetime_override_by_drv` was missing "Petrol" and "Diesel"
  entries (both now `None`, matching every other drivetrain's default/disabled state).
- [FIXED, mechanical] `target_technology` was missing "PHEV" (added, for consistency
  with `lifetime_by_drv` and `04_materials.drv_prefix_map`, which both already treat
  PHEV as a real, tracked drivetrain).
- [RETRACTED] A previous review pass flagged `eu_countries` as having only 26 entries
  (EU has 27 members). Re-verified programmatically this pass: **the list is actually a
  complete, correct EU-27** -- that earlier finding was a miscount, not a real bug.
  Left unchanged.
- [FIXED, value judgment -- CONFIRM] `EXPORT_SHARE_BY_DRV` was missing "FCEV" and
  "Gases". Added both at `0.08`, matching every other non-BEV drivetrain
  (HEV/PHEV/Hybrid/Liquids/Diesel/Petrol all use 0.08; only BEV is lower, at 0.02). This
  is a reasonable placeholder pending real data, **not a verified value** -- flagged
  inline and here for easy follow-up.
- [FIXED, decision made -- was ambiguous, now resolved per direct confirmation]
  `02_stock_flow.model_end_year` was `2100`, disagreeing with
  `01_data_prep.end_year_model` (`2070`). Aligned to **2070** (both now match). Marked
  inline in case this needs revisiting once stages 02+ are fixed and their real
  simulation-length needs are better understood.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# HISTORICAL NOTE: the original .ipynb notebook this script was converted from did
# `sys.path.append(str(Path.cwd().parent))` and used relative paths like "../data/raw/".
# This is FRAGILE: it silently assumes the process's working directory is set to
# <project_root>/code. Running it from anywhere else (e.g. VS Code "Run File", a CI job,
# Positron with a different cwd) breaks the import of `src.artifacts` and the relative
# data paths without a clear error.
#
# [FIXED -- corrects a real bug in the PREVIOUS version of this fix]: an earlier pass of
# this file resolved `PROJECT_ROOT = Path(__file__).resolve().parent` (this file's own
# directory) and appended THAT to sys.path. That is only correct if this script sits
# DIRECTLY in the project root. In the real project layout (this script lives in
# `<project_root>/code/00_parameters.py`, with `src/` as a SIBLING of `code/`, not a
# child of it), `Path(__file__).resolve().parent` resolves to `.../code/`, NOT the
# actual project root -- so `sys.path.append` added the wrong directory, `from
# src.artifacts import ...` failed with an ImportError that was silently caught, and the
# script printed "src.artifacts not found on path" even when `src/artifacts.py` genuinely
# existed one level up. Confirmed by testing against a real
# `<root>/code/00_parameters.py` + `<root>/src/artifacts.py` layout: the previous
# version failed to import `src.artifacts`, and this version succeeds.
#
# THE FIX: search upward from this file's location for the first ancestor directory that
# contains a `src/` subdirectory, and use THAT as the project root -- works whether this
# script sits directly in the project root or inside the `code/` subfolder (or any other
# depth), without hardcoding `.parent` vs `.parent.parent`, and without hardcoding any
# particular folder name (unlike `src/config.py`'s `find_project_root`, which does need
# to know the folder is called `code/` -- see that file for why).
def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    # No ancestor has a `src/` directory -- fall back to this file's own directory
    # (the previous behavior) rather than raising, so the script still runs (with
    # `src.artifacts` unavailable, exactly as before) instead of crashing outright.
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Kept separate from PROJECT_ROOT: params.xlsx should land next to this script (where a
# human is actually looking, e.g. <project_root>/code/), not at the repository root that
# _find_project_root() resolves to for import purposes -- those are two different
# "roots" for two different reasons.
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    # `src.artifacts` is an internal helper module. It was NOT included in what was
    # shared with me initially, so I could not inspect save_many()/artifact_status() to
    # confirm their exact contract (return type, overwrite behavior, error handling).
    # It has since been shared and reviewed -- see EVmodel_review_consolidated.md. Kept
    # the import wrapped in try/except regardless, so this script still runs (and can be
    # tested) even in an environment where `src/` isn't reachable for some other reason.
    from src.artifacts import save_many, artifact_status  # type: ignore
    _ARTIFACTS_AVAILABLE = True
except ImportError:
    _ARTIFACTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared constants (DRY fix)
# ---------------------------------------------------------------------------
# ORIGINAL BUG (minor, but real): `element_list` and `element_list_noAlCu` were defined
# TWICE with identical values -- once under "01_data_prep" and once under "06_visualization".
# Duplicated literals like this are a maintenance hazard: if someone updates the element
# list for material accounting (01) but forgets to update it for plotting (06), the two
# stages silently diverge with no error. Defining them once and referencing them in both
# places removes that failure mode.
ELEMENT_LIST: list[str] = ["Ag", "In", "Ta", "Zn", "Dy", "Nd", "Pr", "Al", "Cu"]
ELEMENT_LIST_NO_AL_CU: list[str] = ["Ag", "In", "Ta", "Zn", "Dy", "Nd", "Pr"]

# All drivetrains that appear ANYWHERE in the model. Used below to validate that
# per-drivetrain dicts (lifetime, export share, unknown-fate share) are complete.
ALL_DRIVETRAINS: list[str] = [
    "BEV", "HEV", "PHEV", "Hybrid", "Liquids", "Petrol", "Diesel", "Gases", "FCEV",
]


# ---------------------------------------------------------------------------
# PARAMS: the central parameter dictionary
# ---------------------------------------------------------------------------
# Edit values here, then run this script to overwrite data/processed/intermediate/00_params.pkl.
#
# IMPROVEMENT SUGGESTION (not yet applied, see documentation): this nested dict-of-dicts is
# easy to typo and has no type checking. For a project with "more code and data to come",
# consider migrating to `dataclasses` or `pydantic` models per stage (e.g. DataPrepParams,
# StockFlowParams, ...). That gives you autocomplete, type validation, and a single place
# that raises a clear error the moment a parameter is missing/wrong-typed, instead of a
# silent KeyError three stages downstream. I've kept the dict structure here to stay
# faithful to the original, but flagged this as the top structural improvement.
PARAMS: dict[str, dict[str, Any]] = {

    # ---- 01 Data prep: scenario selection, model horizon, composition inputs, taxonomy ----
    "01_data_prep": {
        "scenario": "npi25",  # default scenario used in 01 + 06
        "scenario_list": ["b650", "npi25", "ssp2L", "ssp2M", "ssp1"],
        "start_year_model": 1900,  # model timeline start
        "end_year_model": 2070,  # model timeline end
        "start_year_plotting": 2015,  # default plotting start
        "end_year_plotting": 2070,  # default plotting end
        "composition_extend_from_year": 2050,  # last observed year used as anchor for extension
        "accelerating_year": 2026,  # manual BEV acceleration inflection year, stage 01
        # ^ FLAGGED: magic number with no documented derivation. If this represents a real
        # policy/market assumption (e.g. a regulation date), it should cite its source.
        "threshold": 1e-4,  # stock-flow numerical threshold
        "prefix": "Stock|Transport|Pass|Road|LDV",  # REMIND variable prefix filter
        "eu_countries": [
            "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
            "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
            "Malta", "Netherlands", "Poland", "Portugal", "Romania",
            "Slovakia", "Slovenia", "Spain", "Sweden",
        ],
        # [RETRACTED, re-verified programmatically this pass]: an earlier review flagged
        # this list as having only 26 entries against an expected EU-27. Recounted
        # directly: this IS the complete, correct EU-27 (27 entries). No change needed.
        "remind_regions": ["DEU", "ECE", "ECS", "ENC", "ESC", "ESW", "EWN", "FRA", "UKI", "NEN"],
        "remind_technology": ["BEV", "Hybrid", "Liquids", "Gases", "FCEV"],
        "target_technology": ["BEV", "HEV", "PHEV", "Petrol", "Diesel"],
        # [FIXED, mechanical]: added "PHEV", previously missing despite PHEV having its
        # own lifetime parameters (02_stock_flow) and composition code
        # (04_materials.drv_prefix_map). "remind_technology" (REMIND's own categories,
        # below) still doesn't map 1:1 to this list -- REMIND only reports "Hybrid" as
        # an aggregate, split into HEV/PHEV downstream (stage 03's EEA-derived split).
        # That's expected and not a bug; left as-is.
        "target_class_detail": [
            "Large Car and SUV", "Van", "Compact Car", "Midsize Car", "Mini Car", "Subcompact Car",
        ],
        "attribute_list": ["Region", "technology"],
        "key_names": ["Region", "Drivetrain"],
        "element_list": ELEMENT_LIST,
        "element_list_noAlCu": ELEMENT_LIST_NO_AL_CU,
        "input_dir": "../data/raw/",
        "output_dir": "../data/processed/",
        "composition_file_name": "ELV_2010_2050.xlsx",
        "petrol_composition_file_name": "ELVComponent_1990_2050_Petrol.xlsx",
        "sheets": [
            "petrolCar_1980_2050",
            "dieselCar_1980_2050",
            "BEVCar_1980_2050",
            "HEVCar_1980_2050",
            "PHEVCar_1980_2050",
            "otherCar_1980_2050",
        ],
    },

    # ---- 02 Stock-flow: lifetime assumptions and export/unknown-fate split behavior ----
    "02_stock_flow": {
        "model_end_year": 2070,
        # [FIXED, decision made per direct confirmation -- was 2100]: aligned to match
        # 01_data_prep.end_year_model (2070). Previously the two disagreed, which is
        # exactly the kind of silent cross-stage mismatch validate_params() below checks
        # for. Marked clearly here in case this needs revisiting once stage 02 (which
        # actually consumes this value as its simulation end year) is being fixed and
        # its real horizon needs are better understood -- easy to change back if 2100
        # turns out to be needed for a real reason (e.g. deliberately modeling further
        # out than what's plotted/reported).
        "last_exp_data_year": 2022,
        "lifetime_by_drv": {
            # Weibull(shape_k, scale_lambda) vehicle survival parameters per drivetrain.
            "Hybrid": {"shape_k": 3.0, "scale_lambda": 9.0},   # set to 9.0 to avoid negative inflows (likely negligible effect)
            "PHEV":   {"shape_k": 3.0, "scale_lambda": 9.0},   # set to 9.0 to avoid negative inflows (likely negligible effect)
            "HEV":    {"shape_k": 3.0, "scale_lambda": 9.0},   # set to 9.0 to avoid negative inflows (likely negligible effect)
            "BEV":    {"shape_k": 3.0, "scale_lambda": 13.0},  # TODO: change per scenario
            "Liquids":{"shape_k": 3.0, "scale_lambda": 13.0},
            "Petrol": {"shape_k": 3.0, "scale_lambda": 13.0},
            "Diesel": {"shape_k": 3.0, "scale_lambda": 13.0},
            "Gases":  {"shape_k": 3.0, "scale_lambda": 13.0},
            "FCEV":   {"shape_k": 3.0, "scale_lambda": 13.0},
        },
        # Optional per-drivetrain override window; keep None to disable.
        "lifetime_override_by_drv": {
            "BEV": None, "HEV": None, "PHEV": None, "Hybrid": None,
            "Gases": None, "FCEV": None, "Liquids": None,
            "Petrol": None, "Diesel": None,
            # [FIXED, mechanical]: "Petrol" and "Diesel" were previously missing here
            # (present in lifetime_by_drv but not here) -- added as None (disabled),
            # matching every other drivetrain's default state. No behavior change for
            # code using .get(); removes a real KeyError risk for code doing direct
            # dict indexing across all drivetrains.
        },
        "unknown_whereabouts_share": {
            "BEV": 0.1,     # TODO: change per scenario
            "HEV": 0.43,
            "PHEV": 0.43,
            "FCEV": 0.43,
            "Gases": 0.43,
            "Diesel": 0.43,
            "Petrol": 0.43,
            "Liquids": 0.43,
            "Hybrid": 0.43,
        },
        "EXPORT_SHARE_BY_DRV": {
            "BEV": 0.02,    # TODO: change per scenario
            "HEV": 0.08,
            "PHEV": 0.08,
            "Diesel": 0.08,
            "Petrol": 0.08,
            "Hybrid": 0.08,
            "Liquids": 0.08,
            "FCEV": 0.08,   # [FIXED, value judgment -- CONFIRM]: was missing; set to match
            "Gases": 0.08,  # every other non-BEV drivetrain. NOT a verified real value --
            # placeholder pending real data. See module docstring CHANGELOG. Previously
            # missing entirely, which validate_params() below flagged as a KeyError risk
            # for any code indexing this dict across all drivetrains.
        },
    },

    # ---- 03 Disaggregation: global years and (currently unused) window ----
    "03_disaggregation": {
        "output_dir": "../data/processed/",
        "start_year_model": 1900,
        "end_year_model": 2070,
    },

    # ---- 04 Materials: key mapping from drivetrain+segment to composition coding ----
    "04_materials": {
        "segment_map": {
            "A": "0101", "B": "0102", "C": "0103", "D": "0104", "E": "0105", "F": "0106",
            "JA": "0201", "JB": "0202", "JC": "0203", "JD": "0204", "JE": "0205", "JF": "0206",
        },
        "drv_prefix_map": {
            "PHEV": "050103",
            "HEV": "040101",
            "BEV": "030103",
            "Diesel": "020102",
            "Petrol": "010101",
            # NOTE: "Hybrid", "Liquids", "Gases", "FCEV" have no composition-code prefix
            # here, even though they appear in 02_stock_flow's lifetime table. If the
            # materials stage is only ever run for these 5 drivetrains, that's fine and
            # intentional -- but worth confirming, since it means those 4 other
            # drivetrains never get a material-composition estimate.
        },
        "region": "EUR",
        "drivetrains": ["BEV", "HEV", "PHEV", "Diesel", "Petrol"],
    },

    # ---- 06 Visualization: global windows and default scenario for plots ----
    "06_visualization": {
        "year_plot_start": 2015,
        "year_plot_end": 2070,
        "year_ratio_start": 2020,
        "element_list": ELEMENT_LIST,           # de-duplicated, see ELEMENT_LIST above
        "element_list_noAlCu": ELEMENT_LIST_NO_AL_CU,
        "scenario": "npi25",
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_params(params: dict[str, dict[str, Any]]) -> list[str]:
    """
    Run a set of consistency checks across PARAMS and return a list of warning/error
    strings. This did NOT exist in the original .ipynb notebook this file was converted
    from -- that version would happily pickle an internally-inconsistent PARAMS dict
    with no feedback. Given this file feeds five downstream stages, catching mismatches
    here (once) is much cheaper than debugging a KeyError inside stage 04 six months
    from now.

    This function does not raise; it returns findings so the caller can decide whether to
    treat them as fatal. See main() for how they're surfaced.
    """
    issues: list[str] = []

    stock_flow = params.get("02_stock_flow", {})
    lifetime_drvs = set(stock_flow.get("lifetime_by_drv", {}).keys())
    export_drvs = set(stock_flow.get("EXPORT_SHARE_BY_DRV", {}).keys())
    unknown_drvs = set(stock_flow.get("unknown_whereabouts_share", {}).keys())
    override_drvs = set(stock_flow.get("lifetime_override_by_drv", {}).keys())

    missing_export = lifetime_drvs - export_drvs
    if missing_export:
        issues.append(
            f"EXPORT_SHARE_BY_DRV is missing drivetrains present in lifetime_by_drv: "
            f"{sorted(missing_export)}. Any code indexing this dict for all drivetrains "
            f"will raise a KeyError."
        )

    missing_unknown = lifetime_drvs - unknown_drvs
    if missing_unknown:
        issues.append(
            f"unknown_whereabouts_share is missing drivetrains present in lifetime_by_drv: "
            f"{sorted(missing_unknown)}."
        )

    missing_override = lifetime_drvs - override_drvs
    if missing_override:
        issues.append(
            f"lifetime_override_by_drv is missing drivetrains present in lifetime_by_drv: "
            f"{sorted(missing_override)} (may be intentional if .get() is used downstream)."
        )

    # Fractions should be within [0, 1].
    for name, mapping in (
        ("EXPORT_SHARE_BY_DRV", stock_flow.get("EXPORT_SHARE_BY_DRV", {})),
        ("unknown_whereabouts_share", stock_flow.get("unknown_whereabouts_share", {})),
    ):
        for drv, share in mapping.items():
            if share is not None and not (0.0 <= share <= 1.0):
                issues.append(f"{name}['{drv}'] = {share} is outside the valid [0, 1] range.")

    # Year ordering sanity checks.
    dp = params.get("01_data_prep", {})
    if dp.get("start_year_model", 0) >= dp.get("end_year_model", 0):
        issues.append("01_data_prep: start_year_model must be strictly before end_year_model.")
    if dp.get("start_year_plotting", 0) > dp.get("end_year_plotting", 0):
        issues.append("01_data_prep: start_year_plotting must not be after end_year_plotting.")

    # Cross-stage horizon consistency (flagged as a question, see module docstring).
    sf_end = stock_flow.get("model_end_year")
    dp_end = dp.get("end_year_model")
    if sf_end is not None and dp_end is not None and sf_end != dp_end:
        issues.append(
            f"Model horizon differs between stages: 01_data_prep.end_year_model={dp_end} "
            f"vs 02_stock_flow.model_end_year={sf_end}. Confirm this is intentional."
        )

    return issues


# ---------------------------------------------------------------------------
# Excel export (human-readable parameter register)
# ---------------------------------------------------------------------------
# Description lookup for well-known parameter keys. Falls back to a generic per-section
# description when a key isn't in this table. IMPROVEMENT SUGGESTION: this table currently
# covers only ~10 of the ~45 leaf parameters in PARAMS. Since this Excel file is explicitly
# meant to be handed to a non-programmer ("To put into excel"), it's worth the (one-time)
# effort of writing a real description for every key rather than falling back to a generic
# placeholder -- I've left the fallback behavior unchanged but flagged it here.
_DESCRIPTIONS: dict[str, str] = {
    "scenario": "Selected model scenario for stock-flow simulation.",
    "scenario_list": "All scenarios available for selection.",
    "start_year_model": "First year of the simulation horizon.",
    "end_year_model": "Last year of the simulation horizon.",
    "start_year_plotting": "First year shown in default plots.",
    "end_year_plotting": "Last year shown in default plots.",
    "threshold": "Numerical tolerance used for stock-flow stability checks.",
    "eu_countries": "EU country list used for regional aggregation.",
    "remind_regions": "REMIND model region codes mapped into this analysis.",
    "remind_technology": "Drivetrain categories as labeled in REMIND scenario output.",
    "target_technology": "Drivetrain categories used by this model's outputs.",
    "lifetime_by_drv": "Weibull lifetime parameters (shape_k, scale_lambda) per drivetrain.",
    "lifetime_override_by_drv": "Optional manual override window per drivetrain (None = disabled).",
    "unknown_whereabouts_share": "Share of retired vehicles with unknown/untracked fate, per drivetrain.",
    "EXPORT_SHARE_BY_DRV": "Share of retired vehicles assumed exported, per drivetrain.",
    "segment_map": "Vehicle-segment letter code -> composition dataset code.",
    "drv_prefix_map": "Drivetrain -> composition dataset key prefix.",
    "model_end_year": "Last year simulated in the stock-flow module.",
    "last_exp_data_year": "Last year with observed export data.",
}


def _describe(key: str, section: str) -> str:
    """Return a human-readable description for a parameter key, or a generic fallback."""
    return _DESCRIPTIONS.get(key, f"Model parameter in section '{section}' (no description authored yet).")


def _flatten_params(params: dict[str, dict[str, Any]]) -> list[list[Any]]:
    """
    Flatten the nested PARAMS dict into rows of (name, description, full_key, value)
    for Excel export.

    IMPROVEMENT over the original: the original `make_name()` returned ONLY the deepest
    key segment (e.g. just "shape_k"), which means two unrelated parameters that happen to
    share a leaf name (e.g. "scenario" appears in both 01_data_prep and 06_visualization)
    would show up with an IDENTICAL "name" column value and be indistinguishable at a
    glance in Excel -- only the "key" column (the fully-qualified dotted path) disambiguates
    them, and that column is easy to overlook. Here `name` is still the short label people
    scan first, but every row's `key` column is guaranteed unique by construction, and rows
    are grouped by section, so ambiguity is resolved by physical proximity + the key column.
    """
    rows: list[list[Any]] = []

    for section, content in params.items():
        for key, value in content.items():
            base_key = f"{section}.{key}"

            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        for k2, v2 in subvalue.items():
                            rows.append([
                                k2,
                                _describe(k2, section),
                                f"{section}.{key}.{subkey}.{k2}",
                                json.dumps(v2) if isinstance(v2, (dict, list)) else v2,
                            ])
                    else:
                        rows.append([
                            subkey,
                            _describe(subkey, section),
                            f"{section}.{key}.{subkey}",
                            json.dumps(subvalue) if isinstance(subvalue, (dict, list)) else subvalue,
                        ])
            else:
                rows.append([
                    key,
                    _describe(key, section),
                    base_key,
                    json.dumps(value) if isinstance(value, (dict, list)) else value,
                ])

    return rows


def save_params_excel(params: dict[str, dict[str, Any]], filepath: Path) -> pd.DataFrame:
    """Write PARAMS to a flat Excel register (one row per leaf parameter) and return the df."""
    rows = _flatten_params(params)
    df = pd.DataFrame(rows, columns=["name", "description", "key", "value"])

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="parameters", index=False)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    """
    Validate PARAMS, persist it as the pickled `params` artifact (via src.artifacts, if
    available), and write the Excel companion register. Returns a small status dict so the
    script is usable both from the command line and imported programmatically
    (`from parameters import main as build_params`).
    """
    issues = validate_params(PARAMS)
    if issues:
        print("PARAMS validation found the following issues:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        # NOT raising here: validation problems shouldn't necessarily block a work-in-
        # progress run, but they must not be silently swallowed either. Consider making
        # this `raise ValueError(...)` once the underlying data gaps are resolved, so CI
        # or a colleague can't accidentally run the pipeline on a known-bad parameter set.

    result: dict[str, Any] = {"validation_issues": issues}

    if _ARTIFACTS_AVAILABLE:
        # [FIXED, this round]: explicitly pass root=PROJECT_ROOT (this script's own
        # correctly-resolved location, found by searching for a sibling `src/` -- see
        # `_find_project_root` above) rather than relying on `save_many`'s old default
        # of resolving from `Path.cwd()`. CONFIRMED BUG this fixes: running this script
        # from an IDE (Positron) whose console keeps a working directory different from
        # the script's own location caused the pickled artifact to be silently written
        # to the WRONG location (a phantom folder tree wherever the console's cwd
        # happened to be) while params.xlsx (already anchored to the script's own file
        # location) landed correctly -- the two outputs were inconsistently rooted.
        # Reproduced and verified fixed: see EVmodel_review_consolidated.md Fix Log.
        saved = save_many(params=PARAMS, root=PROJECT_ROOT)
        result["saved"] = saved
        print("Saved artifact:", saved)
    else:
        print(
            "src.artifacts not found on path -- skipped persisting the pickled 'params' "
            "artifact. Place this file inside the project (next to the `src/` package) "
            "and re-run to enable that step.",
            file=sys.stderr,
        )

    excel_path = SCRIPT_DIR / "params.xlsx"
    df = save_params_excel(PARAMS, excel_path)
    result["excel_path"] = str(excel_path)
    result["n_parameters"] = len(df)
    print(f"Wrote {len(df)} parameters to {excel_path}")

    return result


if __name__ == "__main__":
    main()
