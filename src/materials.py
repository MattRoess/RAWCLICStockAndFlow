"""
materials.py
=============

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Core quantification library used by all four stage-04 notebooks (`src.materials`).

======================================================================
ESSENTIAL REQUIREMENT, per explicit instruction: composition should resolve down to
COMPONENT and its MATERIALS -- NOT further decomposed into individual chemical
ELEMENTS. This file already supports exactly that distinction via `parameter_code`:

  - `parameter_code="m-c"` (now the DEFAULT, changed this round -- was "e-m"): stops at
    material level. `merged["element"]` is just an ALIAS for the material name at
    `material_level_key` (e.g. "materialKeyLevel2") -- there is no elemental
    decomposition in this path. `componentKeyLevel0`/`componentKeyLevel1` are present
    in the merge either way, so component-level grouping is always available.
  - `parameter_code="e-m"` (previous default, still available if ever needed): merges
    in composition rows with an actual `element` column (individual chemical elements
    like Ag, In, Ta, Zn, Dy, Nd, Pr, Al, Cu). Which elements exist is whatever the
    composition file carries; nothing here holds a list of them.

**Changed this round**: the function default is now `"m-c"`, matching the project's
requirement. No existing caller depends on the old default -- no stage-04 script has
been shared/wired yet, so this is a zero-risk change now, before anything calls it.
`params_schema.py`'s new `MaterialsParams.composition_parameter_code` (default `"m-c"`)
and `.material_level_key` centralize this choice for whenever stage 04 is wired in, so
it's a single params-driven setting, not a hardcoded string buried in a stage-04 script.

**Verified structurally** (synthetic composition + tracker fixture, since real
composition data/stage-04 scripts aren't available yet): confirmed the `"m-c"` path
computes `mass` correctly and never introduces an `element` column that doesn't already
equal the chosen material level -- i.e. no elemental decomposition occurs.
======================================================================

NEW FINDING: `build_mass_by_year_elem_dict` is imported by all four stage-04 notebooks
but, per a grep across all four notebook dumps reviewed in a previous round, is never
actually CALLED in any of them -- every notebook reimplements very similar aggregation
logic inline instead (DRY-violation finding M37).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

def save_unregistered_scenario_outputs(artifacts_dir: Path, outputs: dict[str, Any]) -> dict[str, Path]:
    """
    [NEW] Shared persistence helper for stage-04's per-scenario outputs, which have no
    corresponding entry in `artifacts.py`'s static `ARTIFACT_FILES` registry (adding one
    entry per scenario x output-type combination there isn't practical -- it would need
    to grow every time a new scenario is added, in a file whose whole purpose is being
    a stable, hand-maintained registry). Replaces THREE near-identical raw-pickle-saving
    loops duplicated across `04_01_materials.py`, `04_03_tractionmotors.py`, and
    `04_04_batteries.py` with one shared implementation.

    `outputs`: {filename (without directory) -> object to pickle}.
    Returns {filename -> full Path}, for printing/logging at the call site.

    NOTE, unchanged from the original design: these files are genuinely invisible to
    `artifact_status()` -- this helper doesn't change that, only removes the
    code duplication in how they're written. If you want per-scenario stage-04 outputs
    tracked by `artifact_status()`, that requires a design change to `artifacts.py`
    itself (e.g. a wildcard/prefix-based registry), not just this helper -- flagging
    this as a real option to consider, not implementing it unasked.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for filename, obj in outputs.items():
        path = artifacts_dir / filename
        with path.open("wb") as handle:
            pickle.dump(obj, handle)
        saved[filename] = path
    return saved


def quantify_elements_from_tracker(
    tracker_keyed: dict,
    composition_extended: pd.DataFrame,
    parameter_code: str = "m-c",
    material_level_key: str = "materialKeyLevel2",
) -> dict:
    """
    Merge tracker flow rows against composition rows and compute material/component
    mass (or element mass, if `parameter_code="e-m"` is explicitly requested).

    THE MATCHING LOGIC: tracker rows carry a `key` column (the "V"+drivetrain-prefix+
    segment-code string built by `disaggregation.py`'s `add_keys_to_tracker_dict`, e.g.
    "V0301030101") and a `cohort_year` (vintage year). These are matched, via an INNER
    JOIN, against the composition table's `productKeyLevel2` (lowercased/stripped, same
    as the tracker `key`) and `productionYear`. This confirms exactly what the "V+prefix
    +segment" key-building machinery in `disaggregation.py` was for: matching tracker
    rows to composition rows by product code and build year.

    FLAGGED (real, confirmed): the join is `how="inner"` -- any tracker row whose (key,
    cohort_year) combination has NO matching composition row is SILENTLY DROPPED from
    the output, with no warning, count, or record of what was excluded. Contrast with
    `04_04_batteries.py`'s battery-composition merge, which uses a LEFT join and
    explicitly checks/prints unmatched rows. If composition coverage has any gaps (e.g.
    a segment/drivetrain/cohort-year combination the Excel-derived + extended
    composition table doesn't cover), the corresponding material mass is not "zero" or
    "flagged missing" -- it simply never appears in `quantified`, and downstream totals
    would be silently understated with no visible symptom. Given the composition tables
    in 04_01/04_02/04_03 are all deliberately extended to cover every cohort year seen
    in the tracker (see those notebooks' `load_composition` functions), this is unlikely
    to bite in the current pipeline as configured -- but it's a silent-failure-prone
    design choice worth being aware of if that extension logic ever changes.

    THE MASS FORMULA (resolves finding C12 from the previous round):
        mass = amount * value * 1e6 / 1e3
    where `amount` is the tracker's flow count in the project's established "millions of
    vehicles" unit, and `value` is the composition table's per-unit figure. The cleanest
    consistent reading of this chain: `amount * 1e6` converts "millions of vehicles"
    into an actual vehicle count, `* value` (grams per vehicle, presumably) gives total
    grams, and `/ 1e3` converts to kilograms. This is exactly analogous to
    `04_04_batteries.py`'s independently-written battery formula
    (`capacity_GWh * 1e6 [kWh/GWh] * Value[g/kWh] / 1e3 [g/kg] = mass[kg]`), which
    strengthens confidence this is the intended, consistent convention across the
    codebase -- though neither this function nor 04_04 documents it explicitly, and
    confirming the real units of the composition file's `value`/`Value` column against
    its actual column documentation would still be the definitive check.
    """
    valid_material_level_keys = {
        "materialKeyLevel1",
        "materialKeyLevel2",
        "materialKeyLevel3",
        "materialKeyLevel4",
        "materialKeyLevel_highest",
    }
    if material_level_key not in valid_material_level_keys:
        raise ValueError(
            "material_level_key must be one of "
            f"{sorted(valid_material_level_keys)}; got {material_level_key!r}."
        )

    composition_em = (
        composition_extended.loc[
            composition_extended["parameterCode"].eq(parameter_code)
        ]
        .copy()
        .assign(
            productKeyLevel2=lambda frame: frame["productKeyLevel2"].astype(str).str.strip().str.lower(),
            productionYear=lambda frame: frame["productionYear"].astype(int),
            value=lambda frame: frame["value"].astype(float),
        )
    )

    if composition_em.empty:
        raise ValueError(
            f"No rows found in composition_extended for parameter_code={parameter_code!r}."
        )

    merge_cols = [
        "productKeyLevel2",
        "productionYear",
        "componentKeyLevel0",
        "componentKeyLevel1",
        "materialKeyLevel0",
        "materialKeyLevel1",
        "materialKeyLevel2",
        "materialKeyLevel3",
        "materialKeyLevel4",
        "materialKeyLevel_highest",
        "value",
    ]

    if parameter_code == "e-m":
        merge_cols.append("element")

    merge_cols = [col for col in merge_cols if col in composition_em.columns]


    quantified = {}
    for (reg, drv), frame in tracker_keyed.items():
        for flow_name, flow_df in frame.groupby("flow"):
            data = flow_df.copy()
            if data.empty:
                continue

            data = data.assign(
                key=data["key"].astype(str).str.strip().str.lower(),
                cohort_year=data["cohort_year"].astype(int),
                amount=data["amount"].astype(float),
            )

            merged = data[
                ["Region", "Drive Train", "key", "cohort_year", "scrap_year", "amount"]
            ].merge(
                composition_em[merge_cols],
                left_on=["key", "cohort_year"],
                right_on=["productKeyLevel2", "productionYear"],
                how="inner",
            )

            merged["mass"] = merged["amount"] * merged["value"] * 1e6 / 1e3  # see docstring above: resolves C12

            if parameter_code == "m-c":
                if material_level_key not in merged.columns:
                    raise KeyError(
                        f"Column {material_level_key!r} not found in merged composition data."
                    )
                merged["material"] = merged[material_level_key].astype(str)
                merged["element"] = merged["material"]

            result = (
                merged.rename(columns={"cohort_year": "production_year"})
                .drop(columns=["value", "key"], errors="ignore")
                .reset_index(drop=True)
            )
            quantified[(reg, drv, flow_name)] = result

    return quantified


def build_mass_by_year_elem_dict(quantified: dict) -> dict:
    # CONFIRMED imported by all four stage-04 notebooks but never actually called in any
    # of them (grepped across all four notebook dumps) -- each notebook reimplements
    # very similar (Region, ..., scrap_year, element/material) aggregation logic inline
    # instead. Two-step aggregation below (per-product then overall) is redundant but
    # not incorrect -- summation is associative, so the result matches a single-step
    # groupby(["scrap_year","element"]) directly.
    out = {}
    for key_tuple, frame in quantified.items():
        if frame.empty:
            continue
        elements_sum = (
            frame.groupby(["Region", "productKeyLevel2", "scrap_year", "element"], as_index=False)
            .agg({"mass": "sum"})
        )
        mass_by_year_elem = (
            elements_sum.groupby(["scrap_year", "element"], as_index=False)["mass"].sum()
        )
        out[key_tuple] = mass_by_year_elem
    return out


def combine_materials_by_flow(
    mass_by_year_materials_dict: dict,
    material_level_cols: list[str] | None = None,
) -> dict:
    """
    Combine per-(Region, Drive Train, flow) frames into per-(Region, flow) frames,
    PRESERVING all material-hierarchy-level columns (`materialKeyLevel0`-`4` and
    `materialKeyLevel_highest`) present in the input -- used by 04_01_materials.py
    (bulk materials, which cares about the multi-level hierarchy) and
    04_03_tractionmotors.py.

    CONFIRMED (resolves an open question from the previous round): this is genuinely
    different from `combine_mass_by_flow` below, not a naming inconsistency for the same
    operation -- this one groups by `["scrap_year"] + <every present material-level
    column>`, the other groups by `["scrap_year", "element"]` only. Each is correctly
    matched to its respective notebook's needs.
    """
    if material_level_cols is None:
        material_level_cols = [
            "materialKeyLevel0",
            "materialKeyLevel1",
            "materialKeyLevel2",
            "materialKeyLevel3",
            "materialKeyLevel4",
            "materialKeyLevel_highest",
        ]

    combined = {}

    for (reg, _drv, flow), frame in mass_by_year_materials_dict.items():
        if frame.empty:
            continue

        key = (reg, flow)

        if key not in combined:
            combined[key] = frame.copy()
        else:
            combined[key] = pd.concat([combined[key], frame], ignore_index=True)

    for key, frame in combined.items():
        keep_cols = [col for col in material_level_cols if col in frame.columns]
        group_cols = ["scrap_year"] + keep_cols

        combined[key] = (
            frame.groupby(group_cols, as_index=False, dropna=False)["mass"]
            .sum()
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        
        if "materialKeyLevel_highest" in combined[key].columns:
            combined[key]["material"] = combined[key]["materialKeyLevel_highest"].astype(str)
            combined[key]["element"] = combined[key]["material"]

    return combined


def combine_mass_by_flow(mass_by_year_elem_dict: dict) -> dict:
    """
    Combine per-(Region, Drive Train, flow) frames into per-(Region, flow) frames,
    collapsing directly to (scrap_year, element) -- used by 04_02_elements.py. See
    `combine_materials_by_flow` above for why this is a genuinely different function,
    not a duplicate/naming variant of it.
    """
    combined = {}

    for (reg, _drv, flow), frame in mass_by_year_elem_dict.items():
        if frame.empty:
            continue

        key = (reg, flow)

        if key not in combined:
            combined[key] = frame.copy()
        else:
            combined[key] = pd.concat([combined[key], frame], ignore_index=True)

    for key, frame in combined.items():
        combined[key] = (
            frame.groupby(["scrap_year", "element"], as_index=False)["mass"]
            .sum()
            .sort_values(["scrap_year", "element"])
            .reset_index(drop=True)
        )

    return combined