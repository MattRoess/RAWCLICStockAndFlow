"""
04_03_tractionmotors.py
=======================

Stage 04, part 3: traction-motor material and element flows.

    .venv/bin/python code/04_03_tractionmotors.py

REWRITTEN 2026-09-21, and not edited -- the previous version was wrong in ways
that could not be patched out. It summed across `voltageClass`, adding the
400 V, 800 V and 1000 V variants of one car together and trebling the mass; it
summed across motor type, collapsing five drive configurations into one; it
dropped `materialClass`, so the two newer machines -- which carry their material
name in that column and not in the house keys -- arrived with no material at
all, 32% of rows and 12.8% of mass; and it took ONE anchor year and replicated
it across every cohort, an assumption its own docstring called stronger than
04_01's. The composition file now carries 61 years, so none of that is needed.

WHAT IT PRODUCES, and Matthias asked for both:

  PARALLEL     material mass per (motor type, voltage class), unpartitioned --
               15 states, each answering "if every BEV were this". Nothing is
               mixed, so nothing has to be believed about market shares to read
               it.
  COMBINED     one fleet total, the 15 states weighted by the shares in
               `src/traction.py`. This is the answer to "how much copper and
               how much neodymium", and it depends on every share being right.

  ELEMENTS     Nd, Pr, Dy and Tb inside the magnet, for both -- the element
               layer arrived in the composition file on 2026-09-21 and closes
               what HANDOVER §7.2 of the traction project called its biggest
               gap.

⚠️ MEAN ONLY, FOR NOW, AND IT IS A CHOICE. The composition file carries p025,
p975 and STD from a 200,000-draw bootstrap, and this stage uses `meanValue` and
ignores them. Matthias 2026-09-21: mean only for now. The bands are not lost --
they are in the file and in the draw arrays beside it -- but nothing downstream
of here carries uncertainty until this stage draws instead of multiplying.

⚠️ THE GRADE SCENARIOS ARE NOT AVERAGED. The magnet element sheet carries SH, UH
and EH, and they are three answers to one question rather than a distribution
over it. Every element output is per scenario, and the base is whichever the
composition file marks `is_base`.
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
import src.traction as traction  # type: ignore

save_unregistered_scenario_outputs = materials.save_unregistered_scenario_outputs

# The dimensions that must survive to the parallel output. Losing any of them is
# the defect this rewrite exists to fix.
KEEP = [
    "productKeyLevel2", "productKeyLevel3", "productionYear",
    "componentKeyLevel1", "componentKeyLevel2",
    "materialClass", "materialKeyLevel1", "voltageClass", "torque_nm",
    "parameterCode", "meanValue",
]


def load_composition(p04: dict) -> pd.DataFrame:
    """
    The traction composition, read where it lies, with every dimension kept.

    ⚠️ NO ANCHOR YEAR AND NO REPLICATION. The file has one row per segment,
    motor type, voltage class and YEAR -- 62,403 of them, 61 years -- so the
    year is read, not manufactured. The previous version picked a single year
    and copied it forwards and backwards over the whole cohort range.
    """
    directory = Path(p04["traction_composition_dir"])
    path = directory / p04["traction_composition_file_name"]
    if not path.exists():
        raise FileNotFoundError(
            f"04_03_tractionmotors.py: expected the traction-motor composition "
            f"workbook at {path}, but it doesn't exist. That folder belongs to "
            f"RAWCLICVehicleTractionMotor and is written by its "
            f"01_composition.py -- run that project first, or set "
            f"params.materials.traction_composition_dir if it has moved. Do NOT "
            f"copy the workbook into this project's data/raw."
        )
    print("Traction composition:", path)

    frame = pd.read_excel(path, sheet_name="Consolidated data")
    frame = frame.rename(columns={c: str(c).strip() for c in frame.columns})
    missing = [c for c in KEEP if c not in frame.columns]
    if missing:
        raise KeyError(
            f"the composition file is missing {missing}. This stage needs the "
            f"voltage class, the motor type and materialClass -- summing over "
            f"any of them is what the 2026-09-21 rewrite exists to stop."
        )
    frame = frame[KEEP].copy()

    code = p04["composition_parameter_code"]
    frame["parameterCode"] = frame["parameterCode"].astype(str).str.strip()
    frame = frame[frame["parameterCode"].eq(code)].copy()
    if frame.empty:
        raise ValueError(f"no rows with parameterCode == {code!r}")

    frame["productionYear"] = pd.to_numeric(
        frame["productionYear"], errors="coerce").abs().astype("Int64")
    frame = frame[frame["productionYear"].notna()].copy()
    frame["productionYear"] = frame["productionYear"].astype(int)
    frame["voltageClass"] = pd.to_numeric(frame["voltageClass"]).astype(int)
    frame["meanValue"] = frame["meanValue"].astype(float)
    frame["productKeyLevel2"] = (frame["productKeyLevel2"].astype(str)
                                 .str.strip().str.lower())
    return frame


def load_magnet_elements(p04: dict) -> pd.DataFrame:
    """
    The `e-m` rows: each element as a share of the magnet, per grade scenario.

    Absent from older composition files, so a missing sheet is a warning and
    not a failure -- the material flows still stand without it.
    """
    path = (Path(p04["traction_composition_dir"])
            / p04["traction_composition_file_name"])
    try:
        frame = pd.read_excel(path, sheet_name="Magnet elements")
    except ValueError:
        print("  no 'Magnet elements' sheet -- element flows skipped. Re-run "
              "the traction project to produce it.")
        return pd.DataFrame()
    frame = frame.rename(columns={c: str(c).strip() for c in frame.columns})
    return frame


def quantify_parallel(tracker_keyed: dict, composition: pd.DataFrame) -> dict:
    """
    Material mass per (motor type, voltage class), with nothing mixed together.

    One frame per (region, drive train, flow). Every row answers "if every BEV
    of this segment and cohort had this motor type at this voltage, how many
    kilograms of this material would flow" -- so the 15 states stand side by
    side and none of them has been believed yet.

    ⚠️ THE JOIN IS THE SAME ONE `materials.quantify_elements_from_tracker` MAKES
    -- tracker `key` and `cohort_year` against `productKeyLevel2` and
    `productionYear` -- but it is done here because that helper drops
    `voltageClass` and `materialClass` on the way through, and adding them to a
    shared helper would change 04_01 and 04_04 as well.

    ⚠️ AND IT IS A LEFT JOIN THAT REPORTS WHAT DID NOT MATCH. The shared helper
    uses an inner join, which silently drops any tracker row with no composition
    -- flagged in `materials.py` as a silent-failure design. Here the unmatched
    rows are counted and printed.
    """
    mass_formula_note = "amount [millions] * 1e6 * value [kg per vehicle]"
    out, unmatched_total, matched_total = {}, 0, 0

    for (region, drive), frame in tracker_keyed.items():
        for flow_name, flow in frame.groupby("flow"):
            if flow.empty:
                continue
            data = flow[["Region", "Drive Train", "key", "cohort_year",
                         "scrap_year", "amount"]].copy()
            data["key"] = data["key"].astype(str).str.strip().str.lower()
            data["cohort_year"] = data["cohort_year"].astype(int)
            data["amount"] = data["amount"].astype(float)

            merged = data.merge(
                composition, how="left",
                left_on=["key", "cohort_year"],
                right_on=["productKeyLevel2", "productionYear"])

            missing = merged["meanValue"].isna()
            unmatched_total += int(missing.sum())
            matched_total += int((~missing).sum())
            merged = merged[~missing].copy()
            if merged.empty:
                continue

            # kg: the project's "millions of vehicles" unit times kg per vehicle
            merged["mass"] = merged["amount"] * 1e6 * merged["meanValue"]
            merged = merged.rename(columns={"cohort_year": "production_year"})
            out[(region, drive, flow_name)] = merged.drop(
                columns=["key", "amount", "meanValue", "parameterCode"],
                errors="ignore").reset_index(drop=True)

    if unmatched_total:
        share = unmatched_total / max(1, unmatched_total + matched_total)
        print(f"  ⚠️  {unmatched_total:,} tracker rows ({share:.1%}) had no "
              f"composition and were dropped -- reported, not silent")
    print(f"  parallel: {len(out)} (region, drive train, flow) frames, "
          f"{mass_formula_note}")
    return out


def combine_partitioned(parallel: dict, params) -> dict:
    """
    The 15 states weighted into one fleet total, by segment group and year.

    ⚠️ THIS IS WHERE THE SHARES ARE BELIEVED. Everything above this line is
    arithmetic on the composition file; everything below depends on
    `src/traction.py` being right about how many cars are of each type at each
    voltage. That is why the parallel output is kept and written too: when a
    number here looks wrong, the parallel output says whether the composition or
    the share is responsible.

    The weight is the joint probability of (motor type, voltage class) for that
    car's segment group and COHORT YEAR -- the year it was built, not the year
    it is scrapped, because a car keeps the motor it was made with.
    """
    combined = {}
    for key, frame in parallel.items():
        if frame.empty:
            continue
        block = frame.copy()
        block["group"] = [traction.segment_group(s, params)
                          for s in block["productKeyLevel3"]]
        weights = {}
        for group, year in set(zip(block["group"], block["production_year"])):
            weights[(group, year)] = traction.joint_shares(params, group, year)
        block["share"] = [
            weights[(g, y)].get((m, v), 0.0)
            for g, y, m, v in zip(block["group"], block["production_year"],
                                  block["componentKeyLevel1"],
                                  block["voltageClass"])]
        block["mass"] = block["mass"] * block["share"]
        combined[key] = block[block["mass"] > 0].reset_index(drop=True)
    return combined


def by_material(frames: dict, extra_keys: list[str]) -> pd.DataFrame:
    """One tidy frame: mass by flow, scrap year, material and whatever else."""
    rows = []
    for (region, drive, flow), frame in frames.items():
        if frame.empty:
            continue
        keys = ["scrap_year", "materialClass"] + extra_keys
        grouped = frame.groupby(keys, as_index=False, dropna=False)["mass"].sum()
        grouped["Region"], grouped["Drive Train"], grouped["flow"] = (
            region, drive, flow)
        rows.append(grouped)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())


def element_flows(magnet_mass: pd.DataFrame, elements: pd.DataFrame,
                  params) -> pd.DataFrame:
    """
    Nd, Pr, Dy and Tb, from the magnet mass and the `e-m` shares.

    ⚠️ PER GRADE SCENARIO, NEVER AVERAGED. SH, UH and EH are three answers to
    "which grade does a traction magnet use", and the mean of them is a magnet
    nobody makes. Every row carries `grade_scenario`, and a consumer picks one.

    ⚠️ AND THE SHARE DEPENDS ON THE MOTOR TYPE, because the grade does: the
    scenarios are applied to every magnet-bearing type, and the element sheet
    holds one row per (scenario, motor type, element).
    """
    if magnet_mass.empty or elements.empty:
        return pd.DataFrame()
    magnet = magnet_mass[magnet_mass["materialClass"] == "magnet"]
    if magnet.empty:
        return pd.DataFrame()

    wanted = ["Nd", "Pr", "Dy", "Tb"]
    shares = elements[elements["element"].isin(wanted)][
        ["grade_scenario", "is_base", "componentKeyLevel1", "element",
         "meanValue", "TmaxOperating_C"]].rename(
        columns={"meanValue": "element_share"})

    if "componentKeyLevel1" in magnet.columns:
        merged = magnet.merge(shares, on="componentKeyLevel1", how="inner")
    else:
        # The combined output has already summed over motor type, so the
        # element share cannot be motor-specific. Averaged across the
        # magnet-bearing types, which is exact only when they share a grade --
        # they do today, all SH, and the row says so.
        per_scenario = shares.groupby(
            ["grade_scenario", "is_base", "element", "TmaxOperating_C"],
            as_index=False)["element_share"].mean()
        merged = magnet.merge(per_scenario, how="cross")
    merged["mass"] = merged["mass"] * merged["element_share"]
    return merged.drop(columns=["element_share"]).reset_index(drop=True)


def main() -> dict[str, Any]:
    scenario_names = [
        "BAU", "BEV_only", "stock_lower", "BEV_A_F", "BEV_JA_JF", "BEV_large",
        "BEV_small", "BEV_longer", "ICEV_shorter", "losses_zero", "losses_high",
    ]
    # ⚠️ LOAD WHAT EXISTS. Ten of the eleven scenario trackers are absent
    # whenever stage 03 has not been re-run, and demanding all of them made this
    # stage impossible to run at all. The baseline is required; a missing
    # scenario is named and skipped.
    loaded = load_many("params", "tracker_keyed", root=PROJECT_ROOT)
    params = loaded["params"]
    p04 = params.to_nested_dict()["04_materials"]
    tracker_keyed = loaded["tracker_keyed"]

    print(artifact_status(root=PROJECT_ROOT))

    available, absent = {}, []
    for name in scenario_names:
        try:
            available[name] = load_many(f"tracker_keyed_{name}",
                                        root=PROJECT_ROOT)[f"tracker_keyed_{name}"]
        except Exception:
            absent.append(name)
    if absent:
        print(f"  scenarios not on disk, skipped: {', '.join(absent)}")

    composition = load_composition(p04)
    elements = load_magnet_elements(p04)
    print(f"  {len(composition):,} composition rows, "
          f"{composition.productionYear.nunique()} years, "
          f"{composition.componentKeyLevel1.nunique()} motor types, "
          f"{composition.voltageClass.nunique()} voltage classes")

    # ---------------------------------------------------------------- parallel
    print("\nParallel -- nothing mixed")
    parallel = quantify_parallel(tracker_keyed, composition)
    parallel_tidy = by_material(parallel, ["componentKeyLevel1", "voltageClass"])

    # ---------------------------------------------------------------- combined
    print("\nCombined -- the shares applied")
    combined = combine_partitioned(parallel, params)
    combined_tidy = by_material(combined, [])
    combined_by_type = by_material(combined, ["componentKeyLevel1"])

    # ---------------------------------------------------------------- elements
    element_tidy = element_flows(combined_by_type, elements, params)
    if not element_tidy.empty:
        print(f"  elements: {len(element_tidy):,} rows, "
              f"{element_tidy.grade_scenario.nunique()} grade scenarios")

    artifacts_dir = PROJECT_ROOT / "data" / "processed" / "intermediate"
    saved = save_unregistered_scenario_outputs(artifacts_dir, {
        "04_03_traction_parallel.pkl": parallel_tidy,
        "04_03_traction_combined.pkl": combined_tidy,
        "04_03_traction_combined_by_type.pkl": combined_by_type,
        "04_03_traction_elements.pkl": element_tidy,
    })

    out_dir = PROJECT_ROOT / "data" / "processed"
    for name, frame in (("04_03_traction_parallel", parallel_tidy),
                        ("04_03_traction_combined", combined_tidy),
                        ("04_03_traction_elements", element_tidy)):
        if not frame.empty:
            frame.to_csv(out_dir / f"{name}.csv", index=False)

    scenario_outputs = {}
    for name, tracker in available.items():
        par = quantify_parallel(tracker, composition)
        com = combine_partitioned(par, params)
        scenario_outputs[name] = {
            "parallel": by_material(par, ["componentKeyLevel1", "voltageClass"]),
            "combined": by_material(com, []),
        }
    if scenario_outputs:
        saved.update(save_unregistered_scenario_outputs(artifacts_dir, {
            f"04_03_traction_{name}.pkl": frames
            for name, frames in scenario_outputs.items()}))

    return {"saved": saved, "parallel": parallel_tidy,
            "combined": combined_tidy, "elements": element_tidy}


if __name__ == "__main__":
    main()
