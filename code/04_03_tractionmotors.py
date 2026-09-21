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
    "parameterCode", "meanValue", "p025", "p975",
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
    for column in ("meanValue", "p025", "p975"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # A row with no interval carries the mean as both ends: a point, not a gap.
    frame["p025"] = frame["p025"].fillna(frame["meanValue"])
    frame["p975"] = frame["p975"].fillna(frame["meanValue"])
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


def quantify_parallel(tracker_keyed: dict, composition: pd.DataFrame,
                      params=None) -> dict:
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

    # ⚠️ BEV ONLY, AND BEFORE THE JOIN. The composition describes `elvBEV`, so
    # joining the whole tracker drops every petrol, diesel and hybrid row --
    # 42.9% of all vehicle-flow -- and the warning then reads like a
    # catastrophe when it is a diesel not having a BEV traction motor. See
    # `traction_drive_trains` for why PHEV and HEV being excluded IS a real
    # limitation and not a tautology.
    wanted = (tuple(params.materials.traction_drive_trains) if params
              else ("BEV",))
    fallback = (dict(params.materials.traction_segment_fallback) if params
                else {})
    skipped_drive = sorted({drive for _region, drive in tracker_keyed
                            if drive not in wanted})
    if skipped_drive:
        print(f"  drive trains without a BEV traction motor, not counted: "
              f"{', '.join(skipped_drive)}")

    # ⚠️ AND A SEGMENT THE COMPOSITION DOES NOT HAVE. JA is in the tracker and
    # not in the dataset; without this it was 3.52 million vehicle-flows leaving
    # by the back door.
    if fallback:
        keys = composition[["productKeyLevel2",
                            "productKeyLevel3"]].drop_duplicates()
        by_segment = dict(zip(keys.productKeyLevel3, keys.productKeyLevel2))
        remap = {}
        for absent, stand_in in fallback.items():
            if absent not in by_segment and stand_in in by_segment:
                remap[absent] = by_segment[stand_in]
        if remap:
            print(f"  segments read from a stand-in: "
                  f"{', '.join(f'{a} -> {b}' for a, b in fallback.items())}")

    for (region, drive), frame in tracker_keyed.items():
        if drive not in wanted:
            continue
        for flow_name, flow in frame.groupby("flow"):
            if flow.empty:
                continue
            data = flow[["Region", "Drive Train", "key", "cohort_year",
                         "scrap_year", "amount"]].copy()
            data["key"] = data["key"].astype(str).str.strip().str.lower()
            if fallback and "Segment" in flow.columns:
                stand_in = {absent.lower(): by_segment[name]
                            for absent, name in fallback.items()
                            if name in by_segment}
                segments = flow["Segment"].astype(str).str.strip().str.upper()
                data["key"] = [
                    stand_in.get(str(seg).lower(), key)
                    if str(seg).upper() in
                    {a.upper() for a in fallback} else key
                    for seg, key in zip(segments, data["key"])]
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
            vehicles = merged["amount"] * 1e6
            merged["mass"] = vehicles * merged["meanValue"]
            merged["mass_low"] = vehicles * merged["p025"]
            merged["mass_high"] = vehicles * merged["p975"]
            merged = merged.rename(columns={"cohort_year": "production_year"})
            out[(region, drive, flow_name)] = merged.drop(
                columns=["key", "amount", "meanValue", "p025", "p975",
                         "parameterCode"],
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
        for column in ("mass", "mass_low", "mass_high"):
            block[column] = block[column] * block["share"]
        combined[key] = block[block["mass"] > 0].reset_index(drop=True)
    return combined


def by_material(frames: dict, extra_keys: list[str]) -> pd.DataFrame:
    """One tidy frame: mass by flow, scrap year, material and whatever else."""
    rows = []
    for (region, drive, flow), frame in frames.items():
        if frame.empty:
            continue
        keys = ["scrap_year", "materialClass"] + extra_keys
        columns = [c for c in ("mass", "mass_low", "mass_high")
                   if c in frame.columns]
        grouped = frame.groupby(keys, as_index=False, dropna=False)[columns].sum()
        grouped["Region"], grouped["Drive Train"], grouped["flow"] = (
            region, drive, flow)
        rows.append(grouped)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())


def _grade_metadata(elements: pd.DataFrame | None,
                    scenarios: list[str]) -> dict[str, dict]:
    """
    Which grade class is the base, and what each is rated to.

    Both live in the traction project. They travel in the `Magnet elements`
    sheet, so they are read from there and only guessed at if it is absent --
    and the guess is named as one: the first class alphabetically is nobody's
    base case.
    """
    fallback = {grade: {"is_base": grade == scenarios[0],
                        "TmaxOperating_C": None} for grade in scenarios}
    if elements is None or elements.empty:
        return fallback
    if not {"grade_scenario", "is_base"} <= set(elements.columns):
        return fallback
    out = {}
    for grade in scenarios:
        rows = elements[elements["grade_scenario"] == grade]
        if rows.empty:
            out[grade] = fallback[grade]
            continue
        temperature = (rows["TmaxOperating_C"].iloc[0]
                       if "TmaxOperating_C" in rows.columns else None)
        out[grade] = {"is_base": bool(rows["is_base"].iloc[0]),
                      "TmaxOperating_C": temperature}
    return out


def drawn_distributions(tracker_keyed: dict, composition: pd.DataFrame,
                        params, elements: pd.DataFrame | None = None,
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Material and element flows with their real distributions, from the draws.

    ⚠️ THIS REPLACES CARRYING p025 AND p975 THROUGH THE ARITHMETIC. That was
    wrong twice over: it summed percentiles across segments, motor types and
    components as if every one of them erred in the same direction, and for the
    elements it multiplied the magnet-mass interval by the chemistry MEAN, so a
    terbium line drawn with a +-11% band hid a +-95% specification range.

    Now the sum happens DRAW BY DRAW -- draw i of the magnet mass at this
    segment's torque, times draw i of that grade's chemistry, times the
    deterministic vehicle count, share and year factor -- and only the finished
    200,000-long distribution is reduced to percentiles. That is what the
    traction project persisted its arrays for.

    Returns `(materials, elements)`, both tidy with `mass`, `mass_low`,
    `mass_high`.
    """
    from src import traction_draws as td

    library = td.DrawLibrary(params)
    segment_torque = (composition.groupby("productKeyLevel3")
                      ["torque_nm"].first().to_dict())
    drive_trains = tuple(params.materials.traction_drive_trains)

    material_rows = []
    for material in sorted(composition["materialClass"].dropna().unique()):
        frame, columns = td.coefficients(tracker_keyed, segment_torque, params,
                                         library, material, drive_trains)
        if frame.empty:
            continue
        drawn = td.flow_distribution(frame,
                                     td.draw_matrix(library, columns,
                                                    segment_torque))
        material_rows.append(drawn.assign(materialClass=material))
    materials = (pd.concat(material_rows, ignore_index=True)
                 if material_rows else pd.DataFrame())

    # ---- the elements, magnet mass times chemistry, per draw --------------
    element_rows = []
    frame, columns = td.coefficients(tracker_keyed, segment_torque, params,
                                     library, "magnet", drive_trains)
    if not frame.empty:
        # ⚠️ THE SCENARIOS COME FROM THE ARRAYS ON DISK. They are the traction
        # project's setting, and this model's params have no `run` namespace to
        # read it from -- asking for one returned nothing and quietly left a
        # single scenario, which is how UH and EH disappeared from a figure that
        # had been drawing all three.
        scenarios = library.grade_classes()
        metadata = _grade_metadata(elements, scenarios)
        for grade in scenarios:
            names, _ = library.chemistry(library.motor_of(columns[0][0]), grade)
            for element in ("Nd", "Pr", "Dy", "Tb"):
                if element not in names:
                    continue
                drawn = td.flow_distribution(
                    frame, td.draw_matrix(library, columns, segment_torque,
                                          element=element, grade=grade))
                element_rows.append(drawn.assign(
                    element=element, grade_scenario=grade,
                    is_base=metadata[grade]["is_base"],
                    TmaxOperating_C=metadata[grade]["TmaxOperating_C"]))
    elements = (pd.concat(element_rows, ignore_index=True)
                if element_rows else pd.DataFrame())
    return materials, elements


MATERIAL_COLOURS = {
    "lamination": "#4A6FA5", "copper": "#B5651D", "magnet": "#C0392B",
    "steel": "#7F8C8D", "aluminium": "#7FB3D5",
}
ELEMENT_COLOURS = {"Nd": "#1F4E79", "Pr": "#2E8B57", "Dy": "#D68910",
                   "Tb": "#C0392B"}
GRADE_LINESTYLE = {"SH": "-", "UH": "--", "EH": ":"}


def build_material_ratio_df(combined: pd.DataFrame,
                            region: str = "EUR") -> pd.DataFrame:
    """
    The secondary-supply ratio: collected in year Y over new inflow in year Y.

    "What fraction of this year's new traction-motor material demand could in
    principle be met by this year's collected end-of-life material."

    ⚠️ RESTORED 2026-09-21. The rewrite dropped this and its figure without
    replacing them, which was a deletion dressed as a rewrite. Kept working on
    the tidy combined frame instead of the old dict of frames.
    """
    if combined.empty:
        return pd.DataFrame(columns=["scrap_year", "materialClass", "inflow",
                                     "collected", "ratio"])
    scoped = combined[combined["Region"] == region]
    if scoped.empty:
        return pd.DataFrame(columns=["scrap_year", "materialClass", "inflow",
                                     "collected", "ratio"])

    columns = [c for c in ("mass", "mass_low", "mass_high")
               if c in scoped.columns]

    def side(flow: str, name: str) -> pd.DataFrame:
        block = (scoped[scoped["flow"] == flow]
                 .groupby(["scrap_year", "materialClass"], as_index=False)
                 [columns].sum())
        return block.rename(columns={c: f"{name}_{c}" for c in columns})

    ratio = side("inflow", "inflow").merge(side("collected", "collected"),
                                           on=["scrap_year", "materialClass"],
                                           how="inner")
    positive = ratio["inflow_mass"].where(ratio["inflow_mass"] > 0)
    ratio["ratio"] = ratio["collected_mass"] / positive

    # ⚠️ THE BAND IS NARROW ON PURPOSE, AND THAT IS THE FINDING. A composition
    # error scales BOTH sides of this ratio: if the magnet mass is 10% high, the
    # collected mass and the inflow mass are both 10% high and the ratio is
    # unchanged. So the ends are taken CORRELATED -- low over low, high over
    # high -- and what survives is only the part that does not cancel, which is
    # the composition of a car built twenty years ago differing from one built
    # today. Taking low over high instead would assume the same fit erred in
    # opposite directions on the two flows, which is not a thing that happens.
    if {"collected_mass_low", "inflow_mass_low"} <= set(ratio.columns):
        at_low = (ratio["collected_mass_low"]
                  / ratio["inflow_mass_low"].where(ratio["inflow_mass_low"] > 0))
        at_high = (ratio["collected_mass_high"]
                   / ratio["inflow_mass_high"].where(
                       ratio["inflow_mass_high"] > 0))
        # ⚠️ AND WHICH END IS WHICH IS NOT FIXED. The two flows have different
        # relative widths -- collected comes from cohorts built twenty years
        # earlier, with their own composition -- so low-over-low is not reliably
        # the smaller ratio. Measured: copper 2040 gives 0.228 at the low ends
        # against 0.225 at the high ends, the wrong way round. Named for what
        # they are rather than where they came from.
        # ⚠️ AND THE CENTRAL RATIO IS IN THE SPAN. mean-over-mean is not
        # necessarily between low-over-low and high-over-high -- with different
        # relative widths on the two flows it can sit outside both, which left
        # the line outside its own band on 1.7% of rows. The band spans every
        # outcome considered, central included.
        ends = pd.concat([at_low, at_high, ratio["ratio"]], axis=1)
        ratio["ratio_low"] = ends.min(axis=1)
        ratio["ratio_high"] = ends.max(axis=1)
    return ratio


def figure_secondary_supply(ratio: pd.DataFrame, path: Path) -> Path:
    """Per material, the share of new demand same-year collection could meet."""
    figure, axis = plt.subplots(figsize=(10, 6))
    if ratio.empty:
        axis.text(0.5, 0.5, "No overlapping inflow/collected data",
                  ha="center", va="center")
    else:
        for material, block in ratio.groupby("materialClass"):
            block = block.sort_values("scrap_year")
            colour = MATERIAL_COLOURS.get(material)
            if {"ratio_low", "ratio_high"} <= set(block.columns):
                axis.fill_between(block["scrap_year"], block["ratio_low"],
                                  block["ratio_high"], color=colour,
                                  alpha=0.20, lw=0)
            axis.plot(block["scrap_year"], block["ratio"], lw=1.9,
                      color=colour, label=str(material))
        axis.axhline(1.0, color="#333333", lw=0.9, ls=":")
        axis.legend(frameon=False, fontsize=9, loc="upper left")
        figure.text(0.01, 0.015,
                    "dotted line at 1.0: same-year collection meets that "
                    "year's demand.\n"
                    "band: from the 200,000 draws of both flows -- narrow "
                    "because a composition error scales collected and inflow\n"
                    "together and cancels in their ratio.",
                    fontsize=8.2, color="#555555", ha="left", va="bottom")
    axis.set_title("Traction motor secondary-supply ratio "
                   "(collected / inflow, same year)", fontsize=12)
    axis.set_xlabel("Year")
    axis.set_ylabel("Ratio")
    axis.grid(True, ls="--", alpha=0.3)
    for side_name in ("top", "right"):
        axis.spines[side_name].set_visible(False)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


# ⚠️ THE TRACKER HAS NO "OUTFLOW" COLUMN, AND OUTFLOW IS COLLECTED PLUS LOST.
# Matthias 2026-09-21. End of life splits three ways in the tracker --
# `collected`, `export` and `unknown_whereabouts` -- and the last two are the
# same thing for a European recycler: material that left the fleet and cannot be
# recovered here. An exported car takes its motor with it; an unaccounted one
# takes it nobody knows where. So:
#
#     outflow = collected + lost        lost = export + unknown_whereabouts
#
# Drawn as four series rather than three, because the LOST line is the answer to
# "how much of the neodymium coming out of the fleet can we actually have", and
# leaving it as the white space between two other lines makes the reader
# measure it with a ruler.
END_OF_LIFE = ("collected", "export", "unknown_whereabouts")
LOST = ("export", "unknown_whereabouts")
FLOW_SERIES = {
    "inflow": ("into new vehicles", "#1F4E79", "-"),
    "outflow": ("out of the fleet (collected + lost)", "#7F8C8D", "-"),
    "collected": ("collected", "#2E8B57", "-"),
    "lost": ("lost (exported + unknown whereabouts)", "#C0392B", "--"),
}


def _flow_series(frame: pd.DataFrame, material: str, name: str) -> pd.DataFrame:
    """One of inflow / outflow / collected, summed by year, with its interval."""
    if frame.empty:
        return pd.DataFrame()
    wanted = {"outflow": END_OF_LIFE, "lost": LOST}.get(name, (name,))
    block = frame[(frame["materialClass"] == material)
                  & (frame["flow"].isin(wanted))]
    if block.empty:
        return pd.DataFrame()
    columns = [c for c in ("mass", "mass_low", "mass_high")
               if c in block.columns]
    return block.groupby("scrap_year", as_index=False)[columns].sum()


def figure_material(material: str, parallel: pd.DataFrame,
                    combined: pd.DataFrame, by_type: pd.DataFrame,
                    by_voltage: pd.DataFrame, path: Path) -> Path:
    """
    One material -- copper or magnet -- as development and as distribution.

    Matthias 2026-09-21 asked for exactly this: the developments, and the
    distributions, for copper and the permanent magnets.

    left    DEVELOPMENT. The combined fleet demand over time, on the fan of the
            15 unmixed motor-type x voltage states. How far the solid line sits
            inside that fan is how much of the answer is the share assumption
            rather than the composition.
    middle  DISTRIBUTION BY MOTOR TYPE, stacked. Which drive configurations the
            demand actually sits in, and how that shifts as the fleet turns
            over.
    right   DISTRIBUTION BY VOLTAGE CLASS, stacked -- and for copper this is the
            whole 800 V story: the same car needs two thirds of the copper at
            800 V and 0.585 at 1000 V.

    ⚠️ INFLOW ONLY. These are the materials going INTO new vehicles. Outflow and
    collected are in the same frames and drawn by the secondary-supply figure.
    """
    figure, axes = plt.subplots(1, 4, figsize=(22.5, 5.2))

    # ---- development: the three flows, with their intervals --------------
    axis = axes[0]
    for name, (label, colour, style) in FLOW_SERIES.items():
        series = _flow_series(combined, material, name)
        if series.empty:
            continue
        if {"mass_low", "mass_high"} <= set(series.columns):
            axis.fill_between(series["scrap_year"], series["mass_low"] / 1e6,
                              series["mass_high"] / 1e6, color=colour,
                              alpha=0.18, lw=0, zorder=2)
        axis.plot(series["scrap_year"], series["mass"] / 1e6, lw=2.5,
                  color=colour, ls=style, zorder=3, label=label)
    axis.set_title(f"{material.capitalize()}: in, out, collected and lost",
                   fontsize=11.5)
    axis.set_ylabel("[kt / year]")
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")

    # ---- the unmixed states, for the inflow ------------------------------
    axis = axes[1]
    states = parallel[(parallel["materialClass"] == material)
                      & (parallel["flow"] == "inflow")]
    if not states.empty:
        for _, state in states.groupby(["componentKeyLevel1", "voltageClass"]):
            state = state.groupby("scrap_year", as_index=False)["mass"].sum()
            axis.plot(state["scrap_year"], state["mass"] / 1e6, lw=0.9,
                      color="#AAB2BA", alpha=0.65, zorder=1)
    whole = _flow_series(combined, material, "inflow")
    if not whole.empty:
        axis.plot(whole["scrap_year"], whole["mass"] / 1e6, lw=2.8,
                  color=MATERIAL_COLOURS.get(material), zorder=3,
                  label="combined, shares applied")
        axis.legend(frameon=False, fontsize=9)
    axis.set_title(f"{material.capitalize()} in, against the 15 states",
                   fontsize=11.5)
    axis.set_ylabel("[kt / year]")

    # ---- distribution by motor type, in and collected --------------------
    _stacked(axes[2], by_type, material, "componentKeyLevel1",
             f"{material.capitalize()} IN, by motor type",
             lambda name: str(name).replace("ElectricMotors", ""),
             flows=("inflow",))
    _stacked(axes[3], by_type, material, "componentKeyLevel1",
             f"{material.capitalize()} COLLECTED, by motor type",
             lambda name: str(name).replace("ElectricMotors", ""),
             flows=("collected",))

    for axis in axes:
        axis.set_xlabel("Year")
        axis.set_ylim(bottom=0)
        axis.grid(True, ls="--", alpha=0.25)

    # ⚠️ THE EXPLANATION GOES UNDER THE FIGURE, NOT ON THE DATA. In-axes
    # annotations landed on the lines they were explaining.
    figure.text(0.005, 0.015,
                "outflow = collected + lost;  lost = exported + unknown "
                "whereabouts.      "
                "grey: each of the 15 motor-type x voltage states, unmixed "
                "-- 'if every car were this one'.      "
                "bands: 95% of 200,000 draws, summed draw by draw across "
                "segments, motor types, voltages and cohorts.",
                fontsize=8.2, color="#555555", ha="left")
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _stacked(axis, frame: pd.DataFrame, material: str, column: str,
             title: str, label, flows: tuple[str, ...] = ("inflow",)) -> None:
    """One stacked area: where a material's flow sits, over time."""
    block = frame[(frame["materialClass"] == material)
                  & (frame["flow"].isin(flows))] if not frame.empty else frame
    axis.set_title(title, fontsize=11.5)
    if block.empty:
        axis.text(0.5, 0.5, "no rows", ha="center", va="center")
        return
    wide = (block.groupby(["scrap_year", column], as_index=False)["mass"].sum()
            .pivot(index="scrap_year", columns=column, values="mass")
            .fillna(0.0).sort_index())
    axis.stackplot(wide.index, *[wide[c].values / 1e6 for c in wide.columns],
                   labels=[label(c) for c in wide.columns], alpha=0.9)
    axis.set_ylabel("[kt / year]")
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")


def figure_elements(elements: pd.DataFrame, path: Path) -> Path:
    """
    The magnet's elements: Nd, Pr, Dy and Tb, development and scenario spread.

    left    DEVELOPMENT, one line per element, base grade scenario. Log scale,
            because neodymium and terbium are three orders of magnitude apart
            and a linear axis shows only neodymium.
    right   WHAT THE GRADE COSTS. Dysprosium and terbium under SH, UH and EH --
            the only two elements the grade moves, since the didymium is
            0.29-0.32 of the magnet in every class.

    ⚠️ THREE ANSWERS, NEVER AVERAGED. The scenarios are three readings of "which
    grade does a traction magnet use", and their mean is a magnet nobody makes.
    """
    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.2))
    if elements.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No element rows -- re-run the traction project\n"
                                "to write the 'Magnet elements' sheet",
                      ha="center", va="center")
        figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)
        return path

    base = (elements[elements["is_base"]]
            if "is_base" in elements.columns else elements)

    # ⚠️ LEFT PANEL IS THE THREE FLOWS, not the inflow alone. For a rare earth
    # the gap between what goes in and what comes back out is the whole
    # recovery question, and drawing only the inflow hides it.
    axis = axes[0]
    for name, (label, colour, style) in FLOW_SERIES.items():
        # ⚠️ THE SAME MAPPING AS EVERYWHERE ELSE. This loop had its own copy,
        # written before `lost` existed, so it asked the data for a flow called
        # "lost" -- which does not exist, the tracker calls them `export` and
        # `unknown_whereabouts` -- and silently drew nothing. One mapping now.
        wanted = {"outflow": END_OF_LIFE, "lost": LOST}.get(name, (name,))
        flow_block = (base[base["flow"].isin(wanted)]
                      if "flow" in base.columns else base)
        for element, state in flow_block.groupby("element"):
            if element != "Nd":      # Nd carries the story; the rest crowd it
                continue
            if state.empty:
                continue
            columns = [c for c in ("mass", "mass_low", "mass_high")
                       if c in state.columns]
            state = state.groupby("scrap_year", as_index=False)[columns].sum()
            if {"mass_low", "mass_high"} <= set(state.columns):
                axis.fill_between(state["scrap_year"], state["mass_low"] / 1e6,
                                  state["mass_high"] / 1e6, color=colour,
                                  alpha=0.18, lw=0)
            axis.plot(state["scrap_year"], state["mass"] / 1e6, lw=2.4,
                      color=colour, ls=style, label=f"Nd {label}")
    axis.set_title("Neodymium: in, out, collected and lost", fontsize=11.5)
    axis.set_ylabel("[kt / year]")
    axis.legend(frameon=False, fontsize=8.5)

    axis = axes[1]
    block = (base[base["flow"] == "inflow"]
             if "flow" in base.columns else base)
    for element, state in block.groupby("element"):
        columns = [c for c in ("mass", "mass_low", "mass_high")
                   if c in state.columns]
        state = state.groupby("scrap_year", as_index=False)[columns].sum()
        colour = ELEMENT_COLOURS.get(element, "#555555")
        if {"mass_low", "mass_high"} <= set(state.columns):
            axis.fill_between(state["scrap_year"], state["mass_low"] / 1e6,
                              state["mass_high"] / 1e6, color=colour,
                              alpha=0.20, lw=0)
        axis.plot(state["scrap_year"], state["mass"] / 1e6, lw=2.1,
                  color=colour, label=element)
    # ⚠️ LINEAR, NOT LOG. Matthias 2026-09-21: the bands look terrible on a log
    # axis, and they do -- a 95% interval that starts near zero becomes a wedge
    # covering three decades and reads as an error rather than a range. Linear
    # tells the truth about the magnitudes and hides terbium instead, so the
    # magnitudes are STATED here rather than shown.
    axis.set_title("All four elements in, base grade", fontsize=11.5)
    axis.set_ylabel("[kt / year]")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, fontsize=9)
    peaks = (base[base["flow"] == "inflow"] if "flow" in base.columns else base)
    if not peaks.empty:
        summary = (peaks.groupby("element")["mass"].max() / 1e6
                   ).sort_values(ascending=False)
        axis.annotate(
            "peak, kt/year:  " + ",  ".join(
                f"{name} {value:.3g}" for name, value in summary.items())
            + f"\nterbium is {summary.iloc[0] / summary.iloc[-1]:.0f}x smaller "
              f"than neodymium and flat against this axis -- it is the right "
              f"panel",
            xy=(0.28, 0.42), xycoords="axes fraction", fontsize=8.2,
            color="#555555")

    axis = axes[2]
    scenarios = (elements[elements["flow"] == "inflow"]
                 if "flow" in elements.columns else elements)
    for (element, scenario), state in scenarios[
            scenarios["element"].isin(["Dy", "Tb"])].groupby(
            ["element", "grade_scenario"]):
        columns = [c for c in ("mass", "mass_low", "mass_high")
                   if c in state.columns]
        state = state.groupby("scrap_year", as_index=False)[columns].sum()
        colour = ELEMENT_COLOURS.get(element, "#555555")
        # ⚠️ EVERY SCENARIO BANDED, and they will overlap heavily -- which is
        # the honest picture. The grade moves dysprosium by 64% between SH and
        # EH, and the uncertainty within any one grade is of the same order, so
        # the three are not cleanly separated answers.
        if {"mass_low", "mass_high"} <= set(state.columns):
            axis.fill_between(state["scrap_year"], state["mass_low"] / 1e6,
                              state["mass_high"] / 1e6, color=colour,
                              alpha=0.10, lw=0)
        axis.plot(state["scrap_year"], state["mass"] / 1e6, lw=1.9,
                  color=colour, ls=GRADE_LINESTYLE.get(scenario, "-"),
                  label=f"{element}  {scenario}")
    axis.set_title("What the magnet grade costs: Dy and Tb", fontsize=11.5)
    axis.set_ylabel("[kt / year]")
    axis.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")

    for axis in axes:
        axis.set_xlabel("Year")
        axis.grid(True, ls="--", alpha=0.25)
    figure.text(0.005, 0.015,
                "Neodymium and praseodymium are identical across the three "
                "grade scenarios: the didymium is 0.29-0.32 of the magnet in "
                "every class, so only the heavy rare earths move.\n"
                "bands: 200,000 draws of the magnet mass times the same "
                "draw of that grade's chemistry, summed draw by draw. The "
                "chemistry dominates for terbium, whose SH range is ±95% of "
                "its own value -- which is why the middle axis is linear and "
                "not logarithmic.",
                fontsize=8.2, color="#555555", ha="left")
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _check_params_are_current(params) -> None:
    """
    Fail with an instruction when `00_params.pkl` predates the schema.

    ⚠️ A PICKLED DATACLASS DOES NOT GAIN FIELDS WHEN THE CLASS DOES. The params
    artifact is an INSTANCE frozen at the moment stage 00 last ran, so a field
    added afterwards is simply absent from it, and the first thing to touch that
    field fails -- inside `asdict()`, with an AttributeError naming a dataclass
    internal and not a cause.

    ⚠️ AND THE CHECK IS DERIVED, NOT LISTED. The first version compared against a
    hand-written tuple of field names, which went stale the very next time a
    parameter was added -- the guard against staleness, stale. It now asks the
    LIVE dataclass what fields exist and compares that to what the artifact
    carries, so anything added later is covered without anyone remembering to.

    `vars()`, not `hasattr`: a field with a plain default is also a CLASS
    attribute, so `hasattr` is True on an instance that does not have it.
    `asdict()` reads the instance, so the instance is what to check.
    """
    from dataclasses import fields as dataclass_fields
    from src.params_schema import MaterialsParams  # type: ignore

    declared = {f.name for f in dataclass_fields(MaterialsParams)}
    carried = set(vars(params.materials))
    absent = sorted(declared - carried)
    if not absent:
        return
    shown = ", ".join(absent[:6]) + (" ..." if len(absent) > 6 else "")
    raise SystemExit(
        "04_03_tractionmotors.py: the params artifact predates the schema.\n"
        f"  {len(absent)} field(s) missing from 00_params.pkl: {shown}\n"
        "  A pickled dataclass does not gain fields when the class does.\n\n"
        "  Fix:\n"
        "      .venv/bin/python code/00_parameters.py\n"
        "  then run this stage again."
    )


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
    _check_params_are_current(params)
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
    # Loaded to report whether the traction project has written it; the element
    # FLOWS come from the draw arrays, not from these summary rows.
    elements = load_magnet_elements(p04)
    print(f"  {len(composition):,} composition rows, "
          f"{composition.productionYear.nunique()} years, "
          f"{composition.componentKeyLevel1.nunique()} motor types, "
          f"{composition.voltageClass.nunique()} voltage classes")

    # ---------------------------------------------------------------- parallel
    print("\nParallel -- nothing mixed")
    parallel = quantify_parallel(tracker_keyed, composition, params)
    parallel_tidy = by_material(parallel, ["componentKeyLevel1", "voltageClass"])

    # ---------------------------------------------------------------- combined
    print("\nCombined -- the shares applied")
    combined = combine_partitioned(parallel, params)
    combined_by_type = by_material(combined, ["componentKeyLevel1"])
    combined_by_voltage = by_material(combined, ["voltageClass"])

    # ---------------------------------------------- the real distributions
    #
    # ⚠️ THE MEANS COME FROM THE ARITHMETIC ABOVE AND THE BANDS DO NOT. Summing
    # p025 across segments, motor types and components assumed they all erred
    # together, which roughly doubled the width -- the drawn magnet band is
    # +-5%, the carried one was +-10%. The stacked panels keep using the means,
    # because a stack has no band; everything with a band reads from here.
    print("\nDrawn -- the 200,000 simulations, summed draw by draw")
    drawn_materials, element_tidy = drawn_distributions(
        tracker_keyed, composition, params, elements)
    combined_tidy = (drawn_materials if not drawn_materials.empty
                     else by_material(combined, []))
    if not drawn_materials.empty:
        print(f"  materials: {len(drawn_materials):,} rows with true "
              f"percentiles")
    if not element_tidy.empty:
        print(f"  elements: {len(element_tidy):,} rows, "
              f"{element_tidy.grade_scenario.nunique()} grade scenarios, "
              f"magnet mass x chemistry per draw")

    artifacts_dir = PROJECT_ROOT / "data" / "processed" / "intermediate"
    saved = save_unregistered_scenario_outputs(artifacts_dir, {
        "04_03_traction_parallel.pkl": parallel_tidy,
        "04_03_traction_combined.pkl": combined_tidy,
        "04_03_traction_combined_by_type.pkl": combined_by_type,
        "04_03_traction_elements.pkl": element_tidy,
    })

    # ⚠️ INTO A FOLDER OF ITS OWN. `data/processed/` holds directories --
    # battery_draws, bev_draws, element_draws, figures, intermediate -- and no
    # loose files; the first version of this stage dropped three CSVs straight
    # into it, which is the orphan-data problem by another name.
    out_dir = PROJECT_ROOT / "data" / "processed"
    csv_dir = out_dir / "traction"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("04_03_traction_parallel", parallel_tidy),
                        ("04_03_traction_combined", combined_tidy),
                        ("04_03_traction_combined_by_type", combined_by_type),
                        ("04_03_traction_elements", element_tidy)):
        if not frame.empty:
            frame.to_csv(csv_dir / f"{name}.csv", index=False)

    scenario_outputs = {}
    for name, tracker in available.items():
        par = quantify_parallel(tracker, composition, params)
        com = combine_partitioned(par, params)
        scenario_outputs[name] = {
            "parallel": by_material(par, ["componentKeyLevel1", "voltageClass"]),
            "combined": by_material(com, []),
        }
    if scenario_outputs:
        saved.update(save_unregistered_scenario_outputs(artifacts_dir, {
            f"04_03_traction_{name}.pkl": frames
            for name, frames in scenario_outputs.items()}))

    # ------------------------------------------------------------- figures
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    # ⚠️ THE FIGURES START AFTER THE FIRST YEAR OF THE SERIES. The tracker's
    # 2011 BEV inflow is about twenty times the real one -- see
    # `traction_figure_first_year`. Clipped here, for drawing only: everything
    # written to disk above still carries it.
    first = int(params.materials.traction_figure_first_year)

    def _from(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame[frame["scrap_year"] >= first]
                if not frame.empty and "scrap_year" in frame.columns else frame)

    parallel_shown = _from(parallel_tidy)
    combined_shown = _from(combined_tidy)
    by_type_shown = _from(combined_by_type)
    by_voltage_shown = _from(combined_by_voltage)
    elements_shown = _from(element_tidy)

    ratio = build_material_ratio_df(combined_tidy)
    made = [
        figure_material("copper", parallel_shown, combined_shown, by_type_shown,
                        by_voltage_shown, figure_dir / "04_03_1_copper.png"),
        figure_material("magnet", parallel_shown, combined_shown, by_type_shown,
                        by_voltage_shown, figure_dir / "04_03_2_magnet.png"),
        figure_elements(elements_shown, figure_dir / "04_03_3_rare_earths.png"),
        figure_secondary_supply(_from(ratio),
                                figure_dir / "04_03_4_secondary_supply.png"),
    ]
    print("\nFigures")
    for path in made:
        print(f"  {path}")

    return {"saved": saved, "parallel": parallel_tidy,
            "combined": combined_tidy, "elements": element_tidy,
            "ratio": ratio, "figures": made}


if __name__ == "__main__":
    main()
