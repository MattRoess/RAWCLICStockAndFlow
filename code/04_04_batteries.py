"""
04_04_batteries.py
==================

Battery material flows for BEVs: what enters the fleet, what leaves it, and
what is actually collected from it, in tonnes of each element, per year, per
chemistry, under each chemistry scenario.

THE THIRD SERIES IS THE ONE RECYCLING SEES. Of the BEVs leaving the fleet, 88%
are collected, 2% are exported second-hand and 10% are never traced. Only the
collected ones reach a recycler, so a secondary-supply number built on the
outflow is an upper bound, not a supply.

    .venv/bin/python code/04_04_batteries.py

THREE SCENARIOS x THREE CHEMISTRIES, KEPT APART
------------------------------------------------
The scenario shares decide how many cars carry each chemistry. The output does
NOT sum them: LFP, LMFP and NMC_high are reported separately within each
scenario, so the contribution of each is visible and the totals can be formed
by whoever needs them. Nine (scenario, chemistry) series per flow.

Sodium-ion and solid-state have a share in S2 and S3 and NO COMPOSITION at all.
Their share is reported as an explicit gap rather than dropped -- under S3 that
is most of the market by 2070, and a total that quietly fell would read as a
collapse in demand rather than a hole in the data.

WHAT IS DRAWN AND WHAT IS NOT
------------------------------
  vehicles     per-draw arrays from 03_02, (n_draws, n_years) per segment
  capacity     drawn per segment per draw -- a discrete pack size, held for life
  voltage      drawn per draw -- 400 or 800, never blended
  composition  per-draw element masses from RAWCLICVehicleBattery
  vintages     per draw, from that draw's own inflow history and the lifetime
               curve -- see src/battery_vintage.py
  shares       drawn -- not because the scenario is uncertain, but because a
               share stated for 2070 is a guess made forty-five years early. A
               triangular multiplier per chemistry, nothing in 2020 widening to
               the parameter's full value in 2070, then the group renormalised
               to one. No chemistry is a residual

INFLOW IS BUILT THIS YEAR, OUTFLOW WAS BUILT LONG AGO
------------------------------------------------------
A car scrapped in 2050 was built around 2034 and carries the chemistry mix and
the pack size of 2034. The outflow is therefore spread back over the build
years that could have produced it before any composition is applied. Without
that the mix multiplies both flows identically, cancels out of every
outflow-over-inflow ratio, and all three scenarios collapse onto one curve.

Draw i of every one of those is the same world, because both projects run at the
same number of draws. Nothing is averaged before the end.

WHAT IS WRITTEN
----------------
  data/processed/battery_draws/<flow>/<scenario>/<chemistry>.npy
        (n_draws, n_years, n_elements) in tonnes, float32 -- THE RESULT.
  04_04_battery_material_flows.pkl
        mean, median and the 95% band, taken FROM those draws. A convenience
        summary for reading and plotting, never an input to further maths.

The draws are written because recovery is a RATIO of two of these numbers, and
a ratio of percentiles is not the percentile of a ratio. Secondary supply has
to be formed draw by draw -- outflow of draw i over inflow of draw i, one
world at a time -- and that is impossible from a mean and two percentiles.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # never opens a window -- always safe to save to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.artifacts import load_many, save_many  # noqa: E402
from src.battery_capacity import capacity_draws  # noqa: E402
from src.battery_chemistry import (chemistry_share,  # noqa: E402
                                   chemistry_share_draws)
from src.battery_composition import CompositionAtCapacity, CompositionError  # noqa: E402
from src.battery_vintage import vintage_weights  # noqa: E402
from src.battery_voltage import voltage_draws  # noqa: E402

# Three series, all exported per draw by 03_02. The collected one is the only
# one that reaches a recycler: of the BEVs that leave the fleet, 88% are
# collected, 2% are exported and 10% are never traced -- and 03_02 draws those
# shares, so the collected array carries that uncertainty rather than being
# 0.88 times the outflow.
FLOWS = ("inflow", "outflow", "collected")

# Both levels the battery project writes. The elements do not add up to the
# pack -- the cell casing and the separator have no element rows and the
# electrolyte's cover 1% of its mass -- so 7-11% of every pack, the plastics,
# the separator and the electrolyte, exists only at the component level. That
# is the part a recycler has to deal with rather than sell.
LEVELS = ("element", "component")


def write_recovery_export(directory: Path, flow: str, scenario: str,
                          recovery_years, names: list[str],
                          totals: np.ndarray) -> int:
    """
    One .npy per name, in the shape and the unit RAWCLICRecoveryModel reads.

    Its `src/upstream.py` takes a folder of (draws, years) arrays in KILOTONNES
    whose file names run finest first and end with the group, so a component
    total is `__component____<component>` and an element inside one is
    `<element>__<component>`. Nothing here is summarised: the recovery model
    multiplies these by drawn transfer coefficients, and a mean times a mean is
    not the mean of the product.
    """
    target = directory / scenario / flow
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "years.npy", np.asarray(recovery_years, dtype=int))
    for position, name in enumerate(names):
        element, _, component = name.partition("|")
        stem = (f"__component____{component}" if element == "__component__"
                else f"{element}__{component}")
        np.save(target / f"{stem}.npy", (totals[:, :, position] / 1e3).astype(np.float32))
    return len(names)


def refuse_stale_parameters(params) -> None:
    """
    The saved parameters must be what `src/params_schema.py` says now, or stop.

    `00_parameters.py` builds `Params()` and saves it, so the two can differ for
    exactly one reason: the file predates an edit to the schema. That is not a
    harmless difference. A field declared with `default_factory` lives in the
    PICKLED INSTANCE, so an old file quietly wins over the new default, while a
    field with a plain default is a class attribute and picks the new one up --
    so a stale file does not fail, it half-updates.

    On 2026-09-15 that cost fifty minutes: 04_04 ran to completion, wrote every
    artifact and every figure, and carried three chemistries where the schema
    said five. Nothing anywhere said so. A run that takes most of an hour must
    not be able to answer yesterday's question.
    """
    from dataclasses import fields, is_dataclass
    from src.params_schema import Params

    differences = []

    def walk(saved, fresh, path=""):
        for field in fields(fresh):
            here = f"{path}.{field.name}".lstrip(".")
            new = getattr(fresh, field.name)
            if not hasattr(saved, field.name):
                differences.append(f"{here} (absent from the saved file)")
                continue
            old = getattr(saved, field.name)
            if is_dataclass(new) and not isinstance(new, type):
                walk(old, new, here)
            elif repr(old) != repr(new):
                differences.append(here)

    walk(params, Params())
    if differences:
        shown = "\n  ".join(differences[:8])
        more = "" if len(differences) <= 8 else f"\n  ... and {len(differences)-8} more"
        raise SystemExit(
            "The saved parameters are not what src/params_schema.py says:\n  "
            + shown + more
            + "\n\nRun code/00_parameters.py first. A saved file wins over a new "
              "default for any field with a default_factory, so this stage would "
              "otherwise compute the older answer and say nothing about it.")


def load_flow_draws(root: Path, scenario: str, segment: str, flow: str):
    """(years, draws) from 03_02's BEV export, or (None, None) if absent."""
    directory = root / "data" / "processed" / "bev_draws" / scenario
    path = directory / f"BEV_{segment}_{flow}.npy"
    years_path = directory / "years.npy"
    if not path.exists() or not years_path.exists():
        return None, None
    return np.load(years_path), np.load(path, mmap_mode="r")


def main() -> dict:
    params = load_many("params", root=PROJECT_ROOT)["params"]
    refuse_stale_parameters(params)
    materials = params.materials
    flow_scenario = "BAU"
    # EVERY YEAR. The flows are annual, so anything coarser is this stage
    # throwing away resolution the fleet model already has. It costs: the
    # vintage sum is over every build year for every reported year, which is
    # what makes this the long stage, and the draws it writes are tens of GB.
    #
    # The composition's improvement factor is written every FIVE years, so four
    # years in five are interpolated per draw -- see `_improvement` in
    # src/battery_composition.py. Without that interpolation those years would
    # silently carry no improvement at all.
    years = [y for y in range(2020, 2071)]
    # The build years an outflow can come from. It reaches back further than the
    # reported years because a car scrapped in 2021 was built well before 2020.
    # The pack size is read at the true build year; the composition files start
    # in 2020, so only that part is clamped, and the clamp is measured below.
    vintages = [y for y in range(2004, 2071)]
    vintage_position = {year: vintages.index(year) for year in years}
    recovery_years = [y for y in materials.battery_recovery_years if y in set(years)]
    composition_floor = 2020
    # A chemistry a scenario never uses is ABSENT from that scenario's share
    # dict, not present with a zero -- S1 contains no sodium at all. This is
    # what such a chemistry gets there, and its identity is the test for
    # "this scenario does not use it" below.
    no_share = None
    segments = list(materials.battery_capacity_levels)
    n_draws = params.monte_carlo.n_draws

    composition = CompositionAtCapacity(params)
    draws_dir = PROJECT_ROOT / "data" / "processed" / "battery_draws"
    scenarios = list(materials.battery_chemistry_scenarios)
    named = materials.battery_chemistry_file_names
    active_unknown = set(materials.battery_chemistry_active_material_unknown)
    recovery_dir = (PROJECT_ROOT / "data" / "processed"
                    / materials.battery_recovery_draws_dir)
    groups = materials.battery_chemistry_segment_groups
    lifetime = params.stock_flow.lifetime_by_drv["BEV"]

    print(f"flow scenario {flow_scenario} | {len(scenarios)} chemistry scenarios "
          f"| years {years[0]}-{years[-1]} every "
          f"{years[1]-years[0]} | {n_draws:,} draws")

    # One capacity and voltage draw per segment, reused by every chemistry and
    # scenario: a car's pack size does not change because the market's chemistry
    # mix does, and redrawing would decorrelate things that are one vehicle.
    # The year axis doubles as the VINTAGE axis on the outflow side -- the draws
    # are the same whichever years they are evaluated at, so a car built in 2035
    # and scrapped in 2050 is read off the same drawn vehicle.
    per_segment = {}
    for segment in segments:
        per_segment[segment] = (
            capacity_draws(params, segment, vintages, n_draws=n_draws, seed=404),
            voltage_draws(params, segment, vintages, n_draws=n_draws, seed=404),
        )

    rows, component_rows, gaps = [], [], []
    started = time.time()
    for flow in FLOWS:
        # Opened ONCE per flow, not once per segment per year: memory-mapped,
        # but 2,376 reopenings of the same file is still 2,376 reopenings.
        flow_years = None
        vehicles_by_segment, built_by_segment = {}, {}
        for segment in segments:
            segment_years, drawn = load_flow_draws(PROJECT_ROOT, flow_scenario,
                                                   segment, flow)
            if drawn is None:
                continue
            flow_years = segment_years
            vehicles_by_segment[segment] = drawn
            if flow != "inflow":
                # The draw's own build history, which decides its vintages.
                # Collected cars are scrapped cars: same build years as the
                # outflow they are a part of.
                _, built = load_flow_draws(PROJECT_ROOT, flow_scenario,
                                           segment, "inflow")
                built_by_segment[segment] = built
        if flow_years is None:
            print(f"  no {flow} draws for {flow_scenario} -- skipped")
            continue
        year_index = {int(y): i for i, y in enumerate(flow_years)}

        count_mean, clamped_reported = {}, None
        # The uncovered share is a DISTRIBUTION, because the shares that make it
        # are drawn. Accumulated as a vehicle-weighted numerator and denominator
        # per draw, so the percentiles at the end are percentiles of a share and
        # not a ratio of two percentiles.
        gap_top = {(scenario, group): np.zeros((n_draws, len(years)), dtype=np.float32)
                   for scenario in scenarios for group in set(groups.values())}
        gap_bottom = {group: np.zeros((n_draws, len(years)), dtype=np.float32)
                      for group in set(groups.values())}

        # ONE LEVEL AT A TIME, and that is a memory decision rather than a
        # modelling one. Five chemistries at two levels, 51 years and 200,000
        # draws is 14 GB of accumulators live at once; one level at a time is
        # half of that. The only work paid twice is the vintage weighting, which
        # is one matmul per segment and takes seconds.
        for level in LEVELS:
            accumulated = {
                (scenario, chemistry): np.zeros(
                    (n_draws, len(years), len(composition.names(file_name, level))),
                    dtype=np.float32)
                for scenario in scenarios for chemistry, file_name in named.items()}
            first_pass = level == LEVELS[0]

            for segment in segments:
                vehicles = vehicles_by_segment.get(segment)
                if vehicles is None:
                    continue
                capacity, voltage = per_segment[segment]
                group = groups[segment]
                counts = np.stack([
                    np.asarray(vehicles[:, year_index[int(year)]], dtype=np.float32)
                    if int(year) in year_index else np.zeros(n_draws, dtype=np.float32)
                    for year in years], axis=1)
                if first_pass:
                    count_mean[segment] = counts.mean(axis=0)

                if flow != "inflow":
                    weights, clamped = vintage_weights(
                        built_by_segment[segment], flow_years, years, vintages,
                        lifetime.shape_k, lifetime.scale_lambda)
                    if clamped_reported is None:
                        below = vintages.index(composition_floor)
                        clamped_reported = (
                            clamped, weights[:, :, :below].sum(axis=2).mean(axis=0))
                else:
                    weights = None

                # THE SHARES ARE DRAWN, per group, once for the whole segment:
                # the mix does not depend on the chemistry being accumulated or
                # on the level, and every chemistry of the group has to be in
                # hand at once because the renormalisation decides what gives
                # way.
                group_shares = {
                    scenario: chemistry_share_draws(params, scenario, group, vintages,
                                                    n_draws=n_draws, seed=404)
                    for scenario in scenarios}

                if first_pass:
                    # The share of THESE cars whose ACTIVE MATERIAL nobody has
                    # described, weighted by how many of them there are. On a
                    # retirement flow it follows the same vintage weights as the
                    # material, because it is a property of the year the car was
                    # built. Started from zeros and not from nothing: under S1
                    # the sum is over an empty set, and a bare sum() would hand
                    # back the integer 0 where an array is expected.
                    uncovered = {
                        scenario: sum(
                            (share for chemistry, share in group_shares[scenario].items()
                             if chemistry in active_unknown),
                            np.zeros((n_draws, len(vintages)), dtype=np.float32))
                        for scenario in scenarios}
                    for scenario in scenarios:
                        if flow == "inflow":
                            here = uncovered[scenario][
                                :, [vintage_position[y] for y in years]]
                        else:
                            here = np.einsum("dv,dtv->dt", uncovered[scenario],
                                             weights, optimize=True)
                        gap_top[(scenario, group)] += counts * here
                    gap_bottom[group] += counts
                    del uncovered

                for chemistry, file_name in named.items():
                    shares = {scenario: group_shares[scenario].get(chemistry, no_share)
                              for scenario in scenarios}
                    if all(share is no_share for share in shares.values()):
                        # In no scenario of this group. Its accumulator stays at
                        # zero and is still written, which is the truth: no car
                        # here carries it.
                        continue
                    # (n_draws, n_vintages, n_names) -- one pack per drawn car
                    # for every year it could have been built in.
                    per_car = np.stack([
                        composition.masses(chemistry=file_name,
                                           capacity_kwh=capacity[:, position],
                                           voltage_v=voltage[:, position],
                                           year=max(int(year), composition_floor),
                                           seed=404, level=level).astype(np.float32)
                        for position, year in enumerate(vintages)], axis=1)

                    for year_position, year in enumerate(years):
                        count = counts[:, year_position] * 1e3  # millions, kg -> t
                        for scenario in scenarios:
                            target = accumulated[(scenario, chemistry)]
                            if shares[scenario] is no_share:
                                continue
                            if flow == "inflow":
                                # Built this year: this year's mix and pack.
                                built = vintage_position[year]
                                share = shares[scenario][:, built]
                                if not share.any():
                                    continue
                                target[:, year_position, :] += (
                                    per_car[:, built, :] * (count * share)[:, None])
                            else:
                                # Scrapped this year, built across the vintages.
                                coefficient = (count[:, None] * shares[scenario]
                                               * weights[:, year_position, :])
                                target[:, year_position, :] += np.einsum(
                                    "dv,dve->de", coefficient, per_car, optimize=True)
                    del per_car
                del weights, group_shares
                print(f"  {flow:<9} {level:<9} {segment:<3} done "
                      f"({time.time()-started:5.0f}s)")

            for (scenario, chemistry), per_draw in accumulated.items():
                names = composition.names(named[chemistry], level)
                stem = chemistry if level == "element" else f"{chemistry}_components"
                label = "element" if level == "element" else "component"
                target = draws_dir / flow / scenario
                target.mkdir(parents=True, exist_ok=True)
                np.save(target / f"{stem}.npy", per_draw)
                np.save(target / f"{stem}_names.npy", np.asarray(names, dtype="U64"))
                np.save(draws_dir / "years.npy", np.asarray(years, dtype=int))

                # Summary FROM the draws, so the two can never disagree.
                mean = per_draw.mean(axis=0)
                low, median, high = np.percentile(per_draw, [2.5, 50, 97.5], axis=0)
                into = rows if level == "element" else component_rows
                for year_position, year in enumerate(years):
                    for name_position, name in enumerate(names):
                        into.append({
                            "flow_scenario": flow_scenario,
                            "chemistry_scenario": scenario,
                            "chemistry": chemistry, "flow": flow, "year": int(year),
                            label: name,
                            "mean_tonnes": float(mean[year_position, name_position]),
                            "median_tonnes": float(median[year_position, name_position]),
                            "p2.5": float(low[year_position, name_position]),
                            "p97.5": float(high[year_position, name_position]),
                        })
            accumulated.clear()

        if clamped_reported is not None:
            before_grid, before_composition = clamped_reported
            print(f"    {flow} vintage weight before {vintages[0]}: "
                  + ", ".join(f"{year} {share:.1%}" for year, share
                              in zip(years, before_grid) if share > 0.001))
            print(f"    on build years before {composition_floor}, where the "
                  "composition is clamped: "
                  + ", ".join(f"{year} {share:.1%}" for year, share
                              in zip(years, before_composition) if share > 0.001))

        # ------------------------------------------------------------------
        # THE EXPORT THE RECOVERY MODEL READS. Its own year grid, its own unit,
        # and the CROSS of the two levels: copper in a cable and copper in an
        # electrode foil go through different processes, so an element total
        # could not be given a coefficient that is right for both.
        #
        # Summed over the chemistries, because a recycler receives the mix.
        # Its own pass rather than a branch inside the ones above: it runs on
        # eleven years instead of fifty-one, so it costs about a fifth of one.
        # ------------------------------------------------------------------
        if recovery_years:
            union: dict[str, int] = {}
            for level, prefix in (("component", "__component__"), ("pair", "")):
                for file_name in named.values():
                    for name in composition.names(file_name, level):
                        union.setdefault(f"{prefix}|{name}" if prefix else name,
                                         len(union))
            totals = {scenario: np.zeros((n_draws, len(recovery_years), len(union)),
                                         dtype=np.float32) for scenario in scenarios}

            for segment in segments:
                vehicles = vehicles_by_segment.get(segment)
                if vehicles is None:
                    continue
                capacity, voltage = per_segment[segment]
                group = groups[segment]
                counts = np.stack([
                    np.asarray(vehicles[:, year_index[int(year)]], dtype=np.float32)
                    if int(year) in year_index else np.zeros(n_draws, dtype=np.float32)
                    for year in recovery_years], axis=1)
                weights = None if flow == "inflow" else vintage_weights(
                    built_by_segment[segment], flow_years, recovery_years, vintages,
                    lifetime.shape_k, lifetime.scale_lambda)[0]
                group_shares = {
                    scenario: chemistry_share_draws(params, scenario, group, vintages,
                                                    n_draws=n_draws, seed=404)
                    for scenario in scenarios}

                for chemistry, file_name in named.items():
                    shares = {scenario: group_shares[scenario].get(chemistry)
                              for scenario in scenarios}
                    if all(share is None for share in shares.values()):
                        continue
                    for level, prefix in (("component", "__component__"), ("pair", "")):
                        names = composition.names(file_name, level)
                        columns = [union[f"{prefix}|{n}" if prefix else n]
                                   for n in names]
                        per_car = np.stack([
                            composition.masses(
                                chemistry=file_name, capacity_kwh=capacity[:, position],
                                voltage_v=voltage[:, position],
                                year=max(int(year), composition_floor),
                                seed=404, level=level).astype(np.float32)
                            for position, year in enumerate(vintages)], axis=1)
                        for year_position, year in enumerate(recovery_years):
                            count = counts[:, year_position] * 1e3  # millions, kg -> t
                            for scenario in scenarios:
                                if shares[scenario] is None:
                                    continue
                                if flow == "inflow":
                                    built = vintage_position[year]
                                    share = shares[scenario][:, built]
                                    if not share.any():
                                        continue
                                    piece = (per_car[:, built, :]
                                             * (count * share)[:, None])
                                else:
                                    coefficient = (count[:, None] * shares[scenario]
                                                   * weights[:, year_position, :])
                                    piece = np.einsum("dv,dve->de", coefficient,
                                                      per_car, optimize=True)
                                totals[scenario][:, year_position, columns] += piece
                        del per_car
                del weights, group_shares
                print(f"  {flow:<9} recovery  {segment:<3} done "
                      f"({time.time()-started:5.0f}s)")

            names = list(union)
            for scenario in scenarios:
                written = write_recovery_export(recovery_dir, flow, scenario,
                                                recovery_years, names,
                                                totals[scenario])
            totals.clear()
            print(f"    recovery export: {written} arrays per scenario, "
                  f"{len(recovery_years)} years, kilotonnes")

        # The share whose ACTIVE MATERIAL nobody has described, reported rather
        # than dropped. Sodium-ion and solid-state now carry their packaging --
        # frame, enclosure, cables, collectors, at the mass of the pack they are
        # modelled on -- so their steel, aluminium and copper reach the totals.
        # Their cathode, anode and electrolyte are zero in those arrays, and a
        # zero there means NOT DESCRIBED. This is what keeps that readable.
        #
        # SUMMED over the chemistries that lack one, not maxed: both are missing
        # under S3 and the hole they leave is the two together. On a retirement
        # flow it is the share of the cars' BUILD years, carried forward by the
        # same vintage weights as the material.
        #
        # It is now a BAND, because the shares behind it are drawn. The
        # percentiles are taken of the share itself, per draw, never of a
        # numerator over a denominator that were summarised separately.
        def record(scenario: str, label: str, top: np.ndarray,
                   bottom: np.ndarray) -> None:
            share = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
            mean = share.mean(axis=0)
            low, median, high = np.percentile(share, [2.5, 50, 97.5], axis=0)
            for year_position, year in enumerate(years):
                gaps.append({
                    "flow": flow, "chemistry_scenario": scenario,
                    "year": int(year), "segment_group": label,
                    "share_without_composition": float(mean[year_position]),
                    "median": float(median[year_position]),
                    "p2.5": float(low[year_position]),
                    "p97.5": float(high[year_position])})

        for scenario in scenarios:
            for group in sorted(gap_bottom):
                record(scenario, group, gap_top[(scenario, group)], gap_bottom[group])
            record(scenario, "fleet",
                   sum(gap_top[(scenario, g)] for g in gap_bottom),
                   sum(gap_bottom.values()))
        gap_top.clear(), gap_bottom.clear()

    result = pd.DataFrame(rows)
    component_frame = pd.DataFrame(component_rows)
    gap_frame = pd.DataFrame(gaps).drop_duplicates()
    saved = save_many(battery_material_flows=result,
                      battery_component_flows=component_frame,
                      battery_chemistry_gaps=gap_frame, root=PROJECT_ROOT)
    print(f"\n{len(result):,} element rows | {len(component_frame):,} component "
          f"rows | saved {saved}")
    figures = build_all(params, result, gap_frame)
    return {"saved": saved, "rows": len(result),
            "component_rows": len(component_frame), "figures": figures}


# ===========================================================================
# THE FIGURES
#
# Part of the stage, and drawn at the end of every run.
# `code/test_04_04_figures.py` imports this file to redraw them alone, which
# takes half a minute against the stage's half hour -- a figure is changed far
# more often than a result is recomputed. That file is a bench tool, marked
# `test_` because the numeric prefix belongs to stages that produce results.
# ===========================================================================
FIGURE_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
DRAWS_DIR = PROJECT_ROOT / "data" / "processed" / "battery_draws"

SCENARIO_TITLES = {
    "S1": "S1  incumbents hold",
    "S2": "S2  sodium enters",
    "S3": "S3  sodium and solid-state",
}
SCENARIO_SUBTITLES = {
    "S1": "LFP volume, NMC premium, LMFP growing.\nNothing new ever arrives.",
    "S2": "Sodium-ion takes the small segments.\nNMC shrinks to a niche.",
    "S3": "S2 plus bipolar solid-state from 2040,\nlarge segments first.",
}
SCENARIO_COLORS = {"S1": "#1b9e77", "S2": "#7570b3", "S3": "#d95f02"}

CHEMISTRY_COLORS = {
    "LFP": "#2c7fb8", "LMFP": "#41b6c4", "NMC_high": "#e6550d",
    "Na_ion": "#bdbdbd", "solid_state": "#737373",
}
CHEMISTRY_LABELS = {
    "LFP": "LFP", "LMFP": "LMFP", "NMC_high": "NMC high-Ni",
    "Na_ion": "sodium-ion  (packaging only)",
    "solid_state": "solid-state  (packaging only)",
}
# Short names for the twelve components, and the three the element level cannot
# see at all -- the casing and the separator have no element rows, and the
# electrolyte's cover 1% of its mass.
COMPONENT_LABELS = {
    "cathodeActiveMaterial": "cathode active", "anodeActiveMaterial": "anode active",
    "batteryPackSupportFrame": "support frame",
    "batteryPackThermalConductor": "thermal conductor",
    "batteryPackModuleEnclosuresAndCoolantManifolds": "module enclosure",
    "currentCollectorAnode": "anode collector",
    "currentCollectorCathode": "cathode collector",
    "batteryPackCables": "cables", "batteryPackCellTerminals": "cell terminals",
    "batteryCellElectrolyte": "electrolyte", "batteryCellCasing": "cell casing",
    "batteryCellSeparator": "separator",
}
COMPONENT_COLORS = {
    "cathodeActiveMaterial": "#d94801", "anodeActiveMaterial": "#4a1486",
    "batteryPackSupportFrame": "#6baed6", "batteryPackThermalConductor": "#9ecae1",
    "batteryPackModuleEnclosuresAndCoolantManifolds": "#2171b5",
    "currentCollectorAnode": "#fd8d3c", "currentCollectorCathode": "#fdbe85",
    "batteryPackCables": "#41ab5d", "batteryPackCellTerminals": "#a1d99b",
    "batteryCellElectrolyte": "#d9d9d9", "batteryCellCasing": "#969696",
    "batteryCellSeparator": "#737373",
}
INVISIBLE_TO_ELEMENTS = ("batteryCellElectrolyte", "batteryCellCasing",
                         "batteryCellSeparator")
GROUP_TITLES = {"small": "small  (A, B, JA, JB)",
                "medium": "medium  (C, D, JC, JD)",
                "large": "large  (E, F, JE, JF)"}

OPEN_ITEM = ("Open item, not an oversight: no CELL composition for sodium-ion or "
             "solid-state has been published that survives scrutiny, and inventing "
             "one would be worse than the hole.\nTheir packaging is carried; their "
             "cathode, anode and electrolyte are not. See "
             "documentation/DESIGN_chemistries_without_composition.md.")

GAP_NOTE = ("Sodium-ion and solid-state carry their packaging but no active material: "
            "under S2 and S3 their cathode, anode and electrolyte leave the figure.\n"
            "The dotted line, right axis, is the share of cars whose cell is not "
            "described — read each curve against its own.")


def _style(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# --------------------------------------------------------------------- draws
_LOADED: dict[tuple[str, str], tuple] = {}


def load_elements(flow: str, scenario: str, level: str = "element"
                  ) -> tuple[np.ndarray, dict]:
    """
    Every element -- or component -- as (n_draws, n_years) tonnes, summed over
    the chemistries that have a composition. Summed PER DRAW, so draw i stays
    one world.

    One pass over the three arrays, held for the rest of the run: thirteen
    elements read one at a time would be thirteen passes over two gigabytes.
    """
    if (flow, scenario, level) in _LOADED:
        return _LOADED[(flow, scenario, level)]
    years = np.load(DRAWS_DIR / "years.npy")
    directory = DRAWS_DIR / flow / scenario
    totals: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*.npy")):
        if path.name.endswith("_names.npy"):
            continue
        at_component = path.stem.endswith("_components")
        if at_component != (level == "component"):
            continue
        names = list(np.load(directory / f"{path.stem}_names.npy"))
        drawn = np.load(path, mmap_mode="r")
        for column, name in enumerate(names):
            here = np.asarray(drawn[:, :, column], dtype=np.float32)
            totals[name] = here if name not in totals else totals[name] + here
    if not totals:
        raise SystemExit(f"no {level} draws in {directory}")
    _LOADED[(flow, scenario, level)] = (years, totals)
    return years, totals


def load_element(flow: str, scenario: str, element: str,
                 level: str = "element") -> tuple[np.ndarray, np.ndarray]:
    """(n_draws, n_years) tonnes of one element or component."""
    years, totals = load_elements(flow, scenario, level)
    if element not in totals:
        raise SystemExit(f"{element} appears in no chemistry of {flow}/{scenario}")
    return years, totals[element]


def elements_present(flow: str = "inflow", scenario: str = "S1",
                     level: str = "element") -> list[str]:
    """
    The names that actually carry mass, biggest first.

    The arrays hold the union over the chemistries, so an element only some of
    them contain -- and sulphur and vanadium, which none of the three do -- sits
    there as a column of zeros.

    Oxygen is dropped from the plots. It is bound in the cathode oxides and the
    phosphate, never leaves as oxygen, and nothing recovers it; carrying it into
    a panel of its own only makes the real streams smaller.
    """
    _, totals = load_elements(flow, scenario, level)
    skipped = {"O"} if level == "element" else set()
    carrying = {name: float(values.max()) for name, values in totals.items()
                if name not in skipped}
    return [name for name, top in sorted(carrying.items(), key=lambda kv: -kv[1])
            if top > 0]


def band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, median, high = np.percentile(values, [2.5, 50, 97.5], axis=0)
    return low, median, high


# ------------------------------------------------------------------ figure 1
def figure_scenarios(params) -> Path:
    """What the three scenarios actually assume, group by group."""
    materials = params.materials
    scenarios = list(materials.battery_chemistry_scenarios)
    groups = ["small", "medium", "large"]
    years = np.arange(2025, 2071)
    active_unknown = set(materials.battery_chemistry_active_material_unknown)

    fig, axes = plt.subplots(len(scenarios), len(groups), figsize=(13.5, 9.5),
                             sharex=True, sharey=True)
    for row, scenario in enumerate(scenarios):
        for column, group in enumerate(groups):
            ax = axes[row][column]
            definition = materials.battery_chemistry_scenarios[scenario][group]
            order = [c for c in ("LFP", "LMFP", "NMC_high", "Na_ion", "solid_state")
                     if c in definition]
            stack = np.array([[chemistry_share(params, scenario, group, c, y)
                               for y in years] for c in order]) * 100
            ax.stackplot(years, stack,
                         colors=[CHEMISTRY_COLORS[c] for c in order],
                         edgecolor="white", linewidth=0.4)
            # The boundaries between the bands are drawn, so they are not lines.
            # Each one is a CUMULATIVE share, which is what a boundary is, and
            # its 95% band is how far that boundary can sit from where the
            # scenario puts it. 5,000 draws: this is a picture, not a result.
            drawn = chemistry_share_draws(params, scenario, group, years,
                                          n_draws=5_000, seed=404)
            running = 0.0
            for chemistry in order[:-1]:
                running = running + drawn[chemistry] * 100
                low, high = np.percentile(running, [2.5, 97.5], axis=0)
                # Two thin dashed lines, NOT a filled band. A pale wash over a
                # coloured area reads as another chemistry, which is the one
                # thing this figure must not say.
                for edge in (low, high):
                    ax.plot(years, edge, color="white", linewidth=0.9,
                            linestyle=(0, (3, 2)), alpha=0.95)
            # the part of the stack with no composition, marked in place
            uncovered = stack[[i for i, c in enumerate(order) if c in active_unknown]]
            if len(uncovered):
                ax.fill_between(years, 100 - uncovered.sum(axis=0), 100,
                                facecolor="none", edgecolor="white",
                                hatch="///", linewidth=0.0, alpha=0.9)
            _style(ax)
            ax.set_ylim(0, 100)
            ax.set_xlim(years[0], years[-1])
            if row == 0:
                ax.set_title(GROUP_TITLES[group], fontsize=10.5, pad=8)
            if column == 0:
                ax.set_ylabel("share of new cars  [%]", fontsize=9.5)
                ax.text(-0.34, 0.5, SCENARIO_TITLES[scenario], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11.5,
                        fontweight="bold", color=SCENARIO_COLORS[scenario])
                ax.text(-0.22, 0.5, SCENARIO_SUBTITLES[scenario], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=8.2,
                        color="#555555")

    handles = [Patch(facecolor=CHEMISTRY_COLORS[c], label=CHEMISTRY_LABELS[c])
               for c in ("LFP", "LMFP", "NMC_high", "Na_ion", "solid_state")]
    handles.append(Patch(facecolor="white", edgecolor="#999999", hatch="///",
                         label="active material not described — reported as a gap"))
    handles.append(Line2D([], [], color="#999999", linewidth=0.9,
                          linestyle=(0, (3, 2)), label="95% range of the boundary"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle("Chemistry shares assumed by each scenario, 2025–2070",
                 fontsize=14, fontweight="bold", x=0.55)
    fig.text(0.55, 0.93, "Assumption, not data: the observed record ends in 2026.",
             ha="center", fontsize=9.5, color="#555555")
    fig.text(0.55, 0.905,
             "Each boundary is drawn, not fixed: the dashed pair around it is the 95% "
             "range it moves over — nothing in 2020, ±30% of each chemistry's own "
             "share by 2070.",
             ha="center", fontsize=8.8, color="#777777")
    fig.tight_layout(rect=[0.10, 0.07, 1, 0.92])
    return _save(fig, "04_04_1_chemistry_scenarios.png")


# ------------------------------------------------------------------ figure 2
def figure_element_comparison(gaps: pd.DataFrame,
                              elements=("Li", "Ni", "Cu", "Co")) -> Path:
    """The comparison itself: three scenarios, key elements, inflow and outflow."""
    flows = ("inflow", "outflow", "collected")
    fig, axes = plt.subplots(len(flows), len(elements),
                             figsize=(4.0 * len(elements), 3.7 * len(flows)),
                             sharex=True)
    for row, flow in enumerate(flows):
        for column, element in enumerate(elements):
            ax = axes[row][column]
            for scenario, color in SCENARIO_COLORS.items():
                years, values = load_element(flow, scenario, element)
                low, median, high = band(values)
                ax.fill_between(years, low / 1e3, high / 1e3, color=color, alpha=0.14,
                                linewidth=0)
                ax.plot(years, median / 1e3, color=color, linewidth=2.1,
                        label=SCENARIO_TITLES[scenario])
            gap = gaps[(gaps.flow == flow) & (gaps.segment_group == "fleet")]
            twin = ax.twinx()
            for scenario, color in SCENARIO_COLORS.items():
                here = gap[gap.chemistry_scenario == scenario].sort_values("year")
                twin.plot(here["year"], here["median"] * 100,
                          color=color, linewidth=1.0, linestyle=":", alpha=0.8)
            twin.set_ylim(0, 100)
            twin.set_yticks([0, 50, 100])
            twin.tick_params(labelsize=8, colors="#888888")
            twin.spines["top"].set_visible(False)
            if column == len(elements) - 1:
                twin.set_ylabel("cell not described  [%]", fontsize=8.5,
                                color="#888888")
            else:
                twin.set_yticklabels([])
            _style(ax)
            if row == 0:
                ax.set_title(element, fontsize=13, fontweight="bold", pad=8)
            if column == 0:
                ax.set_ylabel(f"{flow}  [kt / year]", fontsize=10)
            if row == len(flows) - 1:
                ax.set_xlabel("year", fontsize=9.5)

    handles = [Line2D([], [], color=c, linewidth=2.1, label=SCENARIO_TITLES[s])
               for s, c in SCENARIO_COLORS.items()]
    handles += [Line2D([], [], color="#888888", linewidth=2.1, alpha=0.3,
                       label="95% band of the draws"),
                Line2D([], [], color="#888888", linewidth=1.0, linestyle=":",
                       label="share of cars whose cell is not described (right axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Battery material demand and return under the three chemistry scenarios",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925, "median of 200,000 draws, 95% band shaded",
             ha="center", fontsize=9.5, color="#555555")
    fig.text(0.5, 0.095, GAP_NOTE, ha="center", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.055, OPEN_ITEM, ha="center", fontsize=8, color="#888888")
    fig.tight_layout(rect=[0, 0.20, 1, 0.92])
    return _save(fig, "04_04_2_scenario_comparison.png")


# ------------------------------------------------------------------ figure 3
def figure_chemistry_contribution(flows_frame: pd.DataFrame, element: str = "Li") -> Path:
    """Which chemistry carries the demand, and how much of the market is unknown."""
    years = [2030, 2040, 2050, 2070]
    chemistries = ["LFP", "LMFP", "NMC_high"]
    scenarios = list(SCENARIO_COLORS)
    here = flows_frame[(flows_frame.element == element) & (flows_frame.flow == "inflow")]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(13.5, 5),
                             sharey=True)
    for ax, scenario in zip(axes, scenarios):
        rows = here[here.chemistry_scenario == scenario]
        bottom = np.zeros(len(years))
        for chemistry in chemistries:
            values = np.array([
                rows[(rows.chemistry == chemistry) & (rows.year == y)]["mean_tonnes"].sum()
                for y in years]) / 1e3
            ax.bar([str(y) for y in years], values, bottom=bottom, width=0.62,
                   color=CHEMISTRY_COLORS[chemistry], label=CHEMISTRY_LABELS[chemistry],
                   edgecolor="white", linewidth=0.8)
            bottom += values
        for x, total in enumerate(bottom):
            ax.text(x, total * 1.02, f"{total:,.0f}", ha="center", va="bottom",
                    fontsize=8.5, color="#444444")
        _style(ax)
        ax.set_title(SCENARIO_TITLES[scenario], fontsize=11.5, fontweight="bold",
                     color=SCENARIO_COLORS[scenario], pad=8)
        ax.set_xlabel("year", fontsize=9.5)
    axes[0].set_ylabel(f"{element} entering the fleet  [kt / year]", fontsize=10)
    handles = [Patch(facecolor=CHEMISTRY_COLORS[c], label=CHEMISTRY_LABELS[c])
               for c in chemistries]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)

    fig.suptitle(f"Where the {element} demand sits, and what the scenarios take out of view",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915,
             "Only the three chemistries with a described cell are shown — sodium-ion "
             "and solid-state carry no lithium that anyone has counted. The falling "
             "totals in S2 and S3 are cars moving to them, not a falling demand.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.02, OPEN_ITEM, ha="center", fontsize=8, color="#888888")
    fig.tight_layout(rect=[0, 0.16, 1, 0.90])
    return _save(fig, "04_04_3_chemistry_contribution.png")


# ------------------------------------------------------------------ figure 4
def figure_uncovered(gaps: pd.DataFrame) -> Path:
    """How much of each scenario this model cannot yet describe."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    groups = ["small", "medium", "large"]
    styles = {"small": ":", "medium": "--", "large": "-."}

    ax = axes[0]
    for scenario, color in SCENARIO_COLORS.items():
        here = gaps[(gaps.flow == "inflow") & (gaps.segment_group == "fleet")
                    & (gaps.chemistry_scenario == scenario)].sort_values("year")
        ax.plot(here["year"], here["median"] * 100, color=color,
                linewidth=2.4, label=SCENARIO_TITLES[scenario])
        ax.fill_between(here["year"], here["p2.5"] * 100, here["p97.5"] * 100,
                        color=color, alpha=0.16, linewidth=0)
    _style(ax)
    ax.set_title("whole fleet, weighted by cars sold", fontsize=11, pad=8)
    ax.set_ylabel("cars whose cell is not described  [%]", fontsize=10)
    ax.set_xlabel("year", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    ax = axes[1]
    for scenario, color in SCENARIO_COLORS.items():
        for group in groups:
            here = gaps[(gaps.flow == "inflow") & (gaps.segment_group == group)
                        & (gaps.chemistry_scenario == scenario)].sort_values("year")
            ax.plot(here["year"], here["median"] * 100, color=color,
                    linewidth=1.6, linestyle=styles[group], alpha=0.9)
    _style(ax)
    ax.set_title("by segment group", fontsize=11, pad=8)
    ax.set_xlabel("year", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(handles=[Line2D([], [], color="#666666", linestyle=styles[g],
                              linewidth=1.6, label=GROUP_TITLES[g]) for g in groups],
              frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("The hole in the picture: share of cars whose cell is not described",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915,
             "Sodium-ion and solid-state summed, median with the 95% band. Their "
             "packaging IS counted; this is the cathode, anode and electrolyte that "
             "is not. S1 stays at an exact zero, not a narrow band.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.02, OPEN_ITEM, ha="center", fontsize=8, color="#888888")
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    return _save(fig, "04_04_4_uncovered_share.png")


# ------------------------------------------------------------------ figure 5
def figure_secondary_supply(elements=("Li", "Ni", "Cu", "Mn")) -> Path:
    """
    Collected over inflow, formed per draw. The reason the draws are kept: this
    ratio cannot be built from a mean and two percentiles.

    The outflow line above it is what LEAVES the fleet. Only the collected part
    reaches a recycler -- 88% of BEVs, with 2% exported and 10% never traced --
    so the distance between the two lines is material that exists and is lost.
    """
    fig, axes = plt.subplots(1, len(elements),
                             figsize=(3.6 * len(elements), 5.4), sharey=True)
    for ax, element in zip(np.atleast_1d(axes), elements):
        for scenario, color in SCENARIO_COLORS.items():
            years, inflow = load_element("inflow", scenario, element)
            for flow, width, style, alpha in (("outflow", 1.1, (0, (4, 2)), 0.0),
                                              ("collected", 2.2, "-", 0.14)):
                _, leaving = load_element(flow, scenario, element)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(inflow > 0, leaving / inflow, np.nan) * 100
                low, median, high = np.nanpercentile(ratio, [2.5, 50, 97.5], axis=0)
                if alpha:
                    ax.fill_between(years, low, high, color=color, alpha=alpha,
                                    linewidth=0)
                ax.plot(years, median, color=color, linewidth=width, linestyle=style,
                        label=SCENARIO_TITLES[scenario] if flow == "collected" else None)
        _style(ax)
        ax.set_title(element, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("year", fontsize=9.5)
        ax.axhline(100, color="#999999", linewidth=1.0, linestyle="--")
        ax.text(years[0] + 1, 108, "returning as much as entering", fontsize=8.2,
                color="#777777", ha="left")
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("share of the same year's demand  [%]", fontsize=10)
    handles = [Line2D([], [], color=c, linewidth=2.2, label=SCENARIO_TITLES[s])
               for s, c in SCENARIO_COLORS.items()]
    handles += [Line2D([], [], color="#666666", linewidth=2.2,
                       label="collected — reaches a recycler"),
                Line2D([], [], color="#666666", linewidth=1.1, linestyle=(0, (4, 2)),
                       label="outflow — leaves the fleet, 12% of it untraceably")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.045))

    fig.suptitle("What recycling actually gets, formed draw by draw",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925,
             "Collected of draw i over inflow of draw i, then the percentiles — not a "
             "ratio of percentiles, which would be a different and wrong number.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.012,
             "Above 100% more comes back than goes in: the cars being scrapped were "
             "built when lithium chemistries still dominated, while the new ones are "
             "not. The gap between the two lines is\nexport and untraced vehicles — "
             "material that exists and never reaches a recycler. Cobalt is not shown: "
             "it comes only from NMC, so its curve is nickel's to within 0.03 pp.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    return _save(fig, "04_04_5_secondary_supply.png")


# ------------------------------------------------------------------ figure 6
def figure_all_elements(flow: str = "inflow") -> Path:
    """Every element that carries mass, so nothing interesting stays hidden."""
    elements = elements_present(flow)
    columns = 4
    rows = int(np.ceil(len(elements) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.1 * rows),
                             sharex=True)
    flat = np.ravel(axes)
    for position, element in enumerate(elements):
        ax = flat[position]
        for scenario, color in SCENARIO_COLORS.items():
            years, values = load_element(flow, scenario, element)
            low, median, high = band(values)
            ax.fill_between(years, low / 1e3, high / 1e3, color=color, alpha=0.13,
                            linewidth=0)
            ax.plot(years, median / 1e3, color=color, linewidth=1.9,
                    label=SCENARIO_TITLES[scenario])
        _style(ax)
        ax.set_title(element, fontsize=12, fontweight="bold", pad=6)
        ax.set_ylabel("kt / year", fontsize=9)
        ax.margins(y=0.12)
    for spare in flat[len(elements):]:
        spare.set_visible(False)
    # The last row is short, so the bottom panel of a column is not always in
    # it. Label whichever one actually sits at the bottom of each column.
    for column in range(columns):
        lowest = max((position for position in range(len(elements))
                      if position % columns == column), default=None)
        if lowest is None:
            continue
        flat[lowest].set_xlabel("year", fontsize=9.5)
        flat[lowest].tick_params(labelbottom=True)

    handles = [Line2D([], [], color=c, linewidth=2.1, label=SCENARIO_TITLES[s])
               for s, c in SCENARIO_COLORS.items()]
    handles.append(Line2D([], [], color="#888888", linewidth=6, alpha=0.3,
                          label="95% band of the draws"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(f"Every element the three chemistries carry — {flow}",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.958,
             "Each panel has its own scale. Sulphur and vanadium are left out — no "
             "chemistry here contains them — and so is oxygen, which is bound in the "
             "cathode and is recovered by nobody.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.022,
             "S2 and S3 fall because sodium-ion and solid-state carry no described "
             "cell, so their cathode and anode leave the figure — not because the "
             "world needs less. Their packaging is in these totals.",
             ha="center", fontsize=8.5, color="#555555")
    fig.text(0.5, 0.004, OPEN_ITEM.replace(chr(10), " "), ha="center", fontsize=8,
             color="#888888")
    fig.tight_layout(rect=[0, 0.095, 1, 0.945])
    return _save(fig, f"04_04_6_all_elements_{flow}.png")


# ------------------------------------------------------------------ figure 7
def figure_components(flow: str = "collected") -> Path:
    """
    What the flow is made of, component by component.

    The three hatched ones are the reason this level exists: the element arrays
    cannot see them, and they are 7-11% of every pack. They are also the part a
    recycler has to handle rather than sell -- organic electrolyte, polymer
    separator, plastic casing.
    """
    scenarios = list(SCENARIO_COLORS)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 6),
                             sharey=True)
    for ax, scenario in zip(axes, scenarios):
        years, totals = load_elements(flow, scenario, "component")
        order = [name for name in COMPONENT_LABELS if name in totals]
        stack = np.array([np.median(totals[name].astype(float), axis=0) / 1e3
                          for name in order])
        ax.stackplot(years, stack, colors=[COMPONENT_COLORS[n] for n in order],
                     labels=[COMPONENT_LABELS[n] for n in order],
                     edgecolor="white", linewidth=0.4)
        hidden = np.array([stack[position] for position, name in enumerate(order)
                           if name in INVISIBLE_TO_ELEMENTS]).sum(axis=0)
        ax.fill_between(years, stack.sum(axis=0) - hidden, stack.sum(axis=0),
                        facecolor="none", edgecolor="white", hatch="///",
                        linewidth=0.0)
        _style(ax)
        ax.set_xlim(years[0], years[-1])
        ax.set_xlabel("year", fontsize=9.5)
        ax.set_title(SCENARIO_TITLES[scenario], fontsize=11.5, fontweight="bold",
                     color=SCENARIO_COLORS[scenario], pad=8)
    axes[0].set_ylabel(f"{flow}  [kt / year]", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="#999999", hatch="///",
                         label="invisible to the element level"))
    labels.append("invisible to the element level")
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=5,
               frameon=False, fontsize=9)
    fig.suptitle(f"What the {flow} flow is made of, component by component",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925,
             "Each band is that component's own median over 200,000 draws. The "
             "hatched top — electrolyte, separator and cell casing — is 7–11% of a "
             "pack and appears in no element figure at all.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.13, 1, 0.90])
    return _save(fig, f"04_04_7_components_{flow}.png")


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(PROJECT_ROOT)}")
    return path


def build_all(params, flows_frame: pd.DataFrame, gaps: pd.DataFrame) -> list[Path]:
    """Every figure, in the order they are meant to be read."""
    if not (DRAWS_DIR / "years.npy").exists():
        raise SystemExit("no battery draws -- run code/04_04_batteries.py first")
    missing = {"median", "p2.5", "p97.5"} - set(gaps.columns)
    if missing:
        raise SystemExit(
            f"the gap frame has no {sorted(missing)} -- it predates the chemistry "
            "shares becoming drawn, when the uncovered share was one number rather "
            "than a band. Re-run code/04_04_batteries.py; redrawing cannot invent "
            "a band that was never computed.")
    print("figures:")
    return [
        figure_scenarios(params),
        figure_element_comparison(gaps),
        figure_chemistry_contribution(flows_frame),
        figure_uncovered(gaps),
        figure_secondary_supply(),
        figure_all_elements("inflow"),
        # Not for the outflow: measured, the collected flow is 87.9% of it for
        # EVERY element to three decimals -- the share is drawn on vehicles, not
        # on materials, so that panel would be this one times a constant. The
        # difference between the two is shown where it means something, in the
        # secondary-supply figure.
        figure_all_elements("collected"),
        figure_components("inflow"),
        figure_components("collected"),
    ]


if __name__ == "__main__":
    main()
