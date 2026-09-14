"""
04_04_batteries.py
==================

Battery material flows for BEVs: what enters the fleet and what leaves it, in
kilograms of each element, per year, per chemistry, under each chemistry
scenario.

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
  shares       NOT drawn. A scenario stating 30% LFP is an assumption, and the
               fleet of a segment really does contain that mix; drawing it would
               turn a stated input into a spread

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.artifacts import load_many, save_many  # noqa: E402
from src.battery_capacity import capacity_draws  # noqa: E402
from src.battery_chemistry import chemistry_share  # noqa: E402
from src.battery_composition import CompositionAtCapacity, CompositionError  # noqa: E402
from src.battery_figures import build_all  # noqa: E402
from src.battery_vintage import vintage_weights  # noqa: E402
from src.battery_voltage import voltage_draws  # noqa: E402

FLOWS = ("inflow", "outflow")


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
    materials = params.materials
    flow_scenario = "BAU"
    # Every two years. The flows are annual and the chemistry shares move
    # steadily, so a five-year grid was hiding real movement between its points;
    # the composition's improvement factor is written every five years and is
    # interpolated per draw for the years in between.
    years = [y for y in range(2020, 2071, 2)]
    # The build years an outflow can come from. It reaches back further than the
    # reported years because a car scrapped in 2022 was built well before 2020.
    # The pack size is read at the true build year; the composition files start
    # in 2020, so only that part is clamped, and the clamp is measured below.
    vintages = [y for y in range(2004, 2071, 2)]
    vintage_position = {year: vintages.index(year) for year in years}
    composition_floor = 2020
    segments = list(materials.battery_capacity_levels)
    n_draws = params.monte_carlo.n_draws

    composition = CompositionAtCapacity(params)
    draws_dir = PROJECT_ROOT / "data" / "processed" / "battery_draws"
    scenarios = list(materials.battery_chemistry_scenarios)
    named = materials.battery_chemistry_file_names
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

    rows, gaps = [], []
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
            if flow == "outflow":
                # The draw's own build history, which decides its vintages.
                _, built = load_flow_draws(PROJECT_ROOT, flow_scenario,
                                           segment, "inflow")
                built_by_segment[segment] = built
        if flow_years is None:
            print(f"  no {flow} draws for {flow_scenario} -- skipped")
            continue
        year_index = {int(y): i for i, y in enumerate(flow_years)}

        accumulated = {
            (scenario, chemistry): np.zeros(
                (n_draws, len(years), len(composition.elements(file_name))),
                dtype=np.float32)
            for scenario in scenarios for chemistry, file_name in named.items()}
        vintage_mean, count_mean, clamped_reported = {}, {}, None

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
            count_mean[segment] = counts.mean(axis=0)

            if flow == "outflow":
                weights, clamped = vintage_weights(
                    built_by_segment[segment], flow_years, years, vintages,
                    lifetime.shape_k, lifetime.scale_lambda)
                vintage_mean[segment] = weights.mean(axis=0)
                if clamped_reported is None:
                    below = vintages.index(composition_floor)
                    clamped_reported = (clamped,
                                        weights[:, :, :below].sum(axis=2).mean(axis=0))
            else:
                weights = None

            for chemistry, file_name in named.items():
                # (n_draws, n_vintages, n_elements) -- one pack per drawn car for
                # every year it could have been built in.
                per_car = np.stack([
                    composition.masses(chemistry=file_name,
                                       capacity_kwh=capacity[:, position],
                                       voltage_v=voltage[:, position],
                                       year=max(int(year), composition_floor),
                                       seed=404).astype(np.float32)
                    for position, year in enumerate(vintages)], axis=1)
                shares = {scenario: np.array(
                    [chemistry_share(params, scenario, group, chemistry, year)
                     for year in vintages], dtype=np.float32)
                    for scenario in scenarios}

                for year_position, year in enumerate(years):
                    count = counts[:, year_position] * 1e3  # millions of cars, kg -> t
                    for scenario in scenarios:
                        target = accumulated[(scenario, chemistry)]
                        if flow == "inflow":
                            # Built this year: this year's mix and this year's pack.
                            built = vintage_position[year]
                            share = float(shares[scenario][built])
                            if share <= 0:
                                continue
                            target[:, year_position, :] += (
                                per_car[:, built, :] * (count * share)[:, None])
                        else:
                            # Scrapped this year, built across the vintages.
                            coefficient = (count[:, None] * shares[scenario][None, :]
                                           * weights[:, year_position, :])
                            target[:, year_position, :] += np.einsum(
                                "dv,dve->de", coefficient, per_car, optimize=True)
                del per_car
            del weights
            print(f"  {flow:<8} {segment:<3} done ({time.time()-started:5.0f}s)")

        if clamped_reported is not None:
            before_grid, before_composition = clamped_reported
            print(f"    outflow vintage weight before {vintages[0]}: "
                  + ", ".join(f"{year} {share:.1%}" for year, share
                              in zip(years, before_grid) if share > 0.001))
            print(f"    on build years before {composition_floor}, where the "
                  "composition is clamped: "
                  + ", ".join(f"{year} {share:.1%}" for year, share
                              in zip(years, before_composition) if share > 0.001))

        for (scenario, chemistry), per_draw in accumulated.items():
            elements = composition.elements(named[chemistry])
            target = draws_dir / flow / scenario
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / f"{chemistry}.npy", per_draw)
            np.save(target / f"{chemistry}_elements.npy",
                    np.asarray(elements, dtype="U8"))
            np.save(draws_dir / "years.npy", np.asarray(years, dtype=int))

            # Summary FROM the draws, so the two can never disagree.
            mean = per_draw.mean(axis=0)
            low, median, high = np.percentile(per_draw, [2.5, 50, 97.5], axis=0)
            for year_position, year in enumerate(years):
                for element_position, element in enumerate(elements):
                    rows.append({
                        "flow_scenario": flow_scenario, "chemistry_scenario": scenario,
                        "chemistry": chemistry, "flow": flow, "year": int(year),
                        "element": element,
                        "mean_tonnes": float(mean[year_position, element_position]),
                        "median_tonnes": float(median[year_position, element_position]),
                        "p2.5": float(low[year_position, element_position]),
                        "p97.5": float(high[year_position, element_position]),
                    })
        accumulated.clear()

        # The share that has no composition, reported rather than dropped.
        # SUMMED over the chemistries that lack one, not maxed: sodium-ion and
        # solid-state are both missing in S3, and the hole they leave is the two
        # together. On the outflow it is the share of the cars' BUILD years,
        # carried forward by the same vintage weights as the material.
        for scenario in scenarios:
            uncovered = {
                group: np.array([
                    sum(chemistry_share(params, scenario, group, chemistry, year)
                        for chemistry in materials.battery_chemistry_scenarios[scenario][group]
                        if chemistry not in named)
                    for year in (years if flow == "inflow" else vintages)])
                for group in set(groups.values())}

            by_group, group_weight = {}, {}
            for segment in count_mean:
                group = groups[segment]
                here = (uncovered[group] if flow == "inflow"
                        else vintage_mean[segment] @ uncovered[group])
                by_group[group] = by_group.get(group, 0.0) + count_mean[segment] * here
                group_weight[group] = group_weight.get(group, 0.0) + count_mean[segment]

            fleet_top = sum(by_group.values())
            fleet_bottom = sum(group_weight.values())
            for group, weighted in by_group.items():
                share = np.divide(weighted, group_weight[group],
                                  out=np.zeros_like(weighted),
                                  where=group_weight[group] > 0)
                for year_position, year in enumerate(years):
                    gaps.append({"flow": flow, "chemistry_scenario": scenario,
                                 "year": int(year), "segment_group": group,
                                 "share_without_composition": float(share[year_position])})
            fleet = np.divide(fleet_top, fleet_bottom, out=np.zeros_like(fleet_top),
                              where=fleet_bottom > 0)
            for year_position, year in enumerate(years):
                gaps.append({"flow": flow, "chemistry_scenario": scenario,
                             "year": int(year), "segment_group": "fleet",
                             "share_without_composition": float(fleet[year_position])})

    result = pd.DataFrame(rows)
    gap_frame = pd.DataFrame(gaps).drop_duplicates()
    saved = save_many(battery_material_flows=result,
                      battery_chemistry_gaps=gap_frame, root=PROJECT_ROOT)
    print(f"\n{len(result):,} rows | saved {saved}")
    figures = build_all(params, result, gap_frame)
    return {"saved": saved, "rows": len(result), "figures": figures}


if __name__ == "__main__":
    main()
