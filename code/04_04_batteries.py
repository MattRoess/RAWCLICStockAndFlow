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
  shares       NOT drawn. A scenario stating 30% LFP is an assumption, and the
               fleet of a segment really does contain that mix; drawing it would
               turn a stated input into a spread

Draw i of every one of those is the same world, because both projects run at the
same number of draws. Nothing is averaged before the end.
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
from src.battery_composition import CompositionAtCapacity, CompositionError  # noqa: E402
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


def chemistry_share(params, scenario: str, group: str, chemistry: str,
                    year: float) -> float:
    """One chemistry's share of a segment group in a year, renormalised."""
    materials = params.materials
    anchors = np.asarray(materials.battery_chemistry_anchor_years, dtype=float)
    definition = materials.battery_chemistry_scenarios[scenario][group]
    total = sum(np.interp(year, anchors, np.asarray(v, dtype=float))
                for v in definition.values())
    if total <= 0:
        return 0.0
    here = np.interp(year, anchors, np.asarray(definition[chemistry], dtype=float))
    return float(here / total)


def main() -> dict:
    params = load_many("params", root=PROJECT_ROOT)["params"]
    materials = params.materials
    flow_scenario = "BAU"
    years = [y for y in range(2020, 2071, 5)]
    segments = list(materials.battery_capacity_levels)
    n_draws = params.monte_carlo.n_draws

    composition = CompositionAtCapacity(params)
    scenarios = list(materials.battery_chemistry_scenarios)
    named = materials.battery_chemistry_file_names
    groups = materials.battery_chemistry_segment_groups

    print(f"flow scenario {flow_scenario} | {len(scenarios)} chemistry scenarios "
          f"| years {years[0]}-{years[-1]} every 5 | {n_draws:,} draws")

    # One capacity and voltage draw per segment, reused by every chemistry and
    # scenario: a car's pack size does not change because the market's chemistry
    # mix does, and redrawing would decorrelate things that are one vehicle.
    per_segment = {}
    for segment in segments:
        per_segment[segment] = (
            capacity_draws(params, segment, years, n_draws=n_draws, seed=404),
            voltage_draws(params, segment, years, n_draws=n_draws, seed=404),
        )

    rows, gaps = [], []
    started = time.time()
    for flow in FLOWS:
        # Opened ONCE per flow, not once per segment per year: memory-mapped,
        # but 2,376 reopenings of the same file is still 2,376 reopenings.
        flow_years = None
        vehicles_by_segment = {}
        for segment in segments:
            segment_years, drawn = load_flow_draws(PROJECT_ROOT, flow_scenario,
                                                   segment, flow)
            if drawn is None:
                continue
            flow_years = segment_years
            vehicles_by_segment[segment] = drawn
        if flow_years is None:
            print(f"  no {flow} draws for {flow_scenario} -- skipped")
            continue
        year_index = {int(y): i for i, y in enumerate(flow_years)}

        for scenario in scenarios:
            for chemistry, file_name in named.items():
                elements = composition.elements(file_name)
                totals = np.zeros((len(years), len(elements)))
                bands = np.zeros((len(years), len(elements), 2))

                for year_position, year in enumerate(years):
                    accumulated = np.zeros((n_draws, len(elements)))
                    for segment in segments:
                        vehicles = vehicles_by_segment.get(segment)
                        if vehicles is None or int(year) not in year_index:
                            continue
                        share = chemistry_share(params, scenario,
                                                groups[segment], chemistry, year)
                        if share <= 0:
                            continue
                        capacity, voltage = per_segment[segment]
                        per_car = composition.masses(
                            chemistry=file_name,
                            capacity_kwh=capacity[:, year_position],
                            voltage_v=voltage[:, year_position],
                            year=int(year), seed=404)
                        # vehicles are in MILLIONS; kg per car x 1e6 -> tonnes
                        count = np.asarray(vehicles[:, year_index[int(year)]],
                                           dtype=float)
                        accumulated += per_car * (count * share * 1e6 / 1e3)[:, None]

                    totals[year_position] = accumulated.mean(axis=0)
                    bands[year_position, :, 0] = np.percentile(accumulated, 2.5, axis=0)
                    bands[year_position, :, 1] = np.percentile(accumulated, 97.5, axis=0)

                for year_position, year in enumerate(years):
                    for element_position, element in enumerate(elements):
                        rows.append({
                            "flow_scenario": flow_scenario, "chemistry_scenario": scenario,
                            "chemistry": chemistry, "flow": flow, "year": int(year),
                            "element": element,
                            "tonnes": totals[year_position, element_position],
                            "p2.5": bands[year_position, element_position, 0],
                            "p97.5": bands[year_position, element_position, 1],
                        })
                print(f"  {flow:<8} {scenario} {chemistry:<9} done "
                      f"({time.time()-started:5.0f}s)")

            # The share that has no composition, reported rather than dropped.
            for year in years:
                missing = 0.0
                for segment in segments:
                    group = groups[segment]
                    for chemistry in materials.battery_chemistry_scenarios[scenario][group]:
                        if chemistry in named:
                            continue
                        missing = max(missing, chemistry_share(
                            params, scenario, group, chemistry, year))
                gaps.append({"flow": flow, "chemistry_scenario": scenario,
                             "year": int(year), "max_share_without_composition": missing})

    result = pd.DataFrame(rows)
    gap_frame = pd.DataFrame(gaps).drop_duplicates()
    saved = save_many(battery_material_flows=result,
                      battery_chemistry_gaps=gap_frame, root=PROJECT_ROOT)
    print(f"\n{len(result):,} rows | saved {saved}")
    return {"saved": saved, "rows": len(result)}


if __name__ == "__main__":
    main()
