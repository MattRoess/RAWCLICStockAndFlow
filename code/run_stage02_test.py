import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import dataclasses
import numpy as np
import pandas as pd

import src.artifacts as artifacts
import src.params_schema as ps

YEARS = list(range(1990, 2041))
DRIVETRAINS = ["BEV", "Diesel"]


def build_stock_dict():
    stock_dict = {}
    for drv in DRIVETRAINS:
        base = {"BEV": 50.0, "Diesel": 300.0}[drv]
        growth = {"BEV": 1.08, "Diesel": 1.01}[drv]
        values = [base * (growth ** (y - YEARS[0])) for y in YEARS]
        df = pd.DataFrame({"stock": values}, index=pd.Index(YEARS, name="year"))
        stock_dict[("EUR", drv)] = df
    return stock_dict


def build_params(n_draws=200, output_periods=None):
    base = ps.Params()
    if output_periods is None:
        output_periods = [(1990, 2040)]
    mc = dataclasses.replace(
        base.monte_carlo, enabled=True, n_draws=n_draws, seed=7,
        output_periods=output_periods,
    )
    # model_end_year needs to reach at least our synthetic horizon's end.
    sf = dataclasses.replace(base.stock_flow, model_end_year=2040)
    return dataclasses.replace(base, monte_carlo=mc, stock_flow=sf)


def main():
    params = build_params(
        n_draws=200,
        output_periods=[(2000, 2000), (2010, 2020), (1990, 2040)],
    )
    artifacts._seed_store(params=params, stock_dict=build_stock_dict())

    import runpy
    ns = runpy.run_path(str(ROOT / "code" / "02_stockdriven.py"), run_name="stage02_test")
    result = ns["main"]()
    print("\n=== main() completed ===")
    print("Returned keys:", list(result.keys()))

    summary = artifacts._STORE["mc_stage02_period_summary"]
    print("\nperiod_summary_by_drv keys:", list(summary["by_drivetrain"].keys()))
    print("eu_total periods:", list(summary["eu_total"].keys()))
    bev = summary["by_drivetrain"]["BEV"]
    p = (2010, 2020)
    print(f"\nBEV period {p}:")
    print("  cumulative_inflow mean:", bev[p]["cumulative_inflow"]["mean"])
    print("  cumulative_out_survival mean:", bev[p]["cumulative_out_survival"]["mean"])
    print("  stock_end_of_period mean:", bev[p]["stock_end_of_period"]["mean"], "std:", bev[p]["stock_end_of_period"]["std"])
    print("  stock_sum_over_period mean:", bev[p]["stock_sum_over_period"]["mean"])
    print("  stock_per_year years:", sorted(bev[p]["stock_per_year"].keys()))

    single_year = (2000, 2000)
    sy = bev[single_year]
    print(f"\nBEV single-year period {single_year}:")
    print("  end == sum == per_year[2000]?",
          sy["stock_end_of_period"]["mean"] == sy["stock_sum_over_period"]["mean"] == sy["stock_per_year"][2000]["mean"])


if __name__ == "__main__":
    main()
