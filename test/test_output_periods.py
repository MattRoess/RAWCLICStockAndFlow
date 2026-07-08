"""
test_output_periods.py
==========================
Verifies the new `output_periods` feature in `cohort_flow_mc.run_cohort_flow_
monte_carlo`: arbitrary (start, end) year-range outputs for flows (survival/
export/unknown/collected), inflow, and stock (end-of-period snapshot, per-year
series, sum-over-period), specified up front.

Cross-checks everything against `collect_per_year=True`'s ground-truth per-year
arrays (summing the relevant years by hand), rather than trusting the new
accumulation logic on its own.

Run from the project root: python3 test_output_periods.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import src.cohort_flow_mc as cfmc

np.random.seed(0)

years = np.arange(2010, 2041)
rows = []
for drv in ["BEV", "Diesel"]:
    for y in years:
        rows.append({"Region": "EUR", "Drive Train": drv, "year": int(y), "value": 100.0 + float(y - 2010)})
inflow_df = pd.DataFrame(rows)

lifetime_by_drv = {"BEV": {"shape_k": 3.0, "scale_lambda": 13.0}, "Diesel": {"shape_k": 3.0, "scale_lambda": 13.0}}
share_a = {"BEV": 0.05, "Diesel": 0.08}
share_b = {"BEV": 0.15, "Diesel": 0.10}
starting_stock_by_cohort_lookup = {
    ("EUR", "BEV"): {int(y): 20.0 for y in range(1990, 2010)},
    ("EUR", "Diesel"): {int(y): 30.0 for y in range(1990, 2010)},
}

n_draws = 500
periods = [(2020, 2020), (2020, 2030), (2010, 2040)]

# Ground truth: collect_per_year=True, single implicit whole-horizon period.
gt = cfmc.run_cohort_flow_monte_carlo(
    df=inflow_df, years=years, t_end=int(years.max()),
    group_cols=["Region", "Drive Train"], entity_key_col="Drive Train",
    lifetime_by_entity=lifetime_by_drv, share_a_by_entity=share_a, share_b_by_entity=share_b,
    starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
    n_draws=n_draws, seed=99, chunk_size=200, collect_per_year=True, verbose=False,
)

# New: output_periods.
new = cfmc.run_cohort_flow_monte_carlo(
    df=inflow_df, years=years, t_end=int(years.max()),
    group_cols=["Region", "Drive Train"], entity_key_col="Drive Train",
    lifetime_by_entity=lifetime_by_drv, share_a_by_entity=share_a, share_b_by_entity=share_b,
    starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
    n_draws=n_draws, seed=99, chunk_size=200, output_periods=periods, verbose=False,
)

failures = []

for key in [("EUR", "BEV"), ("EUR", "Diesel")]:
    gt_years = gt["by_group"][key]["years"]
    gt_survival = gt["by_group"][key]["per_year_survival"]  # (n_draws, n_years)
    gt_a = gt["by_group"][key]["per_year_a"]
    gt_b = gt["by_group"][key]["per_year_b"]
    gt_collected = gt["by_group"][key]["per_year_collected"]
    year_to_idx = {int(y): i for i, y in enumerate(gt_years)}

    for (start, end) in periods:
        idxs = [year_to_idx[y] for y in gt_years if start <= y <= end]
        expected_survival = gt_survival[:, idxs].sum(axis=1)
        expected_a = gt_a[:, idxs].sum(axis=1)
        expected_b = gt_b[:, idxs].sum(axis=1)
        expected_collected = gt_collected[:, idxs].sum(axis=1)

        period_result = new["by_group"][key]["periods"][(start, end)]

        for label, expected, actual in [
            ("survival", expected_survival, period_result["cumulative_survival"]),
            ("a/export", expected_a, period_result["cumulative_a"]),
            ("b/unknown", expected_b, period_result["cumulative_b"]),
            ("collected", expected_collected, period_result["cumulative_collected"]),
        ]:
            if not np.allclose(expected, actual, atol=1e-6, rtol=1e-6):
                failures.append((key, (start, end), label, np.max(np.abs(expected - actual))))

        # Inflow: deterministic, cross-check against a plain Python sum.
        sub = inflow_df[(inflow_df["Region"] == key[0]) & (inflow_df["Drive Train"] == key[1])]
        expected_inflow = sub[(sub["year"] >= start) & (sub["year"] <= end)]["value"].sum()
        actual_inflow = period_result["cumulative_inflow"]
        if not np.allclose(expected_inflow, actual_inflow, atol=1e-6):
            failures.append((key, (start, end), "inflow", abs(expected_inflow - actual_inflow[0])))
        if actual_inflow.std() != 0.0:
            failures.append((key, (start, end), "inflow should be deterministic (std=0)", actual_inflow.std()))

        # stock_per_year should have exactly the years in [start, end].
        expected_year_set = {y for y in gt_years if start <= y <= end}
        actual_year_set = set(period_result["stock_per_year"].keys())
        if expected_year_set != actual_year_set:
            failures.append((key, (start, end), "stock_per_year year set mismatch", expected_year_set ^ actual_year_set))

        # stock_end_of_period should equal stock_per_year[end].
        if not np.allclose(period_result["stock_end_of_period"], period_result["stock_per_year"][end], atol=1e-9):
            failures.append((key, (start, end), "stock_end_of_period != stock_per_year[end]", None))

        # stock_sum_over_period should equal sum of stock_per_year values.
        summed = sum(period_result["stock_per_year"].values())
        if not np.allclose(period_result["stock_sum_over_period"], summed, atol=1e-6, rtol=1e-6):
            failures.append((key, (start, end), "stock_sum_over_period != sum(stock_per_year)", None))

        # Single-year period: end_of_period, sum_over_period, and the one stock_per_year
        # entry should all be identical.
        if start == end:
            spy_val = period_result["stock_per_year"][start]
            if not (
                np.allclose(period_result["stock_end_of_period"], spy_val, atol=1e-9)
                and np.allclose(period_result["stock_sum_over_period"], spy_val, atol=1e-9)
            ):
                failures.append((key, (start, end), "single-year period: end/sum/per_year not all equal", None))

# Backward compatibility: default (no output_periods) still produces flat top-level aliases.
default_result = cfmc.run_cohort_flow_monte_carlo(
    df=inflow_df, years=years, t_end=int(years.max()),
    group_cols=["Region", "Drive Train"], entity_key_col="Drive Train",
    lifetime_by_entity=lifetime_by_drv, share_a_by_entity=share_a, share_b_by_entity=share_b,
    starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
    n_draws=n_draws, seed=99, chunk_size=200, verbose=False,
)
whole = (int(years.min()), int(years.max()))
for key in [("EUR", "BEV"), ("EUR", "Diesel")]:
    g = default_result["by_group"][key]
    if "cumulative_survival" not in g:
        failures.append((key, "default", "flat alias missing", None))
    elif not np.allclose(g["cumulative_survival"], g["periods"][whole]["cumulative_survival"]):
        failures.append((key, "default", "flat alias != periods[whole_horizon]", None))
if "cumulative_survival" not in default_result["total"]:
    failures.append(("total", "default", "flat alias missing at total level", None))

if failures:
    print("FAILURES:")
    for f in failures:
        print(" ", f)
    sys.exit(1)
else:
    print(
        "PASS: output_periods matches hand-summed per-year ground truth for all "
        "flow metrics, inflow is correctly deterministic, stock end/sum/per-year are "
        "mutually consistent, single-year periods collapse correctly, and the default "
        "(no output_periods) case still produces backward-compatible flat aliases."
    )
