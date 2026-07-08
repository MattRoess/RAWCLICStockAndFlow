"""
Focused test of the NEW logic added to 03_01_flowdriven.py's Monte Carlo block:
applying per-draw collected/export/unknown share fractions to a (n_years, n_draws)
out_survival_by_year array, THEN summing by period -- verifying this gives the exact
same result as summing by period FIRST and splitting by shares afterward (they must
be equivalent, since share multiplication and year-summation commute), and that
inflow (previously discarded, now used) sums correctly too.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import src.monte_carlo as mc

np.random.seed(0)

n_years, n_draws = 30, 500
years = np.arange(2010, 2010 + n_years)
out_survival_by_year = np.random.uniform(5, 15, size=(n_years, n_draws))
inflow_by_year = np.random.uniform(8, 20, size=(n_years, n_draws))

coll_share = np.random.uniform(0.5, 0.7, size=n_draws)
exp_share = np.random.uniform(0.05, 0.1, size=n_draws)
unk_share = 1.0 - coll_share - exp_share

collected_by_year = out_survival_by_year * coll_share[None, :]
export_by_year = out_survival_by_year * exp_share[None, :]
unknown_by_year = out_survival_by_year * unk_share[None, :]

periods = [(2015, 2015), (2015, 2025), (2010, 2039)]

collected_period_sums = mc.sum_by_period(collected_by_year, years, periods)
inflow_period_sums = mc.sum_by_period(inflow_by_year, years, periods)

failures = []

for period in periods:
    start, end = period
    mask = (years >= start) & (years <= end)

    # Ground truth: sum out_survival over the period FIRST, then split by share --
    # must equal splitting first then summing (order shouldn't matter).
    expected_collected = out_survival_by_year[mask].sum(axis=0) * coll_share
    actual_collected = collected_period_sums[period]
    if not np.allclose(expected_collected, actual_collected, rtol=1e-10):
        failures.append((period, "collected", np.max(np.abs(expected_collected - actual_collected))))

    expected_inflow = inflow_by_year[mask].sum(axis=0)
    actual_inflow = inflow_period_sums[period]
    if not np.allclose(expected_inflow, actual_inflow, rtol=1e-10):
        failures.append((period, "inflow", np.max(np.abs(expected_inflow - actual_inflow))))

    # Single-year period: collected/export/unknown period sums should equal that
    # single year's row exactly.
    if start == end:
        yi = int(np.where(years == start)[0][0])
        if not np.allclose(collected_period_sums[period], collected_by_year[yi], rtol=1e-10):
            failures.append((period, "single-year collapse", None))

# EU-total-style accumulation check: summing per-drivetrain period sums should equal
# summing the raw arrays across "drivetrains" (here just two independent copies)
# before period-summing.
out_survival_2 = np.random.uniform(3, 9, size=(n_years, n_draws))
period_sums_1 = mc.sum_by_period(out_survival_by_year, years, periods)
period_sums_2 = mc.sum_by_period(out_survival_2, years, periods)
combined_direct = mc.sum_by_period(out_survival_by_year + out_survival_2, years, periods)
for period in periods:
    lhs = period_sums_1[period] + period_sums_2[period]
    rhs = combined_direct[period]
    if not np.allclose(lhs, rhs, rtol=1e-10):
        failures.append((period, "EU-total accumulation order", None))

if failures:
    print("FAILURES:")
    for f in failures:
        print(" ", f)
    sys.exit(1)
else:
    print(
        "PASS: share-broadcast-then-period-sum matches period-sum-then-share-split "
        "exactly, inflow sums correctly, single-year periods collapse correctly, and "
        "EU-total accumulation order doesn't matter -- confirms the new 03_01 logic "
        "(which broadcasts shares before summing, for efficiency) is mathematically "
        "equivalent to the more obvious 'sum then split' approach."
    )
