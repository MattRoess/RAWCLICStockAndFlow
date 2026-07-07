"""
test_cohort_flow_mc_triangular.py
=====================================
Verifies the generic engine's (1) Triangular scale_lambda sampling stays
within [point*(1-spread), point*(1+spread)] and centers on point, and (2) the
dual scalar-vs-dict acceptance for `lifetime_scale_lambda_relative_spread_by_
entity` (a single float = uniform "general" scenario; a dict = per-entity
"specific" scenario). Run from the project root:

    python3 test_cohort_flow_mc_triangular.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import src.cohort_flow_mc as cfmc

np.random.seed(0)

years_all = np.arange(2010, 2036)
rows = []
for drv in ["BEV", "Diesel"]:
    for y in years_all:
        rows.append({"Region": "EUR", "Drive Train": drv, "year": int(y), "value": 100.0})
inflow_df = pd.DataFrame(rows)

lifetime_by_drv = {"BEV": {"shape_k": 3.0, "scale_lambda": 13.0}, "Diesel": {"shape_k": 3.0, "scale_lambda": 13.0}}
share_a = {"BEV": 0.05, "Diesel": 0.08}
share_b = {"BEV": 0.15, "Diesel": 0.10}
starting_stock_by_cohort_lookup = {}

n_draws = 20000

print("=== Test 1: per-entity dict spread (BEV=0.2, Diesel=0.05) ===")
result = cfmc.run_cohort_flow_monte_carlo(
    df=inflow_df, years=years_all, t_end=int(years_all.max()),
    group_cols=["Region", "Drive Train"], entity_key_col="Drive Train",
    lifetime_by_entity=lifetime_by_drv,
    share_a_by_entity=share_a, share_b_by_entity=share_b,
    starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
    n_draws=n_draws,
    lifetime_scale_lambda_relative_spread_by_entity={"BEV": 0.2, "Diesel": 0.05},
    seed=7,
)
# Recover implied scale_lambda draws is not directly exposed, but we can sanity check
# via the cumulative_survival spread: BEV (wider lifetime spread) should show more
# relative variability in cumulative_survival than Diesel (narrower spread), all else equal.
bev_cv = result["by_group"][("EUR", "BEV")]["cumulative_survival"].std() / result["by_group"][("EUR", "BEV")]["cumulative_survival"].mean()
diesel_cv = result["by_group"][("EUR", "Diesel")]["cumulative_survival"].std() / result["by_group"][("EUR", "Diesel")]["cumulative_survival"].mean()
print(f"BEV coefficient of variation (spread=0.2): {bev_cv:.4f}")
print(f"Diesel coefficient of variation (spread=0.05): {diesel_cv:.4f}")
assert bev_cv > diesel_cv, "Expected BEV (wider spread) to show more relative variability than Diesel"
print("PASS: per-entity dict spread produces distinct, correctly-ordered variability.\n")

print("=== Test 2: single scalar spread applied uniformly (general mode) ===")
result_general = cfmc.run_cohort_flow_monte_carlo(
    df=inflow_df, years=years_all, t_end=int(years_all.max()),
    group_cols=["Region", "Drive Train"], entity_key_col="Drive Train",
    lifetime_by_entity=lifetime_by_drv,
    share_a_by_entity=share_a, share_b_by_entity=share_b,
    starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
    n_draws=n_draws,
    lifetime_scale_lambda_relative_spread_by_entity=0.15,  # single float, not a dict
    seed=7,
)
bev_cv_g = result_general["by_group"][("EUR", "BEV")]["cumulative_survival"].std() / result_general["by_group"][("EUR", "BEV")]["cumulative_survival"].mean()
diesel_cv_g = result_general["by_group"][("EUR", "Diesel")]["cumulative_survival"].std() / result_general["by_group"][("EUR", "Diesel")]["cumulative_survival"].mean()
print(f"BEV coefficient of variation (general spread=0.15): {bev_cv_g:.4f}")
print(f"Diesel coefficient of variation (general spread=0.15): {diesel_cv_g:.4f}")
assert abs(bev_cv_g - diesel_cv_g) < 0.03, "Expected similar variability under a uniform general spread"
print("PASS: a single scalar spread is applied uniformly across all entities.\n")

print("=== Test 3: Triangular bounds sanity (point=13.0, spread=0.2 -> support [10.4, 15.6]) ===")
rng = np.random.default_rng(123)
draws = cfmc._sample_relative_triangular_scale(13.0, 0.2, 100000, rng)
print(f"min={draws.min():.3f} max={draws.max():.3f} mean={draws.mean():.3f} (expected mean ~13.0)")
assert draws.min() >= 13.0 * 0.8 - 1e-6
assert draws.max() <= 13.0 * 1.2 + 1e-6
assert abs(draws.mean() - 13.0) < 0.05
print("PASS: Triangular draws stay within [point*(1-spread), point*(1+spread)] and center on point.\n")

print("ALL TESTS PASSED")
