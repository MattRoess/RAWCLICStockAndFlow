"""
test_m33_cohort_truncation_fix.py
=====================================
Verifies the M33 fix: `starting_stock_by_cohort_lookup` vintages OLDER than
`years.min()` used to be silently dropped (never read at all). This test builds a
starting-stock lookup with real mass at vintages before `years.min()`, then confirms:
  1. That mass is NOT zero in the output (the bug, if reintroduced, would show up as
     year-1 stock/out_survival being exactly `inflow` only, missing the pre-existing
     stock).
  2. The scalar function and the vectorized engine (n_draws=1) still match exactly --
     the fix was applied identically to both.
  3. A LONG lifetime (scale_lambda=25, larger than any point estimate in current use)
     makes the old cohorts' survival meaningfully non-negligible -- this is exactly the
     "would silently lose real stock" scenario the original M33 finding warned about,
     now fixed rather than just documented as low-risk.

Run from the project root: python3 test_m33_cohort_truncation_fix.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import src.flowdriven_model as fdm

np.random.seed(0)

# years starts at 1975 -- mirrors 03_02_adjustedflows.py's real setup where
# starting_stock_by_cohort_lookup (from stage 02) has vintages back to ~1955.
years = np.arange(1975, 2011)
t_end = int(years.max())

rows = []
for y in years:
    rows.append({"Region": "EUR", "Drive Train": "BEV", "Segment": "A", "year": int(y), "value": 10.0})
inflow_df = pd.DataFrame(rows)

lifetime_by_drv = {"BEV": {"shape_k": 3.0, "scale_lambda": 25.0}}  # deliberately LONG
unknown_whereabouts_share = {"BEV": 0.1}
export_share_by_drivetrain = {"BEV": 0.05}

# Starting stock with REAL mass at vintages older than years.min()=1975 -- exactly the
# case M33 flagged: stage 02's cohort matrix goes back further than 03_01's BACKCAST_
# START_YEAR.
starting_stock_by_cohort_lookup = {
    ("EUR", "BEV", "A"): {
        **{int(y): 100.0 for y in range(1955, 1975)},  # 20 years of pre-1975 vintage stock
        **{int(y): 0.0 for y in range(1975, 2011)},     # no additional stock within years range
    }
}

mismatches = []

for outflow_timing in ["post_inflow", "pre_inflow"]:
    scalar_result = fdm.run_flow_driven_model_with_outflow_disaggregation(
        df=inflow_df, years=years, t_end=t_end,
        lifetime_by_drv=lifetime_by_drv,
        export_r_by_drv={"BEV": np.ones(3)},
        age_bins=[(0, 4), (5, 9), (10, 60)],
        unknown_whereabouts_share=unknown_whereabouts_share,
        export_share_by_drivetrain=export_share_by_drivetrain,
        starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
        outflow_value_col="value", year_col="year", inflow_col="value",
        group_cols=["Region", "Drive Train", "Segment"],
        outflow_timing=outflow_timing,
        lifetime_change_by_drv=None, segment_shares_by_drv={},
        stock_modifier_2027=1.0,
    )
    flows_scalar = scalar_result["flows_df"]

    # --- Check 1: pre-1975 stock is NOT dropped -------------------------------------
    # Total pre-existing stock = 20 years x 100.0 = 2000.0. At year 1975 (first
    # simulated year), before any of that stock has had a chance to fully retire, the
    # reported `stock` (post-outflow) should be well above just that single year's
    # inflow (10.0) -- if the bug were present, pre-1975 vintages would never even enter
    # `stock_prev`, and year-1975 stock would be ~inflow-sized, not ~thousands-sized.
    first_year_stock = flows_scalar.loc[flows_scalar["year"] == 1975, "stock"].iloc[0]
    if first_year_stock < 500.0:  # generous threshold -- bug would show ~10, fix shows ~1900+
        mismatches.append((outflow_timing, "pre-1975 stock appears dropped", first_year_stock))

    # --- Check 2: vectorized engine (n_draws=1) matches scalar exactly --------------
    mc_result = fdm.run_flow_driven_model_monte_carlo(
        df=inflow_df, years=years, t_end=t_end,
        lifetime_by_drv=lifetime_by_drv,
        unknown_whereabouts_share=unknown_whereabouts_share,
        export_share_by_drivetrain=export_share_by_drivetrain,
        starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
        n_draws=1, lifetime_scale_lambda_relative_spread=None,
        unknown_whereabouts_share_std=None, export_share_std=None,
        outflow_timing=outflow_timing, lifetime_change_by_drv=None,
        stock_modifier_2027=1.0, seed=42, chunk_size=20000, collect_per_year=True,
        verbose=False,
    )
    key = ("EUR", "BEV", "A")
    mc_group = mc_result["by_group"][key]
    mc_years = mc_group["years"]

    for metric_scalar, metric_mc in [
        ("out_survival", "per_year_survival"),
        ("out_export", "per_year_export"),
        ("out_unknown", "per_year_unknown"),
    ]:
        scalar_vals = flows_scalar.set_index("year")[metric_scalar].reindex(mc_years).values
        mc_vals = mc_group[metric_mc][0]
        if not np.allclose(scalar_vals, mc_vals, atol=1e-6, rtol=1e-6):
            maxdiff = np.nanmax(np.abs(scalar_vals - mc_vals))
            mismatches.append((outflow_timing, "scalar vs vectorized mismatch", metric_scalar, maxdiff))

if mismatches:
    print("FAILURES:")
    for m in mismatches:
        print(" ", m)
    sys.exit(1)
else:
    print(
        "PASS: pre-1975 starting stock is no longer silently dropped (M33 fixed), "
        "and the scalar/vectorized engines still match exactly under this scenario."
    )
