"""
test_flowdriven_mc_regression.py
====================================
Regression test: the vectorized Monte Carlo wrapper
(`fdm.run_flow_driven_model_monte_carlo`), called with `n_draws=1` and no
spread parameters, must reproduce `fdm.run_flow_driven_model_with_outflow_
disaggregation`'s (the scalar, ground-truth function) `flows_df` numbers
EXACTLY. Run this from the project root:

    python3 test_flowdriven_mc_regression.py

Covers both `outflow_timing` modes ("post_inflow", "pre_inflow") and both
with/without a `lifetime_change_by_drv` override, since those are the two
axes of behavior that differ structurally in the recurrence.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import src.flowdriven_model as fdm

np.random.seed(0)

years_all = np.arange(1975, 2036)
rows = []
for drv in ["BEV", "Diesel"]:
    for seg in ["A", "B"]:
        for y in years_all:
            base = 100.0 if drv == "BEV" else 200.0
            rows.append({"Region": "EUR", "Drive Train": drv, "Segment": seg, "year": int(y), "value": base + 5 * (y - 1975)})
inflow_df = pd.DataFrame(rows)

lifetime_by_drv = {"BEV": {"shape_k": 3.0, "scale_lambda": 13.0}, "Diesel": {"shape_k": 3.0, "scale_lambda": 13.0}}
unknown_whereabouts_share = {"BEV": 0.15, "Diesel": 0.10}
export_share_by_drivetrain = {"BEV": 0.05, "Diesel": 0.08}

starting_stock_by_cohort_lookup = {}
for drv in ["BEV", "Diesel"]:
    for seg in ["A", "B"]:
        starting_stock_by_cohort_lookup[("EUR", drv, seg)] = {int(y): 50.0 for y in range(1975, 2006)}

lifetime_change_by_drv = {
    "Diesel": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 9.0},
    "BEV": {"start_year": 2027, "shape_k": 3.0, "scale_lambda": 17.0},
}

mismatches = []

for outflow_timing in ["post_inflow", "pre_inflow"]:
    for use_override in [False, True]:
        years = years_all.copy()
        t_end = int(years.max())

        scalar_result = fdm.run_flow_driven_model_with_outflow_disaggregation(
            df=inflow_df, years=years, t_end=t_end,
            lifetime_by_drv=lifetime_by_drv,
            export_r_by_drv={"BEV": np.ones(3), "Diesel": np.ones(3)},
            age_bins=[(0, 4), (5, 9), (10, 30)],
            unknown_whereabouts_share=unknown_whereabouts_share,
            export_share_by_drivetrain=export_share_by_drivetrain,
            starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
            outflow_value_col="value", year_col="year", inflow_col="value",
            group_cols=["Region", "Drive Train", "Segment"],
            outflow_timing=outflow_timing,
            lifetime_change_by_drv=lifetime_change_by_drv if use_override else None,
            segment_shares_by_drv={},
            stock_modifier_2027=1.0,
        )
        flows_scalar = scalar_result["flows_df"]

        mc_result = fdm.run_flow_driven_model_monte_carlo(
            df=inflow_df, years=years, t_end=t_end,
            lifetime_by_drv=lifetime_by_drv,
            unknown_whereabouts_share=unknown_whereabouts_share,
            export_share_by_drivetrain=export_share_by_drivetrain,
            starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
            n_draws=1,
            lifetime_scale_lambda_relative_spread=None,
            unknown_whereabouts_share_std=None,
            export_share_std=None,
            outflow_timing=outflow_timing,
            lifetime_change_by_drv=lifetime_change_by_drv if use_override else None,
            stock_modifier_2027=1.0,
            seed=42,
            chunk_size=20000,
            collect_per_year=True,
        )

        for drv in ["BEV", "Diesel"]:
            for seg in ["A", "B"]:
                key = ("EUR", drv, seg)
                sub_scalar = flows_scalar[
                    (flows_scalar["Region"] == "EUR") & (flows_scalar["Drive Train"] == drv) & (flows_scalar["Segment"] == seg)
                ].sort_values("year")

                mc_group = mc_result["by_group"][key]
                mc_years = mc_group["years"]

                for metric_scalar, metric_mc in [
                    ("out_survival", "per_year_survival"),
                    ("out_export", "per_year_export"),
                    ("out_unknown", "per_year_unknown"),
                ]:
                    scalar_vals = sub_scalar.set_index("year")[metric_scalar].reindex(mc_years).values
                    mc_vals = mc_group[metric_mc][0]  # draw 0
                    if not np.allclose(scalar_vals, mc_vals, atol=1e-8, rtol=1e-8):
                        maxdiff = np.nanmax(np.abs(scalar_vals - mc_vals))
                        mismatches.append((outflow_timing, use_override, drv, seg, metric_scalar, maxdiff))

if mismatches:
    print("MISMATCHES FOUND:")
    for m in mismatches:
        print(m)
    sys.exit(1)
else:
    print("PASS: vectorized MC engine (n_draws=1) matches the scalar reference exactly "
          "(within 1e-8) across both timing modes, with and without lifetime_change_by_drv override.")
