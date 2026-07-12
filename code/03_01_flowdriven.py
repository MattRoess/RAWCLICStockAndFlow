"""
03_01_flowdriven.py
====================

Stage 03, part 1: takes stage 02's aggregate stock-driven results and:

  1. Splits each drivetrain's survival outflow into "collected" / "unknown fate" /
     "export" sub-flows (`split_outflows_collected_unknown_export`).
  2. Splits the aggregate "Hybrid" drivetrain into HEV/PHEV, and "Liquids" into
     Diesel/Petrol, using time-varying EEA-derived split ratios.
  3. Disaggregates each drivetrain's stock/flows across 12 vehicle segments
     (A-F, JA-JF) using EEA-derived segment shares.
  4. Builds a "tracker" keyed by (region, drivetrain) with a materials-ready key
     (segment code + drivetrain-technology prefix) attached to every row.
  5. SEPARATELY, reconstructs a segment-level starting stock at 2005 and backcasts a
     synthetic 1975-2005 inflow history for it, then re-runs the whole system as a
     **flow-driven** model (inflow is now the prescribed/independent variable; stock and
     outflow are computed FORWARD from it) -- the opposite paradigm from stage 02's
     stock-driven model. Compares the two paradigms' outflow totals as a sanity check.

Persists `matrices_by_key` (now enriched with collected/export/unknown sub-matrices),
`disaggregated`, `segment_shares_ext`, `liquids_shares_ext`, `tracker_keyed`,
`synthetic_pre_2005_inflows`, `starting_stock_2005_segments`, `flows_03`,
`seg_share_by_drv`.

FIXES APPLIED THIS ROUND -- NOW VERIFIED END-TO-END (not just mechanical this time)
--------------------------------------------------------------------------------------
`src/flowdriven_model.py`, `src/disaggregation.py`, and a synthetic EEA dataset are now
available, so this stage was actually RUN for the first time, against synthetic
REMIND+export+EEA data, through the full `00->01->02->03_01` chain. Two real bugs were
found this way (not from reading the code, from running it) and fixed:

- **[NEW BUG, found via this run]**: `add_keys_to_tracker_dict` (in
  `disaggregation.py`) returns `missing` as a pandas DataFrame (always -- even when
  empty), not a list. This file's own `if missing:` check on that DataFrame raises
  `ValueError: The truth value of a DataFrame is ambiguous` UNCONDITIONALLY --
  reproduced directly, this line would fail on every run regardless of whether
  anything was actually missing. Fixed: `if not missing.empty:`.
- Dataclass params access: `params["02_stock_flow"]` -> `params.stock_flow`, etc. --
  same pattern as stages 00-02.
- Project-root resolution: upward-searching `_find_project_root`.
- `root=PROJECT_ROOT` threaded into every `load_many`/`save_many` call.
- `_plot_stockdriven_vs_flowdriven_outflow`'s `plt.show()` replaced with a saved PNG
  (`data/processed/figures/03_01_stockdriven_vs_flowdriven.png`).
- `fdm.build_p02_mapped_inputs` receives `params.to_nested_dict()["02_stock_flow"]` (a
  plain dict) -- CONFIRMED necessary and correct by reading `flowdriven_model.py`'s
  actual source: `build_p02_mapped_inputs` does `p02["lifetime_by_drv"]` internally,
  which would raise `TypeError` on a `StockFlowParams` dataclass instance.

**After both fixes, the full chain runs cleanly end-to-end** (verified with synthetic
data) and saves a real diagnostic plot. See `HOW_TO_RUN_AND_VERIFY.md` for the exact
commands.

See the "EXACT DIFFERENCES FROM 03_02" note in `03_02_adjustedflows.py` for a full,
itemized comparison between this notebook and its sibling.
"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never opens an interactive window -- always saves to file
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Same fix as 01_data_prep.py (see that file for the full explanation): resolve
# relative params like `output_dir` against THIS script's own location, not the
# caller's current working directory at runtime.
SCRIPT_DIR = Path(__file__).resolve().parent

import src.flowdriven_model as fdm  # type: ignore
import src.disaggregation as disagg  # type: ignore
import src.artifacts as artifacts  # type: ignore
from src.monte_carlo import (  # type: ignore
    Normal, summarize_distribution, sum_by_period, sensitivity_correlations, plot_tornado,
)
from src.stockflow_model import build_backcast_state, run_cohort_survival_monte_carlo  # type: ignore

load_many = artifacts.load_many
save_many = artifacts.save_many
artifact_status = artifacts.artifact_status

build_hev_phev_split = disagg.build_hev_phev_split
mask_inflow_before_introduction_year = disagg.mask_inflow_before_introduction_year
build_liquids_split_wide = disagg.build_liquids_split_wide
build_segment_share_wide = disagg.build_segment_share_wide
disaggregate_model_to_segments = disagg.disaggregate_model_to_segments
prepare_eea_share_tables = disagg.prepare_eea_share_tables
split_hybrid_inflow_afterwards = disagg.split_hybrid_inflow_afterwards
split_hybrid_outflows_afterwards = disagg.split_hybrid_outflows_afterwards
split_liquids_inflow_afterwards = disagg.split_liquids_inflow_afterwards
split_liquids_outflows_afterwards = disagg.split_liquids_outflows_afterwards
split_outflows_collected_unknown_export = disagg.split_outflows_collected_unknown_export
build_tracker_from_disaggregated = disagg.build_tracker_from_disaggregated
add_keys_to_tracker_dict = disagg.add_keys_to_tracker_dict


# ---------------------------------------------------------------------------
# BEV segment-share assumption (business-as-usual)
# ---------------------------------------------------------------------------
# The share of NEW BEV inflow going to each of the 12 vehicle segments. FLAGGED: these
# 12 numbers sum to ~1.0 (verified: 0.99999... within float tolerance) but are hardcoded
# directly in this notebook cell rather than sourced from 00_parameters.py -- consistent
# with the "should be centralized but isn't" pattern flagged repeatedly in earlier rounds
# of this review (e.g. M9, M23).
BEV_SEGMENT_SHARES_BAU = {
    "A": 0.11599, "B": 0.07235, "C": 0.17037, "D": 0.06288, "E": 0.04190, "F": 0.02945,
    "JA": 0.00341, "JB": 0.07498, "JC": 0.25116, "JD": 0.15162, "JE": 0.02108, "JF": 0.00481,
}
segment_shares_by_drv = {"BEV": BEV_SEGMENT_SHARES_BAU}


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------
# FLAGGED (DRY violation, verified across both notebooks): `make_shades`,
# `add_clean_legends`, and `build_segment_stock_table` are each (re)defined 2-4 times
# across `03_01_flowdriven.ipynb` and `03_02_adjustedflows.ipynb` (grep-confirmed: 4
# copies of `make_shades`/`make_shades_exports`, 3 of `add_clean_legends`, 3 of
# `build_segment_stock_table`). They're defined once here; if you already have a shared
# `src/plotting.py` (referenced by 03_02) that contains equivalents, these should move
# there instead of staying duplicated per-notebook.
def make_shades(base_color: str, n: int) -> list[np.ndarray]:
    """Generate `n` shades of `base_color`, fading toward white."""
    base = np.array(to_rgb(base_color))
    white = np.array([1, 1, 1])
    return [base * (1 - (i / max(n - 1, 1)) * 0.75) + white * ((i / max(n - 1, 1)) * 0.75) for i in range(n)]


SEGMENT_ORDER = ["A", "B", "C", "D", "E", "F", "JA", "JB", "JC", "JD", "JE", "JF"]
BASE_DRV_COLORS = {"BEV": "#1b9e77", "HEV": "#f2d27d", "PHEV": "#d9b44a", "Diesel": "#4c78a8", "Petrol": "#9ecae1"}


def build_segment_stock_table(flows_df: pd.DataFrame, drv_order: list[str], year_start: int = 2010, year_end: int | None = None):
    """Pivot a flows table into a (year x drivetrain-segment) stock table for area plotting."""
    segment_shades = {drv: dict(zip(SEGMENT_ORDER, make_shades(col, len(SEGMENT_ORDER)))) for drv, col in BASE_DRV_COLORS.items()}
    tmp = flows_df.groupby(["year", "Drive Train", "Segment"], as_index=False)["stock"].sum()
    tmp = tmp[tmp["year"].between(year_start, year_end)] if year_end is not None else tmp[tmp["year"] >= year_start]
    pivot = tmp.pivot_table(index="year", columns=["Drive Train", "Segment"], values="stock", aggfunc="sum", fill_value=0.0).sort_index()
    ordered_cols, ordered_colors = [], []
    for drv in drv_order:
        if drv not in pivot.columns.get_level_values(0):
            continue
        for seg in SEGMENT_ORDER:
            if (drv, seg) in pivot.columns:
                ordered_cols.append((drv, seg))
                ordered_colors.append(segment_shades.get(drv, {}).get(seg, "#999999"))
    pivot = pivot.reindex(columns=ordered_cols, fill_value=0.0)
    pivot.columns = [f"{drv}-{seg}" for drv, seg in pivot.columns]
    return pivot, ordered_colors


def main() -> dict[str, Any]:
    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    loaded = load_many("params", "matrices_by_key", "stock_dict", root=PROJECT_ROOT)
    params = loaded["params"]
    matrices_by_key = loaded["matrices_by_key"]
    stock_dict = loaded["stock_dict"]

    p01 = params.data_prep
    p02 = params.stock_flow
    p03 = params.disaggregation
    p04 = params.materials
    p06 = params.visualization

    # [IMPORTANT, flagged not silently changed]: `fdm.build_p02_mapped_inputs` is a
    # function inside `src/flowdriven_model.py`, which has NOT been shared -- I do not
    # know whether its internals do dict-style access (`p02["lifetime_by_drv"]`) or
    # attribute access. The ORIGINAL code passed a raw dict here (confirmed by this same
    # file's own `p02["EXPORT_SHARE_BY_DRV"]` access a few lines below, pre-dataclass-
    # refactor). To avoid silently breaking `fdm.py` by handing it a `StockFlowParams`
    # dataclass instance it was never written to accept, this passes the DICT VIEW
    # (`Params.to_nested_dict()`, built for exactly this transition purpose -- see
    # `src/params_schema.py`) into every `fdm`/`disagg` call, while the rest of THIS
    # script uses normal dataclass attribute access. Once `flowdriven_model.py` and
    # `disaggregation.py` are shared and reviewed, this indirection can likely be
    # removed in favor of updating those modules to accept the dataclass directly.
    p02_dict = params.to_nested_dict()["02_stock_flow"]

    # [FIXED, same bug class as 01_data_prep.py's input_dir fix this round]:
    # `output_dir` ("../data/processed/") is relative -- resolved against SCRIPT_DIR
    # (this file's own location) instead of the caller's cwd, so
    # `prepare_eea_share_tables`'s `output_dir + "EEA_final_data.csv"` lookup works
    # regardless of where this script is invoked from.
    output_dir = str((SCRIPT_DIR / p03.output_dir).resolve()) + "/"
    start_year_model = int(p03.start_year_model)
    end_year_model = int(p03.end_year_model)
    YEAR_PLOT_START = int(p06.year_plot_start)
    YEAR_PLOT_END = int(p06.year_plot_end)
    segment_map = p04.segment_map
    drv_prefix_map = p04.drv_prefix_map
    materials_region = p04.region
    materials_drivetrains = tuple(p04.drivetrains)

    # FLAGGED (finding M17, now directly confirmed by usage): `export_share_by_drv` and
    # `unknown_whereabouts_share`, both declared under `params.stock_flow` in
    # 00_parameters.py, are read HERE, in stage 03 -- not in 02_stockdriven.py. This is
    # the direct confirmation of what was inferred two rounds ago from the artifact
    # registry alone.
    mapped = fdm.build_p02_mapped_inputs(
        p02_dict, drivetrains=("BEV", "HEV", "PHEV", "Diesel", "Petrol", "FCEV", "Gases", "Liquids", "Hybrid")
    )
    unknown_whereabouts_share = mapped["unknown_whereabouts_share"]
    export_share_by_drivetrain = p02.export_share_by_drv
    # FLAGGED: `mapped` is built here with the FULL 9-drivetrain tuple, but is
    # REASSIGNED later (see below) with only 5 drivetrains ("BEV","HEV","PHEV","Diesel",
    # "Petrol"). Since only `unknown_whereabouts_share` is pulled from THIS version
    # before the reassignment, and `build_p02_mapped_inputs` presumably maps parameters
    # per-drivetrain independently (no evidence of cross-drivetrain interaction), this is
    # very likely safe -- but I can't fully confirm it without `fdm.py`'s source. Worth a
    # single upfront `mapped = fdm.build_p02_mapped_inputs(p02, drivetrains=(5 drvs))`
    # instead of reassigning the same name with two different scopes partway through the
    # notebook, purely for readability.

    artifact_status()

    # -----------------------------------------------------------------------
    # EEA-derived segment/drivetrain split tables + model year range
    # -----------------------------------------------------------------------
    # [NEW] Opt-in synthetic fallback while waiting for the real EEA data. Off by
    # default (params.disaggregation.use_synthetic_eea_fallback = False) -- with the
    # real file in place, this block does nothing and prepare_eea_share_tables reads
    # it exactly as before. Only if the real file is genuinely missing AND the flag is
    # explicitly set does this generate a clearly-labeled, loudly-announced SYNTHETIC
    # placeholder so the rest of the pipeline can still be exercised. See
    # `disaggregation.py`'s `generate_synthetic_eea_data` docstring for exactly what
    # this does and doesn't mean for the resulting numbers.
    eea_path = Path(output_dir) / "EEA_final_data.csv"
    if not eea_path.exists() and p03.use_synthetic_eea_fallback:
        print(
            "=" * 70 + "\n"
            "WARNING: data/processed/EEA_final_data.csv not found. "
            "params.disaggregation.use_synthetic_eea_fallback is True, so a "
            "SYNTHETIC placeholder is being generated instead.\n"
            "EVERY number derived from this file (segment shares, HEV/PHEV split, "
            "Diesel/Petrol split, and everything downstream) is MEANINGLESS until "
            "the real file replaces it. This is for testing the pipeline's "
            "PLUMBING only, not for any real analysis.\n" + "=" * 70
        )
        disagg.generate_synthetic_eea_data(
            output_dir=output_dir,
            start_year=2005,  # EEA registration data realistically wouldn't predate this
            end_year=end_year_model,
            seed=42,
        )

    # [NEW] "EU plus 4" project convention: EU27 + Norway + Iceland (the two of the four
    # EFTA/EEA countries actually present in EEA_final_data.csv -- CH and LI are simply
    # absent from the source, nothing to explicitly exclude there). GB is explicitly
    # DROPPED (14.0% of total registration volume in the raw file -- the single largest
    # non-EU contributor, and no longer an EU market). See
    # `prepare_eea_share_tables`'s own docstring in disaggregation.py for the full
    # rationale and the documented, ACCEPTED limitation around uneven country-year
    # coverage even within this scope (NO from 2019, IS from 2018, HR from 2014).
    # [FIXED, this round -- confirmed via a real diagnostic run] Was
    # `tuple(p01.eu_countries) + ("NO", "IS")` -- `eu_countries` holds FULL COUNTRY
    # NAMES ("Austria", "Belgium", ...), but EEA_final_data.csv's own "Country"
    # column holds ISO2 codes ("AT", "BE", ...). Full names never matched ISO2
    # codes, so this silently filtered down to ~0.5% of total registration volume
    # (effectively just "NO"/"IS", which happen to already be 2-letter codes) --
    # meaning segment shares, the Diesel/Petrol split, AND the HEV/PHEV split were
    # all being computed from Norway+Iceland alone instead of the real EU-27. Now
    # uses the dedicated `eu_countries_iso2` field (same 27 countries, ISO2 form) --
    # `eu_countries` itself is untouched, since `clean_export_data` (data_prep.py)
    # separately needs it in full-name form.
    eea_country_scope = tuple(p01.eu_countries_iso2) + ("NO", "IS")
    # [FIXED, step 2 of agreed plan, same confirmed bug as build_hev_phev_split]
    # introduction_year_by_drv makes segment_shares_ext/liquids_shares_ext give a
    # hard 0 for years before a drivetrain's real introduction year, instead of the
    # old .bfill() backfilling a later real EEA share into years before it existed.
    eea_data, segment_shares_ext, liquids_shares_ext = prepare_eea_share_tables(
        output_dir=output_dir, start_year_model=start_year_model, end_year_model=end_year_model,
        country_scope=eea_country_scope,
        introduction_year_by_drv=p03.introduction_year_by_drv,
    )
    # NOTE: `start_year_model`/`end_year_model` here come from params["03_disaggregation"]
    # (1900 / 2070). But immediately below, `start_year`/`model_end_year` are recomputed
    # DIRECTLY from `matrices_by_key`'s actual data range -- a DIFFERENT pair of values,
    # under DIFFERENTLY-NAMED variables, used for everything else in this notebook. If
    # matrices_by_key's actual coverage doesn't match p03's declared window, the EEA
    # tables get built for one year range while the rest of the model uses another --
    # worth confirming these can never diverge in practice, or aligning the two.
    start_year = int(min(mats["flows_df"].index.min() for mats in matrices_by_key.values()))
    model_end_year = int(max(mats["flows_df"].index.max() for mats in matrices_by_key.values()))
    years_full = pd.Index(range(start_year, model_end_year + 1), name="Year")
    # This is (at least) the THIRD distinct "model horizon" definition seen across the
    # pipeline so far: REMIND raw data to 2150 (stage 01), model_end_year=2100 (stage 02),
    # p03's declared 1900-2070 (used only for the EEA tables above), and NOW this
    # data-derived start_year/model_end_year (used for everything else here). See
    # consolidated review finding M25.

    # -----------------------------------------------------------------------
    # Split stage-02's aggregate outflow into collected / unknown-fate / export
    # -----------------------------------------------------------------------
    # THE MATH MODEL: this is presumably a straightforward multiplicative split --
    #   out_export(t)   = out_survival(t) * export_share[drv]
    #   out_unknown(t)  = out_survival(t) * unknown_whereabouts_share[drv]
    #   out_collected(t)= out_survival(t) * (1 - export_share[drv] - unknown_whereabouts_share[drv])
    # (inferred from how the equivalent computation is done explicitly, in the open, in
    # 03_02's `disaggregated_new` construction: `out_survival * (1 - unknown_whereabouts_share)`
    # for the collected share -- I could not confirm the exact formula used inside
    # `split_outflows_collected_unknown_export` itself without `disaggregation.py`.)
    # This ALSO mutates `matrices_by_key` -- every (region, drivetrain) entry gains new
    # sub-matrices (e.g. `outflow_exp_df`), confirmed by later cells (03_02) reading
    # `mats["outflow_exp_df"]` directly from `matrices_by_key`, a key that did NOT exist
    # when `matrices_by_key` was first built in `02_stockdriven.py`.
    matrices_by_key, unknown_whereabouts_share = split_outflows_collected_unknown_export(
        matrices_by_key=matrices_by_key,
        unknown_whereabouts_share=unknown_whereabouts_share,
        export_share=export_share_by_drivetrain,
    )

    # Hybrid -> HEV/PHEV split, using a time-varying EEA-derived ratio.
    # [FIXED, step 1 of agreed plan, confirmed via real diagnostic run] Pass
    # introduction_year_by_drv so years before a drivetrain's real first sale get a
    # hard 0 share instead of the old .bfill() backfilling a later real value into
    # years before it existed (e.g. PHEV, first sold 2012, no longer shows a nonzero
    # share in 2005). See disaggregation.py's _fill_share_with_introduction_year for
    # the exact logic, including the HEV/PHEV-same-EEA-start-year case.
    split_hp = build_hev_phev_split(
        eea_data=eea_data, years_full=years_full,
        introduction_year_by_drv=p03.introduction_year_by_drv,
    )
    split_hybrid_inflow_afterwards(matrices_by_key=matrices_by_key, split_hp=split_hp, region="EUR", base_drv="Hybrid")
    split_hybrid_outflows_afterwards(matrices_by_key=matrices_by_key, split_hp=split_hp, region="EUR", base_drv="Hybrid")

    # Liquids -> Diesel/Petrol split, same idea.
    liquids_split = build_liquids_split_wide(liquids_shares_ext)
    split_liquids_inflow_afterwards(matrices_by_key=matrices_by_key, liquids_split=liquids_split, region="EUR", base_drv="Liquids")
    split_liquids_outflows_afterwards(matrices_by_key=matrices_by_key, liquids_split=liquids_split, region="EUR", base_drv="Liquids")

    # Segment disaggregation. NOTE: `allowed_drivetrains` here is exactly the 5-drivetrain
    # set (BEV, HEV, PHEV, Diesel, Petrol) -- FCEV and Gases are silently excluded from
    # segment-level (and therefore materials-stage) accounting entirely. This CONFIRMS
    # and resolves earlier open questions (consolidated review M3, M5): FCEV and Gases
    # stock exists in stage 02's matrices_by_key but never gets carried into segment-level
    # or materials modeling from this point forward in the pipeline.
    seg_share_by_drv = build_segment_share_wide(segment_shares_ext)
    disaggregated = disaggregate_model_to_segments(
        matrices_by_key=matrices_by_key, seg_share_by_drv=seg_share_by_drv, region="EUR",
        allowed_drivetrains=("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
    )

    tracker = build_tracker_from_disaggregated(disaggregated, region=materials_region, drivetrains=materials_drivetrains, include_zero=False)
    tracker_keyed, missing = add_keys_to_tracker_dict(tracker, segment_map=segment_map, drv_prefix_map=drv_prefix_map)
    # [FIXED, new bug found via actual end-to-end run this round -- not in the original
    # review]: `add_keys_to_tracker_dict` (disaggregation.py) returns `missing` as a
    # pandas DataFrame (always -- even when empty), not a list. `if missing:` on ANY
    # DataFrame raises `ValueError: The truth value of a DataFrame is ambiguous`,
    # regardless of whether it's actually empty. This line would have failed on every
    # single run, matched-keys or not -- confirmed by reproducing the exact traceback
    # before this fix. Correct check: `.empty`.
    if not missing.empty:
        print(f"WARNING: {len(missing)} tracker entries could not be matched to a materials key:\n{missing}")

    # -----------------------------------------------------------------------
    # Reconstruct 2005 starting stock by segment, backcast synthetic 1975-2005 inflow
    # -----------------------------------------------------------------------
    # THE MATH MODEL: this is a SEPARATE, SEGMENT-LEVEL backcast, independent of stage
    # 02's aggregate-level `prepare_backcasting_state` (which assumes constant historical
    # inflow going back init_max_age=50 years, i.e. to ~1955 for t0~2005). This one:
    #   1. Splits the 2005 total stock into BEV / Hybrid / Liquids base shares.
    #   2. Splits Hybrid into HEV/PHEV and Liquids into Diesel/Petrol using the SAME-YEAR
    #      (2005) EEA split ratios.
    #   3. Multiplies each drivetrain's 2005 stock by that drivetrain's 2005 segment-share
    #      profile (from `segment_shares_ext`) to get a (drivetrain, segment) stock grid.
    #   4. Calls `fdm.build_synthetic_pre_baseyear_inflows` to synthesize an inflow
    #      history from BACKCAST_START_YEAR (1975) through 2005 for EACH
    #      (region, drivetrain, segment) group, presumably again assuming some inflow
    #      pattern (constant, or another assumption -- not verifiable without `fdm.py`)
    #      consistent with the group's Weibull lifetime, that would explain the observed
    #      2005 segment-level stock.
    # This is a DIFFERENT methodology (segment-level, 30-year window, 1975-2005) from
    # stage 02's (aggregate-level, 50-year window, ~1955-2005) -- two independently
    # designed backcasts feeding into what is nominally one coherent pipeline. See the
    # consolidated review for the full comparison across all THREE backcasting variants
    # found in this codebase so far.
    BASE_YEAR = 2005
    BACKCAST_START_YEAR = 1975
    REGION = "EUR"

    stock_parts = []
    for (reg, drv), mats in matrices_by_key.items():
        if reg != "EUR" or "stock_t_tau_df" not in mats:
            continue
        stock_series = mats["stock_t_tau_df"].sum(axis=1)
        stock_parts.append(pd.DataFrame({
            "year": stock_series.index.astype(int), "Region": reg, "Drive Train": drv,
            "stock": stock_series.to_numpy(dtype=float),
        }))
    if not stock_parts:
        raise ValueError("No stock_t_tau_df found in matrices_by_key for EUR.")
    stock_inspect_df = pd.concat(stock_parts, ignore_index=True)

    stock_2005 = stock_inspect_df[stock_inspect_df["Region"].eq(REGION) & stock_inspect_df["year"].eq(BASE_YEAR)].copy()
    if stock_2005.empty:
        raise ValueError(f"No stock data found for {REGION} in {BASE_YEAR}.")

    starting_total_2005 = float(stock_2005["stock"].sum())
    shares_by_drv = stock_2005.groupby("Drive Train", as_index=True)["stock"].sum()
    bev_share = float(shares_by_drv.get("BEV", 0.0))
    hybrid_share = float(shares_by_drv.get("Hybrid", shares_by_drv.get("HEV", 0.0) + shares_by_drv.get("PHEV", 0.0)))
    liquids_share = float(shares_by_drv.get("Liquids", shares_by_drv.get("Diesel", 0.0) + shares_by_drv.get("Petrol", 0.0)))

    base_share_sum = bev_share + hybrid_share + liquids_share
    if base_share_sum <= 0:
        raise ValueError("Could not build BEV/Hybrid/Liquids shares for 2005.")

    base_shares_2005 = pd.Series({"BEV": bev_share / base_share_sum, "Hybrid": hybrid_share / base_share_sum, "Liquids": liquids_share / base_share_sum})
    base_stock_2005 = base_shares_2005 * starting_total_2005

    if BASE_YEAR not in split_hp.index or BASE_YEAR not in liquids_split.index:
        raise ValueError(f"Missing {BASE_YEAR} in split_hp or liquids_split.")

    hybrid_internal_2005 = split_hp.loc[BASE_YEAR, ["HEV", "PHEV"]].astype(float)
    hybrid_internal_2005 = hybrid_internal_2005 / hybrid_internal_2005.sum()
    liquids_internal_2005 = liquids_split.loc[BASE_YEAR, ["Diesel", "Petrol"]].astype(float)
    liquids_internal_2005 = liquids_internal_2005 / liquids_internal_2005.sum()

    starting_stock_2005 = pd.Series({
        "BEV": base_stock_2005["BEV"],
        "HEV": base_stock_2005["Hybrid"] * hybrid_internal_2005["HEV"],
        "PHEV": base_stock_2005["Hybrid"] * hybrid_internal_2005["PHEV"],
        "Diesel": base_stock_2005["Liquids"] * liquids_internal_2005["Diesel"],
        "Petrol": base_stock_2005["Liquids"] * liquids_internal_2005["Petrol"],
    }, name="value")

    segment_shares_2005 = (
        segment_shares_ext[segment_shares_ext["Year"] == BASE_YEAR]
        .pivot(index="Drive Train", columns="Segment", values="segment_share")
        .fillna(0.0)
    )
    segment_shares_2005 = segment_shares_2005.loc[[drv for drv in starting_stock_2005.index if drv in segment_shares_2005.index]]

    starting_stock_2005_segments = (
        segment_shares_2005.mul(starting_stock_2005, axis=0).stack().reset_index().rename(columns={0: "value"})
    )
    starting_stock_2005_segments["Region"] = REGION
    starting_stock_2005_segments = starting_stock_2005_segments[["Region", "Drive Train", "Segment", "value"]].copy()

    importlib.reload(fdm)
    synthetic_pre_2005_inflows = fdm.build_synthetic_pre_baseyear_inflows(
        starting_stock_segments=starting_stock_2005_segments,
        lifetime_by_drv=mapped["lifetime_by_drv"],
        base_year=BASE_YEAR, backcast_start_year=BACKCAST_START_YEAR,
        group_cols=["Region", "Drive Train", "Segment"], value_col="value",
    )
    # [FIXED, step 4 of the agreed plan] Post-hoc zero out any (Drive Train, year)
    # row in the 1975-2004 synthetic backcast that falls before that drivetrain's
    # real introduction year -- e.g. BEV and PHEV (introduced 2011/2012, both after
    # 2004) have their ENTIRE synthetic backcast zeroed; HEV (introduced 2000) keeps
    # 2000-2004 and only 1975-1999 is zeroed. The model itself
    # (build_synthetic_pre_baseyear_inflows) is untouched -- this masks its OUTPUT
    # only. See disaggregation.py's mask_inflow_before_introduction_year.
    synthetic_pre_2005_inflows = mask_inflow_before_introduction_year(
        synthetic_pre_2005_inflows, introduction_year_by_drv=p03.introduction_year_by_drv,
    )

    # -----------------------------------------------------------------------
    # Run the FLOW-DRIVEN model
    # -----------------------------------------------------------------------
    # `mapped` is REASSIGNED here with a narrower 5-drivetrain scope (see note above).
    mapped = fdm.build_p02_mapped_inputs(p02_dict, drivetrains=("BEV", "HEV", "PHEV", "Diesel", "Petrol"))

    inflow_segments_full = pd.concat(
        [synthetic_pre_2005_inflows[["Region", "Drive Train", "Segment", "year", "value"]],
         disaggregated["inflow_segments"][["Region", "Drive Train", "Segment", "year", "value"]]],
        ignore_index=True,
    )
    inflow_segments_full = (
        inflow_segments_full.groupby(["Region", "Drive Train", "Segment", "year"], as_index=False)["value"].sum()
        .sort_values(["Region", "Drive Train", "Segment", "year"]).reset_index(drop=True)
    )

    # [NEW] Fix a known, structural gap: stage 02's cohort recurrence never computes
    # its own t0=BASE_YEAR row (loop starts at i_t=1 by construction), so every
    # drivetrain that genuinely already existed before BASE_YEAR shows a false dip
    # to exactly 0 there (e.g. Diesel/Petrol/HEV), while drivetrains truly
    # introduced at/after BASE_YEAR (BEV, PHEV) correctly stay at 0 -- see
    # flowdriven_model.py's fill_base_year_gap_via_interpolation docstring for the
    # exact, narrow condition used to tell these two cases apart. Does NOT touch
    # _run_cohort_recurrence or any other model code; only BASE_YEAR's row is ever
    # modified, and only for groups matching that condition -- every other
    # (group, year) value in inflow_segments_full is byte-identical to before.
    inflow_segments_full = fdm.fill_base_year_gap_via_interpolation(
        inflow_segments_full, base_year=BASE_YEAR,
        group_cols=["Region", "Drive Train", "Segment"], year_col="year", value_col="value",
    )

    years_model = np.arange(BACKCAST_START_YEAR, years_full.values[-1] + 1, dtype=int)

    # THE MATH MODEL: flow-driven cohort model -- the mirror image of stage 02.
    # Given a PRESCRIBED inflow series (synthetic pre-2005 backcast + stage-02-derived
    # post-2005 segment inflow), simulate forward: each year, apply Weibull-hazard-based
    # survival to existing cohorts, add the prescribed new inflow, and read off the
    # resulting STOCK and OUTFLOW (further split into collected/export/unknown) as
    # OUTPUTS rather than solving for inflow as a residual. `outflow_timing="pre_inflow"`
    # (vs. 03_02's `"post_inflow"`) controls whether that year's hazard is applied to
    # existing cohorts BEFORE or AFTER the new inflow is added -- see the consolidated
    # review and 03_02_adjustedflows.py for why this specific difference matters.
    results_03 = fdm.run_flow_driven_model_with_outflow_disaggregation(
        df=inflow_segments_full, years=years_model, t_end=years_model[-1],
        lifetime_by_drv=mapped["lifetime_by_drv"], export_r_by_drv=mapped["export_r_by_drv"],
        age_bins=mapped["age_bins"], unknown_whereabouts_share=mapped["unknown_whereabouts_share"],
        outflow_value_col="value", year_col="year", inflow_col="value",
        group_cols=["Region", "Drive Train", "Segment"], outflow_timing="pre_inflow",
        segment_shares_by_drv=segment_shares_by_drv, export_share_by_drivetrain=export_share_by_drivetrain,
    )
    flows_03 = results_03["flows_df"].copy()
    outflow_long_03 = results_03["outflow_long_df"].copy()

    # -----------------------------------------------------------------------
    # Validation: stock-driven (stage 02) vs flow-driven (stage 03) outflow comparison
    # -----------------------------------------------------------------------
    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "03_01_stockdriven_vs_flowdriven.png"
    _plot_stockdriven_vs_flowdriven_outflow(disaggregated, flows_03, YEAR_PLOT_START, YEAR_PLOT_END, fig_path)
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------------
    # FLAGGED (real inconsistency, found by comparing this cell to the official save
    # cell further down): the ORIGINAL notebook ALSO has a separate cell that does:
    #     flows_03.to_pickle("../data/processed/intermediate/03_flows_03.pkl")
    #     outflow_long_03.to_pickle("../data/processed/intermediate/03_outflow_long_03.pkl")
    #     disaggregated["inflow_segments"].to_pickle("../data/processed/intermediate/03_inflow_segments.pkl")
    # This BYPASSES the `save_many`/`artifacts.py` registry entirely: hardcoded relative
    # paths (fragile -- depends on cwd being exactly the notebooks folder, unlike
    # `config.py`'s `get_paths()`), not tracked in `ARTIFACT_FILES` (so `artifact_status()`
    # never reports on them), and `flows_03` ends up saved TWICE via two different
    # mechanisms (once here via raw `to_pickle`, once below via the proper `save_many`
    # call) -- while `outflow_long_03` ONLY ever gets the raw/untracked treatment. I have
    # NOT reproduced the raw-to_pickle calls in this converted script -- only the
    # `save_many` call below, which is the mechanism the rest of the project uses
    # consistently. If `outflow_long_03` needs to be a tracked artifact, it should be
    # added to `ARTIFACT_FILES` and passed to `save_many` explicitly.
    saved = save_many(
        matrices_by_key=matrices_by_key, disaggregated=disaggregated,
        segment_shares_ext=segment_shares_ext, liquids_shares_ext=liquids_shares_ext,
        tracker_keyed=tracker_keyed, synthetic_pre_2005_inflows=synthetic_pre_2005_inflows,
        starting_stock_2005_segments=starting_stock_2005_segments, flows_03=flows_03,
        seg_share_by_drv=seg_share_by_drv,
        root=PROJECT_ROOT,
    )
    print("Saved artifacts:", saved)

    # -----------------------------------------------------------------------
    # Monte Carlo (opt-in via params.monte_carlo.enabled, default False -- does not
    # affect or slow down a normal deterministic run above; everything above this
    # point is unchanged whether or not this block runs).
    #
    # PROPAGATION: loads `mc_stage02_draws` (raw per-drivetrain, per-draw
    # `scale_lambda` and `cumulative_out_survival` arrays from 02_stockdriven.py's own
    # Monte Carlo run) and applies THIS stage's own uncertain parameters
    # (unknown_whereabouts_share, export_share) to those SAME draws, using
    # `disaggregation.compute_collected_export_unknown_shares` -- the IDENTICAL
    # function the deterministic split above uses, not a separate implementation.
    # Same draw index throughout: draw #12345's stage-02 lifetime outcome flows into
    # draw #12345's stage-03 split outcome -- genuine propagation, not independent
    # per-stage resampling.
    #
    # PER-YEAR BANDS, not just a cumulative endpoint: stage 02 only PERSISTS the
    # cumulative (2070-total) per-draw array (a full per-year-per-draw array would be
    # ~100MB+ per drivetrain per metric -- too large to pickle by default). To get a
    # genuine year-by-year uncertainty band here too (matching
    # `02_monte_carlo_flows_over_time.png`'s style, not just an isolated histogram),
    # this block RE-RUNS `run_cohort_survival_monte_carlo` using the SAME saved
    # `scale_lambda` draws (deterministic given the same inputs -- this reproduces
    # stage 02's per-draw results exactly, without needing to have persisted the large
    # intermediate array). `src/stockflow_model.py` is what makes this possible: the
    # exact same function stage 02 uses is now a normal library import here too.
    #
    # Uncertainty spreads (`unknown_whereabouts_share_std`, `export_share_std`) come
    # from `params.stock_flow` -- nothing hardcoded here.
    #
    # [FIXED, step 5 of the agreed plan] The `run_cohort_survival_monte_carlo` call
    # below now also passes `hard_zero_inflow_from_year`/`hard_zero_inflow_until_year`
    # (from `params.stock_flow`, same `.get(drivetrain)` pattern as
    # `02_stockdriven.py`'s own MC call) -- previously omitted here, meaning this
    # block's re-derived per-year `out_survival_by_year` (and the collected/export/
    # unknown bands built from it) would silently diverge from what stage 02 actually
    # computed and persisted for Liquids/Hybrid (post-2050 hard zero) and BEV
    # (pre-2011 hard zero), even though it's regenerated from the SAME saved
    # scale_lambda draws. See `params_schema.py`'s
    # `StockFlowParams.hard_zero_inflow_from_year_by_drv`/`hard_zero_inflow_until_
    # year_by_drv` for the full rationale.
    # -----------------------------------------------------------------------
    if params.monte_carlo.enabled:
        try:
            mc_stage02 = load_many("mc_stage02_draws", root=PROJECT_ROOT)["mc_stage02_draws"]
        except FileNotFoundError:
            print(
                "Monte Carlo skipped: 'mc_stage02_draws' not found -- run "
                "02_stockdriven.py with params.monte_carlo.enabled=True first."
            )
            mc_stage02 = None

        if mc_stage02 is not None:
            n_draws = params.monte_carlo.n_draws
            model_end_year_02 = int(p02.model_end_year)
            init_max_age = p02.init_max_age
            rng_seed_seq = np.random.SeedSequence(params.monte_carlo.seed, spawn_key=(1,))
            # spawn_key=(1,) deliberately differs from stage 02's own SeedSequence
            # (spawn_key=(0,), implicit) -- this stage's shares are sampled
            # independently of stage 02's lifetime, not correlated with it by
            # accident of sharing the same raw seed stream.
            drivetrains_present = sorted(mc_stage02.keys())
            child_seeds = rng_seed_seq.spawn(len(drivetrains_present))

            per_drivetrain_mc: dict[str, dict[str, np.ndarray]] = {}
            per_drivetrain_band: dict[str, dict[str, dict]] = {}
            # [NEW] period-based summaries -- see 02_stockdriven.py's Monte Carlo block
            # for the identical convention. Shared `output_periods` setting
            # (`params.monte_carlo.output_periods`) as stage 02 and stage 03_02, so a
            # request like "2030-2040" means the same window everywhere.
            output_periods = params.monte_carlo.output_periods
            per_drivetrain_period_summary: dict[str, dict] = {}
            eu_total_period_sums: dict[tuple[int, int], dict[str, np.ndarray]] = {
                p: {
                    "cumulative_inflow": np.zeros(n_draws), "cumulative_collected": np.zeros(n_draws),
                    "cumulative_export": np.zeros(n_draws), "cumulative_unknown": np.zeros(n_draws),
                }
                for p in output_periods
            }
            # [NEW] Sensitivity analysis inputs: three uncertain parameters per
            # drivetrain at this stage (scale_lambda, inherited from stage 02's own
            # draws; unknown_whereabouts_share and export_share, sampled here).
            sensitivity_input_draws: dict[str, np.ndarray] = {}

            for drivetrain, child_seed in zip(drivetrains_present, child_seeds):
                rng = np.random.default_rng(child_seed)

                # --- cumulative (2070-total) split, using stage 02's saved arrays ---
                cumulative_out_survival = mc_stage02[drivetrain]["cumulative_out_survival"]

                uw_point = unknown_whereabouts_share[drivetrain]
                uw_std = p02.unknown_whereabouts_share_std[drivetrain]
                uw_draws = Normal(uw_point, uw_std, clip_min=0.0, clip_max=1.0).sample(rng, n=n_draws)

                exp_point = export_share_by_drivetrain.get(drivetrain, 0.0)
                exp_std = p02.export_share_std.get(drivetrain, 0.0)
                exp_draws = Normal(exp_point, exp_std, clip_min=0.0, clip_max=1.0).sample(rng, n=n_draws)

                sensitivity_input_draws[f"{drivetrain}_scale_lambda"] = mc_stage02[drivetrain]["scale_lambda"]
                sensitivity_input_draws[f"{drivetrain}_unknown_share"] = uw_draws
                sensitivity_input_draws[f"{drivetrain}_export_share"] = exp_draws

                unk_share, exp_share, coll_share = disagg.compute_collected_export_unknown_shares(uw_draws, exp_draws)
                per_drivetrain_mc[drivetrain] = {
                    "cumulative_collected": cumulative_out_survival * coll_share,
                    "cumulative_export": cumulative_out_survival * exp_share,
                    "cumulative_unknown": cumulative_out_survival * unk_share,
                }

                # --- per-YEAR split, regenerated from the SAME scale_lambda draws ---
                scale_lambda_draws = mc_stage02[drivetrain]["scale_lambda"]
                base_shape_k = p02.lifetime_by_drv[drivetrain].shape_k
                stock_series = pd.to_numeric(stock_dict[("EUR", drivetrain)]["stock"], errors="coerce").fillna(0.0)
                stock_series.index = stock_series.index.astype(int)
                stock_series = stock_series.sort_index()
                backcast = build_backcast_state(
                    stock_series=stock_series, model_end_year=model_end_year_02,
                    shape_k=base_shape_k, scale_lambda=p02.lifetime_by_drv[drivetrain].scale_lambda,
                    init_max_age=init_max_age,
                )
                result_02 = run_cohort_survival_monte_carlo(
                    stock_series=stock_series, model_end_year=model_end_year_02, drivetrain=drivetrain,
                    shape_k_draws=np.full(n_draws, base_shape_k), scale_lambda_draws=scale_lambda_draws,
                    lifetime_override=p02.lifetime_override_by_drv.get(drivetrain), backcast=backcast,
                    negative_inflow_policy=p02.negative_inflow_policy,
                    # [FIXED, step 5 of the agreed plan] This block RE-RUNS
                    # run_cohort_survival_monte_carlo to regenerate a per-year band
                    # from the SAME saved scale_lambda draws -- it's only a faithful
                    # reproduction of stage 02's own run if EVERY argument that
                    # affects the cohort trajectory matches, including the hard-zero
                    # overrides. Without these two, this re-derived out_survival_by_
                    # year (and the collected/export/unknown bands built from it)
                    # would silently diverge from what 02_stockdriven.py actually
                    # computed and persisted for Liquids/Hybrid (post-2050) and BEV
                    # (pre-2011) -- the exact kind of dragged inconsistency this
                    # whole plan exists to avoid. Same params.stock_flow fields,
                    # same .get(drivetrain) pattern, as 02_stockdriven.py's own call.
                    hard_zero_inflow_from_year=p02.hard_zero_inflow_from_year_by_drv.get(drivetrain),
                    hard_zero_inflow_until_year=p02.hard_zero_inflow_until_year_by_drv.get(drivetrain),
                )
                # (n_years, n_draws) x (n_draws,) broadcasts correctly: each draw's own
                # share applies to that same draw's every year.
                out_survival_by_year = result_02["out_survival_by_year"]
                collected_by_year = out_survival_by_year * coll_share[None, :]
                export_by_year = out_survival_by_year * exp_share[None, :]
                unknown_by_year = out_survival_by_year * unk_share[None, :]
                years_list = result_02["t"].tolist()

                per_drivetrain_band[drivetrain] = {
                    "collected": {
                        "years": years_list,
                        "p2_5": np.percentile(collected_by_year, 2.5, axis=1).tolist(),
                        "median": np.percentile(collected_by_year, 50, axis=1).tolist(),
                        "p97_5": np.percentile(collected_by_year, 97.5, axis=1).tolist(),
                    },
                    "export": {
                        "years": years_list,
                        "p2_5": np.percentile(export_by_year, 2.5, axis=1).tolist(),
                        "median": np.percentile(export_by_year, 50, axis=1).tolist(),
                        "p97_5": np.percentile(export_by_year, 97.5, axis=1).tolist(),
                    },
                    "unknown": {
                        "years": years_list,
                        "p2_5": np.percentile(unknown_by_year, 2.5, axis=1).tolist(),
                        "median": np.percentile(unknown_by_year, 50, axis=1).tolist(),
                        "p97_5": np.percentile(unknown_by_year, 97.5, axis=1).tolist(),
                    },
                }

                # --- [NEW] period-based summaries for this drivetrain ---
                # `inflow_by_year` was already computed by `run_cohort_survival_monte_
                # carlo` above (same call, no extra simulation) but previously
                # discarded here -- now actually used, giving "cumulative input" at
                # 03_01 too, not just at stage 02.
                inflow_period_sums = sum_by_period(result_02["inflow_by_year"], result_02["t"], output_periods)
                collected_period_sums = sum_by_period(collected_by_year, result_02["t"], output_periods)
                export_period_sums = sum_by_period(export_by_year, result_02["t"], output_periods)
                unknown_period_sums = sum_by_period(unknown_by_year, result_02["t"], output_periods)

                drv_period_summary: dict[tuple[int, int], dict] = {}
                for period in output_periods:
                    start, end = period
                    mask = (result_02["t"] >= start) & (result_02["t"] <= end)
                    years_in_period = result_02["t"][mask]
                    # stock_t: DETERMINISTIC (stage 02's stock-driven paradigm pins it
                    # to the REMIND target regardless of lifetime draws) -- see
                    # 02_stockdriven.py's Monte Carlo block for the same convention.
                    stock_values_in_period = result_02["stock_t"][mask]

                    eu_total_period_sums[period]["cumulative_inflow"] += inflow_period_sums[period]
                    eu_total_period_sums[period]["cumulative_collected"] += collected_period_sums[period]
                    eu_total_period_sums[period]["cumulative_export"] += export_period_sums[period]
                    eu_total_period_sums[period]["cumulative_unknown"] += unknown_period_sums[period]

                    drv_period_summary[period] = {
                        "cumulative_inflow": summarize_distribution(inflow_period_sums[period]),
                        "cumulative_collected": summarize_distribution(collected_period_sums[period]),
                        "cumulative_export": summarize_distribution(export_period_sums[period]),
                        "cumulative_unknown": summarize_distribution(unknown_period_sums[period]),
                        "stock_end_of_period": summarize_distribution(
                            np.full(n_draws, stock_values_in_period[-1] if stock_values_in_period.size else np.nan)
                        ),
                        "stock_sum_over_period": summarize_distribution(
                            np.full(n_draws, stock_values_in_period.sum())
                        ),
                        "stock_per_year": {
                            int(y): summarize_distribution(np.full(n_draws, v))
                            for y, v in zip(years_in_period, stock_values_in_period)
                        },
                    }
                per_drivetrain_period_summary[drivetrain] = drv_period_summary

            eu_total_period_summary: dict[tuple[int, int], dict] = {
                period: {
                    metric: summarize_distribution(eu_total_period_sums[period][metric])
                    for metric in ["cumulative_inflow", "cumulative_collected", "cumulative_export", "cumulative_unknown"]
                }
                for period in output_periods
            }
            for period in output_periods:
                s = eu_total_period_summary[period]["cumulative_collected"]
                print(
                    f"Monte Carlo [EU total, {period[0]}-{period[1]}, collected]: {n_draws:,} draws -- "
                    f"mean={s['mean']:.2f}, median={s['median']:.2f}, P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
                )

            eu_total_mc = {
                metric: sum(per_drivetrain_mc[drv][metric] for drv in drivetrains_present)
                for metric in ["cumulative_collected", "cumulative_export", "cumulative_unknown"]
            }

            summary_mc: dict[str, dict] = {}
            for drv in drivetrains_present:
                for metric, values in per_drivetrain_mc[drv].items():
                    summary_mc[f"{drv}__{metric}"] = summarize_distribution(values)
                for flow_name, band in per_drivetrain_band[drv].items():
                    summary_mc[f"{drv}__{flow_name}_by_year_band"] = band
            for metric, values in eu_total_mc.items():
                summary_mc[f"EU_total__{metric}"] = summarize_distribution(values)

            # --- [NEW] Sensitivity analysis: which drivetrain's lifetime OR share
            # uncertainty drives EU-total collected volume most. Headline period =
            # widest requested `output_periods` entry (typically the whole horizon).
            headline_period = max(output_periods, key=lambda p: p[1] - p[0])
            sensitivity_df = sensitivity_correlations(
                sensitivity_input_draws, eu_total_period_sums[headline_period]["cumulative_collected"],
            )
            print(f"\nSensitivity [EU total, {headline_period[0]}-{headline_period[1]}, collected]:")
            print(sensitivity_df.to_string(index=False))

            fig_tornado, _ = plot_tornado(
                sensitivity_df,
                title=f"Sensitivity: EU-total collected, {headline_period[0]}-{headline_period[1]}",
            )
            fig_path_tornado = fig_dir / "03_01_monte_carlo_sensitivity_tornado.png"
            fig_tornado.savefig(fig_path_tornado, dpi=150, bbox_inches="tight")
            print(f"Saved diagnostic plot: {fig_path_tornado}")

            saved_mc = save_many(
                mc_stage03_summary=summary_mc,
                mc_stage03_01_sensitivity=sensitivity_df,
                mc_stage03_01_period_summary={
                    "by_drivetrain": per_drivetrain_period_summary, "eu_total": eu_total_period_summary,
                },
                root=PROJECT_ROOT,
            )
            print("Saved Monte Carlo artifacts:", saved_mc)

            s = summary_mc["EU_total__cumulative_collected"]
            print(
                f"Monte Carlo [EU total, collected]: {n_draws:,} draws -- "
                f"mean={s['mean']:.2f}, median={s['median']:.2f}, mode={s['mode']:.2f}, "
                f"std={s['std']:.2f}, P2.5={s['p2_5']:.2f}, P97.5={s['p97_5']:.2f}"
            )

            # -----------------------------------------------------------------------
            # Plot 1: collected/export/unknown OVER TIME, median + P2.5-P97.5 band,
            # all drivetrains overlaid -- the direct Monte Carlo counterpart of this
            # stage's own flow quantities, in the same style as
            # `02_monte_carlo_flows_over_time.png`.
            # -----------------------------------------------------------------------
            fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
            colors = plt.cm.tab10.colors
            for i, drivetrain in enumerate(drivetrains_present):
                color = colors[i % len(colors)]
                for ax, flow_name in zip(axes, ["collected", "export", "unknown"]):
                    band = per_drivetrain_band[drivetrain][flow_name]
                    ax.plot(band["years"], band["median"], color=color, linewidth=1.6, label=drivetrain)
                    ax.fill_between(band["years"], band["p2_5"], band["p97_5"], color=color, alpha=0.2)
            for ax, flow_name in zip(axes, ["Collected", "Export", "Unknown whereabouts"]):
                ax.set_title(f"Monte Carlo: annual {flow_name.lower()} outflow, median + P2.5-P97.5 band", fontsize=12)
                ax.set_ylabel(f"{flow_name} [million/year]")
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
            axes[-1].set_xlabel("Year")
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            # [FIXED, this round] Renamed from "03_monte_carlo_flows_over_time.png" --
            # missing the stage number, inconsistent with this same file's sibling
            # output "03_01_monte_carlo_sensitivity_tornado.png". Confirmed with the
            # user.
            fig_path = fig_dir / "03_01_monte_carlo_flows_over_time.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Saved diagnostic plot: {fig_path}")

            # [REMOVED, this round, confirmed with the user] "Plot 2" -- the EU-total
            # cumulative-collected-by-{end_year_model} histogram (previously saved as
            # "03_monte_carlo_eu_collected.png") is no longer generated. It used
            # `s["bin_edges"]`/`s["frequencies"]` (EU_total__cumulative_collected's
            # summarize_distribution() output) purely for this plot; that summary
            # itself is still computed and saved in `summary_mc` either way (used
            # elsewhere, e.g. the printed mean/median/P2.5/P97.5 line above), so nothing
            # upstream needed to change -- only this figure's generation was deleted.

    return saved


def _plot_stockdriven_vs_flowdriven_outflow(
    disaggregated: dict, flows_03: pd.DataFrame, year_min: int, year_max: int, save_path: Path,
) -> None:
    """
    Compare stage-02's stock-driven outflow (reconstructed from the
    collected/export/unknown segment tables) against stage-03's flow-driven outflow, by
    base drivetrain (BEV, Hybrid=HEV+PHEV, Liquids=Diesel+Petrol) and in total.

    Any discrepancy here is a mix of (a) the intended stock-driven vs flow-driven
    paradigm difference, AND (b) the fact that the flow-driven run's own initial
    conditions come from an INDEPENDENT backcast (segment-level, 1975-2005) rather than
    directly reusing stage 02's (aggregate-level, ~1955-2005) cohort structure -- the two
    effects are not separated by this plot alone.

    [FIXED] Saves to `save_path` instead of calling `plt.show()`, consistent with every
    other stage's diagnostic-plot convention (this file sets `matplotlib.use("Agg")`).
    """
    drv_prefix_map_local = {"BEV": ["BEV"], "Hybrid": ["HEV", "PHEV"], "Liquids": ["Diesel", "Petrol"]}

    stock_parts = []
    for flow_key in ["outflow_coll_segments", "outflow_exp_segments", "outflow_unk_segments"]:
        if flow_key in disaggregated and disaggregated[flow_key] is not None:
            tmp = disaggregated[flow_key].copy()
            if "value" in tmp.columns:
                stock_parts.append(tmp[["Region", "Drive Train", "year", "value"]])
    if not stock_parts:
        raise ValueError("No stock-driven outflow tables found in disaggregated dictionary.")
    stock_by_drv = pd.concat(stock_parts, ignore_index=True).groupby(["Region", "Drive Train", "year"], as_index=False)["value"].sum().rename(columns={"value": "outflow_stockdriven"})

    flow_by_drv = flows_03.groupby(["Region", "Drive Train", "year"], as_index=False)["out_total"].sum().rename(columns={"out_total": "outflow_flowdriven"})

    rows = []
    for base_drv, mapped_drvs in drv_prefix_map_local.items():
        s = stock_by_drv[(stock_by_drv["Region"] == "EUR") & (stock_by_drv["Drive Train"].isin(mapped_drvs))].groupby("year", as_index=False)["outflow_stockdriven"].sum()
        f = flow_by_drv[(flow_by_drv["Region"] == "EUR") & (flow_by_drv["Drive Train"].isin(mapped_drvs))].groupby("year", as_index=False)["outflow_flowdriven"].sum()
        m = pd.merge(s, f, on="year", how="outer").fillna(0.0)
        m["base_drv"] = base_drv
        rows.append(m)
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df = plot_df[(plot_df["year"] >= year_min) & (plot_df["year"] <= year_max)]
    total_df = plot_df.groupby("year", as_index=False)[["outflow_stockdriven", "outflow_flowdriven"]].sum()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"BEV": "#1b9e77", "Hybrid": "#f2c14e", "Liquids": "#4c78a8"}
    for ax, drv in zip(axes.flat, ["BEV", "Hybrid", "Liquids"]):
        d = plot_df[plot_df["base_drv"] == drv].sort_values("year")
        c = colors.get(drv, "#333333")
        ax.plot(d["year"], d["outflow_stockdriven"], color=c, linewidth=2.0, label="Stock-driven")
        ax.plot(d["year"], d["outflow_flowdriven"], color=c, linewidth=1.6, linestyle="--", label="Flow-driven")
        ax.set_title(drv, fontsize=11)
        ax.legend(frameon=False)
    ax_total = axes.flat[3]
    ax_total.plot(total_df["year"], total_df["outflow_stockdriven"], color="#2f6db3", linewidth=2.2, label="Stock-driven total")
    ax_total.plot(total_df["year"], total_df["outflow_flowdriven"], color="#ff8c00", linewidth=1.8, linestyle="--", label="Flow-driven total")
    ax_total.set_title("Total (EUR)")
    ax_total.legend(frameon=False)
    fig.suptitle("Outflow comparison: stock-driven (stage 02) vs flow-driven (stage 03)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()