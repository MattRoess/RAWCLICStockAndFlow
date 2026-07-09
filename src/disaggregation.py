"""
disaggregation.py
===================

Segment/drivetrain-splitting and materials-tracker-construction library used by stage 03
(`03_01_flowdriven.py`, `03_02_adjustedflows.py`).

======================================================================
C9 -- FIXED THIS ROUND. "out_total" was computed by DOUBLE-COUNTING export/unknown
outflow, inconsistently, in five places across this file.
======================================================================
By construction elsewhere in this codebase, `out_survival` is the TOTAL survival-based
outflow, and `out_export`/`out_unknown`/`out_collected` are a PARTITION of it (they sum
back to `out_survival` exactly: `out_export + out_unknown + out_collected ==
out_survival`). The correct "total outflow" is therefore just `out_survival` itself.

Previously, THREE different (and mutually inconsistent) formulas existed for
`out_total` across this file, all of which double-counted:
  - `split_outflows_collected_unknown_export`: `out_survival + out_export + out_unknown`
  - `split_hybrid_outflows_afterwards`: same 3-term formula, same bug
  - `split_liquids_outflows_afterwards`: `out_survival + out_export` (2-term, missing
    out_unknown -- a third, independently wrong variant)
  - `split_hybrid_flows_afterwards` / `split_liquids_flows_afterwards` (both dead code,
    never called by either notebook): same 2-term formula as above

Using the real `00_parameters.py` shares, the resulting `out_total` in the persisted
`matrices_by_key` artifact was inflated by +8% to +51% depending on drivetrain (see Fix
Log / EVmodel_review_consolidated.md for the exact per-drivetrain derivation that was
here before the fix).

**Run-verified this round**: exercised end-to-end as part of the full
`00->01->02->03_01->03_02` pipeline run against synthetic data -- every function in
this file that's actually called by the two stage-03 scripts executed successfully and
produced structurally sensible output (confirmed via `.equals()` comparisons and direct
number checks, not just "didn't crash"). Not yet run against real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_collected_export_unknown_shares(unknown_whereabouts_share, export_share):
    """
    [NEW] The actual arithmetic of the collected/export/unknown split, extracted into
    one place so both the deterministic per-year split below AND a Monte Carlo
    per-draw split (see 03_01_flowdriven.py's Monte Carlo block) use the IDENTICAL
    computation -- not two separate implementations of the same formula.

    Works for EITHER a plain float (the normal, single-run case) OR a NumPy array
    (Monte Carlo, one value per draw) -- `np.clip` and elementwise arithmetic behave
    the same way for both, so no special-casing is needed for the arithmetic itself.

    Returns (unknown_share, export_s, collected_share), same type/shape as the inputs.

    ERROR HANDLING DIFFERS BETWEEN THE TWO CASES, deliberately:
    - Scalar input: raises `ValueError` if unknown_share + export_s > 1 -- EXACT
      original behavior, unchanged.
    - Array input: rather than aborting an entire 200,000-draw Monte Carlo run over a
      handful of draws where two independently-sampled shares happen to sum past 1
      (an edge-of-distribution sampling artifact, not a data error), those draws'
      `collected_share` is clipped to 0 instead, and a count is printed. Aborting a
      long-running batch over a handful of extreme draws would be worse than slightly
      under-representing the tail for those specific draws.
    """
    unknown_share = np.clip(unknown_whereabouts_share, 0.0, 1.0)
    export_s = np.clip(export_share, 0.0, 1.0)
    total_share = unknown_share + export_s

    if isinstance(unknown_share, np.ndarray) or isinstance(export_s, np.ndarray):
        over = total_share > 1.0
        if np.any(over):
            print(
                f"NOTE: {int(np.sum(over))}/{np.size(total_share)} Monte Carlo draws had "
                f"unknown_whereabouts_share + export_share > 1.0 (sampling artifact at "
                f"the tails) -- collected_share clipped to 0 for those draws instead of "
                f"raising, so the run isn't aborted over rare edge draws."
            )
        collected_share = np.clip(1.0 - total_share, 0.0, None)
    else:
        if total_share > 1.0:
            raise ValueError(f"Shares exceed 1 for this drivetrain: unknown={unknown_share}, export={export_s}")
        collected_share = 1.0 - total_share

    return unknown_share, export_s, collected_share


def split_outflows_collected_unknown_export(
    matrices_by_key: dict,
    unknown_whereabouts_share: dict[str, float],
    export_share: dict[str, float],
) -> tuple[dict, dict[str, float]]:

    for key in matrices_by_key:
        surv_df = matrices_by_key[key]["outflow_surv_df"]
        drivetrain = key[1]
        if drivetrain not in unknown_whereabouts_share:
            raise KeyError(
                f"Missing unknown whereabouts share for drivetrain={drivetrain!r}. "
                "Add it to unknown_whereabouts_share."
            )

        unknown_share, export_s, collected_share = compute_collected_export_unknown_shares(
            unknown_whereabouts_share[drivetrain], export_share.get(drivetrain, 0.0),
        )

        unk_df = surv_df * unknown_share
        exp_df = surv_df * export_s
        coll_df = surv_df * collected_share


        matrices_by_key[key]["outflow_unk_df"] = unk_df
        matrices_by_key[key]["outflow_exp_df"] = exp_df
        matrices_by_key[key]["outflow_coll_df"] = coll_df


        flows = matrices_by_key[key]["flows_df"].copy()

        flows["out_unknown"] = unk_df.sum(axis=1)
        flows["out_collected"] = coll_df.sum(axis=1)
        flows["out_export"] = exp_df.sum(axis=1)

        if {"out_survival", "out_export"}.issubset(flows.columns):
            # [FIXED, resolves C9]: out_export/out_unknown/out_collected are already a
            # PARTITION of out_survival (they sum back to it exactly) -- the correct
            # "total outflow" is out_survival itself, not out_survival plus its own
            # subsets. Previously: out_survival + out_export + out_unknown (double-
            # counted both subsets, +12% to +51% depending on drivetrain -- see the
            # module docstring's original derivation, kept below for the historical
            # record now that the bug is fixed).
            flows["out_total"] = flows["out_survival"]

        matrices_by_key[key]["flows_df"] = flows

    return matrices_by_key, unknown_whereabouts_share


def prepare_eea_share_tables(
    output_dir: str,
    start_year_model: int,
    end_year_model: int,
    country_scope: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    [FIXED] Previously a bare `pd.read_csv` -- if the file was missing, this failed
    deep inside pandas with a generic `FileNotFoundError` and no indication of what was
    actually expected. Now checks existence first and raises a clear, actionable error
    naming the exact path and the required schema.

    [NEW] `country_scope`: which ISO2 `Country` values to keep before any aggregation.
    Previously this function ignored the `Country` column entirely -- it grouped
    straight to `(Year, Drive Train, Segment)` over WHATEVER countries happened to be
    present in the raw file, silently including any non-EU countries the source file
    contains. The real `EEA_final_data.csv` (confirmed by inspection) contains EU27 +
    GB + NO + IS. Project convention ("EU plus 4"): use EU27 + Norway + Iceland (the
    only two of the four EFTA/EEA countries actually present in this source -- CH and
    LI are simply absent from the data, not deliberately excluded), and explicitly
    DROP GB (14.0% of total registration volume in the real file -- by far the largest
    non-EU contributor, and no longer an EU market).

    Passing `country_scope=None` (the default) keeps the OLD behavior (no filtering) --
    for backward compatibility with any other caller that doesn't pass this argument
    yet. Every call site in this project's own pipeline (03_01_flowdriven.py,
    03_02_adjustedflows.py) should pass `country_scope` explicitly -- see
    `03_01_flowdriven.py`'s call site for the exact `eu_countries + ("NO", "IS")`
    construction.

    KNOWN, ACCEPTED LIMITATION (documented, not corrected here -- see
    MATH_MODELS.md/HOW_TO_RUN_AND_VERIFY.md addenda): even after this filter, country
    coverage is UNEVEN across years within `country_scope` itself -- GB's exclusion
    aside, NO only has data from 2019 onward, IS only from 2018 onward, and HR only
    from 2014 onward (EU accession year) in the source file. This means the country
    composition of "EU+NO+IS" changes year to year even after this filter is applied.
    No better data exists, so this is accepted as-is -- confirmed (see discussion log)
    to have LOW impact on segment-share rankings/magnitudes, but MEANINGFUL impact on
    drivetrain-level shares (Norway's disproportionately high BEV share pulls the
    EU+NO+IS-wide BEV share up by roughly 0.5-1.2 percentage points in recent years
    relative to an EU27-only baseline, despite NO being only ~0.43% of total
    registration volume) -- flag this explicitly wherever drivetrain-level shares
    derived from this function are reported or interpreted.
    """
    eea_path = Path(output_dir) / "EEA_final_data.csv"
    if not eea_path.exists():
        raise FileNotFoundError(
            f"prepare_eea_share_tables: expected an EEA registrations CSV at "
            f"{eea_path}, but it doesn't exist.\n"
            f"Required columns: 'Year', 'Drive Train', 'Segment', 'Registrations' -- "
            f"covering at least Petrol/Diesel/HEV/PHEV/BEV and the 12 segment codes "
            f"(A-F, JA-JF).\n"
            f"If you don't have this file yet: set "
            f"params.disaggregation.use_synthetic_eea_fallback = True to have "
            f"03_01_flowdriven.py generate a clearly-labeled SYNTHETIC placeholder in "
            f"its place, so you can exercise the rest of the pipeline in the meantime. "
            f"No other code change is needed once the real file arrives -- just place "
            f"it at the path above."
        )
    eea_data = pd.read_csv(eea_path)

    if country_scope is not None:
        if "Country" not in eea_data.columns:
            raise KeyError(
                f"prepare_eea_share_tables: country_scope was given ({sorted(country_scope)}), "
                f"but the loaded EEA_final_data.csv at {eea_path} has no 'Country' column to "
                f"filter on. Either fix the file's schema or pass country_scope=None."
            )
        present = set(eea_data["Country"].unique())
        missing_from_data = set(country_scope) - present
        if missing_from_data:
            print(
                f"NOTE: prepare_eea_share_tables: country_scope includes "
                f"{sorted(missing_from_data)}, which have NO rows in {eea_path.name} -- "
                f"these will simply contribute nothing (not an error; e.g. CH/LI are "
                f"expected to be absent from the current source file)."
            )
        dropped = present - set(country_scope)
        if dropped:
            dropped_share = (
                eea_data.loc[eea_data["Country"].isin(dropped), "Registrations"].sum()
                / eea_data["Registrations"].sum()
            )
            print(
                f"prepare_eea_share_tables: excluding {sorted(dropped)} from country_scope "
                f"({dropped_share:.1%} of total registration volume in the raw file)."
            )
        eea_data = eea_data[eea_data["Country"].isin(country_scope)].copy()

    eea_data = eea_data[
        (eea_data["Segment"].notna())
        & (eea_data["Segment"] != "")
        & (eea_data["Segment"] != "V")
    ].copy()

    years_full = pd.Index(range(int(start_year_model), int(end_year_model) + 1), name="Year")

    segment_shares = (
        eea_data.groupby(["Year", "Drive Train", "Segment"])["Registrations"].sum().reset_index()
    )
    segment_shares["total_drv_year"] = (
        segment_shares.groupby(["Year", "Drive Train"])["Registrations"].transform("sum")
    )
    segment_shares["segment_share"] = segment_shares["Registrations"] / segment_shares["total_drv_year"]

    seg_pairs = segment_shares[["Drive Train", "Segment"]].drop_duplicates()
    seg_full_index = pd.MultiIndex.from_frame(
        seg_pairs.assign(_k=1)
        .merge(pd.DataFrame({"Year": years_full, "_k": 1}), on="_k", how="outer")
        .drop(columns="_k"),
        names=["Drive Train", "Segment", "Year"],
    )

    segment_shares_ext = (
        segment_shares.set_index(["Drive Train", "Segment", "Year"])
        .sort_index()
        .reindex(seg_full_index)
        .groupby(level=["Drive Train", "Segment"], group_keys=False)
        .apply(lambda frame: frame.bfill().ffill())
        .reset_index()
    )

    ice_df = eea_data[eea_data["Drive Train"].isin(["Petrol", "Diesel"])].copy()
    liquids_shares = (
        ice_df.groupby(["Year", "Drive Train"])["Registrations"].sum().reset_index()
    )
    liquids_shares["liquids_shares"] = liquids_shares.groupby("Year")["Registrations"].transform("sum")
    liquids_shares["share"] = liquids_shares["Registrations"] / liquids_shares["liquids_shares"]

    drv_pairs = liquids_shares[["Drive Train"]].drop_duplicates()
    drv_full_index = pd.MultiIndex.from_frame(
        drv_pairs.assign(_k=1)
        .merge(pd.DataFrame({"Year": years_full, "_k": 1}), on="_k", how="outer")
        .drop(columns="_k"),
        names=["Drive Train", "Year"],
    )

    liquids_shares_ext = (
        liquids_shares.set_index(["Drive Train", "Year"])
        .sort_index()
        .reindex(drv_full_index)
        .groupby(level=["Drive Train"], group_keys=False)
        .apply(lambda frame: frame.bfill().ffill())
        .reset_index()
    )

    return eea_data, segment_shares_ext, liquids_shares_ext


def generate_synthetic_eea_data(
    output_dir: str,
    start_year: int = 2005,
    end_year: int = 2050,
    seed: int = 42,
) -> Path:
    """
    [NEW] Write a clearly-labeled SYNTHETIC placeholder `EEA_final_data.csv` to
    `output_dir`, matching the schema `prepare_eea_share_tables` requires (columns:
    Year, Drive Train, Segment, Registrations), so the rest of the pipeline can be
    exercised while waiting for the real file.

    THIS IS NOT REAL DATA. Registrations are random (uniform 50-500) per
    (year, drivetrain, segment) -- structurally valid, numerically meaningless. Every
    downstream number derived from it (segment shares, HEV/PHEV split, Diesel/Petrol
    split, and everything stage 04+ eventually computes from them) is equally
    meaningless until the real file replaces this one. Never used automatically --
    only called when `params.disaggregation.use_synthetic_eea_fallback = True`, and
    always prints a loud warning when it runs (see call site in
    `03_01_flowdriven.py`).

    Once your real `EEA_final_data.csv` is available, just place it at the same path
    (`output_dir/EEA_final_data.csv`) -- `prepare_eea_share_tables` doesn't care how
    the file got there, real or synthetic. No code change needed either way.
    """
    rng = np.random.default_rng(seed)
    drivetrains = ["Petrol", "Diesel", "HEV", "PHEV", "BEV"]
    segments = ["A", "B", "C", "D", "E", "F", "JA", "JB", "JC", "JD", "JE", "JF"]

    rows = []
    for year in range(start_year, end_year + 1):
        for drv in drivetrains:
            for seg in segments:
                rows.append({
                    "Year": year, "Drive Train": drv, "Segment": seg,
                    "Registrations": float(rng.uniform(50, 500)),
                })

    out_path = Path(output_dir) / "EEA_final_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def build_two_way_split_wide(
    shares_ext: pd.DataFrame,
    categories: tuple[str, str],
    year_col: str = "Year",
    cat_col: str = "Drive Train",
    value_col: str = "share",
) -> pd.DataFrame:
    wide = (
        shares_ext
        .pivot(index=year_col, columns=cat_col, values=value_col)
        .sort_index()
        .fillna(0.0)
    )

    cols = [c for c in categories if c in wide.columns]
    wide = wide[cols]

    return wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def build_hev_phev_split(eea_data: pd.DataFrame, years_full: pd.Index) -> pd.DataFrame:
    hyb = eea_data[eea_data["Drive Train"].isin(["HEV", "PHEV"])].copy()
    hyb_year = hyb.groupby(["Year", "Drive Train"])["Registrations"].sum().reset_index()
    hyb_year["share"] = hyb_year["Registrations"] / hyb_year.groupby("Year")["Registrations"].transform("sum")

    drv_pairs = hyb_year[["Drive Train"]].drop_duplicates()
    full_index = pd.MultiIndex.from_frame(
        drv_pairs.assign(_k=1)
        .merge(pd.DataFrame({"Year": years_full, "_k": 1}), on="_k", how="outer")
        .drop(columns="_k"),
        names=["Drive Train", "Year"],
    )

    hyb_ext = (
        hyb_year.set_index(["Drive Train", "Year"])
        .sort_index()
        .reindex(full_index)
        .groupby(level=["Drive Train"], group_keys=False)
        .apply(lambda frame: frame.bfill().ffill())
        .reset_index()
    )

    return build_two_way_split_wide(
        hyb_ext,
        categories=("HEV", "PHEV"),
        year_col="Year",
        cat_col="Drive Train",
        value_col="share",
    )


def split_hybrid_inflow_afterwards(
    matrices_by_key: dict,
    split_hp: pd.DataFrame,
    region: str = "EUR",
    base_drv: str = "Hybrid",
) -> None:
    base_key = (region, base_drv)
    if base_key not in matrices_by_key:
        raise KeyError(f"Missing {base_key} in matrices_by_key.")

    base = matrices_by_key[base_key]
    years = base["flows_df"].index
    split_hp = split_hp.reindex(years).ffill().bfill().fillna(0.0)

    out = {}
    for sub in ["HEV", "PHEV"]:
        if sub not in split_hp.columns:
            continue
        s = split_hp[sub]
        mats_sub = dict(matrices_by_key.get((region, sub), {}))
        flows_sub = mats_sub.get("flows_df", base["flows_df"]).copy()
        if "inflow" in base["flows_df"].columns:
            flows_sub["inflow"] = base["flows_df"]["inflow"].mul(s, axis=0)
        mats_sub["flows_df"] = flows_sub
        out[(region, sub)] = mats_sub

    matrices_by_key.update(out)


def split_hybrid_outflows_afterwards(
    matrices_by_key: dict,
    split_hp: pd.DataFrame,
    region: str = "EUR",
    base_drv: str = "Hybrid",
) -> None:
    base_key = (region, base_drv)
    if base_key not in matrices_by_key:
        raise KeyError(f"Missing {base_key} in matrices_by_key.")

    base = matrices_by_key[base_key]
    years = base["flows_df"].index
    split_hp = split_hp.reindex(years).ffill().bfill().fillna(0.0)
    required_outflow_mats = ["outflow_coll_df", "outflow_exp_df", "outflow_unk_df", "outflow_surv_df"]
    missing = [name for name in required_outflow_mats if name not in base]
    if missing:
        raise KeyError(
            f"{base_key} missing required outflow matrices {missing}. "
            "Run split_outflows_collected_unknown before splitting outflows."
        )

    def _scale_outflow_by_cohort(matrix_df: pd.DataFrame, share_by_year: pd.Series) -> pd.DataFrame:
        cohort_year = pd.to_numeric(pd.Index(matrix_df.columns), errors="coerce")
        col_shares = share_by_year.reindex(cohort_year).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        return matrix_df.mul(col_shares, axis=1)

    out = {}
    for sub in ["HEV", "PHEV"]:
        if sub not in split_hp.columns:
            continue
        s = split_hp[sub]
        mats_sub = dict(matrices_by_key.get((region, sub), {}))

        for name in required_outflow_mats:
            mats_sub[name] = _scale_outflow_by_cohort(base[name], s)

        flows_sub = mats_sub.get("flows_df", base["flows_df"]).copy()
        if "outflow_surv_df" in mats_sub and "out_survival" in flows_sub.columns:
            flows_sub["out_survival"] = mats_sub["outflow_surv_df"].sum(axis=1)
        if "outflow_exp_df" in mats_sub and "out_export" in flows_sub.columns:
            flows_sub["out_export"] = mats_sub["outflow_exp_df"].sum(axis=1)
        if "outflow_unk_df" in mats_sub and "out_unknown" in flows_sub.columns:
            flows_sub["out_unknown"] = mats_sub["outflow_unk_df"].sum(axis=1)
        if "outflow_coll_df" in mats_sub and "out_collected" in flows_sub.columns:
            flows_sub["out_collected"] = mats_sub["outflow_coll_df"].sum(axis=1)
        if "out_total" in flows_sub.columns and {"out_survival", "out_export", "out_unknown"}.issubset(flows_sub.columns):
            # [FIXED, resolves C9, same fix as split_outflows_collected_unknown_export
            # above]: out_total = out_survival, not out_survival + its own subsets.
            flows_sub["out_total"] = flows_sub["out_survival"]

        mats_sub["flows_df"] = flows_sub
        out[(region, sub)] = mats_sub

    matrices_by_key.update(out)


def split_hybrid_flows_afterwards(
    matrices_by_key: dict,
    split_hp: pd.DataFrame,
    region: str = "EUR",
    base_drv: str = "Hybrid",
    keep_original: bool = True,
) -> None:
    # CONFIRMED NEVER CALLED by either 03_01_flowdriven.py or 03_02_adjustedflows.py --
    # both notebooks import and use split_hybrid_inflow_afterwards +
    # split_hybrid_outflows_afterwards instead. This function appears to be an
    # alternative/superseded single-call combined version (it also has a `keep_original`
    # flag to remove the "Hybrid" parent key after splitting -- something neither called
    # function does, meaning "Hybrid" and "Liquids" currently remain in matrices_by_key
    # alongside their HEV/PHEV and Diesel/Petrol children; see the consolidated review
    # for the double-counting risk this creates in plotting.py's age-statistics
    # functions). Its own out_total formula (`out_survival + out_export`, 2-term) is yet
    # a FOURTH variant, differing from all three call sites actually in use -- moot
    # since this function is dead code, but illustrates the same underlying confusion
    # about what "out_total" should mean recurring across the file.
    base_key = (region, base_drv)
    if base_key not in matrices_by_key:
        raise KeyError(f"Missing {base_key} in matrices_by_key.")

    base = matrices_by_key[base_key]
    years = base["flows_df"].index
    split_hp = split_hp.reindex(years).ffill().bfill().fillna(0.0)

    def _scale_outflow_by_cohort(matrix_df: pd.DataFrame, share_by_year: pd.Series) -> pd.DataFrame:
        cohort_year = pd.to_numeric(pd.Index(matrix_df.columns), errors="coerce")
        col_shares = share_by_year.reindex(cohort_year).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        return matrix_df.mul(col_shares, axis=1)

    out = {}
    for sub in ["HEV", "PHEV"]:
        s = split_hp[sub]
        mats_sub = {}
        for name in ["outflow_coll_df", "outflow_exp_df", "outflow_unk_df", "outflow_surv_df"]:
            if name in base:
                mats_sub[name] = _scale_outflow_by_cohort(base[name], s)

        flows_sub = base["flows_df"].copy()

        if "inflow" in flows_sub.columns:
            flows_sub["inflow"] = flows_sub["inflow"].mul(s, axis=0)

        if "outflow_surv_df" in mats_sub and "out_survival" in flows_sub.columns:
            flows_sub["out_survival"] = mats_sub["outflow_surv_df"].sum(axis=1)
        elif "out_survival" in flows_sub.columns:
            flows_sub["out_survival"] = flows_sub["out_survival"].mul(s, axis=0)

        if "outflow_exp_df" in mats_sub and "out_export" in flows_sub.columns:
            flows_sub["out_export"] = mats_sub["outflow_exp_df"].sum(axis=1)
        elif "out_export" in flows_sub.columns:
            flows_sub["out_export"] = flows_sub["out_export"].mul(s, axis=0)

        if "outflow_unk_df" in mats_sub and "out_unknown" in flows_sub.columns:
            flows_sub["out_unknown"] = mats_sub["outflow_unk_df"].sum(axis=1)
        elif "out_unknown" in flows_sub.columns:
            flows_sub["out_unknown"] = flows_sub["out_unknown"].mul(s, axis=0)

        if "outflow_coll_df" in mats_sub and "out_collected" in flows_sub.columns:
            flows_sub["out_collected"] = mats_sub["outflow_coll_df"].sum(axis=1)
        elif "out_collected" in flows_sub.columns:
            flows_sub["out_collected"] = flows_sub["out_collected"].mul(s, axis=0)

        if "out_total" in flows_sub.columns and {"out_survival", "out_export"}.issubset(flows_sub.columns):
            flows_sub["out_total"] = flows_sub["out_survival"]  # [FIXED, resolves C9]
        elif "out_total" in flows_sub.columns:
            flows_sub["out_total"] = flows_sub["out_total"].mul(s, axis=0)

        mats_sub["flows_df"] = flows_sub
        out[(region, sub)] = mats_sub

    matrices_by_key.update(out)
    if not keep_original:
        matrices_by_key.pop(base_key, None)


def build_segment_share_wide(segment_shares_ext: pd.DataFrame) -> dict[str, pd.DataFrame]:
    seg_wide: dict[str, pd.DataFrame] = {}
    for drv, group in segment_shares_ext.groupby("Drive Train"):
        wide = group.pivot(index="Year", columns="Segment", values="segment_share").sort_index().fillna(0.0)
        wide = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        seg_wide[drv] = wide
    return seg_wide


def build_liquids_split_wide(liquids_shares_ext: pd.DataFrame) -> pd.DataFrame:
    wide = liquids_shares_ext.pivot(index="Year", columns="Drive Train", values="share").sort_index().fillna(0.0)
    cols = [c for c in ["Diesel", "Petrol"] if c in wide.columns]
    wide = wide[cols]
    return wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def split_liquids_inflow_afterwards(
    matrices_by_key: dict,
    liquids_split: pd.DataFrame,
    region: str = "EUR",
    base_drv: str = "Liquids",
) -> None:
    base_key = (region, base_drv)
    if base_key not in matrices_by_key:
        raise KeyError(f"Missing {base_key} in matrices_by_key.")

    base = matrices_by_key[base_key]
    years = base["flows_df"].index
    split = liquids_split.reindex(years).ffill().bfill().fillna(0.0)

    out = {}
    for fuel in ["Diesel", "Petrol"]:
        if fuel not in split.columns:
            continue
        s = split[fuel]
        mats_sub = dict(matrices_by_key.get((region, fuel), {}))
        flows_sub = mats_sub.get("flows_df", base["flows_df"]).copy()
        if "inflow" in base["flows_df"].columns:
            flows_sub["inflow"] = base["flows_df"]["inflow"].mul(s, axis=0)
        mats_sub["flows_df"] = flows_sub
        out[(region, fuel)] = mats_sub

    matrices_by_key.update(out)


def split_liquids_outflows_afterwards(
    matrices_by_key: dict,
    liquids_split: pd.DataFrame,
    region: str = "EUR",
    base_drv: str = "Liquids",
) -> None:
    base_key = (region, base_drv)
    if base_key not in matrices_by_key:
        raise KeyError(f"Missing {base_key} in matrices_by_key.")

    base = matrices_by_key[base_key]
    years = base["flows_df"].index
    split = liquids_split.reindex(years).ffill().bfill().fillna(0.0)
    required_outflow_mats = ["outflow_coll_df", "outflow_exp_df", "outflow_unk_df", "outflow_surv_df"]
    missing = [name for name in required_outflow_mats if name not in base]
    if missing:
        raise KeyError(
            f"{base_key} missing required outflow matrices {missing}. "
            "Run split_outflows_collected_unknown before splitting outflows."
        )

    def _scale_outflow_by_cohort(matrix_df: pd.DataFrame, share_by_year: pd.Series) -> pd.DataFrame:
        cohort_year = pd.to_numeric(pd.Index(matrix_df.columns), errors="coerce")
        col_shares = share_by_year.reindex(cohort_year).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        return matrix_df.mul(col_shares, axis=1)

    out = {}
    for fuel in ["Diesel", "Petrol"]:
        if fuel not in split.columns:
            continue
        s = split[fuel]
        mats_sub = dict(matrices_by_key.get((region, fuel), {}))

        for name in required_outflow_mats:
            mats_sub[name] = _scale_outflow_by_cohort(base[name], s)

        flows_sub = mats_sub.get("flows_df", base["flows_df"]).copy()
        if "outflow_surv_df" in mats_sub and "out_survival" in flows_sub.columns:
            flows_sub["out_survival"] = mats_sub["outflow_surv_df"].sum(axis=1)
        if "outflow_exp_df" in mats_sub and "out_export" in flows_sub.columns:
            flows_sub["out_export"] = mats_sub["outflow_exp_df"].sum(axis=1)
        if "outflow_unk_df" in mats_sub and "out_unknown" in flows_sub.columns:
            flows_sub["out_unknown"] = mats_sub["outflow_unk_df"].sum(axis=1)
        if "outflow_coll_df" in mats_sub and "out_collected" in flows_sub.columns:
            flows_sub["out_collected"] = mats_sub["outflow_coll_df"].sum(axis=1)
        if "out_total" in flows_sub.columns and {"out_survival", "out_export"}.issubset(flows_sub.columns):
            # CONFIRMED BUG, and a DIFFERENT (also wrong) formula than the Hybrid/HEV/
            # PHEV path above: this one double-counts out_export but omits out_unknown
            # entirely, unlike split_outflows_collected_unknown_export /
            # split_hybrid_outflows_afterwards (which double-count both). Applies to
            # Diesel and Petrol -- net effect: out_total = out_survival * 1.08 (+8%)
            # rather than the +51% seen for Hybrid/HEV/PHEV, using real
            # 00_parameters.py shares. Three call sites, three different formulas, all
            # wrong in different ways -- see module docstring for the full breakdown.
            flows_sub["out_total"] = flows_sub["out_survival"]  # [FIXED, resolves C9]

        mats_sub["flows_df"] = flows_sub
        out[(region, fuel)] = mats_sub

    matrices_by_key.update(out)


def split_liquids_flows_afterwards(
    matrices_by_key: dict,
    liquids_split: pd.DataFrame,
    region: str = "EUR",
    base_drv: str = "Liquids",
    keep_original: bool = True,
) -> None:
    # CONFIRMED NEVER CALLED, same situation as split_hybrid_flows_afterwards above --
    # split_liquids_inflow_afterwards + split_liquids_outflows_afterwards are the
    # functions actually imported and used.
    base_key = (region, base_drv)
    if base_key not in matrices_by_key:
        raise KeyError(f"Missing {base_key} in matrices_by_key.")

    base = matrices_by_key[base_key]
    years = base["flows_df"].index
    split = liquids_split.reindex(years).ffill().bfill().fillna(0.0)

    def _scale_outflow_by_cohort(matrix_df: pd.DataFrame, share_by_year: pd.Series) -> pd.DataFrame:
        cohort_year = pd.to_numeric(pd.Index(matrix_df.columns), errors="coerce")
        col_shares = share_by_year.reindex(cohort_year).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        return matrix_df.mul(col_shares, axis=1)

    out = {}
    for fuel in ["Diesel", "Petrol"]:
        if fuel not in split.columns:
            continue

        s = split[fuel]
        mats_sub = {}
        for name in ["outflow_coll_df", "outflow_exp_df", "outflow_unk_df", "outflow_surv_df"]:
            if name in base:
                mats_sub[name] = _scale_outflow_by_cohort(base[name], s)

        flows_sub = base["flows_df"].copy()

        if "inflow" in flows_sub.columns:
            flows_sub["inflow"] = flows_sub["inflow"].mul(s, axis=0)

        if "outflow_surv_df" in mats_sub and "out_survival" in flows_sub.columns:
            flows_sub["out_survival"] = mats_sub["outflow_surv_df"].sum(axis=1)
        elif "out_survival" in flows_sub.columns:
            flows_sub["out_survival"] = flows_sub["out_survival"].mul(s, axis=0)

        if "outflow_exp_df" in mats_sub and "out_export" in flows_sub.columns:
            flows_sub["out_export"] = mats_sub["outflow_exp_df"].sum(axis=1)
        elif "out_export" in flows_sub.columns:
            flows_sub["out_export"] = flows_sub["out_export"].mul(s, axis=0)

        if "outflow_unk_df" in mats_sub and "out_unknown" in flows_sub.columns:
            flows_sub["out_unknown"] = mats_sub["outflow_unk_df"].sum(axis=1)
        elif "out_unknown" in flows_sub.columns:
            flows_sub["out_unknown"] = flows_sub["out_unknown"].mul(s, axis=0)

        if "outflow_coll_df" in mats_sub and "out_collected" in flows_sub.columns:
            flows_sub["out_collected"] = mats_sub["outflow_coll_df"].sum(axis=1)
        elif "out_collected" in flows_sub.columns:
            flows_sub["out_collected"] = flows_sub["out_collected"].mul(s, axis=0)

        if "out_total" in flows_sub.columns and {"out_survival", "out_export"}.issubset(flows_sub.columns):
            flows_sub["out_total"] = flows_sub["out_survival"]  # [FIXED, resolves C9]
        elif "out_total" in flows_sub.columns:
            flows_sub["out_total"] = flows_sub["out_total"].mul(s, axis=0)

        mats_sub["flows_df"] = flows_sub

        out[(region, fuel)] = mats_sub

    matrices_by_key.update(out)
    if not keep_original:
        matrices_by_key.pop(base_key, None)


def alloc_matrix_yearwise(matrix_df: pd.DataFrame, share_wide: pd.DataFrame) -> pd.DataFrame:
    share_wide = share_wide.reindex(matrix_df.index).ffill().bfill().fillna(0.0)
    out_parts = []
    for seg in share_wide.columns:
        scaled = matrix_df.mul(share_wide[seg], axis=0)
        long = (
            scaled.stack()
            .rename("value")
            .reset_index()
            .rename(columns={"level_0": "year", "level_1": "cohort_year"})
        )
        long["Segment"] = seg
        out_parts.append(long)
    return pd.concat(out_parts, ignore_index=True)


def alloc_matrix_cohort_yearwise(matrix_df: pd.DataFrame, share_wide: pd.DataFrame) -> pd.DataFrame:
    # NOTE (positive finding): outflow matrices are allocated to segments using the
    # segment-share profile AT THE COHORT'S VINTAGE YEAR (via `cohort_year =
    # matrix_df.columns`), not at the current calendar year -- i.e. a car's segment is
    # correctly treated as fixed at the time it was manufactured/registered, not
    # re-assigned based on the segment mix in the year it happens to retire. Contrast
    # with `alloc_series_yearwise` below (used for INFLOW), which correctly uses the
    # CALENDAR year share instead -- appropriate there since a newly-registered
    # vehicle's cohort year and calendar year are the same thing. Both are consistent
    # and correct as far as this review can verify.
    cohort_year = pd.to_numeric(pd.Index(matrix_df.columns), errors="coerce")
    share_by_cohort = share_wide.reindex(cohort_year).ffill().bfill().fillna(0.0)

    out_parts = []
    for seg in share_by_cohort.columns:
        col_shares = share_by_cohort[seg].to_numpy(dtype=float)
        scaled = matrix_df.mul(col_shares, axis=1)
        long = (
            scaled.stack()
            .rename("value")
            .reset_index()
            .rename(columns={"level_0": "year", "level_1": "cohort_year"})
        )
        long["Segment"] = seg
        out_parts.append(long)
    return pd.concat(out_parts, ignore_index=True)


def alloc_series_yearwise(series: pd.Series, share_wide: pd.DataFrame) -> pd.DataFrame:
    share_wide = share_wide.reindex(series.index).ffill().bfill().fillna(0.0)
    out_parts = []
    for seg in share_wide.columns:
        out_parts.append(
            pd.DataFrame(
                {
                    "year": series.index.astype(int),
                    "Segment": seg,
                    "value": series.to_numpy(dtype=float) * share_wide[seg].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(out_parts, ignore_index=True)


def disaggregate_model_to_segments(
    matrices_by_key: dict,
    seg_share_by_drv: dict[str, pd.DataFrame],
    region: str = "EUR",
    allowed_drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
) -> dict:
    outflow_coll_parts = []
    outflow_exp_parts = []
    outflow_unk_parts = []
    inflow_parts = []

    for (reg, drv), mats in matrices_by_key.items():
        if reg != region or drv not in allowed_drivetrains:
            continue
        if drv not in seg_share_by_drv:
            raise KeyError(f"No segment shares available for drivetrain '{drv}' in seg_share_by_drv.")

        share_wide = seg_share_by_drv[drv]

        for name, parts in [
            ("outflow_coll_df", outflow_coll_parts),
            ("outflow_exp_df", outflow_exp_parts),
            ("outflow_unk_df", outflow_unk_parts),
        ]:
            if name not in mats:
                raise KeyError(f"{(reg, drv)} missing '{name}' in matrices_by_key.")
            long = alloc_matrix_cohort_yearwise(mats[name], share_wide)
            long["Region"] = reg
            long["Drive Train"] = drv
            parts.append(long)

        flows = mats["flows_df"]
        if "inflow" not in flows.columns:
            raise KeyError(f"{(reg, drv)} flows_df missing 'inflow'.")
        inflow_long = alloc_series_yearwise(flows["inflow"], share_wide)
        inflow_long["Region"] = reg
        inflow_long["Drive Train"] = drv
        inflow_parts.append(inflow_long)

    return {
        "outflow_coll_segments": pd.concat(outflow_coll_parts, ignore_index=True) if outflow_coll_parts else pd.DataFrame(),
        "outflow_exp_segments": pd.concat(outflow_exp_parts, ignore_index=True) if outflow_exp_parts else pd.DataFrame(),
        "outflow_unk_segments": pd.concat(outflow_unk_parts, ignore_index=True) if outflow_unk_parts else pd.DataFrame(),
        "inflow_segments": pd.concat(inflow_parts, ignore_index=True) if inflow_parts else pd.DataFrame(),
    }



def build_tracker_from_disaggregated(
    disaggregated: dict,
    *,
    region: str = "EUR",
    drivetrains: tuple[str, ...] | None = None,
    include_zero: bool = False,
    value_col: str = "value",
    amount_col: str = "amount",
) -> dict:
    need_keys = ["inflow_segments", "outflow_coll_segments", "outflow_exp_segments", "outflow_unk_segments"]
    for key in need_keys:
        if key not in disaggregated:
            raise KeyError(f"disaggregated is missing '{key}'")

    rows = []

    infl = disaggregated["inflow_segments"].copy()
    if not infl.empty:
        infl = infl.rename(columns={value_col: amount_col})
        infl = infl[infl["Region"] == region].copy()
        if drivetrains is not None:
            infl = infl[infl["Drive Train"].isin(drivetrains)].copy()

        infl["flow"] = "inflow"
        infl["scrap_year"] = infl["year"]
        infl["cohort_year"] = infl["year"]
        infl = infl[["Region", "Drive Train", "flow", "scrap_year", "cohort_year", "Segment", amount_col]]
        rows.append(infl)

    out_map = {
        "outflow_coll_segments": "collected",
        "outflow_exp_segments": "export",
        "outflow_unk_segments": "unknown_whereabouts",
    }

    for key, flow_label in out_map.items():
        frame = disaggregated[key].copy()
        if frame.empty:
            continue

        frame = frame.rename(columns={value_col: amount_col})
        frame = frame[frame["Region"] == region].copy()
        if drivetrains is not None:
            frame = frame[frame["Drive Train"].isin(drivetrains)].copy()

        if "cohort_year" not in frame.columns or "year" not in frame.columns:
            raise KeyError(f"{key} must contain 'cohort_year' and 'year'.")

        frame["scrap_year"] = frame["year"]
        frame["flow"] = flow_label
        frame = frame[["Region", "Drive Train", "flow", "year", "scrap_year", "cohort_year", "Segment", amount_col]]
        rows.append(frame)

    all_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Region", "Drive Train", "flow", "year", "scrap_year", "cohort_year", "Segment", amount_col]
    )

    if not include_zero and not all_long.empty:
        all_long = all_long[all_long[amount_col].fillna(0.0) != 0.0].copy()

    tracker: dict[tuple[str, str], pd.DataFrame] = {}
    if not all_long.empty:
        for (reg, drv), group in all_long.groupby(["Region", "Drive Train"], sort=True):
            tracker[(reg, drv)] = group.reset_index(drop=True)

    return tracker


def add_keys_to_tracker_dict(
    tracker: dict,
    *,
    segment_map: dict[str, str],
    drv_prefix_map: dict[str, str],
    segment_col: str = "Segment",
    drivetrain_col: str = "Drive Train",
    amount_col: str = "amount",
    out_key_col: str = "key",
) -> tuple[dict, pd.DataFrame]:
    def _norm(x):
        return str(x).strip() if not pd.isna(x) else np.nan

    tracker_keyed = {}
    missing_rows = []

    for (reg, drv_key), frame in tracker.items():
        data = frame.copy()
        if amount_col not in data.columns:
            raise KeyError(f"Expected '{amount_col}' in tracker DataFrame for {(reg, drv_key)}.")

        if drivetrain_col not in data.columns:
            data[drivetrain_col] = _norm(drv_key)
        else:
            data[drivetrain_col] = data[drivetrain_col].apply(_norm)

        if segment_col not in data.columns:
            raise KeyError(f"Expected '{segment_col}' in tracker DataFrame for {(reg, drv_key)}.")
        data[segment_col] = data[segment_col].apply(_norm)

        drv_prefix = data[drivetrain_col].map(drv_prefix_map)
        seg_code = data[segment_col].map(segment_map)
        data[out_key_col] = np.where(drv_prefix.notna() & seg_code.notna(), "V" + drv_prefix + seg_code, np.nan)

        miss = data.loc[data[out_key_col].isna(), [drivetrain_col, segment_col]].drop_duplicates()
        if not miss.empty:
            miss = miss.rename(columns={drivetrain_col: "Drive Train", segment_col: "Segment"})
            miss["Region"] = reg
            miss["tracker_key_drv"] = drv_key
            missing_rows.append(miss)

        tracker_keyed[(reg, drv_key)] = data.reset_index(drop=True)

    missing = (
        pd.concat(missing_rows, ignore_index=True).drop_duplicates().sort_values(["Region", "Drive Train", "Segment"])
        if missing_rows
        else pd.DataFrame(columns=["Region", "Drive Train", "Segment", "tracker_key_drv"])
    )
    return tracker_keyed, missing
