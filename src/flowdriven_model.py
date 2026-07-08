"""
flowdriven_model.py
=====================

Core flow-driven cohort model used by stage 03 (`03_01_flowdriven.py`,
`03_02_adjustedflows.py`). This is the module (`fdm`) referenced but not previously
available in earlier rounds of this review -- now fully annotated below.

NOTE ON THIS FILE: no logic changed, only comments added. Everything below was verified
by reading the actual source, not inferred from usage as in previous rounds.

======================================================================
HEADLINE FINDING: SIX of `run_flow_driven_model_with_outflow_disaggregation`'s ~20
parameters are accepted but NEVER READ inside the function body:
`export_r_by_drv`, `age_bins`, `export_rate`, `export_total_by_year`,
`segment_shares_by_drv`, `allowed_export_segments`.
======================================================================
Confirmed by reading the entire function body (see the function itself, further down,
for exactly where each becomes dead). Practical consequences:

- The elaborate age-bucket export-allocation machinery in this same file
  (`allocate_exports_bucket_rates`, `build_export_probability_lookup`) is NEVER CALLED
  by the model. Exports are actually allocated as a flat, age-independent proportional
  split: `out_export = out_survival * export_share_by_drivetrain[drv]` -- the same
  share for every cohort/age, regardless of how old it is. The entire "export
  probability by age" empirical analysis computed and plotted in `03_02_adjustedflows.py`
  (cells deriving `export_prob_by_age_drv`) has NO effect on any model output -- it's
  disconnected exploratory analysis.
- Passing 5 different `segment_shares_by_drv` profiles (BAU/A_F/JA_JF/Large/Small) into
  `run_adjusted_scenario` in `03_02_adjustedflows.py` does NOTHING inside this function --
  the actual segment-mix change for those scenarios comes entirely from the UPSTREAM
  `tweak_inflow_segment_shares_within_drivetrain()` call that reshapes the input `df`
  BEFORE it reaches this function. This resolves an open question from the previous
  review round: `segment_shares_by_drv`'s extra profiles are confirmed vestigial as far
  as this function is concerned.
======================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NOTE: uses the same `src.xxx` import convention as pipeline stage scripts
# (e.g. `03_02_adjustedflows.py`'s `import src.artifacts as artifacts`), not a
# bare `import cohort_flow_mc` -- a bare import would only work if `src/`
# itself (not just the project root) were on `sys.path`, which it isn't;
# every stage script only appends PROJECT_ROOT.
import src.cohort_flow_mc as _cohort_flow_mc


def build_export_probability_lookup(
    export_prob_by_age_drv_mapped: pd.DataFrame,
) -> dict[tuple[str, str, int], float]:
    """
    Build lookup:
    {
        (Region, Drive Train, age): export_probability
    }

    CONFIRMED ORPHANED: this function's output is never passed into
    `run_flow_driven_model_with_outflow_disaggregation` (grep-confirmed across both
    03_01 and 03_02 call sites) -- the age-based export probabilities computed and
    plotted in `03_02_adjustedflows.py` (`export_prob_by_age_drv`) exist purely for
    visual inspection and have no effect on the model. See module docstring.
    """
    required = {"Region", "Drive Train", "age", "export_probability"}
    missing = required - set(export_prob_by_age_drv_mapped.columns)
    if missing:
        raise KeyError(
            f"Missing required columns in export_prob_by_age_drv_mapped: {sorted(missing)}"
        )

    df = export_prob_by_age_drv_mapped.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype(int)
    df["export_probability"] = (
        pd.to_numeric(df["export_probability"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    lookup: dict[tuple[str, str, int], float] = {}
    for _, row in df.iterrows():
        key = (
            str(row["Region"]),
            str(row["Drive Train"]),
            int(row["age"]),
        )
        lookup[key] = float(row["export_probability"])

    return lookup

def prepare_full_inflow_table(
    synthetic_pre_2005_inflows: pd.DataFrame,
    inflow_segments_scenario: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["Region", "Drive Train", "Segment", "year", "value"]

    pre = synthetic_pre_2005_inflows[cols].copy()
    scen = inflow_segments_scenario[cols].copy()

    df = pd.concat([pre, scen], ignore_index=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    df = (
        df.groupby(["Region", "Drive Train", "Segment", "year"], as_index=False)["value"]
        .sum()
        .sort_values(["Region", "Drive Train", "Segment", "year"])
        .reset_index(drop=True)
    )
    return df


def build_segmented_starting_stock_and_synthetic_inflows(
    stock_inspect_df: pd.DataFrame,
    split_hp: pd.DataFrame,
    liquids_split: pd.DataFrame,
    segment_shares_ext: pd.DataFrame,
    params: dict,
    region: str = "EUR",
    base_year: int = 2005,
    backcast_start_year: int = 1975,
    drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
    display_fn=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build target stock by drivetrain and segment for base_year,
    then backcast synthetic pre-base-year inflows.
    """
    if stock_inspect_df is None or stock_inspect_df.empty:
        raise ValueError("stock_inspect_df is missing or empty.")
    if split_hp is None or liquids_split is None:
        raise ValueError("split_hp or liquids_split missing.")
    if segment_shares_ext is None or segment_shares_ext.empty:
        raise ValueError("segment_shares_ext is missing.")

    stock_base = stock_inspect_df[
        stock_inspect_df["Region"].eq(region) & stock_inspect_df["year"].eq(base_year)
    ].copy()
    if stock_base.empty:
        raise ValueError(f"No stock data found for {region} in {base_year}.")

    starting_total = float(stock_base["stock"].sum())
    shares_by_drv = stock_base.groupby("Drive Train", as_index=True)["stock"].sum()

    bev_share = float(shares_by_drv.get("BEV", 0.0))
    hybrid_share = float(
        shares_by_drv.get("Hybrid", shares_by_drv.get("HEV", 0.0) + shares_by_drv.get("PHEV", 0.0))
    )
    liquids_share = float(
        shares_by_drv.get("Liquids", shares_by_drv.get("Diesel", 0.0) + shares_by_drv.get("Petrol", 0.0))
    )

    share_sum = bev_share + hybrid_share + liquids_share
    if share_sum <= 0:
        raise ValueError(f"Could not build BEV/Hybrid/Liquids shares for {base_year}.")

    base_shares = pd.Series(
        {
            "BEV": bev_share / share_sum,
            "Hybrid": hybrid_share / share_sum,
            "Liquids": liquids_share / share_sum,
        }
    )
    base_stock = base_shares * starting_total

    if base_year not in split_hp.index or base_year not in liquids_split.index:
        raise ValueError(f"Missing {base_year} in split_hp or liquids_split.")

    hybrid_internal = split_hp.loc[base_year, ["HEV", "PHEV"]].astype(float)
    hybrid_internal = hybrid_internal / hybrid_internal.sum()

    liquids_internal = liquids_split.loc[base_year, ["Diesel", "Petrol"]].astype(float)
    liquids_internal = liquids_internal / liquids_internal.sum()

    starting_stock = pd.Series(
        {
            "BEV": base_stock["BEV"],
            "HEV": base_stock["Hybrid"] * hybrid_internal["HEV"],
            "PHEV": base_stock["Hybrid"] * hybrid_internal["PHEV"],
            "Diesel": base_stock["Liquids"] * liquids_internal["Diesel"],
            "Petrol": base_stock["Liquids"] * liquids_internal["Petrol"],
        },
        name="value",
    )

    segment_shares_base = (
        segment_shares_ext[segment_shares_ext["Year"] == base_year]
        .pivot(index="Drive Train", columns="Segment", values="segment_share")
        .fillna(0.0)
    )

    segment_shares_base = segment_shares_base.loc[
        [drv for drv in starting_stock.index if drv in segment_shares_base.index]
    ]

    starting_stock_segments = (
        segment_shares_base.mul(starting_stock, axis=0)
        .stack()
        .reset_index()
        .rename(columns={0: "value"})
    )
    starting_stock_segments["Region"] = region
    starting_stock_segments = starting_stock_segments[
        ["Region", "Drive Train", "Segment", "value"]
    ].copy()

    mapped = build_p02_mapped_inputs(params["02_stock_flow"], drivetrains=drivetrains)

    synthetic_pre_baseyear_inflows = build_synthetic_pre_baseyear_inflows(
        starting_stock_segments=starting_stock_segments,
        lifetime_by_drv=mapped["lifetime_by_drv"],
        base_year=base_year,
        backcast_start_year=backcast_start_year,
        group_cols=["Region", "Drive Train", "Segment"],
        value_col="value",
    )

    if display_fn:
        display_fn(f"Starting total stock ({region}, {base_year}): {starting_total:,.3f}")
        display_fn(f"Segmented total stock ({region}, {base_year}): {starting_stock_segments['value'].sum():,.3f}")
        display_fn(f"Synthetic inflow total: {synthetic_pre_baseyear_inflows['value'].sum():,.3f}")

    return starting_stock_segments, synthetic_pre_baseyear_inflows





def build_stock_by_segment_at_base_year(
    matrices_by_key: dict,
    seg_share_by_drv: dict[str, pd.DataFrame],
    *,
    region: str = "EUR",
    base_year: int = 2005,
    allowed_drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
) -> pd.DataFrame:
    """
    Build stock by cohort and segment at the base year from stock_t_tau_df.

    THE MATH MODEL: reuses stage 02's cohort structure directly (does NOT re-derive a
    backcast). For each cohort (vintage year), its base-year stock is split across
    segments using a SINGLE base-year segment-share profile (`seg_share_by_drv[drv]` at
    `base_year`) applied UNIFORMLY to every cohort regardless of vintage:

        stock[drv, seg, cohort] = stock_t_tau[base_year, cohort] * seg_share[drv, seg, base_year]

    This implicitly assumes a cohort's segment composition doesn't vary by vintage --
    e.g. a surviving 1985-vintage Petrol cohort is assumed to have the SAME segment mix
    as a 2005-vintage Petrol cohort, which is a simplifying assumption (real segment mix
    likely drifted over decades) but a reasonable one absent vintage-specific segment
    data. This is what `03_02_adjustedflows.py`'s scenario re-runs initialize from,
    NOT a fresh backcast -- see the consolidated review, finding C8.
    """
    rows: list[dict] = []

    for (reg, drv), mats in matrices_by_key.items():
        if reg != region or drv not in allowed_drivetrains:
            continue
        if "stock_t_tau_df" not in mats:
            continue
        if drv not in seg_share_by_drv:
            continue

        stock_t_tau = mats["stock_t_tau_df"].copy()
        if base_year not in stock_t_tau.index:
            continue

        stock_base = stock_t_tau.loc[base_year].copy()
        stock_base.index = pd.to_numeric(stock_base.index, errors="coerce").astype(int)

        share_wide = seg_share_by_drv[drv]
        if base_year not in share_wide.index:
            raise ValueError(f"Missing segment shares for {drv} in {base_year}.")

        seg_shares = share_wide.loc[base_year].fillna(0.0)
        seg_shares = seg_shares / seg_shares.sum()

        for cohort_year, stock_value in stock_base.items():
            stock_value = float(stock_value)
            if stock_value == 0.0:
                continue

            for seg, seg_share in seg_shares.items():
                rows.append(
                    {
                        "Region": reg,
                        "Drive Train": drv,
                        "Segment": seg,
                        "year": int(base_year),
                        "cohort_year": int(cohort_year),
                        "stock": stock_value * float(seg_share),
                    }
                )

    return pd.DataFrame(rows)


def build_starting_stock_by_cohort_lookup(
    stock_by_segment_at_base_year: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    cohort_year_col: str = "cohort_year",
    stock_col: str = "stock",
) -> dict[tuple, dict[int, float]]:
    """
    Convert base-year stock by cohort table into nested lookup:
    {
        (Region, Drive Train, Segment): {cohort_year: stock, ...}
    }
    """
    if group_cols is None:
        group_cols = ["Region", "Drive Train", "Segment"]

    required = set(group_cols + [cohort_year_col, stock_col])
    missing = required - set(stock_by_segment_at_base_year.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    lookup: dict[tuple, dict[int, float]] = {}

    for group_key, sub in stock_by_segment_at_base_year.groupby(group_cols, dropna=False):
        tmp = sub.sort_values(cohort_year_col)
        lookup[group_key] = dict(
            zip(
                tmp[cohort_year_col].astype(int),
                tmp[stock_col].astype(float),
            )
        )

    return lookup


def plot_relative_split(
    df_split: pd.DataFrame,
    *,
    title: str,
    year_plot_start: int | None = None,
    year_plot_end: int | None = None,
) -> None:
    """
    Plot the relative share of columns in a DataFrame over time, normalized per row.
    """
    d = df_split.copy()
    d.index = d.index.astype(int)

    if year_plot_start is not None and year_plot_end is not None:
        d = d[(d.index >= int(year_plot_start)) & (d.index <= int(year_plot_end))]

    d = d.div(d.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    _, ax = plt.subplots(figsize=(10, 5))
    d.plot(ax=ax, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative share")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, title="")
    plt.tight_layout()
    plt.show()




def build_p02_mapped_inputs(
    p02: dict,
    drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
) -> dict:
    """
    Build and map required inputs from the p02 parameter dictionary.

    CONFIRMED BUG (real, silent, high severity): `export_r_by_drv` and `age_bins` below
    look for keys `p02["export_r_by_drv"]` and `p02["age_bins"]` -- but
    `00_parameters.py`'s actual `PARAMS["02_stock_flow"]` dict has NEITHER key. It only
    has `EXPORT_SHARE_BY_DRV` (a flat per-drivetrain scalar share, different name AND
    different shape) and no `age_bins` at all. This means:
      - `p02.get("export_r_by_drv", {})` ALWAYS returns `{}`, so `export_r_by_drv[drv]`
        ALWAYS falls back to `np.ones(3, dtype=float)` for every drivetrain, no matter
        what's configured in `00_parameters.py`.
      - `age_bins = p02.get("age_bins")` is ALWAYS `None`, so it ALWAYS falls back to the
        hardcoded default `[(0, 4), (5, 9), (10, 30)]`.
    In other words: two of this function's four outputs are, in practice, NEVER
    configurable from `00_parameters.py` -- they always resolve to the hardcoded
    defaults below, regardless of what anyone edits in the params file. This matters
    because `export_r_by_drv` and `age_bins` are the inputs to
    `allocate_exports_bucket_rates` (age-weighted export allocation) -- which, per the
    module docstring above, is never even called by the model. So this silent fallback
    currently has no downstream numerical consequence (the age-bucket allocation path is
    dead entirely) -- but if that code path is ever activated, anyone editing
    `00_parameters.py` expecting to control it would find their edits have no effect,
    with no error or warning anywhere.
    """
    lifetime_by_drv: dict[str, dict[str, float]] = {}
    for drv in drivetrains:
        drv_params = p02["lifetime_by_drv"].get(drv, {})
        lifetime_by_drv[drv] = {
            # NOTE: default shape_k here is 4.0 -- every real drivetrain in
            # 00_parameters.py uses shape_k=3.0. This default is silently substituted,
            # with no warning, only if a requested `drv` is entirely absent from
            # p02["lifetime_by_drv"] (not currently triggered -- all 9 real drivetrains
            # are present there -- but would silently produce an inconsistent Weibull
            # shape if a typo'd or new drivetrain name were ever passed).
            "shape_k": float(drv_params.get("shape_k", 4.0)),
            "scale_lambda": float(drv_params.get("scale_lambda", 13.0)),
        }

    export_r_by_drv: dict[str, np.ndarray] = {}
    for drv in drivetrains:
        # See CONFIRMED BUG note in this function's docstring: this key never exists in
        # the real p02, so export_r is always the fallback ones(3) below.
        export_r = p02.get("export_r_by_drv", {}).get(drv)
        if export_r is not None:
            export_r_by_drv[drv] = np.array(export_r, dtype=float)
        else:
            export_r_by_drv[drv] = np.ones(3, dtype=float)

    age_bins = p02.get("age_bins")  # see CONFIRMED BUG note above: always None in practice
    if age_bins is None:
        age_bins = [(0, 4), (5, 9), (10, 30)]

    unknown_whereabouts_share = p02.get("unknown_whereabouts_share", {})
    if not unknown_whereabouts_share:
        unknown_whereabouts_share = {drv: 0.0 for drv in drivetrains}

    return {
        "lifetime_by_drv": lifetime_by_drv,
        "export_r_by_drv": export_r_by_drv,
        "age_bins": age_bins,
        "unknown_whereabouts_share": unknown_whereabouts_share,
    }


def calculate_starting_stock_by_segment(
    stock_inspect_df: pd.DataFrame,
    split_hp: pd.DataFrame,
    liquids_split: pd.DataFrame,
    segment_shares_ext: pd.DataFrame,
    region: str,
    year: int,
) -> pd.DataFrame:
    """
    Calculate starting stock by drivetrain and segment for a given year and region.
    Returns a DataFrame with columns: Drive Train, Segment, stock
    """
    stock_year = stock_inspect_df[
        stock_inspect_df["Region"].eq(region) & stock_inspect_df["year"].eq(year)
    ].copy()
    if stock_year.empty:
        raise ValueError(f"No stock data found for {region} in {year}.")

    starting_total = float(stock_year["stock"].sum())
    shares_by_drv = stock_year.groupby("Drive Train", as_index=True)["stock"].sum()
    bev_share = float(shares_by_drv.get("BEV", 0.0))
    hybrid_share = float(shares_by_drv.get("Hybrid", shares_by_drv.get("HEV", 0.0) + shares_by_drv.get("PHEV", 0.0)))
    liquids_share = float(shares_by_drv.get("Liquids", shares_by_drv.get("Diesel", 0.0) + shares_by_drv.get("Petrol", 0.0)))
    base_share_sum = bev_share + hybrid_share + liquids_share
    if base_share_sum <= 0:
        raise ValueError("Could not build BEV/Hybrid/Liquids shares for year.")
    base_shares = pd.Series({
        "BEV": bev_share / base_share_sum,
        "Hybrid": hybrid_share / base_share_sum,
        "Liquids": liquids_share / base_share_sum,
    })
    base_stock = base_shares * starting_total
    if year not in split_hp.index or year not in liquids_split.index:
        raise ValueError(f"Missing {year} in split_hp or liquids_split.")
    hybrid_internal = split_hp.loc[year, ["HEV", "PHEV"]].astype(float)
    hybrid_internal = hybrid_internal / hybrid_internal.sum()
    liquids_internal = liquids_split.loc[year, ["Diesel", "Petrol"]].astype(float)
    liquids_internal = liquids_internal / liquids_internal.sum()
    starting_stock = pd.Series({
        "BEV": base_stock["BEV"],
        "HEV": base_stock["Hybrid"] * hybrid_internal["HEV"],
        "PHEV": base_stock["Hybrid"] * hybrid_internal["PHEV"],
        "Diesel": base_stock["Liquids"] * liquids_internal["Diesel"],
        "Petrol": base_stock["Liquids"] * liquids_internal["Petrol"],
    }, name=f"starting_stock_{year}")
    segment_shares_year = (
        segment_shares_ext[segment_shares_ext["Year"] == year]
        .pivot(index="Drive Train", columns="Segment", values="segment_share")
        .fillna(0.0)
    )
    segment_shares_year = segment_shares_year.loc[
        [drv for drv in starting_stock.index if drv in segment_shares_year.index]
    ]
    starting_stock_segments = (
        segment_shares_year.mul(starting_stock, axis=0)
        .stack()
        .reset_index()
        .rename(columns={0: "stock", "Drive Train": "Drive Train", "Segment": "Segment"})
    )
    return starting_stock_segments


def weibull_survival_lookup(*, shape_k: float, scale_lambda: float, max_age: int) -> np.ndarray:
    ages = np.arange(max_age + 1, dtype=float)
    survival = np.exp(-((ages / scale_lambda) ** shape_k))
    return np.clip(survival, 0.0, 1.0)


def weibull_hazard_lookup(*, shape_k: float, scale_lambda: float, max_age: int) -> np.ndarray:
    ages = np.arange(max_age + 1, dtype=float)
    survival = np.exp(-((ages / scale_lambda) ** shape_k))

    hazard = np.zeros(max_age + 1, dtype=float)
    valid = survival[:-1] > 0
    hazard[:-1][valid] = 1.0 - (survival[1:][valid] / survival[:-1][valid])
    hazard[-1] = 1.0

    return np.clip(hazard, 0.0, 1.0)


def allocate_exports_bucket_rates(
    stock_available: np.ndarray,
    ages: np.ndarray,
    total_export: float,
    age_bins: list[tuple[int, int]],
    r_weights: np.ndarray,
) -> np.ndarray:
    # CONFIRMED NEVER CALLED anywhere in flowdriven_model.py itself, nor in either
    # 03_01_flowdriven.py or 03_02_adjustedflows.py. This age-bucket-weighted export
    # allocation (bucket i's export total is proportional to stock_b[i]*r_weights[i],
    # then distributed within the bucket proportional to each cohort's share of that
    # bucket's stock) is fully implemented but orphaned -- the model actually in use
    # allocates exports with a flat per-cohort proportional share instead (see
    # run_flow_driven_model_with_outflow_disaggregation's docstring, point 3).
    out = np.zeros_like(stock_available, dtype=float)

    if total_export <= 0:
        return out

    stock_available = np.asarray(stock_available, dtype=float)
    ages = np.asarray(ages, dtype=float)
    r_weights = np.asarray(r_weights, dtype=float)

    bucket_cols: list[np.ndarray] = []
    stock_b = np.zeros(len(age_bins), dtype=float)

    for i, (age_min, age_max) in enumerate(age_bins):
        cols = np.where((ages >= age_min) & (ages <= age_max) & (ages >= 0))[0]
        bucket_cols.append(cols)
        stock_b[i] = stock_available[cols].sum() if cols.size else 0.0

    denom = np.sum(stock_b * r_weights)
    if denom <= 0:
        return out

    k = total_export / denom
    e_b = k * r_weights

    for i, cols in enumerate(bucket_cols):
        if cols.size == 0 or stock_b[i] <= 0:
            continue

        export_b = stock_b[i] * e_b[i]
        avail = stock_available[cols]
        avail_sum = avail.sum()

        if avail_sum <= 0:
            continue

        out[cols] = export_b * (avail / avail_sum)

    return out


def aggregate_group_year_totals(
    frame: pd.DataFrame | None,
    *,
    group_cols: list[str],
    year_col: str,
    value_col: str,
) -> dict[tuple, float]:
    if frame is None or frame.empty:
        return {}

    required = set(group_cols + [year_col, value_col])
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns for outflow table: {sorted(missing)}")

    tmp = frame[group_cols + [year_col, value_col]].copy()
    tmp[year_col] = tmp[year_col].astype(int)
    tmp[value_col] = tmp[value_col].astype(float)

    grouped = tmp.groupby(group_cols + [year_col], as_index=False)[value_col].sum()

    key_map: dict[tuple, float] = {}
    for _, row in grouped.iterrows():
        key = tuple(row[col] for col in group_cols) + (int(row[year_col]),)
        key_map[key] = float(row[value_col])

    return key_map


def aggregate_group_totals(
    frame: pd.DataFrame | None,
    *,
    group_cols: list[str],
    value_col: str,
) -> dict[tuple, float]:
    if frame is None or frame.empty:
        return {}

    required = set(group_cols + [value_col])
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns for initial stock table: {sorted(missing)}")

    tmp = frame[group_cols + [value_col]].copy()
    tmp[value_col] = tmp[value_col].astype(float)
    grouped = tmp.groupby(group_cols, as_index=False)[value_col].sum()

    key_map: dict[tuple, float] = {}
    for _, row in grouped.iterrows():
        key = tuple(row[col] for col in group_cols)
        key_map[key] = float(row[value_col])

    return key_map


def build_synthetic_pre_baseyear_inflows(
    starting_stock_segments: pd.DataFrame,
    *,
    lifetime_by_drv: dict,
    base_year: int = 2005,
    backcast_start_year: int = 1975,
    group_cols: list[str] | None = None,
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Build synthetic historical inflows for years [backcast_start_year, base_year - 1]
    such that the surviving stock in base_year matches starting_stock_segments.

    THE MATH MODEL: constant historical inflow (same assumption as stage 02's
    `prepare_backcasting_state`, applied differently)
    ------------------------------------------------------------
    Solves for a single CONSTANT annual inflow `I` over `[backcast_start_year,
    base_year)` such that, under Weibull survival, the resulting base_year stock
    matches the target:

        I * sum_{y in pre_years} S(base_year - y) = target_stock_baseyear
        I = target_stock_baseyear / sum(survival_weights)

    This is the SAME underlying assumption as stage 02's `prepare_backcasting_state`
    (constant historical inflow, Weibull-survival-weighted) -- just solved in the
    opposite direction: `prepare_backcasting_state` takes a known total stock and
    splits it BY AGE assuming constant inflow; this function takes a known total stock
    and solves for the CONSTANT INFLOW RATE that would produce it. Mathematically
    equivalent premise, different output shape, applied at the segment level over a
    shorter window (30 years, backcast_start_year=1975) versus stage 02's aggregate
    level (50 years, back to ~1955). This refines/corrects a characterization from an
    earlier round of this review that described these as two independently-designed
    methodologies -- they share the same core assumption; only the window length and
    level of aggregation differ. The THIRD backcasting-related computation in this
    codebase, `build_stock_by_segment_at_base_year` (below), is genuinely different: it
    does NOT re-derive anything from an inflow assumption at all -- it just slices
    stage 02's already-computed cohort stock by a (cohort-age-invariant) segment share.
    """
    if group_cols is None:
        group_cols = ["Region", "Drive Train", "Segment"]

    required = set(group_cols + [value_col])
    missing = required - set(starting_stock_segments.columns)
    if missing:
        raise KeyError(f"Missing required columns in starting_stock_segments: {sorted(missing)}")

    pre_years = np.arange(backcast_start_year, base_year, dtype=int)
    rows: list[dict] = []

    for group_key, sub in starting_stock_segments.groupby(group_cols, dropna=False):
        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
        else:
            group_dict = {group_cols[0]: group_key}

        drivetrain = group_dict["Drive Train"]
        target_stock_baseyear = float(sub[value_col].sum())

        if target_stock_baseyear <= 0:
            continue

        shape_k = float(lifetime_by_drv[drivetrain]["shape_k"])
        scale_lambda = float(lifetime_by_drv[drivetrain]["scale_lambda"])

        max_age = int(base_year - backcast_start_year)
        survival = weibull_survival_lookup(
            shape_k=shape_k,
            scale_lambda=scale_lambda,
            max_age=max_age,
        )

        ages_in_baseyear = (base_year - pre_years).astype(int)
        surv_weights = survival[ages_in_baseyear]

        denom = float(surv_weights.sum())
        if denom <= 0:
            raise ValueError(
                f"Survival denominator is zero for group {group_key}. "
                f"Check lifetime parameters or backcast_start_year."
            )

        annual_inflow = target_stock_baseyear / denom

        for y in pre_years:
            rows.append(
                {
                    **group_dict,
                    "year": int(y),
                    value_col: float(annual_inflow),
                    "is_synthetic_pre_baseyear": True,
                }
            )

    return pd.DataFrame(rows)


def build_segment_distribution(df: pd.DataFrame) -> dict[str, list[str]]:
    return (
        df.dropna(subset=["Segment"])
        .groupby("Drive Train")["Segment"]
        .unique()
        .apply(list)
        .to_dict()
    )

def run_flow_driven_model_with_outflow_disaggregation(
    df: pd.DataFrame,
    years: np.ndarray,
    t_end: int,
    *,
    lifetime_by_drv: dict,
    export_r_by_drv: dict[str, np.ndarray],
    age_bins: list[tuple[int, int]],
    unknown_whereabouts_share: dict[str, float] | None = None,
    export_share_by_drivetrain: dict[str, float],
    export_rate: float | None = None,
    export_total_by_year: pd.Series | dict[int, float] | None = None,
    starting_stock_by_cohort_lookup: dict[tuple, dict[int, float]] | None = None,
    outflow_value_col: str = "value",
    year_col: str = "year",
    inflow_col: str = "value",
    group_cols: list[str] | None = None,
    outflow_timing: str = "post_inflow",
    lifetime_change_by_drv: dict[str, dict[str, float]] | None = None,
    segment_shares_by_drv: dict[str, dict[str, float]],
    allowed_export_segments: dict[str, list[str]] | None = None,
    stock_modifier_2027: float = 1.0,
    stock_modifier_start_year: int = 2027,
) -> dict:
    """
    Flow-driven model with cohort stock tracking.

    Important:
    starting_stock_by_cohort_lookup is passed explicitly.
    No hidden notebook globals.

    THE MATH MODEL (verified by reading this function in full)
    ------------------------------------------------------------
    For each (Region, Drive Train, Segment) group and each simulated year `t`:

    1. **Effective lifetime**: `lifetime_change_by_drv[drv]` overrides the base
       (`shape_k`, `scale_lambda`) once `t >= lifetime_change_by_drv[drv]["start_year"]`
       -- OPEN-ENDED (no end_year), unlike `00_parameters.py`'s
       `lifetime_override_by_drv` / `02_stockdriven.py`'s windowed
       `get_effective_lifetime_params` (which requires both start_year AND end_year).
       Two genuinely different override schemas coexist in this codebase.

    2. **`outflow_timing` controls exactly what "this year's hazard" is applied to**:
       - `"post_inflow"`: this year's new inflow is added to the newborn cohort's slot
         FIRST, then the Weibull hazard for AGE 0 (`h(0)`, small but NOT exactly zero
         for k=3) is applied to it in the SAME year -- a brand-new vehicle has a tiny but
         nonzero chance of being recorded as retiring in its own registration year.
         Existing cohorts get hazard applied using their CURRENT-year age (`t - tau`).
       - `"pre_inflow"`: existing cohorts get hazard applied using their age AS OF THE
         START of the year (`t - 1 - tau`, i.e. one less than current-year age) -- this
         matches the derivation of `h(a) = 1 - S(a+1)/S(a)` as "probability of dying
         between age a and a+1" more literally. The newborn cohort (age would be -1) is
         excluded from hazard entirely and is added to stock only AFTER hazard is
         computed for everyone else -- no same-year retirement risk for new inflow.
       `03_01_flowdriven.py` uses `"pre_inflow"`; `03_02_adjustedflows.py` uses
       `"post_inflow"` -- a genuine, unexplained methodological difference between the
       two notebooks (see the consolidated review, findings C7).

    3. **Export/unknown/collected split is FLAT and AGE-INDEPENDENT**:
           out_unknown  = out_survival * unknown_whereabouts_share[drv]
           out_export   = out_survival * export_share_by_drivetrain[drv]
           out_collected= out_survival * (1 - unknown_s - export_s)
       exactly like `disaggregation.py`'s `split_outflows_collected_unknown_export` --
       same formula, same shares, applied per-cohort here instead of to the aggregate
       stage-02 matrices. `out_total` in the returned `flows_df` is set to
       `out_surv.sum()` -- i.e. it EQUALS `out_survival`, not
       `out_survival + out_export + out_unknown` (contrast with the DOUBLE-COUNTING bug
       confirmed in `disaggregation.py`'s versions of "out_total" -- this function's
       own "out_total" is correct).

    4. **`stock_modifier_2027`**: for `t >= stock_modifier_start_year` (a real parameter
       now, default `2027` -- was a hardcoded literal until this fix; flagged in an
       earlier round of this docstring as "hardcoded year, not a parameter" but not
       actually fixed until now), `inflow_t *= stock_modifier_2027` -- a flat multiplier
       applied to EVERY drivetrain/segment's inflow uniformly from that year onward.
       Currently `1.0` (no-op) everywhere it's called.

    CONFIRMED DEAD PARAMETERS (accepted in the signature, never read in the function
    body below): `export_r_by_drv`, `age_bins`, `export_rate`, `export_total_by_year`,
    `segment_shares_by_drv`, `allowed_export_segments`. See the module docstring at the
    top of this file for the practical consequences. `export_r` IS computed from
    `export_r_by_drv` a few lines into the loop below (`export_r = np.asarray(...)`) but
    is then never referenced again -- dead local variable, confirms the parameter really
    is unused rather than used indirectly.
    """
    if group_cols is None:
        group_cols = ["Region", "Drive Train", "Segment"]

    if unknown_whereabouts_share is None:
        unknown_whereabouts_share = {}

    if starting_stock_by_cohort_lookup is None:
        starting_stock_by_cohort_lookup = {}

    flows_rows: list[dict] = []
    stock_rows: list[dict] = []
    surv_rows: list[dict] = []
    exp_rows: list[dict] = []
    unknown_rows: list[dict] = []
    outflow_long_rows: list[dict] = []

    for group_key, sub in df.groupby(group_cols, dropna=False):
        sub = sub.sort_values(year_col).reset_index(drop=True)

        if isinstance(group_key, tuple):
            group_dict = dict(zip(group_cols, group_key))
            lookup_key = group_key
        else:
            group_dict = {group_cols[0]: group_key}
            lookup_key = group_key

        drivetrain = group_dict["Drive Train"]

        shape_k = float(lifetime_by_drv[drivetrain]["shape_k"])
        scale_lambda = float(lifetime_by_drv[drivetrain]["scale_lambda"])
        export_r = np.asarray(export_r_by_drv[drivetrain], dtype=float)  # computed, then NEVER used again below -- confirms export_r_by_drv is dead (see docstring)
        inflow_map = dict(zip(sub[year_col], sub[inflow_col]))

        cohort_stock_map = starting_stock_by_cohort_lookup.get(lookup_key, {})

        # [FIXED, resolves M33] `cohort_years` now includes any vintage present in
        # `starting_stock_by_cohort_lookup` that's OLDER than `years.min()`, not just
        # `years` itself. Previously (`cohort_years = years.copy()`), any such older
        # cohort's starting stock was silently dropped -- `starting_stock_by_cohort_
        # lookup.get(...)` values for vintages outside `cohort_years` were never read at
        # all, since the loop building `stock_prev` only iterated over `cohort_years`.
        # This was flagged as numerically negligible under the lifetime parameters in
        # use at the time (Weibull survival ~0 by age 30-40), but is a genuine silent
        # data-loss bug in general -- e.g. under a longer-lifetime Monte Carlo draw
        # (`scale_lambda` can now range well above the historical point estimate via
        # `AsymmetricSpread`'s `upper`), survival at age 30-50+ is no longer negligible,
        # and stock mass from real, older cohorts would have been dropped without any
        # warning. `03_02_adjustedflows.py`'s `starting_stock_by_cohort_lookup` (from
        # stage 02's cohort matrix, vintages back to ~1955) vs. `years` (starting at
        # 1975, from `03_01_flowdriven.py`'s `BACKCAST_START_YEAR`) is exactly the case
        # this was written for.
        #
        # Extending `cohort_years` this way is a strict generalization, not a behavior
        # change, when `starting_stock_by_cohort_lookup` has no vintage older than
        # `years.min()` (the case every existing regression test covers) -- confirmed by
        # `test_flowdriven_mc_regression.py` continuing to match the pre-fix numbers
        # exactly. `max_age_group` (computed from `cohort_years.min()`, below) grows
        # automatically to cover the oldest now-included cohort's true maximum age, so
        # the Weibull hazard lookup table is correctly sized too -- no separate change
        # needed there.
        extra_vintages = sorted(
            int(tau) for tau in cohort_stock_map if int(tau) < int(years.min())
        )
        if extra_vintages:
            cohort_years = np.concatenate([np.array(extra_vintages, dtype=int), years.copy()])
        else:
            cohort_years = years.copy()
        n_cohorts = len(cohort_years)

        max_age_group = int(t_end - int(cohort_years.min()))

        stock_prev = np.zeros(n_cohorts, dtype=float)

        for j, tau in enumerate(cohort_years):
            stock_prev[j] = float(cohort_stock_map.get(int(tau), 0.0))


        for t in years:


                        # choose lifetime depending on year
            if (
                lifetime_change_by_drv
                and drivetrain in lifetime_change_by_drv
                and lifetime_change_by_drv[drivetrain] is not None
                and t >= lifetime_change_by_drv[drivetrain]["start_year"]
            ):
                k_eff = lifetime_change_by_drv[drivetrain]["shape_k"]
                lam_eff = lifetime_change_by_drv[drivetrain]["scale_lambda"]
            else:
                k_eff = shape_k
                lam_eff = scale_lambda

            # recompute hazard for this year
            h = weibull_hazard_lookup(
                shape_k=k_eff,
                scale_lambda=lam_eff,
                max_age=max_age_group,
            )


            stock_start = stock_prev.copy()
            out_surv = np.zeros(n_cohorts, dtype=float)
            out_unknown = np.zeros(n_cohorts, dtype=float)

            inflow_t = float(inflow_map.get(t, 0.0))

            if t >= stock_modifier_start_year:
                inflow_t *= stock_modifier_2027

            if outflow_timing == "post_inflow":
                stock_base = stock_start.copy()
                for j, tau in enumerate(cohort_years):
                    if tau > t:
                        continue
                    if tau == t:
                        stock_base[j] += inflow_t

                ages_base = (t - cohort_years).astype(int)
                active = cohort_years <= t
            else:
                stock_base = stock_start.copy()
                ages_base = (t - 1 - cohort_years).astype(int)
                active = (cohort_years <= (t - 1)) & (ages_base >= 0)

            for j, is_active in enumerate(active):
                if not is_active:
                    continue
                age = int(ages_base[j])
                if age < 0:
                    continue
                age = min(age, len(h) - 1)
                out_surv[j] = stock_base[j] * h[age]

            out_surv = np.minimum(out_surv, stock_base)
            stock_end = stock_base - out_surv

            group_year_key = tuple(group_dict[col] for col in group_cols) + (int(t),)


            if drivetrain not in unknown_whereabouts_share:
                raise KeyError(f"Missing unknown_whereabouts_share for {drivetrain}")

            if export_share_by_drivetrain is None or drivetrain not in export_share_by_drivetrain:
                raise KeyError(f"Missing export_share for {drivetrain}")

            unknown_s = float(np.clip(unknown_whereabouts_share[drivetrain], 0.0, 1.0))
            export_s = float(np.clip(export_share_by_drivetrain[drivetrain], 0.0, 1.0))

            if unknown_s + export_s > 1.0:
                raise ValueError(f"Shares exceed 1 for {drivetrain}")

            out_unknown = out_surv * unknown_s
            out_export = out_surv * export_s
            out_collected = out_surv * (1.0 - unknown_s - export_s)

            if outflow_timing != "post_inflow":
                for j, tau in enumerate(cohort_years):
                    if tau == t:
                        stock_end[j] += inflow_t
                        break

            flows_rows.append(
                {
                    **group_dict,
                    "year": int(t),
                    "inflow": inflow_t,
                    "out_survival": float(out_surv.sum()),
                    "out_export": float(out_export.sum()),
                    "out_unknown": float(out_unknown.sum()),
                    "out_collected": float(out_collected.sum()),
                    "out_total": float(out_surv.sum()),
                    "stock": float(stock_end.sum()),
                }
            )

            for j, tau in enumerate(cohort_years):
                if tau > t:
                    continue

                age = int(t - tau)

                stock_rows.append(
                    {
                        **group_dict,
                        "year": int(t),
                        "cohort_year": int(tau),
                        "age": age,
                        "stock": float(stock_end[j]),
                    }
                )

                for flow_type, val in [
                    ("survival", out_surv[j]),
                    ("export", out_export[j]),
                    ("unknown", out_unknown[j]),
                ]:
                    outflow_long_rows.append(
                        {
                            **group_dict,
                            "year": int(t),
                            "cohort_year": int(tau),
                            "age": age,
                            "flow_type": flow_type,
                            "value": float(val),
                        }
                    )

                surv_rows.append(
                    {
                        **group_dict,
                        "year": int(t),
                        "cohort_year": int(tau),
                        "age": age,
                        "out_survival": float(out_surv[j]),
                    }
                )
                exp_rows.append(
                    {
                        **group_dict,
                        "year": int(t),
                        "cohort_year": int(tau),
                        "age": age,
                        "out_export": float(out_export[j]),
                    }
                )
                unknown_rows.append(
                    {
                        **group_dict,
                        "year": int(t),
                        "cohort_year": int(tau),
                        "age": age,
                        "out_unknown": float(out_unknown[j]),
                    }
                )

            stock_prev = stock_end.copy()

    return {
        "flows_df": pd.DataFrame(flows_rows),
        "stock_by_cohort_df": pd.DataFrame(stock_rows),
        "outflow_surv_df": pd.DataFrame(surv_rows),
        "outflow_exp_df": pd.DataFrame(exp_rows),
        "outflow_unknown_df": pd.DataFrame(unknown_rows),
        "outflow_long_df": pd.DataFrame(outflow_long_rows),
    }


# =============================================================================
# MONTE CARLO -- thin, product-specific wrapper around the generic engine
# =============================================================================
# The actual recurrence, Weibull vectorization, draws-chunking, and
# uncertainty-sampling conventions live ONCE in `cohort_flow_mc.py` (product-
# agnostic, reusable by other pipelines/products). This function only resolves
# this project's vehicle/drivetrain-specific parameter names into the generic
# shape that engine expects, then calls it -- no separate math implementation.
#
# The scalar function above (`run_flow_driven_model_with_outflow_
# disaggregation`) is UNCHANGED and remains the ground-truth reference this
# wrapper (via the generic engine) is cross-validated against -- see
# `test_regression_generic.py`. `03_01_flowdriven.py` continues to call the
# scalar function directly; this wrapper is for Monte Carlo callers only
# (currently `03_02_adjustedflows.py`).
# =============================================================================


def run_flow_driven_model_monte_carlo(
    df: pd.DataFrame,
    years: np.ndarray,
    t_end: int,
    *,
    lifetime_by_drv: dict,
    unknown_whereabouts_share: dict[str, float],
    export_share_by_drivetrain: dict[str, float],
    starting_stock_by_cohort_lookup: dict[tuple, dict[int, float]] | None = None,
    n_draws: int = 1,
    lifetime_scale_lambda_relative_spread: dict[str, float] | float | None = None,
    unknown_whereabouts_share_std: dict[str, float] | None = None,
    export_share_std: dict[str, float] | None = None,
    year_col: str = "year",
    inflow_col: str = "value",
    group_cols: list[str] | None = None,
    outflow_timing: str = "post_inflow",
    lifetime_change_by_drv: dict[str, dict[str, float]] | None = None,
    stock_modifier_2027: float = 1.0,
    stock_modifier_start_year: int = 2027,
    output_periods: list[tuple[int, int]] | None = None,
    seed: int | np.random.SeedSequence | None = None,
    chunk_size: int = 20_000,
    collect_per_year: bool = False,
    verbose: bool = True,
    progress_label: str = "",
) -> dict:
    """
    Vehicle-pipeline-specific Monte Carlo wrapper. See `cohort_flow_mc.
    run_cohort_flow_monte_carlo` for the full model description and
    uncertainty conventions -- this function just maps this project's
    drivetrain-keyed parameter dicts onto that generic API
    (`entity_key_col="Drive Train"`, `share_a="export"`, `share_b="unknown"`)
    and translates `stock_modifier_2027` into the engine's generic
    `period_inflow_multiplier` ({year: multiplier} for every year >=
    `stock_modifier_start_year`, a real parameter, default 2027 -- not a
    hardcoded literal, matching the fix applied to the scalar function).

    `output_periods`: list of `(start_year, end_year)` inclusive ranges to
    report cumulative flows/inflow/stock for -- see `cohort_flow_mc.
    run_cohort_flow_monte_carlo`'s docstring for the full description
    (single-year query: `(2030, 2030)`; multi-year: `(2030, 2040)`). Defaults
    to the whole horizon (`years.min()` to `years.max()`), matching the
    pre-`output_periods` behavior exactly.

    With `n_draws=1` and all spread parameters None/0, reproduces the scalar
    function's `flows_df` numbers exactly (verified by regression test).

    Returns the generic engine's result dict, with vehicle-pipeline-friendly
    aliases added: "cumulative_export"/"cumulative_unknown" (aliases of the
    generic engine's "cumulative_a"/"cumulative_b") at each group level, at
    the top level (only when the whole-horizon period is among
    `output_periods`, same condition the generic engine itself uses), AND
    within EACH requested period's own sub-dict
    (`result["by_group"][key]["periods"][(start,end)]["cumulative_export"]`
    etc.). "eu_total" is an alias of "total" (top level only, for backward
    compatibility -- use `result["total"]["periods"][...]` for per-period
    EU-total access, same structure as any group's).
    """
    if group_cols is None:
        group_cols = ["Region", "Drive Train", "Segment"]

    period_inflow_multiplier = None
    if stock_modifier_2027 != 1.0:
        period_inflow_multiplier = {
            int(t): float(stock_modifier_2027) for t in years if t >= stock_modifier_start_year
        }

    result = _cohort_flow_mc.run_cohort_flow_monte_carlo(
        df=df, years=years, t_end=t_end,
        group_cols=group_cols, entity_key_col="Drive Train",
        lifetime_by_entity=lifetime_by_drv,
        share_a_by_entity=export_share_by_drivetrain,
        share_b_by_entity=unknown_whereabouts_share,
        share_a_name="export", share_b_name="unknown",
        starting_stock_by_cohort_lookup=starting_stock_by_cohort_lookup,
        n_draws=n_draws,
        lifetime_scale_lambda_relative_spread_by_entity=lifetime_scale_lambda_relative_spread,
        share_a_std_by_entity=export_share_std,
        share_b_std_by_entity=unknown_whereabouts_share_std,
        year_col=year_col, inflow_col=inflow_col,
        outflow_timing=outflow_timing,
        lifetime_change_by_entity=lifetime_change_by_drv,
        period_inflow_multiplier=period_inflow_multiplier,
        output_periods=output_periods,
        seed=seed, chunk_size=chunk_size, collect_per_year=collect_per_year,
        verbose=verbose, progress_label=progress_label,
    )

    def _add_aliases(d: dict) -> None:
        """Add cumulative_export/cumulative_unknown aliases to a dict that has
        cumulative_a/cumulative_b -- used for the top level, each group, each
        group's each period, and the total's each period."""
        if "cumulative_a" in d:
            d["cumulative_export"] = d["cumulative_a"]
        if "cumulative_b" in d:
            d["cumulative_unknown"] = d["cumulative_b"]

    for g in result["by_group"].values():
        _add_aliases(g)
        for period_result in g.get("periods", {}).values():
            _add_aliases(period_result)
        if collect_per_year:
            g["per_year_export"] = g["per_year_a"]
            g["per_year_unknown"] = g["per_year_b"]

    _add_aliases(result["total"])
    for period_result in result["total"].get("periods", {}).values():
        _add_aliases(period_result)

    # entity_draws are keyed by drivetrain already (entity_key_col="Drive Train" in
    # the call above) -- add the same export/unknown aliases for consistency with
    # every other share_a/share_b -> export/unknown alias in this wrapper.
    for drv_draws in result.get("entity_draws", {}).values():
        drv_draws["export"] = drv_draws["share_a"]
        drv_draws["unknown"] = drv_draws["share_b"]

    result["eu_total"] = dict(result["total"])

    return result
