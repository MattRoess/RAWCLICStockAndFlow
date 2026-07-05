"""
plotting.py
=============

Visualization library for the EVmodel pipeline (`src.plotting`).

FIXES APPLIED THIS ROUND
--------------------------
- `plot_flows_split_collected_unknown_all_trackers` now returns `(fig, axes)` and
  accepts `show`/`save_path` parameters, instead of unconditionally calling
  `plt.show()` -- same convention as every other stage's diagnostic plots. This is
  what lets `03_02_adjustedflows.py` call it automatically as part of `main()` and save
  a real PNG, rather than it only being usable interactively.

CONFIRMED FINDING (unchanged, still relevant -- affects future stage 04-07 work):
`average_stock_age_total` (below) has no drivetrain filter and sums over EVERY key in
`matrices_by_key` unconditionally. Because `split_hybrid_*_afterwards` and
`split_liquids_*_afterwards` (in `disaggregation.py`) ADD new keys for HEV/PHEV and
Diesel/Petrol WITHOUT removing the original "Hybrid"/"Liquids" parent keys, by the time
this function would be called (post stage-03 splitting), `matrices_by_key` contains BOTH
"Hybrid" and its children "HEV"+"PHEV" as separate, coexisting entries whose stock sums
to the same total twice over (same for "Liquids"/"Diesel"/"Petrol"). Summing all keys
unconditionally therefore DOUBLE-COUNTS Hybrid/HEV/PHEV and Liquids/Diesel/Petrol stock
(BEV/Gases/FCEV are unaffected, since they're never split). `average_scrap_age_total`,
just below it, is aware of this risk (it accepts an `excluded` parameter for exactly
this purpose) but `plot_average_vehicle_age_system` -- the only caller of either
function in this file -- calls it with `excluded` left at its default (empty set), so
the protection isn't actually exercised unless a future caller passes `excluded=
{"Hybrid", "Liquids"}` (or similar) explicitly. Neither function is called from either
`03_01_flowdriven.py` or `03_02_adjustedflows.py` -- presumably intended for a
not-yet-shared stage 04/05/06 notebook, so left unfixed until that context exists.
"""


from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # never opens an interactive window -- always safe to save to file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import math


def plot_vehicle_stock_liq(df: pd.DataFrame, scenario_name: str, year_start: int, year_end: int) -> None:
    summary = df.groupby(["year", "technology"])["value"].sum().reset_index()
    summary = summary[(summary["year"] >= year_start) & (summary["year"] <= year_end)]

    pivot = summary.pivot(index="year", columns="technology", values="value").fillna(0).sort_index()
    tech_order = ["BEV", "Hybrid", "Liquids"]
    pivot = pivot.reindex(columns=[c for c in tech_order if c in pivot.columns])

    colors = {"BEV": "#1b9e77", "Hybrid": "#f2d27d", "Liquids": "#4c78a8"}
    color_list = [colors.get(c, "#333333") for c in pivot.columns]

    _, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.area(ax=ax, color=color_list, alpha=0.9, linewidth=0)

    ax.set_title(f"Vehicle stock by drivetrain ({scenario_name})", fontsize=12)
    ax.set_ylabel("Vehicles [million]")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 320)
    ax.margins(x=0, y=0)

    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(title="Drivetrain", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_vehicle_stock(df: pd.DataFrame, scenario_name: str, year_start: int, year_end: int) -> None:
    summary = df.groupby(["year", "technology"])["value"].sum().reset_index()
    summary = summary[(summary["year"] >= year_start) & (summary["year"] <= year_end)]

    pivot = summary.pivot(index="year", columns="technology", values="value").fillna(0).sort_index()
    tech_order = ["BEV", "Hybrid", "Diesel", "Liquids", "Petrol"]
    pivot = pivot.reindex(columns=[c for c in tech_order if c in pivot.columns])

    colors = {
        "BEV": "#1b9e77",
        "Hybrid": "#f2d27d",
        "Liquids": "#4c78a8",
        "Diesel": "#4c78a8",
        "Petrol": "#9ecae1",
    }
    color_list = [colors.get(c, "#333333") for c in pivot.columns]

    _, ax = plt.subplots(figsize=(12, 6))
    pivot.plot.area(ax=ax, color=color_list, alpha=0.9, linewidth=0)

    ax.set_title(f"Vehicle stock by drivetrain ({scenario_name})", fontsize=12)
    ax.set_ylabel("Vehicles [million]")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 320)
    ax.margins(x=0, y=0)

    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend(title="Drivetrain", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_flows_split_collected_unknown(
    matrices_by_key: dict,
    region: str,
    drivetrains: tuple[str, ...],
    year_min: int,
    year_max: int,
) -> None:
    def _get_series(
        flows: pd.DataFrame,
        mats: dict,
        flow_col: str,
        matrix_key: str | None = None,
    ) -> pd.Series:
        """
        Prefer precomputed columns from flows_df.
        Fall back to summing a matrix only when explicitly given.
        Always align to flows.index.
        """
        if flow_col in flows.columns:
            return flows[flow_col].astype(float).reindex(flows.index).fillna(0.0)

        if matrix_key is not None and matrix_key in mats:
            s = mats[matrix_key].sum(axis=1).astype(float)
            return s.reindex(flows.index).fillna(0.0)

        return pd.Series(0.0, index=flows.index, dtype=float)

    for drv in drivetrains:
        mats = matrices_by_key[(region, drv)]
        flows = mats["flows_df"].copy().sort_index()
        flows = flows[(flows.index >= year_min) & (flows.index <= year_max)].copy()

        x = flows.index.to_numpy()

        # Important: inflow should not fall back to summing flows_df
        inflow = _get_series(flows, mats, "inflow").to_numpy()
        out_coll = _get_series(flows, mats, "out_collected", "outflow_coll_df").to_numpy()
        out_unk = _get_series(flows, mats, "out_unknown", "outflow_unk_df").to_numpy()
        out_exp = _get_series(flows, mats, "out_export", "outflow_exp_df").to_numpy()
        out_total = _get_series(flows, mats, "out_total", "outflow_surv_df").to_numpy()

        _, ax = plt.subplots(figsize=(12, 6))

        red_collect = "#ffc9c9"
        red_unknown = "#ff8787"
        red_export = "#c92a2a"
        red_total = "#7f0000"

        coll_area = ax.fill_between(
            x, 0, out_coll,
            color=red_collect, alpha=0.8, linewidth=0,
            label="Collected ELV outflow", zorder=1
        )
        unk_area = ax.fill_between(
            x, out_coll, out_coll + out_unk,
            color=red_unknown, alpha=0.8, linewidth=0,
            label="Unknown whereabouts", zorder=1
        )
        exp_area = ax.fill_between(
            x, out_coll + out_unk, out_coll + out_unk + out_exp,
            color=red_export, alpha=0.8, linewidth=0,
            label="Export outflow", zorder=1
        )

        tot_line, = ax.plot(
            x, out_total,
            color=red_total, linewidth=2,
            label="Total outflow", zorder=2
        )
        in_line, = ax.plot(
            x, inflow,
            color="#2b8a3e", linewidth=2,
            label="Inflow", zorder=3
        )

        ax.set_title(f"Vehicle inflow and outflows ({drv})", fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Vehicles [million]")
        ax.set_ylim(top=30, bottom=0)

        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            [in_line, tot_line, exp_area, unk_area, coll_area],
            ["Inflow", "Total outflow", "Export outflow", "Unknown whereabouts", "Collected ELV outflow"],
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()


def plot_stacked_segments(
    df_long: pd.DataFrame,
    *,
    region: str,
    drivetrain: str,
    value_col: str = "value",
    year_min: int = 2015,
    year_max: int | None = None,
    title: str = "",
    ylabel: str = "Vehicles [million]",
    x_min: int | None = None,
    x_max: int | None = None,
    box_aspect: float | None = None,
):
    """
    df_long columns expected:
      year, Segment, value, Region, Drive Train
    Data should already be split to final drivetrains before plotting.
    """

    # segment groups and ordering
    seg_AF = ["A","B","C","D","E","F"]
    seg_J  = ["JA","JB","JC","JD","JE","JF"]
    col_order = seg_AF + seg_J

    # colors (exactly like your layout)
    colors_AF = ["#08306B","#08519C","#2171B5","#4292C6","#6BAED6","#9ECAE1"]
    colors_J  = ["#67000D","#A50F15","#CB181D","#EF3B2C","#FB6A4A","#FC9272"]

    color_map = dict(zip(seg_AF, colors_AF))
    color_map.update(dict(zip(seg_J, colors_J)))

    d = df_long.copy()
    d = d[(d["Region"] == region) & (d["Drive Train"] == drivetrain)]

    if len(d) == 0:
        print(f"No data for {(region, drivetrain)}")
        return

    if year_max is None:
        year_max = int(d["year"].max())

    d = d[(d["year"] >= year_min) & (d["year"] <= year_max)].copy()

    # ALWAYS aggregate to unique (year, Segment) to avoid pivot errors
    d = d.groupby(["year", "Segment"], as_index=False)[value_col].sum()

    wide = (
        d.pivot(index="year", columns="Segment", values=value_col)
        .fillna(0.0)
        .sort_index()
    )

    # enforce segment order (keep only those present)
    cols_present = [c for c in col_order if c in wide.columns]
    wide = wide.reindex(columns=cols_present)

    # colors aligned to columns
    colors = [color_map[c] for c in wide.columns]

    _, ax = plt.subplots(figsize=(12, 6))
    wide.plot.area(
        ax=ax,
        color=colors,
        alpha=0.95,
    )

    ax.set_title(title or f"{drivetrain}: segment split")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # optional axis control
    if x_min is not None or x_max is not None:
        ax.set_xlim(x_min if x_min is not None else wide.index.min(),
                    x_max if x_max is not None else wide.index.max())

    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)

    # legend with headings (SUV first, then Regular), like your example
    handles, labels = ax.get_legend_handles_labels()
    h_by_lab = dict(zip(labels, handles))
    dummy = Line2D([], [], linestyle="none")

    suv_labels = [s for s in seg_J if s in h_by_lab]
    reg_labels = [s for s in seg_AF if s in h_by_lab]

    legend_handles = []
    legend_labels = []

    if suv_labels:
        legend_handles += [dummy] + [h_by_lab[s] for s in suv_labels]
        legend_labels += ["SUV"] + suv_labels

    if reg_labels:
        legend_handles += [dummy] + [h_by_lab[s] for s in reg_labels]
        legend_labels += ["Regular"] + reg_labels

    ax.legend(
        legend_handles,
        legend_labels,
        title="Segments",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def plot_all_drivetrains_segments(
    allocated: dict,
    *,
    region: str = "EUR",
    drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol"),
    year_min: int = 2015,
    year_max: int = 2070,
):
    # 1) inflow
    for drv in drivetrains:
        plot_stacked_segments(
            allocated["inflow_segments"],
            region=region,
            drivetrain=drv,
            year_min=year_min,
            year_max=year_max,
            title=f"Inflow ({drv})",
            ylabel="Vehicles [million]",
        )

    # 2) exports
    for drv in drivetrains:
        plot_stacked_segments(
            allocated["outflow_exp_segments"],
            region=region,
            drivetrain=drv,
            year_min=year_min,
            year_max=year_max,
            title=f"Exports ({drv})",
            ylabel="Vehicles [million]",
        )

    # 3) collected
    for drv in drivetrains:
        plot_stacked_segments(
            allocated["outflow_coll_segments"],
            region=region,
            drivetrain=drv,
            year_min=year_min,
            year_max=year_max,
            title=f"Collected ELVs ({drv})",
            ylabel="Vehicles [million]",
        )

    # 4) unknown
    for drv in drivetrains:
        plot_stacked_segments(
            allocated["outflow_unk_segments"],
            region=region,
            drivetrain=drv,
            year_min=year_min,
            year_max=year_max,
            title=f"Unknown whereabouts ({drv})",
            ylabel="Vehicles [million]",
        )



def plot_segment_shares_ext(segment_shares_ext: pd.DataFrame, year_min: int, year_max: int) -> None:
    seg_AF = ["A", "B", "C", "D", "E", "F"]
    seg_J = ["JA", "JB", "JC", "JD", "JE", "JF"]
    col_order = seg_AF + seg_J

    colors_AF = ["#08306B", "#08519C", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1"]
    colors_J = ["#67000D", "#A50F15", "#CB181D", "#EF3B2C", "#FB6A4A", "#FC9272"]
    color_map = dict(zip(seg_AF + seg_J, colors_AF + colors_J))

    drivetrains = sorted(segment_shares_ext["Drive Train"].unique())
    for drv in drivetrains:
        df = segment_shares_ext[segment_shares_ext["Drive Train"] == drv].copy()
        df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)]
        if df.empty:
            continue

        piv = df.pivot(index="Year", columns="Segment", values="segment_share").fillna(0).sort_index()
        cols = [c for c in col_order if c in piv.columns]
        piv = piv[cols]
        piv = piv.div(piv.sum(axis=1), axis=0)
        colors = [color_map[c] for c in piv.columns]

        _, ax = plt.subplots(figsize=(10, 6))
        piv.plot.area(ax=ax, color=colors, alpha=0.9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Segment share")
        ax.set_ylim(0, 1)
        ax.set_title(f"{drv} segment shares")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Segment", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.show()


def plot_elements_by_flow(
    combined_dict: dict,
    flow: str,
    element_list: list[str],
    element_list_noAlCu: list[str],
    year_start: int,
    year_end: int,
    region: str = "EUR",
    ylabel: str = "Mass [t]",
) -> None:
    key = (region, flow)
    if key not in combined_dict:
        print(f"No data found for {key}")
        return

    df = combined_dict[key].copy()
    if df.empty:
        print(f"Empty dataframe for {key}")
        return

    custom_palette = ["#009ADE", "#00CD6C", "#AF58BA", "#F28522", "#FFC61E", "#FF1F5B", "#A6761D"]

    for elems, skip_first_two in ((element_list, False), (element_list_noAlCu, True)):
        sub = df[df["element"].isin(elems)]
        if sub.empty:
            continue

        piv = sub.pivot(index="scrap_year", columns="element", values="mass").fillna(0).sort_index()
        piv = piv.loc[(piv.index >= year_start) & (piv.index <= year_end)]
        if piv.empty:
            continue

        piv = piv[piv.sum().sort_values(ascending=False).index]
        palette = custom_palette[2:] if skip_first_two else custom_palette
        colors = (palette * (len(piv.columns) // len(palette) + 1))[: len(piv.columns)]

        _, ax = plt.subplots(figsize=(10, 6))
        piv.plot.area(ax=ax, alpha=0.8, color=colors)

        label = "rawclic_noAlCu" if skip_first_two else "rawclic"
        ax.set_title(f"{region} {flow} – {label}")
        ax.set_xlabel("Scrap year")
        ax.set_ylabel(ylabel)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Element", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        plt.show()


def plot_total_exports(df_exp_eu: pd.DataFrame) -> None:
    export_by_year = df_exp_eu.groupby("year")["export"].sum().sort_index()
    years = export_by_year.index
    values = export_by_year.values

    plt.figure()
    plt.fill_between(years, values, 0, color="#8e7cc3", alpha=0.6)
    plt.plot(years, values, color="#5e548e", linewidth=0.5)
    plt.xlabel("Year")
    plt.ylabel("Used vehicle exports [million]")
    plt.title("Used vehicle exports outside EU")
    plt.xticks(np.arange(int(years.min()), int(years.max()) + 1, 5))

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(years.min() - 0.3, years.max())
    ax.set_ylim(bottom=0)

    plt.grid(alpha=0.3)
    plt.show()


def plot_exports_vs_unknown_totals(matrices_by_key: dict, start_year: int) -> None:
    exports_total = None
    unknown_total = None

    for _key, mats in matrices_by_key.items():
        flows = mats["flows_df"]

        if "out_export" in flows.columns:
            s_exp = flows["out_export"].astype(float)
        elif "outflow_exp_df" in mats:
            s_exp = mats["outflow_exp_df"].sum(axis=1).astype(float)
        else:
            s_exp = pd.Series(0.0, index=flows.index)

        if "out_unknown" in flows.columns:
            s_unk = flows["out_unknown"].astype(float)
        elif "outflow_unk_df" in mats:
            s_unk = mats["outflow_unk_df"].sum(axis=1).astype(float)
        else:
            s_unk = pd.Series(0.0, index=flows.index)

        exports_total = s_exp.copy() if exports_total is None else exports_total.add(s_exp, fill_value=0.0)
        unknown_total = s_unk.copy() if unknown_total is None else unknown_total.add(s_unk, fill_value=0.0)

    exports_total = exports_total[exports_total.index >= start_year]
    unknown_total = unknown_total[unknown_total.index >= start_year]

    plt.figure(figsize=(9, 5))
    plt.plot(exports_total.index, exports_total.values, label="Exports")
    plt.plot(unknown_total.index, unknown_total.values, label="Unknown whereabouts")
    plt.xlabel("Year")
    plt.ylabel("Vehicles [mio.]")
    plt.ylim(0, 6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def average_stock_age_total(matrices_by_key: dict, region: str = "EUR") -> pd.Series:
    # CONFIRMED double-counting risk if called after stage 03's Hybrid/Liquids split --
    # no drivetrain exclusion mechanism at all here, unlike average_scrap_age_total
    # below. See module docstring.
    years_ref = None
    total_weighted_age = None
    total_stock = None

    for (reg, _drv), mats in matrices_by_key.items():
        if reg != region or "stock_t_tau_df" not in mats:
            continue

        stock_mat = mats["stock_t_tau_df"]
        tau_back = stock_mat.columns.to_numpy(dtype=int)
        years = stock_mat.index.to_numpy(dtype=int)

        if years_ref is None:
            years_ref = years
            total_weighted_age = np.zeros(len(years), dtype=float)
            total_stock = np.zeros(len(years), dtype=float)

        for i, year in enumerate(years):
            stock_vals = stock_mat.loc[year].to_numpy(dtype=float)
            ages = year - tau_back
            total_weighted_age[i] += np.sum(stock_vals * ages)
            total_stock[i] += stock_vals.sum()

    avg_age = np.divide(
        total_weighted_age,
        total_stock,
        out=np.full_like(total_weighted_age, np.nan, dtype=float),
        where=total_stock > 0,
    )
    return pd.Series(avg_age, index=years_ref, name="average_stock_age_total")


def average_scrap_age_total(matrices_by_key: dict, region: str = "EUR", excluded: set | None = None) -> pd.Series:
    excluded = excluded or set()
    years_ref = None
    total_weighted_age = None
    total_scrap = None

    for (reg, drv), mats in matrices_by_key.items():
        if reg != region or drv in excluded or "outflow_surv_df" not in mats:
            continue

        out_surv = mats["outflow_surv_df"]
        tau_back = out_surv.columns.to_numpy(dtype=int)
        years = out_surv.index.to_numpy(dtype=int)

        if years_ref is None:
            years_ref = years
            total_weighted_age = np.zeros(len(years), dtype=float)
            total_scrap = np.zeros(len(years), dtype=float)

        for i, year in enumerate(years):
            flows = out_surv.loc[year].to_numpy(dtype=float)
            ages = year - tau_back
            total_weighted_age[i] += np.sum(flows * ages)
            total_scrap[i] += flows.sum()

    avg = np.divide(
        total_weighted_age,
        total_scrap,
        out=np.full_like(total_weighted_age, np.nan, dtype=float),
        where=total_scrap > 0,
    )
    return pd.Series(avg, index=years_ref, name="average_scrap_age_total")


def plot_average_scrapping_age_by_drivetrain(
    matrices_by_key: dict,
    region: str = "EUR",
    excluded: set | None = None,
) -> None:
    excluded = excluded or set()
    drvs = []
    for (reg, drv), mats in matrices_by_key.items():
        if reg != region or drv in excluded:
            continue
        if "outflow_surv_df" in mats:
            drvs.append(drv)
    drvs = sorted(set(drvs))

    series_scrap_by_drv = {}
    for drv in drvs:
        out_surv = matrices_by_key[(region, drv)]["outflow_surv_df"]
        tau_back = out_surv.columns.to_numpy(dtype=int)
        years = out_surv.index.to_numpy(dtype=int)
        avg = np.full(len(years), np.nan, dtype=float)
        for i, year in enumerate(years):
            flows = out_surv.loc[year].to_numpy(dtype=float)
            ages = year - tau_back
            total = flows.sum()
            if total > 0:
                avg[i] = np.sum(flows * ages) / total
        series_scrap_by_drv[drv] = pd.Series(avg, index=years)

    scrap_total = average_scrap_age_total(matrices_by_key, region=region, excluded=excluded)

    _, ax = plt.subplots(figsize=(11, 7))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, drv in enumerate(drvs):
        c = color_cycle[i % len(color_cycle)]
        series_scrap_by_drv[drv].plot(ax=ax, color=c, linewidth=1.5, label=drv)

    scrap_total.plot(ax=ax, color="black", linewidth=2, label="Total")

    ax.set_title("Average scrapping age by drivetrain and total")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average scrapping age [years]")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.show()


def plot_average_vehicle_age_system(matrices_by_key: dict, region: str = "EUR") -> None:
    # NOTE: calls average_scrap_age_total with no `excluded` set (defaults to none
    # excluded) and average_stock_age_total (no exclusion mechanism at all) -- if this
    # is called on a matrices_by_key that has already been through stage 03's Hybrid/
    # Liquids splitting, both resulting averages are subject to the double-counting
    # described in the module docstring. Pass excluded={"Hybrid","Liquids"} (or
    # equivalent) once that's confirmed as the right fix.
    avg_scrap_total = average_scrap_age_total(matrices_by_key, region=region)
    avg_stock_total = average_stock_age_total(matrices_by_key, region=region)

    _, ax = plt.subplots(figsize=(9, 6))
    avg_stock_total.plot(ax=ax, linewidth=2, label="Average stock age")
    avg_scrap_total.plot(ax=ax, linewidth=2, label="Average scrapping age")

    ax.set_title("Average vehicle age in the European passenger vehicle system")
    ax.set_xlabel("Year")
    ax.set_ylabel("Vehicle age [years]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


def plot_mass_by_drv_flow_elements(
    mass_by_year_elem_dict: dict,
    element_list: list[str],
    element_list_noAlCu: list[str],
    year_min: int,
    year_max: int,
) -> None:
    custom_palette = ["#009ADE", "#00CD6C", "#AF58BA", "#F28522", "#FFC61E", "#FF1F5B", "#A6761D"]

    for (reg, drv, flow), df in mass_by_year_elem_dict.items():
        if df.empty:
            continue

        df_y = df.loc[(df["scrap_year"] >= year_min) & (df["scrap_year"] <= year_max)].copy()
        if df_y.empty:
            continue

        base = df_y[df_y["element"].isin(element_list)].copy()
        if base.empty:
            continue

        elem_order = (
            base.groupby("element")["mass"]
            .sum()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        palette_long = (custom_palette * (len(elem_order) // len(custom_palette) + 1))[: len(elem_order)]
        elem_to_color = dict(zip(elem_order, palette_long))

        for label, elems, skip_first_two in (("rawclic", element_list, False), ("rawclic_noAlCu", element_list_noAlCu, True)):
            sub = df_y[df_y["element"].isin(elems)].copy()
            if sub.empty:
                continue

            piv = sub.pivot(index="scrap_year", columns="element", values="mass").fillna(0).sort_index()
            if piv.empty:
                continue

            ordered_cols = [e for e in elem_order if e in piv.columns]
            piv = piv[ordered_cols]

            if skip_first_two:
                drop_elems = [e for e in elem_order[:2] if e in piv.columns]
                if drop_elems:
                    piv = piv.drop(columns=drop_elems)
                    ordered_cols = [e for e in ordered_cols if e not in drop_elems]

            colors = [elem_to_color[e] for e in ordered_cols]
            _, ax = plt.subplots(figsize=(10, 6))
            piv.plot.area(ax=ax, alpha=0.8, color=colors)

            ax.set_title(f"{reg} {drv} {flow} - {label}")
            ax.set_xlabel("Scrap year")
            ax.set_ylabel("Mass [kg]")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            handles, labels_leg = ax.get_legend_handles_labels()
            order_position = {name: i for i, name in enumerate(ordered_cols)}
            ranked_pairs = []
            for handle, label_name in zip(handles, labels_leg):
                rank = order_position.get(label_name, 10**9)
                ranked_pairs.append((rank, handle, label_name))
            ranked_pairs.sort(key=lambda item: item[0])
            handles = [item[1] for item in ranked_pairs]
            labels_leg = [item[2] for item in ranked_pairs]
            ax.legend(handles, labels_leg, title="Element", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))

            plt.tight_layout()
            plt.show()


def plot_ratio_focus_groups(
    ratio_df: pd.DataFrame,
    start_year: int,
    high_elements: list[str],
    low_elements: list[str],
    group_size: int = 7,
) -> None:
    ratio_cut = ratio_df.loc[
        (ratio_df["scrap_year"] >= start_year) & (ratio_df["collected_to_inflow_ratio"].notna())
    ].copy()

    for group in (high_elements, low_elements):
        _, ax = plt.subplots(figsize=(11, 6))
        focus_df = ratio_cut[ratio_cut["element"].isin(group)].copy()
        for elem in group:
            d = focus_df[focus_df["element"] == elem].sort_values("scrap_year")
            if d.empty:
                continue
            ax.plot(d["scrap_year"], d["collected_to_inflow_ratio"], linewidth=1.5, label=elem)

        ax.set_xlabel("Year")
        ax.set_ylabel("Secondary supply ratio")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False, ncol=3)
        plt.tight_layout()
        plt.show()

    others = ratio_cut[~ratio_cut["element"].isin(high_elements + low_elements)].copy()
    ranked_others = (
        others.groupby("element", as_index=False)["collected_to_inflow_ratio"]
        .max()
        .sort_values("collected_to_inflow_ratio", ascending=True)["element"]
        .tolist()
    )

    for i in range(0, len(ranked_others), group_size):
        group = ranked_others[i : i + group_size]
        _, ax = plt.subplots(figsize=(11, 6))
        for elem in group:
            d = others[others["element"] == elem].sort_values("scrap_year")
            ax.plot(d["scrap_year"], d["collected_to_inflow_ratio"], linewidth=1.5, label=elem)

        ax.set_xlabel("Year")
        ax.set_ylabel("Secondary supply ratio")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.show()


def plot_indicator_ratio(
    ratio_source: pd.DataFrame,
    label_col: str,
    title: str,
    year_start: int,
    top_n: int = 16,
    y_max: float = 1.2,
) -> None:
    required = {"scrap_year", "collected_to_inflow_ratio", label_col}
    missing = required.difference(ratio_source.columns)
    if missing:
        print(f"Skip '{title}' - missing columns: {sorted(missing)}")
        return

    df = ratio_source.copy()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col].notna() & df[label_col].ne("") & df[label_col].ne("nan")]
    df = df[df["scrap_year"] >= year_start]
    df = df[df["collected_to_inflow_ratio"].notna()]
    if df.empty:
        print(f"Skip '{title}' - no non-null values from year >= {year_start}.")
        return

    latest_year = int(df["scrap_year"].max())
    top_labels = (
        df[df["scrap_year"] == latest_year]
        .groupby(label_col, as_index=False)["collected_to_inflow_ratio"]
        .mean()
        .sort_values("collected_to_inflow_ratio", ascending=False)[label_col]
        .head(top_n)
        .tolist()
    )
    if not top_labels:
        print(f"Skip '{title}' - no labels found.")
        return

    piv = (
        df[df[label_col].isin(top_labels)]
        .pivot_table(index="scrap_year", columns=label_col, values="collected_to_inflow_ratio", aggfunc="mean")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    for label in piv.columns:
        ax.plot(piv.index, piv[label], linewidth=1.8, alpha=0.95, linestyle="-", label=str(label))

    ax.axhline(1.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7, label="_nolegend_")
    ax.set_title(title)
    ax.set_xlabel("Scrap year")
    ax.set_ylabel("Collected to inflow ratio")
    ax.set_ylim(0, y_max)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", title=label_col.title())

    fig.subplots_adjust(left=0.10, right=0.72, top=0.93, bottom=0.10)
    plt.show()


def plot_cumulative_demand_by_element(
    mass_by_year_elements_dict: dict,
    year_start: int,
    year_end: int,
    flow: str = "inflow",
) -> pd.DataFrame:
    element_rows = []
    for (region, drivetrain, flow_name), df in mass_by_year_elements_dict.items():
        if flow_name != flow:
            continue
        required = {"scrap_year", "element", "mass"}
        if not required.issubset(df.columns):
            continue
        tmp = df[["scrap_year", "element", "mass"]].copy()
        tmp["Region"] = region
        tmp["Drive Train"] = drivetrain
        element_rows.append(tmp)

    if not element_rows:
        raise ValueError("No inflow element data found in mass_by_year_elements_dict.")

    element_long = pd.concat(element_rows, ignore_index=True)
    element_long["scrap_year"] = pd.to_numeric(element_long["scrap_year"], errors="coerce")
    element_long["mass"] = pd.to_numeric(element_long["mass"], errors="coerce")
    element_long = element_long.dropna(subset=["scrap_year", "element", "mass"]).copy()
    element_long["scrap_year"] = element_long["scrap_year"].astype(int)
    element_long = element_long[(element_long["scrap_year"] >= year_start) & (element_long["scrap_year"] <= year_end)].copy()

    element_yearly = (
        element_long
        .groupby(["scrap_year", "element"], as_index=False)["mass"]
        .sum()
        .sort_values(["element", "scrap_year"])
    )
    element_yearly["cumulative_mass"] = element_yearly.groupby("element")["mass"].cumsum()

    for elem in sorted(element_yearly["element"].astype(str).unique()):
        d = element_yearly[element_yearly["element"].astype(str) == elem].copy()
        if d.empty:
            continue
        _, ax = plt.subplots(figsize=(9, 5))
        ax.plot(d["scrap_year"], d["cumulative_mass"], linewidth=2)
        ax.set_title(f"Cumulative material demand - Element: {elem}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative demand [kg]")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    return element_yearly


def plot_cumulative_demand_by_material(
    mass_by_year_materials_dict: dict,
    year_start: int,
    year_end: int,
    flow: str = "inflow",
    candidate_material_cols: list[str] | None = None,
) -> pd.DataFrame:
    candidate_material_cols = candidate_material_cols or ["materialKeyLevel2", "materialKeyLevel1", "material", "element"]
    material_rows = []

    for (region, drivetrain, flow_name), df in mass_by_year_materials_dict.items():
        if flow_name != flow:
            continue
        if "scrap_year" not in df.columns or "mass" not in df.columns:
            continue

        material_col = next((c for c in candidate_material_cols if c in df.columns), None)
        if material_col is None:
            continue

        tmp = df[["scrap_year", material_col, "mass"]].copy().rename(columns={material_col: "material_label"})
        tmp["Region"] = region
        tmp["Drive Train"] = drivetrain
        material_rows.append(tmp)

    if not material_rows:
        raise ValueError("No inflow material data found in mass_by_year_materials_dict.")

    material_long = pd.concat(material_rows, ignore_index=True)
    material_long["scrap_year"] = pd.to_numeric(material_long["scrap_year"], errors="coerce")
    material_long["mass"] = pd.to_numeric(material_long["mass"], errors="coerce")
    material_long = material_long.dropna(subset=["scrap_year", "material_label", "mass"]).copy()
    material_long["scrap_year"] = material_long["scrap_year"].astype(int)
    material_long = material_long[(material_long["scrap_year"] >= year_start) & (material_long["scrap_year"] <= year_end)].copy()

    material_yearly = (
        material_long
        .groupby(["scrap_year", "material_label"], as_index=False)["mass"]
        .sum()
        .sort_values(["material_label", "scrap_year"])
    )
    material_yearly["cumulative_mass"] = material_yearly.groupby("material_label")["mass"].cumsum()

    for mat in sorted(material_yearly["material_label"].astype(str).unique()):
        d = material_yearly[material_yearly["material_label"].astype(str) == mat].copy()
        if d.empty:
            continue
        _, ax = plt.subplots(figsize=(9, 5))
        ax.plot(d["scrap_year"], d["cumulative_mass"], linewidth=2)
        ax.set_title(f"Cumulative material demand - Material: {mat}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative demand [kg]")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    return material_yearly



def plot_flows_split_collected_unknown_from_tracker(
    tracker_keyed,
    region="EUR",
    drivetrains=("BEV", "Liquids", "Hybrid"),
    year_min=2015,
    year_max=2070,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    green = "#2b8a3e"
    red_collect = "#ffc9c9"
    red_unknown = "#ff8787"
    red_export = "#c92a2a"
    red_total = "#7f0000"

    for drv in drivetrains:
        key = (region, drv)

        if key not in tracker_keyed:
            continue

        df = tracker_keyed[key].copy()

        df["scrap_year"] = pd.to_numeric(df["scrap_year"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        df = df.dropna(subset=["scrap_year"])
        df["scrap_year"] = df["scrap_year"].astype(int)

        df = df[
            (df["scrap_year"] >= year_min) &
            (df["scrap_year"] <= year_max)
        ]

        if df.empty:
            continue

        grouped = df.groupby(["scrap_year", "flow"], as_index=False)["amount"].sum()

        pivot = grouped.pivot(index="scrap_year", columns="flow", values="amount").fillna(0.0)

        years = pd.Index(range(year_min, year_max + 1), name="scrap_year")
        pivot = pivot.reindex(years, fill_value=0.0)

        inflow = pivot.get("inflow", 0)
        out_coll = pivot.get("collected", 0)
        out_unk = pivot.get("unknown_whereabouts", 0)
        out_exp = pivot.get("export", 0)

        x = pivot.index.to_numpy()

        # STACKED AREAS (consistent from version 1)
        ax.fill_between(x, 0, out_coll, color=red_collect, alpha=0.8, linewidth=0)
        ax.fill_between(x, out_coll, out_coll + out_unk, color=red_unknown, alpha=0.8, linewidth=0)
        ax.fill_between(x, out_coll + out_unk, out_coll + out_unk + out_exp, color=red_export, alpha=0.8, linewidth=0)

        ax.plot(x, inflow, color=green, linewidth=2)
        ax.plot(x, out_coll + out_unk + out_exp, color=red_total, linewidth=2)

    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.25)


def plot_flows_split_collected_unknown_all_trackers(
    tracker_keyed_by_scenario,
    region="EUR",
    drivetrains=("BEV", "Diesel", "Petrol", "PHEV", "HEV"),
    year_min=2015,
    year_max=2070,
    scenario_order=None,
    n_cols=4,
    show=True,
    save_path=None,
):
    """
    [FIXED] Now returns `(fig, axes)` instead of unconditionally calling `plt.show()`.
    `show=True` (default) preserves the original interactive behavior. Pass
    `save_path=<path>` to also save a PNG -- this is what lets this function be called
    from a non-interactive pipeline script (see `03_02_adjustedflows.py`'s `main()`),
    consistent with every other stage's diagnostic-plot convention.
    """
    if scenario_order is None:
        scenario_order = list(tracker_keyed_by_scenario.keys())

    scenarios = [
        s for s in scenario_order
        if s in tracker_keyed_by_scenario and tracker_keyed_by_scenario[s] is not None
    ]

    n = len(scenarios)
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for i, scenario_name in enumerate(scenarios):
        plot_flows_split_collected_unknown_from_tracker(
            tracker_keyed=tracker_keyed_by_scenario[scenario_name],
            region=region,
            drivetrains=drivetrains,
            year_min=year_min,
            year_max=year_max,
            ax=axes[i],
        )

        axes[i].set_title(scenario_name)

    # hide unused axes
    for j in range(len(scenarios), len(axes)):
        axes[j].axis("off")

    # SINGLE shared legend (from version 1 style)
    handles = [
        plt.Line2D([0], [0], color="#2b8a3e", lw=2),
        plt.Line2D([0], [0], color="#7f0000", lw=2),
        plt.Rectangle((0, 0), 1, 1, color="#c92a2a", alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, color="#ff8787", alpha=0.8),
        plt.Rectangle((0, 0), 1, 1, color="#ffc9c9", alpha=0.8),
    ]

    labels = [
        "Inflow",
        "Total outflow",
        "Export outflow",
        "Unknown whereabouts",
        "Collected ELV outflow",
    ]

    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes