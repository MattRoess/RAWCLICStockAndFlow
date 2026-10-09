"""
plotting.py
=============

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Visualization library for the EVmodel pipeline (`src.plotting`).

FIXES APPLIED THIS ROUND
--------------------------
- `plot_flows_split_collected_unknown_from_tracker` / `_all_trackers`: CONFIRMED BUG --
  the previous version drew every drivetrain passed in `drivetrains` (5 of them, by
  default: BEV, Diesel, Petrol, PHEV, HEV) onto the SAME axes using the SAME five
  colors (one green for "inflow", one dark red for "total outflow", three reds for the
  stacked collected/unknown/export areas) with NO per-drivetrain distinction at all.
  Concretely: for a scenario like "BAU", the single panel was showing 5 overlapping
  green inflow lines and 5 overlapping dark-red total-outflow lines on top of each
  other in identical colors -- e.g. BEV/HEV/PHEV's much smaller numbers (correctly
  ramping to 0 per their phase-out target years) tangled together visually with
  Diesel/Petrol's much larger, healthy, non-zero numbers. This made it look like
  something was badly wrong with Diesel/Petrol when in fact their underlying data was
  fine (confirmed independently via `tracker_keyed` inspection) -- the chart itself just
  couldn't show which line belonged to which drivetrain. This bug predates and is
  unrelated to any of the HEV/PHEV/BEV backcast or phase-out fixes made elsewhere in the
  pipeline; it was already this way before those fixes changed HEV/PHEV's shape.

  FIX: each drivetrain now gets its own fixed color (`DRIVETRAIN_LINE_COLORS` below).
  Inflow is drawn as a solid line, total outflow as a dashed line, both in that
  drivetrain's color. The stacked collected/unknown/export areas are DROPPED -- with 5
  drivetrains sharing one panel there is no way to stack 5 independent sets of areas
  without them overlapping just as ambiguously as the old lines did; inflow vs. total
  outflow per drivetrain is the comparison that matters here. Legend is now two-part:
  drivetrain color swatches, plus a solid/dashed key for inflow vs. total outflow.

- `plot_flows_split_collected_unknown_all_trackers` now returns `(fig, axes)` and
  accepts `show`/`save_path` parameters, instead of unconditionally calling
  `plt.show()` -- same convention as every other stage's diagnostic plots. This is
  what lets `03_02_adjustedflows.py` call it automatically as part of `main()` and save
  a real PNG, rather than it only being usable interactively.

[2026-10-09] Sixteen functions that no stage called -- the vehicle-stock and flow plots of the
original notebooks, the age, ratio and cumulative-demand plots, and the two average-age
helpers only they used -- were removed as dead code. `git log` has them. What remains is
what `03_02_adjustedflows.py` draws.
"""


from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # never opens an interactive window -- always safe to save to file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# Fixed per-drivetrain colors used by `plot_flows_split_collected_unknown_from_tracker`
# / `_all_trackers` so the same drivetrain always gets the same color across every
# scenario panel. Kept distinct from each other and from the green/dark-red convention
# used elsewhere in this file for single-drivetrain plots. Any drivetrain not in this
# map (shouldn't happen for the current 5-drivetrain pipeline, but kept as a safety net)
# falls back to matplotlib's default color cycle.
DRIVETRAIN_LINE_COLORS = {
    "BEV": "#1b9e77",
    "HEV": "#e6ab02",
    "PHEV": "#d95f02",
    "Diesel": "#7570b3",
    "Petrol": "#377eb8",
}


def plot_flows_split_collected_unknown_from_tracker(
    tracker_keyed,
    region="EUR",
    drivetrains=("BEV", "Liquids", "Hybrid"),
    year_min=2015,
    year_max=2070,
    ax=None,
    # [NEW] Optional `{drivetrain: {"years": array, "inflow_low": array,
    # "inflow_high": array, "outflow_low": array, "outflow_high": array}}`,
    # e.g. `mc_result["per_year_entity_bands"]` from `flowdriven_model.py`'s
    # `run_flow_driven_model_monte_carlo` (only populated when that call was
    # given `per_year_entity_band_pct`, e.g. `(2.5, 97.5)`). When supplied,
    # each drivetrain's inflow/outflow lines get a matching shaded band in
    # the SAME color, at reduced alpha, behind the lines (`zorder` below
    # them). `None` (default) draws exactly as before -- lines only, no
    # shading, fully backward compatible.
    entity_bands=None,
):
    """
    [FIXED] Previously drew every drivetrain in `drivetrains` onto the same `ax` using
    the SAME fixed colors for inflow/outflow/collected/unknown/export -- with no way to
    tell which line/area belonged to which drivetrain. When called with all 5
    drivetrains at once (the default in `plot_flows_split_collected_unknown_all_trackers`
    below), this produced an unreadable overlay where e.g. BEV/HEV/PHEV's much smaller,
    zero-trending numbers were tangled together with Diesel/Petrol's much larger, stable
    numbers in identical colors. See the module docstring for the full diagnosis.

    Now: each drivetrain gets its own fixed color from `DRIVETRAIN_LINE_COLORS`. Inflow
    is a solid line, total outflow (collected + unknown + export) is a dashed line, both
    in that drivetrain's color. The stacked collected/unknown/export breakdown is
    dropped -- with 5 drivetrains sharing one axes there is no non-overlapping way to
    stack 5 independent sets of areas, and inflow-vs-outflow per drivetrain is the
    comparison that actually needs to be readable here.

    [NEW] `entity_bands`, when supplied, additionally shades a 95% (or whatever
    percentiles were requested) Monte Carlo band behind each drivetrain's inflow
    and outflow lines -- see that parameter's docstring above.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

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
        out_total = out_coll + out_unk + out_exp

        x = pivot.index.to_numpy()
        color = DRIVETRAIN_LINE_COLORS.get(drv)

        # [FIXED] Shaded band behind the lines, if MC data is available for this
        # drivetrain. Interpolated onto THIS function's own year grid (`years`,
        # `year_min..year_max`) in case the band's own year range differs
        # (e.g. the band covers the full simulation horizon while this plot is
        # zoomed to a subset) -- `np.interp` requires ascending x, which the
        # band's `years` array already is (built from `np.arange` upstream).
        #
        # PREVIOUSLY the line drawn below was always this deterministic
        # (point-estimate) run's inflow/out_total, even when a band was
        # present -- meaning the line and the shaded region it sat inside came
        # from two DIFFERENT things (one fixed point-estimate run vs. an
        # entire MC ensemble), not one consistent statistic of the same
        # ensemble. A deterministic line drawn inside an MC band can legitimately
        # sit outside that band -- they aren't the same distribution. Now: when
        # a band is available, the line IS the MC median for that same
        # drivetrain/year/metric -- same draws, same reduction, guaranteed
        # consistent with the shading around it. The deterministic
        # `inflow`/`out_total` computed above is used ONLY as a fallback when
        # no MC band exists for this drivetrain (e.g. Monte Carlo was
        # disabled) -- same line as always in that case.
        band = (entity_bands or {}).get(drv)
        if band is not None:
            band_years = np.asarray(band["years"], dtype=float)
            x_f = x.astype(float)
            inflow_low = np.interp(x_f, band_years, band["inflow_low"])
            inflow_line = np.interp(x_f, band_years, band["inflow_median"])
            inflow_high = np.interp(x_f, band_years, band["inflow_high"])
            outflow_low = np.interp(x_f, band_years, band["outflow_low"])
            outflow_line = np.interp(x_f, band_years, band["outflow_median"])
            outflow_high = np.interp(x_f, band_years, band["outflow_high"])
            ax.fill_between(x, inflow_low, inflow_high, color=color, alpha=0.15, linewidth=0, zorder=1)
            ax.fill_between(x, outflow_low, outflow_high, color=color, alpha=0.15, linewidth=0, zorder=1)
        else:
            inflow_line = inflow
            outflow_line = out_total

        ax.plot(x, inflow_line, color=color, linestyle="-", linewidth=2, zorder=3)
        ax.plot(x, outflow_line, color=color, linestyle="--", linewidth=2, zorder=2)

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
    # [NEW] Optional `{scenario_name: entity_bands}`, where `entity_bands` is
    # the same shape `plot_flows_split_collected_unknown_from_tracker` accepts
    # (see that function's `entity_bands` docstring) -- e.g.
    # `{name: r["mc"]["per_year_entity_bands"] for name, r in
    # scenario_results_all.items()}`. A scenario missing from this dict, or
    # with `None`/`{}` MC results (e.g. Monte Carlo was disabled), simply gets
    # no shading for that subplot -- lines only, same as before this
    # parameter existed. `None` (default) draws every subplot unshaded.
    entity_bands_by_scenario=None,
):
    """
    [FIXED] Now returns `(fig, axes)` instead of unconditionally calling `plt.show()`.
    `show=True` (default) preserves the original interactive behavior. Pass
    `save_path=<path>` to also save a PNG -- this is what lets this function be called
    from a non-interactive pipeline script (see `03_02_adjustedflows.py`'s `main()`),
    consistent with every other stage's diagnostic-plot convention.

    [FIXED] Legend rebuilt to match the drivetrain-distinguishable rendering in
    `plot_flows_split_collected_unknown_from_tracker`: one color swatch per drivetrain,
    plus a solid/dashed key for inflow vs. total outflow. Previously this legend
    described a stacked collected/unknown/export breakdown that, with 5 drivetrains
    overlaid in identical colors, could not actually be read off the chart -- see the
    module docstring for the full diagnosis.

    [NEW] `entity_bands_by_scenario`, when supplied, shades each subplot with that
    scenario's 95% (or whatever was requested) Monte Carlo band -- see
    `plot_flows_split_collected_unknown_from_tracker`'s `entity_bands` docstring.
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
            entity_bands=(entity_bands_by_scenario or {}).get(scenario_name),
        )

        axes[i].set_title(scenario_name)
        # One panel per SCENARIO here (the sibling function panels per drivetrain);
        # both show inflow and outflow together, hence the neutral "Vehicles".
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Vehicles [million/year]")

    # hide unused axes
    for j in range(len(scenarios), len(axes)):
        axes[j].axis("off")

    # Two-part shared legend: drivetrain colors, then inflow/outflow linestyle key.
    drv_handles = [
        plt.Line2D([0], [0], color=DRIVETRAIN_LINE_COLORS.get(drv, "#333333"), lw=2)
        for drv in drivetrains
    ]
    drv_labels = list(drivetrains)

    style_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    style_labels = ["Inflow", "Total outflow"]

    handles = drv_handles + style_handles
    labels = drv_labels + style_labels

    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_flows_by_drivetrain_single_scenario(
    tracker_keyed,
    scenario_name,
    region="EUR",
    drivetrains=("BEV", "Diesel", "Petrol", "PHEV", "HEV"),
    year_min=2015,
    year_max=2070,
    entity_bands=None,
    show=False,
    save_path=None,
):
    """
    [NEW] One full-size figure for a SINGLE scenario: small multiples, one subplot
    per drivetrain, each showing that drivetrain's inflow (solid) and total outflow
    (dashed) lines plus its 95% MC band (if `entity_bands` is supplied) -- see
    `plot_flows_split_collected_unknown_from_tracker`'s `entity_bands` docstring for
    the expected shape. This is the per-scenario counterpart to
    `plot_flows_split_collected_unknown_all_trackers`'s grid (which compares ACROSS
    scenarios, one subplot per scenario); this instead goes one level of detail
    deeper INTO a single scenario, one subplot per drivetrain so each drivetrain's
    band is clearly visible on its own axes rather than overlaid with 4 others.
    """
    n_cols = min(3, len(drivetrains))
    n_rows = math.ceil(len(drivetrains) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, drv in enumerate(drivetrains):
        plot_flows_split_collected_unknown_from_tracker(
            tracker_keyed=tracker_keyed,
            region=region,
            drivetrains=(drv,),
            year_min=year_min,
            year_max=year_max,
            ax=axes[i],
            entity_bands=entity_bands,
        )
        axes[i].set_title(drv, fontsize=12)
        # "Vehicles", not "Inflow": each panel shows inflow AND outflow together.
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Vehicles [million/year]")

    for j in range(len(drivetrains), len(axes)):
        axes[j].axis("off")

    style_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    style_labels = ["Inflow", "Total outflow"]
    if entity_bands:
        style_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.3, linewidth=0))
        style_labels.append("95% MC band")

    fig.suptitle(f"{scenario_name}: inflow / outflow by drivetrain", fontsize=14, y=0.99)
    fig.legend(style_handles, style_labels, loc="upper center", ncol=len(style_handles), frameon=False, bbox_to_anchor=(0.5, 0.93))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes