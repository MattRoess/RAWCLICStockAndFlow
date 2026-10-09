"""
stock_flow.py
==============

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Two utilities: `prepare_backcasting_state` (the pre-t0 cohort-age-structure
reconstruction stage 02 depends on) and `plot_bev_stock_compare_grouped` (visual
scenario comparison). A third, `warp_bev_transition_all_segments` (accelerated a BEV
adoption curve), was wired into no stage and was removed on 2026-10-09.

FIXES APPLIED THIS ROUND
--------------------------
- **Import order** (L15-class issue): `import numpy as np` used to appear textually
  AFTER the definitions of `warp_bev_transition_all_segments` (since removed) and
  `plot_bev_stock_compare_grouped`. Not a runtime bug (Python resolves function-body names at call time, and
  the whole module finishes importing before any external caller can invoke these
  functions) -- but moved to the top with the other imports for clarity and to remove
  any doubt.
- **Added input validation** to `plot_bev_stock_compare_grouped`: previously, calling it
  with no `end_year_plotting` (default `None`) would fail deep inside with a confusing
  `TypeError` from `int(None)`. Now raises a clear `ValueError` at the top of the
  function instead.
- **`plot_bev_stock_compare_grouped` now returns `(fig, ax)`** instead of
  unconditionally calling `plt.show()` (was L17 in the review -- `plt.show()` only
  makes sense interactively). A new `show: bool = True` parameter preserves the
  original interactive behavior by default; pass `show=False` to use this from a
  non-interactive script and save the returned `fig` to a file instead.
- **`min_visible_stock` parameter added** (was a hardcoded magic number `0.2`,
  filtering out near-zero BEV stock so early years don't clutter the plot) -- still no
  cited derivation for why `0.2` specifically, but now overridable without touching
  this file.

`prepare_backcasting_state`'s ALGORITHM ITSELF IS UNCHANGED -- this is the function
`02_stockdriven.py` was already built against (see that file's `build_backcast_state`
docstring for the assumed contract, and `MATH_MODELS.md` §2.5 for confirmation this
implementation matches, now that the real file is available). **Verified end-to-end**:
re-ran the full 00→01→02 pipeline, and the same synthetic negative-inflow
demonstration used in `MATH_MODELS.md` §2.3, against this REAL implementation (not a
stand-in) -- identical results (16/61 negative years, max +20.012 surplus under
`"report_only"`, exact target match under `"clip_to_target"`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def plot_bev_stock_compare_grouped(
    scenario_map: dict[str, pd.DataFrame],
    # [REMOVED, per user request] `bev_acc: pd.Series` -- the accelerated-BEV
    # reference curve this function used to plot as a black dashed line
    # (previously sourced from `warp_bev_transition_all_segments`, since removed) -- gone,
    # along with all the drawing/legend code that rendered it. See
    # `selected_scenario` below for the replacement: this plot now marks
    # WHICH of the real scenarios is actually selected, instead of showing a
    # synthetic manual-tweak reference curve alongside them.
    selected_scenario: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    start_year_plotting: int | None = None,
    end_year_plotting: int | None = None,
    min_visible_stock: float = 0.2,
    show: bool = True,
) -> tuple[Figure, plt.Axes]:
    """
    Plot BEV stock transition speed for grouped scenarios.

    Returns `(fig, ax)`. If `show=True` (default, preserves original interactive
    behavior), also calls `plt.show()`. Pass `show=False` from a non-interactive
    script and save `fig` yourself (`fig.savefig(...)`).

    `min_visible_stock`: BEV values below this are floored to 0 before plotting, to
    keep early-year noise from cluttering the chart. Was a hardcoded `0.2` with no
    cited derivation -- now overridable, derivation still not established.

    `selected_scenario`: if given and present in `scenario_map`, that scenario's line
    is drawn thicker and on top of the others, and its legend entry is marked
    "(selected)" -- the direct visual answer to "which of these lines is the one
    actually feeding stage 02", now that there's no separate reference curve to
    compare against.
    """
    if year_min is None:
        year_min = start_year_plotting
    if year_max is None:
        year_max = end_year_plotting
    if year_min is None or year_max is None:
        raise ValueError(
            "plot_bev_stock_compare_grouped: need year_min/year_max, or "
            "start_year_plotting/end_year_plotting -- got None for at least one."
        )
    year_min, year_max = int(year_min), int(year_max)

    groups = {
        "ELV paper": ["b650", "npi25"],
        "Rawclic": ["ssp2M", "ssp2L", "ssp1"],
    }
    colors = {
        "b650": "#7b3294",
        "npi25": "#c2a5cf",
        "ssp2M": "#1f78b4",
        "ssp2L": "#6baed6",
        "ssp1": "#9ecae1",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = []
    for group_name, scen_list in groups.items():
        for scen_name in scen_list:
            if scen_name not in scenario_map:
                continue
            df = scenario_map[scen_name]
            summary = df.groupby(["year", "technology"])["value"].sum().reset_index()
            summary = summary[(summary["year"] >= year_min) & (summary["year"] <= year_max)]
            bev = (
                summary[summary["technology"] == "BEV"]
                .groupby("year")["value"]
                .sum()
                .sort_index()
            )
            bev = bev.where(bev >= min_visible_stock, 0)
            is_selected = scen_name == selected_scenario
            ax.plot(
                bev.index, bev.values,
                linewidth=3.5 if is_selected else 1.5,
                color=colors.get(scen_name, "#333333"), label=scen_name,
                zorder=10 if is_selected else 2,
            )
            plotted.append(scen_name)

    ax.set_title("BEV stock transition speed (scenario comparison)", fontsize=12)
    ax.set_ylabel("BEV stock [million]")
    ax.set_xlabel("Year")
    ax.margins(x=0, y=0)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = []
    for group_name, scen_list in groups.items():
        legend_handles.append(Line2D([], [], linestyle="none", label=f"{group_name}:"))
        for scen_name in scen_list:
            if scen_name not in plotted:
                continue
            is_selected = scen_name == selected_scenario
            label = f"  {scen_name}" + ("  (selected)" if is_selected else "")
            legend_handles.append(
                Line2D(
                    [], [], color=colors.get(scen_name, "#333333"),
                    linewidth=3.5 if is_selected else 2, label=label,
                )
            )
    ax.legend(
        handles=legend_handles, title=None, loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=False, handlelength=2.5, handletextpad=0.8,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if show:
        plt.show()
    return fig, ax


def prepare_backcasting_state(
    *, t0: int, t_end: int, stock0: float, init_max_age: int, shape_k: float, scale_lambda: float
):
    """
    Reconstruct the pre-t0 cohort age structure, assuming CONSTANT historical inflow
    (see MATH_MODELS.md §2.5): the initial stock is split across ages
    0..init_max_age proportional to the Weibull survival curve S(age) itself -- i.e.
    exactly the age distribution you'd get if the same number of vehicles had been
    registered every year for the preceding `init_max_age` years. This is a standard,
    defensible default, but a weak assumption for a genuinely young technology (e.g.
    BEV) whose real historical inflow was near zero for most of that window -- see
    MATH_MODELS.md §2.5 and the original review's M22 for the numeric magnitude.

    Returns
    -------
    tau_back : np.ndarray
        Cohort birth-year axis, from `t0 - init_max_age` through `t_end`.
    stock0_by_cohort : np.ndarray
        Initial stock (at t0) split by cohort, summing to `stock0`.
    h_out_life_age : np.ndarray
        Age-indexed hazard lookup, length `max_age + 1`.
    max_age : int
        `(t_end - t0) + init_max_age` -- the oldest age ANY tracked cohort could reach
        by the end of the simulation (a cohort already `init_max_age` old at t0, aged
        forward to t_end), not just `init_max_age` itself.
    """
    tau_back = np.arange(t0 - init_max_age, t_end + 1, dtype=int)

    ages_init = (t0 - tau_back).astype(float)
    valid = (ages_init >= 0) & (ages_init <= init_max_age)

    weights = np.zeros_like(ages_init, dtype=float)
    weights[valid] = np.exp(-((ages_init[valid] / scale_lambda) ** shape_k))

    if weights.sum() > 0:
        age_shares = weights / weights.sum()
    else:
        age_shares = valid.astype(float) / max(float(valid.sum()), 1.0)

    stock0_by_cohort = float(stock0) * age_shares

    max_age = int(t_end - tau_back[0])
    ages_all = np.arange(max_age + 1, dtype=float)
    survival = np.exp(-((ages_all / scale_lambda) ** shape_k))

    hazard = np.zeros(max_age + 1, dtype=float)
    mask = survival[:-1] > 0
    hazard[:-1][mask] = 1.0 - (survival[1:][mask] / survival[:-1][mask])
    hazard[-1] = 1.0
    hazard = np.clip(hazard, 0.0, 1.0)

    return tau_back, stock0_by_cohort, hazard, max_age