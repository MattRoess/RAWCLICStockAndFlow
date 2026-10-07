"""
stock_flow.py
==============

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Three utilities: `prepare_backcasting_state` (the pre-t0 cohort-age-structure
reconstruction stage 02 depends on), and two optional manual scenario-exploration
tools not currently wired into any pipeline stage: `warp_bev_transition_all_segments`
(accelerates a BEV adoption curve) and `plot_bev_stock_compare_grouped` (visual
scenario comparison).

FIXES APPLIED THIS ROUND
--------------------------
- **Import order** (L15-class issue): `import numpy as np` used to appear textually
  AFTER `warp_bev_transition_all_segments`'s and `plot_bev_stock_compare_grouped`'s own
  definitions. Not a runtime bug (Python resolves function-body names at call time, and
  the whole module finishes importing before any external caller can invoke these
  functions) -- but moved to the top with the other imports for clarity and to remove
  any doubt.
- **`warp_bev_transition_all_segments` no longer mutates its `stock_dict` argument in
  place** (was M24 in the review). It now builds and returns a NEW dict, leaving the
  caller's original untouched. This matters specifically for Monte Carlo / repeated
  scenario exploration: a sampler calling this function many times against the same
  base `stock_dict` must not have to worry about accumulated mutation from a previous
  call, or about remembering to deep-copy first.
- **Added input validation** to both `warp_bev_transition_all_segments` and
  `plot_bev_stock_compare_grouped`: previously, calling either with no `warp_end_year`/
  `end_year_plotting` (both default `None`) would fail deep inside with a confusing
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


def warp_bev_transition_all_segments(
    stock_dict: dict[tuple[str, str], pd.DataFrame],
    region: str = "EUR",
    bev_label: str = "BEV",
    accelerating_year: int = 2026,
    warp_gamma: float = 0.6,
    warp_end_year: int | None = None,
    end_year_plotting: int | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Return a NEW stock_dict where the BEV share-of-total curve for `region` is
    "warped" to front-load its rise: for years between `accelerating_year` and
    `warp_end_year`, the BEV share at year t is replaced by the share the ORIGINAL
    (unwarped) curve reaches at a later year `t_warp`, computed as:

        u = (t - accelerating_year) / (warp_end_year - accelerating_year)   in [0, 1]
        u_warp = u ** warp_gamma                                             (gamma < 1 -> front-loads)
        t_warp = accelerating_year + u_warp * (warp_end_year - accelerating_year)
        new_share(t) = original_share(t_warp)   (via linear interpolation)

    `warp_gamma < 1` pulls later values earlier (front-loads the ramp); `warp_gamma > 1`
    would back-load it. The warped share is floored at the original share
    (`np.maximum`) so warping never produces a DIP below the unwarped curve. The
    remaining (1 - BEV) share is redistributed across every other drivetrain in
    proportion to their original relative shares, then converted back to absolute
    stock using the original (unwarped) total.

    Does NOT mutate `stock_dict` -- returns a new dict. Every drivetrain not present in
    `region`, or if `bev_label` isn't one of `region`'s drivetrains, the function
    returns a dict of unchanged copies (still new objects, still safe to mutate
    independently of the input).

    NOTE: `accelerating_year`, `warp_gamma`, `warp_end_year` are NOT currently sourced
    from `params_schema.py` -- this function isn't wired into any pipeline stage yet
    (per the original review's M23 finding). If/when it is, thread these from params
    rather than relying on the defaults here, so there's one source of truth.
    """
    if warp_end_year is None:
        warp_end_year = end_year_plotting
    if warp_end_year is None:
        raise ValueError(
            "warp_bev_transition_all_segments: either warp_end_year or "
            "end_year_plotting must be provided (both are None)."
        )

    keys = [k for k in stock_dict.keys() if k[0] == region]
    if not keys:
        return {k: v.copy() for k, v in stock_dict.items()}

    drvs = sorted({k[1] for k in keys})
    if bev_label not in drvs:
        return {k: v.copy() for k, v in stock_dict.items()}

    df = pd.concat({d: stock_dict[(region, d)]["stock"] for d in drvs}, axis=1).sort_index()
    total = df.sum(axis=1)
    shares = df.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    s = shares[bev_label].clip(0.0, 1.0).copy()

    years = s.index.to_numpy(dtype=int)
    ta = int(accelerating_year)
    Te = int(min(warp_end_year, years.max()))

    s_new = s.copy()
    mask = (years >= ta) & (years <= Te)
    if np.any(mask):
        t = years[mask].astype(float)
        u = (t - ta) / max(1.0, (Te - ta))
        u_warp = np.power(u, float(warp_gamma))
        t_warp = ta + u_warp * (Te - ta)
        s_warp = np.interp(t_warp, years.astype(float), s.to_numpy())
        s_new.loc[mask] = s_warp
        s_new.loc[mask] = np.maximum(s_new.loc[mask], s.loc[mask])

    others = [d for d in drvs if d != bev_label]
    other_base = shares[others].sum(axis=1).replace(0, np.nan)
    new_shares = shares.copy()
    new_shares[bev_label] = s_new
    for d in others:
        new_shares[d] = (1.0 - s_new) * (shares[d] / other_base)
    new_shares = new_shares.fillna(0.0)
    new_df = new_shares.mul(total, axis=0)

    # [FIXED] build a NEW dict rather than mutating stock_dict's values in place.
    result: dict[tuple[str, str], pd.DataFrame] = {k: v.copy() for k, v in stock_dict.items()}
    for d in drvs:
        out = result[(region, d)].copy()
        out["stock"] = new_df[d].to_numpy()
        result[(region, d)] = out
    return result


def plot_bev_stock_compare_grouped(
    scenario_map: dict[str, pd.DataFrame],
    # [REMOVED, per user request] `bev_acc: pd.Series` -- the accelerated-BEV
    # reference curve this function used to plot as a black dashed line
    # (previously sourced from `warp_bev_transition_all_segments`) -- gone,
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