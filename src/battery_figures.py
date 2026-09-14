"""
battery_figures.py
==================

The figures for the battery material flows: what the three chemistry scenarios
assume, and what they do to the material demand.

Drawn by `code/04_04_batteries.py` at the end of its run, and redrawable on
their own with `code/04_04_figures.py` -- the stage takes minutes and reads
gigabytes of draws, a figure takes seconds and reads the result.

THE THREE SCENARIOS
--------------------
  S1  LFP volume, NMC premium, LMFP growing -- nothing new ever arrives
  S2  sodium enters the small segments, NMC shrinks to a niche
  S3  S2 plus bipolar solid-state from 2040, large segments first

WHY EVERY COMPARISON FIGURE CARRIES THE GAP
--------------------------------------------
Sodium-ion and solid-state have NO composition. Under S2 and S3 they take most
of the market, so their material demand leaves these figures -- the curves fall
because the chemistry is unknown, not because the world needs less. Read S2 and
S3 against the uncovered share drawn with them, never on their own.

Ratios (secondary supply) are formed DRAW BY DRAW from the persisted arrays:
outflow of draw i over inflow of draw i. A ratio of two percentiles is not a
percentile of the ratio, so the summary table cannot answer this question.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # never opens a window -- always safe to save to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.battery_chemistry import chemistry_share

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURE_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
DRAWS_DIR = PROJECT_ROOT / "data" / "processed" / "battery_draws"

SCENARIO_TITLES = {
    "S1": "S1  incumbents hold",
    "S2": "S2  sodium enters",
    "S3": "S3  sodium and solid-state",
}
SCENARIO_SUBTITLES = {
    "S1": "LFP volume, NMC premium, LMFP growing.\nNothing new ever arrives.",
    "S2": "Sodium-ion takes the small segments.\nNMC shrinks to a niche.",
    "S3": "S2 plus bipolar solid-state from 2040,\nlarge segments first.",
}
SCENARIO_COLORS = {"S1": "#1b9e77", "S2": "#7570b3", "S3": "#d95f02"}

CHEMISTRY_COLORS = {
    "LFP": "#2c7fb8", "LMFP": "#41b6c4", "NMC_high": "#e6550d",
    "Na_ion": "#bdbdbd", "solid_state": "#737373",
}
CHEMISTRY_LABELS = {
    "LFP": "LFP", "LMFP": "LMFP", "NMC_high": "NMC high-Ni",
    "Na_ion": "sodium-ion  (no composition)",
    "solid_state": "solid-state  (no composition)",
}
GROUP_TITLES = {"small": "small  (A, B, JA, JB)",
                "medium": "medium  (C, D, JC, JD)",
                "large": "large  (E, F, JE, JF)"}

GAP_NOTE = ("Sodium-ion and solid-state carry no composition: under S2 and S3 their "
            "material leaves the figure.\nThe dotted line, right axis, is the share of "
            "cars whose battery is not accounted for — read each curve against its own.")


def _style(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# --------------------------------------------------------------------- draws
def load_element(flow: str, scenario: str, element: str) -> tuple[np.ndarray, np.ndarray]:
    """
    (n_draws, n_years) tonnes of one element, summed over the chemistries that
    have a composition. Summed per draw, so draw i stays one world.
    """
    years = np.load(DRAWS_DIR / "years.npy")
    directory = DRAWS_DIR / flow / scenario
    total = None
    for path in sorted(directory.glob("*.npy")):
        if path.name.endswith("_elements.npy"):
            continue
        elements = list(np.load(directory / f"{path.stem}_elements.npy"))
        if element not in elements:
            continue
        column = elements.index(element)
        drawn = np.load(path, mmap_mode="r")[:, :, column]
        total = np.asarray(drawn, dtype=np.float64) if total is None else total + drawn
    if total is None:
        raise SystemExit(f"{element} appears in no chemistry of {flow}/{scenario}")
    return years, total


def band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, median, high = np.percentile(values, [2.5, 50, 97.5], axis=0)
    return low, median, high


# ------------------------------------------------------------------ figure 1
def figure_scenarios(params) -> Path:
    """What the three scenarios actually assume, group by group."""
    materials = params.materials
    scenarios = list(materials.battery_chemistry_scenarios)
    groups = ["small", "medium", "large"]
    years = np.arange(2025, 2071)
    named = materials.battery_chemistry_file_names

    fig, axes = plt.subplots(len(scenarios), len(groups), figsize=(13.5, 9.5),
                             sharex=True, sharey=True)
    for row, scenario in enumerate(scenarios):
        for column, group in enumerate(groups):
            ax = axes[row][column]
            definition = materials.battery_chemistry_scenarios[scenario][group]
            order = [c for c in ("LFP", "LMFP", "NMC_high", "Na_ion", "solid_state")
                     if c in definition]
            stack = np.array([[chemistry_share(params, scenario, group, c, y)
                               for y in years] for c in order]) * 100
            ax.stackplot(years, stack,
                         colors=[CHEMISTRY_COLORS[c] for c in order],
                         edgecolor="white", linewidth=0.4)
            # the part of the stack with no composition, marked in place
            uncovered = stack[[i for i, c in enumerate(order) if c not in named]]
            if len(uncovered):
                ax.fill_between(years, 100 - uncovered.sum(axis=0), 100,
                                facecolor="none", edgecolor="white",
                                hatch="///", linewidth=0.0, alpha=0.9)
            _style(ax)
            ax.set_ylim(0, 100)
            ax.set_xlim(years[0], years[-1])
            if row == 0:
                ax.set_title(GROUP_TITLES[group], fontsize=10.5, pad=8)
            if column == 0:
                ax.set_ylabel("share of new cars  [%]", fontsize=9.5)
                ax.text(-0.34, 0.5, SCENARIO_TITLES[scenario], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11.5,
                        fontweight="bold", color=SCENARIO_COLORS[scenario])
                ax.text(-0.22, 0.5, SCENARIO_SUBTITLES[scenario], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=8.2,
                        color="#555555")

    handles = [Patch(facecolor=CHEMISTRY_COLORS[c], label=CHEMISTRY_LABELS[c])
               for c in ("LFP", "LMFP", "NMC_high", "Na_ion", "solid_state")]
    handles.append(Patch(facecolor="white", edgecolor="#999999", hatch="///",
                         label="no composition — reported as a gap"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle("Chemistry shares assumed by each scenario, 2025–2070",
                 fontsize=14, fontweight="bold", x=0.55)
    fig.text(0.55, 0.925, "Assumption, not data: the observed record ends in 2026.",
             ha="center", fontsize=9.5, color="#555555")
    fig.tight_layout(rect=[0.10, 0.07, 1, 0.92])
    return _save(fig, "04_04_1_chemistry_scenarios.png")


# ------------------------------------------------------------------ figure 2
def figure_element_comparison(gaps: pd.DataFrame, elements=("Li", "Ni", "Co")) -> Path:
    """The comparison itself: three scenarios, key elements, inflow and outflow."""
    flows = ("inflow", "outflow")
    fig, axes = plt.subplots(len(flows), len(elements), figsize=(14, 8),
                             sharex=True)
    for row, flow in enumerate(flows):
        for column, element in enumerate(elements):
            ax = axes[row][column]
            for scenario, color in SCENARIO_COLORS.items():
                years, values = load_element(flow, scenario, element)
                low, median, high = band(values)
                ax.fill_between(years, low / 1e3, high / 1e3, color=color, alpha=0.14,
                                linewidth=0)
                ax.plot(years, median / 1e3, color=color, linewidth=2.1,
                        label=SCENARIO_TITLES[scenario])
            gap = gaps[(gaps.flow == flow) & (gaps.segment_group == "fleet")]
            twin = ax.twinx()
            for scenario, color in SCENARIO_COLORS.items():
                here = gap[gap.chemistry_scenario == scenario].sort_values("year")
                twin.plot(here["year"], here["share_without_composition"] * 100,
                          color=color, linewidth=1.0, linestyle=":", alpha=0.8)
            twin.set_ylim(0, 100)
            twin.set_yticks([0, 50, 100])
            twin.tick_params(labelsize=8, colors="#888888")
            twin.spines["top"].set_visible(False)
            if column == len(elements) - 1:
                twin.set_ylabel("cars without composition  [%]", fontsize=8.5,
                                color="#888888")
            else:
                twin.set_yticklabels([])
            _style(ax)
            if row == 0:
                ax.set_title(element, fontsize=13, fontweight="bold", pad=8)
            if column == 0:
                ax.set_ylabel(f"{flow}  [kt / year]", fontsize=10)
            if row == len(flows) - 1:
                ax.set_xlabel("year", fontsize=9.5)

    handles = [Line2D([], [], color=c, linewidth=2.1, label=SCENARIO_TITLES[s])
               for s, c in SCENARIO_COLORS.items()]
    handles += [Line2D([], [], color="#888888", linewidth=2.1, alpha=0.3,
                       label="95% band of the draws"),
                Line2D([], [], color="#888888", linewidth=1.0, linestyle=":",
                       label="share of cars without composition (right axis)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Battery material demand and return under the three chemistry scenarios",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925, "median of 200,000 draws, 95% band shaded",
             ha="center", fontsize=9.5, color="#555555")
    fig.text(0.5, 0.085, GAP_NOTE, ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.155, 1, 0.92])
    return _save(fig, "04_04_2_scenario_comparison.png")


# ------------------------------------------------------------------ figure 3
def figure_chemistry_contribution(flows_frame: pd.DataFrame, element: str = "Li") -> Path:
    """Which chemistry carries the demand, and how much of the market is unknown."""
    years = [2030, 2040, 2050, 2070]
    chemistries = ["LFP", "LMFP", "NMC_high"]
    scenarios = list(SCENARIO_COLORS)
    here = flows_frame[(flows_frame.element == element) & (flows_frame.flow == "inflow")]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(13.5, 5),
                             sharey=True)
    for ax, scenario in zip(axes, scenarios):
        rows = here[here.chemistry_scenario == scenario]
        bottom = np.zeros(len(years))
        for chemistry in chemistries:
            values = np.array([
                rows[(rows.chemistry == chemistry) & (rows.year == y)]["mean_tonnes"].sum()
                for y in years]) / 1e3
            ax.bar([str(y) for y in years], values, bottom=bottom, width=0.62,
                   color=CHEMISTRY_COLORS[chemistry], label=CHEMISTRY_LABELS[chemistry],
                   edgecolor="white", linewidth=0.8)
            bottom += values
        for x, total in enumerate(bottom):
            ax.text(x, total * 1.02, f"{total:,.0f}", ha="center", va="bottom",
                    fontsize=8.5, color="#444444")
        _style(ax)
        ax.set_title(SCENARIO_TITLES[scenario], fontsize=11.5, fontweight="bold",
                     color=SCENARIO_COLORS[scenario], pad=8)
        ax.set_xlabel("year", fontsize=9.5)
    axes[0].set_ylabel(f"{element} entering the fleet  [kt / year]", fontsize=10)
    handles = [Patch(facecolor=CHEMISTRY_COLORS[c], label=CHEMISTRY_LABELS[c])
               for c in chemistries]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)

    fig.suptitle(f"Where the {element} demand sits, and what the scenarios take out of view",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915,
             "Only the three chemistries with a composition are shown. The falling "
             "totals in S2 and S3 are cars moving to sodium-ion and solid-state, "
             "not a falling demand.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    return _save(fig, "04_04_3_chemistry_contribution.png")


# ------------------------------------------------------------------ figure 4
def figure_uncovered(gaps: pd.DataFrame) -> Path:
    """How much of each scenario this model cannot yet describe."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    groups = ["small", "medium", "large"]
    styles = {"small": ":", "medium": "--", "large": "-."}

    ax = axes[0]
    for scenario, color in SCENARIO_COLORS.items():
        here = gaps[(gaps.flow == "inflow") & (gaps.segment_group == "fleet")
                    & (gaps.chemistry_scenario == scenario)].sort_values("year")
        ax.plot(here["year"], here["share_without_composition"] * 100, color=color,
                linewidth=2.4, label=SCENARIO_TITLES[scenario])
        ax.fill_between(here["year"], 0, here["share_without_composition"] * 100,
                        color=color, alpha=0.10, linewidth=0)
    _style(ax)
    ax.set_title("whole fleet, weighted by cars sold", fontsize=11, pad=8)
    ax.set_ylabel("cars whose battery has no composition  [%]", fontsize=10)
    ax.set_xlabel("year", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    ax = axes[1]
    for scenario, color in SCENARIO_COLORS.items():
        for group in groups:
            here = gaps[(gaps.flow == "inflow") & (gaps.segment_group == group)
                        & (gaps.chemistry_scenario == scenario)].sort_values("year")
            ax.plot(here["year"], here["share_without_composition"] * 100, color=color,
                    linewidth=1.6, linestyle=styles[group], alpha=0.9)
    _style(ax)
    ax.set_title("by segment group", fontsize=11, pad=8)
    ax.set_xlabel("year", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.legend(handles=[Line2D([], [], color="#666666", linestyle=styles[g],
                              linewidth=1.6, label=GROUP_TITLES[g]) for g in groups],
              frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("The hole in the picture: share of cars with no battery composition",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.915,
             "Sodium-ion and solid-state summed. S1 stays at zero because nothing new "
             "arrives in it; S3 reaches three quarters of the large segments.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return _save(fig, "04_04_4_uncovered_share.png")


# ------------------------------------------------------------------ figure 5
def figure_secondary_supply(elements=("Li", "Ni")) -> Path:
    """
    Outflow over inflow, formed per draw. The reason the draws are kept: this
    ratio cannot be built from a mean and two percentiles.
    """
    fig, axes = plt.subplots(1, len(elements), figsize=(12.5, 5.2), sharey=True)
    for ax, element in zip(np.atleast_1d(axes), elements):
        for scenario, color in SCENARIO_COLORS.items():
            years, inflow = load_element("inflow", scenario, element)
            _, outflow = load_element("outflow", scenario, element)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(inflow > 0, outflow / inflow, np.nan) * 100
            low, median, high = np.nanpercentile(ratio, [2.5, 50, 97.5], axis=0)
            ax.fill_between(years, low, high, color=color, alpha=0.14, linewidth=0)
            ax.plot(years, median, color=color, linewidth=2.2,
                    label=SCENARIO_TITLES[scenario])
        _style(ax)
        ax.set_title(element, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("year", fontsize=9.5)
        ax.axhline(100, color="#999999", linewidth=1.0, linestyle="--")
        ax.text(years[0] + 1, 108, "returning as much as entering", fontsize=8.2,
                color="#777777", ha="left")
    np.atleast_1d(axes)[0].set_ylabel(
        "end-of-life return as a share of the same year's demand  [%]", fontsize=10)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9.5, loc="upper left")

    fig.suptitle("Secondary supply potential, formed draw by draw",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925,
             "Outflow of draw i over inflow of draw i, then the percentiles — not a "
             "ratio of percentiles, which would be a different and wrong number.",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.015,
             "Above 100% more comes back than goes in: the cars being scrapped were "
             "built when lithium chemistries still dominated, while the new ones are "
             "not.\nThe scenarios separate only because the outflow is given its "
             "BUILD year's chemistry and pack — see src/battery_vintage.py for what "
             "that reconstruction can and cannot do.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    return _save(fig, "04_04_5_secondary_supply.png")


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(PROJECT_ROOT)}")
    return path


def build_all(params, flows_frame: pd.DataFrame, gaps: pd.DataFrame) -> list[Path]:
    """Every figure, in the order they are meant to be read."""
    if not (DRAWS_DIR / "years.npy").exists():
        raise SystemExit("no battery draws -- run code/04_04_batteries.py first")
    print("figures:")
    return [
        figure_scenarios(params),
        figure_element_comparison(gaps),
        figure_chemistry_contribution(flows_frame),
        figure_uncovered(gaps),
        figure_secondary_supply(),
    ]
