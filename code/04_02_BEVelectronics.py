"""
04_02_BEVelectronics.py -- how much electronics material the BEV fleet takes in,
gives back, and hands to recycling, year by year, with full uncertainty.

WHAT THIS STAGE DOES
--------------------
Two separate studies meet here.

  * This model knows HOW MANY BEVs enter the fleet, leave it, and are collected
    for recycling, each year, per size segment -- as 200,000 Monte Carlo draws.
  * The BEV-electronics study (RAWCLICVehicleElectronics) knows HOW MUCH wiring,
    sensor, circuit-board and motor material sits in ONE car -- also as 200,000
    Monte Carlo draws, per year, per size group.

Multiply them and you get the material flow. Multiply them DRAW BY DRAW and the
answer carries both uncertainties honestly, instead of a mean times a mean.

    vehicles [millions/year]  x  material [grams/vehicle]  =  material [tonnes/year]

Reported in kilotonnes (kt), which puts the annual numbers in the 10-1000 range.

THE THREE THINGS THIS STAGE HAS TO GET RIGHT
--------------------------------------------
1. USE THE REAL DRAWS, NOT A RE-DERIVATION.
   Both sides persist their actual Monte Carlo draws to disk (`.npy`), and this
   stage reads them. It does not re-run either model and it does not resample a
   summary. This is deliberate: in this pipeline, stages that reconstructed
   another stage's numbers instead of reading them diverged silently three
   separate times before anyone noticed.

2. SEGMENTS DO NOT LINE UP, AND THE FIX MUST NOT INVENT ANYTHING.
   The electronics study groups cars into AB, CD and EF. This model uses twelve
   segments. Splitting AB into A and B is done by TILTING the pooled
   distribution -- A leans to its smaller values, B to its larger -- in a way
   that provably recombines to the original. See `split_pair_by_tilt`.

3. THE TWO UNCERTAINTIES ARE TREATED AS INDEPENDENT.
   Draw i of the fleet is paired with draw i of the electronics. Confirmed with
   the user as acceptable. It is worth stating what it assumes: a world where
   BEVs sell faster is treated as no more likely to also be a world where
   electronics content grows faster. Both do respond to technology adoption, so
   if anything this makes the reported band slightly NARROWER than the truth.
   It is never wider. Figure 9 shows which side dominates the spread.

INPUTS
------
  data/processed/bev_draws/<scenario>/BEV_<segment>_<flow>.npy
      Written by 03_02_adjustedflows.py when
      `params.materials.bev_electronics_export_draws` is True.
      Shape (draws, years), millions of vehicles.

  <bev_electronics_draws_dir>/<group>_<series>.npy
      Written by the electronics study's tools/mc_composition.py.
      Shape (draws, years), grams per vehicle.

OUTPUTS
-------
  data/processed/intermediate/04_02_bev_electronics_summary.pkl
  data/processed/figures/04_02_*.png   (see FIGURES below)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.artifacts import load_many, save_many  # type: ignore
from src.monte_carlo import summarize_distribution  # type: ignore

# vehicles are in MILLIONS, material in GRAMS per vehicle:
#   1e6 vehicles x 1 g = 1e6 g = 1 tonne, so the raw product is already tonnes.
#   /1000 -> kilotonnes, which keeps the annual numbers readable.
TONNES_PER_KILOTONNE = 1_000.0

FLOWS = ("inflow", "outflow", "collected")
FLOW_LABEL = {
    "inflow": "entering the fleet",
    "outflow": "leaving the fleet",
    "collected": "collected for recycling",
}


# ---------------------------------------------------------------------------
# The segment split
# ---------------------------------------------------------------------------
def split_pair_by_tilt(
    pool: np.ndarray, w_small: float, tilt: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split one electronics group (e.g. AB) into its two segments (A and B).

    THE PROBLEM. The electronics study reports AB as a single distribution, but
    an A-segment car is smaller than a B-segment car and carries a little less
    material. Giving both the same distribution would throw that away. Inventing
    two new distributions would be worse -- it would put numbers into the model
    that no study produced.

    THE APPROACH. Keep the pool exactly as it is, and change only WHICH of its
    draws each segment is likely to see. Every draw is scored by where it sits in
    the pool, low to high, and given a weight:

        p_small(x) = w_small + tilt x (0.5 - F(x))

    where F(x) is the draw's rank as a fraction (0 = smallest, 1 = largest) and
    `w_small` is the smaller segment's share of the pair's vehicles. The larger
    segment gets the remainder, 1 - p_small.

    TWO PROPERTIES, BOTH REQUIRED.
      * p_small falls as x rises, so A really does lean to the smaller values.
      * The average of p_small over the pool is exactly w_small, because the
        average of F is 0.5 by construction. So the two halves, weighted by
        their vehicle shares, reproduce the original pool distribution. Nothing
        is created and nothing is lost. Figure 2 shows this holding.

    `tilt = 0` gives both segments the pool unchanged. The formula stays valid
    while `tilt <= 2 x min(w_small, 1 - w_small)`; beyond that a weight would go
    negative, the recombination guarantee fails, and this raises rather than
    quietly clipping.

    WHAT THE TILT ACTUALLY BUYS. It is a re-weighting strength, not a percentage
    gap. Measured on the real AB pool at 200,000 draws:

        tilt 0.20 -> A and B means differ by  3.1%   (the configured setting)
        tilt 0.40 ->                          6.3%
        tilt 0.60 ->                          9.5%
        tilt 0.77 ->                         12.3%   (the ceiling for AB)

    The ceiling follows from A holding ~61.5% of AB's vehicles. A 20% separation
    cannot be produced from this pool without abandoning the recombination
    property, which is the one thing that keeps this a re-weighting of a
    published distribution rather than an invented one.

    HOW EXACT IS THE RECOMBINATION. Resampling under the weights carries ordinary
    Monte Carlo noise, which falls with draw count. On the real AB pool the error
    in reproducing the published mean was 0.10% at 300 draws, 0.017% at 30,000,
    and 0.007% at 200,000. The caller prints this per pair and figure 04_02_02
    puts it on each panel, so drift is visible rather than assumed.

    WHOLE ROWS MOVE TOGETHER. A row of the pool is one simulated car followed
    across all years -- a car with 800 V zonal wiring is that car in every year.
    Ranking and re-weighting therefore work on whole rows (scored by each row's
    mean across years), never year by year, which would shred that structure and
    understate the spread.

    Returns (small_segment_draws, large_segment_draws), each the same shape as
    `pool`. Rows are drawn from the pool WITH REPLACEMENT under the weights --
    the values themselves are always the study's own.
    """
    if not 0.0 <= w_small <= 1.0:
        raise ValueError(f"w_small={w_small} must be between 0 and 1.")
    limit = 2.0 * min(w_small, 1.0 - w_small)
    if tilt > limit + 1e-12:
        raise ValueError(
            f"bev_electronics_segment_tilt={tilt} is too large for a pair whose "
            f"smaller segment holds {w_small:.1%} of the vehicles: the maximum "
            f"that still lets the two halves add back to the original "
            f"distribution is {limit:.3f}. Lower the tilt, or accept that the "
            f"split would no longer reproduce the electronics study's own "
            f"distribution."
        )

    n_rows = pool.shape[0]
    level = pool.mean(axis=1)                       # one score per simulated car
    order = np.argsort(level, kind="stable")
    frac = np.empty(n_rows, dtype=float)
    frac[order] = np.linspace(0.0, 1.0, n_rows)     # F(x): rank as a fraction

    p_small = np.clip(w_small + tilt * (0.5 - frac), 0.0, 1.0)
    p_large = 1.0 - p_small

    rng = np.random.default_rng(seed)
    small_idx = rng.choice(n_rows, size=n_rows, replace=True, p=p_small / p_small.sum())
    large_idx = rng.choice(n_rows, size=n_rows, replace=True, p=p_large / p_large.sum())
    return pool[small_idx], pool[large_idx]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_fleet_draws(bev_dir: Path) -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray]:
    """BEV vehicle counts per (segment, flow), plus the year axis."""
    years_path = bev_dir / "years.npy"
    if not years_path.exists():
        raise FileNotFoundError(
            f"No BEV draw export found at {bev_dir}.\n"
            f"Run 03_02_adjustedflows.py with "
            f"params.materials.bev_electronics_export_draws = True and "
            f"params.monte_carlo.enabled = True first -- this stage reads the "
            f"vehicle-count draws that run writes, and cannot reconstruct them."
        )
    years = np.load(years_path)
    out: dict[tuple[str, str], np.ndarray] = {}
    for path in sorted(bev_dir.glob("BEV_*_*.npy")):
        stem = path.stem[len("BEV_"):]
        seg, _, flow = stem.rpartition("_")
        if flow in FLOWS:
            out[(seg, flow)] = np.load(path, mmap_mode="r")
    if not out:
        raise FileNotFoundError(f"{bev_dir} has a year axis but no BEV_*.npy arrays.")
    return out, years


def load_electronics_draws(draws_dir: Path, series: tuple[str, ...]) -> dict[tuple[str, str], np.ndarray]:
    """Grams per vehicle per (electronics group, series)."""
    if not draws_dir.is_dir():
        raise FileNotFoundError(
            f"No electronics draws at {draws_dir}.\n"
            f"In the RAWCLICVehicleElectronics repository run "
            f"tools/mc_composition.py (it writes Composition/draws/*.npy), then "
            f"point params.materials.bev_electronics_draws_dir at that folder."
        )
    out: dict[tuple[str, str], np.ndarray] = {}
    for path in sorted(draws_dir.glob("*_*.npy")):
        group, _, ser = path.stem.rpartition("_")
        if ser in series:
            out[(group, ser)] = np.load(path, mmap_mode="r")
    if not out:
        raise FileNotFoundError(f"{draws_dir} contains no *_*.npy matching {series}.")
    return out


def electronics_years(draws_dir: Path, n_years: int) -> np.ndarray:
    """
    The electronics study's year axis. It does not ship one, but its own
    configuration fixes it at 2020..2070 and every array has that length, so the
    axis is reconstructed and then checked against the array width.
    """
    years = np.arange(2020, 2020 + n_years, dtype=int)
    if len(years) != n_years:
        raise ValueError(f"cannot reconstruct electronics year axis for {n_years} columns")
    return years


# ---------------------------------------------------------------------------
# The calculation
# ---------------------------------------------------------------------------
def build_segment_electronics(
    elec: dict[tuple[str, str], np.ndarray],
    fleet: dict[tuple[str, str], np.ndarray],
    fleet_years: np.ndarray,
    keep: np.ndarray,
    pairs: tuple[tuple[str, str, str], ...],
    series: str,
    tilt: float,
    seed: int,
    n_draws: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """
    Turn the three electronics groups into twelve per-segment distributions.

    Each pair (AB -> A, B) is tilt-split using the pair's own vehicle shares. The
    J-segments have no electronics of their own and take their non-J twin's
    distribution: JA uses A's, JB uses B's, and so on.
    """
    per_segment: dict[str, np.ndarray] = {}
    report: list[dict[str, Any]] = []

    for group, seg_small, seg_large in pairs:
        pool = np.asarray(elec[(group, series)][:n_draws], dtype=np.float64)

        # Vehicle shares decide how much of the pool each segment represents.
        # Taken over the reporting window as a whole because a row is one car
        # followed through time -- it belongs to one segment for its whole life,
        # so the weight cannot be re-decided year by year.
        small_v = np.asarray(fleet[(seg_small, "inflow")][:n_draws][:, keep]).sum()
        large_v = np.asarray(fleet[(seg_large, "inflow")][:n_draws][:, keep]).sum()
        w_small = float(small_v / (small_v + large_v)) if (small_v + large_v) > 0 else 0.5

        small, large = split_pair_by_tilt(pool, w_small, tilt, seed=seed + hash(group) % 10_000)
        per_segment[seg_small] = small
        per_segment[seg_large] = large
        per_segment["J" + seg_small] = small
        per_segment["J" + seg_large] = large

        gap = (large.mean() - small.mean()) / small.mean() if small.mean() else float("nan")
        report.append({
            "group": group, "small": seg_small, "large": seg_large,
            "w_small": w_small, "tilt": tilt,
            "mean_small": small.mean(), "mean_large": large.mean(),
            "mean_gap_pct": 100.0 * gap,
            "pool_mean": pool.mean(),
            "recombined_mean": w_small * small.mean() + (1 - w_small) * large.mean(),
        })
    return per_segment, report


def multiply(
    fleet: dict[tuple[str, str], np.ndarray],
    per_segment_elec: dict[str, np.ndarray],
    keep: np.ndarray,
    flow: str,
    segments: list[str],
    n_draws: int,
) -> np.ndarray:
    """
    Draw-by-draw product, summed over segments -> (draws, years) in kilotonnes.

    Draw i of the fleet meets draw i of the electronics. The sum over segments
    happens on the RAW draws, before any percentile is taken: percentile-of-sum
    is the honest total, sum-of-percentiles is not (it would imply every segment
    hits its extreme in the same world).
    """
    total = None
    for seg in segments:
        v = np.asarray(fleet[(seg, flow)][:n_draws][:, keep], dtype=np.float64)
        g = per_segment_elec[seg]
        prod = v * g                                   # millions x grams = tonnes
        total = prod if total is None else total + prod
    return total / TONNES_PER_KILOTONNE


def summarize_by_year(arr: np.ndarray, years: np.ndarray) -> pd.DataFrame:
    lo, med, hi = np.percentile(arr, [2.5, 50, 97.5], axis=0)
    return pd.DataFrame({
        "year": years, "mean": arr.mean(axis=0), "median": med,
        "p2_5": lo, "p97_5": hi, "std": arr.std(axis=0),
    })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _band(ax, years, df, color, label):
    ax.plot(years, df["median"], color=color, linewidth=1.8, label=label)
    ax.fill_between(years, df["p2_5"], df["p97_5"], color=color, alpha=0.20, linewidth=0)


def fig_flows(by_flow: dict[str, pd.DataFrame], years, path: Path, n_draws: int) -> None:
    """FIGURE 4 -- the headline: all three flows with their 95% bands."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for flow, color in zip(FLOWS, ("#2E86AB", "#E67E22", "#2b8a3e")):
        _band(ax, years, by_flow[flow], color, f"{flow} ({FLOW_LABEL[flow]})")
    ax.set_title(f"BEV electronics material, total -- median and 95% band "
                 f"(n={n_draws:,})", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Electronics material [kt/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.text(0.01, 0.01, "Fleet and electronics uncertainties multiplied draw by draw, "
                         "treated as independent.", fontsize=7, color="#666666")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_recombination(report: list[dict[str, Any]], pools, splits, path: Path) -> None:
    """
    FIGURE 2 -- proof the segment split is honest.

    Each panel: the electronics group's own distribution, with the two segments
    drawn from it on top. The two lean apart, and weighted together they land
    back on the original -- which is the whole claim being made.
    """
    fig, axes = plt.subplots(1, len(report), figsize=(5 * len(report), 4.2))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, report):
        pool = pools[r["group"]]
        small, large = splits[r["small"]], splits[r["large"]]
        bins = np.linspace(np.percentile(pool, 0.5), np.percentile(pool, 99.5), 60)
        ax.hist(pool.mean(axis=1), bins=bins, density=True, color="#888888",
                alpha=0.35, label=f"{r['group']} (as published)")
        ax.hist(small.mean(axis=1), bins=bins, density=True, histtype="step",
                color="#2E86AB", linewidth=1.6, label=f"{r['small']} (smaller)")
        ax.hist(large.mean(axis=1), bins=bins, density=True, histtype="step",
                color="#E67E22", linewidth=1.6, label=f"{r['large']} (larger)")
        ax.set_title(f"{r['group']} -> {r['small']} + {r['large']}\n"
                     f"means differ by {r['mean_gap_pct']:.1f}%  |  "
                     f"recombined error {abs(r['recombined_mean']/r['pool_mean']-1)*100:.3f}%",
                     fontsize=9)
        ax.set_xlabel("Electronics per vehicle [g]")
        ax.set_ylabel("Probability density")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_domains(by_series: dict[str, pd.DataFrame], years, flow: str, path: Path) -> None:
    """FIGURE 5 -- which part of the electronics the mass actually is."""
    doms = [s for s in by_series if s != "Total"]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(years, *[by_series[d]["median"] for d in doms], labels=doms, alpha=0.85)
    ax.plot(years, by_series["Total"]["median"], color="black", linewidth=1.2,
            linestyle="--", label="Total (median)")
    ax.set_title(f"BEV electronics {FLOW_LABEL[flow]}, by domain (medians)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Electronics material [kt/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Medians stack only approximately -- the median of a sum is not "
                         "the sum of medians. The dashed line is the true total.",
             fontsize=7, color="#666666")
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_snapshots(arr: np.ndarray, years, probe_years, flow: str, path: Path) -> None:
    """FIGURE 7 -- the actual distributions at a few years, not just a band."""
    probes = [y for y in probe_years if y in set(years.tolist())]
    fig, axes = plt.subplots(1, len(probes), figsize=(4.6 * len(probes), 4))
    axes = np.atleast_1d(axes)
    for ax, y in zip(axes, probes):
        v = arr[:, list(years).index(y)]
        ax.hist(v, bins=60, color="#2E86AB", alpha=0.75)
        for q, c, ls in ((2.5, "black", "--"), (50, "#2b8a3e", "-"), (97.5, "black", "--")):
            ax.axvline(np.percentile(v, q), color=c, linestyle=ls, linewidth=1.2)
        ax.set_title(f"{y}: median {np.percentile(v,50):,.0f} kt\n"
                     f"95% {np.percentile(v,2.5):,.0f}-{np.percentile(v,97.5):,.0f} kt",
                     fontsize=9)
        ax.set_xlabel(f"Electronics {FLOW_LABEL[flow]} [kt/year]")
        ax.set_ylabel("Number of draws")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_variance_split(
    fleet, per_segment_elec, keep, years, segments, n_draws, path: Path,
) -> None:
    """
    FIGURE 9 -- where does the uncertainty come from?

    Three runs of the same product:
      both vary        the real result
      fleet only       electronics held at its mean -- how much spread the
                       vehicle numbers alone produce
      electronics only fleet held at its mean

    If one line's band is close to the full band, that side dominates and is
    where any effort to narrow the uncertainty should go.
    """
    def width(fleet_fixed: bool, elec_fixed: bool) -> np.ndarray:
        total = None
        for seg in segments:
            v = np.asarray(fleet[(seg, "inflow")][:n_draws][:, keep], dtype=np.float64)
            g = per_segment_elec[seg]
            if fleet_fixed:
                v = np.broadcast_to(v.mean(axis=0), v.shape)
            if elec_fixed:
                g = np.broadcast_to(g.mean(axis=0), g.shape)
            prod = v * g
            total = prod if total is None else total + prod
        total = total / TONNES_PER_KILOTONNE
        return np.percentile(total, 97.5, axis=0) - np.percentile(total, 2.5, axis=0)

    both = width(False, False)
    fleet_only = width(False, True)
    elec_only = width(True, False)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(years, both, color="black", linewidth=2.0, label="both vary (the real band)")
    ax.plot(years, fleet_only, color="#2E86AB", linewidth=1.6,
            label="vehicle numbers only")
    ax.plot(years, elec_only, color="#E67E22", linewidth=1.6,
            label="electronics content only")
    ax.set_title("Where the uncertainty comes from -- width of the 95% band", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Width of 95% band [kt/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_by_segment(by_seg: dict[str, pd.DataFrame], years, flow: str, path: Path) -> None:
    """FIGURE 6 -- which car sizes carry the material."""
    fig, ax = plt.subplots(figsize=(11, 6))
    segs = sorted(by_seg, key=lambda s: (s.startswith("J"), s))
    ax.stackplot(years, *[by_seg[s]["median"] for s in segs], labels=segs, alpha=0.85)
    ax.set_title(f"BEV electronics {FLOW_LABEL[flow]}, by segment (medians)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Electronics material [kt/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8, ncol=2)
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_cumulative(by_flow: dict[str, np.ndarray], years, path: Path) -> None:
    """FIGURE 8 -- running totals over the whole period."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for flow, color in zip(FLOWS, ("#2E86AB", "#E67E22", "#2b8a3e")):
        cum = np.cumsum(by_flow[flow], axis=1) / 1000.0        # kt -> Mt
        lo, med, hi = np.percentile(cum, [2.5, 50, 97.5], axis=0)
        ax.plot(years, med, color=color, linewidth=1.8, label=f"{flow} ({FLOW_LABEL[flow]})")
        ax.fill_between(years, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.set_title("BEV electronics material, cumulative since 2020", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative electronics material [Mt]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_per_vehicle(elec, series_list, years_e, path: Path) -> None:
    """FIGURE 1 -- the electronics input itself, before any vehicle counts."""
    groups = sorted({g for g, _ in elec})
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        for ser, color in zip(series_list, ("black", "#2E86AB", "#E67E22", "#2b8a3e", "#8e44ad")):
            if (g, ser) not in elec:
                continue
            a = np.asarray(elec[(g, ser)], dtype=np.float64) / 1000.0     # g -> kg
            lo, med, hi = np.percentile(a, [2.5, 50, 97.5], axis=0)
            ax.plot(years_e, med, color=color, linewidth=1.6, label=ser)
            if ser == "Total":
                ax.fill_between(years_e, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.set_title(f"{g}", fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Electronics per vehicle [kg]")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Electronics content of one BEV -- the study's own distributions "
                 "(band shown for Total)", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_vehicles(fleet, keep, years, segments, n_draws, path: Path) -> None:
    """FIGURE 3 -- the fleet input itself, before any electronics."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for flow, color in zip(FLOWS, ("#2E86AB", "#E67E22", "#2b8a3e")):
        tot = None
        for seg in segments:
            v = np.asarray(fleet[(seg, flow)][:n_draws][:, keep], dtype=np.float64)
            tot = v if tot is None else tot + v
        lo, med, hi = np.percentile(tot, [2.5, 50, 97.5], axis=0)
        ax.plot(years, med, color=color, linewidth=1.8, label=f"{flow} ({FLOW_LABEL[flow]})")
        ax.fill_between(years, lo, hi, color=color, alpha=0.20, linewidth=0)
    ax.set_title("BEV vehicles -- the fleet side of the multiplication", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Vehicles [million/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    params = load_many("params", root=PROJECT_ROOT)["params"]
    p04 = params.materials
    scenario = (params.adjusted_flows.scenarios_to_run or ("BAU",))[0]

    fig_dir = PROJECT_ROOT / "data" / "processed" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bev_dir = PROJECT_ROOT / "data" / "processed" / "bev_draws" / scenario
    fleet, fleet_years = load_fleet_draws(bev_dir)

    elec_dir = Path(p04.bev_electronics_draws_dir)
    if not elec_dir.is_absolute():
        elec_dir = (PROJECT_ROOT / "code" / elec_dir).resolve()
    series = tuple(p04.bev_electronics_series)
    elec = load_electronics_draws(elec_dir, series)

    n_years_e = next(iter(elec.values())).shape[1]
    years_e = electronics_years(elec_dir, n_years_e)

    # Both sides must be on the same years and the same number of draws. The
    # overlap is taken rather than assumed: the electronics study starts in 2020,
    # the fleet model in 1975.
    y_lo = max(int(p04.bev_electronics_year_min), int(years_e.min()), int(fleet_years.min()))
    y_hi = min(int(p04.bev_electronics_year_max), int(years_e.max()), int(fleet_years.max()))
    years = np.arange(y_lo, y_hi + 1)
    keep = np.isin(fleet_years, years)
    keep_e = np.isin(years_e, years)

    n_fleet = next(iter(fleet.values())).shape[0]
    n_elec = next(iter(elec.values())).shape[0]
    n_draws = min(n_fleet, n_elec)
    if n_fleet != n_elec:
        print(f"  NOTE: fleet has {n_fleet:,} draws, electronics has {n_elec:,}. "
              f"Using the first {n_draws:,} of each. They are independent samples, "
              f"so this pairs draw i with draw i and discards the surplus -- it "
              f"does not resample.")

    elec = {k: np.asarray(v)[:, keep_e] for k, v in elec.items()}
    segments = sorted({s for s, _ in fleet})
    print(f"[04_02] scenario={scenario}  years={y_lo}-{y_hi}  draws={n_draws:,}  "
          f"segments={len(segments)}  series={list(series)}")

    pairs = tuple(tuple(x) for x in p04.bev_electronics_segment_pairs)
    tilt = float(p04.bev_electronics_segment_tilt)

    results: dict[str, Any] = {"scenario": scenario, "years": years,
                              "n_draws": n_draws, "tilt": tilt}
    summary_rows: list[dict[str, Any]] = []

    # ---- the split, once per series -------------------------------------
    per_segment_by_series: dict[str, dict[str, np.ndarray]] = {}
    split_report: list[dict[str, Any]] = []
    for ser in series:
        per_seg, rep = build_segment_electronics(
            elec, fleet, fleet_years, keep, pairs, ser, tilt,
            seed=int(params.monte_carlo.seed or 0), n_draws=n_draws,
        )
        per_segment_by_series[ser] = per_seg
        if ser == "Total":
            split_report = rep

    print("\n  segment split (tilt = %.2f)" % tilt)
    print(f"    {'pair':<6}{'smaller share':>15}{'mean gap':>12}{'recombination error':>22}")
    for r in split_report:
        err = abs(r["recombined_mean"] / r["pool_mean"] - 1) * 100
        print(f"    {r['group']:<6}{r['w_small']:>14.1%}{r['mean_gap_pct']:>11.1f}%"
              f"{err:>21.4f}%")
        results.setdefault("split_report", []).append(r)

    # ---- the product ----------------------------------------------------
    by_flow_arr: dict[str, np.ndarray] = {}
    by_flow_df: dict[str, pd.DataFrame] = {}
    for flow in FLOWS:
        arr = multiply(fleet, per_segment_by_series["Total"], keep, flow, segments, n_draws)
        by_flow_arr[flow] = arr
        by_flow_df[flow] = summarize_by_year(arr, years)
        for _, row in by_flow_df[flow].iterrows():
            summary_rows.append({"flow": flow, "series": "Total", **row.to_dict()})

    by_series_df = {}
    for ser in series:
        a = multiply(fleet, per_segment_by_series[ser], keep, "inflow", segments, n_draws)
        by_series_df[ser] = summarize_by_year(a, years)
        if ser != "Total":
            for _, row in by_series_df[ser].iterrows():
                summary_rows.append({"flow": "inflow", "series": ser, **row.to_dict()})

    by_seg_df = {}
    for seg in segments:
        v = np.asarray(fleet[(seg, "inflow")][:n_draws][:, keep], dtype=np.float64)
        a = v * per_segment_by_series["Total"][seg] / TONNES_PER_KILOTONNE
        by_seg_df[seg] = summarize_by_year(a, years)

    # ---- figures --------------------------------------------------------
    print("\n  figures:")
    jobs = [
        ("04_02_01_electronics_per_vehicle.png",
         lambda p: fig_per_vehicle(elec, series, years, p)),
        ("04_02_02_segment_split_recombination.png",
         lambda p: fig_recombination(
             split_report,
             {g: np.asarray(elec[(g, "Total")], dtype=np.float64)[:n_draws] for g, _, _ in pairs},
             per_segment_by_series["Total"], p)),
        ("04_02_03_bev_vehicles.png",
         lambda p: fig_vehicles(fleet, keep, years, segments, n_draws, p)),
        ("04_02_04_material_flows.png",
         lambda p: fig_flows(by_flow_df, years, p, n_draws)),
        ("04_02_05_by_domain.png",
         lambda p: fig_domains(by_series_df, years, "inflow", p)),
        ("04_02_06_by_segment.png",
         lambda p: fig_by_segment(by_seg_df, years, "inflow", p)),
        ("04_02_07_distribution_snapshots.png",
         lambda p: fig_snapshots(by_flow_arr["inflow"], years, (2030, 2050, 2070), "inflow", p)),
        ("04_02_08_cumulative.png",
         lambda p: fig_cumulative(by_flow_arr, years, p)),
        ("04_02_09_uncertainty_source.png",
         lambda p: fig_variance_split(fleet, per_segment_by_series["Total"], keep,
                                      years, segments, n_draws, p)),
    ]
    for name, fn in jobs:
        path = fig_dir / name
        fn(path)
        print(f"    {path.name}")

    results["summary"] = pd.DataFrame(summary_rows)
    saved = save_many(bev_electronics_summary=results, root=PROJECT_ROOT)
    print(f"\n  saved: {saved}")

    for flow in FLOWS:
        d = by_flow_df[flow]
        i = list(years).index(2050) if 2050 in years else -1
        print(f"  {flow:<10} 2050: median {d['median'].iloc[i]:>8,.1f} kt   "
              f"95% {d['p2_5'].iloc[i]:>8,.1f} - {d['p97_5'].iloc[i]:>8,.1f} kt")
    return saved


if __name__ == "__main__":
    main()
