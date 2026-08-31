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

# THE WIRING SERIES IS COPPER. Not "mostly copper" -- the electronics study's wiring
# model reports exactly one quantity, `Cu (kg)`, which mc_composition.py converts to
# grams and stores as the "Wiring" series. So the wiring draws already ARE copper
# mass per vehicle, per draw, per year, at the full draw count, with no element model
# or extra assumption involved. That is why `Wiring` needs no element file, and why
# the element path reproduces the wiring series exactly (5.2e-8, float32 precision).
#
# This stage used to report wiring copper as if it were all the copper, because the
# other three domains had no per-draw element data. They do now, and copper is
# reported across all four. The computed 2050 inflow split -- Wiring 86.2%, Motors
# 12.4%, PCB 0.9%, Sensors 0.25% -- independently reproduces the split measured from
# the study's own element table, from a completely different code path.

# ONE COLOUR PER DOMAIN, EVERYWHERE. Not a style preference -- a correctness one.
#
# These colours used to be handed out by position, `zip(domains, palette)`. That is
# fine only while every figure shows the same domains in the same order, and they do
# not: an element chart only shows the domains that actually contain that element, so
# neodymium (motors + sensors) gave motors the colour wiring had on the copper chart.
# Reading two figures side by side then meant re-learning the legend each time, and
# invited straightforward misreading.
#
# Looking these up by name makes a domain's colour independent of which other domains
# happen to be present.
DOMAIN_COLOR = {
    "Wiring":  "#2E86AB",   # blue
    "Motors":  "#E67E22",   # orange
    "PCB":     "#2b8a3e",   # green
    "Sensors": "#8e44ad",   # purple
    "Total":   "black",
}

# Same reasoning for the three flows, which are drawn together on many charts.
FLOW_COLOR = {"inflow": "#2E86AB", "outflow": "#E67E22", "collected": "#2b8a3e"}

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


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------
# HOW EACH DOMAIN CARRIES ITS ELEMENTS, AND WHY THEY DIFFER.
#
# Three of the four domains hand over a per-draw FRACTION of the domain's mass;
# sensors hand over an absolute per-draw MASS. That asymmetry is not untidiness, it
# is the only way to get sensors right.
#
#   Wiring   Cu = 1.0 exactly. The wiring model reports one quantity, `Cu (kg)`, so
#            the wiring series already IS copper. No element model involved.
#   Motors   fraction of whole-motor mass, including bulk aluminium and plastic.
#   PCB      fraction of board metal mass. Mass and composition come from the same
#            model, so they agree by construction.
#   Sensors  ABSOLUTE mg per vehicle, because mc_composition's Sensors mass is the
#            one domain built from MODES rather than draws or means. Summing modes
#            understates expected mass by 1.61x here (the mean of a triangular is
#            (min+mode+max)/3, and these tables are strongly right-skewed); with the
#            count convention the total gap is 1.73x. A fraction multiplied by that
#            mass would inherit the understatement invisibly. So sensor element mass
#            is taken at its own level and only the SHAPE of the sensor trajectory
#            -- its year-to-year profile, normalised to 1.0 at the base year -- is
#            taken from mc_composition.
ELEMENT_DOMAINS = ("Wiring", "Motors", "PCB", "Sensors")
SENSOR_BASE_YEAR = 2025          # the year the sensor study's composition is fixed at
MG_PER_GRAM = 1_000.0


def load_element_draws(elem_dir: Path, groups: tuple[str, ...]) -> dict:
    """
    Per-draw element data for every domain that has any.

    Returns {"Motors"|"PCB": {group: (elements, fractions)},
             "Sensors": {group: (elements, mg per vehicle)}}.

    Wiring is absent on purpose -- it needs no file. Missing domains raise, because
    silently reporting three domains' copper as if it were all of it is exactly the
    kind of quiet shortfall this stage exists to avoid.
    """
    if not elem_dir.is_dir():
        raise FileNotFoundError(
            f"No element draws at {elem_dir}.\n"
            f"In the RAWCLICVehicleElectronics repository run "
            f"ElectricMotorElementMC.py, PCBElementMC.py and SensorElementsMC.py, "
            f"then point params.materials.bev_electronics_element_draws_dir there."
        )

    def _names(path: Path) -> list[str]:
        return [x for x in path.read_text().split("\n") if x.strip()]

    out: dict = {"Motors": {}, "PCB": {}, "Sensors": {}}
    for g in groups:
        for dom, stem, kind in (("Motors", f"motors_{g}", "fractions"),
                                ("PCB", f"pcb_{g}", "fractions"),
                                ("Sensors", f"sensors_{g}", "mass_mg")):
            arr = elem_dir / f"{stem}_{kind}.npy"
            txt = elem_dir / (f"{stem}_mass_elements.txt" if kind == "mass_mg"
                              else f"{stem}_elements.txt")
            if not arr.exists() or not txt.exists():
                raise FileNotFoundError(
                    f"{dom} element draws missing for group {g}: expected "
                    f"{arr.name} and {txt.name} in {elem_dir}.")
            els = _names(txt)
            a = np.load(arr, mmap_mode="r")
            if a.shape[1] != len(els):
                raise ValueError(
                    f"{arr.name} has {a.shape[1]} columns but {txt.name} names "
                    f"{len(els)} elements.")
            out[dom][g] = (els, a)
    return out


def resolve_elements(wanted: tuple[str, ...], elem: dict) -> dict[str, list[str]]:
    """
    Work out where each requested element can be found, and drop the ones nowhere.

    An element missing from one domain is normal -- platinum is a sensor element and
    appears in no motor. An element missing from ALL of them cannot be answered by
    the element models this run was given, so it is SKIPPED with a note rather than
    stopping the run: which elements the upstream models happen to resolve is a
    property of those files, not of this stage, and the request list is only a
    selection of what to report. Reporting the rest is better than reporting nothing.

    Nothing resolving at all is a different matter -- that means the wrong directory
    or the wrong models -- and still raises.
    """
    available: dict[str, set[str]] = {"Wiring": {"Cu"}}
    for dom in ("Motors", "PCB", "Sensors"):
        available[dom] = set()
        for els, _ in elem[dom].values():
            available[dom].update(els)

    where: dict[str, list[str]] = {}
    unknown = []
    for e in wanted:
        doms = [d for d in ELEMENT_DOMAINS if e in available[d]]
        if doms:
            where[e] = doms
        else:
            unknown.append(e)

    every = sorted(set().union(*available.values()))
    if unknown:
        print(f"\n  NOTE: {unknown} are not resolved by any element model in this "
              f"draws directory, so they are skipped. Everything else is reported "
              f"as usual.\n        Available: {', '.join(every)}")
    if not where:
        raise ValueError(
            f"none of materials.bev_electronics_elements {list(wanted)} is resolved "
            f"by any domain. That points at the wrong element draws directory or the "
            f"wrong models, not at the request list.\nAvailable: {', '.join(every)}")
    return where


def element_flows(
    fleet, per_segment_by_series, elem, keep, flow, segments, seg_group,
    elements, where, years, n_draws, export=None,
):
    """
    Element mass by year, per element and per domain, in kilotonnes.

    Returns {element: {"total": (draws, years), domain: (draws, years), ...}} with
    every array already summarised -- only DataFrames come back, never the raw
    arrays, because holding 18 elements x 4 domains x 200,000 draws would be 60 GB.

    `export`, when given, is (out_dir, year_indices). The raw draws for just
    those years are written to disk on the way past, before the array is
    summarised and dropped. That is the whole point: the recovery model
    multiplies these by transfer coefficients that are themselves drawn, and a
    mean times a mean is not the mean of a product -- so it needs the draws,
    not the percentiles. One year is about 170 MB where the full span is 60 GB,
    which is why the slice is taken here rather than the arrays kept.

    THE ORDER OF OPERATIONS MATTERS. Segments are summed on RAW draws before any
    percentile is taken, exactly as `multiply` does for total mass: percentile-of-sum
    is the honest total, sum-of-percentiles assumes every segment hits its extreme in
    the same world.

    MEMORY. Segments are pooled per (domain, electronics group) once per flow -- 12
    arrays, float32, ~41 MB each at 200,000 draws x 51 years, so ~490 MB -- and
    reused across every element. Only one element's accumulator and one domain's
    contribution are held in float64 at a time, ~82 MB each.

    Those are this function's own arrays. The MEASURED peak for the whole stage is
    5.3 GB at 200,000 draws, most of it the fleet and electronics draws the stage
    already holds. Everything here scales linearly with draw count.
    """
    n_years = len(years)
    i_base = int(np.searchsorted(years, SENSOR_BASE_YEAR))
    i_base = min(max(i_base, 0), n_years - 1)

    # ---- pool the segments once, per (domain, group) ----------------------
    pooled: dict[tuple[str, str], np.ndarray] = {}
    for seg in segments:
        g = seg_group[seg]
        v = np.asarray(fleet[(seg, flow)][:n_draws][:, keep], dtype=np.float64)
        for dom in ELEMENT_DOMAINS:
            grams = per_segment_by_series[dom][seg]
            if dom == "Sensors":
                # Shape, not level: normalise the trajectory to 1.0 at the base year
                # so only its year-to-year profile is used. The sensor study's own
                # masses supply the level. Where the base year is zero there is no
                # trajectory to speak of and the shape is left flat.
                base = grams[:, i_base][:, None]
                contrib = v * np.divide(grams, base, out=np.ones_like(grams),
                                        where=base > 0)
            else:
                contrib = v * grams
            k = (dom, g)
            pooled[k] = contrib.astype(np.float32) if k not in pooled else \
                pooled[k] + contrib.astype(np.float32)

    # The mass of each domain itself, before any element split. The recovery
    # model needs it: its composition is a share of a parent, and the parent
    # there is the domain. Without it the only masses available would be the
    # tracked elements, which are a minority of a motor or a board.
    #
    # THE THREE DOMAINS DO NOT CARRY THEIR MASS THE SAME WAY, and taking the
    # pooled array as the mass is right for two of them and badly wrong for the
    # third:
    #
    #   Wiring   pooled IS the copper mass; the domain is copper by definition.
    #   Motors   pooled is vehicles x grams, so a mass. Elements are fractions
    #   PCB      of it.
    #   Sensors  pooled is vehicles x a NORMALISED TRAJECTORY -- shape only, no
    #            level, because the sensor study's own per-element masses supply
    #            the level further down. Summing it gives a number that is not a
    #            mass at all. The domain's mass is the sum over ALL its elements.
    #
    # Read as a mass, the sensor pooled array made the tracked elements come to
    # 11,861% of their own domain.
    if export is not None:
        for dom in ELEMENT_DOMAINS:
            groups = [(g, v) for (d, g), v in pooled.items() if d == dom]
            if not groups:
                continue

            if dom == "Sensors":
                domain_mass = np.zeros((n_draws, n_years), dtype=np.float64)
                for g, v in groups:
                    els, arr = elem[dom][g]
                    per_vehicle_mg = np.asarray(
                        arr[:n_draws, :], dtype=np.float64).sum(axis=1)[:, None]
                    domain_mass += v * (per_vehicle_mg / MG_PER_GRAM)
            else:
                domain_mass = np.sum([v for _, v in groups], axis=0, dtype=np.float64)

            domain_mass /= TONNES_PER_KILOTONNE
            _write_element_draws(export, flow, "__domain__", dom, domain_mass)
            del domain_mass

    # ---- one element at a time -------------------------------------------
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for e in elements:
        total = np.zeros((n_draws, n_years), dtype=np.float64)
        per_dom: dict[str, pd.DataFrame] = {}
        for dom in where[e]:
            part = np.zeros((n_draws, n_years), dtype=np.float64)
            for g in {seg_group[s] for s in segments}:
                k = (dom, g)
                if k not in pooled:
                    continue
                if dom == "Wiring":
                    part += pooled[k]                       # already copper
                    continue
                els, arr = elem[dom][g]
                if e not in els:
                    continue
                col = np.asarray(arr[:n_draws, els.index(e)], dtype=np.float64)[:, None]
                if dom == "Sensors":
                    part += pooled[k] * (col / MG_PER_GRAM)  # mg/vehicle -> g/vehicle
                else:
                    part += pooled[k] * col                  # fraction of domain mass
            part /= TONNES_PER_KILOTONNE
            total += part
            if export is not None:
                _write_element_draws(export, flow, e, dom, part)
            per_dom[dom] = summarize_by_year(part, years)
            del part
        per_dom["total"] = summarize_by_year(total, years)
        if export is not None:
            _write_element_draws(export, flow, e, "total", total)
        out[e] = per_dom
        del total
    del pooled
    return out


def _write_element_draws(export, flow: str, element: str, domain: str,
                         arr: np.ndarray) -> None:
    """
    Persist one element's draws for the wanted years, in kilotonnes.

    Written float32 to match the fleet draws this stage reads, which halves the
    file for a precision that is far finer than anything the inputs justify.
    """
    out_dir, year_index = export
    folder = out_dir / flow
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{element}__{domain}.npy",
            np.asarray(arr[:, year_index], dtype=np.float32))


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
    for flow in FLOWS:
        color = FLOW_COLOR[flow]
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
    ax.stackplot(years, *[by_series[d]["median"] for d in doms], labels=doms,
                 colors=[DOMAIN_COLOR[d] for d in doms], alpha=0.85)
    tot = by_series["Total"]
    # The band belongs on the total, and its absence was misleading. A stacked area
    # cannot carry a band per layer -- four overlapping bands would be unreadable --
    # but drawing the total as a bare line implied a precision the number does not
    # have. The layers are still medians; the total now shows its real spread.
    # The fill goes BEHIND the stack, so only the part rising above it is visible.
    # Drawn on top it greyed out the layers underneath and made the composition
    # unreadable, which defeats the point of a stacked chart. The lower bound would
    # then be hidden by the stack, so both bounds are also drawn as thin lines above
    # everything -- the lower one is legible against the coloured layers.
    ax.fill_between(years, tot["p2_5"], tot["p97_5"], color="0.45", alpha=0.28,
                    linewidth=0, label="Total, 95% band", zorder=0)
    for bound in ("p2_5", "p97_5"):
        ax.plot(years, tot[bound], color="0.25", linewidth=0.9, linestyle=":", zorder=6)
    ax.plot(years, tot["median"], color="black", linewidth=1.4,
            linestyle="--", label="Total (median)", zorder=7)
    ax.set_title(f"BEV electronics {FLOW_LABEL[flow]}, by domain "
                 f"(layers are medians; band is on the total)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Electronics material [kt/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Layers are medians and stack only approximately -- the median "
                         "of a sum is not the sum of medians. The dashed line and its band "
                         "are the true total and its 95% range.",
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


def fig_by_segment(by_seg: dict[str, pd.DataFrame], years, flow: str, path: Path,
                   total: pd.DataFrame | None = None) -> None:
    """
    FIGURE 6 -- which car sizes carry the material.

    Twelve stacked layers, each a median. As in `fig_domains`, a band per layer is
    not drawable, but the TOTAL carries one when `total` is supplied -- without it
    the figure reads as though the overall quantity were known exactly.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    segs = sorted(by_seg, key=lambda s: (s.startswith("J"), s))
    ax.stackplot(years, *[by_seg[s]["median"] for s in segs], labels=segs, alpha=0.85)
    if total is not None:
        # Behind the stack, with both bounds also drawn as thin lines -- see
        # fig_domains for why a fill drawn on top makes the composition unreadable.
        ax.fill_between(years, total["p2_5"], total["p97_5"], color="0.45", alpha=0.28,
                        linewidth=0, label="Total, 95% band", zorder=0)
        for bound in ("p2_5", "p97_5"):
            ax.plot(years, total[bound], color="0.25", linewidth=0.9, linestyle=":", zorder=6)
        ax.plot(years, total["median"], color="black", linewidth=1.4, linestyle="--",
                label="Total (median)", zorder=7)
    ax.set_title(f"BEV electronics {FLOW_LABEL[flow]}, by segment "
                 f"(layers are medians; band is on the total)", fontsize=12)
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
    for flow in FLOWS:
        color = FLOW_COLOR[flow]
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
        for ser in series_list:
            color = DOMAIN_COLOR[ser]
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
    for flow in FLOWS:
        color = FLOW_COLOR[flow]
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



def fig_domain_recovery(by_series_by_flow, years, series, path: Path) -> None:
    """
    FIGURE 10 -- how much of each kind of electronics is actually recovered.

    One panel per domain, showing what enters the fleet, what leaves it, and what is
    collected for recycling. The gap between the last two is the material that leaves
    the fleet and is not recovered -- exported or untraceable.

    Reading it: outflow lags inflow by roughly a vehicle lifetime, so the two are not
    meant to meet until the fleet stops growing. What matters is the distance between
    outflow and collected, which is set by the collection rate rather than by
    anything about the electronics.
    """
    doms = [s for s in series if s != "Total"]
    fig, axes = plt.subplots(1, len(doms), figsize=(4.6 * len(doms), 4.4))
    axes = np.atleast_1d(axes)
    for ax, dom in zip(axes, doms):
        for flow in FLOWS:
            color = FLOW_COLOR[flow]
            df = by_series_by_flow[flow][dom]
            ax.plot(years, df["median"], color=color, linewidth=1.6, label=flow)
            ax.fill_between(years, df["p2_5"], df["p97_5"], color=color,
                            alpha=0.18, linewidth=0)
        ax.set_title(dom, fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Electronics material [kt/year]")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Each kind of electronics: what enters, what leaves, what is "
                 "collected (median and 95% band)", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)



def _unit(peak: float) -> tuple[float, str]:
    """Pick kt, t or kg so an axis reads in human numbers rather than 0.00004 kt."""
    if peak >= 1.0:
        return 1.0, "kt"
    if peak >= 1e-3:
        return 1e3, "t"
    return 1e6, "kg"


def fig_element_total(el_by_flow, years, element, path: Path, n_draws: int) -> None:
    """
    All three flows for ONE element, summed over every domain that contains it.

    This is the recycling view: what enters the fleet, what leaves it, and what is
    actually collected, for the element itself rather than for a bag of electronics.
    """
    peak = max(el_by_flow[f][element]["total"]["p97_5"].max() for f in FLOWS)
    k, unit = _unit(peak)
    fig, ax = plt.subplots(figsize=(11, 6))
    for flow in FLOWS:
        color = FLOW_COLOR[flow]
        df = el_by_flow[flow][element]["total"]
        ax.plot(years, df["median"] * k, color=color, linewidth=1.8,
                label=f"{flow} ({FLOW_LABEL[flow]})")
        ax.fill_between(years, df["p2_5"] * k, df["p97_5"] * k, color=color,
                        alpha=0.20, linewidth=0)
    ax.set_title(f"{element} in BEV electronics -- median and 95% band "
                 f"(n={n_draws:,})", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{element} [{unit}/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_element_domains(el_by_flow, years, element, flow, path: Path) -> None:
    """Which domain carries this element -- stacked, with the total's band behind."""
    doms = [d for d in ELEMENT_DOMAINS if d in el_by_flow[flow][element]]
    tot = el_by_flow[flow][element]["total"]
    k, unit = _unit(tot["p97_5"].max())
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(years, tot["p2_5"] * k, tot["p97_5"] * k, color="#999999",
                    alpha=0.18, linewidth=0, zorder=0, label="95% band on the total")
    base = np.zeros(len(years))
    for d in doms:
        color = DOMAIN_COLOR[d]
        v = el_by_flow[flow][element][d]["median"].to_numpy() * k
        ax.fill_between(years, base, base + v, color=color, alpha=0.85,
                        linewidth=0, label=d, zorder=2)
        base = base + v
    ax.plot(years, tot["p2_5"] * k, color="#555555", linestyle=":", linewidth=1, zorder=6)
    ax.plot(years, tot["p97_5"] * k, color="#555555", linestyle=":", linewidth=1, zorder=6)
    ax.set_title(f"{element} {FLOW_LABEL[flow]}, by domain -- medians stacked",
                 fontsize=12)
    ax.set_xlabel("Year"); ax.set_ylabel(f"{element} [{unit}/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.text(0.01, 0.01,
             "Stacked medians do not add to the median of the total; the grey band "
             "and dotted lines are the total's own 95% interval, taken on raw draws.",
             fontsize=7, color="#666666")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_element_panel(el_by_flow, years, elements, flow, path: Path, n_draws: int) -> None:
    """Every requested element at once, each on its own axis and its own unit."""
    n = len(elements)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ax, e in zip(axes.ravel(), elements):
        df = el_by_flow[flow][e]["total"]
        k, unit = _unit(df["p97_5"].max())
        ax.plot(years, df["median"] * k, color="#2E86AB", linewidth=1.5)
        ax.fill_between(years, df["p2_5"] * k, df["p97_5"] * k,
                        color="#2E86AB", alpha=0.20, linewidth=0)
        ax.set_title(f"{e}  [{unit}/y]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Critical elements {FLOW_LABEL[flow]} -- median and 95% band "
                 f"(n={n_draws:,})", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_element_recovery(el_by_flow, years, elements, path: Path) -> None:
    """
    Collected as a share of what left the fleet -- ONE curve, not one per element.

    THIS FIGURE DELIBERATELY DOES NOT SHOW EIGHTEEN LINES. The first version did, and
    it was misleading: every element came out between 87.65% and 87.96%, so on an
    axis auto-scaled to a 0.3-point range the lines fanned apart dramatically and
    invited exactly the wrong conclusion -- that some elements are recovered better
    than others.

    They are not, and cannot be, in this model. Collection is applied to VEHICLES:
    a car is collected or it is not, and everything in it goes with it. Every element
    therefore shares one collection rate by construction, and the residual scatter is
    draw noise in a ratio of medians, not a real difference.

    So the honest chart is the single shared rate, drawn on an axis from zero where
    its flatness is visible, with the spread across elements quantified in the caption
    rather than dramatised by the y-axis.

    What is NOT modelled, and matters for recycling: element-specific recovery inside
    the recycling chain. Copper in a harness is recovered at a very different yield
    from neodymium in a bonded magnet or gold on a board. That is a separate process
    step downstream of this model, and nothing here should be read as a recovery
    yield.
    """
    rates = []
    for e in elements:
        out = el_by_flow["outflow"][e]["total"]["median"].to_numpy()
        col = el_by_flow["collected"][e]["total"]["median"].to_numpy()
        rates.append(np.divide(col, out, out=np.full_like(col, np.nan), where=out > 0))
    R = 100 * np.vstack(rates)
    lo, hi, mid = np.nanmin(R, axis=0), np.nanmax(R, axis=0), np.nanmedian(R, axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(years, mid, color="#2b8a3e", linewidth=2.0,
            label="collection rate (identical for every element)")
    ax.fill_between(years, lo, hi, color="#2b8a3e", alpha=0.25, linewidth=0,
                    label="full spread across all elements")
    ax.set_ylim(0, 100)
    ax.set_title("Collected as a share of what leaves the fleet", fontsize=12)
    ax.set_xlabel("Year"); ax.set_ylabel("Collected / outflow [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.01,
             f"Collection is applied to whole vehicles, so all {len(elements)} elements "
             f"share one rate: the spread across them is {np.nanmax(hi - lo):.2f} "
             f"percentage points at its widest, which is draw noise. This is a "
             f"COLLECTION rate, not a recovery yield -- element-specific losses inside "
             f"the recycling chain are not modelled here.",
             fontsize=7, color="#666666")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_copper(el_by_flow, years, path: Path, n_draws: int) -> None:
    """
    FIGURE 11 -- copper, the priority element. ALL of it, across all four domains.

    Until the element models emitted per-draw fractions this figure could only show
    WIRING copper and said so in its caption, because wiring is the one domain whose
    model reports copper directly. That restriction is gone: motors, boards and
    sensors now contribute their own copper, each from its own Monte Carlo.

    The left panel is the recycling view -- in, out, and collected. The right panel
    shows where the copper actually sits, and how that shifts as motors take a larger
    share of a vehicle's copper over time.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    for flow in FLOWS:
        color = FLOW_COLOR[flow]
        df = el_by_flow[flow]["Cu"]["total"]
        ax.plot(years, df["median"], color=color, linewidth=1.8,
                label=f"{flow} ({FLOW_LABEL[flow]})")
        ax.fill_between(years, df["p2_5"], df["p97_5"], color=color, alpha=0.20,
                        linewidth=0)
    ax.set_title("Total copper -- median and 95% band", fontsize=12)
    ax.set_xlabel("Year"); ax.set_ylabel("Copper [kt/year]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    doms = [d for d in ELEMENT_DOMAINS if d in el_by_flow["inflow"]["Cu"]]
    tot = el_by_flow["inflow"]["Cu"]["total"]
    ax2.fill_between(years, tot["p2_5"], tot["p97_5"], color="#999999", alpha=0.18,
                     linewidth=0, zorder=0, label="95% band on the total")
    base = np.zeros(len(years))
    for d in doms:
        color = DOMAIN_COLOR[d]
        v = el_by_flow["inflow"]["Cu"][d]["median"].to_numpy()
        ax2.fill_between(years, base, base + v, color=color, alpha=0.85, linewidth=0,
                         label=d, zorder=2)
        base = base + v
    ax2.plot(years, tot["p2_5"], color="#555555", linestyle=":", linewidth=1, zorder=6)
    ax2.plot(years, tot["p97_5"], color="#555555", linestyle=":", linewidth=1, zorder=6)
    ax2.set_title("Copper entering the fleet, by domain", fontsize=12)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Copper [kt/year]")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, ncol=2, fontsize=9)

    fig.suptitle(f"Copper in BEV electronics (n={n_draws:,})", fontsize=13)
    fig.text(0.01, 0.01,
             "Wiring copper is exact -- that model reports Cu directly. Motor, board "
             "and sensor copper come from their own element Monte Carlos. Stacked "
             "medians do not add to the median of the total; the band is taken on raw "
             "draws.",
             fontsize=7, color="#666666")
    plt.tight_layout(rect=(0, 0.015, 1, 0.96))
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

    # The domain split -- wiring, sensors, circuit boards, motors -- for EVERY flow,
    # not only what enters the fleet. "How much wiring is actually collected for
    # recycling" is a different question from "how much wiring is sold", and the
    # answer matters more: it is the material that can be recovered. Reporting only
    # the inflow split left that question unanswered.
    by_series_by_flow: dict[str, dict[str, pd.DataFrame]] = {}
    for flow in FLOWS:
        per_flow: dict[str, pd.DataFrame] = {}
        for ser in series:
            a = multiply(fleet, per_segment_by_series[ser], keep, flow, segments, n_draws)
            per_flow[ser] = summarize_by_year(a, years)
            if ser != "Total":
                for _, row in per_flow[ser].iterrows():
                    summary_rows.append({"flow": flow, "series": ser, **row.to_dict()})
        by_series_by_flow[flow] = per_flow
    by_series_df = by_series_by_flow["inflow"]

    by_seg_df = {}
    for seg in segments:
        v = np.asarray(fleet[(seg, "inflow")][:n_draws][:, keep], dtype=np.float64)
        a = v * per_segment_by_series["Total"][seg] / TONNES_PER_KILOTONNE
        by_seg_df[seg] = summarize_by_year(a, years)

    # ---- elements -------------------------------------------------------
    missing_series = [d for d in ELEMENT_DOMAINS if d not in per_segment_by_series]
    if missing_series:
        raise ValueError(
            f"element results need every domain in the series list, but "
            f"{missing_series} are absent. Add them to "
            f"materials.bev_electronics_series (currently {list(series)}).")

    seg_group: dict[str, str] = {}
    for group, small, large in pairs:
        for s in (small, large, "J" + small, "J" + large):
            seg_group[s] = group
    groups = tuple(dict.fromkeys(g for g, _, _ in pairs))

    elem_dir = Path(p04.bev_electronics_element_draws_dir)
    if not elem_dir.is_absolute():
        elem_dir = (PROJECT_ROOT / "code" / elem_dir).resolve()
    elem = load_element_draws(elem_dir, groups)
    requested = tuple(p04.bev_electronics_elements)
    where = resolve_elements(requested, elem)
    # Only what the models actually resolve goes on from here, so every downstream
    # consumer -- figures, tables, the draw export -- sees one consistent list.
    elements = tuple(e for e in requested if e in where)

    n_elem = min(a.shape[0] for d in ("Motors", "PCB", "Sensors")
                 for _, a in elem[d].values())
    if n_elem < n_draws:
        raise ValueError(
            f"element draws hold {n_elem:,} rows but this run uses {n_draws:,}. "
            f"Re-run the element models at >= {n_draws:,} draws -- resampling them "
            f"to fit would invent draws the models never made.")

    print(f"\n  elements: {len(elements)} of {len(requested)} requested, resolved "
          f"from {elem_dir.name}")
    for e in elements:
        print(f"    {e:<4} {' + '.join(where[e])}")

    # ---- the slice of draws the recovery model reads --------------------
    # This stage keeps percentiles and drops the draws, which is right for a
    # stage that plots bands. The recovery model cannot use percentiles: it
    # multiplies these by transfer coefficients that are also drawn, and a mean
    # times a mean is not the mean of a product. So the draws for a few named
    # years are written on the way past.
    export = None
    wanted_years = tuple(int(y) for y in (p04.bev_electronics_element_draws_years or ()))
    if wanted_years:
        present = [y for y in wanted_years if y in set(years.tolist())]
        missing = sorted(set(wanted_years) - set(present))
        if missing:
            print(f"\n  NOTE: element draw export skips {missing}, outside "
                  f"{y_lo}-{y_hi}.")
        if present:
            out_dir = (PROJECT_ROOT / "data" / "processed"
                       / p04.bev_electronics_element_draws_out_dir / scenario)
            year_index = np.searchsorted(years, np.array(present))
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "years.npy", np.array(present))
            export = (out_dir, year_index)
            size = (len(elements) * (len(ELEMENT_DOMAINS) + 1) * len(FLOWS)
                    * n_draws * len(present) * 4 / 1e9)
            print(f"\n  element draws -> {out_dir.relative_to(PROJECT_ROOT)}  "
                  f"years {present}  (about {size:.2f} GB)")

    el_by_flow: dict[str, dict] = {}
    for flow in FLOWS:
        el_by_flow[flow] = element_flows(
            fleet, per_segment_by_series, elem, keep, flow, segments, seg_group,
            elements, where, years, n_draws, export=export)
        for e in elements:
            for dom, df in el_by_flow[flow][e].items():
                for _, row in df.iterrows():
                    summary_rows.append({"flow": flow, "series": f"element:{e}",
                                         "domain": dom, **row.to_dict()})

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
        ("04_02_05_by_domain_inflow.png",
         lambda p: fig_domains(by_series_by_flow["inflow"], years, "inflow", p)),
        ("04_02_05b_by_domain_outflow.png",
         lambda p: fig_domains(by_series_by_flow["outflow"], years, "outflow", p)),
        ("04_02_05c_by_domain_collected.png",
         lambda p: fig_domains(by_series_by_flow["collected"], years, "collected", p)),
        ("04_02_11_copper.png",
         lambda p: fig_copper(el_by_flow, years, p, n_draws)),
        ("04_02_12_copper_collected_by_domain.png",
         lambda p: fig_element_domains(el_by_flow, years, "Cu", "collected", p)),
        ("04_02_13_elements_inflow.png",
         lambda p: fig_element_panel(el_by_flow, years, elements, "inflow", p, n_draws)),
        ("04_02_14_elements_collected.png",
         lambda p: fig_element_panel(el_by_flow, years, elements, "collected", p, n_draws)),
        ("04_02_15_element_recovery.png",
         lambda p: fig_element_recovery(el_by_flow, years, elements, p)),
        ("04_02_10_domain_recovery.png",
         lambda p: fig_domain_recovery(by_series_by_flow, years, series, p)),
        ("04_02_06_by_segment.png",
         lambda p: fig_by_segment(by_seg_df, years, "inflow", p,
                                  total=by_flow_df["inflow"])),
        ("04_02_07_distribution_snapshots.png",
         lambda p: fig_snapshots(by_flow_arr["inflow"], years, (2030, 2050, 2070), "inflow", p)),
        ("04_02_08_cumulative.png",
         lambda p: fig_cumulative(by_flow_arr, years, p)),
        ("04_02_09_uncertainty_source.png",
         lambda p: fig_variance_split(fleet, per_segment_by_series["Total"], keep,
                                      years, segments, n_draws, p)),
    ]
    # One figure per element, for the three flows and for the domain split. These
    # are the per-element detail behind the summary panels above.
    for e in elements:
        jobs.append((f"04_02_16_element_{e}.png",
                     lambda p, _e=e: fig_element_total(el_by_flow, years, _e, p, n_draws)))
        jobs.append((f"04_02_17_element_{e}_by_domain.png",
                     lambda p, _e=e: fig_element_domains(el_by_flow, years, _e,
                                                         "collected", p)))

    for name, fn in jobs:
        path = fig_dir / name
        fn(path)
        print(f"    {path.name}")

    results["summary"] = pd.DataFrame(summary_rows)
    saved = save_many(bev_electronics_summary=results, root=PROJECT_ROOT)
    print(f"\n  saved: {saved}")

    results["elements"] = elements
    results["element_domains"] = where

    i = list(years).index(2050) if 2050 in years else -1
    print()
    # The copper read-out is a convenience, not a requirement: it is skipped rather
    # than crashing if copper was left out of the request list.
    if "Cu" in elements:
        print("  COPPER, all four domains [kt/year]:")
        for flow in FLOWS:
            d = el_by_flow[flow]["Cu"]["total"]
            print(f"    {flow:<10} 2050: median {d['median'].iloc[i]:>8,.1f}   "
                  f"95% {d['p2_5'].iloc[i]:>8,.1f} - {d['p97_5'].iloc[i]:>8,.1f}")
        print("    2050 inflow by domain:", ", ".join(
            f"{d} {el_by_flow['inflow']['Cu'][d]['median'].iloc[i]:,.1f}"
            for d in ELEMENT_DOMAINS if d in el_by_flow["inflow"]["Cu"]))
        print()
    print("  ELEMENTS COLLECTED FOR RECYCLING, 2050 -- median [95% band]:")
    for e in elements:
        d = el_by_flow["collected"][e]["total"]
        k, unit = _unit(d["p97_5"].iloc[i])
        print(f"    {e:<4} {d['median'].iloc[i]*k:>10,.2f} {unit:<3} "
              f"[{d['p2_5'].iloc[i]*k:>10,.2f} - {d['p97_5'].iloc[i]*k:>10,.2f}]   "
              f"from {' + '.join(where[e])}")
    print()
    print("  collected for recycling, by domain, 2050 [kt/year]:")
    for ser in series:
        if ser == "Total":
            continue
        d = by_series_by_flow["collected"][ser]
        i = list(years).index(2050) if 2050 in years else -1
        print(f"    {ser:<9} median {d['median'].iloc[i]:>8,.1f}   "
              f"95% {d['p2_5'].iloc[i]:>8,.1f} - {d['p97_5'].iloc[i]:>8,.1f}")
    print()

    for flow in FLOWS:
        d = by_flow_df[flow]
        i = list(years).index(2050) if 2050 in years else -1
        print(f"  {flow:<10} 2050: median {d['median'].iloc[i]:>8,.1f} kt   "
              f"95% {d['p2_5'].iloc[i]:>8,.1f} - {d['p97_5'].iloc[i]:>8,.1f} kt")
    return saved


if __name__ == "__main__":
    main()
