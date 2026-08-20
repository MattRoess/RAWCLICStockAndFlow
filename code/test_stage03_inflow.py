"""
test_stage03_inflow.py -- regression tests for how stage 02's inflow uncertainty is
carried into stage 03_02.

    .venv/bin/python code/test_stage03_inflow.py

RUN THIS AFTER ANY CHANGE to the composition block in `03_02_adjustedflows.py`, to
`build_inflow_draws_by_drivetrain`, or to `inflow_uncertainty_parent_by_drv`. It runs
in about a minute at the reduced draw count below and needs no pipeline run.

WHY THIS FILE EXISTS. That block was wrong for four days and produced inflow figures
that were off by a factor of two, with nothing in the pipeline detecting it: every
stage ran, every artifact was written, every figure rendered. It was found by a
reader comparing two charts by hand. Each test below is one of the things that would
have caught it immediately.

The tests check the RULE, replayed against the real data, not the plumbing. Test 1
is the decisive one -- it fails loudly against the old rule and passes against the
current one.

EXTERNAL VALIDATION. Tests 5 and 6 compare against real EU registration statistics
(the EEA file on disk, itself confirmed against ACEA's published 2023 EU total of
10.5 million to within 2%). Model output that drifts far from the real record is a
result worth knowing about, so those tests carry deliberately wide bounds: they exist
to catch a factor-of-two regression, not to police normal modelling change.
"""
from __future__ import annotations

import sys
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import load_many                                  # noqa: E402
from src.stockflow_model import build_inflow_draws_by_drivetrain     # noqa: E402

# Enough draws for stable means. Every defect these tests target is STRUCTURAL --
# a factor of two, or an exact zero -- so it shows at any draw count. Raising this
# makes the run slower without making the tests stronger.
N_DRAWS = 4000

# Years with real registration data on disk to compare against.
REAL_YEARS = list(range(2010, 2020))     # 2020+ excluded: see test 5's docstring

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}\n         {detail}")


# ---------------------------------------------------------------- fixtures
def build_fixture():
    """The real deterministic inflow, the real stage-02 draws, and the parent map."""
    params = load_many("params", root=ROOT)["params"]
    params = dataclasses.replace(
        params, monte_carlo=dataclasses.replace(params.monte_carlo, n_draws=N_DRAWS)
    )
    stock_dict = load_many("stock_dict", root=ROOT)["stock_dict"]

    years = np.arange(1975, 2071, dtype=int)
    draws = build_inflow_draws_by_drivetrain(
        stock_dict=stock_dict, params=params,
        parent_by_drv=params.stock_flow.inflow_uncertainty_parent_by_drv, years=years)
    yi = {int(y): i for i, y in enumerate(years)}

    tracker = load_many("tracker_keyed_BAU", root=ROOT)["tracker_keyed_BAU"]
    frames = []
    for (region, drv), df in tracker.items():
        g = (df[df["flow"] == "inflow"]
             .groupby(["Segment", "scrap_year"], as_index=False)["amount"].sum())
        g["Region"], g["Drive Train"] = region, drv
        frames.append(g)
    det = (pd.concat(frames, ignore_index=True)
             .rename(columns={"scrap_year": "year", "amount": "value"}))

    parent_of = dict(params.stock_flow.inflow_uncertainty_parent_by_drv)
    det["_parent"] = det["Drive Train"].map(parent_of).fillna(det["Drive Train"])
    det = det.join(det.groupby(["_parent", "year"])["value"].sum().rename("parent_total"),
                   on=["_parent", "year"])
    return det, draws, yi, parent_of


def compose(det, draws, yi, years_wanted):
    """The rule as it stands in 03_02. Returns per-(drivetrain, year) draw vectors."""
    agg: dict[tuple[str, int], np.ndarray] = {}
    share_sum: dict[tuple[str, int], float] = {}
    sub = det[det["year"].isin(years_wanted)]
    for r in sub.to_dict("records"):
        drv, yr, val, par = r["Drive Train"], int(r["year"]), float(r["value"]), r["_parent"]
        vol, i = draws.get(drv), yi.get(yr)
        if vol is None or i is None or not np.isfinite(vol[i]).all():
            continue
        col = np.asarray(vol[i], dtype=np.float64)
        pt = float(r["parent_total"])
        share = val / pt if pt else 0.0
        share_sum[(par, yr)] = share_sum.get((par, yr), 0.0) + share
        raw = val + share * (col - col.mean())
        agg.setdefault((drv, yr), [0.0, 0.0])
        agg[(drv, yr)][0] = agg[(drv, yr)][0] + raw                 # before flooring
        agg[(drv, yr)][1] = agg[(drv, yr)][1] + np.maximum(raw, 0.0)  # as used
    return agg, share_sum


# -------------------------------------------------------------------- tests
def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print(f"\n  draws={N_DRAWS:,}  project={ROOT}\n")

    det, draws, yi, parent_of = build_fixture()
    probe = sorted(set(REAL_YEARS) | {2004, 2005, 2006, 2030, 2040})
    agg, share_sum = compose(det, draws, yi, probe)
    determ = det.groupby(["Drive Train", "year"])["value"].sum()

    # 1. THE DECISIVE ONE. Each parent's volume is divided among its children, so
    #    every share under a (parent, year) must sum to exactly 1. Under the old
    #    rule this summed to 2.0 for Liquids and Hybrid -- both children took the
    #    whole volume.
    worst = max(share_sum.items(), key=lambda kv: abs(kv[1] - 1.0))
    check("shares sum to 1 per (parent, year)",
          abs(worst[1] - 1.0) <= 1e-9,
          f"worst: parent={worst[0][0]!r} year={worst[0][1]} sum={worst[1]:.9f} "
          f"over {len(share_sum)} pairs (old rule gave 2.0 for Liquids and Hybrid)")

    # 2. BEFORE FLOORING, the composed mean must equal the deterministic value
    #    exactly. That is what makes the 2005 boundary continuous by construction
    #    rather than by tuning, and it is the core property of the additive rule.
    ratios = []
    for (drv, yr), (raw, _floored) in agg.items():
        d = float(determ.get((drv, yr), np.nan))
        if np.isfinite(d) and d > 0:
            ratios.append((abs(float(np.mean(raw)) / d - 1.0), drv, yr))
    worst_r = max(ratios) if ratios else (0.0, "-", 0)
    check("pre-floor composed mean == deterministic value",
          worst_r[0] <= 1e-9,
          f"worst deviation {worst_r[0]:.2e} at {worst_r[1]} {worst_r[2]} "
          f"across {len(ratios)} (drivetrain, year) pairs")

    # 2b. FLOORING CAN ONLY LIFT, NEVER LOWER. Flooring per draw is deliberate and
    #     nonlinear -- mean(max(x,0)) >= max(mean(x),0) -- so the mean after
    #     flooring must be >= the deterministic value, never below it. Measured, the
    #     lift is exactly 1.00x wherever there is real volume and only departs from
    #     it in the tail of a phase-out: HEV 2040 has a deterministic 0.0128 million
    #     (12,800 cars), 36.6% of draws go negative there, and the mean lifts to
    #     0.1323. That is the correct Monte Carlo answer -- in a draw where the
    #     fleet target lands higher, hybrid inflow really is still positive that
    #     year -- but any result in a near-zero year should be read with it in mind.
    lifts = []
    for (drv, yr), (_raw, floored) in agg.items():
        d = float(determ.get((drv, yr), np.nan))
        if np.isfinite(d) and d > 0:
            lifts.append((float(np.mean(floored)) / d, drv, yr))
    below = [x for x in lifts if x[0] < 1.0 - 1e-9]
    worst_lift = max(lifts) if lifts else (1.0, "-", 0)
    check("flooring only lifts the mean, never lowers it",
          not below,
          f"largest lift {worst_lift[0]:.2f}x at {worst_lift[1]} {worst_lift[2]} "
          f"(phase-out tail); {len(below)} pairs below the deterministic value "
          f"(must be 0)")

    # 3. No spike at the base year. Stage 02's first modelled year is 2005 and its
    #    inflow there is identically zero by construction; the rule must leave the
    #    deterministic value standing rather than overwrite it with that zero.
    bad = []
    for drv in ("Diesel", "Petrol"):
        v = agg.get((drv, 2005))
        v = v[1] if v is not None else None
        d = float(determ.get((drv, 2005), np.nan))
        if v is None or not np.isfinite(d):
            continue
        if abs(float(np.mean(v)) / d - 1.0) > 1e-9:
            bad.append(f"{drv} {np.mean(v):.4f} vs {d:.4f}")
    check("2005 keeps its deterministic value (no zero spike)",
          not bad,
          "Diesel and Petrol both equal the deterministic value at 2005"
          if not bad else "; ".join(bad))

    # 4. Years stage 02 does not model get no override at all.
    pre = [k for k in agg if k[1] < 2005]
    check("pre-2005 years are never overridden",
          not pre,
          f"{len(pre)} overridden years before 2005 (must be 0) -- stage 02 marks "
          f"them NaN and they keep their deterministic value")

    # 5. Against the real record. Wide bounds on purpose: this catches a
    #    factor-of-two regression, not ordinary modelling drift. Restricted to
    #    2010-2019 because from 2020 the real series collapses with COVID and the
    #    chip shortage, which a stock-driven scenario model does not reproduce.
    eea = (pd.read_csv(ROOT / "data" / "raw" / "EEA_final_data.csv")
             .groupby(["Year", "Drive Train"])["Registrations"].sum().unstack(fill_value=0) / 1e6)
    for drv, lo, hi in (("Diesel", 0.8, 1.5), ("Petrol", 0.8, 1.5), ("BEV", 0.5, 2.5)):
        rs = []
        for y in REAL_YEARS:
            if y not in eea.index or (drv, y) not in agg:
                continue
            real = float(eea.loc[y].get(drv, np.nan))
            if np.isfinite(real) and real > 0:
                rs.append(float(np.mean(agg[(drv, y)][1])) / real)
        if not rs:
            continue
        m = float(np.mean(rs))
        check(f"{drv} within [{lo}, {hi}] x real registrations, 2010-2019",
              lo <= m <= hi,
              f"mean ratio {m:.2f}x (range {min(rs):.2f}-{max(rs):.2f}); "
              f"the old rule gave {'2.2' if drv in ('Diesel','Petrol') else '1.0'}x")

    # 6. The band must be real where stage 02 has information, and zero where it
    #    does not. A zero-width band at 2006 would mean the uncertainty stopped
    #    being carried at all.
    v05 = agg.get(("Diesel", 2005))[1] if ("Diesel", 2005) in agg else None
    v06 = agg.get(("Diesel", 2006))[1] if ("Diesel", 2006) in agg else None
    w05 = float(np.ptp(np.percentile(v05, [2.5, 97.5]))) if v05 is not None else np.nan
    w06 = float(np.ptp(np.percentile(v06, [2.5, 97.5]))) if v06 is not None else np.nan
    check("uncertainty is carried (2006 band > 0, 2005 band == 0)",
          w06 > 0.05 and w05 == 0.0,
          f"Diesel 95% width: 2005 {w05:.4f} (stage 02 has no information there), "
          f"2006 {w06:.4f}")

    # ---- the HEV carve (option B) ------------------------------------------
    # HEV is carved out of Liquids, not split off Hybrid. REMIND has no non-plug-in
    # hybrid category at all -- "Hybrid electric" is the plug-in -- so splitting it
    # into HEV and PHEV invented an HEV series while the real hybrids stayed inside
    # Liquids. See documentation/DESIGN_hev_carve_from_liquids.md.
    import src.disaggregation as dg                                    # noqa: E402
    from src.disaggregation import prepare_eea_share_tables            # noqa: E402

    params = load_many("params", root=ROOT)["params"]
    p03 = params.disaggregation
    if getattr(p03, "hev_carved_from_liquids", False):
        out_dir = str((ROOT / "code" / p03.output_dir).resolve()) + "/"
        eea_in = str((ROOT / "code" / p03.eea_input_dir).resolve()) + "/"
        eea_data, _seg, liq_ext = prepare_eea_share_tables(
            output_dir=out_dir, eea_input_dir=eea_in,
            start_year_model=int(p03.start_year_model),
            end_year_model=int(p03.end_year_model),
            introduction_year_by_drv=p03.introduction_year_by_drv)
        yrs = np.arange(int(p03.start_year_model), int(p03.end_year_model) + 1)
        intro = int(p03.introduction_year_by_drv["HEV"])
        endy = int(p03.hev_share_phaseout_end_year)
        tbl = dg.build_liquids_three_way_split(
            eea_data=eea_data, liquids_shares_ext=liq_ext, years_full=yrs,
            hev_introduction_year=intro, phaseout_end_year=endy)

        worst_c = float((tbl.sum(axis=1) - 1.0).abs().max())
        check("Diesel + Petrol + HEV == 1 for every year",
              worst_c <= 1e-12,
              f"worst deviation {worst_c:.2e} over {len(tbl)} years")

        s_intro = float(tbl.loc[intro, "HEV"])
        s_end = float(tbl.loc[endy, "HEV"]) if endy in tbl.index else 0.0
        s_pre = float(tbl.loc[intro - 1, "HEV"]) if (intro - 1) in tbl.index else 0.0
        check("HEV share is zero before introduction and after phase-out",
              abs(s_intro) < 1e-12 and abs(s_end) < 1e-12 and abs(s_pre) < 1e-12,
              f"s({intro-1})={s_pre:.2e}  s({intro})={s_intro:.2e}  s({endy})={s_end:.2e}")

        mono = tbl.loc[intro:2019, "HEV"]
        check("HEV share increases from introduction to the first observation",
              bool((mono.diff().dropna() >= -1e-12).all()) and float(mono.iloc[-1]) > 0,
              f"s rises {100*float(mono.iloc[0]):.2f}% -> {100*float(mono.iloc[-1]):.2f}% "
              f"across {intro}-2019, never decreasing")

        # Measured from the SAME country-scoped frame the builder uses (EU27 plus
        # Norway and Iceland), not the raw 30-country file -- an earlier version of
        # this check compared against the raw file and failed at 4.569 vs 4.385%,
        # which was the scoping working correctly, not a defect.
        scoped = (eea_data[eea_data["Drive Train"].isin(["HEV", "Petrol", "Diesel"])]
                  .groupby(["Year", "Drive Train"])["Registrations"].sum()
                  .unstack(fill_value=0.0))
        obs19 = float(scoped.loc[2019, "HEV"] / scoped.loc[2019].sum())
        check("HEV share at 2019 matches the observed registration share",
              abs(float(tbl.loc[2019, "HEV"]) - obs19) < 1e-9,
              f"table {100*float(tbl.loc[2019,'HEV']):.3f}%  vs observed (scoped) "
              f"{100*obs19:.3f}%")

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n  {len(_results) - n_fail}/{len(_results)} passed")
    if n_fail:
        print("\n  FAILURES:")
        for s, name, detail in _results:
            if s == FAIL:
                print(f"    {name}: {detail}")
        print("\n  Do not commit a change that leaves these failing. See the protected"
              "\n  note above the composition block in code/03_02_adjustedflows.py.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
