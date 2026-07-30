"""
inspect_mc.py
=============

Read-only browser for the Monte Carlo `.pkl` artifacts.

WHY THIS EXISTS: the point of wanting an Excel export was never Excel -- it was not
having to poke at pickles by hand to see a number. Excel turned out to be the wrong
container for it (the summaries alone need ~7,500 sheets, and the raw 200,000-draw
arrays cannot live in a spreadsheet at all), so this does the same job directly against
the `.pkl` files: no workbook to generate, nothing to go stale, and the raw draws stay
available.

This script NEVER writes to the artifacts -- it only reads. `--csv` writes a separate
file of whatever table you just looked at.

QUICK START
-----------
    python code/inspect_mc.py                       # what exists, and what is in it
    python code/inspect_mc.py 02                    # stage 02 period summary
    python code/inspect_mc.py 02 --drv BEV          # one drivetrain
    python code/inspect_mc.py 03_01                 # stage 03_01 (collected/export/unknown)
    python code/inspect_mc.py 03_02 --scope EU_total
    python code/inspect_mc.py 02 --band             # per-YEAR uncertainty bands
    python code/inspect_mc.py 02 --draws            # raw per-draw arrays
    python code/inspect_mc.py 02 --sens             # sensitivity / tornado ranking
    python code/inspect_mc.py 03_02 --scope BEV --csv bev.csv

Every filter (`--drv`, `--metric`, `--period`, `--scope`, `--scenario`, `--year`) is a
case-insensitive SUBSTRING match, so `--metric export` finds `cumulative_export`, and
`--drv liq` finds `Liquids`. Filters combine.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
INTERMEDIATE = PROJECT_ROOT / "data" / "processed" / "intermediate"

# Which .pkl backs each thing you can ask for, per stage.
ARTIFACTS: dict[str, dict[str, str]] = {
    "02": {
        "periods": "02_mc_period_summary.pkl",
        "bands": "02_mc_summary.pkl",
        "draws": "02_mc_draws.pkl",
        "sens": "02_mc_sensitivity.pkl",
    },
    "03_01": {
        "periods": "03_01_mc_period_summary.pkl",
        "sens": "03_01_mc_sensitivity.pkl",
    },
    "03_02": {
        "flat": "03_02_mc_summary.pkl",
        "sens": "03_02_mc_sensitivity.pkl",
    },
}

# The columns worth seeing; `bin_edges`/`frequencies` are the histogram and are
# deliberately not shown (that is what makes a spreadsheet dump explode).
STAT_COLS = ["mean", "median", "p2_5", "p97_5", "std", "min", "max", "n"]


def load(name: str) -> Any:
    path = INTERMEDIATE / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet.\n"
            f"Run the stage that writes it, with params.monte_carlo.enabled = True."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def is_summary(obj: Any) -> bool:
    return isinstance(obj, dict) and "mean" in obj and "bin_edges" in obj


def keep(value: Any, needle: str | None) -> bool:
    return needle is None or needle.lower() in str(value).lower()


def fmt_period(p: Any) -> str:
    if isinstance(p, tuple) and len(p) == 2:
        return f"{p[0]}-{p[1]}"
    return str(p)


def show(df: pd.DataFrame, args, *, title: str) -> None:
    if df.empty:
        print(f"\n{title}\n  (nothing matched those filters)")
        return
    print(f"\n{title}   [{len(df)} rows]")
    with pd.option_context("display.max_rows", args.max_rows,
                           "display.width", 200,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(df.to_string(index=False))
    if args.csv:
        out = Path(args.csv)
        df.to_csv(out, index=False)
        print(f"\n  written: {out.resolve()}")


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------
def view_periods(stage: str, args) -> None:
    """`{by_drivetrain: {drv: {period: {metric: summary}}}, eu_total: {period: {...}}}`"""
    data = load(ARTIFACTS[stage]["periods"])
    rows = []
    for scope_label, block in (
        ("by_drivetrain", data.get("by_drivetrain", {})),
        ("eu_total", {"EU_total": data.get("eu_total", {})}),
    ):
        for drv, periods in block.items():
            if not keep(drv, args.drv):
                continue
            for period, metrics in periods.items():
                if not keep(fmt_period(period), args.period):
                    continue
                for metric, summary in metrics.items():
                    if not is_summary(summary):
                        continue  # e.g. stock_per_year -- see --year
                    if not keep(metric, args.metric):
                        continue
                    rows.append({
                        "scope": drv, "period": fmt_period(period), "metric": metric,
                        **{c: summary.get(c) for c in STAT_COLS},
                    })
    show(pd.DataFrame(rows), args, title=f"stage {stage} -- period summaries")


def view_stock_per_year(stage: str, args) -> None:
    """`stock_per_year` sits inside the period summaries as {year: summary}."""
    data = load(ARTIFACTS[stage]["periods"])
    rows = []
    for drv, periods in data.get("by_drivetrain", {}).items():
        if not keep(drv, args.drv):
            continue
        for period, metrics in periods.items():
            spy = metrics.get("stock_per_year")
            if not isinstance(spy, dict):
                continue
            for year, summary in sorted(spy.items()):
                if not is_summary(summary) or not keep(year, args.year):
                    continue
                rows.append({
                    "drivetrain": drv, "period": fmt_period(period), "year": year,
                    **{c: summary.get(c) for c in STAT_COLS},
                })
    show(pd.DataFrame(rows), args,
         title=f"stage {stage} -- stock per year (std=0 before the uncertainty cutoff "
               f"is correct, not a bug)")


def view_bands(stage: str, args) -> None:
    """`{drv: {..._band: {years, p2_5, median, p97_5}}}` -- per-year uncertainty bands."""
    data = load(ARTIFACTS[stage]["bands"])
    rows = []
    for drv, block in data.items():
        if not keep(drv, args.drv):
            continue
        for key, band in block.items():
            if not (isinstance(band, dict) and "years" in band):
                continue
            if not keep(key, args.metric):
                continue
            for i, year in enumerate(band["years"]):
                if not keep(year, args.year):
                    continue
                rows.append({
                    "drivetrain": drv, "band": key.replace("_by_year_band", ""),
                    "year": year,
                    "p2_5": band["p2_5"][i], "median": band["median"][i],
                    "p97_5": band["p97_5"][i],
                    "width": band["p97_5"][i] - band["p2_5"][i],
                })
    show(pd.DataFrame(rows), args, title=f"stage {stage} -- per-year bands")


def view_flat(stage: str, args) -> None:
    """03_02: one flat dict keyed `SCENARIO__PERIOD__SCOPE[__SEGMENT]__METRIC`."""
    data = load(ARTIFACTS[stage]["flat"])
    rows = []
    for key, summary in data.items():
        if not is_summary(summary):
            continue
        parts = key.split("__")
        if len(parts) == 4:
            scenario, period, scope, metric = parts
            segment = ""
        elif len(parts) == 5:
            scenario, period, scope, segment, metric = parts
        else:
            scenario, period, scope, segment, metric = key, "", "", "", ""
        if not (keep(scenario, args.scenario) and keep(period, args.period)
                and keep(scope, args.scope) and keep(metric, args.metric)
                and keep(segment, args.segment)):
            continue
        rows.append({
            "scenario": scenario, "period": period, "scope": scope,
            "segment": segment, "metric": metric,
            **{c: summary.get(c) for c in STAT_COLS},
        })
    df = pd.DataFrame(rows)
    if not df.empty and not args.segment and not args.scope:
        # 6,000+ rows unfiltered is not a useful thing to print at someone.
        print(f"\n  note: {len(df)} rows match. Narrow with --scope EU_total, "
              f"--segment A, or --metric export.")
    show(df, args, title=f"stage {stage} -- scenario summaries")


def view_draws(stage: str, args) -> None:
    """Raw per-draw arrays -- the thing a spreadsheet could never hold."""
    data = load(ARTIFACTS[stage]["draws"])
    rows = []
    for drv, block in data.items():
        if not keep(drv, args.drv):
            continue
        for name, arr in block.items():
            if not isinstance(arr, np.ndarray) or not keep(name, args.metric):
                continue
            rows.append({
                "drivetrain": drv, "array": name, "n_draws": arr.size,
                "mean": arr.mean(), "median": float(np.median(arr)),
                "p2_5": float(np.percentile(arr, 2.5)),
                "p97_5": float(np.percentile(arr, 97.5)),
                "std": arr.std(), "min": arr.min(), "max": arr.max(),
            })
    show(pd.DataFrame(rows), args,
         title=f"stage {stage} -- raw per-draw arrays (computed live from the draws)")


def view_sens(stage: str, args) -> None:
    data = load(ARTIFACTS[stage]["sens"])
    frames = []
    if isinstance(data, pd.DataFrame):
        frames.append(("-", data))
    elif isinstance(data, dict):
        for scenario, df in data.items():
            if keep(scenario, args.scenario) and isinstance(df, pd.DataFrame):
                frames.append((scenario, df))
    rows = []
    for scenario, df in frames:
        for _, r in df.iterrows():
            if not keep(r.get("parameter", ""), args.metric):
                continue
            rows.append({"scenario": scenario, **r.to_dict()})
    out = pd.DataFrame(rows)
    if not out.empty and "abs_r" in out:
        out = out.sort_values("abs_r", ascending=False)
    show(out, args,
         title=f"stage {stage} -- sensitivity (Spearman rank corr. with the output)")


def view_overview(args) -> None:
    print("Monte Carlo artifacts in", INTERMEDIATE)
    print()
    for stage, kinds in ARTIFACTS.items():
        print(f"  stage {stage}")
        for kind, fname in kinds.items():
            path = INTERMEDIATE / fname
            if not path.exists():
                print(f"    {kind:<9} {fname:<32} -- MISSING")
                continue
            size = path.stat().st_size / 1e6
            mtime = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            print(f"    {kind:<9} {fname:<32} {size:>7.2f} MB   {mtime}")
    print()
    print("  What you can ask for:")
    for stage, kinds in ARTIFACTS.items():
        opts = []
        if "periods" in kinds:
            opts += ["(default: period summaries)", "--band" if "bands" in kinds else "",
                     "--year", "--draws" if "draws" in kinds else ""]
        if "flat" in kinds:
            opts += ["(default: scenario summaries)"]
        opts += ["--sens"]
        print(f"    {stage:<6} {'  '.join(o for o in opts if o)}")
    print()
    print("  Examples:")
    print("    python code/inspect_mc.py 02 --drv Liquids")
    print("    python code/inspect_mc.py 02 --band --drv BEV --metric inflow_applied")
    print("    python code/inspect_mc.py 03_02 --scope EU_total --metric export")
    print("    python code/inspect_mc.py 02 --sens")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Browse the Monte Carlo .pkl artifacts (read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("stage", nargs="?", choices=sorted(ARTIFACTS),
                    help="02, 03_01 or 03_02. Omit to list what exists.")
    ap.add_argument("--drv", help="drivetrain substring, e.g. BEV / liq")
    ap.add_argument("--metric", help="metric or parameter substring, e.g. export")
    ap.add_argument("--period", help="period substring, e.g. 2030")
    ap.add_argument("--scope", help="03_02 only: EU_total or a drivetrain")
    ap.add_argument("--segment", help="03_02 only: segment code, e.g. A / JB")
    ap.add_argument("--scenario", help="03_02 only: scenario name, e.g. BAU")
    ap.add_argument("--year", help="year substring, for --band / --year views")
    ap.add_argument("--band", action="store_true", help="per-YEAR uncertainty bands")
    ap.add_argument("--draws", action="store_true", help="raw per-draw arrays")
    ap.add_argument("--sens", action="store_true", help="sensitivity ranking")
    ap.add_argument("--stock-per-year", action="store_true",
                    help="per-year stock distribution inside the period summaries")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    ap.add_argument("--max-rows", type=int, default=80,
                    help="rows to print before pandas truncates (default 80)")
    args = ap.parse_args()

    if args.stage is None:
        view_overview(args)
        return 0

    stage = args.stage
    kinds = ARTIFACTS[stage]
    try:
        if args.sens:
            view_sens(stage, args)
        elif args.draws:
            if "draws" not in kinds:
                print(f"stage {stage} has no raw per-draw artifact "
                      f"(only stage 02 persists those).")
                return 1
            view_draws(stage, args)
        elif args.band:
            if "bands" not in kinds:
                print(f"stage {stage} has no per-year band artifact.")
                return 1
            view_bands(stage, args)
        elif args.stock_per_year:
            if "periods" not in kinds:
                print(f"stage {stage} has no period summaries.")
                return 1
            view_stock_per_year(stage, args)
        elif "flat" in kinds:
            view_flat(stage, args)
        else:
            view_periods(stage, args)
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
