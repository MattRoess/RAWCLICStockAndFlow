"""
export_mc_summaries_excel.py
================================

Run this AFTER 02_stockdriven.py, 03_01_flowdriven.py, and 03_02_adjustedflows.py
have all been run with `params.monte_carlo.enabled=True` -- loads their three period-
summary artifacts and writes one combined Excel workbook: a flat "Summary" sheet
(mean/median/mode/std/P2.5/P97.5/min/max per row) plus one sheet per histogram
(bin edges + frequency). See `src/mc_excel_export.py`'s module docstring for the
full layout description and a SCALE WARNING worth reading before running this
against your real 200,000-draw data -- the histogram-sheet count is printed before
anything is written, so you can stop and narrow `params.monte_carlo.output_periods`
first if it looks too large.

Usage:
    python code/export_mc_summaries_excel.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.artifacts import load_many  # type: ignore
from src.mc_excel_export import (  # type: ignore
    flatten_nested_period_summary, flatten_flat_scenario_summary,
    flatten_sensitivity_results, export_mc_summaries_to_excel,
)
import pandas as pd


def main() -> Path:
    loaded = {}
    artifact_names = [
        "mc_stage02_period_summary", "mc_stage03_01_period_summary", "mc_stage03_02_summary",
        "mc_stage02_sensitivity", "mc_stage03_01_sensitivity", "mc_stage03_02_sensitivity",
    ]
    for name in artifact_names:
        try:
            loaded[name] = load_many(name, root=PROJECT_ROOT)[name]
        except FileNotFoundError:
            print(f"Skipping {name}: not found -- run its stage with params.monte_carlo.enabled=True first.")

    records: list[dict] = []
    if "mc_stage02_period_summary" in loaded:
        records += flatten_nested_period_summary(loaded["mc_stage02_period_summary"], stage="02_stockdriven")
    if "mc_stage03_01_period_summary" in loaded:
        records += flatten_nested_period_summary(loaded["mc_stage03_01_period_summary"], stage="03_01_flowdriven")
    if "mc_stage03_02_summary" in loaded:
        records += flatten_flat_scenario_summary(loaded["mc_stage03_02_summary"], stage="03_02_adjustedflows")

    sensitivity_frames = []
    if "mc_stage02_sensitivity" in loaded:
        sensitivity_frames.append(flatten_sensitivity_results(loaded["mc_stage02_sensitivity"], stage="02_stockdriven"))
    if "mc_stage03_01_sensitivity" in loaded:
        sensitivity_frames.append(flatten_sensitivity_results(loaded["mc_stage03_01_sensitivity"], stage="03_01_flowdriven"))
    if "mc_stage03_02_sensitivity" in loaded:
        sensitivity_frames.append(flatten_sensitivity_results(loaded["mc_stage03_02_sensitivity"], stage="03_02_adjustedflows"))
    sensitivity_df = pd.concat(sensitivity_frames, ignore_index=True) if sensitivity_frames else None

    if not records:
        raise RuntimeError(
            "No Monte Carlo period-summary artifacts found at all -- run at least one of "
            "02_stockdriven.py / 03_01_flowdriven.py / 03_02_adjustedflows.py with "
            "params.monte_carlo.enabled=True first."
        )

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mc_summaries.xlsx"
    return export_mc_summaries_to_excel(records, output_path, sensitivity_df=sensitivity_df)


if __name__ == "__main__":
    main()
