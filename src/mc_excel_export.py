"""
mc_excel_export.py
======================

Exports the Monte Carlo period-summary artifacts from stages 02, 03_01, and 03_02
(`mc_stage02_period_summary`, `mc_stage03_01_period_summary`, `mc_stage03_02_summary`)
into ONE Excel workbook:

  - A single "Summary" sheet: one row per (stage, scope, period, [year,] metric)
    with n/mean/median/mode/std/P2.5/P97.5/min/max, plus a "histogram_sheet" column
    naming which sheet holds that row's full 50-bin histogram.
  - A single "Sensitivity" sheet (if sensitivity data is provided): Spearman rank
    correlation between each uncertain input and its stage's headline output metric,
    one row per (stage, [scenario,] parameter) -- see `flatten_sensitivity_results`.
    Small tables; no need for a per-entry sheet the way histograms need.
  - One sheet PER histogram (bin edges + frequency), named sequentially (H00001,
    H00002, ...) since the natural descriptive name (e.g. "BAU__2020-2030__BEV__A__
    cumulative_collected") is both far longer than Excel's 31-character sheet-name
    limit and can contain characters Excel forbids in sheet names. Each histogram
    sheet also repeats its own identifying tags as a small header block at the top,
    so it's self-describing even without cross-referencing the Summary sheet.

SCALE WARNING, read before running against real (200,000-draw) data: the number of
histogram sheets is (stages) x (periods) x (scopes: EU_total + drivetrains [+
segments for 03_02]) x (metrics) [+ one sheet per YEAR for any period with
stock_per_year data]. This can reach the thousands very quickly, especially from
03_02's per-segment breakdown combined with more than one requested
`output_periods` entry. Each additional sheet costs real write time and file size --
this module prints a running progress count (every 200 sheets) and, before writing
anything, prints the TOTAL histogram-sheet count it's about to produce, so you can
Ctrl-C and narrow `output_periods` / scope first if the number looks unreasonable.

USAGE (see also the runnable script `export_mc_summaries_excel.py`):
    from src.mc_excel_export import (
        flatten_nested_period_summary, flatten_flat_scenario_summary, export_mc_summaries_to_excel,
    )
    records = []
    records += flatten_nested_period_summary(mc_stage02_period_summary, stage="02_stockdriven")
    records += flatten_nested_period_summary(mc_stage03_01_period_summary, stage="03_01_flowdriven")
    records += flatten_flat_scenario_summary(mc_stage03_02_summary, stage="03_02_adjustedflows")
    export_mc_summaries_to_excel(records, output_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Font


def flatten_nested_period_summary(period_summary: dict, *, stage: str) -> list[dict[str, Any]]:
    """
    Flattens stage 02's / stage 03_01's period-summary shape:
        {"by_drivetrain": {drv: {(start,end): {metric: summarize_distribution()_dict, ...
                                                 "stock_per_year": {year: summarize_distribution()_dict, ...}}}},
         "eu_total": {(start,end): {metric: ..., "stock_per_year": {...}}}}
    into a flat list of records, one per (scope, period, metric) or
    (scope, period, "stock_per_year", year).
    """
    records: list[dict[str, Any]] = []

    def _emit(scope: str, period: tuple[int, int], metric_dict: dict) -> None:
        for metric, value in metric_dict.items():
            if metric == "stock_per_year":
                for year, summ in value.items():
                    records.append({
                        "stage": stage, "scope": scope, "period": f"{period[0]}-{period[1]}",
                        "year": int(year), "metric": "stock_per_year", "summary": summ,
                    })
            else:
                records.append({
                    "stage": stage, "scope": scope, "period": f"{period[0]}-{period[1]}",
                    "year": None, "metric": metric, "summary": value,
                })

    for drv, by_period in period_summary.get("by_drivetrain", {}).items():
        for period, metric_dict in by_period.items():
            _emit(drv, period, metric_dict)
    for period, metric_dict in period_summary.get("eu_total", {}).items():
        _emit("EU_total", period, metric_dict)

    return records


def flatten_flat_scenario_summary(summary_mc: dict, *, stage: str) -> list[dict[str, Any]]:
    """
    Flattens stage 03_02's flat, string-keyed summary shape:
        {"{scenario}__{start}-{end}__{scope}[__{segment}]__{metric}": summarize_distribution()_dict,
         "{scenario}__{start}-{end}__{scope}[__{segment}]__stock_year_{year}": summarize_distribution()_dict}
    into the same flat record shape `flatten_nested_period_summary` produces, by
    parsing the "__"-delimited key.
    """
    records: list[dict[str, Any]] = []
    for key, summ in summary_mc.items():
        if "_by_year_band" in key:
            continue  # a different (non-summarize_distribution-shaped) artifact; not a histogram record
        parts = key.split("__")
        if len(parts) == 4:
            scenario, period, scope, metric = parts
            segment = None
        elif len(parts) == 5:
            scenario, period, scope, segment, metric = parts
        else:
            continue  # unrecognized key shape -- skip rather than guess
        year = None
        if metric.startswith("stock_year_"):
            year = int(metric.replace("stock_year_", ""))
            metric = "stock_per_year"
        scope_label = scope if segment is None else f"{scope}__{segment}"
        records.append({
            "stage": stage, "scenario": scenario, "scope": scope_label, "period": period,
            "year": year, "metric": metric, "summary": summ,
        })
    return records


def _sanitize_header_value(v: Any) -> str:
    return "" if v is None else str(v)


def flatten_sensitivity_results(sensitivity_data, *, stage: str) -> "pd.DataFrame":
    """
    Normalizes a stage's sensitivity_correlations() output into one flat DataFrame
    with a "stage" (and, for stage 03_02, "scenario") tag column.

    `sensitivity_data` is either:
      - a single DataFrame (stage 02 / stage 03_01: one sensitivity table for the
        whole stage), or
      - a dict {scenario_name: DataFrame} (stage 03_02: one table per scenario).
    """
    if isinstance(sensitivity_data, dict):
        frames = []
        for scenario_name, df in sensitivity_data.items():
            df = df.copy()
            df.insert(0, "scenario", scenario_name)
            df.insert(0, "stage", stage)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = sensitivity_data.copy()
        df.insert(0, "stage", stage)
        return df


def export_mc_summaries_to_excel(
    records: list[dict[str, Any]], output_path: Path, *,
    sensitivity_df: "pd.DataFrame | None" = None, progress_every: int = 200,
) -> Path:
    """
    Writes the "Summary" sheet + one histogram sheet per record, and (if provided) a
    compact "Sensitivity" sheet (Spearman rank correlations -- small tables, no need
    for the one-sheet-per-histogram treatment). Prints the total histogram-sheet
    count before starting (see module docstring's SCALE WARNING) and progress every
    `progress_every` sheets.
    """
    output_path = Path(output_path)
    n_records = len(records)
    print(f"mc_excel_export: about to write {n_records:,} histogram sheets (+ 1 Summary sheet) to {output_path}")

    tag_cols = sorted({k for r in records for k in r.keys() if k != "summary"})
    stat_cols = ["n", "mean", "median", "mode", "std", "p2_5", "p97_5", "min", "max"]

    summary_rows = []
    sheet_names = []
    for i, rec in enumerate(records, start=1):
        sheet_name = f"H{i:05d}"
        sheet_names.append(sheet_name)
        row = {col: rec.get(col) for col in tag_cols}
        for col in stat_cols:
            row[col] = rec["summary"].get(col)
        row["histogram_sheet"] = sheet_name
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows, columns=tag_cols + stat_cols + ["histogram_sheet"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ws = writer.sheets["Summary"]
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col_cells in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells) + 2
            ws.column_dimensions[col_cells[0].column_letter].width = min(width, 40)

        if sensitivity_df is not None and not sensitivity_df.empty:
            sensitivity_df.to_excel(writer, sheet_name="Sensitivity", index=False)
            ws_s = writer.sheets["Sensitivity"]
            for cell in ws_s[1]:
                cell.font = Font(bold=True)
            for col_cells in ws_s.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells) + 2
                ws_s.column_dimensions[col_cells[0].column_letter].width = min(width, 40)

        for i, (rec, sheet_name) in enumerate(zip(records, sheet_names), start=1):
            summ = rec["summary"]
            edges = summ["bin_edges"]
            freqs = summ["frequencies"]
            bin_centers = [(edges[j] + edges[j + 1]) / 2 for j in range(len(freqs))]
            hist_df = pd.DataFrame({
                "bin_low": edges[:-1], "bin_high": edges[1:], "bin_center": bin_centers, "frequency": freqs,
            })

            header_rows = [[col, _sanitize_header_value(rec.get(col))] for col in tag_cols]
            header_rows += [[k, summ.get(k)] for k in stat_cols]
            header_df = pd.DataFrame(header_rows, columns=["field", "value"])

            header_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            hist_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(header_rows) + 2)

            ws_h = writer.sheets[sheet_name]
            for cell in ws_h[1]:
                cell.font = Font(bold=True)
            header_row_of_table = len(header_rows) + 3
            for cell in ws_h[header_row_of_table]:
                cell.font = Font(bold=True)

            if progress_every and i % progress_every == 0:
                print(f"mc_excel_export: {i:,}/{n_records:,} histogram sheets written")

    print(f"mc_excel_export: done -- {n_records:,} histogram sheets + Summary sheet -> {output_path}")
    return output_path
