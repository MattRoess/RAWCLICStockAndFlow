"""
00_parameters.py
=================

Builds, validates, and persists the central `Params` object for the EVmodel pipeline.

STRUCTURE
----------
This file is intentionally thin. Every parameter dataclass (`Params`, `DataPrepParams`,
`StockFlowParams`, ...) lives in `src/params_schema.py`, NOT here.

WHY: this script is always run directly (`python 00_parameters.py`), which Python
executes as module `__main__`. If the dataclasses were defined here, the pickled
`params` artifact would record its classes as `__main__.Params` etc. -- resolvable only
from a process where THIS file happens to be `__main__`. Any other stage script
unpickling that artifact (also run directly, also its own `__main__`) would fail with
`AttributeError: Can't get attribute 'Params' on <module '__main__'...>`. Confirmed by
an actual end-to-end run this round: 00_parameters.py -> pickle -> 01_data_prep.py
failed exactly this way before this file was split. Defining the schema in
`src/params_schema.py` (a module that is only ever imported, never executed as a
script) fixes this structurally -- pickle then records `src.params_schema.Params`,
resolvable identically from every stage, since every stage already puts `src/` on
`sys.path`. See `src/params_schema.py`'s own docstring for the full explanation.

Access pattern for downstream stages: `params.data_prep.scenario`,
`params.stock_flow.export_share_by_drv`, etc. -- NOT the old
`PARAMS["01_data_prep"]["scenario"]` dict-key style.

WHAT'S STILL A PLACEHOLDER / OPEN (see EVmodel_review_consolidated.md for full history)
------------------------------------------------------------------------------------------
- `stock_flow.export_share_by_drv["FCEV"]` / `["Gases"]`: still `0.08`, matching every
  other non-BEV drivetrain -- not a verified real export share.
- `data_prep.accelerating_year = 2026`: no cited source/derivation (L2).
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent

import pandas as pd

from src.params_schema import Params  # type: ignore

try:
    from src.artifacts import save_many, artifact_status  # type: ignore
    _ARTIFACTS_AVAILABLE = True
except ImportError:
    _ARTIFACTS_AVAILABLE = False


PARAMS: Params = Params()


def validate_params(params: Params) -> list[str]:
    """Kept for callers still using the old function name."""
    return params.validate()


# ---------------------------------------------------------------------------
# Excel export (human-readable parameter register)
# ---------------------------------------------------------------------------
_DESCRIPTIONS: dict[str, str] = {
    "scenario": "Selected model scenario for stock-flow simulation.",
    "scenario_list": "All scenarios available for selection.",
    "remind_scenario_files": "Per-scenario (relative .mif path, CSV separator).",
    "start_year_model": "First year of the simulation horizon.",
    "end_year_model": "Last year of the simulation horizon.",
    "start_year_plotting": "First year shown in default plots.",
    "end_year_plotting": "Last year shown in default plots.",
    "threshold": "Numerical tolerance used for stock-flow stability checks.",
    "eu_countries": "EU country list used for regional aggregation.",
    "remind_regions": "REMIND model region codes mapped into this analysis.",
    "remind_technology": "Drivetrain categories as labeled in REMIND scenario output.",
    "target_technology": "Drivetrain categories used by this model's outputs.",
    "lifetime_by_drv": "Weibull lifetime parameters (shape_k, scale_lambda) per drivetrain.",
    "lifetime_override_by_drv": "Optional manual override window per drivetrain (None = disabled).",
    "unknown_whereabouts_share": "Share of retired vehicles with unknown/untracked fate, per drivetrain.",
    "export_share_by_drv": "Share of retired vehicles assumed exported, per drivetrain.",
    "segment_map": "Vehicle-segment letter code -> composition dataset code.",
    "drv_prefix_map": "Drivetrain -> composition dataset key prefix.",
    "model_end_year": "Last year simulated in the stock-flow module.",
    "last_exp_data_year": "Last year with observed export data.",
    "export_data_file_name": "Raw EU used-vehicle export/import trade source file.",
    "iso3_corrections": "Source-data ISO3 code quirks mapped to the real ISO 3166-1 alpha-3 code.",
    "export_min_year": "Earliest year kept from the raw export/import trade data.",
    "export_max_year_exclusive": "Exclusive upper-bound year kept from the raw export/import trade data.",
}


def _describe(key: str, section: str) -> str:
    return _DESCRIPTIONS.get(key, f"Model parameter in section '{section}' (no description authored yet).")


def _flatten_value(name: str, value: Any, section: str, prefix: str, rows: list[list[Any]]) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            _flatten_value(f.name, getattr(value, f.name), section, f"{prefix}.{f.name}", rows)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_value(str(k), v, section, f"{prefix}.{k}", rows)
        return
    rows.append([name, _describe(name, section), prefix, json.dumps(value) if isinstance(value, (list, tuple)) else value])


def _flatten_params(params: Params) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for section_name, section_obj in (
        ("01_data_prep", params.data_prep),
        ("02_stock_flow", params.stock_flow),
        ("03_disaggregation", params.disaggregation),
        ("04_materials", params.materials),
        ("06_visualization", params.visualization),
    ):
        for f in fields(section_obj):
            _flatten_value(f.name, getattr(section_obj, f.name), section_name, f"{section_name}.{f.name}", rows)
    return rows


def save_params_excel(params: Params, filepath: Path) -> pd.DataFrame:
    rows = _flatten_params(params)
    df = pd.DataFrame(rows, columns=["name", "description", "key", "value"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="parameters", index=False)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict[str, Any]:
    issues = validate_params(PARAMS)
    if issues:
        print("PARAMS validation found the following issues:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)

    result: dict[str, Any] = {"validation_issues": issues}

    if _ARTIFACTS_AVAILABLE:
        saved = save_many(params=PARAMS, root=PROJECT_ROOT)
        result["saved"] = saved
        print("Saved artifact:", saved)
    else:
        print(
            "src.artifacts not found on path -- skipped persisting the pickled 'params' "
            "artifact. Place this file inside the project (next to the `src/` package) "
            "and re-run to enable that step.",
            file=sys.stderr,
        )

    excel_path = SCRIPT_DIR / "params.xlsx"
    df = save_params_excel(PARAMS, excel_path)
    result["excel_path"] = str(excel_path)
    result["n_parameters"] = len(df)
    print(f"Wrote {len(df)} parameters to {excel_path}")

    return result


if __name__ == "__main__":
    main()
