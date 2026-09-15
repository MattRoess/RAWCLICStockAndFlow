"""
test_04_04_figures.py
=====================

A BENCH TOOL, NOT A PIPELINE STAGE. Redraws 04_04's figures from the draws
already on disk, without rerunning the stage.

    .venv/bin/python code/test_04_04_figures.py

Named `test_` and not `04_04_` on purpose: the numeric prefix belongs to the
stages that produce results, and this produces none. It is here because a figure
gets adjusted far more often than a result gets recomputed -- thirty seconds
against the stage's half hour.

WHAT IT IS NOT. It asserts nothing and checks nothing, unlike
`test_stage03_inflow.py` and `test_stage04_02_export.py` beside it. Every figure
lives in `04_04_batteries.py` and is drawn at the end of every run of it; this
file holds no plotting code of its own and cannot drift from the stage, because
it calls the stage's own `build_all`.
"""

from __future__ import annotations

import importlib.util
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

from src.artifacts import load_many  # noqa: E402


def stage() -> object:
    """04_04_batteries.py, imported by path -- its name is not an identifier."""
    path = PROJECT_ROOT / "code" / "04_04_batteries.py"
    spec = importlib.util.spec_from_file_location("stage_04_04", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> dict:
    loaded = load_many("params", "battery_material_flows", "battery_chemistry_gaps",
                       root=PROJECT_ROOT)
    return {"figures": stage().build_all(loaded["params"],
                                         loaded["battery_material_flows"],
                                         loaded["battery_chemistry_gaps"])}


if __name__ == "__main__":
    main()
