"""
04_04_figures.py
================

Redraws the battery figures from the saved draws, without rerunning the stage.

    .venv/bin/python code/04_04_figures.py

The figures themselves live in `04_04_batteries.py`, which draws them at the end
of every run. This is for changing one, which happens far more often than the
result changes.
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
