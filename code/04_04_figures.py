"""
04_04_figures.py
================

Redraws the battery figures from the saved draws, without rerunning the stage.

    .venv/bin/python code/04_04_figures.py

`04_04_batteries.py` draws them itself at the end of every run. This is for
changing a figure, which happens far more often than the result changes.
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

from src.artifacts import load_many  # noqa: E402
from src.battery_figures import build_all  # noqa: E402


def main() -> dict:
    loaded = load_many("params", "battery_material_flows", "battery_chemistry_gaps",
                       root=PROJECT_ROOT)
    return {"figures": build_all(loaded["params"],
                                 loaded["battery_material_flows"],
                                 loaded["battery_chemistry_gaps"])}


if __name__ == "__main__":
    main()
