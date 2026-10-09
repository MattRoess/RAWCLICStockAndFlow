"""
materials.py
=============

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

One helper the stage-04 stages share: `save_unregistered_scenario_outputs`.

[2026-10-09] This module used to be the quantification library the stage-04 notebooks
were once going to share: `quantify_elements_from_tracker`, `build_mass_by_year_elem_dict`,
`combine_materials_by_flow` and `combine_mass_by_flow`, with the `"m-c"` / `"e-m"`
composition levels they chose between. No stage called any of them -- each reimplements
its own aggregation (the DRY finding M37 said so) -- and they were removed as dead code.
`git log` has them.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_unregistered_scenario_outputs(artifacts_dir: Path, outputs: dict[str, Any]) -> dict[str, Path]:
    """
    [NEW] Shared persistence helper for stage-04's per-scenario outputs, which have no
    corresponding entry in `artifacts.py`'s static `ARTIFACT_FILES` registry (adding one
    entry per scenario x output-type combination there isn't practical -- it would need
    to grow every time a new scenario is added, in a file whose whole purpose is being
    a stable, hand-maintained registry). Replaces THREE near-identical raw-pickle-saving
    loops duplicated across `04_01_materials.py`, `04_03_tractionmotors.py`, and
    `04_04_batteries.py` with one shared implementation.

    `outputs`: {filename (without directory) -> object to pickle}.
    Returns {filename -> full Path}, for printing/logging at the call site.

    NOTE, unchanged from the original design: these files are genuinely invisible to
    `artifact_status()` -- this helper doesn't change that, only removes the
    code duplication in how they're written. If you want per-scenario stage-04 outputs
    tracked by `artifact_status()`, that requires a design change to `artifacts.py`
    itself (e.g. a wildcard/prefix-based registry), not just this helper -- flagging
    this as a real option to consider, not implementing it unasked.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for filename, obj in outputs.items():
        path = artifacts_dir / filename
        with path.open("wb") as handle:
            pickle.dump(obj, handle)
        saved[filename] = path
    return saved

