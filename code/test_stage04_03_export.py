"""
test_stage04_03_export.py -- what a 04_03 run leaves in the draw folder.

    .venv/bin/python code/test_stage04_03_export.py

RUN THIS AFTER ANY CHANGE to `src/traction_export.py`. It calls `_write` for
real against a small synthetic frame and then LISTS THE FILES, the same
question `test_stage04_02_export.py` asks and for the same reason: checking a
block in isolation tells you the block does what you wrote, listing the folder
tells you the stage does what you meant.

WHY THIS EXISTS
---------------
The export wrote `collected` and `inflow` and no `outflow`, from 2026-09-24
until 2026-09-28. Nothing failed. Downstream, `RAWCLICRecoveryModel` needs all
three -- `outflow - collected` is the mass that left the fleet and never
reached a recycler -- so its `account()` returned None and FOUR figures were
silently not drawn for every traction motor case: account, losses, trapped,
fate. The battery and the electronics cases had them all along. It surfaced
only because somebody asked where copper's account figure was.

The tracker has no `outflow` row. It keys each end-of-life vehicle by where it
went, so the outflow is `collected + export + unknown_whereabouts`, and getting
that sum wrong understates what left the fleet -- which flatters the collection
rate without anything failing. So the third check is arithmetic, not existence.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import traction_export as te  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"\n          {detail}"
                                                     if not ok and detail else ""))


def frame_for(years, flows, per_flow) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per (region, flow, year), and the draw product that goes with it."""
    index = pd.MultiIndex.from_tuples(
        [("EUR", flow, int(year)) for flow in flows for year in years],
        names=["Region", "flow", "scrap_year"])
    frame = pd.DataFrame(np.zeros((len(index), 1)), index=index)
    # (rows, draws): every draw of a row carries that flow's own value, so the
    # sum is checkable by eye.
    product = np.vstack([np.full((1, 4), per_flow[flow] * te.KG_PER_KT)
                         for flow in flows for _ in years])
    return frame, product


def main() -> int:
    years = np.array([2030, 2050], dtype=int)
    tracker_flows = ("collected", "export", "inflow", "unknown_whereabouts")
    per_flow = {"collected": 88.0, "export": 2.0,
                "unknown_whereabouts": 10.0, "inflow": 130.0}
    frame, product = frame_for(years, tracker_flows, per_flow)

    root = Path(tempfile.mkdtemp(prefix="traction-export-test-"))
    try:
        written = te._write(root, "mix", te.EXPORTED_FLOWS, frame, years,
                            product, "__component____copper")
        folders = sorted(p.name for p in (root / "mix").iterdir() if p.is_dir())

        # 1. All three flows the recovery model reads, and no tracker-only name.
        check("the export writes collected, inflow AND outflow",
              folders == ["collected", "inflow", "outflow"],
              f"wrote {folders}")
        check("no tracker-only flow leaks into the export",
              not ({"export", "unknown_whereabouts"} & set(folders)),
              f"wrote {folders}")

        # 2. Every flow carries the same array names, which is what lets the
        #    reader ask one flow for a name it found in another.
        names = {f: sorted(p.name for p in (root / "mix" / f).glob("*.npy"))
                 for f in folders}
        check("every flow holds the same arrays",
              len(set(map(tuple, names.values()))) == 1,
              f"{names}")

        # 3. THE ARITHMETIC. outflow is the three destinations added, in kt.
        out = np.load(root / "mix" / "outflow" / "__component____copper.npy")
        coll = np.load(root / "mix" / "collected" / "__component____copper.npy")
        want = per_flow["collected"] + per_flow["export"] + per_flow["unknown_whereabouts"]
        check("outflow is collected + export + unknown_whereabouts",
              np.allclose(out, want), f"got {out.flat[0]}, wanted {want}")
        check("collected is unchanged by the composition",
              np.allclose(coll, per_flow["collected"]),
              f"got {coll.flat[0]}, wanted {per_flow['collected']}")
        check("outflow is larger than collected, as it must be",
              float(out.mean()) > float(coll.mean()))

        # 4. Shape and unit are the reader's contract: (draws, years) in kt.
        check("shape is (draws, years)", out.shape == (4, len(years)),
              f"got {out.shape}")
        check("the manifest names every array written",
              len(written) == len(folders),
              f"{len(written)} entries for {len(folders)} flows")

        # 5. A composed flow is all of its parts or none. Half of them would be
        #    an outflow that is quietly too small.
        partial, partial_product = frame_for(
            years, ("collected", "inflow"), per_flow)
        try:
            te._parts_of(partial, "outflow", years)
            refused = False
        except ValueError as error:
            refused = "unknown_whereabouts" in str(error)
        check("a tracker missing a part refuses to compose outflow", refused,
              "it composed one from the parts that happened to be there")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failed = [name for ok, name, _ in _results if not ok]
    print(f"\n  {len(_results) - len(failed)}/{len(_results)} passed")
    if failed:
        print("\n  Do not commit a change that leaves these failing: each one\n"
              "  costs a full re-run of the stage to discover any other way.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
