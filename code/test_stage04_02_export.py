"""
test_stage04_02_export.py -- what a 04_02 run actually leaves in the draw folder.

    .venv/bin/python code/test_stage04_02_export.py

RUN THIS AFTER ANY CHANGE to the export block in `04_02_BEVelectronics.py`. It
calls `element_flows` for real, with `export` set, against the committed element
files at 50 draws, and then LISTS THE FILES. About a second, and it needs no
pipeline run.

WHY THIS EXISTS
---------------
On 2026-09-01 the alloy export was added and the element export was not removed,
so a run wrote `fealloy__Motors` AND `Mn__esteel__Motors` AND
`Sr__magnet__Motors` -- the same mass twice, under two names, which is the exact
double-count the alloy export exists to prevent.

It was not caught because the new block was checked in isolation and the only
question that mattered was never asked: WHAT FILES DOES A RUN PRODUCE? That cost
two full runs of the stage. Every test here asks that question and nothing else.

Checking a block in isolation tells you the block does what you wrote. Listing
the folder tells you the stage does what you meant.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGE = ROOT / "code" / "04_02_BEVelectronics.py"


def _element_dir() -> Path:
    """
    Where the electronics models wrote their element draws.

    Taken from the same setting the stage reads, rather than hard-coded here, so
    moving that repository moves this test with it. The setting is relative to
    `data/`, which is where the stage resolves it from.
    """
    try:
        from src.params_schema import Params                          # noqa: E402
        stated = Params().materials.bev_electronics_element_draws_dir
        resolved = (ROOT / "data" / stated).resolve()
        if resolved.is_dir():
            return resolved
    except Exception:
        pass
    return (ROOT.parent / "RAWCLICVehicleElectronics" / "Composition"
            / "element_draws").resolve()


ELEMENTS = _element_dir()

GROUPS = ("AB", "CD", "EF")
SEGMENTS = ("A", "B", "C", "D", "E", "F")
SEG_GROUP = {"A": "AB", "B": "AB", "C": "CD", "D": "CD", "E": "EF", "F": "EF"}
YEARS = np.arange(2020, 2026)
DRAWS = 50
FLOW = "collected"

# Deliberately including the four that were wrongly exported. A test that asks
# only for Cu and Au would have passed against the broken version.
ASK_FOR = {"Cu": ["Wiring", "Motors", "PCB", "Sensors"],
           "Au": ["PCB", "Sensors"],
           "Nd": ["Sensors"],
           "Sr__magnet": ["Motors"],
           "Mn__esteel": ["Motors"],
           "Mn__cfsteel": ["Motors"],
           "Al__bulk": ["Motors"]}

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((ok, name, detail))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail and not ok else ""))


def stage():
    spec = importlib.util.spec_from_file_location("s04_02", STAGE)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:                 # the stage guards its own __main__
        pass
    return module


def written(module) -> set[str]:
    """Run the export for real and return the stems it wrote."""
    def names(path: Path) -> list[str]:
        return [x for x in path.read_text().split("\n") if x.strip()]

    elem: dict = {"Motors": {}, "PCB": {}, "Sensors": {}}
    for group in GROUPS:
        for domain, stem, kind in (("Motors", f"motors_{group}", "fractions"),
                                   ("PCB", f"pcb_{group}", "fractions"),
                                   ("Sensors", f"sensors_{group}", "mass_mg")):
            listing = ELEMENTS / (f"{stem}_mass_elements.txt" if kind == "mass_mg"
                                  else f"{stem}_elements.txt")
            elem[domain][group] = (names(listing),
                                   np.load(ELEMENTS / f"{stem}_{kind}.npy", mmap_mode="r"))

    rng = np.random.default_rng(0)
    fleet = {(s, FLOW): rng.lognormal(np.log(1.0), 0.1, (DRAWS, len(YEARS)))
             for s in SEGMENTS}
    per_segment = {d: {s: rng.lognormal(np.log(500.0), 0.1, (DRAWS, len(YEARS)))
                       for s in SEGMENTS} for d in module.ELEMENT_DOMAINS}

    out = Path(tempfile.mkdtemp())
    try:
        module.element_flows(fleet, per_segment, elem, slice(0, len(YEARS)), FLOW,
                             SEGMENTS, SEG_GROUP, list(ASK_FOR), dict(ASK_FOR),
                             YEARS, DRAWS, export=(out, np.array([0, 1])))
        return {p.name[:-4] for p in (out / FLOW).glob("*.npy")}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main() -> int:
    print(f"\n{STAGE.name}: what one run leaves in the folder\n")
    if not ELEMENTS.is_dir():
        print(f"  element draws not found at {ELEMENTS} -- cannot run")
        return 1

    module = stage()
    files = written(module)
    alloys = set(module.ALLOY_OF.values())
    print(f"  {len(files)} files\n")
    for name in sorted(files):
        print(f"    {name}")
    print()

    # 1. The metals case's inputs exist. Without these it cannot be built at all.
    needed = {"__domain____Motors", "__domain____Wiring", "copper__Wiring",
              "copper__Motors", "alalloy__Motors", "fealloy__Motors"}
    check("an alloy domain writes its domain mass and its alloys",
          needed <= files, f"missing {sorted(needed - files)}")

    # 2. THE ONE THAT WAS MISSED. An alloy domain must write NOTHING else --
    #    `fealloy__Motors` already holds every gram of `Mn__esteel__Motors`.
    doubled = {f for f in files
               if f.rpartition("__")[2] in module.ALLOY_DOMAINS
               and f.rpartition("__")[0] not in ({"__domain__"} | alloys)}
    check("an alloy domain writes no element files",
          not doubled, f"the same mass twice, under: {sorted(doubled)}")

    # 3. Nor under a `__total` name, which consumers skip but a reader does not.
    alloy_only = {e for e, doms in ASK_FOR.items() if set(doms) <= set(module.ALLOY_DOMAINS)}
    leaked = {f"{e}__total" for e in alloy_only} & files
    check("an element that lives only in an alloy writes no total either",
          not leaked, f"still written: {sorted(leaked)}")

    # 4. And the domains that DO separate elements must be untouched by all of it.
    for domain in (d for d in module.ELEMENT_DOMAINS if d not in module.ALLOY_DOMAINS):
        have = {f.rpartition("__")[0] for f in files if f.rpartition("__")[2] == domain}
        want = {e for e, doms in ASK_FOR.items() if domain in doms}
        check(f"{domain} still exports every element asked for",
              want <= have, f"missing {sorted(want - have)}")

    failed = [name for ok, name, _ in _results if not ok]
    print(f"\n  {len(_results) - len(failed)}/{len(_results)} passed")
    if failed:
        print("\n  Do not commit a change that leaves these failing: each one costs a\n"
              "  full re-run of the stage to discover any other way.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
