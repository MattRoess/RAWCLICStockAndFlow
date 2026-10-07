"""
config.py
=========

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Small, self-contained project-path resolver. This is what `get_paths()` (imported by
`artifacts.py`) actually is.

1. `get_paths()` calls `artifacts.mkdir(parents=True, exist_ok=True)` on every call --
   the artifacts directory is always created before anything writes to it (this resolved
   the earlier-flagged finding L10, since retracted).

FIX LOG (this round -- see EVmodel_review_consolidated.md, Fix Log, for the full record)
------------------------------------------------------------------------------------------
- [FIXED] `find_project_root` previously fell back to the current working directory with
  no warning if it couldn't find a real project root. It now emits a `warnings.warn(...)`
  in that branch -- same return value as before (nothing that imports this module needs
  to change), but a misconfigured environment now fails LOUDLY instead of silently
  reading/writing from the wrong place. See finding H3.
- [FIXED, corrected this round] The root-detection marker was hardcoded to a directory
  literally named `notebooks`. This project does not use Jupyter notebooks -- the
  working directory of `.py` scripts is called `code/`. `find_project_root` now looks
  for `code/` (alongside `data/`) as its marker. An earlier version of this fix
  mistakenly used "scripts" instead of the actual folder name -- corrected here.

NOTE ON THIS FILE otherwise: no other logic changed, only comments added/updated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

# The directory name that counts as "this is the project's code folder" for project-root
# detection.
_SCRIPT_DIR_NAME = "code"


@dataclass(frozen=True)
class Paths:
    """Resolved, immutable set of project directories."""
    project_root: Path
    data_raw: Path
    data_processed: Path
    artifacts: Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upward from `start` (default: current working directory) looking for the first
    ancestor directory that contains BOTH a `data/` subdirectory AND a `code/`
    subdirectory -- used as a heuristic fingerprint for "this is the EVmodel project
    root."

    [FIXED, this round]: if no ancestor matches, this now emits a `warnings.warn(...)`
    before falling back to `current` (the resolved `start`/cwd) -- previously this was
    silent. The return value in the fallback case is unchanged (still `current`), so no
    caller needs to change; the only difference is a misconfigured environment (a fresh
    checkout missing `data/`, a CI runner, a typo'd working directory, ...) now makes
    itself visible instead of quietly resolving `data_raw`/`data_processed`/`artifacts`
    to the wrong place.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / _SCRIPT_DIR_NAME).exists() and (candidate / "data").exists():
            return candidate
    warnings.warn(
        f"find_project_root: no ancestor of {current} contains both a 'data/' folder "
        f"and a '{_SCRIPT_DIR_NAME}/' folder -- falling back to {current} as the "
        f"project root. Paths derived from this (data_raw, data_processed, artifacts) "
        f"may be wrong. Pass an explicit `start=` to get_paths()/find_project_root(), "
        f"or run from within the real project tree, to avoid this.",
        stacklevel=2,
    )
    return current


def get_paths(start: Path | None = None) -> Paths:
    """
    Resolve and return the project's key directories, creating the artifacts directory
    if it doesn't exist yet. Called by every `artifacts.py` save/load/status operation
    (see module docstring, point 1).
    """
    root = find_project_root(start)
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    artifacts = data_processed / "intermediate"
    artifacts.mkdir(parents=True, exist_ok=True)
    # NOT changed (finding L16, deliberately left as-is): this mkdir call runs on EVERY
    # invocation of get_paths() -- i.e. every single artifact save/load/status call, not
    # just once per session. Explicitly not "fixing" this: the cost is a single
    # stat+mkdir-if-needed syscall, negligible even called thousands of times, and adding
    # e.g. a module-level cache/flag to skip it after the first call would add real
    # complexity (cache invalidation if the directory is deleted mid-run, thread-safety)
    # for a saving too small to matter. Recorded here so it's a documented decision, not
    # a silently-skipped item.
    return Paths(
        project_root=root,
        data_raw=data_raw,
        data_processed=data_processed,
        artifacts=artifacts,
    )
