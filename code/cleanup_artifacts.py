"""
cleanup_artifacts.py -- find artifact files that are wasting disk, and delete only
the ones you name.

WHY THIS REPORTS RATHER THAN DELETES. `data/processed/intermediate/` reached 71 GB,
and roughly 50 GB of that was per-draw arrays that no code path reads. Deleting
multi-GB scientific output automatically is the wrong default: a run that takes an
hour is expensive to repeat, and this project's folder lives under iCloud, which
produces conflict copies that are SOMETIMES THE ONLY INTACT COPY. Two files here are
zero bytes while their "… 2.pkl" twins hold 8.3 GB of real data -- a rule like
"delete anything ending in 2" would have destroyed them.

So this script prints what it finds, with sizes and a reason, and deletes nothing
unless you pass --delete with an explicit category. Even then it refuses to remove a
file whose only sibling is empty.

USAGE

    python code/cleanup_artifacts.py                    # report only
    python code/cleanup_artifacts.py --delete unread    # delete files no code reads
    python code/cleanup_artifacts.py --delete empty     # delete zero-byte files
    python code/cleanup_artifacts.py --delete conflicts # delete iCloud "… 2" copies
    python code/cleanup_artifacts.py --delete all --yes # everything above, no prompt

CATEGORIES

    unread     Written by a stage, with no load found for it. REPORTED ONLY --
               this tool will not delete these. Finding "nothing reads it" by
               static analysis is unreliable here because artifact keys are built
               dynamically (`*[f"tracker_keyed_{n}" for n in names]`), and during
               development this heuristic wrongly cleared three files that ARE
               loaded. Use it as a hint, verify by hand, delete by hand.
    empty      Zero bytes -- a failed or interrupted write.
    conflicts  iCloud conflict copies ("name 2.pkl") whose original exists AND is
               non-empty. A conflict copy whose original is missing or empty is
               reported as `rescue` instead and is never offered for deletion.
    orphans    Not produced by any current stage -- left over from older code.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


ROOT = _find_project_root(Path(__file__).resolve().parent)
INTERMEDIATE = ROOT / "data" / "processed" / "intermediate"
CODE_DIRS = (ROOT / "code", ROOT / "src")

CONFLICT_RE = re.compile(r"^(?P<stem>.+?) (?P<n>\d+)(?P<ext>\.\w+)$")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def registry_keys_by_filename() -> dict[str, str]:
    """
    Map artifact FILENAME -> registry KEY, from `src/artifacts.py`.

    This matters more than it looks. Registered artifacts are loaded by key --
    `load_many("params")` -- and the filename `00_params.pkl` appears in no source
    file at all. Searching for filenames alone concluded that every registered
    input, `00_params.pkl` included, was unread and safe to delete. It would have
    removed the pipeline's own inputs.
    """
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.artifacts import ARTIFACT_FILES  # type: ignore
    except Exception:
        return {}
    return {fname: key for key, fname in ARTIFACT_FILES.items()}


LOAD_FUNCS = {"load_many", "load", "load_artifact", "read_pickle", "np_load"}


def loaded_keys_and_paths() -> tuple[set[str], list[str]]:
    """
    Every artifact key and path literal that the code actually LOADS.

    Parsed with AST rather than matched line by line, because the loads that matter
    are multi-line:

        loaded = load_many(
            "params", "matrices_by_key", "df_exp_eu", ...
        )

    A line-based search sees `"df_exp_eu"` on a line with no loading verb on it and
    concludes the artifact is never read -- which is exactly what happened, and it
    would have marked a required input as safe to delete. The AST sees one call node
    with all its arguments, wherever they sit.

    Returns (keys passed to a load function, other string literals near loads).
    """
    keys: set[str] = set()
    others: list[str] = []
    for d in CODE_DIRS:
        if not d.is_dir():
            continue
        for f in d.rglob("*.py"):
            if "archive" in f.parts or f.name == Path(__file__).name:
                continue
            try:
                tree = ast.parse(f.read_text())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if fname not in LOAD_FUNCS:
                    continue
                # Walk the ENTIRE call subtree. The keys that matter are often
                # nested well below the argument list:
                #
                #     load_many("params", *[f"tracker_keyed_{n}" for n in names])
                #
                # That f-string sits inside a comprehension inside a starred
                # expression. Looking only at direct arguments missed it, and so
                # declared a 47 MB artifact that stage 04_04 genuinely loads to be
                # unread. Walking the subtree catches every literal in the call.
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        keys.add(sub.value)
                    elif isinstance(sub, ast.JoinedStr):
                        lit = "".join(v.value for v in sub.values
                                      if isinstance(v, ast.Constant) and isinstance(v.value, str))
                        if lit:
                            others.append(lit)
    return keys, others


def is_read_somewhere(filename: str, loaded: tuple[set[str], list[str]],
                      registry: dict[str, str] | None = None) -> bool:
    """
    Is this artifact READ by any source file -- not merely written by one?

    The distinction is the whole point. An earlier version asked only "is the name
    mentioned anywhere", which counted the line that WRITES the file and so marked
    50 GB of write-only arrays as worth keeping.

    Two ways an artifact can be read:
      * by registry KEY -- `load_many("params")`, where the filename never appears;
      * by a path built in an f-string, matched on its literal head.

    Deliberately generous in the safe direction: an unrecognised idiom means the file
    is KEPT. A false "read" wastes disk; a false "unread" could delete something the
    pipeline needs.
    """
    keys, path_literals = loaded
    key = (registry or {}).get(filename)
    if key and key in keys:
        return True
    if filename in keys or Path(filename).stem in keys:
        return True
    stem = Path(filename).stem
    for lit in path_literals:
        if lit and (lit in filename or (len(lit) > 8 and stem.startswith(lit))):
            return True
    return False


def classify(path: Path, loaded: tuple[set[str], list[str]],
             by_name: dict[str, Path], registry: dict[str, str]) -> tuple[str, str]:
    size = path.stat().st_size
    if size == 0:
        return "empty", "zero bytes -- a failed or interrupted write"

    m = CONFLICT_RE.match(path.name)
    if m:
        original = by_name.get(f"{m.group('stem')}{m.group('ext')}")
        if original is None:
            return "rescue", "iCloud conflict copy, but NO original exists -- this may be the only copy"
        if original.stat().st_size == 0:
            return "rescue", "iCloud conflict copy whose original is EMPTY -- this is the only intact copy"
        return "conflicts", f"iCloud conflict copy; original exists at {human(original.stat().st_size)}"

    if not is_read_somewhere(path.name, loaded, registry):
        return "unread", "written by a stage, read by nothing"
    return "keep", "read by at least one stage"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", choices=["empty", "conflicts", "all"],
                    help="actually delete this category (default: report only). "
                         "Note 'unread' is NOT offered -- see the module docstring.")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if not INTERMEDIATE.is_dir():
        print(f"nothing to scan: {INTERMEDIATE} does not exist")
        return 0

    loaded = loaded_keys_and_paths()
    registry = registry_keys_by_filename()
    files = sorted(p for p in INTERMEDIATE.iterdir() if p.is_file())
    by_name = {p.name: p for p in files}

    groups: dict[str, list[tuple[Path, int, str]]] = {}
    for p in files:
        cat, why = classify(p, loaded, by_name, registry)
        groups.setdefault(cat, []).append((p, p.stat().st_size, why))

    total = sum(sz for g in groups.values() for _, sz, _ in g)
    print(f"{INTERMEDIATE}")
    print(f"{len(files)} files, {human(total)} total\n")

    order = ["rescue", "empty", "conflicts", "unread", "keep"]
    for cat in order:
        items = groups.get(cat)
        if not items:
            continue
        items.sort(key=lambda t: -t[1])
        cat_size = sum(sz for _, sz, _ in items)
        banner = {
            "rescue": "DO NOT DELETE -- may be the only intact copy",
            "empty": "zero-byte files",
            "conflicts": "iCloud conflict copies (original exists and is intact)",
            "unread": "written by a stage, read by nothing",
            "keep": "read by at least one stage",
        }[cat]
        print(f"--- {cat.upper()}: {len(items)} files, {human(cat_size)} — {banner}")
        for p, sz, why in items[:12]:
            print(f"      {human(sz):>10}  {p.name}")
            if cat in ("rescue", "empty"):
                print(f"                  {why}")
        if len(items) > 12:
            print(f"      ... and {len(items) - 12} more")
        print()

    if not args.delete:
        safe = sum(sz for cat in ("empty", "conflicts")
                   for _, sz, _ in groups.get(cat, []))
        unread = sum(sz for _, sz, _ in groups.get("unread", []))
        print(f"Mechanically safe to reclaim (empty + conflicts): {human(safe)}")
        print(f"Reported as unread, NOT deletable by this tool:   {human(unread)}")
        print()
        print("Nothing was deleted. --delete accepts only 'empty' and 'conflicts'.")
        print("UNREAD is a heuristic and deliberately cannot be deleted here: this")
        print("codebase builds artifact keys dynamically, and three separate times")
        print("during development the heuristic called a genuinely-loaded file unread.")
        print("Stop 04_01 writing the large ones instead, with")
        print("materials.persist_mc_mass_draws = False (already the default), and")
        print("remove existing ones by hand once you have checked them.")
        if groups.get("rescue"):
            print(f"\nNOTE: {len(groups['rescue'])} file(s) are in RESCUE and are never "
                  f"offered for deletion.\n      Their originals are missing or empty, so "
                  f"the conflict copy holds the only data.\n      Rename them back "
                  f"before cleaning up, or they will be the next thing lost.")
        return 0

    cats = ["empty", "conflicts"] if args.delete == "all" else [args.delete]
    targets = [(p, sz) for c in cats for p, sz, _ in groups.get(c, [])]
    if not targets:
        print(f"nothing to delete in category '{args.delete}'")
        return 0

    freed = sum(sz for _, sz in targets)
    print(f"About to delete {len(targets)} files, freeing {human(freed)}.")
    if not args.yes:
        try:
            if input("Type 'delete' to confirm: ").strip().lower() != "delete":
                print("aborted, nothing deleted")
                return 1
        except EOFError:
            print("aborted (no terminal to confirm on), nothing deleted")
            return 1

    done = 0
    for p, sz in targets:
        try:
            p.unlink()
            done += 1
        except OSError as exc:
            print(f"  could not delete {p.name}: {exc}")
    print(f"deleted {done} files, freed {human(freed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
