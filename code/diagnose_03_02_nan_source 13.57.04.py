"""
diagnose_03_02_nan_source.py
===============================
Follow-up to diagnose_03_02_nan_summary.py, now that we know the failing metric:

    prefix = 'BAU__1975-2070__EU_total'
    metric = 'cumulative_survival'
    -> all 200,000 draws NaN, for the FULL 1975-2070 window, summed across all
       60 (drivetrain, segment) groups.

An EU-wide total is NaN for literally every draw only if AT LEAST ONE of the 60
groups feeding into that sum produced an all-NaN cumulative_survival for every
draw (any NaN + anything = NaN). This script finds WHICH group, WITHOUT another
38-minute wait, by doing two things:

  1. Temporarily overrides params.monte_carlo.n_draws to a small number (default
     200) via a patch on src.artifacts.load_many -- this bug looks structural (ALL
     200,000 draws failed identically, not a rare sampling fluke), so a tiny draw
     count should reproduce it just as reliably, in well under a minute instead of
     ~38 minutes. Nothing else about the real computation changes.
  2. Wraps every function whose name contains "monte_carlo" in src.flowdriven_model
     AND src.cohort_flow_mc (whichever module actually defines the per-group
     Monte Carlo entry point -- I don't know which without seeing the code, so
     both are covered) with a pass-through diagnostic: prints a short summary of
     that call's arguments, then checks whether the return value contains any
     all-NaN array, and if so prints exactly which key and which call. The
     wrapped functions behave EXACTLY as before -- same inputs, same outputs,
     nothing computed differently -- this only adds visibility.

Also no-ops src.artifacts.save_many during this run (prints what WOULD have been
saved instead) so a low-draw diagnostic run doesn't overwrite your real, full
200,000-draw BAU artifact on disk.

USAGE: run exactly like 03_02_adjustedflows.py, e.g.:
    /path/to/.venv/bin/python /path/to/code/diagnose_03_02_nan_source.py
Optional: pass a draw count as the first argument to override the default of 200,
e.g. `... diagnose_03_02_nan_source.py 500` if 200 doesn't reproduce it.
Paste back everything printed between "=====...=====" markers.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
import sys
from pathlib import Path

import numpy as np


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

N_DRAWS_OVERRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 200
print(f"Overriding params.monte_carlo.n_draws to {N_DRAWS_OVERRIDE} for a fast repro.")

# -----------------------------------------------------------------------
# 1. Patch src.artifacts.load_many / save_many
# -----------------------------------------------------------------------
import src.artifacts as artifacts  # type: ignore

_original_load_many = artifacts.load_many
_original_save_many = artifacts.save_many


def _patched_load_many(*keys, **kwargs):
    result = _original_load_many(*keys, **kwargs)
    if "params" in result:
        params = result["params"]
        try:
            new_mc = dataclasses.replace(params.monte_carlo, n_draws=N_DRAWS_OVERRIDE)
            new_params = dataclasses.replace(params, monte_carlo=new_mc)
            result["params"] = new_params
            print(f"Patched params.monte_carlo.n_draws: {params.monte_carlo.n_draws} -> {N_DRAWS_OVERRIDE}")
        except Exception as e:
            print(f"WARNING: could not override params.monte_carlo.n_draws ({e}) -- "
                  f"running with the ORIGINAL draw count, this may take the full ~38 min.")
    return result


def _patched_save_many(**kwargs):
    print(f"[diagnostic run -- NOT actually saving] save_many would have saved: {sorted(kwargs.keys())}")
    return {k: Path(f"<not saved -- diagnostic run>/{k}.pkl") for k in kwargs if k != "root"}


artifacts.load_many = _patched_load_many
artifacts.save_many = _patched_save_many

# -----------------------------------------------------------------------
# 2. Patch src.monte_carlo.summarize_distribution (same as diagnose_03_02_nan_summary.py)
# -----------------------------------------------------------------------
import src.monte_carlo as mc  # type: ignore

_original_summarize_distribution = mc.summarize_distribution


def _describe_array_like(val):
    """Best-effort stats for whatever value is in a dict we're inspecting."""
    try:
        a = np.asarray(val, dtype=float)
        n = a.size
        n_nan = int(np.isnan(a).sum())
        n_inf = int(np.isinf(a).sum())
        n_finite = n - n_nan - n_inf
        if n_finite > 0:
            finite_vals = a[np.isfinite(a)]
            return (f"len={n}, NaN={n_nan}, Inf={n_inf}, finite={n_finite}, "
                    f"finite range=[{finite_vals.min():.6g}, {finite_vals.max():.6g}]")
        return f"len={n}, NaN={n_nan}, Inf={n_inf}, finite={n_finite} (ALL INVALID)"
    except Exception as e:
        return f"<could not summarize: {e}>"


def _deep_scan_for_nan(obj, path: str, depth: int = 0, max_depth: int = 5, _seen=None) -> list[str]:
    """
    Recursively walk dicts (and dict-like containers) looking for array-like leaf
    values that are ALL NaN/Inf. Returns a list of "path: stats" strings for every
    all-invalid leaf found -- this is what lets us find something buried like
    mc['by_group'][('EUR','BEV','A')]['cumulative_survival'] without needing to
    already know that exact structure.
    """
    if _seen is None:
        _seen = set()
    if depth > max_depth:
        return []
    findings = []

    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in _seen:
            return []
        _seen.add(obj_id)
        for k, v in obj.items():
            findings += _deep_scan_for_nan(v, f"{path}[{k!r}]", depth + 1, max_depth, _seen)
        return findings

    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            for col in obj.columns:
                if pd.api.types.is_numeric_dtype(obj[col]):
                    findings += _deep_scan_for_nan(obj[col].to_numpy(), f"{path}[{col!r}]", depth + 1, max_depth, _seen)
            return findings
        if isinstance(obj, pd.Series):
            obj = obj.to_numpy()
    except Exception:
        pass

    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (int, float, np.floating, np.integer)):
        obj = np.asarray(obj)

    if isinstance(obj, np.ndarray) and obj.dtype.kind in "fc":
        if obj.size == 0:
            return []
        n_finite = int(np.isfinite(obj).sum())
        if n_finite == 0:
            findings.append(f"{path}: {_describe_array_like(obj)}")
    return findings


def _diagnostic_summarize_distribution(values, *args, **kwargs):
    try:
        arr = np.asarray(values, dtype=float)
        n_finite = int(np.isfinite(arr).sum())
    except Exception:
        return _original_summarize_distribution(values, *args, **kwargs)
    if n_finite == 0:
        print("=" * 78)
        print("summarize_distribution about to fail -- ALL values NaN/Inf (see per-call")
        print("Monte Carlo diagnostics above for which group caused this).")
        try:
            frame = inspect.currentframe().f_back
            print("  simple local variables in the calling function:")
            for name, val in frame.f_locals.items():
                if isinstance(val, (str, int, float, bool)) or val is None:
                    print(f"    {name} = {val!r}")
        except Exception as e:
            print(f"  (could not inspect the calling frame: {e})")

        # DEEP scan of every dict-typed local in THIS frame and ONE frame up (e.g.
        # `mc`, which the previous run showed has a nested 'by_group' key) -- walks
        # arbitrarily nested dicts/DataFrames looking for all-NaN leaves, so we don't
        # need to already know the exact structure.
        print("  deep scan for all-NaN values, walking every dict-typed local variable")
        print("  (in this function AND the function that called it) up to 5 levels deep:")
        any_found = False
        try:
            frames_to_scan = [inspect.currentframe().f_back]
            if inspect.currentframe().f_back.f_back is not None:
                frames_to_scan.append(inspect.currentframe().f_back.f_back)
            _global_seen = set()
            for fr in frames_to_scan:
                for name, val in fr.f_locals.items():
                    if isinstance(val, dict) and val:
                        results = _deep_scan_for_nan(val, name, 0, 5, _global_seen)
                        for r in results:
                            any_found = True
                            print(f"    ALL-NaN LEAF FOUND: {r}")
        except Exception as e:
            print(f"  (deep scan failed: {e})")
        if not any_found:
            print("    (deep scan found no all-NaN leaves in any dict-typed local -- the NaN may")
            print("     be introduced by a computation whose intermediate result isn't kept in a")
            print("     local variable, e.g. inline inside an expression.)")
        print("=" * 78)
    return _original_summarize_distribution(values, *args, **kwargs)


mc.summarize_distribution = _diagnostic_summarize_distribution


# -----------------------------------------------------------------------
# 3. Wrap every *_monte_carlo* function in src.flowdriven_model / src.cohort_flow_mc
# -----------------------------------------------------------------------
def _short_repr(val) -> str:
    try:
        import pandas as pd
        if isinstance(val, np.ndarray):
            return f"ndarray(shape={val.shape}, dtype={val.dtype})"
        if isinstance(val, pd.DataFrame):
            return f"DataFrame(shape={val.shape}, columns={list(val.columns)[:8]})"
        if isinstance(val, pd.Series):
            return f"Series(len={len(val)}, name={val.name!r})"
        if isinstance(val, dict):
            return f"dict(keys={list(val.keys())[:8]})"
        if isinstance(val, (list, tuple)) and len(val) > 10:
            return f"{type(val).__name__}(len={len(val)})"
    except Exception:
        pass
    r = repr(val)
    return r if len(r) < 200 else r[:200] + "...(truncated)"


def _check_for_all_nan(result, call_desc: str) -> None:
    def _check_array(arr, key_desc):
        try:
            a = np.asarray(arr, dtype=float)
        except Exception:
            return
        if a.size == 0:
            return
        if np.isnan(a).all():
            print("!" * 78)
            print(f"ALL-NaN RESULT FOUND: {call_desc} -> key {key_desc!r}")
            print(f"  shape={a.shape}, first 10 values: {a.flatten()[:10].tolist()}")
            print("!" * 78)

    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, np.ndarray) or (isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float))):
                _check_array(v, k)
    elif isinstance(result, (tuple, list)):
        for i, v in enumerate(result):
            if isinstance(v, np.ndarray):
                _check_array(v, f"[{i}]")
    elif isinstance(result, np.ndarray):
        _check_array(result, "<return value>")


_wrapped_count = 0


def _wrap_monte_carlo_functions(module) -> None:
    global _wrapped_count
    for name in dir(module):
        if "monte_carlo" not in name.lower():
            continue
        obj = getattr(module, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        if getattr(obj, "_is_diagnostic_wrapped", False):
            continue

        @functools.wraps(obj)
        def _wrapper(*args, __orig=obj, __name=f"{module.__name__}.{name}", **kwargs):
            arg_desc = ", ".join(_short_repr(a) for a in args)
            kwarg_desc = ", ".join(f"{k}={_short_repr(v)}" for k, v in kwargs.items())
            call_desc = f"{__name}({arg_desc}{', ' if arg_desc and kwarg_desc else ''}{kwarg_desc})"
            print(f"[monte_carlo call] {call_desc}")

            # [NEW] BEV-specific deep dive: the previous run's deep-NaN-scan showed
            # EVERY BEV segment (and only BEV) with all-NaN cumulative_survival/etc.
            # Print BEV's ACTUAL input values (not just dict keys) for every kwarg
            # that looks like it's keyed by (Region, Drivetrain, Segment) or by
            # drivetrain name directly, so we can see whether something is
            # degenerately zero/empty specifically for BEV.
            print("  --- BEV-specific input values (checking for a degenerate zero vs. already-NaN) ---")

            def _describe_recursive(obj, prefix, depth=0, max_depth=4):
                if depth > max_depth:
                    print(f"      {prefix}: <max depth reached>")
                    return
                if isinstance(obj, dict):
                    if not obj:
                        print(f"      {prefix}: <empty dict>")
                        return
                    for kk, vv in list(obj.items())[:15]:
                        _describe_recursive(vv, f"{prefix}[{kk!r}]", depth + 1, max_depth)
                    if len(obj) > 15:
                        print(f"      {prefix}: ... ({len(obj) - 15} more keys)")
                    return
                try:
                    import pandas as pd
                    if isinstance(obj, (pd.Series, pd.DataFrame)):
                        obj = obj.to_numpy()
                except Exception:
                    pass
                print(f"      {prefix}: {_describe_array_like(obj)}")

            for k, v in kwargs.items():
                if not isinstance(v, dict) or not v:
                    continue
                sample_key = next(iter(v.keys()))
                if isinstance(sample_key, tuple) and len(sample_key) == 3:
                    bev_entries = {gk: gv for gk, gv in v.items() if gk[1] == "BEV"}
                    if bev_entries:
                        print(f"    {k} (BEV groups, recursive):")
                        for gk, gv in bev_entries.items():
                            _describe_recursive(gv, str(gk))
                elif sample_key == "BEV" or "BEV" in v:
                    print(f"    {k}['BEV'] (recursive):")
                    _describe_recursive(v.get("BEV"), "BEV")
            print("  --- end BEV-specific input values ---")

            result = __orig(*args, **kwargs)
            _check_for_all_nan(result, call_desc)
            return result

        _wrapper._is_diagnostic_wrapped = True
        setattr(module, name, _wrapper)
        _wrapped_count += 1
        print(f"Wrapped {module.__name__}.{name} with a diagnostic pass-through.")


try:
    import src.flowdriven_model as fdm  # type: ignore
    _wrap_monte_carlo_functions(fdm)
except ImportError as e:
    print(f"NOTE: could not import src.flowdriven_model ({e}) -- skipping.")

try:
    import src.cohort_flow_mc as cfmc  # type: ignore
    _wrap_monte_carlo_functions(cfmc)
except ImportError as e:
    print(f"NOTE: could not import src.cohort_flow_mc ({e}) -- skipping (may not exist as a separate module).")

if _wrapped_count == 0:
    print("WARNING: no *_monte_carlo* functions found to wrap in either module -- the per-call")
    print("diagnostics below won't appear. The n_draws speedup and summarize_distribution")
    print("diagnostic will still run, though.")

# -----------------------------------------------------------------------
# 4. Run 03_02_adjustedflows.py for real, with all of the above patched in
# -----------------------------------------------------------------------
SCRIPT_CANDIDATES = [
    PROJECT_ROOT / "code" / "03_02_adjustedflows.py",
    SCRIPT_DIR / "03_02_adjustedflows.py",
]
SCRIPT_PATH = next((c for c in SCRIPT_CANDIDATES if c.is_file()), None)
if SCRIPT_PATH is None:
    matches = [m for m in PROJECT_ROOT.rglob("03_02_adjustedflows*.py") if "diagnose_" not in m.name]
    if not matches:
        raise FileNotFoundError("Could not find 03_02_adjustedflows.py under the project root.")
    SCRIPT_PATH = matches[0]

print(f"\nRunning: {SCRIPT_PATH}  (n_draws={N_DRAWS_OVERRIDE}, should be fast)\n")

import runpy
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")