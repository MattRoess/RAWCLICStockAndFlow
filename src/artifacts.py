"""
artifacts.py
============

Central artifact registry + pickle-based persistence layer for the EVmodel pipeline.

PURPOSE
-------
Every intermediate result that flows between stages 00-06 (scenario dataframes, stock
dicts, disaggregated stock, material composition tables, sensitivity-analysis "tracker"
variants, ...) is saved and loaded through this module rather than ad hoc file paths
scattered across scripts. `ARTIFACT_FILES` is the single place that maps a short,
memorable artifact name (e.g. "stock_dict") to its on-disk filename (e.g.
"01_stock_dict_df.pkl"). This is a genuinely good pattern -- it means every stage can
say `load_many("stock_dict")` without knowing or caring about the actual file layout,
and `artifact_status()` gives a one-shot view of what has/hasn't been computed yet.

FIX LOG (this round -- see EVmodel_review_consolidated.md, Fix Log, for the full record)
------------------------------------------------------------------------------------------
- [FIXED, mechanical] `ARTIFACT_FILES` had three keys defined twice each (identical
  values both times) -- deduplicated. No behavior change (Python already silently kept
  the second/last definition; the resulting dict is unchanged). See the comment at
  `ARTIFACT_FILES` for the still-open question of whether a genuinely different,
  currently-missing sensitivity variant was intended instead of a literal duplicate --
  that part is NOT resolved here, since it requires knowing what that variant should be
  called, which nothing in the code or prior review answers.
- [FLAGGED, needs a decision -- not resolved here] `EXPORT_SHARE_BY_DRV` (the artifact
  name registered here) is textually identical to `params["02_stock_flow"]
  ["EXPORT_SHARE_BY_DRV"]` (the parameter defined in `00_parameters.py`). See the comment
  at that entry below for the two possible readings and why I haven't picked one.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .config import get_paths
# `get_paths()` (from a sibling `config.py`, now reviewed) returns a `Paths` object with
# an `.artifacts` attribute -- a `Path` to the directory where all pickles live.

# ---------------------------------------------------------------------------
# Artifact name -> filename registry
# ---------------------------------------------------------------------------
# The "NN_" filename prefix (00, 01, 02, 03, 04, 05) tells you which stage produced the
# artifact -- e.g. everything prefixed "03_" is written by stage 03 (disaggregation).
# Reading down this list is a good proxy for "everything the pipeline eventually
# computes", even for artifacts not yet shown to me in any source file.
#
# [FIXED, mechanical, this round]: three keys were defined TWICE in this dict literal
# with identical values -- deduplicated below (Python silently kept the last definition
# either way, so the resulting dict is unchanged; this only removes the confusing,
# redundant source lines).
#
# [STILL OPEN -- needs your input, not resolved here]: given every other name in this
# stretch of the registry follows a "one entry per sensitivity-analysis variant" pattern
# (BEV_only, BEV_A_F, BEV_JA_JF, BEV_large, BEV_small, BEV_longer, ICEV_shorter,
# stock_lower, unknownwhereabouts_lower/normal, losses_zero/high), the duplicated keys
# --"tracker_keyed_BEV_longer", "tracker_keyed_ICEV_shorter", "tracker_keyed_stock_lower"
# -- may have been intended as placeholders for a DIFFERENT, currently-missing variant
# (e.g. an "unknownwhereabouts_higher" to pair with "_lower"/"_normal", or a
# "losses_..." counterpart) rather than a literal repeat. If you want a new variant
# registered, tell me its name and I'll add it as a real (non-duplicate) entry --
# I can't invent the name myself.
ARTIFACT_FILES: dict[str, str] = {
    "params": "00_params.pkl",
    "scenario_map": "01_scenario_map.pkl",
    "scenario_df": "01_scenario_df.pkl",
    "stock_dict": "01_stock_dict.pkl",
    "stock_dict_df": "01_stock_dict_df.pkl",
    "df_exp_eu": "01_df_exp_eu.pkl",
    "matrices_by_key": "02_matrices_by_key.pkl",
    "tracker_keyed": "03_tracker_keyed.pkl",
    "disaggregated": "03_disaggregated.pkl",
    "stock_2005": "03_stock_2005.pkl",
    "segment_shares_ext": "03_segment_shares_ext.pkl",
    "liquids_shares_ext": "03_liquids_shares_ext.pkl",
    "composition_extended": "04_composition_extended.pkl",
    "ev_share_long": "04_ev_share_long.pkl",
    "bev_surv_outflow": "04_bev_surv_outflow.pkl",
    "mass_by_year_elem_dict": "04_mass_by_year_elem_dict.pkl",
    "combined": "04_combined.pkl",
    "mc_stage02_draws": "02_mc_draws.pkl",
    # [NEW] Raw per-drivetrain, per-draw Monte Carlo arrays from 02_stockdriven.py
    # (cumulative_inflow, cumulative_out_survival, and the scale_lambda draws that
    # produced them). Stage 03's own Monte Carlo block loads this and applies its OWN
    # (unknown_whereabouts_share, export_share) uncertainty to these SAME per-draw
    # values -- that's what makes this genuine propagation, not independent
    # per-stage resampling. Raw arrays, not just a summary, because a summary alone
    # would lose the draw-to-draw correspondence downstream stages need.
    "mc_stage02_summary": "02_mc_summary.pkl",
    # [NEW] mean/median/mode/P2.5/P97.5 + 50-bin histogram per drivetrain, per metric
    # -- for direct inspection/plotting of stage 02's own Monte Carlo output.
    "mc_stage03_summary": "03_mc_summary.pkl",
    # [NEW] Same idea, for stage 03's own Monte Carlo extension: applies THIS stage's
    # (unknown_whereabouts_share, export_share) uncertainty to stage 02's per-draw
    # cumulative_out_survival (loaded from mc_stage02_draws), producing
    # collected/export/unknown summaries per drivetrain and as an EU total.
    "mc_stage03_02_summary": "03_02_mc_summary.pkl",
    # [UPDATED] 03_02_adjustedflows.py's Monte Carlo extension: each of the 11
    # scenarios now re-simulates its OWN lifetime AND share uncertainty per draw
    # via a vectorized cohort-flow engine (`src/cohort_flow_mc.py`, wrapped by
    # `src/flowdriven_model.py`'s `run_flow_driven_model_monte_carlo`) -- NOT a
    # post-hoc re-split of a fixed deterministic total. `scale_lambda` is
    # resampled per draw from a Triangular distribution
    # (`stock_flow.lifetime_scale_lambda_relative_spread`, same convention as
    # stage 02); `unknown_whereabouts_share`/`export_share` are resampled from a
    # clipped Normal (`stock_flow.unknown_whereabouts_share_std`/
    # `export_share_std`). A scenario's own `lifetime_change_by_drv` override
    # (e.g. BEV_longer's scale_lambda=17 from 2027) and `stock_modifier_2027`
    # are preserved in its Monte Carlo run, not just its deterministic one.
    # Stores collected/export/unknown summaries per drivetrain and as an EU
    # total, per scenario -- same output shape as before this update, only the
    # simulation behind it changed.
    # NOTE: stage-04's per-scenario outputs (11 scenarios x several output types across
    # 04_01/04_03/04_04) are deliberately NOT individually registered here -- see
    # `materials.py`'s `save_unregistered_scenario_outputs()` docstring for why (adding
    # one entry per scenario-x-output combination doesn't scale, and would need to grow
    # every time a scenario is added). They're saved via that shared helper instead,
    # invisible to `artifact_status()` by design -- a real, deliberately-deferred option
    # to register them (e.g. a wildcard/prefix-based registry) is flagged there, not
    # implemented unasked.
    "ratio_df": "05_ratio_df.pkl",
    "stock_disagg_df": "05_stock_disagg_df.pkl",
    "synthetic_pre_2005_inflows": "03_synthetic_pre_2005_inflows.pkl",
    "starting_stock_2005_segments": "03_starting_stock_2005_segments.pkl",
    "export_prob_by_age_drv": "03_export_prob_by_age_drv.pkl",
    "flows_03": "03_flows_03.pkl",
    "seg_share_by_drv": "03_seg_share_by_drv.pkl",
    "tracker_keyed_new": "03_tracker_keyed_new.pkl",
    "tracker_keyed_BAU": "03_tracker_keyed_BAU.pkl",
    "tracker_keyed_BEV_only": "03_tracker_keyed_BEV_only.pkl",
    "tracker_keyed_BEV_A_F": "03_tracker_keyed_BEV_A_F.pkl",
    "tracker_keyed_BEV_JA_JF": "03_tracker_keyed_BEV_JA_JF.pkl",
    "tracker_keyed_BEV_large": "03_tracker_keyed_BEV_large.pkl",
    "tracker_keyed_BEV_small": "03_tracker_keyed_BEV_small.pkl",
    "tracker_keyed_BEV_longer": "03_tracker_keyed_BEV_longer.pkl",
    "tracker_keyed_ICEV_shorter": "03_tracker_keyed_ICEV_shorter.pkl",
    "tracker_keyed_stock_lower": "03_tracker_keyed_stock_lower.pkl",
    "tracker_keyed_unknownwhereabouts_lower": "03_tracker_keyed_unknownwhereabouts_lower.pkl",
    "tracker_keyed_unknownwhereabouts_normal": "03_tracker_keyed_unknownwhereabouts_normal.pkl",
    "tracker_keyed_losses_zero": "03_tracker_keyed_losses_zero.pkl",
    "tracker_keyed_losses_high": "03_tracker_keyed_losses_high.pkl",
    "EXPORT_SHARE_BY_DRV": "03_EXPORT_SHARE_BY_DRV.pkl",
    # [STILL OPEN -- needs your input, not resolved here]: this name is identical to
    # params["02_stock_flow"]["EXPORT_SHARE_BY_DRV"] defined in 00_parameters.py. Two
    # readings are both plausible and I can't tell which from this file alone:
    #   (a) this artifact IS that same dict, persisted as-is for stage 03's convenience
    #       (in which case registering it separately here is redundant -- stage 03 could
    #       just read it from the "params" artifact instead of needing its own copy), or
    #   (b) this artifact is a DERIVED/adjusted version -- e.g. stage 03 recomputes or
    #       overrides some drivetrain's export share before saving it under this name,
    #       in which case the two are legitimately different objects that happen to
    #       share a name, and whichever stage-03 code writes this artifact should say so
    #       in a comment there.
    # I don't have visibility into which stage-03 code path actually writes this
    # artifact, so I can't resolve this without you telling me the intent -- happy to
    # rename this entry (e.g. to "export_share_by_drv_stage03") if (b) is correct, to
    # remove the naming collision.
}


def _artifact_path(name: str, root: Path | None = None) -> Path:
    """
    Resolve a registered artifact name to its full on-disk path.

    `root`: optional explicit project root, passed through to `get_paths(start=root)`.
    See `save_many`/`load_many`/`artifact_status` docstrings for why this matters -- in
    short, without it, the artifact directory is resolved from the process's current
    working directory at call time, which is not always the same as where the calling
    script actually lives (confirmed to cause real, silent mislocation in Positron --
    see EVmodel_review_consolidated.md Fix Log for the reproduction).
    """
    paths = get_paths(start=root)
    if name not in ARTIFACT_FILES:
        raise KeyError(f"Unknown artifact '{name}'. Available: {sorted(ARTIFACT_FILES)}")
    return paths.artifacts / ARTIFACT_FILES[name]


def save_artifact(name: str, value: Any, root: Path | None = None) -> Path:
    """
    Pickle `value` to the path registered under `name` in ARTIFACT_FILES.

    `root`: see `_artifact_path`. `get_paths()` (via `config.py`) always creates the
    target directory before writing (confirmed -- see the retracted finding L10 in
    EVmodel_review_consolidated.md), so no separate mkdir is needed here.
    """
    path = _artifact_path(name, root=root)
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return path


def load_artifact(name: str, root: Path | None = None) -> Any:
    """Unpickle and return the artifact registered under `name`. `root`: see `_artifact_path`."""
    path = _artifact_path(name, root=root)
    if not path.exists():
        raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_many(root: Path | None = None, **kwargs: Any) -> dict[str, Path]:
    """
    Save several artifacts in one call; keyword name becomes the artifact name.

    [FIXED, this round]: added an optional `root=` parameter, threaded through to
    `get_paths(start=root)`. WHY THIS MATTERS: without it, the artifact directory is
    resolved from `Path.cwd()` -- the process's working directory AT THE MOMENT this is
    called -- which is not necessarily where the calling script lives. Confirmed to
    cause real breakage: running a script from an IDE (Positron) whose console keeps a
    different working directory than the script's own location silently wrote the
    pickled artifact to a wrong, unrelated folder tree, while the script's OTHER output
    (`params.xlsx`, which is anchored to the script's own file location) landed
    correctly. Passing `root=<the script's own resolved project root>` makes both
    outputs consistently anchored to the same place, regardless of the caller's cwd.
    Existing calls with no `root=` keep the exact previous behavior (cwd-based), so
    nothing that already worked changes.
    """
    saved: dict[str, Path] = {}
    for key, value in kwargs.items():
        saved[key] = save_artifact(key, value, root=root)
    return saved


def load_many(*names: str, root: Path | None = None) -> dict[str, Any]:
    """Load several artifacts in one call, returned as {name: value}. `root`: see `save_many`."""
    return {name: load_artifact(name, root=root) for name in names}


def artifact_status(root: Path | None = None) -> list[dict[str, Any]]:
    """
    Report, for every REGISTERED artifact (not just ones that exist), whether its file
    is currently present on disk. Useful as a one-shot "how far has the pipeline been
    run" dashboard -- e.g. calling this after only running 00 and 01 will show `exists:
    True` for the 00_/01_-prefixed rows and `exists: False` for everything from stage 02
    onward. `root`: see `save_many`.
    """
    rows = []
    paths = get_paths(start=root)
    for name, fname in ARTIFACT_FILES.items():
        path = paths.artifacts / fname
        rows.append(
            {
                "artifact": name,
                "file": fname,
                "exists": path.exists(),
                "path": str(path),
            }
        )
    return rows
