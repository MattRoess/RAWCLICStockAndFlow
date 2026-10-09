"""
01_data_prep.py
================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Loads REMIND scenario output and EU vehicle-export data, reshapes REMIND's variable
structure into a usable stock/inflow/outflow table, and persists the artifacts that
stage 02 (stock-flow) consumes:

    scenario_map, scenario_df, stock_dict, stock_dict_df, df_exp_eu

FIXES APPLIED THIS ROUND
--------------------------
- Reads the new dataclass `params.data_prep` (see 00_parameters.py) instead of a dict
  section -- `p01["scenario"]` -> `p01.scenario`.
- The hardcoded `REMIND_SCENARIOS` list (which duplicated `scenario_list` and needed a
  runtime drift-check against it) is gone -- `p01.remind_scenario_files` is now the
  single source of truth, enforced structurally by `DataPrepParams.validate()`.
- `export_data_file_name` and `iso3_corrections` are now sourced from params rather
  than hardcoded inside `data_prep.clean_export_data`.
- `export_min_year` / `export_max_year_exclusive` are now sourced from params rather
  than hardcoded as literal call arguments here.
- Removed the second, dead import block (json, scipy.stats, seaborn, Line2D,
  weibull_min, nnls, duplicate matplotlib.pyplot) -- none of it was used in this file.
  If a fuller version of this stage genuinely needs Weibull fitting / NNLS here, add
  the specific import back at the point of use, not as a block import up front.
- Removed deprecated `matplotlib.cm.get_cmap` import (unused anyway, since the whole
  dead-import block is gone).
- Removed the `export_full = export_df.copy()` alias (M11) -- `export_df` is the only
  name now; nothing downstream in this file referenced the old name.
- File paths built with `pathlib.Path` instead of string concatenation (L3).
- **[NEW] Integrated diagnostic plots**: right after `stock_dict` is built, this stage
  now generates and saves THREE charts (no separate script to remember to run):
    1. `01_bev_transition_check.png` -- BEV scenario comparison (as before).
    2. `01_stock_by_drivetrain.png` -- **NEW**: the actual stock trajectory per
       drivetrain for the SELECTED scenario, i.e. exactly what this stage hands to
       stage 02. This is the single most direct "does my data look right" check for
       this stage.
    3. `01_export_volume_check.png` -- **NEW**: total EU export volume by year (from
       `df_exp_eu`), a sanity check on the C1/C2 fixes' real-world magnitude (North
       Korea exclusion, dedup fix) -- if this number looks implausible, that's the
       first place to look.
- **[FIXED, this round]** `01_stock_by_drivetrain.png` was plotting the FULL raw REMIND
  data horizon (out to 2150 -- confirmed, REMIND's raw `.mif` files genuinely contain
  data that far out; there was never any year-clipping logic in `plot_stock_by_
  drivetrain` or in `build_stock_dict`/`prepare_remind_scenarios` in `src/data_prep.py`,
  neither of which this fix touches). Confirmed with the user: this stage's own diagnostic
  PLOT should be capped at 2100, matching stage 02's own `model_end_year=2100` (see the
  comment in `03_01_flowdriven.py` noting "REMIND raw data to 2150 (stage 01),
  model_end_year=2100 (stage 02)") -- so this plot now shows the same horizon stage 02
  actually uses, instead of the full uncut raw range. This is DISPLAY-ONLY: `stock_dict`
  and `stock_dict_df` (the artifacts actually saved and handed to stage 02) are
  completely unchanged, still the full raw REMIND range -- only `plot_stock_by_
  drivetrain`'s own chart is capped. See `PLOT_YEAR_MAX` below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # never opens an interactive window -- always saves to file
import matplotlib.pyplot as plt


# Same upward-searching root resolution as 00_parameters.py (see that file's docstring
# for why: this script can live directly in the project root OR in a `code/` subfolder,
# with `src/` as a sibling of `code/`, not a child of it).
def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# The script's own directory, independent of the caller's current working directory.
# `p01.input_dir` ("../data/raw/") is written relative to THIS location (the script
# sitting in code/), not to wherever a terminal happened to be `cd`'d into when Python
# was invoked with a full/relative path to this file.
SCRIPT_DIR = Path(__file__).resolve().parent

from src.artifacts import load_many, save_many  # type: ignore
from src.config import get_paths  # type: ignore
from src.data_prep import (  # type: ignore
    build_stock_dict,
    clean_export_data,
    fill_missing_pre_year_history,
    load_remind_scenarios,
    prepare_remind_scenarios,
)
from src.stock_flow import (  # type: ignore
    plot_bev_stock_compare_grouped,
)


# [FIXED, this round] `01_stock_by_drivetrain.png`'s x-axis cap -- see the module
# docstring's "FIXES APPLIED THIS ROUND" entry above for the full reasoning. Matches
# stage 02's own `model_end_year=2100` (confirmed with the user), NOT the full raw REMIND
# horizon (2150) that `stock_dict_df` itself still contains. DISPLAY-ONLY: only this
# plot function reads it; `stock_dict`/`stock_dict_df` (what actually gets saved for
# stage 02) are untouched. Edit this one number to retune -- no other code change needed.
PLOT_YEAR_MAX = 2100


def plot_stock_by_drivetrain(stock_dict_df, scenario: str, region: str = "EUR", year_max: int = PLOT_YEAR_MAX):
    """
    The single most direct "does my data look right" check for this stage: actual
    stock trajectory per drivetrain, for the scenario actually selected in params --
    exactly what gets handed to stage 02, not a scenario-comparison abstraction.

    `year_max`: DISPLAY-ONLY cap on the plotted x-axis (default `PLOT_YEAR_MAX`, 2100 --
    matches stage 02's own `model_end_year`). `stock_dict_df` itself is NOT filtered or
    modified -- only the rows plotted here are restricted to `year <= year_max`, so the
    y-axis autoscale also reflects just this capped window rather than being stretched
    by the long flat tail out to the raw REMIND data's full 2150 horizon.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for tech, grp in stock_dict_df.groupby("technology"):
        grp = grp[grp["year"] <= year_max].sort_values("year")
        ax.plot(grp["year"], grp["value"], linewidth=1.8, label=tech)
    ax.set_title(f"{region} vehicle stock by drivetrain — scenario '{scenario}'", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stock [million]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig, ax


def plot_export_volume(df_exp_eu):
    """
    Total EU used-vehicle export volume by year -- a real-world-magnitude sanity check
    on the C1 (North Korea exclusion) and C2 (dedup double-count) fixes in
    src/data_prep.py. If this number looks implausible (e.g. an order of magnitude off
    from what you'd expect), that's the first place to look.
    """
    by_year = df_exp_eu.groupby("year")["export"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(by_year.index, by_year.values, color="#4a7fb5", width=0.8)
    ax.set_title("Total EU used-vehicle export volume by year", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Export volume [million vehicles]")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, ax


def main() -> dict[str, Path]:
    """Run the full 01 data-prep stage and persist its artifacts."""

    # [FIXED, found via end-to-end smoke test this round, same class of bug as
    # 00_parameters.py's Fix Log item 13]: load_many/save_many resolve their artifact
    # directory from Path.cwd() unless told otherwise. Running this script from
    # `<project_root>/code/` (exactly what HOW_TO_RUN_AND_VERIFY.md instructs) without
    # `root=` would look for `00_parameters.py`'s pickled artifact in the WRONG place
    # and fail with FileNotFoundError -- reproduced and confirmed before this fix.
    params = load_many("params", root=PROJECT_ROOT)["params"]
    p01 = params.data_prep

    # [FIXED, real bug reported and reproduced this round]: `input_dir` ("../data/raw/")
    # is a RELATIVE path. Passing it straight to pandas/Path means it gets resolved
    # against the process's current working directory AT RUNTIME -- which is wherever
    # the terminal/IDE happened to be `cd`'d into when Python was invoked, NOT
    # necessarily `<project_root>/code/` (confirmed: running via a full path from the
    # project root itself, e.g. `python code/01_data_prep.py` from
    # `<project_root>/`, resolved "../data/raw/" one level ABOVE the project entirely
    # and raised FileNotFoundError, even though the actual REMIND files were correctly
    # in place). Fixed by resolving against `SCRIPT_DIR` (this file's own location,
    # always correct regardless of caller cwd) instead of leaving it as a bare relative
    # string. `input_dir` is now an absolute path from this point on.
    input_dir = str((SCRIPT_DIR / p01.input_dir).resolve())

    # -----------------------------------------------------------------------
    # Load + reshape REMIND scenario output
    # -----------------------------------------------------------------------
    remind_scenarios = [
        (name, rel_path, sep) for name, (rel_path, sep) in p01.remind_scenario_files.items()
    ]
    remind_data = load_remind_scenarios(input_dir=input_dir, scenario_specs=remind_scenarios)

    scenario_df, scenario_map = prepare_remind_scenarios(
        data_dict=remind_data,
        scenario=p01.scenario,
        remind_regions=list(p01.remind_regions),
        prefix=p01.prefix,
        remind_technology=list(p01.remind_technology),
        # [NEW, this round] See params_schema.py's DataPrepParams.norway_iceland_
        # share_of_neu docstring for the full derivation -- fixes the UKI (UK+Ireland)
        # region-taxonomy bug found this round for scenarios that provide REMIND's
        # native EU27/NEU variables (b650, npi25).
        norway_iceland_share_of_neu=p01.norway_iceland_share_of_neu,
        # [NEW, this round] Companion fix for ssp2L/ssp2M/ssp1, which have no native
        # EU27 to use directly -- see params_schema.py's DataPrepParams.uk_ireland_
        # share_of_eur docstring for the derivation.
        uk_ireland_share_of_eur=p01.uk_ireland_share_of_eur,
    )

    # -----------------------------------------------------------------------
    # [NEW, this round] Pre-2015 missing-history fix -- SEPARATE from, and applied
    # AFTER, the region-scope fix above (that fixes WHICH COUNTRIES are included;
    # this fixes MISSING YEARS -- see fill_missing_pre_year_history's docstring for
    # the full "NaN -> 0 -> interpolation-overshoot -> wrong backcast" chain this
    # resolves for ssp2L/ssp2M/ssp1). Operates on the FULL scenario_map (needs the
    # donor scenarios -- b650, npi25 -- already present and already region-scope-
    # corrected), so this has to run after prepare_remind_scenarios returns, not
    # inside it. `scenario_df` is re-derived from the UPDATED scenario_map afterward,
    # in case `p01.scenario` itself is one of the patched (target) scenarios.
    # -----------------------------------------------------------------------
    scenario_map = fill_missing_pre_year_history(
        scenario_map=scenario_map,
        target_scenarios=p01.pre2015_history_target_scenarios,
        donor_scenarios=p01.pre2015_history_donor_scenarios,
        splice_year=p01.pre2015_history_splice_year,
        method=p01.pre2015_history_method,
    )
    scenario_df = scenario_map[p01.scenario].copy()

    # -----------------------------------------------------------------------
    # EU vehicle export/import data
    # -----------------------------------------------------------------------
    export_df, export_df_2005, df_exp_eu, df_imp_eu = clean_export_data(
        input_dir=input_dir,
        eu_countries=p01.eu_countries,
        min_year=p01.export_min_year,
        max_year_exclusive=p01.export_max_year_exclusive,
        export_data_file_name=p01.export_data_file_name,
        iso3_corrections=p01.iso3_corrections,
    )

    # -----------------------------------------------------------------------
    # Aggregate stock/inflow/outflow by (region, technology), summed over segment
    # -----------------------------------------------------------------------
    stock_dict, stock_dict_df = build_stock_dict(
        scenario_df=scenario_df,
        remind_regions=list(p01.remind_regions),
        stock_interpolation_method=p01.stock_interpolation_method,
    )

    print("stock_dict keys (region, technology):", list(stock_dict.keys()))

    fig_dir = get_paths(start=PROJECT_ROOT).figures      # the one folder, src/config.py
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Diagnostic plot 1: BEV stock across scenarios, with the SELECTED scenario
    # (`p01.scenario`) marked -- integrated here (not a separate script), since this
    # is exactly the step that produces scenario_map. See MATH_MODELS.md §2.6.
    # [REMOVED, per user request] Previously also plotted a synthetic "Accelerated
    # BEV" reference curve (a manual what-if, built via `warp_bev_transition_all_
    # segments` above this block) as a black dashed line -- that computation and its
    # line/legend entry are both gone, and so is `warp_bev_transition_all_segments`
    # itself (src/stock_flow.py), which nothing called any more (2026-10-09).
    # -----------------------------------------------------------------------
    if ("EUR", "BEV") in stock_dict:
        fig, ax = plot_bev_stock_compare_grouped(
            scenario_map=scenario_map,
            selected_scenario=p01.scenario,
            start_year_plotting=p01.start_year_plotting,
            end_year_plotting=p01.end_year_plotting,
            show=False,
        )
        fig_path = fig_dir / "01_bev_transition_check.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostic plot: {fig_path}")
    else:
        print("Skipped BEV transition plot: no ('EUR', 'BEV') key in stock_dict.")

    # -----------------------------------------------------------------------
    # Diagnostic plot 2: actual stock by drivetrain for the selected scenario -- the
    # most direct check of what this stage hands to stage 02. [FIXED, this round]:
    # capped at PLOT_YEAR_MAX=2100 (display-only, see plot_stock_by_drivetrain's
    # docstring) instead of the full raw REMIND horizon (2150).
    # -----------------------------------------------------------------------
    fig, ax = plot_stock_by_drivetrain(stock_dict_df, scenario=p01.scenario)
    fig_path = fig_dir / "01_stock_by_drivetrain.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Diagnostic plot 3: export volume sanity check (C1/C2 fixes' real-world magnitude)
    # -----------------------------------------------------------------------
    fig, ax = plot_export_volume(df_exp_eu)
    fig_path = fig_dir / "01_export_volume_check.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostic plot: {fig_path}")

    # -----------------------------------------------------------------------
    # Persist artifacts for stage 02 (stock-flow)
    # -----------------------------------------------------------------------
    # NOTE, still open: export_df_2005 and df_imp_eu are computed but not persisted --
    # only df_exp_eu is. Confirm whether that's intentional (those two are
    # intermediate/debug-only) before this list is extended.
    saved = save_many(
        scenario_map=scenario_map,
        scenario_df=scenario_df,
        stock_dict=stock_dict,
        stock_dict_df=stock_dict_df,
        df_exp_eu=df_exp_eu,
        root=PROJECT_ROOT,
    )
    print("Saved artifacts:", saved)

    return saved


if __name__ == "__main__":
    main()