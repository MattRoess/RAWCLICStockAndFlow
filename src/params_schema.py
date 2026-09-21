"""
src/params_schema.py
======================

Defines every parameter dataclass used by 00_parameters.py, and the `Params` object
that gets pickled as the `params` artifact.

WHY THIS LIVES HERE AND NOT INSIDE 00_parameters.py
-----------------------------------------------------
Found via an actual end-to-end run, not by inspection: when `00_parameters.py` is run
directly (`python 00_parameters.py`), Python assigns it the module name `__main__`.
Pickle then records every dataclass instance's class as `__main__.Params`,
`__main__.DataPrepParams`, etc. When a DIFFERENT script (`01_data_prep.py`, also run
directly, also `__main__` in its own process) unpickles that artifact, Python looks for
`Params` inside ITS OWN `__main__` namespace -- which doesn't define it -- and unpickling
fails with `AttributeError: Can't get attribute 'Params' on <module '__main__'...>`.
This is a standard, well-known pickle gotcha with classes defined in "run as a script"
files, not specific to this refactor -- but the previous dict-based PARAMS never hit it,
since plain dicts don't carry a class reference back to whichever script created them.

THE FIX: define the classes here, in a real package module (`src/params_schema.py`)
that is never itself executed as `__main__` -- only ever imported. Pickle then records
each class as `src.params_schema.Params` etc., which every stage script can resolve
identically, because every stage already ensures `src/` is on `sys.path` (via each
script's own PROJECT_ROOT resolution) before touching the artifact store. No stage needs
to explicitly `import src.params_schema` for this to work -- pickle imports it
automatically at unpickle time, the same way it would for any other library class.

Verified: pickled `Params` in one process (simulating `00_parameters.py`), unpickled in
a separate process with a different `__main__` (simulating `01_data_prep.py`) -- succeeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ALL_DRIVETRAINS: list[str] = [
    "BEV", "HEV", "PHEV", "Hybrid", "Liquids", "Petrol", "Diesel", "Gases", "FCEV",
]

# ---------------------------------------------------------------------------
# HOW TO EDIT THIS FILE
# ---------------------------------------------------------------------------
# This file holds every number the model uses. Nothing here performs a calculation --
# each entry is a setting that the pipeline stages read. Change a value here and the
# model behaves differently; change nothing and the model stays exactly as it is.
#
# You do NOT need to be a programmer to change a value. Three rules:
#
#   1. Change only what is to the RIGHT of the "=" sign.
#      Correct:   n_draws: int = 50000
#      Wrong:     draws: int = 50000        <- renaming the setting breaks the code
#
#   2. Keep the TYPE the same. A number stays a number (2025, 0.15); text stays in
#      quotes ("report_only"); True and False stay capitalised exactly like that.
#
#   3. Keep the punctuation. Entries inside { } need their commas and colons.
#      A missing comma is the single most common way to break this file.
#
# AFTER EDITING, RUN:   .venv/bin/python code/00_parameters.py
# That regenerates the parameter file the stages read AND checks your edit. If you
# broke something it tells you there -- before any long model run starts.
#
# Each setting below says what it does and whether it is safe to change. Where a note
# says a value is load-bearing, changing it alters results that other settings assume;
# read the note before touching it.
#
# LAYOUT: shared building blocks first, then one section per pipeline stage in the
# order the stages actually run, then the cross-cutting Monte Carlo settings, then the
# container object that holds them all.

# ---------------------------------------------------------------------------
# SHARED BUILDING BLOCKS -- small types reused by several stages below
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsymmetricSpread:
    """
    "Give or take, but not evenly."

    Says how far a value might plausibly stray from its best guess, allowing the two
    directions to differ. `lower` and `upper` are FRACTIONS, not absolute amounts:

        AsymmetricSpread(lower=0.10, upper=0.20)
            on a best guess of 15 years means "as short as 13.5 years (10% less),
            as long as 18 years (20% more), most likely 15."

    Wherever one of these is accepted you may instead write a plain number, which
    means the same spread both ways: 0.15 is identical to
    AsymmetricSpread(lower=0.15, upper=0.15).

    WHEN TO USE THE ASYMMETRIC FORM: when the two directions genuinely differ -- for
    example "cars are unlikely to last much less than expected, but could easily last
    a good deal longer". A flat plus-or-minus would misstate that.

    ONE THING TO EXPECT: with an uneven spread the AVERAGE of the drawn values sits
    slightly to the wider side, not exactly on the best guess. That is correct, not a
    bug -- if you say long lifetimes are more plausible than short ones, the average
    lifetime should indeed come out above the best guess.

    SAFE TO CHANGE: yes. Both numbers must be zero or positive. Zero on both sides
    means "no uncertainty at all -- always use the best guess".
    """
    lower: float
    upper: float

    def validate(self, *, field_name: str) -> list[str]:
        issues: list[str] = []
        if self.lower < 0:
            issues.append(f"{field_name}.lower={self.lower} must be >= 0.")
        if self.upper < 0:
            issues.append(f"{field_name}.upper={self.upper} must be >= 0.")
        return issues


@dataclass(frozen=True)
class WeibullLifetime:
    """
    How long vehicles of one drivetrain survive before being scrapped.

    Vehicles do not all die at one age -- some go early, some last far longer. This
    describes that whole spread with two numbers (a Weibull survival curve):

        scale_lambda  roughly the typical lifetime, in years. Bigger = cars last
                      longer. This is the number you would normally adjust.
        shape_k       how tightly deaths cluster around that typical age. Higher
                      values mean most cars die at a similar age; lower values mean
                      a broad spread of ages. Rarely needs changing.

    SAFE TO CHANGE: yes, but this is one of the most influential settings in the whole
    model -- it drives when vehicles leave the fleet, and therefore all the recycling
    and material-recovery numbers downstream. Both values must be above zero.
    """
    shape_k: float
    scale_lambda: float


@dataclass(frozen=True)
class LifetimeOverride:
    """
    "For these years only, use a different lifetime."

    A temporary replacement for a drivetrain's normal survival curve, covering the
    years from `start_year` to `end_year` inclusive. Outside that window the normal
    lifetime applies again.

    Use it when something makes one period genuinely different -- a scrappage scheme,
    a known quality problem in a particular model generation.

    SAFE TO CHANGE: yes. `start_year` must not be after `end_year`. If you want a
    change that starts and never ends, this is the wrong type -- see
    OpenEndedLifetimeChange below.
    """
    start_year: int
    end_year: int
    shape_k: float
    scale_lambda: float


@dataclass(frozen=True)
class OpenEndedLifetimeChange:
    """
    "From this year onward, use a different lifetime -- permanently."

    Same idea as LifetimeOverride above, but with no end: once `start_year` arrives the
    new lifetime applies for the rest of the run. This is what the scenarios in stage
    03_02 use (e.g. "BEV_longer": from 2027, BEVs last longer than in the base case).

    WHY TWO SEPARATE TYPES: stage 02 needs a bounded window, stage 03_02 needs an
    open-ended change. Both conventions are genuinely in use; this is not a duplicate.

    SAFE TO CHANGE: yes.
    """
    start_year: int
    shape_k: float
    scale_lambda: float


@dataclass(frozen=True)
class ScenarioSpec:
    """
    One complete, self-contained definition of a `03_02_adjustedflows.py` sensitivity
    scenario. Every override field is SPARSE and ADDITIVE over this stage's base
    `StockFlowParams` values: a missing drivetrain entry means "use the base value for
    that drivetrain", not "zero it out" -- this mirrors how
    `run_flow_driven_model_with_outflow_disaggregation` itself already treats a missing
    `lifetime_change_by_drv[drv]` entry (falls back to `lifetime_by_drv[drv]`). This is
    what lets e.g. `ICEV_shorter` specify ONLY Diesel/Petrol's changed lifetime, without
    having to also restate BEV's unchanged one (the pre-refactor `LIFETIME_CHANGE_BY_
    DRV_*` dicts redundantly restated every drivetrain, changed or not).

    Exactly one of `inflow_drivetrain_shares_final` / `inflow_segment_shares_final`
    should be set for a scenario that changes the inflow composition; neither set means
    "reuse the BAU scenario's resolved inflow unchanged" (what all five lifetime/loss/
    stock scenarios do).
    """
    # The scenario's identifier. It appears in every artifact and figure
    # filename this scenario produces, so keep it short and filename-safe.
    # It must match the key this spec is stored under in `scenarios`.
    name: str

    # Inflow-composition transform (at most one of the two pairs below should be set):
    #   - `inflow_drivetrain_shares_final`: changes the DRIVETRAIN mix of inflow
    #     (maps onto `tweak_inflow_drivetrain_shares` in 03_02_adjustedflows.py) --
    #     used by BEV_only.
    #   - `inflow_segment_shares_drivetrain` + `inflow_segment_shares_final`: changes
    #     ONE drivetrain's INTERNAL segment mix, holding the overall drivetrain mix
    #     fixed (maps onto `tweak_inflow_segment_shares_within_drivetrain`) -- used by
    #     BAU (its own segment-mix baseline) and the four BEV segment-profile
    #     scenarios (A_F, JA_JF, Large, Small).
    inflow_drivetrain_shares_final: dict[str, float] | None = None
    inflow_segment_shares_drivetrain: str | None = None
    inflow_segment_shares_final: dict[str, float] | None = None

    # [NEW] Monte Carlo uncertainty around `inflow_segment_shares_final` (the FUTURE
    # segment-mix target, applied for years >= AdjustedFlowsParams.scenario_start_year)
    # -- deliberately NOT applied to the real historic segment split (years before
    # scenario_start_year), which comes from actual registration data and stays fully
    # deterministic. `None` (default) means no segment-share uncertainty for this
    # scenario -- opt-in, same sparse convention as every other override on this class.
    #
    # A per-segment Triangular(mode*(1-lower), mode, mode*(1+upper)) is sampled
    # INDEPENDENTLY for each of the 12 segments (reusing this scenario's own
    # `inflow_segment_shares_final` values as each segment's mode), one draw per Monte
    # Carlo trial, then the whole 12-segment draw is renormalized (divided by its own
    # sum) so it sums to exactly 1 -- every segment moves proportionally on every draw,
    # not just whichever one happens to be picked to "absorb" the difference. Requires
    # `inflow_segment_shares_final` to also be set (validated) -- there is nothing to
    # sample around otherwise. Same `AsymmetricSpread` type as
    # `StockFlowParams.lifetime_scale_lambda_relative_spread` (a plain float means
    # symmetric spread); see `03_02_adjustedflows.py`'s
    # `sample_future_segment_share_inflow_draws` for exactly how this gets sampled and
    # turned into per-draw inflow values, and `cohort_flow_mc.py`'s
    # `inflow_draws_by_group` parameter for how it's fed into the vectorized engine.
    inflow_segment_share_spread: AsymmetricSpread | float | None = None

    # Sparse lifetime override: only drivetrains that actually change need an entry.
    lifetime_change_by_drv: dict[str, OpenEndedLifetimeChange] = field(default_factory=dict)

    # Sparse share overrides: only drivetrains that actually change need an entry.
    export_share_overrides: dict[str, float] = field(default_factory=dict)
    unknown_whereabouts_share_overrides: dict[str, float] = field(default_factory=dict)

    # Flat inflow multiplier from `AdjustedFlowsParams.lifetime_change_start_year`
    # onward (the vehicle pipeline's `stock_modifier_2027`, expressed generically here
    # since the year itself is also a parameter, not hardcoded to literally "2027").
    stock_modifier: float = 1.0

    def validate(self, *, valid_segments: set[str]) -> list[str]:
        issues: list[str] = []
        if self.inflow_drivetrain_shares_final is not None and self.inflow_segment_shares_final is not None:
            issues.append(
                f"adjusted_flows.scenarios['{self.name}']: both inflow_drivetrain_shares_final "
                f"and inflow_segment_shares_final are set -- at most one should be."
            )
        if self.inflow_drivetrain_shares_final is not None:
            total = sum(self.inflow_drivetrain_shares_final.values())
            if not (0.999 <= total <= 1.001):
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}'].inflow_drivetrain_shares_final "
                    f"sums to {total:.6f}, expected 1.0."
                )
        if self.inflow_segment_shares_final is not None:
            if self.inflow_segment_shares_drivetrain is None:
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}']: inflow_segment_shares_final is "
                    f"set but inflow_segment_shares_drivetrain is None."
                )
            missing_segs = set(self.inflow_segment_shares_final) - valid_segments
            if missing_segs:
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}'].inflow_segment_shares_final has "
                    f"unrecognized segments: {sorted(missing_segs)}."
                )
            total = sum(self.inflow_segment_shares_final.values())
            if not (0.999 <= total <= 1.001):
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}'].inflow_segment_shares_final sums "
                    f"to {total:.6f}, expected 1.0."
                )
        if self.inflow_segment_share_spread is not None:
            if self.inflow_segment_shares_final is None:
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}']: inflow_segment_share_spread is set "
                    f"but inflow_segment_shares_final is None -- nothing to sample around."
                )
            if isinstance(self.inflow_segment_share_spread, AsymmetricSpread):
                issues += self.inflow_segment_share_spread.validate(
                    field_name=f"adjusted_flows.scenarios['{self.name}'].inflow_segment_share_spread"
                )
            elif self.inflow_segment_share_spread < 0:
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}'].inflow_segment_share_spread = "
                    f"{self.inflow_segment_share_spread} must be >= 0."
                )
        for drv, change in self.lifetime_change_by_drv.items():
            if change.scale_lambda <= 0 or change.shape_k <= 0:
                issues.append(
                    f"adjusted_flows.scenarios['{self.name}'].lifetime_change_by_drv['{drv}'] "
                    f"has non-positive shape_k/scale_lambda."
                )
        for name_, mapping in (
            ("export_share_overrides", self.export_share_overrides),
            ("unknown_whereabouts_share_overrides", self.unknown_whereabouts_share_overrides),
        ):
            for drv, share in mapping.items():
                if not (0.0 <= share <= 1.0):
                    issues.append(
                        f"adjusted_flows.scenarios['{self.name}'].{name_}['{drv}'] = {share} "
                        f"is outside [0, 1]."
                    )
        if self.stock_modifier <= 0:
            issues.append(f"adjusted_flows.scenarios['{self.name}'].stock_modifier must be positive.")
        return issues


# The 12 vehicle segments (A-F, JA-JF) -- matches `MaterialsParams.segment_map`'s keys.
_ADJUSTED_FLOWS_SEGMENTS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "JA", "JB", "JC", "JD", "JE", "JF")

# ---------------------------------------------------------------------------
# STAGE 01 -- Data prep  (code/01_data_prep.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 01 -- Data prep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataPrepParams:
    """
    The very first stage: reads the raw REMIND fleet projections and the trade data,
    trims them to Europe, fills in the years REMIND does not supply, and hands a clean
    fleet trajectory to stage 02.

    The setting that matters most here is `scenario` -- it decides which possible
    future the entire pipeline is built on.
    """

    # WHICH REMIND SCENARIO THE WHOLE MODEL RUNS ON.
    # REMIND is the external energy-system model that tells this one how big the
    # European car fleet should be, year by year, and what mix of drivetrains it holds.
    # Everything downstream inherits this choice.
    # SAFE TO CHANGE: yes -- pick any name from `scenario_list` below. This is one of
    # the most consequential single edits in the file.
    scenario: str = "npi25"

    # The REMIND scenarios available to choose from.
    # SAFE TO CHANGE: only alongside `remind_scenario_files` below -- a name here with
    # no matching file entry cannot be loaded.
    scenario_list: tuple[str, ...] = ("b650", "npi25", "ssp2L", "ssp2M", "ssp1")

    # Where each scenario's REMIND data file actually lives.
    # SAFE TO CHANGE: yes, when new REMIND runs arrive. Paths are relative to the raw
    # data folder set further down.
    remind_scenario_files: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "b650": ("REMIND/REMIND_generic_SSP2-EU21-PkBudg650.mif", ";"),
        "npi25": ("REMIND/REMIND_generic_SSP2-EU21-NPi2025.mif", ";"),
        "ssp2L": ("REMIND/REMIND_generic_C_SMIPv08-L-SSP2-PkPrice400-def-rem-6.mif", ";"),
        "ssp2M": ("REMIND/REMIND_generic_C_SMIPv08-M-SSP2-NPi2025-def-rem-6.mif", ";"),
        "ssp1": ("REMIND/REMIND_generic_C_SMIPv08-VLLO-SSP1-PkPrice500-def-rem-6.mif", ";"),
    })

    # The first and last year the model covers.
    # Years before REMIND's own data begins are reconstructed by the model (see the
    # pre-2015 settings below), so the earliest years are estimates, not data.
    # SAFE TO CHANGE: with care. Start must be before end.
    start_year_model: int = 1950
    end_year_model: int = 2070

    # The first and last year drawn on this stage's charts. Display only -- the model
    # still computes the full range above.
    # SAFE TO CHANGE: yes.
    start_year_plotting: int = 2015
    end_year_plotting: int = 2070

    # From this year onward, vehicle composition is held constant at its last known
    # value -- nobody has credible material-composition forecasts beyond it.
    # SAFE TO CHANGE: yes, but pushing it later means inventing composition trends.
    composition_extend_from_year: int = 2050

    # The year from which the model applies the accelerated fleet-turnover assumption.
    # SAFE TO CHANGE: yes.
    accelerating_year: int = 2026

    # Numbers smaller than this are treated as zero when tidying the REMIND data.
    # Guards against meaningless dust like 0.0000001 vehicles.
    # SAFE TO CHANGE: rarely needed. Too large a value would delete real small
    # quantities -- early-year FCEV or Gases, for instance.
    threshold: float = 1e-4

    # The row label inside the REMIND file identifying passenger-car stock. This is how
    # the loader finds the right rows among thousands.
    # SAFE TO CHANGE: no, unless REMIND itself renames its variables.
    prefix: str = "Stock|Transport|Pass|Road|LDV"

    # The countries counted as "Europe" for this study, by full name.
    # SAFE TO CHANGE: yes, but it must stay consistent with the ISO-2 list below --
    # the two describe the same set in two different notations.
    eu_countries: tuple[str, ...] = (
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
        "Malta", "Netherlands", "Poland", "Portugal", "Romania",
        "Slovakia", "Slovenia", "Spain", "Sweden",
    )

    # The same country set, as two-letter codes, for data files that use codes instead
    # of names.
    # SAFE TO CHANGE: yes, but keep it aligned with the list above.
    eu_countries_iso2: tuple[str, ...] = (
        "AT", "BE", "BG", "HR", "CY", "CZ",
        "DK", "EE", "FI", "FR", "DE", "GR",
        "HU", "IE", "IT", "LV", "LT", "LU",
        "MT", "NL", "PL", "PT", "RO",
        "SK", "SI", "ES", "SE",
    )

    # Which REMIND world regions add up to Europe. REMIND does not have a single
    # "EU" region, so these are summed.
    # SAFE TO CHANGE: no, unless REMIND changes its regional definitions.
    remind_regions: tuple[str, ...] = ("DEU", "ECE", "ECS", "ENC", "ESC", "ESW", "EWN", "FRA", "UKI", "NEN")

    # The drivetrain names REMIND uses. Coarser than ours: it has one "Liquids"
    # category covering both petrol and diesel, and one "Hybrid" covering HEV and PHEV.
    # SAFE TO CHANGE: no, these are REMIND's own names.
    remind_technology: tuple[str, ...] = ("BEV", "Hybrid", "Liquids", "Gases", "FCEV")

    # The drivetrain names THIS model uses -- finer than REMIND's. Splitting REMIND's
    # coarse categories into these is a large part of what stage 03_01 does.
    # SAFE TO CHANGE: no, without matching changes throughout the pipeline.
    target_technology: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Petrol", "Diesel")

    # Norway and Iceland's share of REMIND's Northern-Europe region, used to subtract
    # them out -- they are not in the study's EU scope.
    # SAFE TO CHANGE: only with a better estimate. It is a fraction between 0 and 1.
    norway_iceland_share_of_neu: float = 0.1139

    # The UK and Ireland's share of REMIND's Europe region, used the same way.
    # SAFE TO CHANGE: only with a better estimate.
    uk_ireland_share_of_eur: float = 0.1193

    # HOW TO FILL IN HISTORY BEFORE 2015. Some REMIND scenarios do not supply the past;
    # this decides what to do about that.
    #     "donor_scenario_average"  average the scenarios that DO have history and use
    #                               that for the ones that do not (current)
    # SAFE TO CHANGE: yes, if another method is implemented. History is shared reality,
    # so borrowing it across scenarios is reasonable -- scenarios should differ in the
    # future, not the past.
    pre2015_history_method: str = "donor_scenario_average"

    # The scenarios MISSING pre-2015 history, which therefore receive it.
    # SAFE TO CHANGE: only if REMIND's data coverage changes.
    pre2015_history_target_scenarios: tuple[str, ...] = ("ssp2L", "ssp2M", "ssp1")

    # The scenarios that HAVE pre-2015 history and supply it to the others.
    # SAFE TO CHANGE: only if REMIND's data coverage changes.
    pre2015_history_donor_scenarios: tuple[str, ...] = ("b650", "npi25")

    # The year where borrowed history stops and each scenario's own data takes over.
    # SAFE TO CHANGE: yes, but it must match where the donor data actually ends, or the
    # join leaves a visible step in the fleet curve.
    pre2015_history_splice_year: int = 2015

    # The vehicle size classes used in the source data, before they are mapped onto
    # this model's A-F / JA-JF segments.
    # SAFE TO CHANGE: no, unless the source data changes its class names.
    target_class_detail: tuple[str, ...] = (
        "Large Car and SUV", "Van", "Compact Car", "Midsize Car", "Mini Car", "Subcompact Car",
    )

    # Which columns of the REMIND data identify a row, and what to call them here.
    # SAFE TO CHANGE: no, unless the source format changes.
    attribute_list: tuple[str, ...] = ("Region", "technology")

    key_names: tuple[str, ...] = ("Region", "Drivetrain")

    # Where raw input data is read from, and where processed results are written.
    # Both are relative to the `code/` directory.
    # SAFE TO CHANGE: only if you actually move those folders.
    input_dir: str = "../data/raw/"

    output_dir: str = "../data/processed/"

    # The workbook of used-vehicle export trade data.
    # SAFE TO CHANGE: yes, when a newer version arrives.
    export_data_file_name: str = "usedvehicles_v1.2.xlsx"

    # REMIND reports the fleet only every five years; this decides how the years in
    # between are filled in.
    #     "cubic"   a smooth curve through the known points (current)
    #     "linear"  straight lines between them
    # SAFE TO CHANGE: yes. "cubic" looks more natural but can overshoot slightly where
    # the trend turns sharply; "linear" never overshoots but has visible kinks.
    stock_interpolation_method: str = "cubic"

    # The range of years to keep from the export trade data. The upper bound is
    # EXCLUSIVE -- 2023 means "up to and including 2022".
    # SAFE TO CHANGE: yes, as new trade data arrives.
    export_min_year: int = 2005
    export_max_year_exclusive: int = 2023

    # Fixes for wrong country codes in the source data -- it writes "IRE" for Ireland,
    # where the correct code is "IRL". Without this, Ireland's vehicles are silently
    # dropped.
    # SAFE TO CHANGE: yes, add an entry whenever you find another bad code.
    iso3_corrections: dict[str, str] = field(default_factory=lambda: {"IRE": "IRL"})

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.start_year_model >= self.end_year_model:
            issues.append("data_prep: start_year_model must be strictly before end_year_model.")
        if self.start_year_plotting > self.end_year_plotting:
            issues.append("data_prep: start_year_plotting must not be after end_year_plotting.")
        if self.stock_interpolation_method not in {"cubic", "pchip"}:
            issues.append(
                f"data_prep.stock_interpolation_method={self.stock_interpolation_method!r} "
                f"is not one of ['cubic', 'pchip']."
            )
        if set(self.remind_scenario_files) != set(self.scenario_list):
            issues.append(
                "data_prep: remind_scenario_files keys and scenario_list disagree -- "
                f"files only: {sorted(set(self.remind_scenario_files) - set(self.scenario_list))}, "
                f"list only: {sorted(set(self.scenario_list) - set(self.remind_scenario_files))}."
            )
        if self.scenario not in self.scenario_list:
            issues.append(f"data_prep.scenario={self.scenario!r} is not in scenario_list.")
        if not (0.0 <= self.norway_iceland_share_of_neu < 1.0):
            issues.append(
                f"data_prep.norway_iceland_share_of_neu={self.norway_iceland_share_of_neu!r} "
                f"must be in [0.0, 1.0) -- it's a share of REMIND's 'NEU' region, not a "
                f"percentage (0.1139, not 11.39)."
            )
        if not (0.0 <= self.uk_ireland_share_of_eur < 1.0):
            issues.append(
                f"data_prep.uk_ireland_share_of_eur={self.uk_ireland_share_of_eur!r} "
                f"must be in [0.0, 1.0) -- it's a share of REMIND's 'EUR' region, not a "
                f"percentage (0.1193, not 11.93)."
            )
        _KNOWN_PRE2015_METHODS = {"donor_scenario_average"}
        if self.pre2015_history_method not in _KNOWN_PRE2015_METHODS:
            issues.append(
                f"data_prep.pre2015_history_method={self.pre2015_history_method!r} is "
                f"not one of {sorted(_KNOWN_PRE2015_METHODS)} -- if you've added a new "
                f"method to data_prep.py's fill_missing_pre_year_history, add its name "
                f"here too."
            )
        if self.pre2015_history_method == "donor_scenario_average" and not self.pre2015_history_donor_scenarios:
            issues.append(
                "data_prep.pre2015_history_donor_scenarios is empty -- "
                "pre2015_history_method='donor_scenario_average' needs at least one."
            )
        missing_donors = set(self.pre2015_history_donor_scenarios) - set(self.scenario_list)
        if missing_donors:
            issues.append(
                f"data_prep.pre2015_history_donor_scenarios contains scenario(s) not "
                f"in scenario_list: {sorted(missing_donors)}."
            )
        missing_targets = set(self.pre2015_history_target_scenarios) - set(self.scenario_list)
        if missing_targets:
            issues.append(
                f"data_prep.pre2015_history_target_scenarios contains scenario(s) not "
                f"in scenario_list: {sorted(missing_targets)}."
            )
        overlap_donor_target = set(self.pre2015_history_donor_scenarios) & set(self.pre2015_history_target_scenarios)
        if overlap_donor_target:
            issues.append(
                f"data_prep.pre2015_history_donor_scenarios and _target_scenarios "
                f"overlap ({sorted(overlap_donor_target)}) -- a scenario can't donate "
                f"history to itself."
            )
        if self.export_min_year >= self.export_max_year_exclusive:
            issues.append("data_prep: export_min_year must be strictly before export_max_year_exclusive.")
        if len(self.eu_countries_iso2) != len(self.eu_countries):
            issues.append(
                f"data_prep: eu_countries_iso2 has {len(self.eu_countries_iso2)} entries "
                f"but eu_countries has {len(self.eu_countries)} -- they must list the same "
                f"countries 1:1 (full name vs. ISO2 code)."
            )
        if len(set(self.eu_countries_iso2)) != len(self.eu_countries_iso2):
            issues.append("data_prep: eu_countries_iso2 contains duplicate codes.")
        return issues


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StockFlowParams:
    """
    Everything stage 02 needs: how big the fleet should be, how long vehicles last,
    what happens to them when they are scrapped, and how uncertain each of those is.

    Read it in four parts:
      1. the model's time span and vehicle lifetimes
      2. what happens to retired vehicles (collected / exported / untraceable)
      3. rules for how strictly to follow the prescribed fleet path, including
         phase-outs and hard stops
      4. uncertainty settings, used only when the Monte Carlo analysis runs
    """

    # The last year the model simulates.
    # SAFE TO CHANGE: yes, but everything downstream inherits it -- charts, cumulative
    # totals and the material results in stage 04 all stop here.
    model_end_year: int = 2070

    # The last year for which real vehicle-export trade data exists. After this year
    # the model has to assume export behaviour rather than read it.
    # SAFE TO CHANGE: only when newer trade data actually arrives.
    last_exp_data_year: int = 2022

    # The oldest vehicle age, in years, the model tracks. Anything older is treated as
    # gone. 50 is generous -- almost no car survives that long.
    # SAFE TO CHANGE: rarely needed. Raising it costs memory and time for vehicles that
    # have essentially all been scrapped already.
    init_max_age: int = 50

    # HOW LONG VEHICLES LAST, per drivetrain. See WeibullLifetime in the shared
    # building blocks above for what the two numbers mean -- in short, the second one
    # (scale_lambda) is roughly the typical lifetime in years.
    # Every drivetrain currently uses the same curve: typical life ~18 years.
    # SAFE TO CHANGE: yes -- and this is one of the highest-impact settings in the
    # model. Longer lifetimes mean vehicles leave the fleet later, which delays all the
    # recycling and material-recovery numbers downstream.
    lifetime_by_drv: dict[str, WeibullLifetime] = field(default_factory=lambda: {
        "Hybrid": WeibullLifetime(3.0, 18.0),
        "PHEV":   WeibullLifetime(3.0, 18.0),
        "HEV":    WeibullLifetime(3.0, 18.0),
        "BEV":    WeibullLifetime(3.0, 18.0),
        "Liquids": WeibullLifetime(3.0, 18.0),
        "Petrol": WeibullLifetime(3.0, 18.0),
        "Diesel": WeibullLifetime(3.0, 18.0),
        "Gases":  WeibullLifetime(3.0, 18.0),
        "FCEV":   WeibullLifetime(3.0, 18.0),
    })

    # Optional "for these years only, use a different lifetime" exceptions.
    # `None` for a drivetrain means no exception -- use the normal lifetime above.
    # SAFE TO CHANGE: yes. Set one only if you have a concrete reason (a scrappage
    # scheme, a known bad model generation). See LifetimeOverride above.
    lifetime_override_by_drv: dict[str, LifetimeOverride | None] = field(default_factory=lambda: {
        drv: None for drv in ALL_DRIVETRAINS
    })

    # Of the vehicles that leave the fleet, the fraction that simply cannot be traced --
    # not recorded as recycled, not recorded as exported. 0.43 means 43%.
    # BEV is far lower (10%) because battery vehicles are tracked much more closely.
    # SAFE TO CHANGE: yes. Each value is a fraction between 0 and 1. This is an
    # ESTIMATE, not a measurement -- which is why it also carries the widest
    # uncertainty of the three outflow shares (see further below).
    unknown_whereabouts_share: dict[str, float] = field(default_factory=lambda: {
        "BEV": 0.1,
        "HEV": 0.43, "PHEV": 0.43, "FCEV": 0.43, "Gases": 0.43,
        "Diesel": 0.43, "Petrol": 0.43, "Liquids": 0.43, "Hybrid": 0.43,
    })

    # Of the vehicles that leave the fleet, the fraction exported out of the EU as
    # second-hand cars. 0.08 means 8%. BEVs are exported much less (2%) so far.
    # SAFE TO CHANGE: yes, values between 0 and 1. Note the two entries marked
    # PLACEHOLDER -- those are guesses standing in until real numbers exist.
    export_share_by_drv: dict[str, float] = field(default_factory=lambda: {
        "BEV": 0.02,
        "HEV": 0.08, "PHEV": 0.08, "Diesel": 0.08, "Petrol": 0.08,
        "Hybrid": 0.08, "Liquids": 0.08,
        "FCEV": 0.08,   # PLACEHOLDER, not a verified real value.
        "Gases": 0.08,  # PLACEHOLDER, not a verified real value.
    })

    # Of the vehicles that leave the fleet, the fraction properly collected for
    # recycling. 0.49 means 49%. BEV is much higher (88%) because batteries are
    # valuable and regulated.
    # These three shares -- collected, exported, unknown -- are what every retired
    # vehicle is split into, so they work together. The model normalises them so they
    # add up, but if you set values that are wildly inconsistent you will get a
    # normalised result you did not intend.
    # SAFE TO CHANGE: yes, values between 0 and 1.
    collected_share_by_drv: dict[str, float] = field(default_factory=lambda: {
        "BEV": 0.88,
        "HEV": 0.49, "PHEV": 0.49, "Diesel": 0.49, "Petrol": 0.49,
        "Hybrid": 0.49, "Liquids": 0.49,
        "FCEV": 0.49,   # PLACEHOLDER, matches export_share_by_drv's own placeholder note.
        "Gases": 0.49,  # PLACEHOLDER, matches export_share_by_drv's own placeholder note.
    })

    # WHAT TO DO WHEN THE MATHS ASKS FOR NEGATIVE SALES.
    # The model works out sales as "the fleet size we are told to hit, minus the cars
    # that survived from previous years". When a target fleet shrinks faster than old
    # cars are scrapped, that subtraction goes negative -- which would mean un-selling
    # cars. Two choices:
    #     "report_only"     record the negative number for inspection, but simulate
    #                       zero sales that year. Nothing is forced. (current)
    #     "clip_to_target"  force extra vehicles out of the fleet so the target is met
    #                       exactly.
    # SAFE TO CHANGE: yes, but understand which question you are asking. "report_only"
    # keeps the fleet honest and lets it sit above target; "clip_to_target" keeps the
    # target exact by scrapping vehicles that would not otherwise have gone.
    negative_inflow_policy: str = "report_only"

    # HOW STRICTLY TO FOLLOW THE PRESCRIBED FLEET PATH, per drivetrain. Three options:
    #     "remind_soft"       follow the target, but never force negative sales
    #                         (uses the policy setting above). This is the default.
    #     "remind_literal"    follow the target exactly, forcing vehicles out if needed.
    #     "inflow_phaseout"   follow the target, but additionally cap this drivetrain's
    #                         share of total sales -- see the next setting.
    # A drivetrain not listed here uses "remind_soft".
    # SAFE TO CHANGE: yes. NOTE that "inflow_phaseout" below is configured for Liquids
    # but will do nothing until this is switched from "remind_soft" to
    # "inflow_phaseout".
    inflow_mode_by_drv: dict[str, str] = field(default_factory=lambda: {
        "Liquids": "remind_soft",
    })

    # THE PHASE-OUT RULE: (from this year, at most this share of all new vehicles).
    # (2035, 0.10) reads as "from 2035 onward, Liquids may be at most 10% of total
    # sales" -- the EU phase-out expressed directly.
    # ONLY APPLIES if the drivetrain's mode above is set to "inflow_phaseout".
    # SAFE TO CHANGE: yes. The share is a fraction between 0 and 1.
    inflow_phaseout_by_drv: dict[str, tuple[int, float]] = field(default_factory=lambda: {
        "Liquids": (2035, 0.10),
    })

    # Uncertainty on that cap, as (lowest, most likely, highest).
    # (0.08, 0.10, 0.15) means the 10% cap could plausibly be as tight as 8% or as
    # loose as 15%. Used only by the uncertainty analysis; the plain run uses the
    # middle value.
    # SAFE TO CHANGE: yes. Keep them in order: lowest <= most likely <= highest.
    inflow_phaseout_max_share_triangular_by_drv: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        "Liquids": (0.08, 0.10, 0.15),
    })

    # "NO MORE SALES AT ALL FROM THIS YEAR." An absolute stop, not a cap.
    # Liquids and Hybrid stop in 2050.
    # This overrides everything else -- it is the strongest statement in this file
    # about a drivetrain's future.
    # SAFE TO CHANGE: yes, but be sure you mean a hard stop rather than a phase-out.
    hard_zero_inflow_from_year_by_drv: dict[str, int] = field(default_factory=lambda: {
        "Liquids": 2050,
        "Hybrid": 2050,
    })

    # "NO SALES BEFORE THIS YEAR." The mirror of the setting above, for the past.
    # BEV is 2011 because battery vehicles were not sold in volume before then; without
    # this the model's reconstruction of history would invent early BEVs.
    # SAFE TO CHANGE: only to correct a factual error about when sales really began.
    hard_zero_inflow_until_year_by_drv: dict[str, int] = field(default_factory=lambda: {
        "BEV": 2011,
    })

    # DOES STAGE 02'S INFLOW UNCERTAINTY REACH STAGE 03_02?
    #
    # Stage 02 samples how large the vehicle fleet is, and therefore how many cars
    # are bought each year. For a long time none of that reached stage 03_02: the
    # table between them holds one number per row with no room for draws, so total
    # BEV inflow arrived varying by 0.000001% where stage 02 had it varying by 9.6%.
    # Every downstream inflow band -- 03_02, 04_01, 04_02 -- was too narrow.
    #
    # True  -- stage 02's actual per-draw inflow is carried across and used. This is
    #          the correct behaviour and the default.
    # False -- the old behaviour: 03_02 starts from a single fixed inflow trajectory.
    #          Kept only so a run can be checked against results produced before this
    #          was fixed; it reproduces them exactly.
    #
    # WHAT CHANGES WHEN IT IS ON. Inflow bands widen everywhere downstream. Means
    # also shift slightly, by up to +0.228 million vehicles per year for Liquids
    # around 2035, because each draw is floored at zero individually rather than the
    # single average trajectory being floored once. That is the correct Monte Carlo
    # answer, not a side effect -- see documentation/DESIGN_inflow_uncertainty_
    # propagation.md, section 6.
    #
    # WHAT IT COSTS. One extra stage-02 Monte Carlo pass per 03_02 run.
    #
    # SAFE TO CHANGE: yes. Leave it True unless you are deliberately reproducing old
    # numbers.
    propagate_stage02_inflow_uncertainty: bool = True

    # WHICH COARSE DRIVETRAIN EACH FINE ONE INHERITS ITS INFLOW UNCERTAINTY FROM.
    #
    # THE PROBLEM THIS SOLVES. Stage 02 works with five coarse drivetrains -- BEV,
    # Liquids, Hybrid, Gases, FCEV -- and that is where the uncertainty about HOW
    # MANY vehicles are sold is sampled. Stages 03_01 and 03_02 work with finer
    # ones, splitting Liquids into Petrol and Diesel and Hybrid into HEV and PHEV.
    # Without a mapping, the finer stages have no way to ask "how uncertain was
    # this drivetrain's sales volume?", because the drivetrain they are modelling
    # does not exist upstream.
    #
    # WHAT INHERITANCE MEANS HERE, AND WHAT IT DELIBERATELY DOES NOT TOUCH.
    # Petrol and Diesel both inherit Liquids' volume movement: in a Monte Carlo
    # draw where Europe buys 8% fewer liquid-fuel cars, both petrol and diesel
    # sales move down together, because they are the same market. What this does
    # NOT do is decide how Liquids divides between petrol and diesel -- that split,
    # and its own uncertainty and development over time, is stage 03_01's work and
    # is left completely untouched. The two effects compose: 03_01 says how the
    # cake is cut, stage 02 says how big the cake is.
    #
    # SAFE TO CHANGE: only if the drivetrain lists themselves change. Every fine
    # drivetrain used downstream needs an entry, and every value must be a
    # drivetrain stage 02 actually models.
    # THIS MUST MATCH WHERE EACH DRIVETRAIN'S VOLUME ACTUALLY COMES FROM.
    # HEV maps to Liquids, not Hybrid, because HEV is carved OUT of Liquids -- see
    # `disaggregation.hev_carved_from_liquids`. It was left pointing at Hybrid when
    # that change was made, and the symptom was visible in the figures: HEV's
    # uncertainty band nearly vanished. Measured, HEV's spread was 1.78% against
    # 6.77% for Diesel and 10.93% for Petrol, because it was being handed the Hybrid
    # group's deviation -- sized for a ~1.3 million quantity -- spread across its own
    # ~3.9 million level. Petrol and Diesel were over-spread for the mirror-image
    # reason: the whole Liquids deviation landed on a base that no longer included HEV.
    #
    # The sum-to-one guard in 03_02 cannot catch this. Shares still sum to 1 inside
    # each group whichever group a drivetrain is put in; conservation says nothing
    # about whether it is the RIGHT group. Only the figures showed it.
    #
    # If `hev_carved_from_liquids` is set False to reproduce an old result, HEV must
    # be moved back to Hybrid here or its uncertainty will be wrong again.
    #
    # SAFE TO CHANGE: only if the drivetrain lists themselves change. Every fine
    # drivetrain used downstream needs an entry, and every value must be a
    # drivetrain stage 02 actually models.
    inflow_uncertainty_parent_by_drv: tuple[tuple[str, str], ...] = (
        ("BEV", "BEV"),
        ("Petrol", "Liquids"),
        ("Diesel", "Liquids"),
        ("HEV", "Liquids"),      # carved out of Liquids, so its volume varies with Liquids
        ("PHEV", "Hybrid"),      # REMIND's "Hybrid electric" IS the plug-in hybrid
    )

    # HOW UNCERTAIN THE LIFETIMES ARE, per drivetrain, as a fraction: 0.15 means
    # "give or take 15%". Used only by the uncertainty analysis -- the ordinary run
    # always uses the exact lifetimes set above.
    # This is consistently the single biggest driver of uncertainty in the results.
    # SAFE TO CHANGE: yes. A plain number means the same both ways; use
    # AsymmetricSpread(lower=..., upper=...) when the two directions differ.
    lifetime_scale_lambda_relative_spread: dict[str, float | AsymmetricSpread] = field(default_factory=lambda: {
        # PLACEHOLDER values -- every drivetrain wrapped in AsymmetricSpread(lower,
        # upper) rather than a plain float, so per-drivetrain asymmetry is a one-line
        # edit here (change lower/upper independently) instead of a type change.
        # lower=upper=0.15 for every entry below is numerically IDENTICAL to the old
        # flat `0.15` float default -- this is a mechanical type conversion, not yet a
        # real asymmetric belief about any drivetrain's lifetime. Tune lower/upper
        # per drivetrain once real uncertainty ranges are available (e.g. "BEVs are
        # unlikely to die much earlier than expected, but could plausibly last
        # noticeably longer" -> lower=0.10, upper=0.25).
        "Hybrid":   AsymmetricSpread(lower=0.25, upper=0.40),
        "PHEV":     AsymmetricSpread(lower=0.25, upper=0.40),
        "HEV":      AsymmetricSpread(lower=0.25, upper=0.40),
        "BEV":      AsymmetricSpread(lower=0.25, upper=0.40),
        "Liquids":  AsymmetricSpread(lower=0.25, upper=0.40),
        "Petrol":   AsymmetricSpread(lower=0.25, upper=0.40),
        "Diesel":   AsymmetricSpread(lower=0.25, upper=0.40),
        "Gases":    AsymmetricSpread(lower=0.25, upper=0.40),
        "FCEV":     AsymmetricSpread(lower=0.25, upper=0.40),
    })

    # THE YEAR UNCERTAINTY ABOUT FLEET SIZE BEGINS.
    # Before this year every simulation run uses exactly the known fleet size -- the
    # past is not a forecast, so it gets no error bars. From this year on, runs start
    # to differ from one another.
    # SAFE TO CHANGE: yes. Set it to the last year you consider observed rather than
    # projected.
    stock_target_uncertainty_start_year: int = 2025

    # HOW FAST THAT UNCERTAINTY IS ALLOWED TO GROW: 0.005 means half a percentage point
    # per year. Uncertainty does not appear all at once in the start year -- it widens
    # gradually, because a forecast does not become wrong overnight.
    # A consequence worth knowing: a large spread takes many years to arrive in full
    # (a 15% spread needs ~30 years at this rate), while a small one arrives quickly.
    # SAFE TO CHANGE: yes. Must be above zero, or the uncertainty would never arrive.
    stock_target_ramp_max_rate_per_year: float = 0.005

    # DOES BUYING ONE KIND OF CAR MEAN NOT BUYING ANOTHER?
    # True  -- yes. A buyer picks ONE drivetrain, so if more people choose BEVs, fewer
    #          choose something else. The drivetrain shares are tied together and
    #          always add up. (current, and the realistic assumption)
    # False -- no. Each drivetrain varies independently, as if buyers could choose
    #          several at once. This is the older, simpler behaviour.
    # SAFE TO CHANGE: yes. Setting it to False reproduces the previous results exactly,
    # so it is a safe way to compare against older numbers.
    stock_target_correlated_mix: bool = True

    # HOW UNCERTAIN THE TOTAL SIZE OF THE FLEET IS -- one shared figure covering every
    # drivetrain at once: "there might be 2% more or fewer cars in Europe overall".
    # This is separate from the per-drivetrain uncertainty below, which is about the
    # MIX rather than the TOTAL.
    # WHY 2% AND NOT MORE: the total-fleet effect moves every drivetrain the same way,
    # while the mix effect moves them in opposite directions. Set this much above ~3%
    # and it overwhelms the mix effect -- in years where one drivetrain dominates, the
    # drivetrains start appearing to rise and fall TOGETHER, hiding the substitution
    # this model exists to show. 2% keeps both effects visible.
    # SAFE TO CHANGE: yes, but read the paragraph above before raising it.
    total_fleet_relative_spread: float | AsymmetricSpread = AsymmetricSpread(lower=0.02, upper=0.02)

    # HOW UNCERTAIN EACH DRIVETRAIN'S SHARE OF THE FLEET IS, as a fraction:
    # 0.15 means "give or take 15%". This is uncertainty about the MIX -- which
    # drivetrains people buy -- as opposed to the total-fleet figure above.
    # SAFE TO CHANGE: yes.
    stock_target_relative_spread: dict[str, float | AsymmetricSpread] = field(default_factory=lambda: {
        "Hybrid":   AsymmetricSpread(lower=0.15, upper=0.15),
        "PHEV":     AsymmetricSpread(lower=0.15, upper=0.15),
        "HEV":      AsymmetricSpread(lower=0.15, upper=0.15),
        "BEV":      AsymmetricSpread(lower=0.15, upper=0.15),
        "Liquids":  AsymmetricSpread(lower=0.15, upper=0.15),
        "Petrol":   AsymmetricSpread(lower=0.15, upper=0.15),
        "Diesel":   AsymmetricSpread(lower=0.15, upper=0.15),
        "Gases":    AsymmetricSpread(lower=0.15, upper=0.15),
        "FCEV":     AsymmetricSpread(lower=0.15, upper=0.15),
    })

    # HOW UNCERTAIN THE COLLECTED SHARE IS, per drivetrain.
    # The tightest of the three outflow uncertainties, because collection rates are
    # actually measured. BEV is slightly wider: fewer years of statistics exist.
    # SAFE TO CHANGE: yes.
    collected_share_relative_spread: dict[str, float | AsymmetricSpread] = field(default_factory=lambda: {
        # PLACEHOLDER values, but now DIFFERENTIATED (not flat 0.15 for everyone) --
        # tight, since this is the MEASURED quantity (collection statistics).
        # BEV: newer market, fewer years of collection-statistics history -> a bit
        # less tight than the mature ICE/Hybrid group. FCEV/Gases: same "not a
        # verified real value" caveat as their point estimates above -- widest of
        # the three groups, reflecting that even less is actually known about them.
        "BEV":     0.08,
        "HEV":     0.05, "PHEV": 0.05, "Hybrid": 0.05,
        "Diesel":  0.05, "Petrol": 0.05, "Liquids": 0.05,
        "FCEV":    0.15,  # PLACEHOLDER, not a verified real value.
        "Gases":   0.15,  # PLACEHOLDER, not a verified real value.
    })

    # HOW UNCERTAIN THE EXPORT SHARE IS, per drivetrain.
    # Wider than the collected share above, because trade statistics are noisier than
    # collection statistics.
    # SAFE TO CHANGE: yes.
    export_share_relative_spread: dict[str, float | AsymmetricSpread] = field(default_factory=lambda: {
        # Same grouping logic as collected_share_relative_spread above, but export
        # (trade statistics) is somewhat noisier than collection statistics in
        # general, hence slightly wider than the corresponding collected_share
        # entry in every group.
        "BEV":     0.15,
        "HEV":     0.10, "PHEV": 0.10, "Hybrid": 0.10,
        "Diesel":  0.10, "Petrol": 0.10, "Liquids": 0.10,
        "FCEV":    0.20,  # PLACEHOLDER, not a verified real value.
        "Gases":   0.20,  # PLACEHOLDER, not a verified real value.
    })

    # HOW UNCERTAIN THE UNTRACEABLE SHARE IS, per drivetrain.
    # Deliberately the widest of the three: this quantity is inferred, not measured.
    # Note this is only the LEFTOVER uncertainty -- the setting below already links it
    # to vehicle lifetime, which contributes uncertainty of its own.
    # SAFE TO CHANGE: yes.
    unknown_whereabouts_share_relative_spread: dict[str, float | AsymmetricSpread] = field(default_factory=lambda: {
        # Deliberately the WIDEST of the three, per the reasoning behind this
        # field's existence: unknown_whereabouts is an ESTIMATE, not a measurement
        # -- and this is only the RESIDUAL spread (on top of the lifetime coupling
        # below, which already adds substantial extra uncertainty of its own). Also
        # given asymmetric upper > lower, reflecting that underestimating "how much
        # we don't know" is a more common failure mode than overestimating it.
        "BEV":     AsymmetricSpread(lower=0.20, upper=0.35),
        "HEV":     AsymmetricSpread(lower=0.20, upper=0.35),
        "PHEV":    AsymmetricSpread(lower=0.20, upper=0.35),
        "Hybrid":  AsymmetricSpread(lower=0.20, upper=0.35),
        "Diesel":  AsymmetricSpread(lower=0.20, upper=0.35),
        "Petrol":  AsymmetricSpread(lower=0.20, upper=0.35),
        "Liquids": AsymmetricSpread(lower=0.20, upper=0.35),
        "FCEV":    AsymmetricSpread(lower=0.20, upper=0.35),  # PLACEHOLDER, not a verified real value.
        "Gases":   AsymmetricSpread(lower=0.20, upper=0.35),  # PLACEHOLDER, not a verified real value.
    })

    # LINKS THE UNTRACEABLE SHARE TO VEHICLE LIFETIME, per drivetrain.
    # The reasoning: in a simulation run where cars turn out to be shorter-lived, more
    # vehicles leave the fleet than the records account for -- so the untraceable share
    # in that run should rise too, not stay fixed. 0.8 is a fairly strong link.
    #     0    no link -- untraceable share ignores lifetime entirely
    #     0.8  strong link (current)
    # SAFE TO CHANGE: yes. 0 reproduces the older, uncoupled behaviour.
    unknown_share_lifetime_coupling_k: dict[str, float] = field(default_factory=lambda: {
        # Same k=0.8 baseline as before for the mature ICE/Hybrid/BEV group.
        # Leaning more heavily on "shorter lifetime
        # implies more unexplained outflow" is more defensible than pretending
        # scale_lambda and unknown_share are independent for them.
        "BEV":     0.8,
        "HEV":     0.8, "PHEV": 0.8, "Hybrid": 0.8,
        "Diesel":  0.8, "Petrol": 0.8, "Liquids": 0.8,
        "FCEV":    0.8,  # PLACEHOLDER, not a verified real value.
        "Gases":   0.8,  # PLACEHOLDER, not a verified real value.
    })

    def validate(self) -> list[str]:
        issues: list[str] = []
        lifetime_drvs = set(self.lifetime_by_drv)

        missing = lifetime_drvs - set(self.lifetime_scale_lambda_relative_spread)
        if missing:
            issues.append(
                f"stock_flow.lifetime_scale_lambda_relative_spread is missing "
                f"drivetrains present in lifetime_by_drv: {sorted(missing)}."
            )
        for drv, spread in self.lifetime_scale_lambda_relative_spread.items():
            if isinstance(spread, AsymmetricSpread):
                issues += spread.validate(field_name=f"stock_flow.lifetime_scale_lambda_relative_spread['{drv}']")
            elif spread < 0:
                issues.append(f"stock_flow.lifetime_scale_lambda_relative_spread['{drv}'] = {spread} must be >= 0.")

        for name, mapping in (
            ("collected_share_relative_spread", self.collected_share_relative_spread),
            ("export_share_relative_spread", self.export_share_relative_spread),
            ("unknown_whereabouts_share_relative_spread", self.unknown_whereabouts_share_relative_spread),
            ("stock_target_relative_spread", self.stock_target_relative_spread),
        ):
            missing = lifetime_drvs - set(mapping)
            if missing:
                issues.append(f"stock_flow.{name} is missing drivetrains present in lifetime_by_drv: {sorted(missing)}.")
            for drv, spread in mapping.items():
                if isinstance(spread, AsymmetricSpread):
                    issues += spread.validate(field_name=f"stock_flow.{name}['{drv}']")
                elif spread < 0:
                    issues.append(f"stock_flow.{name}['{drv}'] = {spread} must be >= 0.")

        # [NEW] `total_fleet_relative_spread` is a SINGLE spread shared by every
        # drivetrain (not a per-drivetrain mapping), so it validates on its own rather
        # than in the loop above.
        if isinstance(self.total_fleet_relative_spread, AsymmetricSpread):
            issues += self.total_fleet_relative_spread.validate(
                field_name="stock_flow.total_fleet_relative_spread"
            )
        elif self.total_fleet_relative_spread < 0:
            issues.append(
                f"stock_flow.total_fleet_relative_spread="
                f"{self.total_fleet_relative_spread} must be >= 0."
            )

        if not isinstance(self.stock_target_correlated_mix, bool):
            issues.append(
                f"stock_flow.stock_target_correlated_mix="
                f"{self.stock_target_correlated_mix!r} must be a bool."
            )

        if not isinstance(self.stock_target_uncertainty_start_year, int) or self.stock_target_uncertainty_start_year < 1950:
            issues.append(
                f"stock_flow.stock_target_uncertainty_start_year="
                f"{self.stock_target_uncertainty_start_year!r} must be an int >= 1950."
            )

        if self.stock_target_ramp_max_rate_per_year <= 0:
            issues.append(
                f"stock_flow.stock_target_ramp_max_rate_per_year="
                f"{self.stock_target_ramp_max_rate_per_year!r} must be > 0 "
                f"(0 would mean an infinite ramp -- the multiplier would never "
                f"reach its sampled value)."
            )

        missing_k = lifetime_drvs - set(self.unknown_share_lifetime_coupling_k)
        if missing_k:
            issues.append(
                f"stock_flow.unknown_share_lifetime_coupling_k is missing drivetrains "
                f"present in lifetime_by_drv: {sorted(missing_k)}."
            )

        if self.negative_inflow_policy not in {"report_only", "clip_to_target"}:
            issues.append(
                f"stock_flow.negative_inflow_policy={self.negative_inflow_policy!r} is "
                f"not one of ['report_only', 'clip_to_target']."
            )

        # [NEW] inflow_mode_by_drv / inflow_phaseout_by_drv consistency checks.
        valid_inflow_modes = {"remind_literal", "remind_soft", "inflow_phaseout"}
        for drv, mode in self.inflow_mode_by_drv.items():
            if mode not in valid_inflow_modes:
                issues.append(
                    f"stock_flow.inflow_mode_by_drv[{drv!r}]={mode!r} is not one of "
                    f"{sorted(valid_inflow_modes)}."
                )
            if mode == "inflow_phaseout" and drv not in self.inflow_phaseout_by_drv:
                issues.append(
                    f"stock_flow.inflow_mode_by_drv[{drv!r}]='inflow_phaseout' but "
                    f"'{drv}' has no entry in stock_flow.inflow_phaseout_by_drv."
                )
        for drv, phaseout_spec in self.inflow_phaseout_by_drv.items():
            start_year, max_share = phaseout_spec
            if not (0.0 < max_share < 1.0):
                issues.append(
                    f"stock_flow.inflow_phaseout_by_drv[{drv!r}]'s max_share={max_share} "
                    f"must be strictly between 0 and 1."
                )
            hard_zero_year = self.hard_zero_inflow_from_year_by_drv.get(drv)
            if hard_zero_year is not None and start_year >= hard_zero_year:
                issues.append(
                    f"stock_flow.inflow_phaseout_by_drv[{drv!r}]'s start_year={start_year} "
                    f"must be strictly before hard_zero_inflow_from_year_by_drv[{drv!r}]="
                    f"{hard_zero_year} (the phase-out window would be empty or invalid)."
                )

        # [NEW] inflow_phaseout_max_share_triangular_by_drv consistency checks.
        for drv, triangular in self.inflow_phaseout_max_share_triangular_by_drv.items():
            low, mode, high = triangular
            if not (low <= mode <= high):
                issues.append(
                    f"stock_flow.inflow_phaseout_max_share_triangular_by_drv[{drv!r}]="
                    f"{triangular} must satisfy low <= mode <= high."
                )
            if not (0.0 < low < 1.0) or not (0.0 < high < 1.0):
                issues.append(
                    f"stock_flow.inflow_phaseout_max_share_triangular_by_drv[{drv!r}]="
                    f"{triangular} -- low and high must both be strictly between 0 and 1."
                )

        for drv, year in self.hard_zero_inflow_from_year_by_drv.items():
            if drv not in lifetime_drvs:
                issues.append(
                    f"stock_flow.hard_zero_inflow_from_year_by_drv has key {drv!r}, which "
                    f"is not a drivetrain present in lifetime_by_drv ({sorted(lifetime_drvs)})."
                )
            if not isinstance(year, int) or year < 1950:
                issues.append(
                    f"stock_flow.hard_zero_inflow_from_year_by_drv[{drv!r}] = {year!r} must "
                    f"be a plausible calendar year (int >= 1950)."
                )

        for drv, year in self.hard_zero_inflow_until_year_by_drv.items():
            if drv not in lifetime_drvs:
                issues.append(
                    f"stock_flow.hard_zero_inflow_until_year_by_drv has key {drv!r}, which "
                    f"is not a drivetrain present in lifetime_by_drv ({sorted(lifetime_drvs)})."
                )
            if not isinstance(year, int) or year < 1950:
                issues.append(
                    f"stock_flow.hard_zero_inflow_until_year_by_drv[{drv!r}] = {year!r} must "
                    f"be a plausible calendar year (int >= 1950)."
                )
            from_year = self.hard_zero_inflow_from_year_by_drv.get(drv)
            if from_year is not None and isinstance(year, int) and from_year <= year:
                issues.append(
                    f"stock_flow.hard_zero_inflow_from_year_by_drv[{drv!r}] = {from_year!r} "
                    f"is <= hard_zero_inflow_until_year_by_drv[{drv!r}] = {year!r} -- this "
                    f"would zero EVERY year for {drv!r} (the 'from' window and the 'until' "
                    f"window overlap/cover the whole horizon), which is almost certainly not "
                    f"intended."
                )

        for name, mapping in (
            ("export_share_by_drv", self.export_share_by_drv),
            ("unknown_whereabouts_share", self.unknown_whereabouts_share),
            ("collected_share_by_drv", self.collected_share_by_drv),
        ):
            missing = lifetime_drvs - set(mapping)
            if missing:
                issues.append(f"stock_flow.{name} is missing drivetrains present in lifetime_by_drv: {sorted(missing)}.")
            for drv, share in mapping.items():
                if share is not None and not (0.0 <= share <= 1.0):
                    issues.append(f"stock_flow.{name}['{drv}'] = {share} is outside [0, 1].")

        # [NEW] The three point estimates are supposed to sum to 1 per drivetrain --
        # this isn't ENFORCED at sample time any more (they're independently sampled
        # and normalized, see disaggregation.compute_collected_export_unknown_shares),
        # so a typo'd point estimate here would previously be silently corrected by
        # that normalization at every draw without ever surfacing as an error. Catch
        # it here instead, at the point estimate level, where it's actually a mistake.
        for drv in lifetime_drvs:
            total = (
                self.collected_share_by_drv.get(drv, 0.0)
                + self.export_share_by_drv.get(drv, 0.0)
                + self.unknown_whereabouts_share.get(drv, 0.0)
            )
            if abs(total - 1.0) > 1e-6:
                issues.append(
                    f"stock_flow: collected_share_by_drv['{drv}'] + export_share_by_drv['{drv}'] + "
                    f"unknown_whereabouts_share['{drv}'] = {total}, expected 1.0."
                )

        missing_override = lifetime_drvs - set(self.lifetime_override_by_drv)
        if missing_override:
            issues.append(
                f"stock_flow.lifetime_override_by_drv is missing drivetrains present in "
                f"lifetime_by_drv: {sorted(missing_override)}."
            )
        return issues


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DisaggregationParams:

    # Where this stage reads its input data and writes its results.
    # SAFE TO CHANGE: only if you actually move the data folder. The path is relative
    # to the `code/` directory.
    output_dir: str = "../data/processed/"

    # WHERE THE EEA REGISTRATIONS FILE LIVES.
    #
    # `EEA_final_data.csv` is an INPUT -- real registration statistics that no stage
    # of this pipeline can regenerate. It used to be looked up under `output_dir`,
    # i.e. in `data/processed/`, alongside artifacts that ARE regenerable and that
    # get cleared out periodically. On 2026-08-20 it was deleted along with the
    # generated files, and there was no way to recreate it: the fallback only writes
    # a clearly-labelled synthetic placeholder.
    #
    # An input that lives in the output folder will eventually be deleted with the
    # outputs. It now lives in `data/raw/` with the other inputs, which is the only
    # place clearing generated results cannot reach.
    #
    # SAFE TO CHANGE: only if you actually move the folder. If the file is not found
    # here, the stage also looks in `output_dir` and says so, so an older working
    # copy keeps running until the file is moved across.
    eea_input_dir: str = "../data/raw/"

    # The first and last year the model simulates.
    # SAFE TO CHANGE: with care. Widening the window makes the model reconstruct more
    # history by backcasting -- estimating years it has no data for -- so the earliest
    # years become progressively less reliable. Start must be before end.
    start_year_model: int = 1950
    end_year_model: int = 2070

    # The first and last year drawn on this stage's charts.
    # These change ONLY what you see, never what is computed: the model still runs the
    # full start_year_model..end_year_model range above. Narrow them to zoom in on a
    # period of interest; widen them to see the whole run.
    # SAFE TO CHANGE: yes. The only rule is that start must not be after end.
    year_plot_start: int = 2015
    year_plot_end: int = 2070

    # Use invented stand-in data instead of the real EEA registration file?
    # NORMALLY LEAVE THIS FALSE. The real file (`EEA_final_data.csv`) is what decides
    # how vehicles split across size segments; results built on the stand-in are not
    # meaningful, only structurally valid.
    # Set to True ONLY as a temporary bridge when that file is missing and you want to
    # check the rest of the pipeline runs. The stage then generates a clearly labelled
    # SYNTHETIC placeholder instead of stopping with an error.
    # SAFE TO CHANGE: yes -- but never report numbers produced with this set to True.
    use_synthetic_eea_fallback: bool = False

    # Fixes the random numbers used to generate that stand-in data, so the placeholder
    # is at least reproducible. Irrelevant while the setting above is False.
    # SAFE TO CHANGE: yes, any whole number.
    synthetic_eea_seed: int = 42

    # The real-world year each drivetrain first went on sale.
    # This is a documented historical fact, not a modelling assumption, and it is used
    # in two places that must agree: filling gaps in the registration data, and
    # reconstructing 1975-2004, for which there is no data at all. Before a drivetrain's
    # first year the model puts an exact zero -- the vehicle genuinely did not exist yet
    # -- rather than guessing a number from later data.
    # SAFE TO CHANGE: only to correct a factual error. Getting it wrong invents vehicles
    # in years they could not have existed, or erases real early ones. Each value must
    # be a whole year, 1950 or later.
    introduction_year_by_drv: dict[str, int] = field(default_factory=lambda: {
        "BEV": 2011, "HEV": 2000, "PHEV": 2012,
    })

    # WHERE HYBRIDS COME FROM. This decides whether HEV is carved out of Liquids or
    # split off Hybrid, and it is a correctness switch, not a preference.
    #
    # The REMIND files hold five vehicle technologies -- Liquids, Hybrid electric,
    # Gases, FCEV, BEV -- and NONE of them is a non-plug-in hybrid. In REMIND's
    # taxonomy "Hybrid electric" is the PLUG-IN hybrid; ordinary full and mild hybrids
    # are counted inside Liquids. The model used to split that plug-in class into HEV
    # and PHEV, which invented an HEV series out of plug-in volume while the real
    # hybrids stayed inside Liquids -- counting them twice and neither correctly.
    # ACEA puts hybrids at 25.8% of the 2023 EU market, 2.71 million cars, so this is
    # not a rounding matter.
    #
    # True   HEV is carved out of Liquids using real registration shares, and the
    #        whole of REMIND's Hybrid becomes PHEV. Validated against the EEA file
    #        and ACEA: HEV 1.03x reality in 2019, PHEV 1.02x in 2021.
    # False  the old behaviour, kept only so results published before 20 August 2026
    #        can be reproduced. It is wrong; do not use it for new work.
    #
    # SAFE TO CHANGE: only to reproduce an old result.
    hev_carved_from_liquids: bool = True

    # WHEN HYBRIDS STOP BEING SOLD, as a share of the liquid-fuel market.
    #
    # Real registration data ends in 2023, where hybrids are 36.1% of all petrol,
    # diesel and hybrid sales. Beyond that the share has to be assumed, because REMIND
    # has no hybrid-versus-plain-liquids opinion at all -- that absence is the whole
    # reason this setting exists. The share is ramped straight down from its last
    # observed value to zero in this year.
    #
    # 2035 is not arbitrary: the model's own liquid-fuel sales are last positive in
    # 2034 (0.24 million) and negative after, so hybrids reach zero exactly as the
    # fuel they depend on does. A hybrid cannot outlive petrol.
    #
    # SAFE TO CHANGE: yes, and it is a genuine scenario choice. Later means hybrids
    # linger as a larger slice of a shrinking market; earlier means they give way to
    # battery-electric sooner. It does not change total sales, only their split.
    hev_share_phaseout_end_year: int = 2035

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.start_year_model >= self.end_year_model:
            issues.append("disaggregation: start_year_model must be strictly before end_year_model.")
        if self.year_plot_start > self.year_plot_end:
            issues.append("disaggregation: year_plot_start must not be after year_plot_end.")
        for drv, year in self.introduction_year_by_drv.items():
            if not isinstance(year, int) or year < 1950:
                issues.append(
                    f"disaggregation.introduction_year_by_drv[{drv!r}] = {year!r} must be "
                    f"a plausible calendar year (int >= 1950)."
                )
        return issues

# ---------------------------------------------------------------------------
# STAGE 03_02 -- Adjusted flows / scenarios  (code/03_02_adjustedflows.py)
# ---------------------------------------------------------------------------
# WHAT THIS STAGE DOES: takes the fleet from stage 03_01 and asks "what if?".
# Each SCENARIO is one alternative future -- more BEVs, longer-lasting cars, fewer
# vehicles sold, higher collection losses -- and this stage re-simulates the whole
# fleet under each one, so they can be compared side by side.
#
# THE SETTINGS BELOW ARE THE SHARED RULES. The scenarios themselves are defined
# further down in `scenarios`, each as a ScenarioSpec (see SHARED BUILDING BLOCKS
# near the top of this file for what a ScenarioSpec can contain).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdjustedFlowsParams:

    # When a scenario that changes the MIX of vehicles sold begins, and when it has
    # fully arrived. Between these two years the change phases in gradually.
    # Both are currently the same year, which means the change happens instantly in
    # 2026 with no phase-in. They are kept as two separate settings so a future
    # scenario can spread a change over several years by pushing the second one later.
    # SAFE TO CHANGE: yes. Keep start <= end. History before the start year is never
    # touched by a scenario -- only the future is.
    scenario_start_year: int = 2026
    scenario_ramp_end_year: int = 2026

    # The year that scenarios changing how LONG vehicles last take effect
    # (e.g. "BEV_longer", "ICEV_shorter").
    # SAFE TO CHANGE: yes. Set once here so every lifetime scenario stays in step.
    lifetime_change_start_year: int = 2027

    # The year that scenarios changing HOW MANY vehicles are sold take effect
    # (e.g. "stock_lower", which sells 20% fewer).
    # Deliberately separate from the lifetime year above: "how many cars are bought"
    # and "how long cars last" are unrelated levers, even though both currently
    # start in 2027.
    # SAFE TO CHANGE: yes.
    stock_modifier_start_year: int = 2027

    # The year each scenario starts from. The existing fleet as it stood in this year
    # is the common starting point every scenario builds on, rather than each one
    # re-deriving history from scratch.
    # SAFE TO CHANGE: NOT ON ITS OWN. This must match the base year stage 02 uses. If
    # you change it here and not there, the scenarios start from a fleet that stage 02
    # never produced, and every downstream number is quietly wrong.
    base_year: int = 2005

    # Which scenarios actually get simulated.
    #     ("BAU",)                     just business-as-usual -- fast (the default)
    #     ("BAU", "stock_lower")       two scenarios, comparable against each other
    #     None                         every scenario listed below -- the full sweep
    #
    # IMPORTANT, AND EASY TO MISS: all the scenario-COMPARISON figures need at least
    # TWO scenarios here. With only ("BAU",) those figures cannot be drawn and simply
    # do not appear -- nothing is broken, there is just nothing to compare.
    #
    # Cost warning: each scenario is a full Monte Carlo simulation. The complete sweep
    # is 11 of them and takes hours at 200,000 draws.
    #
    # ⚠️ AND EACH ONE NOW ALSO WRITES THE PER-DRAW BEV ARRAYS that stage 04_04
    # reads -- twelve extra per-segment Monte Carlo runs per scenario, and about
    # 2.7 GB on disk per scenario at 200,000 draws. The full sweep is therefore
    # roughly 30 GB in data/processed/bev_draws/. Set
    # materials.bev_electronics_export_draws to False if you want the sweep
    # without them.
    #
    # A misspelled name is caught immediately when you run code/00_parameters.py,
    # rather than hours into a run.
    # SAFE TO CHANGE: yes -- this is the setting you are most likely to want to edit.
    scenarios_to_run: tuple[str, ...] | None = ("BAU", )

    # EVERY SCENARIO THIS STAGE KNOWS HOW TO RUN, defined in one place.
    #
    # Each entry is one alternative future, described by a `ScenarioSpec`
    # (see the shared building blocks above). A scenario only changes what it
    # explicitly names: leave a field out and the base value applies, so
    # `ICEV_shorter` states only Diesel and Petrol lifetimes and inherits
    # everything else untouched.
    #
    # Defining a scenario here does NOT run it -- `scenarios_to_run` above
    # decides that, and each one costs a full Monte Carlo simulation.
    #
    # SAFE TO CHANGE: yes. Adding a scenario is one new entry here and
    # nothing else; the stage discovers it automatically. A misspelled name in
    # `scenarios_to_run` is caught when you run code/00_parameters.py.
    scenarios: dict[str, ScenarioSpec] = field(default_factory=lambda: {
        "BAU": ScenarioSpec(
            name="BAU",
            inflow_segment_shares_drivetrain="BEV",
            inflow_segment_shares_final={
                "A": 0.11599, "B": 0.07235, "C": 0.17037, "D": 0.06288, "E": 0.04190, "F": 0.02945,
                "JA": 0.00341, "JB": 0.07498, "JC": 0.25116, "JD": 0.15162, "JE": 0.02108, "JF": 0.00481,
            },
            # [NEW, pilot] Future segment-mix uncertainty: 10% lower / 15% upper relative
            # spread around each of the 12 mode values above, e.g. "A"'s Triangular is
            # (0.11599*0.90, 0.11599, 0.11599*1.15) = (0.10439, 0.11599, 0.13339).
            # Applied only to years >= scenario_start_year (2026) -- real historic
            # segment-mix data stays fully deterministic. Not yet set for the other four
            # segment-mix scenarios (BEV_A_F, BEV_JA_JF, BEV_large, BEV_small) -- BAU is
            # a deliberate pilot; extend the same pattern to those once this is verified.
            inflow_segment_share_spread=AsymmetricSpread(lower=0.20, upper=0.40),
        ),
        "BEV_only": ScenarioSpec(
            name="BEV_only",
            inflow_drivetrain_shares_final={"BEV": 1.0, "HEV": 0.0, "PHEV": 0.0, "Diesel": 0.0, "Petrol": 0.0},
        ),
        "BEV_A_F": ScenarioSpec(
            name="BEV_A_F",
            inflow_segment_shares_drivetrain="BEV",
            inflow_segment_shares_final={
                "A": 0.11940, "B": 0.14733, "C": 0.42153, "D": 0.21450, "E": 0.06298, "F": 0.03426,
                "JA": 0.0, "JB": 0.0, "JC": 0.0, "JD": 0.0, "JE": 0.0, "JF": 0.0,
            },
        ),
        "BEV_JA_JF": ScenarioSpec(
            name="BEV_JA_JF",
            inflow_segment_shares_drivetrain="BEV",
            inflow_segment_shares_final={
                "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0,
                "JA": 0.11940, "JB": 0.14733, "JC": 0.42153, "JD": 0.21450, "JE": 0.06298, "JF": 0.03426,
            },
        ),
        "BEV_large": ScenarioSpec(
            name="BEV_large",
            inflow_segment_shares_drivetrain="BEV",
            inflow_segment_shares_final={
                "A": 0.02, "B": 0.05, "C": 0.12, "D": 0.16, "E": 0.12, "F": 0.06,
                "JA": 0.04, "JB": 0.10, "JC": 0.16, "JD": 0.10, "JE": 0.05, "JF": 0.02,
            },
        ),
        "BEV_small": ScenarioSpec(
            name="BEV_small",
            inflow_segment_shares_drivetrain="BEV",
            inflow_segment_shares_final={
                "A": 0.10, "B": 0.22, "C": 0.28, "D": 0.14, "E": 0.04, "F": 0.01,
                "JA": 0.03, "JB": 0.07, "JC": 0.07, "JD": 0.03, "JE": 0.01, "JF": 0.00,
            },
        ),
        "BEV_longer": ScenarioSpec(
            name="BEV_longer",
            lifetime_change_by_drv={
                "BEV": OpenEndedLifetimeChange(start_year=2027, shape_k=3.0, scale_lambda=17.0),
            },
        ),
        "ICEV_shorter": ScenarioSpec(
            name="ICEV_shorter",
            lifetime_change_by_drv={
                "Diesel": OpenEndedLifetimeChange(start_year=2027, shape_k=3.0, scale_lambda=9.0),
                "Petrol": OpenEndedLifetimeChange(start_year=2027, shape_k=3.0, scale_lambda=9.0),
            },
        ),
        "stock_lower": ScenarioSpec(name="stock_lower", stock_modifier=0.8),
        "losses_zero": ScenarioSpec(
            name="losses_zero",
            export_share_overrides={"BEV": 0.0},
            unknown_whereabouts_share_overrides={"BEV": 0.0},
        ),
        "losses_high": ScenarioSpec(
            name="losses_high",
            export_share_overrides={"BEV": 0.08},  # "like ICEV" -- matches every other non-BEV drivetrain
            unknown_whereabouts_share_overrides={"BEV": 0.43},  # "like ICEV"
        ),
    })

    def active_scenario_names(self) -> tuple[str, ...]:
        """
        Resolve `scenarios_to_run` into the actual list of scenario names
        `03_02_adjustedflows.py` should simulate: every name in `scenarios` if
        `scenarios_to_run` is None (the full sweep), otherwise just the requested
        subset -- always in `scenarios`' own dict order (BAU first), which is the
        order `03_02_adjustedflows.py`'s inflow-resolution step relies on. Centralized
        here, not duplicated in `03_02_adjustedflows.py`'s `main()`, so there is one
        place that defines what "select which scenarios run" means.
        """
        if self.scenarios_to_run is None:
            return tuple(self.scenarios.keys())
        requested = set(self.scenarios_to_run)
        return tuple(name for name in self.scenarios if name in requested)

    def validate(self) -> list[str]:
        issues: list[str] = []
        if "BAU" not in self.scenarios:
            issues.append("adjusted_flows.scenarios must include a 'BAU' entry (used as the base inflow for scenarios with no inflow transform of their own).")
        if self.scenarios_to_run is not None:
            if not self.scenarios_to_run:
                issues.append(
                    "adjusted_flows.scenarios_to_run is an empty tuple -- set it to None "
                    "to run every scenario, or list at least one scenario name."
                )
            unknown = set(self.scenarios_to_run) - set(self.scenarios)
            if unknown:
                issues.append(
                    f"adjusted_flows.scenarios_to_run has unknown scenario name(s) "
                    f"{sorted(unknown)} -- available: {sorted(self.scenarios)}."
                )
        for spec in self.scenarios.values():
            issues += spec.validate(valid_segments=set(_ADJUSTED_FLOWS_SEGMENTS))
        for name_, change_start in [("lifetime_change_start_year", self.lifetime_change_start_year)]:
            for scen_name, spec in self.scenarios.items():
                for drv, change in spec.lifetime_change_by_drv.items():
                    if change.start_year != change_start:
                        issues.append(
                            f"adjusted_flows.scenarios['{scen_name}'].lifetime_change_by_drv['{drv}']"
                            f".start_year={change.start_year} != adjusted_flows.{name_}={change_start} "
                            f"-- expected every lifetime-change scenario to share the same start year."
                        )
        if self.scenario_start_year > self.scenario_ramp_end_year:
            issues.append("adjusted_flows.scenario_start_year must not be after scenario_ramp_end_year.")
        if self.base_year > self.scenario_start_year:
            issues.append(
                f"adjusted_flows.base_year={self.base_year} is after scenario_start_year="
                f"{self.scenario_start_year} -- the cohort base year should precede any scenario transform."
            )
        return issues

# ---------------------------------------------------------------------------
# STAGE 04 -- Materials / car composition  (code/04_01_carcomposition.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 04 -- Materials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialsParams:
    """
    Turns vehicles into materials: how much steel, aluminium, copper, battery
    chemistry and so on the fleet contains, and therefore how much becomes available
    for recovery when those vehicles are scrapped.

    Most settings here point at the workbooks holding composition data, or translate
    between this model's names and the codes used inside those files. The ones that
    genuinely change results are the battery sizes and the choice of detail level.
    """

    # Translates the size-segment letters used everywhere else in this model into the
    # numeric codes used inside the composition data files.
    # SAFE TO CHANGE: no, unless the composition files themselves change their codes.
    # A wrong code here silently pulls the composition of the wrong size of car.
    segment_map: dict[str, str] = field(default_factory=lambda: {
        "A": "0101", "B": "0102", "C": "0103", "D": "0104", "E": "0105", "F": "0106",
        "JA": "0201", "JB": "0202", "JC": "0203", "JD": "0204", "JE": "0205", "JF": "0206",
    })

    # The same idea for drivetrains: our name -> the code used in the composition files.
    # SAFE TO CHANGE: no, same warning as above.
    drv_prefix_map: dict[str, str] = field(default_factory=lambda: {
        "PHEV": "050103", "HEV": "040101", "BEV": "030103",
        "Diesel": "020102", "Petrol": "010101",
    })

    # Which region's results this stage processes.
    # SAFE TO CHANGE: only if the upstream stages actually produced another region.
    region: str = "EUR"

    # Which drivetrains get material results.
    # Note these are the FINE-GRAINED names (Petrol and Diesel separately), not the
    # grouped "Liquids" used earlier in the pipeline.
    # SAFE TO CHANGE: yes, to narrow the run. Adding a name only works if composition
    # data exists for it.
    drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol")

    # Which level of detail to read from the car-composition file:
    #     "m-c"  material per component  (current -- what this project needs)
    #     "e-m"  individual chemical elements (finer, much larger)
    # SAFE TO CHANGE: yes, but "m-c" is the level the rest of this analysis assumes.
    composition_parameter_code: str = "m-c"

    # Which column in the composition file holds the material name to group by.
    # SAFE TO CHANGE: no, unless the file's own column naming changes.
    material_level_key: str = "materialKeyLevel_highest"

    # ⚠️ WHERE THE TRACTION-MOTOR PROJECT KEEPS WHAT IT PRODUCES, read where it
    # lies. Matthias 2026-09-21: traction motor information lives in
    # RAWCLICVehicleTractionMotor and nowhere else, so this model is pointed at
    # that folder rather than keeping a copy of the workbook beside its own
    # inputs. Only what THIS model consumes and no sibling produces belongs in
    # data/raw -- the EEA registrations, the used-vehicle trade file, REMIND.
    #
    # An absolute path, for the same reason `battery_composition_dir` is one:
    # that project is a sibling checkout and neither repo may assume where the
    # other sits. Nothing under it is ever written.
    #
    # ⚠️ IT MUST BE THE `data/consolidated` FOLDER AND NOT `data`. The traction
    # project writes a great deal for its own use -- the trajectory by segment,
    # the audit, the corrections log -- and exactly one pair of files for this
    # model. Pointing at the wrong folder would make every one of those look
    # like an interface nobody may change.
    # SAFE TO CHANGE: yes, when that project moves.
    traction_composition_dir: str = (
        "/Users/rm/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/"
        "RAWCLICVehicleTractionMotor/data/consolidated")

    # The file in there that 04_03 reads. Written by that project's
    # `01_composition.py`; the sheet is "Consolidated data".
    # SAFE TO CHANGE: yes, if that project renames what it writes.
    traction_composition_file_name: str = "TractionMotor_for_stockandflow.xlsx"

    # ==================================================================
    # HOW THE FLEET SPLITS ACROSS TRACTION MOTOR TYPES AND VOLTAGES
    # ==================================================================
    #
    # The traction project reports a composition for every combination of motor
    # type, voltage class, torque and year, and takes no view on how many cars
    # are of each. That view is here, because it is a fleet question and this is
    # the fleet model. Matthias 2026-09-21.
    #
    # ⚠️ DRAWN PER CAR, NOT BLENDED. A vehicle is one motor type at one voltage,
    # for life -- the same rule `battery_voltage.py` states for the pack: "THE
    # STATES ARE DISCRETE. A vehicle is one architecture, never a blend." A
    # share is the probability a draw lands on that state, not a fraction of
    # every car.

    # ⚠️ THE REPORT'S OWN SHARES, TRANSCRIBED UNCHANGED, IN ITS OWN CATEGORIES.
    # RAWCLIC_BEV_Motors_Comprehensive_Report_V1 §10.2-10.7, base case where it
    # offers conservative/base/optimistic (2060 and 2070). Percentages, summing
    # to 100 in every row, exactly as printed -- the mapping onto this model's
    # five motor types happens in code and is visible there rather than baked
    # into the numbers.
    #
    # ⚠️ THEY ARE SCENARIO CONSTRUCTION AND THE REVIEW SAYS SO: "the original
    # 2030, 2035 and 2040 architecture shares are scenario construction rather
    # than confirmed forecasts". Treat the trajectory as an argument, not a
    # measurement. The 2025 row is the closest thing to an observation here.
    #
    # BEFORE 2025 the report says nothing, so the 2025 mix is held constant
    # backwards with axial flux at zero -- it was not on the market -- and its
    # share given to PMSM. Early Tesla induction is not separately modelled.
    # SAFE TO CHANGE: yes. A registration-weighted series would beat all of it.
    traction_type_shares: dict[str, dict[int, dict[str, float]]] = field(
        default_factory=lambda: {
            "AB": {
                2010: {"PMSM": 91, "EESM": 5, "ASM": 4, "axial": 0, "SynRM": 0},
                2025: {"PMSM": 88, "EESM": 5, "ASM": 4, "axial": 0, "SynRM": 3},
                2030: {"PMSM": 75, "EESM": 8, "ASM": 5, "axial": 1, "SynRM": 11},
                2040: {"PMSM": 55, "EESM": 12, "ASM": 5, "axial": 2, "SynRM": 26},
                2050: {"PMSM": 42, "EESM": 14, "ASM": 5, "axial": 3, "SynRM": 36},
                2060: {"PMSM": 35, "EESM": 15, "ASM": 5, "axial": 3, "SynRM": 42},
                2070: {"PMSM": 29, "EESM": 16, "ASM": 5, "axial": 5, "SynRM": 45},
            },
            "CD": {
                2010: {"PMSM": 86, "EESM": 7, "ASM": 6, "axial": 0, "SynRM": 0},
                2025: {"PMSM": 84, "EESM": 7, "ASM": 6, "axial": 1, "SynRM": 2},
                2030: {"PMSM": 72, "EESM": 12, "ASM": 7, "axial": 3, "SynRM": 6},
                2040: {"PMSM": 52, "EESM": 21, "ASM": 6, "axial": 8, "SynRM": 13},
                2050: {"PMSM": 43, "EESM": 25, "ASM": 5, "axial": 12, "SynRM": 15},
                2060: {"PMSM": 36, "EESM": 27, "ASM": 5, "axial": 15, "SynRM": 17},
                2070: {"PMSM": 31, "EESM": 30, "ASM": 5, "axial": 17, "SynRM": 17},
            },
            "EF": {
                2010: {"PMSM": 81, "EESM": 10, "ASM": 9, "axial": 0, "SynRM": 0},
                2025: {"PMSM": 78, "EESM": 10, "ASM": 9, "axial": 2, "SynRM": 1},
                2030: {"PMSM": 65, "EESM": 17, "ASM": 8, "axial": 7, "SynRM": 3},
                2040: {"PMSM": 45, "EESM": 25, "ASM": 5, "axial": 19, "SynRM": 6},
                2050: {"PMSM": 37, "EESM": 28, "ASM": 4, "axial": 25, "SynRM": 6},
                2060: {"PMSM": 30, "EESM": 31, "ASM": 4, "axial": 29, "SynRM": 6},
                2070: {"PMSM": 25, "EESM": 32, "ASM": 4, "axial": 33, "SynRM": 6},
            },
        })

    # ⚠️ WHERE SynRM/PMa GOES, BECAUSE THIS PROJECT DOES NOT MODEL IT.
    # Matthias 2026-09-21: "SynRM goes to PMSM for now."
    #
    # The traction project excluded synchronous reluctance deliberately --
    # "demonstrators only, with no bill of material in either source" -- which
    # was right while nothing needed a share. The report gives it one, and it is
    # not small: 45% of AB by 2070 in the base case.
    #
    # ⚠️ AND THE CHOICE IS NOT NEUTRAL. A PMa-SynRM uses far less magnet than an
    # IPM, so sending its share to PMSM keeps the small-car fleet MAGNET-HEAVY
    # exactly where the report had it going magnet-light. This is the
    # conservative direction for rare-earth demand -- it cannot understate it --
    # but it is a real overstatement of AB magnets in the late years, and it is
    # the first thing to revisit when a SynRM bill of material exists.
    # SAFE TO CHANGE: yes -- "EESM" is the other defensible destination, and it
    # would bracket the answer from below.
    traction_synrm_goes_to: str = "PMSM"

    # ⚠️ THE SHARE OF CARS WITH TWO DRIVEN AXLES, MEASURED, BY SEGMENT GROUP.
    # EV Database, 1438 models: AWD 576, front 440, rear 422. By group the
    # gradient is steep and monotonic -- A and B have essentially none, F is
    # 80.7%, JF 88.9%.
    #
    # ⚠️ MODEL-WEIGHTED, NOT REGISTRATION-WEIGHTED, and Matthias chose to keep it
    # that way 2026-09-21. It counts a 200-unit halo trim the same as a 50,000-
    # unit volume seller, and AWD skews to low-volume trims, so the level is
    # probably high even though the gradient is certainly right. The critical
    # review names the same gap: "eligible public evidence does not provide an EU
    # registration-weighted distribution of single-, dual- and multi-motor BEVs
    # by segment and year".
    # SAFE TO CHANGE: yes, and a registration-weighted series would replace it.
    traction_awd_share: dict[str, float] = field(default_factory=lambda: {
        "AB": 0.061, "CD": 0.386, "EF": 0.755,
    })

    # ⚠️ AND OF THOSE TWO-MOTOR CARS, HOW MANY CARRY TWO PERMANENT-MAGNET
    # MACHINES RATHER THAN ONE PM AND ONE INDUCTION.
    #
    # Matthias 2026-09-21: "I assume AWD induction will be the large parts, but
    # E and F might have two permanent ones, due to being very heavy cars."
    #
    # WHY IT DECIDES WHICH CATEGORY THE CAR IS IN. `IMandPMElectricMotors` in the
    # source data is the PM-plus-induction configuration; a PM-plus-PM car is
    # simply a PM car with more torque, and belongs in `PMElectricMotors`. The
    # composition is a function of torque, so a twin-PM car at the vehicle's own
    # torque needs no special treatment -- two machines at half the torque carry
    # about the same magnet as one at full torque, which is this project's own
    # measured finding.
    #
    # WHAT IT COSTS. The two categories differ by a CONSTANT 1.20 kg of magnet at
    # every torque -- the fitted intercepts are +1.107 and -0.093 kg and the
    # slopes are identical -- because the second machine in an IM+PM car has no
    # magnet in it at all. So every percentage point moved between them is 1.2 kg
    # of magnet per car, and in EF that is the single largest lever on
    # rare-earth demand in this model.
    #
    # ⚠️ NOT MEASURED. Nothing in the fleet data says which AWD cars are twin-PM.
    # These three numbers are Matthias's reading of the market, rising with
    # vehicle weight, and they are the only invented figures in this block.
    # SAFE TO CHANGE: yes, and this is the one to vary first.
    traction_twin_pm_share_of_awd: dict[str, float] = field(
        default_factory=lambda: {"AB": 0.20, "CD": 0.40, "EF": 0.70})

    # ⚠️ THE DUAL-ROTOR RADIAL MACHINE GETS NO SHARE, ON PURPOSE.
    # The report has no category for it: its "axial" share is YASA-shaped, and
    # DeepDrive is a separate bet by a separate company. Giving it a share means
    # inventing adoption for a machine with no published composition and no
    # registrations. Zero keeps it in the model -- its composition is computed
    # and its draws are written -- without fabricating a market.
    # SAFE TO CHANGE: yes, and it takes its share from `axial` when it gets one.
    traction_dual_rotor_share: float = 0.0

    # ⚠️ SEGMENTS THE COMPOSITION DOES NOT HAVE, AND WHAT TO READ INSTEAD.
    #
    # The fleet tracker carries twelve segments; the consolidated composition
    # carries eleven. The missing one is JA, the smallest light-commercial
    # class -- it is not in the Zenodo dataset, so the traction project cannot
    # report it without inventing it.
    #
    # Left alone it was a SILENT LOSS: 5,370 tracker rows and 3.52 million
    # vehicle-flows, 0.34% of all BEV flow, cohorts 2011 to 2070, dropped by the
    # join and never counted. Small, and still wrong to lose without saying so.
    #
    # JA reads JB: the next light-commercial size up, and already its partner in
    # the AB voltage group. The alternative was A, the passenger car of similar
    # size, and JB was preferred because a van's duty cycle and torque sit
    # closer to another van's.
    # SAFE TO CHANGE: yes, and the entry disappears the day the composition
    # covers JA.
    traction_segment_fallback: dict[str, str] = field(
        default_factory=lambda: {"JA": "JB"})

    # ⚠️ WHICH DRIVE TRAINS HAVE A TRACTION MOTOR OF THIS KIND AT ALL.
    #
    # The composition describes `elvBEV`, so joining the whole tracker against
    # it drops every petrol, diesel and hybrid row -- 42.9% of all vehicle-flow,
    # which looks like a catastrophe in a warning line and is simply a diesel
    # not having a BEV traction motor. Filtered before the join so the count
    # that gets reported is the count that matters.
    #
    # ⚠️ AND PHEV IS A REAL EXCLUSION, NOT A TAUTOLOGY. A plug-in hybrid has a
    # traction motor -- 46,392 rows and 42.3 million vehicle-flows of them -- and
    # this model does not count it, because the composition covers BEVs only.
    # HEV likewise, 65.1 million. That understates European traction-motor
    # material demand by however much those motors weigh, and it is a scope
    # limit of the source rather than a decision made here.
    # SAFE TO CHANGE: yes, the day a composition exists for a hybrid's motor.
    traction_drive_trains: tuple[str, ...] = ("BEV",)

    # ⚠️ WHERE THE FIGURES START, AND WHY IT IS NOT WHERE THE DATA STARTS.
    #
    # Matthias 2026-09-21: "Can we start in 2012. 2011 numbers do not make
    # sense" -- and then, decisively: "2011 is the initial stock."
    #
    # ⚠️ SO IT IS NOT AN ERROR, IT IS A DIFFERENT QUANTITY. The tracker hands
    # this stage 0.2268 million BEVs for 2011 and 0.0545 for 2012, against real
    # EU BEV registrations of roughly 0.01 million in 2011. The first figure is
    # not a year of registrations at all: it is the BEV fleet that already
    # existed when the series begins, injected as the first cohort. BEV alone
    # shows it because BEV alone starts mid-model -- diesel, petrol and HEV
    # begin in 2006 and PHEV in 2012, each continuous with its second year.
    #
    # Which is exactly why it must not be DRAWN beside annual registrations: a
    # stock and a flow on one line, in the same units, inviting the reader to
    # compare them.
    #
    # ⚠️ THE FIGURES ONLY. The data keeps 2011, and deliberately. It is 0.035% of
    # cumulative BEV inflow, so it moves no total worth reporting -- and a stage
    # that silently dropped a cohort the tracker contains would disagree with
    # 04_01, 04_02 and 04_04, which all read the same tracker. A figure that
    # declines to plot a bad year is honest; a stage that discards data other
    # stages keep is a trap for whoever reconciles them.
    #
    # AND IT MUST STAY IN THE DATA. Those cars are really in the fleet: they
    # carry copper and neodymium, and they come back as outflow fifteen to
    # twenty years later. Dropping the cohort to tidy a figure would remove real
    # material from the stock. SAFE TO CHANGE: yes -- it is a drawing window and
    # nothing else.
    traction_figure_first_year: int = 2012

    # Where the traction project keeps its 200,000-draw arrays, relative to
    # `traction_composition_dir`. 31 mass arrays, 12 chemistry arrays, the
    # torque grid and the year x voltage scale table.
    # SAFE TO CHANGE: only if that project moves them.
    traction_draws_dir: str = "draws"


    # ⚠️ WHAT A BEV OF EACH SEGMENT ACTUALLY CARRIES, as a DISCRETE MIXTURE.
    #
    # A segment does not offer a continuum of pack sizes, it offers a handful.
    # Measured from EV_details.csv at 5 kWh resolution, models introduced from
    # 2022: the kept levels cover 66-87% of a segment, and some have one
    # dominant size -- JC's 80 kWh holds 39% of 235 models, JD's 100 kWh 38%.
    # So the wide capacity range inside a segment IS a mixture of a few real
    # pack sizes, not spread around a single one.
    #
    # ONE LEVEL IS DRAWN PER MONTE CARLO DRAW. That keeps the range, which is
    # the point: a draw says "we do not know which pack this car has". The
    # alternative -- splitting the fleet across the levels -- is what a fleet
    # physically is, but it averages the mixture away and collapses the band.
    # Both give the same mean. See documentation/DESIGN_bev_capacity_for_04_04.md.
    #
    # Rule for the levels: at most five, each holding at least 10% of the
    # segment's models, renormalised. A proportional threshold rather than an
    # absolute count, because "at least 3 models" collapsed segment A to a
    # single level and destroyed the very spread this exists to carry.
    #
    # ⚠️ A and JA are ASSUMPTIONS, not measurements. JA has two models, both
    # Hyundai INSTER, and the 50/50 weighting is a choice; A has ten.
    # SAFE TO CHANGE: yes -- weights must be positive and are renormalised.
    battery_capacity_levels: dict[str, dict] = field(default_factory=lambda: {
        "A":  {"levels_kwh": (25.0, 30.0, 35.0), "weights": (0.600, 0.200, 0.200)},
        "B":  {"levels_kwh": (40.0, 45.0, 50.0, 55.0),
               "weights": (0.275, 0.175, 0.200, 0.350)},
        "C":  {"levels_kwh": (55.0, 60.0, 65.0, 80.0, 85.0),
               "weights": (0.197, 0.268, 0.211, 0.183, 0.141)},
        "D":  {"levels_kwh": (80.0, 85.0, 90.0, 100.0),
               "weights": (0.400, 0.283, 0.167, 0.150)},
        "E":  {"levels_kwh": (85.0, 90.0, 100.0), "weights": (0.246, 0.188, 0.565)},
        "F":  {"levels_kwh": (90.0, 95.0, 105.0, 120.0),
               "weights": (0.208, 0.250, 0.361, 0.181)},
        "JA": {"levels_kwh": (42.0, 49.0), "weights": (0.500, 0.500)},
        "JB": {"levels_kwh": (50.0, 55.0, 60.0, 65.0, 70.0),
               "weights": (0.191, 0.353, 0.118, 0.132, 0.206)},
        "JC": {"levels_kwh": (65.0, 70.0, 80.0, 85.0),
               "weights": (0.170, 0.152, 0.538, 0.140)},
        "JD": {"levels_kwh": (75.0, 80.0, 100.0), "weights": (0.156, 0.278, 0.567)},
        "JE": {"levels_kwh": (100.0, 105.0, 115.0), "weights": (0.436, 0.256, 0.308)},
        "JF": {"levels_kwh": (100.0, 110.0, 120.0, 125.0),
               "weights": (0.306, 0.278, 0.139, 0.278)},
    })

    # The year the levels above describe. They come from models introduced 2022
    # onward, and the fleet median has been flat at 82 kWh since 2023, so 2024
    # is the middle of the window they were measured over.
    # SAFE TO CHANGE: only with the levels themselves.
    battery_capacity_levels_year: int = 2024

    # HOW THE LEVELS MOVE, per decade, as a triangular drawn ONCE PER MONTE
    # CARLO DRAW. Growth runs from battery_capacity_levels_year and stops at the
    # plateau year below; before that year the same rate runs backwards, which
    # is how a car scrapped in 2040 gets the capacity of its own build year.
    # SAFE TO CHANGE: yes.
    battery_capacity_growth_per_decade: dict[str, float] = field(
        default_factory=lambda: {"min": 0.05, "mode": 0.10, "max": 0.20})

    # ⚠️ WHEN CAPACITY STOPS GROWING, drawn per Monte Carlo draw alongside the
    # rate. Capacity levels off because RANGE saturates, not because capacity
    # does: once a segment reaches the range its buyers want, more kWh is dead
    # weight and further efficiency gains show up as range at the same capacity.
    # Fast charging removes the pressure to buy range with capacity.
    #
    # Measured: fleet capacity stopped at 82 kWh (68 -> 82 -> 82 by introduction
    # period) while consumption improved 164 -> 158 Wh/km and range gained only
    # 15 km in the last period against 100 km in the one before. C and JC have
    # plateaued; JD and F have not.
    #
    # It is DRAWN rather than fitted because the record cannot settle it: six or
    # seven usable years, 13-20 models per segment in the early periods against
    # 106-158 now, and models rather than registrations. "Plateaued" and
    # "paused" look identical over that span.
    # SAFE TO CHANGE: yes.
    battery_capacity_plateau_year: dict[str, float] = field(
        default_factory=lambda: {"min": 2035.0, "mode": 2040.0, "max": 2050.0})

    # ⚠️ 800 V ADOPTION, which decides how much copper a pack carries.
    #
    # Same power at double the voltage is less current and less conductor: the
    # composition files give a 400 V and an 800 V row per component, and the
    # 800 V one carries a third less copper in the cables and cell terminals.
    # So this driver moves a real material, not a label.
    #
    # WRITTEN-DOWN COPY, NOT READ ACROSS PROJECTS. The source is
    # RAWCLICVehicleElectronics' Data/18_BEV_technology_penetration.xlsx, sheet
    # "Penetration", Driver="Voltage", State="800V" -- the single source of
    # truth that project validates its own sensor and wiring models against
    # (its check V12). It is copied here so this repo runs without that one
    # present, on the same footing as reference maps elsewhere in this file:
    # IF THAT WORKBOOK CHANGES, THIS HAS TO BE UPDATED BY HAND.
    #
    # Values are (Share_Min, Share_Mode, Share_Max) of new sales. The band is
    # real uncertainty, about +/-0.12 on the share, and a caller that uses only
    # the mode is throwing away spread the source deliberately carries.
    #
    # EF leads by a wide margin -- 0.42 in 2025 against CD 0.07 and AB 0.02 --
    # which matches where 800 V actually appears: Chinese platforms and the
    # large segments first.
    # SAFE TO CHANGE: only to track the source workbook.
    pack_voltage_800v_share: dict[str, dict[int, tuple[float, float, float]]] = field(
        default_factory=lambda: {
        "AB": {2010: (0.0, 0.001, 0.121), 2015: (0.0, 0.003, 0.123),
               2020: (0.0, 0.007, 0.127), 2025: (0.0, 0.019, 0.139),
               2030: (0.0, 0.050, 0.170), 2035: (0.003, 0.123, 0.243),
               2040: (0.153, 0.273, 0.393), 2045: (0.380, 0.500, 0.620),
               2050: (0.607, 0.727, 0.847), 2060: (0.830, 0.950, 1.0),
               2070: (0.873, 0.993, 1.0)},
        "CD": {2010: (0.0, 0.0, 0.120), 2015: (0.0, 0.0, 0.120),
               2020: (0.0, 0.005, 0.125), 2025: (0.0, 0.070, 0.190),
               2030: (0.280, 0.400, 0.520), 2035: (0.600, 0.720, 0.840),
               2040: (0.760, 0.880, 1.0), 2045: (0.820, 0.940, 1.0),
               2050: (0.850, 0.970, 1.0), 2060: (0.870, 0.990, 1.0),
               2070: (0.880, 1.0, 1.0)},
        "EF": {2010: (0.0, 0.005, 0.125), 2015: (0.0, 0.027, 0.147),
               2020: (0.003, 0.123, 0.243), 2025: (0.299, 0.419, 0.539),
               2030: (0.667, 0.787, 0.907), 2035: (0.830, 0.950, 1.0),
               2040: (0.870, 0.990, 1.0), 2045: (0.878, 0.998, 1.0),
               2050: (0.880, 1.0, 1.0), 2060: (0.880, 1.0, 1.0),
               2070: (0.880, 1.0, 1.0)},
    })

    # ⚠️ AND A THIRD VOLTAGE STATE: 1000 V, WHICH THE BATTERY SIDE DOES NOT HAVE.
    #
    # Matthias 2026-09-21: "1000V stays separate. I expect 1000V to show up at E
    # and F segment in limited numbers and could become more dominant."
    #
    # The traction composition carries three voltage classes -- 400, 800 and
    # 1000 -- because copper mass falls with voltage, and `scenario.copper_mass`
    # in that project puts 1000 V at 0.585 of the 400 V copper against 800 V's
    # 0.667. Without a share for it the class exists in the composition and
    # never reaches a car.
    #
    # ⚠️ ENTIRELY UNMEASURED, AND THE SHAPE IS MATTHIAS'S. Only BYD ships a
    # 1000 V-class architecture today, and it is sold mainly outside Europe, so
    # there is no European registration base to fit. These numbers say: nothing
    # before 2030, EF first and fastest because that is where charging power is
    # worth paying for, CD later and smaller, AB barely at all. They are a
    # stated expectation written as a curve, not evidence.
    #
    # THE THREE STATES MUST NOT OVERLAP. 1000 V is taken out of the 800 V
    # population, not added on top: a car draws 1000 V first, then 800 V from
    # what is left, then 400 V. The code does that; these are the raw shares.
    # SAFE TO CHANGE: yes, and the first real registration data replaces it.
    pack_voltage_1000v_share: dict[str, dict[int, tuple[float, float, float]]] = field(
        default_factory=lambda: {
            "AB": {2010: (0.0, 0.0, 0.0), 2025: (0.0, 0.0, 0.0),
                   2030: (0.0, 0.0, 0.0), 2040: (0.0, 0.0, 0.01),
                   2050: (0.0, 0.01, 0.03), 2060: (0.0, 0.02, 0.05),
                   2070: (0.0, 0.03, 0.08)},
            "CD": {2010: (0.0, 0.0, 0.0), 2025: (0.0, 0.0, 0.0),
                   2030: (0.0, 0.0, 0.01), 2040: (0.0, 0.03, 0.08),
                   2050: (0.02, 0.08, 0.18), 2060: (0.03, 0.12, 0.24),
                   2070: (0.05, 0.15, 0.30)},
            "EF": {2010: (0.0, 0.0, 0.0), 2025: (0.0, 0.0, 0.01),
                   2030: (0.0, 0.02, 0.05), 2040: (0.03, 0.10, 0.20),
                   2050: (0.08, 0.20, 0.35), 2060: (0.12, 0.28, 0.45),
                   2070: (0.15, 0.33, 0.50)},
        })

    # Which of the source's three groups each segment belongs to. The voltage
    # driver is resolved at AB/CD/EF because that is the grain it was built at;
    # capacity stays per segment, where the evidence supports it.
    # SAFE TO CHANGE: yes, but every segment in segment_map needs an entry.
    pack_voltage_segment_groups: dict[str, str] = field(default_factory=lambda: {
        "A": "AB", "B": "AB", "JA": "AB", "JB": "AB",
        "C": "CD", "D": "CD", "JC": "CD", "JD": "CD",
        "E": "EF", "F": "EF", "JE": "EF", "JF": "EF",
    })

    # WHERE THE BATTERY COMPOSITION FILES ARE, written by RAWCLICVehicleBattery's
    # 05_composition.py. Nine `consolidated_<chemistry>.csv`, their
    # `<chemistry>_<capacity>kWh_<voltage>V_mass_draws.npy` arrays in kilograms,
    # and `improvement_factor_draws_<year>.npy`.
    #
    # An absolute path, because that project is a sibling checkout and neither
    # repo may assume where the other sits. Nothing under it is ever written.
    # SAFE TO CHANGE: yes, when that project moves.
    battery_composition_dir: str = (
        "/Users/rm/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/"
        "RAWCLICVehicleBattery/data/consolidated")

    # ⚠️ EXTRA UNCERTAINTY FOR EXTRAPOLATED CAPACITY, as a fraction of the mass,
    # per 100 kWh beyond the composition files' top anchor.
    #
    # Those files stop at 100 kWh. The capacity rule puts JE and JF above that
    # in every draw from 2030 and F in three quarters of them, so a large part
    # of the fleet is read off a straight line continued past the last anchor.
    # A linear continuation is the least the data can be made to say, but it is
    # not as well known as an interpolation, and without this it would arrive
    # carrying exactly the same band.
    #
    # Applied as a triangular multiplier centred on 1, widening with distance:
    # at 150 kWh the half-width is half of this, at 200 kWh the whole of it.
    # SAFE TO CHANGE: yes -- and it is a judgement, not a measurement.
    battery_extrapolation_uncertainty_per_100kwh: float = 0.10

    # ⚠️ CHEMISTRY SHARES TO 2070, three scenarios. ASSUMPTION, NOT DATA --
    # the observed record ends in 2026 and everything after it is judgement.
    #
    # WRITTEN-DOWN COPY of RAWCLICVehicleBattery's src/scenarios.py, on the same
    # footing as the voltage table above: copied so this repo runs without that
    # one present, and IF THAT PROJECT CHANGES ITS SCENARIOS THIS MUST BE
    # UPDATED BY HAND.
    #
    # Values are percentages at the anchor years, interpolated between and held
    # flat outside, then renormalised per group and year.
    #
    # S1  LFP volume, NMC premium, LMFP growing, nothing new ever arrives
    # S2  sodium enters the small segments, NMC shrinks to a niche
    # S3  S2 plus bipolar solid-state from 2040, large segments first
    #
    # ⚠️ Na_ion and solid_state HAVE NO COMPOSITION. Their share is carried and
    # reported as a gap rather than silently dropped -- under S3 that is most of
    # the market by 2070, and a total that quietly fell would read as a collapse
    # in demand rather than a hole in the data.
    # SAFE TO CHANGE: yes, to track the battery project.
    battery_chemistry_anchor_years: tuple[int, ...] = (2025, 2035, 2050, 2070)

    battery_chemistry_segment_groups: dict[str, str] = field(default_factory=lambda: {
        "A": "small", "B": "small", "JA": "small", "JB": "small",
        "C": "medium", "D": "medium", "JC": "medium", "JD": "medium",
        "E": "large", "F": "large", "JE": "large", "JF": "large",
    })

    battery_chemistry_scenarios: dict[str, dict[str, dict[str, tuple]]] = field(
        default_factory=lambda: {
        "S1": {
            "small":  {"LFP": (70, 65, 60, 60), "LMFP": (18, 25, 33, 35),
                       "NMC_high": (12, 10, 7, 5)},
            "medium": {"LFP": (50, 45, 42, 40), "LMFP": (22, 32, 42, 45),
                       "NMC_high": (28, 23, 16, 15)},
            "large":  {"LFP": (8, 10, 10, 10), "LMFP": (17, 25, 32, 35),
                       "NMC_high": (75, 65, 58, 55)},
        },
        "S2": {
            "small":  {"Na_ion": (2, 40, 60, 68), "LFP": (68, 40, 25, 20),
                       "LMFP": (18, 15, 12, 10), "NMC_high": (12, 5, 3, 2)},
            "medium": {"Na_ion": (0, 12, 20, 24), "LFP": (50, 48, 42, 38),
                       "LMFP": (22, 32, 34, 35), "NMC_high": (28, 8, 4, 3)},
            "large":  {"Na_ion": (0, 3, 8, 10), "LFP": (8, 18, 24, 25),
                       "LMFP": (17, 49, 58, 57), "NMC_high": (75, 30, 10, 8)},
        },
        "S3": {
            "small":  {"solid_state": (0, 0, 15, 35), "Na_ion": (2, 40, 52, 45),
                       "LFP": (68, 40, 20, 12), "LMFP": (18, 15, 11, 7),
                       "NMC_high": (12, 5, 2, 1)},
            "medium": {"solid_state": (0, 0, 28, 50), "Na_ion": (0, 12, 15, 14),
                       "LFP": (50, 48, 30, 18), "LMFP": (22, 32, 25, 17),
                       "NMC_high": (28, 8, 2, 1)},
            "large":  {"solid_state": (0, 2, 45, 70), "Na_ion": (0, 3, 5, 5),
                       "LFP": (8, 18, 12, 6), "LMFP": (17, 49, 33, 17),
                       "NMC_high": (75, 28, 5, 2)},
        },
    })

    # The scenario names above mapped onto the composition files' own chemistry
    # names. A name with no entry has no composition and is reported as a gap.
    # SAFE TO CHANGE: only to match the composition files.
    battery_chemistry_file_names: dict[str, str] = field(default_factory=lambda: {
        "LFP": "battLiFP_subsub", "LMFP": "battLiMFP_subsub",
        "NMC_high": "battLiNMC_highNi",
        # Packaging only -- see battery_chemistry_active_material_unknown.
        "Na_ion": "Na_ion", "solid_state": "solid_state",
    })

    # ⚠️ CHEMISTRIES WHOSE ACTIVE MATERIAL NOBODY HAS DESCRIBED.
    #
    # These two DO have a composition file, and it is real: the casing, the
    # separator, the cables, the terminals, the enclosure, the frame, the
    # thermal conductor and both current collectors, at the mass of the pack
    # they are modelled on. What it does not have is the cathode, the anode and
    # the electrolyte -- the cell itself -- because nobody has published one
    # that survives scrutiny, and a plausible number borrowed from a lithium
    # chemistry would be a claim nobody made.
    #
    # In their arrays the active materials are ZERO, and zero there means NOT
    # DESCRIBED rather than none present. This list is what stops that zero
    # being read as a fact: the cars carrying these chemistries are reported as
    # a gap, at the same share as before they had any composition at all. What
    # changed is that their steel, aluminium and copper now reach the totals.
    # SAFE TO CHANGE: remove a name the day a real composition arrives for it,
    # and not before.
    battery_chemistry_active_material_unknown: tuple[str, ...] = (
        "Na_ion", "solid_state")

    # WHERE 04_04 WRITES THE DRAWS THE RECOVERY MODEL READS, under
    # data/processed/. One folder per chemistry scenario, then per flow:
    #
    #     <scenario>/<flow>/years.npy
    #     <scenario>/<flow>/__component____<component>.npy   (draws, years)
    #     <scenario>/<flow>/<element>__<component>.npy       (draws, years)
    #
    # In KILOTONNES, which is the unit RAWCLICRecoveryModel's `src/upstream.py`
    # expects, and summed over the chemistries: a recycler receives the mix, not
    # one chemistry at a time.
    #
    # WHY THE CROSS AND NOT THE ELEMENT TOTAL. Copper in a cable and copper in
    # an electrode foil go through different processes and are recovered at
    # different rates. An element total cannot be given one coefficient that is
    # right for both, so the recovery model is handed the element WITHIN the
    # component and decides per component.
    # SAFE TO CHANGE: yes, it is only a folder name.
    battery_recovery_draws_dir: str = "battery_recovery_draws"

    # WHICH YEARS OF THAT EXPORT TO WRITE.
    #
    # Same trade as the other two stages that feed the recovery model: this
    # stage computes every year anyway and the setting only decides which of
    # them survive to disk. Eleven years at 200,000 draws is about 300 MB per
    # scenario and flow; all fifty-one would be five times that for a model
    # that reports a year or a short span.
    # SAFE TO CHANGE: yes. An empty tuple writes nothing.
    battery_recovery_years: tuple[int, ...] = tuple(range(2020, 2071, 5))

    # ⚠️ HOW WRONG THE CHEMISTRY SHARES MIGHT BE, as a multiplier on each one.
    #
    # The scenarios above are stated assumptions and the model does not argue
    # with them. What it does say is that a share stated for 2070 is a guess made
    # forty-five years early, and a share stated for 2021 is nearly a
    # measurement. This is that distance, as a TRIANGULAR multiplier on each
    # chemistry's stated share.
    #
    # EVERY CHEMISTRY IS DRAWN AND THE GROUP IS THEN RENORMALISED TO ONE. None of
    # them is a residual that absorbs whatever the others left over -- that would
    # give one chemistry all the uncertainty and the rest none, and which one got
    # it would depend on the order they were written in. The renormalisation is
    # what makes them correlated: in a draw where sodium runs ahead, the others
    # give way.
    #
    # ONE DRAW PER CHEMISTRY, shared across the segment groups and held across
    # every year. Sodium beating expectations is one event, not twelve.
    #
    # 0.70 / 1.00 / 1.30 means: at full width, a chemistry's share can be 30%
    # below or above what the scenario states -- RELATIVE to that share, so a
    # stated 40% runs 28-52% and a stated 4% runs 2.8-5.2%. Make it asymmetric by
    # moving `mode` off 1.00, which is a statement that the scenario is more
    # likely wrong in one direction.
    # SAFE TO CHANGE: yes. min <= mode <= max, and all three positive.
    battery_chemistry_share_spread: dict[str, float] = field(default_factory=lambda: {
        "min": 0.70, "mode": 1.00, "max": 1.30,
    })

    # Where that width ramps from nothing to its full value. At the first year
    # the shares are exactly what the scenario states; at the second and beyond,
    # the full spread above. Linear between.
    # SAFE TO CHANGE: yes. The first year is "what we can see now".
    battery_chemistry_share_spread_years: tuple[int, int] = (2020, 2070)

    # Fallback battery size used when a vehicle's segment is unknown.
    # SAFE TO CHANGE: yes. Ideally it stays near the middle of the map above.
    average_battery_capacity_kwh: float = 60.0

    # The workbook holding the pre-computed composition statistics (means and spreads)
    # that this stage reads instead of recomputing them.
    # SAFE TO CHANGE: yes, when a newer version arrives.
    composition_summary_file_name: str = "36_MonteCarlo_Summary.xlsx"

    # Which single number to take from that summary file for the ordinary, non-Monte
    # Carlo run: "mean" or "median".
    # SAFE TO CHANGE: yes. Does not affect the uncertainty analysis, which uses the
    # full distributions rather than one number.
    composition_scalar_statistic: str = "mean"

    # The workbook holding the full composition distributions -- the shapes the
    # uncertainty analysis draws from, rather than a single average.
    # SAFE TO CHANGE: yes, when a newer version arrives.
    histogram_file_name: str = "37_MonteCarlo_Histograms.xlsx"

    # Which sheet inside that workbook belongs to which drivetrain.
    # SAFE TO CHANGE: only if the workbook's sheet names change. A wrong name here
    # means the material composition of the wrong vehicle type is used.
    histogram_sheet_names_by_drv: dict[str, list[str]] = field(default_factory=lambda: {
        "Petrol": ["componentCarPetrol"],
        "Diesel": ["componentCarDiesel"],
        "BEV": ["componentCarBEV"],
        "HEV": ["componentCarHEV"],
        "PHEV": ["componentCarPHEV"],
    })

    # How finely material results are reported over time:
    #     "period"  one figure per reporting window  (current -- far smaller and faster)
    #     "year"    a figure for every single year   (much larger, much slower)
    # SAFE TO CHANGE: yes, but "year" combined with 200,000 draws produces very large
    # files -- the existing per-scenario ones already run to several GB.
    material_mc_time_resolution: str = "period"

    # How many uncertainty draws this stage runs. Same meaning as `n_draws` in the
    # Monte Carlo section, but kept separate so this stage can be run at a different
    # cost from the rest of the pipeline.
    # SAFE TO CHANGE: yes -- lower it for a quick check.
    materials_mc_n_draws: int = 200_000

    # Fixes this stage's random numbers so a re-run reproduces identical results.
    # SAFE TO CHANGE: yes, any whole number.
    materials_mc_seed: int | None = 42

    # KEEP THE FULL PER-DRAW MASS ARRAYS ON DISK?
    #
    # This stage builds, for every scenario and flow, the mass of every material in
    # every (drivetrain, segment, component) combination for all 200,000 draws. Those
    # arrays are enormous -- 8 to 17 GB per scenario-flow, and roughly 50 GB in total
    # for a two-scenario run.
    #
    # NOTHING IN THE PIPELINE READS THEM BACK. They were written for a consumer that
    # was never built: no stage loads them, and they are deliberately not in the
    # artifact registry, so `load_many` cannot reach them either. The code frees them
    # from memory the moment they are written, so even this stage is finished with
    # them. Everything downstream uses the summaries, which are small and are always
    # saved.
    #
    # False -- do not write them. Nothing in the pipeline notices, and about 50 GB of
    #          disk is not consumed. This is the default.
    # True  -- write them, for analysis outside this pipeline. Check you have the
    #          disk: at 200,000 draws it is tens of GB per run.
    #
    # SAFE TO CHANGE: yes. Turning it on costs only disk; turning it off costs
    # nothing, because the results the pipeline uses do not come from these files.
    persist_mc_mass_draws: bool = False

    # =======================================================================
    # BEV ELECTRONICS  (code/04_02_BEVelectronics.py)
    # -----------------------------------------------------------------------
    # Links this fleet model to the separate BEV-electronics study: how much
    # wiring, sensor, circuit-board and motor material enters the fleet, leaves
    # it, and is collected from it, year by year.
    #
    # The electronics study reports grams PER VEHICLE as a full distribution;
    # this model supplies HOW MANY vehicles, also as a distribution. Multiplying
    # them draw by draw gives the total, with both uncertainties carried
    # through.
    # =======================================================================

    # Where the electronics study keeps its raw per-draw arrays, written by its
    # `tools/mc_composition.py`. One file per segment and series, e.g.
    # "AB_Total.npy", each shaped (draws, years).
    # SAFE TO CHANGE: only if you move that repository.
    bev_electronics_draws_dir: str = (
        "../../RAWCLICVehicleElectronics/Composition/draws"
    )

    # Which parts of the electronics to report. "Total" is the sum of the other
    # four and is what most results use.
    # SAFE TO CHANGE: yes, to narrow the output. Names must match the .npy files.
    bev_electronics_series: tuple[str, ...] = ("Total", "Wiring", "Sensors", "PCB", "Motors")

    # HOW THE ELECTRONICS SEGMENTS MAP ONTO THIS MODEL'S SEGMENTS.
    # The electronics study groups cars into three sizes -- AB, CD, EF -- while
    # this model uses twelve. Each electronics group covers a PAIR: AB covers
    # both A and B.
    #
    # A and B are not identical, though: within the pair, A is the smaller car
    # and carries slightly less electronics, B slightly more. So rather than
    # giving both the same distribution, the pair's distribution is TILTED --
    # A draws a little more often from its lower end, B from its upper end.
    # Crucially the two halves still add back up to exactly the original AB
    # distribution: nothing is invented and nothing is lost.
    #
    # The J-segments (JA-JF) have no electronics counterpart of their own, so
    # each takes its non-J twin: JA uses A's distribution, JB uses B's, and so
    # on.
    # SAFE TO CHANGE: only if the electronics study changes its own grouping.
    # Written as a plain tuple, not a dict, deliberately: a dataclass field with a
    # dict default needs `default_factory`, and such fields live only on the
    # INSTANCE. An older saved parameter file would then lack this entry entirely
    # and fail on load. A tuple is immutable, so it lives on the class and older
    # files simply inherit it.
    bev_electronics_segment_pairs: tuple[tuple[str, str, str], ...] = (
        ("AB", "A", "B"), ("CD", "C", "D"), ("EF", "E", "F"),
    )

    # HOW STRONGLY TO TILT that split. 0 means no tilt at all -- A and B would
    # get identical distributions. Larger means the two pull further apart.
    #
    # THIS IS NOT A PERCENTAGE, and the difference matters. It is the strength of
    # the re-weighting, not the resulting gap between the two segments' averages.
    # Measured on the real AB pool at 200,000 draws:
    #
    #     tilt      resulting gap between A's and B's mean
    #     0.20                3.1%      <- the current setting
    #     0.40                6.3%
    #     0.60                9.5%
    #     0.77               12.3%      <- the ceiling for this pair
    #
    # THERE IS A HARD CEILING, and it is not arbitrary. The weight given to the
    # smaller segment is `share + tilt x (0.5 - rank)`. Push the tilt past
    # `2 x min(share, 1 - share)` and that weight goes negative for the largest
    # draws, at which point the two halves no longer add back to the electronics
    # study's own published distribution -- the model would be reporting a
    # composition nobody measured. For AB, where A holds about 61.5% of the
    # vehicles, the ceiling is 0.77 and so the widest achievable separation is
    # about 12.3%. A 20% separation is not reachable at all. The code stops with
    # an explanatory error rather than clipping the weight and carrying on.
    #
    # HOW GOOD IS THE RECOMBINATION IN PRACTICE? The split resamples the pool
    # under those weights, so it carries ordinary Monte Carlo noise, which shrinks
    # with draw count. Measured on the real AB pool:
    #
    #        draws        error in reproducing the published mean
    #          300                    0.10%
    #       30,000                    0.017%
    #      200,000                    0.007%
    #
    # At the production draw count the split is faithful to seven parts in
    # 100,000. Figure 04_02_02 prints this error on every panel, so a run that
    # drifts is visible rather than assumed.
    #
    # SAFE TO CHANGE: yes -- this is the dial you are most likely to adjust. Any
    # value from 0 up to the ceiling described above.
    bev_electronics_segment_tilt: float = 0.2

    # The "standard" case: one average BEV rather than twelve segment-specific
    # ones, built as a mixture across A-F so it comes out as a full
    # distribution, not a single number.
    # True  -- weight each segment by how many vehicles are actually in it
    # False -- weight all six equally
    # SAFE TO CHANGE: yes.
    bev_electronics_standard_fleet_weighted: bool = True

    # First and last year to report. The electronics study itself only covers
    # 2020-2070, so asking for earlier years would have nothing to multiply.
    # SAFE TO CHANGE: yes, within the electronics study's own range.
    bev_electronics_year_min: int = 2020
    bev_electronics_year_max: int = 2070

    # Where the electronics study keeps its per-draw ELEMENT arrays, written by
    # the three element models (ElectricMotorElementMC, PCBElementMC,
    # SensorElementsMC). One file per domain and segment.
    # SAFE TO CHANGE: only if you move that repository.
    bev_electronics_element_draws_dir: str = (
        "../../RAWCLICVehicleElectronics/Composition/element_draws"
    )

    # WHICH ELEMENTS TO REPORT.
    #
    # EMPTY MEANS EVERY ELEMENT THE DRAWS RESOLVE, and empty is the default. The
    # stage reads the element names out of the `*_elements.txt` files that sit
    # beside the `.npy` draws, so what gets reported follows whatever the element
    # models produced. Nothing in this repository holds a list of element names.
    #
    # That is deliberate. A hard-coded list made the stage fail outright the first
    # time it met a set of draws without Pr, Tb and Nb in it -- the code was welded
    # to names that are not its to decide. The available set is a property of the
    # upstream models' output files and it changes when those models change.
    #
    # Naming elements here narrows the report to a subset, which is useful when you
    # want the critical raw materials only rather than iron, silicon and the trace
    # `*_ppm` impurities. A name no domain resolves is skipped with a note listing
    # what was available; only a list where NOTHING resolves stops the run, since
    # that means the draws directory is wrong.
    #
    # Two things to know before narrowing it. Some names are not elements at all --
    # `Plastic` and `Unspecified` are real rows in the motor model. And an element
    # that comes from a single domain, such as Pd from PCB or Pt from sensors,
    # carries only that one model's uncertainty, so its band is narrower than a
    # multi-domain element's for a reason that is not physical.
    #
    # THE SELECTION: critical and strategic raw materials only.
    #
    # Empty would report all 62 names the draws carry, which is not useful -- most
    # of them are sulfur, oxygen, carbon and phosphorus, or copper-winding
    # contamination, and none of those is a recovery target.
    #
    # Names ending `__<material>` come from the motor model, where an element is
    # identified by the material it sits in. `Cu` has no suffix because copper is
    # summed across every motor material by decision. PCB and sensor elements have
    # no suffix either -- those models resolve one material each.
    #
    # WHAT WAS LEFT OUT, and why:
    #   the 16 `__copper` entries   contamination in the winding, not a target.
    #                               Bi, Sb, Se, Te and Cd occur ONLY there, so they
    #                               leave the report entirely.
    #   S, O, C, P everywhere       not of interest, whatever material they sit in.
    #   Fe and its variants         bulk, not critical.
    #   Si, Si__esteel, Ba          on the CRM list but alloying or bulk here.
    #   Plastic, Unspecified        not elements.
    #
    # Mn__esteel and Mn__cfsteel ARE included: manganese in the steels is an
    # alloying addition, not contamination, so the rule that drops the `__copper`
    # entries does not reach them.
    #
    # SAFE TO CHANGE: yes. This only selects what is reported; it does not change
    # any calculation. A name no domain resolves is skipped with a note, so this
    # list cannot break a run.
    bev_electronics_elements: tuple[str, ...] = (
        "Cu",                                       # the priority element
        "Nd", "Dy",                                 # magnet rare earths, sensors
        "Sr__magnet",                               # the motor's ferrite magnet
        "Co", "Li",                                 # battery-adjacent, sensors
        "Pt", "Pd", "Au", "Ag",                     # precious, PCB and sensors
        "Ga", "Ge", "In", "Ta", "W", "Ti", "B",     # semiconductor and hard metals
        "Ni", "Mn", "Mn__esteel", "Mn__cfsteel",    # CRM, incl. steel alloying
        "Al__bulk",                                 # housing and frame aluminium
    )

    # Whether stage 03_02 exports the BEV per-year, per-draw vehicle counts that
    # stage 04_02 needs.
    #
    # It costs real time: about twelve extra Monte Carlo calls, one per BEV
    # segment, on top of the normal 03_02 run. Roughly 1.5 GB lands in
    # data/processed/bev_draws/.
    #
    # It has to happen in 03_02 rather than in 04_02 because the inflow these
    # draws come from is the scenario's own resolved inflow, which exists only
    # inside that stage. Exporting from there means 04_02 reads the real numbers
    # instead of trying to reconstruct them.
    #
    # SAFE TO CHANGE: yes. Set False if you are not going to run 04_02 and want
    # 03_02 back at its normal speed. 04_02 stops with a clear message if the
    # files are missing.
    bev_electronics_export_draws: bool = True

    # WHICH YEARS' PER-ELEMENT DRAWS TO WRITE OUT, for the recovery model.
    #
    # This stage already computes element mass as (draws, years) arrays and then
    # throws them away, keeping only percentiles -- holding 18 elements x 4
    # domains x 200,000 draws for every year would be about 60 GB. That is the
    # right default for a stage that only needs to plot bands.
    #
    # The recovery model needs the draws themselves, because it multiplies them
    # by transfer coefficients that are also drawn, and a mean times a mean is
    # not the mean of the product. It does not need every year: one year, or a
    # short span, is what a recovery result is reported for.
    #
    # So a narrow slice is written instead of nothing. MEASURED on the current
    # export, which writes the alloys rather than every element: 131 MB per year
    # at 200,000 draws across 4 domains and 3 flows.
    #
    #     11 years, every fifth 2020-2070    1.4 GB
    #     51 years, every year               6.7 GB     <- the default below
    #
    # EVERY YEAR, BECAUSE A DOWNSTREAM QUESTION MUST NOT COST AN UPSTREAM RUN.
    # This stage computes all 51 years whatever this says; the setting only
    # decides which of them survive to disk. Writing a subset meant that asking
    # the recovery model for a year outside it required re-running this stage --
    # minutes here, and the whole point of exporting draws was to avoid exactly
    # that. 6.7 GB is one-off and the disk has it; a re-run is paid every time
    # somebody changes their mind about a year.
    #
    #     ()            write nothing (the old behaviour)
    #     (2040,)       that one year
    #     (2030, 2040)  those two years
    #     tuple(range(2020, 2071, 5))  a span with a step
    #
    # WHY IT MATTERS WHICH YEARS ARE HERE. The recovery model's `run.years`
    # selects from what this wrote, so a year missing here cannot be run there
    # -- and it cannot be interpolated either, because the model needs the
    # DRAWS and not a summary. Asking that model for 2020-2070 while this said
    # five years returned five years; it now says so plainly rather than
    # narrowing in silence, but the fix is here.
    #
    # SAFE TO CHANGE: yes. Years outside the run's own range are ignored with a
    # note rather than silently dropped.
    bev_electronics_element_draws_years: tuple[int, ...] = tuple(range(2020, 2071))

    # WHERE THOSE PER-ELEMENT DRAWS ARE WRITTEN, under data/processed/.
    # One folder per scenario, then per flow, then one .npy per element and per
    # element-and-domain. The recovery model reads this folder.
    # SAFE TO CHANGE: yes.
    bev_electronics_element_draws_out_dir: str = "element_draws"

    # WHICH YEARS OF 04_01's MASS DRAWS TO WRITE for RAWCLICRecoveryModel.
    #
    # Same trade as bev_electronics_element_draws_years above: the full
    # per-draw arrays are 8 to 17 GB per scenario-flow (see
    # persist_mc_mass_draws, which is off for exactly that reason), while a
    # few named years are affordable and are what a recovery result is
    # reported for. One year of the 476 (drivetrain, component, material)
    # combinations is about 0.4 GB at 200,000 draws.
    #
    # A YEAR HERE MUST ALSO BE A SINGLE-YEAR PERIOD IN
    # monte_carlo.output_periods, because 04_01's draws are cumulative over a
    # period and the recovery model's axis is years. The default
    # output_periods is one entry covering 1975-2070, which is cumulative and
    # therefore exports nothing. To get an annual axis, set:
    #
    #     output_periods = [(y, y) for y in (2030, 2035, 2040, 2045, 2050)]
    #
    # A year with no matching single-year period is skipped with a note.
    #
    #     ()        write nothing
    #     (2040,)   that one year
    #
    # EVERY YEAR, and this setting alone decides it. 04_01 now computes a
    # single-year period for each year named here, writes it, and drops it again
    # before anything reports -- so monte_carlo.output_periods no longer has to
    # carry them and stages 02, 03_01 and 03_02 are untouched.
    #
    # THE COST IS COMPUTE, NOT DISK, and it is the real one. MEASURED on this
    # machine at 50,000 draws: 1.2 minutes per period, and the period loop runs
    # once per flow -- there are two, `inflow` and `collected`. So the run time
    # is roughly 2.4 minutes per year in this list, on top of the two reporting
    # periods:
    #
    #     every 5th year, 2020-2070   11 years   13 periods/flow   ~31 min   1.6 GB
    #     every 2nd year              26 years   28 periods/flow   ~62 min   3.8 GB
    #     every year                  51 years   53 periods/flow   ~2h 06m   7.6 GB
    #
    # Every year was tried on 2026-09-02 and abandoned at the 1h 50m mark. A
    # step of 5 is what the recovery model was actually asked for, and it cannot
    # use many more: five drivetrains across all 51 years is about 84 GB of
    # result against a 4 GB memory budget, so it would refuse to run them anyway.
    #
    # If a specific intermediate year is ever needed, add it here and re-run --
    # that is a targeted 31 minutes, not a standing two-hour tax on every run.
    #
    # SAFE TO CHANGE: yes, on its own. Nothing else moves with it any more.
    carcomposition_draws_years: tuple[int, ...] = tuple(range(2020, 2071, 5))

    # WHERE THOSE DRAWS ARE WRITTEN, under data/processed/.
    # One folder per scenario, then one per <drivetrain>_<flow>, then one .npy
    # per component and per component-and-material. One folder per drivetrain
    # because the recovery model has a single product at Layer 1 and here that
    # product IS the drivetrain -- a battery is pulled from a BEV and a
    # catalytic converter from a Petrol, so they are different studies.
    # SAFE TO CHANGE: yes.
    carcomposition_draws_out_dir: str = "carcomposition_draws"

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not (0.0 <= self.bev_electronics_segment_tilt <= 1.0):
            issues.append(
                f"materials.bev_electronics_segment_tilt="
                f"{self.bev_electronics_segment_tilt} must be between 0 and 1."
            )
        if self.bev_electronics_year_min > self.bev_electronics_year_max:
            issues.append(
                "materials.bev_electronics_year_min must not be after "
                "bev_electronics_year_max."
            )
        if "Total" not in self.bev_electronics_series:
            issues.append(
                "materials.bev_electronics_series must include 'Total' -- the "
                "headline results are built from it."
            )
        if len(set(self.bev_electronics_elements)) != len(self.bev_electronics_elements):
            dupes = sorted({e for e in self.bev_electronics_elements
                            if list(self.bev_electronics_elements).count(e) > 1})
            issues.append(
                f"materials.bev_electronics_elements repeats {dupes}. A repeated "
                f"element would be counted once but plotted twice."
            )
        if self.materials_mc_n_draws <= 0:
            issues.append(f"materials.materials_mc_n_draws={self.materials_mc_n_draws} must be positive.")
        if self.composition_scalar_statistic not in {"mean", "median", "mode"}:
            issues.append(
                f"materials.composition_scalar_statistic={self.composition_scalar_statistic!r} "
                f"is not one of ['mean', 'median', 'mode']."
            )
        if self.material_mc_time_resolution not in {"period", "annual", "both"}:
            issues.append(
                f"materials.material_mc_time_resolution={self.material_mc_time_resolution!r} "
                f"is not one of ['period', 'annual', 'both']."
            )
        missing_histogram_drvs = set(self.drivetrains) - set(self.histogram_sheet_names_by_drv)
        if missing_histogram_drvs:
            issues.append(
                f"materials.histogram_sheet_names_by_drv is missing drivetrain(s) present "
                f"in drivetrains: {sorted(missing_histogram_drvs)}."
            )
        extra_histogram_drvs = set(self.histogram_sheet_names_by_drv) - set(self.drivetrains)
        if extra_histogram_drvs:
            issues.append(
                f"materials.histogram_sheet_names_by_drv has drivetrain(s) not present in "
                f"drivetrains: {sorted(extra_histogram_drvs)} -- these will never be read."
            )
        for drv, sheet_names in self.histogram_sheet_names_by_drv.items():
            if not sheet_names:
                issues.append(f"materials.histogram_sheet_names_by_drv['{drv}'] is empty -- needs at least one sheet name.")
        if self.composition_parameter_code not in {"m-c", "e-m"}:
            issues.append(
                f"materials.composition_parameter_code={self.composition_parameter_code!r} "
                f"is not one of ['m-c', 'e-m']."
            )
        valid_levels = {
            "materialKeyLevel0", "materialKeyLevel1", "materialKeyLevel2",
            "materialKeyLevel3", "materialKeyLevel4", "materialKeyLevel_highest",
        }
        if self.material_level_key not in valid_levels:
            issues.append(
                f"materials.material_level_key={self.material_level_key!r} is not one "
                f"of {sorted(valid_levels)}."
            )
        levels = self.battery_capacity_levels
        missing_levels = set(self.segment_map) - set(levels)
        if missing_levels:
            issues.append(
                f"materials.battery_capacity_levels is missing segments present in "
                f"segment_map: {sorted(missing_levels)}.")
        for segment, entry in levels.items():
            kwh, weights = entry.get("levels_kwh", ()), entry.get("weights", ())
            if len(kwh) != len(weights) or not kwh:
                issues.append(
                    f"materials.battery_capacity_levels[{segment!r}] has "
                    f"{len(kwh)} levels and {len(weights)} weights.")
                continue
            if any(v <= 0 for v in kwh):
                issues.append(
                    f"materials.battery_capacity_levels[{segment!r}] has a "
                    f"non-positive capacity: {kwh}")
            if any(w <= 0 for w in weights):
                issues.append(
                    f"materials.battery_capacity_levels[{segment!r}] has a "
                    f"non-positive weight: {weights}. A level nobody buys should "
                    "be removed, not given weight zero.")
        for name, band in (("battery_capacity_growth_per_decade",
                            self.battery_capacity_growth_per_decade),
                           ("battery_capacity_plateau_year",
                            self.battery_capacity_plateau_year)):
            missing_keys = {"min", "mode", "max"} - set(band)
            if missing_keys:
                issues.append(f"materials.{name} is missing {sorted(missing_keys)}.")
            elif not band["min"] <= band["mode"] <= band["max"]:
                issues.append(
                    f"materials.{name} must satisfy min <= mode <= max: {band}")

        spread = self.battery_chemistry_share_spread
        missing = {"min", "mode", "max"} - set(spread)
        if missing:
            issues.append(
                f"materials.battery_chemistry_share_spread is missing {sorted(missing)}.")
        elif not 0 < spread["min"] <= spread["mode"] <= spread["max"]:
            issues.append(
                "materials.battery_chemistry_share_spread must satisfy "
                f"0 < min <= mode <= max: {spread}")
        first, last = self.battery_chemistry_share_spread_years
        if first >= last:
            issues.append(
                "materials.battery_chemistry_share_spread_years must rise: "
                f"{(first, last)}")
        return issues


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonteCarloParams:
    """
    Settings for the uncertainty analysis, shared by every stage that runs one.

    Turning this off does not change any stage's ordinary single-run results -- it only
    decides whether the stages ALSO do the extra uncertainty pass on top.
    """

    # Run the uncertainty analysis at all?
    # SAFE TO CHANGE: yes. False makes the pipeline much faster and still produces
    # every ordinary result -- you just get no ranges, no shaded bands, no histograms.
    enabled: bool = True

    # How many times to re-run the model with freshly drawn inputs.
    # SAFE TO CHANGE: yes -- this is the main speed/quality dial.
    #     ~1,000     seconds to run, visibly rough ranges; fine for testing a change
    #    ~20,000     a good working compromise
    #    200,000     production quality, smooth distributions (the current setting)
    # More draws never changes the ANSWER, only how precisely the range is pinned down.
    # Memory grows with it too: 200,000 draws peaks at roughly 6.6 GB.
    n_draws: int = 200000

    # Fixes the random numbers, so re-running reproduces identical results.
    # SAFE TO CHANGE: yes, any whole number. Change it only if you deliberately want a
    # different random sample. Setting it to None gives a different sample every run,
    # which makes results non-reproducible -- avoid that for anything you report.
    seed: int | None = 42

    # ---- Separate budget for the standalone stock-and-flow uncertainty script -----
    # `mc_stockflow_uncertainty.py` varies lifetime, unknown-whereabouts share and
    # export share for every drivetrain at once. It has its OWN draw count and seed so
    # that running a quick demo elsewhere can never accidentally start a 200,000-draw
    # job here.
    # SAFE TO CHANGE: yes -- same meaning as `n_draws` / `seed` above.
    stockflow_n_draws: int = 200_000
    stockflow_seed: int | None = 42

    # How wide the uncertainty is for that standalone script, as a fraction:
    # 0.15 means "give or take 15%".
    #     lifetime spread -> Triangular(base x 0.85, base, base x 1.15)
    #     share spread    -> Normal(base, base x 0.15), kept within 0-1
    # IMPORTANT: these two are PLACEHOLDERS. They are not measured uncertainties -- no
    # real ones have been supplied. Replace them once you have real ranges; everything
    # downstream works identically whatever the numbers are.
    # SAFE TO CHANGE: yes. Each must stay above 0 and below 1.
    stockflow_lifetime_spread: float = 0.15
    stockflow_share_spread: float = 0.15

    # How many draws to process at a time, to keep memory under control.
    # Pure performance setting -- it does NOT affect results in any way. Smaller means
    # less memory and slightly slower; larger means more memory and slightly faster.
    # SAFE TO CHANGE: yes, though there is rarely a reason. Lower it only if a run runs
    # out of memory.
    chunk_size: int = 20_000

    # Which time windows to report results for, written as (first year, last year),
    # both years included. Every window listed here gets a full summary: mean, median,
    # the 2.5% and 97.5% bounds, and a histogram.
    #
    # A single year is a window with that year twice, e.g. (2030, 2030). The same list
    # is used by stages 02, 03_01 and 03_02, so "2030-2040" means the identical window
    # everywhere -- which is the whole point of setting it once, here.
    #
    # These must be chosen BEFORE a run. Stage 03_02 accumulates exactly these windows
    # while it simulates, because keeping every year for every draw would not fit in
    # memory at 200,000 draws. You cannot ask for a new window afterwards without
    # re-running.
    #
    # SAFE TO CHANGE: yes -- add as many windows as you like. Each adds reporting work,
    # not simulation work, so a handful costs very little. For example:
    #     [(1975, 2070),   # the whole model horizon
    #      (2030, 2030),   # one single year
    #      (2030, 2040)]   # a decade
    # (1975, 2070) is the cumulative headline period every figure and saved
    # table is keyed on -- keep it, or those keys disappear. (2040, 2040) is
    # added for the recovery-model export, which needs single-year periods
    # (see materials.carcomposition_draws_years).
    #
    # DO NOT ADD SINGLE-YEAR ENTRIES HERE TO SERVE 04_01's DRAW EXPORT. This
    # setting is shared by stages 02, 03_01, 03_02 and 04_01, and every entry is
    # a reporting window each of them computes and writes.
    #
    # (2040, 2040) used to sit here and was removed on 2026-09-02. It was added
    # in 00af52a for one reason: 04_01's draw export could only write a year that
    # had a matching single-year period, so 2040 was put in the shared list to
    # get 2040 exported. Since 7b39946 that stage derives its own single-year
    # periods from materials.carcomposition_draws_years, so the entry bought
    # nothing and cost four stages an extra reporting window on every run.
    #
    # The same shortcut was taken again at 51x the scale earlier the same day and
    # rejected. If a year is wanted in the EXPORT, put it in
    # carcomposition_draws_years. Put a window here only when the window itself
    # is the thing being reported on.
    output_periods: list[tuple[int, int]] = field(
        default_factory=lambda: [(1975, 2070)])

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.n_draws <= 0:
            issues.append(f"monte_carlo.n_draws={self.n_draws} must be positive.")
        if self.stockflow_n_draws <= 0:
            issues.append(f"monte_carlo.stockflow_n_draws={self.stockflow_n_draws} must be positive.")
        if not (0.0 < self.stockflow_lifetime_spread < 1.0):
            issues.append(f"monte_carlo.stockflow_lifetime_spread={self.stockflow_lifetime_spread} must be in (0, 1).")
        if not (0.0 < self.stockflow_share_spread < 1.0):
            issues.append(f"monte_carlo.stockflow_share_spread={self.stockflow_share_spread} must be in (0, 1).")
        if self.chunk_size <= 0:
            issues.append(f"monte_carlo.chunk_size={self.chunk_size} must be positive.")
        if not self.output_periods:
            issues.append("monte_carlo.output_periods must have at least one (start, end) entry.")
        for start, end in self.output_periods:
            if start > end:
                issues.append(f"monte_carlo.output_periods entry ({start}, {end}) has start > end.")
        return issues

# ---------------------------------------------------------------------------
# THE CONTAINER -- every section above, in one object
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Top-level Params
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # One section per pipeline stage, plus the cross-cutting Monte Carlo
    # settings. Each is a group of related parameters, documented in its own
    # class above. Access them as `params.data_prep`, `params.stock_flow` and
    # so on; every stage script does exactly that at the top of its `main()`.
    # SAFE TO CHANGE: no -- these are the sections themselves, not settings.
    # Change values inside the sections, not this list.
    data_prep: DataPrepParams = field(default_factory=DataPrepParams)
    stock_flow: StockFlowParams = field(default_factory=StockFlowParams)
    adjusted_flows: AdjustedFlowsParams = field(default_factory=AdjustedFlowsParams)
    disaggregation: DisaggregationParams = field(default_factory=DisaggregationParams)
    materials: MaterialsParams = field(default_factory=MaterialsParams)
    monte_carlo: MonteCarloParams = field(default_factory=MonteCarloParams)

    def validate(self) -> list[str]:
        issues: list[str] = []
        issues += self.data_prep.validate()
        issues += self.stock_flow.validate()
        issues += self.adjusted_flows.validate()
        issues += self.disaggregation.validate()
        issues += self.materials.validate()
        issues += self.monte_carlo.validate()

        if self.stock_flow.model_end_year != self.data_prep.end_year_model:
            issues.append(
                f"Model horizon differs between stages: data_prep.end_year_model="
                f"{self.data_prep.end_year_model} vs stock_flow.model_end_year="
                f"{self.stock_flow.model_end_year}."
            )
        if self.disaggregation.end_year_model != self.data_prep.end_year_model:
            issues.append(
                f"Model horizon differs between stages: data_prep.end_year_model="
                f"{self.data_prep.end_year_model} vs disaggregation.end_year_model="
                f"{self.disaggregation.end_year_model}."
            )
        return issues

    def to_nested_dict(self) -> dict[str, Any]:
        """Legacy view: same shape as the old PARAMS dict, for transition callers only."""
        return {
            "01_data_prep": asdict(self.data_prep),
            "02_stock_flow": asdict(self.stock_flow),
            "03_02_adjusted_flows": asdict(self.adjusted_flows),
            "03_disaggregation": asdict(self.disaggregation),
            "04_materials": asdict(self.materials),
            "monte_carlo": asdict(self.monte_carlo),
        }