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

ELEMENT_LIST: list[str] = ["Ag", "In", "Ta", "Zn", "Dy", "Nd", "Pr", "Al", "Cu"]
ELEMENT_LIST_NO_AL_CU: list[str] = ["Ag", "In", "Ta", "Zn", "Dy", "Nd", "Pr"]

ALL_DRIVETRAINS: list[str] = [
    "BEV", "HEV", "PHEV", "Hybrid", "Liquids", "Petrol", "Diesel", "Gases", "FCEV",
]


# ---------------------------------------------------------------------------
# Stage 01 -- Data prep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataPrepParams:
    scenario: str = "npi25"
    scenario_list: tuple[str, ...] = ("b650", "npi25", "ssp2L", "ssp2M", "ssp1")

    remind_scenario_files: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "b650": ("REMIND/REMIND_generic_SSP2-EU21-PkBudg650.mif", ";"),
        "npi25": ("REMIND/REMIND_generic_SSP2-EU21-NPi2025.mif", ";"),
        "ssp2L": ("REMIND/REMIND_generic_C_SMIPv08-L-SSP2-PkPrice400-def-rem-6.mif", ";"),
        "ssp2M": ("REMIND/REMIND_generic_C_SMIPv08-M-SSP2-NPi2025-def-rem-6.mif", ";"),
        "ssp1": ("REMIND/REMIND_generic_C_SMIPv08-VLLO-SSP1-PkPrice500-def-rem-6.mif", ";"),
    })

    start_year_model: int = 1900
    end_year_model: int = 2070
    start_year_plotting: int = 2015
    end_year_plotting: int = 2070
    composition_extend_from_year: int = 2050
    accelerating_year: int = 2026  # FLAGGED (L2, still open): no cited derivation.
    threshold: float = 1e-4
    prefix: str = "Stock|Transport|Pass|Road|LDV"

    eu_countries: tuple[str, ...] = (
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
        "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
        "Malta", "Netherlands", "Poland", "Portugal", "Romania",
        "Slovakia", "Slovenia", "Spain", "Sweden",
    )

    remind_regions: tuple[str, ...] = ("DEU", "ECE", "ECS", "ENC", "ESC", "ESW", "EWN", "FRA", "UKI", "NEN")
    remind_technology: tuple[str, ...] = ("BEV", "Hybrid", "Liquids", "Gases", "FCEV")
    target_technology: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Petrol", "Diesel")
    target_class_detail: tuple[str, ...] = (
        "Large Car and SUV", "Van", "Compact Car", "Midsize Car", "Mini Car", "Subcompact Car",
    )

    attribute_list: tuple[str, ...] = ("Region", "technology")
    key_names: tuple[str, ...] = ("Region", "Drivetrain")
    element_list: tuple[str, ...] = tuple(ELEMENT_LIST)
    element_list_noAlCu: tuple[str, ...] = tuple(ELEMENT_LIST_NO_AL_CU)

    input_dir: str = "../data/raw/"
    output_dir: str = "../data/processed/"
    composition_file_name: str = "ELV_2010_2050.xlsx"
    petrol_composition_file_name: str = "ELVComponent_1990_2050_Petrol.xlsx"
    export_data_file_name: str = "usedvehicles_v1.2.xlsx"

    sheets: tuple[str, ...] = (
        "petrolCar_1980_2050", "dieselCar_1980_2050", "BEVCar_1980_2050",
        "HEVCar_1980_2050", "PHEVCar_1980_2050", "otherCar_1980_2050",
    )

    stock_interpolation_method: str = "cubic"  # resolves C3; "cubic" or "pchip"

    export_min_year: int = 2005
    export_max_year_exclusive: int = 2023

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
        if self.export_min_year >= self.export_max_year_exclusive:
            issues.append("data_prep: export_min_year must be strictly before export_max_year_exclusive.")
        return issues


# ---------------------------------------------------------------------------
# Stage 02 -- Stock flow
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeibullLifetime:
    """Vehicle survival curve parameters: Weibull(shape_k, scale_lambda)."""
    shape_k: float
    scale_lambda: float


@dataclass(frozen=True)
class LifetimeOverride:
    """Optional manual override window for a drivetrain's effective lifetime."""
    start_year: int
    end_year: int
    shape_k: float
    scale_lambda: float


@dataclass(frozen=True)
class AsymmetricSpread:
    """
    Asymmetric relative spread around a point estimate: the true value could be
    `lower` fraction SMALLER or `upper` fraction LARGER than the point estimate,
    not assumed equal. Used to build
    `Triangular(point*(1-lower), point, point*(1+upper))`.

    A plain float is still accepted everywhere `lifetime_scale_lambda_relative_
    spread` is used (meaning symmetric: `lower = upper = that float`) -- this
    type exists for when that symmetry assumption doesn't hold, e.g. "10%
    shorter-lived is plausible, but up to 20% longer-lived is also plausible"
    (`AsymmetricSpread(lower=0.1, upper=0.2)`), which a flat +/-15% would
    misrepresent either way.

    NOTE: an asymmetric Triangular's MEAN is `(low + mode + high) / 3`, not
    `point` -- it shifts toward whichever side has the wider spread. This is
    correct distribution behavior: if the belief is "more likely to run longer
    than shorter", the sampled mean SHOULD sit above `point`.
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
class StockFlowParams:
    model_end_year: int = 2070
    last_exp_data_year: int = 2022
    init_max_age: int = 50
    # [NEW, centralized] Was a hardcoded local variable in 02_stockdriven.py's main().
    # Moved here because stage 03's Monte Carlo extension needs the IDENTICAL value to
    # regenerate stage 02's per-year cohort results from the same saved scale_lambda
    # draws -- if this were hardcoded separately in two files, they could silently
    # drift out of sync.

    lifetime_by_drv: dict[str, WeibullLifetime] = field(default_factory=lambda: {
        "Hybrid": WeibullLifetime(3.0, 9.0),
        "PHEV":   WeibullLifetime(3.0, 9.0),
        "HEV":    WeibullLifetime(3.0, 9.0),
        "BEV":    WeibullLifetime(3.0, 13.0),
        "Liquids": WeibullLifetime(3.0, 13.0),
        "Petrol": WeibullLifetime(3.0, 13.0),
        "Diesel": WeibullLifetime(3.0, 13.0),
        "Gases":  WeibullLifetime(3.0, 13.0),
        "FCEV":   WeibullLifetime(3.0, 13.0),
    })

    lifetime_override_by_drv: dict[str, LifetimeOverride | None] = field(default_factory=lambda: {
        drv: None for drv in ALL_DRIVETRAINS
    })

    unknown_whereabouts_share: dict[str, float] = field(default_factory=lambda: {
        "BEV": 0.1,
        "HEV": 0.43, "PHEV": 0.43, "FCEV": 0.43, "Gases": 0.43,
        "Diesel": 0.43, "Petrol": 0.43, "Liquids": 0.43, "Hybrid": 0.43,
    })

    export_share_by_drv: dict[str, float] = field(default_factory=lambda: {
        "BEV": 0.02,
        "HEV": 0.08, "PHEV": 0.08, "Diesel": 0.08, "Petrol": 0.08,
        "Hybrid": 0.08, "Liquids": 0.08,
        "FCEV": 0.08,   # PLACEHOLDER, not a verified real value.
        "Gases": 0.08,  # PLACEHOLDER, not a verified real value.
    })

    negative_inflow_policy: str = "report_only"
    # [NEW] Controls how 02_stockdriven.py's run_cohort_survival_model() handles a year
    # where the residual-inflow formula goes negative (prescribed stock declining faster
    # than natural Weibull attrition explains -- see MATH_MODELS.md 2.3).
    #   "report_only"     -- (default, ORIGINAL behavior, unchanged) add zero that year;
    #                         modeled stock EXCEEDS the falling target from then on (a
    #                         surplus -- corrected this round, the original file's own
    #                         comment had this backwards as a "shortfall").
    #   "clip_to_target"  -- add zero inflow, but also force additional pro-rata outflow
    #                         across all surviving cohorts so modeled stock hits the
    #                         prescribed target exactly that year (removes the surplus).
    #                         Implemented and tested, not yet the default -- switching
    #                         changes real output numbers.
    # See 02_stockdriven.py's NEGATIVE_INFLOW_POLICIES for the implementation, and
    # HOW_TO_RUN_AND_VERIFY.md for how to verify either one.

    # -------------------------------------------------------------------------
    # Monte Carlo uncertainty around the point estimates above.
    # -------------------------------------------------------------------------
    # Centralized HERE, not hardcoded in any script -- changing an uncertainty range is
    # exactly the same kind of edit as changing a point estimate (edit this file,
    # re-run 00_parameters.py). Expressed as SPREADS (a relative fraction, or an
    # absolute standard deviation) rather than baked-in Distribution objects, so the
    # actual distribution is always built fresh from whatever the CURRENT point
    # estimate is at sampling time -- if you change `lifetime_by_drv["BEV"]
    # .scale_lambda`, the Monte Carlo spread around it updates automatically, with no
    # separate value to keep in sync.
    lifetime_scale_lambda_relative_spread: dict[str, float | AsymmetricSpread] = field(default_factory=lambda: {
        drv: 0.15 for drv in ALL_DRIVETRAINS
    })
    # Used to build Triangular(point*(1-lower), point, point*(1+upper)) around
    # lifetime_by_drv[drv].scale_lambda -- `lower=upper=<the float>` when a plain float
    # is given (symmetric, the default here), or genuinely different lower/upper via
    # `AsymmetricSpread(lower=..., upper=...)` for a drivetrain whose uncertainty isn't
    # symmetric, e.g.:
    #     "BEV": AsymmetricSpread(lower=0.10, upper=0.20)  # 10% shorter-lived is
    #                                                        # plausible, up to 20%
    #                                                        # longer-lived also is
    # PLACEHOLDER default (15%, symmetric) -- tune per drivetrain once real uncertainty
    # ranges (e.g. a survival-curve fit's own confidence interval) are available.

    unknown_whereabouts_share_std: dict[str, float] = field(default_factory=lambda: {
        drv: 0.05 for drv in ALL_DRIVETRAINS
    })
    # Used to build Normal(point, std, clip 0..1) around unknown_whereabouts_share[drv].

    export_share_std: dict[str, float] = field(default_factory=lambda: {
        drv: 0.02 for drv in ALL_DRIVETRAINS
    })
    # Used to build Normal(point, std, clip 0..1) around export_share_by_drv[drv].

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
            ("unknown_whereabouts_share_std", self.unknown_whereabouts_share_std),
            ("export_share_std", self.export_share_std),
        ):
            missing = lifetime_drvs - set(mapping)
            if missing:
                issues.append(f"stock_flow.{name} is missing drivetrains present in lifetime_by_drv: {sorted(missing)}.")
            for drv, spread in mapping.items():
                if spread < 0:
                    issues.append(f"stock_flow.{name}['{drv}'] = {spread} must be >= 0.")

        if self.negative_inflow_policy not in {"report_only", "clip_to_target"}:
            issues.append(
                f"stock_flow.negative_inflow_policy={self.negative_inflow_policy!r} is "
                f"not one of ['report_only', 'clip_to_target']."
            )

        for name, mapping in (
            ("export_share_by_drv", self.export_share_by_drv),
            ("unknown_whereabouts_share", self.unknown_whereabouts_share),
        ):
            missing = lifetime_drvs - set(mapping)
            if missing:
                issues.append(f"stock_flow.{name} is missing drivetrains present in lifetime_by_drv: {sorted(missing)}.")
            for drv, share in mapping.items():
                if share is not None and not (0.0 <= share <= 1.0):
                    issues.append(f"stock_flow.{name}['{drv}'] = {share} is outside [0, 1].")

        missing_override = lifetime_drvs - set(self.lifetime_override_by_drv)
        if missing_override:
            issues.append(
                f"stock_flow.lifetime_override_by_drv is missing drivetrains present in "
                f"lifetime_by_drv: {sorted(missing_override)}."
            )
        return issues


# ---------------------------------------------------------------------------
# Stage 03, part 2 -- Adjusted flows (03_02_adjustedflows.py) sensitivity scenarios
# ---------------------------------------------------------------------------
# Per explicit instruction: NO parameter for this stage may be hardcoded in the script
# itself. Previously `03_02_adjustedflows.py` had its 11 scenarios' definitions spread
# across five separate module-level constants (`stock_modifier_2027`,
# `LIFETIME_CHANGE_BY_DRV_{BAU,ICEV_SHORTER,BEV_LONGER}`, five
# `BEV_SEGMENT_SHARES_*` dicts, and two inline `losses_zero`/`losses_high` share
# overrides) -- adding a 12th scenario meant editing several of these in several
# places. `ScenarioSpec` bundles everything ONE scenario needs into ONE object, so
# adding a scenario is one new entry in `AdjustedFlowsParams.scenarios`, not edits
# scattered across the file.
@dataclass(frozen=True)
class OpenEndedLifetimeChange:
    """
    A lifetime override with NO end_year -- the convention `flowdriven_model.py`'s
    `run_flow_driven_model_with_outflow_disaggregation` actually implements (confirmed
    via source: `if t >= lifetime_change_by_drv[drv]["start_year"]`, no `end_year` ever
    read). Deliberately a SEPARATE type from `LifetimeOverride` above (which requires
    both `start_year` AND `end_year`, for stage 02's windowed
    `get_effective_lifetime_params`) -- these are two genuinely different, coexisting
    override conventions in this codebase, not a naming inconsistency for the same
    thing (see the consolidated review, finding H4).
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


@dataclass(frozen=True)
class AdjustedFlowsParams:
    """
    Everything `03_02_adjustedflows.py` needs that isn't already in `StockFlowParams`:
    the shared inflow-tweak timing window, the shared lifetime-change start year, and
    all 11 scenario definitions. See `ScenarioSpec`'s docstring for the sparse-override
    convention every scenario below relies on.
    """
    # Shared by every inflow-composition-tweaking scenario (BAU's own segment-mix
    # baseline, BEV_only, and the four BEV segment-profile scenarios): the transform
    # ramps from the baseline share at `scenario_start_year` to its target share by
    # `scenario_ramp_end_year`. Currently both years coincide (an immediate switch, no
    # ramp) -- kept as two separate parameters since nothing about the transform
    # requires them to be equal, and a future scenario may want a genuine multi-year ramp.
    scenario_start_year: int = 2026
    scenario_ramp_end_year: int = 2026

    # Shared by every scenario with a `lifetime_change_by_drv` entry (BEV_longer,
    # ICEV_shorter): the year that override takes effect. Centralized here rather than
    # repeated inside each `OpenEndedLifetimeChange` so all lifetime-change scenarios
    # stay synchronized to one edit if this ever needs to move.
    lifetime_change_start_year: int = 2027

    scenarios: dict[str, ScenarioSpec] = field(default_factory=lambda: {
        "BAU": ScenarioSpec(
            name="BAU",
            inflow_segment_shares_drivetrain="BEV",
            inflow_segment_shares_final={
                "A": 0.11599, "B": 0.07235, "C": 0.17037, "D": 0.06288, "E": 0.04190, "F": 0.02945,
                "JA": 0.00341, "JB": 0.07498, "JC": 0.25116, "JD": 0.15162, "JE": 0.02108, "JF": 0.00481,
            },
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

    def validate(self) -> list[str]:
        issues: list[str] = []
        if "BAU" not in self.scenarios:
            issues.append("adjusted_flows.scenarios must include a 'BAU' entry (used as the base inflow for scenarios with no inflow transform of their own).")
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
        return issues


# ---------------------------------------------------------------------------
# Stage 03 -- Disaggregation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DisaggregationParams:
    output_dir: str = "../data/processed/"
    start_year_model: int = 1900
    end_year_model: int = 2070

    use_synthetic_eea_fallback: bool = False
    # [NEW] Off by default -- your real EEA registrations data governs stage 03's
    # segment/drivetrain splits, and this should almost always stay False once that
    # file exists. Set to True ONLY as a temporary bridge while waiting for the real
    # `EEA_final_data.csv`: `03_01_flowdriven.py` will then auto-generate a clearly
    # labeled SYNTHETIC placeholder (see `disaggregation.py`'s
    # `generate_synthetic_eea_data`) instead of raising, so you can exercise/test the
    # rest of the pipeline in the meantime. Flip back to False (or just leave it -- see
    # the file-existence check in `03_01_flowdriven.py`) once the real file is in
    # place; no other code change is needed either way.
    synthetic_eea_seed: int = 42

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.start_year_model >= self.end_year_model:
            issues.append("disaggregation: start_year_model must be strictly before end_year_model.")
        return issues


# ---------------------------------------------------------------------------
# Stage 04 -- Materials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaterialsParams:
    segment_map: dict[str, str] = field(default_factory=lambda: {
        "A": "0101", "B": "0102", "C": "0103", "D": "0104", "E": "0105", "F": "0106",
        "JA": "0201", "JB": "0202", "JC": "0203", "JD": "0204", "JE": "0205", "JF": "0206",
    })
    drv_prefix_map: dict[str, str] = field(default_factory=lambda: {
        "PHEV": "050103", "HEV": "040101", "BEV": "030103",
        "Diesel": "020102", "Petrol": "010101",
    })
    region: str = "EUR"
    drivetrains: tuple[str, ...] = ("BEV", "HEV", "PHEV", "Diesel", "Petrol")

    composition_parameter_code: str = "m-c"
    # [NEW] Centralizes the choice `materials.py`'s `quantify_elements_from_tracker()`
    # takes as its `parameter_code` argument, per your explicit, essential requirement:
    # composition resolves down to COMPONENT and its MATERIALS, NOT further decomposed
    # into individual chemical elements.
    #   "m-c" (default) -- stops at material level (`element` column is just an alias
    #                       for the material name at `material_level_key`; component
    #                       grouping via componentKeyLevel0/1 is always available).
    #   "e-m"           -- decomposes further into individual elements (Ag, In, Ta,
    #                       Zn, Dy, Nd, Pr, Al, Cu -- see ELEMENT_LIST). NOT the
    #                       default; only switch to this if element-level detail is
    #                       genuinely needed later.
    material_level_key: str = "materialKeyLevel_highest"
    # [UPDATED] Which material-hierarchy level to report material names at, when
    # composition_parameter_code == "m-c". Changed default from "materialKeyLevel2" to
    # "materialKeyLevel_highest" to match the REAL stage-04 code's own usage (both
    # 04_01_materials.py and 04_03_tractionmotors.py use "materialKeyLevel_highest" --
    # a per-row fallback to the deepest available level, more robust than assuming
    # uniform depth across every composition row). One of "materialKeyLevel0"..
    # "materialKeyLevel4" or "materialKeyLevel_highest".

    # -----------------------------------------------------------------------
    # Stage 04, part 3 -- traction motors (04_03_tractionmotors.py)
    # -----------------------------------------------------------------------
    traction_composition_file_name: str = "20260309-Traction_motors_consolidated.xlsx"
    # [NEW] Centralizes what was hardcoded directly inside 04_03_tractionmotors.py
    # (a date-stamped filename -- same "should be centralized" pattern as M9/M23/M31).

    # -----------------------------------------------------------------------
    # Stage 04, part 4 -- batteries (04_04_batteries.py)
    # -----------------------------------------------------------------------
    battery_share_file_name: str = "BATTKey_xEV_shares_final.xlsx"
    battery_composition_file_name: str = "250318_WP3_MS23_consolidatedComposition_BATT_EV_v7_editable.xlsx"
    # [NEW] Centralizes two more hardcoded filenames, same pattern as above.

    battery_composition_parameter_code: str = "e-m"
    # ======================================================================
    # [FLAGGED -- ESSENTIAL REQUIREMENT CONFLICT, NOT RESOLVED, NEEDS YOUR INPUT]
    # ======================================================================
    # This is "e-m" -- ELEMENT level, the same code used by the excluded
    # 04_02_elements.py -- directly conflicting with your explicit, essential
    # requirement that composition resolve to component + material, NOT elements.
    # UNLIKE the main ELV_2010_2050.xlsx composition file (confirmed, via materials.py,
    # to also offer an "m-c" material-level reading), I have NOT seen the battery
    # composition workbook (`battery_composition_file_name` above) and do NOT know
    # whether it offers an equivalent material-level parameterCode value. I have
    # deliberately NOT guessed a replacement string and silently substituted it --
    # doing so risks silently reading the wrong column of a file I've never seen.
    # ACTION NEEDED: open the battery composition workbook, check its `parameterCode`
    # column's distinct values (04_04_batteries.py already prints
    # "Available parameterCode values" for the main composition file the same way --
    # a similar print could confirm this file's options), and tell me the exact string
    # that means "material/component level, not individual elements". Until then,
    # 04_04_batteries.py's battery-material-mass output remains at ELEMENT level --
    # flagged loudly both here and at the call site in that file.
    # ======================================================================

    battery_size_map: dict[str, float] = field(default_factory=lambda: {
        "A": 25.0, "B": 45.0, "C": 60.0, "D": 80.0, "E": 80.0, "F": 100.0,
        "JA": 25.0, "JB": 45.0, "JC": 60.0, "JD": 80.0, "JE": 80.0, "JF": 100.0,
    })
    # [NEW] Centralizes 04_04_batteries.py's hardcoded per-segment battery capacity
    # (kWh) assumption. Per that file's own TODOs: no year dependence (rising battery
    # capacity over time is not modeled), and D/E segments share the same value --
    # confirm both are intentional once you can check against the segment taxonomy.
    average_battery_capacity_kwh: float = 60.0

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.composition_parameter_code not in {"m-c", "e-m"}:
            issues.append(
                f"materials.composition_parameter_code={self.composition_parameter_code!r} "
                f"is not one of ['m-c', 'e-m']."
            )
        if self.battery_composition_parameter_code == "e-m":
            issues.append(
                "materials.battery_composition_parameter_code='e-m' -- element level, "
                "conflicts with your essential component+material-only requirement. "
                "Not an error (the pipeline still runs), but surfaced here as a "
                "standing reminder until you confirm the battery workbook's "
                "material-level parameterCode value."
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
        missing_battery_segments = set(self.segment_map) - set(self.battery_size_map)
        if missing_battery_segments:
            issues.append(
                f"materials.battery_size_map is missing segments present in "
                f"segment_map: {sorted(missing_battery_segments)}."
            )
        return issues


# ---------------------------------------------------------------------------
# Stage 06 -- Visualization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VisualizationParams:
    year_plot_start: int = 2015
    year_plot_end: int = 2070
    year_ratio_start: int = 2020
    element_list: tuple[str, ...] = tuple(ELEMENT_LIST)
    element_list_noAlCu: tuple[str, ...] = tuple(ELEMENT_LIST_NO_AL_CU)
    scenario: str = "npi25"

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.year_plot_start > self.year_plot_end:
            issues.append("visualization: year_plot_start must not be after year_plot_end.")
        return issues


# ---------------------------------------------------------------------------
# Monte Carlo (cross-cutting, not tied to any one stage)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonteCarloParams:
    """
    Off by default -- enabling this does not change any deterministic stage's output,
    only whether stages ALSO run an additional Monte Carlo pass (see
    `02_stockdriven.py`'s Monte Carlo block for the first concrete usage; `src/
    monte_carlo.py` for the generic sampling machinery this drives).
    """
    enabled: bool = True  # MC
    n_draws: int = 200000
    seed: int | None = 42

    # [NEW] The full stock-and-flow uncertainty analysis (`mc_stockflow_uncertainty.py`)
    # -- varies lifetime + unknown_whereabouts_share + export_share_by_drv for every
    # stage-02-level drivetrain simultaneously. Kept SEPARATE from `n_draws` above
    # (which the lightweight single-parameter demo in 02_stockdriven.py uses) so
    # enabling that quick demo never accidentally triggers a 200,000-draw run.
    stockflow_n_draws: int = 200_000
    stockflow_seed: int | None = 42
    # Placeholder relative spreads for the uncertainty distributions -- NOT derived
    # from any real uncertainty estimate yet (no such estimate has been provided).
    # scale_lambda: Triangular(base*(1-spread), base, base*(1+spread)).
    # unknown_whereabouts_share / export_share_by_drv: Normal(base, base*spread),
    # clipped to [0, 1]. Override these once you have real uncertainty ranges --
    # everything downstream (sampling, propagation, sensitivity analysis) works
    # identically regardless of what the spread actually is.
    stockflow_lifetime_spread: float = 0.15
    stockflow_share_spread: float = 0.15

    # [NEW] Memory-chunking for vectorized Monte Carlo engines (e.g.
    # `cohort_flow_mc.py`): draws are processed in batches of this size so peak memory
    # is bounded by `chunk_size x n_cohorts`, not `n_draws x n_cohorts`. A generic,
    # cross-stage performance knob (not a model assumption), centralized here rather
    # than hardcoded as a function default in any one engine.
    chunk_size: int = 20_000

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
        return issues


# ---------------------------------------------------------------------------
# Top-level Params
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    data_prep: DataPrepParams = field(default_factory=DataPrepParams)
    stock_flow: StockFlowParams = field(default_factory=StockFlowParams)
    adjusted_flows: AdjustedFlowsParams = field(default_factory=AdjustedFlowsParams)
    disaggregation: DisaggregationParams = field(default_factory=DisaggregationParams)
    materials: MaterialsParams = field(default_factory=MaterialsParams)
    visualization: VisualizationParams = field(default_factory=VisualizationParams)
    monte_carlo: MonteCarloParams = field(default_factory=MonteCarloParams)

    def validate(self) -> list[str]:
        issues: list[str] = []
        issues += self.data_prep.validate()
        issues += self.stock_flow.validate()
        issues += self.adjusted_flows.validate()
        issues += self.disaggregation.validate()
        issues += self.materials.validate()
        issues += self.visualization.validate()
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
            "06_visualization": asdict(self.visualization),
            "monte_carlo": asdict(self.monte_carlo),
        }
