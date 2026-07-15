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
    # [NEW] Same 27 countries as `eu_countries` above, but as ISO2 codes -- SEPARATE
    # field, not a replacement. `eu_countries` (full names) is also used directly in
    # `clean_export_data` (data_prep.py), filtering against `Exp_Name`/`Imp_Name`
    # columns that are canonicalized to full names via NAME_MAP -- changing its
    # representation would have broken that call site. This field exists ONLY for
    # `prepare_eea_share_tables`'s `country_scope` (disaggregation.py), which filters
    # EEA_final_data.csv's own "Country" column -- CONFIRMED via a real diagnostic
    # run (2026-07-11) to contain ISO2 codes, not full names. Before this field
    # existed, `03_01_flowdriven.py` passed `tuple(p01.eu_countries) + ("NO", "IS")`
    # directly as `country_scope` -- since full country names never match ISO2 codes,
    # that silently matched almost nothing (0.5% of total registration volume in the
    # real file -- effectively just "NO"/"IS", which happen to already be 2-letter
    # codes by coincidence), meaning segment shares, the Diesel/Petrol split, AND the
    # HEV/PHEV split were all being computed from Norway+Iceland alone instead of the
    # EU-27. Order matches `eu_countries` above 1:1, for easy cross-checking.
    eu_countries_iso2: tuple[str, ...] = (
        "AT", "BE", "BG", "HR", "CY", "CZ",
        "DK", "EE", "FI", "FR", "DE", "GR",
        "HU", "IE", "IT", "LV", "LT", "LU",
        "MT", "NL", "PL", "PT", "RO",
        "SK", "SI", "ES", "SE",
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
        "Hybrid": WeibullLifetime(3.0, 13.0),
        "PHEV":   WeibullLifetime(3.0, 9.0),
        "HEV":    WeibullLifetime(3.0, 13.0),
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

    # [NEW] Previously only ever computed IMPLICITLY as
    # `1 - unknown_whereabouts_share - export_share_by_drv` inside
    # `disaggregation.compute_collected_export_unknown_shares` -- now an explicit
    # point estimate in its own right, so it can carry its own Monte Carlo spread
    # (`collected_share_relative_spread` below) instead of silently absorbing
    # whatever the other two happen to sample. Values below are exactly
    # `1 - unknown_whereabouts_share[drv] - export_share_by_drv[drv]` for every
    # drivetrain -- numerically identical to the old implicit remainder, so the
    # DETERMINISTIC (non-MC) path is byte-identical to before this change. What
    # changes is only how Monte Carlo treats the three shares (see
    # `*_share_relative_spread` below): none of the three is a privileged
    # "remainder" that absorbs the other two's sampling noise -- collected, export,
    # and unknown_whereabouts are all measured/estimated with their OWN
    # uncertainty, and are sampled independently then normalized to sum to 1.
    collected_share_by_drv: dict[str, float] = field(default_factory=lambda: {
        "BEV": 0.88,
        "HEV": 0.49, "PHEV": 0.49, "Diesel": 0.49, "Petrol": 0.49,
        "Hybrid": 0.49, "Liquids": 0.49,
        "FCEV": 0.49,   # PLACEHOLDER, matches export_share_by_drv's own placeholder note.
        "Gases": 0.49,  # PLACEHOLDER, matches export_share_by_drv's own placeholder note.
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

    hard_zero_inflow_from_year_by_drv: dict[str, int] = field(default_factory=lambda: {
        "Liquids": 2050,
        "Hybrid": 2050,
    })
    # [NEW] A hard, unconditional policy override -- NOT a mathematical/statistical
    # correction of the residual-inflow formula's output. For a drivetrain listed here,
    # from the given year onward, `run_cohort_survival_model`/`run_cohort_survival_
    # monte_carlo` force `inflow_applied = 0` regardless of what
    # `inflow(t) = target(t) - remaining_total(t)` computes -- no new registrations are
    # added, the surviving cohort simply decays via ordinary Weibull attrition from
    # that year on.
    #
    # "Hybrid" added 2026-07-11, same year (2050), per direct visual confirmation
    # against the real 02_flows_by_drivetrain_check.png chart -- the same
    # near-zero-then-small-bump shape identified and numerically traced for "Liquids"
    # is also visible on Hybrid's own inflow line. UNLIKE Liquids, this one has NOT
    # been separately traced line-by-line against diag_df/target_stock numbers (no
    # diagnostic run was requested for Hybrid) -- applied directly on your
    # confirmation that the same real-world policy applies. Re-run
    # diagnose_02_liquids_2050_bump.py with DRIVETRAIN="Hybrid" if you want the same
    # numeric trace as Liquids got before trusting this by eye alone.
    #
    # WHY THIS EXISTS: verified directly against real data (2026-07-11 diagnostic run)
    # that for "Liquids", the residual formula manufactures a genuine, non-trivial
    # POSITIVE "phantom" inflow from ~2047 onward (peaking ~0.45M/year around 2054),
    # purely because natural attrition on the aging survivor fleet removes stock FASTER
    # than REMIND's own (smooth, monotonically-declining, NOT an interpolation
    # artifact -- separately confirmed) target is falling. The model has no way to let
    # modeled stock undershoot target, so it invents new-vehicle inflow to hold the
    # target exactly -- with zero basis in any actual registration. Per an actual
    # legal ICE-sales ban (a real, documented external fact, not a data-fitting
    # assumption), real inflow should be a hard 0 from the ban year on, and modeled
    # stock should be ALLOWED to fall below REMIND's target from that point -- REMIND's
    # target simply stops being achievable/binding for this drivetrain past the ban.
    #
    # SCOPE, per explicit decision: applied here at stage 02, on the aggregate
    # "Liquids" category, BEFORE the Diesel/Petrol split happens in 03_01_flowdriven.py
    # -- a single point of truth. 03_01 and 03_02 read stage 02's `flows_df["inflow"]`
    # and only DISAGGREGATE it (never recompute it) for years from the base year
    # onward, so a 0 here becomes a genuine 0 for both Diesel and Petrol downstream,
    # automatically, without any override logic duplicated in those files.
    #
    # NOTHING IS HIDDEN: `run_cohort_survival_model`'s diag_df gains a new
    # "inflow_pre_hard_zero_override" column holding what the raw residual would have
    # been absent this override, for every year -- so the "phantom demand" REMIND's
    # target implies stays fully inspectable, it's just no longer treated as real.
    #
    # Empty dict (`{}`) for a drivetrain not listed here means no override -- byte-
    # identical behavior to before this field existed (verified via regression test in
    # stockflow_model.py's own test suite).

    hard_zero_inflow_until_year_by_drv: dict[str, int] = field(default_factory=lambda: {
        "BEV": 2011,
    })
    # [NEW, step 3 of the agreed plan] The mirror image of
    # `hard_zero_inflow_from_year_by_drv`: for a drivetrain listed here, for every
    # year BEFORE the given year, `run_cohort_survival_model`/`run_cohort_survival_
    # monte_carlo` force `inflow_applied = 0` -- a drivetrain that had not been
    # introduced yet cannot have had real registrations, no matter what
    # `inflow(t) = target(t) - remaining_total(t)` computes from REMIND's own
    # target_stock trajectory.
    #
    # "BEV": 2011 -- first BEV sold in Europe in 2011, per direct user confirmation.
    # (Also confirmed by the user, but NOT YET WIRED as of this dict: "PHEV first
    # sold in 2012", "HEV before 2000 is 0" -- both apply to the HEV/PHEV SPLIT within
    # the "Hybrid" aggregate, which only exists from 03_01_flowdriven.py onward
    # (stage 02 only tracks "Hybrid" as a whole, not HEV/PHEV separately) -- see
    # `disaggregation.py`'s `introduction_year_by_drv` / `build_hev_phev_split` (step
    # 1 of this plan) for where those two are actually enforced; they do NOT belong
    # in this stage-02-only dict, since stage 02 has no "HEV"/"PHEV" key to begin with.
    #
    # SCOPE / propagation: same single-point-of-truth mechanism as
    # `hard_zero_inflow_from_year_by_drv` -- applied here at stage 02, before
    # 03_01_flowdriven.py/03_02_adjustedflows.py disaggregate stage 02's own
    # `flows_df["inflow"]` (they never recompute it), so a 0 here becomes a genuine 0
    # downstream automatically. `diag_df["inflow_pre_hard_zero_override"]` preserves
    # what the raw residual would have been for every year, whether or not this
    # override is active for that drivetrain/year -- nothing is hidden.
    #
    # Empty dict (`{}`) for a drivetrain not listed here means no override -- byte-
    # identical behavior to before this field existed (verified via regression test in
    # stockflow_model.py's own test suite).

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

    # [FIXED, replaces unknown_whereabouts_share_std/export_share_std below] Was:
    # two independent Normal(point, std, clip 0..1) spreads, with `collected_share`
    # computed as whatever's left over (`1 - unknown - export`) -- meaning
    # `collected_share` had NO uncertainty of its own, and silently absorbed both
    # other shares' sampling noise. That was backwards: `collected` and `export`
    # are the two MEASURED quantities (collection statistics, trade statistics),
    # each with real uncertainty; `unknown_whereabouts` is itself an ESTIMATE, not
    # a clean residual computed from a known total (total outflow itself is a
    # model output of the lifetime survival curve, not independently measured
    # either). None of the three has a privileged "remainder" status -- all three
    # now get their OWN Triangular spread (same `float | AsymmetricSpread`
    # convention as `lifetime_scale_lambda_relative_spread` above), sampled
    # independently, then NORMALIZED per draw so they sum to exactly 1 (see
    # `disaggregation.compute_collected_export_unknown_shares`). PLACEHOLDER
    # magnitudes below (15%, symmetric) -- tune per drivetrain once real
    # uncertainty ranges (e.g. from the underlying collection/trade statistics'
    # own confidence intervals) are available.
    #
    # NOT MODELED (a known, flagged simplification, not silently ignored): these
    # three shares were historically part of how the LIFETIME parameters
    # themselves were back-calculated, so a real correlation between
    # `lifetime_scale_lambda_relative_spread` draws and these three shares'
    # draws likely exists. Sampled independently here (different spawn_key,
    # same as before this change) for lack of the historical calibration data
    # needed to model that correlation properly.
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
        "FCEV":    AsymmetricSpread(lower=0.30, upper=0.50),  # PLACEHOLDER, not a verified real value.
        "Gases":   AsymmetricSpread(lower=0.30, upper=0.50),  # PLACEHOLDER, not a verified real value.
    })
    # collected_share_relative_spread / export_share_relative_spread: used to build
    # Triangular(point*(1-lower), point, point*(1+upper)) around their own point
    # estimate (collected_share_by_drv[drv] / export_share_by_drv[drv]) -- `lower=
    # upper=<float>` for symmetric (the default here), or `AsymmetricSpread(lower=...,
    # upper=...)` per drivetrain, exactly like the lifetime spread above.
    #
    # unknown_whereabouts_share_relative_spread: [CHANGED MEANING] no longer built
    # around the flat point estimate `unknown_whereabouts_share[drv]` directly --
    # this drivetrain's `scale_lambda` draw (from stage 02) first shifts the
    # CENTER via `unknown_share_lifetime_coupling_k` below, and THIS spread is the
    # RESIDUAL Triangular noise layered around that shifted center (whatever
    # uncertainty in "unknown" isn't already explained by that draw's own lifetime
    # outcome). See `03_01_flowdriven.py`'s Monte Carlo block for the exact
    # formula. Since part of unknown_share's total uncertainty is now explained by
    # the lifetime coupling rather than sampled directly here, this residual
    # spread being similar in magnitude to collected/export's own spread does NOT
    # mean unknown's TOTAL uncertainty is similar to theirs -- the coupling term
    # adds substantially more spread on top for any drivetrain with a wide
    # lifetime spread.

    # [NEW] Couples this drivetrain's `scale_lambda` Monte Carlo draw (stage 02) to
    # `unknown_whereabouts_share`'s central tendency for that SAME draw, per your
    # own reasoning: a shorter-than-point-estimate lifetime draw implies MORE total
    # outflow than collected+export alone explain, so the inferred "unknown" gap
    # should be LARGER for that draw, not independent of it. Formula (see
    # `03_01_flowdriven.py`):
    #     rel_dev = (scale_lambda_draw - scale_lambda_point) / scale_lambda_point
    #     unknown_share_center = unknown_whereabouts_share[drv] * (1 - k * rel_dev)
    # `rel_dev < 0` (shorter-than-point lifetime draw) with `k > 0` makes
    # `unknown_share_center > unknown_whereabouts_share[drv]` -- larger unknown
    # share for a shorter-lifetime draw, matching the direction of your reasoning.
    # `k = 0` recovers the OLD (pre-coupling) behavior exactly -- unknown_share
    # centered on its own flat point estimate, `scale_lambda` fully independent.
    # PLACEHOLDER (k=1.0 for every drivetrain, i.e. a 10% shorter lifetime draw
    # shifts the center ~10% higher, before the residual Triangular noise and
    # final normalization) -- this is a genuine modeling assumption with no
    # first-principles derivation available; tune per drivetrain once you have a
    # real basis for the coupling strength (e.g. from however "unknown" was
    # historically inferred when the lifetime parameters were originally
    # calibrated).
    unknown_share_lifetime_coupling_k: dict[str, float] = field(default_factory=lambda: {
        # Same k=1.0 baseline as before for the mature ICE/Hybrid/BEV group.
        # FCEV/Gases get a STRONGER coupling (1.5) -- with so little else known
        # about these drivetrains (see their placeholder point estimates and
        # widest-of-all spreads above), leaning more heavily on "shorter lifetime
        # implies more unexplained outflow" is more defensible than pretending
        # scale_lambda and unknown_share are independent for them.
        "BEV":     1.0,
        "HEV":     1.0, "PHEV": 1.0, "Hybrid": 1.0,
        "Diesel":  1.0, "Petrol": 1.0, "Liquids": 1.0,
        "FCEV":    1.5,  # PLACEHOLDER, not a verified real value.
        "Gases":   1.5,  # PLACEHOLDER, not a verified real value.
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
        ):
            missing = lifetime_drvs - set(mapping)
            if missing:
                issues.append(f"stock_flow.{name} is missing drivetrains present in lifetime_by_drv: {sorted(missing)}.")
            for drv, spread in mapping.items():
                if isinstance(spread, AsymmetricSpread):
                    issues += spread.validate(field_name=f"stock_flow.{name}['{drv}']")
                elif spread < 0:
                    issues.append(f"stock_flow.{name}['{drv}'] = {spread} must be >= 0.")

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

        for drv, year in self.hard_zero_inflow_from_year_by_drv.items():
            if drv not in lifetime_drvs:
                issues.append(
                    f"stock_flow.hard_zero_inflow_from_year_by_drv has key {drv!r}, which "
                    f"is not a drivetrain present in lifetime_by_drv ({sorted(lifetime_drvs)})."
                )
            if not isinstance(year, int) or year < 1900:
                issues.append(
                    f"stock_flow.hard_zero_inflow_from_year_by_drv[{drv!r}] = {year!r} must "
                    f"be a plausible calendar year (int >= 1900)."
                )

        for drv, year in self.hard_zero_inflow_until_year_by_drv.items():
            if drv not in lifetime_drvs:
                issues.append(
                    f"stock_flow.hard_zero_inflow_until_year_by_drv has key {drv!r}, which "
                    f"is not a drivetrain present in lifetime_by_drv ({sorted(lifetime_drvs)})."
                )
            if not isinstance(year, int) or year < 1900:
                issues.append(
                    f"stock_flow.hard_zero_inflow_until_year_by_drv[{drv!r}] = {year!r} must "
                    f"be a plausible calendar year (int >= 1900)."
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

    # [NEW, was a hardcoded literal in flowdriven_model.py until this fix] The year
    # `stock_modifier` (any scenario's flat inflow multiplier, e.g. `stock_lower`'s 0.8)
    # starts applying. Independent from `lifetime_change_start_year` above -- they
    # happen to share the same default value today, but a scenario changing "how many
    # vehicles enter the fleet" and one changing "how long vehicles last" are
    # conceptually unrelated knobs that don't need to move together.
    stock_modifier_start_year: int = 2027

    # [NEW, was a hardcoded literal (`base_year=2005`) at every `run_adjusted_scenario`
    # call site in `03_02_adjustedflows.py`'s `main()` until this fix] The stock-flow
    # cohort base year: `run_adjusted_scenario` rebuilds each scenario's starting
    # cohort stock from stage 02's matrix AT this year (see `fdm.build_stock_by_
    # segment_at_base_year`) -- not a fresh 1975 backcast. Matches stage 02's own base
    # year; if that ever changes, this must move with it (not verified against
    # `StockFlowParams` automatically, since stage 02 doesn't currently expose its own
    # base year as a named field either -- flagged here rather than silently assumed
    # in sync).
    base_year: int = 2005

    # [NEW] Which scenario(s) `03_02_adjustedflows.py` actually simulates -- for
    # focused research runs instead of always paying for the full 11-scenario x
    # Monte-Carlo sweep. `None` means "run every scenario in `scenarios` below" (the
    # full sweep); a tuple of names restricts the run to just those.
    #   Default is ("BAU",) -- a fast, minimal run out of the box. Widen it
    #   deliberately, e.g. scenarios_to_run=("BAU", "stock_lower", "losses_high"), or
    #   set it to None to reproduce the full 11-scenario sweep.
    # Names are validated against `scenarios` below in `validate()`, so a typo'd
    # scenario name fails fast at parameter-build time (00_parameters.py) rather than
    # partway through a multi-hour run. NOTE: regardless of this selection, BAU's own
    # resolved INFLOW is always computed by `03_02_adjustedflows.py` (every scenario
    # without its own inflow transform reuses it) -- but BAU is only actually
    # SIMULATED (and its tracker/Monte Carlo output produced) if "BAU" is itself
    # included here. Use `active_scenario_names()` below to resolve this field --
    # don't re-implement the None-vs-tuple logic at the call site.
    scenarios_to_run: tuple[str, ...] | None = ("BAU",)

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

    introduction_year_by_drv: dict[str, int] = field(default_factory=lambda: {
        "BEV": 2011, "HEV": 2000, "PHEV": 2012,
    })
    # [NEW] Real-world first year each drivetrain was ever sold -- a documented
    # external fact, not a data-fitting assumption. Confirmed with you directly
    # (2026-07-11). Used in TWO places, both needing the SAME single source of
    # truth:
    #   1. `build_hev_phev_split` (disaggregation.py): for a (drivetrain, year) with
    #      no real EEA registration data, a year BEFORE that drivetrain's
    #      introduction year is treated as exactly 0 (known fact -- it didn't exist
    #      yet), rather than backfilled from whenever real data happens to start.
    #      Years AT/AFTER the introduction year but still missing data (a genuine
    #      data-coverage gap, not an existence question) still fall back to the old
    #      nearest-available-real-value behavior -- there's no better information
    #      for that case.
    #   2. The synthetic 1975-2004 pre-base-year backcast (03_01_flowdriven.py):
    #      masks any backcast row before a drivetrain's introduction year to 0.
    # "BEV" is not itself split by build_hev_phev_split (that function only handles
    # HEV/PHEV) -- its entry here is for the stage-02 "before year" mechanism and
    # the backcast masking (item 2 above), not this function.

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.start_year_model >= self.end_year_model:
            issues.append("disaggregation: start_year_model must be strictly before end_year_model.")
        for drv, year in self.introduction_year_by_drv.items():
            if not isinstance(year, int) or year < 1900:
                issues.append(
                    f"disaggregation.introduction_year_by_drv[{drv!r}] = {year!r} must be "
                    f"a plausible calendar year (int >= 1900)."
                )
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

    # -----------------------------------------------------------------------
    # Stage 04, part 1 REWRITE -- component-material (C-M) Monte Carlo
    # composition (04_01_materials.py). Replaces the old bulk ELV_2010_2050.xlsx
    # composition table's role for the materials stage with a component x
    # material x drivetrain x segment x year MASS DISTRIBUTION [kg/vehicle]
    # (mean/median/mode/std/P025/P975, plus a full 50-bin histogram for genuine
    # Monte Carlo bootstrap sampling) -- not a single point estimate.
    # -----------------------------------------------------------------------
    composition_summary_file_name: str = "36_MonteCarlo_Summary.xlsx"
    # [NEW] The ~10MB summary workbook, living in `data/raw/composition/`: one tab
    # per drivetrain (componentCarPetrol/Diesel/BEV/HEV/PHEV/Other), columns
    # `components, segment, year, variable, mean, median, mode, std, P025, P975`
    # -- one row per (component, material) mass distribution for a given
    # segment/year/drivetrain. `componentCarOther` is never read (out of scope,
    # matches `drivetrains` below); `segment == "standard"` rows are dropped (a
    # generic/non-segment-specific summary row, not used downstream). Feeds the
    # SCALAR (deterministic) materials path's point estimate -- see
    # `composition_scalar_statistic`.

    composition_scalar_statistic: str = "mean"
    # [NEW] Which column of `composition_summary_file_name` the scalar materials
    # path multiplies vehicle counts by. One of "mean", "median", "mode"
    # (validated). Defaults to "mean" -- change if the scalar path should track
    # the median or mode instead.

    histogram_file_name: str = "37_MonteCarlo_Histograms.xlsx"
    # [NEW] The large (~260MB today, expected to grow toward ~5x once resolution
    # moves from every-5-years to annual) histogram workbook, also in
    # `data/raw/composition/`. Columns: `components, segment, year, variable,
    # bin_lower, bin_upper, count, frequency` -- 50 contiguous, equal-width bins
    # per (components, segment, year, variable) group within a sheet, `count`
    # summing to the number of underlying draws (200,000 in the real data seen
    # so far), `frequency` summing to 1.0. This is what the Monte Carlo path
    # bootstraps composition draws from (see `bootstrap_composition_draws` in
    # `04_01_materials.py`).

    histogram_sheet_names_by_drv: dict[str, list[str]] = field(default_factory=lambda: {
        "Petrol": ["componentCarPetrol"],
        "Diesel": ["componentCarDiesel"],
        "BEV": ["componentCarBEV"],
        "HEV": ["componentCarHEV"],
        "PHEV": ["componentCarPHEV"],
    })
    # [FIXED, this round -- was a real bug, not a stable design] Petrol/Diesel used
    # to list EXACT numbered sheet names here (e.g. "componentCarPetrol_1",
    # "componentCarPetrol_2") because, at the time this was first written, the
    # histogram workbook had just started splitting each drivetrain across multiple
    # sheets (Excel's per-sheet row limit) and nothing auto-discovered siblings yet
    # -- this was a manual stopgap to get past the original crash, using whichever
    # sheets existed at that moment. It silently went stale: the workbook later grew
    # to 5 sheets per drivetrain (still growing as data densifies toward annual
    # resolution), but this list was never updated, so 3 of Petrol's/Diesel's 5
    # sheets' worth of REAL data (including all of JA-JF and "standard" for both)
    # was silently never read -- found via the user manually opening the workbook
    # and spotting real data on a sheet the pipeline was ignoring.
    #
    # 04_01_carcomposition.py's `_stream_histogram_sheets` now auto-discovers the
    # FULL sibling family for any requested name, whether that name is a bare prefix
    # (as used here now, matching BEV/HEV/PHEV) or a specific numbered sheet -- so
    # this field no longer needs to track an exact, fragile sheet count at all. Bare
    # prefixes are used for every drivetrain now specifically so this field can never
    # again silently under-specify the real sheet count the way it just did.
    #
    # TODO(cleanup): once this auto-discovery behavior has been running in
    # production for a while and is trusted, consider whether this field is worth
    # keeping as a dict at all, vs. just deriving "componentCar{drivetrain}" as a
    # bare prefix directly from `drivetrains` below with no separate mapping to
    # maintain. Left as an explicit dict for now (not collapsed automatically) since
    # removing it is a design simplification to make deliberately, not a fix to
    # rush through here.

    material_mc_time_resolution: str = "period"
    # [NEW] Controls what vehicle-count granularity the Monte Carlo materials
    # path combines composition draws against:
    #   "period" (default) -- mc["by_group"][...]["periods"][(start,end)]
    #                          ["cumulative_collected"], the SAME period-level
    #                          draws stage 03_02 already computes. No extra MC
    #                          cost beyond what 03_02 already pays.
    #   "annual"           -- one set of material-mass draws PER YEAR, not just
    #                          per requested period. Requires per-year vehicle-
    #                          count draws upstream in 03_02 (collect_per_year=
    #                          True, or many single-year output_periods entries)
    #                          -- materially more expensive; only turn on once
    #                          year-by-year material mass is actually needed.
    #   "both"             -- compute both of the above.
    # One of "period", "annual", "both" (validated).

    materials_mc_n_draws: int = 200_000
    # [NEW] Number of Monte Carlo draws for the MATERIALS-stage combination (vehicle-
    # count bootstrap x composition bootstrap). Deliberately INDEPENDENT of
    # `monte_carlo.n_draws` (stage 03_02's own resolution, 200,000 by default) -- since
    # both sides of the materials combination are bootstrapped fresh from SAVED
    # histograms/summaries (see the architecture note in `04_01_materials.py`; raw
    # per-draw arrays from stage 03_02 are never persisted to disk), this is a free
    # choice tuned purely for the materials stage's own speed/precision tradeoff.
    # Confirmed with the user: start at 20,000 (verified on the real project: full
    # BEV drivetrain, 936 (component, material, segment) groups, bootstrapped + combined
    # in under a second at this size) -- must stay fully vectorized as this increases
    # (it already is: one `rng.random(n_draws)` call per group, no per-draw Python loop).

    materials_mc_seed: int | None = 42
    # [NEW] Seed for the materials-stage bootstrap RNG stream. Kept SEPARATE from
    # `monte_carlo.seed` (stage 03_02) and `monte_carlo.stockflow_seed` (stage 02) --
    # same "one seed per stage" convention already used elsewhere in this file --
    # so materials-stage randomness never accidentally correlates with (or depends on
    # the exact draw sequence of) an earlier stage's MC run.

    def validate(self) -> list[str]:
        issues: list[str] = []
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

    # [NEW, moved here from being 03_02-only] Which (start_year, end_year) INCLUSIVE
    # year-ranges to report Monte Carlo results for -- cumulative flows, cumulative
    # inflow, and stock (end-of-period snapshot, per-year series, and sum-over-period
    # "stock-years"), broken down as finely as each stage's own model supports (stage
    # 02/03_01: per drivetrain only, no segment concept yet; stage 03_02: per
    # drivetrain AND per segment). A single year is just a range where start == end
    # (e.g. `(2030, 2030)`). SHARED across stages 02, 03_01, and 03_02 -- specifying it
    # once here keeps all three consistent by construction, rather than three
    # independently-edited copies of the same list. Specified UP FRONT (not queryable
    # after a run completes): stage 03_02's vectorized engine (`cohort_flow_mc.py`)
    # accumulates exactly these windows during its one simulation pass to keep memory
    # bounded at 200,000 draws; stages 02/03_01 already track full per-year (not
    # per-cohort) arrays cheaply and sum over these windows post-hoc
    # (`monte_carlo.sum_by_period`) -- same requested periods either way, different
    # (cheaper) implementation because those two stages' per-year arrays were already
    # affordable to keep in full. Every entry gets a full distribution summary
    # (mean/median/mode/std/P2.5/P97.5 + 50-bin histogram, via `monte_carlo.
    # summarize_distribution`). Defaults to just the whole horizon (matching the
    # pre-this-feature behavior) -- add entries for finer-grained queries, e.g.:
    #     output_periods: list[tuple[int, int]] = field(default_factory=lambda: [
    #         (1975, 2070),   # whole horizon (kept for 03_02's flat top-level aliases)
    #         (2030, 2030),   # single year
    #         (2030, 2040),   # a decade
    #     ])
    output_periods: list[tuple[int, int]] = field(default_factory=lambda: [(1975, 2070)])

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