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
class StockFlowParams:
    model_end_year: int = 2070
    last_exp_data_year: int = 2022

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

    def validate(self) -> list[str]:
        issues: list[str] = []
        lifetime_drvs = set(self.lifetime_by_drv)

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
    material_level_key: str = "materialKeyLevel2"
    # [NEW] Which material-hierarchy level to report material names at, when
    # composition_parameter_code == "m-c". One of "materialKeyLevel0".."materialKeyLevel4"
    # or "materialKeyLevel_highest". STILL OPEN: the right default depends on your real
    # composition file's hierarchy (which level is "the material name" you want to see) --
    # "materialKeyLevel2" matches materials.py's own prior default, not independently
    # confirmed against real data yet. Adjust once you can see actual composition rows.

    def validate(self) -> list[str]:
        issues: list[str] = []
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
    enabled: bool = False
    n_draws: int = 200
    seed: int | None = 42

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.n_draws <= 0:
            issues.append(f"monte_carlo.n_draws={self.n_draws} must be positive.")
        return issues


# ---------------------------------------------------------------------------
# Top-level Params
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    data_prep: DataPrepParams = field(default_factory=DataPrepParams)
    stock_flow: StockFlowParams = field(default_factory=StockFlowParams)
    disaggregation: DisaggregationParams = field(default_factory=DisaggregationParams)
    materials: MaterialsParams = field(default_factory=MaterialsParams)
    visualization: VisualizationParams = field(default_factory=VisualizationParams)
    monte_carlo: MonteCarloParams = field(default_factory=MonteCarloParams)

    def validate(self) -> list[str]:
        issues: list[str] = []
        issues += self.data_prep.validate()
        issues += self.stock_flow.validate()
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
            "03_disaggregation": asdict(self.disaggregation),
            "04_materials": asdict(self.materials),
            "06_visualization": asdict(self.visualization),
            "monte_carlo": asdict(self.monte_carlo),
        }
