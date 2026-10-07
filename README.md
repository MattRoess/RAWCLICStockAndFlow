# RAWCLIC stock-and-flow model

How many cars Europe has, buys and scraps, by drivetrain and size segment, and from
that the materials those cars carry in, give back and hand to recycling. Every
result is a distribution: 200,000 Monte Carlo draws, kept as per-draw arrays and
not reduced to percentiles until the end. The fleet comes from a REMIND scenario;
what a car is made of comes from the sibling composition projects; the arrays
written at the end are what
[RAWCLICRecoveryModel](https://github.com/MattRoess/RAWCLICRecoveryModel) reads.

## Running it

```bash
.venv/bin/python code/00_parameters.py          # always first
.venv/bin/python code/01_data_prep.py
.venv/bin/python code/02_stockdriven.py
.venv/bin/python code/03_01_flowdriven.py
.venv/bin/python code/03_02_adjustedflows.py
.venv/bin/python code/04_01_carcomposition.py
.venv/bin/python code/04_02_BEVelectronics.py
.venv/bin/python code/04_03_tractionmotors.py
.venv/bin/python code/04_04_batteries.py
```

Each stage reads what the previous ones wrote, so the order matters. You can stop
anywhere; a later stage says plainly that it has nothing to read and does not
produce wrong numbers. At the default 200,000 draws the whole chain takes hours and
writes tens of gigabytes, and the cost of every stage is in
[documentation/RUNNING.md](documentation/RUNNING.md). To see that a change runs at
all, set `monte_carlo.n_draws` to about 1,000 first.

| | |
|---|---|
| `code/00_parameters.py` | Turn `src/params_schema.py` into the file every stage reads, and validate it. |
| `code/01_data_prep.py` | Read the REMIND scenario files and the used-vehicle export data, trim them to Europe, fill the years REMIND does not supply. |
| `code/02_stockdriven.py` | The fleet: how many cars exist, enter and leave, per drivetrain, from a Weibull cohort-survival model. Samples the stock and lifetime uncertainty. |
| `code/03_01_flowdriven.py` | Split the fleet into twelve size segments (A–F, JA–JF) and the outflow into collected, exported and untraced, then run it again flow-driven. |
| `code/03_02_adjustedflows.py` | Re-simulate the fleet under each scenario. Eleven are declared; `adjusted_flows.scenarios_to_run` picks which run, BAU by default. Writes the per-draw BEV flows the stage-04 stages read. |
| `code/04_01_carcomposition.py` | Turn vehicles into material mass: steel, aluminium, copper, battery chemistry. |
| `code/04_02_BEVelectronics.py` | BEV electronics (wiring, sensors, circuit boards, motors): vehicles times grams per vehicle, draw by draw. |
| `code/04_03_tractionmotors.py` | Traction-motor material and element flows. |
| `code/04_04_batteries.py` | BEV battery element and component flows, kept apart by chemistry, under three chemistry scenarios. Draws the figures and writes the export for the recovery model. |

`code/generate_parameter_reference.py` is not a stage: it rewrites
`documentation/PARAMETER_REFERENCE.md` from the settings. `code/test_04_04_figures.py`
redraws 04_04's figures from the draws already on disk and is a bench tool despite
its name. `code/proof_0403_vehicle_draws.py` writes nothing.

## Checks

Three scripts, each run directly. They exist because defects here once went
unnoticed while every stage ran and every figure rendered; each asks one
question about what a stage actually leaves behind or computes.

```bash
.venv/bin/python code/test_stage04_03_export.py   # needs only the code: 9 checks, a second
.venv/bin/python code/test_stage04_02_export.py   # needs RAWCLICVehicleElectronics beside this repository, its draws written
.venv/bin/python code/test_stage03_inflow.py      # needs the saved output of stages 01 and 03_02; about a second
```

`code/hooks/pre-commit` runs the last two before a commit that touches the files
they protect; install it with the line at the top of that file.

## Where the numbers come from

**The data are not in this repository.** `data/` is ignored except for one file, so a
fresh clone is about 5 MB of code and documentation: `00_parameters.py` runs, and
nothing after it can until the inputs are in place.

| in `data/raw/` | what it is | |
|---|---|---|
| `REMIND/*.mif` | five REMIND scenario outputs, the fleet projections; `data_prep.scenario` picks one (default `npi25`) | not included |
| `usedvehicles_v1.2.xlsx` | EU used-vehicle export data | not included |
| `composition/*.xlsx` | the component–material composition summaries and histograms 04_01 reads | not included |
| `EEA_final_data.csv` | EEA registrations: segment and drivetrain shares, and the check that 03_02's inflow matches the record | **tracked**, 0.67 MB, the one exception: nothing in the pipeline can regenerate it |

Three stages also read a sibling project, in place. They are never copied and
nothing under them is ever written:

| stage | reads | setting |
|---|---|---|
| 04_02 | `RAWCLICVehicleElectronics/Composition/draws` and `.../element_draws`, beside this repository | `materials.bev_electronics_draws_dir`, `materials.bev_electronics_element_draws_dir` (relative) |
| 04_03 | `RAWCLICVehicleTractionMotor/data/consolidated` | `materials.traction_composition_dir` (**absolute: edit it**) |
| 04_04 | `RAWCLICVehicleBattery/data/consolidated` | `materials.battery_composition_dir` (**absolute: edit it**) |

Everything the stages write goes under `data/processed/`: `intermediate/` (every
summary the pipeline reads back), `figures/`, `bev_draws/` (03_02), `carcomposition_draws/`
(04_01), `element_draws/` (04_02), `traction/` and `traction_recovery_draws/` (04_03),
`battery_draws/` and `battery_recovery_draws/` (04_04). The recovery model reads
`element_draws/`, `traction_recovery_draws/` and `battery_recovery_draws/`.

## Settings

Everything is in **`src/params_schema.py`**, each value with a comment saying what it
does and whether it is safe to change. Edit there, then run `00_parameters.py`, which
also validates the edit. `documentation/PARAMETER_REFERENCE.md` is generated from that
file and states its own count; do not edit it.

The ones that change most: `monte_carlo.n_draws` (about 1,000 to check a change runs,
20,000 for daily work, 200,000 for anything reported), `monte_carlo.enabled`,
`data_prep.scenario`, `adjusted_flows.scenarios_to_run` and
`materials.battery_chemistry_scenarios`.

## Layout

```
code/           the stages, the three checks, the parameter-reference generator
src/            the model: params_schema.py (every setting), the cohort-flow engines,
                the composition and battery libraries
documentation/  start at documentation/README.md
data/raw/       inputs (not tracked, except EEA_final_data.csv)
data/processed/ everything the stages write (not tracked)
```

## Two things that are easy to misread

**The draws are the result.** Stages write per-draw arrays (`.npy`) and take
percentiles from them only at the end, because the band of a total cannot be built
from the bands of its parts. That is why `data/processed/` is tens of gigabytes at
200,000 draws.

**Solid-state batteries have no described cell.** Stage 04_04 carries their packaging
but not their cathode, anode or electrolyte, and reports their share of cars as an
explicit gap rather than dropping it. Cell-material curves under the solid-state
scenario (S3) are lower bounds, not forecasts; see
[documentation/DESIGN_chemistries_without_composition.md](documentation/DESIGN_chemistries_without_composition.md).

What is tested, at what draw count, and what is still open is kept in
[documentation/HANDOVER.md](documentation/HANDOVER.md).

## Setup

No conda. Python 3.14 and a pinned `requirements.txt`; see
[documentation/SETUP.md](documentation/SETUP.md).

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## Related projects

| | |
|---|---|
| [RAWCLICRecoveryModel](https://github.com/MattRoess/RAWCLICRecoveryModel) | What is recovered from the end-of-life flows this model writes |
| [RAWCLICVehicleElectronics](https://github.com/MattRoess/RAWCLICVehicleElectronics) | Electronics composition per car, read by 04_02 |
| [RAWCLICVehicleTractionMotor](https://github.com/MattRoess/RAWCLICVehicleTractionMotor) | Traction-motor composition, read by 04_03 |
| [RAWCLICVehicleBattery](https://github.com/MattRoess/RAWCLICVehicleBattery) | Battery composition by capacity and chemistry, read by 04_04 |
