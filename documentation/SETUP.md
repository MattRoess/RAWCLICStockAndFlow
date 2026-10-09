# Setting up on a new machine

**No conda** — a plain virtual environment and a pinned `requirements.txt`.

Checked on 2026-10-07 from a fresh clone: §5's two commands, run with the environment
this repository was developed in (Python 3.14.4). Building a new environment from
`requirements.txt` was checked only in the sense that every pin is already satisfied
there; it was not installed from scratch.

## 1. Python 3.14

```bash
python3 -V
```

If it is not 3.14.x, install it — any of these, none involving conda:

- **python.org installer** (simplest on a fresh Mac): the macOS installer for 3.14.
  It installs to `/Library/Frameworks/Python.framework/Versions/3.14/`.
- **pyenv**: `pyenv install 3.14.4`
- **Homebrew**: `brew install python@3.14`

The environment is verified on **3.14.4**, which `.python-version` records. If only
3.13 is available the pins will most likely still resolve, but that combination is
untested.

## 2. Clone

```bash
git clone https://github.com/MattRoess/RAWCLICStockAndFlow.git
cd RAWCLICStockAndFlow
```

The repository is public and about 5 MB: the code, the documentation and one input
file. It does **not** hold the data (§4). If you will run stages 04_02 to 04_04, clone
the sibling projects (§4) into the same parent folder.

Keep the folder out of iCloud Drive and similar sync services if you can. They restore
files one at a time, in their own order, and that once left this repository's git
database with broken object links; `HANDOVER.md` has the account.

## 3. Create the environment

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

If `python3` is not 3.14, point at the interpreter directly, for example:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -m venv .venv
```

`.venv/` is gitignored and is rebuilt per machine. Never commit it. It is also **not
relocatable**: a venv records the absolute path it was created at, so moving the
project folder breaks it. The symptom is a prompt that still shows `(.venv)` while
`python` is not found, although `./.venv/bin/python` keeps working. Confirm with:

```bash
grep "^command" .venv/pyvenv.cfg && pwd
```

and rebuild, which takes about a minute:

```bash
rm -rf .venv && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

## 4. Editor

There are no workspace settings in the repository, so in Positron or VS Code:

1. **Open the project root**, not a parent folder.
2. **Select the interpreter**: Command Palette → `Python: Select Interpreter` → the one
   at `./.venv/bin/python` inside the project.
3. **Open a new terminal.** The prompt should show `(.venv)`.

To confirm which interpreter a terminal is really using:

```bash
which python && python -V && python -c "import scipy; print('scipy', scipy.__version__)"
```

It must print a path inside the project's `.venv`. Without that, `.venv/bin/python` in
every command below always works.

## 5. The inputs a clone does not have

Nothing after `00_parameters.py` can run until these are in place.

**In `data/raw/`** (created by you; the folder is ignored by git except for the EEA file):

| | read by | setting |
|---|---|---|
| `REMIND/*.mif`, the five scenario files | 01 | `data_prep.remind_scenario_files`, `data_prep.scenario` picks one |
| `usedvehicles_v1.2.xlsx` | 01 | `data_prep.export_data_file_name` |
| `composition/36_MonteCarlo_Summary.xlsx`, `SampleHistogram.xlsx` | 04_01 | |
| `EEA_final_data.csv` | 03_01, `test_stage03_inflow.py` | **already in the clone** |

**Beside the repository**, three sibling projects, each of which has to have been run so
that the files this model reads exist. They are read in place and nothing under them is
ever written:

| stage | project | run first | reads |
|---|---|---|---|
| 04_02 | [RAWCLICVehicleElectronics](https://github.com/MattRoess/RAWCLICVehicleElectronics) | `.venv/bin/python tools/mc_composition.py` | `Composition/draws`, `Composition/element_draws` |
| 04_03 | [RAWCLICVehicleTractionMotor](https://github.com/MattRoess/RAWCLICVehicleTractionMotor) | `01_composition.py` | `data/consolidated/TractionMotor_for_stockandflow.xlsx` |
| 04_04 | [RAWCLICVehicleBattery](https://github.com/MattRoess/RAWCLICVehicleBattery) | `05_composition.py` | `data/consolidated` |

**Two of those locations are absolute paths** in `src/params_schema.py`, written for
the machine this was developed on: `materials.traction_composition_dir` and
`materials.battery_composition_dir`. Set each to the `data/consolidated` folder of your
checkout of that project, and then run `00_parameters.py` again. The electronics
folders are relative (`../../RAWCLICVehicleElectronics/...` from `code/`), which holds
when the two repositories sit side by side.

## 6. The settings

Everything the model reads is in **`src/params_schema.py`**. Each value has a plain
comment above it saying what it does and whether it is safe to change.

After changing a value:

```bash
.venv/bin/python code/00_parameters.py
```

That writes the parameter file every stage reads and **validates the edit**, so a
mistake surfaces in seconds and not forty minutes into a run. It also writes
`code/params.xlsx`, a report that nothing reads. If you changed a parameter's
documentation, regenerate the reference as well:

```bash
.venv/bin/python code/generate_parameter_reference.py
```

## 7. Verify

What a clone can check on its own:

```bash
.venv/bin/python code/00_parameters.py          # exits 0 and says where it saved the parameter file
.venv/bin/python code/test_stage04_03_export.py # every check passes (nine today)
.venv/bin/python code/test_battery_seeds.py     # every check passes (nine today)
```

The other two checks need what §5 describes, and say so rather than failing quietly:
`code/test_stage04_02_export.py` stops with "element draws not found" when
`RAWCLICVehicleElectronics` is missing, and `code/test_stage03_inflow.py` needs the saved
output of stages 01 and 03_02. Then run the stages in the order in
[README.md](../README.md), starting with a small `monte_carlo.n_draws`.

## 8. Why the versions are pinned

So that the same inputs give the same numbers. Library behaviour has bitten this
pipeline before — for example a pandas/pyarrow combination that raised `ArrowTypeError`
on an Excel column holding mixed types (`src/data_prep.py`, `clean_export_data`).
Do not relax a pin without rerunning §7 and the checks in the README.
