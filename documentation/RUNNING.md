# Running the model

Everything you need to run the pipeline, change what it does, and know whether the
result is trustworthy.

---

## 1. The short version

```bash
cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/GitHub/RAWCLICStockAndFlow
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

The repository lives in iCloud Drive. That is where the path above comes from,
and it is also why a `.git` can end up split across two folders and a `.venv` can
come back as evicted placeholders — both have happened.

Each stage reads what the previous ones wrote, so the order matters. You can stop
anywhere; later stages simply have nothing to read and say so clearly rather than
producing wrong numbers.

---

## 2. What each stage does, and what it costs

Times are for 200,000 Monte Carlo draws on a 16 GB machine.

| stage | what it does | time | peak memory |
|---|---|---|---|
| `00_parameters.py` | Turns `src/params_schema.py` into the file every stage reads, and **validates it**. | seconds | — |
| `01_data_prep.py` | Reads the raw REMIND projections and trade data, trims to Europe, fills the years REMIND does not supply. | ~1 min | small |
| `02_stockdriven.py` | The fleet: how many vehicles exist, enter and leave, per drivetrain. Where the stock uncertainty is sampled. | ~68 s | **6.6 GB** |
| `03_01_flowdriven.py` | Splits the fleet into size segments and into collected / exported / untraceable. | ~2 min | moderate |
| `03_02_adjustedflows.py` | Re-simulates the fleet under each scenario. **The long one.** | **~40 min**, plus ~13 min for the BEV export | high |
| `04_01_carcomposition.py` | Turns vehicles into materials — steel, aluminium, copper, battery chemistry. | long | high |
| `04_02_BEVelectronics.py` | BEV electronics material flows, from both studies' real draws. | ~2 min | ~600 MB |
| `04_03_tractionmotors.py` | Traction motor material flows. | minutes | moderate |
| `04_04_batteries.py` | BEV battery material flows: three flows, three chemistry scenarios, elements and components, **every year**. Writes nine figures at the end. | **~6 h** (six chemistries; 47 min with five) | **~13 GB** |

`code/test_04_04_figures.py` redraws 04_04's figures from the draws already on
disk in ~30 s, without rerunning the stage. It is a bench tool, not a stage, and
not a test despite the prefix — the prefix is there because `0x_` belongs to the
things that produce results.

**04_04 writes about 48 GB of per-draw arrays** (40 GB with five chemistries) to `data/processed/battery_draws/`, and
that is deliberate: secondary supply is collected over inflow formed draw by
draw, and a mean with two percentiles cannot answer it. See
`BATTERY_MATERIAL_FLOWS.md`.

### Turning the cost down

The single biggest lever is `params.monte_carlo.n_draws`:

| draws | quality | use for |
|---|---|---|
| ~1,000 | visibly rough bands | checking a change runs at all |
| ~20,000 | good working compromise | day-to-day work |
| 200,000 | smooth distributions | anything reported |

More draws never change the answer, only how precisely the range is pinned down.

Setting `params.monte_carlo.enabled = False` removes the uncertainty analysis
entirely. Every ordinary result is still produced; you simply get no ranges, no
shaded bands and no distribution figures.

---

## 3. Changing what the model does

Every number lives in `src/params_schema.py`. Nothing is hardcoded anywhere else.

1. Edit only what is to the **right** of the `=`. Renaming a parameter breaks the code.
2. Keep the **type** — a number stays a number, text stays quoted, `True`/`False` stay capitalised.
3. Keep the punctuation inside `{ }` and `( )`. A missing comma is the commonest breakage.

Then **always**:

```bash
.venv/bin/python code/00_parameters.py
```

That regenerates the file the stages read *and* validates the edit, so a mistake
surfaces in seconds instead of forty minutes into a run.

If you changed a parameter's documentation, regenerate the reference too:

```bash
.venv/bin/python code/generate_parameter_reference.py
```

Full list of every parameter: `PARAMETER_REFERENCE.md`. It is generated and
states its own count — do not copy that number into another document, which is
how three of them came to disagree.

---

## 4. Things that catch people out

**Scenario-comparison figures need more than one scenario.** With
`scenarios_to_run = ("BAU",)` — the default — every cross-scenario figure has
nothing to compare and simply does not appear. Nothing is broken. Add a second
scenario to get them.

**Stage 04_02 needs two things that are not produced by default.**

- `03_02` must have run with `bev_electronics_export_draws = True` (it is, by
  default), which adds ~13 minutes.
- The electronics study must have written its draws:
  ```bash
  cd ../RAWCLICVehicleElectronics && .venv/bin/python tools/mc_composition.py
  ```
  ~106 s, writes ~584 MB to `Composition/draws/`.

If either is missing, `04_02` stops with a message saying exactly which.

**Stage 04_04 needs two things that are not in this repository.**

- `03_02`'s per-draw BEV export, in `data/processed/bev_draws/BAU/` — the same
  export `04_02` uses.
- The composition arrays written by the sibling battery project:
  ```bash
  cd ../RAWCLICVehicleBattery && .venv/bin/python 05_composition.py
  ```
  found through `materials.battery_composition_dir`.

**Both sides must run at the same number of draws.** 04_04 raises rather than
pairing draw *i* on one side with a different world on the other. If it stops
saying so, re-run whichever side is short.

**Old parameter files still load, but regenerate anyway.** A saved `00_params.pkl`
from before a parameter was added will inherit the new default rather than fail —
but only for simple values. Running `00_parameters.py` removes the ambiguity.

**Disk.** `data/processed` holds **58 GB** (measured 2026-09-15), and per-draw
arrays are all of it:

| | |
|---|---|
| `battery_draws/` | about 48 GB — 04_04, three flows x three scenarios x two levels, every year |
| `carcomposition_draws/` | 9.0 GB — 04_01 |
| `element_draws/` | 5.5 GB — 04_02 |
| `bev_draws/` | 2.6 GB — 03_02's BEV export |
| `intermediate/` | 724 MB — every summary the pipeline reads back |

It once held 71 GB, plus roughly 25 GB of accidental iCloud duplicates with a
trailing " 2" in the filename; both are gone, and `find data/processed -name
"* 2.*"` still returns nothing. Watch it again if
`materials.persist_mc_mass_draws` is switched on — that one adds tens of GB and
nothing reads it back.


Figures land in `figures/`, at the top of the repository (until 2026-10-09: `data/processed/figures/`).

---

## 5. Is the result trustworthy?

Check these before believing a number.

**Did the Monte Carlo actually run?** `params.monte_carlo.enabled` must be `True`.
Without it there are no bands at all.

**Is the inflow uncertainty being carried?**
`params.stock_flow.propagate_stage02_inflow_uncertainty` must be `True`. With it
off, downstream inflow bands are far too narrow — total BEV inflow varies by
0.000001% instead of about 9.6%. **Any result produced before 14 August 2026
predates this fix and should be regenerated.**

**Does the drivetrain mix substitute?** `params.stock_flow.stock_target_correlated_mix`
should be `True`, so that buying more BEVs means buying fewer of something else.

**Do the figures show bands?** A flat line where you expect a band means uncertainty
is not reaching that quantity. That is exactly how the inflow defect was found.

---

## 6. The other documents

| document | what it is for |
|---|---|
| `PARAMETER_REFERENCE.md` | Every parameter, with its explanation. **Generated — do not edit by hand.** |
| `MODEL_DESCRIPTION.md` | How the model works: cohort mechanics, special cases, Monte Carlo methodology. |
| `UNCERTAINTY_MAP.md` | Where uncertainty enters, travels, and stops. Read this before trusting a band. |
| `DESIGN_inflow_uncertainty_propagation.md` | Why the inflow propagation is built the way it is, including two rejected designs. |
| `DESIGN_chemistries_without_composition.md` | Why 04_04 cannot describe solid-state cars, how big that hole is, and why it stays open. |
| `BATTERY_MATERIAL_FLOWS.md` | Stage 04_04 in full: the three chemistry scenarios, why there are two levels of detail, and what the figures do and do not say. |
| `HANDOVER.md` | Current state: what is tested, at what draw count, and what is open. |

---

## 7. The standard this project runs on

**Correct Monte Carlo, or nothing.** In practice:

1. A stage reads another stage's actual draws, or calls the same function that
   produced them. It never re-derives them — that has silently broken this pipeline
   three times.
2. Sums happen on raw draws, before percentiles. Percentile-of-sum is the honest
   total; sum-of-percentiles assumes every part hits its extreme in the same world.
3. Substitution is real: one more A-segment car means one fewer of something else.
4. No invented distributions. Where one must be split, it is re-weighted so the
   parts provably recombine to the published original.
5. Any approximation is stated, with its direction — whether it makes the band too
   narrow or too wide.
6. Claims are backed by measurement on the real data, not by reasoning about the code.
