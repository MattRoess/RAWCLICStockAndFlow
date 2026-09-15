# Handover — updated 2026-09-15

Where the work stands, what is safe, what is not, and what to do next.

The newest work is stage **04_04**, the battery material flow, built on 14–15
September on top of battery-side work from the 9th. It has its own document —
**`BATTERY_MATERIAL_FLOWS.md`** — which is the one to read before touching it.
This file records what changed around it.

---

## 1. State of the repositories

### Since 2026-09-02 — stage 04_04, and what it dragged in with it

04_04 was rewritten from nothing usable into a per-draw battery material flow:
three flows, three chemistry scenarios, three chemistries, two levels of detail,
every year from 2020 to 2070, at 200,000 draws, with the chemistry shares
themselves drawn. `RAWCLICVehicleBattery`
supplies the composition and was changed to match.

| file | change |
|---|---|
| `code/04_04_batteries.py` | the stage, and the nine figures it draws at the end of every run |
| `code/test_04_04_figures.py` | bench tool, not a stage and not a test: redraws those figures alone from the saved draws, ~30 s |
| `src/battery_capacity.py` | **new** — the pack a segment carries, as a five-level discrete mixture drawn per car and held for its life |
| `src/battery_voltage.py` | **new** — 400 or 800 V, drawn per car, never blended |
| `src/battery_composition.py` | **new** — element and component masses at a drawn capacity, read from the battery project |
| `src/battery_chemistry.py` | **new** — the scenario shares, and the share with no composition behind it |
| `src/battery_vintage.py` | **new** — where the cars leaving the fleet were built |
| `src/params_schema.py` | the battery block: capacity levels, voltage shares, three chemistry scenarios, the composition directory; `battery_size_map` and `battery_composition_parameter_code` deleted |
| `src/artifacts.py` | registers `battery_material_flows`, `battery_component_flows`, `battery_chemistry_gaps` |
| `RAWCLICVehicleBattery/05_composition.py` | writes the component level beside the elements, and `check_draws_match_workbook()` — which its own docstring had claimed existed since 10 September and did not |
| `RAWCLICVehicleBattery/06_segment_capacity.py` | **deleted**, with the eleven parameters only it read |

**Three defects were found by building it, each of which produced numbers that
looked fine:**

1. **The outflow carried the scrap year's chemistry and pack.** A car scrapped
   in 2050 was built around 2036 and carries 2036's mix. Giving it 2050's meant
   the mix multiplied inflow and outflow by the same factor, cancelled out of
   every ratio between them, and **all three scenarios produced one identical
   curve**. `src/battery_vintage.py` now spreads each year's outflow back over
   the build years that could have produced it, weighted by the draw's own
   inflow history and the Weibull scrapping density stage 02 already uses.
2. **The elements do not add up to the pack.** The cell casing and the separator
   have no element rows and the electrolyte's cover 1% of its mass, so the
   element arrays miss **7–11% of every pack** — the plastics, the separator and
   the organic electrolyte, which is the part a recycler has to dispose of
   rather than sell. Both levels are now carried, in two frames that are
   deliberately not stacked into one.
3. **The mass-improvement factor is written every five years.** On the two-year
   grid, 2022, 2024, 2026 … silently had none — about 20% too much mass in every
   second year by 2070. `battery_composition.py` now interpolates it **per
   draw**.

Two smaller ones: the gap metric took the **maximum** over the chemistries
without a composition rather than the sum, understating S3's large segments as
70% against a measured 75%; and the outflow was being reported as secondary
supply, when only the **collected** part reaches a recycler — 88% of BEVs, with
2% exported and 10% never traced.

### Earlier work, to 2026-09-02

| file | change | tested |
|---|---|---|
| `src/params_schema.py` | reorganised into stage order; visualization section removed; every parameter rewritten for a non-specialist reader; 9 new BEV-electronics settings | yes — 511 parameters before and after, **zero changed** |
| `src/stockflow_model.py` | `run_stage02_cohort_monte_carlo` (shared orchestration); `sample_stock_target_draws`; `build_inflow_draws_by_drivetrain` | yes — stage 02 byte-identical, 45,121 values |
| `code/02_stockdriven.py` | calls the shared orchestrator; figure renamed and retitled | yes — byte-identical |
| `code/03_01_flowdriven.py` | calls the shared orchestrator (this is the 20.8% bug fix); axis labels | yes — 15/15 arrays exact |
| `code/03_02_adjustedflows.py` | restored the removed by-drivetrain MC and its figures; exports BEV per-draw arrays for 04_02; **now receives stage 02's inflow uncertainty** | yes — 300 draws, plus the four acceptance tests at 2,000 |
| `code/04_02_BEVelectronics.py` | **new** — BEV electronics material flows | yes — at 300 draws |
| `code/04_01_carcomposition.py` | all mass figures switched to Mt; new standard-vs-segments boxplot | yes — rendered |
| `src/artifacts.py` | registers `bev_electronics_summary` | yes |
| `code/04_01_carcomposition.py` | per-year mass draws exported for the recovery model; reads the per-year draws 03_02 already writes instead of asking for a period; every drivetrain gets a single-year vehicle count without re-running 03_02 | yes — re-run 31 Aug |
| `code/04_02_BEVelectronics.py` | an element no domain resolves is now SKIPPED with a note, not fatal (see §4.2) | yes — re-run 31 Aug |
| `src/plotting.py`, `src/params_schema.py`, `src/materials.py` | the two dead element plots deleted, and `element_list` / `element_list_noAlCu` with them (see §8) | yes — `Params()` builds, old pickle still loads |

### RAWCLICVehicleElectronics — 7 files changed by this work

| file | change | tested |
|---|---|---|
| `tools/mc_composition.py` | persists raw per-draw arrays to `Composition/draws/*.npy`; reads the PCB interface as `.pkl`; **sensor mass is now the triangular mean, not the mode** (2026-09-02, see §4.2) | yes — full 200,000-draw run; the CSVs regenerated byte-identical until the sensor fix, which deliberately changes two of the eight |
| `ElectricMotorElementMC/…py` | combined per-segment element fractions, **on whole-motor mass** — the denominator fix | yes — re-run, Cu within 0.15% of the material table, Σfrac = 1.00000000 per draw |
| `SensorElementsMC/…py` | emits per-draw element **masses** (not just fractions); writes one binary draw file instead of two identical CSVs | yes — re-run clean, totals identical to the previous values |
| `PCBElementMC/…py` | bulk results and grand totals → `.pkl`; `element_mass_by_year` → `.pkl` | see §4.2 |
| `SensorNumbersMC/…py` | one binary draw file instead of two identical CSVs | not re-run |
| `tools/build_composition.py` | reads the PCB interface as `.pkl` | not re-run |

**Binary outputs.** Bulk per-draw tables are now pandas pickles rather than CSV.
Two models were writing the *same* frame twice as byte-identical CSVs
(`SensorElementsMC`, `SensorNumbersMC`); each now writes once. Summary CSVs,
`Composition/csv` and the histogram CSVs are deliberately unchanged — the summaries
are meant to be read by a person, and `Composition/csv` is the byte-identical
validation anchor.

5.18 GB of regenerable CSV was deleted after checking that nothing reads it; the
repository went from 6.7 GB to 1.7 GB. The four files that ARE read
(`element_mass_by_year`, `pcb_year_resolved_*`, `elemental_summary`,
`materials_mc_summary`) were verified present afterwards.

`Data/30_BEV_electronics_composition.csv` and `docs/01_USER_GUIDE.md` also show as
modified. **Those are not from this work** — they were already modified at 13:38
and 11:58, before this session, and `mc_composition.py` writes neither.

Work continued on branch `carcomposition-draw-export` (the earlier work is on
`mc-correctness-and-bev-electronics`). A backup of the two electronics CSVs is at
`/private/tmp/elec_backup/` (temporary — copy it somewhere permanent if wanted).

---

## 2. To run the pipeline

```
code/00_parameters.py          FIRST — always, and especially after a schema edit
code/02_stockdriven.py         ~68 s at 200,000 draws, 6.6 GB peak
code/03_01_flowdriven.py
code/03_02_adjustedflows.py    ~40 min, plus ~13 min for the BEV export
code/04_01_carcomposition.py
code/04_02_BEVelectronics.py   needs 03_02's export and the electronics draws
code/04_03_tractionmotors.py
code/04_04_batteries.py        ~31 min, writes 25 GB of draws and nine figures
code/test_04_04_figures.py     ~30 s, redraws those figures without the stage
                               -- a bench tool, not part of the chain
```

**04_04 needs two things that are not in this repository.** 03_02's per-draw BEV
export (`data/processed/bev_draws/BAU/`), and the composition arrays written by
`RAWCLICVehicleBattery/05_composition.py`, found through
`materials.battery_composition_dir`. Both sides must run at the same number of
draws or 04_04 raises rather than pairing draw *i* with a different world.

Run `00_parameters.py` first, and 04_04 now refuses to start if you did not.

**Why it is now fatal there.** A field declared with `default_factory` lives in
the pickled instance, so an old file quietly wins over a new default, while a
field with a plain default is a class attribute and picks the new one up. A
stale file therefore does not fail — it HALF-UPDATES. It happened twice in two
days. On 2026-09-14 the only difference was a deleted parameter nothing reads,
so those results stood, which was luck rather than process. On 2026-09-15 it
cost fifty minutes: 04_04 ran to completion, wrote every artifact and every
figure, and carried three chemistries where the schema said five, with nothing
anywhere saying so.

`refuse_stale_parameters()` in 04_04 compares the saved object against a fresh
`Params()` field by field and stops with the difference named. **Only 04_04 has
it.** Putting it in `src/artifacts.py` so every stage gets it is the obvious
next step and has not been done.

---

## 3. What is verified, and to what standard

- **Stage 02 correlated drivetrain mix.** Byte-identical to the previous behaviour
  with the flag off (45,121 values, zero differences, three separate times).
  Substitution correlation −0.33 to −0.83 across 2030–2050; sum conservation to
  6.6e-16; historic years exactly unchanged.
- **Stage 03_01 fix.** 0 of 15 arrays matched stage 02 before, 15 of 15 after.
- **Electronics draws.** Full 200,000-draw run reproduces both published CSVs
  byte-identically, so the persisted draws are the same Monte Carlo universe as
  the study's own results.
- **Segment split in 04_02.** Reproduces the published AB distribution to **0.007%**
  at 200,000 draws.
- **04_01 and 04_02 figures.** Rendered and inspected.
- **04_04's summary table against its own draws.** 108 spot checks across every
  flow, scenario and chemistry: worst relative difference **1.7e-5**, which is
  float32 storage and nothing else. The table is computed from the draws, so the
  two cannot disagree by construction — this measures that they do not.
- **The component arrays against the consolidated CSVs.** 0.0000%, checked in
  the battery project on every anchor by `check_draws_match_workbook()`, and
  independently here afterwards. The guard was tested by breaking each level in
  turn: it caught 2% at the component level and 1% at the element level.
- **The elements against the components.** They miss 9.0% of the pack in 2020
  rising to 10.1% in 2070, against 7.2–10.9% measured per anchor in the battery
  project. Consistent.
- **Collected against outflow.** 87.9% in 2070, identical for every element to
  three decimals — the collection share is drawn on vehicles, not on materials.
  That is why there is no all-elements figure for the outflow: it would be the
  collected one times a constant.
- **04_04's nine figures.** Rendered and inspected, several of them more than
  once; two were redrawn because looking at them showed the argument was wrong,
  not the code.

Draw counts as actually run, from the output files themselves:

| stage | last run | draws |
|---|---|---|
| 03_02 + BEV export | 2026-08-28 16:37–16:47 | 200,000 |
| 04_01 | 2026-08-31 08:50 | 50,000 (`materials_mc_n_draws`) |
| `mc_composition` (electronics) | 2026-08-31 14:56 | 200,000 |
| `ElectricMotorMC` / `ElectricMotorElementMC` | 2026-08-31 12:43 / 14:09 | 200,000 |
| 04_02 | 2026-08-31 15:43 | **200,000** |
| 03_02 + BEV export (re-run) | 2026-09-14 12:24–12:25 | 200,000 |
| 04_04 | 2026-09-14 15:28 | **200,000** |
| `05_composition` (battery) | 2026-09-14 14:51 | 200,000 |

04_02 pairs draw *i* with draw *i* across independent samples and discards the
surplus rather than resampling, so its run size is the SMALLEST of its inputs. It
sat at 20,000 for a long time because `tools/mc_composition.py` had last been run
with `200000` passed as its argument omitted — the electronics draws were 20,000
rows against the fleet's 200,000, and nine draws in ten were thrown away. Re-running
`python3 tools/mc_composition.py 200000` (1m44s) fixed it. If 04_02 ever prints a
NOTE about mismatched draw counts, that is this, and the fix is to re-run whichever
input is short.

That re-run also regenerated `Composition/csv`. Six of its eight files came back
**byte-identical**; the two built from the Monte Carlo (`joint_mc_stats`,
`joint_mc_histograms`) moved as convergence, not as a shift — median change 0.03%
on the mean and median, 0.08–0.09% on the tails, 0.50% on the mode, which is the
least stable statistic at any draw count.

---

## 4. Open items

### 4.1 CLOSED — stage 02's inflow uncertainty now reaches 03_02

Fixed 14 August 2026. Stage 02's **absolute** per-draw inflow is transferred, split
to the fine drivetrains by 03_01's own shares and composed with 03_02's segment mix.
Governed by `propagate_stage02_inflow_uncertainty`, default on.

Verified by four acceptance tests: total BEV inflow CV rose from 0.000001% to
3.97 / 6.21 / 14.13% at 2030 / 2040 / 2050, against stage 02's own 4.0 / 6.2 / 14.2%;
segment-mix spread preserved (9.5–12.8% → 11.5–14.0%); historic years clean; and
with the switch off, means and CVs return exactly to the previous values.

Design, rejected alternatives and measurements:
`DESIGN_inflow_uncertainty_propagation.md`.

**Consequence: any result produced before 14 August 2026 understates inflow
uncertainty downstream and should be regenerated.**

### 4.1b CLOSED — collected-flow fix is in, and the re-run is done

Fixed 19 August 2026, reported by Yousef and colleagues. `03_02`'s deterministic
tracker computed `collected = out_survival x (1 - unknown)`, dropping the export
share — the only one of four implementations that did. It overstated collected by
2.3% for BEV and **16.3% for every other drivetrain**, and double-counted exported
vehicles. Full account: `DESIGN_collected_flow_definition.md`.

Done: the shared function is now used everywhere, and the partition invariant is
asserted at three choke points. 03_01 re-run clean at 200,000 draws with the guards
active, numbers unchanged.

**DONE 2026-08-28.** 03_02 re-ran with the guards active — `03_02_mc_summary.pkl`
written 16:47 and the whole BEV export 16:37–16:46, well after the last code change
to 03_02 or `stockflow_model.py` (`714fa6b`, 21 Aug 06:30). The run completed and
saved, so the partition assertions did not fire. 04_01 followed on 2026-08-31 07:45,
so its collected material masses are the corrected ones.

Not verified: the *predicted magnitude* — collected falling ~2.3% for BEV and ~16.3%
elsewhere. Nobody has compared the new summary against the old one. If that matters,
it is a direct comparison of two pickles, not another run.

Stage 04_02 is unaffected either way: it reads `per_year_collected` from the Monte
Carlo engine, which always used the correct three-way split.

### 4.1c OPEN — the HEV/PHEV split is a category error, NOT a missing volume

**CORRECTED 2026-08-20, same day.** This entry first said "the model's HEV + PHEV
inflow is 0.66x real EU registrations in 2019 and 0.36x in 2023 — a genuine
modelling error in stage 02's Hybrid volume". **That comparison was invalid** and the
conclusion drawn from it was wrong. There is no missing hybrid volume. The original
text is kept here because the mistake is instructive: two series were compared on the
strength of sharing a label.

WHAT IS ACTUALLY WRONG. The REMIND files contain five LDV technologies —
`Liquids`, `Hybrid electric`, `Gases`, `FCEV`, `BEV`. There is **no non-plug-in
hybrid category**. In REMIND's taxonomy `Hybrid electric` is the PLUG-IN hybrid;
full and mild hybrids sit inside `Liquids`. `data_prep.py:241` renames
`Hybrid electric` to `Hybrid`, and stage 03_01 then splits that into HEV and PHEV --
**manufacturing an HEV series out of plug-in-hybrid volume, while the real HEVs are
still being counted inside Liquids.**

WHY THE ORIGINAL COMPARISON WAS INVALID. In `EEA_final_data.csv`, `HEV` is assigned
from fuel types PETROL / DIESEL / E85 — it is dominated by 48V mild hybrids, which
are petrol and diesel cars. Comparing REMIND's plug-in class against that label
compares different populations. Reclassify the mild hybrids back to Liquids and the
apparent defect inverts:

| | model / real |
|---|---|
| EEA `HEV` at face value | 0.66x (2019), 0.36x (2023) |
| mild hybrids returned to Liquids | 1.14x (2020), 1.02x (2021), 1.57x (2023) |

The model is not short of hybrids. With mild hybrids counted as Liquids, the model's
Liquids matches reality at **1.03–1.09x for 2015–2019**.

VALIDATED AGAINST ACEA. Every category in the EEA file agrees with ACEA's published
2023 shares (10.5 M market): Petrol 3.71 vs 3.693, HEV 2.71 vs 2.855, BEV 1.53 vs
1.669, Diesel 1.43 vs 1.370, PHEV 0.81 vs 0.840 M. The real data is sound; only the
mapping onto REMIND's categories was wrong.

REMIND's single hybrid class (7.278 M stock, 1.319 M inflow in 2023) matches neither
real category — 1.6x real PHEV inflow, 0.37x real HEV+PHEV inflow — and ACEA's HEV,
25.8% of the market at 2.71 M cars a year, has nowhere to live in the model except
inside Liquids.

**IMPLEMENTED 2026-08-20 (option B).** HEV is carved out of Liquids using the real
registration shares; the whole of REMIND's `Hybrid` is PHEV. Governed by
`disaggregation.hev_carved_from_liquids` (default True) and
`hev_share_phaseout_end_year` (default 2035). Measured after the change: HEV 1.06x
real at 2019, PHEV 1.02x at 2021 (0.33x before), conservation 1.11e-16.
`code/test_stage03_inflow.py` is now 13 tests and covers it.
Full account: `DESIGN_hev_carve_from_liquids.md`.

**THE DRIVETRAIN TAXONOMY HAS CHANGED.** `HEV` is now a slice of Liquids, not of
Hybrid; `PHEV` is all of Hybrid. All three stages have now been re-run — 03_01,
then 03_02 on 28 Aug and 04_01 on 31 Aug — so downstream holds the new split.
Anything published *before* those dates with the old HEV/PHEV series should be
regenerated.

Also open there: 2020-2023 runs 1.9-2.2x high because the model does not reproduce
the COVID and chip-shortage collapse, and 1975-2004 is flat because **no pre-2005
data exists in this project** — supplying a historical registrations series would fix
that one.

### 4.2 Smaller open items

- **`04_02` element resolution — DONE, 14 August 2026.** All four domains now carry
  per-draw elements at 200,000 draws. Both blocking constraints were removed by
  having the element models emit per-draw arrays (`Composition/element_draws/`)
  rather than reading the two-year summary CSV.

  Two real defects were found and fixed on the way, both of which would have
  produced plausible-looking wrong numbers with nothing in the output to show for it:
  the **motor fractions used the wrong denominator** (aluminium and plastic are 10–22%
  of a motor and are not elementally resolved, so every motor element was high by
  11–28%), and **sensor mass is estimated from modes** rather than means, which
  understates it by 1.73x. Full account, with measurements:
  `DESIGN_element_resolution.md`.

  **Total copper is now ~16% higher than this stage used to report**, because it
  previously showed wiring only. 2050: inflow 552.6, outflow 422.7, collected 371.1
  kt (medians).

  **CLOSED 2026-09-02 — the Sensors series is no longer mode-based.**
  `sensor_mg_per_type()` summed each element's `_mode_mg`; expected mass is
  `E[Σx] = Σ E[x]` and the mean of a triangular is `(min + mode + max)/3`, so it now
  uses that. Sensors were the only domain estimated from modes.

  Measured at 200,000 draws against the run immediately before: Wiring, PCB and
  Motors **×1.0000 exactly**, Sensors ×1.891 / 1.904 / 1.919 for AB / CD / EF, Total
  ×1.002–1.003. The realised factor is 1.90 rather than the 1.62 the source table
  suggests, because the table counts all 60 sensor types once each while a vehicle
  carries a mix — per-type ratios run 1.37× (ultrasonic) to 3.67× (current sensor),
  and the ratio rising with segment size is that mix changing.

  04_02's ELEMENT masses are unchanged to the fourth digit, as designed: it takes
  sensor element mass from `SensorElementsMC`'s own level and only the shape of this
  series. Its Sensors DOMAIN mass moved 1.6 → 3.1 kt collected in 2050.

  The 1.07× count half of the old 1.73× gap is deliberately NOT fixed:
  `mc_composition` uses `SensorNumbersMC`'s per-draw counts against
  `SensorElementsMC`'s uniform mean, and per-draw counts are the better convention.

  **`Composition/csv` is no longer byte-identical to the published tables**, and that
  was the reason this sat deferred since 14 August — the anchor was worth more than
  the correction while the correction was not needed. The anchor is now spent, on
  purpose. Six of its eight files are still byte-identical; `joint_mc_stats` and
  `joint_mc_histograms` carry the sensor change.
- **`04_02` was welded to its element list — FIXED 2026-08-31 (`cbf8903`).**
  `resolve_elements` raised on any requested element that no domain resolved, so a
  set of element draws without Pr, Tb and Nb stopped the entire stage instead of
  reporting the other fifteen. Which elements exist is a property of the upstream
  models' files, not of this stage, and the request list is only a selection of what
  to report. Unresolved names are now skipped with a note naming them and listing
  what is available; `main` narrows `elements` to what resolved, so the figures, the
  tables and the draw export all see one consistent list. A list where **nothing**
  resolves still raises — that is the wrong draws directory, not a wrong request.
  The closing copper read-out is guarded too, so dropping `Cu` no longer crashes it.
  Verified by a full run: 15 elements reported, Pr/Tb/Nb skipped, every figure drawn.
  See §8 for the rest of the codebase.
- **THE MOTOR MAGNET IS A FERRITE, AND IT IS NAMED FOR THE PART** (2026-08-31).
  The composition switched from NdFeB to strontium ferrite on 13 August, but the
  NAME did not follow: `ratio__NdFeB`, the histogram filenames and the element
  model's `mass_col` all still said NdFeB, because the magnet row was identified by
  its MATERIAL. A motor has a magnet whatever it is made of, and
  `05_VehicleElectricMotorsWeight.xlsx` already carried that in its `component`
  column, unused. Now: workbook `A4` = `Magnet` in all four motor sheets,
  `ElectricMotorMC` keys that row on `component`, exports are `ratio__Magnet` /
  `mass_kg__Magnet`. Changing the magnet material is now a workbook cell and a
  composition sheet — no code, no filenames, no column names. Verified as a pure
  rename: 04_02's output was byte-identical afterwards.

  **This is also why `Pr`, `Tb` and `Nb` stopped resolving** and the old hard-coded
  element list failed — those three existed only in the NdFeB magnet. The rare
  earths still in the results come from Sensors, which is the sensor study's own
  data and a separate question.

  **Known bias, left in deliberately**, recorded in full in
  `ElectricMotorElementMC.py` above `FERRITE_ELEMENTS` — not repeated here. Short
  version: the stepper sheets' magnet mass fractions were measured on NdFeB
  magnets and the model now splits that mass as ferrite, so stepper magnet mass is
  LOW. Direction known, size not; correcting it needs a source, not a multiplier.

  Naming: NdFeB is a **REM**, Fe-Sr is a **ferrite**, and a magnet is called by its
  name, never by its constituent elements.
- **04_01 AND 04_02 DO NOT DOUBLE COUNT — 04_02's mass is ADDITIONAL** (measured
  2026-09-02). 04_01's BEV 2050 inflow is 33,415 kt across twelve components
  (`elvBIW`, `elvBattery`, `elvChassis`, `elvPowertrain`, …). Searching its material
  names for electronics, cable, actuator, controller, light, wiring or harness
  returns **nothing**: vehicle electronics are absent from its composition data,
  which is why 04_02 exists. 04_02's 992.9 kt of BEV electronics is therefore on top
  of 04_01's total, not inside it. Neither stage reads the other and nothing sums
  them, but both now export per-draw arrays for the recovery model, so this is the
  fact that stage needs.

  What misled a reader once: `MATERIALS_ALWAYS_SHOWN_INDIVIDUALLY` in 04_01 lists
  `powerElectronics`, `actuators`, `controllers`, `cableLike`. It is a DISPLAY
  preference, inert at the current settings (`top_n` is None), and it names materials
  the composition data does not contain. It is not a second electronics account.
- `KG_PER_TONNE` in `04_01_carcomposition.py` — **DELETED 2026-09-15**, along
  with `battery_composition_parameter_code` (no reader but its own validation,
  and its value `"e-m"` is not one of the codes the battery workbook carries),
  an unused `gaps` argument and a label map that mapped every key to itself.
  `Params().validate()` now returns nothing at all, which is what makes a real
  complaint visible.
- Two functions vanished from `04_01` on 2026-07-09 (`plot_material_mass_by_year`,
  `quantify_and_aggregate`). Successors appear to exist; not confirmed.
- **`data/processed/intermediate/` is now 724 MB**, and `find` reports **zero**
  files with a trailing " 2" in the name (checked 31 Aug). The 71 GB and the ~25 GB
  of accidental copies are gone. `data/processed` as a whole is 3.6 GB.
- Scenario-comparison figures need two or more entries in `scenarios_to_run`;
  it is currently `("BAU",)`.
- `materials_mc_n_draws` is **200,000** (this entry said 50,000 until
  2026-09-15; it was wrong).
- **The parameter count belongs in one place.** `PARAMETER_REFERENCE.md` is
  generated and says **155**; this file used to say 145 and `README.md` says 140,
  and `00_parameters.py` prints 536 because it counts every dictionary entry.
  Do not repeat the number — point at the generated file.
- **`RUNNING.md` still says 04_03 and 04_04 are "not part of this chain".**
  True when it was written, wrong now: 04_04 consumes 03_02's BEV export and the
  battery project's arrays, takes half an hour and writes 25 GB.

### 4.3 OPEN — what 04_04 cannot yet say

- **Sodium-ion and solid-state have no composition at all.** Their share is
  reported as an explicit gap rather than dropped, but under S3 that gap is
  69.3% [63.9-74.2] of new cars by 2070, and it passes 10% of the fleet in 2031.
  **Left open deliberately** -- nobody has published a composition for either
  that survives scrutiny, and inventing one would be worse than the hole. Full
  account, with the measurements and the rejected alternatives:
  `DESIGN_chemistries_without_composition.md`.
- **Half of that gap is an export, not a data problem, and could be closed now.**
  The battery project DOES model these two: at 60 kWh it gives sodium-ion
  209.5 kg of casing, separator, cables, terminals, enclosure, frame, thermal
  conductor and current collectors -- within 3 kg of LFP's own structure -- and
  leaves only the ACTIVE material empty. It writes no `.npy` draw arrays for
  them, so `src/battery_composition.py` raises and 04_04 drops the whole car,
  structure included. Exporting those draws would move the gap from "the car" to
  "the active material". See §2 of that document.
- **The material level is empty of information.** The battery workbook's `m-c`
  rows carry a mass but no material name, and only for three components — which
  is why the COMPONENT level was built instead. A real material breakdown needs
  the workbook to supply one.
- **The composition is clamped before 2020**, because that is where the battery
  project's improvement factors start. It touches 100% of the 2020 outflow, 40%
  of 2030, 8% of 2036 and under 1% after 2042. The pack size is read at the true
  build year, so only the composition is clamped.
- **The vintage weights use the central lifetime, not the drawn one.** 03_02
  reports a year's outflow as one number rather than a matrix by build year, so
  the mixture is reconstructed. The weights therefore carry less spread than the
  flows they weight. Fixing it properly means exporting a cohort-resolved
  outflow from 03_02.
- The figure loaders in 04_04 raise `SystemExit` where the rest of the codebase
  raises a typed error. Cosmetic, but inconsistent.

---

## 5. The rule this project runs on

**Correct Monte Carlo, or nothing. No compromises, no approximations that quietly
narrow a band.**

Concretely, what that has meant in practice:

1. **A stage reads another stage's actual draws, or calls the same function that
   produced them.** It never re-derives them. Stage 03_01 broke this three times and
   was silently wrong each time, because a re-derivation that is missing an argument
   still produces plausible-looking numbers.
2. **Sums happen on raw draws, before percentiles.** Percentile-of-sum is the
   honest total; sum-of-percentiles assumes every part hits its extreme in the same
   world.
3. **Substitution is real.** One more A-segment car means one fewer of something
   else; one more BEV means one fewer of another drivetrain. Shares are renormalised
   so they sum to one.
4. **No invented distributions.** Where a distribution has to be split, it is
   re-weighted so the parts provably recombine to the published original.
5. **An approximation is stated, with its direction.** Independence between the
   fleet and electronics Monte Carlos is assumed and documented, including the fact
   that it makes the band slightly narrow rather than wide.
6. **Verify against the real data, not against reasoning.** Every claim in these
   documents has a measured number behind it.
7. **Persist the draws, not the statistics.** A ratio of two percentiles is not
   the percentile of a ratio. 04_04 keeps 25 GB of per-draw arrays for exactly
   this reason: secondary supply is collected over inflow, formed draw by draw,
   and a mean with two percentiles cannot answer it.
8. **A discrete state is drawn, never blended.** A pack is 400 V or 800 V and a
   segment offers a handful of pack sizes, not a continuum. Share-weighting a
   bimodal mixture collapses it onto its own mean and destroys the band — which
   is the failure this rule exists to prevent, not a hypothetical one.

---

## 6. How to work on this

Written down because it was learned the hard way in this session.

- **Measure before proposing.** The ratio-based propagation was proposed after
  reading code and before looking at the data. Ten minutes of measurement would have
  killed it. It cost far more than that to walk back.
- **Ask before implementing.** Every parameter and design choice is the user's.
- **Do not claim more than was tested.** "Everything is correct" was said when what
  had been verified was stages 02 and 03_01 only. The 03_02 boundary had never been
  examined, and that is where the defect was.
- **Render figures and look at them.** An audit that greps for `set_ylabel` passes a
  chart whose unit label is rotated 90° and clipped off the page. That happened.
- **Keep answers short.** State the finding, the evidence, the recommendation.
- **A file of its own is a waiting room, not a home.** Cut code may sit in its
  own runnable file while a decision is open. When the decision is made, the
  file goes — `06_segment_capacity.py` sat there for four days and was deleted
  on 2026-09-14, because by then the question it answered was answered better
  elsewhere. Leaving it was not neutral: it was runnable and wrote plausible
  files with two defects stated in its own docstring.
- **When a docstring claims a check exists, check that it exists.**
  `apply_pack_rules_to_draws` said `check_draws_match_workbook()` guarded it on
  every run. That function did not exist for four days. It does now, and it was
  tested by breaking the thing it guards.

---

## 7. Documentation (added 14 August 2026)

The two Word documents were a month out of date and nothing had flagged it. They are
now Markdown, one of them generated, and moved to `superseded/`.

| document | state |
|---|---|
| `README.md` | index — start here |
| `RUNNING.md` | how to run everything, what it costs, what catches people out |
| `PARAMETER_REFERENCE.md` | every parameter, **generated** from `src/params_schema.py`. It states its own count; do not copy that number anywhere else |
| `MODEL_DESCRIPTION.md` | converted from the `.docx`, plus section 7 for everything since 13 July |
| `UNCERTAINTY_MAP.md` | where uncertainty enters, travels and stops |
| `DESIGN_inflow_uncertainty_propagation.md` | the propagation design and two rejected alternatives |
| `DESIGN_collected_flow_definition.md` | the collected-flow defect and the shared definition (§4.1b) |
| `DESIGN_hev_carve_from_liquids.md` | why HEV is carved out of Liquids (§4.1c) |
| `DESIGN_element_resolution.md` | how 04_02 resolves elements, and the two defects found doing it (§4.2, §8) |
| `DESIGN_inflow_parent_split.md` | which stage-02 parent each fine drivetrain draws its uncertainty from |
| `BATTERY_MATERIAL_FLOWS.md` | **stage 04_04** — the three scenarios, the two levels, the vintage rule, and what the figures do and do not say. Read this one first for anything battery |
| `DESIGN_bev_capacity_for_04_04.md` | why the pack a segment carries is a drawn five-level mixture and not a map |
| `DESIGN_discrete_vehicle_states.md` | why voltage and pack size are drawn per car and never share-weighted |
| `DESIGN_chemistries_without_composition.md` | the sodium-ion and solid-state gap: how big, why it stays open, and the half of it that is only an export away |

Regenerate the parameter reference after any parameter change:

```bash
.venv/bin/python code/generate_parameter_reference.py
```

It reports how many parameters have no explanation in the code. That count is **0**
and should stay there — it is the check that stops the reference silently rotting the
way its predecessor did.

---

## 8. Element independence — the rest of the codebase

Written 2026-08-31, after `04_02`'s element list turned out to be a hard gate.

### The principle

**A name list is a request, not a contract.** Which elements exist is a property of
the upstream models' output files. A stage asked for an element those files do not
carry should say so and report the rest. Two failure modes bracket the right
behaviour, and both are wrong:

| | behaviour | why it is wrong |
|---|---|---|
| too strict | raise on any unresolved name | one absent element costs you the other fifteen; the stage cannot run against a different set of draws |
| too loose | drop it silently | the figure comes out with a missing series and nothing says why; this is the failure mode that produces plausible wrong numbers |

The rule: **skip, name what was skipped, list what was available, continue.** That
is what `resolve_elements` now does and it is the pattern to copy.

The distinction that matters: this applies to **name lists** (which elements, which
materials). It does **not** apply to **structural** checks — a missing column, a
missing draws file, a domain absent from the series list. Those stay fatal, and
several of them are (`load_element_draws`, `materials.py`, `04_03`, and in
`04_04` both `src/battery_composition.py`'s `CompositionError` and the "no draws
in this directory" check). A shorter list is a legitimate input; a broken file is
not. Line numbers are deliberately not quoted here — the ones this section used
to carry pointed into a file that has since been rewritten twice.

### What the survey found

**No element list is hard-coded in the code.** The names come from the
`*_elements.txt` files beside the `.npy` draws;
`materials.bev_electronics_elements` selects which of them to REPORT, and empty
would mean all of them.

**An element is identified by the material it sits in.** `S__esteel` and
`S__copper` are both sulfur and are not the same quantity — one is alloyed into the
electrical steel, the other is an impurity in the copper winding. They are different
materials in different parts of the motor and adding them describes nothing. Copper
is the single exception, summed across every motor material as a plain `Cu`, because
one motor total is what it is wanted for. Bulk aluminium is `Al__bulk` under the same
rule. Only the MOTOR names carry a material; the PCB and sensor models resolve one
material each, so their names are plain — which does mean `Fe` (PCB) and `Fe__esteel`
(motors) sit side by side under different conventions. Making that uniform is a
change in those two models.

**The report is the critical and strategic materials, 22 of them** — set 2026-08-31,
with the exclusions and their reasons written into the parameter itself. Left out:
the 16 `__copper` entries as contamination (which removes Bi, Sb, Se, Te and Cd
entirely, as they occur only there); S, O, C and P wherever they sit; Fe as bulk;
Si, `Si__esteel` and Ba as alloying or bulk; Plastic and Unspecified as not elements.
`Mn__esteel` and `Mn__cfsteel` are IN — manganese in the steels is an alloying
addition, not contamination.

**Figures: 17, down from 86.** There used to be two per element, a total and a domain
split, which was 80 of them and grew with the list; they were redundant with the
panels in `04_02_13`/`_14` and the recovery spread in `_15`. `fig_element_total` and
`fig_element_domains` are kept, so a named element goes back on its own page as one
appended job.

One element name remains anywhere in the code: `WIRING_ELEMENT = "Cu"` in `04_02`.
That is a statement about the model rather than a selection — the wiring model
reports a copper mass and nothing else, so Wiring has no `*_elements.txt` and its
single element cannot be read from a file that does not exist. The two dedicated
copper figures are guarded on `"Cu" in elements`, so a request that drops copper
still runs.

| site | state |
|---|---|
| `04_02` `resolve_elements` | FIXED — skips, names the skip, continues (`cbf8903`), and an empty request now means every element the draws carry |
| `materials.bev_electronics_elements` | was 18 hard-coded names; now names the 22 critical and strategic materials, and an empty value would mean "whatever the models resolved" |
| `src/plotting.py` `plot_elements_by_flow` | DELETED — dead code, no callers |
| `src/plotting.py` `plot_mass_by_drv_flow_elements` | DELETED — dead code, no callers |
| `data_prep.element_list` / `element_list_noAlCu` | DELETED with them; the two plotting functions were their only consumers, so they were inert |
| `ELEMENT_LIST` / `ELEMENT_LIST_NO_AL_CU` in `params_schema.py` | DELETED — the two parameters were their only use |
| `04_01` `MATERIALS_ALWAYS_SHOWN_INDIVIDUALLY` | already independent — a "prefer to show" set; an absent name simply never matches, and it only takes effect when `top_n` is an int, which it is not |
| structural checks in `04_01`, `04_03`, `04_04`, `materials.py` | correctly fatal — a missing column or draws file is a broken input, not a shorter list |

The deletion was checked, not assumed. Both functions were greppable to nothing
across `.py` and `.ipynb`; the two parameters were used only by them; the two
module constants were used only by the two parameters. `Params()` still builds, the
parameter count went 147 → 145, and the existing `00_params.pkl` still unpickles —
it simply carries `element_list` and `element_list_noAlCu` as stray attributes on
`data_prep` that no code reads, and `asdict` round-trips unchanged.

The archived copies under `src/archive/` still contain all of this. They were left
alone deliberately: they are snapshots, not live code.

### The workflow, when a new name list appears

1. **Find where the names actually come from.** For elements it is the `.txt` files
   beside the `.npy` draws. The available set is data, so read it; never hard-code
   a mirror of it.
2. **Resolve, do not assert.** Partition the request into resolved and unresolved.
3. **Narrow the working list to what resolved**, once, near the top of `main`, and
   pass that list everywhere downstream. This is the step that keeps the figures,
   the summary tables and the draw export from disagreeing.
4. **Print the skip.** Name what was dropped and list what was available, on one
   line. Silence here is the expensive failure.
5. **Raise only on an empty intersection**, and say in the message that it points at
   the wrong input directory rather than at the request list.
6. **Guard any code that names a specific element**, such as the copper read-out at
   the end of `04_02` — `if "Cu" in elements:`.
7. **Run the stage against a deliberately short list** and confirm it completes.

---

## 9. Reproducibility — one defect, fixed

Found 2026-08-31 while explaining why two 04_02 runs disagreed in the third decimal.

`04_02` seeded each pair's segment split with `seed + hash(group) % 10_000`. **Python
salts `str` hashing per process**, so that seed was different in every run and the
stage was not reproducible. The symptom was small enough to dismiss as ordinary Monte
Carlo noise — the recombination error moved in the fourth decimal, 2050 collected
copper wandered by ~0.07% — which is exactly why it survived this long. It is now
`zlib.crc32(group.encode()) % 10_000`, the same number in every process, forever.

Verified: two consecutive full runs of 04_02 are **byte-identical**, diffed over the
complete log.

**This was the only unseeded randomness in the pipeline.** The sweep behind that
claim: no module in `code/` or `src/` calls the global `np.random.*` samplers or
constructs a bare `default_rng()`; every generator descends from `monte_carlo.seed`
or `materials_mc_seed` through `np.random.SeedSequence`, spawned in sorted order
(`03_01`, over `sorted(mc_stage02.keys())`), in configuration order (`03_02` and
`04_01`, over `active_scenario_names()`), or in the input frame's own row order
(`cohort_flow_mc`, per entity). `hash()` appeared exactly once in the whole
codebase, and that was the one site.

If a stage ever needs a per-name seed offset again: **crc32, never `hash`.**
