# Handover — updated 2026-08-31

Where the work stands, what is safe, what is not, and what to do next.

---

## 1. State of the repositories

### RAWCLICStockAndFlow

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

### RAWCLICVehicleElectronics — 7 files changed by this work

| file | change | tested |
|---|---|---|
| `tools/mc_composition.py` | persists raw per-draw arrays to `Composition/draws/*.npy`; reads the PCB interface as `.pkl` | yes — full 200,000-draw run, **both CSVs regenerate byte-identical** |
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
code/00_parameters.py          FIRST — there are new settings
code/02_stockdriven.py         ~68 s at 200,000 draws, 6.6 GB peak
code/03_01_flowdriven.py
code/03_02_adjustedflows.py    ~40 min, plus ~13 min for the BEV export
code/04_01_carcomposition.py
code/04_02_BEVelectronics.py   needs 03_02's export and the electronics draws
```

Run `00_parameters.py` first. Old saved parameter files do still load — the
`default_factory` trap that would have broken them was found and fixed — but
regenerating is cleaner.

`code/inspect_mc.py` browses any stage's Monte Carlo results without opening a
pickle by hand.

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

Draw counts as actually run, from the output files themselves:

| stage | last run | draws |
|---|---|---|
| 03_02 + BEV export | 2026-08-28 16:37–16:47 | 200,000 (96 years, float32) |
| 04_01 | 2026-08-31 07:45 | 50,000 (`materials_mc_n_draws`) |
| 04_02 | 2026-08-31 07:52 | 20,000 — the electronics models' own count, which caps it |

04_02 is capped by the electronics draws, not by choice: the fleet arrays hold
200,000 and the electronics arrays 20,000, and the stage pairs the first 20,000 of
each rather than resampling. Raising it means re-running the element models at
>= the fleet count.

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

  Still open: `mc_composition`'s Sensors series is itself mode-based and understates
  sensors by ~1.73x. 04_02 works around it by taking sensor element masses at the
  sensor study's own level, but the **domain-mass** figures still carry the
  understatement. Fixing it at source regenerates `Composition/csv`, which has been
  validated byte-identically, so it was deliberately deferred.
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
- `KG_PER_TONNE` in `04_01_carcomposition.py` is still there and still unused
  (checked 31 Aug).
- Two functions vanished from `04_01` on 2026-07-09 (`plot_material_mass_by_year`,
  `quantify_and_aggregate`). Successors appear to exist; not confirmed.
- **`data/processed/intermediate/` is now 724 MB**, and `find` reports **zero**
  files with a trailing " 2" in the name (checked 31 Aug). The 71 GB and the ~25 GB
  of accidental copies are gone. `data/processed` as a whole is 3.6 GB.
- Scenario-comparison figures need two or more entries in `scenarios_to_run`;
  it is currently `("BAU",)`.
- `materials_mc_n_draws` is now **50,000**, down from 200,000.

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

---

## 7. Documentation (added 14 August 2026)

The two Word documents were a month out of date and nothing had flagged it. They are
now Markdown, one of them generated, and moved to `superseded/`.

| document | state |
|---|---|
| `README.md` | index — start here |
| `RUNNING.md` | how to run everything, what it costs, what catches people out |
| `PARAMETER_REFERENCE.md` | all **147** parameters, **generated** from `src/params_schema.py` |
| `MODEL_DESCRIPTION.md` | converted from the `.docx`, plus section 7 for everything since 13 July |
| `UNCERTAINTY_MAP.md` | where uncertainty enters, travels and stops |
| `DESIGN_inflow_uncertainty_propagation.md` | the propagation design and two rejected alternatives |
| `DESIGN_collected_flow_definition.md` | the collected-flow defect and the shared definition (§4.1b) |
| `DESIGN_hev_carve_from_liquids.md` | why HEV is carved out of Liquids (§4.1c) |
| `DESIGN_element_resolution.md` | how 04_02 resolves elements, and the two defects found doing it (§4.2, §8) |
| `DESIGN_inflow_parent_split.md` | which stage-02 parent each fine drivetrain draws its uncertainty from |

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
several of them are (`load_element_draws`, `materials.py:200`, `04_04:136`,
`04_03:207`). A shorter list is a legitimate input; a broken file is not.

### What the survey found

Element names appear in exactly two files — `src/params_schema.py` and
`code/04_02_BEVelectronics.py`. Everything else refers to elements through a
parameter. So the blast radius is small.

| site | state | action |
|---|---|---|
| `04_02` `resolve_elements` | FIXED `cbf8903` | none |
| `src/plotting.py:450` `plot_elements_by_flow` | too loose — `.isin(element_list)`, then `continue` on an empty frame, silently | see below |
| `src/plotting.py:711` `plot_mass_by_drv_flow_elements` | same | see below |
| `general.element_list` / `element_list_noAlCu` | `("Ag","In","Ta","Zn","Dy","Nd","Pr","Al","Cu")` and the same without Al/Cu | see below |
| `04_01` `MATERIALS_ALWAYS_SHOWN_INDIVIDUALLY` | already independent — a "prefer to show" set; an absent name simply never matches, and it only takes effect when `top_n` is an int, which it is not | none |
| structural checks in `04_01`, `04_03`, `04_04`, `materials.py` | correctly fatal | none |

**Both `plotting.py` functions have no callers anywhere in the repository** — grep
across `.py` and `.ipynb` finds only their own definitions. So do not fix them
before deciding whether they should exist at all. Two options, and this is the
user's call:

1. **Delete them**, and `element_list` / `element_list_noAlCu` with them — they are
   the only consumers, so those two parameters are currently inert.
2. **Keep and fix**: replace the silent `continue` with the `04_02` note, and derive
   the "drop the top two" behaviour from the data rather than from a second
   hard-coded list.

Until one is chosen, nothing in the running pipeline depends on either, so nothing
is at risk.

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
