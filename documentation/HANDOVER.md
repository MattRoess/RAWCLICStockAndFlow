# Handover — updated 2026-08-14

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

### RAWCLICVehicleElectronics — 1 file changed by this work

| file | change | tested |
|---|---|---|
| `tools/mc_composition.py` | persists raw per-draw arrays to `Composition/draws/*.npy` | yes — full 200,000-draw run, **both CSVs regenerate byte-identical** |

`Data/30_BEV_electronics_composition.csv` and `docs/01_USER_GUIDE.md` also show as
modified. **Those are not from this work** — they were already modified at 13:38
and 11:58, before this session, and `mc_composition.py` writes neither.

Committed on branch `mc-correctness-and-bev-electronics`. A backup of the two electronics CSVs is at
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

Everything in 03_02 and 04_02 was tested at **300 draws**. The first full-scale
run has not been done.

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

### 4.2 Smaller open items

- **`04_02` element and component resolution — the next piece of work.** It
  currently reports total electronics mass per domain. Wanted: per-element results,
  copper above all, and the domain breakdown (wiring / auxiliary motors / PCB /
  sensors) for the **collected** flow as well as inflow.

  What exists: `Composition/csv/detail_2025_2070.csv` gives Year, Segment, Domain,
  Component_Type, Element. Copper appears in all four domains — one AB vehicle in
  2070 carries 21.6 kg of it, 86% in wiring. Element coverage differs by domain:
  Motors 34 elements, Sensors 31, PCB 10, Wiring 1 (copper only).

  Two constraints block correct Monte Carlo at element level: the file covers **two
  years only** (2025 and 2070), and it holds mean plus P2.5/P97.5, **not draws**.
  Either the electronics models emit per-draw, year-resolved element arrays, or the
  element split is treated as deterministic and said so plainly. Not yet decided.
- `KG_PER_TONNE` in `04_01_carcomposition.py` is now unused.
- Two functions vanished from `04_01` on 2026-07-09 (`plot_material_mass_by_year`,
  `quantify_and_aggregate`). Successors appear to exist; not confirmed.
- `data/processed/intermediate/` is **71 GB**, including duplicated files with a
  trailing " 2" in the name (roughly 25 GB of accidental copies).
- Scenario-comparison figures need two or more entries in `scenarios_to_run`;
  it is currently `("BAU",)`.

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
| `PARAMETER_REFERENCE.md` | all 137 parameters, **generated** from `src/params_schema.py` |
| `MODEL_DESCRIPTION.md` | converted from the `.docx`, plus section 7 for everything since 13 July |
| `UNCERTAINTY_MAP.md` | where uncertainty enters, travels and stops |
| `DESIGN_inflow_uncertainty_propagation.md` | the propagation design and two rejected alternatives |

Regenerate the parameter reference after any parameter change:

```bash
.venv/bin/python code/generate_parameter_reference.py
```

It reports how many parameters have no explanation in the code. That count is **0**
and should stay there — it is the check that stops the reference silently rotting the
way its predecessor did.
