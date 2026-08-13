# Handover — 2026-08-13

Where the work stands, what is safe, what is not, and what to do next.

---

## 1. State of the repositories

### RAWCLICStockAndFlow — 5 files changed, all uncommitted

| file | change | tested |
|---|---|---|
| `src/params_schema.py` | reorganised into stage order; visualization section removed; every parameter rewritten for a non-specialist reader; 9 new BEV-electronics settings | yes — 511 parameters before and after, **zero changed** |
| `src/stockflow_model.py` | `run_stage02_cohort_monte_carlo` (shared orchestration); `sample_stock_target_draws`; `build_inflow_volume_multipliers` **(unused — see §4)** | yes — stage 02 byte-identical, 45,121 values |
| `code/02_stockdriven.py` | calls the shared orchestrator; figure renamed and retitled | yes — byte-identical |
| `code/03_01_flowdriven.py` | calls the shared orchestrator (this is the 20.8% bug fix); axis labels | yes — 15/15 arrays exact |
| `code/03_02_adjustedflows.py` | restored the removed by-drivetrain MC and its figures; exports BEV per-draw arrays for 04_02 | yes — at 300 draws |
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

**Nothing is committed.** A backup of the two electronics CSVs is at
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

### 4.1 The defect — stage 02's inflow uncertainty does not reach 03_02

Documented in full in `UNCERTAINTY_MAP.md`. Summary: total BEV inflow arrives in
03_02 with a coefficient of variation of 0.000001% against stage 02's 9.6%. Inflow
bands downstream are too narrow. Outflow and collected are unaffected.

**Not yet designed.** A ratio-based fix was proposed, tested, found unsound
(inflow is a residual; its relative spread reaches 1775% for a drivetrain being
phased out), and **not implemented**.

**Required before designing anything:** measure whether 03_02's inflow levels still
match stage 02's after 03_01 splits and rescales the coarse drivetrains.

**`build_inflow_volume_multipliers` in `src/stockflow_model.py` is dead code.**
Nothing calls it. Delete it or rewrite it once the design is settled.

### 4.2 Smaller open items

- `04_02` currently reports **total electronics mass**. Requested but not built:
  per-element results for the critical elements, and the domain breakdown
  (wiring / auxiliary motors / PCB / sensors) for the **collected** flow as well as
  inflow. The element data exists at `Composition/csv/detail_2025_2070.csv`
  (Year, Segment, Domain, Component_Type, Element) but only as mean and P2.5/P97.5,
  **not as draws** — so how element-split uncertainty should be treated is an open
  question, not a coding task.
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
