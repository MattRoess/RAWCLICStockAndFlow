# Carving HEV out of Liquids (option B)

**Implemented 2026-08-20.** Governed by `params.disaggregation.hev_carved_from_liquids` (default True) and
`hev_share_phaseout_end_year` (default 2035). Guarded by `code/test_stage03_inflow.py`
(13 tests, 4 of them on this change). Set the flag False only to reproduce a result
published before this date.

This document is written so the change can be built without rediscovering any of it.
Every number below was measured, and the rule is specified precisely enough to code
from directly.

---

## 1. Why this is needed

The REMIND files contain exactly five LDV technologies:

```
Liquids · Hybrid electric · Gases · FCEV · BEV
```

**None of them is a non-plug-in hybrid.** In REMIND's taxonomy `Hybrid electric` is
the *plug-in* hybrid; full and mild hybrids sit inside `Liquids`.
`src/data_prep.py:241` renames `Hybrid electric` → `Hybrid`, and stage 03_01 then
splits that into HEV and PHEV — **inventing an HEV series out of plug-in-hybrid
volume, while the real HEVs remain counted inside Liquids.**

ACEA's HEV is 25.8% of the EU market, 2.71 M cars in 2023. Those cars have nowhere to
live in the model except inside Liquids, and they are simultaneously being
double-represented by a fabricated HEV series.

### What this is NOT

An earlier reading of this — recorded in `HANDOVER.md` §4.1c and withdrawn the same
day — claimed the model was *missing* hybrid volume (0.36× reality in 2023). That
comparison was invalid: `EEA_final_data.csv` assigns `HEV` from fuel types PETROL /
DIESEL / E85, so its HEV is dominated by 48 V mild hybrids, which are petrol and
diesel cars. Reclassify them back to Liquids and the model is not short of hybrids at
all (1.02–1.57×), and its Liquids matches reality at **1.03–1.09× for 2015–2019**.

There is no missing volume. There is a category error.

---

## 2. The rule

Let `s(t)` be the HEV share of the Liquids family.

```
t <  2000          s = 0                        introduction year, params.disaggregation
                                                .introduction_year_by_drv["HEV"]
2000 ≤ t < 2019    s = 0.044 × (t−2000)/19      gradual increase to the first observation
2019 ≤ t ≤ 2023    s = the OBSERVED EEA share
2023 <  t < 2035   s = 0.361 × (2035−t)/12      ramps to zero with the Liquids phase-out
t ≥  2035          s = 0
```

Observed shares, measured from `data/raw/EEA_final_data.csv`:

| year | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|
| s(t) | 4.4% | 13.4% | 24.3% | 31.6% | **36.1%** |

Then, per year:

```
HEV(t)    = s(t) × Liquids(t)
Petrol(t) + Diesel(t) = (1 − s(t)) × Liquids(t)     split by the EXISTING 03_01 logic
PHEV(t)   = the whole REMIND Hybrid volume, NOT split
```

**2035 is not arbitrary**: the model's Liquids inflow is last positive in 2034
(0.2374 M) and negative thereafter. Where the residual is negative, sales are zero
per `stock_flow.negative_inflow_policy = "report_only"` — the negative is recorded
for inspection, not simulated. The HEV ramp reaching zero at 2035 coincides with
that, so no special case is needed.

---

## 3. Proof

Replayed against the real deterministic inflow from `tracker_keyed_BAU`. Script:
`scratchpad/prove_hev_carve.py` (temporary — the specification above is sufficient to
rebuild it).

### Conservation

```
Petrol + Diesel + HEV == Liquids     max error 1.78e-15
```

### Against real EEA registrations

| year | HEV real | HEV new | ratio | Petrol ratio | Diesel ratio |
|---|---|---|---|---|---|
| 2019 | 0.645 | 0.662 | **1.03** | 1.03 | 1.03 |
| 2020 | 1.363 | 1.905 | 1.40 | 1.40 | 1.40 |
| 2021 | 1.888 | 3.173 | 1.68 | 1.68 | 1.68 |
| 2022 | 2.218 | 3.737 | 1.68 | 1.68 | 1.68 |
| 2023 | 2.855 | 3.865 | 1.35 | 1.35 | 1.35 |

**The decisive detail: Petrol, Diesel and HEV carry the identical ratio in every
year.** The three-way split is therefore exact, and the entire residual is the
pre-existing Liquids overshoot — the COVID and chip-shortage collapse the model does
not reproduce. The carve introduces no error of its own. At 2019, before that
collapse, HEV lands at **1.03×**.

### PHEV, taking REMIND's Hybrid unsplit

| year | real | new | current split |
|---|---|---|---|
| 2019 | 0.195 | 0.553 → 2.84× | 0.64× |
| 2020 | 0.637 | 0.724 → 1.14× | 0.38× |
| 2021 | 0.906 | 0.927 → **1.02×** | 0.33× |
| 2022 | 0.900 | 1.128 → 1.25× | 0.36× |
| 2023 | 0.840 | 1.320 → 1.57× | 0.35× |

### Trajectory

| year | s(t) | Liquids | Petrol | Diesel | HEV | PHEV |
|---|---|---|---|---|---|---|
| 2000 | 0.0% | 12.992 | 6.106 | 6.886 | 0.000 | 0.132 |
| 2010 | 2.3% | 12.724 | 5.842 | 6.589 | 0.294 | 0.191 |
| 2019 | 4.4% | 15.099 | 9.443 | 4.993 | 0.662 | 0.553 |
| 2023 | 36.1% | 10.718 | 4.999 | 1.854 | **3.865** | 1.320 |
| 2030 | 15.0% | 3.766 | 2.334 | 0.866 | 0.566 | 1.602 |
| 2035 | 0.0% | 0.000 | 0.000 | 0.000 | 0.000 | 0.533 |

### External validation of the input data

Every category in the EEA file agrees with ACEA's published 2023 shares on a 10.5 M
market: Petrol 3.71 vs 3.693, HEV 2.71 vs 2.855, BEV 1.53 vs 1.669, Diesel 1.43 vs
1.370, PHEV 0.81 vs 0.840 M.

---

## 4. What was built

| file | change |
|---|---|
| `src/params_schema.py` | `hev_carved_from_liquids`, `hev_share_phaseout_end_year` (419 parameters) |
| `src/disaggregation.py` | `build_liquids_three_way_split`, with a sum-to-one guard; the four splitters take a `children` list instead of hardcoding `["Diesel","Petrol"]` / `["HEV","PHEV"]` |
| `code/03_01_flowdriven.py` | branches on the flag; 2005 starting stock split from the same tables, with a conservation guard |
| `code/test_stage03_inflow.py` | 4 new tests |

The observed shares are read from the EEA file at runtime rather than written into
code, so they follow the data. **They are read from the country-scoped frame** (EU27
plus Norway and Iceland), not the raw 30-country file — the 2019 share is 4.569%
scoped against 4.385% unscoped, and an early version of the test compared against the
wrong one and failed.

Measured after implementation: HEV **1.06×** real at 2019, PHEV **1.02×** at 2021,
s(t) 0% at 2000 rising to 4.57% (2019), 36.84% (2023), 0% at 2035, conservation
1.11e-16 across 121 years.

**Pre-2005 inflow does not follow s(t)** — it is the constant backcast solved from the
2005 stock (`build_synthetic_pre_baseyear_inflows`), which is existing, documented
behaviour and unchanged by this work.

### Consequences to expect

- **The drivetrain taxonomy changes from stage 03 onward.** `HEV` stops being a slice
  of `Hybrid` and becomes a slice of `Liquids`; `PHEV` becomes the whole of `Hybrid`.
  Everything downstream that keys on drivetrain is affected, including 04_01 and the
  material stages.
- **A full pipeline re-run and re-validation is required.** 03_02 is the ~56 min run.
- Anything published with the old HEV/PHEV series should be regenerated.

### Known limits, to state rather than hide

- **HEV is 0.000 before 2019 in the source data** — a reporting artefact, not history.
  The back-extrapolation from the introduction year is an assumption chosen
  deliberately; hybrids did exist before 2019.
- **The post-2023 share is an assumption.** REMIND has no view — it has no HEV
  category at all, which is the whole reason for this change.
- **2020–2023 will still run 1.35–1.68× high**, because the model's Liquids total
  does not reproduce the COVID collapse. That is unrelated to this change and is not
  fixed by it.
