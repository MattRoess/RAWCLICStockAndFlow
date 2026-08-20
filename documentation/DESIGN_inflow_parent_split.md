# Carrying stage 02's inflow uncertainty into stage 03_02

**DO NOT CHANGE THE RULE IN THIS DOCUMENT WITHOUT RUNNING
`code/test_stage03_inflow.py`.** The rule is validated against real EU registration
statistics across every year for which data exists. The tests encode each claim made
here; change the rule and they fail, by design.

Settled 20 August 2026, after the defect below produced inflow figures wrong by a
factor of two for four days without anything detecting it.

---

## 1. The rule

Stage 03 owns the volume. Stage 02 contributes only its **deviation** from its own
mean.

```
share = det_value / parent_group_total          # share of the PARENT, not the drivetrain
dev   = draws_parent(y) − mean(draws_parent(y))
value = max(det_value + share × dev, 0)
```

Three properties, all asserted by the test file:

| property | measured |
|---|---|
| shares sum to 1 per (parent, year) | 1.000000000 across 39 pairs |
| pre-floor mean == deterministic value | worst deviation 3.3e-16 across 61 pairs |
| flooring only lifts the mean, never lowers | 0 pairs below; largest lift 10.35× |

**Additive, not the ratio form `draw / mean`.** The ratio form was rejected earlier in
this project: the residual passes through zero as Liquids phases out, so the relative
spread diverges — CV reached 1775%.

---

## 2. The defect this replaced

Three things were wrong at once.

### 2.1 The parent volume was never split between its children

`build_inflow_draws_by_drivetrain` hands **both** children of a coarse group the
parent's array — its docstring says so, and says *"the caller then applies its own
share to divide them."* The caller did not. It computed

```python
parent = det.groupby(["Drive Train", "year"])["value"].sum()   # drivetrain total!
share  = val / parent_total
```

`parent_total` grouped by **Drive Train** is a drivetrain's own total across
*segments* — the segment split. The parent→child split (Liquids → Diesel/Petrol,
Hybrid → HEV/PHEV) was simply absent, so Diesel's segments summed to the entire
Liquids volume, and so did Petrol's.

The variable being named `parent_total` while holding the drivetrain total is what
hid it.

**Measured against real EEA registrations:**

| | 2010 | 2014 | 2018 | 2019 |
|---|---|---|---|---|
| Diesel, old rule / real | 1.99× | 2.01× | **2.75×** | **3.04×** |
| Diesel, current rule / real | 1.12× | 1.14× | 1.04× | 1.04× |
| Petrol, old rule / real | 2.24× | 2.42× | 1.67× | 1.61× |
| Petrol, current rule / real | 1.12× | 1.14× | 1.07× | 1.09× |

In 2018 the old rule reported **15.2 million diesel cars** against an actual EU
diesel market of about 5.6 million — more than the entire European car market of
~15.6 million.

BEV was never affected: its parent is itself, so its share is 1. **Stage 04_02's
copper and element results are therefore untouched by this defect.**

### 2.2 The base year is identically zero

Stage 02's first modelled year is 2005, where `inflow = stock_target − survivors` and
the target *is* the initial stock. The residual is exactly `0.00e+00` for every
drivetrain and every draw — structural, not sampling noise. Writing that over the
deterministic 7.143 produced the spike to zero visible in every inflow figure.

Under the current rule `dev = 0` there, so the deterministic value survives with a
zero-width band. That is the honest answer: stage 02 carries no information about
that year.

### 2.3 The two stages do not share a level

Stage 02's MC mean sits **below** stage 03's deterministic value:

| year | 2006 | 2010 | 2015 | 2020 | 2030 |
|---|---|---|---|---|---|
| stage02 mean / stage03 deterministic | 0.954 | 0.939 | 0.967 | 0.982 | 0.868 |

This is **not** a defect. Inflow is a nonlinear function of the sampled lifetime, so
`E[inflow] ≠ inflow(E[λ])`. But it means *substituting* stage 02's level for stage
03's can never line up at the boundary, however the shares are computed — which is
why the rule carries the deviation instead of the level.

(The module docstring of `build_inflow_draws_by_drivetrain` claims "measured ratio
1.000 to three decimals". That does not hold against the current model.)

---

## 3. External validation

The EEA file on disk was itself confirmed against ACEA: ACEA reports **10.5 million**
EU registrations in 2023, the file gives **10.735 million** — 2% apart.

| check | source | result |
|---|---|---|
| Diesel 2010–2019 | EEA | old **2.21×** → current **1.11×** |
| Petrol 2010–2019 | EEA | old **2.20×** → current **1.13×** |
| BEV 2010–2019 | EEA | 1.26×, unaffected by the rule |
| Diesel 2018 share | ACEA — 35.9% of ~15.6 M | current 5.748 M vs ~5.6 M real |
| pre-2005 level | Trading Economics — EU average 13.08 M/yr (2003–2026) | model 13.12 M/yr, **0.5%** |

The pre-2005 result matters: the flat backcast had been suspected of being far too
low. It is not. It is within 0.5% of the long-run EU average. It looked wrong only
because it was being compared against a post-2005 band that the defect had doubled.

---

## 4. What is NOT fixed by this, and is still open

**These are real disagreements with the observed record. They are not caused by the
rule in this document and are not repaired by it.**

- **Hybrid volume is far too low.** The model's own deterministic HEV + PHEV inflow
  is **0.66×** real registrations in 2019 (0.553 M vs 0.840 M) and **0.36×** in 2023
  (1.319 M vs 3.695 M). This sits in stage 02's Hybrid volume or in stage 03_01's
  Hybrid → HEV/PHEV split.

  Note that fixing the parent split makes PHEV move from 1.56× to 0.41× of real —
  *further* from reality. That is not the fix failing. PHEV previously received the
  whole Hybrid volume, several times its own share, which partly cancelled the fact
  that the Hybrid total is too small. Two errors compensating is not agreement.

- **2020–2023 runs high.** Real diesel registrations collapsed from 4.87 M (2019) to
  1.37 M (2023) with COVID and the chip shortage. A stock-driven scenario model does
  not reproduce that. Model/real reaches 1.9–2.2× in those years even with the rule
  correct.

- **1975–2004 is flat.** Not a code defect: **no pre-2005 data exists in this
  project.** Every input starts in 2005 (`stock_dict` 2005, `df_exp_eu` 2005,
  `EEA_final_data.csv` 2010). `build_synthetic_pre_baseyear_inflows` solves for one
  constant because it has nothing to fit to. Supplying a real pre-2005 registrations
  series would make those years real instead of flat.

---

## 5. Why nothing caught the defect

Recorded because the conditions will recur.

The invariant existed in prose and was never executable. The builder's docstring said
the caller must divide the parent's volume; nothing checked that it did. Every stage
ran, every artifact was written, every figure rendered. It was found by a reader
comparing two charts by hand.

`test_stage03_inflow.py` exists so that never happens again. Test 1 — shares sum to 1
per parent — fails loudly against the old rule and is the single check that would
have caught this on day one.

The test file also caught an overstated claim in this work: the first version asserted
the composed mean equals the deterministic value, full stop. It does so only *before*
flooring. Flooring is deliberately nonlinear and lifts the mean by up to 10.35× in a
phase-out tail. The assertion is now in two parts, and the limitation is documented
rather than hidden.

---

## 6. Running the tests

```bash
.venv/bin/python code/test_stage03_inflow.py
```

About a minute, 4,000 draws, no pipeline run needed. Every defect these tests target
is structural — a factor of two, or an exact zero — so it shows at any draw count.

**Run it after any change to** the composition block in `03_02_adjustedflows.py`,
`build_inflow_draws_by_drivetrain`, or `inflow_uncertainty_parent_by_drv`.
