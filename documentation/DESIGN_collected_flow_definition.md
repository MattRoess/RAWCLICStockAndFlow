# The collected flow: one definition, four implementations

What "collected" means, the defect that existed until 19 August 2026, why nothing
caught it, and the guards that now make the same class of defect fail loudly.

Reported by Yousef and colleagues, 19 August 2026.

---

## 0. The two questions, answered directly

Written so this section can be forwarded on its own.

> **Warum ist der „collected flow" auf zwei verschiedene Arten definiert?**
> In 3.1: `collected = (1 − share of exports − share of unknown whereabouts) × outflow`
> In 3.2: `collected = (1 − unknown whereabouts) × outflow`

**You are right, and it was a defect — not a modelling choice.** There is one correct
definition and 03_02 was not using it.

The correct one is 03_01's: collected, export and unknown **partition** the outflow.
A vehicle leaving the fleet is collected, exported, or unaccounted for; there is no
fourth destination, so the three must sum to the outflow exactly.

03_02's tracker path dropped the export share. It therefore **overstated collected by
2.3% for BEV and 16.3% for every other drivetrain**, and because export was still
reported separately alongside it, exported vehicles were counted twice — the three
sub-flows summed to 1.02 of the outflow for BEV and 1.08 for the others.

One correction to the question itself: 03_01 does not subtract, it **normalises** —
`collected / (collected + export + unknown)`. With the current parameters the three
already sum to 1 for every drivetrain, so normalising and subtracting give the same
number. Normalising is what keeps it right when a scenario changes one share without
restating the others.

Fixed 19 August 2026: 03_02 now calls the same shared function as 03_01, and the
sum-to-one invariant is asserted in three places so this cannot recur silently.
**The affected results — 04_01 and the material stages — still need re-running.**
Stage 04_02 (BEV electronics) was never affected; it reads the Monte Carlo path,
which always used the correct split.

> **Warum heisst es „override"? Warum nicht nur „export_share"?**

Because they are not the share — they are per-scenario **modifications** of it.

The shares themselves live in `StockFlowParams`:
`export_share_by_drv`, `unknown_whereabouts_share`, `collected_share_by_drv`.

`export_share_overrides` and `unknown_whereabouts_share_overrides` live on
`ScenarioSpec`, and are **sparse and additive**: an entry replaces the base value for
that one drivetrain, and a missing drivetrain means "keep the base value", not "zero".
That is what lets `losses_high` say only `{"BEV": 0.43}` instead of restating all nine
drivetrains.

So naming them `export_share` would be actively misleading — it would suggest a second
complete copy of the shares, when a scenario normally sets one or two entries. The
`_overrides` suffix is the accurate name and matches every other override field on
that class (`lifetime_change_by_drv`, and so on).

---

## 1. The definition

Every vehicle leaving the fleet goes to exactly one of three places. There is no
fourth destination, so the three shares are a **partition** of the survival outflow:

```
s_c, s_e, s_u   = (collected, export, unknown) / (collected + export + unknown)

out_collected(t) = out_survival(t) x s_c
out_export(t)    = out_survival(t) x s_e
out_unknown(t)   = out_survival(t) x s_u

out_collected + out_export + out_unknown == out_survival        (exactly)
```

The shares are **normalised**, not subtracted. With the base parameters they already
sum to 1 for every drivetrain, so normalising and subtracting give identical answers.
Normalising is what stays correct when a scenario override changes one share without
restating the others — which is exactly what `ScenarioSpec`'s sparse overrides do.

Base values:

| drivetrain | collected | export | unknown | sum |
|---|---|---|---|---|
| BEV | 0.880 | 0.020 | 0.100 | 1.000 |
| all others | 0.490 | 0.080 | 0.430 | 1.000 |

---

## 2. The defect

Four places in the codebase implemented this split. Three agreed. One did not.

| # | where | formula | BEV | others |
|---|---|---|---|---|
| 1 | `src/disaggregation.py` — 03_01 tracker | normalise | 0.8800 | 0.4900 |
| 2 | `src/cohort_flow_mc.py` — both MC engines | normalise | 0.8800 | 0.4900 |
| 3 | `src/flowdriven_model.py:1149` | `1 − u − e` | 0.8800 | 0.4900 |
| 4 | **`code/03_02_adjustedflows.py:903`** | **`1 − u`** | **0.9000** | **0.5700** |

Implementation 4 dropped the export share. Two consequences:

- **Collected was overstated** by 2.3% for BEV and **16.3% for every other
  drivetrain**.
- **The partition broke.** `outflow_exp_segments` was built right beside it and still
  reported export separately, so collected + export + unknown came to 1.02 for BEV
  and 1.08 for the others. Exported vehicles were counted twice — once inside
  collected, once as export.

**What was affected.** Implementation 4 fed `tracker_keyed`, hence 04_01 and the
material stages: their collected material masses were high by those percentages.

**What was not.** Stage 04_02 reads `per_year_collected` from the Monte Carlo engine
(implementation 2), so the BEV electronics copper and element results are unaffected.
This was checked, not assumed.

---

## 3. Why nothing caught it

Worth recording, because the same conditions will recur.

**The invariant was documented but never asserted.** `disaggregation.py`'s own
docstring states that the parts "sum back to `out_survival` exactly". That is a
testable claim, and nothing tested it. A defect that violates a stated invariant
should not require a human reading two files side by side.

**The broken implementation became the documentation for the correct one.** The
comment at `03_01_flowdriven.py:357` gave the formula as
`(1 - export_share - unknown_whereabouts_share)` and said plainly that it was
"inferred from how the equivalent computation is done ... in 03_02's
`disaggregated_new` construction" and that the author "could not confirm the exact
formula used inside `split_outflows_collected_unknown_export` itself without
`disaggregation.py`" — a file sitting in the same repository. A comment that
announces its own uncertainty is an open question, not background.

**Stage-agreement was verified on two flows and generalised to all.** The 03_01/03_02
reconciliation work compared **stock** and **inflow** — 15 arrays, 0/15 matching
before the fix, 15/15 after. True, and then described as "the stages agree". The
internal split of the outflow was never in the comparison set.

**Downstream work trusted the word without checking it.** Stage 04_02 computed
collected copper, neodymium and platinum "for recycling", with the composition side
verified to 1e-14 — while the definition of the flow it multiplied was never examined.
Precision on one side of a product does not transfer to the other.

**Searches were scoped to the current subtask, not to the concept.** `collected` was
searched twice during the element work. Both searches were shaped by the question in
hand — where do the BEV draws come from, what does 04_02 read — and both returned a
satisfying answer, so both stopped. Four implementations of the same split existed in
files that were open at the time. Searching for the *quantity* rather than for the
*current task* would have listed them side by side, which is all it took.

**The common thread is scope, not care.** Each piece of work verified deeply inside
what it was asked to change and treated everything upstream as given. "Correct" was
applied to the patch, not to the pipeline the patch fed. The reviewers found this in
a morning because they asked a different question — *what does this quantity mean in
each stage* — rather than *does this change work*.

The durable answer is not more diligence. It is that a stated invariant must be
executable: `disaggregation.py` claimed the parts sum to `out_survival` exactly, and
the moment that claim became an assertion, the defect had nowhere to hide.

---

## 4. What changed

1. **One implementation.** `03_02_adjustedflows.py` now calls the shared
   `compute_collected_export_unknown_shares` instead of its own formula.

2. **The invariant is asserted at three choke points**, all of which run once per
   drivetrain rather than per draw, so they cost nothing at 200,000 draws:

   | where | check |
   |---|---|
   | `disaggregation.compute_collected_export_unknown_shares` | normalised shares sum to 1 |
   | `cohort_flow_mc.normalize_three_shares` | normalised shares sum to 1 |
   | `03_02_adjustedflows.py`, before the tracker is built | collected + export + unknown == out_survival on the actual frames |

   Rows whose three inputs are all zero are excluded — they are deliberately forced to
   `(0, 0, 0)` and have no outflow to split.

3. **The stale comment** in `03_01_flowdriven.py` was replaced with the formula read
   from the implementation, and records what it used to say.

---

## 5. Status

| | |
|---|---|
| fix implemented | yes |
| guards implemented | yes |
| 03_01 re-run under the guards | yes — 200,000 draws, guard silent, collected mean 581.85 (95%: 489.57–690.85), numbers unchanged |
| **03_02 re-run** | **no — pending** |
| **04_01 re-run** | **no — pending** |

Until 03_02 runs, the fix is verified as correct in code and consistent with 03_01,
but **not** demonstrated on the tracker path, and 04_01's material masses are still
the old, overstated ones.

Expected on re-run: collected falls ~2.3% for BEV and ~16.3% for every other
drivetrain, and the partition check closes. If the partition check fires instead,
that is a second defect in the same area and should be investigated before anything
else is changed.
