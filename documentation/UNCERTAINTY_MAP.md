# Uncertainty map of the RAWCLIC stock-and-flow pipeline

**Status: 2026-08-14.** Every number in this document was measured on the real
data, not asserted. Where something is broken it says so plainly.

This document exists because a defect went unnoticed for a long time: stage 02
samples how many vehicles are sold, and stage 03_02 never hears about it. It was
found only when a chart showed an inflow line with no uncertainty band. A map of
where uncertainty enters, where it travels, and where it stops would have made it
visible immediately. That is what this is.

---

## 1. The one-paragraph summary

Uncertainty is **sampled on the stock target** in stage 02, and on **lifetimes and
outflow shares** in stages 02 and 03_02. It now reaches every stage. Until
2026-08-14 it stopped at the 03_01 → 03_02 boundary, because the artifact between
them (`flows_03`) is a plain table with one number per row and no draw dimension;
stage 02's per-draw inflow is now transferred alongside it. Results produced before
that date understate inflow uncertainty downstream and should be regenerated.

---

## 2. What is sampled, and where

| stage | quantity sampled | parameter | measured spread |
|---|---|---|---|
| 02 | stock target per drivetrain | `stock_target_relative_spread` | ±15% |
| 02 | total fleet size (shared) | `total_fleet_relative_spread` | ±2% |
| 02 | vehicle lifetime | `lifetime_scale_lambda_relative_spread` | ±15% |
| 02 | phase-out cap (if enabled) | `inflow_phaseout_max_share_triangular_by_drv` | 8/10/15% |
| 03_01 | collected / export / unknown shares | `*_relative_spread` | 5–20% |
| 03_02 | vehicle lifetime (re-sampled) | `lifetime_scale_lambda_relative_spread` | ±15% |
| 03_02 | segment mix within a drivetrain | `inflow_segment_share_spread` | per scenario |
| 04_01 | material composition | composition workbooks | per material |
| 04_02 | electronics per vehicle | the electronics study's own MC | per domain |

Nothing else is uncertain. In particular **inflow volume is never sampled
directly** — see section 4.

---

## 3. What crosses each boundary

```
   stage 02  ──────────────────────────►  stage 03_01
   stock target draws, lifetime draws        CARRIED CORRECTLY
   (fixed 2026-08-13; before that fix,
    03_01 dropped stock_target_draws and
    was up to 20.8% wrong on per-year inflow)

   stage 03_01  ─────────────────────────►  stage 03_02
   flows_03 (deterministic levels) PLUS        CARRIED, since 2026-08-14
   stage 02's per-draw inflow, transferred     (was: NOTHING CROSSED)
   directly as absolute values

   stage 03_02  ─────────────────────────►  04_01, 04_02
   per-draw arrays, correct                   CARRIED CORRECTLY
   for whatever 03_02 itself sampled
```

### The measurement that shows it

Total BEV inflow, summed over all twelve segments, in 2040:

| | coefficient of variation |
|---|---|
| stage 02's own draws | **9.6%** |
| what arrived in 03_02, before the fix | **0.000001%** |
| what arrives now | **matches stage 02** — see below |

Measured after the fix, total BEV inflow in 03_02 against stage 02's own draws:

| year | 03_02 now | stage 02 |
|---|---|---|
| 2030 | 3.97% | 4.0% |
| 2040 | 6.21% | 6.2% |
| 2050 | 14.13% | 14.2% |

The agreement to within 0.1 percentage point is the evidence that real draws were
transferred rather than an approximation fitted. Segment-mix spread was preserved
and in fact rises slightly (9.5–12.8% → 11.5–14.0%), because volume uncertainty now
compounds with mix uncertainty instead of being absent.

The segment shares inside 03_02 do vary — individual segments move by 9.5–12.8% —
but they are renormalised to sum to 1, so they only ever **redistribute** a fixed
total. One more A-segment car means one fewer of something else. That behaviour is
correct and should not be changed. What is missing is that the total itself never
moves.

---

## 4. Why inflow is a difficult quantity, and why the obvious fix is wrong

Inflow is not sampled anywhere. It is a **residual**:

```
inflow(t) = stock target(t) - vehicles that survived from earlier years
```

Two large uncertain numbers subtracted from one another. While a drivetrain is
growing this is well conditioned. While it is being phased out the residual
approaches zero and its *relative* spread diverges.

Measured, stage 02, 4,000 draws:

| drivetrain | year | mean inflow | std | CV |
|---|---|---|---|---|
| BEV | 2030 | 8.594 | 0.346 | 4.0% |
| BEV | 2050 | 14.682 | 2.086 | 14.2% |
| Liquids | 2030 | 3.244 | 1.674 | **51.6%** |
| Liquids | 2040 | 0.000 | 0.005 | **1775%** |
| Hybrid | 2040 | 0.066 | 0.112 | **168%** |

The absolute spread on Liquids in 2040 is 0.005 million vehicles — negligible. The
relative spread is 1775% purely because the denominator is nearly zero.

**Consequence.** Any scheme that carries the uncertainty as a *ratio* to stage
02's inflow is mathematically unsound for exactly those drivetrains being phased
out, which after 2035 is most of them. An earlier proposal in this session did
exactly that and would have injected violent swings into petrol and diesel that
stage 02 never simulated. It was caught by testing before implementation and was
not applied.

### What is well behaved

| quantity | CV range across 2030–2060 | usable as a carrier? |
|---|---|---|
| stock target | 0.8 – 8.6% | yes |
| total inflow, all drivetrains | 6 – 14% | yes |
| BEV share of total inflow | 0.1 – 12.6% | yes |
| per-drivetrain inflow | 4% … 1775% | **no** |

---

## 5. Closed — 2026-08-14

**The boundary now carries the uncertainty.** Stage 02's per-draw inflow is
transferred to stage 03_02 as absolute values, split to the fine drivetrains by
stage 03_01's own shares, and composed with 03_02's segment mix. Governed by
`propagate_stage02_inflow_uncertainty`, default on.

The design, the two rejected alternatives and the measurements behind the choice
are in `DESIGN_inflow_uncertainty_propagation.md`. In short: a relative multiplier
was rejected because inflow is a residual whose relative spread reaches 1775%
during a phase-out and whose value goes genuinely negative; a shared total-fleet
multiplier was rejected because it would have moved every drivetrain the same way
and silently undone the correlated mix.

One consequence is deliberate and visible in the numbers: each draw is floored at
zero individually, which lifts the Liquids mean by about 0.228 million vehicles
around 2035 relative to flooring a single average trajectory. That is the correct
Monte Carlo answer, not an artefact.

### What this changed downstream

Inflow bands widen in 03_02, 04_01 and 04_02. Any result produced before
2026-08-14 understates inflow uncertainty and should be regenerated.

---

## 6. What is still unknown

**Element-level results.** Stage 04_02 currently reports total electronics mass
per domain. Per-element results — copper above all — are wanted but not yet
possible as correct Monte Carlo: the electronics study publishes element data for
two years only (2025 and 2070) and as mean plus P2.5/P97.5, not as draws. Either
those models emit per-draw, year-resolved element arrays, or the element split has
to be treated as deterministic and said so. Not yet decided.

**Stage 02 remains the memory ceiling.** The propagation itself retains only
`n_parents x n_years x n_draws` in float32 — 254 MB at 200,000 draws, 1.3 GB at a
million. The stage-02 pass behind it peaks near 6.6 GB at 200,000 and scales
linearly, so that is what limits a larger run, not the transfer.

---

## 7. Fixed 2026-08-13, for the record

**Stage 03_01 was reproducing stage 02's Monte Carlo without passing
`stock_target_draws`.** Its per-year bands therefore carried none of the
stock-target uncertainty. Measured error up to **20.8%** on per-year inflow; 0 of
15 arrays matched stage 02 before the fix, 15 of 15 after.

The cause was structural: 03_01 rebuilt stage 02's call by hand instead of calling
the same function. It had already diverged twice before by the same mechanism
(missing hard-zero arguments, then missing the phase-out pass). The whole
orchestration now lives in `run_stage02_cohort_monte_carlo`, called by both stages,
so there is nothing left to keep in sync.

**The lesson generalises, and is the reason for section 3 of this document:** a
stage that re-derives another stage's numbers will drift, and the drift is silent
because the output still looks plausible. Read the other stage's actual draws, or
call the same function it called.
