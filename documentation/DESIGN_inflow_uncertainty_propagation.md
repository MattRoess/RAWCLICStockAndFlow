# Design — carrying stage 02's inflow uncertainty into stage 03_02

**Status: 2026-08-14. Design settled on measurement. Implementation follows.**

Companion to `UNCERTAINTY_MAP.md`, which states the defect. This document records
the measurements that decided the design, including the two approaches that were
rejected and why, so neither is proposed again.

---

## 1. The defect being fixed

Stage 02 samples how large the vehicle fleet is and therefore how many vehicles are
bought each year. Stage 03_02 never receives that. Measured on total BEV inflow in
2040:

| | coefficient of variation |
|---|---|
| stage 02's own draws | **9.6%** |
| what arrives in 03_02 | **0.000001%** |

Every downstream inflow band — 03_02, 04_01, 04_02 — is therefore too narrow.
Outflow and collected are unaffected: 03_02 samples lifetime itself.

---

## 2. Measurement A0 — the boundary is clean

The question that blocked the design: after stage 03_01 splits stage 02's coarse
drivetrains into finer ones, are the inflow LEVELS still the same? If 03_01
rescaled, any transplanted uncertainty would need a correction of unknown size.

Measured, EUR, millions of vehicles per year:

| year | drivetrain | stage 02 | flows_03 | ratio |
|---|---|---|---|---|
| 2025 | BEV | 3.004 | 3.004 | 1.000 |
| 2030 | BEV | 8.608 | 8.608 | 1.000 |
| 2030 | Liquids | 3.766 | 3.766 | 1.000 |
| 2035 | Liquids | −0.264 | −0.264 | 1.000 |
| 2040 | BEV | 16.040 | 16.040 | 1.000 |
| 2040 | Liquids | −4.655 | −4.655 | 1.000 |

**The difference is 0.000 everywhere.** Stage 03_01 passes stage 02's inflow
through untouched — it splits Liquids into Petrol/Diesel and Hybrid into HEV/PHEV
by share, but does not rescale the total.

Totals across all drivetrains differ by 0.4–1.1%, solely because Gases and FCEV
exist in stage 02 and are dropped downstream. That is a scope difference, not a
rescaling.

**Consequence: no correction factor is needed.** Stage 02's per-draw inflow can be
transplanted directly onto the same quantity in the same units.

---

## 3. Rejected: a relative multiplier

The first proposal was to carry the uncertainty as a ratio to stage 02's mean
inflow, and multiply 03_02's deterministic inflow by it. **This is unsound and was
never implemented.**

Inflow is not sampled anywhere. It is a residual:

```
inflow(t) = stock target(t) − vehicles surviving from earlier years
```

Two large uncertain numbers subtracted. While a drivetrain grows this is well
conditioned; while it is phased out the residual approaches zero and its relative
spread diverges:

| drivetrain | year | mean | std | CV |
|---|---|---|---|---|
| BEV | 2030 | 8.594 | 0.346 | 4.0% |
| BEV | 2050 | 14.682 | 2.086 | 14.2% |
| Liquids | 2030 | 3.244 | 1.674 | 51.6% |
| Liquids | 2040 | 0.000 | 0.005 | **1775%** |
| Hybrid | 2040 | 0.066 | 0.112 | **168%** |

Liquids' absolute spread in 2040 is 0.005 million vehicles — negligible. The 1775%
is entirely an artefact of dividing by a near-zero mean. A ratio would have
injected violent swings into petrol and diesel that stage 02 never simulated.

A0 made it worse still: `flows_03` carries the **raw residual including negative
values** (Liquids is −4.655 in 2040). A ratio against a negative denominator has no
meaning at all.

---

## 4. Rejected: a total-fleet multiplier

Total inflow across drivetrains is well behaved (CV 6–14%), so a single
total-fleet multiplier applied to every drivetrain would be numerically stable.

Rejected because it discards the part of stage 02 that matters most here: the
**correlated drivetrain mix**. In stage 02 a draw that buys more BEVs buys fewer
of something else. A single shared multiplier moves every drivetrain in the same
direction, which is the opposite behaviour, and would silently undo the
substitution effect that was deliberately built in.

---

## 5. Adopted: absolute per-draw inflow

Carry stage 02's **absolute per-draw inflow**, per coarse drivetrain, per year, and
split it to the fine drivetrains using stage 03_01's own shares.

**Why absolute works where relative fails.** The absolute spread stays finite and
meaningful everywhere, including through zero and into negative territory:
Liquids' 0.005 at 2040 is simply a small number, correctly small. There is no
denominator to blow up.

**Why no correction is needed.** A0 showed the two stages hold the identical
quantity in identical units, so a draw can be transplanted as-is.

**How coarse becomes fine.** Petrol and Diesel both come from Liquids; HEV and
PHEV from Hybrid. Each fine drivetrain receives its parent's per-draw value
multiplied by 03_01's own deterministic share of that parent, so:

- the **volume** uncertainty comes from stage 02, where it is sampled
- the **split** between petrol and diesel remains entirely stage 03_01's, with its
  own uncertainty and time development, untouched
- the **segment mix** inside a drivetrain remains entirely 03_02's, renormalised to
  sum to one, untouched

Three effects compose; none replaces another.

---

## 6. Flooring: the one genuine modelling decision

`flows_03` carries negative inflow where a fleet is shrinking faster than its cars
are scrapped. Today the negative value is floored to zero once, on the
deterministic trajectory. With draws, each draw must be floored individually.

These are not the same, because flooring is nonlinear: `mean(max(x,0)) ≥
max(mean(x),0)`. Measured, 4,000 draws, millions per year:

| drivetrain | year | deterministic (floored once) | per-draw floored mean | difference |
|---|---|---|---|---|
| BEV | 2030–2045 | — | — | **0.000** |
| Liquids | 2030 | 3.243 | 3.244 | +0.001 |
| **Liquids** | **2035** | **0.000** | **0.228** | **+0.228** |
| Liquids | 2040 | 0.000 | 0.000 | 0.000 |
| **Hybrid** | **2040** | **0.000** | **0.066** | **+0.066** |

**Per-draw flooring is adopted, and it is the correct answer rather than a side
effect.** In a Monte Carlo draw where the fleet target lands higher, liquid-fuel
inflow genuinely is still positive that year; flooring each draw preserves that,
and flooring the mean discards it. The deterministic pipeline could not represent
it because it has only one trajectory.

The effect appears only in transition years where the residual straddles zero, and
peaks at 0.228 million vehicles against a total of about 14 million per year —
1.6%. It is reported rather than hidden: the verification step records the change
in the mean, not only the change in the band.

---

## 7. What must be true afterwards

The implementation is not accepted unless all four hold.

1. Total BEV inflow CV in 03_02 rises from 0.000001% to **≈9.6%**, matching stage 02.
2. Segment-mix spread stays at **9.5–12.8%** — the existing segment work preserved,
   not replaced.
3. Petrol and Diesel move together (they share a parent); BEV moves against them
   (the correlated mix survives the crossing).
4. With propagation switched off, 03_02 reproduces today's numbers **byte-identically**.

---

## 8. Notes for whoever maintains this

`build_inflow_volume_multipliers` in `src/stockflow_model.py` implements the
**rejected** ratio approach of section 3. It is called by nothing and is deleted as
part of this work. If it reappears, this document explains why it should not.

The rule this follows is the one that fixed stage 03_01: a stage reads another
stage's actual draws, or calls the same function that produced them. It never
re-derives. The propagation added here is a transfer of real draws, not a
reconstruction.
