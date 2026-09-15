# Sodium-ion and solid-state carry no composition — and that stays open

Recorded 2026-09-15, deliberately and not as an oversight.

Stage 04_04 reports the share of cars whose battery it cannot describe. Under S3
that is most of the market. This document says exactly how big the hole is, why
it is not filled, what was rejected, and what would close it.

---

## 1. How big it is

Share of cars with no composition, median with the 95 % band, 2070. Inflow is
new cars; the collected flow lags it because the cars being scrapped were built
before these chemistries arrived.

| | inflow | collected |
|---|---|---|
| **S2** small | 68.0 % [61.0–73.8] | 60.7 % [55.8–65.1] |
| S2 medium | 24.0 % [18.9–29.4] | 20.5 % [17.4–23.7] |
| S2 large | 10.0 % [7.6–12.8] | 8.2 % [6.8–9.7] |
| **S2 fleet** | **34.4 % [28.8–39.8]** | 30.0 % [26.3–33.7] |
| **S3** small | 80.0 % [75.9–83.6] | 68.5 % [64.8–71.9] |
| S3 medium | 64.0 % [57.8–69.6] | 46.2 % [42.2–50.2] |
| S3 large | 75.0 % [69.1–79.8] | 53.3 % [48.7–57.6] |
| **S3 fleet** | **69.3 % [63.9–74.2]** | 52.8 % [49.0–56.6] |

The fleet gap passes 10 % in **2031** under both scenarios, 25 % in 2044 (S2) and
2038 (S3), and 50 % in **2050** under S3. It never reaches 50 % under S2.

It is a band and not a number because the chemistry shares are drawn — the size
of the hole is itself uncertain.

---

## 2. The gap in the MODEL is bigger than the gap in the DATA

This is the part worth acting on.

The battery project does model these two chemistries. What it leaves empty is the
**active material**, not the whole pack. At 60 kWh, 400 V, 2030:

| chemistry | pack | packaging and structure | active material |
|---|---|---|---|
| LFP | 408.9 kg | 206.7 kg | 202.2 kg |
| sodium-ion | 209.5 kg | 209.5 kg | **0** |
| solid-state | 130.5 kg | 130.5 kg | **0** |

A sodium pack's casing, separator, cables, terminals, module enclosure, support
frame, thermal conductor and both current collectors are all there — 209.5 kg,
within 3 kg of LFP's own structure — scaled by cell mass and carrying a drawn
packaging-trust factor (0.9/1.0/1.3 for sodium, 0.7/0.8/1.1 for solid-state)
that says how much that scaling is believed.

**04_04 cannot read any of it.** The battery project writes per-draw `.npy` mass
arrays for the seven lithium chemistries — twenty arrays each — and **none for
these two**. `src/battery_composition.py` raises rather than hand back zeros, so
04_04 drops the whole car, structure included.

So the model reports 69 % of S3's new cars as entirely unknown when roughly half
of each of those cars, by mass, is modelled and sitting in a CSV. Their iron,
aluminium and copper are missing from every total in this stage for no reason
other than the export.

**What would close this half:** `05_composition.py` writes
`<chemistry>_<capacity>kWh_<voltage>V_mass_draws.npy` and its component twin for
Na_ion and solid_state as well. 04_04 then carries their structure and reports
only the **active materials** as the gap. That is a change in the battery
project, not here, and it needs the draws to go through the same pack rules and
the same `check_draws_match_workbook()` guard as the rest.

---

## 3. Why the active materials are not filled

Because nobody has published a composition for either that survives scrutiny,
and inventing one would be worse than the hole.

**Settled 2026-09-10 in the battery project, and not reopened since.** Sodium-ion
and solid-state are marked unknown there: packaging only, active materials left
empty, with the packaging trust drawn rather than assumed.

**Rejected: deriving them from a base chemistry.** Scaling NMC or LFP active
material onto a sodium pack produces numbers that look like data and are not.
The proposal was made once and refused; it stays refused.

**Rejected: dropping their share.** A total that quietly fell as sodium took the
small segments would read as falling material demand. It is the opposite — the
demand moves to materials this model cannot name.

**What is done instead:** the share is carried as an explicit gap, reported per
segment group and weighted by cars sold, and drawn on every figure that could
otherwise be misread.

---

## 4. What this means for reading the results

- Element flows under S2 and S3 are **lower bounds on demand**, not forecasts.
  The curves fall because cars leave the picture.
- The **covered part is not damaged** by the gap. LFP, LMFP and NMC flows are
  complete for the cars that carry them; the ratios built on them — secondary
  supply, collected over inflow — are sound within that part.
- S1 is unaffected: nothing new arrives in it, and its gap is an exact zero in
  every year and every draw, not a narrow band.
- **Do not sum across scenarios or compare S1 totals against S3 totals** as
  though the difference were a material saving. Much of it is a reporting hole.

---

## 5. What would close it

Data, not modelling. A composition for sodium-ion and for bipolar solid-state at
the workbook's own levels — component, material, element — from a source that can
be cited. Until that exists, the honest thing is the gap, drawn next to every
curve it affects.

Half of it, the structure, could be closed now by exporting draws that already
exist. See §2.
