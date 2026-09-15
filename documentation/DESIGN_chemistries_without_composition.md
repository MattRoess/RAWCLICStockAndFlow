# Sodium-ion and solid-state have no described CELL — and that stays open

Recorded 2026-09-15, deliberately and not as an oversight. **Updated the same
day: the packaging half of this hole is closed.**

Stage 04_04 reports the share of cars whose CELL it cannot describe — their
cathode, anode and electrolyte. Under S3 that is most of the market. Their
packaging it does describe, and since 2026-09-15 it carries it. This document
says how big the remaining hole is, why it is not filled, what was rejected, and
what would close it.

---

## 1. How big it is

Share of cars whose cell is not described, median with the 95 % band, 2070.
Unchanged by carrying the packaging, and measured again after it: the same cars
are counted either way. Inflow is
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

## 2. The packaging half — CLOSED 2026-09-15

This used to say the gap in the model was bigger than the gap in the data. It
was, and it is not any more.

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

**04_04 could not read any of it.** The battery project wrote per-draw `.npy`
mass arrays for the seven lithium chemistries and none for these two, so
`src/battery_composition.py` raised and 04_04 dropped the whole car, structure
included. The model reported 69 % of S3's new cars as entirely unknown while
roughly half of each of those cars, by mass, sat modelled in a CSV it could not
read.

**Fixed, and run through on 2026-09-15.** `05_composition.py` now writes both levels for both chemistries
through the same pack rules as everything else, guarded by
`check_unknown_draws_match_workbook()`, which compares them against the CSV at
2020 — where the improvement factor is exactly 1 in every draw — correcting for
`build_unknown_rows` writing the packaging trust's MODE into the central column
while the draws carry the factor itself. Verified at all five anchors and both
voltages.

In those arrays the **active materials are zero, and zero means NOT DESCRIBED**.
`materials.battery_chemistry_active_material_unknown` is what stops that zero
being read as a fact: the same cars are still reported as a gap, at the same
share — 69.3 % [63.9–74.2] under S3 in 2070, before and after. Verified in the
output: 1,836 sodium rows under S1 and 2,295 active-material rows across both
chemistries, every one of them exactly zero.

**What it recovered**, inflow 2070, kt/year, median:

| | S1 | S2 | S3 |
|---|---|---|---|
| Fe | 1,630 (was 1,638) | **1,695** (was 1,204) | **1,225** (was 553) |
| Al | 886 (was 886) | **970** (was 601) | **882** (was 279) |
| Cu | 445 (was 446) | **345** (was 311) | **267** (was 145) |

Under S3 that is **+122 % iron, +216 % aluminium and +84 % copper** — material
that exists, that a recycler will see, and that was falling out of the totals
because the stage could not read the cars carrying it. S1 moves by a kilogram
in a million, as it must: those chemistries do not exist in it.

By 2070 the two contribute **18 % of S2's total inflow mass and 43 % of S3's**,
all of it packaging.

**And it changes the headline.** Under S3 the fleet's iron and aluminium demand
in 2070 is not far below today's — it moves away from lithium, it does not
disappear. Reading the old output, it did.

---

## 3. Why the active materials are still not filled

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

- **Cell** materials under S2 and S3 — lithium, nickel, cobalt, manganese,
  phosphorus, graphite — are **lower bounds on demand**, not forecasts. The
  curves fall because the cell of those cars is not described.
- **Structural** materials — iron, aluminium, copper — are complete for every
  car in every scenario, since 2026-09-15. Read them as totals. Under S3 in 2070
  that is 1,225 kt of iron entering and 1,374 kt collected, against 553 kt
  entering before the packaging was carried.
- The **covered part is not damaged** by the gap. LFP, LMFP and NMC flows are
  complete for the cars that carry them; the ratios built on them — secondary
  supply, collected over inflow — are sound within that part.
- S1 is unaffected: nothing new arrives in it, and its gap is an exact zero in
  every year and every draw, not a narrow band.
- **Do not sum across scenarios or compare S1 totals against S3 totals** as
  though the difference were a material saving. Much of it is a reporting hole.

---

## 5. What would close what is left

Data, not modelling. A CELL composition for sodium-ion and for bipolar
solid-state — cathode, anode, electrolyte — from a source that can be cited.
Until that exists, the honest thing is the gap, drawn next to every curve it
affects.

The day one arrives: fill it in the battery project, take the name out of
`materials.battery_chemistry_active_material_unknown`, and the gap closes itself
without another line of code here.
