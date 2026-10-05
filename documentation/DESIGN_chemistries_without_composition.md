# Solid-state has no described CELL — and that stays open

Recorded 2026-09-15, deliberately and not as an oversight. **Rewritten 2026-10-05,
when sodium-ion stopped being part of the hole**: the battery project builds both
of its cells, `Na_ion_layered` and `Na_ion_prussian_white`, from the literature (a
scenario, not a bill of materials), and 04_04 counts them like any other chemistry.

Stage 04_04 reports the share of cars whose CELL it cannot describe — their
cathode, anode and electrolyte. That is solid-state alone. Its packaging it does
describe, and carries. This document says how big the remaining hole is, why it is
not filled, what was rejected, and what would close it.

---

## 1. How big it is

Solid-state exists only in S3, so the gap is exactly zero in S1 and S2. Share of
new cars whose cell is not described, S3, whole fleet, inflow, median with the
95 % band:

| year | share |
|---|---|
| 2040 | 8.9 % [8.0–9.8] |
| 2050 | 26.2 % [23.1–29.2] |
| 2070 | 47.9 % [40.9–53.9] |

Measured in a 2,000-draw sandbox run on 2026-10-05; the same three years measured
on 2026-09-15 at 200,000 draws gave 8.9, 26.2 and 48.0 %. The real number to quote
is the one in `04_04_4_uncovered_share.png` after each run.

It is a band and not a number because the chemistry shares are drawn — the size of
the hole is itself uncertain.

---

## 2. The packaging half — closed

The battery project models solid-state's packaging, not its cell: the cables, the
enclosure, the frame, the thermal conductor and both current collectors, at the
mass of the pack it is modelled on. Bipolar stacking removes the casing, the
separator and the per-cell terminals, so there are none. At 60 kWh, 2020, 400 V
that pack is 135.4 kg, all of it packaging.

04_04 carries it, so its steel, aluminium and copper reach the totals. In the
arrays the **active materials are zero, and zero means NOT DESCRIBED**.
`materials.battery_chemistry_active_material_unknown` is what stops that zero being
read as a fact: the cars are still reported as a gap.

---

## 3. Why the active materials are still not filled

Because nobody has published a composition for a bipolar solid-state cell that
survives scrutiny, and inventing one would be worse than the hole.

**Rejected: deriving it from a base chemistry.** Scaling NMC or LFP active material
onto a solid-state pack produces numbers that look like data and are not. The
proposal was made once and refused; it stays refused.

**Rejected: dropping its share.** A total that quietly fell as solid-state took the
large segments would read as falling material demand. It is the opposite — the
demand moves to materials this model cannot name.

**What is done instead:** the share is carried as an explicit gap, reported per
segment group and weighted by cars sold, and drawn on every figure that could
otherwise be misread.

---

## 3b. For recovery, this hole is not harmless

Solid-state's cell is a lithium battery. The battery project's own note says the
anode is **lithium metal**, its template is built on NMC high-nickel, and it runs
at 400–500 Wh/kg on the cell. Lithium is CRM and SRM, nickel and cobalt likewise.
Treating its cell as CRM-free would delete the material this whole analysis is
about, in exactly the scenario where it takes 70 % of the large segments.

So for a RECOVERY question the gap is the share above — 8.9 % in 2040, 48 % in 2070
— and not a rounding error. For a MASS question it is still the whole missing cell.

---

## 4. What this means for reading the results

- **Cell** materials under S3 — lithium, nickel, cobalt, manganese, phosphorus,
  graphite — are **lower bounds on demand**, not forecasts. The curves fall because
  the cell of those cars is not described.
- **Structural** materials — iron, aluminium, copper — are complete for every car in
  every scenario. Read them as totals.
- The **covered part is not damaged** by the gap. LFP, LMFP, NMC and the two sodium
  cells are complete for the cars that carry them.
- **S1 and S2 are unaffected:** the gap is an exact zero in every year and every
  draw, not a narrow band.
- **Do not sum across scenarios or compare S1 totals against S3 totals** as though
  the difference were a material saving. Much of it is a reporting hole.

---

## 5. What would close it

Data, not modelling. A CELL composition for bipolar solid-state — cathode, anode,
electrolyte — from a source that can be cited. Until that exists, the honest thing
is the gap, drawn next to every curve it affects.

The day one arrives: fill it in the battery project, take `solid_state` out of
`materials.battery_chemistry_active_material_unknown`, and the gap closes itself
without another line of code here.
