# Element resolution in stage 04_02

How stage 04_02 gets from "kilotonnes of BEV electronics" to "kilotonnes of copper,
neodymium and platinum", what had to be fixed to make that correct, and the two
defects the work uncovered.

Written 14 August 2026.

---

## 1. What this is for

Recycling questions are element questions. "How much electronics leaves the fleet"
does not tell you how much copper a smelter sees, or whether the neodymium in
end-of-life motors is worth recovering. Stage 04_02 previously reported mass per
domain — wiring, motors, boards, sensors — and copper only for wiring, because
wiring was the one domain whose model reports copper directly.

This document covers the extension to elements across all four domains.

---

## 2. The shape of the calculation

For every element, every year, every draw:

```
element_mass = vehicles x material_per_vehicle x element_content
```

Segments are summed on **raw draws**, before any percentile. Percentile-of-sum is
the honest total; sum-of-percentiles assumes every segment hits its extreme in the
same world.

The fleet Monte Carlo and the electronics Monte Carlo are **independent samples**, so
draw *i* of one is paired with draw *i* of the other. That is valid — any pairing of
two independent streams is a valid joint sample — and it is stated rather than
hidden. It does mean a genuine correlation between "large vehicle" and "heavy
harness" is not captured, which makes the band slightly narrow rather than wide.

### Why sensors are handled differently

Three domains supply a per-draw **fraction** of domain mass. Sensors supply an
absolute per-draw **mass**. The asymmetry is deliberate and section 4 explains why.

| domain | element content | basis |
|---|---|---|
| Wiring | Cu = 1.0 exactly | the model reports `Cu (kg)` and nothing else |
| Motors | fraction of whole-motor mass | `ElectricMotorElementMC` |
| PCB | fraction of board metal mass | `PCBElementMC` |
| Sensors | absolute mg per vehicle | `SensorElementsMC` |

---

## 3. Defect one — the motor denominator

**Found before it produced a published number. It would not have been visible in any
output.**

`ElectricMotorElementMC` resolves **four** material streams into elements: cast Fe
steel, copper, electrical steel, NdFeB. A motor in `ElectricMotorMC` is made of
**six** — those four plus **Aluminium and Plastic**, which have no elemental
breakdown.

The first version of the combined per-segment fractions divided element mass by the
sum of the four streams. Those fractions summed to 1.0000 on every segment and
looked completely healthy. They summed to 1 over the **wrong total**: "fraction of
the elementally-resolved part of a motor", not "fraction of a motor".

Stage 04_02 multiplies them by `mc_composition`'s Motors mass, which is the whole
motor. Mixing the two denominators inflates every motor element by
`1 / (1 - unresolved share)`:

| segment | Al + Plastic | inflation if unfixed |
|---|---|---|
| AB | 10.32% | +11.5% |
| CD | 17.67% | +21.5% |
| EF | 22.18% | +28.5% |

Neodymium, dysprosium and motor copper would all have been high by that much.

**The fix.** Divide by `ElectricMotorMC`'s own per-draw `total_mass_kg`. Bulk
aluminium is folded into the `Al` element — aluminium metal is the element
aluminium. Plastic is carried under its own name. Row *i* is the same simulated
motor in both models (`ElectricMotorElementMC` reads `ElectricMotorMC`'s draws
rather than resampling them), so the division is exact per draw.

**Verification.** Predictions were written down *before* the re-run: motor copper
must land on the copper material share, 2.016/13.044 = 0.15455 for AB.

| | predicted | measured | error |
|---|---|---|---|
| AB Cu | 0.15455 | 0.15467 | 0.079% |
| CD Cu | 0.14885 | 0.14904 | 0.131% |
| EF Cu | 0.14645 | 0.14667 | 0.148% |

The small excess is real and in the right direction — copper also occurs as a trace
in NdFeB, so total Cu slightly exceeds the copper stream's share. `Al` came out at
3.2 / 10.4 / 14.7% and `Plastic` at 7.2 / 7.4 / 7.7%, both matching the material
table. Σfraction = 1.00000000 per draw. **Nd fell from 0.03192 to 0.02861.**

### The residual, and a tolerance set the wrong way round

The reconciliation check first ran with a tolerance of 1e-9, justified in a comment
claiming "nothing but float64 rounding sits between the two sides, so the measured
worst case is ~1e-16". That was **asserted, not measured**, and it was wrong: the
check fired on 199,989 of 200,000 draws.

The real residual is ~5e-5 of motor mass and has a cause: the **copper stream's named
elements sum to 99.9916% of that stream**, because the specification lists Cu plus
impurities in ppm and does not quite close. The other three streams close to ~3e-8.

It is carried as an explicit `Unspecified` entry rather than absorbed into a loosened
tolerance. Fractions then sum to exactly 1 by construction, and the unaccounted part
is visible rather than hidden. `MOTOR_FRAC_TOL` now bounds how large that entry may
get: 1e-3, which is 20x the measured value and still 30x below the smallest resolved
material (aluminium in AB, 3.1%), so a whole missing stream cannot hide inside it.

---

## 4. Defect two — sensors are estimated from modes

`mc_composition` builds its Sensors domain mass as

```
sum over sensor types of  count x sum of each element's MODE mg
```

The mode. Every other domain uses draws or means: motors read `ElectricMotorMC`'s
per-draw samples, PCB uses `Mean_g`, wiring reports copper directly. **Sensors are
the only domain estimated from modes.**

That is the wrong estimator for a mass balance. Expected total mass is
`E[Σx] = Σ E[x]`, and the mean of a `triangular(min, mode, max)` is
`(min + mode + max)/3`, not the mode. These composition tables are strongly
right-skewed, so summing modes understates expected mass. Measured on the source
file:

```
mass convention (triangular mean vs mode)   1.613
count convention (uniform mean vs SN counts) 1.073
                                    product  1.73
```

1.73 is exactly the discrepancy observed between `SensorElementsMC`'s totals and
`mc_composition`'s Sensors series — the gap is fully explained, with nothing left
over.

This also dissolves an apparent "mix inconsistency" seen earlier (Cu +23%, Al −44%,
Mn +69% between the two): it is the same mode-versus-mean effect hitting each element
differently according to its own skew. `SensorElementsMC`'s mix is the mean-based
one, which is the correct one for mass accounting.

**The fix.** Take sensor element mass from `SensorElementsMC`'s per-draw element
masses at their own level, and take only the **shape** of `mc_composition`'s sensor
trajectory — its year-to-year profile normalised to 1.0 at 2025 — to carry them
through time. A fraction multiplied by the mode-based mass would have inherited the
1.73x understatement invisibly, because the bias is in the estimator, not in the
sampling, and no amount of Monte Carlo reveals it.

**Not fixed at source.** `mc_composition`'s Sensors series is still mode-based, so
the *domain mass* figures still understate sensors by ~1.73x. Fixing it there would
regenerate `Composition/csv`, which has been validated byte-identically against the
published study. That is a deliberate deferral, not an oversight.

### What a sensor is made of is only half known

The named elements account for **48.5%** of physical sensor weight in aggregate
(median 40% per type; as little as 4% for a rain/light sensor, 81% for a rotor
position sensor). The rest is plastic, epoxy and glass that the source table never
resolves elementally.

So `sensors_*_fractions.npy` sums to 1 over *the elements this model resolves*, which
is not all of a sensor. It must not be read as "a sensor is 36% copper". The element
**masses** are unaffected by this — they are absolute.

Separately, O, N, C and P are excluded from `SensorElementsMC` by explicit user
instruction and are not re-added here.

---

## 5. Which elements are reported

**Whatever the draws carry.** `materials.bev_electronics_elements` defaults to
**empty**, and empty means every element the element models resolved. The names come
out of the `*_elements.txt` files beside the `.npy` arrays, so the report follows the
upstream models rather than a list held here. At the current draws that is **51
elements**, copper first and the rest alphabetical.

The default used to be eighteen critical and strategic raw materials, hard-coded.
That list made the stage fail outright against a set of draws that had no Pr, Tb or
Nb — the code was welded to names that were never its to decide. The available set is
a property of the upstream models' files and changes when those models change.

Naming elements in the parameter narrows the report to a subset, which is what you
want if you care about the critical raw materials and not about iron, silicon and the
sixteen trace `*_ppm` impurities. Then:

- An element absent from **one** domain contributes nothing there and that is normal
  — platinum is a sensor element and appears in no motor.
- An element **no** domain resolves is **skipped**, with a note naming it and listing
  what was available. A request list is a selection, not a contract.
- A list where **nothing** resolves does stop the run: that means the wrong draws
  directory or the wrong models.

Two cautions when narrowing it. Not every name is an element — `Plastic` and
`Unspecified` are real rows in the motor model. And an element from a single domain,
such as Pd from PCB or Pt from sensors, carries only that one model's uncertainty, so
its band is narrower than a multi-domain element's for a reason that is not physical.

**One element name remains in the code**, in `04_02`'s `WIRING_ELEMENT = "Cu"`. It is
a statement about the model, not a selection: the wiring model reports a copper mass
and nothing else, so Wiring has no `*_elements.txt` and its single element cannot be
read from a file that does not exist. The two dedicated copper figures are guarded on
`"Cu" in elements` so a narrowed request that drops copper still runs.

### Cost of reporting everything

| | 18 requested (15 resolved) | empty — all 51 |
|---|---|---|
| per-element figures | 30 | 102 |
| element draw export, 5 years | 0.09 GB | 0.31 GB |

Narrow the parameter if that matters; nothing about the calculation changes either
way.

---

## 5b. Reproducibility

`04_02` is now deterministic: two consecutive runs are **byte-identical**, checked by
diffing their full logs.

It was not before. The per-pair seed for the segment split was
`seed + hash(group) % 10_000`, and Python salts `str` hashing per process, so every
run split the pairs differently. The symptom was small and easy to dismiss — the
recombination error moved in the fourth decimal, 2050 collected copper wandered by
~0.07% between runs — which is exactly why it survived. `zlib.crc32` of the same
bytes is the same number in every process, forever.

This was the only unseeded randomness in the pipeline. Every other stage derives its
generators from `monte_carlo.seed` or `materials_mc_seed` through
`np.random.SeedSequence`, spawned in sorted or configuration order, and no stage uses
the global `np.random` functions.

Pd and Pt each come from a single domain, so their bands carry only that one model's
uncertainty and are narrower than a multi-domain element's.

---

## 5c. The alloys, and why the draw export also writes them

**Added 2026-09-01, for the recovery model.**

What comes out of a shredder is a **material, not an element**. Magnetic
separation takes a ferrous stream, eddy current a non-ferrous one, and what the
recycler sells is steel scrap, an aluminium alloy and copper. Elements alloyed
into one of those stay in it — nobody separates the manganese out of recovered
steel, so *"manganese recovered"* describes a process that does not happen.

So the draw export writes each alloy's own mass alongside the elements:

    <out>/<flow>/copper__Wiring.npy     the harness
                 copper__Motors.npy     the windings
                 alalloy__Motors.npy
                 fealloy__Motors.npy

`04_01_carcomposition.py` has exported this shape all along, as
`<material>__<component>.npy`, and the recovery model reads it with
`child_layer = material` and no element layer at all. This makes 04_02's export
usable the same way. **Boards and sensors keep their element export and get no
alloy one**: they go to a recycler that genuinely does separate elements, so
gold, silver and palladium come out as themselves.

### The mapping, which is the only domain knowledge involved

`ALLOY_OF` in `code/04_02_BEVelectronics.py`, one line per alloy:

| alloy | from | note |
|---|---|---|
| `copper` | `copper`, and the bare `Cu` column | the element models name copper's own copper **without** a suffix — it is the base metal and the `__copper` columns are its impurities |
| `alalloy` | `bulk` | |
| `fealloy` | `esteel`, `cfsteel`, `magnet` | ferrite is ferrimagnetic, so a magnet leaves the separator **inside the ferrous stream** as an impurity in the steel. It is not a magnet product, and its strontium is not recovered as strontium |

### Nothing is approximated

`motors_<segment>_<alloy>_elements.txt` names an alloy's elements, and every one
of them appears in the main `motors_<segment>_elements.txt` as
`<element>__<alloy>` — esteel's `Fe Si C Mn Al P S` are all there. An alloy's
mass is therefore the **exact** sum of its columns.

The sum is taken over **all** columns, never over the reported `elements`
subset. Narrowing that list would otherwise shrink an alloy without saying so —
and that is not hypothetical: the 2026-09-01 run resolved 24 elements, dropped
`Fe`, and left the two steels holding their manganese alone.

Measured on the committed element files, mean fraction of a motor:

| segment | `copper` | `alalloy` | `fealloy` | unmapped | total |
|---|---|---|---|---|---|
| AB | 0.1546 | 0.0313 | 0.7416 | 0.0725 | **1.000000** |
| CD | 0.1490 | 0.1030 | 0.6737 | 0.0743 | **1.000000** |
| EF | 0.1466 | 0.1458 | 0.6309 | 0.0767 | **1.000000** |

Every material named in the files is mapped; the unmapped remainder is
`Plastic` and `Unspecified`, which is **deliberately not exported**. It is
genuinely unresolved, and the recovery model derives it as a `rest` child and
treats it as unrecovered — the honest reading, and what makes every recovery
figure there a lower bound.

### An alloy domain exports its alloys and nothing else

`Wiring` and `Motors` write their alloys and their domain mass, and **no element
files at all**. Writing both would put the same mass in the folder twice under
two names — `fealloy__Motors` already contains every gram of `Mn__esteel__Motors`
and `Sr__magnet__Motors` — and a consumer reading the folder as materials would
count both. The elements inside an alloy are not recovered separately, so there
is nothing downstream for them to key on either.

The skip is per **domain**, not per element, so it cannot be got wrong by
editing `bev_electronics_elements`. Requesting `Mn__esteel` still reports and
plots it here; it simply does not reach the export. `PCB` and `Sensors` are
unaffected and keep every element file they had.

So the folder after a run holds, per flow:

    __domain____Wiring   __domain____Motors   __domain____PCB   __domain____Sensors
    copper__Wiring
    copper__Motors   alalloy__Motors   fealloy__Motors
    <element>__PCB   <element>__Sensors

Verified by replaying the added block against the real element files rather than
by re-running the stage: `Wiring` writes `copper` at 100% of its domain,
`Motors` writes `fealloy` 68.3%, `copper` 15.0% and `alalloy` 9.3%, no alloy
exceeds its domain on any draw, and the three plus the unmapped remainder come
to the whole.

---

## 6. Verification

| check | result |
|---|---|
| wiring copper via the element path == the wiring series | 5.2e-8 (float32 storage precision) |
| motor Cu against the material table | 0.08 – 0.15% |
| motor Σfraction, per draw | 1.00000000 |
| sensor mass `.npy` against the model's own summary | 2e-16 |
| copper domain split, 2050 inflow | Wiring 86.2%, Motors 12.4%, PCB 0.9%, Sensors 0.25% |

The copper split is the informative one: it independently reproduces the 85–91% /
8–14% / <1% / 0.1% split measured from the study's own element table, from a
completely different code path.

**Total copper is now ~16% higher than this stage used to report**, because it used
to report wiring only.

---

## 7. Memory

Segments are pooled per (domain, electronics group) once per flow — 12 arrays,
float32, ~41 MB each at 200,000 draws x 51 years, so ~490 MB. Only one element's
accumulator and one domain's contribution are held in float64 at a time, ~82 MB each.

Those are the element engine's own arrays. The **measured** peak for the whole stage
is **5.3 GB at 200,000 draws**, most of it the fleet and electronics draws the stage
already holds, and the run takes **92 s**. Everything scales linearly with draw
count, so 500,000 draws would need roughly 13 GB.

An earlier version of this note predicted 650 MB. That was the element engine's own
footprint mistaken for the process peak — the figure above is measured with
`/usr/bin/time -l`.

Element draws are never resampled. If the element models hold fewer rows than the run
needs, the stage stops rather than inventing draws the models never made.
