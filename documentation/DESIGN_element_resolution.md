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

`materials.bev_electronics_elements` — critical and strategic raw materials by
default, not everything the models resolve. Iron and silicon are most of a motor by
mass and are of no interest for criticality.

```
Cu, Nd, Dy, Pr, Tb, Co, Li, Pt, Pd, Au, Ag, Ga, Ge, In, Ta, W, Nb, Al
```

An element absent from a domain contributes nothing there and that is normal —
platinum is a sensor element and appears in no motor. An element **no** domain
resolves stops the run with a message listing what is available, rather than quietly
producing zeros.

Pd and Pt each come from a single domain, so their bands carry only that one model's
uncertainty and are narrower than a multi-domain element's.

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
