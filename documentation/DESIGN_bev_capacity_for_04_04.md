# BEV capacity for stage 04_04 — the discussion, the evidence and the decisions

How many kWh a BEV of a given segment carries, in a given year, and how that
enters the Monte Carlo. Settled 2026-09-14. Implemented 2026-09-14 in `src/battery_capacity.py`; §7 records what it
produces.

Every number below was measured from `EV_details.csv` (the EV-database scrape
held in RAWCLICVehicleBattery), models introduced from 2022 unless stated, on
the **nominal** capacity basis.

---

## 1. Why this is 04_04's problem

The battery project answers *what is inside a battery of capacity X*. It
deliberately does not answer *what capacity a segment carries*, because that
depends on how many cars of what size exist and when — fleet knowledge that
lives here, not there. Its `06_segment_capacity.py` is the cut remains of an
earlier attempt and is explicitly not its deliverable.

So 04_04 must decide the capacity, then interpolate the composition files over
it. Their anchors are 25 / 45 / 60 / 80 / 100 kWh.

**The existing answer, `params.materials.battery_size_map`, is wrong** and by a
lot. Fitted nominal capacity in 2026 against what it assumes:

| A | B | C | D | E | F | JB | JC | JD | JE | JF |
|---|---|---|---|---|---|---|---|---|---|---|
| +16% | +2% | +10% | +2% | +14% | +6% | **+30%** | **+27%** | +10% | **+31%** | +6% |

Every segment sits above it. JC is the most populous in the file, assumed at
60 kWh against a measured 76.5.

---

## 2. Decision 1 — capacity is a DISCRETE MIXTURE, not a continuum

A segment does not offer a continuum of pack sizes; it offers a handful. At
5 kWh resolution, the top four bins cover 66–87% of a segment's models, and
some segments have one dominant size: JC's 80 kWh bin holds 39% of 235 models,
JD's 100 kWh holds 38%.

So the wide capacity range within a segment is **a mixture of a few real pack
sizes**, not spread around a single one. Modelling it as a continuous
distribution would misrepresent what a segment actually is.

### The levels, as agreed

At most **five levels** per segment, each holding **at least 10% of the
segment's models**, renormalised over the kept levels.

```python
"A":  ((25, 30, 35),              (0.600, 0.200, 0.200)),   # n=10  THIN
"B":  ((40, 45, 50, 55),          (0.275, 0.175, 0.200, 0.350)),
"C":  ((55, 60, 65, 80, 85),      (0.197, 0.268, 0.211, 0.183, 0.141)),
"D":  ((80, 85, 90, 100),         (0.400, 0.283, 0.167, 0.150)),
"E":  ((85, 90, 100),             (0.246, 0.188, 0.565)),
"F":  ((90, 95, 105, 120),        (0.208, 0.250, 0.361, 0.181)),
"JA": ((42, 49),                  (0.500, 0.500)),          # ASSUMED, see below
"JB": ((50, 55, 60, 65, 70),      (0.191, 0.353, 0.118, 0.132, 0.206)),
"JC": ((65, 70, 80, 85),          (0.170, 0.152, 0.538, 0.140)),
"JD": ((75, 80, 100),             (0.156, 0.278, 0.567)),
"JE": ((100, 105, 115),           (0.436, 0.256, 0.308)),
"JF": ((100, 110, 120, 125),      (0.306, 0.278, 0.139, 0.278)),
```

> **Why 10% and not "at least 3 models".** An absolute threshold was tried
> first and collapsed segment A to a single level at 25 kWh, destroying the
> spread the whole design exists to keep — A has 10 models, so its 30 and 35 kWh
> levels hold 2 cars each. A proportional rule scales with the sample and keeps
> a level that is 20% of a thin segment.

> **⚠️ JA is an assumption, not a measurement.** It has two models, both Hyundai
> INSTER, at 42 and 49 kWh. The 50/50 weighting is a choice. Its bootstrap CI is
> ±5 kWh on a 7 kWh span — entirely noise. **A is thin too**, at 10 models.

---

## 3. Decision 2 — ONE LEVEL DRAWN PER MONTE CARLO DRAW

This was the fork that mattered, and it was decided deliberately.

| | what it means | effect on the band |
|---|---|---|
| **chosen** — draw one level per draw | "we do not know which pack this car has" | **keeps the wide range** |
| rejected — split the fleet across levels | "we know the mix, and here it is" | collapses the range to near nothing |

Both give the same mean. The second is what a fleet physically is; the first is
what was chosen, because the range is considered essential and must survive into
the output.

> **This has to be said in any output that shows the band.** The width is a
> mixture of real pack sizes being sampled one at a time, not uncertainty about
> how much material a battery contains. A reader who assumes the latter will
> over-read it.

---

## 4. Decision 3 — moderate growth, then a plateau

### The evidence for the plateau

Fleet-wide, by introduction period:

| | capacity | efficiency | real range |
|---|---|---|---|
| 2018–21 | 68 kWh | 164 Wh/km | 310 km |
| 2022–23 | 82 kWh | 162 Wh/km | 410 km |
| 2024–27 | **82 kWh** | **158 Wh/km** | **425 km** |

Capacity stopped at 82 kWh and has not moved, while efficiency kept improving.
Range gained 100 km in the first step and 15 km in the second.

And it arrives segment by segment, not all at once:

| segment | capacity | Wh/km | range | state |
|---|---|---|---|---|
| C | 62 → 61 → 63 | 145 → 142 | 350 → 330 → 360 | flat throughout |
| JC | 70 → 78 → 82 | 163 → 155 | 355 → 400 → **400** | range plateaued |
| JD | 85 → 88 → 90 | 181 → 167 | 380 → 402 → 458 | still growing |
| F | 93 → 93 → 105 | 160 → 161 | 420 → 485 → 532 | still growing |

**The mechanism is that RANGE saturates, not capacity.** Once a segment reaches
the range its buyers want, more kWh is dead weight, and further efficiency gains
show up as range at the same capacity. Fast charging — which Chinese
manufacturers offer and which addresses the European range anxiety directly —
removes the pressure to buy range with capacity. The price evidence points the
same way: an LFP car is 17.7% ± 1.4 pp cheaper at the same capacity and carries
+2.0% ± 1.8 pp more kWh at the same price, so the chemistry saving went to price
and not to capacity.

### The parameters, as agreed

| | min | mode | max |
|---|---|---|---|
| growth per decade | 0.05 | **0.10** | 0.20 |
| plateau year | 2035 | **2040** | 2050 |

**Both are triangular and both are drawn once per Monte Carlo draw**, alongside
the level. So a 2050 capacity carries a real band saying "we do not know whether
this levelled off in 2035 or 2050", rather than a single curve implying we do.

> **Why the plateau is drawn rather than fitted.** The usable record is roughly
> 2020–2026, with 13–20 models per segment in the early periods against 106–158
> now, and it is models rather than registrations. Six or seven years cannot
> distinguish "plateaued" from "paused" — JC going 400 → 400 km could be either.
> The saturation is therefore an assumption, and it is carried as a distribution
> because that is what an untestable structure honestly is.

> **Open, not decided: what happens after the plateau.** If the target range is
> fixed and consumption keeps falling ~4% per period, the same range needs
> slightly less battery each year — capacity would drift *down* about half a
> percent a year, for forty years. Modelled as flat for now.

---

## 5. Decision 4 — the weights are PARAMETERS, not drawn

Asked directly: are the weights an accident of a small sample?

They are noisy. The worst individual weight standard error runs from ±0.033
(JC, n=235) to ±0.155 (A, n=10) and ±0.354 (JA, n=2).

**But that noise barely reaches the answer.** Bootstrapping the models and
refitting gives a 95% CI on the resulting mean capacity of:

| JC | JB | D | E | F | JF | A | JA |
|---|---|---|---|---|---|---|---|
| 2.1 | 3.3 | 3.4 | 3.2 | 4.7 | 6.5 | **5.0** | **10.0** |

kWh wide — against level spans of 15–30 kWh. For JC that is ±1 kWh of sampling
error against ±10 kWh of real mixture, about eleven to one.

**So the weights are fixed as parameters and only the level is drawn.** Drawing
the weights as well would add randomness that comes from having too few cars
rather than from anything real, which is explicitly not wanted. A and JA are the
two segments where this reasoning does not hold, and they are marked as
assumptions above.

---

## 6. A question that was asked and answered: is the mean good enough?

Mass at the mean capacity, against the weighted mean of mass at each level,
LFP:

| seg | mean kWh | mass(mean) | mean(mass) | difference |
|---|---|---|---|---|
| C | 64.3 | 446.6 kg | 447.3 kg | −0.17% |
| JC | 76.6 | 502.9 | 503.1 | −0.03% |
| F | 102.1 | 624.1 | 623.8 | +0.04% |

Under 0.2% everywhere. kg/kWh falls steeply — 9.89 at 25 kWh to 6.14 at 100 —
but that is a fixed intercept, not curvature: mass runs about 125 kg + 4.9 kg
per kWh. Over the narrow span the levels cover there is almost no curvature left.

**Which is exactly why the mean is the wrong choice.** It buys nothing in
accuracy and costs the entire spread. The discrete draw changes the band, not
the central — a good property, and worth knowing: the large change against
today's `battery_size_map` comes from the *level* (JC 76.5 against 60), not from
the discreteness.

> This holds only because the levels span a narrow range. If a segment's levels
> ever spanned 25–100 kWh the curvature would matter and the two would part.

---

## 7. Implemented, 2026-09-14

`src/battery_capacity.py` — `capacity_draws(params, segment, years, n_draws, seed)`
returns `(n_draws, n_years)` in nominal kWh. Parameters are
`materials.battery_capacity_levels`, `…_levels_year`, `…_growth_per_decade` and
`…_plateau_year`. Run the module directly for a summary table.

Verified: the drawn levels are discrete and reproduce the parameter weights to
0.002; at the anchor year every draw equals its own level exactly; growth and
plateau are shared across segments while the level choice is independent.

Mean capacity it produces, against the map it replaces:

| seg | 2020 | 2030 | 2040 | 2050+ | `battery_size_map` |
|---|---|---|---|---|---|
| C | 64.2 | 71.7 | 79.6 | 81.7 | 60 |
| F | 97.8 | 109.2 | 121.2 | 124.3 | 100 |
| JC | 73.3 | 81.9 | 90.9 | 93.3 | 60 |
| JE | 101.3 | 113.1 | 125.6 | 128.8 | 80 |

Band width grows with the year — JC 20.8 kWh in 2020 to 38.1 by 2050 — because
the plateau date is drawn and its effect compounds.

> **⚠️ This pushes the large segments past the composition files' top anchor.**
> Those files run to 100 kWh and 04_04 must interpolate over them. Share of
> draws above 100 kWh: JE and JF **100% from 2030**, F 74% in 2030 rising to
> 99% by 2050, JC 15% by 2040, C 10% by 2040. So a large part of the fleet is
> extrapolated composition, on a linear continuation past the top anchor.
> Whether that is acceptable, or whether the workbook needs an anchor above
> 100 kWh, is not decided.

> **⚠️ The backward arm is an extrapolation.** Growth runs backwards from 2024
> at the same drawn rate, so a car scrapped in 2040 gets its build year's
> capacity. Measured capacity was 68 kWh in 2018–21 against 82 now — a steeper
> rise than 10% per decade — so the early years come out somewhat too high.

## 7b. Discreteness, and the 5 kWh grid that was rejected

Whether a capacity stays discrete once a draw runs forty years forward is its
own question, and the answer is not simply yes: **inside a draw a segment has
exactly its levels and nothing between them, while across draws the values
smear** — 4 distinct capacities at the anchor, ~50,000 at 2040, which are
guesses about what the four sizes will be rather than sizes in one market.

A proposal to round each draw's grown levels to a 5 kWh grid was **refuted by
the data**: only 25.6% of 1,068 models sit on a multiple of 5 kWh, and the
commonest sizes are 82, 100, 84, 54, 105 and 91 kWh. Pack capacity is cell
count times cell capacity, not a round number. Full argument in
[`DESIGN_discrete_vehicle_states.md`](DESIGN_discrete_vehicle_states.md).

## 8. What is still open
2. **Which chemistry scenario drives the run** — S1 / S2 / S3 from the battery
   project, and how it combines with 03_02's six flow scenarios. Six × three is
   eighteen runs.
3. **Only BAU has per-draw flow arrays on disk.** The other five scenarios need
   a 03_02 rerun.
4. **Post-plateau drift** — flat, or down with efficiency (§4).
5. **Sodium and solid-state have no active-material composition.** Under S2 and
   S3 a growing share of the fleet produces packaging masses and no cathode,
   anode or electrolyte — most of the market by 2070 under S3. 04_04 has to show
   that gap rather than let the totals quietly fall.
6. **These are models, not registrations.** Every weight and level above is a
   share of model variants offered, not of cars sold. Sales weighting would
   change them, and the EEA data needed for it is already in this repo.

## 9. The random streams, and their tags (2026-10-09)

Every stream is seeded `[seed, crc32(<its name>)]`, or `[seed, crc32(<its name>), crc32(<key>)]` when it
is per segment or per chemistry -- never with the bare seed, and never with `hash()`, which Python salts
per process. Seeded alike, two streams are one stream; until 2026-10-09 four pairs of them were.

| stream | tag (`crc32` of) | key | shared on purpose |
|---|---|---|---|
| capacity growth rate, plateau year | `battery_capacity.growth` | — | by every segment (§3) |
| pack size | `battery_capacity.level` | segment | no; held across years |
| voltage adoption order | `battery_voltage.adoption` | segment | no; held for the car's life |
| voltage share band | `battery_voltage.band` | — | market-wide: a world in which 800 V is early is early everywhere |
| composition extrapolation factor | `battery_composition.extrapolation` | — | the same uniform for draw *i* in every call with that seed, so both levels agree |
| chemistry share multiplier | `battery_chemistry.share` | chemistry | across groups and years (`battery_chemistry.py`) |

The pack size and the voltage of a segment are therefore independent, which they were not: the same
uniform picked the pack size and placed the car in the adoption order, and a small pack was almost
always 800 V. The capacity growth and the voltage band were tied exactly, rank correlation +1.0000.
A new stream takes a new name. HANDOVER.md of 2026-10-09 has the measurements.

